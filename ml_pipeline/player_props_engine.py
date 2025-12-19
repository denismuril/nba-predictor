import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_pipeline" / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MLflow Setup
# MLflow Setup
try:
    # Usar caminho relativo simples para evitar problemas de protocolo file:// no WSL/Windows
    mlflow.set_tracking_uri("./mlruns")
except Exception as e:
    logger.warning(f"⚠️ Erro ao configurar MLflow URI: {e}")
mlflow.set_experiment("NBA_Player_Props_v1")

def load_and_prepare_data():
    """
    Carrega dados históricos e prepara features para Player Props.
    Idealmente, isso viria de um banco de dados de boxscores históricos.
    Como placeholder, vamos simular a estrutura esperada ou carregar de um CSV se existir.
    """
    # TODO: Implementar carregamento real de histórico de jogadores (boxscores)
    # Por enquanto, vamos criar um script que treina se o arquivo existir, ou avisa que precisa de dados.
    
    history_file = DATA_DIR / "player_boxscores_history.csv"
    
    if not history_file.exists():
        # FASE 5 FIX: Um sistema de apostas NUNCA deve treinar com dados sintéticos/falsos!
        # Isso criaria um modelo que "parece funcionar" mas é inútil em produção.
        raise FileNotFoundError(
            f"❌ Histórico de jogadores não encontrado em: {history_file}\n"
            "📥 Execute o scraper primeiro para coletar dados reais:\n"
            "   python -m data.scrapers.player_boxscores_scraper\n\n"
            "⚠️  Um sistema de apostas NUNCA deve treinar com dados sintéticos."
        )

    df = pd.read_csv(history_file)
    
    # Converter data para datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Player', 'Date'])
    
    # Feature Engineering
    logger.info("🛠️  Gerando Features (Rolling Averages, Rest Days)...")
    
    features = []
    targets = ['PTS', 'REB', 'AST']
    
    for stat in targets:
        # Média Móvel 5 e 10 jogos
        df[f'{stat}_L5'] = df.groupby('Player')[stat].transform(lambda x: x.shift(1).rolling(5).mean())
        df[f'{stat}_L10'] = df.groupby('Player')[stat].transform(lambda x: x.shift(1).rolling(10).mean())
        features.extend([f'{stat}_L5', f'{stat}_L10'])
        
    # Minutos médios L5
    if 'MIN' in df.columns:
        df['MIN_L5'] = df.groupby('Player')['MIN'].transform(lambda x: x.shift(1).rolling(5).mean())
        features.append('MIN_L5')
    
    # Dias de descanso
    if 'Date' in df.columns:
        df['Rest_Days'] = df.groupby('Player')['Date'].diff().dt.days.fillna(3)
        features.append('Rest_Days')
    
    # H2H (se disponível Opponent)
    if 'Opponent' in df.columns:
        for stat in targets:
            # Shift 1 para evitar leak do jogo atual
            df[f'{stat}_H2H'] = df.groupby(['Player', 'Opponent'])[stat].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            features.append(f'{stat}_H2H')
    
    # Home/Away (1 = Home, 0 = Away)
    if 'Location' in df.columns:
        df['Is_Home'] = df['Location'].apply(lambda x: 1 if x == 'Home' else 0)
        features.append('Is_Home')
    
    # Limpar NaNs gerados pelo shift/rolling
    df_clean = df.dropna(subset=features + targets)

    return df_clean, features, targets


# =============================================================================
# FASE 3: Hit Rates - Cálculo de Frequência de Acerto
# =============================================================================

def calculate_hit_rates(
    df_player: pd.DataFrame,
    target_col: str,
    line_value: float
) -> Dict[str, float]:
    """
    FASE 3 IMPLEMENTATION: Calcula hit rates com proteção anti-data-leakage.
    """
    if df_player is None or df_player.empty:
        return {
            'L5': 0.5, 'L10': 0.5, 'L20': 0.5,
            'home': 0.5, 'away': 0.5, 'overall': 0.5
        }

    # Garantir ordenação temporal
    df = df_player.sort_values('Date').copy()

    # Marcar hits ANTES do jogo atual (anti-leakage com shift)
    df['_over_line'] = (df[target_col] > line_value).astype(int)

    # Calcular rolling hit rate com shift
    df['_hit_L5'] = df['_over_line'].shift(1).rolling(5, min_periods=1).mean()
    df['_hit_L10'] = df['_over_line'].shift(1).rolling(10, min_periods=1).mean()
    df['_hit_L20'] = df['_over_line'].shift(1).rolling(20, min_periods=1).mean()

    # Pegar os valores mais recentes (último jogo)
    last_row = df.iloc[-1]

    l5 = last_row.get('_hit_L5', 0.5)
    l10 = last_row.get('_hit_L10', 0.5)
    l20 = last_row.get('_hit_L20', 0.5)

    # Home/Away split (se disponível)
    home_col = 'Is_Home' if 'Is_Home' in df.columns else 'Location'
    
    home_rate = 0.5
    away_rate = 0.5

    if home_col in df.columns:
        if home_col == 'Is_Home':
            home_mask = df[home_col] == 1
        else:
            home_mask = df[home_col] == 'Home'

        home_games = df[home_mask]
        away_games = df[~home_mask]

        home_rate = home_games['_over_line'].shift(1).mean() if len(home_games) > 0 else 0.5
        away_rate = away_games['_over_line'].shift(1).mean() if len(away_games) > 0 else 0.5

    # Overall (histórico completo)
    overall = df['_over_line'].shift(1).mean()

    return {
        'L5': round(float(l5 if pd.notna(l5) else 0.5), 3),
        'L10': round(float(l10 if pd.notna(l10) else 0.5), 3),
        'L20': round(float(l20 if pd.notna(l20) else 0.5), 3),
        'home': round(float(home_rate if pd.notna(home_rate) else 0.5), 3),
        'away': round(float(away_rate if pd.notna(away_rate) else 0.5), 3),
        'overall': round(float(overall if pd.notna(overall) else 0.5), 3)
    }

def train_prop_model(target_col, df, features):
    """
    Treina um modelo XGBoost Regressor para uma estatística específica (PTS, REB, AST).
    """
    logger.info(f"\n🏀 Treinando modelo para: {target_col}")
    
    # Garantir que apenas colunas numéricas vão para o treino
    X = df[features].select_dtypes(include=[np.number])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name=f"Prop_{target_col}"):
        # Parâmetros otimizados para regressão
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50
        }
        
        mlflow.log_params(params)
        
        model = xgb.XGBRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Avaliação
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        logger.info(f"✅ {target_col} Results - MAE: {mae:.2f} | R2: {r2:.3f}")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Salvar modelo localmente
        model_path = MODELS_DIR / f"xgb_{target_col.lower()}.joblib"
        joblib.dump(model, model_path)
        
        # Salvar metadados do modelo (importante para inferência EV)
        metadata = {
            "mae": mae,
            "features": list(X.columns),
            "target": target_col,
            "trained_at": datetime.now().isoformat()
        }
        metadata_path = MODELS_DIR / f"xgb_{target_col.lower()}_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"💾 Modelo e metadados salvos em: {model_path}")
        
        # Logar artefato no MLflow
        mlflow.sklearn.log_model(model, f"model_{target_col}")
        
    return model, mae


# =============================================================================
# FASE 4: INFERÊNCIA EV+ (Expected Value)
# =============================================================================

def calculate_ev(model_probability: float, decimal_odds: float) -> float:
    """
    Calcula Valor Esperado da aposta.
    EV = (Prob_Modelo * Odd) - 1
    
    Args:
        model_probability: Probabilidade estimada pelo modelo (0.0 a 1.0)
        decimal_odds: Odd decimal da casa de apostas (ex: 1.90)
        
    Returns:
        EV como decimal (ex: 0.05 = 5% EV)
    """
    return (model_probability * decimal_odds) - 1

def _get_probability_from_projection(
    projection: float, 
    line: float, 
    mae: float, 
    direction: str = 'over'
) -> float:
    """
    Calcula probabilidade baseada na projeção e MAE (assumindo dist. normal).
    
    Args:
        projection: Valor projetado pelo modelo
        line: Linha da aposta
        mae: Erro médio absoluto do modelo (usado como proxy de desvio padrão)
        direction: 'over' ou 'under'
        
    Returns:
        Probabilidade (0.0 a 1.0)
    """
    if mae <= 0:
        return 0.5
        
    # Z-score: quão longe a linha está da projeção, em unidades de erro
    z_score = (projection - line) / (mae * 1.25) # 1.25 fator de conservadorismo (RMSE ~ 1.25 MAE)
    
    # Probabilidade de ser maior (Over)
    prob_over = norm.cdf(z_score)
    
    if direction.lower() == 'over':
        return prob_over
    else:
        return 1.0 - prob_over

def predict_with_ev(
    props_df: pd.DataFrame,
    models_dir: Path = None,
    ev_threshold: float = 0.05,
    confidence_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Motor Principal de Inferência.
    
    Recebe DataFrame do PropsProcessor, carrega modelos, faz previsões
    e calcula EV para identificar Sniper Bets.
    
    Args:
        props_df: DataFrame processado com features
        models_dir: Diretório dos modelos (opcional)
        ev_threshold: Limite mínimo de EV para recomendar (0.05 = 5%)
        confidence_threshold: Limite de confiança (probabilidade)
        
    Returns:
        DataFrame enriquecido com:
        - prediction (valor projetado)
        - model_prob (probabilidade estimada)
        - ev (valor esperado)
        - recommendation (Sniper, Value, Skip)
    """
    df = props_df.copy()
    models_dir = models_dir or MODELS_DIR
    
    # Inicializar colunas de resultado
    df['prediction'] = np.nan
    df['model_prob'] = np.nan
    df['ev'] = np.nan
    df['recommendation'] = 'Skip'
    df['mae_used'] = np.nan
    
    # Agrupar por tipo de prop para carregar modelo correto
    for prop_type in df['prop_type'].unique():
        # Mapear prop types para modelos (points -> pts, rebounds -> reb)
        model_name = prop_type.lower()
        if model_name == 'points': model_name = 'pts'
        elif model_name == 'rebounds': model_name = 'reb'
        elif model_name == 'assists': model_name = 'ast'
        # Adicionar outros mapeamentos conforme necessário (threes, etc)
        
        model_path = models_dir / f"xgb_{model_name}.joblib"
        meta_path = models_dir / f"xgb_{model_name}_metadata.joblib"
        
        if not model_path.exists():
            logger.warning(f"⚠️ Modelo não encontrado para {prop_type}: {model_path}")
            continue
            
        try:
            model = joblib.load(model_path)
            
            # Tentar carregar metadados para pegar MAE
            mae = 1.0 # Default
            features = []
            
            if meta_path.exists():
                meta = joblib.load(meta_path)
                mae = meta.get('mae', 1.0)
                features = meta.get('features', [])
            
            # Se não temos features salvas, tentar inferir do modelo
            if not features and hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
            
            # Filtrar dados para este prop type
            # prop_type vem do scraper, pode ser 'points', 'rebounds', etc.
            mask = df['prop_type'] == prop_type
            subset = df[mask]
            
            if subset.empty:
                continue
                
            # Verificar se temos todas as features necessárias
            # Mapeamento de colunas do PropsProcessor para features do modelo
            # O modelo treinado espera: PTS_L5, PTS_L10, Rest_Days...
            # O PropsProcessor entrega: season_avg, L5_AVG, H2H_AVG, REST_DAYS...
            
            # ADAPTER: Renomear colunas do PropsProcessor para formato do modelo
            # Isso é crucial! O modelo foi treinado com nomes específicos.
            
            # Mapeamento dinâmico
            adapter_map = {
                'season_avg': f'{model_name.upper()}_L10', # Proxy: season avg ~ L10
                'L5_AVG': f'{model_name.upper()}_L5',
                'REST_DAYS': 'Rest_Days',
                'H2H_AVG': f'{model_name.upper()}_H2H',
                # Adicionar outros conforme feature engineering do treino
            }
            
            X_subset = pd.DataFrame()
            for feat in features:
                # Tentar encontrar a feature correspondente
                source_col = None
                
                # 1. Checar mapeamento direto
                for src, dest in adapter_map.items():
                    if dest == feat:
                        source_col = src
                        break
                        
                # 2. Se não achou, checar se existe direto no df
                if not source_col and feat in subset.columns:
                    source_col = feat
                
                if source_col and source_col in subset.columns:
                    X_subset[feat] = subset[source_col]
                else:
                    # Feature faltante - preencher com média ou valor neutro
                    # logger.debug(f"Feature faltante {feat} para {prop_type}, preenchendo com 0")
                    X_subset[feat] = 0.0
            
            # Fazer predição
            preds = model.predict(X_subset)
            
            # Salvar predições
            df.loc[mask, 'prediction'] = preds
            df.loc[mask, 'mae_used'] = mae
            
            # Calcular probabilidades e EV
            for idx, row in df[mask].iterrows():
                # Determinar direção (Over/Under) com base na odd
                # Se temos over_odds, assumimos que estamos avaliando o Over
                # Se temos under_odds, também podemos avaliar o Under
                # O ideal é gerar linhas para ambos ou pegar a melhor?
                # Simplificação: Avaliar o Over
                
                proj = row['prediction']
                line = row['line']
                
                # Calcular prob pro Over
                prob_over = _get_probability_from_projection(proj, line, mae, 'over')
                
                # Calcular prob pro Under
                prob_under = 1.0 - prob_over
                
                # Decidir qual lado apostar baseado no maior EV
                # (assumindo que temos odds para ambos)
                
                best_ev = -1.0
                best_side = 'Skip'
                best_prob = 0.0
                
                if 'over_odds' in row and pd.notna(row['over_odds']):
                    ev_over = calculate_ev(prob_over, row['over_odds'])
                    if ev_over > best_ev:
                        best_ev = ev_over
                        best_side = 'Over'
                        best_prob = prob_over
                        
                if 'under_odds' in row and pd.notna(row['under_odds']):
                    ev_under = calculate_ev(prob_under, row['under_odds'])
                    if ev_under > best_ev:
                        best_ev = ev_under
                        best_side = 'Under'
                        best_prob = prob_under
                
                df.at[idx, 'ev'] = best_ev
                df.at[idx, 'model_prob'] = best_prob
                df.at[idx, 'recommended_side'] = best_side
                
                # Classificação Final
                if best_ev > ev_threshold and best_prob > confidence_threshold:
                    df.at[idx, 'recommendation'] = '⚡ SNIPER'
                elif best_ev > (ev_threshold * 0.6) and best_prob > (confidence_threshold * 0.9):
                    df.at[idx, 'recommendation'] = '💰 Value'
                else:
                    df.at[idx, 'recommendation'] = 'Skip'
                    
        except Exception as e:
            logger.error(f"❌ Erro na inferência para {prop_type}: {e}")
            continue
            
    return df

if __name__ == "__main__":
    logger.info("🚀 Iniciando Pipeline de Treinamento de Player Props...")
    
    try:
        data = load_and_prepare_data()
        
        if data:
            df, features, targets = data
            
            results = {}
            for target in targets:
                model, mae = train_prop_model(target, df, features)
                results[target] = mae
                
            logger.info("\n📊 Resumo do Treinamento:")
            for k, v in results.items():
                logger.info(f"  - {k}: MAE {v:.2f}")
        else:
            logger.info("⏭️  Pulando treinamento (sem dados históricos).")
            
    except Exception as e:
        logger.error(f"❌ Erro fatal no pipeline: {e}")
