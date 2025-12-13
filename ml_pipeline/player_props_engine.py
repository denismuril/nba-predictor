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

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "ml_pipeline" / "models"
MODELS_DIR.mkdir(exist_ok=True)

# MLflow Setup
mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
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
    df['MIN_L5'] = df.groupby('Player')['MIN'].transform(lambda x: x.shift(1).rolling(5).mean())
    features.append('MIN_L5')
    
    # Dias de descanso
    df['Rest_Days'] = df.groupby('Player')['Date'].diff().dt.days.fillna(3)
    features.append('Rest_Days')
    
    # Home/Away (1 = Home, 0 = Away)
    df['Is_Home'] = df['Location'].apply(lambda x: 1 if x == 'Home' else 0)
    features.append('Is_Home')
    
    # Opponent Defense Rank (Placeholder - idealmente viria de um lookup externo)
    # df = df.merge(defense_ranks, on='Opponent')
    
    # Limpar NaNs gerados pelo shift/rolling
    df_clean = df.dropna(subset=features + targets)

    return df_clean, features, targets


# =============================================================================
# FASE 3: Hit Rates - Cálculo de Frequência de Acerto
# =============================================================================

from typing import Dict


def calculate_hit_rates(
    df_player: pd.DataFrame,
    target_col: str,
    line_value: float
) -> Dict[str, float]:
    """
    FASE 3 IMPLEMENTATION: Calcula hit rates com proteção anti-data-leakage.

    IMPORTANTE: Usa .shift(1) para garantir que apenas jogos ANTES da data atual
    são incluídos no cálculo.

    Args:
        df_player: DataFrame de um jogador, deve estar ordenado por data
        target_col: Coluna target ('PTS', 'REB', 'AST')
        line_value: Linha a avaliar (ex: 25.5 para PTS Over)

    Returns:
        Dict com:
        - 'L5': Hit rate últimos 5 jogos (0.0 a 1.0)
        - 'L10': Hit rate últimos 10 jogos
        - 'L20': Hit rate últimos 20 jogos
        - 'home': Hit rate em jogos em casa
        - 'away': Hit rate em jogos fora
        - 'overall': Hit rate geral

    Zero Data Leakage:
        O .shift(1) garante que estamos olhando para jogos ANTERIORES,
        nunca para o jogo atual ou futuros.

    Exemplo:
        >>> player_df = df[df['Player'] == 'LeBron James'].sort_values('Date')
        >>> rates = calculate_hit_rates(player_df, 'PTS', 25.5)
        >>> if rates['L10'] >= 0.80:
        ...     print("🔥 Jogador está quente!")
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

    if home_col in df.columns:
        if home_col == 'Is_Home':
            home_mask = df[home_col] == 1
        else:
            home_mask = df[home_col] == 'Home'

        home_games = df[home_mask]
        away_games = df[~home_mask]

        home_rate = home_games['_over_line'].shift(1).mean() if len(home_games) > 0 else 0.5
        away_rate = away_games['_over_line'].shift(1).mean() if len(away_games) > 0 else 0.5
    else:
        home_rate = 0.5
        away_rate = 0.5

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


def add_hit_rate_features(df: pd.DataFrame, lines: Dict[str, float] = None) -> pd.DataFrame:
    """
    Adiciona colunas de hit rate para todos os jogadores no DataFrame.

    Args:
        df: DataFrame com boxscores (Player, Date, PTS, REB, AST, etc.)
        lines: Dict com linhas por stat. Ex: {'PTS': 20.5, 'REB': 8.5, 'AST': 5.5}
               Se None, usa médias L10 como linha dinâmica.

    Returns:
        DataFrame com colunas adicionais:
        - {stat}_hit_L5, {stat}_hit_L10, {stat}_hit_L20
        - {stat}_hit_home, {stat}_hit_away
    """
    logger.info("📊 Calculando Hit Rate features para todas as props...")

    df = df.copy()

    # Stats a calcular
    stat_cols = ['PTS', 'REB', 'AST']

    for stat in stat_cols:
        if stat not in df.columns:
            continue

        # Inicializar colunas
        for suffix in ['L5', 'L10', 'L20', 'home', 'away']:
            df[f'{stat}_hit_{suffix}'] = 0.5

        # Calcular por jogador
        for player in df['Player'].unique():
            player_mask = df['Player'] == player
            player_df = df[player_mask].sort_values('Date')

            # Usar linha fornecida ou média L10 como proxy
            if lines and stat in lines:
                line = lines[stat]
            else:
                # Usar média L10 do jogador como linha dinâmica
                line = player_df[stat].shift(1).rolling(10, min_periods=3).mean().iloc[-1]
                if pd.isna(line):
                    line = player_df[stat].mean()

            rates = calculate_hit_rates(player_df, stat, line)

            # Aplicar ao último jogo do jogador
            last_idx = player_df.index[-1]
            df.loc[last_idx, f'{stat}_hit_L5'] = rates['L5']
            df.loc[last_idx, f'{stat}_hit_L10'] = rates['L10']
            df.loc[last_idx, f'{stat}_hit_L20'] = rates['L20']
            df.loc[last_idx, f'{stat}_hit_home'] = rates['home']
            df.loc[last_idx, f'{stat}_hit_away'] = rates['away']

    logger.info(f"✅ Hit Rate features calculadas para {len(stat_cols)} stats")
    return df

def train_prop_model(target_col, df, features):
    """
    Treina um modelo XGBoost Regressor para uma estatística específica (PTS, REB, AST).
    """
    logger.info(f"\n🏀 Treinando modelo para: {target_col}")
    
    X = df[features]
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
        logger.info(f"💾 Modelo salvo em: {model_path}")
        
        # Logar artefato no MLflow
        mlflow.sklearn.log_model(model, f"model_{target_col}")
        
    return model, mae

if __name__ == "__main__":
    logger.info("🚀 Iniciando Pipeline de Treinamento de Player Props...")
    
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
        # Criar arquivos dummy para evitar erro no frontend se não houver modelos
        for t in ['pts', 'reb', 'ast']:
            dummy_path = MODELS_DIR / f"xgb_{t}.joblib"
            if not dummy_path.exists():
                logger.warning(f"⚠️  Criando placeholder vazio para {dummy_path}")
                # Não criamos arquivo vazio pq joblib vai falhar ao carregar.
                # O frontend deve tratar a ausência do arquivo.
