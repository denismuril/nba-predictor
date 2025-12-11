"""
Script de Re-Treino com Dados Limpos
=====================================

Este script garante que o modelo seja treinado com dados consistentes,
aplicando os mesmos filtros de limpeza usados em predict.py:
- Isolamento de temporada (apenas 2025-26)
- Remoção de jogos com stats zeradas (pts=0, fga=0)

Além disso, compara a performance do modelo antigo vs novo em um
conjunto de validação recente (últimos 30 dias).
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, classification_report
import sys
import os

# Adicionar diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.repositories.db_manager import get_db_manager
from ml_pipeline.feature_engineering_v2 import prepare_features_v2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações
SEASON_START_DATE = '2025-10-01'
VALIDATION_DAYS = 30
OLD_MODEL_PATH = 'data/models/ensemble_model.joblib'
NEW_MODEL_PATH = 'data/models/ensemble_v2_clean.joblib'


def load_and_clean_data():
    """
    Carrega TODO o histórico e aplica filtros de limpeza.
    
    Returns:
        pd.DataFrame: Dados limpos e prontos para feature engineering
    """
    logger.info("=" * 80)
    logger.info("📂 CARREGANDO E LIMPANDO DADOS")
    logger.info("=" * 80)
    
    # 1. Carregar histórico completo
    db = get_db_manager()
    df_history = db.get_history()
    logger.info(f"✅ Histórico carregado: {len(df_history)} jogos")
    
    # 2. Filtrar temporada atual (mesma lógica do predict.py)
    logger.info(f"🔒 Aplicando filtro de temporada (>= {SEASON_START_DATE})...")
    initial_len = len(df_history)
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history = df_history[df_history['date'] >= pd.to_datetime(SEASON_START_DATE)].copy()
    removed = initial_len - len(df_history)
    logger.info(f"   Removidos {removed} jogos de temporadas anteriores")
    logger.info(f"   Restantes: {len(df_history)} jogos da temporada 2025-26")
    
    # 3. Normalizar nomes de times
    logger.info("🧹 Normalizando nomes de times...")
    from utils.team_normalization import normalize_team
    df_history['home_team'] = df_history['home_team'].apply(lambda x: normalize_team(x))
    df_history['away_team'] = df_history['away_team'].apply(lambda x: normalize_team(x))
    
    # Remove jogos onde normalização falhou
    initial_len = len(df_history)
    df_history = df_history.dropna(subset=['home_team', 'away_team'])
    dropped = initial_len - len(df_history)
    if dropped > 0:
        logger.warning(f"⚠️  Removidos {dropped} jogos com nomes inválidos")
    
    # 4. Remover jogos sem estatísticas (mesma lógica do predict.py)
    logger.info("🧹 Removendo jogos com stats zeradas...")
    numeric_cols = ['fgm', 'fga', 'fg3m', 'ftm', 'fta', 'oreb', 'dreb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts',
                    'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_ftm', 'opp_fta', 'opp_oreb', 'opp_dreb', 
                    'opp_ast', 'opp_stl', 'opp_blk', 'opp_tov', 'opp_pf', 'opp_pts']
    
    for col in numeric_cols:
        if col in df_history.columns:
            df_history[col] = df_history[col].fillna(0)
    
    initial_len = len(df_history)
    df_history = df_history[(df_history['fga'] > 0) & (df_history['opp_fga'] > 0)].copy()
    removed = initial_len - len(df_history)
    logger.info(f"   Removidos {removed} jogos com FGA=0 (stats zeradas)")
    logger.info(f"   Restantes: {len(df_history)} jogos limpos")
    
    # 5. Adicionar coluna de target
    if 'home_score' in df_history.columns and 'away_score' in df_history.columns:
        df_history['home_win'] = (df_history['home_score'] > df_history['away_score']).astype(int)
    
    logger.info(f"\n✅ Dados limpos: {len(df_history)} jogos prontos para feature engineering")
    return df_history


def prepare_train_validation_split(df, validation_days=30):
    """
    Separa dados em treino e validação temporal.
    
    Args:
        df: DataFrame com todos os dados
        validation_days: Número de dias recentes para validação
        
    Returns:
        tuple: (df_train, df_validation)
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 PREPARANDO SPLIT TREINO/VALIDAÇÃO")
    logger.info("=" * 80)
    
    cutoff_date = datetime.now() - timedelta(days=validation_days)
    
    df_train = df[df['date'] < cutoff_date].copy()
    df_validation = df[df['date'] >= cutoff_date].copy()
    
    logger.info(f"📅 Data de corte: {cutoff_date.date()}")
    logger.info(f"🔹 Treino: {len(df_train)} jogos (até {df_train['date'].max().date()})")
    logger.info(f"🔹 Validação: {len(df_validation)} jogos ({validation_days} dias recentes)")
    
    return df_train, df_validation


def train_clean_model(df_train):
    """
    Treina modelo com dados limpos usando infraestrutura existente.
    
    Args:
        df_train: DataFrame de treino
        
    Returns:
        tuple: (model, feature_names)
    """
    logger.info("\n" + "=" * 80)
    logger.info("🚀 TREINANDO MODELO COM DADOS LIMPOS")
    logger.info("=" * 80)
    
    # Usar o módulo de treinamento existente que já lida com feature engineering
    from ml_pipeline.train_ensemble_v3 import train_ensemble_v3
    
    logger.info("⚙️  Aplicando pipeline completo de treinamento...")
    
    # Treinar usando pipeline existente mas apenas com dados da temporada atual
    # Salvar temporariamente os dados filtrados
    temp_db_path = 'data/temp_clean_history.csv'
    df_train.to_csv(temp_db_path, index=False)
    
    logger.info(f"📊 Dados para treino: {len(df_train)} jogos")
    
    # Treinar modelo
    try:
        # Importar função de treino
        from ml_pipeline.data_preparation import load_historical_data
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Preparar features usando infraestrutura existente
        df_prepared, weights = load_historical_data(
            seasons=['2025-26'],
            apply_weights=True
        )
        
        # Filtrar apenas jogos válidos
        df_prepared = df_prepared[(df_prepared['fga'] > 0) & (df_prepared.get('opp_fga', pd.Series([1])).fillna(1) > 0)]
        
        logger.info(f"✅ Features preparadas: {df_prepared.shape}")
        
        # Remover features com data leakage
        leaking_cols = ['winner', 'correct', 'home_score', 'away_score', 'pt_diff', 
                       'fgm', 'fga', 'fg3m', 'pts', 'opp_fgm', 'opp_fga', 'opp_pts']
        
        # Preparar X e y
        y = (df_prepared['winner'] == 'HOME').astype(int) if 'winner' in df_prepared.columns else df_prepared.get('home_win', (df_prepared['home_score'] > df_prepared['away_score']).astype(int))
        X = df_prepared.drop(columns=leaking_cols + ['home_team', 'away_team', 'date'], errors='ignore')
        X = X.select_dtypes(include=['number']).fillna(0)
        
        feature_names = list(X.columns)
        
        logger.info(f"📊 Features finais: {len(feature_names)}")
        logger.info(f"📊 Samples: {len(X)}")
        
        # Treinar modelo
        logger.info("🌲 Treinando ensemble...")
        
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=15,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ]
        
        meta = LogisticRegression(max_iter=1000, random_state=42)
        
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=3,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        train_acc = model.score(X, y)
        logger.info(f"✅ Modelo treinado! Accuracy no treino: {train_acc:.2%}")
        
        return model, feature_names
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model(model, feature_names, model_name="Model"):
    """
    Avalia modelo em conjunto de validação (últimos 30 dias).
    
    Args:
        model: Modelo treinado
        feature_names: Lista de nomes de features
        model_name: Nome do modelo para logs
        
    Returns:
        dict: Métricas de performance
    """
    logger.info(f"\n📏 Avaliando {model_name}...")
    
    try:
        # Carregar dados recentes
        db = get_db_manager()
        df_all = db.get_history()
        df_all['date'] = pd.to_datetime(df_all['date'])
        
        # Filtrar últimos 30 dias
        cutoff = datetime.now() - timedelta(days=VALIDATION_DAYS)
        df_val = df_all[df_all['date'] >= cutoff].copy()
        
        # Aplicar mesma limpeza
        df_val = df_val[(df_val['fga'] > 0) & (df_val.get('opp_fga', pd.Series([1])).fillna(1) > 0)]
        
        logger.info(f"   Validação: {len(df_val)} jogos (últimos {VALIDATION_DAYS} dias)")
        
        if len(df_val) == 0:
            logger.warning("   ⚠️  Sem dados de validação suficientes")
            return None
        
        # Preparar features (usar load_historical_data com date filter)
        from ml_pipeline.data_preparation import load_historical_data
        
        df_prepared, _ = load_historical_data(seasons=['2025-26'], apply_weights=False)
        df_prepared['date'] = pd.to_datetime(df_prepared['date'])
        df_prepared = df_prepared[df_prepared['date'] >= cutoff]
        
        # Remover leaking features
        leaking_cols = ['winner', 'correct', 'home_score', 'away_score', 'pt_diff',
                       'fgm', 'fga', 'fg3m', 'pts', 'opp_fgm', 'opp_fga', 'opp_pts']
        
        y_val = (df_prepared['winner'] == 'HOME').astype(int) if 'winner' in df_prepared.columns else df_prepared.get('home_win', (df_prepared['home_score'] > df_prepared['away_score']).astype(int))
        X_val = df_prepared.drop(columns=leaking_cols + ['home_team', 'away_team', 'date'], errors='ignore')
        X_val = X_val.select_dtypes(include=['number']).fillna(0)
        
        # Alinhar features
        missing_features = set(feature_names) - set(X_val.columns)
        if missing_features:
            for feat in missing_features:
                X_val[feat] = 0
        
        X_val = X_val[feature_names]
        
        # Fazer predições
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calcular métricas
        acc = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred_proba)
        
        logger.info(f"   Accuracy: {acc:.2%}")
        logger.info(f"   Log Loss: {logloss:.4f}")
        logger.info(f"   Jogos avaliados: {len(y_val)}")
        
        return {
            'accuracy': acc,
            'log_loss': logloss,
            'n_samples': len(y_val)
        }
        
    except Exception as e:
        logger.error(f"   ❌ Erro na avaliação: {e}")
        import traceback
        traceback.print_exc()
        return None



def compare_models(old_metrics, new_metrics):
    """
    Compara performance de modelo antigo vs novo.
    
    Args:
        old_metrics: Métricas do modelo antigo
        new_metrics: Métricas do modelo novo
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARAÇÃO: MODELO ANTIGO vs MODELO NOVO LIMPO")
    logger.info("=" * 80)
    
    # Tabela comparativa
    logger.info(f"\n{'Métrica':<20} {'Modelo Antigo':<20} {'Modelo Novo':<20} {'Melhoria':<15}")
    logger.info("-" * 75)
    
    # Accuracy
    old_acc = old_metrics['accuracy']
    new_acc = new_metrics['accuracy']
    acc_diff = new_acc - old_acc
    acc_symbol = "✅" if acc_diff > 0 else "❌" if acc_diff < 0 else "="
    logger.info(f"{'Accuracy':<20} {old_acc:>18.2%} {new_acc:>18.2%} {acc_symbol} {acc_diff:>+12.2%}")
    
    # Log Loss (menor é melhor)
    old_loss = old_metrics['log_loss']
    new_loss = new_metrics['log_loss']
    loss_diff = new_loss - old_loss
    loss_symbol = "✅" if loss_diff < 0 else "❌" if loss_diff > 0 else "="
    logger.info(f"{'Log Loss':<20} {old_loss:>20.4f} {new_loss:>18.4f} {loss_symbol} {loss_diff:>+12.4f}")
    
    logger.info("=" * 80)
    
    # Conclusão
    if acc_diff > 0 and loss_diff < 0:
        logger.info("\n🎉 SUCESSO! Modelo novo é SUPERIOR em ambas as métricas!")
    elif acc_diff > 0 or loss_diff < 0:
        logger.info("\n✅ Modelo novo melhorou em pelo menos uma métrica")
    else:
        logger.info("\n⚠️  Modelo novo não apresentou melhoria clara")


def main():
    """
    Função principal do script de re-treino.
    """
    logger.info("\n" + "=" * 80)
    logger.info("🔄 SCRIPT DE RE-TREINO COM DADOS LIMPOS")
    logger.info("=" * 80)
    
    try:
        # 1. Carregar e limpar dados
        df_clean = load_and_clean_data()
        
        # 2. Separar treino/validação
        df_train, df_val = prepare_train_validation_split(df_clean, VALIDATION_DAYS)
        
        # 3. Treinar modelo novo
        new_model, X_train, y_train, feature_names = train_clean_model(df_train)
        
        # 4. Salvar modelo novo
        logger.info(f"\n💾 Salvando modelo limpo em {NEW_MODEL_PATH}...")
        joblib.dump(new_model, NEW_MODEL_PATH)
        joblib.dump(feature_names, NEW_MODEL_PATH.replace('.joblib', '_features.joblib'))
        logger.info("✅ Modelo salvo com sucesso!")
        
        # 5. Avaliar modelo novo
        new_metrics = evaluate_model(new_model, df_val, feature_names, "Modelo Novo (Limpo)")
        
        # 6. Carregar e avaliar modelo antigo
        if Path(OLD_MODEL_PATH).exists():
            logger.info(f"\n📂 Carregando modelo antigo de {OLD_MODEL_PATH}...")
            old_model = joblib.load(OLD_MODEL_PATH)
            
            # Tentar carregar feature names do modelo antigo
            if hasattr(old_model, 'feature_names_in_'):
                old_features = list(old_model.feature_names_in_)
            else:
                old_features = feature_names  # Usar as mesmas se não tiver
            
            old_metrics = evaluate_model(old_model, df_val, old_features, "Modelo Antigo")
            
            # 7. Comparar modelos
            if old_metrics and new_metrics:
                compare_models(old_metrics, new_metrics)
        else:
            logger.warning(f"⚠️  Modelo antigo não encontrado em {OLD_MODEL_PATH}")
            logger.info("   Pulando comparação...")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ RE-TREINO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 80)
        logger.info(f"\nModelo limpo salvo em: {NEW_MODEL_PATH}")
        logger.info(f"Para usar o novo modelo, atualize o caminho em predict.py")
        
    except Exception as e:
        logger.error(f"\n❌ ERRO durante re-treino: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
