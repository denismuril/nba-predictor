import pandas as pd
import logging
from datetime import datetime, timedelta
from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.train_model import train_and_save_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# FASE 3 FIX: Importar parâmetros centralizados (Single Source of Truth)
from config.model_config import RF_PARAMS

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(test_days=14):
    """
    Executa um backtest simples:
    - Treina com dados anteriores aos últimos 'test_days'.
    - Testa nos últimos 'test_days'.
    """
    logger.info("🔄 Iniciando Backtest...")
    
    # Carregar dados
    df = load_historical_data()
    
    if df.empty:
        logger.error("❌ Nenhum dado encontrado para backtest.")
        return
    
    # Converter data se necessário
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    # Definir data de corte
    max_date = df['date'].max()
    cutoff_date = max_date - timedelta(days=test_days)
    
    logger.info(f"📅 Período de Teste: {cutoff_date.date()} a {max_date.date()}")
    
    # Split temporal
    train_df = df[df['date'] < cutoff_date]
    test_df = df[df['date'] >= cutoff_date]
    
    logger.info(f"📊 Treino: {len(train_df)} jogos | Teste: {len(test_df)} jogos")
    
    if len(train_df) < 50:
        logger.warning("⚠️  Poucos dados de treino. Resultados podem não ser confiáveis.")
        
    if len(test_df) == 0:
        logger.error("❌ Sem dados de teste. Ajuste o período ou popule mais dados.")
        return

    # Preparar features (mesma lógica do train_model)
    # 2.1 APLICAR FEATURE ENGINEERING V2 (CRÍTICO)
    from ml_pipeline.feature_engineering_v2 import prepare_features_v2
    logger.info("🛠️ Aplicando Feature Engineering V2 no Backtest...")
    df = prepare_features_v2(df)
    
    # Recalcular data de corte após feature engineering (pode ter dropado linhas)
    train_df = df[df['date'] < cutoff_date]
    test_df = df[df['date'] >= cutoff_date]
    
    # Remover colunas de resultado (leakage) e identificadores
    drop_cols = ['winner', 'correct', 'date', 'prediction', 'home_score', 'away_score', 'pt_diff', 
                 'total_points', 'home_team', 'away_team', 'prob_home', 'prob_away']
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[float, int])
    y_train = train_df['winner'].apply(lambda x: 1 if x == 'HOME' else 0)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Target distribution:\n{y_train.value_counts()}")

    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['winner'].apply(lambda x: 1 if x == 'HOME' else 0)
    
    # Alinhar colunas (garantir que teste tenha as mesmas colunas do treino)
    # Adicionar colunas faltantes em X_test com 0
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    # Remover colunas extras em X_test
    extra_cols = set(X_test.columns) - set(X_train.columns)
    X_test = X_test.drop(columns=extra_cols)
    # Reordenar
    X_test = X_test[X_train.columns]
    
    # Treinar Modelo
    # FASE 3 FIX: Usar RF_PARAMS centralizado (idêntico ao train_model.py)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"🏆 Acurácia no Backtest: {acc:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['AWAY', 'HOME']))
    
    # Análise de Confiança
    test_df = test_df.copy()
    test_df['prob_home'] = y_prob
    test_df['pred_backtest'] = ['HOME' if p > 0.5 else 'AWAY' for p in y_prob]
    test_df['correct_backtest'] = (test_df['pred_backtest'] == test_df['winner']).astype(int)
    
    # Agrupar por confiança
    test_df['confidence_bucket'] = pd.cut(test_df['prob_home'], bins=[0, 0.4, 0.6, 1.0], labels=['Baixa (Away)', 'Neutra', 'Alta (Home)'])
    
    print("\nPerformance por Nível de Confiança:")
    print(test_df.groupby('confidence_bucket')['correct_backtest'].mean())

if __name__ == "__main__":
    run_backtest()
