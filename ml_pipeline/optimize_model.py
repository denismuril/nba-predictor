import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ml_pipeline.data_preparation import load_historical_data

logger = logging.getLogger(__name__)

def train_with_optimization():
    """
    Treina o modelo usando TimeSeriesSplit para validação temporal correta.
    V12 Alpha: Sem shuffle, treino no passado, teste no futuro.
    """
    logger.info("🚀 Iniciando Otimização de Modelo (V12 Alpha)...")
    
    # 1. Carregar dados
    df = load_historical_data()
    if df is None or df.empty:
        logger.error("❌ Sem dados para treinar.")
        return
    
    # Garantir ordenação temporal
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. Preparar Features e Target
    # Features numéricas geradas pelo data_preparation (rolling stats, four factors)
    feature_cols = [c for c in df.columns if 'rolling_' in c or 'rest_' in c or 'is_b2b' in c]
    
    # Se não tiver features suficientes, usar fallback ou todas numéricas
    if not feature_cols:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['home_score', 'away_score', 'total_points', 'pt_diff', 'winner', 'correct', 'odds_home', 'odds_away']
        feature_cols = [c for c in feature_cols if c not in exclude]
    
    X = df[feature_cols]
    y = (df['winner'] == 'HOME').astype(int)
    
    logger.info(f"📊 Features: {len(feature_cols)} | Amostras: {len(df)}")
    
    # 3. Time Series Split
    # 5 Splits. Em cada split, o índice de teste é sempre maior que o de treino.
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    log_losses = []
    
    # Modelos Base
    models = {
        'xgb': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
        'lgbm': LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        'rf': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    }
    
    # Loop de Validação
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Treinar modelos base
        fold_probs = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            fold_probs[name] = model.predict_proba(X_test)[:, 1]
            
        # Meta-Model (Média simples ou Regressão Logística treinada no fold anterior - aqui média simples para robustez)
        # V12.1: Média simples dos 3 modelos
        ensemble_prob = (fold_probs['xgb'] + fold_probs['lgbm'] + fold_probs['rf']) / 3
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, ensemble_pred)
        ll = log_loss(y_test, ensemble_prob)
        
        accuracies.append(acc)
        log_losses.append(ll)
        
        logger.info(f"   🔄 Fold {fold}: Acc={acc:.1%} | LogLoss={ll:.4f} | Train Size={len(X_train)} | Test Size={len(X_test)}")
        fold += 1
        
    avg_acc = np.mean(accuracies)
    logger.info(f"📈 Média TimeSeries CV: Acc={avg_acc:.1%} | LogLoss={np.mean(log_losses):.4f}")
    
    # 4. Treino Final (Full History)
    logger.info("🏁 Treinando modelo final com todo o histórico...")
    final_models = {}
    for name, model in models.items():
        model.fit(X, y)
        final_models[name] = model
        
    # Salvar modelos e feature names
    joblib.dump(final_models, 'data/models/ensemble_v12_models.joblib')
    joblib.dump(feature_cols, 'data/models/feature_names_v12.joblib')
    
    logger.info("✅ Modelo V12 Alpha salvo com sucesso.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_with_optimization()
