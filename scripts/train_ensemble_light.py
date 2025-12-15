#!/usr/bin/env python3
"""
Script leve para treinar ensemble blending com menos memória.
Usa apenas 1 temporada para reduzir consumo de RAM.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_light():
    """Treina ensemble com footprint de memória reduzido."""
    logger.info("🧪 Ensemble Blending LIGHT (baixa memória)")
    
    # 1. Carregar apenas temporada atual (menos dados)
    from ml_pipeline.data_preparation import load_historical_data
    
    logger.info("📦 Carregando apenas temporada 2024-25 e 2025-26...")
    df = load_historical_data(seasons=['2024-25', '2025-26'], apply_weights=False)
    
    # Forçar limpeza de memória
    gc.collect()
    
    logger.info(f"   📊 {len(df)} jogos carregados")
    
    # 2. Selecionar features numéricas simples (sem rolling complexo)
    try:
        features = joblib.load('data/models/feature_names_final.joblib')
    except Exception:
        logger.info("   Usando features numéricas básicas...")
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclusions = ['home_score', 'away_score', 'total_points',
                      'winner', 'spread', 'actual_spread', 'game_id']
        features = [f for f in features if f not in exclusions]
    
    # Filtrar apenas features que existem
    features = [f for f in features if f in df.columns]
    
    X = df[features].fillna(0)
    y = df['winner']
    
    logger.info(f"   📊 {len(X)} samples, {len(features)} features")
    
    # Liberar df original
    del df
    gc.collect()
    
    # 3. Modelos com configuração leve
    models = {
        'rf': RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=1),
        'xgb': xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1, n_jobs=1)
    }
    
    # 4. OOF predictions
    logger.info("🔄 Gerando OOF predictions...")
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds = menos memória
    oof_preds = pd.DataFrame(index=X.index)
    
    for name, model in models.items():
        logger.info(f"   Treinando {name}...")
        oof_col = f'pred_{name}'
        oof_preds[oof_col] = 0.0
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            model.fit(X_train, y_train)
            oof_preds.loc[val_idx, oof_col] = model.predict_proba(X_val)[:, 1]
        
        gc.collect()
    
    # 5. Meta-learner
    logger.info("🧠 Treinando Meta-Learner...")
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(oof_preds, y)
    
    # Métricas
    meta_preds = meta_model.predict_proba(oof_preds)[:, 1]
    acc = accuracy_score(y, (meta_preds >= 0.5).astype(int))
    brier = brier_score_loss(y, meta_preds)
    
    logger.info(f"📊 Accuracy: {acc:.2%}, Brier: {brier:.4f}")
    
    # 6. Calibração
    logger.info("🎯 Calibração isotônica...")
    calibrated = CalibratedClassifierCV(estimator=meta_model, method='isotonic', cv='prefit')
    calibrated.fit(oof_preds, y)
    
    cal_preds = calibrated.predict_proba(oof_preds)[:, 1]
    acc_cal = accuracy_score(y, (cal_preds >= 0.5).astype(int))
    brier_cal = brier_score_loss(y, cal_preds)
    
    logger.info(f"📊 Calibrated: Acc={acc_cal:.2%}, Brier={brier_cal:.4f}")
    
    # 7. Treinar modelos finais e salvar
    logger.info("💾 Salvando modelos...")
    final_models = {}
    for name, model in models.items():
        model.fit(X, y)
        final_models[name] = model
    
    joblib.dump(final_models, 'data/models/blending_base_models.joblib')
    joblib.dump(calibrated, 'data/models/blending_meta_model.joblib')
    joblib.dump(calibrated, 'data/models/blending_calibrated_model.joblib')
    
    logger.info("✅ Ensemble Blending LIGHT concluído!")
    logger.info(f"   Final Accuracy: {acc_cal:.2%}")
    
    return {'accuracy': acc_cal, 'brier': brier_cal}

if __name__ == "__main__":
    train_light()
