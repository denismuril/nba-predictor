#!/usr/bin/env python3
"""
Calibração de Totals - Isotonic Regression

Aplica Isotonic Regression para o modelo de Over/Under (Total Points).
Objetivo: Melhorar o MAE das previsões de pontuação total.
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_calibrate_totals():
    logger.info("="*80)
    logger.info("🔢 CALIBRAÇÃO DE TOTALS (ISOTONIC)")
    logger.info("="*80)
    
    # 1. Carregar dados
    from ml_pipeline.data_preparation import load_historical_data
    df = load_historical_data(seasons=['2023-24', '2024-25'], apply_weights=False)
    
    # Target: Total Points
    df['target_total'] = df['home_score'] + df['away_score']
    
    # Features (mesmas do Moneyline por enquanto)
    features_file = Path('data/models/feature_names_final.joblib')
    if features_file.exists():
        features = joblib.load(features_file)
    else:
        logger.error("Features não encontradas.")
        return

    # Preparar X, y
    X = df.copy()
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    y = df['target_total']
    
    # Split Train/Calib/Test
    n = len(df)
    idx_train = int(n * 0.6)
    idx_calib = int(n * 0.8)
    
    X_train, y_train = X.iloc[:idx_train], y.iloc[:idx_train]
    X_calib, y_calib = X.iloc[idx_train:idx_calib], y.iloc[idx_train:idx_calib]
    X_test, y_test = X.iloc[idx_calib:], y.iloc[idx_calib:]
    
    # 2. Treinar Modelo Base (XGBoost Regressor)
    from xgboost import XGBRegressor
    logger.info("🔧 Treinando modelo base (XGBoost)...")
    base_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
    base_model.fit(X_train, y_train)
    
    # Previsões no set de calibração
    pred_calib = base_model.predict(X_calib)
    
    # 3. Treinar Calibrador (Isotonic Regression)
    logger.info("🔧 Ajustando Isotonic Regression...")
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(pred_calib, y_calib)
    
    # 4. Avaliar no Test Set
    pred_test_base = base_model.predict(X_test)
    pred_test_calib = iso_reg.predict(pred_test_base)
    
    mae_base = mean_absolute_error(y_test, pred_test_base)
    mae_calib = mean_absolute_error(y_test, pred_test_calib)
    
    logger.info("\n📊 RESULTADOS (TEST SET):")
    logger.info(f"   MAE Base:      {mae_base:.4f}")
    logger.info(f"   MAE Calibrado: {mae_calib:.4f}")
    
    diff = mae_base - mae_calib
    if diff > 0:
        logger.info(f"   ✅ Melhoria: -{diff:.4f} MAE")
        
        # Salvar modelos
        joblib.dump(base_model, 'data/models/totals_model_base.joblib')
        joblib.dump(iso_reg, 'data/models/totals_calibrator.joblib')
        logger.info("\n💾 Modelos salvos: totals_model_base.joblib, totals_calibrator.joblib")
    else:
        logger.info(f"   ⚠️  Sem melhoria (piora de +{abs(diff):.4f})")
        
    return {'mae_base': mae_base, 'mae_calib': mae_calib}

if __name__ == "__main__":
    train_and_calibrate_totals()
