#!/usr/bin/env python3
"""
Model Calibration Script - Platt Scaling para melhorar underdogs
Objetivo: Cleveland e Denver têm baixa accuracy como underdogs (34-50%)
Solução: Calibrar probabilidades com Isotonic Regression ou Platt Scaling
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import joblib
import numpy as np
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("🎯 MODEL CALIBRATION - Platt Scaling")
logger.info("="*80)

# Carregar dados
logger.info("\n📊 Carregando dados...")
df, weights = load_historical_data(
    seasons=ML_SEASONS,
    apply_weights=True,
    weight_config=ML_SAMPLE_WEIGHT_CONFIG
)

logger.info(f"Total: {len(df)} jogos")

# Preparar dados
drop_cols = ['winner', 'correct', 'date', 'prediction', 
             'home_score', 'away_score', 'pt_diff', 'total_points',
             'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
             'ast', 'stl', 'blk', 'pf', 'pts',
             'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 
             'opp_fta', 'opp_ftm', 'opp_ast', 'opp_stl', 'opp_blk', 'opp_pf', 'opp_pts',
             'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
             'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
             'prob_home', 'prob_away', 'home_team', 'away_team',
             'predicted_spread', 'predicted_total', 'ci_lower', 'ci_upper',
             'model_version', 'created_at', 'odds_home', 'odds_away', 'total_line', 'odds_source',
             'odd_home', 'odd_away', 'confidence']

X = df.drop(columns=drop_cols, errors='ignore')
y = (df['winner'] == 'HOME').astype(int)

# Apenas numéricas
X = X.select_dtypes(include=['number'])
X = X.fillna(0)

logger.info(f"✅ Features: {X.shape[1]}, Amostras: {X.shape[0]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Carregar modelo base
logger.info("\n📦 Carregando modelo base...")
model_path = 'models/ensemble_v7.joblib'  # Changed from _clean
base_model = joblib.load(model_path)

# Avaliar modelo base (sem calibração)
logger.info("\n📊 Performance ANTES da calibração:")
y_proba_base = base_model.predict_proba(X_test)[:, 1]
y_pred_base = base_model.predict(X_test)

acc_base = accuracy_score(y_test, y_pred_base)
brier_base = brier_score_loss(y_test, y_proba_base)
logloss_base = log_loss(y_test, y_proba_base)

logger.info(f"   Accuracy: {acc_base:.4f}")
logger.info(f"   Brier Score: {brier_base:.4f} (menor é melhor)")
logger.info(f"   Log Loss: {logloss_base:.4f}")

# Calibrar modelo com Platt Scaling (sigmoid)
logger.info("\n🎯 Calibrando modelo com Platt Scaling...")
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',  # Platt Scaling
    cv='prefit'  # Modelo já treinado
)

calibrated_model.fit(X_train, y_train)

# Avaliar modelo calibrado
logger.info("\n📊 Performance DEPOIS da calibração:")
y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
y_pred_cal = calibrated_model.predict(X_test)

acc_cal = accuracy_score(y_test, y_pred_cal)
brier_cal = brier_score_loss(y_test, y_proba_cal)
logloss_cal = log_loss(y_test, y_proba_cal)

logger.info(f"   Accuracy: {acc_cal:.4f}")
logger.info(f"   Brier Score: {brier_cal:.4f} (menor é melhor)")
logger.info(f"   Log Loss: {logloss_cal:.4f}")

# Comparação
logger.info("\n" + "="*80)
logger.info("📈 COMPARAÇÃO:")
logger.info("="*80)
logger.info(f"   Accuracy:     {acc_base:.4f} → {acc_cal:.4f} ({acc_cal - acc_base:+.4f})")
logger.info(f"   Brier Score:  {brier_base:.4f} → {brier_cal:.4f} ({brier_cal - brier_base:+.4f})")
logger.info(f"   Log Loss:     {logloss_base:.4f} → {logloss_cal:.4f} ({logloss_cal - logloss_base:+.4f})")

# Análise de Underdogs (probabilidade < 0.5)
logger.info("\n" + "="*80)
logger.info("🎯 ANÁLISE DE UNDERDOGS:")
logger.info("="*80)

underdog_mask = y_proba_base < 0.5
n_underdogs = underdog_mask.sum()

if n_underdogs > 0:
    acc_underdog_base = accuracy_score(y_test[underdog_mask], y_pred_base[underdog_mask])
    acc_underdog_cal = accuracy_score(y_test[underdog_mask], y_pred_cal[underdog_mask])
    
    logger.info(f"   Jogos como underdog: {n_underdogs}/{len(y_test)} ({n_underdogs/len(y_test)*100:.1f}%)")
    logger.info(f"   Accuracy (base): {acc_underdog_base:.4f}")
    logger.info(f"   Accuracy (calibrado): {acc_underdog_cal:.4f}")
    logger.info(f"   Melhoria: {acc_underdog_cal - acc_underdog_base:+.4f}")

# Salvar modelo calibrado
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
calibrated_path = f'models/ensemble_v7_calibrated_{timestamp}.joblib'
joblib.dump(calibrated_model, calibrated_path)

# Salvar como default
joblib.dump(calibrated_model, 'models/ensemble_v7_calibrated.joblib')

logger.info(f"\n💾 Modelo calibrado salvo:")
logger.info(f"   - {calibrated_path}")
logger.info(f"   - models/ensemble_v7_calibrated.joblib (default)")

# Decisão final
if brier_cal < brier_base and logloss_cal < logloss_base:
    logger.info("\n✅ SUCESSO! Calibração melhorou probabilidades.")
    logger.info("   Recomendação: Usar modelo calibrado em produção.")
else:
    logger.warning("\n⚠️  Calibração não trouxe melhoria significativa.")
    logger.warning("   Recomendação: Manter modelo base.")

logger.info("\n✅ Calibração concluída!")
