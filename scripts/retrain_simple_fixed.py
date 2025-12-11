#!/usr/bin/env python3
"""
Retreinamento Simples após correção do rest_days
"""
import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import joblib
from datetime import datetime
from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carregar dados
logger.info("🔄 Carregando dados com rest_days corrigido...")
df, weights = load_historical_data(
    seasons=ML_SEASONS,
    apply_weights=True,
    weight_config=ML_SAMPLE_WEIGHT_CONFIG
)

logger.info(f"📊 Dataset: {len(df)} jogos")
logger.info(f"📊 Features: {len(df.columns)} colunas")

# Preparar dados
drop_cols = ['winner', 'correct', 'date', 'prediction', 
             'home_score', 'away_score', 'pt_diff', 'total_points',
             'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
             'ast', 'stl', 'blk', 'pf', 'pts',
             'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
             'opp_ast', 'opp_stl', 'opp_blk', 'opp_pf', 'opp_pts',
             'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
             'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
             'prob_home', 'prob_away', 'home_team', 'away_team',
             'predicted_spread', 'predicted_total', 'ci_lower', 'ci_upper',
             'model_version', 'created_at', 'odds_home', 'odds_away', 'total_line', 'odds_source',
             'odd_home', 'odd_away', 'confidence']

X = df.drop(columns=drop_cols, errors='ignore')
y = (df['winner'] == 'HOME').astype(int)

logger.info(f"✅ Features finais: {X.shape[1]}")
logger.info(f"✅ Amostras: {X.shape[0]}")

# Converter categoricals
X = X.select_dtypes(include=['number'])
logger.info(f"📊 Apenas numéricas: {X.shape[1]} features")

# Fill NaN
X = X.fillna(0)

# Split
logger.info("📊 Train/Test Split (80/20)...")
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=y
)

# Train modelo
logger.info("🤖 Treinando ensemble...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42))
]
meta = LogisticRegression(max_iter=1000, random_state=42)

model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    cv=3,
    n_jobs=-1
)

model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

logger.info(f"\n{'='*60}")
logger.info(f"📊 RESULTADOS:")
logger.info(f"   Train Accuracy: {train_acc:.4f}")
logger.info(f"   Test Accuracy: {test_acc:.4f}")
logger.info(f"{'='*60}\n")

# Report detalhado
logger.info("Classification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=['Away Win', 'Home Win']))

# Salvar
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
model_path = f'models/ensemble_v7_{timestamp}.joblib'
features_path = f'models/feature_names_v7_{timestamp}.joblib'

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)

# Salvar como default também
joblib.dump(model, 'models/ensemble_v7.joblib')
joblib.dump(list(X.columns), 'models/feature_names_v7.joblib')

logger.info(f"\n💾 Modelos salvos:")
logger.info(f"   - {model_path}")
logger.info(f"   - {features_path}")
logger.info(f"   - models/ensemble_v7.joblib (default)")
logger.info(f"\n✅ Retreinamento concluído! Test Accuracy: {test_acc:.4f}")
