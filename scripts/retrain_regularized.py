#!/usr/bin/env python3
"""
Retreinamento com Regularização Forte
Objetivo: Reduzir overfitting (100% -> ~75-80% test accuracy)
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, log_loss

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

# Converter categoricals e remover NaN
X = X.select_dtypes(include=['number'])
X = X.fillna(0)

logger.info(f"✅ Features: {X.shape[1]}")
logger.info(f"✅ Amostras: {X.shape[0]}")

# Split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=y
)

logger.info(f"\n{'='*60}")
logger.info("🛡️ TREINAMENTO COM REGULARIZAÇÃO FORTE")
logger.info(f"{'='*60}\n")

# Hiperparâmetros REGULARIZADOS (anti-overfitting)
logger.info("📐 Hiperparâmetros:")
logger.info("   RandomForest:")
logger.info("      n_estimators: 100 -> 50 (menos árvores)")
logger.info("      max_depth: 8 -> 6 (árvores mais rasas)")
logger.info("      min_samples_split: 2 -> 20 (evita splits muito específicos)")
logger.info("      min_samples_leaf: 1 -> 10 (folhas mais gerais)")
logger.info("   GradientBoosting:")
logger.info("      n_estimators: 50 -> 30")
logger.info("      max_depth: 5 -> 3")
logger.info("      learning_rate: 0.1 (padrão)")
logger.info("      subsample: 0.8 (80% dos dados por árvore)\n")

estimators = [
    ('rf', RandomForestClassifier(
        n_estimators=50,        # Reduzido de 100
        max_depth=6,            # Reduzido de 8
        min_samples_split=20,   # Aumentado de 2
        min_samples_leaf=10,    # Aumentado de 1
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=30,        # Reduzido de 50
        max_depth=3,            # Reduzido de 5
        learning_rate=0.1,
        subsample=0.8,          # Novo: apenas 80% dos dados
        random_state=42
    ))
]

meta = LogisticRegression(
    max_iter=1000,
    C=1.0,                      # Regularização padrão
    random_state=42
)

model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    cv=5,                       # Aumentado de 3 para 5-fold CV
    n_jobs=-1
)

logger.info("🤖 Treinando modelo regularizado...")
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

y_proba_train = model.predict_proba(X_train)
y_proba_test = model.predict_proba(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

train_logloss = log_loss(y_train, y_proba_train, sample_weight=w_train)
test_logloss = log_loss(y_test, y_proba_test, sample_weight=w_test)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

logger.info(f"\n{'='*60}")
logger.info(f"📊 RESULTADOS:")
logger.info(f"   Train Accuracy: {train_acc:.4f}")
logger.info(f"   Test Accuracy: {test_acc:.4f}")
logger.info(f"   Train Log Loss: {train_logloss:.4f}")
logger.info(f"   Test Log Loss: {test_logloss:.4f}")
logger.info(f"   CV Accuracy (média): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
logger.info(f"   Overfitting Gap: {train_acc - test_acc:.4f}")
logger.info(f"{'='*60}\n")

# Interpretação
gap = train_acc - test_acc
if gap < 0.05:
    logger.info("✅ Overfitting controlado! (gap < 5%)")
elif gap < 0.10:
    logger.info("⚠️ Leve overfitting (gap 5-10%)")
else:
    logger.warning("❌ Overfitting significativo (gap > 10%)")

# Report detalhado
logger.info("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=['Away Win', 'Home Win']))

# Salvar
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
model_path = f'models/ensemble_v7_regularized_{timestamp}.joblib'
features_path = f'models/feature_names_v7_regularized_{timestamp}.joblib'

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)

# Salvar como default
joblib.dump(model, 'models/ensemble_v7.joblib')
joblib.dump(list(X.columns), 'models/feature_names_v7.joblib')

logger.info(f"\n💾 Modelos salvos:")
logger.info(f"   - {model_path}")
logger.info(f"   - {features_path}")
logger.info(f"   - models/ensemble_v7.joblib (default)")
logger.info(f"\n✅ Retreinamento regularizado concluído!")
logger.info(f"   Test Accuracy: {test_acc:.4f}")
logger.info(f"   Overfitting Gap: {gap:.4f}")
