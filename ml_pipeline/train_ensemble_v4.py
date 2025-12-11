import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
import numpy as np

# Adicionar raiz ao path (garantir que funciona independente de onde é chamado)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports do projeto (SÓ DEPOIS DE AJUSTAR O PATH)
from ml_pipeline.data_preparation import load_historical_data

logger = logging.getLogger(__name__)

# Configuração
ML_SEASONS = ['2023-24', '2024-25', '2025-26']
IMPORTANCE_THRESHOLD = 0.0005  # Remover features com menos de 0.05% de importância
ML_SAMPLE_WEIGHT_CONFIG = {
    'enabled': True,
    'recent_30_days': 3.0,
    'recent_60_days': 2.0,
    'recent_90_days': 1.5,
    'default': 1.0
}

def load_features_to_drop():
    """Lê o CSV de importância e retorna lista de features para remover."""
    csv_path = Path('results/feature_importance_permutation.csv')
    if not csv_path.exists():
        logger.error("❌ CSV de importância não encontrado. Rode scripts/analyze_feature_importance.py primeiro.")
        return []
        
    df_imp = pd.read_csv(csv_path)
    
    # Filtrar features ruins
    bad_features = df_imp[df_imp['importance_mean'] < IMPORTANCE_THRESHOLD]['feature'].tolist()
    
    logger.info(f"📉 Features para remover (Threshold < {IMPORTANCE_THRESHOLD}): {len(bad_features)}")
    return bad_features

def train_ensemble_model_v4():
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full_ensemble = True
    except ImportError:
        use_full_ensemble = False
    
    logger.info("="*80)
    logger.info("🚀 TREINANDO ENSEMBLE MODEL V4 (OPTIMIZED FEATURES)")
    logger.info("="*80)
    
    # 1. Identificar features ruins
    features_to_drop = load_features_to_drop()
    if not features_to_drop:
        logger.warning("⚠️ Nenhuma feature para remover. O modelo será idêntico ao V3.")
    
    # 2. Carregar dados
    df, sample_weights = load_historical_data(
        seasons=ML_SEASONS, 
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 3. Pré-processamento Base (Igual V3)
    base_drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=base_drop_cols, errors='ignore')
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # 4. REMOVER FEATURES RUINS
    # Precisamos ter cuidado pois o CSV pode ter nomes de features que não existem no X atual
    # (ex: times que não jogaram nesta seleção de dados)
    existing_bad_features = [f for f in features_to_drop if f in X.columns]
    X = X.drop(columns=existing_bad_features)
    
    logger.info(f"✂️  Features removidas do dataset: {len(existing_bad_features)}")
    logger.info(f"✅ Features finais: {len(X.columns)}")
    
    # Salvar features finais
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'data/models/feature_names_v4.joblib')
    # Também sobrescrever o final para ser usado em produção se aprovado
    # joblib.dump(feature_names, 'data/models/feature_names_final.joblib') 
    
    y = (df['winner'] == 'HOME').astype(int)
    
    # 5. Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    weights_train = sample_weights[:split_idx]
    weights_test = sample_weights[split_idx:]
    
    # 6. Treinamento
    # Random Forest Otimizado
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    base_estimators = [
        ('rf', rf),
        ('extra', ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    ]
    
    if use_full_ensemble:
        base_estimators.append(('xgb', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)))
        base_estimators.append(('lgbm', LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)))
    
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    
    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )
    
    logger.info(f"🔄 Treinando Stack ({len(base_estimators)} modelos)...")
    ensemble.fit(X_train, y_train, sample_weight=weights_train)
    
    # 7. Avaliação
    acc = ensemble.score(X_test, y_test, sample_weight=weights_test) # Acurácia ponderada
    raw_acc = ensemble.score(X_test, y_test) # Acurácia simples
    
    logger.info(f"🏆 Acurácia V4 (Optimized): {raw_acc*100:.2f}% (Weighted: {acc*100:.2f}%)")
    
    # Comparar com V3 (se existir metadados)
    v3_meta_path = Path('data/models/training_metadata.json')
    if v3_meta_path.exists():
        with open(v3_meta_path) as f:
            v3_data = json.load(f)
            v3_acc = v3_data.get('accuracy', 0)
            diff = acc - v3_acc
            logger.info(f"🆚 Comparação V3: {v3_acc*100:.2f}% -> V4: {acc*100:.2f}% (Diff: {diff*100:+.2f}pp)")
            
    # Salvar modelo V4
    joblib.dump(ensemble, 'data/models/ensemble_model_v4.joblib')
    logger.info("💾 Modelo V4 salvo em data/models/ensemble_model_v4.joblib")
    
    return ensemble, raw_acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    train_ensemble_model_v4()
