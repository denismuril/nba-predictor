"""
Ensemble Model V2 - Remove features de box scores brutos que não existem para jogos futuros

Usa stacking de:
- Random Forest
- XGBoost  
- LightGBM
- Extra Trees

Meta-model: Logistic Regression
"""
import joblib
import pandas as pd
import numpy as np
from ml_pipeline.data_preparation import load_historical_data
import logging

logger = logging.getLogger(__name__)

def train_ensemble_model():
    """Treina ensemble com stacking - V2 sem box scores brutos"""
    
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full_ensemble = True
    except ImportError:
        logger.warning("⚠️  XGBoost/LightGBM não disponíveis. Usando RF+ExtraTrees apenas.")
        use_full_ensemble = False
    
    logger.info("🎯 Treinando Ensemble Model V2 (sem box scores brutos)...")
    
    # Carregar dados
    df = load_historical_data()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Features e target
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 # REMOVER BOX SCORES BRUTOS (não existem para jogos futuros)
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 # REMOVER FOUR FACTORS BRUTOS (calculados a partir de box scores)
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 # REMOVER prob_home e prob_away (são do modelo antigo)
                 'prob_home', 'prob_away']
    
    # Remover colunas irrelevantes
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # One-Hot Encoding para times
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    # Salvar nomes das features para garantir alinhamento na predição
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'data/models/feature_names.joblib')
    logger.info(f"📋 Features selecionadas: {len(feature_names)}")
    logger.info(f"   Rolling features: {len([f for f in feature_names if 'rolling' in f])}")
    logger.info(f"   Roster features: {len([f for f in feature_names if 'roster' in f])}")
    logger.info(f"   Team dummies: {len([f for f in feature_names if 'team_' in f])}")
    
    y = (df['winner'] == 'HOME').astype(int)
    
    # Time Series Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Base models
    if use_full_ensemble:
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)),
            ('lgbm', LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)),
            ('extra', ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1))
        ]
    else:
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
            ('extra', ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))
        ]
    
    # Meta model
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    
    # Stacking
    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=5,
        n_jobs=-1
    )
    
    logger.info(f"   Treinando {len(base_estimators)} modelos base + meta-model...")
    ensemble.fit(X_train, y_train)
    
    # Avaliar
    accuracy = ensemble.score(X_test, y_test)
    
    logger.info(f"✅ Ensemble V2 treinado")
    logger.info(f"   Acurácia: {accuracy*100:.2f}%")
    
    # Salvar
    joblib.dump(ensemble, 'data/models/ensemble_model.joblib')
    joblib.dump(ensemble, 'data/models/ml_model.joblib')  # Salvar também como ml_model
    logger.info("💾 Modelo salvo: data/models/ensemble_model.joblib")
    logger.info("💾 Modelo salvo: data/models/ml_model.joblib")
    
    return ensemble, accuracy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, acc = train_ensemble_model()
    print(f"\n✅ Ensemble V2 pronto! Acurácia: {acc*100:.1f}%\n")
