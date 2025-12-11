"""
Ensemble Model V2 - Otimizado para predições de jogos futuros

Remove features de box scores brutos que não existem para jogos futuros.
Inclui otimização de hiperparâmetros e feature importance.

Usa stacking de:
- Random Forest (otimizado)
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
from pathlib import Path

logger = logging.getLogger(__name__)

def train_ensemble_model():
    """Treina ensemble com stacking - V2 otimizado"""
    
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full_ensemble = True
    except ImportError:
        logger.warning("⚠️  XGBoost/LightGBM não disponíveis. Usando RF+ExtraTrees apenas.")
        use_full_ensemble = False
    
    logger.info("🎯 Treinando Ensemble Model V2 (Otimizado)...")
    
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
    
    # Análise de features
    rolling_features = [f for f in feature_names if 'rolling' in f]
    roster_features = [f for f in feature_names if 'roster' in f]
    team_dummies = [f for f in feature_names if 'team_' in f]
    odds_features = [f for f in feature_names if 'odds' in f]
    
    logger.info(f"📋 Features selecionadas: {len(feature_names)}")
    logger.info(f"   Rolling features: {len(rolling_features)}")
    logger.info(f"   Roster features: {len(roster_features)}")
    logger.info(f"   Odds features: {len(odds_features)}")
    logger.info(f"   Team dummies: {len(team_dummies)}")
    
    y = (df['winner'] == 'HOME').astype(int)
    
    # Time Series Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # AJUSTE 4: Parâmetros fixos (Grid Search removido para evitar timeout/lock)
    logger.info("🔧 Usando hiperparâmetros fixos para Random Forest (Grid Search desativado)...")
    best_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    
    logger.info(f"✅ Random Forest treinado")
    
    # Base models com RF otimizado
    if use_full_ensemble:
        base_estimators = [
            ('rf', best_rf),
            ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)),
            ('lgbm', LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)),
            ('extra', ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1))
        ]
    else:
        base_estimators = [
            ('rf', best_rf),
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
    
    logger.info(f"🔄 Treinando {len(base_estimators)} modelos base + meta-model...")
    ensemble.fit(X_train, y_train)
    
    # Avaliar
    accuracy = ensemble.score(X_test, y_test)
    
    logger.info(f"✅ Ensemble V2 treinado")
    logger.info(f"   Acurácia no teste: {accuracy*100:.2f}%")
    
    # AJUSTE 5: Feature Importance
    logger.info("📊 Calculando feature importance...")
    
    # Usar o Random Forest do ensemble para feature importance
    rf_importance = best_rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importance
    }).sort_values('importance', ascending=False)
    
    # Salvar feature importance
    feature_importance_df.to_csv('data/models/feature_importance.csv', index=False)
    
    # Log top 10 features
    logger.info("🏆 Top 10 features mais importantes:")
    for idx, row in feature_importance_df.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Salvar modelos
    joblib.dump(ensemble, 'data/models/ensemble_model.joblib')
    joblib.dump(ensemble, 'data/models/ml_model.joblib')
    logger.info("💾 Modelo salvo: data/models/ensemble_model.joblib")
    logger.info("💾 Modelo salvo: data/models/ml_model.joblib")
    
    return ensemble, accuracy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, acc = train_ensemble_model()
    print(f"\n✅ Ensemble V2 Otimizado pronto! Acurácia: {acc*100:.1f}%\n")
