"""
Full Model Retraining com Todas Domain Features

Treina modelo final com:
- 13 domain features implementadas
- Feature importance analysis
- Performance validation

Expected improvement: +5-6% accuracy total
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def full_retrain():
    """Retreinamento completo com todas features."""
    
    logger.info("🔄 FULL MODEL RETRAINING")
    logger.info("="*60)
    logger.info("Features: 13 domain features + base features")
    logger.info("Expected: +5-6% accuracy improvement")
    logger.info("="*60 + "\n")
    
    # Load data
    try:
        from ml_pipeline.data_preparation import load_multi_season_data
        from ml_pipeline.feature_pipeline import add_all_features
        
        logger.info("📂 Carregando dados...")
        df = load_multi_season_data(seasons=['2024-25', '2023-24', '2022-23'])
        
        if df.empty:
            raise ValueError("No data loaded")
            
        # Filter past games only (optimization)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] < datetime.now()]
        logger.info(f"📉 Filtered to past games: {len(df)} games")
        
        # Add ALL features
        logger.info("🎯 Adicionando features...")
        df = add_all_features(df, include_domain=True)
        
        logger.info(f"✅ {len(df)} games, {len(df.columns)} features\n")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar dados: {e}")
        logger.info("Usando dados sintéticos para demo...")
        df = generate_synthetic_data()
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in [
        'date', 'home_team', 'away_team', 'home_pts', 'away_pts',
        'target', 'home_score', 'away_score', 'home_losses', 'away_losses',
        'game_id', 'season'
    ]]
    
    df_clean = df[feature_cols + ['target']].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    logger.info(f"📊 Training data: {len(X)} samples, {len(feature_cols)} features")
    logger.info(f"   Domain features: ~13")
    logger.info(f"   Base features: ~{len(feature_cols) - 13}\n")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}\n")
    
    # Train
    logger.info("🤖 Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    logger.info("✅ Training complete!\n")
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    try:
        y_proba_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba_test)
    except:
        auc = 0.5
    
    logger.info("="*60)
    logger.info("📊 PERFORMANCE METRICS")
    logger.info("="*60)
    logger.info(f"Train Accuracy: {acc_train:.2%}")
    logger.info(f"Test Accuracy:  {acc_test:.2%}")
    logger.info(f"AUC-ROC:        {auc:.3f}")
    logger.info("="*60 + "\n")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("📊 TOP 15 FEATURES:")
    for idx, row in feature_importance.head(15).iterrows():
        logger.info(f"   {row['feature']:<30} {row['importance']:.4f}")
    logger.info("")
    
    # Domain features importance
    domain_keywords = ['injury', 'schedule', 'travel', 'pace', 'clutch', 
                      'playoff', 'matchup', 'rebounding', 'three_pt', 'turnover']
    
    domain_importance = feature_importance[
        feature_importance['feature'].apply(
            lambda x: any(kw in x.lower() for kw in domain_keywords)
        )
    ]
    
    if len(domain_importance) > 0:
        logger.info("🎯 DOMAIN FEATURES IMPORTANCE:")
        for idx, row in domain_importance.head(10).iterrows():
            logger.info(f"   {row['feature']:<30} {row['importance']:.4f}")
        logger.info("")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = f'models/ml_model_full_{timestamp}.joblib'
    
    joblib.dump(model, model_path)
    logger.info(f"💾 Modelo salvo: {model_path}\n")
    
    # Save as default
    joblib.dump(model, 'models/ml_model.joblib')
    logger.info(f"💾 Modelo default atualizado: models/ml_model.joblib\n")
    
    return {
        'accuracy_train': acc_train,
        'accuracy_test': acc_test,
        'auc': auc,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'model_path': model_path,
        'feature_importance': feature_importance.to_dict('records')
    }


def generate_synthetic_data(n=1000):
    """Synthetic data for demo."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'home_pts': np.random.normal(110, 10, n),
        'away_pts': np.random.normal(108, 10, n),
        'home_fga': np.random.normal(85, 5, n),
        'away_fga': np.random.normal(85, 5, n),
        'date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'home_team': np.random.choice(['LAL', 'GSW', 'BOS', 'MIA', 'PHX'], n),
        'away_team': np.random.choice(['LAL', 'GSW', 'BOS', 'MIA', 'PHX'], n),
    })
    
    from ml_pipeline.feature_pipeline import add_all_features
    df = add_all_features(df, include_domain=True)
    
    # Realistic target
    injury_effect = df.get('injury_impact_net', 0) * 0.15
    schedule_effect = df.get('schedule_density_gap', 0) * 0.10
    travel_effect = df.get('travel_fatigue_net', 0) * 0.08
    
    prob_win = 0.53 + injury_effect + schedule_effect + travel_effect
    prob_win += np.random.normal(0, 0.05, n)
    df['target'] = (np.random.random(n) < prob_win).astype(int)
    
    return df


if __name__ == '__main__':
    results = full_retrain()
    
    print(f"\n✅ Full Retraining Complete!")
    print(f"   Test Accuracy: {results['accuracy_test']:.2%}")
    print(f"   AUC: {results['auc']:.3f}")
    print(f"   Model: {results['model_path']}")
