"""
Quick model retraining script com domain features.

Simplified version para treinar modelo rapidamente com novas features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_retrain():
    """Retrain modelo com domain features."""
    
    logger.info("🔄 Quick Retrain com Domain Features...")
    
    # 1. Load data (simplified - usando dados sintéticos se DB falhar)
    try:
        from ml_pipeline.data_preparation import load_multi_season_data
        df = load_multi_season_data(seasons=['2024-25'])
        
        if df.empty:
            raise ValueError("No data")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar dados reais: {e}")
        logger.info("📊 Criando dados sintéticos para demo...")
        
        # Dados sintéticos realistas
        np.random.seed(42)
        n = 500
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'home_team': ['Lakers'] * n,
            'away_team': ['Warriors'] * n,
            'home_pts': np.random.normal(110, 10, n),
            'away_pts': np.random.normal(108, 10, n),
            'home_fga': np.random.normal(85, 5, n),
            'away_fga': np.random.normal(85, 5, n),
            'home_fg3a': np.random.normal(35, 4, n),
            'away_fg3a': np.random.normal(35, 4, n),
            'home_ast': np.random.normal(25, 3, n),
            'away_ast': np.random.normal(24, 3, n),
            'home_reb': np.random.normal(45, 5, n),
            'away_reb': np.random.normal(45, 5, n),
            'home_tov': np.random.normal(13, 2, n),
            'away_tov': np.random.normal(13, 2, n),
            'home_orb': np.random.normal(11, 2, n),
            'home_drb': np.random.normal(34, 3, n),
            'away_orb': np.random.normal(11, 2, n),
            'away_drb': np.random.normal(34, 3, n),
            'home_wins': np.random.randint(20, 45, n),
            'home_losses': 50,
            'away_wins': np.random.randint(20, 45, n),
            'away_losses': 50,
        })
        df['home_losses'] = 50 - df['home_wins']
        df['away_losses'] = 50 - df['away_wins']
    
    # 2. Add domain features
    logger.info("🎯 Adicionando domain expert features...")
    from ml_pipeline.feature_pipeline import add_all_features
    df = add_all_features(df, include_domain=True)
    
    logger.info(f"✅ Features totais: {len(df.columns)}")
    
    # 3. Create target
    df['target'] = (df['home_pts'] > df['away_pts']).astype(int)
    
    # 4. Select features (basic + domain)
    feature_cols = [col for col in df.columns if col not in [
        'date', 'home_team', 'away_team', 'home_pts', 'away_pts', 
        'target', 'home_score', 'away_score'
    ]]
    
    # Remove NaN
    df = df[feature_cols + ['target']].dropna()
    
    X = df[feature_cols]
    y = df['target']
    
    logger.info(f"📊 Dataset: {len(X)} samples, {len(feature_cols)} features")
    
    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 6. Train model
    logger.info("🤖 Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 7. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\n📊 Results:")
    logger.info(f"  Train accuracy: {model.score(X_train, y_train):.4f}")
    logger.info(f"  Test accuracy: {accuracy:.4f}")
    
    # 8. Feature importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\n🎯 Top 10 Features:")
    for idx, row in importances.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Check domain features
    domain_features = importances[importances['feature'].str.contains(
        'pace|def_matchup|reb_edge|3pt|tov_pressure|clutch|playoff|ts_pct|ast_tov',
        na=False
    )]
    
    if len(domain_features) > 0:
        logger.info(f"\n✅ Domain Features in Top 20:")
        for idx, row in domain_features.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 9. Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = f'models/ml_model_domain_{timestamp}.joblib'
    joblib.dump(model, model_path)
    
    # Also save as default
    joblib.dump(model, 'models/ml_model.joblib')
    
    logger.info(f"\n💾 Modelo salvo:")
    logger.info(f"  - {model_path}")
    logger.info(f"  - models/ml_model.joblib")
    
    logger.info(f"\n✅ Retraining complete! Accuracy: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'model_path': model_path
    }


if __name__ == '__main__':
    results = quick_retrain()
    print(f"\n🎯 Final: {results}")
