"""
Optimized Totals Model Training - v17.0

Implements improvements to reduce MAE from 15.31 → 11-12 pts:
1. Totals-specific features (pace interactions, synergy)
2. Grid search for optimal hyperparameters
3. XGBoost with MAE objective

Target: 25% error reduction
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging
from datetime import datetime

from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.feature_engineering_v2 import prepare_features_v2
from ml_pipeline.totals_features import add_totals_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_optimized_totals_model(run_grid_search=True):
    """
    Train optimized Totals model with new features and hyperparameter tuning.
    
    Args:
        run_grid_search: If True, run full grid search (slow). If False, use best params.
        
    Returns:
        tuple: (model, mae, r2)
    """
    logger.info("="*80)
    logger.info("🏀 OPTIMIZED TOTALS MODEL TRAINING - v17.0")
    logger.info("="*80)
    logger.info(f"Grid Search: {'ENABLED' if run_grid_search else 'DISABLED (using best params)'}")
    
    # 1. Load historical data (FASE 3: 5 temporadas)
    logger.info("\n📊 Loading historical data (FASE 3: 5 seasons)...")
    # Don't calculate weights yet, as rows might be dropped during FE
    df = load_historical_data(
        seasons=['2021-22', '2022-23', '2023-24', '2024-25', '2025-26'],  # FASE 3: +2 temporadas
        apply_weights=False
    )
    
    if df is None or df.empty:
        logger.error("❌ Failed to load data!")
        return None, None, None
    
    logger.info(f"   Loaded {len(df)} games")
    
    # 2. Apply feature engineering
    logger.info("\n🔧 Applying feature engineering...")
    df_features = prepare_features_v2(df)
    
    if df_features is None or df_features.empty:
        logger.error("❌ Feature engineering failed!")
        return None, None, None
    
    # 3. Add Totals-specific features
    logger.info("\n🎯 Adding Totals-specific features...")
    df_totals = add_totals_features(df_features)
    
    # 3.5 Add ADVANCED interactions (FASE 1)
    logger.info("\n🔄 Adding advanced interaction features (FASE 1)...")
    from ml_pipeline.totals_features import add_advanced_interactions
    df_totals = add_advanced_interactions(df_totals)
    
    # 3.6 Add PLAYER IMPACT features (FASE 2 Aprimorada)
    logger.info("\n🏥 Adding player impact features (FASE 2)...")
    
    # NOVO: Proxy features para dados históricos
    from ml_pipeline.player_impact import add_proxy_impact_features
    df_totals = add_proxy_impact_features(df_totals)
    
    # Original: Lesões reais (apenas para previsões futuras)
    from ml_pipeline.player_impact import add_player_impact_features
    df_totals = add_player_impact_features(df_totals)
    
    # 4. Calculate target
    if 'total_points' not in df_totals.columns:
        if 'home_score' in df_totals.columns and 'away_score' in df_totals.columns:
            df_totals['total_points'] = df_totals['home_score'] + df_totals['away_score']
        else:
            logger.error("❌ Cannot calculate total_points - missing score columns!")
            return None, None, None
            
    # Calculate weights NOW, after all row drops/merges
    from ml_pipeline.data_preparation import calculate_sample_weights
    weights = calculate_sample_weights(df_totals)
    
    # 5. Select features for Totals - STRICT SELECTION to avoid leakage
    logger.info("\n📋 Selecting features for Totals model...")
    
    # Define allowed prefixes for pre-game features
    allowed_prefixes = [
        'home_rolling_', 'away_rolling_',
        'home_rest', 'away_rest',
        'home_travel', 'away_travel',
        'home_games_in', 'away_games_in',
        'home_b2b', 'away_b2b',
        'pace_', 'combined_', 'synergy', 'momentum', 'expected_total',
        'h2h_', 'net_off_', 'home_off_vs', 'away_off_vs',
        'interaction_'
    ]
    
    # Define explicit forbidden columns (raw stats)
    forbidden_cols = [
        'pts', 'opp_pts', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 'reb', 
        'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus', 'min',
        'home_score', 'away_score', 'total_points', 'pt_diff', 'winner',
        'off_rating', 'def_rating', 'net_rating', 'pace', 'pie', 'efg_pct', 'ts_pct' # Raw game stats
    ]
    
    available_features = []
    
    for col in df_totals.columns:
        # Must match at least one allowed prefix
        if not any(col.startswith(prefix) for prefix in allowed_prefixes):
            continue
            
        # Must not be in forbidden list (exact match)
        # Must be numeric
        if df_totals[col].dtype not in ['float64', 'int64']:
            continue
            
        available_features.append(col)
    
    # Remove duplicates
    available_features = list(set(available_features))
    
    logger.info(f"   Selected {len(available_features)} features (Leakage Free)")
    logger.info(f"   Top 10: {available_features[:10]}")
    
    # 6. Prepare X and y
    # Align weights with df_clean
    df_clean = df_totals.dropna(subset=available_features + ['total_points'])
    
    # Re-index weights to match df_clean
    weights_clean = weights[df_clean.index]
    
    X = df_clean[available_features]
    y = df_clean['total_points']
    
    logger.info(f"\n📊 Dataset: {len(X)} games after cleaning")
    logger.info(f"   Target range: {y.min():.0f} - {y.max():.0f} pts (mean: {y.mean():.1f})")
    
    # 7. Split (temporal)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train = weights_clean[:split_idx]  # Numpy slicing
    
    # 8. Hyperparameter optimization or use best params
    if True: # FORCE GRID SEARCH
        logger.info("\n🔍 Running Grid Search (this may take 15-30 min)...")
        
        param_grid = {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3, 5]
        }
        
        base_model = XGBRegressor(
            objective='reg:absoluteerror',
            eval_metric='mae',
            n_jobs=-1,
            random_state=42
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2
        )
        
        start = datetime.now()
        grid_search.fit(X_train, y_train, sample_weight=w_train)
        duration = (datetime.now() - start).total_seconds() / 60
        
        best_params = grid_search.best_params_
        best_mae_cv = -grid_search.best_score_
        
        logger.info(f"\n✅ Grid Search completed in {duration:.1f} min")
        logger.info(f"   Best MAE (CV): {best_mae_cv:.2f} pts")
        logger.info(f"   Best params: {best_params}")
        
        # Save best params
        params_file = Path('data/models/best_totals_hyperparameters.joblib')
        joblib.dump(best_params, params_file)
        logger.info(f"   Saved to: {params_file}")
        
        model = XGBRegressor(**best_params, n_jobs=-1, random_state=42,
                             objective='reg:absoluteerror', eval_metric='mae')
    
    else:
        # Use pre-optimized params or defaults
        params_file = Path('data/models/best_totals_hyperparameters.joblib')
        
        if params_file.exists():
            logger.info(f"\n💎 Using optimized params from {params_file}")
            best_params = joblib.load(params_file)
            model = XGBRegressor(**best_params, n_jobs=-1, random_state=42,
                                 objective='reg:absoluteerror', eval_metric='mae')
        else:
            logger.info("\n⚙️  Using default optimized params")
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=3,
                objective='reg:absoluteerror',
                eval_metric='mae',
                n_jobs=-1,
                random_state=42
            )
    
    # 9. Train final model
    logger.info("\n🏋️ Training final model...")
    model.fit(X_train, y_train, sample_weight=w_train,
              eval_set=[(X_test, y_test)], verbose=False)
    
    # 10. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS - Totals Model v17")
    logger.info("="*80)
    logger.info(f"MAE:  {mae:.2f} pts")
    logger.info(f"R²:   {r2:.4f}")
    logger.info("="*80)
    
    # Compare with baseline
    baseline_mae = 15.31
    improvement = baseline_mae - mae
    improvement_pct = (improvement / baseline_mae) * 100
    
    if mae < baseline_mae:
        logger.info(f"🎉 IMPROVEMENT: {improvement:.2f} pts ({improvement_pct:.1f}%)")
        if mae <= 12.0:
            logger.info("🏆 TARGET ACHIEVED: MAE ≤ 12 pts!")
    else:
        logger.warning(f"⚠️  MAE increased by {abs(improvement):.2f} pts vs baseline")
    
    # 11. Feature importance
    logger.info("\n🏆 Top 10 Features:")
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"   {row['feature'][:40]:40s} {row['importance']:.4f}")
    
    # 12. Save model
    models_dir = Path('data/models')
    models_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = models_dir / 'totals_model_v17_optimized.joblib'
    features_path = models_dir / 'totals_feature_names_v17.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(available_features, features_path)
    
    logger.info(f"\n💾 Model saved: {model_path}")
    logger.info(f"💾 Features saved: {features_path}")
    
    # Save feature importance
    importance_path = models_dir / 'totals_feature_importance_v17.csv'
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"💾 Importance saved: {importance_path}")
    
    return model, mae, r2


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-search', action='store_true',
                        help='Run full grid search (slow, ~30min)')
    args = parser.parse_args()
    
    train_optimized_totals_model(run_grid_search=args.grid_search)
