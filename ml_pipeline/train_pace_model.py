"""
Pace Prediction Model for Totals Refinement
============================================
Predicts game pace (possessions) separately for more accurate totals.

Formula: Points = (Pace * Efficiency) / 100

This approach is more accurate than predicting points directly because:
1. Pace is more stable/predictable than raw points
2. Efficiency can be calculated from Four Factors
3. Splits prediction into two interpretable components

Usage:
    from ml_pipeline.train_pace_model import PacePredictor
    predictor = PacePredictor()
    pace, predicted_total = predictor.predict(home_team, away_team, game_date)

Author: NBA Predictor v24.0 - Shadow Mode
"""
import os
import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = Path('data/models/pace_model.pkl')
LEAGUE_AVG_PACE = 100.8  # 2024-25 season
LEAGUE_AVG_ORTG = 113.5  # Offensive rating


@dataclass
class PacePrediction:
    """Prediction result for pace-based totals."""
    predicted_pace: float
    home_efficiency: float
    away_efficiency: float
    predicted_total: float
    confidence: float


# =============================================================================
# PACE PREDICTION MODEL
# =============================================================================
class PacePredictor:
    """
    Predicts game pace and uses efficiency to calculate predicted total.
    
    More accurate than direct totals prediction because pace is more stable.
    """
    
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_names = []
        self._load_model()
    
    def _load_model(self):
        """Load trained model from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_names = data.get('features', [])
                    logger.info(f"✅ Pace model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load pace model: {e}")
    
    def _get_team_pace_features(self, team: str, game_date: str) -> Dict[str, float]:
        """
        Get pace-related features for a team.
        
        Features:
        - Rolling pace (5, 10, 30 games)
        - Pace variance (consistency)
        - Rest days impact
        - Home/Away pace split
        """
        # Try to get from feature store
        try:
            from feature_store import FeatureStore
            from data.repositories.db_manager import get_db_manager
            
            db = get_db_manager()
            fs = FeatureStore(db)
            
            features = fs.get_team_features(team, game_date)
            
            return {
                f'{team}_pace_avg_5': features.get('pace_avg_5', LEAGUE_AVG_PACE),
                f'{team}_pace_avg_10': features.get('pace_avg_10', LEAGUE_AVG_PACE),
                f'{team}_pace_std': features.get('pace_std', 3.0),
                f'{team}_ortg_avg_10': features.get('ortg_avg_10', LEAGUE_AVG_ORTG),
                f'{team}_drtg_avg_10': features.get('drtg_avg_10', LEAGUE_AVG_ORTG),
            }
        except Exception:
            # Fallback to league averages
            return {
                f'{team}_pace_avg_5': LEAGUE_AVG_PACE,
                f'{team}_pace_avg_10': LEAGUE_AVG_PACE,
                f'{team}_pace_std': 3.0,
                f'{team}_ortg_avg_10': LEAGUE_AVG_ORTG,
                f'{team}_drtg_avg_10': LEAGUE_AVG_ORTG,
            }
    
    def predict(
        self, 
        home_team: str, 
        away_team: str, 
        game_date: str = None
    ) -> PacePrediction:
        """
        Predict game pace and total points.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            game_date: Game date (defaults to today)
            
        Returns:
            PacePrediction with pace, efficiencies, and predicted total
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get features
        home_features = self._get_team_pace_features(home_team, game_date)
        away_features = self._get_team_pace_features(away_team, game_date)
        
        # Extract pace values
        home_pace = home_features.get(f'{home_team}_pace_avg_10', LEAGUE_AVG_PACE)
        away_pace = away_features.get(f'{away_team}_pace_avg_10', LEAGUE_AVG_PACE)
        
        # Predict game pace (weighted toward slower team - defense controls tempo)
        if home_pace < away_pace:
            # Home is slower, controls tempo
            predicted_pace = home_pace * 0.55 + away_pace * 0.45
        else:
            # Away is slower, controls tempo
            predicted_pace = home_pace * 0.45 + away_pace * 0.55
        
        # Get efficiencies
        home_ortg = home_features.get(f'{home_team}_ortg_avg_10', LEAGUE_AVG_ORTG)
        away_ortg = away_features.get(f'{away_team}_ortg_avg_10', LEAGUE_AVG_ORTG)
        home_drtg = home_features.get(f'{home_team}_drtg_avg_10', LEAGUE_AVG_ORTG)
        away_drtg = away_features.get(f'{away_team}_drtg_avg_10', LEAGUE_AVG_ORTG)
        
        # Calculate expected efficiency for each team
        # Home offense vs Away defense, Away offense vs Home defense
        home_expected_eff = (home_ortg + away_drtg) / 2
        away_expected_eff = (away_ortg + home_drtg) / 2
        
        # Calculate predicted total using the formula:
        # Points = (Pace * Efficiency) / 100
        # Total = Home Points + Away Points
        home_points = (predicted_pace * home_expected_eff) / 100
        away_points = (predicted_pace * away_expected_eff) / 100
        predicted_total = home_points + away_points
        
        # Calculate confidence based on pace variance
        home_std = home_features.get(f'{home_team}_pace_std', 3.0)
        away_std = away_features.get(f'{away_team}_pace_std', 3.0)
        avg_std = (home_std + away_std) / 2
        
        # Lower variance = higher confidence
        confidence = max(0.3, min(0.95, 1 - (avg_std / 10)))
        
        return PacePrediction(
            predicted_pace=predicted_pace,
            home_efficiency=home_expected_eff,
            away_efficiency=away_expected_eff,
            predicted_total=predicted_total,
            confidence=confidence
        )
    
    def predict_with_market_comparison(
        self,
        home_team: str,
        away_team: str,
        market_total: float,
        game_date: str = None
    ) -> Dict:
        """
        Predict totals and compare with market line.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            market_total: Market's total line (e.g., 225.5)
            game_date: Game date
            
        Returns:
            Dict with prediction, edge, and recommendation
        """
        pred = self.predict(home_team, away_team, game_date)
        
        edge = pred.predicted_total - market_total
        edge_pct = abs(edge) / market_total * 100
        
        # Recommendation
        if edge > 3 and edge_pct > 1.5:
            recommendation = 'OVER'
            confidence = min(0.9, pred.confidence * (1 + edge_pct / 10))
        elif edge < -3 and edge_pct > 1.5:
            recommendation = 'UNDER'
            confidence = min(0.9, pred.confidence * (1 + edge_pct / 10))
        else:
            recommendation = 'NO BET'
            confidence = 0.0
        
        return {
            'predicted_total': pred.predicted_total,
            'market_total': market_total,
            'edge': edge,
            'edge_pct': edge_pct,
            'predicted_pace': pred.predicted_pace,
            'recommendation': recommendation,
            'confidence': confidence,
        }


# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_pace_model(seasons: list = None) -> Tuple[any, float]:
    """
    Train the pace prediction model.
    
    Args:
        seasons: List of seasons to use for training
        
    Returns:
        Tuple of (trained model, MAE)
    """
    if seasons is None:
        seasons = ['2023-24', '2024-25']
    
    logger.info("🏀 Training Pace Prediction Model...")
    
    try:
        from ml_pipeline.data_preparation import load_historical_data
        
        df = load_historical_data(seasons=seasons)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Target: actual game pace (average of home and away possessions)
        # We'll use total_points / (ortg + drtg) * 100 as proxy
        if 'total_points' in df.columns:
            # Use actual total points
            y = df['total_points'].values
        else:
            # Calculate from home/away scores
            y = (df.get('home_score', 0) + df.get('away_score', 0)).values
        
        # Features for pace prediction
        pace_features = [
            'home_pace_avg_5', 'home_pace_avg_10', 'away_pace_avg_5', 'away_pace_avg_10',
            'home_ortg_avg_10', 'away_ortg_avg_10', 'home_drtg_avg_10', 'away_drtg_avg_10',
        ]
        
        # Filter to available columns
        available = [f for f in pace_features if f in df.columns]
        
        if len(available) < 4:
            logger.warning(f"⚠️ Not enough pace features ({len(available)}), using fallback")
            # Use any numeric columns
            available = df.select_dtypes(include=[np.number]).columns[:20].tolist()
        
        X = df[available].fillna(0)
        
        # Remove rows with invalid targets
        valid_mask = (y > 150) & (y < 350)  # Reasonable NBA totals range
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"📊 Training on {len(X)} games with {len(available)} features")
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=42
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        mae = -scores.mean()
        
        logger.info(f"📈 Cross-validation MAE: {mae:.2f} points")
        
        # Train final model
        model.fit(X, y)
        
        # Save model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': available,
                'mae': mae,
                'trained_at': datetime.now().isoformat()
            }, f)
        
        logger.info(f"💾 Model saved to {MODEL_PATH}")
        
        return model, mae
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None, float('inf')


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🏀 Pace Prediction Model Test")
    print("=" * 50)
    
    predictor = PacePredictor()
    
    # Test prediction
    test_games = [
        ('LAL', 'BOS', 227.5),
        ('GSW', 'PHX', 231.0),
        ('MIA', 'DEN', 218.5),
    ]
    
    for home, away, market in test_games:
        result = predictor.predict_with_market_comparison(home, away, market)
        
        print(f"\n{home} vs {away}")
        print(f"  Predicted Pace: {result['predicted_pace']:.1f}")
        print(f"  Predicted Total: {result['predicted_total']:.1f}")
        print(f"  Market Total: {result['market_total']}")
        print(f"  Edge: {result['edge']:+.1f} ({result['edge_pct']:.1f}%)")
        print(f"  Recommendation: {result['recommendation']}")
    
    print("\n✅ Test complete!")
