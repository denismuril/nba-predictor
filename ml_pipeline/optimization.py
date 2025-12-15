#!/usr/bin/env python3
"""
Profit-Based Hyperparameter Optimization (Quant Edge)
======================================================
Otimização bayesiana usando lucro simulado como objetivo principal.

Diferenças do optimize_hyperparameters_v6.py:
- Objetivo: Lucro Simulado (não apenas Log Loss)
- Métrica: Kelly-weighted ROI com odds reais
- Pruner: Cancela trials com ROI negativo 

Usage:
    python ml_pipeline/optimization.py --n_trials 50
    python ml_pipeline/optimization.py --n_trials 5 --timeout 120  # Smoke test

Author: NBA Predictor v24.0 - Quant Edge
"""
import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.data_preparation import load_historical_data, calculate_sample_weights

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================
ML_SEASONS = ['2023-24', '2024-25', '2025-26']
OUTPUT_FILE = Path('data/models/best_hyperparameters_profit.json')

# Kelly Configuration
KELLY_FRACTION = 0.25  # Quarter Kelly (conservative)
MIN_EDGE_PCT = 5.0     # Minimum 5% edge to bet
MAX_BET_PCT = 5.0      # Maximum 5% of bankroll per bet

# Features to drop (same as train_ensemble_v6.py)
BASE_DROP_COLS = [
    'winner', 'correct', 'date', 'prediction',
    'home_score', 'away_score', 'pt_diff', 'point_differential', 'total_points',
    'pts', 'opp_pts',
    'fgm', 'fga', 'fg3m', 'fg3a', 'tov', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'pf', 'fta', 'ftm',
    'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_fg3a', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_reb', 
    'opp_ast', 'opp_stl', 'opp_blk', 'opp_pf', 'opp_fta', 'opp_ftm',
    'home_efg', 'home_efg_pct', 'home_tov_pct', 'home_orb_pct', 'home_ftr', 
    'home_off_rating', 'home_def_rating', 'home_pace', 'home_pie', 
    'home_pos', 'home_ast_ratio', 'home_to_ratio', 'home_ts_pct', 'home_reb_pct',
    'away_efg', 'away_efg_pct', 'away_tov_pct', 'away_orb_pct', 'away_ftr', 
    'away_off_rating', 'away_def_rating', 'away_pace', 'away_pie', 
    'away_pos', 'away_ast_ratio', 'away_to_ratio', 'away_ts_pct', 'away_reb_pct',
    'home_ortg_adj', 'away_ortg_adj', 'home_drtg_adj', 'away_drtg_adj',
    'liga_ortg_avg', 'liga_drtg_avg',
    'home_elo', 'away_elo', 'elo_diff',
    'prob_home', 'prob_away',
    'home_team', 'away_team', 'win', 'opp_win', 'game_id', 'season',
    # Odds columns (used for profit calculation, not features)
    'odds_home', 'odds_away', 'odd_home', 'odd_away',
    'opening_odds', 'closing_odds'
]


@dataclass
class BettingResult:
    """Result of a simulated betting period."""
    total_profit: float
    roi_pct: float
    n_bets: int
    win_rate: float
    avg_edge: float
    max_drawdown_pct: float


# =============================================================================
# DATA LOADING
# =============================================================================
def load_optimization_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data for profit-based optimization.
    
    Returns:
        Tuple of (X, y, sample_weights, odds_data)
    """
    logger.info("📊 Loading data for profit-based optimization...")
    
    df = load_historical_data(seasons=ML_SEASONS)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Extract odds data before dropping
    odds_home = df['odds_home'].fillna(df.get('odd_home', 1.90)).fillna(1.90).values
    odds_away = df['odds_away'].fillna(df.get('odd_away', 1.90)).fillna(1.90).values
    
    # Stack odds [home_odds, away_odds] for each game
    odds_data = np.column_stack([odds_home, odds_away])
    
    # Calculate sample weights
    sample_weights = calculate_sample_weights(df)
    
    # Prepare features
    X = df.drop(columns=BASE_DROP_COLS, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    
    # Create target
    y = (df['winner'] == 'HOME').astype(int).values
    
    logger.info(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"   Odds range: Home [{odds_home.min():.2f}, {odds_home.max():.2f}], "
                f"Away [{odds_away.min():.2f}, {odds_away.max():.2f}]")
    
    return X, y, sample_weights, odds_data


# =============================================================================
# PROFIT SIMULATION ENGINE
# =============================================================================
def simulate_betting(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    odds: np.ndarray,
    kelly_fraction: float = KELLY_FRACTION,
    min_edge_pct: float = MIN_EDGE_PCT
) -> BettingResult:
    """
    Simulate betting strategy on predictions.
    
    Args:
        y_prob: Predicted probabilities for HOME win
        y_true: Actual outcomes (1=HOME win, 0=AWAY win)
        odds: Array of [home_odds, away_odds] for each game
        kelly_fraction: Fraction of Kelly to use
        min_edge_pct: Minimum edge required to bet
        
    Returns:
        BettingResult with profit metrics
    """
    bankroll = 1000.0
    initial_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown = 0.0
    
    profits = []
    bets_placed = 0
    bets_won = 0
    edges = []
    
    for i in range(len(y_prob)):
        prob_home = y_prob[i]
        prob_away = 1 - prob_home
        odd_home, odd_away = odds[i]
        actual = y_true[i]
        
        # Calculate implied probabilities (fair odds)
        implied_home = 1 / odd_home if odd_home > 1 else 1.0
        implied_away = 1 / odd_away if odd_away > 1 else 1.0
        
        # Remove vig proportionally
        total_implied = implied_home + implied_away
        fair_implied_home = implied_home / total_implied if total_implied > 0 else 0.5
        fair_implied_away = implied_away / total_implied if total_implied > 0 else 0.5
        
        # Calculate edge for each side
        edge_home = (prob_home - fair_implied_home) * 100
        edge_away = (prob_away - fair_implied_away) * 100
        
        bet_side = None
        bet_prob = None
        bet_odds = None
        edge = 0
        
        # Decide betting side based on edge
        if edge_home >= min_edge_pct and edge_home > edge_away:
            bet_side = 'HOME'
            bet_prob = prob_home
            bet_odds = odd_home
            edge = edge_home
        elif edge_away >= min_edge_pct:
            bet_side = 'AWAY'
            bet_prob = prob_away
            bet_odds = odd_away
            edge = edge_away
        
        if bet_side and bankroll > 0:
            # Kelly Criterion calculation
            b = bet_odds - 1  # Net odds
            p = bet_prob
            q = 1 - p
            
            kelly_full = (b * p - q) / b if b > 0 else 0
            kelly_stake_pct = max(0, min(kelly_full * kelly_fraction, MAX_BET_PCT / 100))
            
            if kelly_stake_pct > 0:
                bet_amount = bankroll * kelly_stake_pct
                bets_placed += 1
                edges.append(edge)
                
                # Resolve bet
                won = (bet_side == 'HOME' and actual == 1) or (bet_side == 'AWAY' and actual == 0)
                
                if won:
                    profit = bet_amount * (bet_odds - 1)
                    bets_won += 1
                else:
                    profit = -bet_amount
                
                bankroll += profit
                profits.append(profit)
                
                # Track drawdown
                peak_bankroll = max(peak_bankroll, bankroll)
                current_drawdown = (peak_bankroll - bankroll) / peak_bankroll * 100
                max_drawdown = max(max_drawdown, current_drawdown)
    
    # Calculate final metrics
    total_profit = bankroll - initial_bankroll
    roi_pct = (total_profit / initial_bankroll) * 100 if initial_bankroll > 0 else 0
    win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0
    avg_edge = np.mean(edges) if edges else 0
    
    return BettingResult(
        total_profit=total_profit,
        roi_pct=roi_pct,
        n_bets=bets_placed,
        win_rate=win_rate,
        avg_edge=avg_edge,
        max_drawdown_pct=max_drawdown
    )


# =============================================================================
# PROFIT-BASED EVALUATION
# =============================================================================
def evaluate_model_profit(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    odds: np.ndarray,
    sample_weights: np.ndarray = None,
    n_splits: int = 5
) -> Tuple[float, float, float, BettingResult]:
    """
    Evaluate model using profit simulation as primary metric.
    
    Returns:
        Tuple of (profit_score, log_loss, accuracy, betting_result)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_profits = []
    all_ll = []
    all_bets = 0
    all_wins = 0
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        odds_val = odds[val_idx]
        w_train = sample_weights[train_idx] if sample_weights is not None else None
        
        # Train model
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Get probabilities
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Traditional metrics
        ll = log_loss(y_val, y_prob)
        all_ll.append(ll)
        
        # Profit simulation
        result = simulate_betting(y_prob, y_val, odds_val)
        all_profits.append(result.roi_pct)
        all_bets += result.n_bets
        all_wins += int(result.win_rate * result.n_bets / 100) if result.n_bets > 0 else 0
    
    # Aggregate results
    avg_roi = np.mean(all_profits)
    avg_ll = np.mean(all_ll)
    win_rate = (all_wins / all_bets * 100) if all_bets > 0 else 0
    
    # Create aggregate betting result
    aggregate_result = BettingResult(
        total_profit=avg_roi * 10,  # Rough scale
        roi_pct=avg_roi,
        n_bets=all_bets // n_splits,
        win_rate=win_rate,
        avg_edge=0,  # Complex to aggregate
        max_drawdown_pct=0
    )
    
    return avg_roi, avg_ll, win_rate, aggregate_result


# =============================================================================
# OPTUNA OBJECTIVES (Profit-Based)
# =============================================================================
def objective_rf(trial, X, y, odds, weights) -> float:
    """Random Forest with profit objective."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    roi, ll, wr, result = evaluate_model_profit(model, X, y, odds, weights)
    
    # Store secondary metrics
    trial.set_user_attr('log_loss', ll)
    trial.set_user_attr('win_rate', wr)
    trial.set_user_attr('n_bets', result.n_bets)
    
    # Prune if clearly unprofitable
    if result.n_bets > 5 and roi < -20:
        raise optuna.TrialPruned()
    
    return roi


def objective_xgb(trial, X, y, odds, weights) -> float:
    """XGBoost with profit objective."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 3.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params, use_label_encoder=False)
    roi, ll, wr, result = evaluate_model_profit(model, X, y, odds, weights)
    
    trial.set_user_attr('log_loss', ll)
    trial.set_user_attr('win_rate', wr)
    trial.set_user_attr('n_bets', result.n_bets)
    
    if result.n_bets > 5 and roi < -20:
        raise optuna.TrialPruned()
    
    return roi


def objective_lgbm(trial, X, y, odds, weights) -> float:
    """LightGBM with profit objective."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 3.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    roi, ll, wr, result = evaluate_model_profit(model, X, y, odds, weights)
    
    trial.set_user_attr('log_loss', ll)
    trial.set_user_attr('win_rate', wr)
    trial.set_user_attr('n_bets', result.n_bets)
    
    if result.n_bets > 5 and roi < -20:
        raise optuna.TrialPruned()
    
    return roi


def objective_hist(trial, X, y, odds, weights) -> float:
    """HistGradientBoosting with profit objective."""
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.1, 5.0, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'max_bins': trial.suggest_int('max_bins', 128, 255),
        'random_state': 42
    }
    
    model = HistGradientBoostingClassifier(**params)
    roi, ll, wr, result = evaluate_model_profit(model, X, y, odds, weights)
    
    trial.set_user_attr('log_loss', ll)
    trial.set_user_attr('win_rate', wr)
    trial.set_user_attr('n_bets', result.n_bets)
    
    if result.n_bets > 5 and roi < -20:
        raise optuna.TrialPruned()
    
    return roi


def objective_extra(trial, X, y, odds, weights) -> float:
    """ExtraTrees with profit objective."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 25),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = ExtraTreesClassifier(**params)
    roi, ll, wr, result = evaluate_model_profit(model, X, y, odds, weights)
    
    trial.set_user_attr('log_loss', ll)
    trial.set_user_attr('win_rate', wr)
    trial.set_user_attr('n_bets', result.n_bets)
    
    if result.n_bets > 5 and roi < -20:
        raise optuna.TrialPruned()
    
    return roi


# =============================================================================
# OPTIMIZATION ORCHESTRATION
# =============================================================================
def run_profit_optimization(n_trials: int = 50, timeout: Optional[int] = None) -> Dict:
    """
    Run profit-based Bayesian optimization for all models.
    
    Args:
        n_trials: Number of trials per model
        timeout: Timeout in seconds per model
        
    Returns:
        Dict with best parameters and profit metrics
    """
    logger.info("=" * 80)
    logger.info("💰 PROFIT-BASED HYPERPARAMETER OPTIMIZATION (QUANT EDGE)")
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"🔢 Trials per model: {n_trials}")
    logger.info(f"⏱️  Timeout: {timeout}s" if timeout else "⏱️  No timeout")
    logger.info(f"💵 Kelly Fraction: {KELLY_FRACTION}")
    logger.info(f"📊 Min Edge: {MIN_EDGE_PCT}%")
    logger.info("=" * 80)
    
    # Load data
    X, y, weights, odds = load_optimization_data()
    
    best_params = {}
    metrics_summary = {}
    
    # Define models
    models = [
        ('rf', 'Random Forest', objective_rf),
        ('xgb', 'XGBoost', objective_xgb),
        ('lgbm', 'LightGBM', objective_lgbm),
        ('hist', 'HistGradientBoosting', objective_hist),
        ('extra', 'ExtraTrees', objective_extra),
    ]
    
    for model_key, model_name, objective_fn in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 Optimizing {model_name} for PROFIT...")
        logger.info(f"{'='*60}")
        
        study = optuna.create_study(
            direction='maximize',  # Maximize ROI
            study_name=f'Profit_{model_key}',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        study.optimize(
            lambda trial: objective_fn(trial, X, y, odds, weights),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Extract best parameters
        best_params[model_key] = study.best_params
        
        # Extract metrics from best trial
        best_trial = study.best_trial
        metrics_summary[model_key] = {
            'roi_pct': study.best_value,
            'log_loss': best_trial.user_attrs.get('log_loss', 0),
            'win_rate': best_trial.user_attrs.get('win_rate', 0),
            'n_bets': best_trial.user_attrs.get('n_bets', 0),
        }
        
        logger.info(f"✅ {model_name} - Best ROI: {study.best_value:+.2f}%")
        logger.info(f"   Log Loss: {metrics_summary[model_key]['log_loss']:.4f}")
        logger.info(f"   Win Rate: {metrics_summary[model_key]['win_rate']:.1f}%")
        logger.info(f"   Avg Bets/Fold: {metrics_summary[model_key]['n_bets']}")
    
    # Save results
    output_data = {
        'optimized_at': datetime.now().isoformat(),
        'optimization_type': 'PROFIT_BASED',
        'n_trials': n_trials,
        'primary_metric': 'roi_pct',
        'kelly_fraction': KELLY_FRACTION,
        'min_edge_pct': MIN_EDGE_PCT,
        'cv_method': 'TimeSeriesSplit(n_splits=5)',
        'models': best_params,
        'metrics': metrics_summary
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"💾 Results saved to: {OUTPUT_FILE}")
    logger.info(f"{'='*80}")
    
    # Final summary
    logger.info("\n📊 PROFIT OPTIMIZATION SUMMARY:")
    logger.info("-" * 60)
    logger.info(f"{'Model':<10} | {'ROI':>8} | {'LogLoss':>8} | {'WinRate':>8} | {'Bets':>6}")
    logger.info("-" * 60)
    for model_key, metrics in metrics_summary.items():
        logger.info(
            f"{model_key.upper():<10} | "
            f"{metrics['roi_pct']:>+7.2f}% | "
            f"{metrics['log_loss']:>8.4f} | "
            f"{metrics['win_rate']:>7.1f}% | "
            f"{metrics['n_bets']:>6}"
        )
    logger.info("-" * 60)
    
    # Find best model
    best_model = max(metrics_summary.items(), key=lambda x: x[1]['roi_pct'])
    logger.info(f"\n🏆 Best Model: {best_model[0].upper()} with ROI: {best_model[1]['roi_pct']:+.2f}%")
    
    return best_params


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Profit-Based Hyperparameter Optimization (Quant Edge)'
    )
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50,
        help='Number of trials per model (default: 50)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds per model (default: no limit)'
    )
    
    args = parser.parse_args()
    
    run_profit_optimization(n_trials=args.n_trials, timeout=args.timeout)
