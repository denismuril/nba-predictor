#!/usr/bin/env python3
"""
Feature Interactions - Fase 3

Cria features de interação que capturam relações não-lineares:
- Produtos entre features chave (home_efg × away_efg)
- Diferenças (home_sos - away_sos)
- Ratios (home_win_streak / away_win_streak)

Usage:
    python scripts/create_feature_interactions.py [--test-interactions]
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_interaction_features(df):
    """
    Cria features de interação.
    
    Args:
        df: DataFrame com features originais
    
    Returns:
        DataFrame com features de interação adicionadas
    """
    logger.info("🔧 Criando features de interação...")
    
    interactions_created = []
    
    # 1. Produtos (multiplicações)
    logger.info("\n📊 Produtos entre features:")
    
    # eFG% interaction (chave para scoring efficiency)
    if 'home_rolling_10_efg' in df.columns and 'away_rolling_10_efg' in df.columns:
        df['interaction_efg_product'] = df['home_rolling_10_efg'] * df['away_rolling_10_efg']
        interactions_created.append('interaction_efg_product')
        logger.info("   ✅ eFG product (scoring matchup)")
    
    # SOS interaction (strength of schedule matchup)
    if 'home_sos_10' in df.columns and 'away_sos_10' in df.columns:
        df['interaction_sos_product'] = df['home_sos_10'] * df['away_sos_10']
        interactions_created.append('interaction_sos_product')
        logger.info("   ✅ SOS product (schedule difficulty matchup)")
    
    # Win rate interaction
    if 'home_rolling_10_win' in df.columns and 'away_rolling_10_win' in df.columns:
        df['interaction_win_product'] = df['home_rolling_10_win'] * df['away_rolling_10_win']
        interactions_created.append('interaction_win_product')
        logger.info("   ✅ Win rate product (form matchup)")
    
    # 2. Diferenças (subtrações)
    logger.info("\n📊 Diferenças entre features:")
    
    # SOS differential (quem enfrentou oponentes mais difíceis)
    if 'home_sos_10' in df.columns and 'away_sos_10' in df.columns:
        df['interaction_sos_diff'] = df['home_sos_10'] - df['away_sos_10']
        interactions_created.append('interaction_sos_diff')
        logger.info("   ✅ SOS differential (schedule advantage)")
    
    # Win streak differential
    if 'home_win_streak' in df.columns and 'away_win_streak' in df.columns:
        df['interaction_streak_diff'] = df['home_win_streak'] - df['away_win_streak']
        interactions_created.append('interaction_streak_diff')
        logger.info("   ✅ Win streak differential (momentum advantage)")
    
    # Rest days differential (fatigue advantage)
    if 'home_rest_days' in df.columns and 'away_rest_days' in df.columns:
        df['interaction_rest_diff'] = df['home_rest_days'] - df['away_rest_days']
        interactions_created.append('interaction_rest_diff')
        logger.info("   ✅ Rest differential (fatigue advantage)")
    
    # 3. Ratios (divisões)
    logger.info("\n📊 Ratios entre features:")
    
    # Win rate ratio (dominância relativa)
    if 'home_rolling_10_win' in df.columns and 'away_rolling_10_win' in df.columns:
        # Evitar divisão por zero
        df['interaction_win_ratio'] = df['home_rolling_10_win'] / (df['away_rolling_10_win'] + 0.01)
        interactions_created.append('interaction_win_ratio')
        logger.info("   ✅ Win rate ratio (relative dominance)")
    
    # 4. Interactions complexas
    logger.info("\n📊 Interactions complexas:")
    
    # Momentum score (win streak × recent performance)
    if 'home_win_streak' in df.columns and 'home_rolling_5_win' in df.columns:
        df['interaction_home_momentum'] = df['home_win_streak'] * df['home_rolling_5_win']
        interactions_created.append('interaction_home_momentum')
        logger.info("   ✅ Home momentum (streak × recent)")
    
    if 'away_win_streak' in df.columns and 'away_rolling_5_win' in df.columns:
        df['interaction_away_momentum'] = df['away_win_streak'] * df['away_rolling_5_win']
        interactions_created.append('interaction_away_momentum')
        logger.info("   ✅ Away momentum (streak × recent)")
    
    # Advantage composite (SOS × win rate × rest)
    if all(f in df.columns for f in ['home_sos_10', 'home_rolling_10_win', 'home_rest_days']):
        df['interaction_home_composite'] = (
            df['home_sos_10'] * 
            df['home_rolling_10_win'] * 
            np.clip(df['home_rest_days'] / 5, 0, 2)  # Normalizar rest
        )
        interactions_created.append('interaction_home_composite')
        logger.info("   ✅ Home composite advantage")
    
    logger.info(f"\n✅ Total de interactions criadas: {len(interactions_created)}")
    
    return df, interactions_created

def test_interactions_impact():
    """Testa impacto das interactions na accuracy."""
    logger.info("="*80)
    logger.info("🧪 TESTANDO IMPACTO DAS INTERACTIONS")
    logger.info("="*80)
    
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Carregar dados
    df, weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    # Preparar features
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = (df['winner'] == 'HOME').astype(int)
    
    # Baseline sem interactions
    logger.info("\n📊 Baseline (sem interactions):")
    X_baseline = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    
    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    scores_baseline = cross_val_score(rf_baseline, X_baseline, y, cv=5, scoring='accuracy')
    
    logger.info(f"   Features: {X_baseline.shape[1]}")
    logger.info(f"   Accuracy: {scores_baseline.mean()*100:.2f}% (±{scores_baseline.std()*100:.2f}%)")
    
    # Com interactions
    logger.info("\n📊 Com interactions:")
    X_interactions, interaction_list = create_interaction_features(X.copy())
    X_interactions = pd.get_dummies(X_interactions, columns=['home_team', 'away_team'], drop_first=False)
    
    rf_interactions = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    scores_interactions = cross_val_score(rf_interactions, X_interactions, y, cv=5, scoring='accuracy')
    
    logger.info(f"   Features: {X_interactions.shape[1]} (+{len(interaction_list)} interactions)")
    logger.info(f"   Accuracy: {scores_interactions.mean()*100:.2f}% (±{scores_interactions.std()*100:.2f}%)")
    
    # Comparação
    improvement = (scores_interactions.mean() - scores_baseline.mean()) * 100
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📈 RESULTADO:")
    logger.info(f"   Melhoria: {improvement:+.2f}%")
    
    if improvement > 0.5:
        logger.info(f"   ✅ Interactions melhoraram significativamente!")
    elif improvement > 0:
        logger.info(f"   ✅ Melhoria marginal com interactions")
    else:
        logger.info(f"   ⚠️  Interactions não melhoraram (pode ser overfitting)")
    
    logger.info(f"{'='*80}")
    
    # Salvar lista de interactions úteis
    interactions_file = Path('data/models/interaction_features.json')
    with open(interactions_file, 'w') as f:
        import json
        json.dump({
            'interactions': interaction_list,
            'baseline_accuracy': float(scores_baseline.mean()),
            'with_interactions_accuracy': float(scores_interactions.mean()),
            'improvement': float(improvement)
        }, f, indent=2)
    
    logger.info(f"\n💾 Interactions salvadas em: {interactions_file}")
    
    return improvement > 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Feature Interactions - Fase 3')
    parser.add_argument('--test-interactions', action='store_true',
                       help='Testar impacto das interactions')
    
    args = parser.parse_args()
    
    if args.test_interactions:
        success = test_interactions_impact()
        return 0 if success else 1
    else:
        logger.info("Use --test-interactions para testar impacto")
        return 1

if __name__ == "__main__":
    sys.exit(main())
