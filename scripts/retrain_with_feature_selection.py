#!/usr/bin/env python3
"""
Retreina modelos COM Feature Selection aplicada

Executa pipeline completo:
1. Carrega dados com advanced features
2. Aplica feature selection (correlation + importance)
3. Retreina todos os 3 modelos
4. Compara com baseline

Usage:
    python scripts/retrain_with_feature_selection.py [--importance-threshold 0.0005]
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import argparse
import time
import pandas as pd

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retrain_with_selection(importance_threshold=0.0005, corr_threshold=0.95):
    """Retreina modelos com feature selection."""
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("🎯 RETREINAMENTO COM FEATURE SELECTION")
    logger.info("="*80)
    logger.info(f"📊 Correlation threshold: {corr_threshold}")
    logger.info(f"🎯 Importance threshold: {importance_threshold}")
    logger.info("="*80)
    
    # 1. Carregar dados e aplicar feature selection
    logger.info("\n📊 FASE 1: FEATURE SELECTION")
    logger.info("-"*80)
    
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    from ml_pipeline.feature_selection import select_features
    
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
    X = pd.get_dummies(X, columns=['home_team', 'away_team'], drop_first=False)
    y = (df['winner'] == 'HOME').astype(int)
    
    logger.info(f"📊 Features antes da seleção: {X.shape[1]}")
    
    # Proteger team dummies e odds
    team_dummies = [c for c in X.columns if 'team_' in c]
    odds_features = [c for c in X.columns if 'odds' in c]
    protected = team_dummies + odds_features
    
    # Aplicar feature selection
    X_selected, selection_report = select_features(
        X, y,
        method='combined',
        corr_threshold=corr_threshold,
        importance_threshold=importance_threshold,
        sample_weight=weights,
        exclude_from_corr=protected
    )
    
    logger.info(f"\n✅ Features após seleção: {X_selected.shape[1]}")
    logger.info(f"📉 Redução: {100 * (1 - X_selected.shape[1]/X.shape[1]):.1f}%")
    
    # 2. Retreinar Moneyline com features selecionadas
    logger.info("\n📊 FASE 2: RETREINAMENTO DO MONEYLINE")
    logger.info("-"*80)
    
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full_ensemble = True
    except ImportError:
        use_full_ensemble = False
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    
    X_train = X_selected.iloc[:split_idx]
    X_test = X_selected.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    weights_train = weights[:split_idx]
    weights_test = weights[split_idx:]
    
    logger.info(f"📊 Train: {X_train.shape[0]} samples")
    logger.info(f"📊 Test: {X_test.shape[0]} samples")
    
    # Treinar ensemble
    logger.info("\n🔧 Treinando ensemble...")
    
    best_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train, sample_weight=weights_train)
    
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
    
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    ensemble = StackingClassifier(estimators=base_estimators, final_estimator=meta_clf, cv=5, n_jobs=-1)
    
    ensemble.fit(X_train, y_train, sample_weight=weights_train)
    
    # Avaliar
    accuracy = ensemble.score(X_test, y_test, sample_weight=weights_test)
    
    logger.info(f"✅ Ensemble treinado!")
    logger.info(f"   Acurácia: {accuracy*100:.2f}%")
    
    # Salvar modelo e features
    joblib.dump(ensemble, 'data/models/ensemble_model_selected.joblib')
    joblib.dump(X_selected.columns.tolist(), 'data/models/feature_names_selected.joblib')
    
    logger.info(f"💾 Modelo salvo: data/models/ensemble_model_selected.joblib")
    
    # 3. Comparação com baseline
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARAÇÃO COM BASELINE")
    logger.info("="*80)
    
    baseline_acc = 0.7455
    current_acc = 0.7114  # Último treinamento sem seleção
    new_acc = accuracy
    
    logger.info(f"📈 EVOLUÇÃO:")
    logger.info(f"   Baseline (155 features): {baseline_acc*100:.2f}%")
    logger.info(f"   Com novas features (165): {current_acc*100:.2f}% ({(current_acc-baseline_acc)*100:+.2f}%)")
    logger.info(f"   Com feature selection ({X_selected.shape[1]}): {new_acc*100:.2f}% ({(new_acc-baseline_acc)*100:+.2f}%)")
    
    if new_acc > baseline_acc:
        logger.info(f"\n🎉 SUCESSO! Melhoria de {(new_acc-baseline_acc)*100:.2f}% vs baseline!")
    elif new_acc > current_acc:
        logger.info(f"\n✅ MELHOROU! Recuperou {(new_acc-current_acc)*100:.2f}% vs sem seleção")
    else:
        logger.info(f"\n⚠️  Ainda abaixo do baseline ({(baseline_acc-new_acc)*100:.2f}%)")
    
    logger.info("="*80)
    
    # Salvar resultados
    import json
    from pathlib import Path
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'features_before': X.shape[1],
        'features_after': X_selected.shape[1],
        'reduction_pct': 100 * (1 - X_selected.shape[1]/X.shape[1]),
        'baseline_accuracy': float(baseline_acc),
        'no_selection_accuracy': float(current_acc),
        'with_selection_accuracy': float(new_acc),
        'improvement_vs_baseline': float((new_acc - baseline_acc) * 100),
        'improvement_vs_no_selection': float((new_acc - current_acc) * 100),
        'corr_threshold': corr_threshold,
        'importance_threshold': importance_threshold
    }
    
    results_file = Path('data/models/feature_selection_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Resultados salvos em: {results_file}")
    
    elapsed = time.time() - start_time
    logger.info(f"\n⏱️  Tempo total: {elapsed:.1f}s")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Retreinamento com Feature Selection')
    parser.add_argument('--importance-threshold', type=float, default=0.0005,
                       help='Threshold de importância (default: 0.0005)')
    parser.add_argument('--corr-threshold', type=float, default=0.95,
                       help='Threshold de correlação (default: 0.95)')
    
    args = parser.parse_args()
    
    results = retrain_with_selection(
        importance_threshold=args.importance_threshold,
        corr_threshold=args.corr_threshold
    )
    
    # Exit code baseado no resultado
    if results['with_selection_accuracy'] > results['baseline_accuracy']:
        return 0  # Sucesso
    else:
        return 1  # Ainda não bateu baseline

if __name__ == "__main__":
    sys.exit(main())
