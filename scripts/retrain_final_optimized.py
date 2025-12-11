#!/usr/bin/env python3
"""
Retreinamento FINAL - Combinando Feature Selection + Hyperparameter Optimization

Combina os melhores resultados de:
- Opção A: Feature selection (86 features selecionadas)
- Opção B: Hyperparameter optimization (hiperparâmetros otimizados)

Usage:
    python scripts/retrain_final_optimized.py
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import time
import pandas as pd
import joblib
from pathlib import Path
import json

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_optimized_hyperparameters():
    """Carrega hiperparâmetros otimizados se disponíveis."""
    params_file = Path('data/models/best_ensemble_params.joblib')
    
    if params_file.exists():
        logger.info(f"✅ Carregando hiperparâmetros otimizados de: {params_file}")
        params = joblib.load(params_file)
        return params
    else:
        logger.warning(f"⚠️  Hiperparâmetros otimizados não encontrados em: {params_file}")
        logger.warning(f"    Usando hiperparâmetros padrão (Opção B ainda rodando?)")
        return None

def retrain_final_model():
    """Retreina modelo final com features selecionadas + hiperparâmetros otimizados."""
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("🏆 RETREINAMENTO FINAL - FASE 1 COMPLETA")
    logger.info("="*80)
    logger.info("📊 Combinando:")
    logger.info("   ✅ Feature Selection (86 features)")
    logger.info("   ⚙️  Hyperparameter Optimization (se disponível)")
    logger.info("="*80)
    
    # 1. Carregar dados
    from ml_pipeline.data_preparation import load_historical_data
    from ml_pipeline.train_ensemble_v3 import ML_SEASONS, ML_SAMPLE_WEIGHT_CONFIG
    
    df, weights = load_historical_data(
        seasons=ML_SEASONS,
        apply_weights=True,
        weight_config=ML_SAMPLE_WEIGHT_CONFIG
    )
    
    # 2. Carregar features selecionadas
    selected_features_file = Path('data/models/selected_features.joblib')
    
    if not selected_features_file.exists():
        logger.error(f"❌ Features selecionadas não encontradas!")
        logger.error(f"    Execute: python scripts/retrain_with_feature_selection.py")
        return None
    
    selected_features = joblib.load(selected_features_file)
    logger.info(f"\n✅ Carregadas {len(selected_features)} features selecionadas")
    
    # 3. Preparar dados
    drop_cols = ['winner', 'correct', 'date', 'prediction', 
                 'home_score', 'away_score', 'pt_diff', 'total_points',
                 'fgm', 'fga', 'fg3m', 'tov', 'oreb', 'dreb', 'fta', 'ftm',
                 'opp_fgm', 'opp_fga', 'opp_fg3m', 'opp_tov', 'opp_oreb', 'opp_dreb', 'opp_fta', 'opp_ftm',
                 'home_efg', 'home_tov_pct', 'home_orb_pct', 'home_ftr',
                 'away_efg', 'away_tov_pct', 'away_orb_pct', 'away_ftr',
                 'prob_home', 'prob_away']
    
    X_full = df.drop(columns=drop_cols, errors='ignore')
    X_full = pd.get_dummies(X_full, columns=['home_team', 'away_team'], drop_first=False)
    
    # Aplicar feature selection
    X = X_full[selected_features]
    y = (df['winner'] == 'HOME').astype(int)
    
    logger.info(f"📊 Features finais: {X.shape[1]}")
    
    # 4. Carregar hiperparâmetros otimizados
    optimized_params = load_optimized_hyperparameters()
    
    # 5. Train/Test split (time series)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    weights_train = weights[:split_idx]
    weights_test = weights[split_idx:]
    
    logger.info(f"\n📊 Dataset:")
    logger.info(f"   Train: {X_train.shape[0]} samples")
    logger.info(f"   Test: {X_test.shape[0]} samples")
    
    # 6. Treinar ensemble com hiperparâmetros
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        use_full_ensemble = True
    except ImportError:
        use_full_ensemble = False
    
    logger.info(f"\n🔧 Construindo ensemble...")
    
    # Aplicar hiperparâmetros otimizados se disponíveis
    if optimized_params and 'random_forest' in optimized_params:
        logger.info(f"⚙️  Usando hiperparâmetros OTIMIZADOS para Random Forest")
        rf_params = optimized_params['random_forest'].copy()
        rf_params['random_state'] = 42
        rf_params['n_jobs'] = -1
        best_rf = RandomForestClassifier(**rf_params)
    else:
        logger.info(f"⚙️  Usando hiperparâmetros PADRÃO para Random Forest")
        best_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    
    best_rf.fit(X_train, y_train, sample_weight=weights_train)
    
    # Base models
    if use_full_ensemble:
        if optimized_params:
            # Usar params otimizados
            xgb_params = optimized_params.get('xgboost', {}).copy() if 'xgboost' in optimized_params else {}
            xgb_params.update({'random_state': 42, 'n_jobs': -1})
            
            lgbm_params = optimized_params.get('lightgbm', {}).copy() if 'lightgbm' in optimized_params else {}
            lgbm_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            
            extra_params = optimized_params.get('extra_trees', {}).copy() if 'extra_trees' in optimized_params else {}
            extra_params.update({'random_state': 42, 'n_jobs': -1})
            
            logger.info(f"⚙️  XGBoost params: {xgb_params}")
            logger.info(f"⚙️  LightGBM params: {lgbm_params}")
            
            base_estimators = [
                ('rf', best_rf),
                ('xgb', XGBClassifier(**xgb_params) if xgb_params else XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)),
                ('lgbm', LGBMClassifier(**lgbm_params) if lgbm_params else LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)),
                ('extra', ExtraTreesClassifier(**extra_params) if extra_params else ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1))
            ]
        else:
            # Params padrão
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
    
    logger.info(f"🔄 Treinando ensemble ({len(base_estimators)} base models)...")
    ensemble.fit(X_train, y_train, sample_weight=weights_train)
    
    # 7. Avaliar
    accuracy = ensemble.score(X_test, y_test, sample_weight=weights_test)
    
    logger.info(f"\n✅ Ensemble treinado!")
    logger.info(f"   Acurácia: {accuracy*100:.2f}%")
    
    # 8. Salvar modelo final
    final_model_path = Path('data/models/ensemble_model_final.joblib')
    joblib.dump(ensemble, final_model_path)
    joblib.dump(selected_features, 'data/models/feature_names_final.joblib')
    
    logger.info(f"\n💾 Modelo FINAL salvo: {final_model_path}")
    logger.info(f"💾 Features FINAL salvas: data/models/feature_names_final.joblib")
    
    # 9. Comparação completa
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARAÇÃO COMPLETA - FASE 1")
    logger.info("="*80)
    
    baseline_acc = 0.7455
    no_features_acc = 0.7114
    with_selection_acc = 0.7275
    final_acc = accuracy
    
    logger.info(f"\n📈 EVOLUÇÃO COMPLETA:")
    logger.info(f"   1️⃣ Baseline (155 features, sem advanced): {baseline_acc*100:.2f}%")
    logger.info(f"   2️⃣ Com advanced features (165): {no_features_acc*100:.2f}% ({(no_features_acc-baseline_acc)*100:+.2f}%)")
    logger.info(f"   3️⃣ Com feature selection (86): {with_selection_acc*100:.2f}% ({(with_selection_acc-baseline_acc)*100:+.2f}%)")
    logger.info(f"   4️⃣ FINAL otimizado ({len(selected_features)}): {final_acc*100:.2f}% ({(final_acc-baseline_acc)*100:+.2f}%)")
    
    if final_acc > baseline_acc:
        delta = (final_acc - baseline_acc) * 100
        logger.info(f"\n🎉🎉🎉 SUCESSO! Melhorou {delta:.2f}% vs baseline!")
        logger.info(f"✅ META FASE 1 ATINGIDA!")
    elif final_acc > with_selection_acc:
        delta = (final_acc - with_selection_acc) * 100
        logger.info(f"\n✅ Hiperparâmetros ajudaram! +{delta:.2f}% vs feature selection")
        delta_baseline = (baseline_acc - final_acc) * 100
        logger.info(f"⚠️  Mas ainda -{delta_baseline:.2f}% vs baseline")
        logger.info(f"💡 Recomendação: Ir para Fase 2 ou ajustar features")
    else:
        logger.info(f"\n⚠️  Hiperparâmetros não melhoraram significativamente")
        logger.info(f"💡 Recomendação: Rollback seletivo ou Fase 2")
    
    logger.info("="*80)
    
    # 10. Salvar resultados finais
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phase': 'Phase 1 Complete',
        'baseline_accuracy': float(baseline_acc),
        'with_advanced_features': float(no_features_acc),
        'with_feature_selection': float(with_selection_acc),
        'final_optimized': float(final_acc),
        'improvement_vs_baseline': float((final_acc - baseline_acc) * 100),
        'features_count': len(selected_features),
        'hyperparameters_optimized': optimized_params is not None,
        'training_time_seconds': time.time() - start_time
    }
    
    results_file = Path('data/models/phase1_final_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Resultados finais salvos: {results_file}")
    logger.info(f"⏱️  Tempo total: {time.time() - start_time:.1f}s")
    
    return results

if __name__ == "__main__":
    try:
        results = retrain_final_model()
        
        if results and results['final_optimized'] > results['baseline_accuracy']:
            sys.exit(0)  # Sucesso!
        else:
            sys.exit(1)  # Não bateu baseline ainda
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
