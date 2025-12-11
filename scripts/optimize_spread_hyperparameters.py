"""
Hyperparameter Optimization for Spread Model

Performs grid search with cross-validation to find optimal XGBoost parameters
that minimize MAE for spread predictions.

Target: Reduce MAE from 8.46 pts to ~6 pts
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.train_spread_real import prepare_features_v2


def optimize_spread_hyperparameters():
    """
    Otimiza hiperparâmetros do modelo Spread usando Grid Search.
    
    Returns:
        best_params: Dict com os melhores hiperparâmetros
        best_score: MAE do melhor modelo
    """
    logger.info("="*80)
    logger.info("🔍 OTIMIZAÇÃO DE HIPERPARÂMETROS - SPREAD MODEL")
    logger.info("="*80)
    
    # Carregar dados
    logger.info("📊 Carregando dados históricos...")
    df, weights = load_historical_data(
        seasons=['2023-24', '2024-25', '2025-26'],
        apply_weights=True
    )
    
    if df is None or df.empty:
        logger.error("❌ Falha ao carregar dados!")
        return None, None
    
    logger.info(f"   Loaded {len(df)} games")
    
    # Preparar features
    X, y = prepare_features_v2(df)
    logger.info(f"   Features shape: {X.shape}")
    
    # Time Series Split para validação (melhor que KFold para dados temporais)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid de hiperparâmetros focado em MAE
    param_grid = {
        'n_estimators': [300, 500, 700],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.5],
        'min_child_weight': [1, 3, 5]
    }
    
    logger.info(f"🔬 Grid Search com {np.prod([len(v) for v in param_grid.values()])} combinações")
    logger.info(f"   Estimativa de tempo: ~15-30 minutos")
    logger.info(f"   Cross-validation: {tscv.n_splits} splits (Time Series)")
    
    # Modelo base
    base_model = XGBRegressor(
        n_jobs=-1,
        random_state=42,
        objective='reg:absoluteerror',
        eval_metric='mae'
    )
    
    # Grid Search com MAE negativo (sklearn maximiza, então usamos neg_mean_absolute_error)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    logger.info("🏋️‍♂️ Iniciando Grid Search...")
    start_time = datetime.now()
    
    # Fit com sample weights
    grid_search.fit(X, y, sample_weight=weights)
    
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"✅ Grid Search completado em {duration/60:.1f} minutos")
    
    # Melhores resultados
    best_params = grid_search.best_params_
    best_mae = -grid_search.best_score_  # Converter de volta para positivo
    
    logger.info("="*80)
    logger.info("🏆 MELHORES HIPERPARÂMETROS ENCONTRADOS:")
    logger.info("="*80)
    for param, value in best_params.items():
        logger.info(f"   {param:20s}: {value}")
    logger.info(f"   {'MAE (CV)':20s}: {best_mae:.2f} pontos")
    logger.info("="*80)
    
    # Salvar hiperparâmetros
    models_dir = Path('data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    params_file = models_dir / 'best_hyperparameters.joblib'
    joblib.dump(best_params, params_file)
    logger.info(f"💾 Hiperparâmetros salvos em: {params_file}")
    
    # Salvar detalhes do grid search
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    results_file = models_dir / 'grid_search_results.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"📊 Resultados detalhados salvos em: {results_file}")
    
    # Top 5 configurações
    logger.info("\n📈 Top 5 Configurações:")
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        mae_val = -row['mean_test_score']
        logger.info(f"   #{idx+1}: MAE={mae_val:.2f} pts | Params: {row['params']}")
    
    return best_params, best_mae


def retrain_with_best_params():
    """
    Retreina o modelo Spread com os hiperparâmetros otimizados.
    """
    logger.info("\n" + "="*80)
    logger.info("🔄 RETREINANDO MODELO COM HIPERPARÂMETROS OTIMIZADOS")
    logger.info("="*80)
    
    # Importar e executar treino
    from ml_pipeline.train_spread_real import train_spread_model_real
    
    model, mae, rmse = train_spread_model_real()
    
    if model is not None:
        logger.info("="*80)
        logger.info("✅ MODELO SPREAD OTIMIZADO TREINADO COM SUCESSO!")
        logger.info(f"   MAE Final: {mae:.2f} pontos")
        logger.info(f"   RMSE Final: {rmse:.2f} pontos")
        logger.info("="*80)
        
        # Comparar com baseline
        baseline_mae = 8.46
        improvement = baseline_mae - mae
        improvement_pct = (improvement / baseline_mae) * 100
        
        if mae < baseline_mae:
            logger.info(f"🎉 MELHORIA: {improvement:.2f} pts ({improvement_pct:.1f}%)")
            if mae <= 6.5:
                logger.info("🏆 META ATINGIDA: MAE ≤ 6.5 pts!")
        else:
            logger.warning(f"⚠️  MAE piorou em {abs(improvement):.2f} pts")
    
    return model, mae, rmse


if __name__ == "__main__":
    # Executar otimização
    best_params, best_mae = optimize_spread_hyperparameters()
    
    if best_params is not None:
        # Retreinar com os melhores parâmetros
        print("\n" + "="*80)
        input("Pressione ENTER para retreinar o modelo com os parâmetros otimizados...")
        retrain_with_best_params()
