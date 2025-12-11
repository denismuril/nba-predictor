"""
Validation Script - Measure Real Impact of Phase 1 & 2 Improvements

Compara performance dos modelos V16/V6 com features V2.1 nos últimos 30 dias.

Métricas Validadas:
- Totals MAE (Target: 16.5 → 15.2 pts)
- Moneyline Accuracy (Target: 61% → 63%)
- Spread MAE (Target: 5.8 → 5.0 pts)
- Brier Score (calibração)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_pipeline.data_preparation import load_historical_data
from ml_pipeline.feature_pipeline_v3 import prepare_features_v3
from data.repositories.db_manager import get_db_manager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_models():
    """Carrega modelos V18 (Totals) e V6 (Ensemble/Moneyline)."""
    models_dir = Path('data/models')
    
    totals_model_path = models_dir / 'totals_model_v18.joblib'
    ensemble_model_path = models_dir / 'ensemble_model_v6.joblib'
    
    if not totals_model_path.exists():
        logger.warning(f"⚠️ Totals model não encontrado em {totals_model_path}")
        totals_model = None
    else:
        totals_model = joblib.load(totals_model_path)
        logger.info("✅ Totals Model V18 carregado")
    
    if not ensemble_model_path.exists():
        logger.warning(f"⚠️ Ensemble model não encontrado em {ensemble_model_path}")
        ensemble_model = None
    else:
        ensemble_model = joblib.load(ensemble_model_path)
        logger.info("✅ Ensemble Model V6 carregado")
    
    return totals_model, ensemble_model

    
def get_recent_games(days=90):
    """
    Carrega jogos dos últimos N dias.
    """
    logger.info(f"📊 Carregando jogos dos últimos {days} dias...")
    
    # Usar load_historical_data com raw=False para aplicar pipeline completo (Elo, Pace, etc)
    df = load_historical_data(raw=False)
    
    if df is None or df.empty:
        logger.warning("⚠️ Nenhum jogo encontrado no banco")
        return None
    
    # Garantir datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filtrar últimos N dias
    cutoff_date = datetime.now() - timedelta(days=days)
    df = df[df['date'] >= cutoff_date]
    
    # Filtrar apenas jogos completados (tem winner)
    df = df[df['winner'].notna()]
    
    # Calcular total de pontos se não existir
    if 'total_points' not in df.columns:
        if 'pts' in df.columns and 'opp_pts' in df.columns:
            df['total_points'] = df['pts'] + df['opp_pts']
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            df['total_points'] = df['home_score'] + df['away_score']
        else:
            logger.warning("⚠️ Não foi possível calcular total_points")
    
    # Ordenar por data
    df = df.sort_values('date', ascending=False).reset_index(drop=True)
    
    logger.info(f"✅ {len(df)} jogos carregados para validação")
    return df


def calculate_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_brier_score(y_true, y_prob):
    """Brier Score para probabilidades."""
    return np.mean((y_true - y_prob) ** 2)


def calculate_accuracy(y_true, y_pred):
    """Accuracy classification."""
    return np.mean(y_true == y_pred)


def validate_totals_model(df, model):
    """
    Valida Totals Model V18.
    
    Returns:
        dict com métricas
    """
    if model is None:
        logger.warning("⚠️ Totals model não disponível. Pulando validação.")
        return None
    
    logger.info("\n" + "="*60)
    logger.info("📊 VALIDANDO TOTALS MODEL V18")
    logger.info("="*60)
    
    # Features já preparadas em get_recent_games
    df_features = df.copy()
    
    # Carregar feature names
    # Carregar feature names
    logger.info("🔍 Tentando carregar feature names de: data/models/totals_feature_names_v18.joblib")
    try:
        feature_names = joblib.load('data/models/totals_feature_names_v18.joblib')
        logger.info(f"✅ Feature names carregados: {len(feature_names)} features")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Falha ao carregar feature names V18: {e}")
        # Não podemos prosseguir sem feature names, pois o modelo exige alinhamento exato
        return None
    
    # Preparar X e y
    drop_cols = ['winner', 'date', 'home_team', 'away_team', 'total_points', 
                 'home_score', 'away_score', 'pt_diff', 'point_differential']
    
    # Remover colunas que não existem
    drop_cols = [c for c in drop_cols if c in df_features.columns]
    
    X = df_features.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    
    # Garantir mesmas features do treino
    if feature_names is not None:
        missing_cols = set(feature_names) - set(X.columns)
        if missing_cols:
            logger.warning(f"⚠️ Features faltando: {missing_cols}")
            # Adicionar com 0
            for col in missing_cols:
                X[col] = 0
        
        X = X[feature_names]
    
    # Fill NaNs with 0 to avoid error
    X = X.fillna(0)
    
    y_true = df_features['total_points'].values
    
    # Filter out games without total_points
    mask = ~np.isnan(y_true)
    if np.sum(mask) < len(y_true):
        logger.warning(f"⚠️ Dropping {len(y_true) - np.sum(mask)} games with NaN total_points")
        X = X[mask]
        y_true = y_true[mask]
        
    if len(y_true) == 0:
        logger.warning("❌ No valid games for Totals validation")
        return None

    # Predições
    y_pred = model.predict(X)
    
    # Métricas
    mae = calculate_mae(y_true, y_pred)
    
    # Estatísticas
    errors = np.abs(y_true - y_pred)
    
    results = {
        'model': 'Totals V18',
        'mae': mae,
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'games_analyzed': len(df),
        'target_mae': 15.2,
        'improvement_needed': mae - 15.2,
        'performance_vs_target': 'PASS' if mae <= 15.2 else 'NEEDS IMPROVEMENT'
    }
    
    # Log resultados
    logger.info(f"\n📈 Resultados do Totals Model:")
    logger.info(f"   MAE: {mae:.2f} pontos")
    logger.info(f"   MAE Target: 15.2 pontos")
    logger.info(f"   Status: {results['performance_vs_target']}")
    logger.info(f"   Median Error: {results['median_error']:.2f}")
    logger.info(f"   Std Error: {results['std_error']:.2f}")
    logger.info(f"   Max Error: {results['max_error']:.2f}")
    
    return results


def validate_ensemble_model(df, model):
    """
    Valida Ensemble Model V6 (Moneyline).
    
    Returns:
        dict com métricas
    """
    if model is None:
        logger.warning("⚠️ Ensemble model não disponível. Pulando validação.")
        return None
    
    logger.info("\n" + "="*60)
    logger.info("🎯 VALIDANDO ENSEMBLE MODEL V6 (MONEYLINE)")
    logger.info("="*60)
    
    # Preparar features usando Pipeline V3
    df_features = prepare_features_v3(df.copy())
    
    # Carregar feature names
    logger.info("🔍 Tentando carregar feature names de: data/models/ensemble_feature_names_v6.joblib")
    try:
        feature_names = joblib.load('data/models/ensemble_feature_names_v6.joblib')
        logger.info(f"✅ Feature names carregados: {len(feature_names)} features")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: Falha ao carregar feature names V6: {e}")
        return None
    
    # Preparar X e y
    drop_cols = ['winner', 'date', 'home_team', 'away_team', 'total_points',
                 'home_score', 'away_score', 'pt_diff', 'point_differential',
                 'prediction', 'prob_home', 'prob_away']
    
    drop_cols = [c for c in drop_cols if c in df_features.columns]
    
    X = df_features.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    
    # Garantir mesmas features
    if feature_names is not None:
        missing_cols = set(feature_names) - set(X.columns)
        if missing_cols:
            for col in missing_cols:
                X[col] = 0
        X = X[feature_names]
    
    # Fill NaNs with 0 to avoid error
    X = X.fillna(0)
    
    # Ground truth: HOME win = 1, AWAY win = 0
    y_true = (df_features['winner'] == 'HOME').astype(int).values
    
    # Predições
    y_pred_proba = model.predict_proba(X)[:, 1]  # Prob de HOME win
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    
    # Métricas
    accuracy = calculate_accuracy(y_true, y_pred_class)
    brier = calculate_brier_score(y_true, y_pred_proba)
    
    results = {
        'model': 'Ensemble V6',
        'accuracy': accuracy * 100,
        'brier_score': brier,
        'games_analyzed': len(df),
        'target_accuracy': 63.0,
        'improvement_needed': accuracy * 100 - 63.0,
        'performance_vs_target': 'PASS' if accuracy * 100 >= 63.0 else 'NEEDS IMPROVEMENT'
    }
    
    # Log resultados
    logger.info(f"\n📈 Resultados do Ensemble Model:")
    logger.info(f"   Accuracy: {accuracy*100:.2f}%")
    logger.info(f"   Accuracy Target: 63.0%")
    logger.info(f"   Status: {results['performance_vs_target']}")
    logger.info(f"   Brier Score: {brier:.4f}")
    
    return results


def generate_report(totals_results, ensemble_results):
    """
    Gera relatório consolidado e salva em JSON.
    """
    logger.info("\n" + "="*60)
    logger.info("📄 RELATÓRIO CONSOLIDADO")
    logger.info("="*60)
    
    report = {
        'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': 'Last 30 days',
        'totals_model': totals_results,
        'ensemble_model': ensemble_results,
        'overall_status': 'NEEDS_REVIEW'
    }
    
    # Determinar status geral
    if totals_results and ensemble_results:
        if (totals_results['performance_vs_target'] == 'PASS' and 
            ensemble_results['performance_vs_target'] == 'PASS'):
            report['overall_status'] = 'ALL_TARGETS_MET'
        elif (totals_results['performance_vs_target'] == 'PASS' or
              ensemble_results['performance_vs_target'] == 'PASS'):
            report['overall_status'] = 'PARTIAL_SUCCESS'
        else:
            report['overall_status'] = 'BELOW_TARGETS'
    
    # Salvar JSON
    output_path = Path('data/validation_report.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n💾 Relatório salvo em: {output_path}")
    logger.info(f"\n🎯 Status Geral: {report['overall_status']}")
    
    return report


def main():
    logger.info("🚀 Starting Validation Backtest...")
    
    # 1. Load data
    df = get_recent_games(days=30)
    if df is None:
        return
        
    # 2. Load models
    totals_model, ensemble_model = load_models()
    
    # 3. Validate Totals Model
    totals_results = validate_totals_model(df, totals_model)
    
    # 4. Validar Ensemble Model
    ensemble_results = validate_ensemble_model(df, ensemble_model)
    
    # 5. Gerar relatório
    report = generate_report(totals_results, ensemble_results)
    
    logger.info("\n✅ Validação completa!")
    
    return report


if __name__ == '__main__':
    main()
