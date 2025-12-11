#!/usr/bin/env python
"""
Script para treinar o AutoCalibrator com dados históricos.

Usage:
    python scripts/train_calibrator.py

O script:
1. Carrega o modelo treinado
2. Busca jogos históricos recentes (últimos 60 dias)
3. Gera previsões para esses jogos
4. Treina o calibrador comparando previsões vs resultados reais
5. Salva o calibrador treinado
"""
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.repositories.db_manager import get_db_manager
from ml_pipeline.feature_engineering_v2 import prepare_features_v2
from ml_pipeline.calibrator import AutoCalibrator
from utils.logger_config import get_logger

logger = get_logger(__name__)


def train_calibrator(lookback_days=60, min_samples=50):
    """
    Treina o calibrador com dados históricos recentes.
    
    Args:
        lookback_days: Janela de jogos históricos para treinar (dias)
        min_samples: Mínimo de jogos necessários
        
    Returns:
        AutoCalibrator treinado
    """
    logger.info(f"🎓 Iniciando treinamento do calibrador (últimos {lookback_days} dias)...")
    
    # 1. Carregar modelo
    model_path = Path('data/models/ensemble_model.joblib')
    if not model_path.exists():
        model_path = Path('data/models/ensemble_v7.joblib')
    
    logger.info(f"📂 Carregando modelo: {model_path}")
    model = joblib.load(model_path)
    
    # 2. Buscar jogos históricos
    db = get_db_manager()
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    logger.info(f"📅 Buscando jogos desde {cutoff_date}...")
    
    # Obter histórico completo e filtrar
    df_history = db.get_history()
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    # Filtrar para janela de tempo desejada
    df_history = df_history[df_history['date'] >= cutoff_date].copy()
    
    # Remover jogos sem resultados (home_score ou away_score = NaN/0)
    df_history = df_history.dropna(subset=['home_score', 'away_score'])
    df_history = df_history[(df_history['home_score'] > 0) | (df_history['away_score'] > 0)]
    
    logger.info(f"✅ {len(df_history)} jogos encontrados")
    
    if len(df_history) < min_samples:
        raise ValueError(
            f"Apenas {len(df_history)} jogos disponíveis. "
            f"Mínimo necessário: {min_samples}. "
            f"Tente aumentar lookback_days ou aguardar mais jogos."
        )
    
    # 3. Normalizar nomes de times
    from utils.team_normalization import normalize_team
    df_history['home_team'] = df_history['home_team'].apply(normalize_team)
    df_history['away_team'] = df_history['away_team'].apply(normalize_team)
    df_history = df_history.dropna(subset=['home_team', 'away_team'])
    
    # 4. Gerar features
    logger.info("🔧 Gerando features...")
    df_features = prepare_features_v2(df_history)
    
    if df_features is None or df_features.empty:
        raise ValueError("Falha ao gerar features")
    
    # 5. Preparar X (features) e y (target)
    # Target: 1 se home ganhou, 0 se away ganhou
    y_true = (df_features['home_score'] > df_features['away_score']).astype(int)
    
    # Obter feature names do modelo
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = joblib.load('data/models/feature_names_v7.joblib')
    
    # Construir X
    X = pd.DataFrame(index=df_features.index)
    for feature in feature_names:
        if feature in df_features.columns:
            X[feature] = df_features[feature]
        else:
            # Tentar alias
            alias = feature.replace('points', 'pts') if 'points' in feature else feature.replace('pts', 'points')
            if alias in df_features.columns:
                X[feature] = df_features[alias]
            else:
                X[feature] = 0
    
    # 6. Gerar previsões
    logger.info("🔮 Gerando previsões...")
    y_pred = model.predict_proba(X)[:, 1]
    
    # 7. Treinar Calibrador
    logger.info("🎓 Treinando calibrador...")
    calibrator = AutoCalibrator(lookback_days=lookback_days, min_samples=min_samples)
    calibrator.fit(y_pred, y_true, df_features['date'])
    
    # 8. Estatísticas de validação
    from sklearn.metrics import brier_score_loss, log_loss
    
    y_pred_calibrated = calibrator.predict(y_pred)
    
    brier_before = brier_score_loss(y_true, y_pred)
    brier_after = brier_score_loss(y_true, y_pred_calibrated)
    
    logloss_before = log_loss(y_true, y_pred)
    logloss_after = log_loss(y_true, y_pred_calibrated)
    
    ece_before = calibrator.calculate_ece(y_pred, y_true)
    ece_after = calibrator.calculate_ece(y_pred_calibrated, y_true)
    
    logger.info("\n📊 Resultados do Treinamento:")
    logger.info(f"   Jogos usados: {len(y_true)}")
    logger.info(f"   Brier Score:  {brier_before:.4f} → {brier_after:.4f} ({((brier_before - brier_after) / brier_before * 100):+.1f}%)")
    logger.info(f"   Log Loss:     {logloss_before:.4f} → {logloss_after:.4f} ({((logloss_before - logloss_after) / logloss_before * 100):+.1f}%)")
    logger.info(f"   ECE:          {ece_before:.4f} → {ece_after:.4f} ({((ece_before - ece_after) / ece_before * 100):+.1f}%)")
    
    # 9. Salvar Calibrador
    save_path = Path('data/models/calibrator.pkl')
    calibrator.save(save_path)
    logger.info(f"\n✅ Calibrador salvo em: {save_path}")
    
    return calibrator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinar AutoCalibrator')
    parser.add_argument('--lookback', type=int, default=60, help='Dias de histórico (padrão: 60)')
    parser.add_argument('--min-samples', type=int, default=50, help='Mínimo de jogos (padrão: 50)')
    
    args = parser.parse_args()
    
    try:
        calibrator = train_calibrator(
            lookback_days=args.lookback,
            min_samples=args.min_samples
        )
        print("\n🎉 Treinamento concluído com sucesso!")
    except Exception as e:
        logger.error(f"❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
