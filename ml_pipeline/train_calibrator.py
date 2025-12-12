#!/usr/bin/env python3
"""
Script para treinar o calibrador do NBA Predictor.

Usa dados históricos de previsões para calibrar as probabilidades
usando Isotonic Regression.

Usage:
    python ml_pipeline/train_calibrator.py
"""
import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ml_pipeline.calibrator import AutoCalibrator
from ml_pipeline.data_preparation import load_historical_data
from data.repositories.db_manager import get_db_manager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_historical_predictions(lookback_days: int = 60) -> pd.DataFrame:
    """
    Carrega previsões históricas do banco de dados.

    Args:
        lookback_days: Número de dias para trás

    Returns:
        DataFrame com colunas: date, y_pred, y_true
    """
    db = get_db_manager()

    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')

    # Buscar previsões que já têm resultado
    query = f"""
        SELECT
            date,
            prob_home / 100.0 as y_pred,
            CASE WHEN winner = home_team THEN 1 ELSE 0 END as y_true
        FROM predictions
        WHERE date >= '{cutoff_str}'
        AND winner IS NOT NULL
        AND prob_home IS NOT NULL
    """

    try:
        with db.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        logger.info(f"📊 Carregadas {len(df)} previsões históricas")
        return df
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar previsões do DB: {e}")
        return pd.DataFrame()


def train_from_backtest(lookback_days: int = 60) -> tuple:
    """
    Treina calibrador usando backtest no histórico.

    Simula previsões do modelo V6 nos dados históricos
    e compara com resultados reais.

    Args:
        lookback_days: Número de dias de dados para usar

    Returns:
        Tuple (y_pred, y_true, dates)
    """
    logger.info(f"📊 Carregando dados históricos para backtest...")

    # Carregar modelo V6
    try:
        model = joblib.load('data/models/ensemble_model_v6.joblib')
        feature_names = joblib.load('data/models/feature_names_v6.joblib')
        logger.info("✅ Modelo V6 carregado")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return None, None, None

    # Carregar dados históricos (com features)
    df = load_historical_data(raw=False)

    if df is None or df.empty:
        logger.error("❌ Não foi possível carregar dados históricos")
        return None, None, None

    # Filtrar últimos N dias
    df['date'] = pd.to_datetime(df['date'])
    cutoff = df['date'].max() - timedelta(days=lookback_days)
    df_recent = df[df['date'] >= cutoff].copy()

    logger.info(f"📅 Usando {len(df_recent)} jogos dos últimos {lookback_days} dias")

    # Calcular home_win baseado no score (mais confiável)
    if 'home_score' in df_recent.columns and 'away_score' in df_recent.columns:
        df_recent = df_recent.dropna(subset=['home_score', 'away_score'])
        df_recent['home_win'] = (df_recent['home_score'] > df_recent['away_score']).astype(int)
        logger.info(f"   ✅ home_win calculado via score: {df_recent['home_win'].sum()}/{len(df_recent)} vitórias em casa")
    elif 'winner' in df_recent.columns:
        # winner = 1 significa home ganhou, winner = 0 significa away ganhou
        df_recent = df_recent.dropna(subset=['winner'])
        df_recent['home_win'] = df_recent['winner'].astype(int)
        logger.info(f"   ✅ home_win via coluna winner")
    else:
        logger.error("❌ Colunas 'home_score/away_score' ou 'winner' não encontradas")
        return None, None, None

    # Remover jogos sem resultado
    df_recent = df_recent[df_recent['home_win'].notna()]

    if len(df_recent) < 30:
        logger.error(f"❌ Apenas {len(df_recent)} jogos com resultado válido")
        return None, None, None

    # Preparar features
    missing_features = [f for f in feature_names if f not in df_recent.columns]
    for f in missing_features:
        if 'ortg' in f.lower() or 'drtg' in f.lower():
            df_recent[f] = 112.0
        elif 'elo' in f.lower():
            df_recent[f] = 1500.0
        elif 'pct' in f.lower():
            df_recent[f] = 0.5
        else:
            df_recent[f] = 0.0

    X = df_recent[feature_names].fillna(0)
    y_true = df_recent['home_win'].values
    dates = df_recent['date'].values

    # Prever
    y_pred = model.predict_proba(X)[:, 1]

    logger.info(f"✅ Backtest concluído: {len(y_pred)} previsões")
    logger.info(f"   Prob média: {y_pred.mean():.3f}")
    logger.info(f"   Win rate real: {y_true.mean():.3f}")

    return y_pred, y_true, dates


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("🎯 TREINAMENTO DO CALIBRADOR")
    logger.info("=" * 60)

    lookback_days = 120

    # Tentar carregar previsões do DB primeiro
    df_preds = get_historical_predictions(lookback_days)

    if len(df_preds) >= 50:
        # Usar previsões do DB
        logger.info("📂 Usando previsões salvas no banco de dados")
        y_pred = df_preds['y_pred'].values
        y_true = df_preds['y_true'].values
        dates = pd.to_datetime(df_preds['date']).values
    else:
        # Fazer backtest
        logger.info("🔄 Fazendo backtest no histórico para gerar dados de calibração...")
        y_pred, y_true, dates = train_from_backtest(lookback_days)

        if y_pred is None:
            logger.error("❌ Falha ao gerar dados de calibração")
            return

    # Validar dados
    if len(y_pred) < 50:
        logger.warning(f"⚠️ Apenas {len(y_pred)} amostras. Calibração pode ser instável!")

    # Treinar calibrador
    logger.info("\n🔧 Treinando Isotonic Regression...")
    calibrator = AutoCalibrator(lookback_days=lookback_days, min_samples=30)
    calibrator.fit(y_pred, y_true, dates)

    # Salvar
    output_path = Path('data/models/calibrator.pkl')
    calibrator.save(output_path)

    # Métricas finais
    y_calibrated = calibrator.predict(y_pred)

    ece_before = calibrator.calculate_ece(y_pred, y_true)
    ece_after = calibrator.calculate_ece(y_calibrated, y_true)

    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTADOS")
    logger.info("=" * 60)
    logger.info(f"   Amostras: {len(y_pred)}")
    logger.info(f"   ECE antes:  {ece_before:.4f}")
    logger.info(f"   ECE depois: {ece_after:.4f}")
    if ece_before > 0:
        improvement = ((ece_before - ece_after) / ece_before) * 100
        logger.info(f"   Melhoria:   {improvement:+.1f}%")

    logger.info(f"\n💾 Calibrador salvo em: {output_path}")
    logger.info("✅ Treinamento concluído!")


if __name__ == '__main__':
    main()
