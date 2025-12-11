"""
Script de recalibração automática do modelo.

Roda diariamente via cron para manter o calibrator atualizado
com os jogos mais recentes.

Usage:
    python scripts/recalibrate_model.py [--lookback-days 30]
"""
import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.calibrator import AutoCalibrator
from monitoring.calibration_monitor import CalibrationMonitor
from data.repositories.db_manager import get_db_manager
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_recent_predictions(lookback_days=60):
    """
    Carrega predictions recentes do banco de dados.
    
    Args:
        lookback_days: Quantos dias olhar para trás
    
    Returns:
        DataFrame com columns: date, y_pred, y_true
    """
    logger.info(f"📂 Carregando predictions dos últimos {lookback_days} dias...")
    
    db = get_db_manager()
    
    # Query para pegar predictions com resultados conhecidos
    # Assumindo que temos tabela 'predictions' com colunas:
    # date, home_team, away_team, predicted_prob, actual_result
    
    query = f"""
    SELECT 
        date,
        predicted_prob_home as y_pred,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END as y_true
    FROM predictions
    WHERE date >= date('now', '-{lookback_days} days')
      AND home_score IS NOT NULL
      AND away_score IS NOT NULL
      AND predicted_prob_home IS NOT NULL
    ORDER BY date
    """
    
    try:
        conn = db.get_connection()
        df = pd.read_sql_query(query, conn)
        db.return_connection(conn)
        
        logger.info(f"✅ Carregados {len(df)} jogos com predictions")
        return df
    
    except Exception as e:
        logger.error(f"❌ Erro ao carregar predictions: {e}")
        return None


def recalibrate(lookback_days=30, min_samples=50):
    """
    Executa recalibração do modelo.
    
    Args:
        lookback_days: Janela de dados para calibração
        min_samples: Mínimo de samples necessários
    """
    logger.info("🔄 Iniciando recalibração automática...")
    
    # Carregar dados
    df = load_recent_predictions(lookback_days=lookback_days * 2)  # Carregar dobro para ter margem
    
    if df is None or len(df) < min_samples:
        logger.warning(
            f"⚠️ Dados insuficientes para recalibração "
            f"(tem {len(df) if df is not None else 0}, precisa {min_samples})"
        )
        return False
    
    # Converter dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Criar calibrator
    calibrator = AutoCalibrator(lookback_days=lookback_days, min_samples=min_samples)
    
    # Fit
    calibrator.fit(
        y_pred=df['y_pred'].values,
        y_true=df['y_true'].values,
        dates=df['date'].values
    )
    
    # Calcular métricas
    y_pred_calibrated = calibrator.predict(df['y_pred'].values)
    ece_before = calibrator.calculate_ece(df['y_pred'].values, df['y_true'].values)
    ece_after = calibrator.calculate_ece(y_pred_calibrated, df['y_true'].values)
    
    logger.info(f"📊 ECE: {ece_before:.4f} → {ece_after:.4f} ({((ece_before-ece_after)/ece_before)*100:+.1f}%)")
    
    # Salvar calibrator
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    calibrator.save(models_dir / 'calibrator.pkl')
    
    # Atualizar monitor
    monitor_path = Path('monitoring/calibration_metrics.json')
    if monitor_path.exists():
        monitor = CalibrationMonitor.load_metrics(monitor_path)
    else:
        monitor = CalibrationMonitor()
    
    monitor.update(
        y_pred_raw=df['y_pred'].values,
        y_true=df['y_true'].values,
        y_pred_calibrated=y_pred_calibrated
    )
    monitor.save_metrics(monitor_path)
    
    # Gerar dashboard
    monitor.plot_dashboard(save_path='monitoring/calibration_dashboard.png')
    
    logger.info("✅ Recalibração completa!")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recalibrar modelo NBA')
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=30,
        help='Dias de lookback para calibração (default: 30)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=50,
        help='Mínimo de samples necessários (default: 50)'
    )
    
    args = parser.parse_args()
    
    success = recalibrate(
        lookback_days=args.lookback_days,
        min_samples=args.min_samples
    )
    
    sys.exit(0 if success else 1)
