"""
Script de Atualização do Dashboard de Monitoramento

Roda após a recalibração para atualizar visualizações.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.calibration_monitor import CalibrationMonitor
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def update_dashboard():
    """Atualiza dashboard de monitoramento."""
    
    logger.info("📊 Atualizando Dashboard de Monitoramento...")
    
    try:
        # Carregar métricas
        metrics_path = Path('monitoring/calibration_metrics.json')
        
        if not metrics_path.exists():
            logger.warning("⚠️ Nenhuma métrica encontrada ainda")
            logger.info("ℹ️  Execute recalibrate_model.py primeiro")
            return False
        
        monitor = CalibrationMonitor.load_metrics(metrics_path)
        
        # Gerar dashboard
        dashboard_path = 'monitoring/calibration_dashboard.png'
        monitor.plot_dashboard(save_path=dashboard_path)
        
        # Sumário
        summary = monitor.get_summary()
        
        logger.info("✅ Dashboard atualizado!")
        logger.info(f"   Checkpoints: {summary.get('total_checkpoints', 0)}")
        logger.info(f"   Samples: {summary.get('total_samples', 0)}")
        
        if 'latest_brier_calibrated' in summary:
            logger.info(f"   Brier (atual): {summary['latest_brier_calibrated']:.4f}")
        
        if 'avg_improvement_pct' in summary:
            logger.info(f"   Melhoria média: {summary['avg_improvement_pct']:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar dashboard: {e}")
        return False


if __name__ == '__main__':
    success = update_dashboard()
    sys.exit(0 if success else 1)
