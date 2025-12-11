"""
Script de Monitoramento de Produção - 7 Dias

Coleta métricas diárias e gera relatórios semanais.

Usage:
    python monitoring/daily_check.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionMonitor:
    """Monitor de produção para tracking contínuo."""
    
    def __init__(self, metrics_dir: str = 'monitoring'):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.daily_metrics_file = self.metrics_dir / 'daily_metrics.json'
        self.weekly_report_file = self.metrics_dir / 'weekly_report.json'
    
    def collect_daily_metrics(self) -> Dict:
        """Coleta métricas do dia."""
        
        logger.info("📊 Coletando métricas diárias...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
        }
        
        # 1. Calibrator metrics
        try:
            from monitoring.calibration_monitor import CalibrationMonitor
            monitor = CalibrationMonitor.load_metrics('monitoring/calibration_metrics.json')
            latest = monitor.get_latest_metrics()
            
            if latest:
                metrics['calibration'] = {
                    'brier_calibrated': latest.get('brier_calibrated'),
                    'improvement_pct': latest.get('improvement_pct'),
                    'n_samples': latest.get('n_samples')
                }
                logger.info(f"   ✅ Calibration: Brier={latest.get('brier_calibrated', 0):.4f}")
        except Exception as e:
            logger.warning(f"   ⚠️ Calibration metrics error: {e}")
            metrics['calibration'] = None
        
        # 2. Model predictions count
        try:
            # TODO: Query DB para contar predictions hoje
            metrics['predictions_count'] = 0  # Placeholder
            logger.info(f"   ✅ Predictions: {metrics['predictions_count']}")
        except Exception as e:
            logger.warning(f"   ⚠️ Predictions count error: {e}")
            metrics['predictions_count'] = None
        
        # 3. System health
        try:
            import joblib
            model_path = Path('models/ml_model.joblib')
            calibrator_path = Path('models/calibrator.pkl')
            
            metrics['system_health'] = {
                'model_exists': model_path.exists(),
                'calibrator_exists': calibrator_path.exists(),
                'model_size_mb': model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0
            }
            logger.info("   ✅ System health OK")
        except Exception as e:
            logger.warning(f"   ⚠️ System health error: {e}")
            metrics['system_health'] = None
        
        # 4. Logs check
        try:
            log_dir = Path('logs')
            today = datetime.now().strftime('%Y-%m-%d')
            
            metrics['logs'] = {
                'recalibration_ran': (log_dir / 'recalibration.log').exists(),
                'monitoring_ran': (log_dir / 'monitoring.log').exists(),
            }
            logger.info("   ✅ Logs checked")
        except Exception as e:
            logger.warning(f"   ⚠️ Logs check error: {e}")
            metrics['logs'] = None
        
        return metrics
    
    def save_daily_metrics(self, metrics: Dict):
        """Salva métricas diárias."""
        
        # Load existing
        if self.daily_metrics_file.exists():
            with open(self.daily_metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Append
        all_metrics.append(metrics)
        
        # Keep only last 30 days
        if len(all_metrics) > 30:
            all_metrics = all_metrics[-30:]
        
        # Save
        with open(self.daily_metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        logger.info(f"💾 Métricas salvas: {self.daily_metrics_file}")
    
    def generate_weekly_report(self) -> Dict:
        """Gera relatório semanal."""
        
        logger.info("📋 Gerando relatório semanal...")
        
        # Load daily metrics
        if not self.daily_metrics_file.exists():
            logger.warning("⚠️ Sem métricas diárias para relatório")
            return {}
        
        with open(self.daily_metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        # Filter last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        recent_metrics = [
            m for m in all_metrics
            if datetime.fromisoformat(m['timestamp']) >= cutoff
        ]
        
        if not recent_metrics:
            logger.warning("⚠️ Sem métricas dos últimos 7 dias")
            return {}
        
        # Calculate summary
        report = {
            'period_start': recent_metrics[0]['date'],
            'period_end': recent_metrics[-1]['date'],
            'days_tracked': len(recent_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        # Calibration summary
        brier_scores = [
            m['calibration']['brier_calibrated']
            for m in recent_metrics
            if m.get('calibration') and m['calibration'].get('brier_calibrated')
        ]
        
        if brier_scores:
            report['calibration'] = {
                'avg_brier': sum(brier_scores) / len(brier_scores),
                'min_brier': min(brier_scores),
                'max_brier': max(brier_scores),
                'days_with_data': len(brier_scores)
            }
        
        # System health
        health_checks = [
            m['system_health']
            for m in recent_metrics
            if m.get('system_health')
        ]
        
        if health_checks:
            report['system_health'] = {
                'days_model_ok': sum(1 for h in health_checks if h.get('model_exists')),
                'days_calibrator_ok': sum(1 for h in health_checks if h.get('calibrator_exists'))
            }
        
        # Save report
        with open(self.weekly_report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Relatório semanal salvo: {self.weekly_report_file}")
        
        return report
    
    def print_report(self, report: Dict):
        """Imprime relatório formatado."""
        
        print("\n" + "="*60)
        print("📊 RELATÓRIO SEMANAL DE PRODUÇÃO")
        print("="*60)
        print(f"Período: {report.get('period_start')} a {report.get('period_end')}")
        print(f"Dias rastreados: {report.get('days_tracked')}\n")
        
        if 'calibration' in report:
            cal = report['calibration']
            print("🎯 Calibração:")
            print(f"  Brier médio: {cal['avg_brier']:.4f}")
            print(f"  Brier min: {cal['min_brier']:.4f}")
            print(f"  Brier max: {cal['max_brier']:.4f}")
            print(f"  Dias com dados: {cal['days_with_data']}/7\n")
        
        if 'system_health' in report:
            health = report['system_health']
            print("🏥 System Health:")
            print(f"  Modelo OK: {health['days_model_ok']}/{report['days_tracked']} dias")
            print(f"  Calibrator OK: {health['days_calibrator_ok']}/{report['days_tracked']} dias\n")
        
        print("="*60)


def run_daily_check():
    """Executa check diário."""
    
    logger.info("🔍 Executando check diário de produção...")
    
    monitor = ProductionMonitor()
    
    # 1. Collect metrics
    metrics = monitor.collect_daily_metrics()
    
    # 2. Save metrics
    monitor.save_daily_metrics(metrics)
    
    # 3. Check for alerts
    try:
        from monitoring.alert_system import AlertSystem
        alert_system = AlertSystem()
        alerts = alert_system.check_metrics(metrics)
        
        if alerts:
            alert_system.send_alerts(alerts)
        else:
            logger.info("✅ Todas as métricas dentro dos thresholds")
    except Exception as e:
        logger.warning(f"⚠️ Erro no sistema de alertas: {e}")
    
    # 4. Generate dashboard
    try:
        from monitoring.dashboard_generator import save_dashboard
        dashboard_path = save_dashboard()
        logger.info(f"📊 Dashboard atualizado: {dashboard_path}")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao gerar dashboard: {e}")
    
    # 5. Check if should generate weekly report (domingo)
    if datetime.now().weekday() == 6:  # Domingo
        logger.info("📅 Domingo - gerando relatório semanal...")
        report = monitor.generate_weekly_report()
        monitor.print_report(report)
    
    logger.info("✅ Check diário completo!")


if __name__ == '__main__':
    run_daily_check()
