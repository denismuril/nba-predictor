"""
Sistema de Alertas para Monitoring de Produção

Detecta anomalias e envia alertas quando métricas degradam.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSystem:
    """Sistema de alertas para anomalias em produção."""
    
    def __init__(self, config_path: str = 'monitoring/alert_config.json'):
        self.config_path = Path(config_path)
        self.alerts_log = Path('logs/alerts.log')
        self.alerts_log.parent.mkdir(exist_ok=True)
        
        # Load config ou usar defaults
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()
            self._save_config()
    
    def _default_config(self) -> Dict:
        """Configuração default de thresholds."""
        return {
            'thresholds': {
                'ece_max': 0.08,           # ECE > 8% = alerta
                'brier_max': 0.30,         # Brier > 0.30 = alerta
                'accuracy_min': 0.52,      # Accuracy < 52% = alerta
                'predictions_min': 5,      # < 5 predictions/dia = alerta
                'model_age_days_max': 14,  # Modelo > 14 dias sem retrain = alerta
            },
            'alert_cooldown_hours': 24,   # Não repetir alerta em 24h
            'enabled': True
        }
    
    def _save_config(self):
        """Salva configuração."""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def check_metrics(self, metrics: Dict) -> List[Dict]:
        """
        Verifica métricas e retorna alertas.
        
        Args:
            metrics: Dict com métricas coletadas
        
        Returns:
            Lista de alertas disparados
        """
        if not self.config['enabled']:
            return []
        
        alerts = []
        thresholds = self.config['thresholds']
        
        # Check 1: ECE
        if metrics.get('calibration'):
            ece = metrics['calibration'].get('ece')
            if ece and ece > thresholds['ece_max']:
                alerts.append({
                    'type': 'ECE_HIGH',
                    'severity': 'WARNING',
                    'message': f"ECE alto: {ece:.4f} > {thresholds['ece_max']:.4f}",
                    'value': ece,
                    'threshold': thresholds['ece_max']
                })
        
        # Check 2: Brier Score
        if metrics.get('calibration'):
            brier = metrics['calibration'].get('brier_calibrated')
            if brier and brier > thresholds['brier_max']:
                alerts.append({
                    'type': 'BRIER_HIGH',
                    'severity': 'WARNING',
                    'message': f"Brier Score alto: {brier:.4f} > {thresholds['brier_max']:.4f}",
                    'value': brier,
                    'threshold': thresholds['brier_max']
                })
        
        # Check 3: Accuracy
        if metrics.get('accuracy'):
            acc = metrics['accuracy'].get('test_accuracy')
            if acc and acc < thresholds['accuracy_min']:
                alerts.append({
                    'type': 'ACCURACY_LOW',
                    'severity': 'CRITICAL',
                    'message': f"Accuracy baixo: {acc:.2%} < {thresholds['accuracy_min']:.2%}",
                    'value': acc,
                    'threshold': thresholds['accuracy_min']
                })
        
        # Check 4: Volume de predictions
        pred_count = metrics.get('predictions_count', 0)
        if pred_count < thresholds['predictions_min']:
            alerts.append({
                'type': 'PREDICTIONS_LOW',
                'severity': 'INFO',
                'message': f"Poucas predictions: {pred_count} < {thresholds['predictions_min']}",
                'value': pred_count,
                'threshold': thresholds['predictions_min']
            })
        
        # Check 5: Model age
        if metrics.get('system_health'):
            # TODO: Implementar check de idade do modelo
            pass
        
        return alerts
    
    def log_alert(self, alert: Dict):
        """Registra alerta em log."""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            **alert
        }
        
        # Append to log file
        with open(self.alerts_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to console
        severity_emoji = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }
        emoji = severity_emoji.get(alert['severity'], '❓')
        
        logger.warning(f"{emoji} ALERTA [{alert['severity']}]: {alert['message']}")
    
    def send_alerts(self, alerts: List[Dict]):
        """
        Envia alertas (log por enquanto, pode expandir para email/slack).
        
        Args:
            alerts: Lista de alertas a enviar
        """
        if not alerts:
            return
        
        logger.info(f"📢 {len(alerts)} alerta(s) disparado(s)")
        
        for alert in alerts:
            # Check cooldown
            if self._should_send(alert):
                self.log_alert(alert)
                # TODO: Adicionar email/slack notification
    
    def _should_send(self, alert: Dict) -> bool:
        """Verifica se deve enviar alerta (cooldown)."""
        # TODO: Implementar cooldown logic
        return True
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Retorna alertas das últimas N horas."""
        if not self.alerts_log.exists():
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = []
        
        with open(self.alerts_log, 'r') as f:
            for line in f:
                try:
                    alert = json.loads(line)
                    alert_time = datetime.fromisoformat(alert['timestamp'])
                    
                    if alert_time >= cutoff:
                        recent_alerts.append(alert)
                except:
                    continue
        
        return recent_alerts


if __name__ == '__main__':
    # Demo
    print("🔔 Demo: Sistema de Alertas\n")
    
    alert_system = AlertSystem()
    
    # Simular métricas ruins
    bad_metrics = {
        'calibration': {
            'ece': 0.15,  # Alto!
            'brier_calibrated': 0.35  # Alto!
        },
        'accuracy': {
            'test_accuracy': 0.48  # Baixo!
        },
        'predictions_count': 2  # Baixo!
    }
    
    # Check
    alerts = alert_system.check_metrics(bad_metrics)
    
    print(f"Métricas ruins dispararam {len(alerts)} alertas:\n")
    
    # Enviar
    alert_system.send_alerts(alerts)
    
    print("\n✅ Demo completo!")
    print(f"Alertas salvos em: {alert_system.alerts_log}")
