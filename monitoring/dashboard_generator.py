"""
Dashboard HTML para Monitoring de Produção

Gera dashboard visual com métricas em tempo real.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
import base64

logger = logging.getLogger(__name__)


def generate_html_dashboard(metrics_file: str = 'monitoring/daily_metrics.json') -> str:
    """
    Gera dashboard HTML com métricas.
    
    Args:
        metrics_file: Path para arquivo de métricas diárias
    
    Returns:
        HTML string
    """
    # Load metrics
    metrics_path = Path(metrics_file)
    
    if not metrics_path.exists():
        return generate_empty_dashboard()
    
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)
    
    if not all_metrics:
        return generate_empty_dashboard()
    
    # Get latest
    latest = all_metrics[-1]
    
    # Calculate trends (last 7 days)
    recent = all_metrics[-7:] if len(all_metrics) >= 7 else all_metrics
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Predictor - Production Monitoring</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            opacity: 0.8;
            font-size: 1.1rem;
        }}
        
        .header .timestamp {{
            margin-top: 10px;
            opacity: 0.6;
            font-size: 0.9rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-card .label {{
            font-size: 0.9rem;
            opacity: 0.7;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .metric-card .trend {{
            font-size: 0.85rem;
            opacity: 0.8;
        }}
        
        .metric-card.good {{
            border-left: 4px solid #4caf50;
        }}
        
        .metric-card.warning {{
            border-left: 4px solid #ff9800;
        }}
        
        .metric-card.critical {{
            border-left: 4px solid #f44336;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-indicator.good {{
            background: #4caf50;
            box-shadow: 0 0 10px #4caf50;
        }}
        
        .status-indicator.warning {{
            background: #ff9800;
            box-shadow: 0 0 10px #ff9800;
        }}
        
        .status-indicator.critical {{
            background: #f44336;
            box-shadow: 0 0 10px #f44336;
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 30px;
        }}
        
        .system-health {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .health-item {{
            display: flex;
            align-items: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            opacity: 0.6;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏀 NBA Predictor</h1>
            <div class="subtitle">Production Monitoring Dashboard</div>
            <div class="timestamp">Última atualização: {latest.get('timestamp', 'N/A')}</div>
        </div>
        
        <div class="metrics-grid">
            {generate_calibration_card(latest)}
            {generate_predictions_card(latest)}
            {generate_accuracy_card(latest)}
            {generate_system_card(latest)}
        </div>
        
        <div class="chart-container">
            <h2>📊 Histórico (Últimos 7 dias)</h2>
            {generate_trend_info(recent)}
        </div>
        
        <div class="chart-container">
            <h2>🏥 System Health</h2>
            <div class="system-health">
                {generate_health_items(latest)}
            </div>
        </div>
        
        <div class="footer">
            © 2025 NBA Predictor | Auto-generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
    """
    
    return html


def generate_calibration_card(metrics: dict) -> str:
    """Gera card de calibração."""
    cal = metrics.get('calibration', {})
    
    if not cal:
        return """
        <div class="metric-card">
            <div class="label">Calibração</div>
            <div class="value">N/A</div>
            <div class="trend">Sem dados</div>
        </div>
        """
    
    brier = cal.get('brier_calibrated', 0)
    improvement = cal.get('improvement_pct', 0)
    
    # Determine status
    status_class = 'good' if brier < 0.10 else ('warning' if brier < 0.20 else 'critical')
    
    return f"""
    <div class="metric-card {status_class}">
        <div class="label">
            <span class="status-indicator {status_class}"></span>
            Brier Score
        </div>
        <div class="value">{brier:.4f}</div>
        <div class="trend">Melhoria: {improvement:.1f}%</div>
    </div>
    """


def generate_predictions_card(metrics: dict) -> str:
    """Gera card de predictions."""
    count = metrics.get('predictions_count', 0)
    
    status_class = 'good' if count >= 5 else ('warning' if count >= 2 else 'critical')
    
    return f"""
    <div class="metric-card {status_class}">
        <div class="label">
            <span class="status-indicator {status_class}"></span>
            Predictions Hoje
        </div>
        <div class="value">{count}</div>
        <div class="trend">Jogos processados</div>
    </div>
    """


def generate_accuracy_card(metrics: dict) -> str:
    """Gera card de accuracy."""
    acc = metrics.get('accuracy', {})
    
    if not acc:
        return """
        <div class="metric-card">
            <div class="label">Accuracy</div>
            <div class="value">N/A</div>
            <div class="trend">Aguardando dados</div>
        </div>
        """
    
    test_acc = acc.get('test_accuracy', 0)
    status_class = 'good' if test_acc >= 0.58 else ('warning' if test_acc >= 0.52 else 'critical')
    
    return f"""
    <div class="metric-card {status_class}">
        <div class="label">
            <span class="status-indicator {status_class}"></span>
            Test Accuracy
        </div>
        <div class="value">{test_acc:.1%}</div>
        <div class="trend">Últimas validações</div>
    </div>
    """


def generate_system_card(metrics: dict) -> str:
    """Gera card de system health."""
    health = metrics.get('system_health', {})
    
    model_ok = health.get('model_exists', False)
    cal_ok = health.get('calibrator_exists', False)
    
    status = 'good' if (model_ok and cal_ok) else ('warning' if model_ok else 'critical')
    
    return f"""
    <div class="metric-card {status}">
        <div class="label">
            <span class="status-indicator {status}"></span>
            System Status
        </div>
        <div class="value">{'✅ OK' if status == 'good' else '⚠️ CHECK'}</div>
        <div class="trend">Modelo: {'✓' if model_ok else '✗'} | Calibrator: {'✓' if cal_ok else '✗'}</div>
    </div>
    """


def generate_trend_info(recent_metrics: list) -> str:
    """Gera informação de trends."""
    if len(recent_metrics) < 2:
        return "<p>Dados insuficientes para análise de tendências</p>"
    
    # Extract brier scores
    brier_scores = [
        m.get('calibration', {}).get('brier_calibrated')
        for m in recent_metrics
        if m.get('calibration', {}).get('brier_calibrated')
    ]
    
    if not brier_scores:
        return "<p>Sem dados de calibração</p>"
    
    avg_brier = sum(brier_scores) / len(brier_scores)
    
    return f"""
    <p style="font-size: 1.1rem; margin-bottom: 15px;">
        Média Brier Score (7d): <strong>{avg_brier:.4f}</strong>
    </p>
    <p style="opacity: 0.8;">
        Dias com dados: {len(brier_scores)}/7<br>
        Melhor: {min(brier_scores):.4f} | Pior: {max(brier_scores):.4f}
    </p>
    """


def generate_health_items(metrics: dict) -> str:
    """Gera items de health."""
    health = metrics.get('system_health', {})
    logs = metrics.get('logs', {})
    
    items = [
        ('Modelo ML', '✅' if health.get('model_exists') else '❌'),
        ('Calibrator', '✅' if health.get('calibrator_exists') else '❌'),
        ('Recalibração', '✅' if logs.get('recalibration_ran') else '❌'),
        ('Monitoring', '✅' if logs.get('monitoring_ran') else '❌'),
    ]
    
    html = ""
    for label, status in items:
        html += f"""
        <div class="health-item">
            <span style="font-size: 1.5rem; margin-right: 10px;">{status}</span>
            <span>{label}</span>
        </div>
        """
    
    return html


def generate_empty_dashboard() -> str:
    """Gera dashboard vazio."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>NBA Predictor Monitoring</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            padding: 50px;
            background: #1e3c72;
            color: white;
        }
    </style>
</head>
<body>
    <h1>🏀 NBA Predictor Monitoring</h1>
    <p>Aguardando primeiras métricas...</p>
    <p>Execute: <code>python monitoring/daily_check.py</code></p>
</body>
</html>
    """


def save_dashboard(output_path: str = 'monitoring/dashboard.html'):
    """Gera e salva dashboard."""
    html = generate_html_dashboard()
    
    output = Path(output_path)
    output.parent.mkdir(exist_ok=True)
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"📊 Dashboard salvo: {output}")
    return output


if __name__ == '__main__':
    # Generate dashboard
    dashboard_path = save_dashboard()
    print(f"✅ Dashboard gerado: {dashboard_path}")
    print(f"Abra em: file://{dashboard_path.absolute()}")
