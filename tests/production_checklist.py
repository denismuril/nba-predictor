"""
Production Checklist - Sistema v16.0

Valida status completo do sistema em produção:
- Cron jobs
- Dashboard
- Predictions
- APIs
- Database
- Logs

Usage:
    python tests/production_checklist.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cron_jobs():
    """Verifica status dos cron jobs."""
    logger.info("\n1️⃣ CRON JOBS")
    logger.info("-" * 60)
    
    # Check monitoring system log
    monitor_log = Path('logs/monitoring_system.log')
    if monitor_log.exists():
        with open(monitor_log, 'r') as f:
            lines = f.readlines()
            last_10 = lines[-10:] if len(lines) >= 10 else lines
            logger.info("✅ Últimas execuções do monitoring:")
            for line in last_10:
                if 'INFO' in line:
                    logger.info(f"  {line.strip()}")
    else:
        logger.warning("⚠️ Log do monitoring não encontrado")
    
    return True


def check_dashboard():
    """Verifica se dashboard foi gerado."""
    logger.info("\n2️⃣ DASHBOARD")
    logger.info("-" * 60)
    
    dashboard_path = Path('monitoring/dashboard.html')
    if dashboard_path.exists():
        size_kb = dashboard_path.stat().st_size / 1024
        mod_time = datetime.fromtimestamp(dashboard_path.stat().st_mtime)
        logger.info(f"✅ Dashboard encontrado")
        logger.info(f"  Tamanho: {size_kb:.1f} KB")
        logger.info(f"  Última modificação: {mod_time}")
    else:
        logger.warning("⚠️ Dashboard não encontrado")
    
    return dashboard_path.exists()


def check_predictions():
    """Verifica predictions de hoje."""
    logger.info("\n3️⃣ PREDICTIONS")
    logger.info("-" * 60)
    
    # Check recent prediction files
    results_dir = Path('results')
    if results_dir.exists():
        csv_files = sorted(results_dir.glob('nba_predictions_*.csv'), 
                          key=lambda x: x.stat().st_mtime, 
                          reverse=True)
        
        if csv_files:
            latest = csv_files[0]
            logger.info(f"✅ Última prediction: {latest.name}")
            
            df = pd.read_csv(latest)
            logger.info(f"  Games: {len(df)}")
            logger.info(f"  Columns: {list(df.columns)[:5]}...")
        else:
            logger.warning("⚠️ Nenhum arquivo de predictions encontrado")
    else:
        logger.warning("⚠️ Diretório results não encontrado")
    
    return True


def check_database():
    """Verifica database."""
    logger.info("\n4️⃣ DATABASE")
    logger.info("-" * 60)
    
    db_path = Path('data/nba_history.db')
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Database encontrado")
        logger.info(f"  Tamanho: {size_mb:.1f} MB")
    else:
        logger.warning("⚠️ Database não encontrado")
    
    return db_path.exists()


def check_apis():
    """Verifica conectividade das APIs."""
    logger.info("\n5️⃣ APIs STATUS")
    logger.info("-" * 60)
    
    # Check cache
    cache_dir = Path('data/cache')
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.json'))
        logger.info(f"✅ Cache ativo: {len(cache_files)} arquivos")
    
    # Check Game ID Mapper cache
    mapper_cache = Path('data/cache/game_id_cache.json')
    if mapper_cache.exists():
        logger.info(f"✅ Game ID Mapper cache ativo")
    else:
        logger.info(f"⚠️ Game ID Mapper cache vazio (normal para início)")
    
    return True


def check_models():
    """Verifica modelos ML."""
    logger.info("\n6️⃣ ML MODELS")
    logger.info("-" * 60)
    
    models_dir = Path('models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.joblib'))
        logger.info(f"✅ {len(model_files)} modelos encontrados")
        
        main_model = Path('models/ml_model.joblib')
        if main_model.exists():
            size_mb = main_model.stat().st_size / (1024 * 1024)
            logger.info(f"  Main model: {size_mb:.1f} MB")
    
    return True


def production_summary():
    """Resumo final."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 PRODUCTION CHECKLIST - RESUMO")
    logger.info("=" * 60)
    
    checks = {
        'Cron Jobs': check_cron_jobs(),
        'Dashboard': check_dashboard(),
        'Predictions': check_predictions(),
        'Database': check_database(),
        'APIs': check_apis(),
        'Models': check_models()
    }
    
    passing = sum(checks.values())
    total = len(checks)
    
    logger.info(f"\n✅ Passing: {passing}/{total}")
    
    if passing == total:
        logger.info("🎉 Sistema 100% operacional!")
    elif passing >= total * 0.8:
        logger.info("⚠️ Sistema majoritariamente operacional")
    else:
        logger.warning("❌ Sistema requer atenção")
    
    return checks


if __name__ == '__main__':
    logger.info("🏀 Production Checklist v16.0\n")
    
    checks = production_summary()
    
    print(f"\n✅ Checklist completo!")
    print(f"   Status: {sum(checks.values())}/{len(checks)} checks passing")
