"""
Prefect Flows - NBA Predictor v24.0
====================================
Orquestração profissional de workflows com Prefect.

Flows:
- daily_pipeline: Pipeline diário completo
- paper_trading_flow: Captura de sinais para paper trading
- settlement_flow: Liquidação de apostas paper

Deploy:
    prefect deploy --all
    prefect server start
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.concurrency.sync import rate_limit

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# TASKS - Unidades atômicas de trabalho
# =============================================================================

@task(
    name="fetch_games",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    tags=["data", "fetch"]
)
def fetch_todays_games(date_str: str = None) -> dict:
    """Busca jogos do dia via API."""
    logger = get_run_logger()
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"📅 Buscando jogos para {date_str}")
    
    try:
        from data.repositories.db_manager import get_db_manager
        from scripts.fetch_todays_games import fetch_games_for_date
        
        games = fetch_games_for_date(date_str)
        logger.info(f"✅ {len(games)} jogos encontrados")
        return {"date": date_str, "games_count": len(games), "status": "success"}
    except Exception as e:
        logger.error(f"❌ Erro buscando jogos: {e}")
        return {"date": date_str, "games_count": 0, "status": "error", "error": str(e)}


@task(
    name="fetch_odds",
    retries=2,
    retry_delay_seconds=120,
    tags=["data", "odds"]
)
def fetch_odds() -> dict:
    """Busca odds atualizadas."""
    logger = get_run_logger()
    rate_limit("odds-api", occupy=1)  # Rate limit para API
    
    logger.info("💰 Buscando odds...")
    
    try:
        from data.scrapers.odds_scraper import obter_odds
        
        odds = obter_odds()
        count = len(odds) if odds else 0
        logger.info(f"✅ {count} odds obtidas")
        return {"odds_count": count, "status": "success"}
    except Exception as e:
        logger.error(f"❌ Erro buscando odds: {e}")
        return {"odds_count": 0, "status": "error", "error": str(e)}


@task(
    name="run_predictions",
    retries=1,
    tags=["ml", "predictions"]
)
def run_predictions(date_str: str = None) -> dict:
    """Executa previsões do modelo."""
    logger = get_run_logger()
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"🤖 Rodando previsões para {date_str}")
    
    try:
        from ml_pipeline import pipeline
        
        predictions = pipeline.predict(date_str)
        count = len(predictions) if predictions is not None else 0
        logger.info(f"✅ {count} previsões geradas")
        return {"predictions_count": count, "status": "success"}
    except Exception as e:
        logger.error(f"❌ Erro nas previsões: {e}")
        return {"predictions_count": 0, "status": "error", "error": str(e)}


@task(
    name="send_telegram_alerts",
    retries=2,
    retry_delay_seconds=30,
    tags=["notifications"]
)
def send_telegram_alerts(predictions: dict) -> dict:
    """Envia alertas para Telegram."""
    logger = get_run_logger()
    
    if predictions.get("status") != "success":
        logger.warning("⚠️ Sem previsões para enviar")
        return {"sent": 0, "status": "skipped"}
    
    logger.info("📱 Enviando alertas Telegram...")
    
    try:
        from telegram_bot.nba_tigrinho_bot import send_daily_predictions
        
        sent = send_daily_predictions()
        logger.info(f"✅ {sent} alertas enviados")
        return {"sent": sent, "status": "success"}
    except Exception as e:
        logger.error(f"❌ Erro Telegram: {e}")
        return {"sent": 0, "status": "error", "error": str(e)}


@task(
    name="paper_trading_capture",
    tags=["betting", "paper"]
)
async def capture_paper_bets() -> dict:
    """Captura sinais para paper trading."""
    logger = get_run_logger()
    
    # Check stop file
    stop_file = Path('data/.STOP_ALL_BETS')
    if stop_file.exists():
        logger.warning("🛑 STOP_ALL_BETS ativo - pulando captura")
        return {"bets": 0, "status": "stopped"}
    
    logger.info("📝 Iniciando captura paper trading...")
    
    try:
        from betting.paper_trading import PaperTradingEngine
        
        engine = PaperTradingEngine(bankroll=1000.0)
        await engine.initialize()
        
        # Capturar por 5 minutos (durante janela de apostas)
        import asyncio
        await asyncio.sleep(300)
        
        stats = await engine.db.get_stats(1)
        await engine.close()
        
        logger.info(f"✅ {stats['total_bets']} bets capturados")
        return {"bets": stats['total_bets'], "status": "success"}
    except Exception as e:
        logger.error(f"❌ Erro paper trading: {e}")
        return {"bets": 0, "status": "error", "error": str(e)}


@task(
    name="settle_paper_bets",
    tags=["betting", "settlement"]
)
async def settle_bets(date_str: str = None) -> dict:
    """Liquida paper bets do dia anterior."""
    logger = get_run_logger()
    
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"💰 Liquidando bets de {date_str}")
    
    try:
        from betting.settle_paper_bets import PaperBetSettler
        
        settler = PaperBetSettler()
        await settler.initialize()
        
        stats = await settler.settle_date(date_str)
        await settler.close()
        
        logger.info(f"✅ {stats['settled']} bets liquidados, PnL: R$ {stats['pnl']:+.2f}")
        return {
            "settled": stats['settled'],
            "pnl": stats['pnl'],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"❌ Erro settlement: {e}")
        return {"settled": 0, "pnl": 0, "status": "error", "error": str(e)}


@task(
    name="health_check",
    tags=["monitoring"]
)
def system_health_check() -> dict:
    """Verifica saúde do sistema."""
    logger = get_run_logger()
    
    health = {
        "postgres": False,
        "redis": False,
        "models": False,
        "timestamp": datetime.now().isoformat()
    }
    
    # PostgreSQL
    try:
        from data.repositories.db_manager import get_db_manager
        db = get_db_manager()
        db.get_comprehensive_history()
        health["postgres"] = True
    except Exception:
        pass
    
    # Redis
    try:
        import redis
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        r.ping()
        health["redis"] = True
    except Exception:
        pass
    
    # Models
    try:
        model_path = Path('data/models/ensemble_model.pkl')
        health["models"] = model_path.exists()
    except Exception:
        pass
    
    status = "healthy" if all([health["postgres"], health["models"]]) else "degraded"
    health["status"] = status
    
    logger.info(f"🏥 Health: {status} | PG: {health['postgres']} | Redis: {health['redis']}")
    return health


# =============================================================================
# FLOWS - Pipelines completos
# =============================================================================

@flow(
    name="NBA Daily Pipeline",
    description="Pipeline diário completo: fetch → predict → alert",
    retries=1,
    retry_delay_seconds=300
)
def daily_pipeline(date_str: str = None):
    """
    Flow principal - executa todo o pipeline diário.
    
    Schedule: Diariamente às 17:00 BRT
    """
    logger = get_run_logger()
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"🏀 Iniciando Daily Pipeline - {date_str}")
    
    # Health check primeiro
    health = system_health_check()
    if health["status"] != "healthy":
        logger.warning("⚠️ Sistema degradado, continuando com cautela...")
    
    # Fetch games
    games_result = fetch_todays_games(date_str)
    
    # Fetch odds
    odds_result = fetch_odds()
    
    # Run predictions
    predictions = run_predictions(date_str)
    
    # Send alerts
    alerts = send_telegram_alerts(predictions)
    
    logger.info(f"✅ Pipeline concluído: {predictions['predictions_count']} previsões")
    
    return {
        "date": date_str,
        "games": games_result,
        "odds": odds_result,
        "predictions": predictions,
        "alerts": alerts,
        "health": health
    }


@flow(
    name="Paper Trading Flow",
    description="Captura sinais para paper trading durante janela de apostas"
)
async def paper_trading_flow():
    """
    Flow de paper trading - captura sinais do Sniper.
    
    Schedule: Diariamente 18:00-19:00 BRT (janela pré-jogo)
    """
    logger = get_run_logger()
    logger.info("🎮 Iniciando Paper Trading Flow")
    
    # Health check
    health = system_health_check()
    
    # Capture bets
    result = await capture_paper_bets()
    
    logger.info(f"📝 Paper Trading concluído: {result['bets']} sinais capturados")
    return result


@flow(
    name="Settlement Flow",
    description="Liquida paper bets do dia anterior e gera relatório"
)
async def settlement_flow(date_str: str = None):
    """
    Flow de liquidação - processa resultados.
    
    Schedule: Diariamente às 09:00 BRT
    """
    logger = get_run_logger()
    
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"💰 Iniciando Settlement Flow - {date_str}")
    
    result = await settle_bets(date_str)
    
    if result["status"] == "success":
        logger.info(f"✅ Settlement: {result['settled']} bets, PnL: R$ {result['pnl']:+.2f}")
    else:
        logger.error(f"❌ Settlement falhou: {result.get('error')}")
    
    return result


@flow(
    name="Morning Health Check",
    description="Verificação matinal de saúde do sistema"
)
def morning_health_flow():
    """
    Flow de health check matinal.
    
    Schedule: Diariamente às 09:00 BRT
    """
    logger = get_run_logger()
    logger.info("🌅 Morning Health Check")
    
    health = system_health_check()
    
    if health["status"] != "healthy":
        logger.error("🚨 Sistema com problemas! Verificar manualmente.")
        # Aqui poderia enviar alerta Telegram
    
    return health


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Predictor Prefect Flows')
    parser.add_argument('flow', choices=['daily', 'paper', 'settle', 'health'],
                       help='Flow to run')
    parser.add_argument('--date', type=str, help='Date YYYY-MM-DD')
    args = parser.parse_args()
    
    if args.flow == 'daily':
        daily_pipeline(args.date)
    elif args.flow == 'paper':
        import asyncio
        asyncio.run(paper_trading_flow())
    elif args.flow == 'settle':
        import asyncio
        asyncio.run(settlement_flow(args.date))
    elif args.flow == 'health':
        morning_health_flow()
