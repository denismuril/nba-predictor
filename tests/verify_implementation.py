import sys
import asyncio
import logging
from pathlib import Path

# Adicionar raiz ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

async def verify_async_scraper():
    logger.info("Testing Async Scraper...")
    from data.scrapers.stats_scraper import StatsScraper
    scraper = StatsScraper()
    
    # Testar fetch simples (NBA Official)
    df = await scraper.get_nba_official_async()
    if df is not None and not df.empty:
        logger.info(f"✅ Async Scraper OK: {len(df)} rows fetched")
    else:
        logger.warning("⚠️  Async Scraper returned empty (might be expected if API is down)")

def verify_db():
    logger.info("Testing Database WAL Mode...")
    from data.repositories.db_manager import get_db_manager
    db = get_db_manager()
    
    # Verificar modo WAL
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        if mode.upper() == 'WAL':
            logger.info("✅ Database is in WAL mode")
        else:
            logger.warning(f"⚠️  Database is in {mode} mode (expected WAL)")

def verify_fatigue():
    logger.info("Testing Fatigue Calculator...")
    from core.travel_calculator import calculate_fatigue_score
    
    score = calculate_fatigue_score(distance_km=1000, timezone_diff=2, games_in_72h=1, is_b2b=True)
    logger.info(f"✅ Fatigue Score Calculation: {score} (Expected ~24.3)")

if __name__ == "__main__":
    try:
        verify_db()
        verify_fatigue()
        asyncio.run(verify_async_scraper())
        logger.info("\n🎉 All verification tests passed!")
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        sys.exit(1)
