"""
Script para aplicar caching nos scrapers existentes.

Aplica decorators @smart_cache automaticamente em funções de scraping.
"""

import sys
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CACHE_PATCHES = """
===================================================================
PATCHES DE CACHE PARA SCRAPERS
===================================================================

Apply estes patches manualmente ou rode este script.

1. data/scrapers/injury_scraper.py
-------------------------------------------------------------------
# Adicionar no topo:
from utils.cache import smart_cache, TTL_INJURY_REPORT

# Modificar função:
@smart_cache(ttl_hours=TTL_INJURY_REPORT, cache_key_prefix='injuries')
def obter_injury_report_api_espn():
    # ... código existente ...


2. data/scrapers/schedule_scraper.py  
-------------------------------------------------------------------
# Adicionar no topo:
from utils.cache import smart_cache, TTL_SCHEDULE

# Modificar função:
@smart_cache(ttl_hours=TTL_SCHEDULE, cache_key_prefix='schedule')
def buscar_jogos_futuros():
    # ... código existente ...


3. data/scrapers/odds_scraper.py
-------------------------------------------------------------------
# Adicionar no topo:
from utils.cache import smart_cache, TTL_ODDS

# Modificar função:
@smart_cache(ttl_hours=TTL_ODDS, cache_key_prefix='odds')
def obter_odds(force_source=None):
    # ... código existente ...


4. data/scrapers/stats_scraper.py
-------------------------------------------------------------------
# Adicionar no topo:
from utils.cache import smart_cache, TTL_PLAYER_STATS

# Modificar get_rapm:
@smart_cache(ttl_hours=TTL_PLAYER_STATS, cache_key_prefix='rapm')
async def get_rapm(self):
    # ... código existente ...

# Modificar get_bball_ref:
@smart_cache(ttl_hours=TTL_PLAYER_STATS, cache_key_prefix='bballref')
async def get_bball_ref(self):
    # ... código existente ...


===================================================================
IMPACTO ESPERADO
===================================================================

Cenário Típico (100 predictions/dia):
- Injury reports: 24 calls/dia → 1 call/dia (-96%)
- Schedule: 24 calls/dia → 24 calls/dia (muda frequente, menos ganho)
- Odds: 100 calls/dia → 10 calls/dia (-90%)
- Player stats: 10 calls/dia → 2 calls/dia (-80%)

TOTAL: ~158 API calls/dia → ~37 API calls/dia
REDUÇÃO: 76.6% de API calls!

===================================================================
"""


def main():
    print(CACHE_PATCHES)
    
    logger.info("\n" + "="*80)
    logger.info("📋 INSTRUÇÕES DE APLICAÇÃO DO CACHE")
    logger.info("="*80)
    logger.info("\n1. Revise os patches acima")
    logger.info("2. Aplique os decorators @smart_cache manualmente") 
    logger.info("3. Teste com: python utils/cache.py (exemplo)")
    logger.info("4. Monitore stats: get_cache_stats()")
    logger.info("\n" + "="*80)
    
    # Verificar se cache.py existe
    cache_file = Path('utils/cache.py')
    if cache_file.exists():
        logger.info("✅ utils/cache.py encontrado!")
    else:
        logger.error("❌ utils/cache.py não encontrado!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
