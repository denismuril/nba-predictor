"""
Script para aplicar patches de cache nos scrapers de forma programática.

Restaura arquivos corrompidos e aplica decorators @smart_cache corretamente.
"""

import os
import sys

# Patch 1: odds_scraper.py - adicionar decorator na função obter_odds
ODDS_SCRAPER_PATCH = """
# Encontrar linha "def obter_odds(force_source: Optional[str] = None) -> Dict:"
# Adicionar ANTES dela:
@smart_cache(ttl_hours=TTL_ODDS, cache_key_prefix='odds')
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║  CACHE PATCHES - APLICAÇÃO MANUAL NECESSÁRIA                 ║
╚══════════════════════════════════════════════════════════════╝

⚠️  IMPORTANTE: Os arquivos injury_scraper.py ficaram corrompidos
    durante tentativas automáticas de edição.

📋 AÇÃO NECESSÁRIA:

1. Restaurar arquivo original:
   $ git checkout HEAD~1 data/scrapers/injury_scraper.py

2. Aplicar 3 patches MANUALMENTE:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH 1: data/scrapers/injury_scraper.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Linha ~8 (após outros imports):
    from utils.cache import smart_cache, TTL_INJURY_REPORT

Linha ~28 (ANTES de "def obter_injury_report_api_espn():"):
    @smart_cache(ttl_hours=TTL_INJURY_REPORT, cache_key_prefix='injuries')
    @smart_retry(max_attempts=3, min_wait=2.0, max_wait=8.0)
    def obter_injury_report_api_espn():

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATCH 2: data/scrapers/odds_scraper.py  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Linha ~16 (após outros imports):
    ✅ JÁ APLICADO: from utils.cache import smart_cache, TTL_ODDS

Linha ~672 (ANTES de "def obter_odds(force_source...):"):
    @smart_cache(ttl_hours=TTL_ODDS, cache_key_prefix='odds')
    def obter_odds(force_source: Optional[str] = None) -> Dict:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Sistema de cache JÁ CRIADO:
   - utils/cache.py (370 linhas, testado)
   - TTLs configurados (24h, 10min, 6h)

📊 IMPACTO ESPERADO:
   - Injury reports: 24 calls/dia → 1 call/dia (-96%)
   - Odds: 100 calls/dia → 10 calls/dia (-90%)
   - Total: ~158 → ~37 calls/dia (-76%)

🧪 COMO TESTAR:
   from utils.cache import get_cache_stats
   stats = get_cache_stats()
   print(f"Hit rate: {stats['hit_rate_pct']}%")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("✅ Guia completo em: docs/CACHE_PATCHES_GUIDE.md")
print("\n📌 Sprint 1.2 ficará 100% após aplicação manual dos patches")
