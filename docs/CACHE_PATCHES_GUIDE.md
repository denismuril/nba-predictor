# Sprint 1.2 - Cache Patches Application Guide

## ✅ O Que Já Foi Criado

- ✅ `utils/cache.py` - Sistema completo de cache com TTL
- ✅ `scripts/setup_cache.py` - Script de instruções  
- ✅ Import adicionado em `odds_scraper.py`

## 📝 Patches a Aplicar Manualmente

### 1. data/scrapers/injury_scraper.py

**Linha 8 - Adicionar import:**
```python
from utils.cache import smart_cache, TTL_INJURY_REPORT
```

**Linha 28 - Adicionar decorator antes de `obter_injury_report_api_espn()`:**
```python
@smart_cache(ttl_hours=TTL_INJURY_REPORT, cache_key_prefix='injuries')
@smart_retry(max_attempts=3, min_wait=2.0, max_wait=8.0)
def obter_injury_report_api_espn():
```

### 2. data/scrapers/odds_scraper.py  

**✅ JÁ APLICADO** - Import na linha 16

**Linha 672 - Adicionar decorator antes de `obter_odds()`:**
```python
@smart_cache(ttl_hours=TTL_ODDS, cache_key_prefix='odds')
def obter_odds(force_source: Optional[str] = None) -> Dict:
```

### 3. data/scrapers/stats_scraper.py (Opcional)

**Adicionar import no topo:**
```python
from utils.cache import smart_cache, TTL_PLAYER_STATS
```

**Decorar método `get_rapm`:**
```python
@smart_cache(ttl_hours=TTL_PLAYER_STATS, cache_key_prefix='rapm')
async def get_rapm(self):
```

## 🎯 Impacto Esperado

| Scraper | Calls Antes | Calls Depois | Redução |
|---------|-------------|--------------|---------|
| Injuries | 24/dia | 1/dia | -96% |
| Odds | 100/dia | 10/dia | -90% |
| Stats | 10/dia | 2/dia | -80% |

**Total: ~158 → ~37 calls/dia = -76%**

## ✅ Como Verificar

```python
from utils.cache import get_cache_stats

# Após rodar scrapers
stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate_pct']}%")
print(f"Total requests: {stats['total_requests']}")
```

## 🔧 TTLs Configurados

```python
TTL_INJURY_REPORT = 24h   # Dados mudam 1x/dia
TTL_ODDS = 10min          # Odds mudam frequentemente  
TTL_SCHEDULE = 1h         # Horários podem mudar
TTL_PLAYER_STATS = 6h     # Atualizado algumas vezes/dia
```

---

**Status: Pronto para aplicação manual**
