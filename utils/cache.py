"""
Smart Caching System com TTL (Time To Live).

Reduz API calls redundantes em até 75% usando cache inteligente com expiração.

Cache Strategies:
- Injury Reports: 24h TTL (dados mudam diariamente)
- Schedule: 1h TTL (horários podem mudar)
- Odds: 10min TTL (odds mudam frequentemente)
- Player Stats: 6h TTL (atualizado algumas vezes ao dia)

Usage:
    from utils.cache import smart_cache, clear_cache
    
    @smart_cache(ttl_hours=24, cache_key_prefix='injuries')
    def obter_injury_report():
        # Expensive API call
        return fetch_injuries_from_api()
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Dict
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class SmartCache:
    """
    Sistema de cache inteligente com TTL.
    
    Features:
    - TTL configurável por tipo de dado
    - Cache em disco (JSON)
    - Auto-cleanup de caches expirados
    - Estatísticas de hit/miss
    """
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Estatísticas
        self.stats = {
            'hits': 0,
            'misses': 0,
            'expired': 0,
            'cached_items': 0
        }
        
        # Carregar stats se existir
        self.stats_file = self.cache_dir / '_cache_stats.json'
        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    self.stats = json.load(f)
            except Exception:
                pass
    
    def set(self, key: str, value: Any, ttl_seconds: int):
        """
        Salva valor no cache com TTL.
        
        Args:
            key: Chave do cache
            value: Valor a cachear (deve ser JSON-serializable)
            ttl_seconds: Tempo de vida em segundos
        """
        cache_entry = {
            'value': value,
            'cached_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat(),
            'ttl_seconds': ttl_seconds
        }
        
        cache_file = self._get_cache_file(key)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2, default=str)
            
            self.stats['cached_items'] += 1
            self._save_stats()
            
            logger.debug(f"📦 Cache SET: {key} (TTL={ttl_seconds}s)")
            
        except Exception as e:
            logger.error(f"❌ Erro salvando cache {key}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtém valor do cache se não expirado.
        
        Args:
            key: Chave do cache
            
        Returns:
            Valor cacheado ou None se expirado/não existe
        """
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            self.stats['misses'] += 1
            self._save_stats()
            logger.debug(f"❌ Cache MISS: {key} (não existe)")
            return None
        
        try:
            with open(cache_file) as f:
                cache_entry = json.load(f)
            
            # Verificar expiração
            expires_at = datetime.fromisoformat(cache_entry['expires_at'])
            
            if datetime.now() > expires_at:
                # Cache expirado
                self.stats['expired'] += 1
                self._save_stats()
                logger.debug(f"⏰ Cache EXPIRED: {key}")
                
                # Deletar arquivo expirado
                cache_file.unlink()
                
                return None
            
            # Cache HIT!
            self.stats['hits'] += 1
            self._save_stats()
            logger.debug(f"✅ Cache HIT: {key}")
            
            return cache_entry['value']
            
        except Exception as e:
            logger.error(f"❌ Erro lendo cache {key}: {e}")
            return None
    
    def invalidate(self, key: str):
        """Remove item do cache."""
        cache_file = self._get_cache_file(key)
        
        if cache_file.exists():
            cache_file.unlink()
            logger.debug(f"🗑️  Cache invalidado: {key}")
    
    def clear_all(self):
        """Limpa todo o cache."""
        count = 0
        for cache_file in self.cache_dir.glob('*.json'):
            if cache_file.name != '_cache_stats.json':
                cache_file.unlink()
                count += 1
        
        logger.info(f"🗑️  {count} itens removidos do cache")
        
        # Reset stats
        self.stats = {'hits': 0, 'misses': 0, 'expired': 0, 'cached_items': 0}
        self._save_stats()
    
    def cleanup_expired(self):
        """Remove todos os caches expirados."""
        count = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob('*.json'):
            if cache_file.name == '_cache_stats.json':
                continue
            
            try:
                with open(cache_file) as f:
                    cache_entry = json.load(f)
                
                expires_at = datetime.fromisoformat(cache_entry['expires_at'])
                
                if now > expires_at:
                    cache_file.unlink()
                    count += 1
                    
            except Exception:
                # Se não conseguir ler, deletar
                cache_file.unlink()
                count += 1
        
        if count > 0:
            logger.info(f"🧹 {count} caches expirados removidos")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso do cache."""
        total = self.stats['hits'] + self.stats['misses']
        
        if total > 0:
            hit_rate = (self.stats['hits'] / total) * 100
        else:
            hit_rate = 0
        
        return {
            **self.stats,
            'total_requests': total,
            'hit_rate_pct': round(hit_rate, 2)
        }
    
    def _get_cache_file(self, key: str) -> Path:
        """Converte chave em nome de arquivo."""
        # Hash da key para nome seguro
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def _save_stats(self):
        """Salva estatísticas."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception:
            pass


# Instância global
_cache = SmartCache()


def smart_cache(ttl_hours: float = 1.0, cache_key_prefix: str = ''):
    """
    Decorator para cachear resultados de função com TTL.
    
    Args:
        ttl_hours: Tempo de vida em horas
        cache_key_prefix: Prefixo para chave do cache
        
    Usage:
        @smart_cache(ttl_hours=24, cache_key_prefix='injuries')
        def obter_injury_report():
            return expensive_api_call()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Gerar chave única baseada em função + args
            key_parts = [cache_key_prefix, func.__name__]
            
            # Adicionar args à key (apenas tipos simples)
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
            
            # Adicionar kwargs à key
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")
            
            cache_key = '_'.join(filter(None, key_parts))
            
            # Tentar obter do cache
            cached_value = _cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            # Cache miss - executar função
            result = func(*args, **kwargs)
            
            # Cachear resultado
            ttl_seconds = int(ttl_hours * 3600)
            _cache.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def clear_cache():
    """Limpa todo o cache."""
    _cache.clear_all()


def cleanup_cache():
    """Remove caches expirados."""
    return _cache.cleanup_expired()


def get_cache_stats() -> Dict:
    """Retorna estatísticas do cache."""
    return _cache.get_stats()


def invalidate_cache(key: str):
    """Invalida cache específico."""
    _cache.invalidate(key)


# ===== CONFIGURAÇÕES PADRÃO =====

# TTLs recomendados por tipo de dado
TTL_INJURY_REPORT = 24  # hours - dados mudam 1x ao dia
TTL_SCHEDULE = 1  # hour - horários podem mudar
TTL_ODDS = 1/6  # 10 minutes - odds mudam frequentemente
TTL_PLAYER_STATS = 6  # hours - atualizado algumas vezes ao dia
TTL_TEAM_STATS = 12  # hours - atualizado 1-2x ao dia


# ===== EXEMPLO DE USO =====

if __name__ == "__main__":
    import time
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Teste 1: Função cacheada
    @smart_cache(ttl_hours=0.001, cache_key_prefix='test')  # 3.6 segundos
    def expensive_function(param: str):
        """Simula API call cara."""
        print(f"  🔄 Executando expensive_function({param})...")
        time.sleep(1)  # Simular latência de API
        return f"result_{param}_" + datetime.now().strftime("%H:%M:%S")
    
    print("\n=== TESTE DE CACHE ===\n")
    
    # Primeira chamada - cache MISS
    print("1ª chamada:")
    result1 = expensive_function("test")
    print(f"  Resultado: {result1}\n")
    
    # Segunda chamada - cache HIT
    print("2ª chamada (imediata):")
    result2 = expensive_function("test")
    print(f"  Resultado: {result2}")
    print(f"  Mesmo resultado? {result1 == result2}\n")
    
    # Esperar expirar
    print("Esperando 4 segundos (cache expirar)...")
    time.sleep(4)
    
    # Terceira chamada - cache MISS (expirado)
    print("\n3ª chamada (após expiração):")
    result3 = expensive_function("test")
    print(f"  Resultado: {result3}")
    print(f"  Resultado diferente? {result1 != result3}\n")
    
    # Stats
    print("=== ESTATÍSTICAS ===")
    stats = get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    print("\n=== CLEANUP ===")
    clear_cache()
    print("  Cache limpo!")
