"""
Rate Limiter Distribuído - Controle de Taxa de Requisições
==========================================================
Implementa rate limiting usando Redis para evitar bloqueios
por APIs de odds e outras fontes externas.

Usa algoritmo Token Bucket.

Autor: NBA Predictor v22.0
"""

import time
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


# Configurações padrão por API
DEFAULT_LIMITS = {
    'oddspedia': {'max_requests': 10, 'window_seconds': 60},
    'theoddsapi': {'max_requests': 50, 'window_seconds': 60},
    'sportsdataio': {'max_requests': 100, 'window_seconds': 60},
    'nba_api': {'max_requests': 30, 'window_seconds': 60},
    'espn': {'max_requests': 20, 'window_seconds': 60},
    'rotowire': {'max_requests': 15, 'window_seconds': 60},
    'default': {'max_requests': 30, 'window_seconds': 60}
}


class DistributedRateLimiter:
    """
    Rate Limiter distribuído usando Redis.
    
    Características:
    - Algoritmo Token Bucket
    - Distribuído (múltiplas instâncias compartilham o estado)
    - Configurável por API
    - Fallback para limiter local se Redis indisponível
    """
    
    def __init__(self, redis_client=None):
        """
        Inicializa o Rate Limiter.
        
        Args:
            redis_client: Cliente Redis assíncrono. Se None, usa fallback local.
        """
        self._redis = redis_client
        self._local_buckets: Dict[str, list] = {}  # Fallback local
        self._limits = DEFAULT_LIMITS.copy()
    
    async def set_redis(self, redis_client):
        """Define cliente Redis após inicialização"""
        self._redis = redis_client
    
    def configure_limit(self, api_name: str, max_requests: int, window_seconds: int):
        """
        Configura limite customizado para uma API.
        
        Args:
            api_name: Nome da API (ex: 'oddspedia')
            max_requests: Máximo de requests na janela
            window_seconds: Tamanho da janela em segundos
        """
        self._limits[api_name] = {
            'max_requests': max_requests,
            'window_seconds': window_seconds
        }
    
    async def acquire(self, key: str, max_requests: int = None, 
                      window_seconds: int = None) -> bool:
        """
        Tenta adquirir um token para fazer uma requisição.
        
        Args:
            key: Identificador do recurso (ex: 'oddspedia', 'theoddsapi')
            max_requests: Máximo de requests (se None, usa config do key)
            window_seconds: Janela de tempo (se None, usa config do key)
            
        Returns:
            True se pode prosseguir, False se deve esperar
        """
        # Usar configuração padrão se não especificado
        if max_requests is None or window_seconds is None:
            config = self._limits.get(key, self._limits['default'])
            max_requests = max_requests or config['max_requests']
            window_seconds = window_seconds or config['window_seconds']
        
        if self._redis:
            return await self._acquire_redis(key, max_requests, window_seconds)
        else:
            return self._acquire_local(key, max_requests, window_seconds)
    
    async def _acquire_redis(self, key: str, max_requests: int, 
                              window_seconds: int) -> bool:
        """Implementação Redis do rate limiter"""
        bucket_key = f"ratelimit:{key}"
        now = time.time()
        window_start = now - window_seconds
        
        try:
            # Remover tokens expirados (fora da janela)
            await self._redis.zremrangebyscore(bucket_key, '-inf', window_start)
            
            # Contar tokens atuais na janela
            current_count = await self._redis.zcard(bucket_key)
            
            if current_count < max_requests:
                # Adicionar novo token com timestamp como score
                await self._redis.zadd(bucket_key, {str(now): now})
                # Definir expiração da chave
                await self._redis.expire(bucket_key, window_seconds)
                
                logger.debug(f"Rate limit OK: {key} ({current_count + 1}/{max_requests})")
                return True
            
            logger.debug(f"Rate limit EXCEEDED: {key} ({current_count}/{max_requests})")
            return False
            
        except Exception as e:
            logger.warning(f"Erro no rate limiter Redis: {e}. Usando fallback local.")
            return self._acquire_local(key, max_requests, window_seconds)
    
    def _acquire_local(self, key: str, max_requests: int, 
                       window_seconds: int) -> bool:
        """Implementação local (fallback) do rate limiter"""
        now = time.time()
        window_start = now - window_seconds
        
        # Inicializar bucket se não existir
        if key not in self._local_buckets:
            self._local_buckets[key] = []
        
        # Remover tokens expirados
        self._local_buckets[key] = [t for t in self._local_buckets[key] if t > window_start]
        
        # Verificar limite
        if len(self._local_buckets[key]) < max_requests:
            self._local_buckets[key].append(now)
            return True
        
        return False
    
    async def wait_and_acquire(self, key: str, max_requests: int = None,
                                window_seconds: int = None, 
                                max_wait: float = 30.0,
                                poll_interval: float = 1.0) -> bool:
        """
        Espera até conseguir um token ou atingir timeout.
        
        Args:
            key: Identificador do recurso
            max_requests: Máximo de requests
            window_seconds: Janela de tempo
            max_wait: Tempo máximo de espera em segundos
            poll_interval: Intervalo entre tentativas
            
        Returns:
            True se conseguiu token, False se timeout
        """
        import asyncio
        
        start = time.time()
        
        while time.time() - start < max_wait:
            if await self.acquire(key, max_requests, window_seconds):
                return True
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"Rate limit timeout: {key} após {max_wait}s")
        return False
    
    async def get_remaining(self, key: str) -> Dict[str, Any]:
        """
        Retorna informações sobre tokens restantes.
        
        Returns:
            Dict com 'remaining', 'limit', 'reset_in_seconds'
        """
        config = self._limits.get(key, self._limits['default'])
        max_requests = config['max_requests']
        window_seconds = config['window_seconds']
        
        if self._redis:
            try:
                bucket_key = f"ratelimit:{key}"
                now = time.time()
                window_start = now - window_seconds
                
                # Limpar expirados
                await self._redis.zremrangebyscore(bucket_key, '-inf', window_start)
                
                # Contar atuais
                current_count = await self._redis.zcard(bucket_key)
                
                # Calcular reset
                oldest = await self._redis.zrange(bucket_key, 0, 0, withscores=True)
                if oldest:
                    reset_in = max(0, window_seconds - (now - oldest[0][1]))
                else:
                    reset_in = 0
                
                return {
                    'remaining': max(0, max_requests - current_count),
                    'limit': max_requests,
                    'reset_in_seconds': int(reset_in),
                    'window_seconds': window_seconds
                }
            except Exception:
                pass
        
        # Fallback local
        now = time.time()
        window_start = now - window_seconds
        bucket = self._local_buckets.get(key, [])
        active = [t for t in bucket if t > window_start]
        
        return {
            'remaining': max(0, max_requests - len(active)),
            'limit': max_requests,
            'reset_in_seconds': int(window_seconds) if active else 0,
            'window_seconds': window_seconds
        }
    
    async def reset(self, key: str):
        """Reseta o bucket para uma API específica"""
        if self._redis:
            try:
                await self._redis.delete(f"ratelimit:{key}")
            except Exception:
                pass
        
        if key in self._local_buckets:
            del self._local_buckets[key]
    
    async def reset_all(self):
        """Reseta todos os buckets"""
        if self._redis:
            try:
                keys = await self._redis.keys("ratelimit:*")
                if keys:
                    await self._redis.delete(*keys)
            except Exception:
                pass
        
        self._local_buckets.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retorna estatísticas de todos os limiters"""
        stats = {}
        now = time.time()
        
        for key, config in self._limits.items():
            if key == 'default':
                continue
            
            window_start = now - config['window_seconds']
            bucket = self._local_buckets.get(key, [])
            active = [t for t in bucket if t > window_start]
            
            stats[key] = {
                'current_requests': len(active),
                'max_requests': config['max_requests'],
                'window_seconds': config['window_seconds'],
                'utilization': len(active) / config['max_requests'] if config['max_requests'] > 0 else 0
            }
        
        return stats


# ============= DECORATOR PARA RATE LIMITING =============

def rate_limited(api_name: str, limiter: DistributedRateLimiter = None):
    """
    Decorator para aplicar rate limiting a funções async.
    
    Usage:
        @rate_limited('oddspedia')
        async def fetch_odds(game_id: str):
            ...
    """
    def decorator(func):
        import asyncio
        from functools import wraps
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal limiter
            
            # Usar limiter global se não especificado
            if limiter is None:
                limiter = _global_limiter
            
            # Aguardar token
            if not await limiter.wait_and_acquire(api_name):
                raise RateLimitExceededError(f"Rate limit exceeded for {api_name}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class RateLimitExceededError(Exception):
    """Exceção quando rate limit é excedido"""
    pass


# ============= SINGLETON GLOBAL =============

_global_limiter: Optional[DistributedRateLimiter] = None


async def get_rate_limiter() -> DistributedRateLimiter:
    """Retorna instância singleton do rate limiter"""
    global _global_limiter
    
    if _global_limiter is None:
        _global_limiter = DistributedRateLimiter()
        
        # Tentar conectar Redis
        try:
            from .redis_cache import get_redis
            redis = await get_redis()
            if redis._connected:
                await _global_limiter.set_redis(redis._redis)
                logger.info("✅ Rate limiter conectado ao Redis")
        except Exception:
            logger.info("ℹ️ Rate limiter usando modo local")
    
    return _global_limiter
