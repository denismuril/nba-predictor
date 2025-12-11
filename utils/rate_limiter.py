"""
API Rate Limiting - v16.0

Rate limiting para proteger APIs e evitar bans.

Features:
- Rate limiting por API
- Retry logic com backoff
- Cache para reduzir calls
- Monitoring de usage

Usage:
    from utils.rate_limiter import RateLimiter
    
    limiter = RateLimiter(calls=10, period=60)  # 10 calls/min
    limiter.wait_if_needed()
"""
import time
import logging
from functools import wraps
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter para controle de APIs.
    
    Usage:
        limiter = RateLimiter(calls=10, period=60)
        limiter.wait_if_needed()
    """
    
    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter.
        
        Args:
            calls: Número máximo de calls
            period: Período em segundos
        """
        self.calls = calls
        self.period = period
        self.timestamps = deque(maxlen=calls)
        
    def wait_if_needed(self):
        """Aguarda se necessário para respeitar rate limit."""
        now = time.time()
        
        # Remove timestamps antigos
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()
        
        # Se atingiu limite, aguarda
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                logger.warning(f"⏳ Rate limit atingido. Aguardando {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        # Registra chamada
        self.timestamps.append(time.time())


def rate_limit(calls: int, period: int):
    """
    Decorator para rate limiting.
    
    Usage:
        @rate_limit(calls=10, period=60)
        def api_call():
            ...
    """
    limiter = RateLimiter(calls, period)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    return decorator


class APIThrottler:
    """
    Throttler com retry logic e exponential backoff.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def call_with_retry(self, func, *args, **kwargs):
        """
        Executa função com retry logic.
        
        Args:
            func: Função a executar
            *args, **kwargs: Argumentos da função
            
        Returns:
            Resultado da função ou None se falhar
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️ Tentativa {attempt + 1} falhou: {e}")
                    logger.info(f"🔄 Retry em {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Todas as tentativas falharam: {e}")
                    return None


# Rate limiters pré-configurados para cada API
API_FOOTBALL_LIMITER = RateLimiter(calls=100, period=86400)  # 100/day
SPORTDATA_LIMITER = RateLimiter(calls=500, period=86400)  # 500/day
SPORTSBLAZE_LIMITER = RateLimiter(calls=1000, period=86400)  # 1000/day
NBA_API_LIMITER = RateLimiter(calls=60, period=60)  # 60/min


# Decorator helpers
def api_football_rate_limit(func):
    """Rate limit para API-Football."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        API_FOOTBALL_LIMITER.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


def sportdata_rate_limit(func):
    """Rate limit para SportData.io."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        SPORTDATA_LIMITER.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


def sportsblaze_rate_limit(func):
    """Rate limit para SportsBlaze."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        SPORTSBLAZE_LIMITER.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


def nba_api_rate_limit(func):
    """Rate limit para NBA Official API."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        NBA_API_LIMITER.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


if __name__ == '__main__':
    # Demo
    print("🔧 Rate Limiter v16.0")
    print("\nConfiguração:")
    print(f"  API-Football: 100 calls/day")
    print(f"  SportData: 500 calls/day")
    print(f"  SportsBlaze: 1000 calls/day")
    print(f"  NBA API: 60 calls/min")
    print("\n✅ Rate limiting configurado!")
