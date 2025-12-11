"""
Retry utilities para tornar scrapers resilientes a falhas transitórias.

Implementa exponential backoff para:
- Network timeouts
- API rate limits
- Conexões intermitentes
"""

import logging
from functools import wraps
from typing import Callable, Optional, Tuple, Type
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
import requests

logger = logging.getLogger(__name__)


# Exceções que devem acionar retry
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.HTTPError,  # Apenas para 5xx (configurar depois)
    ConnectionError,
    TimeoutError,
)


def should_retry_http_error(exception: Exception) -> bool:
    """
    Determina se um HTTPError deve ser retried.
    
    Retry apenas para:
    - 5xx (server errors)
    - 429 (rate limit)
    - 503 (service unavailable)
    
    NÃO retry para:
    - 4xx (client errors, except 429)
    - 401 (unauthorized - API key inválida)
    - 404 (not found - recurso não existe)
    """
    if isinstance(exception, requests.exceptions.HTTPError):
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            
            # Retry para server errors
            if 500 <= status_code < 600:
                return True
            
            # Retry para rate limit
            if status_code == 429:
                return True
            
            # Não retry para client errors
            return False
    
    # Para outras exceptions, usar default
    return isinstance(exception, RETRYABLE_EXCEPTIONS)


def smart_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    logger_name: Optional[str] = None
):
    """
    Decorator para retry com exponential backoff.
    
    Args:
        max_attempts: Número máximo de tentativas (default: 3)
        min_wait: Tempo mínimo de espera entre retries em segundos (default: 1.0)
        max_wait: Tempo máximo de espera entre retries em segundos (default: 10.0)
        multiplier: Multiplicador para exponential backoff (default: 2.0)
        exceptions: Tupla de exceptions para retry (default: RETRYABLE_EXCEPTIONS)
        logger_name: Nome do logger para usar (default: usa logger do módulo)
    
    Exemplo de backoff com defaults:
        Attempt 1: executa imediatamente
        Attempt 2: espera 1-2s (min_wait * multiplier^0)
        Attempt 3: espera 2-4s (min_wait * multiplier^1)
    
    Usage:
        @smart_retry(max_attempts=3, min_wait=2.0)
        def fetch_data():
            response = requests.get(url, timeout=10)
            return response.json()
    """
    if exceptions is None:
        exceptions = RETRYABLE_EXCEPTIONS
    
    retry_logger = logging.getLogger(logger_name) if logger_name else logger
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=multiplier,
                min=min_wait,
                max=max_wait
            ),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(retry_logger, logging.WARNING),
            reraise=True
        )
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RetryError as e:
                # Extrair exception original
                if hasattr(e, 'last_attempt') and e.last_attempt.exception():
                    original_exception = e.last_attempt.exception()
                    retry_logger.error(
                        f"❌ {func.__name__} falhou após {max_attempts} tentativas. "
                        f"Última exceção: {type(original_exception).__name__}: {original_exception}"
                    )
                    raise original_exception
                else:
                    raise
        
        return wrapper
    
    return decorator


def async_smart_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    multiplier: float = 2.0
):
    """
    Decorator para retry assíncrono.
    
    Uso para funções async/await.
    
    Usage:
        @async_smart_retry(max_attempts=3)
        async def fetch_async(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    """
    from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True
            ):
                with attempt:
                    return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Aliases para conveniência
retry_network = smart_retry  # Alias descritivo
retry_api = lambda **kwargs: smart_retry(max_attempts=3, min_wait=2.0, **kwargs)  # Específico para APIs


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging para ver os retries
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Teste 1: Função que falha 2 vezes antes de suceder
    attempt_count = 0
    
    @smart_retry(max_attempts=3, min_wait=0.5, max_wait=2.0)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        print(f"Tentativa {attempt_count}")
        
        if attempt_count < 3:
            raise requests.exceptions.Timeout("Simulated timeout")
        
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"✅ Resultado: {result}")
    except Exception as e:
        print(f"❌ Falhou: {e}")
    
    # Teste 2: Função que sempre falha
    @smart_retry(max_attempts=2, min_wait=0.5)
    def always_fails():
        raise requests.exceptions.ConnectionError("Always fails")
    
    try:
        always_fails()
    except Exception as e:
        print(f"❌ Esperado falhar: {type(e).__name__}")
