"""
RedisCache - Cache de Alta Velocidade
======================================
Substitui leitura de JSON/CSV do disco por cache Redis.
TTLs configuráveis por tipo de dado.

Autor: NBA Predictor v22.0
"""

import json
from typing import Optional, Any, Dict, List
from datetime import timedelta
import os
import logging

logger = logging.getLogger(__name__)

# TTLs padrão por categoria de dado
DEFAULT_TTLS = {
    'odds': timedelta(minutes=5),       # Odds mudam rápido
    'injuries': timedelta(minutes=30),  # Lesões: 30min
    'features': timedelta(hours=24),    # Features: 1 dia
    'predictions': timedelta(hours=6),  # Previsões: 6h
    'schedule': timedelta(hours=12),    # Schedule: 12h
    'default': timedelta(hours=1)       # Padrão: 1h
}


class RedisCache:
    """
    Cache Redis de alta velocidade.
    
    Características:
    - TTLs automáticos por categoria de dado
    - Serialização JSON automática
    - Fallback gracioso se Redis não disponível
    - Métodos específicos para odds, lesões, etc.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Inicializa o cache Redis.
        
        Args:
            redis_url: URL Redis. Se None, usa variáveis de ambiente.
        """
        self.redis_url = redis_url or self._build_url()
        self.ttls = DEFAULT_TTLS.copy()
        self._redis = None
        self._connected = False
    
    def _build_url(self) -> str:
        """Monta URL de conexão Redis"""
        host = os.getenv('REDIS_HOST', 'localhost')
        port = os.getenv('REDIS_PORT', '6379')
        password = os.getenv('REDIS_PASSWORD', '')
        db = os.getenv('REDIS_DB', '0')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"
    
    async def connect(self):
        """Estabelece conexão com Redis"""
        if self._connected:
            return
        
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Testar conexão
            await self._redis.ping()
            self._connected = True
            logger.info("✅ Redis conectado")
        except ImportError:
            logger.warning("⚠️ redis.asyncio não instalado. Cache desabilitado.")
            self._connected = False
        except Exception as e:
            logger.warning(f"⚠️ Redis não disponível: {e}. Cache desabilitado.")
            self._connected = False
    
    async def disconnect(self):
        """Fecha conexão com Redis"""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("🔒 Redis desconectado")
    
    # ============= MÉTODOS GENÉRICOS =============
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Busca valor do cache.
        
        Args:
            key: Chave do cache
            
        Returns:
            Valor deserializado ou None se não encontrado
        """
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return None
        
        try:
            data = await self._redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, category: str = 'default') -> bool:
        """
        Armazena valor no cache com TTL automático.
        
        Args:
            key: Chave do cache
            value: Valor a armazenar (será serializado em JSON)
            category: Categoria para determinar TTL
            
        Returns:
            True se sucesso, False caso contrário
        """
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return False
        
        try:
            ttl = self.ttls.get(category, self.ttls['default'])
            await self._redis.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Remove chave do cache"""
        if not self._connected:
            return False
        
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.debug(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Verifica se chave existe"""
        if not self._connected:
            return False
        
        try:
            return await self._redis.exists(key) > 0
        except Exception:
            return False
    
    # ============= MÉTODOS ESPECÍFICOS: ODDS =============
    
    async def get_odds(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Busca odds de um jogo do cache.
        
        Args:
            game_id: ID do jogo
            
        Returns:
            Dict com odds ou None
        """
        return await self.get(f"odds:{game_id}")
    
    async def set_odds(self, game_id: str, odds: Dict[str, Any]) -> bool:
        """
        Armazena odds de um jogo (TTL: 5 minutos).
        
        Args:
            game_id: ID do jogo
            odds: Dicionário com odds
        """
        return await self.set(f"odds:{game_id}", odds, 'odds')
    
    async def get_all_odds(self, game_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Busca odds de múltiplos jogos.
        
        Args:
            game_ids: Lista de IDs de jogos
            
        Returns:
            Dict mapeando game_id → odds
        """
        result = {}
        for game_id in game_ids:
            odds = await self.get_odds(game_id)
            if odds:
                result[game_id] = odds
        return result
    
    # ============= MÉTODOS ESPECÍFICOS: LESÕES =============
    
    async def get_injuries(self, team_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Busca lesões de um time do cache.
        
        Args:
            team_id: ID do time (ex: 'LAL', 'BOS')
            
        Returns:
            Lista de lesões ou None
        """
        return await self.get(f"injuries:{team_id}")
    
    async def set_injuries(self, team_id: str, injuries: List[Dict[str, Any]]) -> bool:
        """
        Armazena lesões de um time (TTL: 30 minutos).
        
        Args:
            team_id: ID do time
            injuries: Lista de lesões
        """
        return await self.set(f"injuries:{team_id}", injuries, 'injuries')
    
    async def get_all_injuries(self) -> Dict[str, List[Dict[str, Any]]]:
        """Busca todas as lesões em cache"""
        if not self._connected:
            return {}
        
        try:
            keys = await self._redis.keys("injuries:*")
            result = {}
            for key in keys:
                team_id = key.replace("injuries:", "")
                result[team_id] = await self.get(key)
            return result
        except Exception:
            return {}
    
    # ============= MÉTODOS ESPECÍFICOS: FEATURES =============
    
    async def get_feature(self, entity_id: str, feature_name: str) -> Optional[float]:
        """
        Busca feature do cache.
        
        Args:
            entity_id: ID da entidade (time ou jogo)
            feature_name: Nome da feature
        """
        return await self.get(f"feature:{entity_id}:{feature_name}")
    
    async def set_feature(self, entity_id: str, feature_name: str, value: float) -> bool:
        """
        Armazena feature no cache (TTL: 24 horas).
        """
        return await self.set(f"feature:{entity_id}:{feature_name}", value, 'features')
    
    # ============= MÉTODOS ESPECÍFICOS: PREDICTIONS =============
    
    async def get_predictions(self, date: str) -> Optional[List[Dict[str, Any]]]:
        """Busca previsões de uma data"""
        return await self.get(f"predictions:{date}")
    
    async def set_predictions(self, date: str, predictions: List[Dict[str, Any]]) -> bool:
        """Armazena previsões (TTL: 6 horas)"""
        return await self.set(f"predictions:{date}", predictions, 'predictions')
    
    # ============= UTILITÁRIOS =============
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Remove todas as chaves que correspondem ao padrão.
        
        Args:
            pattern: Padrão glob (ex: 'odds:*', 'injuries:*')
            
        Returns:
            Número de chaves removidas
        """
        if not self._connected:
            return 0
        
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
            return len(keys)
        except Exception:
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do cache Redis"""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return {'status': 'disconnected', 'message': 'Redis não disponível'}
        
        try:
            info = await self._redis.info()
            keys_count = await self._redis.dbsize()
            
            return {
                'status': 'healthy',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'keys_count': keys_count,
                'redis_version': info.get('redis_version', 'N/A')
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# ============= SINGLETON HELPER =============

_redis_instance: Optional[RedisCache] = None


async def get_redis() -> RedisCache:
    """
    Retorna instância singleton do RedisCache.
    
    Uso:
        redis = await get_redis()
        await redis.set_odds('game_123', {'home': 1.85, 'away': 2.05})
    """
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = RedisCache()
        await _redis_instance.connect()
    return _redis_instance


async def reset_redis_instance():
    """Reseta singleton (útil para testes)"""
    global _redis_instance
    if _redis_instance:
        await _redis_instance.disconnect()
    _redis_instance = None
