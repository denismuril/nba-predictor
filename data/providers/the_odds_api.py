"""
TheOddsAPI Provider - BACKUP para odds de apostas NBA.

Este provider encapsula a chamada à TheOddsAPI com uma TRAVA DE SEGURANÇA:
antes de cada requisição, verifica o contador de uso no Redis. Se o limite
diário estiver próximo de ser atingido (>= 450 de 500), lança QuotaExceededException
e NÃO faz a requisição.

Esta abordagem preserva a cota gratuita para situações de emergência.

v24.0: Implementação com controle de cota via Redis.
"""

import logging
import os
from datetime import datetime, date
from typing import List, Optional

import aiohttp
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from dotenv import load_dotenv

from data.interfaces.odds_provider import OddsProvider, GameOdds
from exceptions.odds_exceptions import QuotaExceededException, OddsAPIKeyMissingError
from config.constants import TEAM_ABBREV_MAP

load_dotenv()

logger = logging.getLogger(__name__)

# Configuração de limites
QUOTA_LIMIT = 500  # Limite diário da API
QUOTA_WARNING_THRESHOLD = 450  # Limite para bloquear requisições
REDIS_KEY_PREFIX = "nba_predictor:odds_api"


class TheOddsAPIProvider(OddsProvider):
    """
    Provider para TheOddsAPI com controle de cota via Redis.

    COMPORTAMENTO DE SEGURANÇA:
    - Antes de cada requisição, verifica contador no Redis
    - Se contador >= 450, lança QuotaExceededException
    - Após cada requisição bem-sucedida, incrementa contador
    - Contador reseta automaticamente à meia-noite (TTL no Redis)

    Atributos:
        BASE_URL: URL base da API
        SPORT: Esporte alvo (NBA)
        MARKETS: Mercados de odds (h2h = moneyline)
        REGIONS: Região das casas de aposta
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"
    MARKETS = "h2h,spreads,totals"
    REGIONS = "us"

    def __init__(
        self,
        api_key: Optional[str] = None,
        redis_url: Optional[str] = None,
        quota_limit: int = QUOTA_WARNING_THRESHOLD,
    ):
        """
        Inicializa o provider.

        Args:
            api_key: Chave da API (ou None para buscar do .env)
            redis_url: URL do Redis (ou None para usar configuração padrão)
            quota_limit: Limite de requisições antes de bloquear (default: 450)
        """
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0"
        )
        self.quota_limit = quota_limit
        self._redis_client: Optional[aioredis.Redis] = None

        # Mapeamento de nomes da API para nomes internos
        self._team_lookup = self._build_team_lookup()

    def _build_team_lookup(self) -> dict:
        """Constrói dicionário de lookup para normalização de nomes."""
        lookup = {}
        for full_name, abbrev in TEAM_ABBREV_MAP.items():
            lookup[full_name.lower()] = full_name
            lookup[abbrev.lower()] = full_name
            # Variantes comuns
            if " " in full_name:
                parts = full_name.split()
                lookup[parts[-1].lower()] = full_name
        return lookup

    def _normalize_team_name(self, raw_name: str) -> Optional[str]:
        """Normaliza nome de time da API."""
        if not raw_name:
            return None
        clean = raw_name.strip().lower()
        return self._team_lookup.get(clean, raw_name)

    @property
    def name(self) -> str:
        """Nome identificador do provedor."""
        return "theoddsapi"

    @property
    def priority(self) -> int:
        """Prioridade do provedor (10 = backup de emergência)."""
        return 10

    async def _get_redis_client(self) -> Optional[aioredis.Redis]:
        """Obtém cliente Redis, criando se necessário."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis não disponível - controle de cota desabilitado")
            return None

        if self._redis_client is None:
            try:
                self._redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except Exception as e:
                logger.error(f"Falha ao conectar ao Redis: {e}")
                return None

        return self._redis_client

    def _get_quota_key(self) -> str:
        """Gera chave Redis para contador de hoje."""
        today = date.today().isoformat()
        return f"{REDIS_KEY_PREFIX}:usage:{today}"

    async def _check_quota(self) -> int:
        """
        Verifica uso atual da cota.

        Retorna:
            Número de requisições feitas hoje

        Raises:
            QuotaExceededException: Se limite foi atingido
        """
        redis = await self._get_redis_client()

        if redis is None:
            # Sem Redis, permite requisição mas loga warning
            logger.warning("Sem Redis - não é possível verificar cota da API")
            return 0

        try:
            key = self._get_quota_key()
            current_usage = await redis.get(key)
            current_usage = int(current_usage) if current_usage else 0

            if current_usage >= self.quota_limit:
                logger.error(
                    f"🛑 COTA DA API BLOQUEADA: {current_usage}/{QUOTA_LIMIT} requisições usadas"
                )
                raise QuotaExceededException(
                    current_usage=current_usage,
                    limit=QUOTA_LIMIT,
                )

            return current_usage

        except QuotaExceededException:
            raise
        except Exception as e:
            logger.error(f"Erro ao verificar cota: {e}")
            # Na dúvida, bloqueia para preservar cota
            raise QuotaExceededException(
                message=f"Erro ao verificar cota: {e}. Bloqueando por segurança."
            )

    async def _increment_quota(self):
        """Incrementa contador de uso após requisição bem-sucedida."""
        redis = await self._get_redis_client()

        if redis is None:
            return

        try:
            key = self._get_quota_key()
            new_value = await redis.incr(key)

            # Define TTL para expirar à meia-noite (valor conservador: 25 horas)
            await redis.expire(key, 90000)

            logger.debug(f"Cota da API incrementada: {new_value}/{QUOTA_LIMIT}")

        except Exception as e:
            logger.error(f"Erro ao incrementar cota: {e}")

    async def get_quota_status(self) -> dict:
        """
        Obtém status atual da cota.

        Retorna:
            Dict com informações da cota
        """
        redis = await self._get_redis_client()

        if redis is None:
            return {
                "available": None,
                "used": None,
                "limit": QUOTA_LIMIT,
                "status": "unknown",
                "message": "Redis não disponível",
            }

        try:
            key = self._get_quota_key()
            current_usage = await redis.get(key)
            current_usage = int(current_usage) if current_usage else 0

            remaining = QUOTA_LIMIT - current_usage

            return {
                "available": remaining,
                "used": current_usage,
                "limit": QUOTA_LIMIT,
                "status": "ok" if current_usage < self.quota_limit else "blocked",
                "message": f"Usadas {current_usage} de {QUOTA_LIMIT} requisições",
            }

        except Exception as e:
            return {
                "available": None,
                "used": None,
                "limit": QUOTA_LIMIT,
                "status": "error",
                "message": str(e),
            }

    async def get_odds(self, date: str) -> List[GameOdds]:
        """
        Obtém odds da TheOddsAPI para jogos NBA.

        Args:
            date: Data no formato "YYYY-MM-DD" (não usado pela API, retorna jogos futuros)

        Retorna:
            Lista de GameOdds para todos os jogos encontrados

        Raises:
            QuotaExceededException: Se limite de cota foi atingido
            OddsAPIKeyMissingError: Se API key não configurada
            Exception: Se requisição falhar
        """
        if not self.api_key:
            raise OddsAPIKeyMissingError("TheOddsAPI")

        # TRAVA DE SEGURANÇA: Verifica cota antes de chamar
        current_usage = await self._check_quota()
        logger.info(f"📊 Cota da API: {current_usage}/{QUOTA_LIMIT} requisições usadas")

        # Faz requisição à API
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": self.REGIONS,
            "markets": self.MARKETS,
            "oddsFormat": "decimal",
        }

        games = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"TheOddsAPI retornou status {response.status}: {error_text}")
                        raise Exception(f"API retornou status {response.status}")

                    # Atualiza cota após requisição bem-sucedida
                    await self._increment_quota()

                    # Headers de cota da API
                    remaining = response.headers.get("x-requests-remaining", "?")
                    used = response.headers.get("x-requests-used", "?")
                    logger.info(f"📊 TheOddsAPI: restam {remaining} requisições (usadas: {used})")

                    data = await response.json()

                    for event in data:
                        try:
                            game_odds = self._parse_event(event)
                            if game_odds:
                                games.append(game_odds)
                        except Exception as e:
                            logger.debug(f"Erro ao parsear evento: {e}")

        except QuotaExceededException:
            raise
        except Exception as e:
            logger.error(f"❌ TheOddsAPI falhou: {e}")
            raise

        if games:
            logger.info(f"✅ TheOddsAPI: {len(games)} jogos encontrados")
        else:
            logger.warning("⚠️ TheOddsAPI: Nenhum jogo encontrado")

        return games

    def _parse_event(self, event: dict) -> Optional[GameOdds]:
        """
        Parseia um evento da API para GameOdds.

        Args:
            event: Evento da resposta da API

        Retorna:
            GameOdds ou None se parsing falhar
        """
        try:
            home_team = self._normalize_team_name(event.get("home_team", ""))
            away_team = self._normalize_team_name(event.get("away_team", ""))

            if not home_team or not away_team:
                return None

            # Busca odds do primeiro bookmaker disponível
            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                return None

            bookmaker = bookmakers[0]  # Usa primeiro bookmaker
            bookmaker_name = bookmaker.get("key", "unknown")

            home_odds = None
            away_odds = None
            home_spread = None
            away_spread = None
            total_over = None
            over_odds = None
            under_odds = None

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if market_key == "h2h":
                    for outcome in outcomes:
                        team = outcome.get("name", "")
                        price = outcome.get("price", 1.0)
                        if self._normalize_team_name(team) == home_team:
                            home_odds = price
                        elif self._normalize_team_name(team) == away_team:
                            away_odds = price

                elif market_key == "spreads":
                    for outcome in outcomes:
                        team = outcome.get("name", "")
                        point = outcome.get("point", 0)
                        if self._normalize_team_name(team) == home_team:
                            home_spread = point
                        elif self._normalize_team_name(team) == away_team:
                            away_spread = point

                elif market_key == "totals":
                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        point = outcome.get("point", 0)
                        price = outcome.get("price", 1.0)
                        if name.lower() == "over":
                            total_over = point
                            over_odds = price
                        elif name.lower() == "under":
                            under_odds = price

            if not home_odds or not away_odds:
                return None

            game_id = f"{away_team.replace(' ', '_')}_vs_{home_team.replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d')}"

            return GameOdds(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                home_odds=home_odds,
                away_odds=away_odds,
                home_spread=home_spread,
                away_spread=away_spread,
                total_over=total_over,
                over_odds=over_odds,
                under_odds=under_odds,
                bookmaker=bookmaker_name,
                source=self.name,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.debug(f"Erro ao parsear evento: {e}")
            return None

    async def health_check(self) -> bool:
        """
        Verifica se o provider está funcional.

        Verifica:
        1. API key configurada
        2. Conexão com Redis
        3. Cota disponível

        Retorna:
            True se tudo OK, False caso contrário
        """
        # Verifica API key
        if not self.api_key:
            logger.warning("TheOddsAPI: API key não configurada")
            return False

        # Verifica Redis e cota
        try:
            quota_status = await self.get_quota_status()
            if quota_status["status"] == "blocked":
                logger.warning(f"TheOddsAPI: Cota bloqueada - {quota_status['message']}")
                return False
            elif quota_status["status"] == "error":
                logger.warning(f"TheOddsAPI: Erro de cota - {quota_status['message']}")
                # Retorna True mesmo com erro de Redis (API pode funcionar)

            logger.info(f"✅ TheOddsAPI: Health check OK - {quota_status['message']}")
            return True

        except Exception as e:
            logger.error(f"TheOddsAPI health check falhou: {e}")
            return False

    async def close(self):
        """Fecha conexões abertas."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
