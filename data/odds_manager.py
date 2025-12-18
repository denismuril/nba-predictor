"""
OddsDataManager - Orquestrador de múltiplas fontes de odds.

Este módulo implementa o padrão Chain of Responsibility para gerenciar
múltiplos provedores de odds com prioridade definida:

1. SBR Scraper (Prioridade Alta) - Fonte gratuita via web scraping
2. TheOddsAPI (Prioridade Baixa) - Backup pago, só usado se scraper falhar

A lógica garante economia de créditos da API paga, priorizando sempre
as fontes gratuitas.

v24.0: Arquitetura de orquestração com fallback inteligente.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from data.interfaces.odds_provider import OddsProvider, GameOdds
from data.providers.sbr_scraper import SBRScraper
from data.providers.the_odds_api import TheOddsAPIProvider
from exceptions.odds_exceptions import QuotaExceededException

# Tenta importar OddsPedia como alternativa
try:
    from data.providers.oddspedia_provider import OddsPediaProvider
    ODDSPEDIA_AVAILABLE = True
except ImportError:
    ODDSPEDIA_AVAILABLE = False

logger = logging.getLogger(__name__)


class OddsDataManager:
    """
    Orquestrador de múltiplos provedores de odds.

    Gerencia a cadeia de fallback entre diferentes fontes de dados,
    sempre priorizando fontes gratuitas (scrapers) sobre pagas (APIs).

    Fluxo de execução:
    1. Tenta SBR Scraper (TIER 1 - Gratuito)
    2. Se falhar, tenta TheOddsAPI (TIER 2 - Pago com limite)
    3. Se API falhar ou cota estourada, retorna lista vazia

    Atributos:
        providers: Lista de provedores ordenados por prioridade
        last_source: Nome do último provedor que retornou dados com sucesso
        stats: Estatísticas de uso dos provedores

    Exemplo de uso:
        manager = OddsDataManager()
        odds = await manager.fetch_odds("2024-12-15")
        for game in odds:
            print(f"{game.away_team} @ {game.home_team}: {game.home_odds}")
    """

    def __init__(
        self,
        enable_scraper: bool = True,
        enable_api: bool = True,
        scraper_headless: bool = True,
    ):
        """
        Inicializa o manager com provedores configurados.

        Args:
            enable_scraper: Se True, habilita SBR Scraper
            enable_api: Se True, habilita TheOddsAPI
            scraper_headless: Se True, scraper roda sem GUI
        """
        self.providers: List[OddsProvider] = []
        self.last_source: Optional[str] = None
        self.stats: Dict[str, Any] = {
            "requests": {},
            "successes": {},
            "failures": {},
            "last_fetch": None,
        }

        # Registra provedores por ordem de prioridade
        if enable_scraper:
            self.providers.append(SBRScraper(headless=scraper_headless))
            # OddsPedia como backup do SBR (prioridade 2)
            if ODDSPEDIA_AVAILABLE:
                self.providers.append(OddsPediaProvider(headless=scraper_headless))

        if enable_api:
            self.providers.append(TheOddsAPIProvider())

        # Ordena por prioridade (menor número = maior prioridade)
        self.providers.sort(key=lambda p: p.priority)

        logger.info(
            f"OddsDataManager inicializado com {len(self.providers)} provedores: "
            f"{[p.name for p in self.providers]}"
        )

    def _update_stats(self, provider_name: str, success: bool):
        """Atualiza estatísticas de uso do provedor."""
        if provider_name not in self.stats["requests"]:
            self.stats["requests"][provider_name] = 0
            self.stats["successes"][provider_name] = 0
            self.stats["failures"][provider_name] = 0

        self.stats["requests"][provider_name] += 1

        if success:
            self.stats["successes"][provider_name] += 1
        else:
            self.stats["failures"][provider_name] += 1

    async def fetch_odds(self, date: str) -> List[GameOdds]:
        """
        Busca odds para jogos em uma data específica.

        Implementa Chain of Responsibility: tenta cada provedor em ordem
        de prioridade até um retornar dados com sucesso.

        Args:
            date: Data no formato "YYYY-MM-DD"

        Retorna:
            Lista de GameOdds para todos os jogos encontrados.
            Lista vazia se todos os provedores falharem.
        """
        self.stats["last_fetch"] = datetime.now().isoformat()
        self.last_source = None

        for provider in self.providers:
            logger.info(f"🔄 Tentando provedor: {provider.name} (prioridade {provider.priority})...")

            try:
                odds = await provider.get_odds(date)

                if odds and len(odds) > 0:
                    # Sucesso!
                    self.last_source = provider.name
                    self._update_stats(provider.name, success=True)

                    logger.info(
                        f"✅ {provider.name}: {len(odds)} jogos obtidos com sucesso"
                    )

                    return odds

                else:
                    # Provedor retornou lista vazia
                    logger.warning(
                        f"⚠️ {provider.name}: retornou lista vazia. Tentando próximo..."
                    )
                    self._update_stats(provider.name, success=False)

            except QuotaExceededException as e:
                # Cota da API atingida - não tenta mais APIs pagas
                logger.error(
                    f"🛑 {provider.name}: COTA ESTOURADA - {e.message}"
                )
                self._update_stats(provider.name, success=False)

                # Se for API, não tenta mais (outros provedores pagos também estariam bloqueados)
                if "api" in provider.name.lower():
                    logger.warning("Pulando demais provedores de API devido à cota estourada")
                    continue

            except Exception as e:
                # Falha genérica - tenta próximo provedor
                logger.warning(
                    f"❌ {provider.name} falhou: {e}. Tentando próximo provedor..."
                )
                self._update_stats(provider.name, success=False)

        # Todos os provedores falharam
        logger.error(
            "🚨 TODOS OS PROVEDORES DE ODDS FALHARAM! "
            "Nenhum dado de odds disponível para hoje."
        )

        return []

    async def fetch_odds_with_fallback_info(self, date: str) -> Dict[str, Any]:
        """
        Busca odds e retorna informações detalhadas sobre a fonte.

        Útil para debugging e monitoramento de qual fonte está sendo usada.

        Args:
            date: Data no formato "YYYY-MM-DD"

        Retorna:
            Dict com odds e metadados da busca
        """
        odds = await self.fetch_odds(date)

        return {
            "odds": odds,
            "source": self.last_source,
            "count": len(odds),
            "timestamp": datetime.now().isoformat(),
            "success": len(odds) > 0,
            "date_requested": date,
        }

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Executa health check em todos os provedores.

        Retorna:
            Dict mapeando nome do provedor para status (True/False)
        """
        results = {}

        for provider in self.providers:
            try:
                is_healthy = await provider.health_check()
                results[provider.name] = is_healthy
            except Exception as e:
                logger.error(f"Health check falhou para {provider.name}: {e}")
                results[provider.name] = False

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso dos provedores.

        Retorna:
            Dict com estatísticas detalhadas
        """
        stats_copy = dict(self.stats)

        # Calcula taxas de sucesso
        success_rates = {}
        for provider_name in stats_copy.get("requests", {}):
            total = stats_copy["requests"].get(provider_name, 0)
            if total > 0:
                successes = stats_copy["successes"].get(provider_name, 0)
                success_rates[provider_name] = round(successes / total * 100, 2)
            else:
                success_rates[provider_name] = 0.0

        stats_copy["success_rates"] = success_rates
        stats_copy["providers"] = [p.name for p in self.providers]

        return stats_copy

    async def get_api_quota_status(self) -> Optional[Dict[str, Any]]:
        """
        Obtém status da cota da API (se disponível).

        Retorna:
            Dict com informações da cota ou None se API não configurada
        """
        for provider in self.providers:
            if isinstance(provider, TheOddsAPIProvider):
                return await provider.get_quota_status()

        return None

    async def fetch_player_props(self, date: str) -> List:
        """
        Busca player props (Points, Rebounds, Assists) do Action Network.
        
        Usa o Action Network scraper para extrair props de jogadores.
        Em caso de falha, registra no log mas não interrompe o fluxo.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista de PlayerProp objects. Lista vazia se scraping falhar.
            
        Example:
            props = await manager.fetch_player_props("2024-12-18")
            for prop in props:
                print(f"{prop.player_name}: {prop.prop_type} {prop.line}")
        """
        try:
            # Import aqui para evitar dependência circular
            from data.scrapers.action_network_scraper import ActionNetworkScraper
            
            logger.info(f"🎯 Buscando player props para {date}...")
            
            scraper = ActionNetworkScraper(headless=True)
            props = await scraper.fetch_props(date)
            
            if props:
                logger.info(f"✅ {len(props)} player props encontrados")
                
                # Log estruturado de sucesso
                logger.debug(f"Props por tipo: {self._count_props_by_type(props)}")
            else:
                logger.warning("⚠️ Nenhum player prop encontrado")
            
            return props
            
        except Exception as e:
            logger.error(f"❌ Falha ao buscar player props: {e}", exc_info=True)
            return []
    
    def _count_props_by_type(self, props: List) -> dict:
        """Helper para contar props por tipo para logging."""
        from collections import Counter
        return dict(Counter(prop.prop_type for prop in props))

    async def close(self):
        """Fecha conexões de todos os provedores."""
        for provider in self.providers:
            if hasattr(provider, "close"):
                await provider.close()


# ============================================================================
# Funções de conveniência para uso direto
# ============================================================================

_manager_instance: Optional[OddsDataManager] = None


def get_odds_manager() -> OddsDataManager:
    """
    Obtém instância singleton do OddsDataManager.

    Retorna:
        Instância do OddsDataManager
    """
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = OddsDataManager()

    return _manager_instance


async def fetch_odds_async(date: str) -> List[GameOdds]:
    """
    Função de conveniência para buscar odds de forma assíncrona.

    Args:
        date: Data no formato "YYYY-MM-DD"

    Retorna:
        Lista de GameOdds
    """
    manager = get_odds_manager()
    return await manager.fetch_odds(date)


def fetch_odds_sync(date: str) -> List[GameOdds]:
    """
    Função de conveniência para buscar odds de forma síncrona.

    Útil para scripts que não usam asyncio diretamente.

    Args:
        date: Data no formato "YYYY-MM-DD"

    Retorna:
        Lista de GameOdds
    """
    return asyncio.run(fetch_odds_async(date))


# ============================================================================
# Compatibilidade com código legado
# ============================================================================

def obter_odds_manager(date: str = None) -> Dict[str, Any]:
    """
    Função de compatibilidade com código legado.

    Converte GameOdds para formato Dict usado pelo sistema antigo.

    Args:
        date: Data no formato "YYYY-MM-DD" (ou None para hoje)

    Retorna:
        Dict no formato legado: {game_key: {home_odds, away_odds, ...}}
    """
    from datetime import date as date_class

    if date is None:
        date = date_class.today().isoformat()

    odds_list = fetch_odds_sync(date)

    # Converte para formato legado
    result = {}
    for game in odds_list:
        # Gera chave no formato legado
        home_abbrev = game.home_team[:3].upper()
        away_abbrev = game.away_team[:3].upper()
        game_key = f"{away_abbrev}_{home_abbrev}"

        result[game_key] = {
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_odds": game.home_odds,
            "away_odds": game.away_odds,
            "home_spread": game.home_spread,
            "away_spread": game.away_spread,
            "total_over": game.total_over,
            "over_odds": game.over_odds,
            "under_odds": game.under_odds,
            "bookmaker": game.bookmaker,
            "source": game.source,
            "timestamp": game.timestamp.isoformat() if game.timestamp else None,
        }

    return result
