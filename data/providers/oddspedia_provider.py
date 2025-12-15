"""
OddsPedia Provider - Adaptador do scraper existente para nova interface.

Este módulo adapta o OddsPediaScraper existente para a nova interface OddsProvider,
permitindo sua utilização no OddsDataManager.

v24.0: Adaptador para compatibilidade com arquitetura de provedores.
"""

import logging
from datetime import datetime
from typing import List, Optional

from data.interfaces.odds_provider import OddsProvider, GameOdds

logger = logging.getLogger(__name__)

# Importa scraper existente
try:
    from data.scrapers.odds_web_scraper import OddsPediaScraper as LegacyScraper
    ODDSPEDIA_AVAILABLE = True
except ImportError:
    ODDSPEDIA_AVAILABLE = False
    logger.warning("OddsPediaScraper não disponível")


class OddsPediaProvider(OddsProvider):
    """
    Provider que adapta o OddsPediaScraper existente para a interface OddsProvider.

    Prioridade: 2 (segunda opção após SBR)
    """

    def __init__(self, headless: bool = True):
        """
        Inicializa o provider.

        Args:
            headless: Se True, executa navegador em modo headless
        """
        self.headless = headless
        self._scraper = LegacyScraper(headless=headless) if ODDSPEDIA_AVAILABLE else None

    @property
    def name(self) -> str:
        """Nome identificador do provedor."""
        return "oddspedia_scraper"

    @property
    def priority(self) -> int:
        """Prioridade do provedor (2 = segunda opção)."""
        return 2

    def _convert_to_game_odds(self, legacy_dict: dict) -> List[GameOdds]:
        """
        Converte formato legado do OddsPediaScraper para List[GameOdds].

        Args:
            legacy_dict: Dict retornado pelo scraper legado

        Returns:
            Lista de GameOdds
        """
        games = []

        for game_key, data in legacy_dict.items():
            try:
                home_team = data.get("home_team", "")
                away_team = data.get("away_team", "")

                if not home_team or not away_team:
                    continue

                game_id = f"{away_team.replace(' ', '_')}_vs_{home_team.replace(' ', '_')}_{datetime.now().strftime('%Y-%m-%d')}"

                game = GameOdds(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    home_odds=data.get("home_odds", 1.90),
                    away_odds=data.get("away_odds", 1.90),
                    bookmaker="oddspedia",
                    source=self.name,
                    timestamp=datetime.now(),
                )
                games.append(game)

            except Exception as e:
                logger.debug(f"Erro ao converter jogo: {e}")
                continue

        return games

    async def get_odds(self, date: str) -> List[GameOdds]:
        """
        Obtém odds do OddsPedia.

        Args:
            date: Data no formato "YYYY-MM-DD" (não usado pelo OddsPedia)

        Returns:
            Lista de GameOdds

        Raises:
            RuntimeError: Se scraper não disponível
        """
        if not ODDSPEDIA_AVAILABLE or self._scraper is None:
            raise RuntimeError("OddsPediaScraper não está disponível")

        logger.info(f"🔍 OddsPedia Scraper: Buscando odds...")

        try:
            # O scraper legado é síncrono
            legacy_result = self._scraper.fetch_odds()

            if not legacy_result:
                logger.warning("⚠️ OddsPedia retornou resultado vazio")
                return []

            games = self._convert_to_game_odds(legacy_result)

            if games:
                logger.info(f"✅ OddsPedia: {len(games)} jogos encontrados")
            else:
                logger.warning("⚠️ OddsPedia: Nenhum jogo convertido com sucesso")

            return games

        except Exception as e:
            logger.error(f"❌ OddsPedia Scraper falhou: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Verifica se o provider está funcional.

        Returns:
            True se disponível, False caso contrário
        """
        return ODDSPEDIA_AVAILABLE and self._scraper is not None
