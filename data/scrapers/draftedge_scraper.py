"""
DraftEdge Scraper - Projeções de fantasy e props de volume.

NOTA: DraftEdge requer autenticação/assinatura para acesso completo.
Esta implementação fornece um stub que pode ser expandido com credenciais.

v26.3: Implementação skeleton para futura expansão.
"""

import logging
from datetime import datetime
from typing import List

from data.interfaces.player_props_provider import PlayerPropsProvider, PlayerProp

logger = logging.getLogger(__name__)


class DraftEdgeScraper(PlayerPropsProvider):
    """
    Scraper para DraftEdge.com - projeções fantasy e props.
    
    IMPORTANTE: Requer autenticação para acesso completo.
    Atualmente retorna lista vazia - implemente login para produção.
    """
    
    BASE_URL = "https://www.draftedge.com/nba-projections"
    
    def __init__(self, headless: bool = True, credentials: dict = None):
        """
        Inicializa o scraper.
        
        Args:
            headless: Modo headless
            credentials: Dict com 'username' e 'password' (opcional)
        """
        self.headless = headless
        self.credentials = credentials
    
    @property
    def name(self) -> str:
        return "draftedge"
    
    @property
    def priority(self) -> int:
        return 6  # Baixa prioridade devido a restrições de acesso
    
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Obtém props do DraftEdge.
        
        NOTA: Requer implementação de login para funcionar.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista vazia (implemente login para produção)
        """
        if not self.credentials:
            logger.info("⚠️ DraftEdge: Credenciais não fornecidas, pulando...")
            return []
        
        # TODO: Implementar login e extração quando credenciais disponíveis
        logger.warning("⚠️ DraftEdge: Implementação de login pendente")
        return []
    
    async def health_check(self) -> bool:
        """Verifica disponibilidade."""
        return False  # Não funcional sem credenciais
    
    def get_supported_prop_types(self) -> List[str]:
        return ["points", "rebounds", "assists", "fantasy_points"]
