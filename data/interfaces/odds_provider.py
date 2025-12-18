"""
Interface abstrata para provedores de odds.

Este módulo define o contrato que todos os provedores de odds devem implementar,
garantindo consistência entre diferentes fontes de dados (scrapers, APIs, etc).

v24.0: Arquitetura de provedores com prioridade definida.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class GameOdds:
    """
    Representa as odds de um jogo NBA.
    
    Atributos:
        game_id: Identificador único do jogo (ex: "LAL_vs_BOS_2024-12-15")
        home_team: Nome do time da casa
        away_team: Nome do time visitante
        home_odds: Odds decimal para vitória do time da casa
        away_odds: Odds decimal para vitória do time visitante
        home_spread: Spread (handicap) do time da casa (ex: -5.5)
        away_spread: Spread (handicap) do time visitante (ex: +5.5)
        total_over: Linha de total para Over
        total_under: Linha de total para Under
        over_odds: Odds para Over
        under_odds: Odds para Under
        bookmaker: Nome da casa de apostas que originou as odds
        source: Fonte dos dados ("sbr_scraper", "theoddsapi", etc.)
        timestamp: Momento da coleta dos dados
    """
    game_id: str
    home_team: str
    away_team: str
    home_odds: float
    away_odds: float
    home_spread: Optional[float] = None
    away_spread: Optional[float] = None
    total_over: Optional[float] = None
    total_under: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    bookmaker: str = "consensus"
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Converte para dicionário para serialização."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "home_spread": self.home_spread,
            "away_spread": self.away_spread,
            "total_over": self.total_over,
            "total_under": self.total_under,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "bookmaker": self.bookmaker,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GameOdds":
        """Cria instância a partir de dicionário."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            game_id=data["game_id"],
            home_team=data["home_team"],
            away_team=data["away_team"],
            home_odds=data["home_odds"],
            away_odds=data["away_odds"],
            home_spread=data.get("home_spread"),
            away_spread=data.get("away_spread"),
            total_over=data.get("total_over"),
            total_under=data.get("total_under"),
            over_odds=data.get("over_odds"),
            under_odds=data.get("under_odds"),
            bookmaker=data.get("bookmaker", "consensus"),
            source=data.get("source", "unknown"),
            timestamp=timestamp or datetime.now(),
        )


class OddsProvider(ABC):
    """
    Interface abstrata para provedores de odds.
    
    Todos os provedores (scrapers, APIs) devem implementar esta interface
    para garantir consistência e permitir intercâmbio fácil entre fontes.
    
    Exemplo de implementação:
        class SBRScraper(OddsProvider):
            async def get_odds(self, date: str) -> List[GameOdds]:
                # Implementação do scraper...
                return [GameOdds(...), GameOdds(...)]
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nome identificador do provedor.
        
        Retorna:
            String identificando o provedor (ex: "sbr_scraper", "theoddsapi")
        """
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Prioridade do provedor (menor = maior prioridade).
        
        Retorna:
            Inteiro indicando prioridade (1 = mais alta, 10 = mais baixa)
        """
        pass
    
    @abstractmethod
    async def get_odds(self, date: str) -> List[GameOdds]:
        """
        Obtém odds para jogos em uma data específica.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Retorna:
            Lista de GameOdds para todos os jogos encontrados
            
        Raises:
            Exception: Se a obtenção de dados falhar
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica se o provedor está funcional.
        
        Retorna:
            True se o provedor está operacional, False caso contrário
        """
        pass
    
    async def get_player_props(self, date: str):
        """
        Obtém player props (Points, Rebounds, Assists) para uma data específica.
        
        OPCIONAL: Providers que não suportam player props não precisam implementar.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista de PlayerProp objects
            
        Raises:
            NotImplementedError: Se o provider não implementa player props
        """
        raise NotImplementedError(f"{self.name} does not support player props")
