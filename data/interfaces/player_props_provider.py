"""
Player Props Provider Interface - Contrato para scrapers de props NBA.

Define a interface abstrata que todos os provedores de player props devem implementar,
garantindo consistência entre diferentes fontes (Linemate, BettingPros, Covers, etc).

v26.3: Criado para arquitetura de múltiplos scrapers de player props.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class PlayerProp:
    """
    Representa um prop bet de jogador NBA.
    
    Atributos:
        player_name: Nome canônico do jogador (normalizado)
        prop_type: Tipo do prop ('points', 'rebounds', 'assists', 'threes', etc)
        line: Valor da linha (ex: 25.5 pontos)
        over_odds: Odds decimal para Over
        under_odds: Odds decimal para Under
        bookmaker: Casa de apostas fonte (opcional)
        source: Nome do scraper/provedor
        timestamp: Momento da coleta
        game_info: Informações do jogo (opcional)
    """
    player_name: str
    prop_type: str
    line: float
    over_odds: float
    under_odds: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    bookmaker: Optional[str] = None
    game_info: Optional[str] = None
    
    def __post_init__(self):
        """Valida campos obrigatórios após criação."""
        if self.over_odds <= 1.0 or self.under_odds <= 1.0:
            raise ValueError(f"Odds inválidas: over={self.over_odds}, under={self.under_odds}")
        if self.line < 0:
            raise ValueError(f"Linha negativa não permitida: {self.line}")
    
    def to_dict(self) -> dict:
        """Converte para dicionário para serialização."""
        return {
            "player_name": self.player_name,
            "prop_type": self.prop_type,
            "line": self.line,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "bookmaker": self.bookmaker,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "game_info": self.game_info,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProp":
        """Cria instância a partir de dicionário."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            player_name=data["player_name"],
            prop_type=data["prop_type"],
            line=data["line"],
            over_odds=data["over_odds"],
            under_odds=data["under_odds"],
            bookmaker=data.get("bookmaker"),
            source=data.get("source", "unknown"),
            timestamp=timestamp or datetime.now(),
            game_info=data.get("game_info"),
        )


class PlayerPropsProvider(ABC):
    """
    Interface abstrata para provedores de player props.
    
    Todos os scrapers de player props (Linemate, BettingPros, Covers, etc)
    devem implementar esta interface para garantir consistência.
    
    Exemplo de implementação:
        class LinemateScraper(PlayerPropsProvider):
            async def get_props(self, date: str) -> List[PlayerProp]:
                # Implementação do scraper...
                return [PlayerProp(...), PlayerProp(...)]
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nome identificador do provedor.
        
        Returns:
            String identificando o provedor (ex: "linemate", "bettingpros")
        """
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Prioridade do provedor (menor = maior prioridade).
        
        Returns:
            Inteiro indicando prioridade (1 = mais alta, 10 = mais baixa)
        """
        pass
    
    @abstractmethod
    async def get_props(self, date: str) -> List[PlayerProp]:
        """
        Obtém player props para jogos em uma data específica.
        
        Args:
            date: Data no formato "YYYY-MM-DD"
            
        Returns:
            Lista de PlayerProp para todos os jogadores encontrados.
            Lista vazia se nenhum prop encontrado (NÃO retorna valores fictícios).
            
        Raises:
            Exception: Se a obtenção de dados falhar completamente
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica se o provedor está funcional.
        
        Returns:
            True se o provedor está operacional, False caso contrário
        """
        pass
    
    def get_supported_prop_types(self) -> List[str]:
        """
        Retorna tipos de props suportados por este provider.
        
        Returns:
            Lista de tipos suportados (ex: ["points", "rebounds", "assists"])
        """
        return ["points", "rebounds", "assists"]  # Default - override se necessário
