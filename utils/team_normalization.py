"""
Team Name Normalization - Módulo Centralizado

Resolve inconsistências na representação de times através de normalização única e robusta.

Problema resolvido:
- Múltiplas formas do mesmo time: "LAL", "Lakers", "Los Angeles Lakers", "LA Lakers"
- Falsos negativos no matching de resultados (~5-10 jogos/mês)
- Código duplicado em 5+ arquivos diferentes

Solução:
- Singleton pattern para garantir consistência em todo o sistema
- Mapeamento bidirecional (ID ↔ Full Name)
- Cache para performance
- Validação de IDs
"""
import logging
from typing import Dict, List, Optional, Set
from functools import lru_cache

logger = logging.getLogger(__name__)


class TeamNormalizer:
    """
    Singleton para normalização centralizada de nomes de times NBA.
    
    Features:
    - Conversão entre todas as formas válidas de um time
    - Cache automático para performance
    - Validação de IDs e nomes
    - Thread-safe (imutável após inicialização)
    
    Usage:
        normalizer = TeamNormalizer.get_instance()
        
        # Normalizar para ID de 3 letras
        normalizer.normalize("Lakers")  # → "LAL"
        normalizer.normalize("Los Angeles Lakers")  # → "LAL"
        normalizer.normalize("LAL")  # → "LAL"
        
        # Converter para nome completo
        normalizer.to_full_name("LAL")  # → "Los Angeles Lakers"
        
        # Obter todas as formas válidas
        normalizer.all_forms("LAL")  # → ["LAL", "Lakers", "Los Angeles Lakers", "LA Lakers"]
    """
    
    _instance: Optional['TeamNormalizer'] = None
    
    # Mapa canônico: Full Name → 3-letter ID
    TEAM_ID_MAP: Dict[str, str] = {
        # Eastern Conference - Atlantic
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",  # Updated to modern BKN
        "New York Knicks": "NYK",
        "Philadelphia 76ers": "PHI",
        "Toronto Raptors": "TOR",
        
        # Eastern Conference - Central
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Detroit Pistons": "DET",
        "Indiana Pacers": "IND",
        "Milwaukee Bucks": "MIL",
        
        # Eastern Conference - Southeast
        "Atlanta Hawks": "ATL",
        "Charlotte Hornets": "CHA",  # Updated to modern CHA
        "Miami Heat": "MIA",
        "Orlando Magic": "ORL",
        "Washington Wizards": "WAS",
        
        # Western Conference - Northwest
        "Denver Nuggets": "DEN",
        "Minnesota Timberwolves": "MIN",
        "Oklahoma City Thunder": "OKC",
        "Portland Trail Blazers": "POR",
        "Utah Jazz": "UTA",
        
        # Western Conference - Pacific
        "Golden State Warriors": "GSW",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Phoenix Suns": "PHX",  # Updated to modern PHX
        "Sacramento Kings": "SAC",
        
        # Western Conference - Southwest
        "Dallas Mavericks": "DAL",
        "Houston Rockets": "HOU",
        "Memphis Grizzlies": "MEM",
        "New Orleans Pelicans": "NOP",
        "San Antonio Spurs": "SAS",
    }
    
    # Aliases e variações comuns (lowercase para case-insensitive matching)
    TEAM_ALIASES: Dict[str, str] = {
        # Lakers
        "lakers": "Los Angeles Lakers",
        "la lakers": "Los Angeles Lakers",
        "l.a. lakers": "Los Angeles Lakers",
        
        # Clippers
        "clippers": "Los Angeles Clippers",
        "la clippers": "Los Angeles Clippers",
        "l.a. clippers": "Los Angeles Clippers",
        
        # Warriors
        "warriors": "Golden State Warriors",
        "gsw": "Golden State Warriors",
        "golden state": "Golden State Warriors",
        
        # Celtics
        "celtics": "Boston Celtics",
        "boston": "Boston Celtics",
        
        # 76ers variations
        "76ers": "Philadelphia 76ers",
        "sixers": "Philadelphia 76ers",
        "philadelphia": "Philadelphia 76ers",
        
        # Nets
        "nets": "Brooklyn Nets",
        "brooklyn": "Brooklyn Nets",
        
        # Knicks
        "knicks": "New York Knicks",
        "new york": "New York Knicks",
        "ny knicks": "New York Knicks",
        
        # Cavaliers
        "cavaliers": "Cleveland Cavaliers",
        "cavs": "Cleveland Cavaliers",
        "cleveland": "Cleveland Cavaliers",
        
        # Bulls
        "bulls": "Chicago Bulls",
        "chicago": "Chicago Bulls",
        
        # Heat
        "heat": "Miami Heat",
        "miami": "Miami Heat",
        
        # Bucks
        "bucks": "Milwaukee Bucks",
        "milwaukee": "Milwaukee Bucks",
        
        # Hawks
        "hawks": "Atlanta Hawks",
        "atlanta": "Atlanta Hawks",
        
        # Hornets
        "hornets": "Charlotte Hornets",
        "charlotte": "Charlotte Hornets",
        
        # Magic
        "magic": "Orlando Magic",
        "orlando": "Orlando Magic",
        
        # Wizards
        "wizards": "Washington Wizards",
        "washington": "Washington Wizards",
        
        # Raptors
        "raptors": "Toronto Raptors",
        "toronto": "Toronto Raptors",
        
        # Pistons
        "pistons": "Detroit Pistons",
        "detroit": "Detroit Pistons",
        
        # Pacers
        "pacers": "Indiana Pacers",
        "indiana": "Indiana Pacers",
        
        # Nuggets
        "nuggets": "Denver Nuggets",
        "denver": "Denver Nuggets",
        
        # Timberwolves
        "timberwolves": "Minnesota Timberwolves",
        "twolves": "Minnesota Timberwolves",
        "minnesota": "Minnesota Timberwolves",
        
        # Thunder
        "thunder": "Oklahoma City Thunder",
        "okc": "Oklahoma City Thunder",
        "oklahoma city": "Oklahoma City Thunder",
        
        # Trail Blazers
        "trail blazers": "Portland Trail Blazers",
        "blazers": "Portland Trail Blazers",
        "portland": "Portland Trail Blazers",
        
        # Jazz
        "jazz": "Utah Jazz",
        "utah": "Utah Jazz",
        
        # Suns
        "suns": "Phoenix Suns",
        "phoenix": "Phoenix Suns",
        "phx": "Phoenix Suns",  # Common abbreviation
        
        # Kings
        "kings": "Sacramento Kings",
        "sacramento": "Sacramento Kings",
        "sac": "Sacramento Kings",
        
        # Mavericks
        "mavericks": "Dallas Mavericks",
        "mavs": "Dallas Mavericks",
        "dallas": "Dallas Mavericks",
        
        # Rockets
        "rockets": "Houston Rockets",
        "houston": "Houston Rockets",
        
        # Grizzlies
        "grizzlies": "Memphis Grizzlies",
        "grizz": "Memphis Grizzlies",
        "memphis": "Memphis Grizzlies",
        
        # Pelicans
        "pelicans": "New Orleans Pelicans",
        "pels": "New Orleans Pelicans",
        "new orleans": "New Orleans Pelicans",
        "no": "New Orleans Pelicans",
        
        # Spurs
        "spurs": "San Antonio Spurs",
        "san antonio": "San Antonio Spurs",
        "sa": "San Antonio Spurs",
        
        # Hornets (Extra)
        "cha": "Charlotte Hornets",
        
        # Nets (Extra)
        "bkn": "Brooklyn Nets",
        
        # Suns (Extra - variação comum)
        "phx": "Phoenix Suns",
    }
    
    def __init__(self):
        """Inicialização privada - use get_instance() ao invés."""
        # Criar mapa reverso: ID → Full Name
        self._id_to_full = {v: k for k, v in self.TEAM_ID_MAP.items()}
        
        # Criar set de IDs válidos para validação rápida
        self._valid_ids: Set[str] = set(self.TEAM_ID_MAP.values())
        
        # Criar mapa de todas as variações → Full Name
        self._all_variations: Dict[str, str] = {}
        
        # Adicionar Full Names (case-insensitive)
        for full_name in self.TEAM_ID_MAP.keys():
            self._all_variations[full_name.lower()] = full_name
        
        # Adicionar IDs (já uppercase)
        for team_id in self._valid_ids:
            self._all_variations[team_id.lower()] = self._id_to_full[team_id]
        
        # Adicionar aliases
        self._all_variations.update(self.TEAM_ALIASES)
        
        logger.info(
            f"✅ TeamNormalizer inicializado: {len(self.TEAM_ID_MAP)} times, "
            f"{len(self._all_variations)} variações mapeadas"
        )
    
    @classmethod
    def get_instance(cls) -> 'TeamNormalizer':
        """
        Obtém a instância singleton do normalizador.
        
        Returns:
            Instância única de TeamNormalizer
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @lru_cache(maxsize=128)
    def normalize(self, team_name: Optional[str]) -> Optional[str]:
        """
        Normaliza qualquer forma de nome de time para ID de 3 letras.
        
        ROBUSTNESS FIX: Implementa fuzzy matching com difflib para variações
        como "LA Clippers" vs "L.A. Clippers".
        
        Args:
            team_name: Nome do time em qualquer formato
            
        Returns:
            ID de 3 letras (ex: "LAL", "GSW") ou None se não encontrado
            
        Examples:
            >>> normalizer.normalize("Lakers")
            'LAL'
            >>> normalizer.normalize("Los Angeles Lakers")
            'LAL'
            >>> normalizer.normalize("LAL")
            'LAL'
            >>> normalizer.normalize("LA Lakers")
            'LAL'
            >>> normalizer.normalize("L.A. Clippers")  # Fuzzy match
            'LAC'
        """
        if not team_name:
            return None
        
        # Limpar entrada
        clean_name = team_name.strip()
        lookup_key = clean_name.lower()
        
        # 1. Verificar se já é um ID válido (ex: "LAL")
        if clean_name.upper() in self._valid_ids:
            return clean_name.upper()
        
        # 2. Buscar match exato em variações
        full_name = self._all_variations.get(lookup_key)
        
        if full_name:
            return self.TEAM_ID_MAP[full_name]
        
        # 3. ROBUSTNESS FIX: Fuzzy Match com difflib (similaridade >= 80%)
        import difflib
        all_keys = list(self._all_variations.keys())
        matches = difflib.get_close_matches(
            lookup_key,
            all_keys,
            n=1,
            cutoff=0.80  # 80% similaridade mínima
        )
        
        if matches:
            matched_key = matches[0]
            full_name = self._all_variations[matched_key]
            similarity = difflib.SequenceMatcher(
                None, lookup_key, matched_key
            ).ratio()
            
            logger.info(
                f"🔍 Fuzzy match: '{team_name}' → '{matched_key}' "
                f"(score: {similarity:.2%})"
            )
            return self.TEAM_ID_MAP[full_name]
        
        # 4. Não encontrado (nem com fuzzy)
        logger.warning(f"❌ Team não encontrado (nem fuzzy): '{team_name}'")
        return None
    
    @lru_cache(maxsize=32)
    def to_full_name(self, team_id: Optional[str]) -> Optional[str]:
        """
        Converte ID de 3 letras para nome completo.
        
        Args:
            team_id: ID do time (ex: "LAL", "GSW")
            
        Returns:
            Nome completo (ex: "Los Angeles Lakers") ou None
            
        Examples:
            >>> normalizer.to_full_name("LAL")
            'Los Angeles Lakers'
            >>> normalizer.to_full_name("GSW")
            'Golden State Warriors'
        """
        if not team_id:
            return None
        
        team_id_upper = team_id.strip().upper()
        return self._id_to_full.get(team_id_upper)
    
    def all_forms(self, identifier: str) -> List[str]:
        """
        Retorna todas as formas válidas de um time.
        
        Args:
            identifier: Qualquer identificador do time (ID, nome, alias)
            
        Returns:
            Lista de todas as formas conhecidas
            
        Examples:
            >>> normalizer.all_forms("LAL")
            ['LAL', 'Los Angeles Lakers', 'Lakers', 'LA Lakers', 'L.A. Lakers']
        """
        # Primeiro normalizar para ID
        team_id = self.normalize(identifier)
        
        if not team_id:
            return []
        
        full_name = self.to_full_name(team_id)
        full_name_lower = full_name.lower()
        
        # Coletar todas as formas
        forms = [team_id, full_name]
        
        # Adicionar aliases que apontam para este time
        for alias, target_full_name in self.TEAM_ALIASES.items():
            if target_full_name.lower() == full_name_lower:
                forms.append(alias.title())  # Capitalize first letter
        
        return list(set(forms))  # Remove duplicatas
    
    def is_valid_id(self, team_id: str) -> bool:
        """Verifica se um ID é válido."""
        return team_id.upper() in self._valid_ids if team_id else False
    
    def get_all_ids(self) -> List[str]:
        """Retorna lista de todos os IDs válidos."""
        return sorted(self._valid_ids)
    
    def get_all_teams(self) -> List[str]:
        """Retorna lista de todos os nomes completos."""
        return sorted(self.TEAM_ID_MAP.keys())


# Convenience functions para uso direto
def normalize_team(team_name: Optional[str]) -> Optional[str]:
    """Atalho para TeamNormalizer.get_instance().normalize()"""
    return TeamNormalizer.get_instance().normalize(team_name)


def team_to_full_name(team_id: Optional[str]) -> Optional[str]:
    """Atalho para TeamNormalizer.get_instance().to_full_name()"""
    return TeamNormalizer.get_instance().to_full_name(team_id)


def get_team_forms(identifier: str) -> List[str]:
    """Atalho para TeamNormalizer.get_instance().all_forms()"""
    return TeamNormalizer.get_instance().all_forms(identifier)
