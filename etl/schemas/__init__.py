"""
Schemas Pydantic - Validação de Dados de Entrada
=================================================
Schemas para validação robusta de dados da NBA.
Dados inválidos são rejeitados e vão para Dead Letter Queue.

Autor: NBA Predictor v22.0
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum


# ============= CONSTANTES =============

VALID_NBA_TEAMS = {
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
}


class GameStatus(str, Enum):
    """Status possíveis de um jogo"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class BetType(str, Enum):
    """Tipos de aposta suportados"""
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTALS = "totals"
    PLAYER_PROPS = "player_props"


class BetResult(str, Enum):
    """Resultados possíveis de uma aposta"""
    PENDING = "PENDING"
    WON = "WON"
    LOST = "LOST"
    PUSH = "PUSH"
    VOID = "VOID"


# ============= SCHEMAS DE JOGOS =============

class GameSchema(BaseModel):
    """
    Schema de validação para jogos da NBA.
    
    Valida:
    - Times válidos da NBA (30 times)
    - Datas no formato correto
    - Placares não negativos
    """
    game_id: str = Field(..., min_length=5, description="ID único do jogo")
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$', description="Data YYYY-MM-DD")
    home_team: str = Field(..., description="Time da casa (3 letras)")
    away_team: str = Field(..., description="Time visitante (3 letras)")
    home_score: Optional[int] = Field(None, ge=0, description="Placar casa")
    away_score: Optional[int] = Field(None, ge=0, description="Placar visitante")
    status: GameStatus = Field(default=GameStatus.SCHEDULED)
    season: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}$')
    
    @validator('home_team', 'away_team')
    def validate_team(cls, v):
        """Valida se o time é uma equipe válida da NBA"""
        v = v.upper().strip()
        if v not in VALID_NBA_TEAMS:
            raise ValueError(f"Time inválido: '{v}'. Deve ser um dos: {', '.join(sorted(VALID_NBA_TEAMS))}")
        return v
    
    @validator('date')
    def validate_date(cls, v):
        """Valida se a data é uma data válida"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Data inválida: '{v}'. Formato esperado: YYYY-MM-DD")
        return v
    
    @root_validator
    def validate_scores(cls, values):
        """Valida que se um placar existe, os dois devem existir"""
        home = values.get('home_score')
        away = values.get('away_score')
        
        if (home is None) != (away is None):
            raise ValueError("Ambos os placares devem ser fornecidos ou nenhum")
        
        return values
    
    class Config:
        use_enum_values = True


# ============= SCHEMAS DE ODDS =============

class OddsSchema(BaseModel):
    """
    Schema de validação para odds de apostas.
    
    Valida:
    - Odds dentro de range realista (1.01 a 100.0)
    - Formato decimal europeu
    """
    game_id: str = Field(..., min_length=5)
    home_odds: float = Field(..., gt=1.0, le=100.0, description="Odd casa (decimal)")
    away_odds: float = Field(..., gt=1.0, le=100.0, description="Odd visitante (decimal)")
    draw_odds: Optional[float] = Field(None, gt=1.0, le=100.0)
    spread_home: Optional[float] = Field(None, ge=-30.0, le=30.0)
    spread_away: Optional[float] = Field(None, ge=-30.0, le=30.0)
    spread_home_odds: Optional[float] = Field(None, gt=1.0, le=10.0)
    spread_away_odds: Optional[float] = Field(None, gt=1.0, le=10.0)
    total_line: Optional[float] = Field(None, ge=150.0, le=300.0, description="Linha de total")
    over_odds: Optional[float] = Field(None, gt=1.0, le=10.0)
    under_odds: Optional[float] = Field(None, gt=1.0, le=10.0)
    source: str = Field(default="unknown", description="Fonte das odds")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('home_odds', 'away_odds')
    def validate_odds(cls, v):
        """Valida odds dentro de limites realistas"""
        if v <= 1.0:
            raise ValueError(f"Odd inválida: {v}. Deve ser maior que 1.0")
        if v > 100.0:
            raise ValueError(f"Odd muito alta: {v}. Máximo permitido: 100.0")
        return round(v, 2)
    
    @root_validator
    def validate_market_consistency(cls, values):
        """Valida consistência do mercado (overround razoável)"""
        home = values.get('home_odds')
        away = values.get('away_odds')
        
        if home and away:
            # Calcula overround (margem da casa)
            overround = (1/home + 1/away) * 100 - 100
            
            # Overround típico: 2-10%
            if overround < -5 or overround > 30:
                raise ValueError(f"Overround suspeito: {overround:.1f}%. Odds podem estar erradas")
        
        return values
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============= SCHEMAS DE PREVISÕES =============

class PredictionSchema(BaseModel):
    """
    Schema de validação para previsões do modelo.
    """
    game_id: str = Field(...)
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    home_team: str = Field(...)
    away_team: str = Field(...)
    prob_home: float = Field(..., ge=0.01, le=0.99, description="Probabilidade casa")
    prob_away: float = Field(..., ge=0.01, le=0.99, description="Probabilidade visitante")
    prediction: str = Field(..., description="Previsão: HOME ou AWAY")
    confidence: Optional[str] = Field(None)
    predicted_spread: Optional[float] = Field(None, ge=-50.0, le=50.0)
    predicted_total: Optional[float] = Field(None, ge=150.0, le=300.0)
    model_version: str = Field(default="v22.0")
    
    @validator('home_team', 'away_team')
    def validate_team(cls, v):
        v = v.upper().strip()
        if v not in VALID_NBA_TEAMS:
            raise ValueError(f"Time inválido: '{v}'")
        return v
    
    @root_validator
    def validate_probabilities(cls, values):
        """Valida que probabilidades somam ~100%"""
        home = values.get('prob_home', 0)
        away = values.get('prob_away', 0)
        
        total = home + away
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"Probabilidades devem somar ~100%: {total*100:.1f}%")
        
        return values


# ============= SCHEMAS DE APOSTAS =============

class BetSchema(BaseModel):
    """
    Schema de validação para apostas.
    """
    game_id: str = Field(...)
    bet_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    home_team: str = Field(...)
    away_team: str = Field(...)
    side: str = Field(..., pattern=r'^(HOME|AWAY|OVER|UNDER)$')
    bet_type: BetType = Field(...)
    line: Optional[float] = Field(None)
    bet_odds: float = Field(..., gt=1.0, le=100.0)
    stake_pct: float = Field(..., gt=0.0, le=0.25, description="% do bankroll (max 25%)")
    stake_amount: float = Field(..., gt=0.0)
    model_prob: float = Field(..., ge=0.0, le=1.0)
    ev_pct: float = Field(..., ge=-100.0, le=500.0, description="Expected Value %")
    result: BetResult = Field(default=BetResult.PENDING)
    
    @validator('stake_pct')
    def validate_stake(cls, v):
        """Limite de stake para gestão de risco"""
        if v > 0.25:
            raise ValueError(f"Stake muito alto: {v*100:.1f}%. Máximo: 25%")
        return v
    
    @validator('ev_pct')
    def validate_ev(cls, v):
        """Valida que EV está em range razoável"""
        if v < -10:
            raise ValueError(f"EV muito negativo: {v:.1f}%. Aposta não recomendada")
        return v
    
    class Config:
        use_enum_values = True


# ============= SCHEMAS DE LESÕES =============

class InjurySchema(BaseModel):
    """
    Schema de validação para dados de lesões.
    """
    player_name: str = Field(..., min_length=2)
    team: str = Field(...)
    status: str = Field(..., pattern=r'^(OUT|DOUBTFUL|QUESTIONABLE|PROBABLE|AVAILABLE)$')
    injury_type: Optional[str] = Field(None)
    return_date: Optional[str] = Field(None)
    impact_score: float = Field(default=0.08, ge=0.0, le=0.35)
    source: str = Field(default="unknown")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('team')
    def validate_team(cls, v):
        v = v.upper().strip()
        if v not in VALID_NBA_TEAMS:
            raise ValueError(f"Time inválido: '{v}'")
        return v


# ============= SCHEMA DE FEATURES =============

class FeatureSchema(BaseModel):
    """
    Schema de validação para features do Feature Store.
    """
    entity_id: str = Field(..., min_length=3)
    entity_type: str = Field(..., pattern=r'^(team|game|player)$')
    feature_name: str = Field(..., min_length=3, max_length=100)
    feature_value: float = Field(...)
    valid_from: datetime = Field(...)
    valid_to: Optional[datetime] = Field(None)
    version: str = Field(default="1.0")
    
    @root_validator
    def validate_dates(cls, values):
        """Valida que valid_from < valid_to"""
        from_date = values.get('valid_from')
        to_date = values.get('valid_to')
        
        if from_date and to_date and from_date >= to_date:
            raise ValueError("valid_from deve ser anterior a valid_to")
        
        return values


# ============= HELPERS DE VALIDAÇÃO =============

def validate_game(data: Dict[str, Any]) -> GameSchema:
    """Valida dados de jogo e retorna schema"""
    return GameSchema(**data)


def validate_odds(data: Dict[str, Any]) -> OddsSchema:
    """Valida dados de odds e retorna schema"""
    return OddsSchema(**data)


def validate_prediction(data: Dict[str, Any]) -> PredictionSchema:
    """Valida dados de previsão e retorna schema"""
    return PredictionSchema(**data)


def validate_bet(data: Dict[str, Any]) -> BetSchema:
    """Valida dados de aposta e retorna schema"""
    return BetSchema(**data)
