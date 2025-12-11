# Dataclasses para o domínio do NBA Predictor

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class Team:
    name: str
    abbreviation: str
    net_rating: Optional[float] = None
    power_rating: Optional[float] = None


@dataclass
class Player:
    name: str
    team: str
    position: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


@dataclass
class Game:
    date: datetime
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    odds: Optional[dict] = None
    net_rating_home: Optional[float] = None
    net_rating_away: Optional[float] = None
