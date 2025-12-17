# Constantes compartilhadas do projeto
SYSTEM_VERSION = "v26.2"

TEAM_ABBREV_MAP = {
    "Indiana Pacers": "IND", "Cleveland Cavaliers": "CLE", "Brooklyn Nets": "BRK",
    "Boston Celtics": "BOS", "Washington Wizards": "WAS", "Toronto Raptors": "TOR",
    "Miami Heat": "MIA", "Chicago Bulls": "CHI", "New York Knicks": "NYK",
    "Atlanta Hawks": "ATL", "Orlando Magic": "ORL", "Detroit Pistons": "DET",
    "Charlotte Hornets": "CHO",  # FIX: Excel usa CHO, não CHA
    "Philadelphia 76ers": "PHI", "Milwaukee Bucks": "MIL",
    
    "Dallas Mavericks": "DAL", "Minnesota Timberwolves": "MIN", 
    "Phoenix Suns": "PHO",  # FIX: Excel usa PHO, não PHX
    "Denver Nuggets": "DEN", "Houston Rockets": "HOU", "Oklahoma City Thunder": "OKC",
    "Utah Jazz": "UTA", "Portland Trail Blazers": "POR", "Golden State Warriors": "GSW",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "New Orleans Pelicans": "NOP",
}

TEAMS_MAP = {
    "indiana": "Indiana Pacers", "pacers": "Indiana Pacers",
    "cleveland": "Cleveland Cavaliers", "cavaliers": "Cleveland Cavaliers", "cavs": "Cleveland Cavaliers",
    "brooklyn": "Brooklyn Nets", "nets": "Brooklyn Nets",
    "boston": "Boston Celtics", "celtics": "Boston Celtics",
    "washington": "Washington Wizards", "wizards": "Washington Wizards",
    "toronto": "Toronto Raptors", "raptors": "Toronto Raptors",
    "miami": "Miami Heat", "heat": "Miami Heat",
    "chicago": "Chicago Bulls", "bulls": "Chicago Bulls",
    "new york": "New York Knicks", "knicks": "New York Knicks",
    "atlanta": "Atlanta Hawks", "hawks": "Atlanta Hawks",
    "orlando": "Orlando Magic", "magic": "Orlando Magic",
    "detroit": "Detroit Pistons", "pistons": "Detroit Pistons",
    "charlotte": "Charlotte Hornets", "hornets": "Charlotte Hornets",
    "philadelphia": "Philadelphia 76ers", "76ers": "Philadelphia 76ers", "sixers": "Philadelphia 76ers",
    
    "dallas": "Dallas Mavericks", "mavericks": "Dallas Mavericks",
    "minnesota": "Minnesota Timberwolves", "timberwolves": "Minnesota Timberwolves",
    "phoenix": "Phoenix Suns", "suns": "Phoenix Suns",
    "denver": "Denver Nuggets", "nuggets": "Denver Nuggets",
    "houston": "Houston Rockets", "rockets": "Houston Rockets",
    "oklahoma city": "Oklahoma City Thunder", "thunder": "Oklahoma City Thunder",
    "utah": "Utah Jazz", "jazz": "Utah Jazz",
    "portland": "Portland Trail Blazers", "trail blazers": "Portland Trail Blazers",
    "golden state": "Golden State Warriors", "warriors": "Golden State Warriors",
    "los angeles clippers": "Los Angeles Clippers", "clippers": "Los Angeles Clippers",
   "los angeles lakers": "Los Angeles Lakers", "lakers": "Los Angeles Lakers",
    "memphis": "Memphis Grizzlies", "grizzlies": "Memphis Grizzlies",
    "sacramento": "Sacramento Kings", "kings": "Sacramento Kings",
    "san antonio": "San Antonio Spurs", "spurs": "San Antonio Spurs",
    "new orleans": "New Orleans Pelicans", "pelicans": "New Orleans Pelicans",
}

# ==============================================================================
# ⚠️ DEPRECATION WARNING (v22.1+)
# ==============================================================================
# Esta lista estática está DEPRECIADA. Jogadores podem se lesionar ou cair de
# produção durante a temporada, gerando ruído nas features de impacto de estrelas.
#
# TODO (Sprint futura): Substituir por lógica dinâmica baseada em:
#   1. USG% (Usage Rate) > 25% nos últimos 15 jogos
#   2. PER (Player Efficiency Rating) > 20 na temporada
#   3. Minutos jogados > 28 MPG
#
# Isso garantirá que apenas jogadores ATUALMENTE em alta forma sejam considerados
# "estrelas" para fins de cálculo de impacto em lesões.
# ==============================================================================
ALL_STARS_2025 = [
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo", "Nikola Jokic",
    "Luka Doncic", "Jayson Tatum", "Joel Embiid", "Shai Gilgeous-Alexander", "Anthony Edwards",
    "Devin Booker", "Anthony Davis", "Kawhi Leonard", "Paul George", "Donovan Mitchell",
    "Damian Lillard", "Jimmy Butler", "Bam Adebayo", "Jaylen Brown", "Tyrese Haliburton",
    "Jalen Brunson", "Tyrese Maxey", "Paolo Banchero", "Victor Wembanyama", "Ja Morant",
    "Zion Williamson", "Kyrie Irving", "Trae Young", "De'Aaron Fox", "Domantas Sabonis"
]

# --- Line Shopping Config ---
import os
from pathlib import Path
from typing import Optional

# Carregar de .env usando python-dotenv se disponível
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_file, override=False)
except ImportError:
    # Fallback: carregamento manual
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        try:
            with open(env_file, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception:
            pass  # Silenciosamente falhar se não conseguir carregar

ODDS_API_KEY: str = os.getenv('ODDS_API_KEY', 'SUA_CHAVE_AQUI')
RAPIDAPI_KEY: str = os.getenv('RAPIDAPI_KEY', '')
TWITTER_BEARER_TOKEN: str = os.getenv('TWITTER_BEARER_TOKEN', '')

BOOKMAKERS = [
    'pinnacle',
    'bet365',
    'betano',
    'draftkings',
    'fanduel'
]

# --- Rating Algorithm Constants ---

RATING_WEIGHTS = {
    'rapm': 3.0,
    'lebron': 2.0,
    'bpm_min': 0.5,
    'bpm_max': 3.0,
    'bpm_divisor': 1000.0
}

NORMALIZATION_LIMITS = {
    'min': -10.0,
    'max': 10.0
}

REFEREE_ADJUSTMENTS = {
    'high_win_pct_threshold': 0.60,
    'low_win_pct_threshold': 0.54,
    'high_adjustment': 1.5,
    'low_adjustment': -1.0,
    'default_home_win_pct': 0.58
}

HCA_VALUE = 3.0
