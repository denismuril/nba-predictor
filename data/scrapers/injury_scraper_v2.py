"""
Injury Scraper para NBA Predictor

Obtém dados de injury reports de múltiplas fontes.

Data Sources (em ordem de prioridade):
1. Reddit Community API (RapidAPI) - Real-time, historical desde 2021
2. GitHub mxufc29/nbainjuries - Package Python
3. Fallback: Manual scraping de NBA official

Usage:
    from data.scrapers.injury_scraper_v2 import InjuryScraper
    
    scraper = InjuryScraper()
    injuries = scraper.get_current_injuries()
"""
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class InjuryScraper:
    """Scraper para injury reports da NBA."""
    
    def __init__(self, cache_file: str = 'data/injury_cache.json'):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # Team name normalization
        self.team_mapping = {
            'LAL': 'Lakers', 'LAC': 'Clippers', 'GSW': 'Warriors',
            'BOS': 'Celtics', 'MIA': 'Heat', 'PHX': 'Suns',
            'MIL': 'Bucks', 'DEN': 'Nuggets', 'DAL': 'Mavericks',
            # ... adicionar resto do mapping conforme necessário
        }
    
    def get_current_injuries(self, use_cache: bool = True) -> List[Dict]:
        """
        Obtém lista de lesões atuais.
        
        Prioridade:
        1. PDF oficial da NBA (funciona bem!)
        2. RapidAPI (se configurado)
        3. GitHub package
        4. Cache/fallback
        
        Returns:
            Lista de dicts com injury information
        """
        # Check cache first
        if use_cache and self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                
                # Cache válido por 3 horas
                if (datetime.now() - cache_time).seconds < 3 * 3600:
                    logger.info("📋 Usando injury cache")
                    return cache_data['injuries']
        
        injuries = None
        
        # PRIORITY 1: PDF Oficial (funciona bem!)
        try:
            injuries = self._fetch_from_pdf()
            if injuries:
                logger.info(f"✅ {len(injuries)} injuries obtidos via PDF oficial")
        except Exception as e:
            logger.warning(f"⚠️ PDF scraping failed: {e}")
        
        # PRIORITY 2: RapidAPI (se configurado)
        if not injuries:
            try:
                injuries = self._fetch_from_rapidapi()
                if injuries:
                    logger.info("✅ Injuries obtidos via RapidAPI")
            except Exception as e:
                logger.debug(f"RapidAPI não disponível: {e}")
        
        # PRIORITY 3: GitHub Package
        if not injuries:
            try:
                injuries = self._fetch_from_github_package()
                if injuries:
                    logger.info("✅ Injuries obtidos via GitHub package")
            except Exception as e:
                logger.debug(f"GitHub package não disponível: {e}")
        
        # Fallback: Empty list
        if not injuries:
            logger.warning("⚠️ Nenhuma fonte disponível, usando lista vazia")
            injuries = []
            
            # AUDIT FIX: DEAD MAN'S SWITCH
            # Verificar se cache existe e está muito antigo (>12h)
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data_check = json.load(f)
                    cache_time = datetime.fromisoformat(cache_data_check['timestamp'])
                    cache_age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                    
                    if cache_age_hours > 12:
                        logger.error(
                            f"🚨 DEAD MAN'S SWITCH TRIGGERED: Injury data obsoleto! "
                            f"Cache age: {cache_age_hours:.1f}h (threshold: 12h). "
                            f"Todas as fontes falharam (PDF, API, GitHub). "
                            f"AÇÃO NECESSÁRIA: Verificar conectividade e fontes de dados."
                        )
                        # Retornar flag especial que o predict.py pode detetar
                        return [{
                            '_data_age_warning': True,
                            '_cache_age_hours': cache_age_hours,
                            '_status': 'STALE_DATA'
                        }]
        
        # Cache result
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'injuries': injuries
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return injuries
    
    def _fetch_from_pdf(self) -> Optional[List[Dict]]:
        """
        Fetch via PDF oficial da NBA (método que já funciona!).
        
        Usa scrape_injury_report_pdf() do injury_scraper.py existente.
        """
        try:
            # Import scraper original
            import sys
            from pathlib import Path
            
            # Add data dir to path
            data_dir = Path(__file__).parent
            if str(data_dir) not in sys.path:
                sys.path.insert(0, str(data_dir))
            
            # Import função do scraper existente
            from injury_scraper import scrape_injury_report_pdf
            
            # Get data do PDF
            pdf_data = scrape_injury_report_pdf()
            
            if not pdf_data:
                return None
            
            # Converter formato para nosso padrão
            # pdf_data format: {'TeamName': {'PlayerName': 'STATUS'}}
            injuries = []
            
            for team_name, players in pdf_data.items():
                # Normalizar team name para código
                team_code = self._normalize_team_name(team_name)
                
                for player_name, status in players.items():
                    injuries.append({
                        'player_name': player_name,
                        'team': team_code,
                        'status': status,
                        'description': 'Injury',  # PDF não tem detalhes
                        'date_updated': datetime.now().strftime('%Y-%m-%d')
                    })
            
            return injuries if len(injuries) > 0 else None
            
        except ImportError as e:
            logger.debug(f"injury_scraper.py não encontrado: {e}")
            return None
        except Exception as e:
            logger.warning(f"Erro no PDF scraper: {e}")
            return None
    
    def _normalize_team_name(self, team_full_name: str) -> str:
        """
        Converte nome completo do time para código de 3 letras.
        
        Args:
            team_full_name: 'Los Angeles Lakers', etc
        
        Returns:
            Código 3 letras: 'LAL', etc
        """
        # Mapping básico
        team_map = {
            'Los Angeles Lakers': 'LAL',
            'Los Angeles Clippers': 'LAC',
            'Golden State Warriors': 'GSW',
            'Boston Celtics': 'BOS',
            'Miami Heat': 'MIA',
            'Phoenix Suns': 'PHX',
            'Milwaukee Bucks': 'MIL',
            'Denver Nuggets': 'DEN',
            'Dallas Mavericks': 'DAL',
            'Philadelphia 76ers': 'PHI',
            'Cleveland Cavaliers': 'CLE',
            'New York Knicks': 'NYK',
            'Brooklyn Nets': 'BKN',
            'Chicago Bulls': 'CHI',
            'Atlanta Hawks': 'ATL',
            'Toronto Raptors': 'TOR',
            'Detroit Pistons': 'DET',
            'Indiana Pacers': 'IND',
            'Charlotte Hornets': 'CHA',
            'Washington Wizards': 'WAS',
            'Orlando Magic': 'ORL',
            'Minnesota Timberwolves': 'MIN',
            'Oklahoma City Thunder': 'OKC',
            'Portland Trail Blazers': 'POR',
            'Utah Jazz': 'UTA',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Memphis Grizzlies': 'MEM',
            'New Orleans Pelicans': 'NOP',
            'Houston Rockets': 'HOU',
        }
        
        return team_map.get(team_full_name, 'UNK')
    
    def _fetch_from_rapidapi(self) -> Optional[List[Dict]]:
        """Fetch via Reddit Community API (RapidAPI)."""
        # TODO: Implementar quando user fornecer API key
        logger.debug("RapidAPI não configurado (requer API key)")
        return None
    
    def _fetch_from_github_package(self) -> Optional[List[Dict]]:
        """Fetch via mxufc29/nbainjuries package."""
        try:
            import nbainjuries
            df = nbainjuries.get_injuries()
            
            injuries = []
            for _, row in df.iterrows():
                injuries.append({
                    'player_name': row.get('player', 'Unknown'),
                    'team': row.get('team', 'UNK'),
                    'status': row.get('status', 'QUESTIONABLE'),
                    'description': row.get('injury', 'Unknown injury'),
                    'date_updated': row.get('date', datetime.now().strftime('%Y-%m-%d'))
                })
            
            return injuries if len(injuries) > 0 else None
            
        except ImportError:
            return None
        except Exception as e:
            logger.error(f"Erro no GitHub package: {e}")
            return None
    
    def calculate_team_injury_impact(
        self,
        team_code: str,
        injuries: List[Dict],
        player_values: Optional[Dict] = None
    ) -> float:
        """
        Calcula impacto total de injuries para um time.
        
        Args:
            team_code: Código do time (3 letras)
            injuries: Lista de injuries
            player_values: Dict opcional com {player_name: value_score}
        
        Returns:
            Impact score (-0.5 a 0, negativo = prejudicado)
        """
        team_injuries = [inj for inj in injuries if inj.get('team') == team_code]
        
        if not team_injuries:
            return 0.0
        
        total_impact = 0.0
        
        for injury in team_injuries:
            status = injury.get('status', '').upper()
            player = injury.get('player_name', '')
            
            # Player value
            if player_values and player in player_values:
                player_value = player_values[player]
            else:
                player_value = 0.1  # Default: jogador médio
            
            # Status weights
            status_weight = {
                'OUT': 1.0,
                'DOUBTFUL': 0.7,
                'QUESTIONABLE': 0.3,
                'AVAILABLE': 0.0,
                'PROBABLE': 0.1
            }.get(status, 0.3)
            
            impact = -(player_value * status_weight)
            total_impact += impact
        
        return max(total_impact, -0.5)  # Cap em -50%


def get_player_importance_scores() -> Dict[str, float]:
    """
    Retorna scores de importância para key players.
    
    Baseado em aproximação de Win Shares / 48.
    Top stars: 0.2-0.3
    All-stars: 0.15-0.20
    Starters: 0.08-0.12
    Role players: 0.03-0.07
    """
    # TODO: Carregar de database ou API
    # Por agora, hardcode alguns exemplos
    
    return {
        # MVP candidates (2024-25)
        'Nikola Jokic': 0.30,
        'Luka Doncic': 0.28,
        'Giannis Antetokounmpo': 0.27,
        'Joel Embiid': 0.26,
        'Kevin Durant': 0.24,
        'Stephen Curry': 0.24,
        'LeBron James': 0.22,
        
        # All-Stars
        'Anthony Davis': 0.20,
        'Jayson Tatum': 0.19,
        'Damian Lillard': 0.18,
        'Jimmy Butler': 0.17,
        'Devin Booker': 0.16,
        
        # Default for unlisted players
        # (será tratado como 0.1 no código)
    }


if __name__ == '__main__':
    # Demo
    print("🏥 Demo: Injury Scraper\n")
    
    scraper = InjuryScraper()
    
    # Get current injuries
    injuries = scraper.get_current_injuries()
    
    print(f"📋 {len(injuries)} injuries encontrados\n")
    
    if injuries:
        # Mostrar primeiros 5
        for injury in injuries[:5]:
            print(f"  {injury['player_name']} ({injury['team']})")
            print(f"    Status: {injury['status']}")
            print(f"    Injury: {injury['description']}\n")
    
    # Demo impact calculation
    print("📊 Calculando impact para times...\n")
    
    # Simular injuries
    demo_injuries = [
        {'player_name': 'LeBron James', 'team': 'LAL', 'status': 'OUT', 'description': 'ankle', 'date_updated': '2024-11-29'},
        {'player_name': 'Anthony Davis', 'team': 'LAL', 'status': 'QUESTIONABLE', 'description': 'back', 'date_updated': '2024-11-29'},
    ]
    
    player_scores = get_player_importance_scores()
    
    lal_impact = scraper.calculate_team_injury_impact('LAL', demo_injuries, player_scores)
    print(f"Lakers injury impact: {lal_impact:.3f}")
    print(f"  (LeBron OUT + AD QUESTIONABLE)")
    
    print("\n✅ Demo completo!")
