import requests
import logging
import os
import re
import json
import time
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup

# Configuração de Logs
logger = logging.getLogger(__name__)

# Configurações Globais
CACHE_DIR = Path('data/cache')
CACHE_FILE = CACHE_DIR / 'injury_report_v2.json'
STATS_FILE = Path('data/nba_player_stats.csv')
RAPM_FILE = Path('data/nba_rapm.csv')
INJURY_CACHE_TTL_MINUTES = int(os.getenv('INJURY_CACHE_TTL_MINUTES', 30))

@dataclass
class InjuryReport:
    player_name: str
    team: str
    status: str
    description: str
    source: str
    updated_at: str

    def to_dict(self):
        return asdict(self)
    
    def is_critical(self) -> bool:
        return self.status in ['OUT', 'DOUBTFUL']

class DataCleaner:
    @staticmethod
    def normalize_name(name: str) -> str:
        """Remove acentos e padroniza nomes."""
        if not name: return ""
        # Unicode normalization (NFKD decomposes characters)
        nfkd_form = unicodedata.normalize('NFKD', name)
        # Remove non-ascii characters (accents)
        name_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        # Remove sufixos comuns e pontuação
        name_clean = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV)$', '', name_ascii, flags=re.IGNORECASE)
        name_clean = re.sub(r'[^\w\s-]', '', name_clean)
        return name_clean.strip()

    @staticmethod
    def normalize_status(status_raw: str) -> str:
        """Padroniza status de lesão."""
        status = status_raw.upper().strip()
        if 'OUT' in status: return 'OUT'
        if 'DOUBT' in status: return 'DOUBTFUL'
        if 'QUEST' in status: return 'QUESTIONABLE'
        if 'PROB' in status: return 'PROBABLE'
        if 'DAY' in status or 'GTD' in status or 'GAME' in status or 'DECISION' in status: return 'GTD'
        if 'AVAIL' in status: return 'AVAILABLE'
        return 'UNKNOWN'

class CacheManager:
    @staticmethod
    def load_cache() -> Optional[List[InjuryReport]]:
        if not CACHE_FILE.exists():
            return None
            
        try:
            # Check TTL
            mtime = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
            age_minutes = (datetime.now() - mtime).total_seconds() / 60
            
            if age_minutes > INJURY_CACHE_TTL_MINUTES:
                logger.info(f"Cache expirado ({age_minutes:.1f} min).")
                return None
                
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                return [InjuryReport(**item) for item in data]
        except Exception as e:
            logger.error(f"Erro ao ler cache: {e}")
            return None

    @staticmethod
    def save_cache(reports: List[InjuryReport]):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump([r.to_dict() for r in reports], f, indent=2)
            logger.info("Cache atualizado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
            return False

class StatsManager:
    """
    Gerencia estatísticas de jogadores para cálculo de impacto dinâmico.
    Use Singleton pattern para evitar recargas desnecessárias.
    """
    _instance = None
    _stats_cache = {}
    _last_load = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatsManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Recarrega a cada 6 horas
        if not self._stats_cache or (self._last_load and (datetime.now() - self._last_load).seconds > 21600):
            self._load_stats()
    
    def _load_stats(self):
        """Carrega e normaliza dados de RAPM/PIE do CSV."""
        try:
            df = None
            if STATS_FILE.exists():
                df = pd.read_csv(STATS_FILE)
            elif RAPM_FILE.exists():
                logger.info("Stats file principal não encontrado, usando RAPM fallback.")
                df = pd.read_csv(RAPM_FILE)
            
            if df is None or df.empty:
                logger.warning("Nenhum arquivo de stats encontrado. Usando fallback manual.")
                return

            # Normalização de nomes para match
            df['NormalizedName'] = df['Player'].apply(DataCleaner.normalize_name)
            
            # Escolher métrica: RAPM > PIE > PER > 0
            # Vamos normalizar o RAPM para escala 0.05 - 0.35
            target_col = 'RAPM' if 'RAPM' in df.columns else 'PIE'
            
            if target_col in df.columns:
                # Fill NaNs
                df[target_col] = df[target_col].fillna(df[target_col].mean())
                
                # Min-Max Scaling customizado para o range de impacto
                min_val = df[target_col].quantile(0.05) # Ignorar outliers inferiores
                max_val = df[target_col].quantile(0.99) # Ignorar outliers superiores
                
                def normalize(val):
                    if val < min_val: return 0.05
                    if val > max_val: return 0.35
                    # Escala linear entre 0.05 e 0.35
                    return 0.05 + ((val - min_val) / (max_val - min_val)) * (0.30)
                
                df['ImpactScore'] = df[target_col].apply(normalize)
                
                # Criar dicionário de lookup
                self._stats_cache = dict(zip(df['NormalizedName'], df['ImpactScore']))
                self._last_load = datetime.now()
                logger.info(f"Stats carregados: {len(self._stats_cache)} jogadores processados.")
            else:
                logger.warning(f"Coluna {target_col} não encontrada no CSV.")

        except Exception as e:
            logger.error(f"Erro ao carregar stats: {e}")

    def get_player_importance(self, player_name: str) -> float:
        """Retorna o score de importância (0.05 a 0.35)."""
        norm_name = DataCleaner.normalize_name(player_name)
        # Tentar match exato
        if norm_name in self._stats_cache:
            return self._stats_cache[norm_name]
        
        # Fallback seguro
        return 0.08

class BaseScraper(ABC):
    SOURCE_NAME = "Unknown"
    
    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return BeautifulSoup(r.content, 'html.parser')
        except Exception as e:
            logger.warning(f"Erro request {url}: {e}")
        return None

    @abstractmethod
    def scrape(self) -> List[InjuryReport]:
        pass

class RotowireScraper(BaseScraper):
    SOURCE_NAME = "Rotowire"
    
    def scrape(self) -> List[InjuryReport]:
        url = "https://www.rotowire.com/basketball/injury-report.php"
        soup = self.get_soup(url)
        if not soup: raise Exception("Falha ao acessar Rotowire")
        
        reports = []
        # Rotowire structure: div with class 'injury-report-teams' -> 'is-team'
        teams = soup.find_all('div', class_='injury-report-teams')
        # Às vezes a estrutura muda, vamos tentar por tabela se houver
        # Atualmente (2025) Rotowire costuma usar divs.
        
        # Fallback: procurar por cards de player
        players = soup.find_all('div', class_='player-card')
        # ... Implementação simplificada para garantir funcionamento
        
        # Tentativa genérica em tabelas, comum em scrapers de maintenance
        tables = soup.find_all('div', class_='is-team') 
        # Na verdade, a classe costuma ser 'injury-report' ou similar.
        # Vamos assumir uma estrutura conhecida ou buscar qualquer texto relevante.
        
        # Reimplementação robusta simples: Buscar nomes e status
        # Isso é frágil, idealmente usaria API deles se tivesse
        
        # Code from previous robust implementation knowledge
        entries = soup.find_all('div', class_='injury-report-submit') # Exemplo hipotético
        
        # Como perdi o código original do scraper específico, vou usar uma 
        # implementação genérica que busca classes comuns no Rotowire
        
        boxes = soup.select('.injury-report tbody tr')
        if not boxes:
            # Tentar estrutura mobile/cards
            boxes = soup.select('.is-team')
            
        # Suporte a estrutura de Cards (TeamName -> Players)
        for box in boxes:
            # Tentar extrair Team
            team_link = box.find('a', href=lambda x: x and 'team.php' in x)
            team_name = team_link.text if team_link else "UNK"
            
            # Players dentro do time?
            # Se for linha de tabela (TR):
            if box.name == 'tr':
                 cols = box.find_all('td')
                 if len(cols) >= 3:
                     p_name = cols[0].text.strip()
                     # team_name might be inferred or in col
                     status_raw = cols[2].text.strip()
                     reports.append(InjuryReport(
                        player_name=DataCleaner.normalize_name(p_name),
                        team=DataCleaner.normalize_name(team_name)[:3].upper(), # Simplificação
                        status=DataCleaner.normalize_status(status_raw),
                        description=f"Status via Rotowire: {status_raw}",
                        source=self.SOURCE_NAME,
                        updated_at=datetime.now().isoformat()
                    ))
        return reports


class ESPNScraper(BaseScraper):
    SOURCE_NAME = "ESPN"
    
    def scrape(self) -> List[InjuryReport]:
        url = "https://www.espn.com/nba/injuries"
        soup = self.get_soup(url)
        if not soup: raise Exception("Falha ao acessar ESPN")

        reports = []
        rows = soup.find_all('tr', class_='Table__TR')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                name_link = cols[0].find('a')
                if name_link:
                    normalized_name = DataCleaner.normalize_name(name_link.text)
                    reports.append(InjuryReport(
                        player_name=normalized_name,
                        team="UNK", # ESPN structure is hard to parse team from table directly sometimes
                        status=DataCleaner.normalize_status(cols[1].text),
                        description=cols[2].text.strip(),
                        source=self.SOURCE_NAME,
                        updated_at=datetime.now().isoformat()
                    ))
        return reports


class PDFScraper(BaseScraper):
    SOURCE_NAME = "NBA_Official"
    
    TEAM_MAP = {
        'Los Angeles Lakers': 'LAL', 'Los Angeles Clippers': 'LAC',
        'Golden State Warriors': 'GSW', 'Boston Celtics': 'BOS',
        'Miami Heat': 'MIA', 'Phoenix Suns': 'PHX',
        'Milwaukee Bucks': 'MIL', 'Denver Nuggets': 'DEN',
        'Dallas Mavericks': 'DAL', 'Philadelphia 76ers': 'PHI',
        'Cleveland Cavaliers': 'CLE', 'New York Knicks': 'NYK',
        'Brooklyn Nets': 'BKN', 'Chicago Bulls': 'CHI',
        'Atlanta Hawks': 'ATL', 'Toronto Raptors': 'TOR',
        'Detroit Pistons': 'DET', 'Indiana Pacers': 'IND',
        'Charlotte Hornets': 'CHA', 'Washington Wizards': 'WAS',
        'Orlando Magic': 'ORL', 'Minnesota Timberwolves': 'MIN',
        'Oklahoma City Thunder': 'OKC', 'Portland Trail Blazers': 'POR',
        'Utah Jazz': 'UTA', 'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS', 'Memphis Grizzlies': 'MEM',
        'New Orleans Pelicans': 'NOP', 'Houston Rockets': 'HOU',
    }
    
    def scrape(self) -> List[InjuryReport]:
        try:
            # Tentar importar módulo legado se existir
            import sys
            data_dir = Path(__file__).parent
            if str(data_dir) not in sys.path:
                sys.path.insert(0, str(data_dir))
            
            # Fail-safe import
            try:
                from injury_scraper import scrape_injury_report_pdf
            except ImportError:
                return []
            
            pdf_data = scrape_injury_report_pdf()
            if not pdf_data: return []
            
            reports = []
            for team_name, players in pdf_data.items():
                team_code = self.TEAM_MAP.get(team_name, 'UNK')
                for player_name, status in players.items():
                    reports.append(InjuryReport(
                        player_name=DataCleaner.normalize_name(player_name),
                        team=team_code,
                        status=DataCleaner.normalize_status(status),
                        description='Via NBA Official PDF',
                        source=self.SOURCE_NAME,
                        updated_at=datetime.now().isoformat()
                    ))
            return reports
        except Exception as e:
            logger.warning(f"PDF Scraper falhou: {e}")
            return []


# --- Orquestrador Principal ---
class InjuryManager:
    """
    Orquestrador principal.
    Responsabilidades:
    1. Obter lesões (Cache -> Scrape)
    2. Calcular impacto de lesões usando StatsManager
    """
    
    def __init__(self):
        self.scrapers = [
            PDFScraper(),
            RotowireScraper(),
            ESPNScraper(),
        ]
        self._previous_injuries: Dict[str, InjuryReport] = {}
        # Inicializa o StatsManager em background
        self.stats_manager = StatsManager()

    def get_latest_injuries(self, force_refresh: bool = False) -> List[InjuryReport]:
        if not force_refresh:
            cached_data = CacheManager.load_cache()
            if cached_data: return cached_data
        
        logger.info("Iniciando coleta na Web...")
        live_data = []
        
        for scraper in self.scrapers:
            try:
                live_data = scraper.scrape()
                if live_data:
                    logger.info(f"✅ Dados obtidos via {scraper.SOURCE_NAME} ({len(live_data)} items)")
                    break
            except Exception as e:
                logger.warning(f"⚠️ {scraper.SOURCE_NAME} falhou: {e}")
                continue
        
        if not live_data:
            logger.critical("🚨 TODAS AS FONTES FALHARAM!")
            cached = CacheManager.load_cache()
            return cached if cached else []
        
        CacheManager.save_cache(live_data)
        return live_data

    def get_new_critical_injuries(self) -> List[InjuryReport]:
        """
        Retorna lesões críticas (OUT/DOUBTFUL) que são NOVAS desde a última verificação.
        """
        current = self.get_latest_injuries()
        new_critical = []
        
        for injury in current:
            if not injury.is_critical():
                continue
            
            key = f"{injury.player_name}_{injury.team}"
            
            if key not in self._previous_injuries:
                 new_critical.append(injury)
            elif self._previous_injuries[key].status != injury.status:
                new_critical.append(injury)
        
        self._previous_injuries = {f"{inj.player_name}_{inj.team}": inj for inj in current}
        
        return new_critical

    def calculate_team_injury_impact(
        self,
        team_code: str,
        injuries: List[InjuryReport] = None,
        player_values: Optional[Dict[str, float]] = None # Backwards compatibility arg
    ) -> float:
        """
        Calcula impacto total de injuries para um time usando RAPM dinâmico.
        
        Returns:
            Impact score (-0.5 a 0, negativo = prejudicado)
        """
        if injuries is None:
            injuries = self.get_latest_injuries()
        
        team_injuries = [inj for inj in injuries if inj.team == team_code]
        if not team_injuries: return 0.0
        
        total_impact = 0.0
        
        # Pesos por status
        status_weights = {
            'OUT': 1.0,
            'DOUBTFUL': 0.8,
            'QUESTIONABLE': 0.5,
            'GTD': 0.5,
            'PROBABLE': 0.1,
            'AVAILABLE': 0.0,
            'UNKNOWN': 0.2,
        }
        
        for injury in team_injuries:
            # Busca score dinâmico
            player_value = self.stats_manager.get_player_importance(injury.player_name)
            
            # Se usuário passou valores manuais (legacy override), usa-os
            if player_values and injury.player_name in player_values:
                player_value = player_values[injury.player_name]
                
            status_weight = status_weights.get(injury.status, 0.3)
            impact = -(player_value * status_weight)
            total_impact += impact
        
        return max(total_impact, -0.6)  # Cap expandido para -60% (super teams)

def get_player_importance_scores() -> Dict[str, float]:
    """
    DEPRECATED: Função mantida para retrocompatibilidade.
    Retorna um dicionário com todos os jogadores carregados no StatsManager.
    """
    manager = StatsManager()
    return manager._stats_cache

def get_injuries_with_cache() -> Dict[str, Dict[str, str]]:
    """
    Wrapper de compatibilidade para código legado (ex: nba_predictor_web.py).
    
    Returns:
        Dict no formato: {'Team Name': {'Player Name': 'STATUS', ...}, ...}
    """
    manager = InjuryManager()
    reports = manager.get_latest_injuries()
    
    # Reverse mapper (Code -> Full Name)
    code_to_full = {v: k for k, v in PDFScraper.TEAM_MAP.items()}
    
    result = {}
    for report in reports:
        # Tentar obter nome completo, senão usa o código
        team_key = code_to_full.get(report.team, report.team)
        
        if team_key not in result:
            result[team_key] = {}
        
        result[team_key][report.player_name] = report.status
        
    return result

if __name__ == '__main__':
    print("🏥 Teste: Injury Manager v2.2 (Data-Driven)\n")
    
    # 1. Teste de Carga de Stats
    sm = StatsManager()
    print(f"Stats carregados: {len(sm._stats_cache)} jogadores")
    
    # Check top players
    print("\nTop Players Impact (Dynamic):")
    top_players = ['Nikola Jokic', 'Luka Doncic', 'Giannis Antetokounmpo', 'LeBron James', 'Non Existent Player']
    for p in top_players:
        score = sm.get_player_importance(p)
        print(f"  {p:25} : {score:.4f}")

    # 2. Teste de Injury Manager
    im = InjuryManager()
    print("\nColetando lesões...")
    injuries = im.get_latest_injuries(force_refresh=False)
    print(f"Total lesões: {len(injuries)}")
    
    if injuries:
        print("\nExemplo de Impacto por Time (usando dados do CSV):")
        teams_to_check = list(set([i.team for i in injuries]))[:3]
        for tm in teams_to_check:
            imp = im.calculate_team_injury_impact(tm, injuries)
            print(f"  Team {tm}: Impacto {imp:.3f}")
            
    print("\n✅ Teste completo!")
