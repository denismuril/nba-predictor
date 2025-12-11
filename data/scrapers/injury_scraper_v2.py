"""
Injury Scraper para NBA Predictor - v2.1 (Cache-First Strategy)

Implementa estratégia de Cache-First para reduzir scraping e risco de detecção.

Arquitetura:
    - CacheManager: Gerencia persistência local em JSON com TTL configurável
    - BaseScraper: Interface abstrata para scrapers (Strategy Pattern)
    - RotowireScraper: Scraper primário (rotowire.com)
    - ESPNScraper: Scraper secundário/fallback (ESPN)
    - PDFScraper: Scraper de PDF oficial da NBA
    - InjuryManager: Orquestra Cache -> Scraping -> Save

Usage:
    from data.scrapers.injury_scraper_v2 import InjuryManager
    
    manager = InjuryManager()
    injuries = manager.get_latest_injuries()  # Usa cache se válido
"""
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import re
import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from pathlib import Path
from threading import Lock

# --- Configurações do Sistema ---
CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "injuries.json"
CACHE_TTL_MINUTES = int(os.getenv("INJURY_CACHE_TTL_MINUTES", 30))

# --- Configuração de Logs ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("NBA_Injury_Monitor")

# --- Estrutura de Dados ---
@dataclass
class InjuryReport:
    """Estrutura de dados para relatório de lesão."""
    player_name: str
    team: str
    status: str
    description: str
    source: str
    updated_at: str
    
    def is_critical(self) -> bool:
        """Retorna True se o status indica lesão crítica (OUT/DOUBTFUL)."""
        return self.status.upper() in ('OUT', 'DOUBTFUL')


# --- Gerenciador de Cache (Thread-Safe) ---
class CacheManager:
    """
    Gerenciador de cache com persistência em JSON e TTL configurável.
    
    Thread-safe para uso em múltiplos processos.
    """
    _lock = Lock()
    
    @classmethod
    def load_cache(cls) -> Optional[List[InjuryReport]]:
        """
        Tenta carregar dados do disco se forem recentes.
        
        Returns:
            Lista de InjuryReport se cache válido, None caso contrário
        """
        with cls._lock:
            if not CACHE_FILE.exists():
                logger.info("Cache: Arquivo não encontrado. Necessário scraping.")
                return None

            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Verifica a idade do cache
                cached_time = datetime.fromisoformat(data['timestamp'])
                age = datetime.now() - cached_time
                
                if age > timedelta(minutes=CACHE_TTL_MINUTES):
                    logger.info(f"Cache: Expirado (Idade: {age}). Necessário atualizar.")
                    return None
                
                logger.info(f"Cache: VÁLIDO (Idade: {age}). Usando dados locais.")
                
                # Reconstrói os objetos InjuryReport
                reports = [InjuryReport(**item) for item in data['data']]
                return reports

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Cache: Erro ao ler arquivo ({e}). Ignorando cache corrompido.")
                return None
            except Exception as e:
                logger.error(f"Cache: Erro inesperado ({e}). Ignorando.")
                return None

    @classmethod
    def save_cache(cls, reports: List[InjuryReport]) -> bool:
        """
        Salva os dados no disco para uso futuro.
        
        Returns:
            True se salvou com sucesso, False caso contrário
        """
        with cls._lock:
            try:
                # Garantir que diretório existe
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'count': len(reports),
                    'ttl_minutes': CACHE_TTL_MINUTES,
                    'data': [asdict(r) for r in reports]
                }
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Cache: {len(reports)} lesões salvas em '{CACHE_FILE}'.")
                return True
            except Exception as e:
                logger.error(f"Cache: Erro ao salvar arquivo ({e}).")
                return False

    @classmethod
    def get_cache_age_hours(cls) -> Optional[float]:
        """Retorna a idade do cache em horas, ou None se não existir."""
        if not CACHE_FILE.exists():
            return None
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cached_time = datetime.fromisoformat(data['timestamp'])
            return (datetime.now() - cached_time).total_seconds() / 3600
        except Exception:
            return None


# --- Utils de Normalização ---
class DataCleaner:
    """Utilitários para limpeza e normalização de dados."""
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Remove sufixos como Jr., Sr., II, III, IV."""
        if not name:
            return ""
        clean = re.sub(r'\s+(Jr\.?|Sr\.?|I{2,3}|IV)\.?$', '', name, flags=re.IGNORECASE)
        clean = re.sub(r'[^\w\s\'-]', '', clean)  # Permite apóstrofos e hífens
        return clean.strip()

    @staticmethod
    def normalize_status(status_text: str) -> str:
        """Normaliza status de lesão para valores padrão."""
        s = status_text.lower()
        if 'out' in s:
            return 'OUT'
        if 'questionable' in s:
            return 'QUESTIONABLE'
        if 'doubtful' in s:
            return 'DOUBTFUL'
        if 'game time' in s or 'gtd' in s:
            return 'GTD'
        if 'probable' in s:
            return 'PROBABLE'
        if 'available' in s:
            return 'AVAILABLE'
        return 'UNKNOWN'


# --- Scrapers (Strategy Pattern) ---
class BaseScraper(ABC):
    """Classe base abstrata para todos os scrapers de lesões."""
    
    SOURCE_NAME = "Base"
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Faz request HTTP e retorna BeautifulSoup."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Erro de conexão com {url}: {e}")
            return None

    @abstractmethod
    def scrape(self) -> List[InjuryReport]:
        """Método abstrato para realizar o scraping."""
        pass


class RotowireScraper(BaseScraper):
    """Scraper para Rotowire NBA Lineups (fonte primária)."""
    
    SOURCE_NAME = "Rotowire"
    
    def scrape(self) -> List[InjuryReport]:
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        logger.info(f"Iniciando coleta primária: {url}")
        soup = self.get_soup(url)
        reports = []

        if not soup:
            raise Exception("Falha ao acessar Rotowire")

        lineup_boxes = soup.find_all('div', class_='lineup__box')
        for box in lineup_boxes:
            team_name = box.find('div', class_='lineup__team-name')
            if not team_name:
                continue
            team_code = team_name.text.strip()
            
            # Busca jogadores com a div de injury
            players = box.find_all('li', class_='lineup__player')
            for p in players:
                is_injured = p.find('div', class_='lineup__injuries')
                if is_injured:
                    name_tag = p.find('a')
                    if not name_tag:
                        continue
                    
                    status_raw = is_injured.text.strip()
                    report = InjuryReport(
                        player_name=DataCleaner.normalize_name(name_tag.text.strip()),
                        team=team_code,
                        status=DataCleaner.normalize_status(status_raw),
                        description=f"Status via Rotowire: {status_raw}",
                        source=self.SOURCE_NAME,
                        updated_at=datetime.now().isoformat()
                    )
                    reports.append(report)
        
        logger.info(f"Rotowire: {len(reports)} lesões encontradas")
        return reports


class ESPNScraper(BaseScraper):
    """Scraper para ESPN Injuries (fonte secundária/fallback)."""
    
    SOURCE_NAME = "ESPN"
    
    def scrape(self) -> List[InjuryReport]:
        url = "https://www.espn.com/nba/injuries"
        logger.info(f"Iniciando coleta secundária: {url}")
        soup = self.get_soup(url)
        reports = []
        
        if not soup:
            raise Exception("Falha ao acessar ESPN")

        rows = soup.find_all('tr', class_='Table__TR')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                name_link = cols[0].find('a')
                if name_link:
                    report = InjuryReport(
                        player_name=DataCleaner.normalize_name(name_link.text),
                        team="UNK",  # ESPN não mostra team code facilmente
                        status=DataCleaner.normalize_status(cols[1].text),
                        description=cols[2].text.strip() if len(cols) > 2 else "Unknown",
                        source=self.SOURCE_NAME,
                        updated_at=datetime.now().isoformat()
                    )
                    reports.append(report)
        
        logger.info(f"ESPN: {len(reports)} lesões encontradas")
        return reports


class PDFScraper(BaseScraper):
    """Scraper para PDF oficial da NBA (mais confiável)."""
    
    SOURCE_NAME = "NBA_Official"
    
    # Team name to code mapping
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
        """Usa scrape_injury_report_pdf() do injury_scraper.py original."""
        try:
            import sys
            data_dir = Path(__file__).parent
            if str(data_dir) not in sys.path:
                sys.path.insert(0, str(data_dir))
            
            from injury_scraper import scrape_injury_report_pdf
            
            pdf_data = scrape_injury_report_pdf()
            
            if not pdf_data:
                return []
            
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
            
            logger.info(f"PDF Oficial: {len(reports)} lesões encontradas")
            return reports
            
        except ImportError as e:
            logger.debug(f"injury_scraper.py não encontrado: {e}")
            return []
        except Exception as e:
            logger.warning(f"Erro no PDF scraper: {e}")
            return []


# --- Orquestrador Principal ---
class InjuryManager:
    """
    Orquestrador do sistema de lesões com estratégia Cache-First.
    
    Fluxo:
        1. Verifica cache local (se válido, retorna imediatamente)
        2. Se cache inválido/expirado, tenta scrapers em ordem
        3. Salva resultado em cache para próxima consulta
    """
    
    def __init__(self):
        self.scrapers = [
            PDFScraper(),      # Prioridade 1: PDF oficial
            RotowireScraper(), # Prioridade 2: Rotowire
            ESPNScraper(),     # Prioridade 3: ESPN (fallback)
        ]
        self._previous_injuries: Dict[str, InjuryReport] = {}

    def get_latest_injuries(self, force_refresh: bool = False) -> List[InjuryReport]:
        """
        Obtém lista de lesões mais recentes.
        
        Args:
            force_refresh: Se True, ignora cache e força scraping
            
        Returns:
            Lista de InjuryReport
        """
        # 1. Tenta carregar do Cache primeiro (se não forçar refresh)
        if not force_refresh:
            cached_data = CacheManager.load_cache()
            if cached_data:
                return cached_data
        
        # 2. Se não tem cache válido, vai para a Internet
        logger.info("Iniciando coleta na Web...")
        live_data = []
        
        for scraper in self.scrapers:
            try:
                live_data = scraper.scrape()
                if live_data:
                    logger.info(f"✅ Dados obtidos via {scraper.SOURCE_NAME}")
                    break
            except Exception as e:
                logger.warning(f"⚠️ {scraper.SOURCE_NAME} falhou: {e}")
                continue
        
        if not live_data:
            logger.critical("🚨 TODAS AS FONTES FALHARAM!")
            
            # DEAD MAN'S SWITCH: Verificar idade do cache
            cache_age = CacheManager.get_cache_age_hours()
            if cache_age and cache_age > 12:
                logger.error(
                    f"🚨 DEAD MAN'S SWITCH: Cache com {cache_age:.1f}h de idade! "
                    "Verificar fontes de dados."
                )
            
            # Retornar cache antigo como fallback
            cached = CacheManager.load_cache()
            return cached if cached else []
        
        # 3. Salva no Cache para a próxima vez
        CacheManager.save_cache(live_data)
        
        return live_data

    def get_new_critical_injuries(self) -> List[InjuryReport]:
        """
        Retorna lesões críticas (OUT/DOUBTFUL) que são NOVAS desde a última verificação.
        
        Útil para alertas de Telegram.
        """
        current = self.get_latest_injuries()
        new_critical = []
        
        for injury in current:
            if not injury.is_critical():
                continue
                
            key = f"{injury.player_name}_{injury.team}"
            
            # Verifica se é novo ou mudou para pior
            if key not in self._previous_injuries:
                new_critical.append(injury)
            elif self._previous_injuries[key].status != injury.status:
                # Status mudou (ex: QUESTIONABLE -> OUT)
                new_critical.append(injury)
        
        # Atualiza cache interno
        self._previous_injuries = {
            f"{inj.player_name}_{inj.team}": inj for inj in current
        }
        
        return new_critical

    def calculate_team_injury_impact(
        self,
        team_code: str,
        injuries: List[InjuryReport] = None,
        player_values: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calcula impacto total de injuries para um time.
        
        Args:
            team_code: Código do time (3 letras)
            injuries: Lista de injuries (se None, busca automaticamente)
            player_values: Dict opcional com {player_name: value_score}
        
        Returns:
            Impact score (-0.5 a 0, negativo = prejudicado)
        """
        if injuries is None:
            injuries = self.get_latest_injuries()
        
        if player_values is None:
            player_values = get_player_importance_scores()
        
        team_injuries = [inj for inj in injuries if inj.team == team_code]
        
        if not team_injuries:
            return 0.0
        
        total_impact = 0.0
        
        status_weights = {
            'OUT': 1.0,
            'DOUBTFUL': 0.7,
            'QUESTIONABLE': 0.3,
            'GTD': 0.4,
            'PROBABLE': 0.1,
            'AVAILABLE': 0.0,
            'UNKNOWN': 0.2,
        }
        
        for injury in team_injuries:
            player_value = player_values.get(injury.player_name, 0.1)
            status_weight = status_weights.get(injury.status, 0.3)
            impact = -(player_value * status_weight)
            total_impact += impact
        
        return max(total_impact, -0.5)  # Cap em -50%


# --- Compatibilidade com código legado ---
class InjuryScraper(InjuryManager):
    """
    Alias para InjuryManager para manter compatibilidade com código existente.
    
    DEPRECATED: Use InjuryManager diretamente.
    """
    
    def __init__(self, cache_file: str = None):
        super().__init__()
        if cache_file:
            logger.warning(
                "InjuryScraper(cache_file=...) está DEPRECATED. "
                "Use InjuryManager() e configure via env var INJURY_CACHE_TTL_MINUTES."
            )
    
    def get_current_injuries(self, use_cache: bool = True) -> List[Dict]:
        """
        Método legado para compatibilidade.
        
        DEPRECATED: Use get_latest_injuries() que retorna List[InjuryReport].
        """
        injuries = self.get_latest_injuries(force_refresh=not use_cache)
        
        # Converter para formato dict legado
        return [asdict(inj) for inj in injuries]


def get_player_importance_scores() -> Dict[str, float]:
    """
    Retorna scores de importância para key players.
    
    Baseado em aproximação de Win Shares / 48.
    Top stars: 0.2-0.3
    All-stars: 0.15-0.20
    Starters: 0.08-0.12
    Role players: 0.03-0.07
    """
    return {
        # MVP candidates (2024-25)
        'Nikola Jokic': 0.30,
        'Luka Doncic': 0.28,
        'Giannis Antetokounmpo': 0.27,
        'Joel Embiid': 0.26,
        'Kevin Durant': 0.24,
        'Stephen Curry': 0.24,
        'LeBron James': 0.22,
        'Shai Gilgeous-Alexander': 0.25,
        'Jayson Tatum': 0.22,
        
        # All-Stars
        'Anthony Davis': 0.20,
        'Damian Lillard': 0.18,
        'Jimmy Butler': 0.17,
        'Devin Booker': 0.16,
        'Donovan Mitchell': 0.16,
        'Tyrese Haliburton': 0.15,
        'Paolo Banchero': 0.15,
        'Ja Morant': 0.18,
        'Trae Young': 0.16,
        'De\'Aaron Fox': 0.15,
        
        # Key starters
        'Karl-Anthony Towns': 0.14,
        'Kawhi Leonard': 0.18,
        'Paul George': 0.15,
        'Zion Williamson': 0.16,
        'Chet Holmgren': 0.14,
        'Victor Wembanyama': 0.17,
    }


if __name__ == '__main__':
    # Demo
    print("🏥 Demo: Injury Manager v2.1 (Cache-First)\n")
    
    manager = InjuryManager()
    
    # Primeira execução
    print("--- Chamada 1 (primeira execução) ---")
    injuries = manager.get_latest_injuries()
    print(f"📋 {len(injuries)} lesões encontradas\n")
    
    if injuries:
        for injury in injuries[:5]:
            status_icon = "🔴" if injury.is_critical() else "🟡"
            print(f"  {status_icon} {injury.player_name} ({injury.team})")
            print(f"      Status: {injury.status}")
            print(f"      Fonte: {injury.source}\n")
    
    # Segunda execução (deve usar cache)
    print("--- Chamada 2 (teste de cache) ---")
    injuries_cached = manager.get_latest_injuries()
    print(f"📋 {len(injuries_cached)} lesões (do cache)\n")
    
    # Demo impacto
    print("--- Cálculo de Impacto ---")
    for team in ['LAL', 'BOS', 'DEN']:
        impact = manager.calculate_team_injury_impact(team)
        print(f"  {team}: {impact:.3f}")
    
    print("\n✅ Demo completo!")
