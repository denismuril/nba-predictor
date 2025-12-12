"""
Historical NBA Injury Scraper - v1.0

Coleta dados retroativos de lesões da NBA para treinar o modelo.

Fontes (em ordem de prioridade):
1. PDFs Oficiais da NBA (archive via Wayback Machine ou CDN)
2. nbainjuries Python package (desde 2021-22)
3. API-BASKETBALL via RapidAPI (free tier)
4. ESPN API histórica (se disponível)

Uso:
    python scripts/scrape_historical_injuries.py --start 2023-10-01 --end 2024-04-15
"""
import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import io

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import TEAM_ABBREV_MAP

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "injuries_historical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SOURCE 1: NBA Official PDF (Historical Archive)
# =============================================================================

def fetch_nba_pdf_for_date(date: datetime, 
                           hours_to_try: List[str] = None) -> Optional[Dict]:
    """
    Tenta baixar o PDF oficial da NBA para uma data específica.
    
    Args:
        date: Data para buscar
        hours_to_try: Lista de horários (ex: ['01PM', '05PM'])
    
    Returns:
        Dict com lesões ou None se não encontrado
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber não instalado. Use: pip install pdfplumber")
        return None
    
    date_str = date.strftime("%Y-%m-%d")
    
    if hours_to_try is None:
        # Horários típicos de publicação (ET)
        hours_to_try = ['01PM', '05PM', '06PM', '07PM', '08PM', '09PM', '10PM', '11PM',
                        '12PM', '02PM', '03PM', '04PM']
    
    for time_str in hours_to_try:
        url = f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date_str}_{time_str}.pdf"
        
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info(f"  ✅ PDF encontrado: {date_str} {time_str}")
                
                # Parse PDF
                injuries = parse_nba_pdf(io.BytesIO(r.content))
                if injuries:
                    return {
                        'date': date_str,
                        'source': 'nba_official_pdf',
                        'source_url': url,
                        'injuries': injuries
                    }
        except Exception as e:
            pass
    
    return None


def parse_nba_pdf(pdf_content: io.BytesIO) -> Dict[str, Dict[str, str]]:
    """Parseia o PDF da NBA e extrai lesões."""
    import pdfplumber
    import re
    
    injuries = {}
    
    try:
        with pdfplumber.open(pdf_content) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                current_team = None
                
                for line in lines:
                    # Skip headers
                    if any(x in line for x in ["Injury Report:", "Page", "GameDate", "GameTime", "Matchup"]):
                        continue
                    
                    # Detectar time
                    for team_full, team_abbr in TEAM_ABBREV_MAP.items():
                        if team_full.replace(" ", "") in line.replace(" ", ""):
                            current_team = team_full
                            break
                    
                    # Detectar status de lesão
                    status_patterns = ['Out', 'Questionable', 'Doubtful', 'Probable', 'Available']
                    for status in status_patterns:
                        if status in line and current_team:
                            # Extrair nome do jogador
                            pattern = rf'([A-Z][a-z]+,\s*[A-Z][a-z]+)\s+{status}'
                            match = re.search(pattern, line)
                            if match:
                                player_raw = match.group(1)
                                # Converter "Last, First" para "First Last"
                                if ',' in player_raw:
                                    parts = player_raw.split(',')
                                    player_name = f"{parts[1].strip()} {parts[0].strip()}"
                                else:
                                    player_name = player_raw
                                
                                if current_team not in injuries:
                                    injuries[current_team] = {}
                                injuries[current_team][player_name] = status.upper()
                                
    except Exception as e:
        logger.error(f"  Erro parsing PDF: {e}")
    
    return injuries


# =============================================================================
# SOURCE 2: nbainjuries Python Package
# =============================================================================

def fetch_nbainjuries_historical(start_date: datetime, end_date: datetime) -> List[Dict]:
    """
    Usa o pacote nbainjuries para obter dados históricos.
    Dados disponíveis desde 2021-22.
    
    pip install nbainjuries
    """
    try:
        from nbainjuries import Injuries
    except ImportError:
        logger.warning("Pacote nbainjuries não instalado. Use: pip install nbainjuries")
        return []
    
    logger.info(f"📦 Buscando dados via nbainjuries ({start_date} - {end_date})...")
    
    all_injuries = []
    
    try:
        inj = Injuries()
        
        # O pacote pode ter método para buscar por range de datas
        # Consultar documentação específica
        current_date = start_date
        while current_date <= end_date:
            try:
                # Tentar buscar para data específica
                date_str = current_date.strftime("%Y-%m-%d")
                data = inj.get_injuries(date=date_str)
                
                if data:
                    all_injuries.append({
                        'date': date_str,
                        'source': 'nbainjuries_package',
                        'injuries': data
                    })
                    logger.info(f"  ✅ {date_str}: {len(data)} lesões")
                
            except Exception as e:
                pass  # Data não disponível
            
            current_date += timedelta(days=1)
            time.sleep(0.1)  # Rate limiting
            
    except Exception as e:
        logger.error(f"Erro nbainjuries: {e}")
    
    logger.info(f"  Total coletado via nbainjuries: {len(all_injuries)} dias")
    return all_injuries


# =============================================================================
# SOURCE 3: API-BASKETBALL (RapidAPI)
# =============================================================================

def fetch_api_basketball_injuries(season: str = "2023-2024") -> Optional[Dict]:
    """
    Busca lesões via API-BASKETBALL no RapidAPI.
    
    Requer API key em RAPIDAPI_KEY env var.
    Free tier: 100 requests/day
    """
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        logger.warning("RAPIDAPI_KEY não configurada. Pulando API-BASKETBALL.")
        return None
    
    logger.info(f"🏀 Buscando dados via API-BASKETBALL (season {season})...")
    
    url = "https://api-basketball.p.rapidapi.com/injuries"
    
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
    }
    
    params = {
        "league": "12",  # NBA
        "season": season
    }
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        
        if r.status_code == 200:
            data = r.json()
            injuries = data.get('response', [])
            logger.info(f"  ✅ API-BASKETBALL: {len(injuries)} registros")
            return {
                'source': 'api_basketball',
                'season': season,
                'injuries': injuries
            }
        else:
            logger.warning(f"  ⚠️ API-BASKETBALL retornou {r.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"  Erro API-BASKETBALL: {e}")
        return None


# =============================================================================
# SOURCE 4: ESPN API
# =============================================================================

def fetch_espn_injuries() -> Optional[Dict]:
    """
    Busca lesões via ESPN Hidden API.
    Nota: pode não ter dados históricos, apenas current.
    """
    logger.info("📺 Buscando dados via ESPN API...")
    
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    
    try:
        r = requests.get(url, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            logger.info(f"  ✅ ESPN API: dados recebidos")
            return {
                'source': 'espn_api',
                'data': data
            }
        else:
            logger.warning(f"  ⚠️ ESPN retornou {r.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"  Erro ESPN: {e}")
        return None


# =============================================================================
# MAIN: Scraper Controller
# =============================================================================

def scrape_historical_injuries(start_date: datetime, end_date: datetime,
                                use_pdf: bool = True,
                                use_nbainjuries: bool = True,
                                use_rapidapi: bool = True) -> List[Dict]:
    """
    Scraper principal que coleta dados de múltiplas fontes.
    
    Args:
        start_date: Data inicial
        end_date: Data final
        use_pdf: Tentar PDFs oficiais da NBA
        use_nbainjuries: Usar pacote nbainjuries
        use_rapidapi: Usar API-BASKETBALL via RapidAPI
    
    Returns:
        Lista de registros de lesões por data
    """
    logger.info("="*60)
    logger.info("🏥 HISTORICAL NBA INJURY SCRAPER")
    logger.info("="*60)
    logger.info(f"  Período: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"  Fontes: PDF={use_pdf}, nbainjuries={use_nbainjuries}, RapidAPI={use_rapidapi}")
    
    all_data = []
    
    # 1. PDFs Oficiais
    if use_pdf:
        logger.info("\n📄 FONTE 1: PDFs Oficiais da NBA")
        logger.info("-"*40)
        current = start_date
        pdf_count = 0
        while current <= end_date:
            result = fetch_nba_pdf_for_date(current)
            if result:
                all_data.append(result)
                pdf_count += 1
            current += timedelta(days=1)
            time.sleep(0.5)  # Gentle rate limiting
        logger.info(f"  Total PDFs coletados: {pdf_count}")
    
    # 2. nbainjuries package
    if use_nbainjuries:
        logger.info("\n📦 FONTE 2: nbainjuries Package")
        logger.info("-"*40)
        nba_data = fetch_nbainjuries_historical(start_date, end_date)
        all_data.extend(nba_data)
    
    # 3. API-BASKETBALL
    if use_rapidapi:
        logger.info("\n🏀 FONTE 3: API-BASKETBALL (RapidAPI)")
        logger.info("-"*40)
        # Determinar seasons a buscar
        seasons = set()
        current = start_date
        while current <= end_date:
            year = current.year
            month = current.month
            if month >= 10:
                season = f"{year}-{year+1}"
            else:
                season = f"{year-1}-{year}"
            seasons.add(season)
            current += timedelta(days=365)
        
        for season in seasons:
            api_data = fetch_api_basketball_injuries(season)
            if api_data:
                all_data.append(api_data)
            time.sleep(1)  # Rate limiting
    
    logger.info("\n" + "="*60)
    logger.info(f"✅ COLETA CONCLUÍDA: {len(all_data)} registros")
    logger.info("="*60)
    
    return all_data


def save_historical_injuries(data: List[Dict], output_file: str = None):
    """Salva os dados coletados em JSON."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"injuries_historical_{timestamp}.json"
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Dados salvos em: {output_file}")
    return output_file


def merge_injuries_to_training_data(injuries_file: Path, games_db_path: str = None):
    """
    Merge os dados de lesões com os jogos históricos para treino.
    
    Adiciona colunas home_injury_impact e away_injury_impact ao DataFrame de jogos.
    """
    import pandas as pd
    
    logger.info("🔄 Fazendo merge de lesões com dados de treino...")
    
    # Carregar lesões
    with open(injuries_file, 'r') as f:
        injuries_data = json.load(f)
    
    # Criar mapping date -> injuries
    injury_map = {}
    for record in injuries_data:
        date = record.get('date')
        if date and 'injuries' in record:
            if date not in injury_map:
                injury_map[date] = {}
            
            injuries = record['injuries']
            if isinstance(injuries, dict):
                for team, players in injuries.items():
                    # Normalizar team name para código
                    team_code = TEAM_ABBREV_MAP.get(team, team[:3].upper())
                    if team_code not in injury_map[date]:
                        injury_map[date][team_code] = []
                    
                    if isinstance(players, dict):
                        for player, status in players.items():
                            injury_map[date][team_code].append({
                                'player': player,
                                'status': status
                            })
    
    logger.info(f"  Mapeadas lesões para {len(injury_map)} datas")
    
    # Salvar mapping para uso futuro
    mapping_file = OUTPUT_DIR / "injury_date_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(injury_map, f, indent=2)
    logger.info(f"  Mapping salvo em: {mapping_file}")
    
    return injury_map


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Scrape historical NBA injuries')
    parser.add_argument('--start', type=str, default='2023-10-01', 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD), defaults to yesterday')
    parser.add_argument('--no-pdf', action='store_true', help='Skip PDF scraping')
    parser.add_argument('--no-nbainjuries', action='store_true', help='Skip nbainjuries')
    parser.add_argument('--no-rapidapi', action='store_true', help='Skip RapidAPI')
    parser.add_argument('--merge', action='store_true', help='Merge with training data')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now() - timedelta(days=1)
    
    # Scrape
    data = scrape_historical_injuries(
        start_date=start_date,
        end_date=end_date,
        use_pdf=not args.no_pdf,
        use_nbainjuries=not args.no_nbainjuries,
        use_rapidapi=not args.no_rapidapi
    )
    
    # Save
    if data:
        output_file = save_historical_injuries(data)
        
        if args.merge:
            merge_injuries_to_training_data(output_file)
    else:
        logger.warning("⚠️ Nenhum dado coletado!")


if __name__ == '__main__':
    main()
