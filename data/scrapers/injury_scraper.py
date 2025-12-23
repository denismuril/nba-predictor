import requests
import logging
import os
import re
import io
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from config.constants import TEAM_ABBREV_MAP
from utils.retry_utils import smart_retry
from utils.cache import smart_cache, TTL_INJURY_REPORT

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    logger.warning("pdfplumber não instalado. PDF parsing não disponível.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

@smart_cache(ttl_hours=TTL_INJURY_REPORT, cache_key_prefix='injuries')
@smart_retry(max_attempts=3, min_wait=2.0, max_wait=8.0)
def obter_injury_report_api_espn():
    """Tenta ESPN Hidden API para Injury Report"""
    try:
        logger.info("🔍 Tentando ESPN API (Injury Report)...")
        
        url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        r = requests.get(url, headers=HEADERS, timeout=10)
        
        if r.status_code == 200:
            logger.info("✅ ESPN API (Injury Report) funcionou!")
            return r.json()
        else:
            logger.warning(f"⚠️  ESPN API retornou {r.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning("⚠️  ESPN API timeout")
        raise  # Re-raise para acionar retry
    except Exception as e:
        logger.warning(f"⚠️  ESPN API erro: {str(e)[:100]}")
        raise  # Re-raise para acionar retry

def obter_injury_report_twitter():
    """
    Busca lesões recentes via Twitter API v2.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        logger.warning("⚠️  TWITTER_BEARER_TOKEN não configurado. Pulando Twitter.")
        return None

    logger.info("🔍 Tentando Twitter API (Lesões)...")
    
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    
    query = '(injury OR out OR questionable OR doubtful) (from:Underdog__NBA) -is:retweet'
    
    params = {
        'query': query,
        'max_results': 20,
        'tweet.fields': 'created_at,text'
    }
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            tweets = data.get('data', [])
            
            injuries = {}
            # Lógica simplificada de parsing (placeholder)
            if injuries:
                logger.info(f"✅ Twitter API: {len(injuries)} lesões encontradas.")
                return injuries
            else:
                logger.warning("⚠️  Twitter API retornou tweets, mas parser não extraiu dados estruturados.")
                return None
        else:
            logger.warning(f"⚠️  Twitter API retornou {r.status_code}: {r.text}")
            return None
            
    except Exception as e:
        logger.warning(f"⚠️  Erro Twitter API: {e}")
        return None

@smart_retry(max_attempts=2, min_wait=1.0, max_wait=5.0)
def _fetch_pdf_with_retry(url):
    """Helper para fetch de PDF com retry."""
    r = requests.get(url, timeout=5, headers=HEADERS)
    if r.status_code == 200:
        return io.BytesIO(r.content)
    else:
        raise requests.exceptions.HTTPError(f"PDF not found: {r.status_code}")

def scrape_injury_report_pdf():
    """
    Baixa e analisa o PDF oficial de lesões da NBA.
    """
    if not HAS_PDFPLUMBER:
        logger.warning("⚠️  Biblioteca 'pdfplumber' não instalada. Pulando PDF oficial.")
        return None

    logger.info("🔍 Tentando PDF Oficial da NBA...")
    
    # Usar timezone ET (Eastern Time) - horário oficial da NBA
    try:
        et_tz = ZoneInfo('America/New_York')
        now_et = datetime.now(et_tz)
    except Exception:
        # Fallback: assumir 3h de diferença do Brasil
        now_et = datetime.now() - timedelta(hours=3)
    
    date_str = now_et.strftime("%Y-%m-%d")
    
    found_pdf = False
    pdf_content = None
    
    # Gerar lista de horários a tentar (do mais recente para o mais antigo)
    # PDFs são publicados a cada 15 minutos: 12:00, 12:15, 12:30, 12:45, 1:00...
    current_hour_et = now_et.hour
    current_minute_et = now_et.minute
    
    # Arredondar minuto atual para o intervalo de 15 min mais próximo (para baixo)
    current_minute_rounded = (current_minute_et // 15) * 15
    
    # Gerar todos os slots de 15 min do horário atual para trás
    times_to_try = []
    for h in range(current_hour_et, -1, -1):
        if h == current_hour_et:
            # Para a hora atual, começar do minuto arredondado
            minutes_range = range(current_minute_rounded, -1, -15)
        else:
            # Para horas anteriores, tentar todos os minutos (45, 30, 15, 00)
            minutes_range = [45, 30, 15, 0]
        
        for m in minutes_range:
            times_to_try.append((h, m))
    
    for h, m in times_to_try:
        # Formato: 12_45PM, 01_00PM, 11_30AM, etc.
        if h == 0:
            time_str = f"12_{m:02d}AM"
        elif h < 12:
            time_str = f"{h:02d}_{m:02d}AM"
        elif h == 12:
            time_str = f"12_{m:02d}PM"
        else:
            time_str = f"{h-12:02d}_{m:02d}PM"
        
        url = (
            f"https://ak-static.cms.nba.com/referee/injury/"
            f"Injury-Report_{date_str}_{time_str}.pdf"
        )
        try:
            logger.debug(f"    Verificando: {url}")
            pdf_content = _fetch_pdf_with_retry(url)
            logger.info(f"✅ PDF Encontrado: {url}")
            found_pdf = True
            break
        except Exception:
            pass
            
    if not found_pdf:
        logger.warning("⚠️  Nenhum PDF de Injury Report encontrado para hoje (ainda).")
        return None

    # Parsear PDF
    injuries: dict[str, dict[str, str]] = {}
    try:
        with pdfplumber.open(pdf_content) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                lines = text.split('\n')
                
                for line in lines:
                    # Pular cabeçalhos
                    if any(x in line for x in ["Injury Report:", "Page", "GameDate", "GameTime", "Matchup"]):
                        continue
                    
                    # Procurar times no formato "TeamName PlayerName,FirstName Status"
                    for team_full, team_abbr in TEAM_ABBREV_MAP.items():
                        team_no_space = team_full.replace(" ", "")
                        
                        if team_no_space in line:
                            # Extrair jogador e status
                            pattern = rf'{team_no_space}\s+([A-Z][a-z]+,[A-Z][a-z]+)\s+(Out|Available|Questionable|Doubtful|Probable)'
                            match = re.search(pattern, line)
                            
                            if match:
                                player_name = match.group(1)
                                status_raw = match.group(2)
                                
                                # Converter "Last,First" para "First Last"
                                if "," in player_name:
                                    p_parts = player_name.split(",")
                                    player_name = f"{p_parts[1].strip()} {p_parts[0].strip()}"
                                
                                # Mapear status
                                status_map = {
                                    "Out": "OUT",
                                    "Available": "AVAILABLE",
                                    "Questionable": "QUESTIONABLE",
                                    "Doubtful": "DOUBTFUL",
                                    "Probable": "PROBABLE"
                                }
                                status = status_map.get(status_raw, status_raw.upper())
                                
                                if team_full not in injuries:
                                    injuries[team_full] = {}
                                
                                injuries[team_full][player_name] = status
                                logger.debug(f"    Encontrado: {team_full} - {player_name}: {status}")
                            
        if injuries:
            total_injuries = sum(len(v) for v in injuries.values())
            logger.info(f"✅ PDF Parseado: {total_injuries} lesões encontradas em {len(injuries)} times.")
            return injuries
        else:
            logger.warning("⚠️  PDF lido mas nenhuma lesão extraída (layout mudou?).")
            return None
            
    except Exception as e:
        logger.error(f"❌ Erro ao processar PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Cache global para injuries
_INJURY_CACHE = None

def obter_injury_report():
    """Pipeline: PDF Oficial → API ESPN → Twitter (Futuro)"""
    global _INJURY_CACHE
    if _INJURY_CACHE is not None:
        return _INJURY_CACHE

    logger.info("\n" + "="*80)
    logger.info("MÓDULO 1: INJURY REPORT (Lesões)")
    logger.info("="*80)
    
    # 1. PDF Oficial (Prioridade Máxima)
    pdf_data = scrape_injury_report_pdf()
    if pdf_data:
        _INJURY_CACHE = pdf_data
        return pdf_data

    # 2. API ESPN (Fallback 1)
    try:
        api_data = obter_injury_report_api_espn()
        if api_data:
            _INJURY_CACHE = api_data
            return api_data
    except Exception as e:
        logger.warning(f"ESPN API falhou após retries: {e}")

    # 3. Twitter API (Fallback 2)
    twitter_data = obter_injury_report_twitter()
    if twitter_data:
        _INJURY_CACHE = twitter_data
        return twitter_data
    
    # 4. Sem dados (Não inventar!)
    logger.error("❌ Nenhuma fonte de Injury Report disponível! (PDF/API/Twitter falharam)")
    _INJURY_CACHE = {}
    return {}


# Alias for backward compatibility with predict.py and player_impact.py
def get_injuries_with_cache():
    """Alias for obter_injury_report() - provides cached injury data."""
    return obter_injury_report()

