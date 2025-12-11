import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

def scrape_referees(date_str):
    """
    Faz scraping do official.nba.com para obter os árbitros do dia.
    Retorna dict: { 'HomeTeam': ['Ref1', 'Ref2', 'Ref3'], ... }
    """
    logger.info(f"🔍 Buscando escala de árbitros para {date_str}...")
    
    try:
        # Formato da URL: https://official.nba.com/referee-assignments/
        # A página mostra os assignments do dia atual.
        # Se date_str for hoje, ok. Se for futuro/passado, pode não funcionar corretamente via scraping simples dessa página.
        # Mas vamos tentar a página principal.
        
        url = "https://official.nba.com/referee-assignments/"
        r = requests.get(url, headers=HEADERS, timeout=10)
        
        if r.status_code != 200:
            logger.warning(f"⚠️  Falha ao acessar official.nba.com: {r.status_code}")
            return {}
            
        soup = BeautifulSoup(r.content, 'html.parser')
        
        assignments = {}
        
        # A estrutura da tabela pode variar, mas geralmente é uma tabela com colunas Game, Official 1, Official 2, Official 3
        # Vamos procurar por tabelas
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 4:
                    game_text = cols[0].get_text(strip=True) # Ex: "LAL @ BOS" ou "Lakers @ Celtics"
                    
                    # Tentar extrair times
                    if '@' in game_text:
                        teams = game_text.split('@')
                        home_team_raw = teams[1].strip()
                        
                        refs = []
                        for i in range(1, 4): # Colunas 1, 2, 3 são refs
                            if i < len(cols):
                                ref_name = cols[i].get_text(strip=True)
                                if ref_name:
                                    refs.append(ref_name)
                        
                        if refs:
                            # Mapear nome do time se necessário, mas vamos usar o raw por enquanto e tentar match depois
                            assignments[home_team_raw] = refs
                            
        if assignments:
            logger.info(f"✅ Árbitros encontrados para {len(assignments)} jogos.")
            return assignments
        else:
            logger.warning("⚠️  Nenhum árbitro encontrado na página (pode não ter saído ainda).")
            return {}
            
    except Exception as e:
        logger.error(f"❌ Erro ao fazer scraping de árbitros: {e}")
        return {}
