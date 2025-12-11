import pandas as pd
import logging
from data.scrapers.injury_scraper import obter_injury_report
from data.scrapers.stats_scraper import obter_player_stats

logger = logging.getLogger(__name__)

def normalize_team_name(team_name):
    """
    Extrai a palavra-chave principal do nome do time para facilitar matching.
    Ex: 'Los Angeles Lakers' -> 'Lakers'
    """
    team_keywords = {
        "Lakers": ["Los Angeles Lakers", "L.A. Lakers", "LAL"],
        "Clippers": ["Los Angeles Clippers", "LA Clippers", "LAC"],
        "Celtics": ["Boston Celtics", "BOS"],
        "Warriors": ["Golden State Warriors", "GSW"],
        "Nets": ["Brooklyn Nets", "BKN", "BRK"],
        "Bulls": ["Chicago Bulls", "CHI"],
        "Heat": ["Miami Heat", "MIA"],
        "Mavericks": ["Dallas Mavericks", "DAL"],
        "Nuggets": ["Denver Nuggets", "DEN"],
        "Suns": ["Phoenix Suns", "PHX"],
        "Spurs": ["San Antonio Spurs", "SAS"],
        "Bucks": ["Milwaukee Bucks", "MIL"],
        "76ers": ["Philadelphia 76ers", "PHI"],
        "Sixers": ["Philadelphia 76ers", "PHI"],
        "Rockets": ["Houston Rockets", "HOU"],
        "Knicks": ["New York Knicks", "NYK"],
        "Raptors": ["Toronto Raptors", "TOR"],
        "Jazz": ["Utah Jazz", "UTA"],
        "Grizzlies": ["Memphis Grizzlies", "MEM"],
        "Pelicans": ["New Orleans Pelicans", "NOP"],
        "Trail Blazers": ["Portland Trail Blazers", "POR"],
        "Blazers": ["Portland Trail Blazers", "POR"],
        "Kings": ["Sacramento Kings", "SAC"],
        "Hawks": ["Atlanta Hawks", "ATL"],
        "Wizards": ["Washington Wizards", "WAS"],
        "Pistons": ["Detroit Pistons", "DET"],
        "Hornets": ["Charlotte Hornets", "CHA"],
        "Magic": ["Orlando Magic", "ORL"],
        "Pacers": ["Indiana Pacers", "IND"],
        "Cavaliers": ["Cleveland Cavaliers", "CLE"],
        "Cavs": ["Cleveland Cavaliers", "CLE"],
        "Thunder": ["Oklahoma City Thunder", "OKC"],
        "Timberwolves": ["Minnesota Timberwolves", "MIN"],
        "Wolves": ["Minnesota Timberwolves", "MIN"],
    }
    
    # Buscar a palavra-chave que corresponde ao nome completo
    for keyword, variations in team_keywords.items():
        if any(variation.lower() in team_name.lower() for variation in variations):
            return keyword
    
    # Fallback: retornar o nome original
    return team_name

def get_roster_impact(team_name, game_date=None):
    """
    Calcula o impacto do elenco disponível para um jogo, ponderado pelos minutos jogados.
    
    Lógica V12.1:
    - Cruza elenco com Injury Report.
    - Remove jogadores 'Out'.
    - Calcula Score = Soma(PIE * (MIN / 48)) para os Top 10 jogadores disponíveis (rotação principal).
    
    Args:
        team_name (str): Nome do time (ex: "Los Angeles Lakers").
        game_date (str, optional): Data do jogo.
        
    Returns:
        float: Roster Strength Score (escala aproximada 0-100).
    """
    try:
        # 1. Obter dados
        injuries = obter_injury_report()
        stats_dfs = obter_player_stats()
        
        # Extrair DataFrame com PIE (prioridade: ALL_PLAYERS > pie > NBA_OFFICIAL)
        stats_df = None
        if 'ALL_PLAYERS' in stats_dfs and stats_dfs['ALL_PLAYERS'] is not None and not stats_dfs['ALL_PLAYERS'].empty:
            stats_df = stats_dfs['ALL_PLAYERS']
        elif 'pie' in stats_dfs and stats_dfs['pie'] is not None and not stats_dfs['pie'].empty:
            stats_df = stats_dfs['pie']
        elif 'NBA_OFFICIAL' in stats_dfs and stats_dfs['NBA_OFFICIAL'] is not None:
            stats_df = stats_dfs['NBA_OFFICIAL']
        
        if stats_df is None or stats_df.empty:
            logger.warning(f"⚠️  Sem estatísticas de jogadores para calcular roster impact de {team_name}.")
            return 50.0
            
        # 2. Filtrar jogadores do time
        team_col = None
        for col in ['TEAM', 'Team', 'TEAM_NAME', 'TEAM_ABBREVIATION']:
            if col in stats_df.columns:
                team_col = col
                break
        
        if team_col is None:
            logger.warning("Coluna de time não encontrada no DataFrame.")
            return 50.0
            
        # Normalizar nome para busca
        normalized_name = normalize_team_name(team_name)
        
        # Criar mapeamento de abreviações
        abbrev_map = {
            "Lakers": "LAL", "Clippers": "LAC", "Celtics": "BOS",
            "Warriors": "GSW", "Nets": "BKN", "Bulls": "CHI",
            "Heat": "MIA", "Mavericks": "DAL", "Nuggets": "DEN",
            "Suns": "PHX", "Spurs": "SAS", "Bucks": "MIL",
            "76ers": "PHI", "Sixers": "PHI", "Rockets": "HOU",
            "Knicks": "NYK", "Raptors": "TOR", "Jazz": "UTA",
            "Grizzlies": "MEM", "Pelicans": "NOP", "Trail Blazers": "POR",
            "Blazers": "POR", "Kings": "SAC", "Hawks": "ATL",
            "Wizards": "WAS", "Pistons": "DET", "Hornets": "CHA",
            "Magic": "ORL", "Pacers": "IND", "Cavaliers": "CLE",
            "Cavs": "CLE", "Thunder": "OKC", "Timberwolves": "MIN",
            "Wolves": "MIN"
        }
        
        # Tentar match com palavra-chave primeiro
        team_stats = stats_df[stats_df[team_col].str.contains(normalized_name, case=False, na=False)].copy()
        
        # Se não achou, tentar com abreviação
        if team_stats.empty and normalized_name in abbrev_map:
            abbrev = abbrev_map[normalized_name]
            team_stats = stats_df[stats_df[team_col].str.contains(abbrev, case=False, na=False)].copy()
            if not team_stats.empty:
                logger.debug(f"✅ Match encontrado com abreviação '{abbrev}' para {team_name}")
        
        if team_stats.empty:
            logger.warning(f"⚠️  Time {team_name} ('{normalized_name}') não encontrado nas stats de jogadores.")
            return 50.0
            
        # 3. Identificar Desfalques
        team_injuries = injuries.get(team_name, {})
        out_players = []
        doubtful_players = []
        
        for player, status in team_injuries.items():
            status_upper = status.upper()
            if 'OUT' in status_upper:
                out_players.append(player)
            elif 'DOUBTFUL' in status_upper:
                doubtful_players.append(player)
        
        # 4. Calcular Impacto Ponderado
        player_col = None
        for col in ['PLAYER', 'Player', 'PLAYER_NAME', 'NAME']:
            if col in team_stats.columns:
                player_col = col
                break
        
        if player_col is None:
            logger.warning("Coluna de jogador não encontrada no DataFrame.")
            return 50.0
        
        # Detectar colunas de PIE e MIN
        pie_col = None
        min_col = None
        
        for col in ['PIE', 'Pie', 'PLAYER_IMPACT', 'PTS']:  # Priorizar PIE, fallback PTS
            if col in team_stats.columns:
                pie_col = col
                break
        
        for col in ['MIN', 'Min', 'MINUTES', 'MP']:
            if col in team_stats.columns:
                min_col = col
                break
        
        if pie_col is None or min_col is None:
            logger.warning(f"⚠️  Colunas PIE/PTS ou MIN não encontradas para {team_name}.")
            return 50.0
        
        # Garantir tipos numéricos
        team_stats[pie_col] = pd.to_numeric(team_stats[pie_col], errors='coerce').fillna(0)
        team_stats[min_col] = pd.to_numeric(team_stats[min_col], errors='coerce').fillna(10.0)
        
        # Filtrar jogadores disponíveis (remove OUT)
        def is_player_out(name, out_list):
            return name in out_list or any(out in name for out in out_list)

        available_players = team_stats[~team_stats[player_col].apply(lambda x: is_player_out(x, out_players))].copy()
        
        # Ordenar por minutos para pegar a rotação principal (Top 10)
        rotation = available_players.sort_values(min_col, ascending=False).head(10)
        
        total_impact = 0.0
        for _, player in rotation.iterrows():
            name = player[player_col]
            pie = player[pie_col]
            minutes = player[min_col]
            
            # Penalidade para Doubtful (50% do impacto)
            weight_factor = 1.0
            if any(d in name for d in doubtful_players):
                weight_factor = 0.5
            
            # Fórmula V12.1: Impacto Ponderado
            weight = (minutes / 48.0) * weight_factor
            weighted_impact = pie * weight
            total_impact += weighted_impact
            
        return round(total_impact, 2)

    except Exception as e:
        logger.error(f"❌ Erro ao calcular Roster Impact para {team_name}: {e}")
        return 50.0

def get_roster_strengths_for_game(home_team, away_team, injury_report=None, player_stats=None):
    """
    Wrapper para calcular para os dois times de um jogo.
    Retorna dict compatível com ml_pipeline/predict.py
    """
    home_strength = get_roster_impact(home_team)
    away_strength = get_roster_impact(away_team)
    
    return {
        'home_roster_strength': home_strength,
        'away_roster_strength': away_strength,
        'home_injury_impact': 0.0,  # Placeholder (não usado no core)
        'away_injury_impact': 0.0,  # Placeholder
        'home_available_bpm': home_strength,  # Usar PIE como proxy (PIE é comparável)
        'away_available_bpm': away_strength
    }
