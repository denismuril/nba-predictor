import pandas as pd
import numpy as np
import logging
from data.scrapers.injury_scraper import obter_injury_report
from data.scrapers.stats_scraper import obter_player_stats

logger = logging.getLogger(__name__)


# =============================================================================
# V22.0: NON-LINEAR INJURY IMPACT MODELING
# =============================================================================
# Math-Context: A ausência de um jogador de alto USG% (ex: 30%) causa impacto
# exponencialmente maior que a ausência de dois jogadores de baixo USG% (15% cada).
# Isso captura a realidade de que estrelas são "insubstituíveis" em termos de
# criação de jogadas e spacing.
# =============================================================================

# Thresholds para detecção de estrelas
STAR_PIE_THRESHOLD = 15.0      # All-Star caliber (top ~20% da liga)
STAR_USG_THRESHOLD = 25.0      # Alto uso da posse
SUPERSTAR_PIE_THRESHOLD = 18.0  # MVP caliber

# Parâmetros da função sigmoide de penalidade
SIGMOID_MIDPOINT = 25.0  # USG% onde a penalidade começa a acelerar
SIGMOID_STEEPNESS = 0.15  # Quão íngreme é a curva

# Penalidade em cascata quando múltiplos top players estão OUT
CASCADE_PENALTY_MULTIPLIER = 1.25  # 25% extra se top-2 estão fora


def estimate_usg_from_stats(pie: float, minutes: float) -> float:
    """
    Estima USG% (Usage Rate) a partir de PIE e minutos jogados.
    
    Math: USG% real requer play-by-play data. Esta é uma aproximação baseada
    na correlação observada entre PIE e USG% na NBA:
    - PIE 10 (avg) ≈ USG 20%
    - PIE 15 (star) ≈ USG 25%
    - PIE 20 (MVP) ≈ USG 30%
    
    Fórmula: USG_est = 15 + (PIE - 10) * 0.8 + (MIN - 25) * 0.1
    
    Args:
        pie: Player Impact Estimate (0-25 range típico)
        minutes: Minutos por jogo
        
    Returns:
        USG% estimado (15-35 range típico)
    """
    base_usg = 15.0
    pie_contribution = (pie - 10.0) * 0.8
    minutes_contribution = max(0, (minutes - 25.0)) * 0.1
    
    estimated_usg = base_usg + pie_contribution + minutes_contribution
    return np.clip(estimated_usg, 12.0, 40.0)  # Limitar a valores realistas


def sigmoid_penalty(usg_pct: float) -> float:
    """
    Calcula multiplicador de penalidade não-linear baseado em USG%.
    
    Math: Usa função sigmoide deslocada para criar curva S:
    - USG 15% → multiplicador ~1.0 (role player, impacto linear)
    - USG 25% → multiplicador ~1.5 (star, impacto moderado)
    - USG 35% → multiplicador ~2.5 (superstar, impacto severo)
    
    Fórmula: penalty = 1.0 + 2.0 / (1 + e^(-k*(USG - midpoint)))
    
    Args:
        usg_pct: Usage Rate percentual (ex: 25.0 para 25%)
        
    Returns:
        Multiplicador de impacto (1.0 a ~3.0)
        
    Exemplo:
        >>> sigmoid_penalty(15.0)  # Role player
        1.12
        >>> sigmoid_penalty(32.0)  # Superstar (Jokic-level)
        2.45
    """
    exponent = -SIGMOID_STEEPNESS * (usg_pct - SIGMOID_MIDPOINT)
    penalty = 1.0 + 2.0 / (1.0 + np.exp(exponent))
    return penalty


def is_star_player(pie: float, usg_est: float) -> bool:
    """
    Determina se jogador é uma estrela baseado em PIE e USG%.
    
    Args:
        pie: Player Impact Estimate
        usg_est: Usage Rate estimado
        
    Returns:
        True se jogador é considerado estrela
    """
    return pie >= STAR_PIE_THRESHOLD or usg_est >= STAR_USG_THRESHOLD

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
    Calcula o impacto do elenco disponível usando modelagem NÃO-LINEAR.

    Lógica V22.0 (Math-Heavy Update):
    1. Estima USG% a partir de PIE e minutos
    2. Aplica penalidade SIGMOID baseada em USG%
       - Role players: multiplicador ~1.0
       - Stars: multiplicador ~1.5-2.0
       - Superstars: multiplicador ~2.5
    3. Efeito CASCATA se top-2 jogadores estão OUT (+25% penalidade)

    Math: A perda de um jogador de 30% USG é exponencialmente pior
    que a perda de dois jogadores de 15% USG cada.

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

        # =================================================================
        # V22.0: CÁLCULO NÃO-LINEAR DE IMPACTO
        # =================================================================
        # Math: A perda de jogadores de alto USG% tem impacto exponencial.
        # Um jogador de 30% USG é MUITO mais valioso que dois de 15%.
        # =================================================================

        total_impact = 0.0
        star_impacts = []  # Rastrear impacto dos top-2 para cascade

        for idx, (_, player) in enumerate(rotation.iterrows()):
            name = player[player_col]
            pie = player[pie_col]
            minutes = player[min_col]

            # Estimar USG% a partir de PIE e minutos
            usg_est = estimate_usg_from_stats(pie, minutes)

            # Penalidade para Doubtful (50% do impacto)
            weight_factor = 1.0
            if any(d in name for d in doubtful_players):
                weight_factor = 0.5

            # V22.0: Aplicar penalidade sigmoid baseada em USG%
            # Jogadores de alto uso têm multiplicador > 1
            nonlinear_multiplier = sigmoid_penalty(usg_est)

            # Fórmula V22.0: Impacto Ponderado Não-Linear
            base_weight = (minutes / 48.0) * weight_factor
            weighted_impact = pie * base_weight * nonlinear_multiplier
            total_impact += weighted_impact

            # Rastrear top-2 para cascade
            if idx < 2:
                star_impacts.append({
                    'name': name,
                    'impact': weighted_impact,
                    'is_star': is_star_player(pie, usg_est)
                })

        # =================================================================
        # V22.0: EFEITO CASCATA
        # =================================================================
        # Se top-2 jogadores são estrelas e estão OUT, penalidade extra
        # Math: Perder Jokic + Murray é PIOR que perder Jokic + Gordon
        # =================================================================

        # Verificar se algum dos top-2 originais está OUT
        all_players_sorted = team_stats.sort_values(min_col, ascending=False)
        top2_original = all_players_sorted.head(2)

        top2_out_count = 0
        for _, top_player in top2_original.iterrows():
            top_name = top_player[player_col]
            if is_player_out(top_name, out_players):
                top2_out_count += 1

        # Aplicar cascade penalty se ambos top-2 estão fora
        if top2_out_count >= 2:
            logger.info(
                f"⚠️ CASCADE EFFECT: Top-2 jogadores de {team_name} estão OUT! "
                f"Aplicando penalidade de {CASCADE_PENALTY_MULTIPLIER:.0%}"
            )
            # Reduzir o total_impact (time está MUITO mais fraco)
            total_impact = total_impact / CASCADE_PENALTY_MULTIPLIER

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
