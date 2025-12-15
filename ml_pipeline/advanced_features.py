"""
Domain Expert Features para NBA Predictor

Implementa 15 features avançadas baseadas em conhecimento de domínio:
- Grupo 1: Matchup-Specific (5 features)
- Grupo 2: Situational (5 features)  
- Grupo 3: Advanced Metrics (5 features)

Usage:
    from ml_pipeline.advanced_features import add_domain_expert_features
    
    df = add_domain_expert_features(df)
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def add_domain_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona todas as 15 domain expert features.
    
    Args:
        df: DataFrame com features existentes
    
    Returns:
        DataFrame com features adicionadas
    """
    logger.info("🎯 Adicionando Domain Expert Features...")
    
    # ========== PACE FEATURES (Dean Oliver) ==========
    # Importar e usar PaceCalculator para rolling features
    try:
        from ml_pipeline.pace_calculator import add_pace_features, add_rolling_pace_features
        df = add_pace_features(df)
        df = add_rolling_pace_features(df, windows=[5, 10])
        logger.info("✅ Pace features (rolling) adicionadas via PaceCalculator")
    except Exception as e:
        logger.warning(f"⚠️ PaceCalculator falhou, usando fallback: {e}")
    
    # Grupo 1: Matchup-Specific
    df = add_pace_differential(df)
    df = add_defensive_rating_matchup(df)
    df = add_rebounding_edge(df)
    df = add_three_point_matchup(df)
    df = add_turnover_pressure(df)
    
    # Grupo 2: Situational
    df = add_clutch_performance(df)
    df = add_travel_fatigue(df)
    df = add_schedule_density(df)
    df = add_playoff_contention(df)
    df = add_injury_impact(df)
    
    # Grupo 3: Advanced Metrics (SLOW - Disabled for speed)
    # df = add_fastbreak_paint_points(df)
    # df = add_second_chance_points(df)
    # df = add_ts_ast_tov_metrics(df)
    
    n_new = 15  # Total de features adicionadas
    logger.info(f"✅ {n_new} domain expert features adicionadas")
    
    return df


# ============================================================================
# GRUPO 1: MATCHUP-SPECIFIC FEATURES
# ============================================================================

def add_pace_differential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: Pace Differential
    
    Diferença de ritmo de jogo entre os times.
    Times rápidos vs lentos = mais posses = mais variância.
    
    VERSÃO  2.0: Usa fórmula oficial da NBA de utils/nba_formulas.py
    - Elimina training-serving skew
    - Cálculo correto de Possessions com ajuste de ORB
    """
    from utils.nba_formulas import calculate_possessions, calculate_pace
    
    try:
        # Verificar se temos todas as colunas necessárias para método standard
        required_cols = ['home_fga', 'home_fta', 'home_oreb', 'away_dreb', 
                        'home_tov', 'home_fgm', 'away_fga', 'away_fta',
                        'away_oreb', 'home_dreb', 'away_tov', 'away_fgm']
        
        if all(col in df.columns for col in required_cols):
            # Usar método standard (oficial)
            home_poss = calculate_possessions(
                fga=df['home_fga'],
                fta=df['home_fta'],
                orb=df['home_oreb'],
                drb_opp=df['away_dreb'],
                tov=df['home_tov'],
                fgm=df['home_fgm'],
                method='standard'
            )
            
            away_poss = calculate_possessions(
                fga=df['away_fga'],
                fta=df['away_fta'],
                orb=df['away_oreb'],
                drb_opp=df['home_dreb'],
                tov=df['away_tov'],
                fgm=df['away_fgm'],
                method='standard'
            )
            
            logger.debug("✅ Pace calculado com método standard (ORB-adjusted)")
        else:
            # Fallback: método simplificado (mas ainda canonical)
            logger.warning("⚠️ Colunas de ORB/DRB ausentes, usando método simplificado")
            home_poss = calculate_possessions(
                fga=df['home_fga'],
                fta=df['home_fta'],
                orb=0,  # Placeholder
                drb_opp=0,  # Placeholder
                tov=df['home_tov'],
                fgm=0,  # Not used in simplified
                method='simplified'
            )
            
            away_poss = calculate_possessions(
                fga=df['away_fga'],
                fta=df['away_fta'],
                orb=0,
                drb_opp=0,
                tov=df['away_tov'],
                fgm=0,
                method='simplified'
            )
        
        # Calculate pace (normalized to 48 minutes)
        df['home_pace'] = calculate_pace(home_poss)
        df['away_pace'] = calculate_pace(away_poss)
        df['pace_differential'] = df['home_pace'] - df['away_pace']
        df['pace_average'] = (df['home_pace'] + df['away_pace']) / 2
        
        logger.debug(f"✅ Pace differential calculado (avg: {df['pace_average'].mean():.1f})")
        
    except Exception as e:
        logger.warning(f"⚠️ Erro ao calcular pace com fórmula canonical: {e}")
        logger.warning("   Usando fallback para pace neutro")
        df['home_pace'] = 100
        df['away_pace'] = 100
        df['pace_differential'] = 0
        df['pace_average'] = 100
    
    return df


def add_defensive_rating_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: Defensive Rating Matchup
    
    Compara defensive rating do time com offensive rating do oponente.
    """
    if 'home_drtg' in df.columns and 'away_ortg' in df.columns:
        df['def_matchup_home'] = df['home_drtg'] - df['away_ortg']
        df['def_matchup_away'] = df['away_drtg'] - df['home_ortg']
        df['def_matchup_net'] = df['def_matchup_home'] - df['def_matchup_away']
        logger.debug("✅ Defensive matchup calculado")
    else:
        if 'home_pts' in df.columns:
            home_ortg = df.get('home_pts', 110) * 100 / df.get('home_fga', 85)
            away_ortg = df.get('away_pts', 110) * 100 / df.get('away_fga', 85)
            home_drtg = df.get('away_pts', 110) * 100 / df.get('away_fga', 85)
            away_drtg = df.get('home_pts', 110) * 100 / df.get('home_fga', 85)
            
            df['def_matchup_home'] = home_drtg - away_ortg
            df['def_matchup_away'] = away_drtg - home_ortg
            df['def_matchup_net'] = df['def_matchup_home'] - df['def_matchup_away']
        else:
            df['def_matchup_home'] = 0
            df['def_matchup_away'] = 0
            df['def_matchup_net'] = 0
    
    return df


def add_rebounding_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: Rebounding Edge
    
    Vantagem em rebotes (ofensivos e defensivos).
    """
    if 'home_orb' in df.columns and 'home_drb' in df.columns:
        home_orb_pct = df['home_orb'] / (df['home_orb'] + df['away_drb'] + 1)
        away_orb_pct = df['away_orb'] / (df['away_orb'] + df['home_drb'] + 1)
        
        home_drb_pct = df['home_drb'] / (df['home_drb'] + df['away_orb'] + 1)
        away_drb_pct = df['away_drb'] / (df['away_drb'] + df['home_orb'] + 1)
        
        df['orb_edge'] = home_orb_pct - away_drb_pct
        df['drb_edge'] = home_drb_pct - away_orb_pct
        df['total_reb_edge'] = df['orb_edge'] + df['drb_edge']
        
        logger.debug("✅ Rebounding edge calculado")
    else:
        if 'home_reb' in df.columns:
            df['total_reb_edge'] = (df['home_reb'] - df['away_reb']) / (df['home_reb'] + df['away_reb'] + 1)
            df['orb_edge'] = df['total_reb_edge'] * 0.3
            df['drb_edge'] = df['total_reb_edge'] * 0.7
        else:
            df['orb_edge'] = 0
            df['drb_edge'] = 0
            df['total_reb_edge'] = 0
    
    return df


def add_three_point_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 4: Three-Point Dependency
    
    Matchup de dependência de 3-pontos vs defesa de 3-pontos.
    """
    if 'home_fg3a' in df.columns and 'home_fga' in df.columns:
        home_3pt_rate = df['home_fg3a'] / (df['home_fga'] + 1)
        away_3pt_rate = df['away_fg3a'] / (df['away_fga'] + 1)
        
        away_3pt_def = df.get('away_opp_fg3_pct', 0.36)
        home_3pt_def = df.get('home_opp_fg3_pct', 0.36)
        
        df['home_3pt_dependency'] = home_3pt_rate * away_3pt_def
        df['away_3pt_dependency'] = away_3pt_rate * home_3pt_def
        df['three_pt_matchup_gap'] = df['home_3pt_dependency'] - df['away_3pt_dependency']
        
        logger.debug("✅ Three-point matchup calculado")
    else:
        df['home_3pt_dependency'] = 0.12
        df['away_3pt_dependency'] = 0.12
        df['three_pt_matchup_gap'] = 0
    
    return df


def add_turnover_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 5: Turnover Pressure
    
    Pressure causado por forçar turnovers vs tendência de cometer.
    """
    if 'home_tov' in df.columns and 'away_tov' in df.columns:
        home_poss = df['home_fga'] + 0.44 * df.get('home_fta', 0) + df['home_tov']
        away_poss = df['away_fga'] + 0.44 * df.get('away_fta', 0) + df['away_tov']
        
        home_tov_pct = df['home_tov'] / (home_poss + 1)
        away_tov_pct = df['away_tov'] / (away_poss + 1)
        
        home_force_tov = df.get('home_stl', df['home_tov'] * 0.5) / (away_poss + 1)
        away_force_tov = df.get('away_stl', df['away_tov'] * 0.5) / (home_poss + 1)
        
        df['tov_pressure_home'] = home_force_tov - away_tov_pct
        df['tov_pressure_away'] = away_force_tov - home_tov_pct
        df['tov_pressure_net'] = df['tov_pressure_home'] - df['tov_pressure_away']
        
        logger.debug("✅ Turnover pressure calculado")
    else:
        df['tov_pressure_home'] = 0
        df['tov_pressure_away'] = 0
        df['tov_pressure_net'] = 0
    
    return df


# ============================================================================
# GRUPO 2: SITUATIONAL FEATURES
# ============================================================================

def add_clutch_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 6: Clutch Performance
    
    Performance em momentos decisivos.
    """
    if 'home_wins' in df.columns:
        df['clutch_score_home'] = df.get('home_wins', 0) / (df.get('home_wins', 0) + df.get('home_losses', 1) + 1) * 0.1
        df['clutch_score_away'] = df.get('away_wins', 0) / (df.get('away_wins', 0) + df.get('away_losses', 1) + 1) * 0.1
        df['clutch_differential'] = df['clutch_score_home'] - df['clutch_score_away']
    else:
        df['clutch_score_home'] = 0
        df['clutch_score_away'] = 0
        df['clutch_differential'] = 0
    
    logger.debug("✅ Clutch performance calculado (aproximação)")
    return df


def add_travel_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 7: Travel Fatigue
    
    Fadiga de viagem baseada em distância e dias de descanso.
    """
    try:
        NBA_CITIES = {
            'LAL': (34.0430, -118.2673), 'LAC': (34.0430, -118.2673),
            'GSW': (37.7680, -122.3878), 'BOS': (42.3662, -71.0621),
            'MIA': (25.7814, -80.1870), 'PHX': (33.4457, -112.0712),
            'MIL': (43.0439, -87.9172), 'DEN': (39.7487, -105.0076),
            'DAL': (32.7904, -96.8104), 'PHI': (39.9012, -75.1720),
            'CLE': (41.4964, -81.6879), 'NYK': (40.7505, -73.9934),
            'BKN': (40.6828, -73.9754), 'CHI': (41.8807, -87.6742),
            'ATL': (33.7573, -84.3963), 'TOR': (43.6435, -79.3791),
            'DET': (42.3408, -83.0553), 'IND': (39.7640, -86.1555),
            'CHA': (35.2251, -80.8391), 'WAS': (38.8992, -77.0211),
            'ORL': (28.5392, -81.3839), 'MIN': (44.9795, -93.2760),
            'OKC': (35.4634, -97.5151), 'POR': (45.5316, -122.6668),
            'UTA': (40.7683, -111.9011), 'SAC': (38.5801, -121.4997),
            'SAS': (29.4271, -98.4375), 'MEM': (35.1382, -90.0506),
            'NOP': (29.9489, -90.0821), 'HOU': (29.7508, -95.3621),
        }
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c
        
        def calculate_travel_fatigue(distance_km, rest_days):
            if distance_km < 100:
                return 0.0
            if distance_km < 500:
                distance_factor = 0.02
            elif distance_km < 1500:
                distance_factor = 0.08
            elif distance_km < 3000:
                distance_factor = 0.15
            else:
                distance_factor = 0.25
            rest_factor = 1.0 / (rest_days + 1)
            fatigue = -(distance_factor * rest_factor)
            return max(fatigue, -0.3)
        
        if 'home_team' not in df.columns or 'away_team' not in df.columns:
            raise ValueError("Sem team columns")
        
        travel_fatigue_home = []
        travel_fatigue_away = []
        
        for idx, row in df.iterrows():
            home_team = row.get('home_team', 'UNK')
            away_team = row.get('away_team', 'UNK')
            
            try:
                from utils.team_normalization import normalize_team
                home_code = normalize_team(home_team) if home_team != 'UNK' else 'UNK'
                away_code = normalize_team(away_team) if away_team != 'UNK' else 'UNK'
            except:
                home_code = home_team
                away_code = away_team
            
            home_fatigue = 0.0
            
            if home_code in NBA_CITIES and away_code in NBA_CITIES:
                home_coords = NBA_CITIES[home_code]
                away_coords = NBA_CITIES[away_code]
                distance = haversine_distance(away_coords[0], away_coords[1], home_coords[0], home_coords[1])
                rest_days = row.get('rest_days_away', 1)
                away_fatigue = calculate_travel_fatigue(distance, rest_days)
            else:
                away_fatigue = -0.1
            
            travel_fatigue_home.append(home_fatigue)
            travel_fatigue_away.append(away_fatigue)
        
        df['travel_fatigue_home'] = travel_fatigue_home
        df['travel_fatigue_away'] = travel_fatigue_away
        df['travel_fatigue_net'] = df['travel_fatigue_home'] - df['travel_fatigue_away']
        
        avg_away_fatigue = np.mean([f for f in travel_fatigue_away if f < 0])
        logger.info(f"✅ Travel fatigue calculado: média away = {avg_away_fatigue:.3f}")
    
    except Exception as e:
        logger.warning(f"⚠️ Erro ao calcular travel fatigue: {e}")
        df['travel_fatigue_home'] = 0
        df['travel_fatigue_away'] = -0.1
        df['travel_fatigue_net'] = -0.1
    
    return df


def add_schedule_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 8: Schedule Density
    
    Densidade de jogos (back-to-back, 3 em 4 dias, etc).
    """
    try:
        if 'date' not in df.columns and 'game_date' not in df.columns:
            logger.debug("⚠️ Sem coluna de data, usando density neutro")
            df['schedule_density_home'] = 0
            df['schedule_density_away'] = 0
            df['schedule_density_gap'] = 0
            return df
        
        date_col = 'date' if 'date' in df.columns else 'game_date'
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.sort_values(date_col)
        
        df['back_to_back_home'] = 0
        df['back_to_back_away'] = 0
        df['games_last_4d_home'] = 1
        df['games_last_4d_away'] = 1
        df['rest_days_home'] = 2
        df['rest_days_away'] = 2
        
        if 'home_team' in df.columns and 'away_team' in df.columns:
            for team_col, prefix in [('home_team', 'home'), ('away_team', 'away')]:
                for team in df[team_col].unique():
                    if pd.isna(team):
                        continue
                    
                    team_games = df[df[team_col] == team].copy()
                    
                    if len(team_games) < 2:
                        continue
                    
                    for i in range(1, len(team_games)):
                        current_idx = team_games.index[i]
                        prev_idx = team_games.index[i-1]
                        
                        current_date = team_games.loc[current_idx, date_col]
                        prev_date = team_games.loc[prev_idx, date_col]
                        
                        if pd.notna(current_date) and pd.notna(prev_date):
                            days_diff = (current_date - prev_date).days
                            
                            df.loc[current_idx, f'rest_days_{prefix}'] = max(0, days_diff - 1)
                            
                            if days_diff == 1:
                                df.loc[current_idx, f'back_to_back_{prefix}'] = 1
                            
                            last_4d = team_games[
                                (team_games[date_col] <= current_date) &
                                (team_games[date_col] > current_date - pd.Timedelta(days=4))
                            ]
                            df.loc[current_idx, f'games_last_4d_{prefix}'] = len(last_4d)
        
        def calculate_density_score(row, prefix):
            score = 0.0
            if row.get(f'back_to_back_{prefix}', 0) == 1:
                score -= 0.20
            games_4d = row.get(f'games_last_4d_{prefix}', 1)
            if games_4d >= 4:
                score -= 0.15
            elif games_4d >= 3:
                score -= 0.08
            rest = row.get(f'rest_days_{prefix}', 2)
            if rest == 0:
                score -= 0.15
            elif rest == 1:
                score -= 0.10
            elif rest >= 3:
                score += 0.05
            return np.clip(score, -0.5, 0.1)
        
        df['schedule_density_home'] = df.apply(lambda row: calculate_density_score(row, 'home'), axis=1)
        df['schedule_density_away'] = df.apply(lambda row: calculate_density_score(row, 'away'), axis=1)
        df['schedule_density_gap'] = df['schedule_density_home'] - df['schedule_density_away']
        
        n_b2b_home = (df['back_to_back_home'] == 1).sum()
        n_b2b_away = (df['back_to_back_away'] == 1).sum()
        logger.info(f"✅ Schedule density calculado: {n_b2b_home} home B2B, {n_b2b_away} away B2B")
    
    except Exception as e:
        logger.warning(f"⚠️ Erro ao calcular schedule density: {e}")
        df['schedule_density_home'] = 0
        df['schedule_density_away'] = 0
        df['schedule_density_gap'] = 0
    
    return df


def add_playoff_contention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 9: Playoff Contention
    
    Status de classificação para playoffs (desperation factor).
    """
    if 'home_wins' in df.columns and 'home_losses' in df.columns:
        home_win_pct = df['home_wins'] / (df['home_wins'] + df['home_losses'] + 1)
        away_win_pct = df['away_wins'] / (df['away_wins'] + df['away_losses'] + 1)
        
        home_desperation = 1 - abs(home_win_pct - 0.45) / 0.45
        away_desperation = 1 - abs(away_win_pct - 0.45) / 0.45
        
        df['playoff_desperation_home'] = np.clip(home_desperation, 0, 1)
        df['playoff_desperation_away'] = np.clip(away_desperation, 0, 1)
        df['playoff_desperation_gap'] = df['playoff_desperation_home'] - df['playoff_desperation_away']
    else:
        df['playoff_desperation_home'] = 0.5
        df['playoff_desperation_away'] = 0.5
        df['playoff_desperation_gap'] = 0
    
    logger.debug("✅ Playoff contention calculado")
    return df


def add_injury_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 10: Injury Impact (RAPM/PPG Based)
    
    Impacto de lesões calculado somando o RAPM (Real Plus-Minus) e PPG dos jogadores ausentes.
    Substitui a heurística de "Player Importance" por dados reais de eficiência.
    
    Features geradas:
    - home_missing_rapm: Soma do RAPM dos jogadores fora (quanto talento o time perdeu)
    - away_missing_rapm: Soma do RAPM dos jogadores fora
    - missing_rapm_diff: home_missing_rapm - away_missing_rapm (positivo = home perdeu mais talento)
    
    Suporta arquivo histórico (injury_date_mapping.json) e scraper tempo real.
    """
    try:
        import sys
        import json
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from utils.team_normalization import normalize_team
        
        # 1. Carregar Estatísticas de Jogadores (RAPM/PPG)
        player_stats_file = project_root / 'data' / 'nba_player_stats.csv'
        player_stats = {}
        
        if player_stats_file.exists():
            try:
                # Carregar CSV ignorando erros de linha
                stats_df = pd.read_csv(player_stats_file)
                # Normalizar nomes de colunas
                stats_df.columns = [c.lower() for c in stats_df.columns]
                
                for _, row in stats_df.iterrows():
                    p_name = row.get('player')
                    if pd.notna(p_name):
                        # Extrair metricas com defaults seguros
                        # RAPM > 0 = bom, RAPM < 0 = ruim.
                        # Se um jogador ruim (RAPM -2) não joga, teoricamente o time melhora? 
                        # Geralmente sim, mas para simplificar e evitar ruído,
                        # consideramos impacto 0 se RAPM < 0 (ausência de bagre não é "reforço" imediato na NBA)
                        # ou mantemos o valor real? Vamos manter o valor real, mas clipado em -2.
                        rapm = pd.to_numeric(row.get('rapm'), errors='coerce')
                        pts = pd.to_numeric(row.get('pts'), errors='coerce')
                        
                        if pd.isna(rapm): rapm = 0.0
                        if pd.isna(pts): pts = 5.0 # Média de bench warmer
                        
                        player_stats[p_name] = {'rapm': rapm, 'pts': pts}
                        
                logger.info(f"   📊 Stats de jogadores carregados: {len(player_stats)} atletas")
            except Exception as e:
                logger.warning(f"   ⚠️ Erro ao carregar player stats: {e}")
        else:
            logger.warning("   ⚠️ Arquivo nba_player_stats.csv não encontrado. Usando fallback.")

        
        STATUS_WEIGHTS = {
            'OUT': 1.0, 'DOUBTFUL': 0.8, 'QUESTIONABLE': 0.5,
            'GTD': 0.5, 'PROBABLE': 0.2, 'AVAILABLE': 0.0, 'UNKNOWN': 0.2
        }
        
        # 2. Carregar Histórico de Lesões
        historical_file = project_root / 'data' / 'injuries_historical' / 'injury_date_mapping.json'
        historical_injuries = {}
        if historical_file.exists():
            try:
                with open(historical_file, 'r') as f:
                    historical_injuries = json.load(f)
            except Exception:
                pass

        # Se não temos histórico E não temos date column, tentar scraper tempo real
        current_injuries_cache = {}
        if not historical_injuries and 'date' not in df.columns:
            from data.scrapers.injury_scraper_v2 import InjuryScraper
            scraper = InjuryScraper()
            current_injuries_cache = scraper.get_current_injuries(use_cache=True)

        def calculate_missing_production(team_code: str, injuries_list: list) -> tuple:
            """Retorna (missing_rapm, missing_ppg)"""
            if not injuries_list:
                return 0.0, 0.0
            
            total_rapm = 0.0
            total_ppg = 0.0
            
            for injury in injuries_list:
                player = injury.get('player', '')
                status = injury.get('status', 'UNKNOWN').upper()
                weight = STATUS_WEIGHTS.get(status, 0.2)
                
                # Buscar stats reais
                stats = player_stats.get(player)
                if stats:
                    p_rapm = stats['rapm']
                    p_ppg = stats['pts']
                else:
                    # Fallback heurístico se não achar nome exato
                    p_rapm = 0.5 # Levemente positivo (rotação média)
                    p_ppg = 8.0
                    
                    # Tentar fuzzy match simples se necessário (ex: "Luka Dončić" vs "Luka Doncic")
                    # (Omitido para performance, assumindo scrapers alinhados)
                
                # Só conta impacto se o jogador for POSITIVO (RAPM > -1)
                # Jogadores muito ruins (-2.0) fora não contam como "perda de talento" significativa
                clean_rapm = max(p_rapm, -1.0)
                
                total_rapm += (clean_rapm * weight)
                total_ppg += (p_ppg * weight)
            
            return total_rapm, total_ppg

        # Listas para novas colunas
        home_missing_rapm_list = []
        away_missing_rapm_list = []
        # Mantemos nomes antigos para compatibilidade com whitelist existente, 
        # mas populamos com RAPM (mais eficiente)
        
        for idx, row in df.iterrows():
            home_team = row.get('home_team', 'UNK')
            away_team = row.get('away_team', 'UNK')
            home_code = normalize_team(home_team) if home_team != 'UNK' else 'UNK'
            away_code = normalize_team(away_team) if away_team != 'UNK' else 'UNK'
            
            h_rapm, h_ppg = 0.0, 0.0
            a_rapm, a_ppg = 0.0, 0.0
            
            # 1. Tentar Histórico
            injuries_found = False
            if 'date' in df.columns and historical_injuries:
                 date_val = row.get('date')
                 if pd.notna(date_val):
                     date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
                     if date_str in historical_injuries:
                         date_data = historical_injuries[date_str]
                         if home_code in date_data:
                             h_rapm, h_ppg = calculate_missing_production(home_code, date_data[home_code])
                         if away_code in date_data:
                             a_rapm, a_ppg = calculate_missing_production(away_code, date_data[away_code])
                         injuries_found = True
            
            # 2. Tentar Tempo Real (apenas se não achou histórico e é data recente/hoje)
            if not injuries_found and current_injuries_cache:
                # Assumindo que current_injuries_cache é {team: [players]} para HOJE
                if home_code in current_injuries_cache:
                    h_m = [] # Converter formato scraper v2 se necessário
                    # O scraper v2 retorna dict {player: status}, adapter necessário?
                    # O código anterior usava get_current_injuries retornando dict?
                    # ScraperV2 retorna dict {Team: {Player: Status}} ou {Team: [List]}?
                    # Revisando ScraperV2... ele retorna {Team: {Player: Status}}
                    # O helper calculate_missing espera lista de dicts.
                    
                    raw_injuries = current_injuries_cache.get(home_code, {})
                    formatted = [{'player': k, 'status': v} for k,v in raw_injuries.items()]
                    h_rapm, h_ppg = calculate_missing_production(home_code, formatted)

                if away_code in current_injuries_cache:
                     raw_injuries = current_injuries_cache.get(away_code, {})
                     formatted = [{'player': k, 'status': v} for k,v in raw_injuries.items()]
                     a_rapm, a_ppg = calculate_missing_production(away_code, formatted)
            
            home_missing_rapm_list.append(h_rapm)
            away_missing_rapm_list.append(a_rapm)

        # Atribuir às colunas existentes (reutilizando nomes para não quebrar pipeline)
        # injury_impact_home agora é "Home Missing RAPM" (Continuous, >0)
        df['injury_impact_home'] = home_missing_rapm_list
        df['injury_impact_away'] = away_missing_rapm_list
        # Net: Home Missing - Away Missing. Quanto maior, pior pra Home.
        # Mas features geralmente são "Home Advantage". 
        # Se 'injury_impact_net' for usado positivamente pelo modelo, 
        # vamos inverter: Away Missing - Home Missing (Vantagem de Talento Disponível)
        # Se Home falta 10 RAPM e Away falta 0, Net = 0 - 10 = -10 (Desvantagem Home).
        df['injury_impact_net'] = df['injury_impact_away'] - df['injury_impact_home']
        
        # Logging estatísticas
        n_inj = sum([1 for x in home_missing_rapm_list if x > 0])
        logger.info(f"✅ Injury Impact (RAPM) calculado: {n_inj} jogos com desfalques de impacto")
        
    except Exception as e:
        logger.warning(f"⚠️ Erro ao calcular injury impact (V2): {e}")
        df['injury_impact_home'] = 0.0
        df['injury_impact_away'] = 0.0
        df['injury_impact_net'] = 0.0
    
    return df


# ============================================================================
# GRUPO 3: ADVANCED METRICS
# ============================================================================

def add_fastbreak_paint_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 11-12: Fast Break & Paint Points
    
    Pontos de contra-ataque e pontos no garrafão via API-Sports.
    """
    try:
        from data.scrapers.multi_api_scraper import get_advanced_stats
        from data.game_id_mapper import get_game_ids
        
        if 'date' not in df.columns or 'home_team' not in df.columns:
            logger.warning("⚠️ Sem date/teams para buscar IDs reais")
            raise ValueError("Missing date or teams")
        
        fastbreak_home = []
        fastbreak_away = []
        paint_home = []
        paint_away = []
        
        for idx, row in df.iterrows():
            date = row.get('date')
            home = row.get('home_team')
            away = row.get('away_team')
            
            stats = None
            
            # Buscar IDs reais das APIs
            if pd.notna(date) and pd.notna(home) and pd.notna(away):
                date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                
                # Get real IDs    
                ids = get_game_ids(date_str, home, away)
                
                if ids:
                    # Tentar com IDs reais
                    if ids.get('api_football'):
                        stats = get_advanced_stats(game_id=ids['api_football'])
                    elif ids.get('sportdata'):
                        # TODO: implementar sportdata
                        pass
                    elif ids.get('sportsblaze'):
                        stats = get_advanced_stats(game_date=date_str)
            
            # Processar stats ou None
            if stats:
                fastbreak_home.append(stats['home']['fast_break'])
                fastbreak_away.append(stats['away']['fast_break'])
                paint_home.append(stats['home']['paint'])
                paint_away.append(stats['away']['paint'])
            else:
                # SEM dados = NaN (melhor que sintético)
                fastbreak_home.append(None)
                fastbreak_away.append(None)
                paint_home.append(None)
                paint_away.append(None)
        
        df['fastbreak_home'] = fastbreak_home
        df['fastbreak_away'] = fastbreak_away
        df['paint_home'] = paint_home
        df['paint_away'] = paint_away
        
        # Log quantos dados reais conseguimos
        real_data_pct = (df['fastbreak_home'].notna().sum() / len(df)) * 100
        logger.info(f"✅ Fast break/paint: {real_data_pct:.1f}% dados REAIS obtidos")
        
    except Exception as e:
        logger.warning(f"⚠️ Erro: {e}")
        # SEM fallback sintético - deixar como NaN
        df['fastbreak_home'] = None
        df['fastbreak_away'] = None
        df['paint_home'] = None
        df['paint_away'] = None

    
    # Normalize by possessions
    possessions_avg = 100
    df['fastbreak_diff_norm'] = (df['fastbreak_home'] - df['fastbreak_away']) / possessions_avg
    df['paint_diff_norm'] = (df['paint_home'] - df['paint_away']) / possessions_avg
    
    return df


def add_second_chance_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 13: Second Chance Points Differential
    
    Pontos de segunda chance (offensive rebounds → points) via API-Sports.
    """
    try:
        from data.scrapers.multi_api_scraper import get_advanced_stats
        from data.game_id_mapper import get_game_ids
        
        if 'date' not in df.columns or 'home_team' not in df.columns:
            logger.warning("⚠️ Sem date/teams para buscar IDs reais")
            raise ValueError("Missing date or teams")
        
        second_chance_home = []
        second_chance_away = []
        
        for idx, row in df.iterrows():
            date = row.get('date')
            home = row.get('home_team')
            away = row.get('away_team')
            
            stats = None
            
            # Buscar IDs reais
            if pd.notna(date) and pd.notna(home) and pd.notna(away):
                date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                
                ids = get_game_ids(date_str, home, away)
                
                if ids:
                    if ids.get('api_football'):
                        stats = get_advanced_stats(game_id=ids['api_football'])
                    elif ids.get('sportsblaze'):
                        stats = get_advanced_stats(game_date=date_str)
            
            # Processar ou None
            if stats:
                second_chance_home.append(stats['home']['second_chance'])
                second_chance_away.append(stats['away']['second_chance'])
            else:
                # SEM dados = NaN
                second_chance_home.append(None)
                second_chance_away.append(None)
        
        df['second_chance_home'] = second_chance_home
        df['second_chance_away'] = second_chance_away
        
        real_data_pct = (df['second_chance_home'].notna().sum() / len(df)) * 100
        logger.info(f"✅ Second chance: {real_data_pct:.1f}% dados REAIS obtidos")
        
    except Exception as e:
        logger.warning(f"⚠️ Erro: {e}")
        df['second_chance_home'] = None
        df['second_chance_away'] = None

    
    # Normalize
    possessions_avg = 100
    df['second_chance_diff_norm'] = (df['second_chance_home'] - df['second_chance_away']) / possessions_avg
    
    return df


def add_ts_ast_tov_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features 14-15: True Shooting % & AST/TOV Ratio
    
    Métricas avançadas de eficiência.
    """
    # True Shooting %
    if 'home_pts' in df.columns and 'home_fga' in df.columns:
        home_ts = df['home_pts'] / (2 * (df['home_fga'] + 0.44 * df.get('home_fta', 0)) + 1)
        away_ts = df['away_pts'] / (2 * (df['away_fga'] + 0.44 * df.get('away_fta', 0)) + 1)
        df['ts_pct_differential'] = home_ts - away_ts
    else:
        df['ts_pct_differential'] = 0
    
    # Assist-to-Turnover Ratio
    if 'home_ast' in df.columns and 'home_tov' in df.columns:
        home_ast_tov = df['home_ast'] / (df['home_tov'] + 1)
        away_ast_tov = df['away_ast'] / (df['away_tov'] + 1)
        df['ast_tov_ratio_gap'] = home_ast_tov - away_ast_tov
    else:
        df['ast_tov_ratio_gap'] = 0
    
    logger.debug("✅ TS% e AST/TOV calculados")
    return df


if __name__ == '__main__':
    print("🔍 Demo: Domain Expert Features\n")
    
    test_df = pd.DataFrame({
        'home_fga': [85, 90, 82],
        'away_fga': [88, 85, 86],
        'home_pts': [110, 115, 105],
        'away_pts': [108, 110, 112],
        'home_orb': [10, 12, 8],
        'home_drb': [32, 35, 30],
        'away_orb': [8, 10, 11],
        'away_drb': [35, 33, 34],
        'home_fg3a': [35, 40, 32],
        'away_fg3a': [38, 36, 35],
        'home_tov': [12, 10, 14],
        'away_tov': [14, 11, 13],
        'home_ast': [25, 28, 22],
        'away_ast': [23, 24, 25],
        'home_wins': [30, 25, 40],
        'home_losses': [15, 20, 10],
        'away_wins': [20, 35, 15],
        'away_losses': [25, 10, 35]
    })
    
    result_df = add_domain_expert_features(test_df)
    
    new_cols = [col for col in result_df.columns if col not in test_df.columns]
    print(f"✅ {len(new_cols)} features adicionadas:")
    for col in new_cols[:10]:
        print(f"  - {col}")
    
    print(f"\n📊 Sample values (primeira linha):")
    for col in new_cols[:5]:
        print(f"  {col}: {result_df[col].iloc[0]:.4f}")
    
    print("\n✅ Demo completo!")
