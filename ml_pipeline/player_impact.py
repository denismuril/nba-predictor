"""
Player Impact Module

Calcula o impacto de lesões e minutos jogados no modelo de Totals.
Integra dados de lesões com estatísticas de jogadores.

V2: Adicionado PlayerImpactCalculator com RAPM e fuzzy matching.
"""
import pandas as pd
import numpy as np
import logging
import unicodedata
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# RAPM-Based Player Impact Calculator (V2)
# =============================================================================

class PlayerImpactCalculator:
    """
    Calcula o impacto de lesões usando RAPM (Regularized Adjusted Plus-Minus).
    
    Carrega dados do arquivo CSV local e implementa fuzzy matching para
    garantir que nomes com acentos (ex: 'Luka Dončić') correspondam corretamente.
    
    Usage:
        calculator = PlayerImpactCalculator()
        impact = calculator.calculate_missing_impact(['LeBron James', 'Luka Dončić'])
        print(f"Impacto total: {impact:.2f} RAPM")  # Negativo = time piora
    """
    
    def __init__(self, rapm_path: str = 'data/nba_rapm.csv', rapm_column: str = 'RAPM'):
        """
        Inicializa o calculador.
        
        Args:
            rapm_path: Caminho para o arquivo CSV com dados RAPM
            rapm_column: Nome da coluna com valores RAPM (default: RAPM)
        """
        self.rapm_path = Path(rapm_path)
        self.rapm_column = rapm_column
        self.rapm_df = self._load_rapm()
        self.name_cache: Dict[str, Optional[str]] = {}  # Cache de fuzzy matches
        
        if self.rapm_df is not None:
            logger.info(f"✅ PlayerImpactCalculator: {len(self.rapm_df)} jogadores carregados")
        else:
            logger.warning("⚠️ PlayerImpactCalculator: Sem dados RAPM")
    
    def _load_rapm(self) -> Optional[pd.DataFrame]:
        """Carrega o arquivo CSV de RAPM."""
        try:
            if not self.rapm_path.exists():
                logger.warning(f"⚠️ Arquivo RAPM não encontrado: {self.rapm_path}")
                return None
            
            df = pd.read_csv(self.rapm_path)
            
            # Validar colunas requeridas (Support both old and new format)
            player_col = 'Player' if 'Player' in df.columns else 'player_name'
            
            if self.rapm_column not in df.columns and 'rapm_darko' in df.columns:
                 self.rapm_column = 'rapm_darko'

            required_cols = [player_col, self.rapm_column]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"❌ Colunas faltando no CSV: {missing}")
                return None
            
            # Criar coluna normalizada para busca
            df['name_normalized'] = df[player_col].apply(self._normalize_name)
            df['player_name'] = df[player_col] # Ensure standard column exists
            
            # Converter RAPM para numérico (tratar valores faltantes)
            df[self.rapm_column] = pd.to_numeric(df[self.rapm_column], errors='coerce').fillna(0.0)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar RAPM: {e}")
            return None
    
    def _normalize_name(self, name: str) -> str:
        """
        Normaliza nome para busca: remove acentos, lowercase, remove pontuação.
        
        Exemplos:
            'Luka Dončić' -> 'luka doncic'
            'LeBron James' -> 'lebron james'
            'Nikola Jokić' -> 'nikola jokic'
        """
        if not isinstance(name, str):
            return ''
        
        # Normalizar Unicode (remover acentos)
        normalized = unicodedata.normalize('NFKD', name)
        normalized = normalized.encode('ASCII', 'ignore').decode('ASCII')
        
        # Lowercase e remover pontuação extra
        normalized = normalized.lower()
        normalized = re.sub(r'[^a-z\s]', '', normalized)
        normalized = ' '.join(normalized.split())  # Normalizar espaços
        
        return normalized
    
    def _fuzzy_match(self, name: str, threshold: int = 80) -> Optional[str]:
        """
        Busca aproximada para encontrar jogador mesmo com variações no nome.
        
        Args:
            name: Nome do jogador a buscar
            threshold: Score mínimo para considerar match (0-100)
            
        Returns:
            Nome original no CSV ou None se não encontrado
        """
        if self.rapm_df is None:
            return None
        
        # Verificar cache primeiro
        if name in self.name_cache:
            return self.name_cache[name]
        
        normalized = self._normalize_name(name)
        
        # Busca exata primeiro (mais rápido)
        exact_match = self.rapm_df[self.rapm_df['name_normalized'] == normalized]
        if not exact_match.empty:
            result = exact_match.iloc[0]['player_name']
            self.name_cache[name] = result
            return result
        
        # Fuzzy match (usando Levenshtein simplificado)
        best_match = None
        best_score = 0
        
        for _, row in self.rapm_df.iterrows():
            score = self._similarity_score(normalized, row['name_normalized'])
            if score > best_score and score >= threshold:
                best_score = score
                best_match = row['player_name']
        
        self.name_cache[name] = best_match
        
        if best_match:
            logger.debug(f"   Fuzzy match: '{name}' -> '{best_match}' (score: {best_score})")
        else:
            logger.debug(f"   ⚠️ Sem match para: '{name}'")
        
        return best_match
    
    def _similarity_score(self, s1: str, s2: str) -> int:
        """
        Calcula score de similaridade entre dois nomes (0-100).
        Implementação simples baseada em tokens comuns.
        """
        if not s1 or not s2:
            return 0
        
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        
        if not tokens1 or not tokens2:
            return 0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        jaccard = intersection / union if union > 0 else 0
        
        # Bonus por primeiro/último nome corresponder exatamente
        s1_parts = s1.split()
        s2_parts = s2.split()
        
        bonus = 0
        if s1_parts and s2_parts:
            if s1_parts[-1] == s2_parts[-1]:  # Sobrenome igual
                bonus = 20
            elif s1_parts[0] == s2_parts[0]:  # Primeiro nome igual
                bonus = 10
        
        return min(100, int(jaccard * 80) + bonus)
    
    def get_player_rapm(self, player_name: str) -> float:
        """
        Obtém o RAPM de um jogador específico.
        
        Args:
            player_name: Nome do jogador
            
        Returns:
            Valor RAPM ou 0.0 se não encontrado
        """
        if self.rapm_df is None:
            return 0.0
        
        matched_name = self._fuzzy_match(player_name)
        
        # AUDIT FIX: Adicionar warning quando jogador não é encontrado
        # Isto ajuda a detetar nomes novos (rookies, trades) ou erros de normalização
        if matched_name is None:
            logger.warning(
                f"⚠️ Player not found in RAPM database: '{player_name}'. "
                f"Retornando RAPM=0.0. Verificar se é rookie, trade recente, "
                f"ou erro de escrita no injury report."
            )
            return 0.0
        
        player = self.rapm_df[self.rapm_df['player_name'] == matched_name]
        
        if player.empty:
            return 0.0
        
        return float(player[self.rapm_column].iloc[0])
    
    def calculate_missing_impact(self, injured_list: List[str]) -> float:
        """
        Calcula o impacto total dos jogadores lesionados/ausentes.
        
        Retorna a soma do RAPM dos jogadores na lista.
        Valor POSITIVO significa que o time perde jogadores BOM (impacto negativo no time).
        
        Args:
            injured_list: Lista de nomes de jogadores ausentes
            
        Returns:
            Soma do RAPM (valor positivo = time piora sem esses jogadores)
            
        Example:
            >>> calc = PlayerImpactCalculator()
            >>> impact = calc.calculate_missing_impact(['LeBron James', 'Anthony Davis'])
            >>> print(impact)  # Ex: 4.5 (time perde 4.5 RAPM)
        """
        if not injured_list:
            return 0.0
        
        total_impact = 0.0
        found_players = []
        not_found = []
        
        for player in injured_list:
            if not player or not isinstance(player, str):
                continue
            
            rapm = self.get_player_rapm(player)
            
            if rapm != 0.0:
                total_impact += rapm
                found_players.append((player, rapm))
            else:
                not_found.append(player)
        
        # Logging
        if found_players:
            logger.debug(f"   RAPM calculado para {len(found_players)} jogadores:")
            for name, rapm in found_players:
                logger.debug(f"     • {name}: {rapm:+.2f}")
        
        if not_found:
            logger.debug(f"   ⚠️ Não encontrados: {not_found}")
        
        return total_impact
    
    def _convert_injury_dict_to_df(self, injuries_input) -> pd.DataFrame:
        """
        Converte dict de lesões → DataFrame.
        
        Args:
            injuries_input: dict {'Team Name': {'Player': 'STATUS'}} OU DataFrame
            
        Returns:
            DataFrame com colunas ['team', 'player', 'status']
        """
        # Se já é DataFrame, retornar direto
        if isinstance(injuries_input, pd.DataFrame):
            return injuries_input
        
        # Se é dict, converter
        if isinstance(injuries_input, dict):
            from config.constants import TEAM_ABBREV_MAP
            
            rows = []
            for team_name, players in injuries_input.items():
                # Converter nome completo → abreviação
                team_abbr = TEAM_ABBREV_MAP.get(team_name, team_name)
                
                for player, status in players.items():
                    rows.append({
                        'team': team_abbr,
                        'player': player,
                        'status': status
                    })
            
            return pd.DataFrame(rows)
        
        # Se não é dict nem DataFrame, retornar vazio
        return pd.DataFrame(columns=['team', 'player', 'status'])
    
    def get_team_impact_penalty(self, team: str, injuries_df: pd.DataFrame) -> float:
        """
        Calcula penalidade de impacto para um time específico.
        
        Args:
            team: Código do time (ex: 'LAL')
            injuries_df: DataFrame com colunas ['team', 'player', 'status']
            
        Returns:
            RAPM perdido pelo time (positivo = time piora)
        """
        if injuries_df is None or injuries_df.empty:
            return 0.0
        
        # Filtrar lesões do time (apenas OUT ou DOUBTFUL)
        team_injuries = injuries_df[
            (injuries_df['team'] == team) &
            (injuries_df['status'].str.lower().isin(['out', 'doubtful']))
        ]
        
        if team_injuries.empty:
            return 0.0

        injured_players = team_injuries['player'].tolist()
        return self.calculate_missing_impact(injured_players)


# =============================================================================
# FASE 2: StatLineShift - Análise de Impacto de Ausência
# =============================================================================


def calculate_injury_impact(
    player_name: str,
    team_code: str,
    df_boxscores: pd.DataFrame = None
) -> Dict:
    """
    FASE 2 IMPLEMENTATION: Analisa o impacto histórico de um jogador estar OUT.

    Lógica:
    1. Filtra jogos anteriores onde o jogador NÃO jogou (minutos = 0 ou ausente)
    2. Compara performance do time COM vs SEM o jogador
    3. Calcula "Usage Rate Lift" dos companheiros restantes

    Args:
        player_name: Nome do jogador lesionado (ex: 'Luka Doncic')
        team_code: Código do time (ex: 'DAL')
        df_boxscores: DataFrame com histórico de boxscores
                      Requer colunas: Date, Player, Team, MIN, PTS, USG_PCT

    Returns:
        Dict com:
        - 'pts_impact': Diferença média de pontos/jogo do time (negativo = piora)
        - 'win_pct_without': Win% do time sem este jogador
        - 'win_pct_with': Win% do time com este jogador
        - 'games_without': Número de jogos analisados sem o jogador
        - 'usage_lift': Dict de jogadores com aumento de usage quando ausente
        - 'beneficiaries': Lista top 3 jogadores que mais se beneficiam

    Exemplo:
        >>> impact = calculate_injury_impact('Luka Doncic', 'DAL', df_boxscores)
        >>> print(impact['usage_lift'])  # {'Kyrie Irving': +4.5, 'PJ Washington': +2.1}
        >>> print(impact['pts_impact'])   # -8.3 (time marca 8.3 pontos a menos)
    """
    logger.info(f"📊 Calculando impacto de ausência: {player_name} ({team_code})...")

    # Resultado padrão (neutro)
    default_result = {
        'player_name': player_name,
        'team_code': team_code,
        'pts_impact': 0.0,
        'win_pct_without': 0.5,
        'win_pct_with': 0.5,
        'games_without': 0,
        'games_with': 0,
        'usage_lift': {},
        'beneficiaries': [],
        'error': None
    }

    # Carregar boxscores se não fornecido
    if df_boxscores is None:
        boxscores_path = Path('data/player_boxscores_history.csv')
        if not boxscores_path.exists():
            logger.warning(f"⚠️ Arquivo de boxscores não encontrado: {boxscores_path}")
            default_result['error'] = 'boxscores_not_found'
            return default_result

        try:
            df_boxscores = pd.read_csv(boxscores_path)
            logger.info(f"   Carregados {len(df_boxscores)} registros de boxscores")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar boxscores: {e}")
            default_result['error'] = str(e)
            return default_result

    # Validar colunas necessárias
    required_cols = ['Player', 'Team', 'MIN', 'PTS']
    date_col = 'Date' if 'Date' in df_boxscores.columns else 'date'
    required_cols.append(date_col)

    missing_cols = [c for c in required_cols if c not in df_boxscores.columns]
    if missing_cols:
        logger.warning(f"⚠️ Colunas faltando no boxscore: {missing_cols}")
        default_result['error'] = f'missing_columns: {missing_cols}'
        return default_result

    # Normalizar nome para busca
    def normalize(name):
        if not isinstance(name, str):
            return ''
        normalized = unicodedata.normalize('NFKD', name)
        normalized = normalized.encode('ASCII', 'ignore').decode('ASCII')
        return normalized.lower().strip()

    player_normalized = normalize(player_name)

    # Filtrar boxscores do time
    team_games = df_boxscores[df_boxscores['Team'] == team_code].copy()

    if team_games.empty:
        logger.warning(f"⚠️ Nenhum jogo encontrado para {team_code}")
        default_result['error'] = 'no_team_games'
        return default_result

    # Converter datas
    team_games[date_col] = pd.to_datetime(team_games[date_col])

    # Identificar jogos ÚNICOS (agrupar por data)
    game_dates = team_games.groupby(date_col)

    # Classificar jogos: COM vs SEM o jogador
    games_with_player = []
    games_without_player = []

    for game_date, game_df in game_dates:
        # Verificar se o jogador jogou neste jogo
        player_rows = game_df[
            game_df['Player'].apply(lambda x: normalize(x) == player_normalized if isinstance(x, str) else False)
        ]

        if not player_rows.empty:
            # Jogador presente: verificar se jogou minutos
            player_mins = pd.to_numeric(player_rows['MIN'].iloc[0], errors='coerce')
            if pd.notna(player_mins) and player_mins > 0:
                games_with_player.append(game_df)
            else:
                games_without_player.append(game_df)
        else:
            # Jogador ausente do boxscore
            games_without_player.append(game_df)

    # Precisamos de pelo menos 3 jogos em cada categoria para análise significativa
    if len(games_without_player) < 3:
        logger.info(f"   ⚠️ Poucos jogos sem {player_name} ({len(games_without_player)}). Análise limitada.")
        default_result['games_without'] = len(games_without_player)
        default_result['games_with'] = len(games_with_player)
        return default_result

    # ========== CALCULAR MÉTRICAS ==========

    # 1. Pontos por jogo do time
    def get_team_pts(game_dfs):
        """Soma pontos de todos jogadores do time em cada jogo."""
        totals = []
        for gdf in game_dfs:
            team_pts = pd.to_numeric(gdf['PTS'], errors='coerce').sum()
            totals.append(team_pts)
        return np.mean(totals) if totals else 0

    pts_with = get_team_pts(games_with_player)
    pts_without = get_team_pts(games_without_player)
    pts_impact = pts_without - pts_with  # Negativo = time piora sem jogador

    # 2. Win% (se temos coluna de resultado)
    def get_win_pct(game_dfs):
        """Calcula win% baseado nos jogos."""
        # Método: se tiver PTS suficiente, considerar vitória
        # (Ideal seria ter coluna W/L, mas como proxy usamos pontos > 100)
        if not game_dfs:
            return 0.5
        wins = sum(1 for gdf in game_dfs if pd.to_numeric(gdf['PTS'], errors='coerce').sum() > 105)
        return wins / len(game_dfs) if game_dfs else 0.5

    win_pct_with = get_win_pct(games_with_player)
    win_pct_without = get_win_pct(games_without_player)

    # 3. Usage Rate Lift dos companheiros
    usage_lift = {}
    if 'USG_PCT' in df_boxscores.columns:
        # Agrupar uso médio de cada jogador COM e SEM o star
        all_teammates = set()
        for gdf in games_with_player + games_without_player:
            all_teammates.update(gdf['Player'].tolist())

        # Remover o próprio jogador
        all_teammates.discard(player_name)

        for teammate in all_teammates:
            teammate_norm = normalize(teammate)

            # Uso médio quando star jogou
            usage_with = []
            for gdf in games_with_player:
                tm_row = gdf[gdf['Player'].apply(lambda x: normalize(x) == teammate_norm if isinstance(x, str) else False)]
                if not tm_row.empty:
                    usg = pd.to_numeric(tm_row['USG_PCT'].iloc[0], errors='coerce')
                    if pd.notna(usg):
                        usage_with.append(usg)

            # Uso médio quando star ausente
            usage_without = []
            for gdf in games_without_player:
                tm_row = gdf[gdf['Player'].apply(lambda x: normalize(x) == teammate_norm if isinstance(x, str) else False)]
                if not tm_row.empty:
                    usg = pd.to_numeric(tm_row['USG_PCT'].iloc[0], errors='coerce')
                    if pd.notna(usg):
                        usage_without.append(usg)

            # Calcular lift
            if usage_with and usage_without:
                avg_with = np.mean(usage_with)
                avg_without = np.mean(usage_without)
                lift = avg_without - avg_with
                if abs(lift) > 1.0:  # Apenas lifts significativos (>1%)
                    usage_lift[teammate] = round(lift, 2)

    # Ordenar por lift descendente e pegar top 3
    sorted_by_lift = sorted(usage_lift.items(), key=lambda x: x[1], reverse=True)
    beneficiaries = [
        {'player': name, 'usage_lift': lift, 'pts_boost_estimate': round(lift * 0.3, 1)}
        for name, lift in sorted_by_lift[:3]
    ]

    result = {
        'player_name': player_name,
        'team_code': team_code,
        'pts_impact': round(pts_impact, 1),
        'win_pct_without': round(win_pct_without, 3),
        'win_pct_with': round(win_pct_with, 3),
        'games_without': len(games_without_player),
        'games_with': len(games_with_player),
        'usage_lift': usage_lift,
        'beneficiaries': beneficiaries,
        'error': None
    }

    logger.info(f"   ✅ Análise completa: {len(games_without_player)} jogos sem, {len(games_with_player)} com")
    logger.info(f"   📊 Impacto: {pts_impact:+.1f} pts | Win%: {win_pct_with:.1%} → {win_pct_without:.1%}")

    if beneficiaries:
        logger.info(f"   🔝 Top beneficiários: {[b['player'] for b in beneficiaries]}")

    return result


# =============================================================================
# Funções Legacy (compatibilidade com código existente)
# =============================================================================


def calculate_team_injury_impact(
    team: str,
    injuries_df: pd.DataFrame,
    player_stats_df: pd.DataFrame
) -> float:
    """
    Calcula impacto total de lesões no scoring do time.
    
    Args:
        team: Código do time (ex: 'LAL')
        injuries_df: DataFrame com lesões atuais
        player_stats_df: Stats dos jogadores (PTS, MIN, etc.)
    
    Returns:
        injury_impact: Pontos esperados perdidos (negativo)
    
    Exemplo:
        LeBron James lesionado (27 PPG) → impact = -27
        Austin Reaves questionable (15 PPG × 0.5) → impact = -7.5
        Total LAL injury impact: -34.5 pontos
    """
    if injuries_df.empty or player_stats_df.empty:
        return 0.0
    
    team_injuries = injuries_df[injuries_df['team'] == team]
    
    if team_injuries.empty:
        return 0.0
    
    total_impact = 0.0
    
    for _, injury in team_injuries.iterrows():
        player = injury['player']
        status = injury['status']
        
        # Buscar stats do jogador
        player_stats = player_stats_df[
            (player_stats_df['team'] == team) &
            (player_stats_df['player_name'].str.contains(player.split()[0], case=False, na=False))
        ]
        
        if player_stats.empty:
            continue
            
        ppg = player_stats['pts'].iloc[0] if 'pts' in player_stats.columns else 0
        
        # Aplicar probabilidade baseada no status
        multiplier = {
            'out': 1.0,           # 100% ausente
            'doubtful': 0.75,     # 75% chance de ficar fora
            'questionable': 0.5,  # 50% chance
            'day-to-day': 0.25,   # 25% chance
            'probable': 0.1       # 10% chance
        }.get(status, 0.5)  # Default: questionable
        
        impact = ppg * multiplier
        total_impact += impact
        
        logger.debug(f"   {player} ({team}): {ppg:.1f} PPG × {multiplier} = -{impact:.1f} pts")
    
    return -total_impact  # Negativo pois reduz pontos esperados


def add_player_impact_features(
    df: pd.DataFrame,
    injuries_df: pd.DataFrame = None,
    player_stats_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Adiciona features de impacto de jogadores (lesões, minutos).
    
    Args:
        df: DataFrame com jogos
        injuries_df: DataFrame com lesões atuais (opcional, será buscado se None)
        player_stats_df: DataFrame com stats dos jogadores (opcional)
        
    Returns:
        DataFrame com features adicionadas:
        - home_injury_impact: Impacto de lesões do time da casa (negativo)
        - away_injury_impact: Impacto de lesões do time visitante (negativo)
        - total_injury_impact: Soma dos impactos
        - home_minutes_load: Carga de minutos (fadiga) - últimos 5 jogos
        - away_minutes_load: Carga de minutos (fadiga) - últimos 5 jogos
    """
    logger.info("🏥 Adicionando features de player impact...")
    
    df = df.copy()
    
    # Inicializar colunas
    df['home_injury_impact'] = 0.0
    df['away_injury_impact'] = 0.0
    df['total_injury_impact'] = 0.0
    
    # Se não fornecido, tentar carregar lesões
    if injuries_df is None:
        try:
            from data.scrapers.injury_scraper_v2 import get_injuries_with_cache
            injuries_dict = get_injuries_with_cache()
            
            # Converter dict → DataFrame
            if isinstance(injuries_dict, dict):
                from config.constants import TEAM_ABBREV_MAP
                rows = []
                for team_name, players in injuries_dict.items():
                    team_abbr = TEAM_ABBREV_MAP.get(team_name, team_name)
                    for player, status in players.items():
                        rows.append({'team': team_abbr, 'player': player, 'status': status})
                injuries_df = pd.DataFrame(rows)
            else:
                injuries_df = injuries_dict  # Já é DataFrame
            
            logger.info(f"   ✅ Lesões carregadas: {len(injuries_df)} jogadores")
        except Exception as e:
            logger.warning(f"   ⚠️  Não foi possível carregar lesões: {e}")
            injuries_df = pd.DataFrame()
    
    # Se não fornecido, tentar carregar stats de jogadores
    if player_stats_df is None:
        try:
            player_stats_df = load_player_season_stats()
            logger.info(f"   ✅ Stats de jogadores carregadas: {len(player_stats_df)} jogadores")
        except Exception as e:
            logger.warning(f"   ⚠️  Não foi possível carregar stats: {e}")
            player_stats_df = pd.DataFrame()
    
    # Se temos dados de lesões e stats, calcular impacto
    if not injuries_df.empty and not player_stats_df.empty:
        # Para jogos futuros (previsão)
        today = pd.Timestamp.now().normalize()
        
        # Determinar jogos futuros vs históricos
        if 'date' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df['date'])
                future_mask = df['date_parsed'] >= today
            except:
                # Se não conseguir parsear, assumir todos futuros
                future_mask = pd.Series([True] * len(df), index=df.index)
        else:
            # Sem coluna de data, assumir todos futuros
            future_mask = pd.Series([True] * len(df), index=df.index)
        
        future_games = df[future_mask]
        
        impacts_calculated = 0
        for idx, game in future_games.iterrows():
            home_team = game.get('home_team', game.get('home', ''))
            away_team = game.get('away_team', game.get('away', ''))
            
            if home_team and away_team:
                home_impact = calculate_team_injury_impact(
                    home_team, injuries_df, player_stats_df
                )
                away_impact = calculate_team_injury_impact(
                    away_team, injuries_df, player_stats_df
                )
                
                df.loc[idx, 'home_injury_impact'] = home_impact
                df.loc[idx, 'away_injury_impact'] = away_impact
                impacts_calculated += 1
        
        logger.info(f"   ✅ Impactos calculados para {impacts_calculated} jogos futuros")
    else:
        logger.info("   ℹ️  Sem dados de lesões/stats - usando valores neutros (0)")
    
    # Feature combinada
    df['total_injury_impact'] = df['home_injury_impact'] + df['away_injury_impact']
    
    # Minutes Load (se disponível)
    if 'home_rolling_5_min' in df.columns:
        df['home_minutes_load'] = df['home_rolling_5_min']
        df['away_minutes_load'] = df['away_rolling_5_min']
        logger.info("   ✅ Minutes load features adicionadas")
    else:
        df['home_minutes_load'] = 0
        df['away_minutes_load'] = 0
        logger.info("   ℹ️  Minutes load não disponível")
    
    logger.info("✅ Player impact features completas (+5 features)")
    
    return df


def load_player_season_stats(season: str = '2025-26') -> pd.DataFrame:
    """
    Carrega estatísticas de jogadores da temporada atual.
    
    Args:
        season: Temporada (ex: '2025-26')
        
    Returns:
        DataFrame com colunas:
        - player_name: Nome do jogador
        - team: Time (3 letras)
        - pts: Pontos por jogo
        - min: Minutos por jogo
        - games: Jogos jogados
    """
    try:
        # Tentar carregar do banco de dados
        from data.repositories.db_manager import get_db_manager
        
        db = get_db_manager()
        
        # Query para agregar stats dos jogadores
        query = """
        SELECT 
            player_name,
            team,
            AVG(pts) as pts,
            AVG(min) as min,
            COUNT(*) as games
        FROM player_stats
        WHERE season = %s
        GROUP BY player_name, team
        HAVING COUNT(*) >= 5
        ORDER BY pts DESC
        """
        
        df = pd.read_sql(query, db.get_connection(), params=(season,))
        
        return df
        
    except Exception as e:
        logger.warning(f"Erro ao carregar stats de jogadores: {e}")
        # Retornar DataFrame vazio
        return pd.DataFrame(columns=['player_name', 'team', 'pts', 'min', 'games'])


def calculate_injury_adjusted_total(
    base_total: float,
    home_injury_impact: float,
    away_injury_impact: float
) -> float:
    """
    Ajusta o total esperado baseado no impacto de lesões.
    
    Args:
        base_total: Total base previsto pelo modelo
        home_injury_impact: Impacto das lesões do time da casa (negativo)
        away_injury_impact: Impacto das lesões do visitante (negativo)
        
    Returns:
        Total ajustado
        
    Exemplo:
        base_total = 220
        home_injury_impact = -15  # Star player out
        away_injury_impact = -5   # Rotation player questionable
        
        adjusted = 220 + (-15) + (-5) = 200 pts
    """
    adjusted = base_total + home_injury_impact + away_injury_impact
    
    # Garantir que não fique absurdamente baixo
    min_total = 150  # NBA min realistic total
    max_total = 300  # NBA max realistic total
    
    return np.clip(adjusted, min_total, max_total)


def add_proxy_impact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona proxy features para capturar impacto de lesões em dados históricos.
    
    Como não temos dados de lesões históricos, usamos variação de performance
    como proxy para detectar possíveis lesões/mudanças na rotação.
    
    Args:
        df: DataFrame com rolling stats
        
    Returns:
        DataFrame com +10 proxy features adicionadas
        
    Features:
        - Performance drops (pts, ts%)
        - Rotation stability
        - Efficiency changes
        - Combined impact proxies
    """
    logger.info("📊 Adicionando proxy impact features (variação de performance)...")
    
    df = df.copy()
    features_added = 0
    
    # ==================== PERFORMANCE DROPS ====================
    # Queda súbita em pontos = possível lesão
    
    if all(col in df.columns for col in ['home_rolling_5_pts', 'home_rolling_30_pts']):
        df['home_pts_drop'] = df['home_rolling_5_pts'] - df['home_rolling_30_pts']
        df['away_pts_drop'] = df['away_rolling_5_pts'] - df['away_rolling_30_pts']
        features_added += 2
        logger.info(f"   ✅ Performance drops: {features_added} features")
    
    # ==================== EFFICIENCY DROPS ====================
    # Queda em TS% = perda de scorers eficientes
    
    if all(col in df.columns for col in ['home_rolling_5_ts_pct', 'home_rolling_30_ts_pct']):
        df['home_ts_drop'] = df['home_rolling_5_ts_pct'] - df['home_rolling_30_ts_pct']
        df['away_ts_drop'] = df['away_rolling_5_ts_pct'] - df['away_rolling_30_ts_pct']
        features_added += 2
        logger.info(f"   ✅ Efficiency drops: {features_added} total")
    
    # ==================== COMBINED IMPACT PROXY ====================
    # pts_drop × ts_drop = impacto combinado
    
    if all(col in df.columns for col in ['home_pts_drop', 'home_ts_drop']):
        df['home_impact_proxy'] = df['home_pts_drop'] * df['home_ts_drop'] * 100
        df['away_impact_proxy'] = df['away_pts_drop'] * df['away_ts_drop'] * 100
        df['total_impact_proxy'] = df['home_impact_proxy'] + df['away_impact_proxy']
        features_added += 3
        logger.info(f"   ✅ Combined proxy: {features_added} total")
    
    # ==================== ROTATION STABILITY ====================
    # PIE consistency = estabilidade da rotação
    
    if all(col in df.columns for col in ['home_rolling_5_pie', 'home_rolling_30_pie']):
        # Ratio próximo de 1.0 = rotação estável
        df['home_lineup_consistency'] = df['home_rolling_5_pie'] / (df['home_rolling_30_pie'] + 0.01)
        df['away_lineup_consistency'] = df['away_rolling_5_pie'] / (df['away_rolling_30_pie'] + 0.01)
        
        # Clamp to reasonable range
        df['home_lineup_consistency'] = np.clip(df['home_lineup_consistency'], 0.5, 1.5)
        df['away_lineup_consistency'] = np.clip(df['away_lineup_consistency'], 0.5, 1.5)
        
        features_added += 2
        logger.info(f"   ✅ Lineup consistency: {features_added} total")
    
    # ==================== IMPACT DIFFERENTIAL ====================
    # Diferença de impacto entre times
    
    if 'home_impact_proxy' in df.columns:
        df['impact_differential'] = np.abs(df['home_impact_proxy'] - df['away_impact_proxy'])
        features_added += 1
    
    logger.info(f"✅ Proxy features completas: {features_added} novas features")
    
    return df



if __name__ == "__main__":
    # Teste do módulo
    logging.basicConfig(level=logging.INFO)
    
    print("🏥 Testando Player Impact Module...")
    print("=" * 60)
    
    # Criar dados de teste
    test_df = pd.DataFrame({
        'date': ['2025-12-03', '2025-12-04'],
        'home_team': ['LAL', 'GSW'],
        'away_team': ['BOS', 'PHX'],
        'home_rolling_5_min': [240, 235],
        'away_rolling_5_min': [238, 242]
    })
    
    print("\n📊 DataFrame de teste:")
    print(test_df)
    
    # Adicionar features
    df_with_impact = add_player_impact_features(test_df)
    
    print("\n✅ Features adicionadas:")
    print(df_with_impact[[
        'home_team', 'away_team',
        'home_injury_impact', 'away_injury_impact', 'total_injury_impact'
    ]])
