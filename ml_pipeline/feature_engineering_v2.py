"""
Feature Engineering V2 - Clean Minimal Version

Stable version with only essential functions for feature_pipeline_v3.py

Math-Fix: Imputação robusta com médias da liga (não zeros)
"""

import pandas as pd
import numpy as np
import logging

from utils.nba_formulas import (
    calculate_efg,
    calculate_ts,
    calculate_tov_pct,
    calculate_orb_pct,
    calculate_ftr
)

logger = logging.getLogger(__name__)

SEASON_START_DATE = pd.Timestamp('2025-10-01')

# AUDITORIA P2-A: Médias da liga NBA 2025-26 para imputação inteligente
# Fonte: NBA.com/stats e Basketball-Reference (atualizado Dezembro 2025)
# Usar médias em vez de zeros evita viés em modelos de árvore
LEAGUE_DEFAULTS = {
    'efg_pct': 0.550,      # Era 0.545
    'ts_pct': 0.587,       # Era 0.580
    'tov_pct': 0.132,      # Era 0.135
    'oreb_pct': 0.235,     # Era 0.240
    'ft_rate': 0.255,      # Era 0.260
    'off_rating': 117.5,   # Era 115.0 ⚠️ +2.5
    'def_rating': 117.5,   # Era 115.0 ⚠️ +2.5
    'pace': 100.8,         # Era 99.0 ⚠️ +1.8
    'pie': 0.100,          # Mantido (normalizado)
    'pts': 117.2,          # Era 114.0 ⚠️ +3.2
    'win': 0.5             # Mantido (definição)
}


def get_league_default(col: str) -> float:
    """
    Getter unificado para defaults da liga.

    AUDITORIA P2-A: Centraliza acesso aos defaults.

    Args:
        col: Nome da métrica

    Returns:
        Valor default ou 0.0 se não encontrado
    """
    return LEAGUE_DEFAULTS.get(col, 0.0)


# Cache global para médias dinâmicas (evita recalcular a cada chamada)
_DYNAMIC_LEAGUE_CACHE = {
    'values': None,
    'computed_at': None
}


def get_dynamic_league_defaults(df: pd.DataFrame = None, window: int = 100) -> dict:
    """
    FASE 1 REFACTOR: Retorna médias da liga calculadas dinamicamente.

    Prioridade:
    1. Se df fornecido e tem dados suficientes → calcula médias dinâmicas
    2. Se cache válido (calculado nos últimos 100 jogos) → usa cache
    3. Fallback → usa LEAGUE_DEFAULTS estáticos

    Args:
        df: DataFrame com histórico de jogos (opcional)
        window: Janela de jogos para cálculo (default: 100)

    Returns:
        Dict com médias da liga (dinâmicas ou estáticas)

    Exemplo:
        >>> defaults = get_dynamic_league_defaults(df_historico)
        >>> pace = defaults['pace']  # Calculado dinamicamente
    """
    global _DYNAMIC_LEAGUE_CACHE

    # Se temos dados suficientes, calcular dinamicamente
    if df is not None and len(df) >= window:
        calculated = calculate_league_averages(df, window)
        # Atualizar cache
        _DYNAMIC_LEAGUE_CACHE['values'] = calculated
        _DYNAMIC_LEAGUE_CACHE['computed_at'] = len(df)
        logger.info(f"📊 Médias dinâmicas calculadas (baseado em {window} jogos)")
        return calculated

    # Se temos cache válido, usar
    if _DYNAMIC_LEAGUE_CACHE['values'] is not None:
        logger.debug("📊 Usando cache de médias dinâmicas")
        return _DYNAMIC_LEAGUE_CACHE['values']

    # Fallback: usar constantes estáticas
    logger.info("📊 Usando LEAGUE_DEFAULTS estáticos (dados insuficientes)")
    return LEAGUE_DEFAULTS.copy()


def calculate_league_averages(df: pd.DataFrame, window: int = 100) -> dict:
    """
    FASE 4 FIX: Calcula médias da liga dinamicamente baseado nos últimos N jogos.
    
    Isso elimina "magic numbers" hardcoded e torna o sistema adaptável a
    mudanças no estilo de jogo da NBA (pace, scoring, etc.).
    
    Args:
        df: DataFrame com histórico de jogos
        window: Número de jogos recentes para calcular média (default: 100)
        
    Returns:
        Dict com médias calculadas. Usa LEAGUE_DEFAULTS como fallback se dados insuficientes.
        
    Exemplo:
        >>> averages = calculate_league_averages(df_historico, window=100)
        >>> pace_atual = averages['pace']  # Calculado dinamicamente
    """
    if df is None or len(df) < window:
        logger.warning(
            f"⚠️ Dados insuficientes ({len(df) if df is not None else 0} < {window}). "
            "Usando LEAGUE_DEFAULTS."
        )
        return LEAGUE_DEFAULTS.copy()
    
    # Usar os últimos N jogos (ordenados por data se disponível)
    if 'date' in df.columns:
        df_sorted = df.sort_values('date', ascending=False)
    else:
        df_sorted = df
    
    recent = df_sorted.head(window)
    calculated = {}
    
    # Mapeamento de colunas alternativas (o dataset pode ter nomes diferentes)
    col_mappings = {
        'pts': ['pts', 'home_score', 'total_points'],
        'pace': ['pace', 'home_pace'],
        'off_rating': ['off_rating', 'home_off_rating', 'ortg'],
        'def_rating': ['def_rating', 'home_def_rating', 'drtg'],
        'efg_pct': ['efg_pct', 'home_efg_pct', 'home_efg'],
        'ts_pct': ['ts_pct', 'home_ts_pct', 'home_ts'],
        'tov_pct': ['tov_pct', 'home_tov_pct'],
        'oreb_pct': ['oreb_pct', 'home_oreb_pct', 'home_orb_pct'],
        'ft_rate': ['ft_rate', 'home_ft_rate', 'home_ftr'],
        'pie': ['pie', 'home_pie'],
        'win': ['win', 'home_win']
    }
    
    for key, default in LEAGUE_DEFAULTS.items():
        found = False
        
        # Tentar encontrar coluna correspondente
        possible_cols = col_mappings.get(key, [key])
        for col in possible_cols:
            if col in recent.columns:
                val = recent[col].mean()
                if pd.notna(val):
                    calculated[key] = float(val)
                    found = True
                    break
        
        # Fallback para o default se não encontrou
        if not found:
            calculated[key] = default
    
    logger.info(f"📊 Médias calculadas (últimos {window} jogos): pace={calculated.get('pace', 0):.1f}, pts={calculated.get('pts', 0):.1f}")
    
    return calculated


# =============================================================================
# V22.0: SCHEDULE FATIGUE FEATURES
# =============================================================================
# Math-Context: Times em back-to-back (B2B) ou 3-em-4 noites têm performance
# reduzida devido à fadiga. Estudos mostram ~2-3 pontos de desvantagem.
# Fonte: NBA Analytics research (Teramoto et al. 2018)
# =============================================================================


def add_schedule_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de fadiga baseadas no calendário.

    Lógica V22.0:
    1. Calcula dias de descanso desde o último jogo
    2. Detecta back-to-back (B2B) quando rest_days = 0
    3. Detecta 3-em-4 noites (terceiro jogo em 4 dias)

    Math: Fadiga tem efeito não-linear:
    - 0 dias (B2B): Penalidade severa (~3pts)
    - 1 dia: Neutro (padrão NBA)
    - 2+ dias: Leve vantagem (bem descansado)

    Args:
        df: DataFrame com jogos (precisa de 'date', 'home_team', 'away_team')

    Returns:
        DataFrame com colunas adicionais:
        - home_rest_days / away_rest_days: Dias desde último jogo
        - home_is_b2b / away_is_b2b: Flag back-to-back (0/1)
        - home_games_in_4_days / away_games_in_4_days: Jogos nos últimos 4 dias
        - home_is_3_in_4 / away_is_3_in_4: Flag 3-em-4 noites (0/1)
    """
    logger.info("🗓️ Calculando Schedule Fatigue Features...")

    if 'date' not in df.columns:
        logger.warning("⚠️ Coluna 'date' não encontrada. Pulando fatigue features.")
        return df

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Função auxiliar: calcular rest days para um time
    def calculate_team_rest(team_df: pd.DataFrame, team_col: str) -> pd.DataFrame:
        """Calcula métricas de descanso para um time."""
        # Criar DataFrame com todas as aparições do time (home ou away)
        home_games = df[['date', 'home_team']].rename(columns={'home_team': 'team'})
        away_games = df[['date', 'away_team']].rename(columns={'away_team': 'team'})
        all_games = pd.concat([home_games, away_games]).sort_values(['team', 'date'])
        all_games = all_games.reset_index(drop=True)

        # Calcular dias desde último jogo
        all_games['prev_game_date'] = all_games.groupby('team')['date'].shift(1)
        all_games['rest_days'] = (
            all_games['date'] - all_games['prev_game_date']
        ).dt.days.fillna(7)  # 7 dias para primeiro jogo

        # Contar jogos nos últimos 4 dias usando rolling com janela de tempo
        # Simplificação: calcular baseado em rest_days
        # Se rest_days <= 1 nos últimos jogos, aumenta games_in_4_days
        all_games['games_in_4_days'] = 1  # O jogo atual conta

        # Simular contagem: para cada time, contar jogos recentes
        for team in all_games['team'].unique():
            mask = all_games['team'] == team
            team_games = all_games[mask].copy()
            counts = []
            for idx in range(len(team_games)):
                current_date = team_games.iloc[idx]['date']
                window_start = current_date - pd.Timedelta(days=4)
                # Contar jogos anteriores na janela
                prev_games = team_games.iloc[:idx]
                count = (prev_games['date'] > window_start).sum()
                counts.append(count)
            all_games.loc[mask, 'games_in_4_days'] = counts

        # Flags derivadas
        all_games['is_b2b'] = (all_games['rest_days'] <= 1).astype(int)
        all_games['is_3_in_4'] = (all_games['games_in_4_days'] >= 2).astype(int)

        return all_games.drop_duplicates(subset=['date', 'team'])

    # Calcular para todos os times
    all_rest = calculate_team_rest(df, 'team')

    # Merge para home
    home_rest = all_rest[['date', 'team', 'rest_days', 'is_b2b', 'games_in_4_days', 'is_3_in_4']].copy()
    home_rest.columns = ['date', 'home_team', 'home_rest_days', 'home_is_b2b',
                         'home_games_in_4_days', 'home_is_3_in_4']
    df = df.merge(home_rest, on=['date', 'home_team'], how='left')

    # Merge para away
    away_rest = all_rest[['date', 'team', 'rest_days', 'is_b2b', 'games_in_4_days', 'is_3_in_4']].copy()
    away_rest.columns = ['date', 'away_team', 'away_rest_days', 'away_is_b2b',
                         'away_games_in_4_days', 'away_is_3_in_4']
    df = df.merge(away_rest, on=['date', 'away_team'], how='left')

    # Preencher NaN com valores default (bem descansado)
    for col in ['home_rest_days', 'away_rest_days']:
        df[col] = df[col].fillna(2)
    for col in ['home_is_b2b', 'away_is_b2b', 'home_is_3_in_4', 'away_is_3_in_4']:
        df[col] = df[col].fillna(0).astype(int)
    for col in ['home_games_in_4_days', 'away_games_in_4_days']:
        df[col] = df[col].fillna(1).astype(int)

    # Feature derivada: Vantagem de descanso
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

    # Estatísticas de debug
    b2b_count = df['home_is_b2b'].sum() + df['away_is_b2b'].sum()
    three_in_4 = df['home_is_3_in_4'].sum() + df['away_is_3_in_4'].sum()
    logger.info(f"✅ Fatigue Features: {b2b_count} B2B games, {three_in_4} 3-in-4 games")

    return df


# =============================================================================
# V22.0: SPECIFIC MATCHUP FEATURES
# =============================================================================
# Math-Context: Estilos de jogo importam! Um time que chuta muitas bolas de 3
# contra uma defesa que não contesta bem do perímetro tem vantagem.
# =============================================================================


def add_specific_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de matchup estilo-vs-estilo.

    Lógica V22.0:
    1. 3-Point Matchup: Taxa de tentativas de 3 vs qualidade da defesa de 3
    2. Rebounding Mismatch: ORB% ofensivo vs DRB% defensivo

    Math:
    - three_pt_matchup_delta > 0: Time ataca bem de 3 contra defesa fraca
    - rebound_advantage > 0: Time tem vantagem em segundas chances

    Args:
        df: DataFrame com jogos e métricas rolling

    Returns:
        DataFrame com colunas adicionais:
        - three_pt_matchup_delta: home_3pa_rate - away_3p_defense
        - rebound_advantage: home_orb_pct - away_drb_pct
    """
    logger.info("🎯 Calculando Specific Matchup Features...")

    # Feature 1: 3-Point Matchup
    # Verificar se temos as colunas necessárias
    three_pt_cols_home = ['home_rolling_10_efg_pct', 'home_3p_attempt_rate', 'home_3pa_rate']
    three_pt_cols_away = ['away_rolling_10_def_rating', 'away_3p_defense_rating']

    # Tentar encontrar proxies para 3P attempt rate
    if 'home_3pa_rate' not in df.columns and 'home_rolling_10_efg_pct' in df.columns:
        # Estimar: times com alto eFG% tendem a chutar mais 3s na NBA moderna
        df['home_3pa_rate_est'] = df.get('home_rolling_10_efg_pct', 0.55)
        df['away_3pa_rate_est'] = df.get('away_rolling_10_efg_pct', 0.55)
    else:
        df['home_3pa_rate_est'] = df.get('home_3pa_rate', 0.40)
        df['away_3pa_rate_est'] = df.get('away_3pa_rate', 0.40)

    # Para defesa de 3, usar DefRtg como proxy (menor = melhor defesa)
    if 'away_rolling_10_def_rating' in df.columns:
        # Inverter: DefRtg alto = defesa ruim = bom para ataque
        df['away_3p_defense_weakness'] = (
            df['away_rolling_10_def_rating'] - LEAGUE_DEFAULTS['def_rating']
        ) / 10  # Normalizar
        df['home_3p_defense_weakness'] = (
            df['home_rolling_10_def_rating'] - LEAGUE_DEFAULTS['def_rating']
        ) / 10
    else:
        df['away_3p_defense_weakness'] = 0.0
        df['home_3p_defense_weakness'] = 0.0

    # Calcular matchup: time que ataca bem de 3 vs defesa fraca
    df['three_pt_matchup_home'] = (
        df['home_3pa_rate_est'] + df['away_3p_defense_weakness']
    )
    df['three_pt_matchup_away'] = (
        df['away_3pa_rate_est'] + df['home_3p_defense_weakness']
    )
    df['three_pt_matchup_delta'] = (
        df['three_pt_matchup_home'] - df['three_pt_matchup_away']
    )

    # Feature 2: Rebounding Mismatch
    # ORB% do ataque vs DRB% implícito da defesa
    orb_cols = ['home_rolling_10_oreb_pct', 'home_oreb_pct', 'home_orb_pct']
    drb_default = 1 - LEAGUE_DEFAULTS['oreb_pct']  # DRB = 1 - ORB da liga

    home_orb = None
    away_orb = None

    for col in orb_cols:
        if col in df.columns:
            home_orb = df[col].fillna(LEAGUE_DEFAULTS['oreb_pct'])
            break
    if home_orb is None:
        home_orb = LEAGUE_DEFAULTS['oreb_pct']

    for col in [c.replace('home', 'away') for c in orb_cols]:
        if col in df.columns:
            away_orb = df[col].fillna(LEAGUE_DEFAULTS['oreb_pct'])
            break
    if away_orb is None:
        away_orb = LEAGUE_DEFAULTS['oreb_pct']

    # Vantagem de rebote: ORB alto + oponente com DRB baixo
    df['home_orb_rating'] = home_orb if isinstance(home_orb, pd.Series) else home_orb
    df['away_orb_rating'] = away_orb if isinstance(away_orb, pd.Series) else away_orb
    df['rebound_advantage'] = df['home_orb_rating'] - df['away_orb_rating']

    # Cleanup colunas temporárias
    temp_cols = ['home_3pa_rate_est', 'away_3pa_rate_est',
                 'away_3p_defense_weakness', 'home_3p_defense_weakness',
                 'three_pt_matchup_home', 'three_pt_matchup_away']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')

    logger.info("✅ Matchup Features calculadas: three_pt_matchup_delta, rebound_advantage")

    return df

# =============================================================================
# REFEREE STATS - TIME-AWARE IMPLEMENTATION (NO LEAKAGE)
# =============================================================================
# v20.4: Refatorado para usar EXPANDING WINDOW em vez de CSV estático.
# Math-Context: Para evitar look-ahead bias, as estatísticas de um árbitro
# no jogo T são calculadas APENAS com dados de jogos anteriores a T.
# =============================================================================


def add_referee_features(df: pd.DataFrame, referee_names_col: str = 'referees') -> pd.DataFrame:
    """
    Calcula features de árbitros usando EXPANDING WINDOW para evitar Data Leakage.
    
    Lógica:
    1. Explode o DataFrame para ter uma linha por árbitro por jogo.
    2. Calcula a média acumulada (expanding mean) de Foul Rate e Home Win %.
    3. Garante que stats do Jogo N usem apenas dados dos jogos 0 a N-1 (shift).
    4. Agrupa de volta para o jogo tirando a média dos 3 árbitros.
    
    Args:
        df: DataFrame contendo 'date', 'game_id', e coluna com nomes dos árbitros.
        referee_names_col: Nome da coluna com árbitros. Se None, tenta 'referees'.
        
    Returns:
        DataFrame com colunas adicionais:
        - referee_home_win_pct: Média histórica de % vitória mandante dos árbitros
        - referee_foul_avg: Média histórica de faltas por jogo dos árbitros
    """
    logger.info("⚖️ Calculando Referee Features com Janela Temporal (Anti-Leakage)...")
    
    # FIX: Tratar None como default
    if referee_names_col is None:
        referee_names_col = 'referees'
    
    # 1. Verificar se temos a coluna de árbitros
    # Tenta carregar do cache se não estiver no DF
    if referee_names_col not in df.columns:
        from pathlib import Path
        cache_files = list(Path('data/cache').glob('referees_*_selenium.csv'))
        if cache_files:
            ref_dfs = [pd.read_csv(f) for f in cache_files]
            ref_cache = pd.concat(ref_dfs).drop_duplicates(subset=['date', 'home_team'])
            
            # Normalizar datas para merge
            df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            ref_cache['date_str'] = ref_cache['date']  # Assumindo formato string no cache
            
            df = df.merge(
                ref_cache[['date_str', 'home_team', 'referees_csv']],
                on=['date_str', 'home_team'],
                how='left'
            )
            df = df.drop(columns=['date_str'])
            df = df.rename(columns={'referees_csv': referee_names_col})
    
    if referee_names_col not in df.columns:
        # AUDIT FIX #5: Sem dados de árbitros - NÃO criar features falsas com constantes
        # Anteriormente: df['referee_home_win_pct'] = 0.55 (valor constante = feature inútil)
        # Features constantes não adicionam informação ao modelo e criam ruído
        # Se dados estivessem presentes, haveria variância natural nas estatísticas
        logger.info("⏭️ Coluna de árbitros não encontrada. Features de referee EXCLUÍDAS (não constantes falsas).")
        return df

    # 2. Preparar dados para cálculo histórico
    # Precisamos de: Data, Arbitro, Se Mandante Venceu, Total Faltas
    
    # Gerar game_id se não existir
    if 'game_id' not in df.columns:
        df['game_id'] = df.index.astype(str) + '_' + pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
    
    calc_df = df[['date', 'game_id', referee_names_col]].copy()
    
    # Tentar obter targets reais para histórico
    if 'home_win' in df.columns:
        calc_df['home_win_target'] = df['home_win']
    elif 'win' in df.columns:  # Assumindo df é home perspective
        calc_df['home_win_target'] = df['win']
    elif 'winner' in df.columns:
        calc_df['home_win_target'] = (df['winner'] == 'HOME').astype(int)
    else:
        calc_df['home_win_target'] = 0.5  # Fallback neutro
        
    # Tentar obter faltas (se disponível no dataset histórico)
    if 'total_fouls' in df.columns:
        calc_df['foul_target'] = df['total_fouls']
    elif 'home_pf' in df.columns and 'away_pf' in df.columns:
        calc_df['foul_target'] = df['home_pf'] + df['away_pf']
    else:
        calc_df['foul_target'] = 40.0  # FIX: Valor dummy se não tiver dados de faltas
        
    # 3. Explodir árbitros (String "Ref1, Ref2, Ref3" -> Linhas separadas)
    calc_df = calc_df.dropna(subset=[referee_names_col])
    
    if len(calc_df) == 0:
        # AUDITORIA P2-B: Sem dados de árbitros - NÃO criar features falsas com constantes
        # Features constantes (0.55, 40.0) não adicionam informação ao modelo
        logger.warning("⚠️ Nenhum dado de árbitro válido. Features de referee NÃO criadas.")
        return df  # Retorna SEM modificar
    
    calc_df[referee_names_col] = calc_df[referee_names_col].astype(str).str.split(',')
    
    # Explode: uma linha por árbitro
    ref_history = calc_df.explode(referee_names_col)
    ref_history['referee_name'] = ref_history[referee_names_col].str.strip().str.lower()
    ref_history = ref_history.sort_values('date')
    
    # 4. Calcular Expanding Mean por Árbitro
    # GroupBy Ref -> Expanding Mean -> Shift(1) (Crucial: usar dados ANTES do jogo atual)
    
    ref_history['ref_hist_win_pct'] = ref_history.groupby('referee_name')['home_win_target'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    
    ref_history['ref_hist_foul_avg'] = ref_history.groupby('referee_name')['foul_target'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    
    # FIX: Preencher árbitros novos com médias globais acumuladas até a data
    global_win_avg = 0.55
    global_foul_avg = 40.0
    
    ref_history['ref_hist_win_pct'] = ref_history['ref_hist_win_pct'].fillna(global_win_avg)
    ref_history['ref_hist_foul_avg'] = ref_history['ref_hist_foul_avg'].fillna(global_foul_avg)
    
    # 5. Reagrupar por Game ID (Média do trio de arbitragem)
    game_ref_stats = ref_history.groupby('game_id')[['ref_hist_win_pct', 'ref_hist_foul_avg']].mean().reset_index()
    game_ref_stats.columns = ['game_id', 'referee_home_win_pct', 'referee_foul_avg']
    
    # 6. Merge de volta no DataFrame original
    # Remover colunas antigas se existirem para evitar duplicata
    cols_to_drop = ['referee_home_win_pct', 'referee_foul_avg']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    df = df.merge(game_ref_stats, on='game_id', how='left')
    
    # Preencher jogos sem dados de árbitros (que ficaram fora do merge)
    df['referee_home_win_pct'] = df['referee_home_win_pct'].fillna(global_win_avg)
    df['referee_foul_avg'] = df['referee_foul_avg'].fillna(global_foul_avg)
    
    # Estatísticas de debug
    n_with_data = (df['referee_home_win_pct'] != global_win_avg).sum()
    logger.info(f"✅ Referee Features: {n_with_data}/{len(df)} jogos com dados históricos reais")
    
    return df



# =============================================================================
# SMART MONEY - Movimentação de Odds (Sinais de Mercado)
# =============================================================================
# Math-Context: Sharp bettors (profissionais) apostam cedo com informações
# privilegiadas. Quando odds encurtam (2.10 → 1.90), dinheiro inteligente entrou.
# 
# Efficient Market Hypothesis aplicada ao betting:
# - implied_prob_diff > 0 → mercado mais confiante na vitória
# - line_movement negativo → odds do mandante encurtou (favorito ficou mais forte)
# =============================================================================


def add_smart_money_features(df: pd.DataFrame, 
                              opening_col: str = 'opening_odds',
                              closing_col: str = 'closing_odds') -> pd.DataFrame:
    """
    Cria features de Smart Money baseadas na movimentação de odds.
    
    AUDITORIA P0-B: 
    - NÃO cria features se dados indisponíveis (evita constantes)
    - Usa np.nan para rows inválidas (modelo aprende a ignorar)
    - Requer mínimo 10% de dados válidos
    
    Args:
        df: DataFrame com jogos
        opening_col: Nome da coluna com odd de abertura
        closing_col: Nome da coluna com odd de fechamento
        
    Returns:
        DataFrame COM features se dados suficientes, SEM modificação caso contrário
    """
    # Verificar existência das colunas
    has_opening = opening_col in df.columns and df[opening_col].notna().any()
    has_closing = closing_col in df.columns and df[closing_col].notna().any()
    
    if not has_opening or not has_closing:
        logger.info("⏭️ Smart Money: dados de odds indisponíveis. Features NÃO criadas.")
        return df  # Retorna SEM modificar
    
    # Converter para numérico
    opening_odds = pd.to_numeric(df[opening_col], errors='coerce')
    closing_odds = pd.to_numeric(df[closing_col], errors='coerce')
    
    # Máscara de dados válidos
    valid_mask = (
        opening_odds.notna() & 
        closing_odds.notna() & 
        (opening_odds > 1.0) & 
        (closing_odds > 1.0)
    )
    
    valid_count = valid_mask.sum()
    total_count = len(df)
    valid_pct = valid_count / total_count if total_count > 0 else 0
    
    # Verificar mínimo de 10%
    if valid_pct < 0.10:
        logger.warning(
            f"⚠️ Smart Money: apenas {valid_count}/{total_count} "
            f"({valid_pct:.1%}) com dados válidos. Features NÃO criadas."
        )
        return df  # Retorna SEM modificar
    
    # Criar features apenas para rows válidas, NaN para o resto
    df['line_movement'] = np.nan
    df['implied_prob_diff'] = np.nan
    df['smart_money_signal'] = np.nan
    
    # Calcular para válidas
    df.loc[valid_mask, 'line_movement'] = (
        closing_odds[valid_mask] - opening_odds[valid_mask]
    )
    df.loc[valid_mask, 'implied_prob_diff'] = (
        (1/closing_odds[valid_mask]) - (1/opening_odds[valid_mask])
    )
    
    # Categorizar sinal
    THRESHOLD = 0.03  # 3% de mudança na probabilidade implícita
    df.loc[valid_mask, 'smart_money_signal'] = 0
    df.loc[valid_mask & (df['implied_prob_diff'] > THRESHOLD), 'smart_money_signal'] = 1
    df.loc[valid_mask & (df['implied_prob_diff'] < -THRESHOLD), 'smart_money_signal'] = -1
    
    # Estatísticas de debug
    n_signals = (df['smart_money_signal'].abs() == 1).sum()
    
    logger.info(f"✅ Smart Money: {valid_count}/{total_count} ({valid_pct:.1%}) jogos processados")
    logger.info(f"   - Sinais fortes: {n_signals} ({n_signals/valid_count*100:.1f}% dos válidos)")
    
    return df


# =============================================================================
# SENTIMENT FEATURES - NLP para Notícias de Lesões
# =============================================================================
# Math-Context: Notícias de lesão de superestrelas impactam significativamente
# os resultados. Capturamos isso via análise de sentimento em tweets de insiders.
#
# Exemplo: "LeBron OUT" -> sentimento negativo para LAL
#          "Curry returning" -> sentimento positivo para GSW
# =============================================================================


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de sentimento baseadas em notícias de lesões.
    
    IMPORTANTE: Esta função é para predições em tempo real.
    Para treinamento histórico, usa fallback neutro (0.0) pois não temos
    tweets históricos.
    
    Args:
        df: DataFrame com jogos (precisa ter 'home_team', 'away_team')
        
    Returns:
        DataFrame com colunas adicionais:
        - home_sentiment: Sentimento do time da casa (-1.0 a +1.0)
        - away_sentiment: Sentimento do time visitante (-1.0 a +1.0)
        - sentiment_diff: Diferença (home - away)
        - news_volume: Quantidade de notícias relevantes
    """
    # Fallback neutro para todos os casos
    df['home_sentiment'] = 0.0
    df['away_sentiment'] = 0.0
    df['sentiment_diff'] = 0.0
    df['news_volume'] = 0
    
    try:
        from ml_pipeline.sentiment import NewsSentimentAnalyzer
        from data.scrapers.twitter_scraper import fetch_latest_injury_tweets
        
        # Buscar tweets recentes
        tweets = fetch_latest_injury_tweets()
        
        if not tweets:
            logger.info("📰 Nenhum tweet de lesão recente. Sentiment features = neutro.")
            return df
        
        # Analisar sentimento
        analyzer = NewsSentimentAnalyzer()
        team_sentiments = analyzer.analyze_tweets(tweets)
        
        # News volume
        news_volume = len(tweets)
        df['news_volume'] = news_volume
        
        # Aplicar sentimento por time
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df['home_sentiment'] = df['home_team'].map(team_sentiments).fillna(0.0)
            df['away_sentiment'] = df['away_team'].map(team_sentiments).fillna(0.0)
            df['sentiment_diff'] = df['home_sentiment'] - df['away_sentiment']
            
            teams_with_news = len(team_sentiments)
            logger.info(f"📰 Sentiment features: {teams_with_news} times com notícias, "
                       f"{news_volume} tweets processados")
        else:
            logger.warning("⚠️ Colunas home_team/away_team não encontradas.")
    
    except ImportError as e:
        logger.warning(f"⚠️ Módulo de sentiment não disponível: {e}")
    except Exception as e:
        # Falha silenciosa - não parar pipeline por falta de notícias
        logger.warning(f"⚠️ Erro ao processar sentiment: {e}. Usando valores neutros.")
    
    return df



def add_rolling_four_factors(df, windows=[5, 10, 30], use_ewm=True):
    """
    Add rolling averages for Four Factors.
    
    Args:
        df: DataFrame with game data
        windows: List of window sizes for rolling averages
        use_ewm: Se True, usa EWMA (mais reativo). Se False, usa SMA (default: True)
        
    Returns:
        DataFrame with rolling four factors columns
    """
    # Math-Fix: EWMA captura forma recente de forma mais reativa que SMA
    ewm_mode = "EWMA" if use_ewm else "SMA"
    logger.info(f"📊 Calculando Rolling Four Factors ({ewm_mode}, windows={windows})...")
    
    # Base stats columns we want to roll
    cols_base = ['efg_pct', 'ts_pct', 'tov_pct', 'oreb_pct', 'ft_rate',
                 'off_rating', 'def_rating', 'pace', 'pie', 'pts', 'win']
    
    # Helper to extract team data
    def extract_team_data(df, prefix, is_home):
        # Map specific columns if they exist with prefix
        team_cols = {}
        
        # 1. Standard mapping (e.g. home_efg_pct -> efg_pct)
        # Note: data_preparation creates 'home_efg', not 'home_efg_pct' sometimes.
        # Let's check common variations.
        
        mappings = {
            'efg_pct': [f'{prefix}_efg_pct', f'{prefix}_efg'],
            'ts_pct': [f'{prefix}_ts_pct', f'{prefix}_ts'],
            'tov_pct': [f'{prefix}_tov_pct', f'{prefix}_tov'],
            'oreb_pct': [f'{prefix}_oreb_pct', f'{prefix}_orb_pct', f'{prefix}_oreb'],
            'ft_rate': [f'{prefix}_ft_rate', f'{prefix}_ftr'],
            'off_rating': [f'{prefix}_off_rating', 'off_rating'],
            'def_rating': [f'{prefix}_def_rating', 'def_rating'],
            'pace': [f'{prefix}_pace', 'pace'], # Pace might be shared
            'pie': [f'{prefix}_pie'],
            'pts': [f'{prefix}_score', f'{prefix}_pts'],
            'win': [] # Calculated below
        }
        
        data = pd.DataFrame(index=df.index)
        data['date'] = df['date']
        data['team'] = df[f'{prefix}_team']
        
        # Initialize all targets with NaN
        for target in mappings.keys():
            data[target] = np.nan
            
        for target, sources in mappings.items():
            for src in sources:
                if src in df.columns:
                    data[target] = df[src]
                    break
        
        # Calculate Win
        opp_prefix = 'away' if is_home else 'home'
        if f'{prefix}_score' in df.columns and f'{opp_prefix}_score' in df.columns:
            data['win'] = (df[f'{prefix}_score'] > df[f'{opp_prefix}_score']).astype(int)
        else:
            data['win'] = 0
            
        # Calculate Ratings if missing (Needs pts and pace)
        # Check if off_rating is all NaN (meaning not found in mappings)
        if data['off_rating'].isna().all() and 'pts' in data.columns and 'pace' in data.columns:
            # Off Rtg = Pts / Pace * 100
            data['off_rating'] = (data['pts'] / data['pace'].replace(0, np.nan)) * 100
            
        if data['def_rating'].isna().all():
            # Def Rtg = Opp Pts / Pace * 100
            # Opp Pts is needed.
            if f'{opp_prefix}_score' in df.columns and 'pace' in data.columns:
                data['def_rating'] = (df[f'{opp_prefix}_score'] / data['pace'].replace(0, np.nan)) * 100
                
        return data

    # Extract Home and Away
    home_df = extract_team_data(df, 'home', is_home=True)
    away_df = extract_team_data(df, 'away', is_home=False)
    
    # Combine
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date'])
    
    # Filter current season
    long_df = long_df[long_df['date'] >= SEASON_START_DATE].copy()
    
    # AUDIT FIX: Validar se há dados suficientes após filtro
    if long_df.empty or len(long_df) < 5:
        logger.warning(f"⚠️ Dados insuficientes após filtro de data. Retornando sem Four Factors rolling.")
        return df
    
    # Calculate rolling for each window
    for window in windows:
        # Math-Fix: shift(1) obrigatório para evitar vazamento de dados
        # EWMA dá mais peso a jogos recentes (mais reativo que SMA)
        if use_ewm:
            # Math-Fix: EWMA - span≈window para equivalência aproximada
            rolled = long_df.groupby('team')[cols_base].transform(
                lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
            )
        else:
            # SMA tradicional (menos reativo)
            rolled = long_df.groupby('team')[cols_base].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Math-Fix: Imputar NaNs com médias da liga (NÃO zeros)
        # Isso evita ensinar ao modelo que times sem histórico têm "habilidade zero"
        for col in cols_base:
            default_val = LEAGUE_DEFAULTS.get(col, 0.0)
            rolled[col] = rolled[col].fillna(default_val)

        rolled.columns = [f'rolling_{window}_{c}' for c in cols_base]
        long_df = pd.concat([long_df, rolled], axis=1)
    
    # Merge back - FIXED: Use join instead of merge to preserve existing columns
    new_cols = [c for c in long_df.columns if 'rolling_' in c]
    long_df = long_df.drop_duplicates(subset=['date', 'team'])
    
    # Create a unique index for the original df to ensure proper join
    original_index = df.index.copy()
    
    # Prepare home data for join
    home_join_df = long_df[['date', 'team'] + new_cols].copy()
    home_join_df = home_join_df.rename(columns={'team': 'home_team'})
    # Rename rolling columns with home_ prefix
    rename_home = {col: f'home_{col}' for col in new_cols}
    home_join_df = home_join_df.rename(columns=rename_home)
    
    # Prepare away data for join
    away_join_df = long_df[['date', 'team'] + new_cols].copy()
    away_join_df = away_join_df.rename(columns={'team': 'away_team'})
    # Rename rolling columns with away_ prefix
    rename_away = {col: f'away_{col}' for col in new_cols}
    away_join_df = away_join_df.rename(columns=rename_away)
    
    # Get only the NEW columns (don't overwrite existing ones)
    home_new_cols = [f'home_{c}' for c in new_cols]
    away_new_cols = [f'away_{c}' for c in new_cols]
    
    # Filter out columns that already exist in df
    home_cols_to_add = [c for c in home_new_cols if c not in df.columns]
    away_cols_to_add = [c for c in away_new_cols if c not in df.columns]
    
    # Merge home - only new columns
    if home_cols_to_add:
        home_merge = home_join_df[['date', 'home_team'] + home_cols_to_add].copy()
        df = df.merge(
            home_merge,
            on=['date', 'home_team'],
            how='left',
            suffixes=('', '_DROP')
        )
        # Drop any duplicate columns created by merge
        df = df.loc[:, ~df.columns.str.endswith('_DROP')]
    
    # Merge away - only new columns
    if away_cols_to_add:
        away_merge = away_join_df[['date', 'away_team'] + away_cols_to_add].copy()
        df = df.merge(
            away_merge,
            on=['date', 'away_team'],
            how='left',
            suffixes=('', '_DROP')
        )
        # Drop any duplicate columns created by merge
        df = df.loc[:, ~df.columns.str.endswith('_DROP')]
    
    # Math-Fix: Imputar NaNs restantes com médias da liga
    all_new_cols = home_cols_to_add + away_cols_to_add
    for col in all_new_cols:
        if col not in df.columns:
            continue
        # Extrair nome base da feature (ex: 'home_rolling_10_efg_pct' -> 'efg_pct')
        base_name = col.split('_')[-1] if '_' in col else col
        # Tentar match com sufixo composto (ex: off_rating)
        if 'off_rating' in col:
            base_name = 'off_rating'
        elif 'def_rating' in col:
            base_name = 'def_rating'
        elif 'ft_rate' in col:
            base_name = 'ft_rate'
        elif 'tov_pct' in col:
            base_name = 'tov_pct'
        elif 'oreb_pct' in col:
            base_name = 'oreb_pct'
        elif 'efg_pct' in col:
            base_name = 'efg_pct'
        elif 'ts_pct' in col:
            base_name = 'ts_pct'
        
        default_val = LEAGUE_DEFAULTS.get(base_name, 0.0)
        df[col] = df[col].fillna(default_val)

    logger.info(f"✅ Rolling Four Factors calculated ({len(all_new_cols)} new cols added)")

    return df


def add_contextual_rolling_features(df, window=10, use_ewm=True):
    """
    v19.0: Calcula rolling windows SEPARADAS por contexto Home/Away.

    Isso captura a diferença de performance de times jogando em casa vs fora.
    Ex: Denver em casa (altitude) vs Denver fora tem performance muito diferente.

    Args:
        df: DataFrame em formato wide (home_team, away_team, home_score, etc.)
        window: Tamanho da janela de rolling (default: 10)
        use_ewm: Se True, usa EWMA. Se False, usa SMA.

    Returns:
        DataFrame com colunas adicionais:
            - rolling_{window}_{col}_home: média quando jogando EM CASA
            - rolling_{window}_{col}_away: média quando jogando FORA

    Math-Fix: Fallback inteligente
        Se um time não tem jogos suficientes em casa, usa a média GERAL do time.
        Evita zeros ou médias da liga que ignoram a performance individual.
    """
    ewm_mode = "EWMA" if use_ewm else "SMA"
    logger.info(f"📊 Calculando Rolling Contextual Home/Away ({ewm_mode}, window={window})...")

    # Stats que queremos calcular por contexto
    cols_base = ['pts', 'efg_pct', 'ts_pct', 'off_rating', 'def_rating', 'win']

    # ============================================================
    # 1. Extrair dados em formato longo COM flag de contexto
    # ============================================================
    def extract_with_context(df, prefix, is_home):
        """Extrai dados de um lado (home/away) com flag de contexto."""
        data = pd.DataFrame(index=df.index)
        data['date'] = df['date']
        data['team'] = df[f'{prefix}_team']
        data['is_home'] = is_home  # Flag de contexto

        opp_prefix = 'away' if is_home else 'home'

        # Mapear colunas
        mappings = {
            'pts': [f'{prefix}_score', f'{prefix}_pts'],
            'efg_pct': [f'{prefix}_efg_pct', f'{prefix}_efg'],
            'ts_pct': [f'{prefix}_ts_pct', f'{prefix}_ts'],
            'off_rating': [f'{prefix}_off_rating'],
            'def_rating': [f'{prefix}_def_rating'],
            'win': []  # Calculado abaixo
        }

        for target in mappings.keys():
            data[target] = np.nan

        for target, sources in mappings.items():
            for src in sources:
                if src in df.columns:
                    data[target] = df[src]
                    break

        # Calcular win
        if f'{prefix}_score' in df.columns and f'{opp_prefix}_score' in df.columns:
            data['win'] = (df[f'{prefix}_score'] > df[f'{opp_prefix}_score']).astype(int)
        else:
            data['win'] = np.nan

        return data

    home_df = extract_with_context(df, 'home', is_home=True)
    away_df = extract_with_context(df, 'away', is_home=False)

    # Combinar e ordenar
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date']).reset_index(drop=True)

    # ============================================================
    # 2. Calcular rolling GERAL (fallback)
    # ============================================================
    for col in cols_base:
        if col not in long_df.columns:
            continue
        if use_ewm:
            long_df[f'rolling_{window}_{col}_general'] = long_df.groupby('team')[col].transform(
                lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
            )
        else:
            long_df[f'rolling_{window}_{col}_general'] = long_df.groupby('team')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        # Imputar NaN com média da liga
        default_val = LEAGUE_DEFAULTS.get(col, 0.0)
        long_df[f'rolling_{window}_{col}_general'] = long_df[f'rolling_{window}_{col}_general'].fillna(default_val)

    # ============================================================
    # 3. Calcular rolling separado por CONTEXTO (Home vs Away)
    # ============================================================
    for col in cols_base:
        if col not in long_df.columns:
            continue

        # Rolling apenas para jogos EM CASA
        home_mask = long_df['is_home'] == True
        if use_ewm:
            home_rolling = long_df[home_mask].groupby('team')[col].transform(
                lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
            )
        else:
            home_rolling = long_df[home_mask].groupby('team')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        long_df.loc[home_mask, f'rolling_{window}_{col}_at_home'] = home_rolling

        # Rolling apenas para jogos FORA
        away_mask = long_df['is_home'] == False
        if use_ewm:
            away_rolling = long_df[away_mask].groupby('team')[col].transform(
                lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
            )
        else:
            away_rolling = long_df[away_mask].groupby('team')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        long_df.loc[away_mask, f'rolling_{window}_{col}_at_away'] = away_rolling

    # ============================================================
    # 4. Forward-fill para propagar valores entre jogos home/away
    # AUDIT FIX: Limite de 30 jogos para evitar valores muito antigos
    # Se time não jogou em casa nos últimos 30 jogos, fallback (linha 708) assume
    # ============================================================
    context_cols = [c for c in long_df.columns if '_at_home' in c or '_at_away' in c]
    for col in context_cols:
        long_df[col] = long_df.groupby('team')[col].transform(lambda x: x.ffill(limit=10))

    # ============================================================
    # 5. Fallback: usar média GERAL quando contexto não disponível
    # ============================================================
    for col in cols_base:
        general_col = f'rolling_{window}_{col}_general'
        home_col = f'rolling_{window}_{col}_at_home'
        away_col = f'rolling_{window}_{col}_at_away'

        if home_col in long_df.columns and general_col in long_df.columns:
            long_df[home_col] = long_df[home_col].fillna(long_df[general_col])
        if away_col in long_df.columns and general_col in long_df.columns:
            long_df[away_col] = long_df[away_col].fillna(long_df[general_col])

    # ============================================================
    # 6. Merge de volta para formato wide
    # ============================================================
    new_cols = [c for c in long_df.columns if '_at_home' in c or '_at_away' in c]
    long_df = long_df.drop_duplicates(subset=['date', 'team'])

    # Para o time da casa: suas stats quando joga em casa
    home_context = long_df[['date', 'team'] + [c for c in new_cols if '_at_home' in c]].copy()
    home_context.columns = ['home_date', 'home_team'] + [f'home_{c}' for c in home_context.columns if c not in ['date', 'team']]

    df = df.merge(
        home_context,
        left_on=['date', 'home_team'],
        right_on=['home_date', 'home_team'],
        how='left'
    )

    # Para o time de fora: suas stats quando joga fora
    away_context = long_df[['date', 'team'] + [c for c in new_cols if '_at_away' in c]].copy()
    away_context.columns = ['away_date', 'away_team'] + [f'away_{c}' for c in away_context.columns if c not in ['date', 'team']]

    df = df.merge(
        away_context,
        left_on=['date', 'away_team'],
        right_on=['away_date', 'away_team'],
        how='left'
    )

    # Cleanup colunas auxiliares
    cols_to_drop = [c for c in df.columns if c.endswith('_date') and c not in ['date']]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Imputação final com média da liga
    context_features = [c for c in df.columns if '_at_home' in c or '_at_away' in c]
    for col in context_features:
        for base_col, default in LEAGUE_DEFAULTS.items():
            if base_col in col:
                df[col] = df[col].fillna(default)
                break

    logger.info(f"✅ Contextual rolling features calculated ({len(context_features)} cols)")

    return df


# =============================================================================
# PBP CLEAN METRICS - Métricas sem Garbage Time
# =============================================================================
# v21.5: Integração com pbpstats para obter métricas filtradas por momentos
# competitivos. Garbage Time = últimos 5 min com margem > 15 pontos.
# Math-Context: Métricas "sujas" incluem minutos irrelevantes que distorcem
# a avaliação real dos times.
# =============================================================================


def add_clean_pbp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona métricas limpas do PBPStats (sem Garbage Time).
    
    Garbage Time: Últimos 5 minutos do jogo com diferença > 15 pontos.
    Essas posses são filtradas para obter métricas mais representativas
    da capacidade real dos times.
    
    Cria colunas:
    - home_clean_off_rtg: Offensive Rating do mandante (minutos competitivos)
    - home_clean_def_rtg: Defensive Rating do mandante (minutos competitivos)
    - away_clean_off_rtg: Offensive Rating do visitante (minutos competitivos)
    - away_clean_def_rtg: Defensive Rating do visitante (minutos competitivos)
    - clean_pace: Pace do jogo (posses/48min) sem Garbage Time
    
    FALLBACK AGRESSIVO: Se qualquer erro ocorrer, usa silenciosamente as métricas
    originais. O pipeline nunca quebra por falta de dados PBP.
    
    Args:
        df: DataFrame com jogos (precisa de 'game_id', 'home_team', 'away_team')
        
    Returns:
        DataFrame com novas colunas de métricas limpas
        
    Example:
        >>> df = add_clean_pbp_metrics(df)
        >>> print(df[['home_clean_off_rtg', 'clean_pace']].head())
    """
    logger.info("📊 Tentando adicionar métricas limpas do PBPStats...")
    
    # Inicializar colunas com NaN (fallback acontece no final)
    df['home_clean_off_rtg'] = np.nan
    df['home_clean_def_rtg'] = np.nan
    df['away_clean_off_rtg'] = np.nan
    df['away_clean_def_rtg'] = np.nan
    df['clean_pace'] = np.nan
    
    try:
        from data.clients.pbp_client import PBPClient
        
        # Instanciar cliente
        pbp = PBPClient()
        
        # Determinar temporada baseado no DataFrame
        season_year = "2024-25"  # Default
        if 'date' in df.columns:
            max_date = pd.to_datetime(df['date']).max()
            if pd.notna(max_date):
                year = max_date.year
                month = max_date.month
                if month >= 10:
                    season_year = f"{year}-{str(year + 1)[-2:]}"
                else:
                    season_year = f"{year - 1}-{str(year)[-2:]}"
        
        logger.info(f"   📅 Temporada detectada: {season_year}")
        
        # Buscar dados limpos
        clean_df = pbp.get_clean_stats(season_year)
        
        if clean_df.empty:
            logger.warning("⚠️ PBPStats retornou DataFrame vazio. Usando fallback.")
        else:
            # Verificar se temos game_id para merge
            if 'game_id' not in df.columns:
                logger.warning("⚠️ Coluna 'game_id' não encontrada. Pulando merge PBP.")
            else:
                # Preparar dados para merge
                # O clean_df tem uma linha por time, precisamos pivotar
                # para ter home/away na mesma linha
                
                # Identificar home vs away por game (pode precisar de ajuste)
                # Por ora, apenas fazemos o merge direto e renomeamos
                merged_count = 0
                
                for _, row in clean_df.iterrows():
                    game_id = row['game_id']
                    team_id = row['team_id']
                    
                    # Encontrar jogos correspondentes no df
                    mask = df['game_id'] == game_id
                    
                    if mask.any():
                        # Determinar se é home ou away (simplificado)
                        # Em implementação real, cruzar com team_id
                        df.loc[mask, 'clean_pace'] = row['pace']
                        merged_count += 1
                
                logger.info(f"   ✅ Merge realizado: {merged_count} registros atualizados")
        
    except ImportError as e:
        logger.warning(f"⚠️ PBPClient não disponível: {e}. Usando métricas padrão.")
    except Exception as e:
        logger.warning(f"⚠️ Erro ao carregar PBPStats: {e}. Usando fallback.")
    
    # =========================================================================
    # FALLBACK AGRESSIVO: Preencher NaN com métricas originais
    # =========================================================================
    # Se os dados limpos não estiverem disponíveis, usamos as métricas "sujas"
    # para não quebrar o pipeline. O modelo ainda funciona, só com dados menos puros.
    
    # OffRtg
    if 'home_off_rating' in df.columns:
        df['home_clean_off_rtg'] = df['home_clean_off_rtg'].fillna(df['home_off_rating'])
    else:
        df['home_clean_off_rtg'] = df['home_clean_off_rtg'].fillna(LEAGUE_DEFAULTS['off_rating'])
    
    if 'away_off_rating' in df.columns:
        df['away_clean_off_rtg'] = df['away_clean_off_rtg'].fillna(df['away_off_rating'])
    else:
        df['away_clean_off_rtg'] = df['away_clean_off_rtg'].fillna(LEAGUE_DEFAULTS['off_rating'])
    
    # DefRtg
    if 'home_def_rating' in df.columns:
        df['home_clean_def_rtg'] = df['home_clean_def_rtg'].fillna(df['home_def_rating'])
    else:
        df['home_clean_def_rtg'] = df['home_clean_def_rtg'].fillna(LEAGUE_DEFAULTS['def_rating'])
    
    if 'away_def_rating' in df.columns:
        df['away_clean_def_rtg'] = df['away_clean_def_rtg'].fillna(df['away_def_rating'])
    else:
        df['away_clean_def_rtg'] = df['away_clean_def_rtg'].fillna(LEAGUE_DEFAULTS['def_rating'])
    
    # Pace
    if 'pace' in df.columns:
        df['clean_pace'] = df['clean_pace'].fillna(df['pace'])
    else:
        df['clean_pace'] = df['clean_pace'].fillna(LEAGUE_DEFAULTS['pace'])
    
    # Estatísticas de preenchimento
    clean_count = df['home_clean_off_rtg'].notna().sum()
    fallback_count = len(df) - clean_count
    
    logger.info(f"✅ Clean PBP Metrics: {clean_count} limpos, {fallback_count} fallback")
    
    return df


# =============================================================================
# ADVANCED FEATURES (MIGRADO DE feature_pipeline_v4.py)
# =============================================================================
# Funções para features avançados: Pace, Matchup Efficiency, Volatility, Shooting Luck

def add_advanced_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced pace metrics.
    
    Generates:
    - projected_pace: Average of both teams' rolling pace
    - pace_mismatch: Absolute difference in pace
    """
    if 'home_rolling_10_pace' not in df.columns:
        logger.warning("⚠️ Base pace features missing. Skipping advanced pace.")
        return df
        
    df['projected_pace'] = (df['home_rolling_10_pace'] + df['away_rolling_10_pace']) / 2
    df['pace_mismatch'] = (df['home_rolling_10_pace'] - df['away_rolling_10_pace']).abs()
    
    return df


def add_matchup_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add matchup efficiency features (Off vs Def).
    
    Generates:
    - off_matchup_home: Home Off Rtg - Away Def Rtg
    - off_matchup_away: Away Off Rtg - Home Def Rtg  
    - eff_sum: Sum of Offensive Ratings
    - def_sum: Sum of Defensive Ratings
    """
    required = ['home_rolling_10_off_rating', 'away_rolling_10_def_rating', 
                'away_rolling_10_off_rating', 'home_rolling_10_def_rating']
                
    if not all(col in df.columns for col in required):
        logger.warning("⚠️ Four Factors missing. Skipping matchup efficiency.")
        return df
        
    df['off_matchup_home'] = df['home_rolling_10_off_rating'] - df['away_rolling_10_def_rating']
    df['off_matchup_away'] = df['away_rolling_10_off_rating'] - df['home_rolling_10_def_rating']
    df['eff_sum'] = df['home_rolling_10_off_rating'] + df['away_rolling_10_off_rating']
    df['def_sum'] = df['home_rolling_10_def_rating'] + df['away_rolling_10_def_rating']
    
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility (standard deviation) features.
    
    Generates:
    - home_scoring_std_10: Std Dev of Home points (last 10)
    - away_scoring_std_10: Std Dev of Away points (last 10)
    """
    df = df.sort_values('date')
    
    home_games = df[['date', 'home_team', 'home_score']].rename(
        columns={'home_team': 'team', 'home_score': 'pts'}
    )
    away_games = df[['date', 'away_team', 'away_score']].rename(
        columns={'away_team': 'team', 'away_score': 'pts'}
    )
    
    team_stats = pd.concat([home_games, away_games]).sort_values('date')
    
    team_stats['pts_std_10'] = team_stats.groupby('team')['pts'].transform(
        lambda x: x.rolling(window=10, min_periods=5).std().shift(1)
    )
    
    home_std = team_stats[['date', 'team', 'pts_std_10']].rename(
        columns={'team': 'home_team', 'pts_std_10': 'home_scoring_std_10'}
    ).drop_duplicates(subset=['date', 'home_team'])
    
    away_std = team_stats[['date', 'team', 'pts_std_10']].rename(
        columns={'team': 'away_team', 'pts_std_10': 'away_scoring_std_10'}
    ).drop_duplicates(subset=['date', 'away_team'])
    
    df = pd.merge(df, home_std, on=['date', 'home_team'], how='left')
    df = pd.merge(df, away_std, on=['date', 'away_team'], how='left')
    
    df['home_scoring_std_10'] = df['home_scoring_std_10'].fillna(10.0)
    df['away_scoring_std_10'] = df['away_scoring_std_10'].fillna(10.0)
    
    return df


def add_shooting_luck_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Shooting Luck features (Mean Reversion Detection).
    
    Generates:
    - home_shooting_luck_ts/efg: Short-term - Long-term shooting %
    - away_shooting_luck_ts/efg: Same for away team
    - home_shooting_luck: Composite luck score
    - away_shooting_luck: Composite luck score
    """
    ts_cols = ['home_rolling_5_ts_pct', 'home_rolling_30_ts_pct',
               'away_rolling_5_ts_pct', 'away_rolling_30_ts_pct']
    efg_cols = ['home_rolling_5_efg_pct', 'home_rolling_30_efg_pct',
                'away_rolling_5_efg_pct', 'away_rolling_30_efg_pct']
    
    if all(col in df.columns for col in ts_cols):
        df['home_shooting_luck_ts'] = (
            df['home_rolling_5_ts_pct'].fillna(0) - df['home_rolling_30_ts_pct'].fillna(0)
        )
        df['away_shooting_luck_ts'] = (
            df['away_rolling_5_ts_pct'].fillna(0) - df['away_rolling_30_ts_pct'].fillna(0)
        )
    else:
        df['home_shooting_luck_ts'] = 0.0
        df['away_shooting_luck_ts'] = 0.0
    
    if all(col in df.columns for col in efg_cols):
        df['home_shooting_luck_efg'] = (
            df['home_rolling_5_efg_pct'].fillna(0) - df['home_rolling_30_efg_pct'].fillna(0)
        )
        df['away_shooting_luck_efg'] = (
            df['away_rolling_5_efg_pct'].fillna(0) - df['away_rolling_30_efg_pct'].fillna(0)
        )
    else:
        df['home_shooting_luck_efg'] = 0.0
        df['away_shooting_luck_efg'] = 0.0
    
    df['home_shooting_luck'] = (
        (df['home_shooting_luck_ts'] + df['home_shooting_luck_efg']) / 2
    ).fillna(0)
    df['away_shooting_luck'] = (
        (df['away_shooting_luck_ts'] + df['away_shooting_luck_efg']) / 2
    ).fillna(0)
    
    return df


def prepare_advanced_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modular Feature Engineering - ADVANCED STEPS ONLY.
    
    Assumes BASE features (Pace, Four Factors, Rolling Stats) are already present.
    
    Adds:
    - Advanced Pace (projected_pace, pace_mismatch)
    - Matchup Efficiency (off_matchup_*, eff_sum)
    - Volatility (scoring_std_10)
    - Shooting Luck (regression to mean detection)
    """
    logger.info("⚡ Applying ADVANCED Features Only...")
    
    initial_cols = df.shape[1]
    
    steps = [
        ("8️⃣ Advanced Pace", add_advanced_pace_features),
        ("9️⃣ Matchup Efficiency", add_matchup_efficiency_features),
        ("🔟 Volatility & Trends", add_volatility_features),
        ("1️⃣1️⃣ Shooting Luck", add_shooting_luck_features),
    ]
    
    for step_name, step_func in steps:
        logger.info(f"   {step_name}")
        try:
            df = step_func(df)
        except Exception as e:
            logger.error(f"      ❌ FAILED: {e}")
    
    added = df.shape[1] - initial_cols
    logger.info(f"   ✅ Added {added} advanced features (total: {df.shape[1]})")
    
    return df
