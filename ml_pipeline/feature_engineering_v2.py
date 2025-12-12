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
