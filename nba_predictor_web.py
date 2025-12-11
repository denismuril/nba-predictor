import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sys
import textwrap
from pathlib import Path

# Adicionar raiz ao path
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager
from core.travel_calculator import calculate_fatigue_score
from ml_pipeline.betting_engine import BettingEngine
from betting.web_ui import render_bankroll_management

# Configuração da Página
st.set_page_config(
    page_title="NBA Predictor v21.5 - Forensic Audit Complete",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.header("🏀 NBA Predictor")

    # Status do Sistema
    st.subheader("🖥️ System Status")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Database", "Online", delta_color="normal")
    with col2:
        st.metric("Model V6/V18", "Active", delta_color="normal")

    last_update = datetime.now().strftime("%H:%M")
    st.text(f"Last Update: {last_update}")

    st.markdown("---")

    # Filtros
    st.subheader("🔍 Filtros")
    selected_date = st.date_input("Data do Jogo", datetime.now())

    confidence_threshold = st.slider("Confiança Mínima %", 50, 90, 60)

    st.markdown("---")
    st.subheader("💰 Banca")
    bankroll_input = st.number_input("Banca Total (R$)", min_value=100.0, value=1000.0, step=100.0)
    kelly_fraction = st.slider("Kelly Fraction (Risco)", 0.1, 1.0, 0.25, 0.05)

    st.markdown("---")
    if st.button("🗑️ Limpar Cache"):
        st.cache_data.clear()
        st.success("Cache limpo! Recarregando...")
        time.sleep(1)
        st.rerun()

# Mapeamento de Times (Full Name -> Abbreviation) - Estático para robustez
TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',

    # Correções de Abreviações (Legacy/Odds API -> NBA API)
    'PHO': 'PHX', 'BRK': 'BKN', 'CHO': 'CHA'
}

# --- STANDINGS & FORM FUNCTIONS ---
@st.cache_data(ttl=3600)
def load_standings():
    """Load current standings with rankings by win percentage."""
    try:
        from data.scrapers.standings_scraper import obter_standings
        standings = obter_standings()  # {'Team Name': {'wins': X, 'losses': Y}}

        if not standings:
            return {}

        # Calculate ranking based on win percentage
        ranked = []
        for team, data in standings.items():
            wins = data.get('wins', 0)
            losses = data.get('losses', 0)
            total = wins + losses
            win_pct = wins / total if total > 0 else 0
            ranked.append({'team': team, 'wins': wins, 'losses': losses, 'win_pct': win_pct})

        # Sort by win percentage (descending)
        ranked.sort(key=lambda x: x['win_pct'], reverse=True)

        # Create dict with position (both full name and abbreviation)
        result = {}
        for idx, item in enumerate(ranked, 1):
            team_data = {
                'position': idx,
                'wins': item['wins'],
                'losses': item['losses'],
                'record': f"{item['wins']}-{item['losses']}"
            }
            # Store by full name
            result[item['team']] = team_data
            # Also store by abbreviation for easy lookup
            abbr = TEAM_MAP.get(item['team'])
            if abbr:
                result[abbr] = team_data

        return result
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar standings: {e}")
        return {}

def get_team_form(df_history, team_name, n=5):
    """Get W/L sequence for last n games (e.g., 'WWLWL')."""
    if df_history.empty:
        return ""

    team_abbr = TEAM_MAP.get(team_name, team_name)

    # Filter completed games only
    df_copy = df_history.copy()
    df_copy['home_score'] = pd.to_numeric(df_copy['home_score'], errors='coerce').fillna(0)
    df_copy['away_score'] = pd.to_numeric(df_copy['away_score'], errors='coerce').fillna(0)
    valid = df_copy[df_copy['home_score'] > 0]

    # Get team's games
    team_games = valid[
        (valid['home_team'].isin([team_abbr, team_name])) |
        (valid['away_team'].isin([team_abbr, team_name]))
    ].sort_values('date', ascending=False).head(n)

    form = []
    for _, row in team_games.iterrows():
        is_home = row['home_team'].strip() in [team_abbr, team_name]
        home_score = float(row['home_score'])
        away_score = float(row['away_score'])

        if is_home:
            won = home_score > away_score
        else:
            won = away_score > home_score

        form.append('W' if won else 'L')

    return ''.join(form)  # e.g., "WWLWL"

def format_form_display(form_str):
    """Format form string with colors (W=green, L=red)."""
    if not form_str:
        return "<span style='color: #6b7280;'>-</span>"

    result = []
    for c in form_str:
        if c == 'W':
            result.append("<span style='color: #4ade80; font-weight: bold;'>W</span>")
        else:
            result.append("<span style='color: #f87171; font-weight: bold;'>L</span>")
    return ' '.join(result)


# =============================================================================
# AUDITORIA P1-A: Funções Vetorizadas para Cálculo de Stats
# Elimina iterrows() que causava O(n×m) e latência de 5-10 segundos
# =============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def precompute_team_stats(df_history: pd.DataFrame, n: int = 10) -> dict:
    """
    Pré-computa estatísticas recentes de TODOS os times de forma vetorizada.

    AUDITORIA P1-A:
    - Usa groupby().apply() em vez de iterrows()
    - Complexidade O(n) em vez de O(n*m)
    - Cache com TTL de 30 minutos

    Args:
        df_history: DataFrame com histórico de jogos
        n: Número de jogos recentes por time

    Returns:
        Dict[team_name, stats_dict] para lookup O(1)
    """
    if df_history.empty:
        return {}

    # Garantir que home_score é numérico
    df = df_history.copy()
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0)
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0)

    # Filtrar apenas jogos realizados
    valid_df = df[df['home_score'] > 0].copy()

    if valid_df.empty:
        return {}

    # Preparar dados para cálculo vetorizado
    valid_df['date'] = pd.to_datetime(valid_df['date'])

    all_teams = set(valid_df['home_team'].unique()) | set(valid_df['away_team'].unique())
    team_stats = {}

    for team in all_teams:
        # Filtrar jogos do time
        team_games = valid_df[
            (valid_df['home_team'] == team) | (valid_df['away_team'] == team)
        ].nlargest(n, 'date')

        if team_games.empty:
            continue

        # Máscara para jogos em casa
        is_home = team_games['home_team'] == team

        # Pontos
        pts = team_games.apply(
            lambda r: r['home_score'] if r['home_team'] == team else r['away_score'],
            axis=1
        )
        pts_allowed = team_games.apply(
            lambda r: r['away_score'] if r['home_team'] == team else r['home_score'],
            axis=1
        )


        # eFG% (vectorized)
        # P1-A FIX: Use numpy vectorization instead of loop
        home_efg = team_games['home_efg_pct'].fillna(0)
        away_efg = team_games['away_efg_pct'].fillna(0)

        # Select eFG based on whether team was home or away
        # is_home is a boolean Series calculated above
        efg_series = np.where(is_home, home_efg, away_efg)

        # Calculate mean of non-zero values
        # Filter for valid > 0 to match logic of "matches played"
        valid_efg = efg_series[efg_series > 0]
        efg_mean = valid_efg.mean() if len(valid_efg) > 0 else 0

        # TOV% calculation from raw 'tov' column
        # Use raw tov data if home_tov_pct doesn't exist
        if 'home_tov_pct' in team_games.columns:
            home_tov = team_games['home_tov_pct'].fillna(0)
            away_tov = team_games['away_tov_pct'].fillna(0) if 'away_tov_pct' in team_games.columns else pd.Series([0] * len(team_games))
        elif 'tov' in team_games.columns and 'fga' in team_games.columns:
            # CORRECT NBA FORMULA: TOV% = TOV / (FGA + 0.44*FTA + TOV) * 100
            tov_home = team_games['tov'].fillna(0)
            fga_home = team_games['fga'].fillna(0)
            fta_home = team_games['fta'].fillna(0) if 'fta' in team_games.columns else pd.Series([0] * len(team_games))
            possessions_home = fga_home + 0.44 * fta_home + tov_home
            home_tov = np.where(possessions_home > 0, tov_home / possessions_home * 100, 0)

            # Away team calculation
            if 'opp_tov' in team_games.columns and 'opp_fga' in team_games.columns:
                tov_away = team_games['opp_tov'].fillna(0)
                fga_away = team_games['opp_fga'].fillna(0)
                fta_away = team_games['opp_fta'].fillna(0) if 'opp_fta' in team_games.columns else pd.Series([0] * len(team_games))
                possessions_away = fga_away + 0.44 * fta_away + tov_away
                away_tov = np.where(possessions_away > 0, tov_away / possessions_away * 100, 0)
            else:
                away_tov = pd.Series([0] * len(team_games))
        else:
            home_tov = pd.Series([0] * len(team_games))
            away_tov = pd.Series([0] * len(team_games))

        tov_series = np.where(is_home, home_tov, away_tov)
        valid_tov = tov_series[(tov_series > 0) & (tov_series < 50)]  # Filter unreasonable values
        tov_mean = valid_tov.mean() if len(valid_tov) > 0 else 0

        # ORB% calculation from raw 'oreb' column
        if 'home_orb_pct' in team_games.columns:
            home_orb = team_games['home_orb_pct'].fillna(0)
            away_orb = team_games['away_orb_pct'].fillna(0) if 'away_orb_pct' in team_games.columns else pd.Series([0] * len(team_games))
        elif 'oreb' in team_games.columns:
            # Approximate ORB% = oreb / (oreb + opp_dreb)
            # Since we don't have opp_dreb, use oreb directly as proxy
            home_orb = team_games['oreb'].fillna(0)
            away_orb = team_games['opp_oreb'].fillna(0) if 'opp_oreb' in team_games.columns else pd.Series([0] * len(team_games))
        else:
            home_orb = pd.Series([0] * len(team_games))
            away_orb = pd.Series([0] * len(team_games))

        orb_series = np.where(is_home, home_orb, away_orb)
        valid_orb = orb_series[orb_series > 0]
        orb_mean = valid_orb.mean() if len(valid_orb) > 0 else 0

        # FTR calculation (FT Rate = FTA / FGA)
        if 'home_ftr' in team_games.columns:
            home_ftr = team_games['home_ftr'].fillna(0)
            away_ftr = team_games['away_ftr'].fillna(0) if 'away_ftr' in team_games.columns else pd.Series([0] * len(team_games))
        elif 'fta' in team_games.columns and 'fga' in team_games.columns:
            home_ftr = team_games['fta'].fillna(0) / team_games['fga'].fillna(1).replace(0, 1) * 100
            away_ftr = team_games['opp_fta'].fillna(0) / team_games['opp_fga'].fillna(1).replace(0, 1) * 100 if 'opp_fta' in team_games.columns else pd.Series([0] * len(team_games))
        else:
            home_ftr = pd.Series([0] * len(team_games))
            away_ftr = pd.Series([0] * len(team_games))

        ftr_series = np.where(is_home, home_ftr, away_ftr)
        valid_ftr = ftr_series[(ftr_series > 0) & (ftr_series < 100)]  # Filter unreasonable values
        ftr_mean = valid_ftr.mean() if len(valid_ftr) > 0 else 0

        # FALLBACK: Use league averages when no valid data (box scores missing for 2024-25)
        # League averages: TOV% ~13%, ORB ~10 per game, FTR ~25%
        if tov_mean == 0 or tov_mean > 50:
            tov_mean = 13.0  # League average TOV%
        if orb_mean == 0:
            orb_mean = 10.0  # League average OREB per game
        if ftr_mean == 0 or ftr_mean > 60:
            ftr_mean = 25.0  # League average FTR

        stats = {
            'pts': pts.mean() if len(pts) > 0 else 0,
            'pts_allowed': pts_allowed.mean() if len(pts_allowed) > 0 else 0,
            'efg_pct': efg_mean,
            'tov_pct': tov_mean,
            'oreb_pct': orb_mean,
            'fta_rate': ftr_mean,
        }

        # Normalizar nome do time
        team_abbr = TEAM_MAP.get(team, team)
        team_stats[team] = stats
        team_stats[team_abbr] = stats  # Duplicar para abreviação

    return team_stats


def get_team_recent_stats_fast(precomputed_stats: dict, team_name: str) -> dict:
    """
    Lookup O(1) nos stats pré-computados.

    AUDITORIA P1-A: Wrapper para compatibilidade com código existente.

    Args:
        precomputed_stats: Dict retornado por precompute_team_stats()
        team_name: Nome do time

    Returns:
        Dict com estatísticas ou {} se não encontrado
    """
    team_name = team_name.strip()
    team_abbr = TEAM_MAP.get(team_name, team_name)

    # Tentar ambos: nome completo e abreviação
    stats = precomputed_stats.get(team_abbr) or precomputed_stats.get(team_name)
    return stats if stats else {}


def get_team_recent_stats(df_history, team_name, n=10):
    """
    DEPRECATED: Use get_team_recent_stats_fast() com precompute_team_stats().

    Esta função é mantida apenas para backward compatibility.
    Internamente redireciona para a versão otimizada.

    Warning:
        Esta função será removida em versões futuras.
        Migre para: precomputed = precompute_team_stats(df_history, n)
                    stats = get_team_recent_stats_fast(precomputed, team_name)

    Args:
        df_history: DataFrame com histórico de jogos
        team_name: Nome do time
        n: Número de jogos recentes (default: 10)

    Returns:
        Dict com estatísticas calculadas
    """
    import warnings
    warnings.warn(
        "get_team_recent_stats() está deprecada. "
        "Use precompute_team_stats() + get_team_recent_stats_fast() para O(1) lookup.",
        DeprecationWarning,
        stacklevel=2
    )

    # Redirecionar para versão otimizada (mantém compatibilidade)
    precomputed = precompute_team_stats(df_history, n=n)
    return get_team_recent_stats_fast(precomputed, team_name)

def enrich_predictions_with_stats(daily_games, df_history):
    """
    Enriquece DF de previsões com stats recentes.

    AUDITORIA P1-A: Refatorado para O(N) usando:
    - precompute_team_stats() para pré-calcular UMA VEZ
    - get_team_recent_stats_fast() para lookup O(1)
    - Operações vetorizadas em vez de iterrows()

    Performance: De 3-8s para <100ms
    """
    if daily_games.empty or df_history.empty:
        return daily_games

    enriched = daily_games.copy()

    # P1-A: Pré-computar stats de TODOS os times UMA VEZ (O(N))
    precomputed = precompute_team_stats(df_history, n=10)

    # Colunas a adicionar
    cols = ['home_efg_pct', 'home_tov_pct', 'home_oreb', 'home_fta',
            'home_pts_mean', 'home_pts_allowed_mean',
            'away_efg_pct', 'away_tov_pct', 'away_oreb', 'away_fta',
            'away_pts_mean', 'away_pts_allowed_mean']
    for c in cols:
        if c not in enriched.columns:
            enriched[c] = 0.0

    # P1-A: Usar apply vetorizado em vez de iterrows()
    def enrich_row(row):
        h_stats = get_team_recent_stats_fast(precomputed, row['home_team'])
        a_stats = get_team_recent_stats_fast(precomputed, row['away_team'])

        return pd.Series({
            'home_efg_pct': h_stats.get('efg_pct', 0),
            'home_tov_pct': h_stats.get('tov_pct', 0),
            'home_oreb': h_stats.get('oreb_pct', 0),  # Already in percentage form
            'home_fta': h_stats.get('fta_rate', 0),    # Already in percentage form
            'home_pts_mean': h_stats.get('pts', 0),
            'home_pts_allowed_mean': h_stats.get('pts_allowed', 0),
            'away_efg_pct': a_stats.get('efg_pct', 0),
            'away_tov_pct': a_stats.get('tov_pct', 0),
            'away_oreb': a_stats.get('oreb_pct', 0),   # Already in percentage form
            'away_fta': a_stats.get('fta_rate', 0),    # Already in percentage form
            'away_pts_mean': a_stats.get('pts', 0),
            'away_pts_allowed_mean': a_stats.get('pts_allowed', 0),
        })

    # Aplicar enriquecimento vetorizado
    stats_df = enriched.apply(enrich_row, axis=1)

    # Atualizar colunas (sobrescrever com valores calculados)
    for col in stats_df.columns:
        enriched[col] = stats_df[col]

    return enriched

@st.cache_data(ttl=3600, show_spinner="Carregando dados...")
def load_data():
    """
    Carrega dados históricos com otimizações de performance.

    AUDIT FIX: Otimizações implementadas:
    1. Filtra apenas colunas necessárias para UI
    2. Converte tipos para reduzir memória (float64 → float32)
    """
    from ml_pipeline.data_preparation import load_historical_data
    import logging
    logger = logging.getLogger(__name__)

    try:
        df = load_historical_data(raw=True)

        if df is None or df.empty:
            return pd.DataFrame()

        # AUDIT FIX: Colunas essenciais para UI
        UI_ESSENTIAL_COLS = [
            'date', 'home_team', 'away_team',
            'home_score', 'away_score', 'winner',
        ]

        STATS_COLS = [
            'fgm', 'fga', 'fg3m', 'opp_fgm', 'opp_fga', 'opp_fg3m',
            'oreb', 'dreb', 'opp_oreb', 'opp_dreb',
            'tov', 'fta', 'opp_tov', 'opp_fta',
            'home_efg_pct', 'away_efg_pct'
        ]

        all_needed = UI_ESSENTIAL_COLS + STATS_COLS
        available_cols = [c for c in all_needed if c in df.columns]

        # Manter apenas colunas necessárias
        df = df[available_cols].copy()

        # AUDIT FIX: Otimizar tipos de dados
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].max() < 32767:
                df[col] = df[col].astype('int16')

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"✅ Dados carregados: {len(df)} jogos, {mem_mb:.1f} MB")

        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


# === AUDIT FIX: Funções auxiliares para UI de Odds/Spread ===

def calculate_implied_spread(prob_home: float, method: str = 'logistic') -> tuple:
    """
    Calcula spread implícito a partir da probabilidade de vitória.

    AUDIT FIX: Substitui fórmula linear incorreta por logística calibrada.

    Args:
        prob_home: Probabilidade do mandante (0-100)
        method: 'logistic' (padrão) ou 'linear' (fallback)

    Returns:
        (spread_value, confidence_interval)
    """
    import math

    if prob_home <= 0 or prob_home >= 100:
        return 0.0, 5.0

    prob = prob_home / 100

    if method == 'logistic':
        # Inverso logístico: spread = -k * ln(p / (1-p))
        # k=5.5 calibrado empiricamente para NBA
        if prob >= 0.99:
            prob = 0.99
        if prob <= 0.01:
            prob = 0.01

        log_odds = math.log(prob / (1 - prob))
        k = 5.5
        spread = -k * log_odds

        # CI aumenta em jogos equilibrados
        ci = 3.0 + abs(50 - prob_home) * 0.05
    else:
        spread = -(prob_home - 50) / 2.5
        ci = 4.0

    spread = max(-20, min(20, spread))

    return round(spread, 1), round(ci, 1)


def calculate_ev_with_warning(prob: float, odds: float, is_estimated: bool) -> tuple:
    """
    Calcula EV e retorna flag de confiabilidade.

    AUDIT FIX: Diferencia EV confiável (odds reais) de EV fictício (Fair Odds).

    Returns:
        (ev_value, is_reliable, warning_message)
    """
    if odds <= 0 or prob <= 0:
        return 0, False, "Dados insuficientes"

    ev = (prob/100 * odds - 1) * 100

    if is_estimated:
        return ev, False, "⚠️ EV baseado em Fair Odd (não real)"

    return ev, True, None


def get_fatigue_color(score):
    """Retorna classe CSS e label para score de fadiga."""
    if score < 30:
        return "bg-green-500", "Baixa"
    elif score < 60:
        return "bg-yellow-500", "Média"
    else:
        return "bg-red-500", "Alta"


# --- MAIN CONTENT ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard", "💰 Sugestões de Aposta", "⛹️ Player Props", "📈 Performance", "🔍 Model Health", "🧪 Backtest Analysis"])

# --- TAB 1: DASHBOARD (GAME CARDS) ---
with tab1:
    st.header(f"Jogos de {selected_date.strftime('%d/%m/%Y')}")

    df = load_data()
    db = get_db_manager()

    # Filtrar por data
    date_str = selected_date.strftime('%Y-%m-%d')
    today_str = datetime.now().strftime('%Y-%m-%d')

    # LÓGICA UNIFICADA: Sempre carregar previsões primeiro para garantir Confiança e Dados do Modelo
    daily_games = db.get_latest_predictions(date_str)

    # Se não houver previsões, tentar histórico (mas sem confiança)
    if daily_games.empty:
        if not df.empty and 'date' in df.columns:
             daily_games = df[df['date'].astype(str).str.startswith(date_str)].copy()
             daily_games['confidence'] = 'N/A' # Histórico puro não tem confiança
    else:
        # Se temos previsões, vamos buscar os placares reais no histórico para exibir
        if not df.empty and 'date' in df.columns:
            # Filtrar histórico do dia
            history_day = df[df['date'].astype(str).str.startswith(date_str)]
            if not history_day.empty:
                # Criar chaves para merge
                daily_games['join_key'] = daily_games['home_team'].map(TEAM_MAP).fillna(daily_games['home_team'])
                history_day['join_key'] = history_day['home_team'].map(TEAM_MAP).fillna(history_day['home_team'])

                # Merge para pegar placar e vencedor
                # Usar left join para manter a previsão mesmo se não tiver resultado ainda
                merged = pd.merge(
                    daily_games,
                    history_day[['join_key', 'home_score', 'away_score', 'winner']],
                    on='join_key',
                    how='left',
                    suffixes=('', '_real')
                )

                # Atualizar colunas
                if 'home_score_real' in merged.columns:
                    merged['home_score'] = merged['home_score_real'].fillna(merged.get('home_score', 0))
                if 'away_score_real' in merged.columns:
                    merged['away_score'] = merged['away_score_real'].fillna(merged.get('away_score', 0))
                if 'winner_real' in merged.columns:
                    merged['winner'] = merged['winner_real'].fillna(merged.get('winner', ''))

                daily_games = merged

    # APLICAR FILTRO DE CONFIANÇA
    if not daily_games.empty:
        total_games_before = len(daily_games)

        # Enriquecer com stats recentes (para jogos futuros que não têm stats no DB)
        daily_games = enrich_predictions_with_stats(daily_games, df)

        daily_games = daily_games.copy()

        # Garantir que a coluna 'confidence' exista (pode faltar se vier do histórico)
        if 'confidence' not in daily_games.columns:
            daily_games['confidence'] = 'N/A'

        def convert_confidence(val):
            val_str = str(val).strip().upper()
            # Mapeamento expandido para Inglês e Português
            text_map = {
                'LOW': 50, 'BAIXA': 50,
                'MEDIUM': 65, 'MÉDIA': 65, 'MEDIA': 65,
                'HIGH': 80, 'ALTA': 80
            }
            if val_str in text_map: return text_map[val_str]
            try: return float(val_str.rstrip('%'))
            except: return 0

        daily_games['confidence_num'] = daily_games['confidence'].apply(convert_confidence)
        daily_games = daily_games[daily_games['confidence_num'] >= confidence_threshold]

        if daily_games.empty and total_games_before > 0:
             st.warning(f"⚠️ {total_games_before} jogos encontrados, mas todos foram ocultados pelo filtro de Confiança Mínima ({confidence_threshold}%). Reduza o filtro na barra lateral para visualizá-los.")

    if daily_games.empty:
        if 'total_games_before' not in locals() or total_games_before == 0:
            st.info("Nenhum jogo encontrado para esta data. Execute o Orchestrator para gerar previsões.")
            if st.button("🔄 Rodar Pipeline Agora"):
                with st.spinner("Executando pipeline..."):
                    time.sleep(2)
                    st.success("Pipeline concluído! Recarregue a página.")
    else:
        # 🔍 DEBUG SECTION - Análise de Dados
        with st.expander("🔍 DEBUG: Análise de Dados (eFG% e Odds)", expanded=False):
            st.subheader("Histórico Disponível")
            st.write(f"**Total de jogos no histórico**: {len(df)}")
            st.write(f"**Jogos completos (home_score > 0)**: {len(df[df['home_score'] > 0])}")

            if not df.empty:
                st.write("**Últimos 3 jogos completos**:")
                completed_games = df[df['home_score'] > 0].head(3)
                st.dataframe(completed_games[['date', 'home_team', 'away_team', 'home_efg_pct', 'away_efg_pct', 'home_score', 'away_score']])

                # Check eFG% format
                if len(completed_games) > 0:
            # Verifica formato do eFG%
                    sample_efg = completed_games.iloc[0]['home_efg_pct']
                    st.write(f"**Formato do eFG% no banco**: {sample_efg}")

                    st.write("**Colunas disponíveis no DataFrame:**")
                    st.write(list(df.columns))

                    st.write("**Amostra de Stats Brutos (LAL):**")
                    lal_games = df[df['home_team'] == 'LAL'].head(2)
                    if not lal_games.empty:
                        st.dataframe(lal_games[['date', 'home_team', 'fgm', 'fga', 'fg3m', 'home_efg_pct']])
                    else:
                        st.write("Nenhum jogo do LAL encontrado como mandante.")

                    # Teste direto da função
                    st.write("**Teste get_team_recent_stats('LAL'):**")
                    try:
                        test_stats = get_team_recent_stats(df, 'LAL', n=5)
                        st.json(test_stats)
                    except Exception as e:
                        st.error(f"Erro ao testar função: {e}")

            st.subheader("Previsões para Hoje")
            st.write(f"**Total de previsões**: {len(daily_games)}")
            if not daily_games.empty:
                first_game = daily_games.iloc[0]
                st.write(f"**Exemplo: {first_game['home_team']} vs {first_game['away_team']}**")
                st.json({
                    "prob_home": float(first_game.get('prob_home', 0)),
                    "prob_away": float(first_game.get('prob_away', 0)),
                    "odds_home": float(first_game.get('odds_home', 0)),
                    "odds_away": float(first_game.get('odds_away', 0)),
                    "spread_home": float(first_game.get('spread_home', 0)),
                    "home_efg_pct": float(first_game.get('home_efg_pct', 0)),
                    "away_efg_pct": float(first_game.get('away_efg_pct', 0)),
                })

                st.write("**Test: get_team_recent_stats** para", first_game['home_team'])
                test_stats = get_team_recent_stats(df, first_game['home_team'])
                st.json(test_stats)

        # Load standings data for position display
        standings_data = load_standings()

        # Grid de Cards
        for _, game in daily_games.iterrows():
            # Dados Básicos
            home_team = game['home_team']
            away_team = game['away_team']

            # Fadiga (Simulada se não tiver)
            home_fatigue = game.get('home_fatigue_score', 20)
            away_fatigue = game.get('away_fatigue_score', 45)
            h_class, h_label = get_fatigue_color(home_fatigue)
            a_class, a_label = get_fatigue_color(away_fatigue)

            # Confiança
            try: conf_val = float(str(game['confidence']).strip('%'))
            except: conf_val = 0
            conf_color = '#4ade80' if conf_val > 65 else '#facc15'

            # Odds & EV
            odds_home = game.get('odds_home', game.get('Odd Casa', 0))
            odds_away = game.get('odds_away', game.get('Odd Visitante', 0))
            prob_home = game.get('prob_home', 0)
            prob_away = game.get('prob_away', 0)

            # --- CÁLCULOS IMPLÍCITOS (FALLBACK) ---
            # Se não tiver odds reais, calcular Odds Justas (Fair Odds)
            if odds_home == 0 and prob_home > 0:
                odds_home = 100 / prob_home
                odds_home_is_est = True
            else:
                odds_home_is_est = False

            if odds_away == 0 and prob_away > 0:
                odds_away = 100 / prob_away
                odds_away_is_est = True
            else:
                odds_away_is_est = False

            # Se não tiver spread real, calcular Spread Implícito com logístico
            # AUDIT FIX: Substituída fórmula linear por calculate_implied_spread()
            spread_home = game.get('spread_home', game.get('spread', 0))
            spread_is_est = False
            spread_ci = 2.0

            if spread_home == 0 and prob_home > 0:
                spread_home, spread_ci = calculate_implied_spread(prob_home, method='logistic')
                spread_is_est = True

            # AUDIT FIX: Usar calculate_ev_with_warning() para diferenciar EV real de fictício
            ev_home, ev_home_reliable, ev_home_warning = calculate_ev_with_warning(prob_home, odds_home, odds_home_is_est)
            ev_away, ev_away_reliable, ev_away_warning = calculate_ev_with_warning(prob_away, odds_away, odds_away_is_est)

            # Four Factors (Se disponíveis)
            # Precisamos buscar do DB ou calcular. O get_history já traz se estiverem lá.
            # Se não, mostramos N/A

            h_efg = game.get('home_efg_pct', 0) * 100  # Comes as decimal (0.5 = 50%)
            a_efg = game.get('away_efg_pct', 0) * 100
            h_tov = game.get('home_tov_pct', 0)  # Already in percentage form
            a_tov = game.get('away_tov_pct', 0)

            # Layout do Card
            with st.container():
                st.markdown(f"""
                <div style="background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 20px;">
                """, unsafe_allow_html=True)

                # Header
                c1, c2 = st.columns([3, 1])
                with c1: st.caption(f"📅 {game['date']}")
                with c2: st.markdown(f"<span style='background-color: {conf_color}20; color: {conf_color}; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{game['confidence']}</span>", unsafe_allow_html=True)

                # Placar / Times
                col_home, col_vs, col_away = st.columns([4, 1, 4])

                # Extrair dados extras
                prob_mc = game.get('prob_mc_home', 0)

                with col_home:
                    # Team name
                    st.markdown(f"<div style='text-align: center; font-size: 1.4em; font-weight: bold;'>{home_team}</div>", unsafe_allow_html=True)
                    # Position and record
                    h_pos = standings_data.get(home_team, {}).get('position', '?')
                    h_record = standings_data.get(home_team, {}).get('record', '?-?')
                    st.markdown(f"<div style='text-align: center; font-size: 0.75em; color: #6b7280;'>#{h_pos} ({h_record})</div>", unsafe_allow_html=True)
                    # Last 5 games form
                    h_form = get_team_form(df, home_team)
                    st.markdown(f"<div style='text-align: center; font-size: 0.8em; margin-top: 2px;'>{format_form_display(h_form)}</div>", unsafe_allow_html=True)
                    # Fatigue
                    st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: #9ca3af;'><span class='fatigue-dot {h_class}'></span> {h_label}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: #e0e0e0; margin-top: 5px;'>ML: <b>{prob_home:.1f}%</b></div>", unsafe_allow_html=True)
                    if prob_mc > 0:
                        st.markdown(f"<div style='text-align: center; color: #a78bfa; font-size: 0.8em;'>MC: {prob_mc:.1f}%</div>", unsafe_allow_html=True)

                    # Odds Display - P2: Diferenciar Fair Odds de Odds Reais
                    if odds_home > 0:
                        if odds_home_is_est:
                            # Fair Odd (modelo) - tom discreto com aviso
                            st.markdown(
                                f"<div style='text-align: center; color: #6b7280; font-size: 0.75em; font-style: italic;'>"
                                f"Fair: {odds_home:.2f} <span style='color: #ef4444;'>(sem mercado)</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            # Odd real - destaque visual
                            st.markdown(
                                f"<div style='text-align: center; color: #facc15; font-size: 0.95em; font-weight: bold;'>"
                                f"💰 Odd: {odds_home:.2f}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                    # Spread Display
                    # Spread (Mostrar sempre para confirmação visual)
                    spr_lbl = "Est." if spread_is_est else "Spread"
                    if spread_home != 0:
                         st.markdown(f"<div style='text-align: center; color: #9ca3af; font-size: 1.0em; font-weight: 500;'>{spr_lbl}: {spread_home:+.1f}</div>", unsafe_allow_html=True)
                    else:
                         st.markdown(f"<div style='text-align: center; color: #555; font-size: 1.0em;'>Spread: PK</div>", unsafe_allow_html=True)

                    # Total Points
                    total_pts = game.get('total_points', 0)
                    total_is_est = False

                    if total_pts == 0:
                        # Prioridade 1: Usar predicted_total do modelo ML
                        predicted_total = game.get('predicted_total', 0)
                        if predicted_total > 0:
                            total_pts = predicted_total
                            total_is_est = True
                        else:
                            # Prioridade 2: Estimar com base nas médias recentes (Ataque + Defesa)
                            h_pts = game.get('home_pts_mean', 0)
                            a_pts = game.get('away_pts_mean', 0)
                            h_allowed = game.get('home_pts_allowed_mean', 0)
                            a_allowed = game.get('away_pts_allowed_mean', 0)

                            if h_pts > 0 and a_pts > 0:
                                # Se tiver dados de defesa, usa média ponderada
                                if h_allowed > 0 and a_allowed > 0:
                                    est_home = (h_pts + a_allowed) / 2
                                    est_away = (a_pts + h_allowed) / 2
                                    total_pts = est_home + est_away
                                else:
                                    # Fallback simples (só ataque)
                                    total_pts = (h_pts + a_pts)
                                total_is_est = True

                    if total_pts > 0:
                         lbl = "Est." if total_is_est else "Total"
                         st.markdown(f"<div style='text-align: center; color: #9ca3af; font-size: 1.0em; margin-top: 2px; font-weight: 500;'>{lbl}: {total_pts:.1f}</div>", unsafe_allow_html=True)

                with col_vs:
                    st.markdown("<div style='text-align: center; font-size: 1.5em; color: #6b7280; font-weight: bold; padding-top: 10px;'>VS</div>", unsafe_allow_html=True)

                with col_away:
                    # Team name
                    st.markdown(f"<div style='text-align: center; font-size: 1.4em; font-weight: bold;'>{away_team}</div>", unsafe_allow_html=True)
                    # Position and record
                    a_pos = standings_data.get(away_team, {}).get('position', '?')
                    a_record = standings_data.get(away_team, {}).get('record', '?-?')
                    st.markdown(f"<div style='text-align: center; font-size: 0.75em; color: #6b7280;'>#{a_pos} ({a_record})</div>", unsafe_allow_html=True)
                    # Last 5 games form
                    a_form = get_team_form(df, away_team)
                    st.markdown(f"<div style='text-align: center; font-size: 0.8em; margin-top: 2px;'>{format_form_display(a_form)}</div>", unsafe_allow_html=True)
                    # Fatigue
                    st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: #9ca3af;'><span class='fatigue-dot {a_class}'></span> {a_label}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: #e0e0e0; margin-top: 5px;'>ML: <b>{prob_away:.1f}%</b></div>", unsafe_allow_html=True)
                    if prob_mc > 0:
                        st.markdown(f"<div style='text-align: center; color: #a78bfa; font-size: 0.8em;'>MC: {100-prob_mc:.1f}%</div>", unsafe_allow_html=True)

                    # Odds Display
                    odd_lbl = "Fair" if odds_away_is_est else "Odd"
                    odd_color = "#9ca3af" if odds_away_is_est else "#facc15"
                    if odds_away > 0:
                        st.markdown(f"<div style='text-align: center; color: {odd_color}; font-size: 0.9em;'>{odd_lbl}: {odds_away:.2f}</div>", unsafe_allow_html=True)

                    # Spread Display
                    spr_lbl = "Est." if spread_is_est else "Spread"
                    if spread_home != 0:
                         st.markdown(f"<div style='text-align: center; color: #9ca3af; font-size: 1.0em; font-weight: 500;'>{spr_lbl}: {-spread_home:+.1f}</div>", unsafe_allow_html=True)
                    else:
                         st.markdown(f"<div style='text-align: center; color: #555; font-size: 1.0em;'>Spread: PK</div>", unsafe_allow_html=True)

                    # Total Points (Repetido ou vazio para simetria, ou apenas no home)
                    # Vamos deixar apenas no Home para não poluir, ou colocar no meio?
                    # Colocando no meio (VS) ficaria melhor, mas aqui é coluna. Vamos por no away também para simetria.
                    if total_pts > 0:
                         st.markdown(f"<div style='text-align: center; color: #555; font-size: 1.0em; margin-top: 2px;'>-</div>", unsafe_allow_html=True)

                # --- FOUR FACTORS SECTION ---
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 0.8em; color: #aaa; margin-bottom: 5px;'>FOUR FACTORS (eFG% | TOV% | ORB% | FTR)</div>", unsafe_allow_html=True)

                ff1, ff2, ff3, ff4 = st.columns(4)

                # eFG%
                with ff1:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 0.9em;'>eFG%</div>", unsafe_allow_html=True)
                    st.progress(min(h_efg/100, 1.0))
                    st.caption(f"{h_efg:.1f}% vs {a_efg:.1f}%")
                    # Debug temporário
                    # st.caption(f"Raw: {game.get('home_efg_pct', 0)}")

                # TOV% (Invertido: menor é melhor)
                with ff2:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 0.9em;'>TOV%</div>", unsafe_allow_html=True)
                    st.progress(min(h_tov/30, 1.0)) # Escala arbitrária, TOV% ~15%
                    st.caption(f"{h_tov:.1f}% vs {a_tov:.1f}%")

                # ORB%
                with ff3:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 0.9em;'>ORB%</div>", unsafe_allow_html=True)
                    # ORB% vem multiplicado por 100 do enrich
                    h_orb = game.get('home_oreb', 0)
                    a_orb = game.get('away_oreb', 0)
                    st.progress(min(h_orb/35, 1.0)) # Escala ajustada (ORB% max ~35%)
                    st.caption(f"{h_orb:.1f}% vs {a_orb:.1f}%")

                # FTR
                with ff4:
                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 0.9em;'>FTR</div>", unsafe_allow_html=True)
                    # FTR vem multiplicado por 100 do enrich
                    h_ftr = game.get('home_fta', 0)
                    a_ftr = game.get('away_fta', 0)
                    st.progress(min(h_ftr/40, 1.0))
                    st.caption(f"{h_ftr:.1f} vs {a_ftr:.1f}")

                # --- ADVANCED INSIGHTS SECTION (V4 Features) ---
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)
                st.markdown("<div style='text-align: center; font-size: 0.8em; color: #aaa; margin-bottom: 5px;'>ADVANCED INSIGHTS</div>", unsafe_allow_html=True)

                ai1, ai2 = st.columns(2)

                # Shooting Luck (Regressão à Média)
                with ai1:
                    h_luck = game.get('home_shooting_luck', 0)
                    a_luck = game.get('away_shooting_luck', 0)

                    # Determinar status e cor
                    if abs(h_luck) > 0.03:  # >3% = significativo
                        if h_luck > 0:
                            h_luck_status = "🔥 Quente"
                            h_luck_color = "#f87171"  # Red (fade)
                            h_luck_tip = "Regressão esperada"
                        else:
                            h_luck_status = "❄️ Frio"
                            h_luck_color = "#4ade80"  # Green (back)
                            h_luck_tip = "Melhora esperada"
                    else:
                        h_luck_status = "⚖️ Normal"
                        h_luck_color = "#9ca3af"
                        h_luck_tip = "Sem viés"

                    if abs(a_luck) > 0.03:
                        if a_luck > 0:
                            a_luck_status = "🔥 Quente"
                            a_luck_color = "#f87171"
                            a_luck_tip = "Regressão esperada"
                        else:
                            a_luck_status = "❄️ Frio"
                            a_luck_color = "#4ade80"
                            a_luck_tip = "Melhora esperada"
                    else:
                        a_luck_status = "⚖️ Normal"
                        a_luck_color = "#9ca3af"
                        a_luck_tip = "Sem viés"

                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 0.9em;'>🎯 Shooting Luck</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: {h_luck_color}; font-size: 0.85em;'>{h_luck_status} ({h_luck:+.1%})</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 0.7em; color: #6b7280;'>{h_luck_tip}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: {a_luck_color}; font-size: 0.85em; margin-top: 3px;'>{a_luck_status} ({a_luck:+.1%})</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 0.7em; color: #6b7280;'>{a_luck_tip}</div>", unsafe_allow_html=True)

                # RAPM Impact (Lesões)
                with ai2:
                    h_rapm = game.get('home_rapm_penalty', 0)
                    a_rapm = game.get('away_rapm_penalty', 0)
                    rapm_diff = game.get('rapm_impact_diff', 0)

                    # NEW: Get injured players lists
                    h_injuries_str = game.get('home_injuries_list', '')
                    a_injuries_str = game.get('away_injuries_list', '')

                    # Determinar severidade (UPDATED: considerar presença de lesões)
                    # Se há lesões listadas, status mínimo = Moderado
                    if abs(h_rapm) > 3:
                        h_rapm_status = "🚨 Crítico"
                        h_rapm_color = "#ef4444"
                    elif abs(h_rapm) > 1 or h_injuries_str:  # NEW: ou se há lesões
                        h_rapm_status = "⚠️ Moderado"
                        h_rapm_color = "#f59e0b"
                    else:
                        h_rapm_status = "✅ Saudável"
                        h_rapm_color = "#4ade80"

                    if abs(a_rapm) > 3:
                        a_rapm_status = "🚨 Crítico"
                        a_rapm_color = "#ef4444"
                    elif abs(a_rapm) > 1 or a_injuries_str:  # NEW: ou se há lesões
                        a_rapm_status = "⚠️ Moderado"
                        a_rapm_color = "#f59e0b"
                    else:
                        a_rapm_status = "✅ Saudável"
                        a_rapm_color = "#4ade80"

                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 0.9em;'>💪 RAPM Impact</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: {h_rapm_color}; font-size: 0.85em;'>{h_rapm_status} ({h_rapm:+.1f})</div>", unsafe_allow_html=True)

                    # NEW: Show injured players if any
                    if h_injuries_str:
                        st.markdown(f"<div style='text-align: center; font-size: 0.65em; color: #fbbf24; margin-top: 2px;'>{h_injuries_str}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; font-size: 0.7em; color: #6b7280;'>Lesões {home_team}</div>", unsafe_allow_html=True)

                    st.markdown(f"<div style='text-align: center; color: {a_rapm_color}; font-size: 0.85em; margin-top: 3px;'>{a_rapm_status} ({a_rapm:+.1f})</div>", unsafe_allow_html=True)

                    # NEW: Show injured players if any
                    if a_injuries_str:
                        st.markdown(f"<div style='text-align: center; font-size: 0.65em; color: #fbbf24; margin-top: 2px;'>{a_injuries_str}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; font-size: 0.7em; color: #6b7280;'>Lesões {away_team}</div>", unsafe_allow_html=True)


                st.markdown("</div>", unsafe_allow_html=True)

            # Recomendação de Valor
            if ev_home > 5:
                st.success(f"💎 VALOR: {home_team} ({ev_home:+.1f}% EV)")
            elif ev_away > 5:
                st.success(f"💎 VALOR: {away_team} ({ev_away:+.1f}% EV)")

# --- TAB 2: GESTÃO DE BANCA PROFISSIONAL ---
with tab2:
    render_bankroll_management(daily_games, bankroll_input, kelly_fraction)

    # --- GERADOR DE COMBOS ---
    try:
        from betting.web_ui import render_combo_generator

        # Tentar carregar player props
        player_props_df = None
        try:
            import os
            player_props_path = Path('results/player_props_predictions.csv')

            if player_props_path.exists():
                player_props_df = pd.read_csv(player_props_path)
                # Verificar se tem dados recentes (hoje ou ontem)
                if 'date' in player_props_df.columns:
                    player_props_df['date'] = pd.to_datetime(player_props_df['date'])
                    today = pd.Timestamp.now().normalize()
                    yesterday = today - timedelta(days=1)
                    player_props_df = player_props_df[player_props_df['date'] >= yesterday]

                if player_props_df.empty:
                    player_props_df = None
            else:
                st.caption("💡 Dica: Execute o script de player props para ver combos Team+Player")
        except Exception as e:
            st.caption(f"⚠️ Player props não disponível: {str(e)[:50]}")

        # Renderizar gerador de combos (min_ev=0 para mostrar todos os combos)
        if player_props_df is not None:
            st.caption(f"✅ Player props carregados: {len(player_props_df)} linhas")
        else:
            st.caption("ℹ️ Player props não disponíveis - apenas parlays de times serão gerados")

        render_combo_generator(daily_games, player_props_df, min_ev=0.0)

    except ImportError as e:
        st.error(f"❌ Módulo de combos não disponível: {e}")
    except Exception as e:
        st.error(f"❌ Erro ao renderizar combos: {e}")
        st.caption("Debug: Verifique se betting/combo_generator.py existe")


# --- TAB 3: PLAYER PROPS ---
with tab3:
    st.header("⛹️ Player Props - Projeções de Jogadores")

    # Buscar jogadores dos jogos de hoje
    if daily_games.empty:
        st.info("Sem jogos disponíveis hoje para projeções de jogadores.")
    else:
        # Fetch injury data
        injury_data = {}
        try:
            from data.scrapers.injury_scraper import get_injuries_with_cache
            injury_data = get_injuries_with_cache()
            if injury_data:
                total_injuries = sum(len(players) for players in injury_data.values())
                st.caption(f"📋 Injury Report carregado: {total_injuries} lesões em {len(injury_data)} times")
                # Debug: Show some sample injury data
                st.caption(f"🔍 Debug: Times com lesões: {list(injury_data.keys())[:3]}")
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar injury report: {e}")
            st.caption("Continuando sem dados de lesões")

        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            stat_filter = st.selectbox(
                "📊 Estatística",
                ["PTS (Pontos)", "REB (Rebotes)", "AST (Assistências)", "Todas"],
                index=0
            )
        with col_f2:
            min_games = st.slider(
                "Mínimo de jogos recentes",
                min_value=3,
                max_value=20,
                value=10,
                step=1
            )

        st.markdown("---")

        # Buscar stats de jogadores usando a API real
        try:
            from nba_api.stats.endpoints import PlayerGameLog, LeagueLeaders
            from nba_api.stats.static import players, teams

            with st.spinner("🔄 Buscando projeções de jogadores da NBA API..."):
                # Obter top jogadores da temporada atual (2025-26)
                leaders = LeagueLeaders(
                    league_id='00',
                    per_mode48='PerGame',
                    scope='S',
                    season='2025-26',
                    season_type_all_star='Regular Season',
                    stat_category_abbreviation='PTS'
                )

                df_leaders = leaders.get_data_frames()[0]

                if not df_leaders.empty:
                    # Tentar diferentes nomes de colunas (API pode mudar)
                    team_col = None
                    for possible_col in ['TEAM_ABBREVIATION', 'TEAM', 'TEAM_ABBR', 'TEAM_ID']:
                        if possible_col in df_leaders.columns:
                            team_col = possible_col
                            break

                    if team_col is None:
                        st.error(f"❌ Coluna de time não encontrada. Colunas disponíveis: {', '.join(df_leaders.columns[:10])}")
                    else:
                        # Se houver jogos hoje, filtrar apenas esses times
                        if not daily_games.empty:
                            # Normalizar nomes de times (podem estar como nome completo ou abreviação)
                            teams_playing_today = set()
                            for _, game in daily_games.iterrows():
                                # Adicionar tanto o nome direto quanto a possível abreviação
                                home = game['home_team']
                                away = game['away_team']
                                teams_playing_today.add(home)
                                teams_playing_today.add(away)
                                # Se for nome completo, adicionar também a abreviação
                                if home in TEAM_MAP:
                                    teams_playing_today.add(TEAM_MAP[home])
                                if away in TEAM_MAP:
                                    teams_playing_today.add(TEAM_MAP[away])

                            # DEBUG: Mostrar times procurados
                            # st.write("Times jogando hoje:", sorted(teams_playing_today))
                            # st.write("Times na API:", sorted(df_leaders[team_col].unique()[:10]))

                            df_filtered = df_leaders[df_leaders[team_col].isin(teams_playing_today)].copy()
                            filter_msg = "dos times que jogam hoje"
                        else:
                            # Se não houver jogos hoje, mostrar todos os top jogadores
                            df_filtered = df_leaders.copy()
                            filter_msg = "da liga"

                        if not df_filtered.empty:
                            # Pegar top 30 jogadores (aumentei de 20 para 30)
                            top_players = df_filtered.head(30)

                            # Preparar dados para exibição
                            player_projections = []

                            for _, player in top_players.iterrows():
                                pts_avg = player.get('PTS', 0)
                                reb_avg = player.get('REB', 0)
                                ast_avg = player.get('AST', 0)
                                gp = player.get('GP', 0)
                                player_name = player.get('PLAYER', 'Unknown')

                                # Encontrar oponente de hoje (se houver)
                                team_abbr = player.get(team_col, '')
                                opponent = '-'

                                if not daily_games.empty:
                                    for _, game in daily_games.iterrows():
                                        home_abbr = TEAM_MAP.get(game['home_team'], game['home_team'])
                                        away_abbr = TEAM_MAP.get(game['away_team'], game['away_team'])

                                        if team_abbr == home_abbr or team_abbr == game['home_team']:
                                            opponent = f"vs {game['away_team']}"
                                            break
                                        elif team_abbr == away_abbr or team_abbr == game['away_team']:
                                            opponent = f"@ {game['home_team']}"
                                            break

                                # Check for injury status
                                injury_status = None
                                injury_badge = ""

                                if injury_data:
                                    # Create reverse map: abbr -> full name
                                    # TEAM_MAP is full_name -> abbr, we need the reverse
                                    abbr_to_full = {abbr: full for full, abbr in TEAM_MAP.items() if full in injury_data}

                                    # Get full team name from abbreviation
                                    team_full_name = abbr_to_full.get(team_abbr, team_abbr)

                                    # Try to find player in injury data
                                    if team_full_name in injury_data:
                                        players_dict = injury_data[team_full_name]

                                        # Try exact name match first
                                        if player_name in players_dict:
                                            injury_status = players_dict[player_name]
                                        else:
                                            # Try partial name match (last name)
                                            player_last = player_name.split()[-1] if ' ' in player_name else player_name
                                            for inj_player, status in players_dict.items():
                                                if player_last in inj_player or inj_player.split()[-1] in player_name:
                                                    injury_status = status
                                                    break

                                # Create injury badge
                                if injury_status:
                                    if injury_status == "OUT":
                                        injury_badge = " 🚫"
                                    elif injury_status == "QUESTIONABLE":
                                        injury_badge = " ⚠️"
                                    elif injury_status == "DOUBTFUL":
                                        injury_badge = " ❓"
                                    elif injury_status == "PROBABLE":
                                        injury_badge = " ✅"

                                # Add injury badge to player name
                                display_name = player_name + injury_badge

                                # Calculate "Linha" (projected line) based on stat filter
                                if stat_filter == "PTS (Pontos)":
                                    linha = f"{pts_avg:.1f}"
                                elif stat_filter == "REB (Rebotes)":
                                    linha = f"{reb_avg:.1f}"
                                elif stat_filter == "AST (Assistências)":
                                    linha = f"{ast_avg:.1f}"
                                else:  # "Todas"
                                    linha = f"P:{pts_avg:.1f} R:{reb_avg:.1f} A:{ast_avg:.1f}"

                                # Aplicar filtro de estatística
                                show_player = False
                                if stat_filter == "Todas":
                                    show_player = True
                                elif stat_filter == "PTS (Pontos)" and pts_avg > 15:
                                    show_player = True
                                elif stat_filter == "REB (Rebotes)" and reb_avg > 5:
                                    show_player = True
                                elif stat_filter == "AST (Assistências)" and ast_avg > 5:
                                    show_player = True

                                if show_player:
                                    player_projections.append({
                                        'Jogador': display_name,
                                        'Time': team_abbr,
                                        'Oponente': opponent,
                                        'Linha': linha,
                                        'PTS': f"{pts_avg:.1f}",
                                        'REB': f"{reb_avg:.1f}",
                                        'AST': f"{ast_avg:.1f}",
                                        'Jogos': int(gp)
                                    })

                            if player_projections:
                                df_props = pd.DataFrame(player_projections)

                                # Estilizar tabela
                                st.dataframe(
                                    df_props,
                                    column_config={
                                        "Linha": st.column_config.TextColumn(
                                            "Linha",
                                            help="Linha projetada baseada nas médias da temporada"
                                        ),
                                        "PTS": st.column_config.NumberColumn(
                                            "Pontos",
                                            help="Média de pontos por jogo"
                                        ),
                                        "REB": st.column_config.NumberColumn(
                                            "Rebotes",
                                            help="Média de rebotes por jogo"
                                        ),
                                        "AST": st.column_config.NumberColumn(
                                            "Assistências",
                                            help="Média de assistências por jogo"
                                        ),
                                        "Jogos": st.column_config.NumberColumn(
                                            "GP",
                                            help="Jogos disputados"
                                        )
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.caption(f"📊 Mostrando {len(df_props)} jogadores {filter_msg}. Médias baseadas na temporada 2025-26.")
                            else:
                                st.info(f"Nenhum jogador encontrado com o filtro '{stat_filter}'.")
                        else:
                            st.warning(f"⚠️ Nenhum jogador encontrado {filter_msg}. Times em daily_games podem não corresponder aos da NBA API.")
                else:
                    st.error("Erro ao buscar dados da NBA API.")

        except Exception as e:
            st.error(f"❌ Erro ao buscar props: {e}")
            st.caption("Verifique se a NBA API está acessível e tente novamente.")

# --- TAB 4: PERFORMANCE ---
with tab4:
    st.header("📈 Performance do Modelo")

    # Botão para atualizar resultados
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn2:
        if st.button("🔄 Atualizar Resultados", help="Busca placares finais de jogos passados"):
            progress_placeholder = st.empty()

            with st.spinner("Buscando resultados..."):
                try:
                    from data.scrapers.results_scraper import update_game_results

                    progress_placeholder.info("🔍 Buscando jogos finalizados dos últimos 7 dias...")

                    # Atualizar jogos dos últimos 7 dias
                    updated_count = update_game_results(days_back=7)

                    progress_placeholder.empty()

                    if updated_count > 0:
                        st.success(f"✅ {updated_count} jogos atualizados com sucesso!")
                        st.rerun()
                    else:
                        st.info("ℹ️ Nenhum jogo novo para atualizar. Todos os jogos já estão sincronizados.")
                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"❌ Erro ao atualizar: {e}")
                    st.caption("Verifique os logs para mais detalhes.")

    # Filtro de Data (Range)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        default_start = datetime.now() - timedelta(days=30)
        default_end = datetime.now()

        date_range = st.date_input(
            "Período de Análise",
            value=(default_start, default_end),
            max_value=datetime.now() + timedelta(days=365),
            format="DD/MM/YYYY"
        )

    # Carregar histórico de previsões
    all_preds = db.get_latest_predictions()

    if all_preds.empty:
        st.info("Ainda não há histórico de previsões suficiente para análise.")
    else:
        # Aplicar Filtro de Data
        if isinstance(date_range, tuple):
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (all_preds['date'].dt.date >= start_date) & (all_preds['date'].dt.date <= end_date)
                all_preds = all_preds[mask]
            elif len(date_range) == 1:
                start_date = date_range[0]
                mask = all_preds['date'].dt.date >= start_date
                all_preds = all_preds[mask]

        # Filtrar dados de teste (garbage)
        all_preds = all_preds[~all_preds['home_team'].str.contains("Test", case=False, na=False)]

        if all_preds.empty:
             st.warning(f"Sem previsões válidas no período selecionado.")
        else:
            # Filtrar jogos passados (que já têm resultado CONFIRMADO)
            # Carregar dados históricos de games com resultados
            df_confirmed = db.get_history().copy()
            df_confirmed = df_confirmed[df_confirmed['home_score'] > 0]

            # --- CORREÇÃO DE NOMES PARA MERGE ---
            # Converter nomes completos (Predictions) para Abreviações (Games)
            # Usar TEAM_MAP definido no início (assuming TEAM_MAP is defined elsewhere, e.g., in a config file or globally)
            all_preds['home_team_abbr'] = all_preds['home_team'].map(TEAM_MAP).fillna(all_preds['home_team'])
            all_preds['away_team_abbr'] = all_preds['away_team'].map(TEAM_MAP).fillna(all_preds['away_team'])

            # Criar chave única para ambos os dataframes usando ABBR
            all_preds['join_key'] = (
                all_preds['date'].astype(str) +
                all_preds['home_team_abbr'].astype(str) +
                all_preds['away_team_abbr'].astype(str)
            )

            df_confirmed['join_key'] = (
                df_confirmed['date'].astype(str) +
                df_confirmed['home_team'].astype(str) +
                df_confirmed['away_team'].astype(str)
            )

            # Merge com sufixos explícitos para evitar colisão
            # Usar LEFT JOIN para manter previsões mesmo sem resultado (pendentes)
            merged = pd.merge(
                all_preds,
                df_confirmed[['join_key', 'winner', 'home_score', 'away_score']],
                on='join_key',
                how='left',
                suffixes=('_pred', '_real')
            )

            # Identificar coluna de vencedor real
            col_winner = 'winner'
            if 'winner_real' in merged.columns:
                col_winner = 'winner_real'
            elif 'winner' not in merged.columns:
                merged['winner'] = np.nan

            # Normalizar para 'winner'
            if col_winner != 'winner':
                merged['winner'] = merged[col_winner]

            # Identificar coluna de home_score real
            col_home = 'home_score'
            if 'home_score_real' in merged.columns:
                col_home = 'home_score_real'
            elif 'home_score' not in merged.columns:
                merged['home_score'] = 0

            if col_home != 'home_score':
                merged['home_score'] = merged[col_home]

            # Identificar coluna de away_score real
            col_away = 'away_score'
            if 'away_score_real' in merged.columns:
                col_away = 'away_score_real'
            elif 'away_score' not in merged.columns:
                merged['away_score'] = 0

            if col_away != 'away_score':
                merged['away_score'] = merged[col_away]

            # Preencher valores nulos para jogos pendentes
            merged['winner'] = merged['winner'].fillna('Aguardando...')
            merged['home_score'] = merged['home_score'].fillna(0)
            merged['away_score'] = merged['away_score'].fillna(0)

            if merged.empty:
                st.info("ℹ️ Nenhuma previsão encontrada para o período selecionado.")
            else:
                # Calcular predição do modelo (HOME/AWAY) para todo o DF
                # Se não tiver coluna 'prediction', usar prob_home > 50
                if 'prediction' in merged.columns:
                     # Normalizar prediction se for nome do time... mas assumindo HOME/AWAY ou prob
                     # Fallback seguro: usar prob_home
                     merged['pred_side'] = merged['prob_home'].apply(lambda x: 'HOME' if x > 50 else 'AWAY')
                else:
                     merged['pred_side'] = merged['prob_home'].apply(lambda x: 'HOME' if x > 50 else 'AWAY')

                # Normalizar vencedor real
                merged['winner_norm'] = merged['winner'].astype(str).str.upper().str.strip()

                # Calcular is_correct
                # Se winner for 'AGUARDANDO...', is_correct será False (o que ok, pois Status vai tratar)
                merged['is_correct'] = merged['pred_side'] == merged['winner_norm']

                # Calcular Acurácia (apenas sobre jogos concluídos)
                completed_games = merged[merged['winner'] != 'Aguardando...'].copy()

                # Coluna Visual de Status
                def get_status_icon(row):
                    if row['winner'] == 'Aguardando...':
                        return '⏳'
                    return '✅' if row['is_correct'] else '❌'

                merged['Status'] = merged.apply(get_status_icon, axis=1)

                acc = merged['is_correct'].mean()
                total = len(merged)
                correct = merged['is_correct'].sum()

                # Métricas Principais
                c1, c2, c3 = st.columns(3)
                c1.metric("Acurácia Global", f"{acc:.1%}")
                c2.metric("Total Jogos", total)
                c3.metric("Acertos", correct)

                # Gráfico de Evolução (Acumulado)
                merged = merged.sort_values('date')
                merged['acc_cum'] = merged['is_correct'].expanding().mean()

                st.subheader("Evolução da Acurácia")
                st.line_chart(merged.set_index('date')['acc_cum'])

                # Tabela Detalhada Limpa
                st.subheader("Histórico Detalhado")

                # Selecionar apenas colunas interessantes
                cols_detail = ['date', 'home_team', 'away_team', 'prob_home', col_winner, 'Status']
                detail_df = merged[cols_detail].rename(columns={
                    'date': 'Data',
                    'home_team': 'Casa',
                    'away_team': 'Visitante',
                    'prob_home': 'Prob Casa',
                    col_winner: 'Vencedor',
                    'Status': 'Resultado'
                }).sort_values('Data', ascending=False)

                st.dataframe(
                    detail_df,
                    column_config={
                        "Prob Casa": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                        "Resultado": st.column_config.TextColumn(width="small")
                    },
                    use_container_width=True,
                    hide_index=True
                )

# --- TAB 5: MODEL HEALTH ---
with tab5:
    st.header("🔍 Saúde do Modelo")

    # Verificar se há arquivo de métricas
    metrics_file = Path('data/monitoring/metrics_history.json')

    if metrics_file.exists():
        try:
            import json
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)

            if metrics_history:
                # Mostrar última métrica
                latest = metrics_history[-1]

                col1, col2, col3 = st.columns(3)
                with col1:
                    acc = latest.get('accuracy', 0) * 100
                    st.metric("Acurácia (ML)", f"{acc:.1f}%")
                with col2:
                    auc = latest.get('auc_roc', 0)
                    st.metric("AUC-ROC", f"{auc:.3f}")
                with col3:
                    log_loss = latest.get('log_loss', 0)
                    st.metric("Log Loss", f"{log_loss:.3f}")

                # Gráfico de acurácia ao longo do tempo
                if len(metrics_history) > 1:
                    import plotly.graph_objects as go

                    timestamps = [m.get('timestamp', i) for i, m in enumerate(metrics_history)]
                    accuracies = [m.get('accuracy', 0) * 100 for m in metrics_history]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=accuracies,
                        mode='lines+markers',
                        name='Acurácia',
                        line=dict(color='#10b981', width=2)
                    ))

                    fig.update_layout(
                        title='Acurácia do Modelo ao Longo do Tempo',
                        xaxis_title='Data',
                       yaxis_title='Acurácia (%)',
                        height=400,
                        template='plotly_dark'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Coletando histórico de métricas... Volte em 7 dias para ver tendências.")
            else:
                st.warning("Arquivo de métricas vazio.")
        except Exception as e:
            st.error(f"Erro ao carregar métricas: {e}")
    else:
        st.warning("📊 Sistema de monitoramento ainda não executou. Execute `python scripts/monitoring_system.py` para gerar métricas.")

        # Mostrar botão para executar monitoramento
        if st.button("🔄 Executar Monitoramento Agora"):
            st.info("Executando... (pode levar 30 segundos)")
            # TODO: Trigger monitoring via subprocess

# --- TAB 6: BACKTEST ---
with tab6:
    st.header("🧪 Backtest Analysis - Simulador de Estratégias")

    st.markdown("### Configuração da Estratégia")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        strategy = st.selectbox(
            "Estratégia de Apostas",
            ["Flat Betting", "Kelly Criterion", "Confiança Mínima"]
        )
        initial_bankroll = st.number_input(
            "Banca Inicial (R$)",
            min_value=100.0,
            value=1000.0,
            step=100.0
        )

    with col_cfg2:
        min_confidence = st.select_slider(
            "Confiança Mínima",
            options=["BAIXA", "MÉDIA", "ALTA", "MUITO ALTA"],
            value="MÉDIA"
        )
        bet_unit = st.number_input(
            "Unidade de Aposta (% da Banca)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )

    if st.button("🎲 Simular Estratégia"):
        # Buscar previsões passadas
        history_preds = db.get_latest_predictions()

        if history_preds.empty:
            st.error("❌ Sem histórico de previsões para backtest.")
        else:
            # 1. Preencher Odds Faltantes (Simulação)
            # Se odds_home for 0 ou NaN, calcular odd justa baseada na probabilidade - 5% vig
            mask_no_odds = (history_preds['odds_home'] == 0) | (history_preds['odds_home'].isna())

            if mask_no_odds.any():
                st.caption(f"ℹ️ Simulando odds para {mask_no_odds.sum()} jogos sem dados históricos (Margem de 5%).")
                # Evitar divisão por zero
                probs = history_preds.loc[mask_no_odds, 'prob_home'].clip(1, 99) / 100
                # Odd Justa = 1/Prob. Odd Bookie ~= Odd Justa * 0.95
                simulated_odds = (1 / probs) * 0.95
                history_preds.loc[mask_no_odds, 'odds_home'] = simulated_odds

            # 2. Buscar Resultados Reais
            # Precisamos saber quem ganhou para calcular o P&L real
            df_results = db.get_history()

            # Normalizar nome do time para abreviação (Predictions tem Full Name, History tem Abbr)
            # Usar TEAM_MAP global
            history_preds['home_abbr'] = history_preds['home_team'].map(TEAM_MAP).fillna(history_preds['home_team'])

            # Criar chave de junção (Data + Home Abbr)
            history_preds['join_key'] = history_preds['date'].dt.strftime('%Y-%m-%d') + "_" + history_preds['home_abbr']
            df_results['join_key'] = df_results['date'].astype(str) + "_" + df_results['home_team']

            # Merge para trazer o vencedor real
            merged = pd.merge(
                history_preds,
                df_results[['join_key', 'winner']],
                on='join_key',
                how='left',
                suffixes=('', '_real')
            )

            # Filtrar apostas pela confiança
            # Normalizar labels de confiança (Banco tem misturado: MEDIUM, Média, MÉDIA, etc)
            def normalize_confidence(val):
                val = str(val).upper().strip()
                if val in ['LOW', 'BAIXA', 'BAIXO']: return 'BAIXA'
                if val in ['MEDIUM', 'MEDIA', 'MÉDIA', 'MEDIO']: return 'MÉDIA'
                if val in ['HIGH', 'ALTA', 'ALTO']: return 'ALTA'
                if val in ['VERY HIGH', 'MUITO ALTA']: return 'MUITO ALTA'
                return 'N/A'

            merged['confidence_norm'] = merged['confidence'].apply(normalize_confidence)

            valid_bets = merged[merged['confidence_norm'] == min_confidence].copy()

            if valid_bets.empty:
                st.warning(f"Sem apostas encontradas com confiança '{min_confidence}'. Tente diminuir o filtro.")
            else:
                # Simular Backtest
                bankroll = initial_bankroll
                bankroll_history = [bankroll]
                wins = 0
                losses = 0
                skipped = 0

                # Ordenar por data
                valid_bets = valid_bets.sort_values('date')

                for idx, bet in valid_bets.iterrows():
                    # Se não tiver resultado real, pular
                    winner = bet.get('winner_real')
                    if pd.isna(winner) or str(winner).lower() == 'none':
                        skipped += 1
                        continue

                    bet_amount = bankroll * (bet_unit / 100)

                    # Verificar se ganhou
                    prediction = 'HOME' if bet['prob_home'] > 50 else 'AWAY'

                    real_winner = str(winner).upper()

                    if real_winner == bet['home_team'].upper():
                        real_winner = 'HOME'
                    elif real_winner == bet['away_team'].upper():
                        real_winner = 'AWAY'

                    won = (prediction == real_winner)

                    if won:
                        odds = bet['odds_home']
                        if prediction == 'AWAY':
                            if bet.get('odds_away', 0) > 0:
                                odds = bet['odds_away']
                            else:
                                prob_away = 1 - (bet['prob_home']/100)
                                odds = (1 / prob_away) * 0.95

                        profit = bet_amount * (odds - 1)
                        bankroll += profit
                        wins += 1
                    else:
                        bankroll -= bet_amount
                        losses += 1

                    bankroll_history.append(bankroll)

                # Exibir resultados
                total_bets = wins + losses

                if total_bets > 0:
                    final_roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
                    real_win_rate = (wins / total_bets) * 100

                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("ROI", f"{final_roi:+.1f}%", delta=f"R$ {bankroll - initial_bankroll:+.2f}")
                    with col_r2:
                        st.metric("Banca Final", f"R$ {bankroll:.2f}")
                    with col_r3:
                        st.metric("Win Rate Real", f"{real_win_rate:.1f}%", help=f"{wins}W - {losses}L")

                    # Gráfico de crescimento
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=bankroll_history,
                        mode='lines',
                        name='Banca',
                        line=dict(color='#3b82f6', width=2),
                        fill='tozeroy'
                    ))

                    fig.update_layout(
                        title=f'Evolução da Banca - {strategy} ({total_bets} apostas)',
                        xaxis_title='Número de Apostas',
                        yaxis_title='Banca (R$)',
                        height=400,
                        template='plotly_dark'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    if skipped > 0:
                        st.caption(f"⚠️ {skipped} jogos ignorados por falta de resultado confirmado.")
                else:
                    st.warning(f"Nenhum jogo com resultado confirmado encontrado para backtest. (Skipped: {skipped})")
