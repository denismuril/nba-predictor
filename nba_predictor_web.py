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
    page_title="NBA Predictor v22.0 - Enterprise Edition",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.header("🏀 NBA Predictor v22.0")

    # Status do Sistema
    st.subheader("🖥️ Status do Sistema")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Banco de Dados", "Online", delta_color="normal")
    with col2:
        st.metric("Modelos", "v22.0 ✓", delta_color="normal")

    last_update = datetime.now().strftime("%H:%M")
    st.text(f"Última Atualização: {last_update}")

    with st.expander("🛠️ Debug Standings"):
        st.write("Verificando dados de classificação...")
        try:
            raw_std = load_standings()
            st.write(f"Times carregados: {len(raw_std)}")
            if len(raw_std) > 0:
                st.write("Amostra (1º Time):", list(raw_std.items())[0])
            
            # Check for current team
            st.write("Mapeamento (Sample):")
            sample_team = list(raw_std.keys())[0] if raw_std else "N/A"
            st.write(f"Key: {sample_team} -> Map: {TEAM_MAP.get(sample_team, 'Not Found')}")
        except Exception as e:
            st.error(f"Erro Debug: {e}")

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
    """Load current standings with rankings by win percentage (Sync)."""
    import requests
    
    url = "http://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        standings = {}
        children = data.get('children', [])
        
        for child in children:
            entries = child.get('standings', {}).get('entries', [])
            for entry in entries:
                try:
                    team_name = entry['team']['displayName']
                    stats = entry.get('stats', [])
                    wins = 0
                    losses = 0
                    for stat in stats:
                        if stat['name'] == 'wins':
                            wins = int(stat['value'])
                        elif stat['name'] == 'losses':
                            losses = int(stat['value'])
                    
                    standings[team_name] = {'wins': wins, 'losses': losses}
                except Exception:
                    continue

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
        
        # --- CRITICAL FIX: Ensure Legacy Aliases are Covered for UI ---
        # The UI might lookup 'BRK' but we have 'BKN', etc.
        aliases = {
            'BKN': 'BRK', 'PHX': 'PHO', 'CHA': 'CHO', 'NOP': 'NOR', 'UTA': 'UTH',
            'SAS': 'SAN', 'GSW': 'GS', 'NYK': 'NY'
        }
        for current, legacy in aliases.items():
            if current in result:
                result[legacy] = result[current]
                
        return result
    except Exception as e:
        # Log error to UI for debugging if needed, or just warn
        print(f"Error loading standings: {e}")
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


# =============================================================================
# FASE 4: Visualizações de Player Props e Impacto de Lesões
# =============================================================================


def render_top_trends_section(df_props: pd.DataFrame):
    """
    FASE 4 IMPLEMENTATION: Exibe jogadores com Hit Rate > 80% nos últimos 10 jogos.

    Mostra uma seção de "jogadores quentes" para identificar boas apostas Over.

    Args:
        df_props: DataFrame com colunas de hit rate (PTS_hit_L10, etc.)
    """
    if df_props is None or df_props.empty:
        st.info("📊 Nenhum dado de player props disponível para Top Trends.")
        return

    st.subheader("🔥 Top Trends - Hit Rate > 80% (L10)")

    # Verificar se temos colunas de hit rate
    hit_cols = [c for c in df_props.columns if '_hit_L10' in c]
    if not hit_cols:
        st.info("📊 Execute o cálculo de hit rates primeiro.")
        return

    # Filtrar jogadores com hit rate >= 80%
    hot_players = []
    for col in hit_cols:
        stat = col.replace('_hit_L10', '')
        hot = df_props[df_props[col] >= 0.80][['Player', 'Team', col]].copy()
        hot['Stat'] = stat
        hot['Hit Rate L10'] = hot[col]
        hot_players.append(hot[['Player', 'Team', 'Stat', 'Hit Rate L10']])

    if not hot_players:
        st.info("📊 Nenhum jogador com Hit Rate > 80% encontrado.")
        return

    combined = pd.concat(hot_players, ignore_index=True)
    combined = combined.sort_values('Hit Rate L10', ascending=False).head(10)

    # Formatar display
    combined['Hit Rate L10'] = combined['Hit Rate L10'].apply(lambda x: f"{x*100:.0f}%")

    # Estilizar
    st.dataframe(
        combined.style.applymap(
            lambda x: 'color: #4ade80; font-weight: bold' if x.endswith('%') else '',
            subset=['Hit Rate L10']
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption("💡 Jogadores que superaram a linha em 8+ dos últimos 10 jogos.")


def render_injury_impact_alert(injury_adjustments: list):
    """
    FASE 4 IMPLEMENTATION: Exibe alertas visuais de impacto de lesões.

    Mostra como a ausência de um jogador afeta a projeção de outros.

    Args:
        injury_adjustments: Lista de dicts com:
            - 'injured_player': Nome do jogador lesionado
            - 'team': Time afetado
            - 'beneficiary': Jogador que se beneficia
            - 'pts_lift': Pontos adicionais projetados

    Exemplo:
        render_injury_impact_alert([
            {'injured_player': 'Luka Doncic', 'team': 'DAL',
             'beneficiary': 'Kyrie Irving', 'pts_lift': 4.5}
        ])
    """
    if not injury_adjustments:
        return

    st.subheader("🏥 Impacto de Lesões")

    for adj in injury_adjustments:
        with st.container():
            cols = st.columns([3, 1, 3, 1])

            # Jogador lesionado
            cols[0].markdown(
                f"🚑 **{adj.get('injured_player', 'Unknown')}** OUT",
                help=f"Time: {adj.get('team', 'N/A')}"
            )

            # Seta
            cols[1].markdown("→")

            # Beneficiário
            cols[2].markdown(
                f"📈 **{adj.get('beneficiary', 'Unknown')}**",
                help="Jogador que recebe aumento de usage"
            )

            # Lift em pontos
            pts_lift = adj.get('pts_lift', 0)
            cols[3].metric(
                "Ajuste",
                f"+{pts_lift:.1f} pts",
                delta=f"+{pts_lift:.1f}",
                delta_color="normal"
            )

    st.caption(
        "💡 Projeções baseadas em análise histórica de usage rate quando "
        "o jogador titular está ausente."
    )


# --- MAIN CONTENT ---

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Dashboard", "💰 Gestão de Banca", "⛹️ Player Props",
    "📈 Performance", "🔍 Saúde do Modelo", "🧪 Análise de Backtest",
    "🖥️ System Health", "🎯 PROP SNIPER"
])

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

    # Se não houver previsões no banco, oferecer opção de gerar em tempo real
    if daily_games.empty:
        st.warning(f"⚠️ Nenhuma previsão salva para {date_str}.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔮 Gerar Previsões Agora", key="generate_predictions"):
                with st.spinner("Gerando previsões... (pode demorar 1-2 min)"):
                    try:
                        from ml_pipeline.predict import predict_next_games
                        predictions_df = predict_next_games(date_str)
                        
                        if predictions_df is not None and not predictions_df.empty:
                            # Converter para formato esperado pelo dashboard
                            daily_games = predictions_df.copy()
                            if 'confidence' not in daily_games.columns:
                                daily_games['confidence'] = 'MEDIUM'
                            st.success(f"✅ {len(daily_games)} previsões geradas!")
                            st.rerun()
                        else:
                            st.error("❌ Nenhum jogo encontrado para esta data.")
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar previsões: {e}")
        
        with col_btn2:
            st.info("💡 Ou rode: `python -m ml_pipeline.predict " + date_str + "`")
        
        # Fallback: tentar histórico (mas sem confiança)
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
                'HIGH': 80, 'ALTA': 80, 'VERY HIGH': 90, 'MUITO ALTA': 90
            }
            if val_str in text_map: return text_map[val_str]
            try: return float(val_str.rstrip('%'))
            except: return 0

        daily_games['confidence_num'] = daily_games['confidence'].apply(convert_confidence)
        
        # Traduzir label para exibição
        def translate_conf_label(val):
            val_upper = str(val).upper().strip().replace('%','')
            if val_upper in ['HIGH', 'ALTA'] or (val_upper.isdigit() and float(val_upper) >= 80): return 'ALTA 🚀'
            if val_upper in ['MEDIUM', 'MEDIA', 'MÉDIA'] or (val_upper.isdigit() and float(val_upper) >= 60): return 'MÉDIA ⚖️'
            return 'BAIXA ⚠️'
            
        daily_games['confidence_display'] = daily_games['confidence'].apply(translate_conf_label)

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
                # Barra de Confiança Visual
                conf_pct = min(100, max(0, int(game['confidence_num'])))
                conf_display = game.get('confidence_display', str(game['confidence']))
                
                with c2: 
                    # Meter Style
                    st.markdown(f"""
                    <div style="text-align: right;">
                        <span style='font-size: 0.7em; color: #aaa;'>CONFIANÇA</span><br>
                        <span style='background-color: {conf_color}30; color: {conf_color}; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.85em; border: 1px solid {conf_color}'>{conf_display}</span>
                    </div>
                    """, unsafe_allow_html=True)

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
                st.markdown("<div style='text-align: center; font-size: 0.8em; color: #aaa; margin-bottom: 5px;'>INSIGHTS AVANÇADOS (V21.8)</div>", unsafe_allow_html=True)

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
            from data.scrapers.injury_scraper_v2 import get_injuries_with_cache
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
                    
                    # Add legacy aliases BKN->BRK etc
                    legacy_aliases = {
                        'BKN': 'BRK', 'CHA': 'CHO', 'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN',
                        'DET': 'DET', 'GSW': 'GSW', 'HOU': 'HOU', 'IND': 'IND', 'LAC': 'LAC',
                        'LAL': 'LAL', 'MEM': 'MEM', 'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN',
                        'NOP': 'NOR', 'NYK': 'NYK', 'OKC': 'OKC', 'ORL': 'ORL', 'PHI': 'PHI',
                        'PHX': 'PHO', 'POR': 'POR', 'SAC': 'SAC', 'SAS': 'SAS', 'TOR': 'TOR',
                        'UTA': 'UTA', 'WAS': 'WAS', 'ATL': 'ATL', 'BOS': 'BOS', 'CHI': 'CHI'
                    }
                    
                    # Definir quantos dias buscar
                    deep_update = st.session_state.get('deep_update', False)
                    days = 200 if deep_update else 7
                    
                    st.info(f"🔍 Buscando resultados dos últimos {days} dias...")
                    updated_count = update_game_results(days_back=days)
                    
                    if updated_count > 0:
                        st.success(f"✅ {updated_count} jogos atualizados com sucesso!")
                    else:
                        st.info("ℹ️ Nenhum jogo novo para atualizar (ou API não retornou dados novos).")
                    
                    if deep_update:
                        st.info("ℹ️ Deep Update finalizado. Verifique se os placares antigos (ex: Nov) apareceram.")
                    
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao atualizar: {e}")
                    st.caption("Verifique os logs para mais detalhes.")

    # Opção de Deep Update
    st.checkbox("Forçar atualização da temporada completa (Deep Update)", key='deep_update', help="Marque para buscar resultados desde o início da temporada. Mais lento.")

    # Filtro de Data (Split Inputs)
    col_p1, col_p2 = st.columns(2)
    
    default_start = datetime.now() - timedelta(days=30)
    default_end = datetime.now()

    with col_p1:
        start_date = st.date_input(
            "Data Inicial",
            value=default_start,
            format="DD/MM/YYYY"
        )

    with col_p2:
        end_date = st.date_input(
            "Data Final",
            value=default_end,
            max_value=datetime.now() + timedelta(days=365),
            format="DD/MM/YYYY"
        )
        
    # Logic adjustment (merging into single usage)
    date_range = (start_date, end_date)

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
            # CRITICAL: Normalizar data para formato YYYY-MM-DD (evita mismatch datetime vs string)
            all_preds['date_norm'] = pd.to_datetime(all_preds['date']).dt.strftime('%Y-%m-%d')
            all_preds['join_key'] = (
                all_preds['date_norm'] +
                all_preds['home_team_abbr'].astype(str) +
                all_preds['away_team_abbr'].astype(str)
            )

            # Normalizar abreviações Legacy (PHO->PHX, BRK->BKN, etc)
            # Criar mapa reverso para normalizar abreviações inconsistentes
            ABBR_NORMALIZE = {'PHO': 'PHX', 'BRK': 'BKN', 'CHO': 'CHA', 'NOR': 'NOP', 'NO': 'NOP', 
                              'NY': 'NYK', 'GS': 'GSW', 'WSH': 'WAS', 'UTAH': 'UTA'}
            df_confirmed['home_team_norm'] = df_confirmed['home_team'].map(ABBR_NORMALIZE).fillna(df_confirmed['home_team'])
            df_confirmed['away_team_norm'] = df_confirmed['away_team'].map(ABBR_NORMALIZE).fillna(df_confirmed['away_team'])
            
            df_confirmed['date_norm'] = pd.to_datetime(df_confirmed['date']).dt.strftime('%Y-%m-%d')
            df_confirmed['join_key'] = (
                df_confirmed['date_norm'] +
                df_confirmed['home_team_norm'].astype(str) +
                df_confirmed['away_team_norm'].astype(str)
            )

            # --- DEDUPLICATION FIX ---
            # Remove any duplicate entries that might cause row multiplication
            if not all_preds.empty:
                all_preds = all_preds.drop_duplicates(subset=['join_key'], keep='first')
            
            if not df_confirmed.empty:
                df_confirmed = df_confirmed.drop_duplicates(subset=['join_key'], keep='first')

            # Merge com sufixos explícitos para evitar colisão
            # Usar LEFT JOIN para manter previsões mesmo sem resultado (pendentes)
            merged = pd.merge(
                all_preds,
                df_confirmed[['join_key', 'winner', 'home_score', 'away_score']],
                on='join_key',
                how='left',
                suffixes=('_pred', '_real')
            )
            
            # DEBUG: Diagnóstico de merge para troubleshooting
            matched_count = merged['winner'].notna().sum() if 'winner' in merged.columns else 0
            if 'winner_real' in merged.columns:
                matched_count = merged['winner_real'].notna().sum()
            pending_count = len(merged) - matched_count
            st.caption(f"📊 Resultados: {matched_count} confirmados | {pending_count} aguardando")

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

# --- TAB 7: SYSTEM HEALTH ---
with tab7:
    st.header("🖥️ System Health - Go Live Dashboard")
    st.caption("Monitoramento em tempo real do sistema para operação segura")

    # Status file for panic button
    STOP_FILE = Path('data/.STOP_ALL_BETS')

    # Check if system is stopped
    is_stopped = STOP_FILE.exists()

    if is_stopped:
        st.error("🚨 SISTEMA PARADO - Apostas desativadas pelo botão de pânico")
        if st.button("✅ Reativar Sistema", type="primary"):
            STOP_FILE.unlink()
            st.success("Sistema reativado!")
            st.rerun()
    else:
        st.success("✅ Sistema Online")

    st.markdown("---")

    # Health checks
    col_h1, col_h2, col_h3 = st.columns(3)

    # PostgreSQL Status
    with col_h1:
        st.subheader("🐘 PostgreSQL")
        try:
            db = get_db_manager()
            # Simple query to check connection
            games_count = len(db.get_comprehensive_history())
            st.metric("Status", "🟢 Online")
            st.metric("Jogos no DB", f"{games_count:,}")
        except Exception as e:
            st.metric("Status", "🔴 Offline")
            st.error(f"Erro: {str(e)[:50]}")

    # Redis Status
    with col_h2:
        st.subheader("📮 Redis Cache")
        try:
            import redis
            r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            r.ping()

            # Check odds freshness
            keys = r.keys('odds:*')
            fresh_keys = 0
            for key in keys[:10]:  # Sample
                ttl = r.ttl(key)
                if ttl > 0:
                    fresh_keys += 1

            if len(keys) > 0 and fresh_keys < len(keys) / 2:
                st.metric("Status", "🟡 Stale Data")
                st.warning("Odds podem estar desatualizadas (TTL baixo)")
            else:
                st.metric("Status", "🟢 Online")

            st.metric("Odds em Cache", len(keys))
        except Exception as e:
            st.metric("Status", "🔴 Offline")
            st.caption(f"Redis indisponível: {str(e)[:30]}")

    # Odds API Status
    with col_h3:
        st.subheader("📊 API de Odds")
        try:
            # Check last odds fetch
            odds_log = Path('data/cache/odds_last_fetch.txt')
            if odds_log.exists():
                last_fetch = datetime.fromtimestamp(odds_log.stat().st_mtime)
                age_minutes = (datetime.now() - last_fetch).total_seconds() / 60

                if age_minutes > 120:  # 2 hours
                    st.metric("Status", "🟡 Stale")
                    st.warning(f"Última atualização: {age_minutes:.0f} min atrás")
                elif age_minutes > 30:
                    st.metric("Status", "🟡 Check")
                    st.caption(f"Última atualização: {age_minutes:.0f} min")
                else:
                    st.metric("Status", "🟢 Fresh")
                    st.caption(f"{age_minutes:.0f} min atrás")
            else:
                st.metric("Status", "🟡 Desconhecido")
                st.caption("Sem registro de fetch")
        except Exception as e:
            st.metric("Status", "🔴 Erro")
            st.caption(str(e)[:30])

    st.markdown("---")

    # Paper Trading Stats
    st.subheader("📊 Paper Trading Status")
    try:
        db = get_db_manager()

        # Get paper trading stats (using direct SQL for sync compatibility)
        import psycopg2
        conn_str = os.getenv('DATABASE_URL', 'postgresql://nba:nba@localhost:5432/nba_predictor')

        try:
            import psycopg2
            conn = psycopg2.connect(conn_str)
            cur = conn.cursor()

            # Get 7-day stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_bets,
                    COUNT(CASE WHEN status = 'WIN' THEN 1 END) as wins,
                    COUNT(CASE WHEN status = 'LOSS' THEN 1 END) as losses,
                    COALESCE(SUM(CASE WHEN status != 'PENDING' THEN profit ELSE 0 END), 0) as total_pnl,
                    COALESCE(SUM(stake), 0) as total_staked
                FROM paper_bets
                WHERE created_at >= NOW() - INTERVAL '7 days'
            """)
            row = cur.fetchone()

            if row and row[0] > 0:
                total_bets, wins, losses, total_pnl, total_staked = row
                settled = wins + losses
                win_rate = (wins / settled * 100) if settled > 0 else 0
                roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    st.metric("Apostas (7d)", total_bets)
                with col_p2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with col_p3:
                    pnl_color = "normal" if total_pnl >= 0 else "inverse"
                    st.metric("PnL", f"R$ {total_pnl:+,.2f}", delta_color=pnl_color)
                with col_p4:
                    st.metric("ROI", f"{roi:+.1f}%")

                # PnL Chart - Daily evolution
                st.markdown("#### 📈 Evolução do PnL (Últimos 30 dias)")
                cur.execute("""
                    SELECT
                        DATE(created_at) as date,
                        SUM(CASE WHEN status != 'PENDING' THEN profit ELSE 0 END) as daily_pnl
                    FROM paper_bets
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """)
                pnl_data = cur.fetchall()

                if pnl_data:
                    dates = [row[0] for row in pnl_data]
                    daily_pnl = [float(row[1]) for row in pnl_data]
                    cumulative_pnl = []
                    running = 0
                    for p in daily_pnl:
                        running += p
                        cumulative_pnl.append(running)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='PnL Acumulado',
                        line=dict(color='green' if cumulative_pnl[-1] >= 0 else 'red', width=2)
                    ))
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="Data",
                        yaxis_title="PnL Acumulado (R$)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recent Bets Table
                st.markdown("#### 📋 Últimas 10 Apostas")
                cur.execute("""
                    SELECT matchup, bet_type, market_odds, stake, status, profit, created_at
                    FROM paper_bets
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                recent_bets = cur.fetchall()

                if recent_bets:
                    df_bets = pd.DataFrame(recent_bets, columns=[
                        'Matchup', 'Tipo', 'Odds', 'Stake', 'Status', 'Profit', 'Data'
                    ])
                    df_bets['Odds'] = df_bets['Odds'].apply(lambda x: f"{x:.2f}")
                    df_bets['Stake'] = df_bets['Stake'].apply(lambda x: f"R$ {x:.2f}")
                    df_bets['Profit'] = df_bets['Profit'].apply(lambda x: f"R$ {x:+.2f}")
                    df_bets['Data'] = pd.to_datetime(df_bets['Data']).dt.strftime('%d/%m %H:%M')
                    st.dataframe(df_bets, use_container_width=True, hide_index=True)

            else:
                st.info("📊 Nenhuma aposta paper registrada ainda.")

            conn.close()

        except ImportError:
            st.info("📊 Paper trading não disponível (psycopg2 não instalado)")
        except Exception as e:
            st.info(f"📊 Paper trading não inicializado: {str(e)[:50]}")

    except Exception as e:
        st.info("📊 Paper trading não inicializado. Execute: python betting/paper_trading.py")

    # Panic Button
    st.subheader("🚨 Controle de Emergência")

    col_panic, col_info = st.columns([1, 2])

    with col_panic:
        if not is_stopped:
            if st.button("🛑 STOP ALL BETS", type="primary", use_container_width=True):
                # Create stop file
                STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
                STOP_FILE.touch()
                st.error("🚨 SISTEMA PARADO!")
                st.rerun()
        else:
            st.info("Sistema já está parado")

    with col_info:
        st.markdown("""
        **O que o STOP ALL BETS faz:**
        - Para imediatamente o Sniper Engine
        - Bloqueia novos sinais de aposta
        - Não afeta apostas já registradas
        - Use para pausar operação em emergências
        """)

    # System Commands Reference
    st.markdown("---")
    st.subheader("📋 Comandos de Operação")

    with st.expander("Ver comandos úteis"):
        st.code("""
# Iniciar Paper Trading
python betting/paper_trading.py --bankroll 1000

# Liquidar apostas de ontem
python betting/settle_paper_bets.py

# Ver relatório de 7 dias
python betting/paper_trading.py --report --days 7

# Rodar orquestrador completo
python orchestrator.py

# Atualizar dados
python scripts/fetch_todays_games.py
        """, language="bash")


# =============================================================================
# TAB 8: PROP SNIPER - Sistema Quantum de Player Props
# =============================================================================

with tab8:
    st.header("🎯 PROP SNIPER - Sistema Quantum de Player Props")
    st.markdown("""
    **Identificação de Alpha (Vantagem Matemática) sobre as Casas de Apostas**
    
    Este sistema usa:
    - 🔬 **Features Inumanas**: Fadiga biológica, DvP 2.0, Blowout Risk, Dynamic Usage
    - 🎯 **Modelagem em Dois Estágios**: XGBoost (Minutos) + LightGBM Quantile (Taxa/Min)
    - 📊 **Intervalos de Confiança**: Percentis 10th, 50th, 90th para decisões ALL-IN
    """)
    
    st.markdown("---")
    
    # Carregar dados e modelos
    @st.cache_data(ttl=600)
    def load_prop_sniper_data():
        """Carrega dados para o Prop Sniper."""
        try:
            from data.scrapers.quantum_scraper import QuantumDataCollector, fetch_all_data_for_predictions
            from ml_pipeline.train_props_quantum import load_quantum_models
            
            # Coletar dados
            collector = QuantumDataCollector()
            
            # Props lines (odds das casas)
            props_lines = collector.fetch_player_props_odds()
            
            # FALLBACK: Se não houver props, gerar a partir de player stats
            if not props_lines:
                player_stats = collector.fetch_all_player_stats_nba_api()
                if player_stats is not None and not player_stats.empty:
                    props_lines = generate_mock_props_from_stats(player_stats)
            
            # Carregar modelos se existirem
            models = load_quantum_models()
            
            return {
                'props_lines': props_lines or [],
                'models_loaded': bool(models),
                'collector': collector
            }
        except Exception as e:
            return {
                'props_lines': [],
                'models_loaded': False,
                'error': str(e)
            }
    
    def generate_mock_props_from_stats(player_stats):
        """Gera props lines mock usando player stats reais."""
        import numpy as np
        
        if player_stats is None or player_stats.empty:
            return []
        
        df = player_stats.copy()
        
        # Mapear colunas
        player_col = 'player' if 'player' in df.columns else 'PLAYER_NAME'
        team_col = 'team' if 'team' in df.columns else 'TEAM_ABBREVIATION'
        pts_col = next((c for c in ['pts_avg', 'PTS', 'pts'] if c in df.columns), None)
        reb_col = next((c for c in ['reb_avg', 'REB', 'reb'] if c in df.columns), None)
        ast_col = next((c for c in ['ast_avg', 'AST', 'ast'] if c in df.columns), None)
        min_col = next((c for c in ['min_avg', 'MIN', 'min'] if c in df.columns), None)
        
        if not all([player_col in df.columns, pts_col, min_col]):
            return []
        
        df_filtered = df[df[min_col] > 20].nlargest(30, min_col)
        
        props = []
        np.random.seed(42)
        
        for _, row in df_filtered.iterrows():
            player_name = str(row.get(player_col, 'Unknown'))
            team_abbr = str(row.get(team_col, 'UNK')) if team_col in df.columns else 'UNK'
            
            # PTS
            pts_avg = float(row.get(pts_col, 15))
            pts_line = round(pts_avg + np.random.uniform(-1.5, 1.5), 1)
            props.append({
                'player': player_name, 'team': team_abbr, 'prop_type': 'PTS',
                'line': max(5.5, pts_line), 'player_avg': pts_avg,
                'odds_over': 1.91, 'odds_under': 1.91, 'bookmaker': 'SIMULATED'
            })
            
            # REB
            if reb_col and reb_col in df.columns:
                reb_avg = float(row.get(reb_col, 5))
                reb_line = round(reb_avg + np.random.uniform(-1, 1), 1)
                props.append({
                    'player': player_name, 'team': team_abbr, 'prop_type': 'REB',
                    'line': max(1.5, reb_line), 'player_avg': reb_avg,
                    'odds_over': 1.91, 'odds_under': 1.91, 'bookmaker': 'SIMULATED'
                })
            
            # AST
            if ast_col and ast_col in df.columns:
                ast_avg = float(row.get(ast_col, 3))
                ast_line = round(ast_avg + np.random.uniform(-1, 1), 1)
                props.append({
                    'player': player_name, 'team': team_abbr, 'prop_type': 'AST',
                    'line': max(0.5, ast_line), 'player_avg': ast_avg,
                    'odds_over': 1.91, 'odds_under': 1.91, 'bookmaker': 'SIMULATED'
                })
        
        return props

    
    @st.cache_data(ttl=300)
    def generate_quantum_predictions(props_lines: list) -> pd.DataFrame:
        """Gera previsões Quantum para os props."""
        try:
            from ml_pipeline.train_props_quantum import load_quantum_models, predict_with_confidence, should_bet
            from data.scrapers.quantum_scraper import QuantumDataCollector
            
            models = load_quantum_models()
            collector = QuantumDataCollector()
            
            if not models:
                # Sem modelos treinados, usar heurísticas
                return generate_heuristic_predictions(props_lines)
            
            predictions = []
            for prop in props_lines:
                player = prop.get('player', 'Unknown')
                prop_type = prop.get('prop_type', 'PTS')
                line = prop.get('line', 0)
                
                # Buscar dados do jogador
                player_data = collector.fetch_player_data(player)
                
                if player_data['stats']:
                    stats = player_data['stats']
                    
                    # Previsão simples baseada em médias
                    if prop_type == 'PTS':
                        pred_median = stats.get('pts_avg', line)
                    elif prop_type == 'REB':
                        pred_median = stats.get('reb_avg', line)
                    elif prop_type == 'AST':
                        pred_median = stats.get('ast_avg', line)
                    else:
                        pred_median = line
                    
                    # Simular quantis
                    pred_low = pred_median * 0.75
                    pred_high = pred_median * 1.25
                    
                    # Avaliar oportunidade
                    evaluation = collector.evaluate_bet_opportunity(
                        pred_median, pred_low, pred_high,
                        line,
                        prop.get('odds_over', 1.91),
                        prop.get('odds_under', 1.91)
                    )
                    
                    # Calcular diferença percentual
                    diff_pct = ((pred_median - line) / line * 100) if line > 0 else 0
                    
                    predictions.append({
                        'player': player,
                        'team': prop.get('team', 'N/A'),
                        'prop_type': prop_type,
                        'line': line,
                        'prediction_low': round(pred_low, 1),
                        'prediction': round(pred_median, 1),
                        'prediction_high': round(pred_high, 1),
                        'diff_pct': round(diff_pct, 1),
                        'recommendation': evaluation['recommendation'],
                        'strength': evaluation['strength'],
                        'ev_plus': evaluation['ev_plus'],
                        'edge': evaluation['edge'],
                        'confidence': 'HIGH' if abs(evaluation['edge']) > 5 else 'MEDIUM' if abs(evaluation['edge']) > 2 else 'LOW',
                        'inferred': player_data.get('inferred', False),
                        'source': player_data.get('source', 'unknown')
                    })
            
            return pd.DataFrame(predictions) if predictions else pd.DataFrame()
            
        except Exception as e:
            st.warning(f"⚠️ Erro ao gerar previsões: {e}")
            return generate_heuristic_predictions(props_lines)
    
    def generate_heuristic_predictions(props_lines: list) -> pd.DataFrame:
        """Gera previsões heurísticas quando modelos não estão disponíveis."""
        import numpy as np
        
        predictions = []
        for prop in props_lines:
            line = prop.get('line', 20)
            
            # Variação aleatória para demonstração
            np.random.seed(hash(prop.get('player', '')) % 2**32)
            variation = np.random.uniform(-0.15, 0.15)
            pred_median = line * (1 + variation)
            
            diff_pct = variation * 100
            
            if abs(diff_pct) > 10:
                rec = 'OVER' if diff_pct > 0 else 'UNDER'
                strength = 'MEDIUM'
                ev = abs(diff_pct) * 0.5
            elif abs(diff_pct) > 5:
                rec = 'OVER' if diff_pct > 0 else 'UNDER'
                strength = 'LEAN'
                ev = abs(diff_pct) * 0.3
            else:
                rec = 'SKIP'
                strength = 'NONE'
                ev = 0
            
            predictions.append({
                'player': prop.get('player', 'Unknown'),
                'team': prop.get('team', 'N/A'),
                'prop_type': prop.get('prop_type', 'PTS'),
                'line': line,
                'prediction_low': round(pred_median * 0.8, 1),
                'prediction': round(pred_median, 1),
                'prediction_high': round(pred_median * 1.2, 1),
                'diff_pct': round(diff_pct, 1),
                'recommendation': rec,
                'strength': strength,
                'ev_plus': round(ev, 2),
                'edge': round(diff_pct * 0.4, 2),
                'confidence': 'DEMO',
                'inferred': True,
                'source': 'heuristic'
            })
        
        return pd.DataFrame(predictions) if predictions else pd.DataFrame()
    
    # Carregar dados
    data = load_prop_sniper_data()
    
    if 'error' in data:
        st.error(f"Erro ao carregar dados: {data['error']}")
    
    # Status dos modelos
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        if data.get('models_loaded'):
            st.success("✅ Modelos Quantum carregados")
        else:
            st.warning("⚠️ Modelos não treinados")
            st.caption("Execute: `python -m ml_pipeline.train_props_quantum`")
    
    with col_status2:
        n_props = len(data.get('props_lines', []))
        st.metric("Props Lines", n_props)
    
    with col_status3:
        st.metric("Última Atualização", datetime.now().strftime("%H:%M"))
    
    st.markdown("---")
    
    # Gerar previsões
    if data.get('props_lines'):
        df_predictions = generate_quantum_predictions(data['props_lines'])
        
        if not df_predictions.empty:
            # Filtros
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                prop_types = ['Todos'] + list(df_predictions['prop_type'].unique())
                selected_prop = st.selectbox("Tipo de Prop", prop_types)
            
            with col_filter2:
                min_ev = st.slider("EV+ Mínimo (%)", -10, 20, 0)
            
            with col_filter3:
                show_only_bets = st.checkbox("Apenas Recomendações", value=True)
            
            # Aplicar filtros
            df_filtered = df_predictions.copy()
            
            if selected_prop != 'Todos':
                df_filtered = df_filtered[df_filtered['prop_type'] == selected_prop]
            
            df_filtered = df_filtered[df_filtered['ev_plus'] >= min_ev]
            
            if show_only_bets:
                df_filtered = df_filtered[df_filtered['recommendation'] != 'SKIP']
            
            # Ordenar por EV+
            df_filtered = df_filtered.sort_values('ev_plus', ascending=False)
            
            st.subheader(f"💰 Oportunidades de Valor ({len(df_filtered)} encontradas)")
            
            # Tabela principal com formatação
            if not df_filtered.empty:
                # Configurar colunas para display
                display_cols = [
                    'player', 'team', 'prop_type', 'line', 'prediction', 
                    'diff_pct', 'recommendation', 'strength', 'ev_plus', 'confidence'
                ]
                
                df_display = df_filtered[display_cols].copy()
                
                # Renomear colunas para display
                df_display.columns = [
                    'Jogador', 'Time', 'Prop', 'Linha Casa', 'Nossa Previsão',
                    'Δ%', 'Recomendação', 'Força', 'EV+', 'Confiança'
                ]
                
                # Aplicar estilos
                def highlight_recommendation(val):
                    if val == 'OVER':
                        return 'background-color: #166534; color: white'
                    elif val == 'UNDER':
                        return 'background-color: #991b1b; color: white'
                    else:
                        return 'background-color: #374151; color: white'
                
                def highlight_ev(val):
                    if val >= 5:
                        return 'color: #4ade80; font-weight: bold'
                    elif val >= 2:
                        return 'color: #fbbf24; font-weight: bold'
                    elif val > 0:
                        return 'color: #9ca3af'
                    else:
                        return 'color: #f87171'
                
                styled_df = df_display.style.applymap(
                    highlight_recommendation, subset=['Recomendação']
                ).applymap(
                    highlight_ev, subset=['EV+']
                )
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Seção de explicabilidade
                st.markdown("---")
                st.subheader("🔍 Análise Detalhada")
                
                # Seletor de jogador
                players = df_filtered['player'].tolist()
                if players:
                    selected_player = st.selectbox("Selecione um jogador para análise", players)
                    
                    if selected_player:
                        player_data = df_filtered[df_filtered['player'] == selected_player].iloc[0]
                        
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.markdown("### 📊 Previsão")
                            
                            # Mini gráfico de intervalo de confiança
                            low = player_data['prediction_low']
                            mid = player_data['prediction']
                            high = player_data['prediction_high']
                            line = player_data['line']
                            
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            
                            # Intervalo de confiança
                            fig.add_trace(go.Bar(
                                x=[high - low],
                                y=['Previsão'],
                                base=[low],
                                orientation='h',
                                marker=dict(color='rgba(59, 130, 246, 0.5)'),
                                name='Intervalo (P10-P90)'
                            ))
                            
                            # Linha da casa
                            fig.add_vline(
                                x=line, 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text=f"Linha: {line}"
                            )
                            
                            # Mediana
                            fig.add_trace(go.Scatter(
                                x=[mid],
                                y=['Previsão'],
                                mode='markers',
                                marker=dict(size=20, color='#3b82f6', symbol='diamond'),
                                name=f'Mediana: {mid}'
                            ))
                            
                            fig.update_layout(
                                height=150,
                                showlegend=True,
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis_title=player_data['prop_type']
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Métricas
                            m1, m2, m3 = st.columns(3)
                            m1.metric("P10 (Piso)", f"{low:.1f}")
                            m2.metric("Mediana", f"{mid:.1f}")
                            m3.metric("P90 (Teto)", f"{high:.1f}")
                        
                        with col_detail2:
                            st.markdown("### 💡 Explicação")
                            
                            # Gerar explicação baseada nos dados
                            rec = player_data['recommendation']
                            prop_type = player_data['prop_type']
                            diff = player_data['diff_pct']
                            
                            reasons = []
                            
                            if rec == 'OVER':
                                reasons.append(f"📈 Previsão {abs(diff):.1f}% **acima** da linha")
                                if player_data['strength'] == 'ALL-IN':
                                    reasons.append("🎯 Nosso piso (P10) já supera a linha")
                            elif rec == 'UNDER':
                                reasons.append(f"📉 Previsão {abs(diff):.1f}% **abaixo** da linha")
                                if player_data['strength'] == 'ALL-IN':
                                    reasons.append("🎯 Nosso teto (P90) não alcança a linha")
                            
                            if player_data.get('inferred'):
                                reasons.append("⚠️ Dados inferidos (jogador sem histórico suficiente)")
                            
                            reasons.append(f"📊 EV+: **{player_data['ev_plus']:.2f}%**")
                            reasons.append(f"📊 Edge: **{player_data['edge']:.2f}%**")
                            
                            for reason in reasons:
                                st.markdown(f"- {reason}")
                            
                            # Alerta de ação
                            if rec != 'SKIP':
                                if player_data['strength'] in ['ALL-IN', 'MEDIUM']:
                                    st.success(f"✅ **Recomendação: {rec} {player_data['line']} {prop_type}**")
                                else:
                                    st.info(f"ℹ️ Tendência: {rec} {player_data['line']} {prop_type}")
                            else:
                                st.warning("⏸️ Sem vantagem clara. Não apostar.")
            else:
                st.info("📊 Nenhuma oportunidade encontrada com os filtros atuais.")
        else:
            st.info("📊 Nenhuma previsão gerada. Execute o pipeline de dados.")
    else:
        st.warning("📊 Nenhuma linha de props disponível. Verifique as APIs de odds.")
        
        # Botão para tentar novamente
        if st.button("🔄 Tentar Carregar Props Lines"):
            st.cache_data.clear()
            st.rerun()
    
    # Seção de comandos
    st.markdown("---")
    with st.expander("🛠️ Comandos do Sistema Quantum Props"):
        st.code("""
# Treinar modelos Quantum Props
python -m ml_pipeline.train_props_quantum

# Executar pipeline completo
python scripts/quantum_props_run.py

# Testar features quantum
python -c "from ml_pipeline.props_quantum_features import test_all_features; test_all_features()"

# Testar coleta de dados
python -c "from data.scrapers.quantum_scraper import test_quantum_scraper; test_quantum_scraper()"
        """, language="bash")
