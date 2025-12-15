"""
UI Components para Gestão de Banca no Web App
==============================================

Componentes do Streamlit para exibir stakes sugeridos,
alertas de correlação e monitor de CLV.

Uso:
    import streamlit as st
    from betting.web_ui import render_bankroll_management
    
    render_bankroll_management(daily_games, bankroll, kelly_fraction)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SafeKellyResult:
    """Resultado do cálculo de stake seguro."""
    stake: float
    stake_pct: float
    recommendation: str  # 'BET', 'SKIP', 'REDUCE'
    reason: Optional[str] = None


class SafeKellyStrategy:
    """
    AUDIT FIX: Estratégia Kelly com proteções contra ruína.
    
    Proteções implementadas:
    1. Stop-loss diário: Máx 10% da banca por dia
    2. Ajuste por perdas consecutivas: Reduz 20% a cada 3 perdas
    3. Hard cap dinâmico: Reduz proporcional ao drawdown
    4. Validação de odds: Rejeita odds estimadas
    5. Edge mínimo: Requer 5% de edge para apostar
    """
    
    def __init__(
        self,
        bankroll: float,
        kelly_fraction: float = 0.25,
        daily_stop_loss_pct: float = 0.10,
        min_edge_pct: float = 0.05,
        min_confidence: float = 0.60,
        hard_cap_pct: float = 0.03,
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.daily_stop_loss_pct = daily_stop_loss_pct
        self.min_edge_pct = min_edge_pct
        self.min_confidence = min_confidence
        self.base_hard_cap_pct = hard_cap_pct
        
        # Estado de proteção
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.peak_bankroll = bankroll
        
    @property
    def daily_stop_loss_remaining(self) -> float:
        """Quanto ainda pode perder hoje."""
        return max(0, self.daily_stop_loss_pct * self.initial_bankroll - self.daily_loss)
    
    @property
    def drawdown_pct(self) -> float:
        """Drawdown atual desde o pico."""
        if self.peak_bankroll <= 0:
            return 0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll
    
    @property
    def effective_kelly_fraction(self) -> float:
        """Kelly efetivo após ajustes por perdas consecutivas."""
        # Reduz 20% a cada 3 perdas consecutivas
        penalty = (self.consecutive_losses // 3) * 0.20
        return max(0.05, self.kelly_fraction * (1 - penalty))
    
    @property
    def dynamic_hard_cap(self) -> float:
        """Hard cap dinâmico que reduz com drawdown."""
        # Cap reduz 50% para cada 10% de drawdown
        drawdown_penalty = 1 - (self.drawdown_pct * 5)
        return self.base_hard_cap_pct * max(0.2, drawdown_penalty)
    
    def calculate_stake(
        self, 
        prob: float, 
        odds: float, 
        is_odds_estimated: bool = False,
        confidence: float = 0.70,
        odds_source: str = 'unknown'  # AUDITORIA P0-A: Novo parâmetro
    ) -> SafeKellyResult:
        """
        Calcula stake seguro considerando todas as proteções.
        
        Args:
            prob: Probabilidade do modelo (0-100)
            odds: Odds decimais
            is_odds_estimated: Se True, odds são Fair Odds (não reais)
            confidence: Nível de confiança do modelo (0-1)
            odds_source: Nome da casa de apostas ou 'estimated'/'unknown'
            
        Returns:
            SafeKellyResult com stake calculado e recomendação
        """
        # ========================================
        # AUDITORIA P0-A: Rejeitar odds não verificadas
        # ========================================
        FONTES_SUSPEITAS = ['estimated', 'fair', 'unknown', 'Estimado', 'Fair', 'Calculado', '']
        
        if is_odds_estimated or odds_source in FONTES_SUSPEITAS:
            return SafeKellyResult(
                stake=0, 
                stake_pct=0, 
                recommendation='SKIP',
                reason='🚫 BLOQUEADO: Odds não verificadas (estimada/fictícia). Aguarde odds reais de uma casa de apostas.'
            )
        
        # AUDITORIA P0-A: Validação de range realista (1.01 a 50.0)
        if not (1.01 <= odds <= 50.0):
            return SafeKellyResult(
                stake=0, 
                stake_pct=0,
                recommendation='SKIP',
                reason=f'⚠️ Odds fora do range válido (1.01-50.0): {odds:.2f}'
            )
        
        # Proteção 2: Verificar stop-loss diário
        if self.daily_stop_loss_remaining <= 0:
            return SafeKellyResult(
                stake=0, stake_pct=0,
                recommendation='SKIP',
                reason='🛑 Stop-loss diário atingido'
            )
        
        # Proteção 3: Verificar confiança mínima
        if confidence < self.min_confidence:
            return SafeKellyResult(
                stake=0, stake_pct=0,
                recommendation='SKIP',
                reason=f'📉 Confiança {confidence:.0%} < mínimo {self.min_confidence:.0%}'
            )
        
        # Calcular edge
        prob_decimal = prob / 100
        implied_prob = 1 / odds if odds > 0 else 0
        edge = prob_decimal - implied_prob
        
        # Proteção 4: Edge mínimo
        if edge < self.min_edge_pct:
            return SafeKellyResult(
                stake=0, stake_pct=0,
                recommendation='SKIP',
                reason=f'📊 Edge {edge:.1%} < mínimo {self.min_edge_pct:.0%}'
            )
        
        # Kelly: f* = (bp - q) / b
        b = odds - 1
        q = 1 - prob_decimal
        kelly_fraction_raw = (b * prob_decimal - q) / b if b > 0 else 0
        
        # Aplicar fração Kelly efetiva (ajustada por perdas)
        stake_pct = kelly_fraction_raw * self.effective_kelly_fraction
        
        # Proteção 5: Hard cap dinâmico
        stake_pct = min(stake_pct, self.dynamic_hard_cap)
        
        # Proteção 6: Limitar ao stop-loss restante
        max_stake_pct = self.daily_stop_loss_remaining / self.bankroll
        stake_pct = min(stake_pct, max_stake_pct)
        
        stake = stake_pct * self.bankroll
        
        # Gerar recomendação
        recommendation = 'BET'
        reason = None
        
        if self.consecutive_losses >= 3:
            recommendation = 'REDUCE'
            reason = f'⚡ Kelly reduzido por {self.consecutive_losses} perdas consecutivas'
        
        if self.drawdown_pct > 0.15:
            recommendation = 'REDUCE'
            reason = f'📉 Hard cap reduzido por drawdown de {self.drawdown_pct:.0%}'
        
        return SafeKellyResult(
            stake=round(stake, 2),
            stake_pct=stake_pct,
            recommendation=recommendation,
            reason=reason
        )
    
    def record_result(self, won: bool, stake: float):
        """Registra resultado de uma aposta."""
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.daily_loss += stake
            
        # Atualizar pico se necessário
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
    
    def reset_daily_state(self):
        """Reseta estado diário (chamar no início de cada dia)."""
        self.daily_loss = 0.0
    
    def adjust_for_correlation(self, bets: list) -> list:
        """
        Ajusta stakes para apostas correlacionadas.
        
        P0-A FIX: Implementação mínima para compatibilidade.
        Reduz stake em 20% quando há múltiplas apostas no mesmo jogo.
        
        Args:
            bets: Lista de dicts com campos 'stake_amount', 'stake_pct', 
                  'game_id', 'team', 'market'
                  
        Returns:
            Lista de bets com stakes ajustados e 'correlation_alert' adicionado
        """
        if len(bets) <= 1:
            for bet in bets:
                bet['correlation_alert'] = None
            return bets
        
        # Detectar correlação simples: múltiplas apostas = maior risco
        correlation_factor = 0.80  # Reduz em 20%
        
        adjusted_bets = []
        for bet in bets:
            adjusted_bet = bet.copy()
            adjusted_bet['stake_amount'] = bet['stake_amount'] * correlation_factor
            adjusted_bet['stake_pct'] = bet['stake_pct'] * correlation_factor
            adjusted_bet['correlation_alert'] = (
                f"⚠️ Stake reduzido 20% (correlação: {len(bets)} apostas simultâneas)"
            )
            adjusted_bets.append(adjusted_bet)
        
        return adjusted_bets


def render_bankroll_management(daily_games, bankroll, kelly_fraction=0.25):
    """
    Renderiza seção completa de gestão de banca no Streamlit.
    
    Args:
        daily_games: DataFrame com jogos do dia
        bankroll: Banca atual (float)
        kelly_fraction: Fração do Kelly a usar (default: 0.25)
    """
    st.header("💰 Gestão de Banca Profissional")
    
    # AUDIT FIX: Usar SafeKellyStrategy
    safe_strategy = SafeKellyStrategy(
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        daily_stop_loss_pct=0.10,
        min_edge_pct=0.05,
        min_confidence=0.60,
        hard_cap_pct=0.03
    )
    
    # Métricas de configuração com proteções
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Banca", f"R$ {bankroll:,.2f}")
    with col2:
        st.metric(
            "Kelly Efetivo", 
            f"{safe_strategy.effective_kelly_fraction:.0%}",
            help="Fração ajustada por perdas consecutivas"
        )
    with col3:
        st.metric(
            "Stop-Loss Restante",
            f"R$ {safe_strategy.daily_stop_loss_remaining:,.2f}",
            help="Máximo que pode perder hoje (10% da banca)"
        )
    with col4:
        st.metric(
            "Hard Cap",
            f"{safe_strategy.dynamic_hard_cap:.1%}",
            help="Limite máximo por aposta (dinâmico)"
        )
    
    st.markdown("---")
    
    if daily_games.empty:
        st.info("Sem jogos disponíveis para análise hoje.")
        return
    
    try:
        from market.odds_shopping import compare_lines
        
        # P0-A FIX: Usar SafeKellyStrategy local (instanciada acima como safe_strategy)
        # Remove dependência de KellyCriterionStrategy legada
        
        st.subheader("🎯 Oportunidades Identificadas")
        
        # Processar jogos
        all_opportunities = []
        
        for _, game in daily_games.iterrows():
            # Mock de odds baseado nas probs
            prob_home = game.get('prob_home', 50) / 100.0
            prob_away = game.get('prob_away', 50) / 100.0
            
            # Odds justas com margem
            fair_odds_home = (1 / prob_home) * 0.95 if prob_home > 0 else 2.0
            fair_odds_away = (1 / prob_away) * 0.95 if prob_away > 0 else 2.0
            
            market_odds = [{
                'home_team_id': game['home_team'],
                'bookmakers': [{
                    'title': 'Estimado',
                    'markets': [{
                        'key': 'h2h',
                        'outcomes': [
                            {'name': game['home_team'], 'price': fair_odds_home},
                            {'name': game['away_team'], 'price': fair_odds_away}
                        ]
                    }]
                }]
            }]
            
            prediction = {
                'Casa': game['home_team'],
                'Prob Casa %': game.get('prob_home', 50),
                'Spread Previsto': game.get('spread_home', 0)
            }
            
        # V21 RISK FIX: Calculate stakes locally using SafeKellyStrategy
        # This prevents "Estimado" odds from generating positive stakes
        # because we will pass is_odds_estimated=True/False correctly.
        
        # 1. Get Opportunities (WITHOUT internal staking)
        # Pass calculate_stakes=False to avoid legacy logic polluting the results
        opps = compare_lines(prediction, market_odds, bankroll, calculate_stakes=False)
        all_opportunities.extend(opps)
        
        for opp in all_opportunities:
            # 2. Explicit Staking Calculation Check
            is_estimated = (opp.get('bookie') == 'Estimado')
            
            # Use SafeKellyStrategy to calculate strict stake
            # This respects dynamic hard caps and blocks estimated odds
            stake_result = safe_strategy.calculate_stake(
                prob=opp.get('model_prob', 0),
                odds=opp.get('odds', 0),
                is_odds_estimated=is_estimated,
                confidence=0.80, # Default confidence if not passed
                odds_source=opp.get('bookie', 'unknown')
            )
            
            opp['suggested_stake_amount'] = stake_result.stake
            opp['suggested_stake_pct'] = stake_result.stake_pct
            opp['recommendation'] = 'APOSTAR' if stake_result.recommendation == 'BET' else 'CONSIDERAR' if stake_result.recommendation == 'REDUCE' else 'IGNORE'
            opp['kelly_full'] = 0 # Not exposed in SafeKellyResult yet, irrelevant for UI safety
            opp['kelly_fraction'] = safe_strategy.effective_kelly_fraction
            
            # Add blocking reason if skipped
            if stake_result.recommendation == 'SKIP':
                opp['recommendation'] = 'EVITAR'
                # Optional: Add alert info if needed
        
        # Filter viable
        viable = [o for o in all_opportunities if o.get('recommendation') in ['APOSTAR', 'CONSIDERAR']]
        
        if not viable:
            st.info("🔍 Nenhuma oportunidade com edge positivo encontrada.")
            st.caption("Requer edge mínimo 5% e confidence 60%")
        else:
            _render_bet_cards(viable, kelly_fraction)
            _render_session_summary(viable, bankroll)
    
    except ImportError as e:
        st.error(f"❌ Módulo de staking indisponível: {e}")
        _render_fallback_ev(daily_games)
    
    # Monitor CLV
    st.markdown("---")
    _render_clv_monitor()


def _render_bet_cards(viable_bets, kelly_fraction):
    """Renderiza cards de apostas sugeridas."""
    for opp in viable_bets:
        with st.container():
            st.markdown("""
            <div style="background-color: #1f2937; padding: 15px; border-radius: 10px; 
                        border-left: 4px solid #4ade80; margin-bottom: 15px;">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"### {opp['team']} - {opp['market']}")
                st.markdown(f"**Odds:** {opp['odds']:.2f}")
                st.markdown(f"**Edge:** {opp['ev']:.1f}%")
            
            with col2:
                stake = opp.get('suggested_stake_amount', 0)
                stake_pct = opp.get('suggested_stake_pct', 0)
                
                st.markdown(f"### R$ {stake:.2f}")
                st.caption(f"{stake_pct:.2f}% da banca")
                
                kelly_full = opp.get('kelly_full', 0)
                kelly_frac = opp.get('kelly_fraction', 0)
                
                if kelly_full > 0:
                    st.progress(min(kelly_frac / 5.0, 1.0))
                    st.caption(f"Kelly: {kelly_full:.1f}% → {kelly_frac:.2f}% (Kelly/{int(1/kelly_fraction)})")
            
            with col3:
                if opp.get('recommendation') == 'APOSTAR':
                    st.markdown("**✅ APOSTAR**")
                else:
                    st.markdown("**⚠️ CONSIDERAR**")
            
            if opp.get('correlation_alert'):
                st.warning(opp['correlation_alert'])
            
            st.markdown("</div>", unsafe_allow_html=True)


def _render_session_summary(viable_bets, bankroll):
    """Renderiza resumo da sessão."""
    st.markdown("---")
    st.subheader("📊 Resumo da Sessão")
    
    total_stakes = sum(o.get('suggested_stake_amount', 0) for o in viable_bets)
    total_exp = (total_stakes / bankroll * 100) if bankroll > 0 else 0
    avg_ev = sum(o.get('ev', 0) for o in viable_bets) / len(viable_bets) if viable_bets else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Apostas Sugeridas", len(viable_bets))
    with col2:
        st.metric("Total Alocado", f"R$ {total_stakes:.2f}")
    with col3:
        st.metric("Exposição", f"{total_exp:.1f}%",
                 delta="Seguro" if total_exp < 10 else "Alto",
                 delta_color="normal" if total_exp < 10 else "inverse")
    with col4:
        st.metric("EV Médio", f"{avg_ev:.1f}%")


def _render_fallback_ev(daily_games):
    """Fallback se staking não disponível - mostrar apenas EV."""
    st.warning("Sistema de staking indisponível. Mostrando apenas EV.")
    
    suggestions = []
    for _, game in daily_games.iterrows():
        prob_home = game.get('prob_home', 0)
        prob_away = game.get('prob_away', 0)
        
        odds_home = game.get('odds_home', 100 / prob_home if prob_home > 0 else 0)
        odds_away = game.get('odds_away', 100 / prob_away if prob_away > 0 else 0)
        
        ev_home = (prob_home/100 * odds_home - 1) * 100 if odds_home > 0 else 0
        ev_away = (prob_away/100 * odds_away - 1) * 100 if odds_away > 0 else 0
        
        if ev_home > 0:
            suggestions.append({
                'Time': game['home_team'],
                'Tipo': 'ML Casa',
                'Prob': f"{prob_home:.1f}%",
                'EV': round(ev_home, 1)
            })
        
        if ev_away > 0:
            suggestions.append({
                'Time': game['away_team'],
                'Tipo': 'ML Visitante',
                'Prob': f"{prob_away:.1f}%",
                'EV': round(ev_away, 1)
            })
    
    if suggestions:
        st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)


def _render_clv_monitor():
    """Renderiza monitor de CLV."""
    st.subheader("📈 Monitor CLV (Closing Line Value)")
    
    try:
        clv_file = Path('data/bet_tracking.csv')
        
        if not clv_file.exists():
            st.info("💡 Nenhum histórico encontrado. Execute `python scripts/monitor_clv.py` para rastrear CLV.")
            return
        
        df_bets = pd.read_csv(clv_file)
        
        if df_bets.empty or 'clv' not in df_bets.columns:
            st.info("Arquivo existe mas sem dados de CLV.")
            return
        
        df_clv = df_bets[df_bets['clv'].notna()].copy()
        
        if df_clv.empty:
            st.info("Nenhuma aposta com CLV calculado ainda.")
            return
        
        # Métricas
        avg_clv = df_clv['clv'].mean() * 100
        positive = len(df_clv[df_clv['clv'] > 0])
        total = len(df_clv)
        positive_pct = (positive / total * 100) if total > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CLV Médio", f"{avg_clv:+.2f}%",
                     delta="Bom" if avg_clv > 0 else "Revisar",
                     delta_color="normal" if avg_clv > 0 else "inverse")
        with col2:
            st.metric("Apostas Rastreadas", total)
        with col3:
            st.metric("CLV Positivo", f"{positive_pct:.0f}%")
        
        # Gráfico
        if len(df_clv) >= 5:
            fig = px.line(
                df_clv.tail(20),
                x='date',
                y='clv',
                title='Evolução do CLV (Últimas 20 Apostas)'
            )
            fig.update_traces(line_color='#4ade80')
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.caption(f"CLV Monitor indisponível: {str(e)[:100]}")


def render_combo_generator(daily_games, player_props=None, min_ev=5.0):
    """
    Renderiza o gerador de combos de apostas.
    
    Args:
        daily_games: DataFrame com previsões de jogos
        player_props: DataFrame com previsões de player props (opcional)
        min_ev: EV mínimo para filtrar combos (default: 5%)
    """
    from betting.combo_generator import generate_smart_combos
    
    st.markdown("---")
    st.header("🎯 Gerador de Combos e Parlays")
    st.caption("Crie combinações inteligentes de apostas para maximizar valor")
    
    if daily_games.empty:
        st.info("Sem jogos disponíveis para gerar combos.")
        return
    
    # Tabs para diferentes tipos de combos
    combo_tab1, combo_tab2, combo_tab3 = st.tabs([
        "🔥 Combos Sugeridos",
        "🛠️ Criador Personalizado",
        "📊 Parlays Multi-Time"
    ])
    
    try:
        # Gerar todos os combos
        all_combos = generate_smart_combos(
            daily_games,
            player_props,
            min_ev=min_ev,
            include_team_player=True,
            include_parlays=True
        )
        
        # Tab 1: Combos Sugeridos (Top de todos os tipos)
        with combo_tab1:
            _render_suggested_combos(all_combos, min_ev, daily_games)
        
        # Tab 2: Criador Personalizado
        with combo_tab2:
            _render_combo_builder(daily_games, player_props)
        
        # Tab 3: Parlays Multi-Time
        with combo_tab3:
            _render_multi_team_parlays(all_combos)
    
    except Exception as e:
        st.error(f"Erro ao gerar combos: {e}")
        st.caption("Verifique se os dados estão disponíveis.")


def _render_suggested_combos(all_combos, min_ev, daily_games):
    """Renderiza combos sugeridos automaticamente."""
    st.subheader("🔥 Top Combos de Maior EV")
    st.caption(f"Filtrado por EV ≥ {min_ev}%")
    
    # Combinar todos os combos e ordenar por EV
    all_suggestions = []
    
    if all_combos.get('team_player'):
        all_suggestions.extend(all_combos['team_player'])
    if all_combos.get('parlay_2'):
        all_suggestions.extend(all_combos['parlay_2'])
    if all_combos.get('parlay_3'):
        all_suggestions.extend(all_combos['parlay_3'])
    if all_combos.get('parlay_4'):
        all_suggestions.extend(all_combos['parlay_4'])
    
    if not all_suggestions:
        st.info("Nenhum combo encontrado com os critérios especificados.")
        # Debug info
        st.caption(f"📊 Debug: {len(daily_games)} jogos disponíveis")
        st.caption(f"🎯 Team+Player combos: {len(all_combos.get('team_player', []))}")
        st.caption(f"📊 2-team parlays: {len(all_combos.get('parlay_2', []))}")
        st.caption(f"📈 3-team parlays: {len(all_combos.get('parlay_3', []))}")
        st.caption(f"🚀 4-team parlays: {len(all_combos.get('parlay_4', []))}")
        st.caption("Tente diminuir o EV mínimo ou verifique se há jogos disponíveis.")
        return
    
    # Ordenar por EV
    all_suggestions.sort(key=lambda x: x['ev'], reverse=True)
    
    # Filtro de tipo
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        combo_type_filter = st.selectbox(
            "Tipo de Combo",
            ["Todos", "Time + Jogador", "Parlay 2 Times", "Parlay 3 Times", "Parlay 4 Times"],
            key="combo_type_filter"
        )
    
    with col_filter2:
        max_display = st.slider("Máximo de combos", 5, 20, 10, key="max_combos_display")
    
    # Filtrar por tipo
    filtered_combos = all_suggestions
    
    # Debug: mostrar tipos disponíveis
    if combo_type_filter != "Todos":
        types_count = {}
        for c in all_suggestions:
            t = c['type']
            types_count[t] = types_count.get(t, 0) + 1
        st.caption(f"🔍 Debug: Tipos disponíveis: {types_count}")
    
    if combo_type_filter == "Time + Jogador":
        filtered_combos = [c for c in all_suggestions if c['type'] == 'team_player']
    elif combo_type_filter == "Parlay 2 Times":
        filtered_combos = [c for c in all_suggestions if c['type'] == '2_team_parlay']
    elif combo_type_filter == "Parlay 3 Times":
        filtered_combos = [c for c in all_suggestions if c['type'] == '3_team_parlay']
    elif combo_type_filter == "Parlay 4 Times":
        filtered_combos = [c for c in all_suggestions if c['type'] == '4_team_parlay']
    
    # Exibir combos
    if not filtered_combos:
        st.info(f"Nenhum combo do tipo '{combo_type_filter}' encontrado.")
    else:
        for i, combo in enumerate(filtered_combos[:max_display]):
            _render_combo_card(combo, i)


def _render_combo_card(combo, index):
    """Renderiza card individual de um combo."""
    # Determinar cor baseado em EV
    ev = combo['ev']
    if ev >= 20:
        border_color = "#4ade80"  # Verde forte
        ev_label = "🔥 EXCELENTE"
    elif ev >= 10:
        border_color = "#facc15"  # Amarelo
        ev_label = "💎 BOM"
    else:
        border_color = "#9ca3af"  # Cinza
        ev_label = "⚡ VALOR"
    
    # Tipo badge
    type_map = {
        'team_player': '🏀 Time + Jogador',
        '2_team_parlay': '📊 Parlay 2x',
        '3_team_parlay': '📈 Parlay 3x',
        '4_team_parlay': '🚀 Parlay 4x',
        'custom': '🛠️ Customizado'
    }
    type_label = type_map.get(combo['type'], combo['type'])
    
    with st.container():
        st.markdown(f"""
        <div style="background-color: #1f2937; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid {border_color}; margin-bottom: 15px;">
        """, unsafe_allow_html=True)
        
        # Header
        col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
        with col_h1:
            st.markdown(f"### {combo['description']}")
        with col_h2:
            st.markdown(f"<span style='background-color: {border_color}20; color: {border_color}; "
                       f"padding: 4px 8px; border-radius: 4px; font-weight: bold;'>{type_label}</span>",
                       unsafe_allow_html=True)
        with col_h3:
            st.markdown(f"<span style='background-color: {border_color}20; color: {border_color}; "
                       f"padding: 4px 8px; border-radius: 4px; font-weight: bold;'>{ev_label}</span>",
                       unsafe_allow_html=True)
        
        # Components
        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
        for comp in combo['components']:
            event = comp['event']
            prob = comp['prob'] * 100
            odd = comp['odd']
            
            st.markdown(f"""
            <div style='padding: 5px 10px; background-color: #111827; border-radius: 5px; margin: 5px 0;'>
                <span style='color: #e0e0e0;'>✓ {event}</span>
                <span style='color: #9ca3af; margin-left: 10px;'>Prob: {prob:.1f}% | Odd: {odd:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Metrics
        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Probabilidade Combinada", f"{combo['combined_prob']*100:.1f}%")
        with col_m2:
            st.metric("Odd Combinada", f"{combo['combined_odd']:.2f}")
        with col_m3:
            st.metric("Expected Value", f"{ev:+.1f}%",
                     delta="Positivo" if ev > 0 else "Negativo")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Copy button
        combo_text = f"{combo['description']} | Odd: {combo['combined_odd']:.2f} | Prob: {combo['combined_prob']*100:.1f}% | EV: {ev:+.1f}%"
        button_key = f"copy_combo_{index}"
        if st.button("📋 Copiar Combo", key=button_key):
            st.code(combo_text, language=None)
            st.success("✅ Copiado! Cole onde quiser.")
        
        st.markdown("</div>", unsafe_allow_html=True)


def _render_combo_builder(daily_games, player_props):
    """Renderiza interface de criação personalizada de combos."""
    st.subheader("🛠️ Crie Seu Próprio Combo")
    st.caption("Selecione os eventos que deseja combinar")
    
    # Preparar opções de picks
    available_picks = []
    
    # Adicionar picks de times
    for _, game in daily_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        prob_home = game.get('prob_home', 50) / 100.0
        prob_away = game.get('prob_away', 50) / 100.0
        odds_home = game.get('odds_home', 0) or (1 / prob_home if prob_home > 0.01 else 2.0)
        odds_away = game.get('odds_away', 0) or (1 / prob_away if prob_away > 0.01 else 2.0)
        
        available_picks.append({
            'event': f'{home_team} vence (vs {away_team})',
            'prob': prob_home,
            'odd': odds_home,
            'type': 'team'
        })
        
        available_picks.append({
            'event': f'{away_team} vence (@ {home_team})',
            'prob': prob_away,
            'odd': odds_away,
            'type': 'team'
        })
    
    # Adicionar player props (se disponível)
    if player_props is not None and not player_props.empty and 'player' in player_props.columns:
        for _, prop in player_props.iterrows():
            player_name = prop.get('player', 'Unknown')
            stat_type = prop.get('stat_type', 'PTS')
            line = prop.get('line', 0)
            prob_over = prop.get('prob_over', 50) / 100.0
            odd_prop = 1 / prob_over if prob_over > 0.01 else 2.0
            
            available_picks.append({
                'event': f'{player_name} {stat_type} Over {line:.1f}',
                'prob': prob_over,
                'odd': odd_prop,
                'type': 'player_prop'
            })
    
    # Multi-select
    selected_events = st.multiselect(
        "Selecione os eventos para combinar (mínimo 2)",
        options=[pick['event'] for pick in available_picks],
        key="custom_combo_selector"
    )
    
    if len(selected_events) >= 2:
        # Filtrar picks selecionados
        selected_picks = [pick for pick in available_picks if pick['event'] in selected_events]
        
        # Criar combo customizado
        from betting.combo_generator import create_custom_combo
        custom_combo = create_custom_combo(selected_picks)
        
        if custom_combo:
            st.markdown("### Preview do Combo")
            _render_combo_card(custom_combo, 999)
        else:
            st.error("Erro ao criar combo. Verifique os dados selecionados.")
    else:
        st.info("👆 Selecione pelo menos 2 eventos acima para criar um combo.")


def _render_multi_team_parlays(all_combos):
    """Renderiza parlays multi-time organizados."""
    st.subheader("📊 Parlays Multi-Time")
    st.caption("Combos de múltiplos times ordenados por EV")
    
    # Tabs por tamanho de parlay
    if all_combos.get('parlay_2') or all_combos.get('parlay_3') or all_combos.get('parlay_4'):
        parlay_tab1, parlay_tab2, parlay_tab3 = st.tabs([
            f"2 Times ({len(all_combos.get('parlay_2', []))})",
            f"3 Times ({len(all_combos.get('parlay_3', []))})",
            f"4 Times ({len(all_combos.get('parlay_4', []))})"
        ])
        
        with parlay_tab1:
            if all_combos.get('parlay_2'):
                st.caption(f"Top {len(all_combos['parlay_2'])} parlays de 2 times")
                for i, combo in enumerate(all_combos['parlay_2']):
                    _render_combo_card(combo, f"2team_{i}")
            else:
                st.info("Nenhum parlay de 2 times encontrado.")
        
        with parlay_tab2:
            if all_combos.get('parlay_3'):
                st.caption(f"Top {len(all_combos['parlay_3'])} parlays de 3 times")
                for i, combo in enumerate(all_combos['parlay_3']):
                    _render_combo_card(combo, f"3team_{i}")
            else:
                st.info("Nenhum parlay de 3 times encontrado.")
        
        with parlay_tab3:
            if all_combos.get('parlay_4'):
                st.caption(f"Top {len(all_combos['parlay_4'])} parlays de 4 times")
                for i, combo in enumerate(all_combos['parlay_4']):
                    _render_combo_card(combo, f"4team_{i}")
            else:
                st.info("Nenhum parlay de 4 times encontrado.")
    else:
        st.info("Nenhum parlay multi-time encontrado. Verifique se há jogos suficientes disponíveis.")


# =============================================================================
# SHADOW MODE: Profit Dashboard
# =============================================================================
def render_profit_dashboard():
    """
    Renderiza dashboard de lucro acumulado para Shadow Mode.
    
    Lê dados de:
    1. data/betting_log.csv (preferencial)
    2. data/backtest_bets.db (fallback)
    3. data/bet_tracking.csv (fallback 2)
    """
    st.header("📈 Dashboard de Lucro - Shadow Mode")
    st.caption("Acompanhe o desempenho simulado em tempo real")
    
    # Tentar carregar dados de várias fontes
    df_bets = None
    source = None
    
    # Fonte 1: betting_log.csv
    log_file = Path('data/betting_log.csv')
    if log_file.exists():
        try:
            df_bets = pd.read_csv(log_file)
            source = 'betting_log.csv'
        except Exception as e:
            logger.debug(f"Erro lendo betting_log: {e}")
    
    # Fonte 2: backtest_bets.db
    if df_bets is None:
        db_file = Path('data/backtest_bets.db')
        if db_file.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_file))
                df_bets = pd.read_sql("SELECT * FROM bets ORDER BY date", conn)
                conn.close()
                source = 'backtest_bets.db'
            except Exception as e:
                logger.debug(f"Erro lendo backtest_bets.db: {e}")
    
    # Fonte 3: bet_tracking.csv
    if df_bets is None:
        tracking_file = Path('data/bet_tracking.csv')
        if tracking_file.exists():
            try:
                df_bets = pd.read_csv(tracking_file)
                source = 'bet_tracking.csv'
            except Exception as e:
                logger.debug(f"Erro lendo bet_tracking: {e}")
    
    if df_bets is None or df_bets.empty:
        st.info("📊 Nenhum dado de apostas encontrado.")
        st.caption("Execute o backtest ou faça apostas simuladas para ver o dashboard.")
        st.code("python ml_pipeline/backtest_betting.py", language="bash")
        return
    
    st.caption(f"📁 Fonte: {source}")
    
    # Preparar dados
    if 'date' in df_bets.columns:
        df_bets['date'] = pd.to_datetime(df_bets['date'])
        df_bets = df_bets.sort_values('date')
    
    # Calcular lucro acumulado
    profit_col = None
    for col in ['profit', 'pnl', 'result', 'return']:
        if col in df_bets.columns:
            profit_col = col
            break
    
    if profit_col:
        df_bets['cumulative_profit'] = df_bets[profit_col].cumsum()
    else:
        st.warning("⚠️ Coluna de lucro não encontrada no arquivo.")
        return
    
    # Métricas principais
    total_bets = len(df_bets)
    total_profit = df_bets[profit_col].sum()
    win_rate = 0
    
    if 'won' in df_bets.columns:
        wins = df_bets['won'].sum()
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    elif profit_col:
        wins = (df_bets[profit_col] > 0).sum()
        win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    # Peak e Drawdown
    peak = df_bets['cumulative_profit'].cummax()
    drawdown = df_bets['cumulative_profit'] - peak
    max_drawdown = drawdown.min()
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Lucro Total",
            f"R$ {total_profit:+,.2f}",
            delta="Positivo" if total_profit > 0 else "Negativo",
            delta_color="normal" if total_profit >= 0 else "inverse"
        )
    with col2:
        st.metric("Total Apostas", f"{total_bets}")
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric(
            "Max Drawdown",
            f"R$ {max_drawdown:,.2f}",
            delta_color="inverse" if max_drawdown < 0 else "normal"
        )
    
    st.markdown("---")
    
    # Gráfico de lucro acumulado
    st.subheader("📈 Lucro Acumulado")
    
    fig = px.line(
        df_bets,
        x='date' if 'date' in df_bets.columns else df_bets.index,
        y='cumulative_profit',
        title='Evolução do Lucro (Shadow Mode)'
    )
    
    # Colorir baseado em positivo/negativo
    fig.update_traces(
        line=dict(color='#4ade80', width=2),
        fill='tozeroy',
        fillcolor='rgba(74, 222, 128, 0.1)'
    )
    
    # Linha de referência no zero
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Layout
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Lucro Acumulado (R$)",
        template="plotly_dark",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas adicionais
    st.subheader("📊 Estatísticas Detalhadas")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        if 'stake' in df_bets.columns:
            total_staked = df_bets['stake'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            st.metric("ROI", f"{roi:+.2f}%")
        
        avg_profit = total_profit / total_bets if total_bets > 0 else 0
        st.metric("Lucro Médio/Aposta", f"R$ {avg_profit:+.2f}")
    
    with col_s2:
        if 'odds' in df_bets.columns:
            avg_odds = df_bets['odds'].mean()
            st.metric("Odds Média", f"{avg_odds:.2f}")
        
        # Sharpe simplificado
        if len(df_bets) > 1:
            returns = df_bets[profit_col]
            sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col_s3:
        # Melhor e pior resultado
        best = df_bets[profit_col].max()
        worst = df_bets[profit_col].min()
        st.metric("Melhor Aposta", f"R$ {best:+.2f}")
        st.metric("Pior Aposta", f"R$ {worst:+.2f}")
    
    # Tabela de apostas recentes
    st.markdown("---")
    st.subheader("🕐 Apostas Recentes")
    
    display_cols = [c for c in ['date', 'home_team', 'away_team', 'side', 'odds', 'stake', profit_col, 'won'] 
                   if c in df_bets.columns]
    
    if display_cols:
        st.dataframe(
            df_bets[display_cols].tail(10).sort_index(ascending=False),
            use_container_width=True,
            hide_index=True
        )
    
    # Auto-refresh hint
    st.caption("💡 Dica: Clique em 'Rerun' no canto superior direito para atualizar os dados.")

