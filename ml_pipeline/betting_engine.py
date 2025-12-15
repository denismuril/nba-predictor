"""
Betting Engine - Gestão de Banca e Valor Esperado (EV)

Este módulo é responsável por:
1. Calcular Valor Esperado (EV) das apostas
2. Sugerir tamanho da aposta (Kelly Criterion com ajuste para apostas simultâneas)
3. Identificar oportunidades de valor (+EV)

Versão 2.0 - Concurrent Betting Support:
- Implementa ajuste de variância para apostas simultâneas
- Reduz risco de ruína em noites com múltiplos jogos
- Baseado em "Kelly Capital Growth" (Thorp, MacLean et al.)
"""
import logging
import math
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BettingEngine:
    """
    Motor de Apostas com Gestão de Risco Conservadora.
    
    Travas de Segurança Implementadas:
        1. Filtro de EV+ Rígido: Edge mínimo de 3% sobre a casa
        2. Kelly Fracionado Conservador: Kelly/8 (0.125)
        3. Hard Cap de 3%: Nunca apostar mais que 3% da banca
        4. Exposição Máxima: 15% total por noite em múltiplos jogos
    """
    
    # ==========================================================================
    # CONSTANTES DE SEGURANÇA - Ajuste Fino para Banca Conservadora
    # ==========================================================================
    MIN_EDGE_THRESHOLD = 0.03  # Edge mínimo de 3% (filtro de ruído)
    MAX_STAKE_CAP = 0.03       # Hard cap de 3% da banca por aposta
    
    def __init__(
        self,
        bankroll: float = 100.0,              # Banca inicial conservadora
        kelly_fraction: float = 0.125,         # Kelly/8 para minimizar risco de ruína
        min_ev: float = 0.02,                  # EV mínimo para registro
        max_total_exposure: float = 0.15,      # Exposição máxima 15% por noite
        min_edge_threshold: float = 0.03       # Limiar de confiança: 3% edge mínimo
    ):
        """
        Inicializa o Motor de Apostas Conservador.
        
        Args:
            bankroll: Valor total da banca (padrão: R$100)
            kelly_fraction: Fração do Kelly (padrão: 0.125 = Kelly/8)
            min_ev: EV mínimo para logging (padrão: 2%)
            max_total_exposure: Exposição máxima total por noite (padrão: 15%)
            min_edge_threshold: Edge mínimo para apostar (padrão: 3%)
        
        Notas de Segurança:
            - Kelly/8 reduz variância em ~75% comparado a Kelly/2
            - Edge 3% filtra "ruído" de probabilidades incertas
            - Hard cap 3% protege contra over-betting em alta confiança
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_ev = min_ev
        self.max_total_exposure = max_total_exposure
        self.min_edge_threshold = min_edge_threshold
        
        logger.info(f"🏦 Betting Engine inicializado (Modo Conservador):")
        logger.info(f"   💰 Bankroll: R${bankroll:.2f}")
        logger.info(f"   📊 Kelly Fraction: {kelly_fraction:.3f} (Kelly/{int(1/kelly_fraction)})")
        logger.info(f"   🎯 Edge Mínimo: {min_edge_threshold:.1%} (filtro de ruído)")
        logger.info(f"   🔒 Max Stake Cap: {self.MAX_STAKE_CAP:.1%}")
        logger.info(f"   📈 Max Exposure/Noite: {max_total_exposure:.1%}")
        
    def calculate_ev(self, prob_win, decimal_odds):
        """
        Calcula Expected Value (EV).
        EV = (Prob_Win * (Odds - 1)) - (Prob_Loss * 1)
        """
        if decimal_odds <= 1:
            return -1.0
            
        prob_loss = 1.0 - prob_win
        profit = decimal_odds - 1.0
        
        ev = (prob_win * profit) - (prob_loss * 1.0)
        return ev
        
    def calculate_kelly_stake(self, prob_win, decimal_odds):
        """
        Calcula a porcentagem da banca a apostar usando Kelly Criterion (método legado).
        
        ⚠️ DEPRECATED: Use calculate_kelly_stake_concurrent() para apostas simultâneas.
        
        f* = (bp - q) / b
        b = odds - 1
        p = probabilidade de vitória
        q = probabilidade de derrota (1-p)
        """
        if decimal_odds <= 1:
            return 0.0
            
        b = decimal_odds - 1.0
        p = prob_win
        q = 1.0 - p
        
        kelly_pct = (b * p - q) / b
        
        # Aplicar fração (Kelly fracionário é mais seguro)
        adj_kelly = max(0.0, kelly_pct * self.kelly_fraction)
        
        # Limitar aposta máxima (ex: 5% da banca)
        adj_kelly = min(adj_kelly, 0.05)
        
        return adj_kelly
    
    def calculate_kelly_stake_concurrent(self, prob_win, decimal_odds, n_concurrent_bets=1):
        """
        Calcula Kelly Criterion ajustado para apostas simultâneas.
        
        Formula:
            f* = (bp - q) / (b * sqrt(n))
            
        Onde:
            b = decimal_odds - 1
            p = probabilidade de vitória
            q = probabilidade de derrota (1-p)
            n = número de apostas simultâneas
        
        O ajuste sqrt(n) reduz a variância quando múltiplas apostas correlacionadas
        ocorrem simultaneamente (ex: 10 jogos na mesma noite da NBA).
        
        Args:
            prob_win: Probabilidade de vitória (0-1)
            decimal_odds: Odds decimais (ex: 2.0 para +100)
            n_concurrent_bets: Número de apostas simultâneas (default: 1)
        
        Returns:
            Porcentagem da banca a apostar (0-1)
        
        References:
            - Thorp, E. (2008): "Kelly Capital Growth Investment Criterion"
            - MacLean et al. (2011): "The Kelly Criterion in Sports Betting"
        
        Examples:
            >>> engine = BettingEngine(bankroll=1000, kelly_fraction=0.25)
            >>> # Single game
            >>> engine.calculate_kelly_stake_concurrent(0.55, 2.0, n_concurrent_bets=1)
            0.025  # 2.5% da banca
            
            >>> # 10 games same night
            >>> engine.calculate_kelly_stake_concurrent(0.55, 2.0, n_concurrent_bets=10)
            0.0079  # 0.79% por jogo (total ~7.9%)
        """
        if decimal_odds <= 1:
            return 0.0
        
        if prob_win <= 0 or prob_win >= 1:
            logger.warning(f"⚠️ Invalid prob_win={prob_win:.3f}, clamping to [0.01, 0.99]")
            prob_win = np.clip(prob_win, 0.01, 0.99)
        
        b = decimal_odds - 1.0
        p = prob_win
        q = 1.0 - p
        
        # Standard Kelly
        kelly_pct = (b * p - q) / b
        
        # Concurrent adjustment: reduce by sqrt(n) to account for correlation
        concurrent_adjustment = 1.0 / math.sqrt(max(n_concurrent_bets, 1))
        kelly_concurrent = kelly_pct * concurrent_adjustment
        
        # Apply fractional Kelly (conservative multiplier)
        adj_kelly = max(0.0, kelly_concurrent * self.kelly_fraction)
        
        # Global constraint: ensure total exposure doesn't exceed limit
        # If we have 10 games at 2% each, total = 20%, which exceeds 15% limit
        # So we cap each individual bet to: max_total_exposure / n_concurrent_bets
        max_per_bet = self.max_total_exposure / max(n_concurrent_bets, 1)
        adj_kelly = min(adj_kelly, max_per_bet)
        
        # ======================================================================
        # TRAVA DE SEGURANÇA: Hard Cap de 3% da banca
        # Independente do Kelly, nunca apostar mais que MAX_STAKE_CAP
        # ======================================================================
        adj_kelly = min(adj_kelly, self.MAX_STAKE_CAP)
        
        return adj_kelly
        
    def analyze_bet(self, game_info, model_prob, market_odds, bet_type='Moneyline', n_concurrent_bets=1):
        """
        Analisa uma aposta e retorna recomendação com filtros de segurança.
        
        FILTROS DE SEGURANÇA APLICADOS:
            1. Edge mínimo de 3% (abs(prob_modelo - prob_implícita) > 0.03)
            2. Kelly/8 fracionado
            3. Hard cap de 3% da banca
        
        Args:
            game_info: Dict com 'home_team' e 'away_team'
            model_prob: Probabilidade do modelo (0-1)
            market_odds: Odds do mercado (decimal)
            bet_type: Tipo de aposta (Moneyline, Spread, etc)
            n_concurrent_bets: Número de jogos simultâneos na mesma noite
        
        Returns:
            Dict com recomendação de aposta (ou SKIP se edge insuficiente)
        """
        # Calcular probabilidade implícita das odds
        implied_prob = 1 / market_odds if market_odds > 1 else 1.0
        
        # ======================================================================
        # TAREFA 1: FILTRO DE EV+ RÍGIDO (Confidence Threshold)
        # Só apostar se: abs(prob_modelo - prob_implícita) > 3%
        # ======================================================================
        edge_over_market = abs(model_prob - implied_prob)
        
        if edge_over_market < self.min_edge_threshold:
            # SKIP: Vantagem insuficiente sobre a casa
            reason = (
                f"Edge {edge_over_market:.1%} < threshold {self.min_edge_threshold:.1%} "
                f"(Modelo: {model_prob:.1%} vs Mercado: {implied_prob:.1%})"
            )
            logger.info(f"⏭️ SKIP: Low Edge - {game_info.get('home_team', 'N/A')} vs {game_info.get('away_team', 'N/A')} [{bet_type}]: {reason}")
            
            return {
                'game': f"{game_info.get('home_team', 'N/A')} vs {game_info.get('away_team', 'N/A')}",
                'bet_type': bet_type,
                'model_prob': round(model_prob * 100, 1),
                'market_odds': market_odds,
                'implied_prob': round(implied_prob * 100, 1),
                'edge_over_market': round(edge_over_market * 100, 2),
                'ev': 0,
                'kelly_pct': 0,
                'suggested_stake': 0,
                'is_value': False,
                'skip_reason': 'SKIP: Low Edge',
                'concurrent_games': n_concurrent_bets,
                'total_exposure_pct': 0
            }
        
        # ======================================================================
        # Aposta passou no filtro - calcular EV e Kelly
        # ======================================================================
        ev = self.calculate_ev(model_prob, market_odds)
        kelly_pct = self.calculate_kelly_stake_concurrent(model_prob, market_odds, n_concurrent_bets)
        stake_amount = self.bankroll * kelly_pct
        
        recommendation = {
            'game': f"{game_info.get('home_team', 'N/A')} vs {game_info.get('away_team', 'N/A')}",
            'bet_type': bet_type,
            'model_prob': round(model_prob * 100, 1),
            'market_odds': market_odds,
            'implied_prob': round(implied_prob * 100, 1),
            'edge_over_market': round(edge_over_market * 100, 2),
            'ev': round(ev * 100, 2),
            'kelly_pct': round(kelly_pct * 100, 2),
            'suggested_stake': round(stake_amount, 2),
            'is_value': ev >= self.min_ev and kelly_pct > 0,
            'concurrent_games': n_concurrent_bets,
            'total_exposure_pct': round(kelly_pct * n_concurrent_bets * 100, 2)
        }
        
        if recommendation['is_value']:
            logger.info(
                f"✅ VALUE BET: {recommendation['game']} [{bet_type}] - "
                f"Edge {edge_over_market:.1%} | EV {ev:.1%} | "
                f"Stake R${stake_amount:.2f} ({kelly_pct:.2%})"
            )
        
        return recommendation

    def process_games(self, df_games):
        """
        Processa um DataFrame de jogos e retorna oportunidades de aposta.
        
        Aplica Kelly Criterion ajustado para apostas simultâneas.
        
        Args:
            df_games: DataFrame com colunas 'prob_home', 'odds_home', 'odds_away'
        
        Returns:
            DataFrame com oportunidades de aposta (+EV)
        """
        opportunities = []
        
        # Count concurrent games for Kelly adjustment
        n_concurrent_bets = len(df_games)
        
        logger.info(f"📊 Processing {n_concurrent_bets} concurrent games")
        
        for idx, row in df_games.iterrows():
            # Home Bet
            if 'odds_home' in row and row['odds_home'] > 1:
                rec_home = self.analyze_bet(
                    row, 
                    row['prob_home'], 
                    row['odds_home'], 
                    'Home Win',
                    n_concurrent_bets=n_concurrent_bets
                )
                if rec_home['is_value']:
                    opportunities.append(rec_home)
                    
            # Away Bet
            prob_away = 1.0 - row['prob_home']
            if 'odds_away' in row and row['odds_away'] > 1:
                rec_away = self.analyze_bet(
                    row, 
                    prob_away, 
                    row['odds_away'], 
                    'Away Win',
                    n_concurrent_bets=n_concurrent_bets
                )
                if rec_away['is_value']:
                    opportunities.append(rec_away)
        
        if opportunities:
            total_suggested = sum(opp['suggested_stake'] for opp in opportunities)
            total_exposure_pct = (total_suggested / self.bankroll) * 100
            
            logger.info(f"✅ Found {len(opportunities)} +EV bets")
            logger.info(f"💰 Total suggested stake: ${total_suggested:.2f} ({total_exposure_pct:.1f}% of bankroll)")
            
            if total_exposure_pct > self.max_total_exposure * 100:
                logger.warning(
                    f"⚠️ Total exposure ({total_exposure_pct:.1f}%) exceeds limit ({self.max_total_exposure*100:.1f}%). "
                    f"Consider reducing bet sizes or skipping lower-EV bets."
                )
        else:
            logger.info("ℹ️ No +EV opportunities found")
                    
        return pd.DataFrame(opportunities)
