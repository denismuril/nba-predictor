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
    def __init__(self, bankroll=1000.0, kelly_fraction=0.25, min_ev=0.02, max_total_exposure=0.15):
        """
        Args:
            bankroll: Valor total da banca
            kelly_fraction: Fração do Kelly a ser usada (ex: 0.25 = 1/4 Kelly)
            min_ev: EV mínimo para sugerir aposta (ex: 0.02 = 2%)
            max_total_exposure: Exposição máxima total por noite (ex: 0.15 = 15%)
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_ev = min_ev
        self.max_total_exposure = max_total_exposure
        
        logger.info(f"🏦 Betting Engine initialized: Bankroll=${bankroll:.2f}, Kelly={kelly_fraction:.2%}, Max Exposure={max_total_exposure:.2%}")
        
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
        
        # Additional safety: never bet more than 5% on a single game
        adj_kelly = min(adj_kelly, 0.05)
        
        return adj_kelly
        
    def analyze_bet(self, game_info, model_prob, market_odds, bet_type='Moneyline', n_concurrent_bets=1):
        """
        Analisa uma aposta e retorna recomendação.
        
        Args:
            game_info: Dict com 'home_team' e 'away_team'
            model_prob: Probabilidade do modelo (0-1)
            market_odds: Odds do mercado (decimal)
            bet_type: Tipo de aposta (Moneyline, Spread, etc)
            n_concurrent_bets: Número de jogos simultâneos na mesma noite
        
        Returns:
            Dict com recomendação de aposta
        """
        ev = self.calculate_ev(model_prob, market_odds)
        kelly_pct = self.calculate_kelly_stake_concurrent(model_prob, market_odds, n_concurrent_bets)
        stake_amount = self.bankroll * kelly_pct
        
        recommendation = {
            'game': f"{game_info['home_team']} vs {game_info['away_team']}",
            'bet_type': bet_type,
            'model_prob': round(model_prob * 100, 1),
            'market_odds': market_odds,
            'implied_prob': round((1/market_odds) * 100, 1) if market_odds > 0 else 0,
            'ev': round(ev * 100, 2),
            'kelly_pct': round(kelly_pct * 100, 2),
            'suggested_stake': round(stake_amount, 2),
            'is_value': ev >= self.min_ev and kelly_pct > 0,
            'concurrent_games': n_concurrent_bets,
            'total_exposure_pct': round(kelly_pct * n_concurrent_bets * 100, 2)  # Total if all bets placed
        }
        
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
