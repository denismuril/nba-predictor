"""
Betting Tracker System - Rastreamento completo de apostas e P&L.

Funcionalidades:
- Registrar apostas com odds, stakes e EV
- Atualizar resultados automaticamente
- Calcular ROI, Sharpe Ratio, Drawdown
- Tracking de Closing Line Value (CLV)

REFATORADO: Agora usa DatabaseManager para suportar SQLite/PostgreSQL
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from data.repositories.db_manager import get_db_manager

logger = logging.getLogger(__name__)

class BettingTracker:
    def __init__(self):
        """Inicializa o Betting Tracker usando DatabaseManager"""
        self.db = get_db_manager()
        self.db.init_db()  # Garante que schema está criado
        logger.info(f"✅ Betting Tracker inicializado com {self.db.db_type.upper()}")
    
    def log_bet(self, 
                game_date, 
                game_id,
                home_team,
                away_team,
                side,
                bet_type='MONEYLINE',
                line=None,
                bet_odds=None,
                stake_pct=None,
                bankroll=1000,
                model_prob=None,
                ev_pct=None,
                kelly_fraction=None,
                opening_odds=None,
                closing_odds=None):
        """
        Registra uma aposta no sistema.
        
        Returns:
            int: ID da aposta registrada
        """
        if bet_odds is None or stake_pct is None:
            logger.warning("⚠️  Aposta ignorada: odds ou stake não fornecidos")
            return None
        
        stake_amount = bankroll * (stake_pct / 100)
        
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO bets (
                    bet_date, game_id, home_team, away_team,
                    side, bet_type, line,
                    opening_odds, bet_odds, closing_odds,
                    stake_pct, stake_amount, bankroll_at_bet,
                    model_prob, ev_pct, kelly_fraction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                game_date, game_id, home_team, away_team,
                side, bet_type, line,
                opening_odds, bet_odds, closing_odds,
                stake_pct, stake_amount, bankroll,
                model_prob, ev_pct, kelly_fraction
            )
            
            cursor.execute(self.db._prepare_query(query), values)
            bet_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"📝 Aposta #{bet_id} registrada: {side} {home_team} vs {away_team} (${stake_amount:.2f} @ {bet_odds})")
            return bet_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro ao registrar aposta: {e}")
            raise
        finally:
            self.db.return_connection(conn)
    
    def update_result(self, bet_id, won):
        """Atualiza resultado de uma aposta após o jogo"""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            
            # Buscar dados da aposta
            cursor.execute(
                self.db._prepare_query("SELECT stake_amount, bet_odds FROM bets WHERE id = ?"),
                (bet_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                logger.error(f"❌ Aposta #{bet_id} não encontrada")
                return
            
            stake_amount, bet_odds = row
            
            if won:
                payout = stake_amount * bet_odds
                profit = payout - stake_amount
                result = 'WIN'
            else:
                payout = 0
                profit = -stake_amount
                result = 'LOSS'
            
            query = """
                UPDATE bets 
                SET result = ?, payout = ?, profit = ?, settled_at = ?
                WHERE id = ?
            """
            
            cursor.execute(
                self.db._prepare_query(query),
                (result, payout, profit, datetime.now().isoformat(), bet_id)
            )
            
            conn.commit()
            logger.info(f"✅ Aposta #{bet_id} atualizada: {result} (${profit:+.2f})")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro ao atualizar resultado: {e}")
            raise
        finally:
            self.db.return_connection(conn)
    
    def get_bet(self, bet_id):
        """Retorna dados de uma aposta específica"""
        conn = self.db.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute(
                self.db._prepare_query("SELECT * FROM bets WHERE id = ?"),
                (bet_id,)
            )
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            
            if row:
                return dict(zip(columns, row))
            return None
            
        finally:
            self.db.return_connection(conn)
    
    def get_performance_metrics(self, period='30d', bet_type=None):
        """
        Calcula métricas de performance.
        
        Parameters:
            period: '7d', '30d', '90d', 'all'
            bet_type: Filter by bet type or None for all
        """
        conn = self.db.get_connection()
        
        try:
            # Calcular data de início
            if period == 'all':
                query = "SELECT * FROM bets WHERE result IN ('WIN', 'LOSS')"
                params = []
            else:
                days = int(period.replace('d', ''))
                cutoff_date = (datetime.now() - timedelta(days=days)).date().isoformat()
                query = "SELECT * FROM bets WHERE result IN ('WIN', 'LOSS') AND bet_date >= ?"
                params = [cutoff_date]
            
            if bet_type:
                query += " AND bet_type = ?"
                params.append(bet_type)
            
            df = pd.read_sql_query(self.db._prepare_query(query), conn, params=params)
            
            if df.empty:
                return {
                    'period': period,
                    'total_bets': 0,
                    'win_rate': 0,
                    'total_staked': 0,
                    'total_returned': 0,
                    'total_profit': 0,
                    'roi': 0,
                    'avg_odds': 0,
                    'avg_ev': 0,
                    'avg_clv': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'current_streak': 0
                }
            
            # Métricas básicas
            total_bets = len(df)
            wins = (df['result'] == 'WIN').sum()
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
            
            total_staked = df['stake_amount'].sum()
            total_returned = df['payout'].sum()
            total_profit = df['profit'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            avg_odds = df['bet_odds'].mean()
            avg_ev = df['ev_pct'].mean() if 'ev_pct' in df.columns else 0
            
            # Closing Line Value
            df_with_clv = df[df['closing_odds'].notna()]
            if not df_with_clv.empty:
                clv_values = ((df_with_clv['bet_odds'] - df_with_clv['closing_odds']) / df_with_clv['closing_odds'] * 100)
                avg_clv = clv_values.mean()
            else:
                avg_clv = 0
            
            # Sharpe Ratio
            if len(df) > 1:
                returns = df['profit'] / df['stake_amount']
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            cumulative_profit = df['profit'].cumsum()
            running_max = cumulative_profit.cummax()
            drawdown = cumulative_profit - running_max
            max_drawdown = drawdown.min()
            
            # Current Streak
            current_streak = 0
            for result in reversed(df['result'].tolist()):
                if result == df['result'].iloc[-1]:
                    current_streak += 1
                else:
                    break
            if df['result'].iloc[-1] == 'LOSS':
                current_streak = -current_streak
            
            # Best/Worst bet
            best_bet_idx = df['profit'].idxmax()
            worst_bet_idx = df['profit'].idxmin()
            
            return {
                'period': period,
                'total_bets': total_bets,
                'win_rate': round(win_rate, 1),
                'total_staked': round(total_staked, 2),
                'total_returned': round(total_returned, 2),
                'total_profit': round(total_profit, 2),
                'roi': round(roi, 2),
                'avg_odds': round(avg_odds, 2),
                'avg_ev': round(avg_ev, 2),
                'avg_clv': round(avg_clv, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'current_streak': current_streak,
                'best_bet': {
                    'profit': round(df.loc[best_bet_idx, 'profit'], 2),
                    'odds': df.loc[best_bet_idx, 'bet_odds'],
                    'game': f"{df.loc[best_bet_idx, 'home_team']} vs {df.loc[best_bet_idx, 'away_team']}"
                },
                'worst_bet': {
                    'profit': round(df.loc[worst_bet_idx, 'profit'], 2),
                    'odds': df.loc[worst_bet_idx, 'bet_odds'],
                    'game': f"{df.loc[worst_bet_idx, 'home_team']} vs {df.loc[worst_bet_idx, 'away_team']}"
                }
            }
        finally:
            self.db.return_connection(conn)
    
    def get_pending_bets(self):
        """Retorna apostas pendentes"""
        conn = self.db.get_connection()
        try:
            query = "SELECT * FROM bets WHERE result = 'PENDING' ORDER BY bet_date DESC"
            df = pd.read_sql_query(self.db._prepare_query(query), conn)
            return df
        finally:
            self.db.return_connection(conn)
    
    def print_performance_summary(self, period='30d'):
        """Imprime resumo de performance formatado"""
        metrics = self.get_performance_metrics(period)
        
        print(f"\n{'='*60}")
        print(f"💰 PERFORMANCE - Últimos {period}")
        print(f"{'='*60}")
        print(f"Total Apostas:    {metrics['total_bets']}")
        print(f"Win Rate:         {metrics['win_rate']:.1f}%")
        print(f"Total Apostado:   ${metrics['total_staked']:,.2f}")
        print(f"Total Retornado:  ${metrics['total_returned']:,.2f}")
        print(f"Lucro Total:      ${metrics['total_profit']:+,.2f}")
        print(f"ROI:              {metrics['roi']:+.2f}%")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     ${metrics['max_drawdown']:,.2f}")
        print(f"Avg EV:           {metrics['avg_ev']:+.2f}%")
        print(f"Avg CLV:          {metrics['avg_clv']:+.2f} cents")
        print(f"Current Streak:   {metrics['current_streak']:+d}")
        print(f"\nMelhor Aposta:    {metrics['best_bet']['game']}")
        print(f"                  ${metrics['best_bet']['profit']:+.2f} @ {metrics['best_bet']['odds']}")
        print(f"Pior Aposta:      {metrics['worst_bet']['game']}")
        print(f"                  ${metrics['worst_bet']['profit']:+.2f} @ {metrics['worst_bet']['odds']}")
        print(f"{'='*60}\n")
