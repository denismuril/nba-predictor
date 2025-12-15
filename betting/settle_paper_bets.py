"""
Paper Bet Settlement Script - NBA Predictor v25.0
==================================================
Script para liquidar apostas paper com resultados reais.
Roda diariamente via cron/Prefect para calcular PnL.

Uso:
    python betting/settle_paper_bets.py
    python betting/settle_paper_bets.py --date 2024-12-14
    python betting/settle_paper_bets.py --days 7

Autor: NBA Predictor v25.0 - Go Live Edition
"""
import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from sqlalchemy import text

# Infrastructure imports
from infrastructure.database import AsyncDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('settle_paper_bets')


class PaperBetSettler:
    """
    Classe para liquidar paper bets comparando com resultados reais.
    """

    def __init__(self):
        self.db: Optional[AsyncDataManager] = None
        self.settled_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

    async def initialize(self) -> bool:
        """Inicializa conexão com banco de dados."""
        try:
            self.db = AsyncDataManager()
            await self.db.init_db()
            logger.info("✅ PaperBetSettler inicializado")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar: {e}")
            return False

    async def get_pending_bets(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Busca apostas pendentes para liquidação.

        Args:
            date_str: Data específica (YYYY-MM-DD) ou None para todas
        """
        try:
            if date_str:
                # Buscar apostas de uma data específica
                query = """
                SELECT id, game_id, matchup, bet_type, market_odds, stake,
                       model_prob, edge, confidence, created_at
                FROM paper_bets
                WHERE status = 'PENDING'
                  AND DATE(created_at) = :target_date
                ORDER BY created_at
                """
                params = {'target_date': date_str}
            else:
                # Buscar todas pendentes
                query = """
                SELECT id, game_id, matchup, bet_type, market_odds, stake,
                       model_prob, edge, confidence, created_at
                FROM paper_bets
                WHERE status = 'PENDING'
                ORDER BY created_at
                """
                params = {}

            async with self.db.get_session() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]

        except Exception as e:
            logger.error(f"❌ Erro ao buscar apostas pendentes: {e}")
            return []

    async def get_game_result(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Busca resultado real de um jogo."""
        try:
            query = """
            SELECT game_id, home_team, away_team, home_score, away_score, winner
            FROM games
            WHERE game_id = :game_id
              AND status = 'Final'
              AND home_score IS NOT NULL
            """

            async with self.db.get_session() as session:
                result = await session.execute(text(query), {'game_id': game_id})
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
                return None

        except Exception as e:
            logger.error(f"❌ Erro ao buscar resultado do jogo {game_id}: {e}")
            return None

    def determine_bet_outcome(
        self,
        bet: Dict[str, Any],
        result: Dict[str, Any]
    ) -> tuple[str, float]:
        """
        Determina se aposta ganhou ou perdeu.

        Returns:
            Tuple (status, profit)
        """
        bet_type = bet['bet_type']
        stake = bet['stake']
        odds = bet['market_odds']

        home_score = result['home_score']
        away_score = result['away_score']
        winner = result['winner']

        # Determinar vencedor
        home_won = home_score > away_score

        # Avaliar aposta
        won = False

        if bet_type == 'moneyline_home':
            won = home_won
        elif bet_type == 'moneyline_away':
            won = not home_won
        elif bet_type == 'spread_home':
            # Assumindo spread básico (sem pontos)
            # Para spread real, precisaríamos do spread line
            won = home_won
        elif bet_type == 'spread_away':
            won = not home_won
        elif bet_type == 'over':
            total = home_score + away_score
            # Precisaríamos da linha de total
            won = total > 220  # Placeholder
        elif bet_type == 'under':
            total = home_score + away_score
            won = total < 220  # Placeholder

        if won:
            profit = stake * (odds - 1)  # Lucro = stake * (odds - 1)
            return 'WIN', round(profit, 2)
        else:
            return 'LOSS', round(-stake, 2)

    async def settle_bet(
        self,
        bet_id: int,
        status: str,
        profit: float,
        result_score: str
    ) -> bool:
        """Atualiza status da aposta no banco."""
        try:
            update_sql = """
            UPDATE paper_bets
            SET status = :status,
                profit = :profit,
                result_score = :result_score,
                settled_at = :settled_at
            WHERE id = :bet_id
            """

            async with self.db.get_session() as session:
                await session.execute(text(update_sql), {
                    'bet_id': bet_id,
                    'status': status,
                    'profit': profit,
                    'result_score': result_score,
                    'settled_at': datetime.utcnow()
                })
                await session.commit()

            return True

        except Exception as e:
            logger.error(f"❌ Erro ao atualizar aposta {bet_id}: {e}")
            return False

    async def settle_date(self, date_str: str) -> Dict[str, Any]:
        """
        Liquida todas as apostas de uma data específica.

        Args:
            date_str: Data no formato YYYY-MM-DD

        Returns:
            Dict com estatísticas
        """
        logger.info(f"📅 Liquidando apostas de {date_str}")

        pending_bets = await self.get_pending_bets(date_str)

        if not pending_bets:
            logger.info("📭 Nenhuma aposta pendente para liquidar")
            return {'settled': 0, 'pnl': 0.0}

        logger.info(f"📋 {len(pending_bets)} apostas pendentes encontradas")

        for bet in pending_bets:
            game_result = await self.get_game_result(bet['game_id'])

            if not game_result:
                logger.warning(f"⚠️ Resultado não encontrado: {bet['game_id']}")
                continue

            # Determinar outcome
            status, profit = self.determine_bet_outcome(bet, game_result)

            # Score string
            result_score = f"{game_result['home_score']}-{game_result['away_score']}"

            # Atualizar banco
            success = await self.settle_bet(bet['id'], status, profit, result_score)

            if success:
                self.settled_count += 1
                self.total_pnl += profit

                if status == 'WIN':
                    self.wins += 1
                    logger.info(
                        f"✅ WIN: {bet['matchup']} | {bet['bet_type']} | "
                        f"Profit: R$ {profit:+.2f}"
                    )
                else:
                    self.losses += 1
                    logger.info(
                        f"❌ LOSS: {bet['matchup']} | {bet['bet_type']} | "
                        f"Profit: R$ {profit:+.2f}"
                    )

        return {
            'settled': self.settled_count,
            'wins': self.wins,
            'losses': self.losses,
            'pnl': self.total_pnl
        }

    async def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Gera relatório de performance dos últimos N dias."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            query = """
            SELECT
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status = 'WIN' THEN 1 END) as wins,
                COUNT(CASE WHEN status = 'LOSS' THEN 1 END) as losses,
                COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending,
                COALESCE(SUM(CASE WHEN status != 'PENDING' THEN profit ELSE 0 END), 0) as total_pnl,
                COALESCE(SUM(stake), 0) as total_staked,
                COALESCE(AVG(CASE WHEN status != 'PENDING' THEN edge ELSE NULL END), 0) as avg_edge,
                COALESCE(AVG(market_odds), 0) as avg_odds
            FROM paper_bets
            WHERE created_at >= :cutoff
            """

            async with self.db.get_session() as session:
                result = await session.execute(text(query), {'cutoff': cutoff})
                row = result.fetchone()

                if row:
                    data = dict(row._mapping)
                    settled = data['wins'] + data['losses']
                    data['win_rate'] = data['wins'] / settled if settled > 0 else 0
                    data['roi'] = data['total_pnl'] / data['total_staked'] if data['total_staked'] > 0 else 0
                    return data

            return {}

        except Exception as e:
            logger.error(f"❌ Erro ao gerar relatório: {e}")
            return {}

    async def close(self):
        """Fecha conexões."""
        if self.db:
            await self.db.close()


def print_report(stats: Dict[str, Any], days: int):
    """Imprime relatório formatado."""
    print("\n" + "=" * 60)
    print(f"📊 PAPER TRADING - RELATÓRIO DE PERFORMANCE ({days} DIAS)")
    print("=" * 60)
    print(f"")
    print(f"📈 Estatísticas Gerais:")
    print(f"   Total de Apostas: {stats.get('total_bets', 0)}")
    print(f"   Vitórias (WIN):   {stats.get('wins', 0)}")
    print(f"   Derrotas (LOSS):  {stats.get('losses', 0)}")
    print(f"   Pendentes:        {stats.get('pending', 0)}")
    print(f"")
    print(f"💰 Resultados Financeiros:")
    print(f"   Total Apostado:   R$ {stats.get('total_staked', 0):,.2f}")
    print(f"   Lucro/Prejuízo:   R$ {stats.get('total_pnl', 0):+,.2f}")
    print(f"   ROI:              {stats.get('roi', 0):.2%}")
    print(f"")
    print(f"📊 Métricas de Qualidade:")
    print(f"   Win Rate:         {stats.get('win_rate', 0):.1%}")
    print(f"   Edge Médio:       {stats.get('avg_edge', 0):.2%}")
    print(f"   Odds Médias:      {stats.get('avg_odds', 0):.2f}")
    print("=" * 60)

    # Veredicto
    roi = stats.get('roi', 0)
    if roi > 0.05:
        print("✅ VEREDICTO: Sistema LUCRATIVO! Considere Go Live.")
    elif roi > 0:
        print("🟡 VEREDICTO: Sistema positivo, mas edge pequeno.")
    else:
        print("❌ VEREDICTO: Sistema em prejuízo. Revisar estratégia.")
    print("")


async def main():
    parser = argparse.ArgumentParser(description='Settle Paper Bets - NBA Predictor')
    parser.add_argument('--date', type=str, default=None,
                        help='Data específica (YYYY-MM-DD). Default: ontem')
    parser.add_argument('--days', type=int, default=7,
                        help='Dias para relatório (default: 7)')
    parser.add_argument('--report-only', action='store_true',
                        help='Apenas mostrar relatório, não liquidar')

    args = parser.parse_args()

    settler = PaperBetSettler()

    if not await settler.initialize():
        logger.error("❌ Falha ao inicializar")
        sys.exit(1)

    if args.report_only:
        # Apenas relatório
        stats = await settler.get_performance_report(args.days)
        print_report(stats, args.days)
        await settler.close()
        return

    # Determinar data para liquidação
    if args.date:
        target_date = args.date
    else:
        # Ontem por padrão
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime('%Y-%m-%d')

    # Liquidar apostas
    result = await settler.settle_date(target_date)

    print("\n" + "=" * 50)
    print(f"📊 SETTLEMENT - {target_date}")
    print("=" * 50)
    print(f"Apostas Liquidadas: {result['settled']}")
    print(f"Vitórias: {result.get('wins', 0)}")
    print(f"Derrotas: {result.get('losses', 0)}")
    print(f"PnL do Dia: R$ {result['pnl']:+,.2f}")
    print("=" * 50)

    # Mostrar relatório completo
    stats = await settler.get_performance_report(args.days)
    print_report(stats, args.days)

    await settler.close()


if __name__ == "__main__":
    asyncio.run(main())
