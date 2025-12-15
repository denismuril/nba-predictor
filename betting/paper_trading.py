"""
Paper Trading Daemon - NBA Predictor v25.0
===========================================
Daemon que roda 24/7 monitorando oportunidades e registrando
apostas simuladas em PostgreSQL.

Características:
- Async com asyncio e AsyncDataManager
- Lê odds do Redis (RedisCache)
- Registra paper bets em tabela SQL (auto-criada)
- Não envia para Telegram (apenas logging)

Uso:
    python betting/paper_trading.py --bankroll 1000
    python betting/paper_trading.py --daemon  # Modo 24/7

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
from dataclasses import dataclass, field

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Infrastructure imports
from infrastructure.database import AsyncDataManager, Base
from infrastructure.redis_cache import get_redis, RedisCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('paper_trading')


# =============================================================================
# PAPER BET MODEL
# =============================================================================

class PaperBet(Base):
    """Modelo para apostas simuladas (paper trading)."""
    __tablename__ = 'paper_bets'

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(50), index=True, nullable=False)
    matchup = Column(String(50), nullable=False)  # "LAL @ BOS"
    bet_type = Column(String(30), nullable=False)  # "moneyline_home", "spread_away"
    market_odds = Column(Float, nullable=False)
    fair_odds = Column(Float, nullable=True)
    stake = Column(Float, nullable=False)
    model_prob = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    status = Column(String(20), default='PENDING', index=True)  # PENDING, WIN, LOSS
    profit = Column(Float, default=0.0)
    result_score = Column(String(30), nullable=True)  # "105-102"
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime, nullable=True)


# =============================================================================
# PAPER TRADING ENGINE
# =============================================================================

@dataclass
class PaperTradingConfig:
    """Configuração do Paper Trading."""
    bankroll: float = 1000.0
    min_edge: float = 0.03  # 3%
    min_confidence: float = 0.60  # 60%
    max_stake_pct: float = 0.05  # 5% max per bet
    kelly_fraction: float = 0.25  # Quarter Kelly
    poll_interval: int = 60  # Segundos entre verificações
    stop_file: str = "data/.STOP_ALL_BETS"


class PaperTradingEngine:
    """
    Engine de Paper Trading que monitora odds e registra apostas simuladas.
    """

    def __init__(self, config: Optional[PaperTradingConfig] = None):
        self.config = config or PaperTradingConfig()
        self.db: Optional[AsyncDataManager] = None
        self.redis: Optional[RedisCache] = None
        self.running = False
        self.total_bets = 0
        self.session_pnl = 0.0

    async def initialize(self) -> bool:
        """Inicializa conexões com banco e Redis."""
        try:
            # Conectar banco de dados
            self.db = AsyncDataManager()
            await self.db.init_db()

            # Criar tabela paper_bets se não existir
            await self._ensure_paper_bets_table()

            # Conectar Redis
            self.redis = await get_redis()
            await self.redis.connect()

            logger.info("✅ PaperTradingEngine inicializado com sucesso")
            logger.info(f"   💰 Bankroll: R$ {self.config.bankroll:,.2f}")
            logger.info(f"   📊 Min Edge: {self.config.min_edge:.1%}")
            logger.info(f"   🎯 Min Confidence: {self.config.min_confidence:.1%}")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao inicializar PaperTradingEngine: {e}")
            return False

    async def _ensure_paper_bets_table(self):
        """Cria tabela paper_bets se não existir."""
        # Separar statements - asyncpg não suporta múltiplos em um execute
        statements = [
            """
            CREATE TABLE IF NOT EXISTS paper_bets (
                id SERIAL PRIMARY KEY,
                game_id VARCHAR(50) NOT NULL,
                matchup VARCHAR(50) NOT NULL,
                bet_type VARCHAR(30) NOT NULL,
                market_odds FLOAT NOT NULL,
                fair_odds FLOAT,
                stake FLOAT NOT NULL,
                model_prob FLOAT NOT NULL,
                confidence FLOAT,
                edge FLOAT,
                status VARCHAR(20) DEFAULT 'PENDING',
                profit FLOAT DEFAULT 0.0,
                result_score VARCHAR(30),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settled_at TIMESTAMP
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_paper_bets_status ON paper_bets(status)",
            "CREATE INDEX IF NOT EXISTS idx_paper_bets_game ON paper_bets(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_bets_created ON paper_bets(created_at)"
        ]
        try:
            async with self.db.get_session() as session:
                for stmt in statements:
                    await session.execute(text(stmt))
                await session.commit()
            logger.info("✅ Tabela paper_bets verificada/criada")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao criar tabela paper_bets: {e}")

    def _check_stop_file(self) -> bool:
        """Verifica se arquivo de parada existe."""
        stop_file = PROJECT_ROOT / self.config.stop_file
        if stop_file.exists():
            logger.warning("🛑 STOP_ALL_BETS ativo - operações suspensas")
            return True
        return False

    def _calculate_kelly_stake(self, prob: float, odds: float) -> float:
        """
        Calcula stake usando Kelly Criterion fracionado.

        Formula: f* = (bp - q) / b
        onde: b = odds - 1, p = probability, q = 1 - p
        """
        b = odds - 1
        p = prob
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b

        # Aplicar fração Kelly (1/4)
        kelly_fraction = kelly * self.config.kelly_fraction

        # Limitar ao máximo permitido
        stake_pct = min(kelly_fraction, self.config.max_stake_pct)
        stake_pct = max(0, stake_pct)  # Não apostar se Kelly negativo

        return self.config.bankroll * stake_pct

    def _calculate_edge(self, model_prob: float, market_odds: float) -> float:
        """Calcula edge: prob_model / prob_implicita - 1."""
        implied_prob = 1 / market_odds
        return (model_prob / implied_prob) - 1 if implied_prob > 0 else 0

    async def check_opportunities(self) -> List[Dict[str, Any]]:
        """
        Verifica oportunidades de aposta no Redis.
        Retorna lista de bets que atendem aos critérios.
        """
        opportunities = []

        if self._check_stop_file():
            return opportunities

        try:
            # Buscar todas as odds do Redis
            # Pattern: odds:GAME_ID
            all_keys = []
            if self.redis._connected and self.redis._redis:
                all_keys = await self.redis._redis.keys("odds:*")

            if not all_keys:
                logger.debug("📭 Nenhuma odd encontrada no Redis")
                return opportunities

            for key in all_keys:
                try:
                    key_str = key if isinstance(key, str) else key
                    game_id = key_str.replace("odds:", "")

                    odds_data = await self.redis.get(key_str)
                    if not odds_data:
                        continue

                    # Verificar oportunidades para cada mercado
                    for market_type in ['moneyline_home', 'moneyline_away', 'spread_home', 'spread_away']:
                        opp = self._evaluate_opportunity(game_id, odds_data, market_type)
                        if opp:
                            opportunities.append(opp)

                except Exception as e:
                    logger.debug(f"Erro processando {key}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Erro ao verificar oportunidades: {e}")

        return opportunities

    def _evaluate_opportunity(
        self,
        game_id: str,
        odds_data: Dict[str, Any],
        market_type: str
    ) -> Optional[Dict[str, Any]]:
        """Avalia se uma odd representa uma oportunidade."""

        # Extrair dados necessários
        market_odds = odds_data.get(f'{market_type}_odds')
        model_prob = odds_data.get(f'{market_type}_prob')
        confidence = odds_data.get('confidence', 0.65)

        if not market_odds or not model_prob:
            return None

        # Calcular edge
        edge = self._calculate_edge(model_prob, market_odds)

        # Verificar critérios
        if edge < self.config.min_edge:
            return None

        if confidence < self.config.min_confidence:
            return None

        # Calcular stake
        stake = self._calculate_kelly_stake(model_prob, market_odds)

        if stake <= 0:
            return None

        # Construir aposta
        home_team = odds_data.get('home_team', 'HOME')
        away_team = odds_data.get('away_team', 'AWAY')
        matchup = f"{away_team} @ {home_team}"

        fair_odds = 1 / model_prob if model_prob > 0 else None

        return {
            'game_id': game_id,
            'matchup': matchup,
            'bet_type': market_type,
            'market_odds': market_odds,
            'fair_odds': fair_odds,
            'stake': round(stake, 2),
            'model_prob': model_prob,
            'confidence': confidence,
            'edge': edge
        }

    async def record_bet(self, bet_data: Dict[str, Any]) -> bool:
        """Registra aposta simulada no banco de dados."""
        try:
            insert_sql = """
            INSERT INTO paper_bets
            (game_id, matchup, bet_type, market_odds, fair_odds, stake,
             model_prob, confidence, edge, status, created_at)
            VALUES
            (:game_id, :matchup, :bet_type, :market_odds, :fair_odds, :stake,
             :model_prob, :confidence, :edge, 'PENDING', :created_at)
            """

            async with self.db.get_session() as session:
                await session.execute(text(insert_sql), {
                    **bet_data,
                    'created_at': datetime.utcnow()
                })
                await session.commit()

            self.total_bets += 1

            logger.info(
                f"📝 PAPER BET #{self.total_bets}: {bet_data['matchup']} | "
                f"{bet_data['bet_type']} @ {bet_data['market_odds']:.2f} | "
                f"Stake: R$ {bet_data['stake']:.2f} | "
                f"Edge: {bet_data['edge']:.1%}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao registrar aposta: {e}")
            return False

    async def get_pending_bets(self) -> List[Dict[str, Any]]:
        """Busca apostas pendentes."""
        try:
            query = """
            SELECT id, game_id, matchup, bet_type, market_odds, stake,
                   model_prob, edge, created_at
            FROM paper_bets
            WHERE status = 'PENDING'
            ORDER BY created_at DESC
            """
            async with self.db.get_session() as session:
                result = await session.execute(text(query))
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]
        except Exception as e:
            logger.error(f"❌ Erro ao buscar apostas pendentes: {e}")
            return []

    async def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Retorna estatísticas dos últimos N dias."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            query = """
            SELECT
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status = 'WIN' THEN 1 END) as wins,
                COUNT(CASE WHEN status = 'LOSS' THEN 1 END) as losses,
                COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending,
                COALESCE(SUM(profit), 0) as total_pnl,
                COALESCE(SUM(stake), 0) as total_staked
            FROM paper_bets
            WHERE created_at >= :cutoff
            """

            async with self.db.get_session() as session:
                result = await session.execute(text(query), {'cutoff': cutoff})
                row = result.fetchone()

                if row:
                    data = dict(row._mapping)
                    total = data['wins'] + data['losses']
                    data['win_rate'] = data['wins'] / total if total > 0 else 0
                    data['roi'] = data['total_pnl'] / data['total_staked'] if data['total_staked'] > 0 else 0
                    return data

                return {
                    'total_bets': 0, 'wins': 0, 'losses': 0, 'pending': 0,
                    'total_pnl': 0, 'total_staked': 0, 'win_rate': 0, 'roi': 0
                }

        except Exception as e:
            logger.error(f"❌ Erro ao buscar stats: {e}")
            return {}

    async def run_once(self) -> int:
        """Executa uma verificação de oportunidades."""
        opportunities = await self.check_opportunities()

        for opp in opportunities:
            await self.record_bet(opp)

        return len(opportunities)

    async def run_daemon(self):
        """Executa daemon em loop infinito."""
        self.running = True
        logger.info("🚀 Paper Trading Daemon iniciado")

        while self.running:
            try:
                if self._check_stop_file():
                    await asyncio.sleep(60)  # Check again in 1 min
                    continue

                bets_made = await self.run_once()

                if bets_made > 0:
                    logger.info(f"✅ {bets_made} oportunidade(s) registrada(s)")

                await asyncio.sleep(self.config.poll_interval)

            except asyncio.CancelledError:
                logger.info("🛑 Daemon interrompido")
                break
            except Exception as e:
                logger.error(f"❌ Erro no daemon: {e}")
                await asyncio.sleep(30)

        await self.close()

    async def close(self):
        """Fecha conexões."""
        if self.db:
            await self.db.close()
        if self.redis:
            await self.redis.disconnect()
        logger.info("👋 PaperTradingEngine encerrado")


# =============================================================================
# CLI
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='Paper Trading Daemon - NBA Predictor')
    parser.add_argument('--bankroll', type=float, default=1000.0,
                        help='Bankroll inicial (default: 1000)')
    parser.add_argument('--min-edge', type=float, default=0.03,
                        help='Edge mínimo (default: 0.03 = 3%%)')
    parser.add_argument('--min-confidence', type=float, default=0.60,
                        help='Confiança mínima (default: 0.60)')
    parser.add_argument('--daemon', action='store_true',
                        help='Rodar em modo daemon (loop infinito)')
    parser.add_argument('--report', action='store_true',
                        help='Mostrar relatório dos últimos 7 dias')
    parser.add_argument('--days', type=int, default=7,
                        help='Dias para relatório (default: 7)')

    args = parser.parse_args()

    config = PaperTradingConfig(
        bankroll=args.bankroll,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence
    )

    engine = PaperTradingEngine(config)

    if not await engine.initialize():
        logger.error("❌ Falha ao inicializar engine")
        sys.exit(1)

    if args.report:
        # Modo relatório
        stats = await engine.get_stats(args.days)
        print("\n" + "=" * 50)
        print(f"📊 PAPER TRADING - ÚLTIMOS {args.days} DIAS")
        print("=" * 50)
        print(f"Total Apostas: {stats.get('total_bets', 0)}")
        print(f"Wins: {stats.get('wins', 0)}")
        print(f"Losses: {stats.get('losses', 0)}")
        print(f"Pendentes: {stats.get('pending', 0)}")
        print(f"Win Rate: {stats.get('win_rate', 0):.1%}")
        print(f"PnL Total: R$ {stats.get('total_pnl', 0):+,.2f}")
        print(f"ROI: {stats.get('roi', 0):.2%}")
        print("=" * 50 + "\n")
        await engine.close()

    elif args.daemon:
        # Modo daemon
        try:
            await engine.run_daemon()
        except KeyboardInterrupt:
            logger.info("🛑 Interrompido pelo usuário")
            await engine.close()

    else:
        # Executar uma vez
        bets = await engine.run_once()
        print(f"\n✅ {bets} oportunidade(s) registrada(s)\n")

        # Mostrar apostas pendentes
        pending = await engine.get_pending_bets()
        if pending:
            print("📋 Apostas Pendentes:")
            for bet in pending[:10]:
                print(f"  - {bet['matchup']} | {bet['bet_type']} @ {bet['market_odds']:.2f}")

        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
