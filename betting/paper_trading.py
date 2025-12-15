"""
Paper Trading Engine - NBA Predictor Go Live v24.0
===================================================
Simula apostas reais sem arriscar dinheiro.
Registra sinais do SniperEngine em PostgreSQL para análise posterior.

MODO: Paper Trading (7 dias de validação)
"""
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
PAPER_TRADING_ENABLED = os.getenv('PAPER_TRADING_MODE', 'true').lower() == 'true'
POSTGRES_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/nba_predictor')


@dataclass
class PaperBet:
    """Representa uma aposta em paper trading."""
    id: Optional[int] = None
    timestamp_sinal: datetime = None
    game_id: str = ""
    home_team: str = ""
    away_team: str = ""
    game_date: str = ""
    tipo_aposta: str = ""  # 'moneyline_home', 'moneyline_away', 'over', 'under'
    odd_capturada: float = 0.0
    stake_sugerida: float = 0.0
    stake_pct: float = 0.0
    modelo_confianca: float = 0.0
    edge_pct: float = 0.0
    fair_odds: float = 0.0
    status: str = "PENDENTE"  # PENDENTE, WON, LOST, VOID
    resultado_real: Optional[str] = None
    pnl: Optional[float] = None
    settled_at: Optional[datetime] = None


class PaperTradingDB:
    """Gerencia conexão com PostgreSQL para paper bets."""
    
    def __init__(self, postgres_url: str = POSTGRES_URL):
        self.postgres_url = postgres_url
        self._pool = None
    
    async def initialize(self):
        """Inicializa pool de conexões e cria tabela."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.postgres_url, min_size=1, max_size=5)
            await self._create_tables()
            logger.info("✅ PaperTradingDB inicializado")
        except ImportError:
            logger.error("❌ asyncpg não instalado. Execute: pip install asyncpg")
            raise
        except Exception as e:
            logger.error(f"❌ Erro conectando ao PostgreSQL: {e}")
            raise
    
    async def _create_tables(self):
        """Cria tabela paper_bets se não existir."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_bets (
                    id SERIAL PRIMARY KEY,
                    timestamp_sinal TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    game_id VARCHAR(100),
                    home_team VARCHAR(10),
                    away_team VARCHAR(10),
                    game_date DATE,
                    tipo_aposta VARCHAR(50),
                    odd_capturada DECIMAL(6,3),
                    stake_sugerida DECIMAL(10,2),
                    stake_pct DECIMAL(5,4),
                    modelo_confianca DECIMAL(5,4),
                    edge_pct DECIMAL(5,2),
                    fair_odds DECIMAL(6,3),
                    status VARCHAR(20) DEFAULT 'PENDENTE',
                    resultado_real VARCHAR(100),
                    pnl DECIMAL(12,2),
                    settled_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_paper_bets_status ON paper_bets(status);
                CREATE INDEX IF NOT EXISTS idx_paper_bets_game_date ON paper_bets(game_date);
            """)
            logger.info("📊 Tabela paper_bets criada/verificada")
    
    async def insert_bet(self, bet: PaperBet) -> int:
        """Insere nova aposta paper."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO paper_bets (
                    timestamp_sinal, game_id, home_team, away_team, game_date,
                    tipo_aposta, odd_capturada, stake_sugerida, stake_pct,
                    modelo_confianca, edge_pct, fair_odds, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
            """, 
                bet.timestamp_sinal or datetime.now(timezone.utc),
                bet.game_id, bet.home_team, bet.away_team, bet.game_date,
                bet.tipo_aposta, bet.odd_capturada, bet.stake_sugerida, bet.stake_pct,
                bet.modelo_confianca, bet.edge_pct, bet.fair_odds, bet.status
            )
            return row['id']
    
    async def get_pending_bets(self, game_date: str = None) -> List[PaperBet]:
        """Busca apostas pendentes de liquidação."""
        async with self._pool.acquire() as conn:
            if game_date:
                rows = await conn.fetch("""
                    SELECT * FROM paper_bets 
                    WHERE status = 'PENDENTE' AND game_date = $1
                    ORDER BY timestamp_sinal
                """, game_date)
            else:
                rows = await conn.fetch("""
                    SELECT * FROM paper_bets 
                    WHERE status = 'PENDENTE'
                    ORDER BY game_date, timestamp_sinal
                """)
            return [PaperBet(**dict(row)) for row in rows]
    
    async def settle_bet(self, bet_id: int, won: bool, resultado: str, pnl: float):
        """Liquida uma aposta com resultado."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE paper_bets 
                SET status = $1, resultado_real = $2, pnl = $3, settled_at = $4
                WHERE id = $5
            """, 'WON' if won else 'LOST', resultado, pnl, datetime.now(timezone.utc), bet_id)
    
    async def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Retorna estatísticas de paper trading."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_bets,
                    COUNT(*) FILTER (WHERE status = 'WON') as wins,
                    COUNT(*) FILTER (WHERE status = 'LOST') as losses,
                    COUNT(*) FILTER (WHERE status = 'PENDENTE') as pending,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(SUM(stake_sugerida), 0) as total_staked,
                    COALESCE(AVG(edge_pct), 0) as avg_edge
                FROM paper_bets
                WHERE created_at >= NOW() - INTERVAL '%s days'
            """ % days)
            
            total = row['total_bets'] or 0
            wins = row['wins'] or 0
            
            return {
                'total_bets': total,
                'wins': wins,
                'losses': row['losses'] or 0,
                'pending': row['pending'] or 0,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl': float(row['total_pnl'] or 0),
                'total_staked': float(row['total_staked'] or 0),
                'roi': (float(row['total_pnl'] or 0) / float(row['total_staked'] or 1)) * 100,
                'avg_edge': float(row['avg_edge'] or 0)
            }
    
    async def close(self):
        """Fecha pool de conexões."""
        if self._pool:
            await self._pool.close()


class PaperTradingEngine:
    """
    Engine de Paper Trading que intercepta sinais do SniperEngine.
    
    Modo:
    - Recebe ValueAlerts do SniperEngine
    - Registra em PostgreSQL (não envia para Telegram/API)
    - Gera relatórios de performance simulada
    """
    
    def __init__(self, bankroll: float = 1000.0, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.db = PaperTradingDB()
        self._running = False
    
    async def initialize(self):
        """Inicializa conexões."""
        await self.db.initialize()
        logger.info(f"🎮 Paper Trading Engine iniciado (Banca: R$ {self.bankroll:,.2f})")
    
    async def process_alert(self, alert: Dict[str, Any]) -> int:
        """
        Processa um alerta do SniperEngine e registra como paper bet.
        
        Args:
            alert: Dict com dados do ValueAlert
            
        Returns:
            ID da aposta registrada
        """
        # Construir PaperBet a partir do alert
        bet = PaperBet(
            timestamp_sinal=datetime.now(timezone.utc),
            game_id=alert.get('game_id', ''),
            home_team=alert.get('home_team', ''),
            away_team=alert.get('away_team', ''),
            game_date=alert.get('game_date', datetime.now().strftime('%Y-%m-%d')),
            tipo_aposta=f"moneyline_{alert.get('bet_side', 'home')}",
            odd_capturada=alert.get('market_odds', 0),
            stake_sugerida=alert.get('kelly_stake_pct', 0) * self.bankroll / 100,
            stake_pct=alert.get('kelly_stake_pct', 0) / 100,
            modelo_confianca=alert.get('confidence', 0),
            edge_pct=alert.get('edge_pct', 0),
            fair_odds=alert.get('fair_odds', 0),
            status='PENDENTE'
        )
        
        bet_id = await self.db.insert_bet(bet)
        
        logger.info(
            f"📝 Paper Bet #{bet_id}: {bet.home_team} vs {bet.away_team} | "
            f"{bet.tipo_aposta} @ {bet.odd_capturada:.2f} | "
            f"Stake: R$ {bet.stake_sugerida:.2f} | Edge: {bet.edge_pct:.1f}%"
        )
        
        return bet_id
    
    async def start_with_sniper(self):
        """
        Inicia paper trading integrado com SniperEngine.
        Intercepta alertas e registra como paper bets.
        """
        from betting.sniper_engine import SniperEngine
        
        sniper = SniperEngine(bankroll=self.bankroll, kelly_fraction=self.kelly_fraction)
        
        # Override do método _send_alert para capturar sinais
        original_send_alert = sniper._send_alert
        
        async def paper_send_alert(alert):
            # Registrar como paper bet
            await self.process_alert(asdict(alert))
            # Log em vez de enviar para Telegram
            logger.info(f"📤 [PAPER] Alerta capturado: {alert.game_id}")
        
        sniper._send_alert = paper_send_alert
        
        logger.info("🎮 Paper Trading Mode: Sinais serão registrados, não enviados")
        
        self._running = True
        await sniper.start()
    
    async def get_report(self, days: int = 7) -> str:
        """Gera relatório de performance."""
        stats = await self.db.get_stats(days)
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║          📊 PAPER TRADING REPORT - {days} DIAS             ║
╠════════════════════════════════════════════════════════════╣
║  Total de Apostas:     {stats['total_bets']:>5}                         ║
║  Vitórias:             {stats['wins']:>5}                              ║
║  Derrotas:             {stats['losses']:>5}                              ║
║  Pendentes:            {stats['pending']:>5}                              ║
║  Win Rate:             {stats['win_rate']:>5.1f}%                            ║
╠════════════════════════════════════════════════════════════╣
║  Total Apostado:       R$ {stats['total_staked']:>10,.2f}                  ║
║  PnL Total:            R$ {stats['total_pnl']:>+10,.2f}                  ║
║  ROI:                  {stats['roi']:>+6.2f}%                            ║
║  Edge Médio:           {stats['avg_edge']:>5.1f}%                            ║
╚════════════════════════════════════════════════════════════╝

{'✅ LUCRATIVO!' if stats['total_pnl'] > 0 else '⚠️ REVISAR ESTRATÉGIA' if stats['total_pnl'] < 0 else '➖ NEUTRO'}

Se tivéssemos apostado: R$ {stats['total_pnl']:+,.2f}
"""
        return report
    
    async def close(self):
        """Encerra engine."""
        self._running = False
        await self.db.close()


# =============================================================================
# CLI
# =============================================================================
async def main():
    """Entry point para paper trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Paper Trading Engine')
    parser.add_argument('--bankroll', type=float, default=1000.0, help='Banca inicial')
    parser.add_argument('--kelly', type=float, default=0.25, help='Kelly fraction')
    parser.add_argument('--report', action='store_true', help='Gera relatório')
    parser.add_argument('--days', type=int, default=7, help='Dias para relatório')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    engine = PaperTradingEngine(bankroll=args.bankroll, kelly_fraction=args.kelly)
    await engine.initialize()
    
    if args.report:
        report = await engine.get_report(args.days)
        print(report)
    else:
        print("🎮 Iniciando Paper Trading Mode...")
        print("   Apostas serão simuladas, não reais.")
        print("   Pressione Ctrl+C para parar.\n")
        
        try:
            await engine.start_with_sniper()
        except KeyboardInterrupt:
            print("\n🛑 Paper Trading encerrado")
        finally:
            await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
