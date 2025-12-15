"""
Settle Paper Bets - NBA Predictor Go Live v24.0
================================================
Liquida apostas de paper trading com resultados reais.
Gera relatório de PnL: "Se tivéssemos apostado, teríamos lucrado R$ X"

Usar: python betting/settle_paper_bets.py --date 2024-12-14
"""
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from betting.paper_trading import PaperTradingDB, PaperBet

logger = logging.getLogger(__name__)


class PaperBetSettler:
    """Liquida paper bets com resultados reais."""
    
    def __init__(self):
        self.db = PaperTradingDB()
        self._results_cache: Dict[str, Dict] = {}
    
    async def initialize(self):
        """Inicializa conexões."""
        await self.db.initialize()
        logger.info("✅ PaperBetSettler inicializado")
    
    async def fetch_real_results(self, game_date: str) -> Dict[str, Dict]:
        """
        Busca resultados reais dos jogos.
        
        Returns:
            Dict[game_id, {'home_score': int, 'away_score': int, 'winner': str}]
        """
        results = {}
        
        try:
            # Tentar buscar do PostgreSQL (tabela games)
            from data.repositories.db_manager import get_db_manager
            
            db = get_db_manager()
            games = db.get_games_by_date(game_date)
            
            for game in games:
                game_id = game.get('game_id', f"{game.get('home_team')}_{game.get('away_team')}_{game_date}")
                home_score = game.get('home_score') or game.get('pts_home')
                away_score = game.get('away_score') or game.get('pts_away')
                
                if home_score is not None and away_score is not None:
                    winner = 'home' if home_score > away_score else 'away'
                    total = home_score + away_score
                    
                    results[game_id] = {
                        'home_team': game.get('home_team'),
                        'away_team': game.get('away_team'),
                        'home_score': home_score,
                        'away_score': away_score,
                        'winner': winner,
                        'total': total
                    }
                    
                    # Também criar chave por times
                    team_key = f"{game.get('home_team')}_{game.get('away_team')}"
                    results[team_key] = results[game_id]
            
            logger.info(f"📊 Encontrados {len(results)} resultados para {game_date}")
            
        except Exception as e:
            logger.error(f"❌ Erro buscando resultados: {e}")
        
        return results
    
    def _determine_bet_outcome(
        self, 
        bet: PaperBet, 
        result: Dict[str, Any]
    ) -> tuple:
        """
        Determina se a aposta foi vencedora.
        
        Returns:
            (won: bool, resultado_str: str, pnl: float)
        """
        tipo = bet.tipo_aposta.lower()
        
        # Moneyline
        if 'moneyline_home' in tipo or tipo == 'home':
            won = result['winner'] == 'home'
            resultado = f"{result['home_team']} {result['home_score']} x {result['away_score']} {result['away_team']}"
        elif 'moneyline_away' in tipo or tipo == 'away':
            won = result['winner'] == 'away'
            resultado = f"{result['home_team']} {result['home_score']} x {result['away_score']} {result['away_team']}"
        # Over/Under (se implementado)
        elif 'over' in tipo:
            line = float(bet.tipo_aposta.split('_')[-1]) if '_' in bet.tipo_aposta else 220
            won = result['total'] > line
            resultado = f"Total: {result['total']} (Linha: {line})"
        elif 'under' in tipo:
            line = float(bet.tipo_aposta.split('_')[-1]) if '_' in bet.tipo_aposta else 220
            won = result['total'] < line
            resultado = f"Total: {result['total']} (Linha: {line})"
        else:
            # Fallback: assume moneyline home
            won = result['winner'] == 'home'
            resultado = f"{result['home_team']} {result['home_score']} x {result['away_score']} {result['away_team']}"
        
        # Calcular PnL
        if won:
            # Lucro = Stake * (Odds - 1)
            pnl = bet.stake_sugerida * (bet.odd_capturada - 1)
        else:
            # Perda = -Stake
            pnl = -bet.stake_sugerida
        
        return won, resultado, round(pnl, 2)
    
    async def settle_date(self, game_date: str) -> Dict[str, Any]:
        """
        Liquida todas as apostas de uma data.
        
        Args:
            game_date: Data no formato YYYY-MM-DD
            
        Returns:
            Dict com estatísticas da liquidação
        """
        logger.info(f"🔄 Liquidando apostas de {game_date}...")
        
        # Buscar apostas pendentes
        pending_bets = await self.db.get_pending_bets(game_date)
        
        if not pending_bets:
            logger.info(f"ℹ️ Nenhuma aposta pendente para {game_date}")
            return {'settled': 0, 'pnl': 0}
        
        logger.info(f"📋 {len(pending_bets)} apostas pendentes")
        
        # Buscar resultados reais
        results = await self.fetch_real_results(game_date)
        
        if not results:
            logger.warning(f"⚠️ Nenhum resultado encontrado para {game_date}")
            return {'settled': 0, 'pnl': 0, 'message': 'Sem resultados disponíveis'}
        
        # Liquidar cada aposta
        stats = {
            'settled': 0,
            'won': 0,
            'lost': 0,
            'not_found': 0,
            'pnl': 0.0,
            'details': []
        }
        
        for bet in pending_bets:
            # Tentar encontrar resultado
            game_key = f"{bet.home_team}_{bet.away_team}"
            result = results.get(bet.game_id) or results.get(game_key)
            
            if not result:
                stats['not_found'] += 1
                logger.warning(f"⚠️ Resultado não encontrado: {bet.home_team} vs {bet.away_team}")
                continue
            
            # Determinar outcome
            won, resultado_str, pnl = self._determine_bet_outcome(bet, result)
            
            # Atualizar no banco
            await self.db.settle_bet(bet.id, won, resultado_str, pnl)
            
            # Estatísticas
            stats['settled'] += 1
            stats['pnl'] += pnl
            if won:
                stats['won'] += 1
            else:
                stats['lost'] += 1
            
            stats['details'].append({
                'bet_id': bet.id,
                'game': f"{bet.home_team} vs {bet.away_team}",
                'tipo': bet.tipo_aposta,
                'odd': bet.odd_capturada,
                'stake': bet.stake_sugerida,
                'won': won,
                'pnl': pnl
            })
            
            emoji = "✅" if won else "❌"
            logger.info(
                f"{emoji} #{bet.id}: {bet.home_team} vs {bet.away_team} | "
                f"{bet.tipo_aposta} | PnL: R$ {pnl:+.2f}"
            )
        
        logger.info(
            f"\n📊 Liquidação concluída: "
            f"{stats['won']}W/{stats['lost']}L | "
            f"PnL: R$ {stats['pnl']:+.2f}"
        )
        
        return stats
    
    async def generate_report(self, game_date: str = None) -> str:
        """Gera relatório detalhado."""
        if game_date is None:
            game_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Liquidar primeiro
        stats = await self.settle_date(game_date)
        
        # Estatísticas gerais (últimos 7 dias)
        overall_stats = await self.db.get_stats(7)
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║         📊 RELATÓRIO DE PAPER TRADING - {game_date}          ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  📅 RESULTADOS DO DIA                                              ║
║  ─────────────────────────────────────────────────────────────────  ║
║  Apostas Liquidadas:    {stats['settled']:>3}                                      ║
║  Vitórias:              {stats.get('won', 0):>3}                                      ║
║  Derrotas:              {stats.get('lost', 0):>3}                                      ║
║  PnL do Dia:            R$ {stats['pnl']:>+10,.2f}                           ║
║                                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  📈 PERFORMANCE GERAL (7 DIAS)                                     ║
║  ─────────────────────────────────────────────────────────────────  ║
║  Total de Apostas:      {overall_stats['total_bets']:>3}                                      ║
║  Win Rate:              {overall_stats['win_rate']:>5.1f}%                                   ║
║  PnL Acumulado:         R$ {overall_stats['total_pnl']:>+10,.2f}                           ║
║  ROI:                   {overall_stats['roi']:>+6.2f}%                                   ║
║  Edge Médio:            {overall_stats['avg_edge']:>5.1f}%                                   ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

💡 CONCLUSÃO:
"""
        if overall_stats['total_pnl'] > 0:
            report += f"""
   ✅ Se tivéssemos apostado nos últimos 7 dias, teríamos LUCRADO: 
      R$ {overall_stats['total_pnl']:,.2f}
   
   📈 O sistema está performando bem. Considere iniciar apostas reais
      com frações menores do Kelly.
"""
        elif overall_stats['total_pnl'] < 0:
            report += f"""
   ⚠️ Se tivéssemos apostado nos últimos 7 dias, teríamos PERDIDO: 
      R$ {abs(overall_stats['total_pnl']):,.2f}
   
   📉 Revisar estratégia antes de operar com dinheiro real.
      Verificar: calibração do modelo, edge mínimo, seleção de jogos.
"""
        else:
            report += """
   ➖ Resultado neutro. Mais dados necessários para conclusão.
"""
        
        return report
    
    async def close(self):
        """Encerra conexões."""
        await self.db.close()


# =============================================================================
# CLI
# =============================================================================
async def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Liquidar Paper Bets')
    parser.add_argument('--date', type=str, help='Data YYYY-MM-DD (default: ontem)')
    parser.add_argument('--all', action='store_true', help='Liquidar todas as pendentes')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    settler = PaperBetSettler()
    await settler.initialize()
    
    try:
        if args.date:
            game_date = args.date
        elif args.all:
            # Liquidar todos os dias pendentes
            game_date = None
        else:
            # Default: ontem
            game_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        report = await settler.generate_report(game_date)
        print(report)
        
    finally:
        await settler.close()


if __name__ == "__main__":
    asyncio.run(main())
