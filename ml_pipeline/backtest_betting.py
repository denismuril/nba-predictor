"""
Financial Backtesting - Simula apostas históricas para calcular P&L real.

Simula Kelly Criterion nos últimos 400 jogos para ver quanto teríamos lucrado/perdido.
"""
import logging
from datetime import datetime
from data.repositories.betting_tracker import BettingTracker
from data.repositories.db_manager import get_db_manager
from utils.kelly import get_bet_recommendation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger  = logging.getLogger(__name__)

def run_financial_backtest(start_date='2024-10-22', initial_bankroll=1000, kelly_fraction=0.25):
    """
    Simula apostas históricas usando Kelly Criterion.
    
    Parameters:
        start_date: Data inicial do backtest
        initial_bankroll: Banca inicial em $
        kelly_fraction: Fração do Kelly (0.25 = Quarter Kelly)
    """
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🎲 BACKTESTING FINANCEIRO - Kelly Criterion")
    logger.info(f"{'='*70}")
    logger.info(f"Período: {start_date} até hoje")
    logger.info(f"Banca Inicial: ${initial_bankroll:,.2f}")
    logger.info(f"Estratégia: {kelly_fraction*100:.0f}% Kelly (conservador)")
    logger.info(f"{'='*70}\n")
    
    # Criar tracker de backtest
    tracker = BettingTracker('data/backtest_bets.db')
    db = get_db_manager()
    
    # Buscar jogos históricos com resultados
    games = db.get_comprehensive_history()
    games = games[games['winner'].notna()].copy()  # Apenas jogos finalizados
    
    if start_date:
        games = games[games['date'] >= start_date]
    
    games = games.sort_values('date')
    
    logger.info(f"📊 {len(games)} jogos encontrados para backtest\n")
    
    bankroll = initial_bankroll
    bets_placed = 0
    
    for idx, game in games.iterrows():
        # Simular decisão de Kelly usando probabilidades do jogo
        # (Em produção, usaríamos previsão point-in-time, mas para simplificação usamos prob armazenada)
        prob_home = game.get('prob_home', 50)
        prob_away = game.get('prob_away', 50)
        
        odd_home = game.get('odds_home', 1.90)
        odd_away = game.get('odds_away', 1.90)
        
        # Decisão de Kelly
        kelly_rec = get_bet_recommendation(
            prob_home,
            prob_away,
            odd_home,
            odd_away,
            fractional=kelly_fraction
        )
        
        if kelly_rec['recommendation'] != 'NO BET' and float(kelly_rec['stake_pct']) > 0:
            # Registrar aposta
            side = kelly_rec['recommendation']
            bet_odds = odd_home if side == 'HOME' else odd_away
            
            bet_id = tracker.log_bet(
                game_date=str(game['date'].date()),
                game_id=f"{game['home_team']}_{game['away_team']}_{str(game['date'].date())}",
                home_team=game['home_team'],
                away_team=game['away_team'],
                side=side,
                bet_odds=bet_odds,
                stake_pct=kelly_rec['stake_pct'],
                bankroll=bankroll,
                model_prob=prob_home if side == 'HOME' else prob_away,
                ev_pct=kelly_rec['ev']
            )
            
            # Resultado real
            won = (side == 'HOME' and game['winner'] == 'HOME') or \
                  (side == 'AWAY' and game['winner'] != 'HOME')
            
            tracker.update_result(bet_id, won)
            
            # Atualizar banca
            bet = tracker.get_bet(bet_id)
            bankroll += bet['profit']
            bets_placed += 1
            
            if bets_placed % 10 == 0:
                logger.info(f"   Progresso: {bets_placed} apostas, Banca: ${bankroll:,.2f}")
    
    # Relatório final
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ RESULTADO DO BACKTEST")
    logger.info(f"{'='*70}\n")
    
    final_metrics = tracker.get_performance_metrics('all')
    
    print(f"Banca Inicial:       ${initial_bankroll:,.2f}")
    print(f"Banca Final:         ${bankroll:,.2f}")
    print(f"Lucro/Prejuízo:      ${bankroll - initial_bankroll:+,.2f}")
    print(f"ROI:                 {((bankroll/initial_bankroll - 1) * 100):+.2f}%")
    print(f"\nTotal Apostas:       {final_metrics['total_bets']}")
    print(f"Win Rate:            {final_metrics['win_rate']:.1f}%")
    print(f"Total Apostado:      ${final_metrics['total_staked']:,.2f}")
    print(f"Sharpe Ratio:        {final_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:        ${final_metrics['max_drawdown']:,.2f}")
    print(f"Avg EV:              {final_metrics['avg_ev']:+.2f}%")
    print(f"\nMelhor Aposta:       ${final_metrics['best_bet']['profit']:+.2f} @ {final_metrics['best_bet']['odds']}")
    print(f"Pior Aposta:         ${final_metrics['worst_bet']['profit']:+.2f} @ {final_metrics['worst_bet']['odds']}")
    print(f"\n{'='*70}\n")
    
    # Análise
    if final_metrics['roi'] > 5:
        print("✅ Sistema LUCRATIVO! ROI acima de 5%")
    elif final_metrics['roi'] > 0:
        print("⚠️  Sistema levemente lucrativo. Considere melhorias.")
    else:
        print("❌ Sistema PERDEDOR. Necessário revisar modelo/estratégia.")
    
    logger.info(f"\n📁 Dados salvos em: data/backtest_bets.db")
    logger.info(f"   Use BettingTracker('data/backtest_bets.db') para análises detalhadas.\n")
    
    return {
        'initial_bankroll': initial_bankroll,
        'final_bankroll': bankroll,
        'profit': bankroll - initial_bankroll,
        'roi': ((bankroll/initial_bankroll - 1) * 100),
        'metrics': final_metrics
    }

if __name__ == "__main__":
    run_financial_backtest()
