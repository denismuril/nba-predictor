"""
Backtest Histórico - Pipeline de Produção (v16.0)

Valida a precisão do algoritmo `calcular_power_rating_v11` usando jogos reais dos últimos 30 dias.
NOTA: Usa estatísticas ATUAIS como proxy para o momento do jogo (limitação aceita para validação de lógica).

Usage:
    python tests/backtest_historico.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Importações do Sistema
from core.algorithms import calcular_power_rating_v11
from data.scrapers.stats_scraper import StatsScraper
from data.scrapers.standings_scraper import StandingsScraper
from utils.team_normalization import normalize_team

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_production_data():
    """Busca todos os dados necessários para o algoritmo (Stats, Standings)."""
    logger.info("📥 Carregando dados de produção (Stats & Standings)...")
    
    stats_scraper = StatsScraper()
    standings_scraper = StandingsScraper()
    
    # Executar em paralelo
    dfs_stats, standings = await asyncio.gather(
        stats_scraper.get_stats(),
        standings_scraper.get_standings()
    )
    
    logger.info("✅ Dados de produção carregados.")
    return dfs_stats, standings

def get_recent_games(days=30):
    """Busca jogos recentes via NBA API."""
    logger.info(f"📅 Buscando jogos dos últimos {days} dias...")
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2025-26', league_id_nullable='00')
        games = gamefinder.get_data_frames()[0]
        
        # Filtrar e limpar
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        cutoff = datetime.now() - timedelta(days=days)
        recent = games[games['GAME_DATE'] >= cutoff].copy()
        
        logger.info(f"DEBUG: Total games found: {len(games)}")
        logger.info(f"DEBUG: Recent games found: {len(recent)}")
        if not recent.empty:
            logger.info(f"DEBUG: Sample MATCHUPs: {recent['MATCHUP'].head().tolist()}")
        
        # Manter apenas jogos finalizados e únicos (remover duplicatas de HOME/AWAY)
        # O endpoint retorna 2 linhas por jogo (uma para cada time). Vamos pegar apenas onde MATCHUP contém '@' (Away view) ou vs (Home view)
        # Melhor: Pegar apenas linhas onde o time é HOME para evitar duplicatas
        # NBA API convention: 'Team vs. Opponent' = Home, 'Team @ Opponent' = Away
        recent = recent[recent['MATCHUP'].str.contains(' vs. ')]
        
        logger.info(f"✅ {len(recent)} jogos únicos encontrados.")
        return recent
    except Exception as e:
        logger.error(f"❌ Erro ao buscar jogos: {e}")
        return pd.DataFrame()

def run_backtest():
    """Executa o backtest."""
    # 1. Carregar dados do sistema
    loop = asyncio.get_event_loop()
    dfs_stats, standings = loop.run_until_complete(get_production_data())
    
    if not dfs_stats or not standings:
        logger.error("❌ Falha ao carregar dados do sistema.")
        return

    # 2. Carregar jogos
    games_df = get_recent_games(days=30)
    if games_df.empty:
        return

    results = []
    
    logger.info("\n🚀 Iniciando Simulação de Jogos...")
    
    for _, game in games_df.iterrows():
        # Identificar times
        try:
            # MATCHUP ex: "BOS vs. NYK" -> Home: BOS, Away: NYK
            matchup = game['MATCHUP']
            parts = matchup.split(' vs. ')
            team_home_code = parts[0]
            team_away_code = parts[1]
            
            # Normalizar nomes (O gamefinder usa abrevs, nosso sistema usa nomes completos ou abrevs mapeadas)
            # Precisamos converter 'BOS' -> 'Boston Celtics' para o sistema funcionar bem se ele esperar nomes completos
            # Mas o normalize_team lida com abrevs.
            
            # O sistema espera nomes que o normalize_team entenda.
            home_team = normalize_team(team_home_code)
            away_team = normalize_team(team_away_code)
            
            # Resultado real
            # WL 'W' -> Home Win (já que filtramos por Home games)
            actual_winner = 1 if game['WL'] == 'W' else 0
            
            # 3. Executar Algoritmo
            # Passamos None para injuries/referees/shot_quality para simplificar (ou poderíamos tentar mockar)
            # Para backtest robusto, ideal seria ter histórico, mas usaremos 'neutral' (None)
            prediction = calcular_power_rating_v11(
                home_team, away_team,
                injuries=None, # Sem dados históricos de lesão fáceis
                standings=standings,
                dfs=dfs_stats,
                referees=None,
                shot_quality_data=None
            )
            
            prob_home = prediction['prob_casa']
            predicted_winner = 1 if prob_home > 50 else 0
            confidence = prob_home if prob_home > 50 else (100 - prob_home)
            
            results.append({
                'date': game['GAME_DATE'],
                'home': home_team,
                'away': away_team,
                'prob_home': prob_home / 100.0, # Normalizar para 0-1
                'actual_home_win': actual_winner,
                'predicted_home_win': predicted_winner,
                'correct': 1 if actual_winner == predicted_winner else 0,
                'confidence': confidence
            })
            
        except Exception as e:
            logger.warning(f"Erro no jogo {game['MATCHUP']}: {e}")
            continue

    # 4. Análise de Resultados
    if not results:
        logger.error("❌ Nenhum resultado gerado.")
        return

    df_res = pd.DataFrame(results)
    
    acc = accuracy_score(df_res['actual_home_win'], df_res['predicted_home_win'])
    loss = log_loss(df_res['actual_home_win'], df_res['prob_home'])
    brier = brier_score_loss(df_res['actual_home_win'], df_res['prob_home'])
    
    logger.info("\n" + "="*60)
    logger.info("📊 RESULTADOS DO BACKTEST (Production Pipeline)")
    logger.info("="*60)
    logger.info(f"✅ Jogos Analisados: {len(df_res)}")
    logger.info(f"🏆 Acurácia: {acc:.2%}")
    logger.info(f"📉 Log Loss: {loss:.4f}")
    logger.info(f"🎯 Brier Score: {brier:.4f}")
    logger.info("-" * 60)
    
    # Confiança vs Acurácia
    high_conf = df_res[df_res['confidence'] > 60]
    if not high_conf.empty:
        acc_hc = accuracy_score(high_conf['actual_home_win'], high_conf['predicted_home_win'])
        logger.info(f"🔥 High Confidence (>60%): {len(high_conf)} jogos | Acurácia: {acc_hc:.2%}")
    
    logger.info("\nExemplo de Erros (Top 3 maior confiança):")
    errors = df_res[df_res['correct'] == 0].sort_values('confidence', ascending=False).head(3)
    for _, row in errors.iterrows():
        logger.info(f"❌ {row['home']} vs {row['away']} | Previsto: {row['predicted_home_win']} ({row['confidence']:.1f}%) | Real: {row['actual_home_win']}")

if __name__ == "__main__":
    run_backtest()

