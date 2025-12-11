#!/usr/bin/env python3
"""
Script para Atualizar Jogos com Scores Zerados

Busca jogos no banco que têm score = 0 e atualiza com
os resultados corretos via ESPN API.

Usage:
    python scripts/fix_missing_scores.py [--limit N]
"""

import sys
import os
import argparse
from datetime import datetime
import time
import logging
import requests

# Adicionar diretório raiz ao path
sys.path.insert(0, '/home/denis/nba-predictor')

from data.repositories.db_manager import get_db_manager
import pandas as pd

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def get_games_with_missing_scores():
    """
    Busca jogos no banco que têm score = 0.
    
    Returns:
        DataFrame com jogos sem scores
    """
    db = get_db_manager()
    df = db.get_comprehensive_history()
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Filtrar jogos com scores zerados
    missing = df[(df['home_score'] == 0) | (df['away_score'] == 0)].copy()
    
    return missing


def fetch_game_result(game_date, home_team, away_team):
    """
    Busca resultado de um jogo específico via ESPN API.
    
    Args:
        game_date: Data do jogo (YYYY-MM-DD)
        home_team: Nome do time da casa
        away_team: Nome do time visitante
    
    Returns:
        Dict com {home_score, away_score} ou None se não encontrado
    """
    try:
        # Converter data para formato ESPN (YYYYMMDD)
        date_str = game_date.replace('-', '')
        
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        events = data.get('events', [])
        
        for event in events:
            try:
                status = event.get('status', {}).get('type', {}).get('state', '')
                
                # Apenas jogos finalizados
                if status.lower() != 'post':
                    continue
                
                competitors = event.get('competitions', [{}])[0].get('competitors', [])
                
                if len(competitors) != 2:
                    continue
                
                home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                
                if not home or not away:
                    continue
                
                event_home_team = home.get('team', {}).get('displayName', '')
                event_away_team = away.get('team', {}).get('displayName', '')
                
                # Match teams (case insensitive)
                if (event_home_team.lower() == home_team.lower() and 
                    event_away_team.lower() == away_team.lower()):
                    
                    home_score = int(home.get('score', 0))
                    away_score = int(away.get('score', 0))
                    
                    return {
                        'home_score': home_score,
                        'away_score': away_score
                    }
                    
            except Exception as e:
                logger.debug(f"Erro ao processar evento: {e}")
                continue
        
        return None
        
    except Exception as e:
        logger.debug(f"Erro ao buscar jogo {game_date}: {e}")
        return None


def update_game_scores(game_id, home_score, away_score):
    """
    Atualiza scores de um jogo no banco.
    
    Args:
        game_id: ID do jogo
        home_score: Score do time da casa
        away_score: Score do time visitante
    
    Returns:
        bool: True se sucesso
    """
    try:
        db = get_db_manager()
        db.update_game_result(game_id, home_score, away_score)
        return True
    except Exception as e:
        logger.error(f"Erro ao atualizar jogo {game_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Corrigir jogos com scores faltando')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limitar número de jogos a processar')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay entre requests em segundos (default: 0.5)')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🔧 CORRIGINDO JOGOS COM SCORES FALTANDO")
    logger.info("="*80)
    
    # Buscar jogos com scores faltando
    logger.info("📊 Buscando jogos com scores zerados no banco...")
    missing_games = get_games_with_missing_scores()
    
    if missing_games.empty:
        logger.info("✅ Nenhum jogo com score faltando encontrado!")
        return 0
    
    total_missing = len(missing_games)
    logger.info(f"⚠️  Encontrados {total_missing} jogos com scores faltando")
    
    if args.limit:
        missing_games = missing_games.head(args.limit)
        logger.info(f"   Processando apenas {len(missing_games)} jogos (--limit {args.limit})")
    
    logger.info("")
    
    # Processar cada jogo
    updated_count = 0
    failed_count = 0
    
    for idx, row in missing_games.iterrows():
        game_date = str(row['date'])[:10]  # YYYY-MM-DD
        home_team = row['home_team']
        away_team = row['away_team']
        game_id = f"{game_date}_{home_team}_{away_team}".replace(" ", "")
        
        logger.info(f"[{idx+1}/{len(missing_games)}] {game_date}: {home_team} vs {away_team}")
        
        # Buscar resultado
        result = fetch_game_result(game_date, home_team, away_team)
        
        if result and result['home_score'] > 0 and result['away_score'] > 0:
            # Atualizar banco
            success = update_game_scores(
                game_id,
                result['home_score'],
                result['away_score']
            )
            
            if success:
                logger.info(f"   ✅ Atualizado: {result['home_score']}-{result['away_score']}")
                updated_count += 1
            else:
                logger.warning(f"   ⚠️  Falha ao salvar no banco")
                failed_count += 1
        else:
            logger.warning(f"   ❌ Resultado não encontrado")
            failed_count += 1
        
        # Delay para não sobrecarregar API
        time.sleep(args.delay)
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"✅ PROCESSO CONCLUÍDO!")
    logger.info(f"   Total processado: {len(missing_games)}")
    logger.info(f"   Atualizados: {updated_count}")
    logger.info(f"   Falhas: {failed_count}")
    logger.info("="*80)
    logger.info("")
    
    if updated_count > 0:
        logger.info("Próximos passos:")
        logger.info("  1. Verificar dados: PYTHONPATH=/home/denis/nba-predictor python -c \"from ml_pipeline.data_preparation import load_historical_data; df = load_historical_data(seasons=['2023-24', '2024-25', '2025-26']); print(f'Total: {len(df)} jogos válidos')\"")
        logger.info("  2. Retreinar modelo: cd /home/denis/nba-predictor && PYTHONPATH=/home/denis/nba-predictor ./venv/bin/python ml_pipeline/train_ensemble_v3.py")
        logger.info("")
    
    return 0 if updated_count > 0 else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Processo interrompido pelo usuário.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
