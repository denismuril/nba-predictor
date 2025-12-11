#!/usr/bin/env python3
"""
Script para Popular Banco de Dados com Jogos Históricos

Busca jogos finalizados de temporadas anteriores (2023-24, 2024-25)
via ESPN API e salva no banco de dados para melhorar o treinamento do modelo ML.

Usage:
    python scripts/populate_historical_data.py [--seasons 2023-24 2024-25 2025-26]
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import time
import logging

# Adicionar diretório raiz ao path
sys.path.insert(0, '/home/denis/nba-predictor')

from data.scrapers.results_scraper import get_game_results
from data.repositories.db_manager import get_db_manager
from data.scrapers.schedule_scraper import obter_schedule
import requests

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

def get_season_date_range(season):
    """
    Retorna range de datas para uma temporada NBA.
    
    Args:
        season: String no formato '2023-24'
    
    Returns:
        Tuple: (start_date, end_date) como strings YYYYMMDD
    """
    year_start = int(season.split('-')[0])
    year_end = int(season.split('-')[1]) + 2000
    
    # NBA seasons: October to June
    start = datetime(year_start, 10, 1)
    end = datetime(year_end, 6, 30)
    
    # Não buscar futuro
    today = datetime.now()
    if end > today:
        end = today
    
    return start, end


def fetch_games_for_date_range(start_date, end_date, delay=1.0):
    """
    Busca todos os jogos em um intervalo de datas.
    
    Args:
        start_date: datetime object
        end_date: datetime object
        delay: Delay entre requests (em segundos)
    
    Returns:
        List de dicts com jogos encontrados
    """
    all_games = []
    current_date = start_date
    
    total_days = (end_date - start_date).days
    logger.info(f"📅 Buscando jogos de {start_date.date()} até {end_date.date()} ({total_days} dias)")
    
    days_processed = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    try:
                        status = event.get('status', {}).get('type', {}).get('state', '')
                        
                        # Apenas jogos finalizados
                        if status.lower() != 'post':
                            continue
                        
                        game_date = event.get('date', '')[:10]
                        competitors = event.get('competitions', [{}])[0].get('competitors', [])
                        
                        if len(competitors) != 2:
                            continue
                        
                        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                        
                        if not home or not away:
                            continue
                        
                        home_team = home.get('team', {}).get('displayName', '')
                        away_team = away.get('team', {}).get('displayName', '')
                        home_score = int(home.get('score', 0))
                        away_score = int(away.get('score', 0))
                        
                        game_id = f"{game_date}_{home_team}_{away_team}".replace(" ", "")
                        
                        all_games.append({
                            'id': game_id,
                            'date': game_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'winner': 'HOME' if home_score > away_score else 'AWAY'
                        })
                        
                    except Exception as e:
                        logger.debug(f"Erro ao processar evento: {e}")
                        continue
            
            days_processed += 1
            if days_processed % 30 == 0:
                logger.info(f"  Progresso: {days_processed}/{total_days} dias ({len(all_games)} jogos encontrados)")
            
            # Delay para não sobrecarregar API
            time.sleep(delay)
            
        except Exception as e:
            logger.warning(f"Erro ao buscar {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    logger.info(f"✅ Busca concluída: {len(all_games)} jogos finalizados encontrados")
    return all_games


def save_games_to_db(games):
    """
    Salva jogos no banco de dados.
    
    Args:
        games: Lista de dicts com dados dos jogos
    
    Returns:
        int: Número de jogos salvos
    """
    if not games:
        logger.info("Nenhum jogo para salvar")
        return 0
    
    db = get_db_manager()
    
    # Converter para formato de predições (para compatibilidade com DB)
    predictions = []
    for game in games:
        pred = {
            'Data': game['date'],
            'Casa': game['home_team'],
            'Visitante': game['away_team'],
            'Prob Casa %': 50.0,  # Placeholder (não temos predição histórica)
            'Prob Visitante %': 50.0,
            'Odd Casa': 0.0,
            'Odd Visitante': 0.0,
            'Confiança': 'N/A'
        }
        predictions.append(pred)
    
    # Salvar predições (criará registros no DB)
    db.save_predictions(predictions)
    
    # Atualizar com resultados
    saved_count = 0
    for game in games:
        try:
            db.update_game_result(
                game_id=game['id'],
                home_score=game['home_score'],
                away_score=game['away_score']
            )
            saved_count += 1
        except Exception as e:
            logger.warning(f"Erro ao salvar jogo {game['id']}: {e}")
    
    logger.info(f"💾 {saved_count} jogos salvos no banco de dados")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='Popular banco com dados históricos da NBA')
    parser.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25'],
                       help='Temporadas para buscar (ex: 2023-24 2024-25)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay entre requests em segundos (default: 0.5)')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("📥 POPULANDO BANCO COM DADOS HISTÓRICOS DA NBA")
    logger.info("="*80)
    logger.info(f"Temporadas: {', '.join(args.seasons)}")
    logger.info(f"Delay entre requests: {args.delay}s")
    logger.info("")
    
    total_games_saved = 0
    
    for season in args.seasons:
        logger.info(f"\n🏀 Processando temporada {season}...")
        
        try:
            # Obter range de datas
            start_date, end_date = get_season_date_range(season)
            
            logger.info(f"   Período: {start_date.date()} a {end_date.date()}")
            
            # Buscar jogos
            games = fetch_games_for_date_range(start_date, end_date, delay=args.delay)
            
            if not games:
                logger.warning(f"⚠️  Nenhum jogo encontrado para {season}")
                continue
            
            # Salvar no banco
            saved = save_games_to_db(games)
            total_games_saved += saved
            
            logger.info(f"✅ Temporada {season}: {saved} jogos salvos")
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar temporada {season}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"✅ PROCESSO CONCLUÍDO!")
    logger.info(f"   Total de jogos salvos: {total_games_saved}")
    logger.info("="*80)
    logger.info("")
    logger.info("Próximos passos:")
    logger.info("  1. Verificar dados: PYTHONPATH=/home/denis/nba-predictor python -c \"from ml_pipeline.data_preparation import load_multi_season_data; df = load_multi_season_data(['2023-24', '2024-25', '2025-26']); print(f'Total: {len(df)} jogos')\"")
    logger.info("  2. Treinar modelo: cd /home/denis/nba-predictor && PYTHONPATH=/home/denis/nba-predictor ./venv/bin/python ml_pipeline/train_ensemble_v3.py")
    logger.info("")
    
    return 0 if total_games_saved > 0 else 1


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
