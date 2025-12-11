#!/usr/bin/env python3
"""
Importação de Dados NBA via CSV (Kaggle ou Outros)

Importa dados de jogos NBA a partir de arquivos CSV para o banco normalizado.
Suporta múltiplos formatos de CSV (Basketball-Reference, Kaggle, etc.)

Dataset Sugerido: https://www.kaggle.com/datasets/kevinpickelman/nba-data-2012-2024
Instruções:
1. Baixe o CSV do Kaggle
2. Coloque em data/csv/games.csv
3. Execute: python3 scripts/import_csv_nba.py
"""
import sys
import os
from pathlib import Path

# Adicionar raiz ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

import pandas as pd
import logging
from datetime import datetime
from data.repositories.db_manager import get_db_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_csv_game(row):
    """
    Converte linha CSV para formato do banco.
    
    Ajuste os nomes das colunas conforme o CSV que você baixar.
    """
    try:
        # Exemplo de mapeamento - AJUSTE conforme seu CSV
        # Kaggle NBA dataset geralmente tem: Date, Home, Away, HomeScore, AwayScore
        
        game_date = pd.to_datetime(row.get('Date') or row.get('GAME_DATE') or row.get('date')).strftime('%Y-%m-%d')
        
        home_team = str(row.get('Home') or row.get('HOME_TEAM') or row.get('home_team', '')).strip().upper()[:3]
        away_team = str(row.get('Away') or row.get('AWAY_TEAM') or row.get('away_team', '')).strip().upper()[:3]
        
        home_score = int(row.get('HomeScore') or row.get('HOME_PTS') or row.get('home_score', 0))
        away_score = int(row.get('AwayScore') or row.get('AWAY_PTS') or row.get('away_score', 0))
        
        # Game ID único
        game_id = f"{game_date}_{away_team}_{home_team}".replace("-", "")
        
        winner = 'HOME' if home_score > away_score else 'AWAY'
        
        season = row.get('Season') or row.get('SEASON') or '2024-25'
        
        game_data = {
            'id': game_id,
            'date': game_date,
            'season': str(season),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'winner': winner
        }
        
        # Stats básicos (se disponíveis no CSV)
        home_stats = {
            'PTS': home_score,
            'FGM': int(row.get('HomeFGM', 0)),
            'FGA': int(row.get('HomeFGA', 0)),
            'FG_PCT': float(row.get('HomeFG%', 0.0)),
            'FG3M': int(row.get('Home3PM', 0)),
            'FG3A': int(row.get('Home3PA', 0)),
            'FG3_PCT': float(row.get('Home3P%', 0.0)),
            'FTM': int(row.get('HomeFTM', 0)),
            'FTA': int(row.get('HomeFTA', 0)),
            'FT_PCT': float(row.get('HomeFT%', 0.0)),
            'OREB': int(row.get('HomeOREB', 0)),
            'DREB': int(row.get('HomeDREB', 0)),
            'REB': int(row.get('HomeTREB', 0)),
            'AST': int(row.get('HomeAST', 0)),
            'STL': int(row.get('HomeSTL', 0)),
            'BLK': int(row.get('HomeBLK', 0)),
            'TOV': int(row.get('HomeTOV', 0)),
            'PF': int(row.get('HomePF', 0)),
        }
        
        away_stats = {
            'PTS': away_score,
            'FGM': int(row.get('AwayFGM', 0)),
            'FGA': int(row.get('AwayFGA', 0)),
            'FG_PCT': float(row.get('AwayFG%', 0.0)),
            'FG3M': int(row.get('Away3PM', 0)),
            'FG3A': int(row.get('Away3PA', 0)),
            'FG3_PCT': float(row.get('Away3P%', 0.0)),
            'FTM': int(row.get('AwayFTM', 0)),
            'FTA': int(row.get('AwayFTA', 0)),
            'FT_PCT': float(row.get('AwayFT%', 0.0)),
            'OREB': int(row.get('AwayOREB', 0)),
            'DREB': int(row.get('AwayDREB', 0)),
            'REB': int(row.get('AwayTREB', 0)),
            'AST': int(row.get('AwayAST', 0)),
            'STL': int(row.get('AwaySTL', 0)),
            'BLK': int(row.get('AwayBLK', 0)),
            'TOV': int(row.get('AwayTOV', 0)),
            'PF': int(row.get('AwayPF', 0)),
        }
        
        return game_data, home_stats, away_stats
        
    except Exception as e:
        logger.error(f"Erro ao parsear linha: {e}")
        logger.debug(f"Dados da linha: {row.to_dict()}")
        return None, None, None

def import_csv(csv_path, limit=None):
    """
    Importa dados de CSV para o banco.
    
    Args:
        csv_path: Caminho para o arquivo CSV
        limit: Limite de jogos para importar (None = todos)
    """
    logger.info("="*80)
    logger.info("📥 IMPORTAÇÃO DE DADOS NBA VIA CSV")
    logger.info(f"   Arquivo: {csv_path}")
    logger.info("="*80)
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ Arquivo não encontrado: {csv_path}")
        logger.info("\n📋 Instruções:")
        logger.info("1. Acesse: https://www.kaggle.com/datasets/kevinpickelman/nba-data-2012-2024")
        logger.info("2. Baixe o arquivo CSV")
        logger.info(f"3. Salve em: {csv_path}")
        return 0
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 CSV carregado: {len(df)} linhas")
        logger.info(f"📋 Colunas: {list(df.columns)}")
        
        if limit:
            df = df.head(limit)
            logger.info(f"⚠️  Limitando importação a {limit} jogos")
        
        db = get_db_manager()
        
        inserted = 0
        skipped = 0
        errors = 0
        
        for idx, row in df.iterrows():
            game_data, home_stats, away_stats = parse_csv_game(row)
            
            if game_data:
                try:
                    db.insert_game_stats(game_data, home_stats, away_stats)
                    inserted += 1
                    
                    if inserted % 100 == 0:
                        logger.info(f"   Processados: {inserted} jogos...")
                        
                except Exception as e:
                    if 'UNIQUE constraint' in str(e) or 'duplicate key' in str(e):
                        skipped += 1
                    else:
                        logger.error(f"❌ Erro ao inserir jogo {game_data.get('id')}: {e}")
                        errors += 1
            else:
                errors += 1
        
        logger.info("="*80)
        logger.info("✅ IMPORTAÇÃO CONCLUÍDA")
        logger.info(f"   Inseridos: {inserted}")
        logger.info(f"   Duplicados: {skipped}")
        logger.info(f"   Erros: {errors}")
        logger.info("="*80)
        
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Erro fatal na importação: {e}")
        return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Importar dados NBA via CSV')
    parser.add_argument('--file', default='data/csv/nba_games.csv', help='Caminho do CSV')
    parser.add_argument('--limit', type=int, help='Limitar número de jogos')
    
    args = parser.parse_args()
    
    count = import_csv(args.file, limit=args.limit)
    sys.exit(0 if count > 0 else 1)
