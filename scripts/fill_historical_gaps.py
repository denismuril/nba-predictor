#!/usr/bin/env python3
"""
Preenchimento de Gaps com Dados Históricos REAIS

Identifica gaps no histórico (ex: 187 dias) e busca dados REAIS via SportsBlaze API
para preencher completamente o histórico.

Usage:
    python scripts/fill_historical_gaps.py [--min-gap-days 30] [--dry-run]
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
from scripts.sportsblaze_integration import SportsBlazeClient

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_gaps(df, min_gap_days=30):
    """
    Identifica gaps temporais nos dados.
    
    Args:
        df: DataFrame com dados históricos
        min_gap_days: Mínimo de dias para considerar um gap
    
    Returns:
        Lista de tuplas (start_date, end_date, gap_days)
    """
    logger.info("="*80)
    logger.info("🔍 IDENTIFICANDO GAPS NO HISTÓRICO")
    logger.info("="*80)
    
    df = df.sort_values('date').reset_index(drop=True)
    df['days_since_prev'] = df['date'].diff().dt.days
    
    # Gaps significativos
    gaps = df[df['days_since_prev'] > min_gap_days].copy()
    
    gap_list = []
    for idx, row in gaps.iterrows():
        if idx > 0:
            start_date = df.loc[idx-1, 'date']
            end_date = row['date']
            gap_days = row['days_since_prev']
            
            gap_list.append({
                'start': start_date,
                'end': end_date,
                'days': int(gap_days)
            })
            
            logger.info(f"\n⚠️  Gap detectado:")
            logger.info(f"   De: {start_date.date()}")
            logger.info(f"   Até: {end_date.date()}")
            logger.info(f"   Duração: {gap_days:.0f} dias")
    
    logger.info(f"\n📊 Total de gaps encontrados: {len(gap_list)}")
    
    return gap_list

def fetch_gap_data(gap, client):
    """
    Busca dados REAIS para preencher um gap.
    
    Args:
        gap: Dict com 'start', 'end', 'days'
        client: SportsBlazeClient
    
    Returns:
        Lista de jogos
    """
    logger.info(f"\n🔄 Buscando dados para gap de {gap['days']} dias...")
    logger.info(f"   Período: {gap['start'].date()} a {gap['end'].date()}")
    
    all_games = []
    current_date = gap['start'] + timedelta(days=1)
    end_date = gap['end']
    
    request_count = 0
    max_requests = 100  # Limitar para não exceder API quota
    
    while current_date < end_date and request_count < max_requests:
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            data = client.get_nba_boxscores(date_str)
            
            if data and 'games' in data:
                games = data['games']
                if games:
                    logger.info(f"   {date_str}: {len(games)} jogos encontrados")
                    all_games.extend(games)
            
            request_count += 1
            
            # Rate limiting (1 req/segundo)
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"   ❌ Erro em {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    logger.info(f"✅ Total coletado: {len(all_games)} jogos em {request_count} requests")
    
    return all_games

def parse_sportsblaze_game(game):
    """
    Converte jogo do SportsBlaze para formato do nosso DB.
    
    Args:
        game: Dict com dados do jogo
    
    Returns:
        Dict com dados formatados ou None se inválido
    """
    try:
        # Extrair informações básicas
        home_team = game['teams']['home']['name']
        away_team = game['teams']['away']['name']
        game_date = pd.to_datetime(game['date']).date()
        status = game.get('status', '')
        
        # Só processar jogos finalizados
        if status != 'Final':
            return None
        
        # Scores (se disponíveis)
        score = game.get('score', {})
        home_score = score.get('home', 0)
        away_score = score.get('away', 0)
        
        # Box scores detalhados (se disponíveis)
        stats = game.get('stats', {})
        
        return {
            'date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'source': 'sportsblaze',
            'raw_data': game  # Guardar dados brutos para referência
        }
        
    except Exception as e:
        logger.warning(f"⚠️  Erro ao parsear jogo: {e}")
        return None

def save_to_database(games, dry_run=False):
    """
    Salva jogos no banco de dados.
    
    Args:
        games: Lista de dicts com dados dos jogos
        dry_run: Se True, apenas simula sem salvar
    
    Returns:
        Número de jogos salvos
    """
    if dry_run:
        logger.info(f"\n🔍 DRY RUN: {len(games)} jogos seriam salvos")
        return 0
    
    logger.info(f"\n💾 Salvando {len(games)} jogos no banco de dados...")
    
    from data.repositories.db_manager import get_db_manager
    db = get_db_manager()
    
    saved_count = 0
    
    for game in games:
        try:
            # Verificar se já existe
            existing = db.conn.execute('''
                SELECT COUNT(*) FROM predictions 
                WHERE date = ? AND home_team = ? AND away_team = ?
            ''', (game['date'], game['home_team'], game['away_team'])).fetchone()[0]
            
            if existing > 0:
                logger.debug(f"   Jogo já existe: {game['home_team']} vs {game['away_team']} em {game['date']}")
                continue
            
            # Inserir novo jogo
            db.conn.execute('''
                INSERT INTO predictions 
                (date, home_team, away_team, home_score, away_score, prediction, correct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                game['date'],
                game['home_team'],
                game['away_team'],
                game['home_score'],
                game['away_score'],
                'HOME' if game['home_score'] > game['away_score'] else 'AWAY',  # Placeholder
                0  # Será recalculado depois
            ))
            
            saved_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ Erro ao salvar jogo: {e}")
    
    db.conn.commit()
    logger.info(f"✅ {saved_count} novos jogos salvos!")
    
    return saved_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preencher Gaps com Dados Históricos REAIS')
    parser.add_argument('--min-gap-days', type=int, default=30,
                       help='Mínimo de dias para considerar gap (default: 30)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simular sem salvar dados')
    parser.add_argument('--max-gaps', type=int, default=5,
                       help='Máximo de gaps para processar (default: 5)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🔄 PREENCHIMENTO DE GAPS COM DADOS REAIS")
    logger.info("="*80)
    logger.info(f"Configuração:")
    logger.info(f"   Gap mínimo: {args.min_gap_days} dias")
    logger.info(f"   Dry run: {args.dry_run}")
    logger.info(f"   Max gaps: {args.max_gaps}")
    
    # Carregar dados atuais
    from data.repositories.db_manager import get_db_manager
    db = get_db_manager()
    df = db.get_comprehensive_history()
    
    if df is None or df.empty:
        logger.error("❌ Nenhum dado histórico encontrado")
        return 1
    
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"\n📊 Dados atuais: {len(df)} jogos")
    logger.info(f"   Período: {df['date'].min().date()} a {df['date'].max().date()}")
    
    # Identificar gaps
    gaps = identify_gaps(df, min_gap_days=args.min_gap_days)
    
    if not gaps:
        logger.info("\n✅ Nenhum gap significativo encontrado!")
        return 0
    
    # Limitar número de gaps
    if len(gaps) > args.max_gaps:
        logger.warning(f"\n⚠️  Encontrados {len(gaps)} gaps, processando apenas {args.max_gaps}")
        gaps = gaps[:args.max_gaps]
    
    # Buscar dados via SportsBlaze
    client = SportsBlazeClient()
    
    all_new_games = []
    
    for i, gap in enumerate(gaps, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📥 Gap {i}/{len(gaps)}")
        logger.info(f"{'='*80}")
        
        games = fetch_gap_data(gap, client)
        
        # Parsear jogos
        parsed_games = []
        for game in games:
            parsed = parse_sportsblaze_game(game)
            if parsed:
                parsed_games.append(parsed)
        
        logger.info(f"   Jogos válidos parseados: {len(parsed_games)}")
        all_new_games.extend(parsed_games)
    
    # Salvar no banco
    if all_new_games:
        saved = save_to_database(all_new_games, dry_run=args.dry_run)
        
        if not args.dry_run and saved > 0:
            logger.info(f"\n🎉 SUCESSO! {saved} novos jogos adicionados ao histórico")
            logger.info(f"\n💡 Próximo passo:")
            logger.info(f"   python scripts/train_all_models.py  # Re-treinar com dados completos")
    else:
        logger.warning("\n⚠️  Nenhum jogo novo encontrado")
    
    logger.info("\n" + "="*80)
    logger.info("✅ PREENCHIMENTO CONCLUÍDO")
    logger.info("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
