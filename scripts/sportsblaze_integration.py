#!/usr/bin/env python3
"""
SportsBlaze API Integration

Integra com SportsBlaze API para enriquecer dados com métricas adicionais.

API Key: sbfxqpy6v6fjljvobf61a5o
Docs: https://docs.sportsblaze.com/

Funcionalidades:
1. Buscar boxscores detalhados
2. Enriquecer dados históricos
3. Fetch dados em tempo real

Usage:
    python scripts/sportsblaze_integration.py --fetch-recent
"""
import sys
import os
sys.path.insert(0, '/home/denis/nba-predictor')

import logging
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import time

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
SPORTSBLAZE_API_KEY = "sbfxqpy6v6fjljvobf61a5o"
BASE_URL = "https://api.sportsblaze.com"

class SportsBlazeClient:
    """Cliente para SportsBlaze API."""
    
    def __init__(self, api_key=SPORTSBLAZE_API_KEY):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.session = requests.Session()
        
    def _make_request(self, endpoint, params=None):
        """Faz request à API com rate limiting."""
        if params is None:
            params = {}
        
        params['key'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro na API: {e}")
            return None
    
    def get_nba_boxscores(self, date):
        """
        Busca boxscores de NBA para uma data específica.
        
        Args:
            date: datetime ou string 'YYYY-MM-DD'
        
        Returns:
            Dict com dados dos jogos
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date
        
        endpoint = f"/nba/v1/boxscores/daily/{date_str}.json"
        logger.info(f"📥 Buscando boxscores para {date_str}...")
        
        data = self._make_request(endpoint)
        
        if data:
            logger.info(f"✅ {len(data.get('games', []))} jogos encontrados")
        
        return data
    
    def get_season_schedule(self, season='2024-25'):
        """
        Busca calendário da temporada.
        
        Args:
            season: String da temporada (ex: '2024-25')
        
        Returns:
            Dict com dados do calendário
        """
        # Nota: Endpoint específico depende da documentação
        endpoint = f"/nba/v1/schedule/{season}.json"
        logger.info(f"📅 Buscando calendário da temporada {season}...")
        
        return self._make_request(endpoint)
    
    def get_team_stats(self, team_id, season='2024-25'):
        """
        Busca estatísticas de um time.
        
        Args:
            team_id: ID do time
            season: String da temporada
        
        Returns:
            Dict com estatísticas do time
        """
        endpoint = f"/nba/v1/teams/{team_id}/stats.json"
        params = {'season': season}
        logger.info(f"📊 Buscando stats do time {team_id}...")
        
        return self._make_request(endpoint, params=params)
    
    def fetch_recent_games(self, days=7):
        """
        Busca jogos dos últimos N dias.
        
        Args:
            days: Número de dias para buscar
        
        Returns:
            Lista de jogos
        """
        logger.info(f"🔄 Buscando jogos dos últimos {days} dias...")
        
        all_games = []
        today = datetime.now()
        
        for i in range(days):
            date = today - timedelta(days=i)
            data = self.get_nba_boxscores(date)
            
            if data and 'games' in data:
                all_games.extend(data['games'])
            
            # Rate limiting (1 req/segundo)
            time.sleep(1)
        
        logger.info(f"✅ Total: {len(all_games)} jogos coletados")
        return all_games

def enrich_historical_data():
    """Enriquece dados históricos com informações do SportsBlaze."""
    logger.info("="*80)
    logger.info("🔄 ENRIQUECIMENTO DE DADOS HISTÓRICOS")
    logger.info("="*80)
    
    client = SportsBlazeClient()
    
    # Carregar dados atuais
    from data.repositories.db_manager import get_db_manager
    db = get_db_manager()
    df = db.get_comprehensive_history()
    
    if df is None or df.empty:
        logger.error("❌ Nenhum dado histórico encontrado")
        return
    
    logger.info(f"📊 Dados atuais: {len(df)} jogos")
    
    # Identificar datas para buscar dados adicionais
    df['date'] = pd.to_datetime(df['date'])
    recent_games = df[df['date'] >= (datetime.now() - timedelta(days=30))]
    
    logger.info(f"🎯 Jogos recentes para enriquecer: {len(recent_games)}")
    
    # Buscar dados adicionais (exemplo)
    enriched_count = 0
    
    for date in recent_games['date'].dt.date.unique()[:5]:  # Limitar a 5 dias para teste
        boxscores = client.get_nba_boxscores(date.strftime('%Y-%m-%d'))
        
        if boxscores:
            enriched_count += len(boxscores.get('games', []))
        
        time.sleep(1)  # Rate limiting
    
    logger.info(f"✅ Enriquecidos: {enriched_count} jogos")
    
    return enriched_count

def test_api_connection():
    """Testa conexão com a API."""
    logger.info("="*80)
    logger.info("🧪 TESTE DE CONEXÃO - SPORTSBLAZE API")
    logger.info("="*80)
    
    client = SportsBlazeClient()
    
    # Testar com data recente
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"\n📡 Testando busca de boxscores para {yesterday}...")
    
    data = client.get_nba_boxscores(yesterday)
    
    if data:
        logger.info("✅ Conexão bem-sucedida!")
        logger.info(f"   Jogos encontrados: {len(data.get('games', []))}")
        
        # Mostrar exemplo de jogo
        if data.get('games'):
            game = data['games'][0]
            logger.info(f"\n📋 Exemplo de jogo:")
            logger.info(f"   {json.dumps(game, indent=2)[:500]}...")
    else:
        logger.error("❌ Falha na conexão")
    
    return data is not None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SportsBlaze API Integration')
    parser.add_argument('--test', action='store_true', help='Testar conexão')
    parser.add_argument('--fetch-recent', action='store_true', help='Buscar jogos recentes')
    parser.add_argument('--enrich', action='store_true', help='Enriquecer dados históricos')
    parser.add_argument('--days', type=int, default=7, help='Dias para buscar (default: 7)')
    
    args = parser.parse_args()
    
    client = SportsBlazeClient()
    
    if args.test:
        success = test_api_connection()
        return 0 if success else 1
    
    elif args.fetch_recent:
        games = client.fetch_recent_games(days=args.days)
        
        # Salvar em JSON
        output_file = Path('data/sportsblaze/recent_games.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(games, f, indent=2)
        
        logger.info(f"💾 Jogos salvos em: {output_file}")
        return 0
    
    elif args.enrich:
        count = enrich_historical_data()
        return 0 if count > 0 else 1
    
    else:
        logger.info("Use --test, --fetch-recent ou --enrich")
        return 1

if __name__ == "__main__":
    sys.exit(main())
