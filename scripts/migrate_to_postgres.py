"""
Script de Migração: JSON/CSV → PostgreSQL
==========================================
Migra dados históricos de arquivos locais para PostgreSQL
sem perder o histórico de apostas.

Uso:
    python scripts/migrate_to_postgres.py
    python scripts/migrate_to_postgres.py --dry-run  # Simula sem gravar

Autor: NBA Predictor v22.0
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import argparse

# Adicionar raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Carregar variáveis de ambiente do .env
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgresMigrator:
    """
    Migrador de dados JSON/CSV para PostgreSQL.
    
    Etapas:
    1. Migrar bankroll.json → tabela de configuração
    2. Migrar cache/injuries.json → Redis
    3. Migrar prepared_games.csv → tabela games + game_stats
    4. Migrar predictions do SQLite → PostgreSQL
    5. Migrar bets do SQLite → PostgreSQL
    """
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'bankroll': 0,
            'injuries': 0,
            'games': 0,
            'predictions': 0,
            'bets': 0,
            'errors': []
        }
    
    async def run(self):
        """Executa migração completa"""
        logger.info("="*60)
        logger.info("🚀 INICIANDO MIGRAÇÃO JSON/CSV → POSTGRESQL")
        logger.info(f"   Modo: {'DRY-RUN (simulação)' if self.dry_run else 'PRODUÇÃO'}")
        logger.info("="*60)
        
        # Verificar pré-requisitos
        await self._check_prerequisites()
        
        # Executar migrações
        await self._migrate_bankroll()
        await self._migrate_injuries_to_redis()
        await self._migrate_prepared_games()
        await self._migrate_sqlite_predictions()
        await self._migrate_sqlite_bets()
        
        # Relatório final
        self._print_report()
    
    async def _check_prerequisites(self):
        """Verifica se PostgreSQL e Redis estão acessíveis"""
        logger.info("\n📋 Verificando pré-requisitos...")
        
        # Verificar variáveis de ambiente
        required_vars = ['DB_TYPE', 'DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASS']
        missing = [v for v in required_vars if not os.getenv(v)]
        
        if os.getenv('DB_TYPE', 'sqlite').lower() != 'postgres':
            logger.warning("⚠️ DB_TYPE não está definido como 'postgres'")
            logger.info("   Defina: export DB_TYPE=postgres")
        
        if missing:
            logger.warning(f"⚠️ Variáveis faltando: {missing}")
        
        # Testar conexão PostgreSQL
        try:
            from infrastructure.database import get_async_db
            db = await get_async_db()
            health = await db.health_check()
            
            if health['status'] == 'healthy':
                logger.info(f"✅ PostgreSQL conectado: {health.get('db_type', 'N/A')}")
            else:
                raise Exception(health.get('error', 'Erro desconhecido'))
                
        except Exception as e:
            logger.error(f"❌ Erro conectando PostgreSQL: {e}")
            if not self.dry_run:
                raise
        
        # Testar conexão Redis
        try:
            from infrastructure.redis_cache import get_redis
            redis = await get_redis()
            health = await redis.health_check()
            
            if health['status'] == 'healthy':
                logger.info(f"✅ Redis conectado: v{health.get('redis_version', 'N/A')}")
            else:
                logger.warning(f"⚠️ Redis não disponível: {health.get('message', '')}")
                
        except Exception as e:
            logger.warning(f"⚠️ Redis não disponível: {e}")
    
    async def _migrate_bankroll(self):
        """Migra bankroll.json"""
        logger.info("\n💰 Migrando bankroll.json...")
        
        bankroll_path = Path("data/bankroll.json")
        if not bankroll_path.exists():
            logger.info("   ⚪ Arquivo não encontrado, pulando")
            return
        
        try:
            with open(bankroll_path) as f:
                data = json.load(f)
            
            logger.info(f"   📄 Dados: {data}")
            
            if not self.dry_run:
                # Armazenar como configuração no Redis
                from infrastructure.redis_cache import get_redis
                redis = await get_redis()
                await redis.set("config:bankroll", data, category="default")
            
            self.stats['bankroll'] = 1
            logger.info("   ✅ Bankroll migrado")
            
        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            self.stats['errors'].append(f"bankroll: {e}")
    
    async def _migrate_injuries_to_redis(self):
        """Migra cache de lesões para Redis"""
        logger.info("\n🏥 Migrando cache de lesões para Redis...")
        
        injuries_path = Path("data/cache/injuries.json")
        if not injuries_path.exists():
            injuries_path = Path("data/injury_cache.json")
        
        if not injuries_path.exists():
            logger.info("   ⚪ Arquivo não encontrado, pulando")
            return
        
        try:
            with open(injuries_path) as f:
                data = json.load(f)
            
            count = 0
            if not self.dry_run:
                from infrastructure.redis_cache import get_redis
                redis = await get_redis()
                
                for team_id, injuries in data.items():
                    if isinstance(injuries, list):
                        await redis.set_injuries(team_id, injuries)
                        count += len(injuries)
            else:
                for team_id, injuries in data.items():
                    if isinstance(injuries, list):
                        count += len(injuries)
            
            self.stats['injuries'] = count
            logger.info(f"   ✅ {count} lesões de {len(data)} times migradas")
            
        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            self.stats['errors'].append(f"injuries: {e}")
    
    async def _migrate_prepared_games(self):
        """Migra prepared_games.csv para PostgreSQL"""
        logger.info("\n📊 Migrando prepared_games.csv...")
        
        csv_path = Path("data/prepared_games.csv")
        if not csv_path.exists():
            logger.info("   ⚪ Arquivo não encontrado, pulando")
            return
        
        try:
            import pandas as pd
            
            df = pd.read_csv(csv_path)
            logger.info(f"   📄 {len(df)} registros encontrados")
            
            # Converter para formato de jogos
            games = []
            for _, row in df.iterrows():
                game = {
                    'game_id': row.get('game_id', f"{row.get('date', '')}_{row.get('home_team', '')}_{row.get('away_team', '')}"),
                    'date': str(row.get('date', ''))[:10],
                    'home_team': row.get('home_team', row.get('home', '')),
                    'away_team': row.get('away_team', row.get('away', '')),
                    'home_score': int(row['home_score']) if pd.notna(row.get('home_score')) else None,
                    'away_score': int(row['away_score']) if pd.notna(row.get('away_score')) else None,
                    'season': row.get('season', '2024-25'),
                    'status': 'Final' if pd.notna(row.get('home_score')) else 'scheduled'
                }
                
                # Determinar vencedor
                if game['home_score'] and game['away_score']:
                    game['winner'] = 'HOME' if game['home_score'] > game['away_score'] else 'AWAY'
                
                games.append(game)
            
            if not self.dry_run and games:
                from infrastructure.database import get_async_db
                db = await get_async_db()
                
                # Inserir em lotes de 100
                batch_size = 100
                for i in range(0, len(games), batch_size):
                    batch = games[i:i+batch_size]
                    await db.bulk_insert_games(batch)
                    logger.info(f"   └─ Inserido lote {i//batch_size + 1}: {len(batch)} jogos")
            
            self.stats['games'] = len(games)
            logger.info(f"   ✅ {len(games)} jogos migrados")
            
        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            self.stats['errors'].append(f"prepared_games: {e}")
    
    async def _migrate_sqlite_predictions(self):
        """Migra predictions do SQLite para PostgreSQL"""
        logger.info("\n🎯 Migrando predictions do SQLite...")
        
        sqlite_paths = [
            Path("nba_predictor.db"),
            Path("data/nba_games.db"),
            Path("data/nba_history.db")
        ]
        
        sqlite_path = None
        for path in sqlite_paths:
            if path.exists():
                sqlite_path = path
                break
        
        if not sqlite_path:
            logger.info("   ⚪ Banco SQLite não encontrado, pulando")
            return
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Verificar se tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
            if not cursor.fetchone():
                logger.info("   ⚪ Tabela predictions não existe")
                conn.close()
                return
            
            # Buscar predictions
            cursor.execute("SELECT * FROM predictions")
            rows = cursor.fetchall()
            
            logger.info(f"   📄 {len(rows)} predictions encontradas")
            
            if not self.dry_run and rows:
                from infrastructure.database import get_async_db
                db = await get_async_db()
                
                predictions = []
                for row in rows:
                    pred = dict(row)
                    # Normalizar campos para novo formato
                    predictions.append({
                        'Data': pred.get('date', ''),
                        'Casa': pred.get('home_team', ''),
                        'Visitante': pred.get('away_team', ''),
                        'Prob Casa %': pred.get('prob_home', 0),
                        'Prob Visitante %': pred.get('prob_away', 0),
                        'Odd Casa': pred.get('odd_home', 0),
                        'Odd Visitante': pred.get('odd_away', 0),
                        'Previsão': pred.get('prediction', 'N/A'),
                        'Confiança': pred.get('confidence', 'N/A')
                    })
                
                await db.save_predictions(predictions)
            
            self.stats['predictions'] = len(rows)
            conn.close()
            logger.info(f"   ✅ {len(rows)} predictions migradas")
            
        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            self.stats['errors'].append(f"predictions: {e}")
    
    async def _migrate_sqlite_bets(self):
        """Migra bets do SQLite para PostgreSQL"""
        logger.info("\n💵 Migrando bets do SQLite...")
        
        sqlite_paths = [
            Path("nba_predictor.db"),
            Path("data/betting_tracker.db"),
            Path("data/backtest_bets.db")
        ]
        
        sqlite_path = None
        for path in sqlite_paths:
            if path.exists():
                sqlite_path = path
                break
        
        if not sqlite_path:
            logger.info("   ⚪ Banco de apostas não encontrado, pulando")
            return
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Verificar se tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bets'")
            if not cursor.fetchone():
                logger.info("   ⚪ Tabela bets não existe")
                conn.close()
                return
            
            # Buscar bets
            cursor.execute("SELECT * FROM bets")
            rows = cursor.fetchall()
            
            logger.info(f"   📄 {len(rows)} bets encontradas")
            
            if not self.dry_run and rows:
                from infrastructure.database import get_async_db
                db = await get_async_db()
                
                for row in rows:
                    bet = dict(row)
                    await db.save_bet(bet)
            
            self.stats['bets'] = len(rows)
            conn.close()
            logger.info(f"   ✅ {len(rows)} bets migradas")
            
        except Exception as e:
            logger.error(f"   ❌ Erro: {e}")
            self.stats['errors'].append(f"bets: {e}")
    
    def _print_report(self):
        """Imprime relatório final"""
        logger.info("\n" + "="*60)
        logger.info("📊 RELATÓRIO DE MIGRAÇÃO")
        logger.info("="*60)
        logger.info(f"   💰 Bankroll:    {self.stats['bankroll']} configuração(ões)")
        logger.info(f"   🏥 Lesões:      {self.stats['injuries']} registros → Redis")
        logger.info(f"   📅 Jogos:       {self.stats['games']} registros → PostgreSQL")
        logger.info(f"   🎯 Predictions: {self.stats['predictions']} registros → PostgreSQL")
        logger.info(f"   💵 Bets:        {self.stats['bets']} registros → PostgreSQL")
        
        if self.stats['errors']:
            logger.info("\n⚠️ ERROS:")
            for error in self.stats['errors']:
                logger.info(f"   - {error}")
        else:
            logger.info("\n✅ Migração concluída sem erros!")
        
        if self.dry_run:
            logger.info("\n⚠️ MODO DRY-RUN: Nenhum dado foi gravado.")
            logger.info("   Para migrar de verdade, rode sem --dry-run")


async def main():
    parser = argparse.ArgumentParser(description='Migração JSON/CSV → PostgreSQL')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Simula migração sem gravar dados')
    args = parser.parse_args()
    
    migrator = PostgresMigrator(dry_run=args.dry_run)
    await migrator.run()


if __name__ == "__main__":
    asyncio.run(main())
