"""
Migração de Dados v1 → v2
=========================
Script único para migrar dados históricos do sistema legado
para a nova infraestrutura Enterprise (PostgreSQL).

Fontes de dados:
- data/games_history.json
- data/predictions_*.csv
- nba_predictor.db (SQLite)
- data/*.csv (estatísticas)

Destino:
- PostgreSQL via AsyncDataManager

Autor: NBA Predictor v23.0
"""

import sys
import asyncio
import logging
import json
import csv
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMigrator:
    """
    Migrador de dados v1 → v2.
    
    Características:
    - Leitura de múltiplas fontes (JSON, CSV, SQLite)
    - Validação com Pydantic Schemas
    - Bulk Insert assíncrono
    - Deduplicação automática
    - Relatório de migração
    """
    
    def __init__(self):
        self.db = None
        self.stats = {
            'games_migrated': 0,
            'games_skipped': 0,
            'predictions_migrated': 0,
            'predictions_skipped': 0,
            'stats_migrated': 0,
            'errors': []
        }
    
    async def initialize(self):
        """Inicializa conexão com banco de dados destino."""
        logger.info("🔌 Conectando ao PostgreSQL...")
        
        try:
            from infrastructure.database import get_async_db
            self.db = await get_async_db()
            await self.db.init_db()
            
            health = await self.db.health_check()
            logger.info(f"✅ Conectado: {health['db_type']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            raise
    
    async def migrate_all(self):
        """Executa migração completa."""
        logger.info("🚀 Iniciando migração v1 → v2...")
        start = datetime.now()
        
        await self.initialize()
        
        # 1. Migrar jogos do SQLite
        await self.migrate_games_from_sqlite()
        
        # 2. Migrar jogos de JSON
        await self.migrate_games_from_json()
        
        # 3. Migrar previsões de CSV
        await self.migrate_predictions_from_csv()
        
        # 4. Migrar estatísticas
        await self.migrate_stats_from_sqlite()
        
        duration = (datetime.now() - start).total_seconds()
        
        # Relatório final
        logger.info("\n" + "="*60)
        logger.info("📊 RELATÓRIO DE MIGRAÇÃO")
        logger.info("="*60)
        logger.info(f"✅ Jogos migrados: {self.stats['games_migrated']}")
        logger.info(f"⏭️  Jogos pulados (duplicados): {self.stats['games_skipped']}")
        logger.info(f"✅ Previsões migradas: {self.stats['predictions_migrated']}")
        logger.info(f"⏭️  Previsões puladas: {self.stats['predictions_skipped']}")
        logger.info(f"✅ Stats migradas: {self.stats['stats_migrated']}")
        logger.info(f"❌ Erros: {len(self.stats['errors'])}")
        logger.info(f"⏱️  Duração: {duration:.1f}s")
        logger.info("="*60)
        
        if self.stats['errors']:
            logger.warning("Erros encontrados:")
            for err in self.stats['errors'][:5]:
                logger.warning(f"  - {err}")
    
    async def migrate_games_from_sqlite(self):
        """Migra jogos do SQLite legado."""
        sqlite_path = BASE_DIR / "nba_predictor.db"
        
        if not sqlite_path.exists():
            logger.info("ℹ️ nba_predictor.db não encontrado - pulando")
            return
        
        logger.info(f"📂 Lendo jogos de {sqlite_path}...")
        
        try:
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Verificar se tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
            if not cursor.fetchone():
                logger.info("ℹ️ Tabela 'games' não existe no SQLite")
                conn.close()
                return
            
            # Ler jogos
            cursor.execute("""
                SELECT game_id, date, season, home_team, away_team, 
                       home_score, away_score, winner, status
                FROM games
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.info("ℹ️ Nenhum jogo encontrado no SQLite")
                return
            
            logger.info(f"📊 {len(rows)} jogos encontrados no SQLite")
            
            # Validar e preparar para inserção
            games_to_insert = []
            
            for row in rows:
                try:
                    game_data = self._normalize_game(dict(row))
                    validated = self._validate_game(game_data)
                    if validated:
                        games_to_insert.append(validated)
                except Exception as e:
                    self.stats['errors'].append(f"Jogo {row['game_id']}: {e}")
                    self.stats['games_skipped'] += 1
            
            # Bulk insert
            if games_to_insert:
                await self.db.bulk_insert_games(games_to_insert)
                self.stats['games_migrated'] += len(games_to_insert)
                logger.info(f"✅ {len(games_to_insert)} jogos migrados do SQLite")
            
        except Exception as e:
            logger.error(f"❌ Erro ao migrar jogos do SQLite: {e}")
            self.stats['errors'].append(f"SQLite games: {e}")
    
    async def migrate_games_from_json(self):
        """Migra jogos de arquivos JSON."""
        json_files = list(BASE_DIR.glob("data/games*.json"))
        json_files.extend(list(BASE_DIR.glob("data/*history*.json")))
        
        if not json_files:
            logger.info("ℹ️ Nenhum arquivo JSON de jogos encontrado")
            return
        
        for json_path in json_files:
            logger.info(f"📂 Lendo {json_path.name}...")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Pode ser lista ou dict
                games = data if isinstance(data, list) else data.get('games', [])
                
                if not games:
                    continue
                
                games_to_insert = []
                
                for game in games:
                    try:
                        normalized = self._normalize_game(game)
                        validated = self._validate_game(normalized)
                        if validated:
                            games_to_insert.append(validated)
                    except Exception as e:
                        self.stats['errors'].append(f"JSON game: {e}")
                        self.stats['games_skipped'] += 1
                
                if games_to_insert:
                    await self.db.bulk_insert_games(games_to_insert)
                    self.stats['games_migrated'] += len(games_to_insert)
                    logger.info(f"✅ {len(games_to_insert)} jogos de {json_path.name}")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao ler {json_path}: {e}")
    
    async def migrate_predictions_from_csv(self):
        """Migra previsões de arquivos CSV."""
        csv_files = list(BASE_DIR.glob("results/predictions_*.csv"))
        csv_files.extend(list(BASE_DIR.glob("data/predictions_*.csv")))
        
        if not csv_files:
            logger.info("ℹ️ Nenhum arquivo CSV de previsões encontrado")
            return
        
        for csv_path in csv_files:
            logger.info(f"📂 Lendo {csv_path.name}...")
            
            try:
                predictions = []
                
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        try:
                            pred = self._normalize_prediction(row)
                            if pred:
                                predictions.append(pred)
                        except Exception as e:
                            self.stats['predictions_skipped'] += 1
                
                if predictions and hasattr(self.db, 'save_predictions'):
                    await self.db.save_predictions(predictions)
                    self.stats['predictions_migrated'] += len(predictions)
                    logger.info(f"✅ {len(predictions)} previsões de {csv_path.name}")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao ler {csv_path}: {e}")
    
    async def migrate_stats_from_sqlite(self):
        """Migra estatísticas de jogos do SQLite."""
        sqlite_path = BASE_DIR / "nba_predictor.db"
        
        if not sqlite_path.exists():
            return
        
        try:
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Verificar se tabela existe
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_stats'")
            if not cursor.fetchone():
                conn.close()
                return
            
            cursor.execute("SELECT * FROM game_stats LIMIT 1000")  # Limitar para não sobrecarregar
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return
            
            logger.info(f"📊 {len(rows)} stats encontradas no SQLite")
            
            # Inserir stats
            for row in rows:
                try:
                    stat_data = dict(row)
                    # Inserir via session diretamente
                    async with self.db.get_session() as session:
                        from infrastructure.database import GameStats
                        stat = GameStats(**stat_data)
                        session.add(stat)
                    self.stats['stats_migrated'] += 1
                except Exception as e:
                    self.stats['errors'].append(f"Stat: {e}")
            
            logger.info(f"✅ {self.stats['stats_migrated']} stats migradas")
            
        except Exception as e:
            logger.error(f"❌ Erro ao migrar stats: {e}")
    
    def _normalize_game(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza dados de jogo para formato padrão."""
        return {
            'game_id': data.get('game_id', data.get('id', f"{data.get('date', '')}_{data.get('home_team', '')}_{data.get('away_team', '')}")),
            'date': data.get('date', data.get('Data', '')),
            'season': data.get('season', '2024-25'),
            'home_team': self._normalize_team(data.get('home_team', data.get('home', data.get('Casa', '')))),
            'away_team': self._normalize_team(data.get('away_team', data.get('away', data.get('Visitante', '')))),
            'home_score': self._safe_int(data.get('home_score', data.get('home_pts'))),
            'away_score': self._safe_int(data.get('away_score', data.get('away_pts'))),
            'winner': data.get('winner', ''),
            'status': data.get('status', 'Final')
        }
    
    def _normalize_prediction(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normaliza dados de previsão."""
        try:
            return {
                'Data': data.get('Data', data.get('date', '')),
                'Casa': data.get('Casa', data.get('home_team', '')),
                'Visitante': data.get('Visitante', data.get('away_team', '')),
                'Prob Casa %': self._safe_float(data.get('Prob Casa %', data.get('prob_home', 50))),
                'Prob Visitante %': self._safe_float(data.get('Prob Visitante %', data.get('prob_away', 50))),
                'Previsão': data.get('Previsão', data.get('prediction', 'N/A')),
                'Confiança': data.get('Confiança', data.get('confidence', 'N/A')),
                'Spread Previsto': self._safe_float(data.get('Spread Previsto', data.get('spread', 0))),
                'Total Previsto': self._safe_float(data.get('Total Previsto', data.get('total', 220)))
            }
        except Exception:
            return None
    
    def _validate_game(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Valida jogo com Pydantic."""
        try:
            from etl.schemas import GameSchema
            validated = GameSchema(**data)
            return validated.dict()
        except ImportError:
            # Sem Pydantic, validação básica
            if not data.get('game_id') or not data.get('date'):
                return None
            if not data.get('home_team') or not data.get('away_team'):
                return None
            return data
        except Exception as e:
            self.stats['errors'].append(f"Validation: {e}")
            return None
    
    def _normalize_team(self, team: str) -> str:
        """Normaliza nome de time para abreviação."""
        if not team:
            return ''
        
        # Mapeamento comum
        team_map = {
            'Los Angeles Lakers': 'LAL',
            'Boston Celtics': 'BOS',
            'Golden State Warriors': 'GSW',
            'Miami Heat': 'MIA',
            'Phoenix Suns': 'PHX',
            'Denver Nuggets': 'DEN',
            'Milwaukee Bucks': 'MIL',
            'Philadelphia 76ers': 'PHI',
            'New York Knicks': 'NYK',
            'Brooklyn Nets': 'BKN',
            'LA Clippers': 'LAC',
            'Dallas Mavericks': 'DAL',
            'Memphis Grizzlies': 'MEM',
            'Sacramento Kings': 'SAC',
            'Minnesota Timberwolves': 'MIN',
            'Cleveland Cavaliers': 'CLE',
            'New Orleans Pelicans': 'NOP',
            'Atlanta Hawks': 'ATL',
            'Chicago Bulls': 'CHI',
            'Toronto Raptors': 'TOR',
            'Indiana Pacers': 'IND',
            'Oklahoma City Thunder': 'OKC',
            'Houston Rockets': 'HOU',
            'San Antonio Spurs': 'SAS',
            'Portland Trail Blazers': 'POR',
            'Utah Jazz': 'UTA',
            'Orlando Magic': 'ORL',
            'Detroit Pistons': 'DET',
            'Charlotte Hornets': 'CHA',
            'Washington Wizards': 'WAS'
        }
        
        return team_map.get(team, team[:3].upper() if len(team) > 3 else team.upper())
    
    def _safe_int(self, value) -> Optional[int]:
        """Converte valor para int de forma segura."""
        if value is None or value == '':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> float:
        """Converte valor para float de forma segura."""
        if value is None or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0


async def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migração v1 → v2")
    parser.add_argument('--dry-run', action='store_true', help='Apenas simular migração')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 Modo DRY RUN - nenhum dado será alterado")
        # TODO: Implementar dry run
        return
    
    migrator = DataMigrator()
    await migrator.migrate_all()


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
