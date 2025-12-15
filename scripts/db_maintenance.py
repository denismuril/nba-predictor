#!/usr/bin/env python3
"""
Database Maintenance Script - NBA Predictor v25.0
==================================================
Script de manutenção semanal para:
1. VACUUM ANALYZE nas tabelas principais
2. Arquivar odds antigas (>30 dias) para tabela fria
3. Limpar logs de execução antigos
4. Gerar relatório de saúde do banco

Uso:
    python scripts/db_maintenance.py              # Modo dry-run
    python scripts/db_maintenance.py --execute    # Executar de verdade
    python scripts/db_maintenance.py --archive    # Apenas arquivar
    python scripts/db_maintenance.py --vacuum     # Apenas VACUUM

Cron (Domingo 03:00):
    0 3 * * 0 cd /app && python scripts/db_maintenance.py --execute >> logs/maintenance.log 2>&1

Autor: NBA Predictor v25.0 - Go Live Edition
"""

import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('db_maintenance')

# Tabelas principais para VACUUM
MAIN_TABLES = [
    'games',
    'predictions', 
    'paper_bets',
    'bets',
    'game_stats',
    'feature_store'
]

# Tabelas para arquivar dados antigos
ARCHIVE_CONFIG = {
    'odds_cache': {
        'archive_table': 'odds_archive',
        'date_column': 'created_at',
        'retention_days': 30
    }
}


class DatabaseMaintenance:
    """Gerencia manutenção do banco de dados."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.stats = {
            'vacuum_tables': 0,
            'archived_rows': 0,
            'deleted_rows': 0,
            'errors': []
        }
    
    async def run_vacuum_analyze(self, conn):
        """Executa VACUUM ANALYZE nas tabelas principais."""
        logger.info("🧹 Iniciando VACUUM ANALYZE...")
        
        for table in MAIN_TABLES:
            try:
                if self.dry_run:
                    logger.info(f"  [DRY-RUN] VACUUM ANALYZE {table}")
                else:
                    # VACUUM precisa ser executado fora de transação
                    await conn.execute(f"VACUUM ANALYZE {table}")
                    logger.info(f"  ✅ VACUUM ANALYZE {table}")
                self.stats['vacuum_tables'] += 1
            except Exception as e:
                if 'does not exist' in str(e):
                    logger.debug(f"  ⏭️ Tabela {table} não existe, pulando")
                else:
                    logger.warning(f"  ⚠️ Erro em {table}: {e}")
                    self.stats['errors'].append(f"VACUUM {table}: {e}")
    
    async def archive_old_data(self, conn):
        """Arquiva dados antigos para tabelas frias."""
        logger.info("📦 Arquivando dados antigos...")
        
        for source_table, config in ARCHIVE_CONFIG.items():
            archive_table = config['archive_table']
            date_col = config['date_column']
            retention = config['retention_days']
            cutoff = datetime.now() - timedelta(days=retention)
            
            try:
                # Verificar se tabela fonte existe
                check_sql = f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{source_table}'
                )
                """
                
                if self.dry_run:
                    logger.info(f"  [DRY-RUN] Arquivar {source_table} -> {archive_table}")
                    logger.info(f"            Dados antes de {cutoff.strftime('%Y-%m-%d')}")
                    continue
                
                result = await conn.fetchval(check_sql)
                if not result:
                    logger.debug(f"  ⏭️ Tabela {source_table} não existe")
                    continue
                
                # Criar tabela de arquivo se não existir
                create_archive = f"""
                CREATE TABLE IF NOT EXISTS {archive_table} (LIKE {source_table} INCLUDING ALL)
                """
                await conn.execute(create_archive)
                
                # Copiar dados antigos para arquivo
                insert_sql = f"""
                INSERT INTO {archive_table}
                SELECT * FROM {source_table}
                WHERE {date_col} < $1
                ON CONFLICT DO NOTHING
                """
                result = await conn.execute(insert_sql, cutoff)
                archived = int(result.split()[-1]) if result else 0
                
                # Deletar originais
                delete_sql = f"""
                DELETE FROM {source_table}
                WHERE {date_col} < $1
                """
                result = await conn.execute(delete_sql, cutoff)
                deleted = int(result.split()[-1]) if result else 0
                
                self.stats['archived_rows'] += archived
                self.stats['deleted_rows'] += deleted
                logger.info(f"  ✅ {source_table}: {deleted} linhas arquivadas")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Erro arquivando {source_table}: {e}")
                self.stats['errors'].append(f"Archive {source_table}: {e}")
    
    async def cleanup_old_logs(self, conn):
        """Remove logs de execução antigos."""
        logger.info("🗑️ Limpando logs antigos...")
        
        # Limpar paper_bets muito antigos (>90 dias)
        cutoff_90d = datetime.now() - timedelta(days=90)
        
        try:
            if self.dry_run:
                logger.info(f"  [DRY-RUN] DELETE FROM paper_bets WHERE settled_at < {cutoff_90d}")
            else:
                result = await conn.execute("""
                    DELETE FROM paper_bets 
                    WHERE status != 'PENDING' 
                    AND settled_at < $1
                """, cutoff_90d)
                deleted = int(result.split()[-1]) if result else 0
                logger.info(f"  ✅ Paper bets antigos removidos: {deleted}")
                self.stats['deleted_rows'] += deleted
        except Exception as e:
            logger.debug(f"  ⏭️ Limpeza de paper_bets: {e}")
    
    async def get_database_stats(self, conn) -> dict:
        """Retorna estatísticas do banco."""
        stats = {}
        
        try:
            # Tamanho total do banco
            size_query = """
            SELECT pg_size_pretty(pg_database_size(current_database()))
            """
            stats['database_size'] = await conn.fetchval(size_query)
            
            # Contagem de registros por tabela
            for table in MAIN_TABLES:
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                    stats[f'{table}_count'] = count
                except Exception:
                    stats[f'{table}_count'] = 'N/A'
            
            # Verificar índices não utilizados
            unused_idx = await conn.fetch("""
                SELECT schemaname, relname, indexrelname, idx_scan
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0 AND indexrelname NOT LIKE '%pkey%'
                LIMIT 5
            """)
            stats['unused_indexes'] = len(unused_idx)
            
        except Exception as e:
            logger.warning(f"Erro ao obter stats: {e}")
        
        return stats
    
    async def run(self, vacuum: bool = True, archive: bool = True, cleanup: bool = True):
        """Executa manutenção completa."""
        import asyncpg
        
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            # Construir URL a partir de variáveis individuais
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            user = os.getenv('POSTGRES_USER', 'nba_admin')
            password = os.getenv('POSTGRES_PASSWORD', '')
            db = os.getenv('POSTGRES_DB', 'nba_predictor_db')
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
        logger.info("=" * 60)
        logger.info("🔧 NBA Predictor - Database Maintenance")
        logger.info(f"   Mode: {'DRY-RUN' if self.dry_run else 'EXECUTE'}")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            conn = await asyncpg.connect(database_url)
            
            # Stats antes
            stats_before = await self.get_database_stats(conn)
            logger.info(f"📊 Database size: {stats_before.get('database_size', 'N/A')}")
            
            # Executar tarefas
            if vacuum:
                await self.run_vacuum_analyze(conn)
            
            if archive:
                await self.archive_old_data(conn)
            
            if cleanup:
                await self.cleanup_old_logs(conn)
            
            # Stats depois
            stats_after = await self.get_database_stats(conn)
            
            await conn.close()
            
            # Relatório final
            self.print_report(stats_before, stats_after)
            
        except Exception as e:
            logger.error(f"❌ Erro fatal: {e}")
            self.stats['errors'].append(f"Connection: {e}")
    
    def print_report(self, before: dict, after: dict):
        """Imprime relatório final."""
        print("\n" + "=" * 60)
        print("📋 RELATÓRIO DE MANUTENÇÃO")
        print("=" * 60)
        print(f"Modo: {'DRY-RUN (nada executado)' if self.dry_run else 'EXECUTE'}")
        print(f"Tabelas VACUUM: {self.stats['vacuum_tables']}")
        print(f"Linhas arquivadas: {self.stats['archived_rows']}")
        print(f"Linhas deletadas: {self.stats['deleted_rows']}")
        print(f"Database size: {before.get('database_size', 'N/A')} -> {after.get('database_size', 'N/A')}")
        
        if self.stats['errors']:
            print("\n⚠️ Erros:")
            for err in self.stats['errors']:
                print(f"  - {err}")
        else:
            print("\n✅ Sem erros!")
        
        print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description='Database Maintenance - NBA Predictor')
    parser.add_argument('--execute', action='store_true',
                        help='Executar manutenção (default: dry-run)')
    parser.add_argument('--vacuum', action='store_true',
                        help='Apenas VACUUM ANALYZE')
    parser.add_argument('--archive', action='store_true',
                        help='Apenas arquivar dados antigos')
    parser.add_argument('--cleanup', action='store_true',
                        help='Apenas limpar logs antigos')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    # Se nenhuma flag específica, executar tudo
    run_all = not (args.vacuum or args.archive or args.cleanup)
    
    maintenance = DatabaseMaintenance(dry_run=dry_run)
    
    await maintenance.run(
        vacuum=args.vacuum or run_all,
        archive=args.archive or run_all,
        cleanup=args.cleanup or run_all
    )


if __name__ == "__main__":
    asyncio.run(main())
