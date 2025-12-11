"""
Script para otimizar banco de dados com índices estratégicos.

REFATORADO: Suporta SQLite e PostgreSQL via DatabaseManager.

Impacto esperado:
- Queries em predictions: ~5s → ~50ms (50x faster)
- Historical data load: ~30s → ~3s (10x faster)

Usage:
    python scripts/optimize_database.py --add-indexes
    python scripts/optimize_database.py --analyze
    python scripts/optimize_database.py --all
"""

import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time
from data.repositories.db_manager import get_db_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Otimizador de banco de dados (SQLite e PostgreSQL)."""
    
    def __init__(self):
        self.db = get_db_manager()
        self.db.init_db()
        logger.info(f"📂 Conectado ao banco: {self.db.db_type.upper()}")
    
    def add_indexes(self):
        """Adiciona índices estratégicos para queries comuns."""
        logger.info("="*80)
        logger.info("🚀 OTIMIZANDO BANCO DE DADOS")
        logger.info("="*80)
        
        # Índices para tabela predictions (além dos já criados no init_db)
        indexes = [
            {
                'name': 'idx_predictions_home_team',
                'table': 'predictions',
                'columns': 'home_team',
                'rationale': 'Filtrar jogos de um time específico em casa'
            },
            {
                'name': 'idx_predictions_away_team',
                'table': 'predictions',
                'columns': 'away_team',
                'rationale': 'Filtrar jogos de um time específico fora'
            },
            {
                'name': 'idx_predictions_composite',
                'table': 'predictions',
                'columns': 'date, home_team, away_team',
                'rationale': 'Query completa: jogo específico em data específica'
            }
        ]
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        created_count = 0
        skipped_count = 0
        
        try:
            for idx in indexes:
                try:
                    # Verificar se índice já existe
                    if self.db.db_type == 'sqlite':
                        cursor.execute("""
                            SELECT name FROM sqlite_master 
                            WHERE type='index' AND name=?
                        """, (idx['name'],))
                    else:  # postgres
                        cursor.execute("""
                            SELECT indexname FROM pg_indexes 
                            WHERE indexname = %s
                        """, (idx['name'],))
                    
                    if cursor.fetchone():
                        logger.info(f"⏭️  {idx['name']} já existe - skipping")
                        skipped_count += 1
                        continue
                    
                    # Criar índice
                    logger.info(f"📊 Criando {idx['name']}...")
                    logger.info(f"   Tabela: {idx['table']}")
                    logger.info(f"   Colunas: {idx['columns']}")
                    logger.info(f"   Motivo: {idx['rationale']}")
                    
                    start_time = time.time()
                    
                    query = f"""
                        CREATE INDEX IF NOT EXISTS {idx['name']}
                        ON {idx['table']}({idx['columns']})
                    """
                    cursor.execute(self.db._prepare_query(query))
                    
                    elapsed = time.time() - start_time
                    
                    logger.info(f"   ✅ Criado em {elapsed:.2f}s")
                    created_count += 1
                    
                except Exception as e:
                    logger.error(f"   ❌ Erro criando {idx['name']}: {e}")
            
            # Commit
            conn.commit()
            
            logger.info(f"\n📊 Resumo:")
            logger.info(f"   Índices criados: {created_count}")
            logger.info(f"   Já existiam: {skipped_count}")
            logger.info(f"   Total: {created_count + skipped_count}")
            
        finally:
            self.db.return_connection(conn)
    
    def analyze_tables(self):
        """Analisa tabelas e atualiza estatísticas."""
        logger.info("\n🔍 Analisando tabelas...")
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.db.db_type == 'sqlite':
                # SQLite: ANALYZE atualiza estatísticas internas
                cursor.execute("ANALYZE predictions")
                cursor.execute("ANALYZE games")
                cursor.execute("ANALYZE game_stats")
                cursor.execute("ANALYZE bets")
            else:  # postgres
                # PostgreSQL: ANALYZE também atualiza estatísticas
                cursor.execute("ANALYZE predictions")
                cursor.execute("ANALYZE games")
                cursor.execute("ANALYZE game_stats")
                cursor.execute("ANALYZE bets")
            
            conn.commit()
            logger.info("✅ Estatísticas atualizadas")
            
        except Exception as e:
            logger.error(f"❌ Erro no ANALYZE: {e}")
        finally:
            self.db.return_connection(conn)
    
    def benchmark_queries(self):
        """Testa performance de queries comuns."""
        logger.info("\n⚡ BENCHMARK DE QUERIES")
        logger.info("="*80)
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            # Query 1: Buscar jogos de hoje
            logger.info("\n1️⃣  Query: Jogos de hoje")
            if self.db.db_type == 'sqlite':
                query1 = "SELECT * FROM predictions WHERE date = date('now')"
            else:
                query1 = "SELECT * FROM predictions WHERE date = CURRENT_DATE"
            
            start = time.time()
            cursor.execute(query1)
            results = cursor.fetchall()
            elapsed1 = time.time() - start
            
            logger.info(f"   Tempo: {elapsed1*1000:.2f}ms")
            logger.info(f"   Resultados: {len(results)} jogos")
            
            # Query 2: Jogos do Lakers
            logger.info("\n2️⃣  Query: Jogos do Lakers")
            query2 = self.db._prepare_query(
                "SELECT * FROM predictions WHERE home_team = ? OR away_team = ?"
            )
            
            start = time.time()
            cursor.execute(query2, ('LAL', 'LAL'))
            results = cursor.fetchall()
            elapsed2 = time.time() - start
            
            logger.info(f"   Tempo: {elapsed2*1000:.2f}ms")
            logger.info(f"   Resultados: {len(results)} jogos")
            
            # Query 3: Últimos 30 dias
            logger.info("\n3️⃣  Query: Últimos 30 dias")
            if self.db.db_type == 'sqlite':
                query3 = """
                    SELECT * FROM predictions 
                    WHERE date >= date('now', '-30 days')
                    ORDER BY date DESC
                """
            else:
                query3 = """
                    SELECT * FROM predictions 
                    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                    ORDER BY date DESC
                """
            
            start = time.time()
            cursor.execute(query3)
            results = cursor.fetchall()
            elapsed3 = time.time() - start
            
            logger.info(f"   Tempo: {elapsed3*1000:.2f}ms")
            logger.info(f"   Resultados: {len(results)} jogos")
            
            # Resumo
            avg_time = (elapsed1 + elapsed2 + elapsed3) / 3
            logger.info(f"\n📊 RESUMO:")
            logger.info(f"   Tempo médio: {avg_time*1000:.2f}ms")
            
            if avg_time < 0.1:  # < 100ms
                logger.info(f"   Status: ✅ EXCELENTE (< 100ms)")
            elif avg_time < 0.5:  # < 500ms
                logger.info(f"   Status: ✅ BOM (< 500ms)")
            else:
                logger.info(f"   Status: ⚠️  LENTO (> 500ms) - considere mais otimizações")
                
        finally:
            self.db.return_connection(conn)
    
    def show_index_info(self):
        """Mostra informações sobre índices existentes."""
        logger.info("\n📋 ÍNDICES EXISTENTES")
        logger.info("="*80)
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            if self.db.db_type == 'sqlite':
                # SQLite: buscar de sqlite_master
                cursor.execute("""
                    SELECT name, tbl_name, sql 
                    FROM sqlite_master 
                    WHERE type='index' AND name LIKE 'idx_%'
                    ORDER BY name
                """)
            else:  # postgres
                # PostgreSQL: buscar de pg_indexes
                cursor.execute("""
                    SELECT indexname, tablename, indexdef 
                    FROM pg_indexes 
                    WHERE indexname LIKE 'idx_%'
                    ORDER BY indexname
                """)
            
            indexes = cursor.fetchall()
            
            if not indexes:
                logger.warning("⚠️  Nenhum índice customizado encontrado")
                return
            
            for idx_name, table, sql in indexes:
                logger.info(f"\n📌 {idx_name}")
                logger.info(f"   Tabela: {table}")
                if sql:
                    logger.info(f"   SQL: {sql[:100]}...")
            
            logger.info(f"\n   Total: {len(indexes)} índices customizados")
            
        finally:
            self.db.return_connection(conn)
    
    def vacuum_database(self):
        """Executa VACUUM para otimizar espaço em disco (SQLite only)."""
        if self.db.db_type != 'sqlite':
            logger.warning("⚠️  VACUUM é específico para SQLite. PostgreSQL usa VACUUM FULL.")
            logger.info("   Para PostgreSQL, execute: VACUUM FULL manualmente se necessário.")
            return
        
        logger.info("\n🗜️  Executando VACUUM...")
        
        # Verificar tamanho antes
        size_before = self.db.db_path.stat().st_size
        logger.info(f"   Tamanho antes: {size_before / 1024 / 1024:.2f} MB")
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("VACUUM")
            conn.commit()
            
            # Verificar tamanho depois
            size_after = self.db.db_path.stat().st_size
            logger.info(f"   Tamanho depois: {size_after / 1024 / 1024:.2f} MB")
            
            saved = size_before - size_after
            if saved > 0:
                logger.info(f"   Espaço economizado: {saved / 1024 / 1024:.2f} MB")
            
            logger.info("   ✅ VACUUM completo")
            
        except Exception as e:
            logger.error(f"   ❌ Erro no VACUUM: {e}")
        finally:
            self.db.return_connection(conn)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Otimizador de Banco de Dados')
    parser.add_argument('--add-indexes', action='store_true',
                       help='Adiciona índices otimizados')
    parser.add_argument('--analyze', action='store_true',
                       help='Analisa tabelas e atualiza estatísticas')
    parser.add_argument('--benchmark', action='store_true',
                       help='Executa benchmark de queries')
    parser.add_argument('--show-indexes', action='store_true',
                       help='Mostra índices existentes')
    parser.add_argument('--vacuum', action='store_true',
                       help='Executa VACUUM (otimiza espaço - SQLite only)')
    parser.add_argument('--all', action='store_true',
                       help='Executa todas as otimizações')
    
    args = parser.parse_args()
    
    try:
        optimizer = DatabaseOptimizer()
        
        if args.all:
            optimizer.add_indexes()
            optimizer.analyze_tables()
            optimizer.benchmark_queries()
            optimizer.show_index_info()
            optimizer.vacuum_database()
        else:
            if args.add_indexes:
                optimizer.add_indexes()
            
            if args.analyze:
                optimizer.analyze_tables()
            
            if args.benchmark:
                optimizer.benchmark_queries()
            
            if args.show_indexes:
                optimizer.show_index_info()
            
            if args.vacuum:
                optimizer.vacuum_database()
        
        logger.info("\n" + "="*80)
        logger.info("✅ OTIMIZAÇÃO COMPLETA!")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
