import os
import sqlite3
import psycopg2
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

import shutil

# Configurações do Banco
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nba_history.db')
SQLITE_TEMP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nba_history_temp.db')

# Configurações Postgres (defaults ou do .env)
PG_HOST = os.getenv('DB_HOST', 'localhost')
PG_PORT = os.getenv('DB_PORT', '5432')
PG_NAME = os.getenv('DB_NAME', 'nba_predictor_db')
PG_USER = os.getenv('DB_USER', 'nba_admin')
PG_PASS = os.getenv('DB_PASS', 'password')

def get_sqlite_connection():
    if not os.path.exists(SQLITE_DB_PATH):
        logger.error(f"❌ Banco SQLite não encontrado em: {SQLITE_DB_PATH}")
        return None
    
    # Copiar para temp usando Backup API para evitar lock
    try:
        if os.path.exists(SQLITE_TEMP_PATH):
            os.remove(SQLITE_TEMP_PATH)
            
        src = sqlite3.connect(SQLITE_DB_PATH)
        dst = sqlite3.connect(SQLITE_TEMP_PATH)
        
        with dst:
            src.backup(dst)
            
        dst.close()
        src.close()
        logger.info(f"📋 Banco copiado para temporário via Backup API: {SQLITE_TEMP_PATH}")
    except Exception as e:
        logger.error(f"❌ Erro ao fazer backup do banco: {e}")
        return None
        
    return sqlite3.connect(SQLITE_TEMP_PATH)

def get_postgres_connection():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_NAME,
            user=PG_USER,
            password=PG_PASS
        )
        return conn
    except Exception as e:
        logger.error(f"❌ Erro ao conectar no PostgreSQL: {e}")
        return None

def create_postgres_schema(pg_conn):
    """Cria a tabela predictions no PostgreSQL compatível com o esquema atual."""
    drop_schema = "DROP TABLE IF EXISTS predictions CASCADE;"
    
    schema = """
    CREATE TABLE IF NOT EXISTS predictions (
        id VARCHAR(255) PRIMARY KEY,
        date TIMESTAMP,
        home_team TEXT,
        away_team TEXT,
        prob_home DOUBLE PRECISION,
        prob_away DOUBLE PRECISION,
        prob_mc_home DOUBLE PRECISION,
        prob_mc_away DOUBLE PRECISION,
        odd_home DOUBLE PRECISION,
        odd_away DOUBLE PRECISION,
        prediction TEXT,
        confidence TEXT,
        home_score BIGINT,
        away_score BIGINT,
        winner TEXT,
        correct BIGINT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        fgm DOUBLE PRECISION, fga DOUBLE PRECISION, fg3m DOUBLE PRECISION, 
        tov DOUBLE PRECISION, oreb DOUBLE PRECISION, dreb DOUBLE PRECISION, 
        fta DOUBLE PRECISION, ftm DOUBLE PRECISION,
        ast DOUBLE PRECISION, stl DOUBLE PRECISION, blk DOUBLE PRECISION, pf DOUBLE PRECISION, pts DOUBLE PRECISION,
        opp_fgm DOUBLE PRECISION, opp_fga DOUBLE PRECISION, opp_fg3m DOUBLE PRECISION, 
        opp_tov DOUBLE PRECISION, opp_oreb DOUBLE PRECISION, opp_dreb DOUBLE PRECISION, 
        opp_fta DOUBLE PRECISION, opp_ftm DOUBLE PRECISION,
        opp_ast DOUBLE PRECISION, opp_stl DOUBLE PRECISION, opp_blk DOUBLE PRECISION, opp_pf DOUBLE PRECISION, opp_pts DOUBLE PRECISION,
        predicted_spread DOUBLE PRECISION DEFAULT 0,
        predicted_total DOUBLE PRECISION DEFAULT 0
    );
    """
    try:
        with pg_conn.cursor() as cur:
            cur.execute(drop_schema)
            cur.execute(schema)
        pg_conn.commit()
        logger.info("✅ Schema criado/verificado no PostgreSQL.")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao criar schema: {e}")
        pg_conn.rollback()
        return False

def migrate_data():
    logger.info("🚀 Iniciando migração SQLite -> PostgreSQL...")
    
    # 1. Conectar SQLite
    sqlite_conn = get_sqlite_connection()
    if not sqlite_conn:
        return

    # 2. Ler dados
    try:
        logger.info("📦 Lendo dados do SQLite...")
        df = pd.read_sql_query("SELECT * FROM predictions", sqlite_conn)
        logger.info(f"📊 {len(df)} registros encontrados.")
    except Exception as e:
        logger.error(f"❌ Erro ao ler SQLite: {e}")
        return
    finally:
        sqlite_conn.close()

    if df.empty:
        logger.warning("⚠️ Tabela vazia. Nada para migrar.")
        return

    # 3. Conectar Postgres
    pg_conn = get_postgres_connection()
    if not pg_conn:
        logger.error("❌ Abortando migração: Falha na conexão Postgres.")
        logger.info("💡 Dica: Verifique se o PostgreSQL está rodando e as credenciais no .env estão corretas.")
        return

    # 4. Criar Schema
    if not create_postgres_schema(pg_conn):
        return

    # 5. Inserir Dados
    try:
        logger.info("🔄 Inserindo dados no PostgreSQL...")
        
        # Converter colunas de data para datetime nativo do Python (para o adaptador psycopg2)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Converter colunas inteiras explicitamente para evitar problemas de range/tipo
        int_cols = ['home_score', 'away_score', 'correct']
        for col in int_cols:
            if col in df.columns:
                # Converter para numérico, forçando erros para NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Converter para Int64 (nullable int do pandas)
                df[col] = df[col].astype('Int64')
                # Substituir <NA> por None para o SQL
                df[col] = df[col].astype(object).where(df[col].notnull(), None)
            
        # Converter NaNs para None (NULL no SQL) para o resto
        df = df.where(pd.notnull(df), None)
        
        # Preparar dados para execute_values
        columns = list(df.columns)
        values = [tuple(x) for x in df.to_numpy()]
        
        insert_query = f"""
        INSERT INTO predictions ({', '.join(columns)}) 
        VALUES %s
        ON CONFLICT (id) DO NOTHING;
        """
        
        with pg_conn.cursor() as cur:
            execute_values(cur, insert_query, values)
        
        pg_conn.commit()
        logger.info(f"✅ Migração concluída! {len(df)} registros processados.")
        
    except Exception as e:
        logger.error(f"❌ Erro na inserção de dados: {e}")
        pg_conn.rollback()
    finally:
        pg_conn.close()

if __name__ == "__main__":
    migrate_data()
