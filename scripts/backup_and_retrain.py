#!/usr/bin/env python3
"""
Script para retreinar o modelo ML com backup do banco de dados.

REFATORADO: Suporta SQLite e PostgreSQL via DatabaseManager.
"""
import shutil
import os
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_and_retrain():
    """Faz backup do banco e retreina o modelo."""
    
    logger.info("🔄 Fazendo backup do banco de dados...")
    
    try:
        db = get_db_manager()
        logger.info(f"📂 Banco de dados: {db.db_type.upper()}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if db.db_type == 'sqlite':
            # Backup SQLite usando método nativo
            db_path = db.db_path
            backup_path = db_path.parent / f"nba_history_backup_{timestamp}.db"
            
            logger.info(f"   Origem: {db_path}")
            logger.info(f"   Destino: {backup_path}")
            
            if db_path.exists():
                # Usar sqlite3.backup() para backup online
                import sqlite3
                source_conn = sqlite3.connect(str(db_path))
                backup_conn = sqlite3.connect(str(backup_path))
                
                source_conn.backup(backup_conn)
                
                source_conn.close()
                backup_conn.close()
                
                logger.info(f"✅ Backup SQLite criado: {backup_path}")
                
                # Remover arquivos WAL se existirem
                for ext in ['-wal', '-shm']:
                    wal_file = str(db_path) + ext
                    if os.path.exists(wal_file):
                        os.remove(wal_file)
                        logger.info(f"🗑️  Removido: {wal_file}")
            else:
                logger.warning(f"⚠️  Banco de dados não encontrado: {db_path}")
                return False
                
        elif db.db_type == 'postgres':
            # Backup PostgreSQL usando pg_dump
            backup_path = Path(f"nba_history_backup_{timestamp}.sql")
            
            logger.info(f"   Database: {db.pg_name}")
            logger.info(f"   Destino: {backup_path}")
            
            # Construir comando pg_dump
            cmd = [
                'pg_dump',
                '-h', db.pg_host,
                '-p', db.pg_port,
                '-U', db.pg_user,
                '-d', db.pg_name,
                '-f', str(backup_path),
                '--no-owner',
                '--no-privileges'
            ]
            
            # Definir senha via env var
            env = os.environ.copy()
            env['PGPASSWORD'] = db.pg_pass
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Backup PostgreSQL criado: {backup_path}")
            else:
                logger.error(f"❌ Erro no pg_dump: {result.stderr}")
                return False
        
        else:
            logger.error(f"❌ Tipo de banco desconhecido: {db.db_type}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no backup: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n🎯 Iniciando retreinamento do modelo...")
    try:
        # Executar treinamento
        project_root = Path(__file__).parent.parent
        
        result = subprocess.run(
            [sys.executable, "-m", "ml_pipeline.train_ensemble"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            logger.info("\n✅ Modelo retreinado com sucesso!")
            return True
        else:
            logger.error(f"\n❌ Falha no retreinamento (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro no retreinamento: {e}")
        return False


if __name__ == "__main__":
    success = backup_and_retrain()
    sys.exit(0 if success else 1)
