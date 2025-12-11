#!/usr/bin/env python3
"""
Orquestrador de Limpeza e Retreino - NBA Predictor

REFATORADO: Suporta SQLite e PostgreSQL via DatabaseManager.

Este script realiza uma manutenção completa no sistema:
1. Limpa o banco de dados (remove duplicados e jogos inválidos)
2. Verifica a integridade dos dados restantes
3. Executa o retreinamento do modelo ML (V3)

Usage:
    python scripts/clean_and_retrain.py
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_database():
    """Executa limpeza profunda no banco de dados via SQL"""
    logger.info("🧹 Iniciando limpeza do banco de dados...")
    
    db = get_db_manager()
    logger.info(f"📂 Banco: {db.db_type.upper()}")
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Contar jogos antes
        cursor.execute(db._prepare_query("SELECT COUNT(*) FROM predictions"))
        total_before = cursor.fetchone()[0]
        
        # 2. Remover jogos com scores zerados (inválidos)
        cursor.execute(db._prepare_query(
            "DELETE FROM predictions WHERE home_score = 0 OR away_score = 0"
        ))
        zeros_removed = cursor.rowcount
        
        # 3. Remover duplicados (adaptado para ambos databases)
        if db.db_type == 'sqlite':
            # SQLite usa rowid
            cursor.execute("""
            DELETE FROM predictions 
            WHERE rowid NOT IN (
                SELECT MAX(rowid) 
                FROM predictions 
                GROUP BY date, home_team, away_team
            )
            """)
        else:  # postgres
            # PostgreSQL usa ctid
            cursor.execute("""
            DELETE FROM predictions p1
            USING predictions p2
            WHERE p1.ctid < p2.ctid
              AND p1.date = p2.date
              AND p1.home_team = p2.home_team
              AND p1.away_team = p2.away_team
            """)
        
        dupes_removed = cursor.rowcount
        
        # 4. Contar jogos depois
        cursor.execute(db._prepare_query("SELECT COUNT(*) FROM predictions"))
        total_after = cursor.fetchone()[0]
        
        conn.commit()
        
        logger.info(f"✅ Limpeza concluída:")
        logger.info(f"   Jogos antes: {total_before}")
        logger.info(f"   Scores zerados removidos: {zeros_removed}")
        logger.info(f"   Duplicados removidos: {dupes_removed}")
        logger.info(f"   Jogos restantes: {total_after}")
        
        return total_after
        
    except Exception as e:
        logger.error(f"❌ Erro ao limpar banco: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return 0
    finally:
        db.return_connection(conn)

def run_training():
    """Executa o script de treinamento V3"""
    logger.info("\n🚀 Iniciando retreinamento do modelo (V3)...")
    
    # Usar o mesmo interpretador python atual
    python_exe = sys.executable
    project_root = Path(__file__).parent.parent
    script_path = project_root / 'ml_pipeline' / 'train_ensemble_v3.py'
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    try:
        # Executar como subprocesso para garantir isolamento
        result = subprocess.run(
            [python_exe, str(script_path)],
            env=env,
            check=True,
            capture_output=False  # Deixar output ir para o console
        )
        logger.info("✅ Treinamento concluído com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Falha no treinamento (Exit code {e.returncode})")
        return False

def main():
    logger.info("="*80)
    logger.info("🤖 ORQUESTRADOR DE MANUTENÇÃO DO SISTEMA")
    logger.info("="*80)
    
    # 1. Limpar Banco
    remaining_games = clean_database()
    
    if remaining_games < 1000:
        logger.warning("⚠️  Atenção: Poucos jogos restantes no banco. O modelo pode não treinar bem.")
        logger.warning("   Considere rodar: python scripts/populate_historical_data.py")
        
        response = input("   Deseja continuar mesmo assim? (s/n): ")
        if response.lower() != 's':
            logger.info("🛑 Operação cancelada pelo usuário.")
            return
    
    # 2. Retreinar
    success = run_training()
    
    if success:
        logger.info("\n✨ SISTEMA ATUALIZADO E LIMPO COM SUCESSO!")
        logger.info("   Agora você tem um banco sem lixo e um modelo treinado com dados limpos.")
    else:
        logger.error("\n❌ Ocorreram erros durante o processo.")

if __name__ == "__main__":
    main()
