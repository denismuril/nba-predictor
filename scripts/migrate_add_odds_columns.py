#!/usr/bin/env python3
"""
Migration: Add Odds Columns to Predictions Table
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from data.repositories.db_manager import get_db_manager

def run_migration():
    db = get_db_manager()
    
    print("🔄 Iniciando Migration: Adicionar Colunas de Odds")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Verificar se colunas já existem
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='predictions'
        """)
        existing_cols = [row[0] for row in cursor.fetchall()]
        
        migrations = []
        
        if 'odds_home' not in existing_cols:
            migrations.append("ALTER TABLE predictions ADD COLUMN odds_home REAL DEFAULT 0.0")
        
        if 'odds_away' not in existing_cols:
            migrations.append("ALTER TABLE predictions ADD COLUMN odds_away REAL DEFAULT 0.0")
        
        if 'total_line' not in existing_cols:
            migrations.append("ALTER TABLE predictions ADD COLUMN total_line REAL DEFAULT 0.0")
        
        if 'odds_source' not in existing_cols:
            migrations.append("ALTER TABLE predictions ADD COLUMN odds_source TEXT DEFAULT 'none'")
        
        if not migrations:
            print("✅ Colunas já existem. Nada a fazer.")
            return
        
        for sql in migrations:
            print(f"   Executando: {sql}")
            cursor.execute(sql)
        
        conn.commit()
        print(f"✅ Migration concluída! {len(migrations)} colunas adicionadas.")

if __name__ == "__main__":
    run_migration()
