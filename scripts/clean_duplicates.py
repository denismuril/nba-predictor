#!/usr/bin/env python3
"""Remove duplicatas da tabela predictions"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.repositories.db_manager import get_db_manager

db = get_db_manager()

# Deletar TODAS as previsões de 2025-12-08
conn = db.get_connection()
try:
    cursor = conn.cursor()
    
    # Contar antes
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE date = '2025-12-08'")
    before = cursor.fetchone()[0]
    
    # Deletar
    cursor.execute("DELETE FROM predictions WHERE date = '2025-12-08'")
    conn.commit()
    
    print(f"✅ Removidas {before} previsões de 2025-12-08")
    print("🔄 Agora execute: python3 scripts/force_db_update.py")
    
except Exception as e:
    conn.rollback()
    print(f"❌ Erro: {e}")
finally:
    db.return_connection(conn)
