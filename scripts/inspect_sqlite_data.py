import sqlite3
import pandas as pd
import os

import uuid

# Configurações do Banco
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nba_history.db')
SQLITE_TEMP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'nba_history_temp_{uuid.uuid4().hex}.db')

def inspect_data():
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"❌ Banco SQLite não encontrado em: {SQLITE_DB_PATH}")
        return

    # Backup para evitar lock
    try:
        if os.path.exists(SQLITE_TEMP_PATH):
            os.remove(SQLITE_TEMP_PATH)
        
        src = sqlite3.connect(SQLITE_DB_PATH)
        dst = sqlite3.connect(SQLITE_TEMP_PATH)
        with dst:
            src.backup(dst)
        dst.close()
        src.close()
    except Exception as e:
        print(f"❌ Erro ao fazer backup: {e}")
        return

    conn = sqlite3.connect(SQLITE_TEMP_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        for col in df.columns:
            # Tentar converter para numérico
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # Se tiver valores válidos
                if numeric_series.notna().sum() > 0:
                    max_val = numeric_series.max()
                    min_val = numeric_series.min()
                    
                    # Checar limites de INTEGER (Postgres: -2147483648 a +2147483647)
                    if max_val > 2147483647 or min_val < -2147483648:
                        print(f"❌ OUTLIER DETECTED in '{col}'!")
                        print(f"   Min: {min_val}, Max: {max_val}")
                        outliers = df[(numeric_series > 2147483647) | (numeric_series < -2147483648)]
                        print(f"   Count: {len(outliers)}")
                        print(outliers[[col, 'id']].head().to_string())
                    else:
                        pass
                        # print(f"✅ '{col}' OK (Min: {min_val}, Max: {max_val})")
            except Exception:
                pass # Não é numérico

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_data()
