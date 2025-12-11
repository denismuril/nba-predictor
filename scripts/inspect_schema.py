import sqlite3
import os

db_path = 'data/nba_history.db'

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(predictions)")
    columns = cursor.fetchall()
    print("Columns in 'predictions' table:")
    for col in columns:
        print(col)
    conn.close()
