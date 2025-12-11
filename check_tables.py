import sqlite3
conn = sqlite3.connect('data/nba_games.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tabelas no banco:')
for table in cursor.fetchall():
    print(f'  - {table[0]}')
conn.close()
