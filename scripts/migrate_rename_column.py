import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'nba_predictor_db')
DB_USER = os.getenv('DB_USER', 'nba_admin')
DB_PASS = os.getenv('DB_PASS', 'password')

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()
    
    print("Connected to Postgres.")
    print("Renaming column 'id' to 'game_id'...")
    
    cursor.execute("ALTER TABLE predictions RENAME COLUMN id TO game_id;")
    conn.commit()
    
    print("✅ Column renamed successfully!")
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
