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
    
    # Add model_version
    try:
        print("Adding column 'model_version'...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT;")
    except psycopg2.errors.DuplicateColumn:
        print("Column 'model_version' already exists.")
        conn.rollback()
    else:
        conn.commit()
        print("✅ Column 'model_version' added.")

    # Add created_at
    try:
        print("Adding column 'created_at'...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
    except psycopg2.errors.DuplicateColumn:
        print("Column 'created_at' already exists.")
        conn.rollback()
    else:
        conn.commit()
        print("✅ Column 'created_at' added.")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
