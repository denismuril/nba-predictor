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
    
    # Check columns
    cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'predictions'")
    columns = cursor.fetchall()
    print("Columns:")
    for col in columns:
        print(f"  - {col[0]} ({col[1]})")
    
    # Check PK
    cursor.execute("""
        SELECT a.attname
        FROM   pg_index i
        JOIN   pg_attribute a ON a.attrelid = i.indrelid
                             AND a.attnum = ANY(i.indkey)
        WHERE  i.indrelid = 'predictions'::regclass
        AND    i.indisprimary;
    """)
    pk = cursor.fetchall()
    print("Primary Key:", pk)
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
