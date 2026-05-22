import os
import time
import psycopg2
from pathlib import Path

def wait_for_db():
    dsn = os.getenv("DB_URI", "postgresql://saci:saci_secret@localhost:5432/saci_core")
    retries = 30
    while retries > 0:
        try:
            conn = psycopg2.connect(dsn)
            print("✅ Database connected!")
            return conn
        except psycopg2.OperationalError as e:
            print(f"⏳ Waiting for DB... ({30-retries}/30)")
            time.sleep(2)
            retries -= 1
    raise Exception("❌ Database Not Ready")

def init_db():
    conn = wait_for_db()
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Load Schema
    schema_path = Path(__file__).parent / "schema.sql"
    print(f"Applying schema from {schema_path}...")
    
    with open(schema_path, "r") as f:
        sql = f.read()
        
    try:
        cursor.execute(sql)
        print("✅ Schema applied successfully!")
    except Exception as e:
        print(f"⚠️ Schema error (might be already applied): {e}")
        
    conn.close()

if __name__ == "__main__":
    init_db()
