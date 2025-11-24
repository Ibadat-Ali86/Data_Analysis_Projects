"""
Load pre-processed COVID monthly summary data into MySQL
"""
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import os

# MySQL connection details
MYSQL_USER = os.getenv("DB_USER", "ibadat")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD", "ibadat")
MYSQL_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_NAME = os.getenv("DB_NAME", "covid_db")

print("Connecting to database...")
try:
    # Create SQLAlchemy engine
    engine = create_engine(f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{DATABASE_NAME}")
    
    # Load CSV
    print("Loading COVID monthly summary CSV...")
    csv_path = os.path.join(os.path.dirname(__file__), '../datasets/covid_monthly_summary.csv')
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    # Load to database
    print("Loading data to MySQL final_covid table...")
    df.to_sql('final_covid', engine, if_exists='replace', index=False, chunksize=1000)
    
    print("✓ Successfully loaded covid_monthly_summary.csv to final_covid table")
    
    # Verify
    result_df = pd.read_sql("SELECT COUNT(*) as count FROM final_covid", engine)
    print(f"✓ Verified: {result_df['count'][0]} records in final_covid table")
    
except Exception as e:
    print(f"✗ Error: {str(e)}")
    raise
