import pandas as pd
import os
from sqlalchemy import create_engine
import logging
import time

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Corrected logging setup
logging.basicConfig(
    filename='logs/ingestion_db.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Optional: List files in current directory
for file in os.listdir('.'):
    print(file)

# Create SQLite engine
engine = create_engine('sqlite:///inventory.db')

def ingest_db(df, table_name, engine):
    """Ingests a DataFrame into the specified database table."""
    try:
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        logging.info(f"Table '{table_name}' ingested successfully.")
    except Exception as e:
        logging.error(f"Error ingesting table '{table_name}': {e}")

def load_raw_data():
    """Loads CSV files from a directory and ingests them into the database."""
    start = time.time()
    directory = r'D:\DATA SCIENCE\Data Analyst\Projects\Vendor Performance Analysis'
    
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return

    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_csv(file_path)
                print(f"{file}: {df.shape}")
                logging.info(f"Ingesting file: {file} with shape: {df.shape}")
                ingest_db(df, file[:-4], engine)
            except Exception as e:
                logging.error(f"Failed to process file {file}: {e}")
    
    end = time.time()
    logging.info(f"Total time taken for ingestion: {(end - start)/60:.2f} minutes")

if __name__ == '__main__':
    load_raw_data()
