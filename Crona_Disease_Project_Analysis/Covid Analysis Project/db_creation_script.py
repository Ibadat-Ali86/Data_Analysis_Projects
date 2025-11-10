import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import os
import logging
import time

# ----------------------------
# Logging setup
# ----------------------------
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, "covid_db.log"),
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filemode='a'
)

# MySQL connection details (using password-based user)
MYSQL_USER = "ibadat"
MYSQL_PASSWORD = "ibadat"
MYSQL_HOST = "localhost"
DATABASE_NAME = "covid_db"

# Step 1-2: Connect to MySQL Server
# Step 3: Create covid_db database
# Step 4: Switch to covid_db
# Step 5: Create SQLAlchemy engine
try:
    # Connect to MySQL server (without specifying database)
    mydb = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    logging.info("Connected to MySQL server")
    print("Connected to MySQL server")

    mycursor = mydb.cursor()

    # Create database if not exists
    mycursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME};")
    mycursor.execute(f"USE {DATABASE_NAME};")

    # Create SQLAlchemy engine pointing to covid_db
    engine = create_engine(f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{DATABASE_NAME}")

    logging.info(f"Database '{DATABASE_NAME}' created/verified")
    print(f"Database '{DATABASE_NAME}' created/verified")

except mysql.connector.Error as e:
    error_msg = f"MySQL connection failed: {str(e)}"
    logging.error(error_msg)
    print(f"\n{error_msg}")
    print("\nFix: Run the user creation command first!")
    raise

except Exception as e:
    error_msg = f"Unexpected error: {str(e)}"
    logging.error(error_msg)
    print(f"\n{error_msg}")
    raise

# Function to ingest dataframe into MySQL
def ingest_db(df, table, engine):
    try:
        df.to_sql(table, engine, if_exists='replace', index=False, chunksize=1000)
        logging.info(f"Table '{table}' ingested successfully")
        print(f" Table '{table}' ingested successfully.")
    except Exception as e:
        logging.error(f"Error ingesting table '{table}': {str(e)}")
        print(f" Error ingesting table '{table}': {str(e)}")

# Load CSVs from current directory and push into DB
def load_raw_data():
    start = time.time()
    directory = os.getcwd()

    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        print(f" Directory does not exist: {directory}")
        return

    # Essential CSV files
    essential_files = ['CovidDeaths.csv', 'CovidVaccinations.csv']

    for file in essential_files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            try:
                print(f" Loading {file}...")
                df = pd.read_csv(file_path)
                table_name = file.replace(".csv", "")  # Remove .csv
                logging.info(f"Ingesting file: {file} | Shape: {df.shape}")
                print(f" Ingesting file: {file} | Shape: {df.shape}")

                ingest_db(df, table_name, engine)

                logging.info(f"{file} loaded successfully")
                print(f" {file} loaded successfully")

            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
                print(f" Error processing {file}: {str(e)}")
        else:
            logging.warning(f"File not found: {file}")
            print(f" File not found: {file}")

    end = time.time()
    duration = (end - start) / 60
    logging.info(f"Total ingestion time: {duration:.2f} minutes")
    print(f" Total ingestion time: {duration:.2f} minutes")

# Run ingestion
if __name__ == "__main__":
    load_raw_data()