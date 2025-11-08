
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import os
import logging
import time


# Logging setup
os.makedirs(os.path.join(os.getcwd(), 'logs'), exist_ok=True)

logging.basicConfig(
    filename="logs/covid_db.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filemode='a'
)

# ----------------------------
# MySQL connection details
# Step 1-2: Connect to MySQL Server
#          ↓
# Step 3: Create covid_db database
#          ↓
# Step 4: Switch to covid_db
#          ↓
# Step 5: Create engine pointing to covid_db
#          ↓
# Next: Use engine with pandas.to_sql()
# ----------------------------

#creating connection with python and sql server
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="ibadat"
)

#creating database through cursor
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE IF NOT EXISTS covid_db;")
mycursor.execute("USE covid_db;")
engine = create_engine("mysql+mysqlconnector://root:ibadat@localhost/covid_db")
# ----------------------------
# Function to ingest dataframe into MySQL
# ----------------------------
def ingest_db(df, table, engine):
    try:
        df.to_sql(table, engine, if_exists='replace', index=False)
        logging.info(f"Table {table} ingested successfully ..")
        print(f" Table '{table}' ingested successfully.")
    except Exception as e:
        logging.error(f"Error in ingesting table {table} .. {str(e)}")
        print(f" Error in ingesting table {table}: {str(e)}")

# Load CSVs from directory and push into DB
def load_raw_data():
    start = time.time()
    directory = r"D:\DATA SCIENCE\Data Analyst\Projects\Data_Analysis\Projects\Covid Analysis Project"

    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        print(f" Directory does not exist: {directory}")
        return

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            try:
                df = pd.read_csv(file_path)
                logging.info(f"Ingesting file: {file} with shape: {df.shape}")
                print(f" Ingesting file: {file} with shape {df.shape}")
                ingest_db(df, file[:-4], engine)  # remove ".csv" for table name
                logging.info(f"Ingesting file: {file} completed")
            except Exception as e:
                logging.error(f"Error occurred while ingesting file: {file} .. {str(e)}")
                print(f" Error occurred while ingesting file {file}: {str(e)}")

    end = time.time()
    logging.info(f"Total Time taken for ingestion: {(end-start)/60:.2f} minutes")
    print(f" Total Time taken for ingestion: {(end-start)/60:.2f} minutes")

# Run ingestion
if __name__ == "__main__":
    load_raw_data()

