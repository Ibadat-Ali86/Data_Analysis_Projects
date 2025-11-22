import sqlite3
import pandas as pd
import logging
from ingestion_db import ingest_db
import numpy as np
import os
from sqlalchemy import create_engine
from ingestion_db import ingest_db, DB_PATH

logging.basicConfig(
    filename='logs/get_vendor_summary.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)


def create_vendor_summary(conn):
    '''this function will merge the different tables to get the overall vendor summary and adding new columns in the resultant data'''
    vendor_sales_summary = pd.read_sql_query("""SELECT 
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM (
    -- Purchase Summary
    SELECT 
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity) AS TotalPurchaseQuantity,
        SUM(p.Dollars) AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
) ps
LEFT JOIN (
    -- Sales Summary
    SELECT 
        VendorNo,
        Brand,
        SUM(SalesQuantity) AS TotalSalesQuantity,
        SUM(SalesDollars) AS TotalSalesDollars,
        SUM(SalesPrice) AS TotalSalesPrice,
        SUM(ExciseTax) AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
) ss ON ps.VendorNumber = ss.VendorNo AND ps.Brand = ss.Brand
LEFT JOIN (
    -- Freight Summary
    SELECT 
        VendorNumber,
        SUM(Freight) AS FreightCost
    FROM vendor_invoice
    GROUP BY VendorNumber
) fs ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC""", conn)
    return vendor_sales_summary


def clean_data(df):
    """Cleans the summary DataFrame and calculates new metrics."""
    # Avoid inplace modification on the original DataFrame by creating a copy
    df = df.copy()
    df['Volume'] = df['Volume'].astype('float')
    df.fillna(0, inplace=True)  # null values removed
    df.fillna(0, inplace=True)
    df['VendorName'] = df['VendorName'].str.strip()  # extra white spaces removed
    
    # Calculate metrics, safely handling potential division by zero
    df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
    df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
    df['StockTurnover'] = df['TotalSalesQuantity'] / df['TotalPurchaseQuantity']
    df['SalesToPurchaseRatio'] = (df['TotalSalesDollars'] / df['TotalPurchaseDollars'])

    # Use np.divide for safe division, replacing potential infinity with 0
    df['ProfitMargin'] = np.divide(df['GrossProfit'], df['TotalSalesDollars'], out=np.zeros_like(df['GrossProfit'], dtype=float), where=df['TotalSalesDollars']!=0) * 100
    df['StockTurnover'] = np.divide(df['TotalSalesQuantity'], df['TotalPurchaseQuantity'], out=np.zeros_like(df['TotalSalesQuantity'], dtype=float), where=df['TotalPurchaseQuantity']!=0)
    df['SalesToPurchaseRatio'] = np.divide(df['TotalSalesDollars'], df['TotalPurchaseDollars'], out=np.zeros_like(df['TotalSalesDollars'], dtype=float), where=df['TotalPurchaseDollars']!=0)
    
    return df


if __name__ == '__main__':
    # creating db connection
    conn = sqlite3.connect('inventory.db')
    # This configuration only runs when the script is executed directly
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/get_vendor_summary.log', mode='a'),
            logging.StreamHandler() # Log to console as well
        ]
    )

    logging.info('Creating vendor summary table*******************')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())
    # Use SQLAlchemy engine for consistency and resource management
    engine = create_engine(DB_PATH)
    with engine.connect() as connection:
        logger.info('Creating vendor summary table...')
        summary_df = create_vendor_summary(connection)
        logger.info(f"Created summary with {len(summary_df)} rows. Head:\n{summary_df.head()}")

    logging.info('Cleaning Data***************')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())
        logger.info('Cleaning data and calculating metrics...')
        clean_df = clean_data(summary_df)
        logger.info(f"Cleaned data head:\n{clean_df.head()}")

    logging.info('Ingesting into db***************')
    ingest_db(clean_df, 'vendor_sales_summary', conn)
    logging.info('Process Completed!')
        logger.info('Ingesting summary table into database...')
        ingest_db(clean_df, 'vendor_sales_summary', engine)

    logger.info('Process Completed!')
