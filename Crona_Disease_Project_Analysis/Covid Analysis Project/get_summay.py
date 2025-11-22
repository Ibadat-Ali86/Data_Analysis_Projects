"""
COVID-19 Data Processing Pipeline
*************************************************
Automated ETL pipeline for COVID-19 data analysis
Author: IBADAT ALI
Date: 10 November 2025
"""
import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import os
import logging
import warnings
from datetime import datetime
from typing import Tuple

warnings.filterwarnings('ignore')

# CONFIGURATION
class Config:
    """Pipeline configuration settings"""
    # Database Configuration
    DB_HOST = "localhost"
    DB_USER = "ibadat"
    DB_PASSWORD = "ibadat"
    DB_NAME = "covid_db"

    # Logging Configuration
    LOG_DIR = "logs"
    LOG_FILE = "final_covid_summary.log"

    # Data Quality Thresholds
    MAX_MISSING_PERCENT = 80
    MIN_DATA_QUALITY_SCORE = 0.7

    # Date Configuration
    VACCINE_START_DATE = '2021-01-01'
    EARLY_PANDEMIC_DATE = '2020-03-01'


# LOGGING SETUP
def setup_logging():
    """Initialize logging configuration"""
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(Config.LOG_DIR, Config.LOG_FILE),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("=" * 80)
    logging.info("COVID-19 DATA PIPELINE STARTED")
    logging.info("=" * 80)


# DATABASE CONNECTION
class DatabaseConnection:
    """Manages database connections and operations"""
    def __init__(self):
        self.conn = None
        self.engine = None

    def connect(self) -> Tuple[mysql.connector.connection.MySQLConnection, create_engine]:
        """Establish database connections"""
        try:
            # MySQL Connector
            self.conn = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME
            )
            # SQLAlchemy Engine
            connection_string = f"mysql+mysqlconnector://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}/{Config.DB_NAME}"
            self.engine = create_engine(connection_string)
            logging.info(f"Connected to database: {Config.DB_NAME}")
            return self.conn, self.engine
        except Exception as e:
            logging.error(f"Database connection failed: {str(e)}")
            raise

    def close(self):
        """Close database connections"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
        if self.engine:
            self.engine.dispose()
        logging.info("Database connections closed")


# DATA EXTRACTION
def extract_covid_data(conn) -> pd.DataFrame:
    """
    Extract and aggregate COVID-19 data from database
    Returns:
        pd.DataFrame: Monthly aggregated COVID data
    """
    logging.info("STAGE 1: Data Extraction Started")

    query = """
    SELECT
        -- Geographic Identifiers
        d.iso_code AS country_code,
        d.continent AS continent_name,
        d.location AS country_name,
        DATE_FORMAT(STR_TO_DATE(d.date, '%m/%d/%Y'), '%Y-%m-01') AS month_start_date,
        YEAR(STR_TO_DATE(d.date, '%m/%d/%Y')) AS year,
        MONTH(STR_TO_DATE(d.date, '%m/%d/%Y')) AS month,

        -- Demographics
        MAX(d.population) AS total_population,
        MAX(d.population_density) AS population_per_sq_km,
        MAX(d.median_age) AS median_age_years,
        MAX(d.aged_65_older) AS percent_aged_65_plus,
        MAX(d.aged_70_older) AS percent_aged_70_plus,

        -- Economic Indicators
        MAX(d.gdp_per_capita) AS gdp_per_capita_usd,
        MAX(d.extreme_poverty) AS percent_in_extreme_poverty,

        -- Health Infrastructure
        MAX(d.cardiovasc_death_rate) AS cardiovascular_death_rate,
        MAX(d.diabetes_prevalence) AS diabetes_prevalence_percent,
        MAX(d.female_smokers) AS female_smoking_rate,
        MAX(d.male_smokers) AS male_smoking_rate,
        MAX(d.handwashing_facilities) AS handwashing_facilities_percent,
        MAX(d.hospital_beds_per_thousand) AS hospital_beds_per_1000_pop,
        MAX(d.life_expectancy) AS life_expectancy_years,
        MAX(d.human_development_index) AS human_development_index_score,

        -- COVID Cases Metrics
        MAX(d.total_cases) AS total_confirmed_cases_month_end,
        SUM(d.new_cases) AS monthly_new_cases,
        AVG(d.new_cases_smoothed) AS avg_daily_cases_7day,
        MAX(d.total_cases_per_million) AS cases_per_million_month_end,
        AVG(d.new_cases_per_million) AS avg_daily_cases_per_million,
        AVG(d.new_cases_smoothed_per_million) AS avg_daily_cases_per_million_7day,

        -- COVID Deaths Metrics
        MAX(d.total_deaths) AS total_confirmed_deaths_month_end,
        SUM(d.new_deaths) AS monthly_new_deaths,
        AVG(d.new_deaths_smoothed) AS avg_daily_deaths_7day,
        MAX(d.total_deaths_per_million) AS deaths_per_million_month_end,
        AVG(d.new_deaths_per_million) AS avg_daily_deaths_per_million,
        AVG(d.new_deaths_smoothed_per_million) AS avg_daily_deaths_per_million_7day,

        -- Case Fatality Rate
        CASE
            WHEN MAX(d.total_cases) > 0
            THEN (MAX(d.total_deaths) / MAX(d.total_cases)) * 100
            ELSE NULL
        END AS case_fatality_rate_percent,

        -- Epidemiological Indicators
        AVG(d.reproduction_rate) AS avg_virus_reproduction_rate,
        MAX(d.reproduction_rate) AS max_virus_reproduction_rate,

        -- Healthcare System Load
        AVG(d.icu_patients) AS avg_icu_patients,
        MAX(d.icu_patients) AS peak_icu_patients,
        AVG(d.icu_patients_per_million) AS avg_icu_patients_per_million,
        MAX(d.icu_patients_per_million) AS peak_icu_patients_per_million,
        AVG(d.hosp_patients) AS avg_hospital_patients,
        MAX(d.hosp_patients) AS peak_hospital_patients,
        AVG(d.hosp_patients_per_million) AS avg_hospital_patients_per_million,
        MAX(d.hosp_patients_per_million) AS peak_hospital_patients_per_million,
        SUM(d.weekly_icu_admissions) AS total_monthly_icu_admissions,
        SUM(d.weekly_hosp_admissions) AS total_monthly_hospital_admissions,

        -- Testing Metrics
        SUM(v.new_tests) AS total_monthly_tests_conducted,
        MAX(v.total_tests) AS total_tests_conducted_month_end,
        MAX(v.total_tests_per_thousand) AS tests_per_thousand_month_end,
        AVG(v.new_tests_per_thousand) AS avg_daily_tests_per_thousand,
        AVG(v.new_tests_smoothed) AS avg_daily_tests_7day,
        AVG(v.positive_rate) AS avg_test_positivity_rate,
        MIN(v.positive_rate) AS min_test_positivity_rate,
        MAX(v.positive_rate) AS max_test_positivity_rate,
        AVG(v.tests_per_case) AS avg_tests_required_per_case,

        -- Vaccination Metrics
        MAX(v.total_vaccinations) AS total_vaccine_doses_month_end,
        MAX(v.people_vaccinated) AS people_with_at_least_one_dose_month_end,
        MAX(v.people_fully_vaccinated) AS people_fully_vaccinated_month_end,
        SUM(v.new_vaccinations) AS monthly_vaccinations,
        AVG(v.new_vaccinations_smoothed) AS avg_daily_vaccinations_7day,
        MAX(v.total_vaccinations_per_hundred) AS vaccine_doses_per_100_month_end,
        MAX(v.people_vaccinated_per_hundred) AS people_vaccinated_per_100_month_end,
        MAX(v.people_fully_vaccinated_per_hundred) AS people_fully_vaccinated_per_100_month_end,

        -- Policy Response
        AVG(v.stringency_index) AS avg_government_response_stringency,
        MAX(v.stringency_index) AS max_government_response_stringency,
        MIN(v.stringency_index) AS min_government_response_stringency,

        -- Record Count
        COUNT(*) AS days_in_month
    FROM
        CovidDeaths d
        INNER JOIN CovidVaccinations v
            ON d.location = v.location AND d.date = v.date
    WHERE
        d.continent IS NOT NULL
    GROUP BY
        d.iso_code, d.continent, d.location,
        DATE_FORMAT(STR_TO_DATE(d.date, '%m/%d/%Y'), '%Y-%m-01')
    ORDER BY
        d.location, year, month
    """

    df = pd.read_sql(query, conn)
    logging.info(f"Extracted {len(df):,} records")
    if not df.empty:
        # Convert month_start_date to datetime to handle mixed types
        df['month_start_date'] = pd.to_datetime(df['month_start_date'], errors='coerce')
        logging.info(f"Date Range: {df['month_start_date'].min()} to {df['month_start_date'].max()}")
        logging.info(f"Countries: {df['country_name'].nunique()}")
    return df


# DATA TRANSFORMATION
class DataTransformer:
    """Handles all data transformation operations"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not self.df.empty:
            self.df['month_start_date'] = pd.to_datetime(self.df['month_start_date'])

    def transform(self) -> pd.DataFrame:
        """Execute full transformation pipeline"""
        if self.df.empty:
            logging.warning("Empty DataFrame received. Skipping transformation.")
            return self.df

        logging.info("STAGE 2: Data Transformation Started")
        self.clean_missing_values()
        self.validate_business_logic()
        self.clean_outliers()
        self.engineer_features()
        logging.info("Transformation completed successfully")
        return self.df

    def clean_missing_values(self):
        """Strategic missing value imputation"""
        logging.info("Cleaning missing values...")

        # Core Demographics & COVID Metrics
        core_cols = [
            'total_population', 'human_development_index_score', 'gdp_per_capita_usd',
            'total_confirmed_cases_month_end', 'monthly_new_cases',
            'total_confirmed_deaths_month_end', 'monthly_new_deaths'
        ]
        for col in core_cols:
            if col in self.df.columns:
                if any(x in col for x in ['population', 'gdp', 'human_development']):
                    self.df[col] = self.df.groupby('country_name')[col].ffill()
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    early_mask = self.df['month_start_date'] < Config.EARLY_PANDEMIC_DATE
                    self.df.loc[early_mask, col] = self.df.loc[early_mask, col].fillna(0)
                    self.df[col] = self.df.groupby('country_name')[col].ffill()
                    self.df[col] = self.df[col].fillna(0)

        # Vaccination Data (Timeline-Aware)
        vaccination_cols = [col for col in self.df.columns if 'vaccin' in col.lower()]
        for col in vaccination_cols:
            pre_vaccine = self.df['month_start_date'] < Config.VACCINE_START_DATE
            self.df.loc[pre_vaccine, col] = 0
            vaccine_era = self.df['month_start_date'] >= Config.VACCINE_START_DATE
            if vaccine_era.any():
                self.df.loc[vaccine_era, col] = self.df.loc[vaccine_era].groupby('country_name')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
            self.df[col] = self.df[col].fillna(0)

        # Testing & Policy Data
        testing_policy_cols = [col for col in self.df.columns if 'test' in col.lower() or 'stringency' in col.lower()]
        for col in testing_policy_cols:
            self.df[col] = self.df.groupby('country_name')[col].ffill()
            if 'positivity' in col.lower():
                self.df[col] = self.df[col].fillna(10.0)
            elif 'stringency' in col.lower():
                self.df[col] = self.df[col].fillna(self.df[col].median())
            self.df[col] = self.df[col].clip(0, 100)
            if 'test' in col.lower() and 'rate' not in col.lower():
                self.df[col] = self.df[col].fillna(0)

        # Healthcare System
        healthcare_cols = [col for col in self.df.columns if 'icu' in col.lower() or 'hosp' in col.lower()]
        for col in healthcare_cols:
            self.df[f'{col}_available'] = (~self.df[col].isnull()).astype(int)
            self.df[col] = self.df[col].fillna(0)

        logging.info("Missing values handled")

    def validate_business_logic(self):
        """Validate and fix business rule violations"""
        logging.info("Validating business logic...")
        fixes = 0

        # Rule 1: Deaths ≤ Cases
        if all(c in self.df.columns for c in ['total_confirmed_deaths_month_end', 'total_confirmed_cases_month_end']):
            mask = self.df['total_confirmed_deaths_month_end'] > self.df['total_confirmed_cases_month_end']
            if mask.any():
                self.df.loc[mask, 'total_confirmed_deaths_month_end'] = self.df.loc[mask, 'total_confirmed_cases_month_end']
                fixes += mask.sum()

        # Percentages 0-100%
        percentage_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['percent', 'per_100', 'positivity', 'fatality'])]
        for col in percentage_cols:
            invalid = (self.df[col] > 100) | (self.df[col] < 0)
            if invalid.any():
                self.df[col] = self.df[col].clip(0, 100)
                fixes += invalid.sum()

        # No negative counts
        count_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['total_', 'monthly_', 'avg_', 'peak_'])]
        for col in count_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                negative = self.df[col] < 0
                if negative.any():
                    self.df.loc[negative, col] = 0
                    fixes += negative.sum()

        logging.info(f"Fixed {fixes} business logic violations")

    def clean_outliers(self):
        """Clean critical data quality outliers"""
        logging.info("Cleaning outliers...")
        corrections = 0

        if 'monthly_new_cases' in self.df.columns:
            self.df['is_high_transmission_month'] = (
                self.df['monthly_new_cases'] >
                self.df.groupby('country_name')['monthly_new_cases'].transform('quantile', 0.90)
            ).astype(int)

        if all(c in self.df.columns for c in ['max_government_response_stringency', 'min_government_response_stringency']):
            delta = self.df['max_government_response_stringency'] - self.df['min_government_response_stringency']
            self.df['is_policy_change_month'] = (delta > 30).astype(int)

        if 'avg_test_positivity_rate' in self.df.columns:
            mask = self.df['avg_test_positivity_rate'] > 50
            corrections = mask.sum()
            if corrections > 0:
                self.df.loc[mask, 'avg_test_positivity_rate'] = 50.0
            logging.info(f"Capped {corrections} extreme positivity rates at 50%")

    def engineer_features(self):
        """Create business-relevant features"""
        logging.info("Engineering features...")
        self.df = self.df.sort_values(['country_name', 'month_start_date'])

        self.df['quarter'] = self.df['month_start_date'].dt.quarter
        self.df['pandemic_year'] = self.df['year'] - 2020
        self.df['pandemic_phase'] = pd.cut(
            self.df['pandemic_year'],
            bins=[-1, 1, 2, 3, 10],
            labels=['Year_1', 'Year_2', 'Year_3', 'Later']
        )

        # Growth rates
        for col in ['total_confirmed_cases_month_end', 'people_vaccinated_per_100_month_end']:
            if col in self.df.columns:
                growth = self.df.groupby('country_name')[col].pct_change()
                growth = growth.replace([np.inf, -np.inf], np.nan)
                self.df[f'{col}_growth_rate'] = growth

        # Policy effectiveness
        if all(c in self.df.columns for c in ['monthly_new_cases', 'avg_government_response_stringency']):
            denom = self.df['avg_government_response_stringency'] + 1
            self.df['policy_effectiveness'] = np.divide(
                self.df['monthly_new_cases'], denom,
                out=np.zeros_like(self.df['monthly_new_cases'], dtype=float),
                where=denom != 0
            )

        # Development & Vaccination Levels
        if 'human_development_index_score' in self.df.columns:
            self.df['development_level'] = pd.cut(
                self.df['human_development_index_score'],
                bins=[0, 0.55, 0.70, 0.80, 1.0],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
        if 'people_vaccinated_per_100_month_end' in self.df.columns:
            self.df['vaccination_level'] = pd.cut(
                self.df['people_vaccinated_per_100_month_end'],
                bins=[0, 30, 60, 80, 100],
                labels=['Low', 'Medium', 'High', 'Very_High'],
                include_lowest=True
            )


# DATA LOADING
def load_to_database(df: pd.DataFrame, table_name: str, engine):
    """Load transformed data to database"""
    logging.info(f"STAGE 3: Loading to table '{table_name}'")
    try:
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        inf_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if np.isinf(df[col]).any()]
        if inf_cols:
            logging.info(f"Replaced inf in {len(inf_cols)} columns")

        df_clean.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )
        logging.info(f"Loaded {len(df_clean):,} records to {table_name}")
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise


# PIPELINE ORCHESTRATION
def run_pipeline():
    """Main pipeline orchestration"""
    start_time = datetime.now()
    db = DatabaseConnection()
    try:
        setup_logging()
        conn, engine = db.connect()

        raw_data = extract_covid_data(conn)
        if raw_data.empty:
            logging.warning("No data extracted. Exiting.")
            return pd.DataFrame()

        transformer = DataTransformer(raw_data)
        clean_data = transformer.transform()

        load_to_database(clean_data, 'covid_monthly_summary', engine)

        duration = (datetime.now() - start_time).total_seconds()
        logging.info("=" * 80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY")
        logging.info(f"Duration: {duration:.2f} seconds")
        logging.info(f"Final Records: {len(clean_data):,}")
        logging.info(f"Final Columns: {len(clean_data.columns)}")
        logging.info("=" * 80)

        return clean_data

    except Exception as e:
        logging.error(f"PIPELINE FAILED: {str(e)}")
        raise
    finally:
        db.close()


# MAIN EXECUTION
if __name__ == '__main__':
    final_data = run_pipeline()
    if not final_data.empty:
        print(f"\nPipeline completed! Data shape: {final_data.shape}")
        print(f"Check logs at: {os.path.join(Config.LOG_DIR, Config.LOG_FILE)}")
    else:
        print("\nPipeline completed but no data was processed.")