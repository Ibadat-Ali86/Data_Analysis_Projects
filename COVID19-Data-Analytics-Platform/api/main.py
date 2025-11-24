"""
COVID-19 Data Analysis Platform - FastAPI Backend
Author: IBADAT ALI
Date: November 2025
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="COVID-19 Global Analysis API",
    description="RESTful API for COVID-19 data analysis and insights",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "ibadat"),
    "password": os.getenv("DB_PASSWORD", "ibadat"),
    "database": os.getenv("DB_NAME", "covid_db")
}

# Helper functions
def get_db_connection():
    """Create database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def query_to_json(query: str):
    """Execute query and return JSON"""
    conn = None
    try:
        conn = get_db_connection()
        df = pd.read_sql(query, conn)
        # Convert datetime columns to string
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
        # Replace NaN with None for JSON serialization
        df = df.replace({np.nan: None})
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        if conn:
            conn.close()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "COVID-19 Global Analysis API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "author": "IBADAT ALI"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except:
        return {"status": "unhealthy", "database": "disconnected"}

@app.get("/api/stats/global")
async def get_global_stats():
    """Get global COVID-19 statistics"""
    query = """
    SELECT 
        SUM(total_confirmed_cases_month_end) as total_cases,
        SUM(total_confirmed_deaths_month_end) as total_deaths,
        SUM(people_fully_vaccinated_month_end) as total_vaccinated,
        AVG(case_fatality_rate_percent) as avg_cfr,
        COUNT(DISTINCT country_name) as countries_count,
        COUNT(DISTINCT continent_name) as continents_count,
        MAX(month_start_date) as latest_date
    FROM final_covid
    WHERE month_start_date = (SELECT MAX(month_start_date) FROM final_covid)
    """
    result = query_to_json(query)
    return result[0] if result else {}

@app.get("/api/stats/timeline")
async def get_timeline(
    metric: str = Query(..., description="Metric to track (monthly_new_cases, monthly_new_deaths, etc.)"),
    limit: int = Query(100, description="Number of months to return")
):
    """Get timeline data for a specific metric"""
    allowed_metrics = [
        'monthly_new_cases', 'monthly_new_deaths', 
        'people_fully_vaccinated_per_100_month_end',
        'avg_government_response_stringency'
    ]
    
    if metric not in allowed_metrics:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Allowed: {allowed_metrics}")
    
    query = f"""
    SELECT 
        month_start_date,
        SUM({metric}) as value
    FROM final_covid
    GROUP BY month_start_date
    ORDER BY month_start_date DESC
    LIMIT {limit}
    """
    return query_to_json(query)

@app.get("/api/countries")
async def get_countries():
    """Get list of all countries"""
    query = """
    SELECT DISTINCT 
        country_name,
        continent_name,
        total_population
    FROM final_covid
    WHERE month_start_date = (SELECT MAX(month_start_date) FROM final_covid)
    ORDER BY country_name
    """
    return query_to_json(query)

@app.get("/api/countries/{country_name}")
async def get_country_data(
    country_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get detailed data for a specific country"""
    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND month_start_date BETWEEN '{start_date}' AND '{end_date}'"
    
    query = f"""
    SELECT *
    FROM final_covid
    WHERE country_name = '{country_name}'
    {date_filter}
    ORDER BY month_start_date
    """
    return query_to_json(query)

@app.get("/api/continents/comparison")
async def get_continent_comparison(
    metric: str = Query("total_confirmed_cases_month_end", description="Metric to compare")
):
    """Compare continents by metric"""
    query = f"""
    SELECT 
        continent_name,
        SUM({metric}) as total_value,
        AVG({metric}) as avg_value,
        COUNT(DISTINCT country_name) as country_count
    FROM final_covid
    WHERE month_start_date = (SELECT MAX(month_start_date) FROM final_covid)
    GROUP BY continent_name
    ORDER BY total_value DESC
    """
    return query_to_json(query)

@app.get("/api/top-countries")
async def get_top_countries(
    metric: str = Query("total_confirmed_cases_month_end", description="Metric to rank by"),
    limit: int = Query(10, description="Number of countries to return")
):
    """Get top countries by metric"""
    query = f"""
    SELECT 
        country_name,
        continent_name,
        {metric} as value,
        total_population
    FROM final_covid
    WHERE month_start_date = (SELECT MAX(month_start_date) FROM final_covid)
    ORDER BY {metric} DESC
    LIMIT {limit}
    """
    return query_to_json(query)

@app.get("/api/vaccination/progress")
async def get_vaccination_progress():
    """Get global vaccination progress over time"""
    query = """
    SELECT 
        month_start_date,
        AVG(people_vaccinated_per_100_month_end) as avg_one_dose,
        AVG(people_fully_vaccinated_per_100_month_end) as avg_fully_vaccinated,
        SUM(monthly_vaccinations) as total_monthly_doses
    FROM final_covid
    WHERE people_vaccinated_per_100_month_end IS NOT NULL
    GROUP BY month_start_date
    ORDER BY month_start_date
    """
    return query_to_json(query)

@app.get("/api/correlation/metrics")
async def get_correlation_data():
    """Get data for correlation analysis"""
    query = """
    SELECT 
        monthly_new_cases,
        monthly_new_deaths,
        case_fatality_rate_percent,
        avg_government_response_stringency,
        people_fully_vaccinated_per_100_month_end,
        avg_test_positivity_rate,
        total_population
    FROM final_covid
    WHERE month_start_date = (SELECT MAX(month_start_date) FROM final_covid)
    AND monthly_new_cases IS NOT NULL
    AND monthly_new_deaths IS NOT NULL
    """
    return query_to_json(query)

@app.get("/api/business-problems/{problem_id}")
async def get_business_problem_data(problem_id: int):
    """Get data specific to business problems (1-10)"""
    
    business_queries = {
        1: """
            SELECT continent_name, month_start_date, 
                   SUM(monthly_new_cases) as cases,
                   AVG(monthly_reproduction_rate) as reproduction_rate
            FROM final_covid
            GROUP BY continent_name, month_start_date
            ORDER BY month_start_date
        """,
        2: """
            SELECT country_name, 
                   AVG(avg_icu_patients) as avg_icu,
                   AVG(avg_hospital_patients) as avg_hospital,
                   AVG(hospital_beds_per_thousand) as beds_per_1000
            FROM final_covid
            WHERE avg_icu_patients IS NOT NULL
            GROUP BY country_name
            ORDER BY avg_icu DESC
            LIMIT 20
        """,
        3: """
            SELECT month_start_date,
                   AVG(people_fully_vaccinated_per_100_month_end) as vaccination_rate,
                   AVG(monthly_new_deaths) as avg_deaths
            FROM final_covid
            WHERE people_fully_vaccinated_per_100_month_end > 0
            GROUP BY month_start_date
            ORDER BY month_start_date
        """
    }
    
    if problem_id not in business_queries:
        raise HTTPException(status_code=404, detail="Business problem not found")
    
    return query_to_json(business_queries[problem_id])

@app.get("/api/search")
async def search_data(
    query_text: str = Query(..., description="Search term"),
    field: str = Query("country_name", description="Field to search in")
):
    """Search functionality"""
    query = f"""
    SELECT DISTINCT {field}
    FROM final_covid
    WHERE {field} LIKE '%{query_text}%'
    LIMIT 20
    """
    return query_to_json(query)

@app.get("/api/export/summary")
async def export_summary():
    """Export summary data"""
    query = """
    SELECT *
    FROM final_covid
    WHERE month_start_date = (SELECT MAX(month_start_date) FROM final_covid)
    ORDER BY total_confirmed_cases_month_end DESC
    LIMIT 100
    """
    return query_to_json(query)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

