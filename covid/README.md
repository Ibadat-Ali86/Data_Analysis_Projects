# 🌍 COVID-19 Global Analysis Platform

> **Interactive, Professional, and Creative COVID-19 Data Analysis Platform**

[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This project is a comprehensive **Data Engineering and Analysis Platform** designed to process, analyze, and visualize global COVID-19 data. It features a robust ETL pipeline, advanced statistical analysis of 10 critical business problems, and a modern, interactive web application.

## 📁 Project Structure

The project is organized into professional modules:

```text
Covid_Analysis_Project/
├── analysis/                   # 📊 EDA, Notebooks & Insights
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── Visualization.ipynb
│   └── INSIGHTS_SUMMARY.md
├── api/                        # 🚀 FastAPI Backend
├── data_engineering/           # ⚙️ ETL Pipelines & Scripts
│   ├── db_creation_script.py
│   └── get_summay.py
├── datasets/                   # 💾 Raw & Processed Data
├── frontend/                   # 🎨 React/Vite Frontend
├── logs/                       # 📝 System Logs
└── scripts/                    # 🛠️ Utility Scripts
```

---

## ✨ Key Features

### 1. Data Engineering
- **Automated ETL Pipeline**: Extracts data from CSVs, transforms it with cleaning and feature engineering, and loads it into a MySQL database.
- **Data Quality Checks**: Handles missing values, outliers, and ensures data consistency.

### 2. Advanced Analysis
- **10 Business Problems Solved**: In-depth analysis of transmission dynamics, healthcare capacity, vaccination impact, and more.
- **Statistical Modeling**: Correlation analysis, regression models, and time-series forecasting.

### 3. Modern Web Platform
- **Interactive Dashboard**: Real-time global statistics and visualizations.
- **Business Insights**: Dedicated section showcasing findings for each business problem.
- **Vaccination Tracker**: Global rollout progress and impact analysis.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Node.js 18+**
- **MySQL 8.0+**

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Covid_Analysis_Project
   ```

2. **Setup Database**
   Ensure MySQL is running and you have a user `ibadat` with password `ibadat` (or update `data_engineering/db_creation_script.py`).
   ```bash
   # Create DB and load raw data
   python data_engineering/db_creation_script.py
   
   # Run ETL pipeline
   python data_engineering/get_summay.py
   ```

3. **Run the Application**
   Use the helper script to start both backend and frontend:
   ```bash
   ./scripts/start_dev.sh
   ```

   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/api/docs

---

## 📊 Business Problems Analyzed

1. **Disease Spread Patterns**: Analysis of waves and reproduction rates.
2. **Healthcare Capacity**: Impact of ICU beds and resources on mortality.
3. **Vaccination Effectiveness**: Correlation between rollout speed and lives saved.
4. **Socioeconomic Impact**: How GDP and HDI correlated with outcomes.
5. **Regional Disparities**: Comparative analysis of continents.
6. **Policy Effectiveness**: Impact of government stringency measures.
7. **Pandemic Progression**: Temporal evolution from 2020 to 2022.
8. **Testing Strategy**: Relationship between testing rates and detection.
9. **Demographic Risks**: Vulnerability analysis by age and comorbidities.
10. **Mortality Determinants**: Factors influencing Case Fatality Rates.

---

## 🛠️ Technology Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Recharts, Framer Motion
- **Backend**: FastAPI, Python, Pandas, NumPy, SQLAlchemy
- **Database**: MySQL
- **DevOps**: Docker, Shell Scripting

---

## 👤 Author

**IBADAT ALI**
- Data Engineering & Analysis
- Full Stack Development

---

*Last Updated: November 2025*
