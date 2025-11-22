# 🌍 COVID-19 Global Analytics Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📖 Overview

The **COVID-19 Global Analytics Platform** is a comprehensive, full-stack data engineering and analysis solution designed to track, process, and visualize global pandemic trends.

This project features an automated **ETL pipeline** that ingests raw data, a high-performance **FastAPI backend** for data serving, and an interactive **React dashboard** for exploring key metrics like vaccination rates, case fatality ratios, and government response stringency.

## ✨ Key Features

- **🚀 Automated ETL Pipeline**: Robust Python scripts to extract, transform, and load (ETL) complex COVID-19 datasets into a MySQL warehouse.
- **📊 Interactive Dashboard**: Modern React frontend with dynamic charts and maps to visualize global and country-specific trends.
- **⚡ High-Performance API**: RESTful API built with FastAPI, offering low-latency access to aggregated pandemic data.
- **🐳 Fully Containerized**: Docker Compose setup for one-command deployment of the entire stack (Database, Backend, Frontend).
- **📈 Advanced Analytics**: Insights into vaccination progress, correlation between policy stringency and infection rates, and more.

## 🛠️ Tech Stack

- **Data Engineering**: Python, Pandas, NumPy, SQLAlchemy
- **Backend**: FastAPI, Uvicorn, MySQL Connector
- **Frontend**: React, TypeScript, Tailwind CSS, Vite
- **Database**: MySQL 8.0
- **DevOps**: Docker, Docker Compose

## 📂 Folder Structure

```
.
├── api/                 # FastAPI backend application
├── data_engineering/    # ETL scripts and data processing logic
├── frontend/            # React frontend application
├── datasets/            # Raw CSV data files
├── docker-compose.yml   # Container orchestration config
└── scripts/             # Utility scripts
```

## 🚀 Installation & Usage

### Prerequisites
- Docker and Docker Compose installed on your machine.

### 🚀 Installation & Usage

You can run this project using **Docker** (Recommended) or **Manually**.

#### Option 1: Docker (Recommended)
This is the easiest way to get everything running.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ibadat-Ali86/Data_Analysis_Projects.git
    cd Data_Analysis_Projects/COVID19-Data-Analytics-Platform
    ```

2.  **Start the application**
    ```bash
    docker-compose up --build
    ```
    *This will start the MySQL database, FastAPI backend, and React frontend.*

3.  **Load Data (First Time Only)**
    The database will be empty initially. To populate it:
    - Open a new terminal.
    - Run the data loader script inside the backend container:
      ```bash
      # Enter the backend container
      docker exec -it covid_backend bash

      # Run the database creation and data loading script
      python ../data_engineering/db_creation_script.py
      ```

4.  **Access the services**
    - **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
    - **API Documentation**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

#### Option 2: Manual Installation
If you prefer to run services individually or don't have Docker.

**Prerequisites:**
- Python 3.9+
- Node.js 16+
- MySQL 8.0

**Step 1: Database Setup**
1.  Install and start MySQL.
2.  Create a user `ibadat` with password `ibadat` (or update `api/main.py` and `data_engineering/db_creation_script.py` with your credentials).
3.  Create the database:
    ```sql
    CREATE DATABASE covid_db;
    ```

**Step 2: Backend Setup**
1.  Navigate to the project root.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install fastapi uvicorn mysql-connector-python pandas sqlalchemy
    ```
4.  Run the data ingestion script:
    ```bash
    python data_engineering/db_creation_script.py
    ```
5.  Start the API server:
    ```bash
    cd api
    uvicorn main:app --reload
    ```

**Step 3: Frontend Setup**
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
4.  Open [http://localhost:5173](http://localhost:5173) (or the port shown in your terminal).

## 🔌 API Reference

The backend provides a comprehensive set of endpoints for accessing COVID-19 data. Full documentation is available via Swagger UI at `/api/docs`.

**Key Endpoints:**
- `GET /api/stats/global`: Global summary statistics.
- `GET /api/countries/{country_name}`: Detailed time-series data for a specific country.
- `GET /api/vaccination/progress`: Global vaccination trends over time.
- `GET /api/correlation/metrics`: Data for analyzing correlations between various metrics.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Ibadat Ali** - [LinkedIn Profile](https://linkedin.com/in/mirzaibadatali)

Project Link: [https://github.com/Ibadat-Ali86/Data_Analysis_Projects](https://github.com/Ibadat-Ali86/Data_Analysis_Projects/tree/main/COVID19-Data-Analytics-Platform)
