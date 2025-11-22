import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export interface GlobalStats {
  total_cases: number;
  total_deaths: number;
  total_vaccinated: number;
  avg_cfr: number;
  countries_count: number;
  continents_count: number;
  latest_date: string;
}

export interface TimelineData {
  month_start_date: string;
  value: number;
}

export interface Country {
  country_name: string;
  continent_name: string;
  total_population: number;
}

export interface ContinentComparison {
  continent_name: string;
  total_value: number;
  avg_value: number;
  country_count: number;
}

export interface TopCountry {
  country_name: string;
  continent_name: string;
  value: number;
  total_population: number;
}

export interface VaccinationProgress {
  month_start_date: string;
  avg_one_dose: number;
  avg_fully_vaccinated: number;
  total_monthly_doses: number;
}

// API Methods
export const apiService = {
  // Health check
  healthCheck: async () => {
    const response = await api.get('/api/health');
    return response.data;
  },

  // Global statistics
  getGlobalStats: async (): Promise<GlobalStats> => {
    const response = await api.get('/api/stats/global');
    return response.data;
  },

  // Timeline data
  getTimeline: async (metric: string, limit: number = 100): Promise<TimelineData[]> => {
    const response = await api.get('/api/stats/timeline', {
      params: { metric, limit }
    });
    return response.data;
  },

  // Countries
  getCountries: async (): Promise<Country[]> => {
    const response = await api.get('/api/countries');
    return response.data;
  },

  // Country data
  getCountryData: async (countryName: string, startDate?: string, endDate?: string) => {
    const response = await api.get(`/api/countries/${encodeURIComponent(countryName)}`, {
      params: { start_date: startDate, end_date: endDate }
    });
    return response.data;
  },

  // Continent comparison
  getContinentComparison: async (metric: string): Promise<ContinentComparison[]> => {
    const response = await api.get('/api/continents/comparison', {
      params: { metric }
    });
    return response.data;
  },

  // Top countries
  getTopCountries: async (metric: string, limit: number = 10): Promise<TopCountry[]> => {
    const response = await api.get('/api/top-countries', {
      params: { metric, limit }
    });
    return response.data;
  },

  // Vaccination progress
  getVaccinationProgress: async (): Promise<VaccinationProgress[]> => {
    const response = await api.get('/api/vaccination/progress');
    return response.data;
  },

  // Correlation data
  getCorrelationData: async () => {
    const response = await api.get('/api/correlation/metrics');
    return response.data;
  },

  // Business problems
  getBusinessProblemData: async (problemId: number) => {
    const response = await api.get(`/api/business-problems/${problemId}`);
    return response.data;
  },

  // Search
  search: async (queryText: string, field: string = 'country_name') => {
    const response = await api.get('/api/search', {
      params: { query_text: queryText, field }
    });
    return response.data;
  },

  // Export
  exportSummary: async () => {
    const response = await api.get('/api/export/summary');
    return response.data;
  }
};

export default api;

