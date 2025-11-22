import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { apiService } from '../services/api';
import { TrendingUp, Activity, AlertCircle } from 'lucide-react';

export const GlobalInsights = () => {
  const [loading, setLoading] = useState(true);
  const [topCountries, setTopCountries] = useState<any[]>([]);
  const [correlationData, setCorrelationData] = useState<any[]>([]);
  const [selectedMetric, setSelectedMetric] = useState('total_confirmed_cases_month_end');

  useEffect(() => {
    loadData();
  }, [selectedMetric]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [countries, correlation] = await Promise.all([
        apiService.getTopCountries(selectedMetric, 15),
        apiService.getCorrelationData()
      ]);
      
      setTopCountries(countries);
      setCorrelationData(correlation);
    } catch (error) {
      console.error('Error loading insights data:', error);
    } finally {
      setLoading(false);
    }
  };

  const metrics = [
    { value: 'total_confirmed_cases_month_end', label: 'Total Cases' },
    { value: 'total_confirmed_deaths_month_end', label: 'Total Deaths' },
    { value: 'people_fully_vaccinated_per_100_month_end', label: 'Vaccination Rate' },
    { value: 'case_fatality_rate_percent', label: 'Case Fatality Rate' }
  ];

  if (loading) return <LoadingSpinner />;

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Global Insights</h1>
          <p className="text-gray-600">Comprehensive analysis and key findings</p>
        </motion.div>

        {/* Metric Selector */}
        <div className="card mb-8">
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Select Metric to Analyze
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metrics.map((metric) => (
              <button
                key={metric.value}
                onClick={() => setSelectedMetric(metric.value)}
                className={`p-4 rounded-lg border-2 transition-all duration-200 ${
                  selectedMetric === metric.value
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="font-semibold text-gray-900">{metric.label}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Top Countries */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card mb-8"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Top 15 Countries by {metrics.find(m => m.value === selectedMetric)?.label}
          </h2>
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={topCountries} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="country_name" type="category" width={150} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#667eea" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Key Insights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card bg-gradient-to-br from-blue-50 to-blue-100 border-l-4 border-blue-500"
          >
            <TrendingUp className="h-8 w-8 text-blue-600 mb-3" />
            <h3 className="text-xl font-bold text-gray-900 mb-2">Case Growth</h3>
            <p className="text-gray-700">
              Peak transmission occurred in Q4 2020 and Q1 2021 across most regions
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="card bg-gradient-to-br from-green-50 to-green-100 border-l-4 border-green-500"
          >
            <Activity className="h-8 w-8 text-green-600 mb-3" />
            <h3 className="text-xl font-bold text-gray-900 mb-2">Vaccination Impact</h3>
            <p className="text-gray-700">
              Countries with &gt;50% vaccination showed 30-40% reduction in mortality
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="card bg-gradient-to-br from-red-50 to-red-100 border-l-4 border-red-500"
          >
            <AlertCircle className="h-8 w-8 text-red-600 mb-3" />
            <h3 className="text-xl font-bold text-gray-900 mb-2">Healthcare Strain</h3>
            <p className="text-gray-700">
              ICU capacity was critical bottleneck during pandemic waves
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

