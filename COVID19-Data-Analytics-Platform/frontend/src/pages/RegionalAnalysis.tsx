import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { apiService } from '../services/api';

export const RegionalAnalysis = () => {
  const [loading, setLoading] = useState(true);
  const [continentData, setContinentData] = useState<any[]>([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const data = await apiService.getContinentComparison('total_confirmed_cases_month_end');
      setContinentData(data);
    } catch (error) {
      console.error('Error loading regional data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Regional Analysis</h1>
          <p className="text-gray-600">Continental and regional comparisons</p>
        </motion.div>

        <div className="card">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Cases by Continent</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={continentData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="continent_name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_value" fill="#667eea" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

