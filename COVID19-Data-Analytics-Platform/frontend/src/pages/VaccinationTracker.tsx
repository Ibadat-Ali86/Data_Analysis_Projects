import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { apiService } from '../services/api';
import { Syringe } from 'lucide-react';

export const VaccinationTracker = () => {
  const [loading, setLoading] = useState(true);
  const [vaccinationData, setVaccinationData] = useState<any[]>([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const data = await apiService.getVaccinationProgress();
      setVaccinationData(data);
    } catch (error) {
      console.error('Error loading vaccination data:', error);
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
          <div className="flex items-center space-x-3">
            <Syringe className="h-10 w-10 text-green-600" />
            <div>
              <h1 className="text-4xl font-bold text-gray-900">Vaccination Tracker</h1>
              <p className="text-gray-600">Global vaccination progress and trends</p>
            </div>
          </div>
        </motion.div>

        <div className="card">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Global Vaccination Progress Over Time
          </h2>
          <ResponsiveContainer width="100%" height={500}>
            <LineChart data={vaccinationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="month_start_date" 
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis label={{ value: '% of Population', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="avg_one_dose" 
                stroke="#4facfe" 
                strokeWidth={2}
                name="At Least One Dose (%)"
              />
              <Line 
                type="monotone" 
                dataKey="avg_fully_vaccinated" 
                stroke="#00f2fe" 
                strokeWidth={2}
                name="Fully Vaccinated (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

