import { motion } from 'framer-motion';
import { Database, Code, BarChart, Zap } from 'lucide-react';

export const About = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            About This Platform
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A comprehensive COVID-19 data analysis platform providing actionable insights 
            through advanced analytics and interactive visualizations
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card"
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Project Overview</h2>
            <p className="text-gray-600 mb-4">
              This platform transforms raw COVID-19 data into actionable intelligence 
              through advanced data engineering, statistical analysis, and business intelligence.
            </p>
            <ul className="space-y-2 text-gray-600">
              <li>• 200+ countries analyzed</li>
              <li>• 2793+ comprehensive data points</li>
              <li>• 10 critical business problems solved</li>
              <li>• Production-ready ETL pipeline</li>
              <li>• Multi-stage data validation</li>
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card"
          >
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Technologies</h2>
            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <Code className="h-6 w-6 text-blue-600 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-gray-900">Frontend</div>
                  <div className="text-sm text-gray-600">
                    React, TypeScript, Tailwind CSS, Recharts
                  </div>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <Database className="h-6 w-6 text-green-600 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-gray-900">Backend</div>
                  <div className="text-sm text-gray-600">
                    FastAPI, Python, MySQL, Pandas, NumPy
                  </div>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <BarChart className="h-6 w-6 text-purple-600 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-gray-900">Analytics</div>
                  <div className="text-sm text-gray-600">
                    SciPy, Scikit-learn, Statistical Modeling
                  </div>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <Zap className="h-6 w-6 text-yellow-600 flex-shrink-0" />
                <div>
                  <div className="font-semibold text-gray-900">Infrastructure</div>
                  <div className="text-sm text-gray-600">
                    Docker, SQLAlchemy, Uvicorn
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card text-center"
        >
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Author</h2>
          <p className="text-xl text-gray-700 mb-2">IBADAT ALI</p>
          <p className="text-gray-600">November 2025</p>
          <p className="text-gray-600 mt-4">
            Built with ❤️ for Global Health Intelligence
          </p>
        </motion.div>
      </div>
    </div>
  );
};

