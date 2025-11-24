import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Globe,
  BarChart3,
  Activity,
  TrendingUp,
  Shield,
  Database,
  Zap,
  Users,
  CheckCircle
} from 'lucide-react';

export const Home = () => {
  const features = [
    {
      icon: Database,
      title: 'Comprehensive Data',
      description: '200+ countries analyzed with 2793+ data points covering the entire pandemic timeline'
    },
    {
      icon: BarChart3,
      title: 'Advanced Analytics',
      description: 'Statistical modeling, correlation analysis, and predictive insights'
    },
    {
      icon: Globe,
      title: 'Global Coverage',
      description: 'Multi-continental analysis with regional and country-level granularity'
    },
    {
      icon: Activity,
      title: 'Real-time Insights',
      description: 'Interactive visualizations and dynamic data exploration'
    },
    {
      icon: Shield,
      title: 'Data Quality',
      description: 'Multi-stage validation and automated ETL pipeline ensuring accuracy'
    },
    {
      icon: TrendingUp,
      title: 'Business Intelligence',
      description: '10 critical business problems solved with actionable recommendations'
    }
  ];

  const metrics = [
    { value: '200+', label: 'Countries Analyzed' },
    { value: '2793+', label: 'Data Points' },
    { value: '10', label: 'Business Problems Solved' },
    { value: '99.9%', label: 'Data Accuracy' }
  ];

  const problemsSolved = [
    'Disease Spread Patterns & Transmission Analysis',
    'Healthcare System Capacity & Resource Planning',
    'Vaccination Effectiveness & Rollout Optimization',
    'Socioeconomic Impact Assessment',
    'Regional Disparities & Equity Analysis',
    'Policy Effectiveness & Intervention Impact',
    'Pandemic Progression & Phase Analysis',
    'Testing Strategy Optimization',
    'Demographic Risk Factor Identification',
    'Case Fatality Rate Analysis'
  ];

  return (
    <div className="bg-gray-50">
      {/* Hero Section */}
      <section className="gradient-bg text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            {/* Author Badge */}
            <div className="flex items-center justify-center gap-4 mb-6">
              <div className="inline-block bg-white/10 backdrop-blur-sm px-6 py-3 rounded-full">
                <div className="flex items-center gap-3">
                  <span className="text-white font-semibold">Created by Ibadat Ali</span>
                  <div className="flex items-center gap-2">
                    <a
                      href="https://linkedin.com/in/mirzaibadatali"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="bg-white/20 hover:bg-white/30 p-2 rounded-full transition-all duration-300 hover:scale-110"
                      title="LinkedIn Profile"
                    >
                      <svg className="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                      </svg>
                    </a>
                    <a
                      href="https://github.com/Ibadat-Ali86"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="bg-white/20 hover:bg-white/30 p-2 rounded-full transition-all duration-300 hover:scale-110"
                      title="GitHub Profile"
                    >
                      <svg className="h-4 w-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                      </svg>
                    </a>
                  </div>
                </div>
              </div>
            </div>

            <div className="inline-block bg-white/10 backdrop-blur-sm px-4 py-2 rounded-full mb-6">
              <span className="text-yellow-300 font-semibold">🔬 Professional Data Engineering Project</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              COVID-19 Global Data
              <br />
              <span className="text-yellow-300">Analysis & Insights Platform</span>
            </h1>
            <p className="text-xl md:text-2xl text-blue-100 mb-8 max-w-3xl mx-auto">
              Transforming pandemic data into actionable intelligence through advanced analytics,
              interactive visualizations, and comprehensive business intelligence
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/dashboard"
                className="bg-white text-primary-600 px-8 py-4 rounded-lg font-semibold hover:bg-blue-50 transition-all duration-300 shadow-lg hover:shadow-xl flex items-center justify-center space-x-2"
              >
                <span>Explore Dashboard</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                to="/about"
                className="bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-lg font-semibold hover:bg-white/20 transition-all duration-300 border-2 border-white/30"
              >
                Learn More
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Metrics Section */}
      <section className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {metrics.map((metric, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold gradient-text mb-2">
                  {metric.value}
                </div>
                <div className="text-gray-600 font-medium">{metric.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Powerful Features
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Everything you need for comprehensive COVID-19 data analysis and insights
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="card hover:scale-105 transition-transform duration-300"
              >
                <div className="gradient-bg p-3 rounded-lg inline-block mb-4">
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Business Problems Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                10 Critical Business Problems Solved
              </h2>
              <p className="text-xl text-gray-600 mb-8">
                Data-driven solutions addressing real-world challenges in pandemic management,
                healthcare planning, and policy decision-making.
              </p>
              <Link
                to="/dashboard"
                className="btn-primary inline-flex items-center space-x-2"
              >
                <span>View Solutions</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="space-y-3"
            >
              {problemsSolved.map((problem, index) => (
                <div
                  key={index}
                  className="flex items-start space-x-3 bg-gray-50 p-4 rounded-lg hover:bg-primary-50 transition-colors duration-300"
                >
                  <CheckCircle className="h-6 w-6 text-green-500 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700 font-medium">{problem}</span>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Technology Stack Section */}
      <section className="py-20 bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Built with Modern Technologies
            </h2>
            <p className="text-xl text-gray-600">
              Production-ready stack for reliability and performance
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="card text-center">
              <Zap className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Frontend</h3>
              <p className="text-gray-600">
                React, TypeScript, Tailwind CSS, Recharts, Framer Motion
              </p>
            </div>
            <div className="card text-center">
              <Database className="h-12 w-12 text-blue-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Backend</h3>
              <p className="text-gray-600">
                FastAPI, Python, MySQL, Pandas, NumPy, SQLAlchemy
              </p>
            </div>
            <div className="card text-center">
              <Users className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Analytics</h3>
              <p className="text-gray-600">
                SciPy, Scikit-learn, Statistical Modeling, Machine Learning
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="gradient-bg text-white py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl font-bold mb-6">
              Ready to Explore the Data?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Dive into comprehensive COVID-19 analysis with interactive visualizations
              and actionable insights
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/dashboard"
                className="bg-white text-primary-600 px-8 py-4 rounded-lg font-semibold hover:bg-blue-50 transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                Launch Dashboard
              </Link>
              <Link
                to="/explorer"
                className="bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-lg font-semibold hover:bg-white/20 transition-all duration-300 border-2 border-white/30"
              >
                Explore Data
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};
