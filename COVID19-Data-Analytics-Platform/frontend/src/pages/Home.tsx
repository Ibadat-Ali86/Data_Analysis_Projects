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

