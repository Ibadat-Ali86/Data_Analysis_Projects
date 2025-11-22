import { ReactNode, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Globe, BarChart3, Map, Syringe, Database, Info, Home } from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { name: 'Home', href: '/', icon: Home },
    { name: 'Dashboard', href: '/dashboard', icon: BarChart3 },
    { name: 'Global Insights', href: '/insights', icon: Globe },
    { name: 'Regional Analysis', href: '/regional', icon: Map },
    { name: 'Vaccination', href: '/vaccination', icon: Syringe },
    { name: 'Business Problems', href: '/business-problems', icon: BarChart3 },
    { name: 'Data Explorer', href: '/explorer', icon: Database },
    { name: 'About', href: '/about', icon: Info },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="gradient-bg text-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2">
              <Globe className="h-8 w-8" />
              <div className="flex flex-col">
                <span className="text-xl font-bold">COVID-19 Analysis</span>
                <span className="text-xs opacity-90">Global Insights Platform</span>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex space-x-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`flex items-center space-x-1 px-4 py-2 rounded-lg transition-all duration-200 ${
                      isActive(item.href)
                        ? 'bg-white/20 font-semibold'
                        : 'hover:bg-white/10'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </nav>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t border-white/20">
            <nav className="px-4 py-4 space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`flex items-center space-x-2 px-4 py-3 rounded-lg transition-all duration-200 ${
                      isActive(item.href)
                        ? 'bg-white/20 font-semibold'
                        : 'hover:bg-white/10'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </nav>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-3">COVID-19 Analysis Platform</h3>
              <p className="text-gray-400 text-sm">
                Data-driven insights for global health intelligence and pandemic response.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3">Quick Links</h3>
              <ul className="space-y-2 text-sm">
                <li><Link to="/dashboard" className="text-gray-400 hover:text-white transition-colors">Dashboard</Link></li>
                <li><Link to="/insights" className="text-gray-400 hover:text-white transition-colors">Global Insights</Link></li>
                <li><Link to="/vaccination" className="text-gray-400 hover:text-white transition-colors">Vaccination Tracker</Link></li>
                <li><Link to="/about" className="text-gray-400 hover:text-white transition-colors">About</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3">Technologies</h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>React + TypeScript + Tailwind CSS</li>
                <li>FastAPI + MySQL + Python</li>
                <li>Recharts + Framer Motion</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-6 text-center text-sm text-gray-400">
            <p>© 2025 IBADAT ALI | COVID-19 Global Analysis Platform</p>
            <p className="mt-1">Built with ❤️ for Global Health Intelligence</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

