import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Download } from 'lucide-react';
import { apiService } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';

export const DataExplorer = () => {
  const [loading, setLoading] = useState(true);
  const [countries, setCountries] = useState<any[]>([]);
  const [selectedCountry, setSelectedCountry] = useState<string>('');
  const [countryData, setCountryData] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadCountries();
  }, []);

  useEffect(() => {
    if (selectedCountry) {
      loadCountryData();
    }
  }, [selectedCountry]);

  const loadCountries = async () => {
    try {
      setLoading(true);
      const data = await apiService.getCountries();
      setCountries(data);
      if (data.length > 0) {
        setSelectedCountry(data[0].country_name);
      }
    } catch (error) {
      console.error('Error loading countries:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadCountryData = async () => {
    try {
      const data = await apiService.getCountryData(selectedCountry);
      setCountryData(data);
    } catch (error) {
      console.error('Error loading country data:', error);
    }
  };

  const filteredCountries = countries.filter(c => 
    c.country_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) return <LoadingSpinner />;

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Data Explorer</h1>
          <p className="text-gray-600">Explore detailed data for any country</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Country Selector */}
          <div className="lg:col-span-1">
            <div className="card sticky top-24">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Select Country</h3>
              <div className="relative mb-4">
                <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search countries..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {filteredCountries.map((country) => (
                  <button
                    key={country.country_name}
                    onClick={() => setSelectedCountry(country.country_name)}
                    className={`w-full text-left px-4 py-2 rounded-lg transition-colors ${
                      selectedCountry === country.country_name
                        ? 'bg-primary-500 text-white'
                        : 'hover:bg-gray-100'
                    }`}
                  >
                    {country.country_name}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Data Display */}
          <div className="lg:col-span-3">
            <div className="card">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">
                  {selectedCountry}
                </h2>
                <button className="btn-primary flex items-center space-x-2">
                  <Download className="h-5 w-5" />
                  <span>Export Data</span>
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Date
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Total Cases
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Total Deaths
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Vaccination %
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {countryData.slice(0, 20).map((row, idx) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {row.month_start_date}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {row.total_confirmed_cases_month_end?.toLocaleString() || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {row.total_confirmed_deaths_month_end?.toLocaleString() || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {row.people_fully_vaccinated_per_100_month_end?.toFixed(1) || 'N/A'}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

