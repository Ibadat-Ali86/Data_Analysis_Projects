import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Home } from './pages/Home'
import { Dashboard } from './pages/Dashboard'
import { GlobalInsights } from './pages/GlobalInsights'
import { RegionalAnalysis } from './pages/RegionalAnalysis'
import { VaccinationTracker } from './pages/VaccinationTracker'
import { DataExplorer } from './pages/DataExplorer'
import { BusinessProblems } from './pages/BusinessProblems'
import { About } from './pages/About'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/insights" element={<GlobalInsights />} />
          <Route path="/regional" element={<RegionalAnalysis />} />
          <Route path="/vaccination" element={<VaccinationTracker />} />
          <Route path="/explorer" element={<DataExplorer />} />
          <Route path="/business-problems" element={<BusinessProblems />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

