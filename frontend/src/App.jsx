import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import PredictionPage from './pages/PredictionPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/predict" element={<PredictionPage />} />
      </Routes>
    </Router>
  );
}

export default App;
