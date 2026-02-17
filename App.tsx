
import React from 'react';
import FaceLivenessDetector from './components/FaceLivenessDetector';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center p-4">
      <FaceLivenessDetector />
    </div>
  );
};

export default App;
