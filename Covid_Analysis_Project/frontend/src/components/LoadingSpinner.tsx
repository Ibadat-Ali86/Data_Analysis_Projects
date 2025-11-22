export const LoadingSpinner = () => {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="loading-spinner"></div>
    </div>
  );
};

export const LoadingScreen = () => {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <div className="loading-spinner mx-auto mb-4"></div>
        <p className="text-gray-600">Loading data...</p>
      </div>
    </div>
  );
};

