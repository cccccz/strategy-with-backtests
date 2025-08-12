import React from 'react';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';

// This component only displays status info and provides controls.
// It receives all its data and functions as props from App.jsx.
const StatusHeader = ({ connected, lastUpdated, autoRefresh, setAutoRefresh, fetchData }) => (
  <div className="bg-white shadow-sm border rounded-lg p-4 mb-6">
    <div className="flex justify-between items-center">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          {connected ? (
            <Wifi className="w-5 h-5 text-green-500" />
          ) : (
            <WifiOff className="w-5 h-5 text-red-500" />
          )}
          <span className={`font-medium ${connected ? 'text-green-600' : 'text-red-600'}`}>
            {connected ? '已连接' : '未连接'}
          </span>
        </div>
        
        {lastUpdated && (
          <div className="text-gray-500 text-sm">
            最后更新: {new Date(lastUpdated).toLocaleTimeString()}
          </div>
        )}
      </div>
      
      <div className="flex items-center space-x-3">
        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
            autoRefresh 
              ? 'bg-green-600 hover:bg-green-700 text-white' 
              : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
          }`}
        >
          自动刷新 {autoRefresh ? '开' : '关'}
        </button>
        
        <button
          onClick={fetchData}
          className="p-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          aria-label="Manual Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
    </div>
  </div>
);

export default StatusHeader;
