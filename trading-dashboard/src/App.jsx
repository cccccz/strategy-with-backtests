import React, { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';

// Import the newly created components
import StatusHeader from './components/StatusHeader';
import LatestOpportunityPanel from './components/LatestOpportunityPanel';
import CurrentPositionPanel from './components/CurrentPositionPanel';
import BalancePanel from './components/BalancePanel';
import TradeHistoryTable from './components/TradeHistoryTable';

const App = () => {
  // --- STATE MANAGEMENT ---
  // All state is managed in the parent App component.
  const [data, setData] = useState({
    balance: null,
    currentPosition: null,
    tradeHistory: null,
    latestOpportunity: null,
    lastUpdated: null,
    connected: false
  });
  
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // --- API & DATA FETCHING ---
  const API_BASE = 'http://localhost:5036/api';

  const fetchData = async () => {
    try {
      const endpoints = [
        'balance',
        'current-position',
        'trade-history',
        'latest-opportunity'
      ];

      // Fetch all data points concurrently
      const responses = await Promise.allSettled(
        endpoints.map(endpoint => 
          fetch(`${API_BASE}/${endpoint}`)
            .then(res => res.ok ? res.json() : null)
        )
      );

      const [balance, currentPosition, tradeHistory, latestOpportunity] = responses.map(
        (result) => result.status === 'fulfilled' ? result.value : null
      );

      setData({
        balance,
        currentPosition,
        tradeHistory,
        latestOpportunity,
        lastUpdated: new Date().toISOString(),
        connected: true
      });
      
    } catch (error) {
      console.error('Failed to fetch data:', error);
      setData(prev => ({ ...prev, connected: false }));
    } finally {
        setLoading(false);
    }
  };

  // Effect for initial data load and auto-refreshing
  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchData, 2000); // Update every 2 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // --- HELPER FUNCTIONS ---
  // These are kept in the parent to be passed down as props where needed.
  const formatPrice = (price) => {
    if (typeof price !== 'number') return 'N/A';
    return price.toFixed(6);
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000); // Assuming Unix timestamp
    return date.toLocaleString('en-US', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatPnL = (pnl) => {
    if (typeof pnl !== 'number') return 'N/A';
    const formatted = pnl.toFixed(2);
    return pnl >= 0 ? `+${formatted}` : formatted;
  };

  // --- RENDER LOGIC ---
  // Display a loading indicator while fetching initial data.
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">加载交易面板中...</p>
        </div>
      </div>
    );
  }
  
  // Render the main dashboard layout with child components
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">套利交易面板</h1>
          <p className="text-gray-600 mt-2">实时监控跨交易所交易机会</p>
        </div>
  
        {/* Render child components and pass necessary data and functions as props */}
        <StatusHeader 
          connected={data.connected}
          lastUpdated={data.lastUpdated}
          autoRefresh={autoRefresh}
          setAutoRefresh={setAutoRefresh}
          fetchData={fetchData}
        />
  
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <LatestOpportunityPanel 
            opportunity={data.latestOpportunity}
            formatPrice={formatPrice}
            formatTime={formatTime}
          />
          <CurrentPositionPanel 
            position={data.currentPosition}
            formatPrice={formatPrice}
            formatTime={formatTime}
          />
        </div>
        
        <BalancePanel 
          balance={data.balance}
          formatPnL={formatPnL}
        />
        
        <TradeHistoryTable 
          tradeHistory={data.tradeHistory}
          formatPrice={formatPrice}
          formatTime={formatTime}
          formatPnL={formatPnL}
        />
      </div>
    </div>
  );
};

export default App;
