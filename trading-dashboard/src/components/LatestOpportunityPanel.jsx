import React from 'react';
import { TrendingUp, Activity } from 'lucide-react';

const LatestOpportunityPanel = ({ opportunity, formatPrice, formatTime }) => {
  // Display a placeholder if there is no opportunity data.
  if (!opportunity) {
    return (
      <div className="bg-white border-2 border-dashed border-gray-300 rounded-lg p-6 text-center flex flex-col justify-center items-center h-full">
        <Activity className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-gray-500 font-medium">正在扫描套利机会...</p>
      </div>
    );
  }

  const spreadPct = (opportunity.open_spread_pct * 100).toFixed(3);

  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border-l-4 border-blue-500 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
        <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
        最新套利机会
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
        <div className="text-center">
          <p className="text-sm text-gray-600">交易对</p>
          <p className="text-xl font-bold text-gray-900">{opportunity.symbol}</p>
        </div>
        
        <div className="text-center space-y-1">
          <p className="text-sm text-gray-600">买入 @ {opportunity.best_buy_exchange}</p>
          <p className="text-lg font-mono text-green-600">{formatPrice(opportunity.best_buy_price)}</p>
          <p className="text-sm text-gray-600">卖出 @ {opportunity.best_sell_exchange}</p>
          <p className="text-lg font-mono text-red-600">{formatPrice(opportunity.best_sell_price)}</p>
        </div>
        
        <div className="text-center">
          <p className="text-sm text-gray-600">价差比</p>
          <p className="text-2xl font-bold text-blue-600">{spreadPct}%</p>
          <p className="text-xs text-gray-500">{formatTime(opportunity.time_stamp_opportunity)}</p>
        </div>
      </div>
    </div>
  );
};

export default LatestOpportunityPanel;
