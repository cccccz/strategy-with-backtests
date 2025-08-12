import React from 'react';
import { Activity, DollarSign } from 'lucide-react';

const CurrentPositionPanel = ({ position, formatPrice, formatTime }) => {
  // Display a placeholder if there is no current position.
  if (!position) {
    return (
      <div className="bg-white border-2 border-dashed border-gray-300 rounded-lg p-6 text-center flex flex-col justify-center items-center h-full">
        <DollarSign className="w-8 h-8 text-gray-400 mx-auto mb-2" />
        <p className="text-gray-500 font-medium">暂无持仓</p>
      </div>
    );
  }

  const spreadPct = (position.open_spread_pct * 100).toFixed(3);

  return (
    <div className="bg-gradient-to-br from-yellow-50 to-orange-50 border-l-4 border-yellow-500 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
        <Activity className="w-5 h-5 mr-2 text-yellow-600" />
        当前持仓
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div>
          <p className="text-sm text-gray-600">交易对</p>
          <p className="text-lg font-bold text-gray-800">{position.symbol}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-600">买入 @ {position.best_buy_exchange}</p>
          <p className="text-lg font-mono text-green-600">{formatPrice(position.best_buy_price)}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-600">卖出 @ {position.best_sell_exchange}</p>
          <p className="text-lg font-mono text-red-600">{formatPrice(position.best_sell_price)}</p>
        </div>
        
        <div>
          <p className="text-sm text-gray-600">开仓价差比</p>
          <p className="text-lg font-bold text-yellow-600">{spreadPct}%</p>
          <p className="text-xs text-gray-500">{formatTime(position.trade_time)}</p>
        </div>
      </div>
    </div>
  );
};

export default CurrentPositionPanel;
