import React from 'react';

const BalancePanel = ({ balance, formatPnL }) => {
  // Render nothing if balance data is not yet available.
  if (!balance) return null;

  const roiColor = balance.roi_percentage >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">账户余额信息</h3>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">初始资金</p>
          <p className="text-xl font-bold text-gray-800">{balance.initial_capital?.toLocaleString()} USDT</p>
        </div>
        
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">当前余额</p>
          <p className="text-xl font-bold text-gray-800">{balance.total_balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT</p>
        </div>
        
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">总盈亏</p>
          <p className={`text-xl font-bold ${roiColor}`}>{formatPnL(balance.total_pnl)} USDT</p>
        </div>
        
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">投资回报率 (ROI)</p>
          <p className={`text-xl font-bold ${roiColor}`}>{balance.roi_percentage?.toFixed(4)}%</p>
        </div>
      </div>

      {/* Exchange Balances Table */}
      {balance.exchange_balances && Object.keys(balance.exchange_balances).length > 0 && (
        <div>
          <h4 className="font-medium text-gray-700 mb-3">各交易所资产分布</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-50">
                <tr className="border-b">
                  <th className="font-semibold p-3">交易所</th>
                  <th className="font-semibold p-3 text-right">总额</th>
                  <th className="font-semibold p-3 text-right">可用</th>
                  <th className="font-semibold p-3 text-right">已用</th>
                  <th className="font-semibold p-3 text-right">利用率</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(balance.exchange_balances).map(([exchange, bal]) => {
                  const utilization = bal.total > 0 ? ((bal.used / bal.total) * 100).toFixed(2) : 0;
                  return (
                    <tr key={exchange} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="p-3 font-medium">{exchange.toUpperCase()}</td>
                      <td className="p-3 text-right font-mono">{bal.total?.toFixed(2)}</td>
                      <td className="p-3 text-right font-mono text-green-600">{bal.available?.toFixed(2)}</td>
                      <td className="p-3 text-right font-mono text-red-600">{bal.used?.toFixed(2)}</td>
                      <td className="p-3 text-right font-mono">{utilization}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default BalancePanel;
