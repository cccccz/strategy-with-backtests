import React from 'react';

const TradeHistoryTable = ({ tradeHistory, formatPrice, formatTime, formatPnL }) => {
  // Display a placeholder if there is no trade history.
  if (!tradeHistory || !tradeHistory.trade_pairs) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">交易历史</h3>
        <p className="text-gray-500">暂无交易记录</p>
      </div>
    );
  }

  const { trade_pairs, summary } = tradeHistory;

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">交易历史</h3>
      
      {/* Trade History Table */}
      <div className="overflow-x-auto mb-6">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr className="border-b">
              <th className="text-left p-3 font-semibold">序号</th>
              <th className="text-left p-3 font-semibold">交易对</th>
              <th className="text-left p-3 font-semibold">买入(开/平)@交易所</th>
              <th className="text-left p-3 font-semibold">卖出(开/平)@交易所</th>
              <th className="text-left p-3 font-semibold">开仓时间</th>
              <th className="text-left p-3 font-semibold">平仓时间</th>
              <th className="text-right p-3 font-semibold">盈亏 (USDT)</th>
              <th className="text-center p-3 font-semibold">状态</th>
            </tr>
          </thead>
          <tbody>
            {trade_pairs.slice().reverse().map((trade, index) => {
              const buyInfo = `${trade.buy_exchange?.slice(0, 3).toUpperCase()}@${formatPrice(trade.open_buy_price)} / ${trade.close_buy_price ? formatPrice(trade.close_buy_price) : formatPrice(trade.current_buy_price) || 'N/A'}`;
              const sellInfo = `${trade.sell_exchange?.slice(0, 3).toUpperCase()}@${formatPrice(trade.open_sell_price)} / ${trade.close_sell_price ? formatPrice(trade.close_sell_price) : formatPrice(trade.current_sell_price) || 'N/A'}`;
              const pnlColor = trade.pnl >= 0 ? 'text-green-600' : 'text-red-600';
              
              return (
                <tr key={index} className="border-b hover:bg-gray-50 last:border-0">
                  <td className="p-3">{trade_pairs.length - index}</td>
                  <td className="p-3 font-medium">{trade.symbol}</td>
                  <td className="p-3 font-mono text-xs">{buyInfo}</td>
                  <td className="p-3 font-mono text-xs">{sellInfo}</td>
                  <td className="p-3">{formatTime(trade.open_time)}</td>
                  <td className="p-3">{trade.close_time ? formatTime(trade.close_time) : '未平仓'}</td>
                  <td className={`p-3 text-right font-mono ${pnlColor}`}>{formatPnL(trade.pnl)}</td>
                  <td className="p-3 text-center">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      trade.status === 'OPEN' ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'
                    }`}>
                      {trade.status === 'OPEN' ? '持仓中' : '已平仓'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary Section */}
      {summary && (
        <div className="p-4 bg-gray-50 rounded-lg border">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="text-center">
              <p className="text-gray-600">交易总数</p>
              <p className="font-bold text-lg text-gray-800">{summary.total_trades}</p>
            </div>
            <div className="text-center">
              <p className="text-gray-600">未平仓</p>
              <p className="font-bold text-lg text-yellow-600">{summary.open_trades}</p>
            </div>
            <div className="text-center">
              <p className="text-gray-600">盈利交易</p>
              <p className="font-bold text-lg text-green-600">{summary.profitable_trades}</p>
            </div>
            <div className="text-center">
              <p className="text-gray-600">总盈亏</p>
              <p className={`font-bold text-lg ${summary.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPnL(summary.total_pnl)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradeHistoryTable;
