import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, TrendingDown, DollarSign, RefreshCw, Wifi, WifiOff } from 'lucide-react';
<script src="https://cdn.tailwindcss.com"></script>
const TradingDashboard = () => {
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

  const API_BASE = 'http://localhost:5036/api';

  const fetchData = async () => {
    try {
      const endpoints = [
        'balance',
        'current-position',
        'trade-history',
        'latest-opportunity'
      ];

      const responses = await Promise.allSettled(
        endpoints.map(endpoint => 
          fetch(`${API_BASE}/${endpoint}`)
            .then(res => res.ok ? res.json() : null)
        )
      );

      const [balance, currentPosition, tradeHistory, latestOpportunity] = responses.map(
        (result, index) => result.status === 'fulfilled' ? result.value : null
      );

      setData(prev => ({
        ...prev,
        balance,
        currentPosition,
        tradeHistory,
        latestOpportunity,
        lastUpdated: new Date().toISOString(),
        connected: true
      }));
      
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch data:', error);
      setData(prev => ({ ...prev, connected: false }));
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchData, 2000); // Update every 2 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

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

  const StatusHeader = () => (
    <div className="bg-gray-800 text-white p-4 rounded-lg mb-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            {data.connected ? (
              <Wifi className="w-5 h-5 text-green-400" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-400" />
            )}
            <span className={`font-medium ${data.connected ? 'text-green-400' : 'text-red-400'}`}>
              {data.connected ? '已连接' : '未连接'}
            </span>
          </div>
          
          {data.lastUpdated && (
            <div className="text-gray-300 text-sm">
              最后更新: {new Date(data.lastUpdated).toLocaleTimeString()}
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1 rounded text-sm font-medium ${
              autoRefresh ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}
          >
            自动刷新 {autoRefresh ? '开' : '关'}
          </button>
          
          <button
            onClick={fetchData}
            className="p-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
  
  const LatestOpportunityPanel = () => {
    if (!data.latestOpportunity) {
      return (
        <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <Activity className="w-8 h-8 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-500">暂无套利机会</p>
        </div>
      );
    }
  
    const opp = data.latestOpportunity;
    const spreadPct = (opp.open_spread_pct * 100).toFixed(2);
  
    return (
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 border-l-4 border-blue-500 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
          最新套利机会
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-600">交易对</p>
            <p className="text-xl font-bold text-gray-800">{opp.symbol}</p>
          </div>
          
          <div className="text-center">
            <p className="text-sm text-gray-600">买入 @ {opp.best_buy_exchange}</p>
            <p className="text-lg font-mono text-green-600">{formatPrice(opp.best_buy_price)}</p>
            <p className="text-sm text-gray-600">卖出 @ {opp.best_sell_exchange}</p>
            <p className="text-lg font-mono text-red-600">{formatPrice(opp.best_sell_price)}</p>
          </div>
          
          <div className="text-center">
            <p className="text-sm text-gray-600">价差比</p>
            <p className="text-2xl font-bold text-blue-600">{spreadPct}%</p>
            <p className="text-sm text-gray-500">{formatTime(opp.time_stamp_opportunity)}</p>
          </div>
        </div>
      </div>
    );
  };
  
  const CurrentPositionPanel = () => {
    if (!data.currentPosition) {
      return (
        <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <DollarSign className="w-8 h-8 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-500">暂无持仓</p>
        </div>
      );
    }
  
    const pos = data.currentPosition;
    const spreadPct = (pos.open_spread_pct * 100).toFixed(2);
  
    return (
      <div className="bg-yellow-50 border-l-4 border-yellow-500 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          <Activity className="w-5 h-5 mr-2 text-yellow-600" />
          当前持仓
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-600">交易对</p>
            <p className="text-lg font-bold text-gray-800">{pos.symbol}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-600">买入 @ {pos.best_buy_exchange}</p>
            <p className="text-lg font-mono text-green-600">{formatPrice(pos.best_buy_price)}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-600">卖出 @ {pos.best_sell_exchange}</p>
            <p className="text-lg font-mono text-red-600">{formatPrice(pos.best_sell_price)}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-600">开仓价差比</p>
            <p className="text-lg font-bold text-yellow-600">{spreadPct}%</p>
            <p className="text-xs text-gray-500">{formatTime(pos.trade_time)}</p>
          </div>
        </div>
      </div>
    );
  };
  
  const BalancePanel = () => {
    if (!data.balance) return null;
  
    const balance = data.balance;
    const roiColor = balance.roi_percentage >= 0 ? 'text-green-600' : 'text-red-600';
  
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">账户余额信息</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-4 bg-gray-50 rounded">
            <p className="text-sm text-gray-600">初始资金</p>
            <p className="text-xl font-bold text-gray-800">{balance.initial_capital?.toLocaleString()} USDT</p>
          </div>
          
          <div className="text-center p-4 bg-gray-50 rounded">
            <p className="text-sm text-gray-600">当前余额</p>
            <p className="text-xl font-bold text-gray-800">{balance.total_balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} USDT</p>
          </div>
          
          <div className="text-center p-4 bg-gray-50 rounded">
            <p className="text-sm text-gray-600">总盈亏</p>
            <p className={`text-xl font-bold ${roiColor}`}>{formatPnL(balance.total_pnl)} USDT</p>
          </div>
          
          <div className="text-center p-4 bg-gray-50 rounded">
            <p className="text-sm text-gray-600">投资回报率 (ROI)</p>
            <p className={`text-xl font-bold ${roiColor}`}>{balance.roi_percentage?.toFixed(4)}%</p>
          </div>
        </div>
  
        {balance.exchange_balances && Object.keys(balance.exchange_balances).length > 0 && (
          <div>
            <h4 className="font-medium text-gray-700 mb-3">各交易所资产分布</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">交易所</th>
                    <th className="text-right py-2">总额</th>
                    <th className="text-right py-2">可用</th>
                    <th className="text-right py-2">已用</th>
                    <th className="text-right py-2">利用率</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(balance.exchange_balances).map(([exchange, bal]) => {
                    const utilization = ((bal.used / bal.total) * 100).toFixed(2);
                    return (
                      <tr key={exchange} className="border-b">
                        <td className="py-2 font-medium">{exchange.toUpperCase()}</td>
                        <td className="text-right py-2">{bal.total?.toFixed(2)}</td>
                        <td className="text-right py-2">{bal.available?.toFixed(2)}</td>
                        <td className="text-right py-2">{bal.used?.toFixed(2)}</td>
                        <td className="text-right py-2">{utilization}%</td>
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
  
  const TradeHistoryTable = () => {
    if (!data.tradeHistory || !data.tradeHistory.trade_pairs) {
      return (
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">交易历史</h3>
          <p className="text-gray-500">暂无交易记录</p>
        </div>
      );
    }
  
    const { trade_pairs, summary } = data.tradeHistory;
  
    return (
      
      <div className="bg-white rounded-lg shadow-sm border p-6">
                {summary && (
          <div className="mt-6 p-4 bg-gray-50 rounded border-t">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="text-center">
                <p className="text-gray-600">交易总数</p>
                <p className="font-bold text-lg">{summary.total_trades}</p>
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
        <h3 className="text-lg font-semibold text-gray-800 mb-4">交易历史</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-gray-50">
                <th className="text-left p-3">序号</th>
                <th className="text-left p-3">交易对</th>
                <th className="text-left p-3">买入@交易所</th>
                <th className="text-left p-3">卖出@交易所</th>
                <th className="text-left p-3">开仓时间</th>
                <th className="text-left p-3">平仓时间</th>
                <th className="text-right p-3">盈亏</th>
                <th className="text-center p-3">状态</th>
              </tr>
            </thead>
            <tbody>
              {trade_pairs.slice().reverse().map((trade, index) => {
                const buyInfo = `${trade.buy_exchange?.slice(0, 3).toUpperCase()}@${formatPrice(trade.open_buy_price)}${trade.close_buy_price ? `/${formatPrice(trade.close_buy_price)}` : trade.current_buy_price ? `/${formatPrice(trade.current_buy_price)}` : '/无'}`;
                const sellInfo = `${trade.sell_exchange?.slice(0, 3).toUpperCase()}@${formatPrice(trade.open_sell_price)}${trade.close_sell_price ? `/${formatPrice(trade.close_sell_price)}` : trade.current_sell_price ? `/${formatPrice(trade.current_sell_price)}` : '/无'}`;
                const pnlColor = trade.pnl >= 0 ? 'text-green-600' : 'text-red-600';
                
                return (
                  <tr key={index} className="border-b hover:bg-gray-50">
                    <td className="p-3">{index + 1}</td>
                    <td className="p-3 font-medium">{trade.symbol}</td>
                    <td className="p-3 font-mono text-xs">{buyInfo}</td>
                    <td className="p-3 font-mono text-xs">{sellInfo}</td>
                    <td className="p-3">{formatTime(trade.open_time)}</td>
                    <td className="p-3">{trade.close_time ? formatTime(trade.close_time) : '未平仓'}</td>
                    <td className={`p-3 text-right font-mono ${pnlColor}`}>{formatPnL(trade.pnl)}</td>
                    <td className="p-3 text-center">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        trade.status === 'OPEN' ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {trade.status === 'OPEN' ? '持仓中' : trade.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
  

      </div>
    );
  };
  
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
  
  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">套利交易面板</h1>
          <p className="text-gray-600 mt-2">实时监控跨交易所交易机会</p>
        </div>
  
        <StatusHeader />
  
        <div className="grid grid-cols-2 gap-4 sm:gap-8 md:gap-12 px-4 mx-auto" style={{ maxWidth: '100%' }}>
  <LatestOpportunityPanel />
  <CurrentPositionPanel />
</div>
        <BalancePanel />
        <TradeHistoryTable />
      </div>
    </div>
  );
};
export default TradingDashboard;