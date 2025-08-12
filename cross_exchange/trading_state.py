from shared_imports import asyncio, Config,datetime,redis,get_common_symbols,json
## 状态管理类
class TradingState:
    def __init__(self):
        self.shared_data = {}
        self.active_trades = []
        self.trade_history = []
        self.opening_positions = 0
        self.lock = asyncio.Lock()
        self.decision_id = 0
        self.decision_id_lock = asyncio.Lock()
        self.symbols = []
        self.okx_symbols = []
        self.latest_opportunity = None
        self.opportunity_lock = asyncio.Lock()
        self.redis_client = redis.Redis(host = 'localhost', port = 6379, db=0, decode_responses=True)

        # 资金管理
        self.initial_capital = Config.INITIAL_CAPITAL
        self.exchange_balances = {}
        self.total_balance = Config.INITIAL_CAPITAL
        self.total_pnl = 0.0
        self.balance_lock = asyncio.Lock()

    def init_exchange_balances(self):
        """初始化各交易所资金分配"""
        for exchange, allocation in Config.EXCHANGE_CAPITAL_ALLOCATION.items():
            self.exchange_balances[exchange] = {
                'available': self.initial_capital * allocation,
                'used': 0.0,
                'total': self.initial_capital * allocation
            }
    
    def get_available_capital(self, exchange):
        """获取指定交易所的可用资金"""
        return self.exchange_balances[exchange]['available']
    
    def calculate_trade_amount(self, buy_exchange, buy_price, sell_exchange, sell_price):
        buy_available = self.get_available_capital(buy_exchange)
        sell_available = self.get_available_capital(sell_exchange)

        # max_trade_capital = available_capital * Config.MAX_TRADE_CAPITAL_PCY

        # if max_trade_capital < Config.MIN_TRADE_AMOUNT:
        #     return 0.0
        
        buy_amount = buy_available / buy_price
        sell_amount = sell_available / sell_price
        trade_amount = min(buy_amount, sell_amount)
        return trade_amount

    
    async def get_next_decision_id(self):
        async with self.decision_id_lock:
            current_id = self.decision_id
            self.decision_id += 1
            return current_id

    def init_symbols(self):
        """初始化交易对"""
        self.symbols = get_common_symbols()
        self.okx_symbols = [symbol.replace('USDT','-USDT').replace('USDT','USDT-SWAP') for symbol in self.symbols]
        
        exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
        self.shared_data = {
            symbol: {exchange: {"bid": None, "ask": None} for exchange in exchanges}
            for symbol in self.symbols
        }

    async def update_redis_data(self):
        """Update Redis with data matching your display functions"""
        try:
            # 1. Market data (your existing shared_data)
            self.redis_client.set("trading:exchange_data", json.dumps(self.shared_data))
            
            # 2. Current active position (from display_trade)
            if self.active_trades:
                current_position = self.active_trades[0]  # Assuming single position
                self.redis_client.set("trading:current_position", json.dumps(current_position))
            else:
                self.redis_client.delete("trading:current_position")
            
            # 3. Formatted trade history (matching display_trade_history logic)
            formatted_history = self._format_trade_history()
            self.redis_client.set("trading:formatted_history", json.dumps(formatted_history))
            
            # 4. Balance information (matching display_balance_info)
            balance_info = {
                'initial_capital': self.initial_capital,
                'total_balance': self.total_balance,
                'total_pnl': self.total_pnl,
                'roi_percentage': (self.total_pnl / self.initial_capital) * 100,
                'exchange_balances': self.exchange_balances,
                'last_updated': datetime.now().isoformat()
            }
            self.redis_client.set("trading:balance", json.dumps(balance_info))
            
            # 5. Latest opportunity with spread info
            if self.latest_opportunity:
                self.redis_client.set("trading:latest_opportunity", json.dumps(self.latest_opportunity))
                
            # 6. Update metrics hash
            metrics = {
                'initial_capital': str(self.initial_capital),
                'total_balance': str(self.total_balance),
                'total_pnl': str(self.total_pnl),
                'roi_percentage': str((self.total_pnl / self.initial_capital) * 100),
                'active_trades_count': str(len(self.active_trades)),
                'total_trades_count': str(len(self.trade_history)),
                'last_updated': datetime.now().isoformat()
            }
            
            for key, value in metrics.items():
                self.redis_client.hset("trading:metrics", key, value)
                
        except Exception as e:
            print(f"Error updating Redis: {e}")

    def _format_trade_history(self):
        """Format trade history matching display_trade_history logic"""
        if not self.trade_history:
            return {
                'trade_pairs': [],
                'summary': {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            }

        # Group trades into pairs (same logic as display function)
        trade_pairs = []
        open_trades = {}

        for trade in self.trade_history:
            if trade['action'] == 'open':
                open_trades[trade['time_stamp_opportunity']] = trade
            elif trade['action'] == 'close':
                matching_open = None
                for timestamp, open_trade in open_trades.items():
                    if (open_trade['symbol'] == trade['symbol'] and 
                        open_trade['best_buy_exchange'] == trade['best_buy_exchange'] and
                        open_trade['best_sell_exchange'] == trade['best_sell_exchange']):
                        matching_open = open_trade
                        break

                if matching_open:
                    # Create trade pair with formatted data
                    pair = {
                        'symbol': matching_open['symbol'],
                        'buy_exchange': matching_open['best_buy_exchange'],
                        'sell_exchange': matching_open['best_sell_exchange'],
                        'open_buy_price': matching_open['best_buy_price'],
                        'open_sell_price': matching_open['best_sell_price'],
                        'close_buy_price': trade.get('current_buy_price'),
                        'close_sell_price': trade.get('current_sell_price'),
                        'open_time': matching_open['trade_time'],
                        'close_time': trade.get('close_time'),
                        'pnl': trade['pnl'],
                        'status': 'CLOSED',
                        'estimated_profit': matching_open.get('estimated_net_profit', 0)
                    }
                    trade_pairs.append(pair)
                    del open_trades[timestamp]

        # Add open trades
        for open_trade in open_trades.values():
            # Get current market prices for open trades
            current_buy_price = None
            current_sell_price = None
            
            symbol = open_trade['symbol']
            buy_ex = open_trade['best_buy_exchange']
            sell_ex = open_trade['best_sell_exchange']
            
            if symbol in self.shared_data:
                if buy_ex in self.shared_data[symbol] and self.shared_data[symbol][buy_ex].get('ask'):
                    current_buy_price = self.shared_data[symbol][buy_ex]['ask']
                if sell_ex in self.shared_data[symbol] and self.shared_data[symbol][sell_ex].get('bid'):
                    current_sell_price = self.shared_data[symbol][sell_ex]['bid']

            pair = {
                'symbol': open_trade['symbol'],
                'buy_exchange': open_trade['best_buy_exchange'],
                'sell_exchange': open_trade['best_sell_exchange'],
                'open_buy_price': open_trade['best_buy_price'],
                'open_sell_price': open_trade['best_sell_price'],
                'current_buy_price': current_buy_price,
                'current_sell_price': current_sell_price,
                'open_time': open_trade['trade_time'],
                'close_time': None,
                'pnl': str(open_trade.get('estimated_net_profit', 0)) + " (estimated)",
                'status': 'OPEN',
                'estimated_profit': open_trade.get('estimated_net_profit', 0)
            }
            trade_pairs.append(pair)

        # Calculate summary
        closed_trades = [pair for pair in trade_pairs if pair['status'] == 'CLOSED']
        if closed_trades:
            total_pnl = sum(trade['pnl'] for trade in closed_trades)
            avg_pnl = total_pnl / len(closed_trades)
            profitable_trades = len([trade for trade in closed_trades if trade['pnl'] > 0])
        else:
            total_pnl = avg_pnl = profitable_trades = 0

        return {
            'trade_pairs': trade_pairs,
            'summary': {
                'total_trades': len(closed_trades),
                'open_trades': len([pair for pair in trade_pairs if pair['status'] == 'OPEN']),
                'profitable_trades': profitable_trades,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            },
            'last_updated': datetime.now().isoformat()
        }

    def calculate_metrics(self):
        """Calculate trading performance metrics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit_per_trade': 0
            }
            
        profitable_trades = [t for t in self.trade_history if t.get('profit', 0) > 0]
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': (len(profitable_trades) / len(self.trade_history)) * 100,
            'total_profit': sum(t.get('profit', 0) for t in self.trade_history),
            'avg_profit_per_trade': sum(t.get('profit', 0) for t in self.trade_history) / len(self.trade_history),
            'best_trade': max((t.get('profit', 0) for t in self.trade_history), default=0),
            'worst_trade': min((t.get('profit', 0) for t in self.trade_history), default=0)
        }

    def add_trade(self, trade_data):
        """Add a new trade to history"""
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trade_history.append(trade_data)

    def update_balance(self, exchange, amount):
        """Update exchange balance"""
        if exchange in self.exchange_balances:
            self.exchange_balances[exchange]['available'] += amount
            self.total_balance = sum(bal['available'] + bal['used'] 
                                   for bal in self.exchange_balances.values())
            self.total_pnl = self.total_balance - self.initial_capital
