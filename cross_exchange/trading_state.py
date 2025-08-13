from typing import Any, Dict, Union
from shared_imports import asyncio, Config,datetime,redis,get_common_symbols,json
## 状态管理类
class TradingState:
    """
    Trading state management class for cryptocurrency arbitrage trading.
    
    Manages trading positions, balances, market data, and performance metrics
    across multiple exchanges. Provides thread-safe operations and Redis
    integration for data persistence and monitoring.
    
    Attributes:
        # shared_data (Dict[str, Dict[str, Dict[str, Optional[float]]]]): Market data structure
            {symbol: {exchange: {
                                    "bid": price,     # 保持兼容
                                    "ask": price,     # 保持兼容  
                                    "orderbook": {    # 新增深度数据
                                        "bids": [(price, quantity), ...],  # 按价格降序
                                        "asks": [(price, quantity), ...],  # 按价格升序
                                        "timestamp": timestamp
                                    }
                                }}  
        active_trades (List[Dict[str, Any]]): Currently open trading positions
        trade_history (List[Dict[str, Any]]): Complete history of all trades
        opening_positions (int): Count of currently opening positions
        lock (asyncio.Lock): Async lock for thread-safe operations
        decision_id (int): Counter for unique decision identification
        decision_id_lock (asyncio.Lock): Lock for decision ID generation
        symbols (List[str]): List of trading symbol pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        okx_symbols (List[str]): OKX-formatted symbol pairs for API compatibility
        # latest_opportunity (Optional[Dict[str, Any]]): Most recent arbitrage opportunity
        opportunity_lock (asyncio.Lock): Lock for opportunity updates
        redis_client (redis.Redis): Redis client for data persistence
        initial_capital (float): Starting capital amount
        exchange_balances (Dict[str, Dict[str, float]]): Balance tracking per exchange
            {exchange: {"available": float, "used": float, "total": float}}
        total_balance (float): Current total balance across all exchanges
        total_pnl (float): Total profit and loss since inception
        balance_lock (asyncio.Lock): Lock for balance operations
    """
    def __init__(self):
        """
        Initialize TradingState with default values and Redis connection.
        
        Sets up initial state, locks, counters, and establishes Redis connection
        for data persistence and monitoring.
        """
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
        """
        Initialize exchange balance allocation based on configuration.
        
        Distributes initial capital across exchanges according to 
        Config.EXCHANGE_CAPITAL_ALLOCATION ratios. Sets up balance tracking
        structure for each configured exchange.
        """
        for exchange, allocation in Config.EXCHANGE_CAPITAL_ALLOCATION.items():
            self.exchange_balances[exchange] = {
                'available': self.initial_capital * allocation,
                'used': 0.0,
                'total': self.initial_capital * allocation
            }
    
    def get_available_capital(self, exchange: str) -> float:
        """
        Get available capital for specified exchange.
        
        Args:
            exchange: Exchange name (e.g., "Binance", "OKX", "Bitget", "Bybit")
            
        Returns:
            Available capital amount for the exchange
            
        Raises:
            KeyError: If exchange is not configured in exchange_balances
        """
        return self.exchange_balances[exchange]['available']
    
    def calculate_trade_amount(self, buy_exchange: str, buy_price: float, 
                            sell_exchange: str, sell_price: float) -> float:
        """
        Calculate maximum tradeable amount based on available capital.
        
        Determines the maximum quantity that can be traded considering
        available capital on both buy and sell exchanges.
        
        Args:
            buy_exchange: Exchange where the buy order will be placed
            buy_price: Price at which to buy (ask price)
            sell_exchange: Exchange where the sell order will be placed  
            sell_price: Price at which to sell (bid price)
            
        Returns:
            Maximum tradeable quantity (in base currency units)
        """
        buy_available = self.get_available_capital(buy_exchange)
        sell_available = self.get_available_capital(sell_exchange)

        # max_trade_capital = available_capital * Config.MAX_TRADE_CAPITAL_PCY

        # if max_trade_capital < Config.MIN_TRADE_AMOUNT:
        #     return 0.0
        
        buy_amount = buy_available / buy_price
        sell_amount = sell_available / sell_price
        trade_amount = min(buy_amount, sell_amount)
        return trade_amount

    
    async def get_next_decision_id(self) -> int:
        """
        Generate next unique decision ID in thread-safe manner.
        
        Returns:
            Unique integer decision ID for trade tracking
        """
        async with self.decision_id_lock:
            current_id = self.decision_id
            self.decision_id += 1
            return current_id

    def init_symbols(self):
        """
        Initialize trading symbols and market data structure.
        
        Sets up symbols list from common_symbols, creates OKX-compatible
        symbol format, and initializes shared_data structure for market
        data storage across all exchanges.
        """
        self.symbols = get_common_symbols()
        self.okx_symbols = [symbol.replace('USDT','-USDT').replace('USDT','USDT-SWAP') for symbol in self.symbols]
        
        exchanges = ['Binance', 'OKX', 'Bitget', 'Bybit']
        self.shared_data = {
            symbol: {exchange: {"bid": None, "ask": None, "orderbook":{"bids":None,"asks":None,"timestamp":None}                
            } for exchange in exchanges}for symbol in self.symbols
        }

    async def update_redis_data(self):
        """
        Update Redis with current trading state data.
        
        Synchronizes multiple Redis keys with current state including:
        - Market data (exchange_data)
        - Current active position
        - Formatted trade history
        - Balance information and metrics
        - Latest arbitrage opportunity
        - Performance metrics hash
        
        Raises:
            Exception: Logs error if Redis update fails
        """
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

    def _format_trade_history(self) -> Dict[str, Any]:
        """
        Format trade history into paired open/close trades with summary.
        
        Groups trade history into matched pairs of open/close actions,
        calculates current market prices for open trades, and generates
        performance summary statistics.
        
        Returns:
            Dictionary containing:
            - trade_pairs: List of formatted trade pair dictionaries
            - symbol (str): Trading symbol
            - buy_exchange (str): Buy exchange name
            - sell_exchange (str): Sell exchange name  
            - open_buy_price (float): Opening buy price
            - open_sell_price (float): Opening sell price
            - close_buy_price (Optional[float]): Closing buy price
            - close_sell_price (Optional[float]): Closing sell price
            - current_buy_price (Optional[float]): Current buy price for open trades
            - current_sell_price (Optional[float]): Current sell price for open trades
            - open_time (float): Opening timestamp
            - close_time (Optional[float]): Closing timestamp
            - pnl (Union[float, str]): Realized PnL or estimated for open trades
            - status (str): "OPEN" or "CLOSED"
            - estimated_profit (float): Expected profit amount
            - summary: Performance summary dictionary
            - total_trades (int): Count of closed trades
            - open_trades (int): Count of currently open trades
            - profitable_trades (int): Count of profitable closed trades
            - total_pnl (float): Sum of all realized PnL
            - avg_pnl (float): Average PnL per closed trade
            - last_updated (str): ISO format timestamp
        """
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

    def calculate_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Calculate comprehensive trading performance metrics.
        
        Returns:
            Performance metrics dictionary:
            - total_trades (int): Total number of trades executed
            - win_rate (float): Percentage of profitable trades (0-100)
            - total_profit (float): Sum of all trade profits
            - avg_profit_per_trade (float): Average profit per trade
            - best_trade (float): Highest single trade profit
            - worst_trade (float): Lowest single trade profit (most negative loss)
        """
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

    def add_trade(self, trade_data: Dict[str, Any]):
        """
        Add new trade record to history with timestamp.
        
        Args:
            trade_data: Trade information dictionary containing:
                Required fields depend on action:
                - action (str): "open" or "close"
                - symbol (str): Trading symbol
                - decision_id (int): Unique decision identifier
                - For "open": best_buy_exchange, best_sell_exchange, 
                            best_buy_price, best_sell_price, trade_amount, etc.
                - For "close": pnl, exit_reason, current_buy_price, 
                            current_sell_price, close_time
        """
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trade_history.append(trade_data)

    def update_balance(self, exchange: str, amount: float):
        """
        Update exchange balance and recalculate totals.
        
        Args:
            exchange: Exchange name to update
            amount: Amount to add to available balance (can be negative)
            
        Updates total_balance and total_pnl automatically based on
        new balance calculations.
        
        Raises:
            KeyError: If exchange is not found in exchange_balances
        """
        if exchange in self.exchange_balances:
            self.exchange_balances[exchange]['available'] += amount
            self.total_balance = sum(bal['available'] + bal['used'] 
                                   for bal in self.exchange_balances.values())
            self.total_pnl = self.total_balance - self.initial_capital
