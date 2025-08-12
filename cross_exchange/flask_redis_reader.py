from flask import Flask, jsonify, request
import redis
import json
from flask_cors import CORS
import math

app = Flask(__name__)
# fix for status: CORS(Cross-Origin Resource Sharing) Error
CORS(app)

redis_client = redis.Redis(host='localhost', port = 6379,db=0,decode_responses=True)

@app.route('/health')
def health():
    try:
        redis_client.ping()
        return{"status": "healthy", "redis":"connected"}
    
    except:
        return {"status": "error", "redis":"disconnected"}, 500
    
@app.route('/exchange_data')
def get_exchange_data():
    """"""
    try:
        market_json = redis_client.get("trading:exchange_data")
        if market_json:
            return jsonify(json.loads(market_json))
        return {"error": "No market data available"}

    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/market-data')
def get_market_data():
    try:
        market_json = redis_client.get("trading:exchange_data")
        if market_json:
            return jsonify(json.loads(market_json))
        return {"error": "No market data available"}, 404
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.route('/api/balance')
def get_balance():
    try:
        balance_data = redis_client.get("trading:balance")
        if balance_data:
            return jsonify(json.loads(balance_data))
        
        metrics = redis_client.hgetall("trading:metrics")
        if metrics:
            return jsonify({
                "initial_capital": float(metrics.get('initial_capital', 0)),
                "total_balance": float(metrics.get('total_balance', 0)),
                "total_pnl": float(metrics.get('total_pnl', 0)),
                "roi_percentage": float(metrics.get('roi_percentage', 0)),
                "exchange_balances": {},
                "last_updated": metrics.get('last_updated', '')
            })
        return {"error": "No balance data available"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/current-position')
def get_current_position():
    """Get current active position - matches display_trade structure"""
    try:
        positions_data = redis_client.get("trading:current_position")
        if positions_data:
            return jsonify(json.loads(positions_data))
        return {"error": "No current position"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/trade-history')
def get_trade_history():
    """Get paginated trade history sorted by latest trade"""
    try:
        # Pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        history_data = redis_client.get("trading:formatted_history")
        if not history_data:
            return {"error": "No trade history available"}, 404

        history = json.loads(history_data)

        trade_pairs = history.get("trade_pairs", [])

        # Sort trades by close_time (fall back to open_time if missing)
        trade_pairs_sorted = sorted(
            trade_pairs,
            key=lambda t: t.get("close_time") or t.get("open_time", 0),
            reverse=True
        )

        total_items = len(trade_pairs_sorted)
        total_pages = math.ceil(total_items / per_page)

        start = (page - 1) * per_page
        end = start + per_page
        paginated_trades = trade_pairs_sorted[start:end]

        return jsonify({
            "last_updated": history.get("last_updated"),
            "summary": history.get("summary"),
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "trade_pairs": paginated_trades
        })

    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/latest-opportunity')
def get_latest_opportunity():
    """Get latest arbitrage opportunity with spread info"""
    try:
        opportunity_data = redis_client.get("trading:latest_opportunity")
        if opportunity_data:
            return jsonify(json.loads(opportunity_data))
        return {"error": "No opportunities available"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/metrics')
def get_metrics():
    """Get trading metrics and statistics"""
    try:
        metrics = redis_client.hgetall("trading:metrics")
        return jsonify(metrics)
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/opportunities')
def get_opportunities():
    """Get latest arbitrage opportunities"""
    try:
        opportunities = redis_client.get("trading:opportunities")
        if opportunities:
            return jsonify(json.loads(opportunities))
        return {"error": "No opportunities data available"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port = 5036)