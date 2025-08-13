from shared_imports import Config,math
from typing import Dict, Any

def calculate_available_liquidity(orderbook: Dict, target_amount: float, side: str = 'buy') -> Dict[str, Any]:
    """
    计算指定数量下的可用流动性
    
    Returns:
    {
        'available_quantity': float,      # 可用数量
        'average_price': float,           # 加权平均价格
        'is_sufficient': bool,            # 是否充足
        'depth_levels': int,              # 使用的价格档位数
        'total_value': float              # 总价值 USDT
    }
    """
    available_quantity = 0.0
    total_value = 0.0
    if side == 'buy':
        orderbook_side = 'asks'
    else:
        orderbook_side = 'bids'
    depth = 0

    if orderbook_side not in orderbook or not orderbook[orderbook_side]:
        return {
            'available_quantity': 0,
            'average_price': 0,
            'is_sufficient': False,
            'depth_levels': 0,
            'total_value': 0
        }
    
    while available_quantity < target_amount and depth < min(Config.MAX_ORDERBOOK_DEPTH, len(orderbook[orderbook_side])):
        price , quantity = orderbook[orderbook_side][depth]

        remaining_needed = target_amount - available_quantity
        quantity_to_take = min(quantity,remaining_needed)
        available_quantity += quantity_to_take
        total_value += price * quantity_to_take
        depth += 1

    average_price = total_value / available_quantity if available_quantity > 0.0 else 0.0

    return {'available_quantity': available_quantity,
            'average_price': average_price,
            'is_sufficient': available_quantity >= target_amount,
            'depth_levels': depth,
            'total_value': total_value}

def estimate_slippage_cost(orderbook: Dict, trade_amount: float, side: str = 'buy') -> Dict[str, float]:
    """
    估算滑点成本
    假设一定够？
    Returns:
    {
        'absolute_slippage': float,       # 绝对滑点 USDT，绝对值
        'percentage_slippage': float,     # 百分比滑点,绝对值
        'effective_price': float,         # 有效成交价
        'best_price': float,              # 最佳价格
        'price_impact': float             # 价格影响度，绝对值
    }
    """
    if side == 'buy':
        orderbook_side = 'asks'
    else:
        orderbook_side = 'bids'
    available_quantity = 0.0
    total_value = 0.0
    depth = 0

    if orderbook_side not in orderbook or not orderbook[orderbook_side]:
        return {
            'absolute_slippage': 0.0,
            'percentage_slippage':  0.0,
            'best_price': 0.0,
            'price_impact': 0.0,
            'effective_price': 0.0
        }
    
    best_price = orderbook[orderbook_side][0][0]

    while available_quantity < trade_amount and depth < min(Config.MAX_ORDERBOOK_DEPTH, len(orderbook[orderbook_side])):
        price , quantity = orderbook[orderbook_side][depth]

        remaining_needed = trade_amount - available_quantity
        quantity_to_take = min(quantity,remaining_needed)
        available_quantity += quantity_to_take
        total_value += price * quantity_to_take
        depth += 1
    
    effective_price = total_value / available_quantity if available_quantity > 0.0 else 0.0
    effective_price = abs(effective_price)
    slippage_abs = abs((effective_price - best_price) * available_quantity)
    slippage_pct = slippage_abs / (best_price * available_quantity)
    price_impact = abs(effective_price - best_price) / best_price

    return {
        'absolute_slippage': slippage_abs,
        'percentage_slippage':  slippage_pct,
        'best_price': best_price,
        'price_impact': price_impact,
        'effective_price': effective_price,
    }

def calculate_effective_price(orderbook: Dict, quantity: float, side: str = 'buy') -> Dict[str, Any]:
    """
    计算实际成交的有效价格
    
    Returns:
    {
        'effective_price': float,         # 加权平均价格
        'can_fill_completely': bool,      # 是否能完全成交
        'filled_quantity': float,         # 实际能成交数量
        'unfilled_quantity': float,       # 未成交数量
        'execution_breakdown': List[Tuple[float, float]]  # [(price, qty), ...]
    }
    """
    if side == 'buy':
        orderbook_side = 'asks'
    else:
        orderbook_side = 'bids'
    
    filled_quantity = 0.0
    total_cost = 0.0
    remaining_quantity = quantity
    execution_breakdown = []
    can_fill_completely = True
    depth = 0

    if orderbook_side not in orderbook or not orderbook[orderbook_side]:
        return {
            'effective_price': 0.0,
            'can_fill_completely': False,
            'filled_quantity': 0.0,
            'unfilled_quantity': 0.0,
            'execution_breakdown': []
        }

    while remaining_quantity > 0 and depth < min(Config.MAX_ORDERBOOK_DEPTH, len(orderbook[orderbook_side])):
        price , available_quantity = orderbook[orderbook_side][depth]
        quantity_to_fill = min(remaining_quantity,available_quantity)
        total_cost += quantity_to_fill * price
        filled_quantity += quantity_to_fill
        remaining_quantity -= quantity_to_fill
        depth += 1
        execution_breakdown.append((price,quantity_to_fill))
    
    effective_price = total_cost / filled_quantity if filled_quantity > 0.0 else 0.0
    if remaining_quantity > 0:

        return {
            'effective_price': effective_price,
            'can_fill_completely': False,
            'filled_quantity': quantity-remaining_quantity,
            'unfilled_quantity': remaining_quantity,
            'execution_breakdown': execution_breakdown
        }
    
    else:
        return {
            'effective_price': effective_price,
            'can_fill_completely': True,
            'filled_quantity': filled_quantity,
            'unfilled_quantity': remaining_quantity,
            'execution_breakdown': execution_breakdown
        }

def evaluate_liquidity(orderbook: Dict, target_amount: float, side: str = 'buy', max_slippage_pct: float = 0.001) -> Dict[str, Any]:
    """
    综合评估流动性 - 整合所有流动性分析功能
    
    Args:
        orderbook: 订单簿数据
        target_amount: 目标交易数量
        side: 交易方向 'buy' 或 'sell'
        max_slippage_pct: 最大允许滑点百分比，默认0.1% (0.001)
    
    Returns:
    {
        'available_quantity': float,      # 可用数量
        'effective_price': float,         # 有效成交价格
        'is_sufficient': bool,            # 是否充足
        'depth_levels': int,              # 使用的价格档位数
        'total_value': float,             # 总价值 USDT
        'absolute_slippage': float,       # 绝对滑点 USDT
        'percentage_slippage': float,     # 百分比滑点
        'execution_breakdown': List[Tuple[float, float]],  # [(price, qty), ...]
        'stopped_by_slippage': bool,      # 是否因滑点限制而停止
        'best_price': float,              # 最佳价格
        'price_impact': float,            # 价格影响度
        'can_fill_completely': bool       # 是否能完全成交
    }
    """
    if side == 'buy':
        orderbook_side = 'asks'
    else:
        orderbook_side = 'bids'
    
    filled_quantity = 0.0
    total_value = 0.0
    execution_breakdown = []
    depth = 0
    stopped_by_slippage = False

    # 处理空订单簿情况
    if orderbook_side not in orderbook or not orderbook[orderbook_side]:
        return {
            'available_quantity': 0.0,
            'effective_price': 0.0,
            'is_sufficient': False,
            'depth_levels': 0,
            'total_value': 0.0,
            'absolute_slippage': 0.0,
            'percentage_slippage': 0.0,
            'execution_breakdown': [],
            'stopped_by_slippage': False,
            'best_price': 0.0,
            'price_impact': 0.0,
            'can_fill_completely': False
        }

    best_price = orderbook[orderbook_side][0][0]
    remaining_quantity = target_amount

    # 主循环：模拟订单执行
    while remaining_quantity > 0 and depth < min(Config.MAX_ORDERBOOK_DEPTH, len(orderbook[orderbook_side])):
        price, available_quantity = orderbook[orderbook_side][depth]
        
        # 检查滑点限制
        if side == 'buy':
            current_slippage_pct = (price - best_price) / best_price
        else:
            current_slippage_pct = (best_price - price) / best_price
            
        if current_slippage_pct > max_slippage_pct:
            stopped_by_slippage = True
            break
            
        quantity_to_fill = min(remaining_quantity, available_quantity)
        total_value += quantity_to_fill * price
        filled_quantity += quantity_to_fill
        remaining_quantity -= quantity_to_fill
        execution_breakdown.append((price, quantity_to_fill))
        depth += 1

    # 计算各种指标
    effective_price = total_value / filled_quantity if filled_quantity > 0.0 else 0.0
    is_sufficient = filled_quantity >= target_amount
    can_fill_completely = remaining_quantity == 0 and not stopped_by_slippage
    
    # 计算滑点指标
    absolute_slippage = 0.0
    percentage_slippage = 0.0
    price_impact = 0.0
    
    if filled_quantity > 0.0 and best_price > 0.0:
        absolute_slippage = abs(effective_price - best_price) * filled_quantity
        percentage_slippage = absolute_slippage / (best_price * filled_quantity)
        price_impact = abs(effective_price - best_price) / best_price

    return {
        'available_quantity': filled_quantity,
        'effective_price': effective_price,
        'is_sufficient': is_sufficient,
        'depth_levels': depth,
        'total_value': total_value,
        'absolute_slippage': absolute_slippage,
        'percentage_slippage': percentage_slippage,
        'execution_breakdown': execution_breakdown,
        'stopped_by_slippage': stopped_by_slippage,
        'best_price': best_price,
        'price_impact': price_impact,
        'can_fill_completely': can_fill_completely
    }

def create_test_orderbook():
    """创建测试用的订单簿数据"""
    return {
        'bids': [  # 买盘 - 价格从高到低排序
            (50000.0, 0.5),   # 价格, 数量
            (49990.0, 1.2),
            (49980.0, 0.8),
            (49970.0, 2.0),
            (49960.0, 1.5),
            (49950.0, 3.0),
            (49940.0, 0.7),
            (49930.0, 1.8),
            (49920.0, 2.2),
            (49910.0, 4.0)
        ],
        'asks': [  # 卖盘 - 价格从低到高排序
            (50010.0, 0.3),
            (50020.0, 1.5),
            (50030.0, 0.9),
            (50040.0, 2.1),
            (50050.0, 1.0),
            (50060.0, 2.5),
            (50070.0, 1.7),
            (50080.0, 3.2),
            (50090.0, 1.1),
            (50100.0, 5.0)
        ]
    }

def create_shallow_orderbook():
    """创建流动性较浅的订单簿"""
    return {
        'bids': [
            (50000.0, 0.1),
            (49990.0, 0.2),
            (49980.0, 0.15)
        ],
        'asks': [
            (50010.0, 0.05),
            (50020.0, 0.1),
            (50030.0, 0.08)
        ]
    }

def create_empty_orderbook():
    """创建空的订单簿"""
    return {
        'bids': [],
        'asks': []
    }

def create_high_slippage_orderbook():
    """创建高滑点测试用的订单簿数据"""
    return {
        'bids': [  # 买盘 - 价格从高到低排序，每档价格差异较大
            (50000.0, 0.5),   # 价格, 数量
            (49900.0, 1.2),   # 0.2% 滑点
            (49800.0, 0.8),   # 0.4% 滑点
            (49700.0, 2.0),   # 0.6% 滑点
            (49600.0, 1.5),   # 0.8% 滑点
            (49500.0, 3.0),   # 1.0% 滑点
            (49400.0, 0.7),   # 1.2% 滑点
        ],
        'asks': [  # 卖盘 - 价格从低到高排序，每档价格差异较大
            (50000.0, 0.3),
            (50100.0, 1.5),   # 0.2% 滑点
            (50200.0, 0.9),   # 0.4% 滑点
            (50300.0, 2.1),   # 0.6% 滑点
            (50400.0, 1.0),   # 0.8% 滑点
            (50500.0, 2.5),   # 1.0% 滑点
            (50600.0, 1.7),   # 1.2% 滑点
        ]
    }


def assert_almost_equal(a, b, places=4):
    """自定义断言，用于浮点数比较"""
    assert math.isclose(a, b, rel_tol=1e-05, abs_tol=1e-08), f"{a} is not approximately equal to {b}"

def assert_liquidity_result(result, expected):
    for key, value in expected.items():
        if isinstance(value, float):
            assert_almost_equal(result[key], value)
        else:
            assert result[key] == value, f"{key} 期望 {value}，实际 {result[key]}"

# ================================================================
# 运行测试
# ================================================================

def run_tests():
    """运行各种测试用例"""
    
    print("=== 流动性计算函数测试 ===\n")
    
    # 测试1：正常订单簿，买入小数量
    print("测试1：正常流动性，买入0.5 BTC")
    orderbook = create_test_orderbook()
    result = calculate_available_liquidity(orderbook, 0.5, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['average_price'], 50014.0)
    assert result['is_sufficient'] == True
    assert result['depth_levels'] == 2
    print("测试通过\n")
    
    # 测试2：正常订单簿，买入大数量
    print("测试2：正常流动性，买入5.0 BTC")
    result = calculate_available_liquidity(orderbook, 5.0, 'buy')
    print(f"结果: {result}")
    answer = 50010.0 * 0.3 + 50020.0 * 1.5 + 50030.0 * 0.9 + 50040.0 * 2.1 + 50050.0 * 0.2
    answer = answer / 5.0
    assert_almost_equal(result['average_price'], answer)
    assert result['is_sufficient'] == True
    assert result['depth_levels'] == 5
    print("测试通过\n")
    
    # 测试3：卖出操作
    print("测试3：正常流动性，卖出1.0 BTC")
    result = calculate_available_liquidity(orderbook, 1.0, 'sell')
    print(f"结果: {result}")
    answer = 50000.0 * 0.5 + 49990.0 * 0.5
    answer = answer / 1.0
    assert_almost_equal(result['average_price'], answer)
    assert result['is_sufficient'] == True
    assert result['depth_levels'] == 2
    print("测试通过\n")
    
    # 测试4：流动性不足
    print("测试4：浅流动性，买入1.0 BTC")
    shallow_orderbook = create_shallow_orderbook()
    result = calculate_available_liquidity(shallow_orderbook, 1.0, 'buy')
    print(f"结果: {result}")
    answer = 50010.0 * 0.1 + 50020.0 * 0.2 + 50030.0 * 0.15
    answer = answer / (0.1 + 0.2 + 0.15)
    assert_almost_equal(result['average_price'], answer)
    assert result['is_sufficient'] == False
    assert_almost_equal(result['available_quantity'], 0.23)
    print("测试通过\n")
    
    # 测试5：空订单簿
    print("测试5：空订单簿，买入0.1 BTC")
    empty_orderbook = create_empty_orderbook()
    result = calculate_available_liquidity(empty_orderbook, 0.1, 'buy')
    print(f"结果: {result}")
    assert result['is_sufficient'] == False
    print("测试通过\n")
    
    # 测试6：精确匹配
    print("测试6：精确匹配第一档数量，买入0.3 BTC")
    result = calculate_available_liquidity(orderbook, 0.3, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['average_price'], 50010.0)
    assert result['is_sufficient'] == True
    assert result['depth_levels'] == 1
    print("测试通过\n")

    print("\n\n=== 有效价格计算函数测试 ===\n")
    
    # 测试7：买入小数量
    print("测试7：买入0.5 BTC，正常订单簿")
    result = calculate_effective_price(orderbook, 0.5, 'buy')
    print(f"结果: {result}")
    assert result['can_fill_completely'] == True
    answer = (50010.0 * 0.3 + 50020.0*0.2)/0.5
    assert_almost_equal(result['effective_price'], answer)
    assert_almost_equal(result['filled_quantity'], 0.5)
    assert result['unfilled_quantity'] == 0.0
    print("测试通过\n")

    # 测试8：买入大数量，无法完全成交
    print("测试8：买入100 BTC，正常订单簿 (无法完全成交)")
    result = calculate_effective_price(orderbook, 100.0, 'buy')
    print(f"结果: {result}")
    assert result['can_fill_completely'] == False
    total = 0.0
    total_q = 0.0
    for p,q in orderbook["asks"]:
        total += p*q
        total_q += q
    answer = total / total_q
    
    assert_almost_equal(result['effective_price'], answer)
    assert_almost_equal(result['filled_quantity'], total_q)
    assert_almost_equal(result['unfilled_quantity'], 100.0-total_q)
    print("测试通过\n")
    
    # 测试9：卖出操作
    print("测试9：卖出1.0 BTC，正常订单簿")
    result = calculate_effective_price(orderbook, 1.0, 'sell')
    print(f"结果: {result}")
    assert result['can_fill_completely'] == True
    answer = 50000.0* 0.5 + 49990.0* 0.5
    assert_almost_equal(result['effective_price'], answer)
    print("测试通过\n")
    
    # 测试10：空订单簿
    print("测试10：买入0.1 BTC，空订单簿")
    result = calculate_effective_price(empty_orderbook, 0.1, 'buy')
    print(f"结果: {result}")
    assert result['can_fill_completely'] == False
    assert_almost_equal(result['filled_quantity'], 0.0)
    print("测试通过\n")

    print("\n\n=== 滑点成本计算函数测试 ===\n")

    # 测试11：买入小数量，滑点小
    print("测试11：买入0.5 BTC")
    result = estimate_slippage_cost(orderbook, 0.5, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['best_price'], 50010.0)
    answer = (50014.0-50010.0)*0.5
    assert_almost_equal(result['absolute_slippage'], answer)
    answer = answer / (50010.0* 0.5)
    assert_almost_equal(result['percentage_slippage'], answer)
    print("测试通过\n")
    
    # 测试12：买入大数量，滑点大
    print("测试12：买入5.0 BTC")
    result = estimate_slippage_cost(orderbook, 5.0, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['best_price'], 50010.0)
    answer = 50010.0 * 0.3 + 50020.0 * 1.5 + 50030.0 * 0.9 + 50040.0 * 2.1 + 50050.0 * 0.2
    answer = answer / 5.0
    assert_almost_equal(result['effective_price'], answer)

    answer = (answer-50010.0) * 5.0
    assert_almost_equal(result['absolute_slippage'], answer)
    answer = answer/ (50010.0 * 5.0)
    assert_almost_equal(result['percentage_slippage'], answer)
    print("测试通过\n")
    
    # 测试13：卖出操作
    print("测试13：卖出1.0 BTC")
    result = estimate_slippage_cost(orderbook, 1.0, 'sell')
    print(f"结果: {result}")
    assert_almost_equal(result['best_price'], 50000.0)
    answer = (50000.0 * 0.5 + 49990.0 * 0.5) / 1.0
    assert_almost_equal(result['effective_price'], answer)
    answer = abs(answer - 50000.0) * 1.0
    assert_almost_equal(result['absolute_slippage'], answer)
    answer = answer / (50000.0 * 1.0)
    assert_almost_equal(result['percentage_slippage'], answer)
    print("测试通过\n")

    # 测试14：浅流动性订单簿
    print("测试14：买入1.0 BTC，浅流动性")
    shallow_orderbook = create_shallow_orderbook()
    result = estimate_slippage_cost(shallow_orderbook, 1.0, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['best_price'], 50010.0)
    answer = (50010.0 * 0.05 + 50020.0 * 0.1 + 50030.0 * 0.08) / (0.05+0.1+0.08)
    assert_almost_equal(result['effective_price'], answer)
    answer = abs(answer - 50010.0) * (0.05+0.1+0.08)
    assert_almost_equal(result['absolute_slippage'], answer)
    answer = answer / (50010.0*(0.05+0.1+0.08))
    assert_almost_equal(result['percentage_slippage'], answer)
    print("测试通过\n")

    # 测试15：空订单簿
    print("测试15：买入0.1 BTC，空订单簿")
    empty_orderbook = create_empty_orderbook()
    result = estimate_slippage_cost(empty_orderbook, 0.1, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['absolute_slippage'], 0.0)
    assert_almost_equal(result['percentage_slippage'], 0.0)
    print("测试通过\n")

    print("\n\n=== 综合流动性评估函数测试 ===\n")

    # 测试16：正常流动性，买入1.0 BTC
    print("测试16：正常订单簿，买入1.0 BTC，无滑点限制")
    result = evaluate_liquidity(orderbook, 1.0, 'buy', max_slippage_pct=1.0)
    print(f"结果: {result}")
    answer_price = (50010.0 * 0.3 + 50020.0 * 0.7) / 1.0
    assert_almost_equal(result['effective_price'], answer_price)
    assert result['is_sufficient'] == True
    assert result['stopped_by_slippage'] == False
    assert result['can_fill_completely'] == True
    print("测试通过\n")

    # 测试17：正常流动性，滑点限制 0.001 导致提前停止
    print("测试17：正常订单簿，买入2.0 BTC，滑点限制 0.001")
    result = evaluate_liquidity(orderbook, 2.0, 'buy', max_slippage_pct=0.001)
    print(f"结果: {result}")
    # 50030 相比 50010 的涨幅 = 0.0004 < 0.001，所以会成交到 50030 档
    assert result['stopped_by_slippage'] == False  # 不会触发滑点限制
    # 再试更小的滑点限制
    result_slip = evaluate_liquidity(orderbook, 2.0, 'buy', max_slippage_pct=0.0001)
    print(f"结果(滑点限制0.0001): {result_slip}")
    assert result_slip['stopped_by_slippage'] == True
    assert result_slip['available_quantity'] < 2.0
    print("测试通过\n")

    # 测试18：浅流动性，买入1.0 BTC
    print("测试18：浅流动性订单簿，买入1.0 BTC")
    result = evaluate_liquidity(shallow_orderbook, 1.0, 'buy', max_slippage_pct=1.0)
    print(f"结果: {result}")
    total_q = 0.05 + 0.1 + 0.08
    answer_price = (50010.0 * 0.05 + 50020.0 * 0.1 + 50030.0 * 0.08) / total_q
    assert_almost_equal(result['effective_price'], answer_price)
    assert result['is_sufficient'] == False
    assert_almost_equal(result['available_quantity'], total_q)
    print("测试通过\n")

    # 测试19：空订单簿
    print("测试19：空订单簿，买入0.5 BTC")
    result = evaluate_liquidity(empty_orderbook, 0.5, 'buy')
    print(f"结果: {result}")
    assert result['is_sufficient'] == False
    assert result['available_quantity'] == 0.0
    assert result['stopped_by_slippage'] == False
    print("测试通过\n")

    # 测试20：卖出方向
    print("测试20：正常订单簿，卖出1.0 BTC")
    result = evaluate_liquidity(orderbook, 1.0, 'sell', max_slippage_pct=1.0)
    print(f"结果: {result}")
    answer_price = (50000.0 * 0.5 + 49990.0 * 0.5) / 1.0
    assert_almost_equal(result['effective_price'], answer_price)
    assert result['can_fill_completely'] == True
    assert result['stopped_by_slippage'] == False
    print("测试通过\n")

    # 测试21：精确匹配第一档
    print("测试21：精确匹配第一档，买入0.3 BTC")
    result = evaluate_liquidity(orderbook, 0.3, 'buy')
    print(f"结果: {result}")
    assert_almost_equal(result['effective_price'], 50010.0)
    assert result['can_fill_completely'] == True
    assert result['depth_levels'] == 1
    print("测试通过\n")

    # 测试22：高滑点订单簿，滑点限制触发
    print("测试22：高滑点订单簿，买入2.0 BTC，滑点限制0.003 (0.3%)")
    high_slip_orderbook = create_high_slippage_orderbook()
    result = evaluate_liquidity(high_slip_orderbook, 2.0, 'buy', max_slippage_pct=0.003)
    print(f"结果: {result}")
    # 50100 相比 50000 = 0.2% < 0.3%，所以第一档和第二档可以成交
    # 50200 相比 50000 = 0.4% > 0.3%，应当停止
    assert result['stopped_by_slippage'] == True
    assert result['available_quantity'] == 0.3 + 1.5  # 只成交前两档
    assert_almost_equal(result['effective_price'], (50000.0 * 0.3 + 50100.0 * 1.5) / 1.8)
    print("测试通过\n")

    # 测试23：高滑点订单簿，滑点限制放宽
    print("测试23：高滑点订单簿，买入2.0 BTC，滑点限制0.005 (0.5%)")
    result = evaluate_liquidity(high_slip_orderbook, 2.0, 'buy', max_slippage_pct=0.005)
    print(f"结果: {result}")
    # 50200 相比 50000 = 0.4% < 0.5%，前三档可成交
    # 50300 相比 50000 = 0.6% > 0.5%，应当停止
    assert result['stopped_by_slippage'] == False
    filled_qty = 0.3 + 1.5 + 0.2  # 第三档只能成交 0.2 BTC 来补够 2.0
    assert_almost_equal(result['available_quantity'], 2.0)
    answer_price = (50000.0 * 0.3 + 50100.0 * 1.5 + 50200.0 * 0.2) / 2.0
    assert_almost_equal(result['effective_price'], answer_price)
    print("测试通过\n")

    # 测试24：高滑点订单簿，卖出2.0 BTC，滑点限制0.003 (0.3%)
    print("测试24：高滑点订单簿，卖出2.0 BTC，滑点限制0.003 (0.3%)")
    result = evaluate_liquidity(high_slip_orderbook, 2.0, 'sell', max_slippage_pct=0.003)
    print(f"结果: {result}")
    # 49900 相比 50000 = 0.2% < 0.3%，第一档和第二档可以成交
    # 49800 相比 50000 = 0.4% > 0.3%，应当停止
    assert result['stopped_by_slippage'] == True
    assert result['available_quantity'] == 0.5 + 1.2  # 只成交前两档
    answer_price = (50000.0 * 0.5 + 49900.0 * 1.2) / 1.7
    assert_almost_equal(result['effective_price'], answer_price)
    print("测试通过\n")

    # 测试25：高滑点订单簿，卖出2.0 BTC，滑点限制0.005 (0.5%)
    print("测试25：高滑点订单簿，卖出2.0 BTC，滑点限制0.005 (0.5%)")
    result = evaluate_liquidity(high_slip_orderbook, 2.0, 'sell', max_slippage_pct=0.005)
    print(f"结果: {result}")
    # 49800 相比 50000 = 0.4% < 0.5%，前三档可成交
    # 49700 相比 50000 = 0.6% > 0.5%，应当停止
    assert result['stopped_by_slippage'] == False
    filled_qty = 0.5 + 1.2 + 0.3  # 第三档只能成交 0.3 BTC
    assert_almost_equal(result['available_quantity'], 2.0)
    answer_price = (50000.0 * 0.5 + 49900.0 * 1.2 + 49800.0 * 0.3) / 2.0
    assert_almost_equal(result['effective_price'], answer_price)
    print("测试通过\n")

# 运行所有测试
if __name__ == "__main__":
    run_tests()