import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

test_dir = './future_L2/test'  # 修正路径：从 quant_camp 根目录开始
pred_dir = './positions'  # 修
def generate_signals(df, window=3600):

    df = df.copy()
    
    # 计算动态区间
    df['rolling_high'] = df['HIGHPRICE'].shift(1).rolling(window).max()
    df['rolling_low'] = df['LOWPRICE'].shift(1).rolling(window).min()
    df['volume_ma'] = df['TRADEVOLUME'].rolling(window).mean()

    # 突破信号
    df['break_high'] = (df['LASTPRICE'] > df['rolling_high']) & (df['LASTPRICE'] > df['LASTPRICE'].shift(1))
    df['break_low'] = (df['LASTPRICE'] < df['rolling_low']) & (df['LASTPRICE'] < df['LASTPRICE'].shift(1))

    df['valid_break_high'] = df['break_high'] & (df['TRADEVOLUME'] > 1.2 * df['volume_ma'])
    df['valid_break_low'] = df['break_low'] & (df['TRADEVOLUME'] > 1.2 * df['volume_ma'])
    
    # 初始化持仓列
    df['position'] = 0

    # df['position'] = np.where(df['break_high'], 1, np.where(df['break_low'], -1, 0))
    
    current_position = 0
    entry_price = 0  

    for i in range(len(df)):
        # 空仓时检测突破
        if current_position == 0:
            if df['valid_break_high'].iloc[i]:
                current_position = 1
                entry_price = df['LASTPRICE'].iloc[i]
            elif df['valid_break_low'].iloc[i]:
                current_position = -1
                entry_price = df['LASTPRICE'].iloc[i]
        
        # 持多仓时检测平仓
        elif current_position == 1:
            if (df['LASTPRICE'].iloc[i] >= entry_price * 1.003) | \
               (df['LASTPRICE'].iloc[i] <= entry_price * 0.999):
                current_position = 0
                
        # 持空仓时检测平仓
        elif current_position == -1:
            if (df['LASTPRICE'].iloc[i] <= entry_price * 0.997) | \
               (df['LASTPRICE'].iloc[i] >= entry_price * 1.001):
                current_position = 0
        
        df['position'].iloc[i] = current_position
    
    
    ## 注意仓位的时间戳和主力合约的时间戳要对齐!
    return df['position']


def generate_signals_optimized(
    df: pd.DataFrame,

    window: int = 7200,                 # ≈ 60-minute rolling window for local highs/lows
    ema_period: int = 1080,             # trend filter (≈ 0.15×window ≈ 9 min)
    vol_multiplier: float = 1.8,        # breakout must occur on ≥1.8×平均成交量
    atr_period: int = 240,              # ATR look-back (≈ 0.033×window ≈ 2 min)
    stop_loss_mult: float = 1.0,        # SL = entry ±1×ATR
    take_profit_mult: float = 3.0,      # TP = entry ±3×ATR
    time_stop: int = 21600,             # flat exit after 3×window bars (≈ 3 hr)
    use_obi: bool = True,               # 默认启用盘口不均衡过滤
    obi_threshold: float = 0.15,        # |OBI|阈值

    bid_col: str = "BUYVOLUME01",
    ask_col: str = "SELLVOLUME01",
    high_col: str = "HIGHPRICE",
    low_col: str = "LOWPRICE",
    close_col: str = "LASTPRICE",
    vol_col: str = "TRADEVOLUME",
) -> pd.Series:
    """High‑frequency breakout strategy with trend/VWAP/OBI filters & ATR risk control.

    Returns
    -------
    pd.Series
        Position series aligned with *df* (1 = long, −1 = short, 0 = flat).
    """

    data = df.copy()

    # ─────────────────────────────
    # 0. VWAP  (全局成交量加权均价)
    cum_vol = data[vol_col].cumsum().replace(0, np.nan)
    data["vwap"] = (data[close_col] * data[vol_col]).cumsum() / cum_vol

    # 1. Rolling extrema & volume reference
    data["rolling_high"] = data[high_col].shift().rolling(window).max()
    data["rolling_low"]  = data[low_col].shift().rolling(window).min()
    data["vol_ma"]        = data[vol_col].rolling(window).mean()

    # 2. Trend filter (EMA)
    data["ema"] = data[close_col].ewm(span=ema_period, adjust=False).mean()

    # 3. ATR
    high, low, close = data[high_col], data[low_col], data[close_col]
    true_range = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    data["atr"] = true_range.rolling(atr_period).mean()

    # 4. Order‑book imbalance (optional)
    if use_obi:
        data["obi"] = (data[bid_col] - data[ask_col]) / (
            data[bid_col] + data[ask_col] + 1e-12
        )
    else:
        data["obi"] = 0.0  # dummy

    # 5. Breakout conditions
    cond_long = (
        (close > data["rolling_high"]) &
        (data[vol_col] > vol_multiplier * data["vol_ma"]) &
        (close > data["ema"]) &
        (close > data["vwap"]) &
        (~use_obi | (data["obi"] >  obi_threshold))
    )

    cond_short = (
        (close < data["rolling_low"]) &
        (data[vol_col] > vol_multiplier * data["vol_ma"]) &
        (close < data["ema"]) &
        (close < data["vwap"]) &
        (~use_obi | (data["obi"] < -obi_threshold))
    )

    data["break_high"], data["break_low"] = cond_long, cond_short

    # 6. Position management with trailing ATR stop
    pos = np.zeros(len(data), dtype=np.int8)
    current_pos, entry_price, entry_idx = 0, 0.0, -1
    peak, trough = 0.0, 0.0  # track extremes for trailing stop

    for i, (price, atr_val) in enumerate(zip(close.to_numpy(), data["atr"].to_numpy())):
        if not np.isfinite(price):
            pos[i] = current_pos
            continue

        if current_pos == 0:
            if data["break_high"].iat[i]:
                current_pos = 1
                entry_price, entry_idx, peak = price, i, price
            elif data["break_low"].iat[i]:
                current_pos = -1
                entry_price, entry_idx, trough = price, i, price

        elif current_pos == 1:
            peak = max(peak, price)
            sl = max(entry_price - stop_loss_mult * atr_val, peak - stop_loss_mult * atr_val)
            tp = entry_price + take_profit_mult * atr_val
            if price <= sl or price >= tp or i - entry_idx >= time_stop:
                current_pos = 0

        elif current_pos == -1:
            trough = min(trough, price)
            sl = min(entry_price + stop_loss_mult * atr_val, trough + stop_loss_mult * atr_val)
            tp = entry_price - take_profit_mult * atr_val
            if price >= sl or price <= tp or i - entry_idx >= time_stop:
                current_pos = 0

        pos[i] = current_pos

    return pd.Series(pos, index=data.index, name="position").ffill().fillna(0)


# Quick example usage (assuming *df* is your high‑frequency DataFrame):
# positions = generate_signals_optimized(df, use_obi=True)
# df["position"] = positions


def pred(date):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    Mfiles = os.listdir(f'{test_dir}/{date}')
    Mfiles = [f for f in Mfiles if '_M' in f]

    result_dict = {}

    for f in Mfiles:
        df = pd.read_parquet(f'{test_dir}/{date}/{f}')
        # result = generate_signals(df)
        result = generate_signals_optimized(df)
        result_dict[f.split('.')[0]] = result

    os.makedirs(f'{pred_dir}/{date}', exist_ok=True)
    for code, result in result_dict.items():
        result.to_csv(f'{pred_dir}/{date}/{code}.csv')





if __name__ == '__main__':
    test_dates = sorted(os.listdir(test_dir))[1:]
    os.makedirs(pred_dir, exist_ok=True)

    # for date in test_dates:
    #     print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    #     pred(date)

    from multiprocessing import Pool
    with Pool(20) as p:
        p.map(pred, test_dates)