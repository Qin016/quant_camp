import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')


def generate_signals_advanced(
    df: pd.DataFrame,
    window: int = 7200,              # rolling high/low window for breakout
    ema_period: int = 1800,          # EMA trend filter
    vol_ma_period: int = 900,        # volume moving average period
    atr_period: int = 240,           # ATR period for risk sizing
    vol_multiplier: float = 1.8,     # volume must exceed vol_multiplier * vol_ma to qualify
    obi_threshold: float = 0.12,     # order-book imbalance threshold
    stop_atr_mult: float = 1.0,      # initial stop = entry ± stop_atr_mult * ATR
    trail_atr_mult: float = 1.0,     # trailing stop uses ATR * trail_atr_mult
    tp_atr_mult: float = 3.0,        # take profit = entry ± tp_atr_mult * ATR
    time_stop: int = 21600,          # max bars to hold a trade (safety)
    cooldown_bars: int = 60,         # wait bars after exit before re-entering same side
    high_col: str = "HIGHPRICE",
    low_col: str = "LOWPRICE",
    close_col: str = "LASTPRICE",
    vol_col: str = "TRADEVOLUME",
    bid_col: str = "BUYVOLUME01",
    ask_col: str = "SELLVOLUME01",
) -> pd.Series:
    """
    Advanced hybrid strategy:
      - breakout vs rolling highs/lows
      - trend filter (EMA) + VWAP trend confirmation
      - volume spike confirmation
      - order-book imbalance (OBI) filter
      - ATR-based SL/TP and trailing stop
      - cooldown after exit to avoid whipsaw
    Returns position series aligned with df (1 long, -1 short, 0 flat)
    """

    data = df.copy()

    # Basic safety: ensure required columns exist; fill missing OB columns with 0
    for col in [high_col, low_col, close_col, vol_col]:
        if col not in data.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame")

    if bid_col not in data.columns:
        data[bid_col] = 0.0
    if ask_col not in data.columns:
        data[ask_col] = 0.0

    # precompute numpy arrays for speed
    close = data[close_col].astype(float)
    high = data[high_col].astype(float)
    low = data[low_col].astype(float)
    vol = data[vol_col].astype(float)
    bid = data[bid_col].astype(float)
    ask = data[ask_col].astype(float)

    # 0) VWAP (global cumulative) - used as additional trend confirmation
    cum_vol = vol.cumsum()
    with np.errstate(invalid='ignore', divide='ignore'):
        vwap = (close * vol).cumsum() / (cum_vol.replace(0, np.nan))
    data["vwap"] = vwap

    # 1) rolling extrema & volume reference
    data["rolling_high"] = high.shift(1).rolling(window=min(window, len(data))).max()
    data["rolling_low"]  = low.shift(1).rolling(window=min(window, len(data))).min()
    data["vol_ma"] = vol.rolling(window=min(vol_ma_period, len(data))).mean().fillna(method="ffill").fillna(0.0)

    # 2) EMA trend filter
    data["ema"] = close.ewm(span=max(2, ema_period), adjust=False).mean()

    # 3) ATR (simple rolling TR mean)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = tr.rolling(window=max(2, atr_period)).mean()

    # 4) Order Book Imbalance (OBI)
    with np.errstate(divide='ignore', invalid='ignore'):
        data["obi"] = (bid - ask) / (bid + ask + 1e-12)
    data["obi"] = data["obi"].fillna(0.0)

    # 5) Breakout candidate booleans
    cond_break_long = (close > data["rolling_high"]) & (vol > vol_multiplier * data["vol_ma"])
    cond_break_short = (close < data["rolling_low"]) & (vol > vol_multiplier * data["vol_ma"])

    # 6) Final entry filters (trend + vwap + obi)
    long_qual = cond_break_long & (close > data["ema"]) & (close > data["vwap"]) & (data["obi"] > -obi_threshold)
    short_qual = cond_break_short & (close < data["ema"]) & (close < data["vwap"]) & (data["obi"] < obi_threshold)

    # convert booleans to numpy arrays
    long_qual_arr = long_qual.to_numpy(dtype=bool)
    short_qual_arr = short_qual.to_numpy(dtype=bool)
    atr_arr = data["atr"].to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)

    n = len(data)
    pos = np.zeros(n, dtype=np.int8)
    current_pos = 0
    entry_price = 0.0
    entry_idx = -1
    peak = -np.inf
    trough = np.inf
    cooldown = 0  # positive for long-cooldown, negative for short-cooldown; magnitude = remaining bars

    for i in range(n):
        price = close_arr[i]
        atr_val = atr_arr[i]

        # if price/atr not finite, keep previous position
        if not np.isfinite(price) or not np.isfinite(atr_val):
            pos[i] = current_pos
            # decrement cooldown
            if cooldown > 0:
                cooldown = max(0, cooldown - 1)
            elif cooldown < 0:
                cooldown = min(0, cooldown + 1)
            continue

        # reduce cooldown counter each bar
        if cooldown > 0:
            cooldown -= 1
        elif cooldown < 0:
            cooldown += 1

        # ENTRY: only if currently flat and not in cooldown (cooldown prevents immediate re-entry same side)
        if current_pos == 0:
            # Long entry
            if long_qual_arr[i] and cooldown >= 0:
                # Confirm ATR is reasonable (avoid too small atr)
                if atr_val > 0 and price > 0:
                    current_pos = 1
                    entry_price = price
                    entry_idx = i
                    peak = price
                    # set initial cooldown to 0 (only set on exit)
            # Short entry
            elif short_qual_arr[i] and cooldown <= 0:
                if atr_val > 0 and price > 0:
                    current_pos = -1
                    entry_price = price
                    entry_idx = i
                    trough = price

        # MANAGE LONG
        elif current_pos == 1:
            # update peak
            peak = max(peak, price)
            # dynamic stop: tighter of (entry - stop_atr_mult*atr) and (peak - trail_atr_mult*atr)
            sl_price = max(entry_price - stop_atr_mult * atr_val, peak - trail_atr_mult * atr_val)
            tp_price = entry_price + tp_atr_mult * atr_val

            # safety: if price hits tp or below sl -> exit
            if price >= tp_price or price <= sl_price or (i - entry_idx) >= time_stop:
                current_pos = 0
                # set cooldown to avoid immediate same-side re-entry
                cooldown = cooldown_bars

        # MANAGE SHORT
        elif current_pos == -1:
            trough = min(trough, price)
            sl_price = min(entry_price + stop_atr_mult * atr_val, trough + trail_atr_mult * atr_val)
            tp_price = entry_price - tp_atr_mult * atr_val

            if price <= tp_price or price >= sl_price or (i - entry_idx) >= time_stop:
                current_pos = 0
                cooldown = -cooldown_bars

        pos[i] = current_pos

    # return series aligned with original index
    return pd.Series(pos, index=data.index, name="position").ffill().fillna(0)


def pred(date):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    Mfiles = os.listdir(f'{test_dir}/{date}')
    Mfiles = [f for f in Mfiles if '_M' in f]

    result_dict = {}

    for f in Mfiles:
        df = pd.read_parquet(f'{test_dir}/{date}/{f}')
        result = generate_signals_advanced(df)
        result_dict[f.split('.')[0]] = result

    os.makedirs(f'{pred_dir}/{date}', exist_ok=True)
    for code, result in result_dict.items():
        result.to_csv(f'{pred_dir}/{date}/{code}.csv')


if __name__ == '__main__':

    test_dir = '../future_L2/test'
    pred_dir = './positions'
    test_dates = sorted(os.listdir(test_dir))[1:]
    os.makedirs(pred_dir, exist_ok=True)

    from multiprocessing import Pool
    with Pool(20) as p:
        p.map(pred, test_dates)
