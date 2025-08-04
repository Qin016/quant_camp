import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')


def generate_signals(df):
    df = df.copy()
    # 随机生成[0, 1， -1]之间的数, 取到0的概率是0.999, 取到1的概率是0.0005, 取到-1的概率是0.0005
    df['position'] = np.random.choice([0, 1, -1], size=len(df), p=[0.999, 0.0005, 0.0005])
    ## 注意仓位的时间戳和主力合约的时间戳要对齐!
    return df['position']




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
        result = generate_signals(df)
        result_dict[f.split('.')[0]] = result

    os.makedirs(f'{pred_dir}/{date}', exist_ok=True)
    for code, result in result_dict.items():
        result.to_csv(f'{pred_dir}/{date}/{code}.csv')



if __name__ == '__main__':

    test_dir = '../future_L2/test'
    pred_dir = './positions'
    test_dates = sorted(os.listdir(test_dir))[1:]
    os.makedirs(pred_dir, exist_ok=True)

    # for date in test_dates:
    #     print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} start to pred {date}')
    #     pred(date)

    from multiprocessing import Pool
    with Pool(20) as p:
        p.map(pred, test_dates)
