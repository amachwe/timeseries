
from matplotlib import pyplot as plt
import pandas as pd

from ts_fun import build_lag_data, corrcof, delta, gen_ols, lag_ols, linefit,evaluate_ts, train_test_split, predict, sample


import data_download as dd

def update(sym="XRP-USD"):
    d1,d2 = dd.to_csv(sym)

    d1.to_csv(f"data/{sym}.csv")
    d2.to_csv(f"data/{sym}-price.csv")
    print(f"{sym} files written.")

def load_from_file(sym="XRP-USD"):
    price = pd.read_csv(f"data/{sym}-price.csv",index_col="time",parse_dates=True)
    daily = pd.read_csv(f"data/{sym}.csv",index_col="time",parse_dates=True)

    return price,daily


def build_model():
    pass



if __name__ == "__main__":

    # update(sym="XRP-USD")
    # update(sym="BTC-USD")
    # update(sym="ETH-USD")
    # update(sym="BCH-USD")
    
    price, daily = load_from_file()
    
    



    