import tslibs.data_client as dc
import ts_fun as ts
import data_download as dd
import pandas as pd
from matplotlib import pyplot as plt

def update():
    d1,d2 = dd.to_csv("XRP-USD")

    d1.to_csv("data/XRP-USD.csv")
    d2.to_csv("data/XRP-USD-price.csv")

def get_seq(col):
    df = pd.read_csv("data/XRP-USD.csv")
    return df[col]
    
def strategy(seq):
    HORIZON = 30
    ser = seq.rolling(HORIZON).mean()
    print(ser)
    print(seq)
    val = [0 for i in range(0,HORIZON+1)]
    buy_sell = [0 for i in range(0,HORIZON+1)]
    for i in range(HORIZON+1,len(seq)):
        val.append(seq[i])
        if ser[i-HORIZON-1] -ser[i-HORIZON] < 0:
            buy_sell.append(0.5)
        else:
            buy_sell.append(-0.5)
            
    
    plt.plot(ser)
    plt.plot(buy_sell)
    plt.plot(val)
    plt.show()


    

if __name__ == "__main__":

    # update()
    # strategy(get_seq("low"))

    xrp = ts.Engine("data/XRP-USD.csv")
    print(xrp.get_max_seek_index())
    cost = xrp.set_seek_index(100).buy(1000)
    assert(xrp.set_seek_index(xrp.get_seek_index()+900).sell(1000)-cost==xrp.buy_sell(1000,100,900))

    print(xrp.buy_sell(1000,300,2300))


    
    
