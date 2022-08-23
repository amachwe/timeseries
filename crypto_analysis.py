import data_download as dd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def update():
    d1,d2 = dd.to_csv("XRP-USD")

    d1.to_csv("data/XRP-USD.csv")
    d2.to_csv("data/XRP-USD-price.csv")

def prob_plot(hist, be):
    total = np.sum(hist)

    csum = 0
    X = []
    Y = []
    for i, h in enumerate(hist):
        csum += h
        Y.append(csum/total)
        X.append(be[i])
        print((csum/total)*100,"%     < ",be[i+1])
    
    
    plt.bar(X,Y)
    plt.show()

if __name__ == "__main__":

    df = pd.read_csv("data/XRP-USD.csv")

    df["hl"] = (df["high"] - df["low"])*100/df["low"]
    df["co"] = (df["close"] - df["open"])*100/df["open"]

    print(df["hl"].max(),df["hl"].min(),df["co"].max(),df["co"].min())

    hist1, be1 = np.histogram(df["hl"],bins=100)
    hist2, be2 = np.histogram(df["co"],bins=100)
    print(len(hist1),len(be1))
    prob_plot(hist1,be1)
    prob_plot(hist2,be2)
    

    
        
        


