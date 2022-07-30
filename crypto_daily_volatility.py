
from dataclasses import replace
from cluster import dbscan
import tslibs.data_client as dc
import tslibs.measure as mes

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import tslibs.accum as acc
import tensorflow as tf
import keras as ks
from scipy.optimize import curve_fit
from sklearn import model_selection

from ts_fun import build_lag_data, corrcof, delta, gen_ols, lag_ols, linefit,evaluate_ts, train_test_split, predict, sample, dbscan
import ts_fun

import data_download as dd

def update():
    d1,d2 = dd.to_csv("XRP-USD")

    d1.to_csv("data/XRP-USD.csv")
    d2.to_csv("data/XRP-USD-price.csv")

def label_sampling(ds,labels, size=1000):
    d = []
    for l in labels:
        sample = ds[ds["lbl"]==l]
        d.append(sample.sample(n=size, replace=True))
    return pd.concat(d)

def build_model():
    #Counter({0: 2772, -1: 78, 1: 9})
    model = ks.Sequential(
        [ks.layers.Input(6),ks.layers.Dense(6),ks.layers.Dense(6),ks.layers.Dense(6),ks.layers.Dense(6),ks.layers.Dense(1)]
    )
    
    ds = pd.read_csv("data/XRP_learn1.csv")
    
    XY = label_sampling(ds,[0,-1])
    XY.to_csv("xy_test.csv")
    Y = XY.pop("lbl")
    X = (XY-XY.min())/(XY.max()-XY.min())
   

    
    XYt = label_sampling(ds,[0,-1],size=100)
    Yt = XYt.pop("lbl")
    Xt = (XYt-XYt.min())/(XYt.max()-XYt.min())
    
    print(X.shape,Xt.shape,Y.shape,Yt.shape)

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    model.fit(X,Y,epochs=10)
    model.evaluate(Xt,Yt,verbose=2)


    

def norm(x):
    return x/x.max()


if __name__ == "__main__":

    #update()
    
    build_model()
    exit()
    
    price = pd.read_csv("data/XRP-USD-price.csv",index_col="time",parse_dates=True)
    daily = pd.read_csv("data/XRP-USD.csv",index_col="time",parse_dates=True)

    print(daily.shape)

    high = norm(daily["high"][:-1])
    low = norm(daily["low"][:-1])
    open = norm(daily["open"][:-1])
    close = norm(daily["close"][:-1])
    volume = norm(daily["volume"][:-1])
    
    time = daily.index[:-1]

    move = np.diff(daily["low"])
  
    ds, lbl = dbscan(np.array([volume,low,high,open,close,move]).transpose(),eps=0.1,numpts=8)

    ds1 = np.array([volume,low,high,open,close,move,lbl]).transpose()
    print(ds1.shape)
    pd.DataFrame(ds1).to_csv("data/xrp_learn.csv")
    from collections import Counter

    print(Counter(lbl))

    fig, ax = plt.subplots(2,1)
    ax[0].scatter(time, volume,c=lbl,cmap="copper")
    ax[1].scatter(time, low,c=lbl,cmap="copper")
    plt.show()

    
  

    

    

    