
import sklearn.model_selection as ms
import numpy as np
import keras.layers as layers
import keras.optimizers as opt
import keras.losses as loss
import keras.models as mods
import data_download as dd
from matplotlib import pyplot as plt
import pandas as pd

def update():
    d1,d2 = dd.to_csv("XRP-USD")

    d1.to_csv("data/XRP-USD.csv")
    d2.to_csv("data/XRP-USD-price.csv")

def build_dataset(data,max_ma=250,interval=3):
    
    Y = np.array([int(i>0.05) for i in data.diff()])
    
    X = []
    for i in range(1,max_ma,interval):
        
        x = data.rolling(i).mean().fillna(0)
        
        
        X.append(x)
    X = np.array(X)
    X=(X/X.max()).transpose()
    print(X.shape,Y.shape)
    return X,Y
        


if __name__ == "__main__":
    update()
    
    data = pd.read_csv("data/XRP-USD.csv")
    
    low = data["low"]
    high = data["high"]
    close = data["close"]
    open = data["open"]
    time = data["time"]

    X,Y = build_dataset(low)

    x,xt,y,yt = ms.train_test_split(X,Y,test_size=0.3,random_state=300)
    print(x.shape,y.shape)
 
    model = mods.Sequential(
        (
            layers.Dense(len(x[1])),
            layers.Dropout(0.1),
            layers.Dense(len(x[1])),
            #layers.Dense(len(x[1])),

            layers.Dense(int(len(x[1])/2)),
            layers.Dense(1)
        )
    )
    model.compile(opt.Adam(learning_rate=0.001),loss=loss.BinaryCrossentropy(),metrics=['accuracy'])
    model.fit(x,y,epochs=200,validation_split=0.2,verbose=1)
    
    
    # xx = x.transpose()
    # plt.scatter(xx[3],xx[6],c=y,cmap="copper")
    # plt.show()
    ths = np.arange(0.1,0.9,0.01)
    perf = []
    for th in ths:
        yp = np.array([int(i>th) for i in model.predict(xt,verbose=0)])
        # plt.plot(yp)
        # plt.plot(yt)
        # plt.show()
        perf.append((yt-yp).sum())
        if perf[-1] <= 3 and perf[-1] >= -3:
            print(th,": ",perf[-1])

    plt.plot(perf,ths)
    plt.show()
        
    pred = model.predict(X)
    print("Predictions,   Low,    Time")
    
    for i in range(len(X)-1,len(X)-4,-1):
        print(pred[i][0],low[i],time[i])



    
   


    

