import tslibs.data_client as dc
import ts_fun as ts
import data_download as dd
import pandas as pd
import numpy as np
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
    cost = xrp.buy(1000,100)
    assert(xrp.sell(1000,1000)-cost==xrp.buy_sell(1000,100,900))

    print(xrp.buy_sell(1000,300,2300))

    X = []
    Y = []
   
    for i in range(0,xrp.get_max_seek_index()):
        
        
        
        for j in range(1,14):
            try:
                X.append([i,j])
                
                bs = xrp.buy_sell(1000,i,j)
                Y.append(bs)
            
                
                

            except:
                print(i,j)
        
        
       
    
    
    # print(len(X),len(D))
    # f, a = plt.subplots(1,2,sharey=True)
    # a[0].scatter(MA,D)
    # a[1].scatter(P,D)
    # plt.show()

    X = np.array(X)
    Y = np.array(Y)
    Y=Y/Y.max()

    import sklearn.model_selection as m
    import tensorflow as tf
    import keras.layers as layers
    import keras.models as models
    
    x,xt,y,yt = m.train_test_split(X,Y,test_size=0.33,random_state=433)

    print(x.size,y.size)
    lnorm = layers.Normalization(axis=-1)
    lnorm.adapt(x)
    
    mod = models.Sequential(
        (
            lnorm,
            layers.Dense(units=2),
            layers.Dense(units=10),
            layers.Dropout(0.1),
            layers.Dense(units=5),
            layers.Dense(units=1)
    )
    )

    #print(mod.summary())
    
    mod.compile(tf.keras.optimizers.Adam(learning_rate=0.1),loss='mean_absolute_error')
    hist = mod.fit(x,y,epochs=100,verbose=1,validation_split=0.2)

    # plt.plot(hist.history["loss"])
    # plt.plot(hist.history["val_loss"])
    # plt.show()

    plt.plot(mod.predict(xt))
    plt.plot(yt)
    plt.show()

    
    








    
    
