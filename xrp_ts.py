
import sklearn.model_selection as ms
import numpy as np
import keras.layers as layers
import keras.optimizers as opt
import keras.losses as loss
import keras.models as mods
import data_download as dd
from matplotlib import pyplot as plt
import pandas as pd
import ts_fun as ts
import pymongo as pm

def update():
    d1,d2 = dd.to_csv("XRP-USD")

    d1.to_csv("data/XRP-USD.csv")
    d2.to_csv("data/XRP-USD-price.csv")


def build_dataset(dataset,predict_length = 100):
    MAX = dataset.max()
    MIN = dataset.min()

    

    dataset = (dataset-MIN)/MAX
    X0 = dataset.loc[0]
    Y0 = dataset.loc[1]
    print(X0)
    df = dataset.diff()
    df = df.dropna()
    

    Y = df["low"][1:]
    X = df[:-1]


    x = X[:-predict_length]
    xt = X[predict_length:]
    y = Y[:-predict_length]
    yt = Y[predict_length:]

    
    

    return x.to_numpy(),xt.to_numpy(),y.to_numpy(),yt.to_numpy(), df, X0, Y0["low"] , MAX, MIN
    
def plot(dataset):
    cols = dataset.keys()
    
    f, a = plt.subplots(len(cols),1)
    for idx, c in enumerate(cols):
        a[idx].plot(dataset[c])
    plt.show()

def clean(x):
    xx = []
    for i in x:
        xx.append(i[0])
    return xx

def build_model(x,y,xt,yt, file="mod1.m", save=True):
    width = x.shape[1]

    model = mods.Sequential(
        (
            layers.Dense(width),
            layers.Dropout(0.1),
            layers.Dense(width),
            layers.Dense(width),
            layers.Dropout(0.1),
            layers.Dense(width),
            layers.Dense(1)
        )
    )

    model.compile(optimizer=opt.Adam(learning_rate=0.001),loss=loss.MSE,metrics=['mse'])
    model.fit(x,y, epochs=100, validation_split=0.2, verbose=1)
    model.evaluate(xt,yt,verbose=1)
    model.save(file)

    return model

def unnorm(Y,ymx,ymn):
    return (Y*ymx)+ymn

def norm(y):
    ymn = y.min()
    ymx = y.max()
    yy = y - ymn
    return yy/ymx, ymx,ymn

if __name__ == "__main__":

    update()

    FORCE_REBUILD = False
    predict_th = 50

    data = pd.read_csv("data/XRP-USD.csv")
    data.pop("time")
    data.pop("seq")
    data.pop("Unnamed: 0")

    
    
    x,xt,y,yt, df, X0, Y0, MAX, MIN = build_dataset(data,predict_length=predict_th)
    print(yt)
    # plot(df)
    # x,_,_ = norm(x)
    # y,ymx,ymn = norm(y)
    # xt,_,_ = norm(xt)
    # yt,ytmx,ytmn = norm(yt)

    
    #x,xt,y,yt =  #ms.train_test_split(X,Y,test_size=0.3,random_state=4)
    print("Shapes:",x.shape,y.shape,xt.shape,yt.shape)
    rm = None
    try:
        if not FORCE_REBUILD:
            rm = mods.load_model("mod1.m")
    except:
        print("Error loading model, building new model.")
        
    
    if not rm or FORCE_REBUILD:
        
        rm = build_model(x,y,xt,yt,save=False)


    yh = rm.predict(df)
    yhp = clean(yh)
    yt = yt.transpose()
    yhp = np.array(yhp)
    print(">>",yt.shape, MAX, MIN, Y0)

    print(yhp)
    
    # print(df["low"][0:5])
    # print(unnorm(df["low"],Y0)[0:5])
    # print(ts.undelta(Y0,unnorm(df["low"],MAX[0]))[0:5])
    #plt.plot(df["low"])
    #plt.plot(yhp)
    plt.plot(unnorm(np.array(ts.undelta(Y0,df["low"])),MAX["low"],MIN["low"]))
    #plt.plot(ts.undelta(Y0,yhp))
    plt.plot(unnorm(np.array(ts.undelta(Y0,yhp)),MAX["low"],MIN["low"]))
    plt.show()

    print(np.mean(np.abs(df["low"]-yhp)))
    print(rm.predict(np.array([df.loc[len(df)-1]])))

    








