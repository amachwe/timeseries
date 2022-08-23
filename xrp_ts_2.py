
import sklearn.model_selection as ms
import statsmodels.tsa.statespace.sarimax as tsa
import statsmodels.graphics.gofplots as gof
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

def update(sym="XRP-USD"):
    d1,d2 = dd.to_csv(sym)

    d1.to_csv(f"data/{sym}.csv")
    d2.to_csv(f"data/{sym}-price.csv")


def data_set(df,test_len=100):
    # d_df = df.diff()
    # d_df = d_df.dropna()
    d_df = df
    return d_df,d_df.iloc[:-test_len], d_df.iloc[-test_len:]

def build_model(series,train_len,total,predict_threshold=3,extn=2):
    
    preds = []
    res = None
    perf = {}
    min_aic = np.inf
    best_perf = None
    for p in range(0,5):
        for q in range(0,5):
            model = tsa.SARIMAX(series[:train_len],order=(q,0,p))
            res = model.fit(disp=False)
            perf[(q,p)] = res.aic

            if min_aic > res.aic:
                min_aic = res.aic
                best_perf = (q,p)
            

    print("Best:",best_perf,min_aic)
    for i in range(train_len,total+extn,predict_threshold):
 
        if i <= total:
            model = tsa.SARIMAX(series[:i],order=(best_perf[0],0,best_perf[1]))
            
            res = model.fit(disp=False)
    
        predictions = res.get_prediction(0,i-1+predict_threshold)

        preds.extend(predictions.predicted_mean.iloc[-predict_threshold:])
    return preds, res
    

    

def predict(test,t0,max,min,mean):
    xpred = []
    prev = t0
    for _ in range(0,len(test)):
        x = mean*(1-np.random.rand())
        xpred.append((x+prev)/2)
        prev = x
        
    return xpred
    



if __name__ == "__main__":

    predict_threshold = 1
    sym = "XRP-USD"
    #update(sym=sym)
    test_len = 100

    df = pd.read_csv(f"data/{sym}.csv")
    df.pop("time")
    df.pop("seq")
    df.pop("Unnamed: 0")
    # ts.plot_pacf(df["low"])
    # ts.plot_pacf(df["high"])
    # ts.plot_pacf(df["close"])
    preds = []
    actuals = []
    for i in ["low","high","close"]:
        d_df,d_df_train,d_df_test = data_set(df[i], test_len=test_len)
        
        pred, res = build_model(d_df,len(d_df_train),len(d_df),predict_threshold=predict_threshold)
        
        print(i,"\nData:",len(d_df_train),len(d_df_test)) 
        print("Pred:",len(pred))

        t = d_df_test.to_numpy()
        #res.plot_diagnostics()
        plt.show()
        preds.append(pred)
        actuals.append(t)
    for p in preds:
        plt.plot(p,'.')
    for a in actuals:
        plt.plot(a)
    plt.show()
    
    