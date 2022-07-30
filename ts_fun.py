import enum
from re import S
import numpy as np
import random
import tqdm
import pandas as pd
from torch import threshold

class Engine(object):

    def __init__(self,file,seq_col="low"):
        self.ds = pd.read_csv(file)
        self.seek_index = 0
        self.max_seek_index = len(self.ds.index)-1
        self.seq_col = seq_col
    
    def get_cols(self):
        return self.ds.columns

    def get_rows(self):
        return self.max_seek_index

    def set_seek_index(self, index):
        self.seek_index = index
        return self

    def get_seek_index(self):
        return self.seek_index

    def get_max_seek_index(self):
        return self.max_seek_index

    def reset(self):
        self.seek_index_index = 0
        return self

    def advance(self):
        if self.seek_index < self.max_seek_index:
            self.seek_index += 1
        return self

    def reverse(self):
        if self.seek_index > 0:
            self.seek_index -= 1
        return self

    def buy(self,qty):
        pr = self.ds[self.seq_col][self.seek_index]
        return qty*pr

    def sell(self,qty):
        return self.buy(qty)

    def buy_sell(self,qty,start, duration):
        cost = self.set_seek_index(start).buy(qty)
        return self.set_seek_index(duration+start).sell(qty)-cost

def nbrs(a,ds,eps):
    __nbrs = []
    dist = []
    
    for idx,b in enumerate(ds):
        #distance metric
        d = np.linalg.norm(b-a)
        
        if d <= eps:
            dist.append(d)
            __nbrs.append(idx)
        
    return __nbrs, dist

def dbscan(ds, eps=0.5, numpts = 3):
    
    c = 0
    ia = 0
    len_ds = len(ds)
    label = np.full(len_ds,-2)
    
    
    for idx, ia in enumerate(ds):
        #if labelled already
        if label[idx]>-2:
            continue
        
        _nbrs,_ = nbrs(ia,ds,eps)
        #if not enough neighbours then noise
        if len(_nbrs)+1 < numpts:
            label[idx] = -1
            continue
        #since it is not a noise point - assign current point to cluster
        label[idx] = c
        seedset = _nbrs
        #build neighbours seedset
        for s in seedset:
            
            if label[s] == -1:
                label[s]=c
            if label[s] > -2:
                continue
            
            #add to current set
            label[s] = c
            _nbrs, _ = nbrs(ds[s],ds,eps)
            
            if len(_nbrs)+1 >= numpts:
                #extends set
                ss = set(seedset)
                for i in _nbrs:
                    if i not in ss:
                        seedset.append(i)

        c = c+1
    
    return ds,label
    
def linefit(x,y, xm=None,ym=None):
    ## return Constant, Slope
    if not xm:
        xm = np.mean(x)
    if not ym:
        ym = np.mean(y)

    m = np.sum((x-xm)*(y-ym))/np.sum((x-xm)*(x-xm))
    c = ym - m*xm

    return c,m

def sample(x, size):
    L = len(x)
    start = np.random.choice(range(0,L-size-1))

    return x[start:start+size], x[start+size+1]

def corrcof(x,y):
    N = x.shape[0]


    sx = np.sum(x)
    sy = np.sum(y)
    n = N*np.sum(x*y)-sx*sy

    d = np.math.sqrt(N*np.sum(x*x)-sx*sx)*np.math.sqrt(N*np.sum(y*y)-sy*sy)

    return n/d

def build_lag_data(x, lag):
    
    X = []
    Y = []
    for i in range(lag,len(x)):
        Y.append(x[i])
        X.append(x[i-lag:i])
    
    return np.array(X),np.array(Y)

def lag_ols(x, lags=40):
    X = []
    Y = []
    for i in range(lags,len(x)):
        Y.append(x[i])
        X.append(x[i-lags:i])

    X = np.array(X)
    xt = np.transpose(X)
    r = np.matmul(xt,X)

    a = np.matmul(np.matmul(np.linalg.inv(r),xt),Y)
    #a = np.append(a,1.0)
    return np.flip(a)

def gen_ols(X,Y):
    

    X = np.array(X)
    xt = np.transpose(X)
    r = np.matmul(xt,X)

    a = np.matmul(np.matmul(np.linalg.inv(r),xt),Y)
    
    return np.flip(a)

def undelta(x0,x):
    und = [x0]
    sum = 0
    for i in x:
        sum += i
        und.append(x0+sum)
    return und



def delta(x,interval=1):
    
    
    delta = []
    for idx,i in enumerate(x):
        
        if idx >= interval:
            
            delta.append(i-x[idx-interval])
        
    return np.array(delta)

def random_walk(size=1000,c=0.5, max=1,min=-1):
    y0 = (max-min)*(0.5-random.random())
    y = []
    for i in range(0,size):
        y.append(y0)
        y0 = c*y0+((max-min)*(0.5-random.random()))

    return np.array(y)

def predict(model, xn, forecast_len=5):
    xn = [x[0] for x in xn]
    LAG = len(model)

    
    for i in range(0, forecast_len):
        f=0
        xx = xn[i:i+LAG]
        
        for j in range(0, LAG):
            f += xx[j]*model[j]
        xn.append(f)
        

    return xn
        
    

from statsmodels.tsa.stattools import adfuller
def evaluate_ts(ts, lags=30, plot=False, dominance_threshold=0.5):
    diff = 0
    res = adfuller(ts,maxlag=lags)
    tmp = ts

    print("Diff:", diff, "  Res:",res)
    stationary = True 
    while res[1] > 0.01:
        tmp = delta(tmp)
        diff += 1
        res = adfuller(tmp, maxlag=lags)
        print("Diff:", diff, "  Res`:",res)
        if diff > 5:
            print("Not stationary till diff 5")
            stationary = False
            break

    cc = []
    lag = []
    N = len(tmp)
    if plot:
        cc,lag = plot_acf(tmp,lags=lags)
    else:   
        for i in range(0,lags):
            
            shft = np.roll(tmp,-i)
            
            cc.append(corrcof(shft,tmp))
            lag.append(i)

    threshold = 2/np.math.sqrt(N)
  
    cnt_signf=0
    lag_signf=[]
    for idx, l in enumerate(lag):
        if cc[idx] > threshold:
            cnt_signf += 1
            lag_signf.append(l)
        elif cc[idx] < -threshold:
            cnt_signf += 1
            lag_signf.append(-l)
        

    dominance = cnt_signf/len(cc)
    
    

    return {
        "stationary":stationary,
        "diff": diff,
        "ratio":dominance,
        "significant_lags":lag_signf,
        "corr_coffs":cc,
        "lags":lag,
        "threshold": threshold
    }
        
from tslibs import measure
from matplotlib import pyplot as plt  

def train_test_split(x, split=0.8):
    S = int(split*len(x))
    train = x[:S]
    test = x[S:]

    return np.array(train),np.array(test)

def data_set_sample(x,y,train_ratio=0.8, sample_length=3):
    l = x.shape[0]
    if x.shape[0] < x.shape[1]:
        l = x.shape[1]

    train_size = int(train_ratio*l)
    test_size = int(abs(1-train_ratio)*l)
    index_choice = [i for i in range(0,l-sample_length*2)]
   
    
    x_tr = []
    y_tr = []
    x_ts = []
    y_ts = []

    collected = 0
    while collected < train_size:
        idx = np.random.randint(0,len(index_choice))
        
        i = index_choice[idx]
        x_tr.extend(x[i:i+sample_length])
        y_tr.extend(y[i:i+sample_length])
        
        if collected < test_size:
            x_ts.extend(x[i+sample_length:i+2*sample_length])
            y_ts.extend(y[i+sample_length:i+2*sample_length])
        
        index_choice.pop(idx)

        collected += sample_length
    
    return np.array(x_tr),np.array(y_tr),np.array(x_ts),np.array(y_ts)
        
            
        
        


def ma(x, q, train_size):
    
    train = np.array(x[:train_size], dtype=np.float)
    
    mu = np.mean(x)
    max = np.max(x)

    K = max
    
    
    e = np.array([random.gauss(0,1) for i in range(train_size)],dtype=np.float)
 
  
    for _ in tqdm.tqdm(range(0,10)):
        pred = []
        X = []
        Y = []
        for i in range(q, train_size):
          
            X.append((e[i-q:i]-train[i-q:i]))
            Y.append(x[i])
            

        vars = gen_ols(X,Y)

        for i in range(0,q):
            pred.append(e[i])
        for i in range(q,train_size):
            yt = ma_mod(mu,vars,e[i-q:i])
            pred.append(yt)
        
        pred = np.array(pred, dtype=np.float)
    
        e = train-pred
        pe = np.sum(np.abs(e))/len(e)
        if pe<0.001: 
            print("BREAK!", pe)
            break
    print(vars,mu)
    return vars, mu, e

def ma_mod(mu,vars,errors):

    return mu + np.sum(vars*errors)

def build_model(x,MA, TRAIN_SIZE):
    v, mu, e = ma(x,MA,TRAIN_SIZE)
    
    return run_model(mu,v,x, TRAIN_SIZE),mu,v, e

def run_model(mu, v, x, TRAIN_SIZE):
    pred = []
    ma = len(v)
    for i in range(TRAIN_SIZE, len(x)):
        x1 = ma_mod(mu,v,x[i-ma:i])
        xp.append(x1)
        pred.append(x1)
    return pred



from matplotlib import pyplot as plt
def plot_acf(x, lags=30):
    cc = []
    lag = []
    plt.ylim(-1,1)
    N = len(x)
    for i in range(0,lags):

        shft = np.roll(x,-i)
        
        cc.append(corrcof(shft,x))
        lag.append(i)

    plt.plot(lag,cc,'.')
    plt.hlines(0,0,lags+1)
    plt.hlines(1.96/np.math.sqrt(N),0,lags+1)
    plt.hlines(-1.96/np.math.sqrt(N),0,lags+1)
    plt.show()

    return np.array(cc), lag


def plot_pacf(x, lags=30):
    
    lag = [i for i in range(0, lags+1)]
    
    plt.ylim(-1,1)
    N = len(x)
    dd = lag_ols(x,lags=lags)

    plt.plot(lag,dd,'.')
    plt.hlines(0,0,lags+1)
    plt.hlines(1.96/np.math.sqrt(N),0,lags+1)
    plt.hlines(-1.96/np.math.sqrt(N),0,lags+1)
    plt.show()

    return np.array(dd), lag

def ma_gen_sample(vars,mu=0,sample_len=1000):

    samples = []
    v = np.array(vars)
    e = np.random.normal(size=sample_len)
    for i in range(sample_len):
        
        x1 = mu
        for j,v in enumerate(vars):
            if i-j-1>=0:
                x1+=e[i-j-1]*v
        
        samples.append(x1)


    return np.array(samples)

if __name__ == "__main__":

    #test sample data set 
    ds = np.array([[i-1,i,i+1] for i in range(0,200)])
    y = np.array([int(i%2==0) for i in range(0,200)])

    xt,yt,xts,yts = data_set_sample(ds,y)
    print(xt[0:3],"\n",xts[0:3])
