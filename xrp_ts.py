import tslibs.data_client as dc
import tslibs.measure as mes
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from ts_fun import corrcof, delta, linefit
import ts_fun
        

# delta([1,2,3,4,5,6,7,8])
# delta([1,2,3,4,5,6,7,8], interval=2)
# delta([1,2,3,4,5,6,7,8], interval=3)
# delta([1,2,3,4,5,6,7,8], interval=4)
# delta([1,2,3,4,5,6,7,8], interval=5)


TEST = 30
MX = []
for i in range(1,365):
    stream = dc.get_data_single("XRP-USD",agg_dur=f"{i}d",agg_fn="mean")[0]



    raw_df = pd.DataFrame(stream,columns=["time","open","close"])
    raw_df["seq"] = [i for i in range(raw_df.shape[0])]
    dl = np.array(delta(raw_df["open"]))

    MX.append(dl.max())

plt.plot(MX)
plt.show()
test = raw_df[-TEST:]
train = raw_df[:-TEST]

print(test.shape,train.shape)





v = delta(raw_df["open"])

m = []
s = []
t = 0

# for i in range(0,v.shape[0]):
    
#     m.append(v[:i].mean())
#     s.append(v[:i].std())

# plt.plot(s)
# plt.plot(m)
# plt.show()
    
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# adr = []
# ii = []
# LEN = raw_df.shape[0]
# for i in range(10,LEN,6):
#     res = adfuller(raw_df["price"][:i])
#     adr.append(res[1])
#     ii.append(i)

# plt.plot(raw_df["price"])
# plt.hlines(0.05,0,LEN)
# plt.plot(ii,adr)

# plt.show()


Y = raw_df["open"]
N = raw_df.shape[0]
dl = np.array(delta(Y))
ts_fun.plot_acf(dl)
print(dl.max())



plot_acf(Y, lags=30)
plt.show()
plot_acf(v, lags=30)
plt.show()


ts_fun.plot_pacf(dl, lags=30)
plot_pacf(dl,lags=30)
plt.show()



# # plt.plot(test["gmean"])
# # plt.plot(test["lNmean"])
# # plt.plot(test["price"])
# a,b = linefit(raw_df["seq"][900:1200],raw_df["price"][900:1200])
# print(a,b)
# plt.plot(raw_df["seq"],raw_df["price"])
# plt.plot(raw_df["seq"][900:1200],a+raw_df["seq"][900:1200]*b)
# plt.show()
# m = []
# s = []
# for i in range(1,366):
#     v = delta(raw_df["price"], interval=i)
#     m.append(v.mean())
#     s.append(v.std())
    

# plt.plot(m)
# plt.plot(s)
# plt.show()





