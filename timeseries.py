import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def meap(y, yp):
    return np.mean(np.abs((y-yp)/y))*100

def meanf(y):
    v_over_m = 0
    o = 0
    v_under_m = 0
    u = 0


    for i in range(0,y.shape[0]-1):
        m = y["value"][:i].mean()
        if m != m: 
            continue

        if y["value"][i+1] > m:
            v_over_m += y["value"][i+1] - m
            o += 1
        else:
            v_under_m += m - y["value"][i+1]
            u += 1

    return v_over_m/o, v_under_m/u




raw_df = pd.read_csv("data/jandj.csv")

print(raw_df.head())

test = raw_df[-4:]
train = raw_df[:-4]

print(train.shape, test.shape)

mean = train["value"].mean()

test["pred"] = mean

print(">>",meap(test["value"],test["pred"]))
v,o = meanf(raw_df)

p = train["value"]
p = p.values[-1]
pp = []

f = False
for i in range(test.shape[0]):
    if f == False:
        p = p + v
        f = True
    else:
        p = p - o
        f = False
    
    pp.append(p)

print(">",meap(test["value"],pp))

plt.plot(pp, label="pp")
plt.plot(test["value"].values, label="Val")
plt.plot(test["pred"].values, label="Pred")
plt.legend()
plt.show()
