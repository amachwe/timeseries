from matplotlib import pyplot as plt

import statsmodels.graphics.tsaplots as tsp
import statsmodels.tsa.stattools as st
import numpy as np
from ts_fun import delta, corrcof, random_walk, plot_acf, lag_ols,evaluate_ts

adf = []
cd = []

c = 0.98


rw = random_walk(c=c,size=1000)

evaluate_ts(rw)

drw = delta(rw)
plt.plot(rw)
plt.show()

cc, lags = plot_acf(rw,lags=100)
dcc, lags = plot_acf(drw, lags=100)


test = np.math.sqrt(rw.shape[0])*dcc
# plt.hist(test,bins=30)
# plt.show()
print(np.mean(lag_ols(rw)-st.pacf(rw)))
print(np.mean(lag_ols(drw)-st.pacf(drw)))
plt.show()








