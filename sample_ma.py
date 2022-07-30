from pickle import TRUE
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ts_fun import evaluate_ts, delta, ma_gen_sample
from ts_fun import measure

import numpy as np

np.random.seed(42)

MA2P = ma_gen_sample([1.0,0.6,0.73,0.86,0.7], sample_len=10000)# ArmaProcess([1,0,0],[1,0.9,0.3,0.6,0.7]).generate_sample(nsample=10000)

from matplotlib import pyplot as plt


print(evaluate_ts(MA2P, plot=TRUE))

TOTAL_LEN = len(MA2P)
TRAIN_LEN = TOTAL_LEN - 100
TEST_LEN = TOTAL_LEN - TRAIN_LEN
WINDOW = 2

pred_data = []
for i in range(TRAIN_LEN, TOTAL_LEN, WINDOW):
    model = SARIMAX(MA2P[:i],order = (0,0,2))
    res = model.fit(disp=False)

    pred = res.get_prediction(0,i+WINDOW-1)

    pred_data.extend(pred.predicted_mean[-WINDOW:])

print(measure.mape(MA2P[TRAIN_LEN:],pred_data))

model = SARIMAX(MA2P[:TRAIN_LEN], order=(0,0,2))
res = model.fit(disp=False)
pred_2 = res.get_prediction(0,TEST_LEN-1)

print(measure.mape(MA2P[TRAIN_LEN:], pred_2.predicted_mean))
plt.plot(MA2P[TRAIN_LEN:],'.')
plt.plot(pred_data,'.')
plt.plot(pred_2.predicted_mean,'.')
plt.show()


