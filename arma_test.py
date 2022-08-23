from email.errors import NonPrintableDefect
import ts_fun as ts
import statsmodels.graphics.gofplots as gof
import statsmodels.tsa.statespace.sarimax as sar

import numpy as np

def arma(coeff_ar, coeff_ma, steps=500):
    p = len(coeff_ar)
    q = len(coeff_ma)
    print(f"ARMA({p},{q})")

    x = np.array([0 for i in range(0,p)])
    for s in range(len(x)-1,steps):
        temp = 0
        for i in range(0,p):
            temp+= x[s-i]*coeff_ar[i]
        if q > 0:
            temp += np.random.normal(loc=0,scale=2)
        for i in range(0,q):
            temp += np.random.normal(loc=0,scale=2)*coeff_ma[i]
        
        x = np.append(x,temp)

    return x


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    x = arma([0.33],[0.9],steps=1000)
    ts.plot_acf(x)
    ts.plot_pacf(x)
    perf = {}
    min_perf = None
    min_aic = np.inf
    for i in range(0,4):
        for j in range(0,4):
            mod = sar.SARIMAX(x,order=(i,0,j))
            res = mod.fit(disp=False)
            perf[(i,j)] = res.aic
            if min_aic > res.aic:
                min_aic = res.aic
                min_perf = (i,j)
    import pprint
    pprint.pprint(perf)
    print(min_aic, min_perf)
    model = sar.SARIMAX(x,order=(min_perf[0],0,min_perf[1]))
    res = model.fit(disp=False)
    res.plot_diagnostics()
    gof.qqplot(res.resid,line='45')
    plt.show()
            







