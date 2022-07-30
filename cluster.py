import numpy as np
import pandas as pd

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


    