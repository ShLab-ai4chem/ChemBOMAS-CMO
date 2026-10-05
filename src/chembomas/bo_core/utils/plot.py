import matplotlib.pyplot as plt
from torchtyping import TensorType
from typing import List
import torch
import os
import datetime
import numpy as np
def plot_results(results:List[List[TensorType['num','value']]],batch_size:int)->None:
    if results[0][0].shape[1] != 1:
        raise NotImplementedError("Only support 1D data")
    # batch the results
    results = [[v.view(1) for v in result] for result in results]
    results = np.array(results)
    batched_results = []
    for i in range(batch_size,results.shape[1],batch_size):
        batched_results.append(np.max(results[:,i-batch_size:i]
                               ,axis=1))
    batched_results = np.hstack(batched_results)
    max_till_now = np.maximum.accumulate(batched_results,axis=1)
    avg_results = np.mean(max_till_now,axis=0)
    error_bar = np.std(max_till_now,axis=0)
    
    plt.plot(avg_results)
    plt.fill_between(range(len(avg_results)),avg_results-error_bar,avg_results+error_bar,alpha=0.5)

def plot_resultses(resultses:List[List[TensorType['num','value']]],batch_size:int,name:List[str],save_dir=None)->None:
    plt.figure()
    avg_resultses = []
    for results in resultses:
        if results[0][0].shape[1] != 1:
            raise NotImplementedError("Only support 1D data")
        # batch the results
        results = [[v.view(1) for v in result] for result in results]
        results = np.array(results)
        batched_results = []
        for i in range(batch_size,results.shape[1]+1,batch_size):
            batched_results.append(np.max(results[:,i-batch_size:i]
                                ,axis=1))
        batched_results = np.hstack(batched_results)
      
        max_till_now = np.maximum.accumulate(batched_results,axis=1)
 
        avg_results = np.mean(max_till_now,axis=0)
        error_bar = np.std(max_till_now,axis=0)
        avg_resultses.append(avg_results)

    for i,avg_results in enumerate(avg_resultses):
        plt.plot(avg_results,label=name[i])
        # plt.fill_between(range(len(avg_results)),avg_results-error_bar,avg_results+error_bar,alpha=0.5)
    plt.legend()
    if save_dir is not None:
        os.makedirs(save_dir,exist_ok=True)
        path = os.path.join(save_dir,f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        plt.savefig(path)
        plt.close()
   
    
