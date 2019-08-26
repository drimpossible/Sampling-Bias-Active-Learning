import numpy as np
from numba import njit, prange
import torch
from datetime import datetime
from joblib import Parallel

asize = 1000000
bsize = 3000000

# CUDA
a = torch.ones((asize,300)).cuda()
b = torch.zeros((bsize,300)).cuda()

start=datetime.now()
def calculate_distance_cuda(selected_set, unselected_set):
    tot = [torch.min(torch.sqrt(torch.norm((unselected_set[i,:].unsqueeze(0)-selected_set[:,:]),dim=1)+2.0e-16)) for i in range(unselected_set.shape[0])]
    return tot

calculate_distance_cuda(a,b)
print(datetime.now()-start)

# Parallel
start=datetime.now()
a = np.random.random((asize,300))
b = np.random.random((bsize,300))

@njit(parallel=True,fastmath=True)
def calculate_distance(selected_set, unselected_set):
    tot = np.zeros(unselected_set.shape[0])
    for i in range(unselected_set.shape[0]):
        dist = np.zeros(selected_set.shape[0])
        for j in prange(selected_set.shape[0]):
            dist[j] = np.sqrt(np.square(unselected_set[i,:]-selected_set[j,:]).sum()+2.0e-16)
        tot[i] = np.min(dist)
    return tot

calculate_distance(a,b)
print(datetime.now()-start)

#print(a.shape,b.shape)
#calculate_distance(a,b)
#dist_vectorized(b,a[0])


