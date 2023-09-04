import numpy as np

N = 10000
n = 100

flag = False
arr = np.random.rand(N)

idx = np.random.randint(N, size=n)
idxsort = np.sort(idx)

if flag:
    arr[(idxsort, )] = 100
else:
    arr[(idx, )] = 100

print(f"N = {N}, n = {n}, sort = {flag}")
