import numpy as np

x = [1,2,3,4,5]
dimension_matrix = np.size(x)
phi = np.ones((dimension_matrix,2))
i = 0
while i < len(x):
    phi[i,1] = x[i]
    i += 1

print(phi)