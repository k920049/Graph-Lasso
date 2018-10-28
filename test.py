import numpy as np

slice_index = [i for i in range(3)]
index = np.delete(arr=slice_index, obj=1, axis=0)

matrix = np.random.rand(3, 3)
vector = np.random.rand(2)

print(matrix, vector)

matrix[index, 1] = vector
matrix[1, index] = vector

print(matrix)





