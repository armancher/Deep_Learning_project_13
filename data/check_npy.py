import numpy as np

arr = np.load("byte_bpe_test.npy")
print(arr[:200])         # print first 200 tokens
print(len(arr))          # length of dataset
print(type(arr))         # should be numpy.ndarray
print(arr.dtype)         # should be int64