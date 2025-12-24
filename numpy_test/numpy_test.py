import numpy as np
import timeit

SCALE = 1000000

arr = [i for i in range(SCALE)]
brr = [i for i in range(SCALE)]

def product(arr, brr):
    sum = 0
    for i in range(SCALE):
        sum += arr[i] * brr[i]
    return sum

np_arr = np.array(arr)
np_brr = np.array(brr)

print(product(arr, brr) == np.dot(np_arr, np_brr))
print(timeit.timeit(lambda :product(arr, brr), number=1))
print(timeit.timeit(lambda :np.dot(np_arr, np_brr), number=1))
