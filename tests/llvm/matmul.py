#!/usr/bin/python3

import ctypes
import numpy as np
import PsyNeuLink.llvm as pnlvm
import timeit

ITERATIONS=100
DIM=1000

matrix = np.random.rand(DIM, DIM)
vector = np.random.rand(DIM)
llvm_res = np.random.rand(DIM)

start = timeit.default_timer()
for _ in range(ITERATIONS):
    result = np.dot(vector, matrix)
stop = timeit.default_timer()
print("Numpy time elapsed {:f}".format(stop-start))

start = timeit.default_timer()

ct_vec = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_mat = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_res = llvm_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x, y = matrix.shape

stop = timeit.default_timer()
print("Convert time elapsed {:f}".format(stop-start))

llvm_fun = pnlvm.get_mxv()

start = timeit.default_timer()
for _ in range(ITERATIONS):
    llvm_fun(ct_vec, ct_mat, x, y, ct_res)
stop = timeit.default_timer()
print("LLVM time elapsed {:f}".format(stop-start))

# Use all close to ignore rounding errors
if not np.allclose(llvm_res, result):
    print("TEST FAILED results differ!")
    print(llvm_res)
    print(result)
else:
    print("TEST PASSED")
