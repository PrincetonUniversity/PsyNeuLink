#!/usr/bin/python3

import ctypes
import numpy as np
import PsyNeuLink.llvm as pnlvm
import timeit
from llvmlite import ir

ITERATIONS=100
DIM_X=1000
DIM_Y=2000

matrix = np.random.rand(DIM_X, DIM_Y)
vector = np.random.rand(DIM_X)
llvm_res = np.random.rand(DIM_Y)

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

llvm_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_vxm')

start = timeit.default_timer()
for _ in range(ITERATIONS):
    llvm_fun(ct_vec, ct_mat, x, y, ct_res)
stop = timeit.default_timer()
print("LLVM time elapsed {:f}".format(stop-start))

if not np.allclose(llvm_res, result):
    print("TEST FAILED LLVM results differ!")
    print(llvm_res)
    print(result)

start = timeit.default_timer()

with pnlvm.LLVMBuilderContext() as ctx:
    double_ptr_ty = ctx.float_ty.as_pointer()
    func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, double_ptr_ty))

    # get builtin IR
    builtin = ctx.get_llvm_function('__pnl_builtin_vxm')

    # Create square vector matrix multiply
    function = ir.Function(ctx.module, func_ty, name="vxsqm")
    _x = ctx.int32_ty(x)
    _y = ctx.int32_ty(y)
    _v, _m, _o = function.args
    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.call(builtin, [_v, _m, _x, _y, _o])
    builder.ret_void()

stop = timeit.default_timer()
print("Build time elapsed {:f}".format(stop-start))

binf2 = pnlvm.LLVMBinaryFunction.get('vxsqm')
start = timeit.default_timer()
for _ in range(ITERATIONS):
    binf2(ct_vec, ct_mat, ct_res)
stop = timeit.default_timer()
print("LLVM-custom time elapsed {:f}".format(stop-start))

# Use all close to ignore rounding errors
if not np.allclose(llvm_res, result):
    print("TEST FAILED LLVM-custom results differ!")
    print(llvm_res)
    print(result)
else:
    print("TEST PASSED")
