import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

from llvmlite import ir

DIM_X = 1000
DIM_Y = 2000
u = np.random.rand(DIM_X, DIM_Y)
v = np.random.rand(DIM_X, DIM_Y)
vector = np.random.rand(DIM_X)
trans_vector = np.random.rand(DIM_Y)
scalar = np.random.rand()


llvm_mat_res = np.random.rand(DIM_X, DIM_Y)
llvm_vec_res = np.random.rand(DIM_Y)
llvm_tvec_res = np.random.rand(DIM_X)


mat_add_res = np.add(u,v)
mat_sub_res = np.subtract(u,v)
mat_mul_res = np.multiply(u, v)
dot_res = np.dot(vector, u)
trans_dot_res = np.dot(trans_vector, u.transpose())
mat_sadd_res = np.add(u, scalar)
mat_smul_res = np.multiply(u, scalar)


ct_u = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_vec = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_tvec = trans_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_mat_res = llvm_mat_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_vec_res = llvm_vec_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ct_tvec_res = llvm_tvec_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.mark.benchmark(group="Hadamard")
@pytest.mark.parametrize("op, builtin, result", [
                         (np.add, "__pnl_builtin_mat_add", mat_add_res),
                         (np.subtract, "__pnl_builtin_mat_sub", mat_sub_res),
                         (np.multiply, "__pnl_builtin_mat_hadamard", mat_mul_res),
                         ], ids=["ADD", "SUB", "MUL"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm)])
def test_mat_hadamard(benchmark, op, builtin, result, mode):
    if mode == 'Python':
        res = benchmark(op, u, v)
    elif mode == 'LLVM':
        llvm_fun = pnlvm.LLVMBinaryFunction.get(builtin)
        benchmark(llvm_fun, ct_u, ct_v, DIM_X, DIM_Y, ct_mat_res)
        res = llvm_mat_res
    assert np.allclose(res, result)


@pytest.mark.benchmark(group="Scalar")
@pytest.mark.parametrize("op, builtin, result", [
                         (np.add, "__pnl_builtin_mat_scalar_add", mat_sadd_res),
                         (np.multiply, "__pnl_builtin_mat_scalar_mult", mat_smul_res),
                         ], ids=["ADD", "MUL"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm)])
def test_mat_scalar(benchmark, op, builtin, result, mode):
    if mode == 'Python':
        res = benchmark(op, u, scalar)
    elif mode == 'LLVM':
        llvm_fun = pnlvm.LLVMBinaryFunction.get(builtin)
        benchmark(llvm_fun, ct_u, scalar, DIM_X, DIM_Y, ct_mat_res)
        res = llvm_mat_res
    assert np.allclose(res, result)


@pytest.mark.benchmark(group="Dot")
def test_dot_numpy(benchmark):
    numpy_res = benchmark(np.dot, vector, u)
    assert np.allclose(numpy_res, dot_res)


@pytest.mark.llvm
@pytest.mark.benchmark(group="Dot")
def test_dot_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vxm")
    benchmark(llvm_fun, ct_vec, ct_u, DIM_X, DIM_Y, ct_vec_res)
    assert np.allclose(llvm_vec_res, dot_res)


@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.benchmark(group="Dot")
def test_dot_cuda(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vxm")
    cuda_vec = pnlvm.jit_engine.pycuda.driver.In(vector)
    cuda_mat = pnlvm.jit_engine.pycuda.driver.In(u)
    cuda_res = pnlvm.jit_engine.pycuda.driver.Out(llvm_vec_res)
    benchmark(llvm_fun.cuda_call, cuda_vec, cuda_mat, np.int32(DIM_X), np.int32(DIM_Y), cuda_res)
    assert np.allclose(llvm_vec_res, dot_res)


@pytest.mark.llvm
@pytest.mark.benchmark(group="Dot")
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_dot_llvm_constant_dim(benchmark, mode):
    custom_name = None

    with pnlvm.LLVMBuilderContext() as ctx:
        custom_name = ctx.get_unique_name("vxsqm")
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, double_ptr_ty))

        # get builtin IR
        builtin = ctx.import_llvm_function("__pnl_builtin_vxm")

        # Create square vector matrix multiply
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        _x = ctx.int32_ty(DIM_X)
        _y = ctx.int32_ty(DIM_Y)
        _v, _m, _o = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.call(builtin, [_v, _m, _x, _y, _o])
        builder.ret_void()

    binf2 = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        benchmark(binf2, ct_vec, ct_u, ct_vec_res)
    else:
        import pycuda
        cuda_vec = pycuda.driver.In(vector)
        cuda_mat = pycuda.driver.In(u)
        cuda_res = pycuda.driver.Out(llvm_vec_res)
        benchmark(binf2.cuda_call, cuda_vec, cuda_mat, cuda_res)
    assert np.allclose(llvm_vec_res, dot_res)


@pytest.mark.benchmark(group="Dot")
def test_dot_transposed_numpy(benchmark):
    numpy_res = benchmark(np.dot, trans_vector, u.transpose())
    assert np.allclose(numpy_res, trans_dot_res)


@pytest.mark.llvm
@pytest.mark.benchmark(group="Dot")
def test_dot_transposed_llvm(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vxm_transposed")
    benchmark(llvm_fun, ct_tvec, ct_u, DIM_X, DIM_Y, ct_tvec_res)
    assert np.allclose(llvm_tvec_res, trans_dot_res)


@pytest.mark.llvm
@pytest.mark.cuda
@pytest.mark.benchmark(group="Dot")
def test_dot_transposed_cuda(benchmark):
    llvm_fun = pnlvm.LLVMBinaryFunction.get("__pnl_builtin_vxm_transposed")
    cuda_vec = pnlvm.jit_engine.pycuda.driver.In(trans_vector)
    cuda_mat = pnlvm.jit_engine.pycuda.driver.In(u)
    cuda_res = pnlvm.jit_engine.pycuda.driver.Out(llvm_tvec_res)
    benchmark(llvm_fun.cuda_call, cuda_vec, cuda_mat,
              np.int32(DIM_X), np.int32(DIM_Y), cuda_res)
    assert np.allclose(llvm_tvec_res, trans_dot_res)


@pytest.mark.llvm
@pytest.mark.benchmark(group="Dot")
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_dot_transposed_llvm_constant_dim(benchmark, mode):
    custom_name = None

    with pnlvm.LLVMBuilderContext() as ctx:
        custom_name = ctx.get_unique_name("vxsqm")
        double_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, double_ptr_ty))

        # get builtin IR
        builtin = ctx.import_llvm_function("__pnl_builtin_vxm_transposed")

        # Create square vector matrix multiply
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        _x = ctx.int32_ty(DIM_X)
        _y = ctx.int32_ty(DIM_Y)
        _v, _m, _o = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.call(builtin, [_v, _m, _x, _y, _o])
        builder.ret_void()

    binf2 = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        benchmark(binf2, ct_tvec, ct_u, ct_tvec_res)
    else:
        import pycuda
        cuda_vec = pycuda.driver.In(trans_vector)
        cuda_mat = pycuda.driver.In(u)
        cuda_res = pycuda.driver.Out(llvm_tvec_res)
        benchmark(binf2.cuda_call, cuda_vec, cuda_mat, cuda_res)
    assert np.allclose(llvm_tvec_res, trans_dot_res)
