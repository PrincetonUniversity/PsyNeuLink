import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

from llvmlite import ir

DIM_X = 1000
DIM_Y = 2000
# These are just basic tests to check that matrix indexing and operations
# work correctly when compiled. The values don't matter much.
# Might as well make them representable in fp32 for single precision testing.
u = np.random.rand(DIM_X, DIM_Y).astype(np.float32).astype(np.float64)
v = np.random.rand(DIM_X, DIM_Y).astype(np.float32).astype(np.float64)
trans_u = u.transpose()
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
trans_dot_res = np.dot(trans_vector, trans_u)
mat_sadd_res = np.add(u, scalar)
mat_smul_res = np.multiply(u, scalar)

def _get_const_dim_func(builtin, *dims):
    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        custom_name = ctx.get_unique_name("cont_dim" + builtin)
        # get builtin function
        builtin = ctx.import_llvm_function(builtin)
        pointer_arg_types = [a for a in builtin.type.pointee.args if pnlvm.helpers.is_pointer(a)]

        func_ty = ir.FunctionType(ir.VoidType(), pointer_arg_types)


        # Create square vector matrix multiply
        function = ir.Function(ctx.module, builtin.type.pointee, name=custom_name)
        const_dims = (ctx.int32_ty(d) for d in dims)
        *inputs, output = (a for a in function.args if pnlvm.helpers.is_floating_point(a))
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.call(builtin, [*inputs, *const_dims, output])
        builder.ret_void()

    return custom_name

@pytest.mark.benchmark
@pytest.mark.parametrize("op, x, y, builtin, result", [
                         (np.add, u, v, "__pnl_builtin_mat_add", mat_add_res),
                         (np.subtract, u, v, "__pnl_builtin_mat_sub", mat_sub_res),
                         (np.multiply, u, v, "__pnl_builtin_mat_hadamard", mat_mul_res),
                         (np.add, u, scalar, "__pnl_builtin_mat_scalar_add", mat_sadd_res),
                         (np.multiply, u, scalar, "__pnl_builtin_mat_scalar_mult", mat_smul_res),
                         (np.dot, vector, u, "__pnl_builtin_vxm", dot_res),
                         (np.dot, trans_vector, trans_u, "__pnl_builtin_vxm_transposed", trans_dot_res),
                         ], ids=["ADD", "SUB", "MUL", "ADDS", "MULS", "DOT", "TRANS DOT"])
@pytest.mark.parametrize("dims", [(DIM_X, DIM_Y), (0, 0)], ids=["VAR-DIM", "CONST-DIM"])
def test_matrix_op(benchmark, op, x, y, builtin, result, func_mode, dims):
    if func_mode == 'Python':
        def ex():
            return op(x, y)

    elif func_mode == 'LLVM':
        if dims == (0, 0):
            func_name = _get_const_dim_func(builtin, DIM_X, DIM_Y)
        else:
            func_name = builtin

        bin_f = pnlvm.LLVMBinaryFunction.get(func_name)
        dty = np.dtype(bin_f.byref_arg_types[0])
        assert dty == np.dtype(bin_f.byref_arg_types[1])
        assert dty == np.dtype(bin_f.byref_arg_types[4])

        lx = x.astype(dty)
        ly = dty.type(y) if np.isscalar(y) else y.astype(dty)
        lres = np.empty_like(result, dtype=dty)

        ct_x = lx.ctypes.data_as(bin_f.c_func.argtypes[0])
        ct_y = ly if np.isscalar(ly) else ly.ctypes.data_as(bin_f.c_func.argtypes[1])
        ct_res = lres.ctypes.data_as(bin_f.c_func.argtypes[4])

        def ex():
            bin_f(ct_x, ct_y, *dims, ct_res)
            return lres

    elif func_mode == 'PTX':
        if dims == (0, 0):
            func_name = _get_const_dim_func(builtin, DIM_X, DIM_Y)
        else:
            func_name = builtin

        bin_f = pnlvm.LLVMBinaryFunction.get(func_name)
        dty = np.dtype(bin_f.byref_arg_types[0])
        assert dty == np.dtype(bin_f.byref_arg_types[1])
        assert dty == np.dtype(bin_f.byref_arg_types[4])

        lx = x.astype(dty)
        ly = dty.type(y) if np.isscalar(y) else y.astype(dty)
        lres = np.empty_like(result, dtype=dty)

        cuda_x = pnlvm.jit_engine.pycuda.driver.In(lx)
        cuda_y = ly if np.isscalar(ly) else pnlvm.jit_engine.pycuda.driver.In(ly)
        cuda_res = pnlvm.jit_engine.pycuda.driver.Out(lres)
        def ex():
            bin_f.cuda_call(cuda_x, cuda_y, np.int32(dims[0]), np.int32(dims[1]), cuda_res)
            return lres

    res = benchmark(ex)
    assert np.allclose(res, result)
