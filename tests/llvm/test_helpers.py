import ctypes
import ctypes.util
import copy
import numpy as np
import pytest
import sys

from psyneulink.core import llvm as pnlvm
from llvmlite import ir


DIM_X = 1000
TST_MIN = 1.0
TST_MAX = 3.0

VECTOR = np.random.rand(DIM_X)

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_fclamp(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        double_ptr_ty = ir.DoubleType().as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, ctx.int32_ty,
                                                  double_ptr_ty))

        # Create clamp function
        custom_name = ctx.get_unique_name("clamp")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        vec, count, bounds = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        tst_min = builder.load(builder.gep(bounds, [ctx.int32_ty(0)]))
        tst_max = builder.load(builder.gep(bounds, [ctx.int32_ty(1)]))

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "linear") as (b1, index):
            val_ptr = b1.gep(vec, [index])
            val = b1.load(val_ptr)
            val = pnlvm.helpers.fclamp(b1, val, tst_min, tst_max)
            b1.store(val, val_ptr)

        builder.ret_void()

    ref = np.clip(VECTOR, TST_MIN, TST_MAX)
    bounds = np.asfarray([TST_MIN, TST_MAX])
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    local_vec = copy.deepcopy(VECTOR)
    if mode == 'CPU':
        ct_ty = ctypes.POINTER(bin_f.byref_arg_types[0])
        ct_vec = local_vec.ctypes.data_as(ct_ty)
        ct_bounds = bounds.ctypes.data_as(ct_ty)

        bin_f(ct_vec, DIM_X, ct_bounds)
    else:
        bin_f.cuda_wrap_call(local_vec, np.int32(DIM_X), bounds)

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_fclamp_const(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        double_ptr_ty = ir.DoubleType().as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, ctx.int32_ty))

        # Create clamp function
        custom_name = ctx.get_unique_name("clamp")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        vec, count = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "linear") as (b1, index):
            val_ptr = b1.gep(vec, [index])
            val = b1.load(val_ptr)
            val = pnlvm.helpers.fclamp(b1, val, TST_MIN, TST_MAX)
            b1.store(val, val_ptr)

        builder.ret_void()

    local_vec = copy.deepcopy(VECTOR)
    ref = np.clip(VECTOR, TST_MIN, TST_MAX)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = ctypes.POINTER(bin_f.byref_arg_types[0])
        ct_vec = local_vec.ctypes.data_as(ct_ty)

        bin_f(ct_vec, DIM_X)
    else:
        bin_f.cuda_wrap_call(local_vec, np.int32(DIM_X))

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_is_close(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        double_ptr_ty = ir.DoubleType().as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), [double_ptr_ty, double_ptr_ty,
                                                  double_ptr_ty, ctx.int32_ty])

        # Create clamp function
        custom_name = ctx.get_unique_name("all_close")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        in1, in2, out, count = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        index = None
        with pnlvm.helpers.for_loop_zero_inc(builder, count, "compare") as (b1, index):
            val1_ptr = b1.gep(in1, [index])
            val2_ptr = b1.gep(in2, [index])
            val1 = b1.load(val1_ptr)
            val2 = b1.load(val2_ptr)
            close = pnlvm.helpers.is_close(b1, val1, val2)
            out_ptr = b1.gep(out, [index])
            out_val = b1.select(close, val1.type(1), val1.type(0))
            res = b1.select(close, out_ptr.type.pointee(1),
                                   out_ptr.type.pointee(0))
            b1.store(out_val, out_ptr)

        builder.ret_void()

    vec1 = copy.deepcopy(VECTOR)
    tmp = np.random.rand(DIM_X)
    tmp[0::2] = vec1[0::2]
    vec2 = np.asfarray(tmp)
    assert len(vec1) == len(vec2)
    res = np.empty_like(vec2)

    ref = np.isclose(vec1, vec2)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = ctypes.POINTER(bin_f.byref_arg_types[0])
        ct_vec1 = vec1.ctypes.data_as(ct_ty)
        ct_vec2 = vec2.ctypes.data_as(ct_ty)
        ct_res = res.ctypes.data_as(ct_ty)

        bin_f(ct_vec1, ct_vec2, ct_res, DIM_X)
    else:
        bin_f.cuda_wrap_call(vec1, vec2, res, np.int32(DIM_X))

    assert np.array_equal(res, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_helper_all_close(mode):

    with pnlvm.LLVMBuilderContext() as ctx:
        arr_ptr_ty = ir.ArrayType(ir.DoubleType(), DIM_X).as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), [arr_ptr_ty, arr_ptr_ty,
                                                  ir.IntType(32).as_pointer()])

        custom_name = ctx.get_unique_name("all_close")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        in1, in2, out = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        all_close = pnlvm.helpers.all_close(builder, in1, in2)
        res = builder.select(all_close, out.type.pointee(1), out.type.pointee(0))
        builder.store(res, out)
        builder.ret_void()

    vec1 = copy.deepcopy(VECTOR)
    vec2 = copy.deepcopy(VECTOR)

    ref = np.allclose(vec1, vec2)
    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_ty = ctypes.POINTER(bin_f.byref_arg_types[0])
        ct_vec1 = vec1.ctypes.data_as(ct_ty)
        ct_vec2 = vec2.ctypes.data_as(ct_ty)
        res = ctypes.c_int32()

        bin_f(ct_vec1, ct_vec2, ctypes.byref(res))
    else:
        res = np.array([5], dtype=np.int32)
        bin_f.cuda_wrap_call(vec1, vec2, res)
        res = res[0]

    assert np.array_equal(res, ref)

@pytest.mark.llvm
@pytest.mark.parametrize("ir_argtype,format_spec,values_to_check", [
    (pnlvm.ir.IntType(32), "%u", range(0, 100)),
    (pnlvm.ir.IntType(64), "%ld", [int(-4E10), int(-3E10), int(-2E10)]),
    (pnlvm.ir.DoubleType(), "%lf", [x *.5 for x in range(0, 10)]),
    ], ids=["i32", "i64", "double"])
@pytest.mark.skipif(sys.platform == 'win32', reason="Loading C library is complicated on windows")
def test_helper_printf(capfd, ir_argtype, format_spec, values_to_check):
    format_str = f"Hello {(format_spec+' ')*len(values_to_check)} \n"
    with pnlvm.LLVMBuilderContext() as ctx:
        func_ty = ir.FunctionType(ir.VoidType(), [])
        ir_values_to_check = [ir_argtype(i) for i in values_to_check]
        custom_name = ctx.get_unique_name("test_printf")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        pnlvm.helpers.printf(builder, format_str, *ir_values_to_check, override_debug=True)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)


    # Printf is buffered in libc.
    bin_f()
    libc = ctypes.util.find_library("c")
    libc = ctypes.CDLL(libc)
    libc.fflush(0)
    assert capfd.readouterr().out == format_str % tuple(values_to_check)

class TestHelperTypegetters:
    FLOAT_TYPE = pnlvm.ir.FloatType()
    FLOAT_PTR_TYPE = pnlvm.ir.PointerType(FLOAT_TYPE)
    DOUBLE_TYPE = pnlvm.ir.DoubleType()
    DOUBLE_PTR_TYPE = pnlvm.ir.PointerType(DOUBLE_TYPE)
    DOUBLE_VECTOR_TYPE = pnlvm.ir.ArrayType(DOUBLE_TYPE, 1)
    DOUBLE_VECTOR_PTR_TYPE = pnlvm.ir.PointerType(DOUBLE_VECTOR_TYPE)
    DOUBLE_MATRIX_TYPE = pnlvm.ir.ArrayType(pnlvm.ir.ArrayType(DOUBLE_TYPE, 1), 1)
    DOUBLE_MATRIX_PTR_TYPE = pnlvm.ir.PointerType(DOUBLE_MATRIX_TYPE)
    INT_TYPE = pnlvm.ir.IntType(32)
    INT_PTR_TYPE = pnlvm.ir.PointerType(INT_TYPE)
    BOOL_TYPE = pnlvm.ir.IntType(1)
    BOOL_PTR_TYPE = pnlvm.ir.PointerType(BOOL_TYPE)

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 0),
        (FLOAT_PTR_TYPE, 1),
        (DOUBLE_TYPE, 0),
        (DOUBLE_PTR_TYPE, 1),
        (DOUBLE_VECTOR_TYPE, 0),
        (DOUBLE_VECTOR_PTR_TYPE, 1),
        (DOUBLE_MATRIX_TYPE, 0),
        (DOUBLE_MATRIX_PTR_TYPE, 1),
        (INT_TYPE, 0),
        (INT_PTR_TYPE, 1),
        (BOOL_TYPE, 0),
        (BOOL_PTR_TYPE, 1),
    ], ids=str)
    def test_helper_is_pointer(self, ir_type, expected):
        assert pnlvm.helpers.is_pointer(ir_type) == expected
        assert pnlvm.helpers.is_pointer(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 1),
        (FLOAT_PTR_TYPE, 1),
        (DOUBLE_TYPE, 1),
        (DOUBLE_PTR_TYPE, 1),
        (DOUBLE_VECTOR_TYPE, 0),
        (DOUBLE_VECTOR_PTR_TYPE, 0),
        (DOUBLE_MATRIX_TYPE, 0),
        (DOUBLE_MATRIX_PTR_TYPE, 0),
        (INT_TYPE, 1),
        (INT_PTR_TYPE, 1),
        (BOOL_TYPE, 1),
        (BOOL_PTR_TYPE, 1),
    ], ids=str)
    def test_helper_is_scalar(self, ir_type, expected):
        assert pnlvm.helpers.is_scalar(ir_type) == expected
        assert pnlvm.helpers.is_scalar(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 1),
        (FLOAT_PTR_TYPE, 1),
        (DOUBLE_TYPE, 1),
        (DOUBLE_PTR_TYPE, 1),
        (DOUBLE_VECTOR_TYPE, 0),
        (DOUBLE_VECTOR_PTR_TYPE, 0),
        (DOUBLE_MATRIX_TYPE, 0),
        (DOUBLE_MATRIX_PTR_TYPE, 0),
        (INT_TYPE, 0),
        (INT_PTR_TYPE, 0),
        (BOOL_TYPE, 0),
        (BOOL_PTR_TYPE, 0),
    ], ids=str)
    def test_helper_is_floating_point(self, ir_type, expected):
        assert pnlvm.helpers.is_floating_point(ir_type) == expected
        assert pnlvm.helpers.is_floating_point(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 0),
        (FLOAT_PTR_TYPE, 0),
        (DOUBLE_TYPE, 0),
        (DOUBLE_PTR_TYPE, 0),
        (DOUBLE_VECTOR_TYPE, 0),
        (DOUBLE_VECTOR_PTR_TYPE, 0),
        (DOUBLE_MATRIX_TYPE, 0),
        (DOUBLE_MATRIX_PTR_TYPE, 0),
        (INT_TYPE, 1),
        (INT_PTR_TYPE, 1),
        (BOOL_TYPE, 1),
        (BOOL_PTR_TYPE, 1),
    ], ids=str)
    def test_helper_is_integer(self, ir_type, expected):
        assert pnlvm.helpers.is_integer(ir_type) == expected
        assert pnlvm.helpers.is_integer(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 0),
        (FLOAT_PTR_TYPE, 0),
        (DOUBLE_TYPE, 0),
        (DOUBLE_PTR_TYPE, 0),
        (DOUBLE_VECTOR_TYPE, 1),
        (DOUBLE_VECTOR_PTR_TYPE, 1),
        (DOUBLE_MATRIX_TYPE, 0),
        (DOUBLE_MATRIX_PTR_TYPE, 0),
        (INT_TYPE, 0),
        (INT_PTR_TYPE, 0),
        (BOOL_TYPE, 0),
        (BOOL_PTR_TYPE, 0),
    ], ids=str)
    def test_helper_is_vector(self, ir_type, expected):
        assert pnlvm.helpers.is_vector(ir_type) == expected
        assert pnlvm.helpers.is_vector(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 0),
        (FLOAT_PTR_TYPE, 0),
        (DOUBLE_TYPE, 0),
        (DOUBLE_PTR_TYPE, 0),
        (DOUBLE_VECTOR_TYPE, 0),
        (DOUBLE_VECTOR_PTR_TYPE, 0),
        (DOUBLE_MATRIX_TYPE, 1),
        (DOUBLE_MATRIX_PTR_TYPE, 1),
        (INT_TYPE, 0),
        (INT_PTR_TYPE, 0),
        (BOOL_TYPE, 0),
        (BOOL_PTR_TYPE, 0),
    ], ids=str)
    def test_helper_is_2d_matrix(self, ir_type, expected):
        assert pnlvm.helpers.is_2d_matrix(ir_type) == expected
        assert pnlvm.helpers.is_2d_matrix(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (FLOAT_TYPE, 0),
        (FLOAT_PTR_TYPE, 0),
        (DOUBLE_TYPE, 0),
        (DOUBLE_PTR_TYPE, 0),
        (DOUBLE_VECTOR_TYPE, 0),
        (DOUBLE_VECTOR_PTR_TYPE, 0),
        (DOUBLE_MATRIX_TYPE, 0),
        (DOUBLE_MATRIX_PTR_TYPE, 0),
        (INT_TYPE, 0),
        (INT_PTR_TYPE, 0),
        (BOOL_TYPE, 1),
        (BOOL_PTR_TYPE, 1),
    ], ids=str)
    def test_helper_is_boolean(self, ir_type, expected):
        assert pnlvm.helpers.is_boolean(ir_type) == expected
        assert pnlvm.helpers.is_boolean(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,expected', [
        (DOUBLE_VECTOR_TYPE, [1]),
        (DOUBLE_VECTOR_PTR_TYPE, [1]),
        (DOUBLE_MATRIX_TYPE, [1, 1]),
        (DOUBLE_MATRIX_PTR_TYPE, [1, 1]),
    ], ids=str)
    def test_helper_get_array_shape(self, ir_type, expected):
        assert pnlvm.helpers.get_array_shape(ir_type(None)) == expected

    @pytest.mark.llvm
    @pytest.mark.parametrize('ir_type,shape', [
        (DOUBLE_VECTOR_TYPE, (1,)),
        (DOUBLE_MATRIX_TYPE, (1,1)),
    ], ids=str)
    def test_helper_array_from_shape(self, ir_type, shape):
        assert ir_type == pnlvm.helpers.array_from_shape(shape, self.DOUBLE_TYPE)

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('op,var,expected', [
    (pnlvm.helpers.tanh, 1.0, 0.7615941559557649),
    (pnlvm.helpers.exp, 1.0, 2.718281828459045),
    (pnlvm.helpers.coth, 1.0, 1.3130352854993313),
    (pnlvm.helpers.csch, 1.0, 0.8509181282393215),
])
def test_helper_numerical(mode, op, var, expected):
    with pnlvm.LLVMBuilderContext() as ctx:
        func_ty = ir.FunctionType(ir.VoidType(), [ctx.float_ty.as_pointer()])

        custom_name = ctx.get_unique_name("numerical")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        in_out = function.args[0]
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        variable = builder.load(in_out)
        result = op(ctx, builder, variable)
        builder.store(result, in_out)

        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        res = bin_f.byref_arg_types[0](var)
        bin_f(ctypes.byref(res))
        res = res.value
    else:
        # FIXME: this needs to consider ctx.float_ty
        res = np.array([var], dtype=np.float64)
        bin_f.cuda_wrap_call(res)
        res = res[0]

    assert res == expected

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('var,expected', [
    (np.array([1,2,3], dtype=np.float), np.array([2,3,4], dtype=np.float)),
    (np.array([[1,2],[3,4]], dtype=np.float), np.array([[2,3],[4,5]], dtype=np.float)),
], ids=["vector", "matrix"])
def test_helper_elementwise_op(mode, var, expected):
    with pnlvm.LLVMBuilderContext() as ctx:
        arr_ptr_ty = ctx.convert_python_struct_to_llvm_ir(var).as_pointer()

        func_ty = ir.FunctionType(ir.VoidType(), [arr_ptr_ty, arr_ptr_ty])

        custom_name = ctx.get_unique_name("elementwise_op")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        inp, out = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        pnlvm.helpers.call_elementwise_operation(ctx, builder, inp,
            lambda ctx, builder, x: builder.fadd(x.type(1.0), x), out)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_vec = np.ctypeslib.as_ctypes(var)
        res = bin_f.byref_arg_types[1]()
        bin_f(ct_vec, ctypes.byref(res))
    else:
        res = copy.deepcopy(var)
        bin_f.cuda_wrap_call(var, res)

    assert np.array_equal(res, expected)

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('var1,var2,expected', [
    (np.array([1.,2.,3.]), np.array([1.,2.,3.]), np.array([2.,4.,6.])),
    (np.array([1.,2.,3.]), np.array([0.,1.,2.]), np.array([1.,3.,5.])),
    (np.array([[1.,2.,3.],
               [4.,5.,6.],
               [7.,8.,9.]]),
     np.array([[10.,11.,12.],
               [13.,14.,15.],
               [16.,17.,18.]]),
     np.array([[11.,13.,15.],
               [17.,19.,21.],
               [23.,25.,27.]])),
])
def test_helper_recursive_iterate_arrays(mode, var1, var2, expected):
    with pnlvm.LLVMBuilderContext() as ctx:
        arr_ptr_ty = ctx.convert_python_struct_to_llvm_ir(var1).as_pointer()

        func_ty = ir.FunctionType(ir.VoidType(), [arr_ptr_ty, arr_ptr_ty, arr_ptr_ty])

        custom_name = ctx.get_unique_name("elementwise_op")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        u, v, out = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        for (a_ptr, b_ptr, o_ptr) in pnlvm.helpers.recursive_iterate_arrays(ctx, builder, u, v, out):
            a = builder.load(a_ptr)
            b = builder.load(b_ptr)
            builder.store(builder.fadd(a,b), o_ptr)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    if mode == 'CPU':
        ct_vec = np.ctypeslib.as_ctypes(var1)
        ct_vec_2 = np.ctypeslib.as_ctypes(var2)
        res = bin_f.byref_arg_types[2]()
        bin_f(ct_vec, ct_vec_2, ctypes.byref(res))
    else:
        res = copy.deepcopy(var1)
        bin_f.cuda_wrap_call(var1, var2, res)

    assert np.array_equal(res, expected)
