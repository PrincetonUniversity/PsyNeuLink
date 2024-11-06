import ctypes
import ctypes.util
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
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
def test_helper_fclamp(mode):

    with pnlvm.LLVMBuilderContext.get_current() as ctx:
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

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name, ctype_ptr_args=(0, 2))
    local_vec = VECTOR.copy()

    if mode == 'CPU':
        ct_vec = local_vec.ctypes.data_as(bin_f.c_func.argtypes[0])
        ct_bounds = bounds.ctypes.data_as(bin_f.c_func.argtypes[2])

        bin_f(ct_vec, DIM_X, ct_bounds)
    else:
        bin_f.cuda_wrap_call(local_vec, np.int32(DIM_X), bounds)

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
def test_helper_fclamp_const(mode):

    with pnlvm.LLVMBuilderContext.get_current() as ctx:
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

    local_vec = VECTOR.copy()
    ref = np.clip(VECTOR, TST_MIN, TST_MAX)

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name, ctype_ptr_args=(0,))
    if mode == 'CPU':
        ct_vec = local_vec.ctypes.data_as(bin_f.c_func.argtypes[0])

        bin_f(ct_vec, DIM_X)
    else:
        bin_f.cuda_wrap_call(local_vec, np.int32(DIM_X))

    assert np.array_equal(local_vec, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('rtol,atol', [[0, 0], [None, None], [None, 100], [2, None]])
@pytest.mark.parametrize('var1,var2',
                         [[1, 1], [1, 100], [1,2], [-4,5], [0, -100], [-1,-2],
                          [[1,1,1,-4,0,-1], [1,100,2,5,-100,-2]]
                         ])
@pytest.mark.parametrize('fp_type', [ir.DoubleType, ir.FloatType])
def test_helper_is_close(mode, var1, var2, rtol, atol, fp_type):

    # Instantiate LLVMBuilderContext using the preferred fp type
    pnlvm.builder_context.LLVMBuilderContext(fp_type())

    tolerance = {}
    if rtol is not None:
        tolerance['rtol'] = rtol
    if atol is not None:
        tolerance['atol'] = atol

    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        float_ptr_ty = ctx.float_ty.as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), [float_ptr_ty, float_ptr_ty, float_ptr_ty, ctx.int32_ty])

        custom_name = ctx.get_unique_name("is_close")
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
            close = pnlvm.helpers.is_close(ctx, b1, val1, val2, **tolerance)
            out_ptr = b1.gep(out, [index])
            out_val = b1.select(close, out_ptr.type.pointee(1), out_ptr.type.pointee(0))
            b1.store(out_val, out_ptr)

        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name, ctype_ptr_args=(0, 1, 2))

    vec1 = np.atleast_1d(np.asfarray(var1, dtype=bin_f.np_arg_dtypes[0].base))
    vec2 = np.atleast_1d(np.asfarray(var2, dtype=bin_f.np_arg_dtypes[1].base))
    assert len(vec1) == len(vec2)
    res = np.empty_like(vec2)

    ref = np.isclose(vec1, vec2, **tolerance)
    if mode == 'CPU':
        ct_vec1 = vec1.ctypes.data_as(bin_f.c_func.argtypes[0])
        ct_vec2 = vec2.ctypes.data_as(bin_f.c_func.argtypes[1])
        ct_res = res.ctypes.data_as(bin_f.c_func.argtypes[2])

        bin_f(ct_vec1, ct_vec2, ct_res, len(res))
    else:
        bin_f.cuda_wrap_call(vec1, vec2, res, np.int32(DIM_X))

    assert np.array_equal(res, ref)


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('rtol,atol', [[0, 0], [None, None], [None, 100], [2, None]])
@pytest.mark.parametrize('var1,var2',
                         [[1, 1], [1, 100], [1,2], [-4,5], [0, -100], [-1,-2],
                          [[1,1,1,-4,0,-1], [1,100,2,5,-100,-2]]
                         ])
def test_helper_all_close(mode, var1, var2, atol, rtol):

    tolerance = {}
    if rtol is not None:
        tolerance['rtol'] = rtol
    if atol is not None:
        tolerance['atol'] = atol

    vec1 = np.atleast_1d(np.asfarray(var1))
    vec2 = np.atleast_1d(np.asfarray(var2))
    assert len(vec1) == len(vec2)

    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        arr_ptr_ty = ir.ArrayType(ir.DoubleType(), len(vec1)).as_pointer()
        func_ty = ir.FunctionType(ir.VoidType(), [arr_ptr_ty, arr_ptr_ty, ir.IntType(32).as_pointer()])

        custom_name = ctx.get_unique_name("all_close")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        in1, in2, out = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        all_close = pnlvm.helpers.all_close(ctx, builder, in1, in2, **tolerance)
        res = builder.select(all_close, out.type.pointee(1), out.type.pointee(0))
        builder.store(res, out)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    res = bin_f.np_buffer_for_arg(2)

    ref = np.allclose(vec1, vec2, **tolerance)

    if mode == 'CPU':
        bin_f(vec1, vec2, res)
    else:
        bin_f.cuda_wrap_call(vec1, vec2, res)

    assert np.array_equal(res, ref)

@pytest.mark.llvm
@pytest.mark.parametrize("ir_argtype,format_spec,values_to_check", [
    pytest.param(pnlvm.ir.IntType(32), "%u", range(0, 20), id="i32"),
    pytest.param(pnlvm.ir.IntType(64), "%lld", [int(-4E10), int(-3E10), int(-2E10)], id="i64"),
    pytest.param(pnlvm.ir.DoubleType(), "%lf", [x *.5 for x in range(0, 5)], id="double"),
    pytest.param(pnlvm.ir.FloatType(), "%lf", [x *.5 for x in range(0, 5)], id="float"),
    ])
def test_helper_printf(capfd, ir_argtype, format_spec, values_to_check):

    format_str = f"Hello {(format_spec + ' ') * len(values_to_check)}\n"

    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        func_ty = ir.FunctionType(ir.VoidType(), [])
        ir_values_to_check = [ir_argtype(i) for i in values_to_check]
        custom_name = ctx.get_unique_name("test_printf")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        pnlvm.helpers.printf(ctx, builder, format_str, *ir_values_to_check, tags={"always"})
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)
    bin_f()

    # Printf is buffered in libc.
    libc = ctypes.util.find_library("msvcrt" if sys.platform == "win32" else "c")
    libc = ctypes.CDLL(libc)
    libc.fflush(0)

    # Convert format specifier to Python compatible
    python_format_spec = {"%lld":"%ld"}.get(format_spec, format_spec)

    # The string below omits the newline character used above and the check below
    # uses 'startswith' to avoid issues with different newline encoding across OSes.
    python_format_str = f"Hello {(python_format_spec + ' ') * len(values_to_check)}"
    assert capfd.readouterr().out.startswith(python_format_str % tuple(values_to_check))

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
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('op,var,expected', [
    (pnlvm.helpers.tanh, 1.0, 0.7615941559557649),
    (pnlvm.helpers.exp, 1.0, 2.718281828459045),
    (pnlvm.helpers.coth, 1.0, 1.3130352854993313),
    (pnlvm.helpers.csch, 1.0, 0.8509181282393215),
    (pnlvm.helpers.log, 1.0, 0.0),
    (pnlvm.helpers.log1p, 1.0, 0.6931471805599453),
])
@pytest.mark.parametrize('fp_type', [pnlvm.ir.DoubleType(), pnlvm.ir.FloatType()], ids=str)
def test_helper_numerical(mode, op, var, expected, fp_type):
    with pnlvm.LLVMBuilderContext(fp_type) as ctx:
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

    res = np.asfarray(var, dtype=bin_f.np_arg_dtypes[0])

    if mode == 'CPU':
        bin_f(res)
    else:
        bin_f.cuda_wrap_call(res)

    np.testing.assert_allclose(res, expected)

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('var,expected', [
    (np.asfarray([1,2,3]), np.asfarray([2,3,4])),
    (np.asfarray([[1,2],[3,4]]), np.asfarray([[2,3],[4,5]])),
], ids=["vector", "matrix"])
def test_helper_elementwise_op(mode, var, expected):
    with pnlvm.LLVMBuilderContext.get_current() as ctx:
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

    vec = np.asfarray(var, dtype=bin_f.np_arg_dtypes[0].base)
    res = bin_f.np_buffer_for_arg(1)

    if mode == 'CPU':
        bin_f(vec, res)
    else:
        bin_f.cuda_wrap_call(vec, res)

    assert np.array_equal(res, expected)

@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
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
    with pnlvm.LLVMBuilderContext.get_current() as ctx:
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

    vec1 = np.asfarray(var1, dtype=bin_f.np_arg_dtypes[0].base)
    vec2 = np.asfarray(var2, dtype=bin_f.np_arg_dtypes[1].base)
    res = bin_f.np_buffer_for_arg(2)

    if mode == 'CPU':
        bin_f(vec1, vec2, res)
    else:
        bin_f.cuda_wrap_call(vec1, vec2, res)

    assert np.array_equal(res, expected)


_fp_types = [ir.DoubleType, ir.FloatType, ir.HalfType]


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('t1', _fp_types)
@pytest.mark.parametrize('t2', _fp_types)
@pytest.mark.parametrize('val', [1.0, '-Inf', 'Inf', 'NaN', 16777216, 16777217, -1.0])
def test_helper_convert_fp_type(t1, t2, mode, val):
    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        func_ty = ir.FunctionType(ir.VoidType(), [t1().as_pointer(), t2().as_pointer()])
        custom_name = ctx.get_unique_name("fp_convert")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        x, y = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        x_val = builder.load(x)
        conv_x = pnlvm.helpers.convert_type(builder, x_val, y.type.pointee)
        builder.store(conv_x, y)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)

    # Get the argument numpy dtype
    np_dt1, np_dt2 = (np.dtype(bin_f.np_arg_dtypes[i]) for i in (0, 1))

    # instantiate value, result and reference
    x = np.asfarray(val, dtype=np_dt1)
    y = bin_f.np_buffer_for_arg(1)
    ref = x.astype(np_dt2)

    if mode == 'CPU':
        bin_f(x, y)
    else:
        bin_f.cuda_wrap_call(x, y)

    np.testing.assert_allclose(y, ref, equal_nan=True)


_int_types = [ir.IntType(64), ir.IntType(32), ir.IntType(16), ir.IntType(8)]


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU', pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('t1', _int_types, ids=str)
@pytest.mark.parametrize('t2', _int_types, ids=str)
@pytest.mark.parametrize('val', [0, 1, -1, 127, -128, 255, -32768, 32767, 65535, np.iinfo(np.int32).min, np.iinfo(np.int32).max])
def test_helper_convert_int_type(t1, t2, mode, val):
    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        func_ty = ir.FunctionType(ir.VoidType(), [t1.as_pointer(), t2.as_pointer()])
        custom_name = ctx.get_unique_name("int_convert")
        function = ir.Function(ctx.module, func_ty, name=custom_name)
        x, y = function.args
        block = function.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        x_val = builder.load(x)
        conv_x = pnlvm.helpers.convert_type(builder, x_val, y.type.pointee)
        builder.store(conv_x, y)
        builder.ret_void()

    bin_f = pnlvm.LLVMBinaryFunction.get(custom_name)

    # Get the argument numpy dtype
    np_dt1, np_dt2 = (np.dtype(bin_f.np_arg_dtypes[i]) for i in (0, 1))

    # instantiate value, result and reference
    x = np.asarray(val).astype(np_dt1)
    y = bin_f.np_buffer_for_arg(1)
    ref = x.astype(np_dt2)

    if mode == 'CPU':
        bin_f(x, y)
    else:
        bin_f.cuda_wrap_call(x, y)

    np.testing.assert_array_equal(y, ref)
