# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM builtins **************************************************************

from llvmlite import ir


from . import helpers
from .builder_context import LLVMBuilderContext, _BUILTIN_PREFIX


def _setup_builtin_func_builder(ctx, name, args, *, return_type=ir.VoidType()):
    builder = ctx.create_llvm_function(args, None, _BUILTIN_PREFIX + name,
                                       return_type=return_type)

    # Add noalias attribute
    for a in builder.function.args:
        if isinstance(a.type, ir.PointerType):
            a.attributes.add('noalias')

    return builder


def setup_vxm(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()
    # Arguments (given a vector of size X, and X by Y matrix):
    # 1) Vector ptr
    # 2) Matrix ptr
    # 3) X dimension size
    # 4) Y dimension size
    # 5) Output vector pointer
    builder = _setup_builtin_func_builder(ctx, "vxm", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    v, m, x, y, o = builder.function.args

    # zero the output array
    with helpers.for_loop_zero_inc(builder, y, "zero") as (b1, index):
        ptr = b1.gep(o, [index])
        b1.store(ctx.float_ty(0), ptr)

    # Multiplication
    with helpers.for_loop_zero_inc(builder, x, "vxm_outer") as (b1, index_i):
        with helpers.for_loop_zero_inc(b1, y, "vxm_inner") as (b2, index_j):
            # Multiplication and accumulation
            vector_ptr = builder.gep(v, [index_i])
            matrix_index = builder.mul(index_i, y)
            matrix_index = builder.add(matrix_index, index_j)
            matrix_ptr = builder.gep(m, [matrix_index])
            out_ptr = builder.gep(o, [index_j])

            vector_el = builder.load(vector_ptr)
            matrix_el = builder.load(matrix_ptr)
            out_el = builder.load(out_ptr)

            new_el = builder.fmul(vector_el, matrix_el)
            new_el = builder.fadd(new_el, out_el)

            builder.store(new_el, out_ptr)

    builder.ret_void()


def setup_vxm_transposed(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()
    # Arguments (given a vector of size Y, and X by Y matrix):
    # 1) Vector ptr
    # 2) Matrix ptr
    # 3) X dimension size
    # 4) Y dimension size
    # 5) Output vector pointer
    builder = _setup_builtin_func_builder(ctx, "vxm_transposed", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    v, m, x, y, o = builder.function.args

    # zero the output array
    with helpers.for_loop_zero_inc(builder, x, "zero") as (b1, index):
        ptr = b1.gep(o, [index])
        b1.store(ctx.float_ty(0), ptr)

    # Multiplication
    with helpers.for_loop_zero_inc(builder, x, "trans_vxm_outer") as (b1, index_j):
        with helpers.for_loop_zero_inc(b1, y, "trans_vxm_inner") as (b2, index_i):

            # Multiplication and accumulation
            vector_ptr = builder.gep(v, [index_i])
            matrix_index = builder.mul(index_j, y)
            matrix_index = builder.add(matrix_index, index_i)
            matrix_ptr = builder.gep(m, [matrix_index])
            out_ptr = builder.gep(o, [index_j])

            vector_el = builder.load(vector_ptr)
            matrix_el = builder.load(matrix_ptr)
            out_el = builder.load(out_ptr)

            new_el = builder.fmul(vector_el, matrix_el)
            new_el = builder.fadd(new_el, out_el)

            builder.store(new_el, out_ptr)

    builder.ret_void()


# Setup vector addition builtin
def setup_vec_add(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector addition func
    # param1: ptr to vector 1
    # param2: ptr to vector 2
    # param3: sizeof vectors (must be the same)
    # param4: ptr to output vector (make sure this is same size as param3)
    builder = _setup_builtin_func_builder(ctx, "vec_add", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, double_ptr_ty))
    u, v, x, o = builder.function.args

    # Addition
    with helpers.for_loop_zero_inc(builder, x, "addition") as (b1, index):
        u_ptr = b1.gep(u, [index])
        v_ptr = b1.gep(v, [index])
        o_ptr = b1.gep(o, [index])
        u_val = b1.load(u_ptr)
        v_val = b1.load(v_ptr)

        u_v_sum = b1.fadd(u_val, v_val)
        b1.store(u_v_sum, o_ptr)

    builder.ret_void()

# Setup vector sum builtin
def setup_vec_sum(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector sum func (i.e. sum(vec))
    # param1: ptr to vector 1
    # param2: sizeof vector
    # param3: scalar output ptr

    builder = _setup_builtin_func_builder(ctx, "vec_sum", (double_ptr_ty, ctx.int32_ty, double_ptr_ty))
    u, x, o = builder.function.args

    # Sum
    builder.store(ctx.float_ty(-0), o)
    with helpers.for_loop_zero_inc(builder, x, "sum") as (b1, index):
        u_ptr = b1.gep(u, [index])
        u_val = b1.load(u_ptr)
        u_sum = b1.fadd(u_val, builder.load(o))
        b1.store(u_sum, o)

    builder.ret_void()


# Setup vector subtraction builtin
def setup_vec_sub(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector addition func
    # param1: ptr to vector 1
    # param2: ptr to vector 2
    # param3: sizeof vectors (must be the same)
    # param4: ptr to output vector (make sure this is same size as param3)
    builder = _setup_builtin_func_builder(ctx, "vec_sub", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, double_ptr_ty))
    u, v, x, o = builder.function.args

    # Subtraction
    with helpers.for_loop_zero_inc(builder, x, "subtraction") as (b1, index):
        u_ptr = b1.gep(u, [index])
        v_ptr = b1.gep(v, [index])
        o_ptr = b1.gep(o, [index])
        u_val = b1.load(u_ptr)
        v_val = b1.load(v_ptr)

        u_v_sum = b1.fsub(u_val, v_val)
        b1.store(u_v_sum, o_ptr)

    builder.ret_void()


# Setup vector hadamard product (ie elementwise product)
def setup_vec_hadamard(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector addition func
    # param1: ptr to vector 1
    # param2: ptr to vector 2
    # param3: sizeof vectors (must be the same)
    # param4: ptr to output vector (make sure this is same size as param3)
    builder = _setup_builtin_func_builder(ctx, "vec_hadamard", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, double_ptr_ty))
    u, v, x, o = builder.function.args

    # Hadamard
    with helpers.for_loop_zero_inc(builder, x, "mult") as (b1, index):
        u_ptr = b1.gep(u, [index])
        v_ptr = b1.gep(v, [index])
        o_ptr = b1.gep(o, [index])
        u_val = b1.load(u_ptr)
        v_val = b1.load(v_ptr)

        u_v_product = b1.fmul(u_val, v_val)
        b1.store(u_v_product, o_ptr)

    builder.ret_void()


# vec multiply by scalar constant
def setup_vec_scalar_mult(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector addition func
    # param1: ptr to vector 1
    # param2: scalar to multiply by
    # param3: sizeof vectors (must be the same)
    # param4: ptr to output vector (make sure this is same size as param3)
    builder = _setup_builtin_func_builder(ctx, "vec_scalar_mult", (double_ptr_ty, ctx.float_ty, ctx.int32_ty, double_ptr_ty))
    u, s, x, o = builder.function.args

    # mult
    with helpers.for_loop_zero_inc(builder, x, "scalar_mult_loop") as (b1, index):
        u_ptr = b1.gep(u, [index])
        o_ptr = b1.gep(o, [index])
        u_val = b1.load(u_ptr)
        u_product = b1.fmul(u_val, s)
        b1.store(u_product, o_ptr)

    builder.ret_void()


# hadamard multiplication for matrices
def setup_mat_scalar_mult(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector magnitude func
    # param1: ptr to matrix 1
    # param2: scalar
    # param3: dim_x of matrix
    # param4: dim_y of matrix
    # param5: output ptr
    builder = _setup_builtin_func_builder(ctx, "mat_scalar_mult", (double_ptr_ty, ctx.float_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    m1, s, dim_x, dim_y, o = builder.function.args

    with helpers.for_loop_zero_inc(builder, dim_x, "zero") as (b1, x):
        with helpers.for_loop_zero_inc(b1, dim_y, "zero_inner") as (b2, y):
            matrix_index = b2.mul(x, dim_y)
            matrix_index = b2.add(matrix_index, y)

            m1_ptr = b2.gep(m1, [matrix_index])
            o_ptr = b2.gep(o, [matrix_index])

            m1_val = b2.load(m1_ptr)
            o_val = b2.fmul(s, m1_val)

            b2.store(o_val, o_ptr)

    builder.ret_void()


# scalar add a value to a matrix
def setup_mat_scalar_add(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector magnitude func
    # param1: ptr to matrix 1
    # param2: scalar
    # param3: dim_x of matrix
    # param4: dim_y of matrix
    # param5: output ptr
    builder = _setup_builtin_func_builder(ctx, "mat_scalar_add", (double_ptr_ty, ctx.float_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    m1, s, dim_x, dim_y, o = builder.function.args

    with helpers.for_loop_zero_inc(builder, dim_x, "mat_scalar_add_outer") as (b1, x):
        with helpers.for_loop_zero_inc(b1, dim_y, "mat_scalar_add_inner") as (b2, y):
            matrix_index = b2.mul(x, dim_y)
            matrix_index = b2.add(matrix_index, y)

            m1_ptr = b2.gep(m1, [matrix_index])
            o_ptr = b2.gep(o, [matrix_index])

            m1_val = b2.load(m1_ptr)
            o_val = b2.fadd(s, m1_val)

            b2.store(o_val, o_ptr)

    builder.ret_void()


# hadamard multiplication for matrices
def setup_mat_hadamard(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector magnitude func
    # param1: ptr to matrix 1
    # param2: ptr to matrix 2
    # param3: dim_x of matrix
    # param4: dim_y of matrix
    # param5: output ptr
    builder = _setup_builtin_func_builder(ctx, "mat_hadamard", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    m1, m2, dim_x, dim_y, o = builder.function.args

    with helpers.for_loop_zero_inc(builder, dim_x, "mat_hadamard_outer") as (b1, x):
        with helpers.for_loop_zero_inc(b1, dim_y, "mat_hadamard_inner") as (b2, y):
            matrix_index = b2.mul(x, dim_y)
            matrix_index = b2.add(matrix_index, y)
            m1_ptr = b2.gep(m1, [matrix_index])
            m2_ptr = b2.gep(m2, [matrix_index])
            o_ptr = b2.gep(o, [matrix_index])

            m1_val = b2.load(m1_ptr)
            m2_val = b2.load(m2_ptr)
            o_val = b2.fmul(m1_val, m2_val)
            b2.store(o_val, o_ptr)

    builder.ret_void()


# matrix subtraction
def setup_mat_sub(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector magnitude func
    # param1: ptr to matrix 1
    # param2: ptr to matrix 2
    # param3: dim_x of matrix
    # param4: dim_y of matrix
    # param5: output ptr
    builder = _setup_builtin_func_builder(ctx, "mat_sub", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    m1, m2, dim_x, dim_y, o = builder.function.args

    with helpers.for_loop_zero_inc(builder, dim_x, "mat_sub_outer") as (b1, x):
        with helpers.for_loop_zero_inc(b1, dim_y, "mat_sub_inner") as (b2, y):
            matrix_index = b2.mul(x, dim_y)
            matrix_index = b2.add(matrix_index, y)
            m1_ptr = b2.gep(m1, [matrix_index])
            m2_ptr = b2.gep(m2, [matrix_index])
            o_ptr = b2.gep(o, [matrix_index])

            m1_val = b2.load(m1_ptr)
            m2_val = b2.load(m2_ptr)
            o_val = b2.fsub(m1_val, m2_val)
            b2.store(o_val, o_ptr)

    builder.ret_void()


# matrix addition
def setup_mat_add(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector magnitude func
    # param1: ptr to matrix 1
    # param2: ptr to matrix 2
    # param3: dim_x of matrix
    # param4: dim_y of matrix
    # param5: output ptr
    builder = _setup_builtin_func_builder(ctx, "mat_add", (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))
    m1, m2, dim_x, dim_y, o = builder.function.args

    with helpers.for_loop_zero_inc(builder, dim_x, "zero") as (b1, x):
        with helpers.for_loop_zero_inc(b1, dim_y, "zero_inner") as (b2, y):
            matrix_index = b2.mul(x, dim_y)
            matrix_index = b2.add(matrix_index, y)
            m1_ptr = b2.gep(m1, [matrix_index])
            m2_ptr = b2.gep(m2, [matrix_index])
            o_ptr = b2.gep(o, [matrix_index])

            m1_val = b2.load(m1_ptr)
            m2_val = b2.load(m2_ptr)
            o_val = b2.fadd(m1_val, m2_val)
            b2.store(o_val, o_ptr)

    builder.ret_void()


def setup_is_close(ctx):
    # Make sure we always have fp64 variant
    for float_ty in {ctx.float_ty, ir.DoubleType()}:
        name = "is_close_{}".format(float_ty)
        builder = _setup_builtin_func_builder(ctx, name, [float_ty,
                                                          float_ty,
                                                          float_ty,
                                                          float_ty],
                                              return_type=ctx.bool_ty)
        val1, val2, rtol, atol = builder.function.args

        fabs_f = ctx.get_builtin("fabs", [val2.type])

        diff = builder.fsub(val1, val2, "is_close_diff")
        abs_diff = builder.call(fabs_f, [diff], "is_close_abs")

        abs2 = builder.call(fabs_f, [val2], "abs_val2")

        rtol = builder.fmul(rtol, abs2, "is_close_rtol")
        tol = builder.fadd(rtol, atol, "is_close_atol")
        res  = builder.fcmp_ordered("<=", abs_diff, tol, "is_close_cmp")
        builder.ret(res)


def setup_csch(ctx):
    builder = _setup_builtin_func_builder(ctx, "csch", (ctx.float_ty,),
                                          return_type=ctx.float_ty)
    x = builder.function.args[0]
    exp_f = ctx.get_builtin("exp", [x.type])
    # (2e**x)/(e**2x - 1)
    # 2/(e**x - e**-x)
    ex = builder.call(exp_f, [x])

    nx = helpers.fneg(builder, x)
    enx = builder.call(exp_f, [nx])
    den = builder.fsub(ex, enx)
    num = den.type(2)

    res = builder.fdiv(num, den)
    builder.ret(res)


def setup_tanh(ctx):
    builder = _setup_builtin_func_builder(ctx, "tanh", (ctx.float_ty,),
                                          return_type=ctx.float_ty)
    x = builder.function.args[0]
    exp_f = ctx.get_builtin("exp", [x.type])
    # (e**2x - 1)/(e**2x + 1) is faster but doesn't handle large inputs (exp -> Inf) well (Inf/Inf = NaN)
    # (1 - (2/(exp(2*x) + 1))) is a bit slower but handles large inputs better
    _2x = builder.fmul(x.type(2), x)
    e2x = builder.call(exp_f, [_2x])
    den = builder.fadd(e2x, e2x.type(1))
    res = builder.fdiv(den.type(2), den)
    res = builder.fsub(res.type(1), res)
    builder.ret(res)


def setup_coth(ctx):
    builder = _setup_builtin_func_builder(ctx, "coth", (ctx.float_ty,),
                                          return_type=ctx.float_ty)
    x = builder.function.args[0]
    exp_f = ctx.get_builtin("exp", [x.type])
    # (e**2x + 1)/(e**2x - 1) is faster but doesn't handle large inputs (exp -> Inf) well (Inf/Inf = NaN)
    # (1 + (2/(exp(2*x) - 1))) is a bit slower but handles large inputs better
    # (e**2x + 1)/(e**2x - 1)
    _2x = builder.fmul(x.type(2), x)
    e2x = builder.call(exp_f, [_2x])
    den = builder.fsub(e2x, e2x.type(1))
    res = builder.fdiv(den.type(2), den)
    res = builder.fadd(res.type(1), res)
    builder.ret(res)


def setup_pnl_intrinsics(ctx):
    # Setup types
    single_intr_ty = ir.FunctionType(ctx.float_ty, [ctx.float_ty])
    double_intr_ty = ir.FunctionType(ctx.float_ty, (ctx.float_ty, ctx.float_ty))

    # Create function declarations
    ir.Function(ctx.module, single_intr_ty, name=_BUILTIN_PREFIX + "exp")
    ir.Function(ctx.module, single_intr_ty, name=_BUILTIN_PREFIX + "log")
    ir.Function(ctx.module, double_intr_ty, name=_BUILTIN_PREFIX + "pow")



def _generate_intrinsic_wrapper(module, name, ret, args):
    intrinsic = module.declare_intrinsic("llvm." + name, list(set(args)))

    func_ty = ir.FunctionType(ret, args)
    function = ir.Function(module, func_ty, name=_BUILTIN_PREFIX + name)
    function.attributes.add('alwaysinline')
    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(function, None)
    builder.ret(builder.call(intrinsic, function.args))

def _generate_cpu_builtins_module(_float_ty):
    """Generate function wrappers for log, exp, and pow intrinsics."""
    module = ir.Module(name="cpu_builtins")
    for intrinsic in ('exp', 'log'):
        _generate_intrinsic_wrapper(module, intrinsic, _float_ty, [_float_ty])

    _generate_intrinsic_wrapper(module, "pow", _float_ty, [_float_ty, _float_ty])
    return module


_MERSENNE_N = 624
_MERSENNE_M = 397


def _setup_mt_rand_init_scalar(ctx, state_ty):
    seed_ty = state_ty.elements[0].element
    builder = _setup_builtin_func_builder(ctx, "mt_rand_init_scalar", (state_ty.as_pointer(), seed_ty))
    state, seed = builder.function.args

    array = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0)])

    # Store seed to the 0-th element
    a_0 = builder.gep(array, [ctx.int32_ty(0), ctx.int32_ty(0)])
    seed_lo = builder.and_(seed, seed.type(0xffffffff))
    seed_lo = builder.trunc(seed_lo, a_0.type.pointee)
    builder.store(seed_lo, a_0)

    # clear gauss helpers
    last_g_avail = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(2)])
    builder.store(last_g_avail.type.pointee(0), last_g_avail)
    last_g = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(3)])
    builder.store(last_g.type.pointee(0), last_g)

    with helpers.for_loop(builder,
                          ctx.int32_ty(1),
                          ctx.int32_ty(_MERSENNE_N),
                          ctx.int32_ty(1), "init_seed") as (b, i):
        a_i = b.gep(array, [ctx.int32_ty(0), i])
        i_m1 = b.sub(i, ctx.int32_ty(1))
        a_i_m1 = b.gep(array, [ctx.int32_ty(0), i_m1])
        val = b.load(a_i_m1)
        val_shift = b.lshr(val, val.type(30))

        val = b.xor(val, val_shift)
        val = b.mul(val, val.type(1812433253))
        i_ext = b.zext(i, val.type)
        val = b.add(val, i_ext)
        val = b.and_(val, val.type(0xffffffff))
        b.store(val, a_i)

    pidx = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1)])
    builder.store(pidx.type.pointee(_MERSENNE_N), pidx)
    seed_p = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(4)])
    builder.store(seed, seed_p)
    builder.ret_void()

    return builder.function


def _setup_mt_rand_init(ctx, state_ty, init_scalar):
    seed_ty = state_ty.elements[0].element
    builder = _setup_builtin_func_builder(ctx, "mt_rand_init", (state_ty.as_pointer(), seed_ty))
    state, seed = builder.function.args

    default_seed = seed.type(19650218)
    builder.call(init_scalar, [state, default_seed])

    # python considers everything to be an array
    key_array = builder.alloca(ir.ArrayType(seed.type, 1), name="key_array")
    key_p = builder.gep(key_array, [ctx.int32_ty(0), ctx.int32_ty(0)])
    builder.store(seed, key_p)

    pi = builder.alloca(ctx.int32_ty, name="pi_slot")
    builder.store(ctx.int32_ty(1), pi)
    pj = builder.alloca(ctx.int32_ty, name="pj_slot")
    builder.store(ctx.int32_ty(0), pj)
    array = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0)])
    a_0 = builder.gep(array, [ctx.int32_ty(0), ctx.int32_ty(0)])

    # This loop should go from max(N, len(key)) -> 0,
    # but we know the key length so we can hardcode it
    with helpers.for_loop_zero_inc(builder, ctx.int32_ty(_MERSENNE_N),
                                   "add_key") as (b, _):
        i = builder.load(pi)
        i_m1 = b.sub(i, ctx.int32_ty(1))
        pa_i = b.gep(array, [ctx.int32_ty(0), i])
        pa_i_m1 = b.gep(array, [ctx.int32_ty(0), i_m1])

        # Load key element
        j = b.load(pj)
        pkey = b.gep(key_array, [ctx.int32_ty(0), j])

        # Update key index
        j_new = b.add(j, ctx.int32_ty(1))
        j_ovf = b.icmp_unsigned(">=", j_new, ctx.int32_ty(1))
        j_new = b.select(j_ovf, ctx.int32_ty(0), j_new)
        b.store(j_new, pj)

        # Mix in the key
        val = b.load(pa_i_m1)
        val = b.xor(val, b.lshr(val, val.type(30)))
        val = b.mul(val, val.type(1664525))
        val = b.xor(b.load(pa_i), val)
        val = b.add(val, b.load(pkey))
        val = b.add(val, b.zext(j, val.type))
        val = b.and_(val, val.type(0xffffffff))
        b.store(val, pa_i)

        # Update the index
        i = b.add(i, ctx.int32_ty(1))
        b.store(i, pi)
        i_ovf = b.icmp_unsigned(">=", i, ctx.int32_ty(_MERSENNE_N))
        with b.if_then(i_ovf, likely=False):
            b.store(ctx.int32_ty(1), pi)
            b.store(val, a_0)

    with helpers.for_loop_zero_inc(builder,
                                   ctx.int32_ty(_MERSENNE_N - 1),
                                   "second_shuffle") as (b, _):
        i = builder.load(pi)
        i_m1 = b.sub(i, ctx.int32_ty(1))
        pa_i = b.gep(array, [ctx.int32_ty(0), i])
        pa_i_m1 = b.gep(array, [ctx.int32_ty(0), i_m1])

        val = b.load(pa_i_m1)
        val = b.xor(val, b.lshr(val, val.type(30)))
        val = b.mul(val, val.type(1566083941))
        val = b.xor(b.load(pa_i), val)
        val = b.sub(val, b.zext(i, val.type))
        val = b.and_(val, val.type(0xffffffff))
        b.store(val, pa_i)

        # Update the index
        i = b.add(i, ctx.int32_ty(1))
        b.store(i, pi)
        i_ovf = b.icmp_unsigned(">=", i, ctx.int32_ty(_MERSENNE_N))
        with b.if_then(i_ovf, likely=False):
            b.store(ctx.int32_ty(1), pi)
            b.store(val, a_0)

    # set the 0th element to INT_MIN
    builder.store(a_0.type.pointee(0x80000000), a_0)

    # store used seed
    used_seed_p = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(4)])
    builder.store(seed, used_seed_p)
    builder.ret_void()

    return builder.function


def _setup_mt_rand_integer(ctx, state_ty):
    int64_ty = ir.IntType(64)
    # Generate random number generator function.
    # It produces random 32bit numberin a 64bit word
    builder = _setup_builtin_func_builder(ctx, "mt_rand_int32", (state_ty.as_pointer(), int64_ty.as_pointer()))
    state, out = builder.function.args

    array = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0)])
    pidx = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1)])
    idx = builder.load(pidx)

    cond = builder.icmp_signed(">=", idx, ctx.int32_ty(_MERSENNE_N))
    with builder.if_then(cond, likely=False):
        mag01 = ir.ArrayType(array.type.pointee.element, 2)([0, 0x9908b0df])
        mag0 = builder.extract_value(mag01, [0])
        mag1 = builder.extract_value(mag01, [1])

        with helpers.for_loop_zero_inc(builder,
                                       ctx.int32_ty(_MERSENNE_N - _MERSENNE_M),
                                       "first_half") as (b, kk):
            pkk = b.gep(array, [ctx.int32_ty(0), kk])
            pkk_1 = b.gep(array, [ctx.int32_ty(0), b.add(kk, ctx.int32_ty(1))])

            val_kk = b.and_(b.load(pkk), pkk.type.pointee(0x80000000))
            val_kk_1 = b.and_(b.load(pkk_1), pkk_1.type.pointee(0x7fffffff))
            val = b.or_(val_kk, val_kk_1)

            val_i1 = b.and_(val, val.type(1))
            val_b = b.trunc(val_i1, ctx.bool_ty)
            val_mag = b.select(val_b, mag1, mag0)

            val_shift = b.lshr(val, val.type(1))

            kk_m = b.add(kk, ctx.int32_ty(_MERSENNE_M))
            pval_kk_m = b.gep(array, [ctx.int32_ty(0), kk_m])
            val_kk_m = b.load(pval_kk_m)

            val = b.xor(val_kk_m, val_shift)
            val = b.xor(val, val_mag)

            b.store(val, pkk)

        with helpers.for_loop(builder,
                              ctx.int32_ty(_MERSENNE_N - _MERSENNE_M),
                              ctx.int32_ty(_MERSENNE_N),
                              ctx.int32_ty(1), "second_half") as (b, kk):
            pkk = b.gep(array, [ctx.int32_ty(0), kk])
            is_last = b.icmp_unsigned("==", kk, ctx.int32_ty(_MERSENNE_N - 1))
            idx_1 = b.select(is_last, ctx.int32_ty(0), b.add(kk, ctx.int32_ty(1)))
            pkk_1 = b.gep(array, [ctx.int32_ty(0), idx_1])

            val_kk = b.and_(b.load(pkk), pkk.type.pointee(0x80000000))
            val_kk_1 = b.and_(b.load(pkk_1), pkk.type.pointee(0x7fffffff))
            val = b.or_(val_kk, val_kk_1)

            val_i1 = b.and_(val, val.type(1))
            val_b = b.trunc(val_i1, ctx.bool_ty)
            val_mag = b.select(val_b, mag1, mag0)

            val_shift = b.lshr(val, val.type(1))

            kk_m = b.add(kk, ctx.int32_ty(_MERSENNE_M - _MERSENNE_N))
            pval_kk_m = b.gep(array, [ctx.int32_ty(0), kk_m])
            val_kk_m = b.load(pval_kk_m)

            val = b.xor(val_kk_m, val_shift)
            val = b.xor(val, val_mag)

            b.store(val, pkk)

        builder.store(pidx.type.pointee(0), pidx)

    # Get pointer and update index
    idx = builder.load(pidx)
    pval = builder.gep(array, [ctx.int32_ty(0), idx])
    idx = builder.add(idx, idx.type(1))
    builder.store(idx, pidx)

    # Load and temper
    val = builder.load(pval)
    tmp = builder.lshr(val, val.type(11))
    val = builder.xor(val, tmp)

    tmp = builder.shl(val, val.type(7))
    tmp = builder.and_(tmp, tmp.type(0x9d2c5680))
    val = builder.xor(val, tmp)

    tmp = builder.shl(val, val.type(15))
    tmp = builder.and_(tmp, tmp.type(0xefc60000))
    val = builder.xor(val, tmp)

    tmp = builder.lshr(val, val.type(18))
    val = builder.xor(val, tmp)

    # val is now random 32bit integer
    val = builder.zext(val, out.type.pointee)
    builder.store(val, out)
    builder.ret_void()

    return builder.function


def _setup_mt_rand_float(ctx, state_ty, gen_int):
    """
    Mersenne Twister double prcision random number generation.

    LLVM IR implementation of the MT19937 algorithm from [0],
    also used by CPython and numpy.

    [0] http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
    """
    # Generate random float number generator function
    builder = _setup_builtin_func_builder(ctx, "mt_rand_double", (state_ty.as_pointer(), ctx.float_ty.as_pointer()))
    state, out = builder.function.args

    al = builder.alloca(gen_int.args[1].type.pointee, name="al_gen_int")
    builder.call(gen_int, [state, al])

    bl = builder.alloca(gen_int.args[1].type.pointee, name="bl_gen_int")
    builder.call(gen_int, [state, bl])

    a = builder.load(al)
    b = builder.load(bl)

    a = builder.lshr(a, a.type(5))  # 27bit random value
    b = builder.lshr(b, b.type(6))  # 26bit random value

    af = builder.uitofp(a, ctx.float_ty)
    bf = builder.uitofp(b, ctx.float_ty)

    # NOTE: The combination below could be implemented using bit ops,
    # but due to floating point rounding it'd give slightly different
    # random numbers
    val = builder.fmul(af, ctx.float_ty(67108864.0))           # Shift left 26
    val = builder.fadd(val, bf)                                # Combine
    val = builder.fdiv(val, ctx.float_ty(9007199254740992.0))  # Scale

    # The value is in interval [0, 1)
    lower_bound = builder.fcmp_ordered(">=", val, val.type(0.0))
    builder.assume(lower_bound)
    upper_bound = builder.fcmp_ordered("<", val, val.type(1.0))
    builder.assume(upper_bound)

    builder.store(val, out)
    builder.ret_void()

    return builder.function


def _setup_mt_rand_normal(ctx, state_ty, gen_float):
    """
    Generate random float from Normal distribution generator.

    The implementation uses polar method [0], same as CPython and Numpy.
    The range is -Inf to Inf.
    [0] https://en.wikipedia.org/wiki/Marsaglia_polar_method
    """
    builder = _setup_builtin_func_builder(ctx, "mt_rand_normal", (state_ty.as_pointer(), ctx.float_ty.as_pointer()))
    state, out = builder.function.args

    p_last = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(3)])
    p_last_avail = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(2)])
    last_avail = builder.load(p_last_avail)

    cond = builder.icmp_signed("==", last_avail, ctx.int32_ty(1))
    with builder.if_then(cond, likely=False):
        builder.store(builder.load(p_last), out)
        builder.store(ctx.float_ty(0), p_last)
        builder.store(p_last_avail.type.pointee(0), p_last_avail)
        builder.ret_void()

    loop_block = builder.append_basic_block("gen_loop_gauss")
    out_block = builder.append_basic_block("gen_gauss_out")

    builder.branch(loop_block)
    builder.position_at_end(loop_block)
    tmp = builder.alloca(out.type.pointee, name="mt_rand_normal_tmp")

    # X1 is in (-1, 1)
    builder.call(gen_float, [state, tmp])
    x1 = builder.load(tmp)
    x1 = builder.fmul(x1, ctx.float_ty(2.0))
    x1 = builder.fsub(x1, ctx.float_ty(1.0))

    # x2 is in (-1, 1)
    builder.call(gen_float, [state, tmp])
    x2 = builder.load(tmp)
    x2 = builder.fmul(x2, ctx.float_ty(2.0))
    x2 = builder.fsub(x2, ctx.float_ty(1.0))

    r2 = builder.fmul(x1, x1)
    r2 = builder.fadd(r2, builder.fmul(x2, x2))

    loop_cond1 = builder.fcmp_unordered(">=", r2, r2.type(1.0))
    loop_cond2 = builder.fcmp_unordered("==", r2, r2.type(0.0))
    loop_cond = builder.or_(loop_cond1, loop_cond2)
    builder.cbranch(loop_cond, loop_block, out_block).set_weights([1, 99])

    builder.position_at_end(out_block)
    log_f = ctx.get_builtin("log", [r2.type])
    f = builder.call(log_f, [r2])
    f = builder.fmul(f, f.type(-2.0))
    f = builder.fdiv(f, r2)

    sqrt_f = ctx.get_builtin("sqrt", [f.type])
    f = builder.call(sqrt_f, [f])

    val = builder.fmul(f, x2)
    builder.store(val, out)

    next_val = builder.fmul(f, x1)
    builder.store(next_val, p_last)
    builder.store(p_last_avail.type.pointee(1), p_last_avail)

    builder.ret_void()


def get_mersenne_twister_state_struct(ctx):
    return ir.LiteralStructType([
        ir.ArrayType(ctx.int32_ty, _MERSENNE_N),  # array
        ctx.int32_ty,   # index
        ctx.int32_ty,   # last_gauss available
        ctx.float_ty,   # last_gauss
        ctx.int32_ty])  # used seed


def setup_mersenne_twister(ctx):
    state_ty = get_mersenne_twister_state_struct(ctx)

    init_scalar = _setup_mt_rand_init_scalar(ctx, state_ty)
    _setup_mt_rand_init(ctx, state_ty, init_scalar)

    gen_int = _setup_mt_rand_integer(ctx, state_ty)
    gen_float = _setup_mt_rand_float(ctx, state_ty, gen_int)
    _setup_mt_rand_normal(ctx, state_ty, gen_float)


_PHILOX_DEFAULT_ROUNDS = 10
_PHILOX_DEFAULT_BUFFER_SIZE = 4
_PHILOX_INIT_A = 0x43b0d7e5
_PHILOX_MULT_A = 0x931e8875
_PHILOX_MIX_MULT_L = 0xca01f9dd
_PHILOX_MIX_MULT_R = 0x4973f715
_PHILOX_INIT_B = 0x8b51f9dd
_PHILOX_MULT_B = 0x58f38ded


def _hash_mix(builder, a, hash_const):
    val = builder.xor(a, hash_const)
    hash_const = builder.mul(hash_const, hash_const.type(_PHILOX_MULT_A))
    val = builder.mul(val, hash_const)
    # XSHIFT sizeof(uint32) * 8 // 2 == 16
    val_sh = builder.lshr(val, val.type(16))
    val = builder.xor(val, val_sh)
    return val, hash_const

def _mix(builder, a, b):
    val_a = builder.mul(a, a.type(_PHILOX_MIX_MULT_L))
    val_b = builder.mul(b, b.type(_PHILOX_MIX_MULT_R))

    val = builder.sub(val_a, val_b)
    # XSHIFT sizeof(uint32) * 8 // 2 == 16
    val_sh = builder.lshr(val, val.type(16))
    return builder.xor(val, val_sh)


def _setup_philox_rand_init(ctx, state_ty):
    seed_ty = ir.IntType(64)
    builder = _setup_builtin_func_builder(ctx, "philox_rand_init", (state_ty.as_pointer(), seed_ty))
    state, seed = builder.function.args

    # Most of the state is set to 0
    builder.store(state.type.pointee(None), state)

    # reset buffer position to max
    buffer_pos_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(4)])
    assert buffer_pos_ptr.type.pointee.width == 16
    builder.store(buffer_pos_ptr.type.pointee(_PHILOX_DEFAULT_BUFFER_SIZE),
                  buffer_pos_ptr)

    # np calls '_seed_seq.generate_state(2, np.int64) to get the key
    # the passed seed is used as an entropy array
    # for np.SeedSeq, which in turn generates 2x64 bit words to
    # use as key.

    # 1.) Generate SeedSeq entropy pool
    # 1. a) Generate assembled entropy based on the provided seed
    assembled_entropy = ir.ArrayType(ctx.int32_ty, 4)(None)
    seed_lo = builder.trunc(seed, ctx.int32_ty)
    seed_hi = builder.lshr(seed, seed.type(32))
    seed_hi = builder.trunc(seed_hi, ctx.int32_ty)

    assembled_entropy = builder.insert_value(assembled_entropy, seed_lo, 0)
    assembled_entropy = builder.insert_value(assembled_entropy, seed_hi, 1)

    # 1. b) Mix assembled entropy to the pool
    entropy_pool = ir.ArrayType(ctx.int32_ty, 4)(None)
    # any diff would be filled with 0,
    # so we might as well force the same size
    assert len(entropy_pool.type) == len(assembled_entropy.type)

    # First perturb the entropy with some magic constants
    hash_const = ctx.int32_ty(_PHILOX_INIT_A)
    for i in range(len(entropy_pool.type)):
        ent_val = builder.extract_value(assembled_entropy, i)
        new_val, hash_const = _hash_mix(builder, ent_val, hash_const)

        entropy_pool = builder.insert_value(entropy_pool, new_val, i)

    # Next perturb the entropy with itself
    for i_src in range(len(entropy_pool.type)):
        for i_dst in range(len(entropy_pool.type)):
            if i_src != i_dst:
                src_val = builder.extract_value(entropy_pool, i_src)
                dst_val = builder.extract_value(entropy_pool, i_dst)

                new_val, hash_const = _hash_mix(builder, src_val, hash_const)
                new_val = _mix(builder, dst_val, new_val)
                entropy_pool = builder.insert_value(entropy_pool, new_val, i_dst)

    # 2.) Use the mixed entropy pool to generate 2xi64 keys
    hash_const = ctx.int32_ty(_PHILOX_INIT_B)
    key_state = ir.ArrayType(ctx.int32_ty, 4)(None)
    for i in range(len(key_state.type)):
        pool_val = builder.extract_value(entropy_pool, i)
        val = builder.xor(pool_val, hash_const)
        hash_const = builder.mul(hash_const, hash_const.type(_PHILOX_MULT_B))
        val = builder.mul(val, hash_const)
        # XSHIFT sizeof(uint32) * 8 // 2 == 16
        val_sh = builder.lshr(val, val.type(16))
        val = builder.xor(val, val_sh)
        key_state = builder.insert_value(key_state, val, i)

    key_state_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1)])
    for i in range(len(key_state_ptr.type.pointee)):
        key_ptr = builder.gep(key_state_ptr, [ctx.int32_ty(0), ctx.int32_ty(i)])
        key_lo = builder.extract_value(key_state, i * 2)
        key_lo = builder.zext(key_lo, key_ptr.type.pointee)
        key_hi = builder.extract_value(key_state, i * 2 + 1)
        key_hi = builder.zext(key_hi, key_ptr.type.pointee)
        key_hi = builder.shl(key_hi, key_hi.type(32))
        key = builder.or_(key_lo, key_hi)
        builder.store(key, key_ptr)

    # Store used seed
    used_seed_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(6)])
    builder.store(seed, used_seed_ptr)

    builder.ret_void()

    return builder.function


def _philox_encode(builder, rounds, value, key):
    assert len(value.type) == 4
    assert len(key.type) == 2

    for i in range(rounds):
        # One round of encoding
        keys = [builder.extract_value(key, j) for j in range(len(key.type))]
        vals = [builder.extract_value(value, k) for k in range(len(value.type))]
        lo0, hi0 = helpers.umul_lo_hi(builder, vals[0].type(0xD2E7470EE14C6C93), vals[0])
        lo1, hi1 = helpers.umul_lo_hi(builder, vals[2].type(0xCA5A826395121157), vals[2])

        new_vals = [None] * len(vals)
        new_vals[0] = builder.xor(hi1, vals[1])
        new_vals[0] = builder.xor(new_vals[0], keys[0])
        new_vals[1] = lo1
        new_vals[2] = builder.xor(hi0, vals[3])
        new_vals[2] = builder.xor(new_vals[2], keys[1])
        new_vals[3] = lo0
        for l, new_val in enumerate(new_vals):
            value = builder.insert_value(value, new_val, l)

        # Now bump the key
        new_key0 = builder.add(keys[0], keys[0].type(0x9E3779B97F4A7C15))
        new_key1 = builder.add(keys[1], keys[1].type(0xBB67AE8584CAA73B))
        key = builder.insert_value(key, new_key0, 0)
        key = builder.insert_value(key, new_key1, 1)

    return value


def _setup_philox_rand_int64(ctx, state_ty):
    int64_ty = ir.IntType(64)
    # Generate random number generator function.
    builder = _setup_builtin_func_builder(ctx, "philox_rand_int64", (state_ty.as_pointer(), int64_ty.as_pointer()))
    state, out = builder.function.args

    counter_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0)])
    key_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(1)])
    buffer_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(2)])
    buffer_pos_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(4)])

    assert buffer_pos_ptr.type.pointee.width == 16

    # Check if there is a pre-generated value
    buffer_pos = builder.load(buffer_pos_ptr)
    already_generated = builder.icmp_unsigned("<", buffer_pos, buffer_pos.type(len(buffer_ptr.type.pointee)))
    with builder.if_then(already_generated, likely=True):
        # Get value from pre-generated buffer
        val_ptr = builder.gep(buffer_ptr, [ctx.int32_ty(0), buffer_pos])
        builder.store(builder.load(val_ptr), out)

        # Update buffer position
        buffer_pos = builder.add(buffer_pos, buffer_pos.type(1))
        builder.store(buffer_pos, buffer_pos_ptr)
        builder.ret_void()


    # Generate 4 new numbers

    # "counter" is 256 bit wide split into 4 64b integers.
    # field i should only be incremented if all fields <i
    # were incremented and wrapped around.
    cond = ctx.bool_ty(True)
    for i in range(len(counter_ptr.type.pointee)):
        counter_el_ptr = builder.gep(counter_ptr, [ctx.int32_ty(0), ctx.int32_ty(i)])
        counter_el = builder.load(counter_el_ptr)
        new_counter = builder.add(counter_el, counter_el.type(1))
        with builder.if_then(cond):
            builder.store(new_counter, counter_el_ptr)

        carry = builder.icmp_unsigned("==", new_counter, new_counter.type(0))
        cond = builder.and_(cond, carry)

    # generate 4 new numbers by encrypting the counter using 'key'
    counter = builder.load(counter_ptr)
    key = builder.load(key_ptr)
    new_buffer = _philox_encode(builder, _PHILOX_DEFAULT_ROUNDS, counter, key)

    # Store the newly generated numbers
    builder.store(new_buffer, buffer_ptr)

    # Return the first one and set the counter
    builder.store(buffer_pos.type(1), buffer_pos_ptr)
    val = builder.extract_value(new_buffer, 0)
    builder.store(val, out)

    builder.ret_void()

    return builder.function


def _setup_philox_rand_int32(ctx, state_ty, gen_int64):
    # Generate random number generator function.
    builder = _setup_builtin_func_builder(ctx, "philox_rand_int32", (state_ty.as_pointer(), ctx.int32_ty.as_pointer()))
    state, out = builder.function.args

    buffered_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(3)])
    has_buffered_ptr = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(5)])
    has_buffered = builder.load(has_buffered_ptr)
    with builder.if_then(has_buffered):
        buffered = builder.load(buffered_ptr)
        builder.store(buffered, out)
        builder.store(has_buffered.type(False), has_buffered_ptr)
        builder.ret_void()


    val_ptr = builder.alloca(gen_int64.args[1].type.pointee, name="rand_i64")
    builder.call(gen_int64, [state, val_ptr])
    val = builder.load(val_ptr)

    val_lo = builder.trunc(val, out.type.pointee)
    builder.store(val_lo, out)

    val_hi = builder.lshr(val, val.type(val.type.width // 2))
    val_hi = builder.trunc(val_hi, buffered_ptr.type.pointee)
    builder.store(val_hi, buffered_ptr)
    builder.store(has_buffered.type(True), has_buffered_ptr)

    builder.ret_void()

    return builder.function


def _setup_philox_rand_double(ctx, state_ty, gen_int64):
    # Generate random float number generator function
    double_ty = ir.DoubleType()
    builder = _setup_builtin_func_builder(ctx, "philox_rand_double", (state_ty.as_pointer(), double_ty.as_pointer()))
    state, out = builder.function.args

    # (rnd >> 11) * (1.0 / 9007199254740992.0)
    rhs = double_ty(1.0 / 9007199254740992.0)

    # Generate random integer
    lhs_ptr = builder.alloca(gen_int64.args[1].type.pointee, name="rand_int64")
    builder.call(gen_int64, [state, lhs_ptr])

    # convert to float
    lhs_int = builder.load(lhs_ptr)
    lhs_shift = builder.lshr(lhs_int, lhs_int.type(11))
    lhs = builder.uitofp(lhs_shift, double_ty)

    res = builder.fmul(lhs, rhs)
    builder.store(res, out)

    builder.ret_void()

    return builder.function


def _setup_philox_rand_float(ctx, state_ty, gen_int32):
    # Generate random float number generator function
    float_ty = ir.FloatType()
    builder = _setup_builtin_func_builder(ctx, "philox_rand_float", (state_ty.as_pointer(), float_ty.as_pointer()))
    state, out = builder.function.args

    # (next_uint32(bitgen_state) >> 9) * (1.0f / 8388608.0f);
    rhs = float_ty(1.0 / 8388608.0)

    # Generate random integer
    lhs_ptr = builder.alloca(gen_int32.args[1].type.pointee, name="rand_int32")
    builder.call(gen_int32, [state, lhs_ptr])

    # convert to float
    lhs_int = builder.load(lhs_ptr)
    lhs_shift = builder.lshr(lhs_int, lhs_int.type(9))
    lhs = builder.uitofp(lhs_shift, float_ty)

    res = builder.fmul(lhs, rhs)
    builder.store(res, out)

    builder.ret_void()

    return builder.function


# Taken from numpy
_wi_double_data = [
    8.68362706080130616677e-16, 4.77933017572773682428e-17,
    6.35435241740526230246e-17, 7.45487048124769627714e-17,
    8.32936681579309972857e-17, 9.06806040505948228243e-17,
    9.71486007656776183958e-17, 1.02947503142410192108e-16,
    1.08234302884476839838e-16, 1.13114701961090307945e-16,
    1.17663594570229211411e-16, 1.21936172787143633280e-16,
    1.25974399146370927864e-16, 1.29810998862640315416e-16,
    1.33472037368241227547e-16, 1.36978648425712032797e-16,
    1.40348230012423820659e-16, 1.43595294520569430270e-16,
    1.46732087423644219083e-16, 1.49769046683910367425e-16,
    1.52715150035961979750e-16, 1.55578181694607639484e-16,
    1.58364940092908853989e-16, 1.61081401752749279325e-16,
    1.63732852039698532012e-16, 1.66323990584208352778e-16,
    1.68859017086765964015e-16, 1.71341701765596607184e-16,
    1.73775443658648593310e-16, 1.76163319230009959832e-16,
    1.78508123169767272927e-16, 1.80812402857991522674e-16,
    1.83078487648267501776e-16, 1.85308513886180189386e-16,
    1.87504446393738816849e-16, 1.89668097007747596212e-16,
    1.91801140648386198029e-16, 1.93905129306251037069e-16,
    1.95981504266288244037e-16, 1.98031606831281739736e-16,
    2.00056687762733300198e-16, 2.02057915620716538808e-16,
    2.04036384154802118313e-16, 2.05993118874037063144e-16,
    2.07929082904140197311e-16, 2.09845182223703516690e-16,
    2.11742270357603418769e-16, 2.13621152594498681022e-16,
    2.15482589785814580926e-16, 2.17327301775643674990e-16,
    2.19155970504272708519e-16, 2.20969242822353175995e-16,
    2.22767733047895534948e-16, 2.24552025294143552381e-16,
    2.26322675592856786566e-16, 2.28080213834501706782e-16,
    2.29825145544246839061e-16, 2.31557953510408037008e-16,
    2.33279099280043561128e-16, 2.34989024534709550938e-16,
    2.36688152357916037468e-16, 2.38376888404542434981e-16,
    2.40055621981350627349e-16, 2.41724727046750252175e-16,
    2.43384563137110286400e-16, 2.45035476226149539878e-16,
    2.46677799523270498158e-16, 2.48311854216108767769e-16,
    2.49937950162045242375e-16, 2.51556386532965786439e-16,
    2.53167452417135826983e-16, 2.54771427381694417303e-16,
    2.56368581998939683749e-16, 2.57959178339286723500e-16,
    2.59543470433517070146e-16, 2.61121704706701939097e-16,
    2.62694120385972564623e-16, 2.64260949884118951286e-16,
    2.65822419160830680292e-16, 2.67378748063236329361e-16,
    2.68930150647261591777e-16, 2.70476835481199518794e-16,
    2.72019005932773206655e-16, 2.73556860440867908686e-16,
    2.75090592773016664571e-16, 2.76620392269639032183e-16,
    2.78146444075954410103e-16, 2.79668929362423005309e-16,
    2.81188025534502074329e-16, 2.82703906432447923059e-16,
    2.84216742521840606520e-16, 2.85726701075460149289e-16,
    2.87233946347097994381e-16, 2.88738639737848191815e-16,
    2.90240939955384233230e-16, 2.91741003166694553259e-16,
    2.93238983144718163965e-16, 2.94735031409293489611e-16,
    2.96229297362806647792e-16, 2.97721928420902891115e-16,
    2.99213070138601307081e-16, 3.00702866332133102993e-16,
    3.02191459196806151971e-16, 3.03678989421180184427e-16,
    3.05165596297821922381e-16, 3.06651417830895451744e-16,
    3.08136590840829717032e-16, 3.09621251066292253306e-16,
    3.11105533263689296831e-16, 3.12589571304399892784e-16,
    3.14073498269944617203e-16, 3.15557446545280064031e-16,
    3.17041547910402852545e-16, 3.18525933630440648871e-16,
    3.20010734544401137886e-16, 3.21496081152744704901e-16,
    3.22982103703941557538e-16, 3.24468932280169778077e-16,
    3.25956696882307838340e-16, 3.27445527514370671802e-16,
    3.28935554267536967851e-16, 3.30426907403912838589e-16,
    3.31919717440175233652e-16, 3.33414115231237245918e-16,
    3.34910232054077845412e-16, 3.36408199691876507948e-16,
    3.37908150518594979994e-16, 3.39410217584148914282e-16,
    3.40914534700312603713e-16, 3.42421236527501816058e-16,
    3.43930458662583133920e-16, 3.45442337727858401604e-16,
    3.46957011461378353333e-16, 3.48474618808741370700e-16,
    3.49995300016538099813e-16, 3.51519196727607440975e-16,
    3.53046452078274009054e-16, 3.54577210797743572160e-16,
    3.56111619309838843415e-16, 3.57649825837265051035e-16,
    3.59191980508602994994e-16, 3.60738235468235137839e-16,
    3.62288744989419151904e-16, 3.63843665590734438546e-16,
    3.65403156156136995766e-16, 3.66967378058870090021e-16,
    3.68536495289491401456e-16, 3.70110674588289834952e-16,
    3.71690085582382297792e-16, 3.73274900927794352614e-16,
    3.74865296456848868882e-16, 3.76461451331202869131e-16,
    3.78063548200896037651e-16, 3.79671773369794425924e-16,
    3.81286316967837738238e-16, 3.82907373130524317507e-16,
    3.84535140186095955858e-16, 3.86169820850914927119e-16,
    3.87811622433558721164e-16, 3.89460757048192620674e-16,
    3.91117441837820542060e-16, 3.92781899208054153270e-16,
    3.94454357072087711446e-16, 3.96135049107613542983e-16,
    3.97824215026468259474e-16, 3.99522100857856502444e-16,
    4.01228959246062907451e-16, 4.02945049763632792393e-16,
    4.04670639241074995115e-16, 4.06406002114225038723e-16,
    4.08151420790493873480e-16, 4.09907186035326643447e-16,
    4.11673597380302570170e-16, 4.13450963554423599878e-16,
    4.15239602940268833891e-16, 4.17039844056831587498e-16,
    4.18852026071011229572e-16, 4.20676499339901510978e-16,
    4.22513625986204937320e-16, 4.24363780509307796137e-16,
    4.26227350434779809917e-16, 4.28104737005311666397e-16,
    4.29996355916383230161e-16, 4.31902638100262944617e-16,
    4.33824030562279080411e-16, 4.35760997273684900553e-16,
    4.37714020125858747008e-16, 4.39683599951052137423e-16,
    4.41670257615420348435e-16, 4.43674535190656726604e-16,
    4.45696997211204306674e-16, 4.47738232024753387312e-16,
    4.49798853244554968009e-16, 4.51879501313005876278e-16,
    4.53980845187003400947e-16, 4.56103584156742206384e-16,
    4.58248449810956667052e-16, 4.60416208163115281428e-16,
    4.62607661954784567754e-16, 4.64823653154320737780e-16,
    4.67065065671263059081e-16, 4.69332828309332890697e-16,
    4.71627917983835129766e-16, 4.73951363232586715165e-16,
    4.76304248053313737663e-16, 4.78687716104872284247e-16,
    4.81102975314741720538e-16, 4.83551302941152515162e-16,
    4.86034051145081195402e-16, 4.88552653135360343280e-16,
    4.91108629959526955862e-16, 4.93703598024033454728e-16,
    4.96339277440398725619e-16, 4.99017501309182245754e-16,
    5.01740226071808946011e-16, 5.04509543081872748637e-16,
    5.07327691573354207058e-16, 5.10197073234156184149e-16,
    5.13120268630678373200e-16, 5.16100055774322824569e-16,
    5.19139431175769859873e-16, 5.22241633800023428760e-16,
    5.25410172417759732697e-16, 5.28648856950494511482e-16,
    5.31961834533840037535e-16, 5.35353631181649688145e-16,
    5.38829200133405320160e-16, 5.42393978220171234073e-16,
    5.46053951907478041166e-16, 5.49815735089281410703e-16,
    5.53686661246787600374e-16, 5.57674893292657647836e-16,
    5.61789555355541665830e-16, 5.66040892008242216739e-16,
    5.70440462129138908417e-16, 5.75001376891989523684e-16,
    5.79738594572459365014e-16, 5.84669289345547900201e-16,
    5.89813317647789942685e-16, 5.95193814964144415532e-16,
    6.00837969627190832234e-16, 6.06778040933344851394e-16,
    6.13052720872528159123e-16, 6.19708989458162555387e-16,
    6.26804696330128439415e-16, 6.34412240712750598627e-16,
    6.42623965954805540945e-16, 6.51560331734499356881e-16,
    6.61382788509766415145e-16, 6.72315046250558662913e-16,
    6.84680341756425875856e-16, 6.98971833638761995415e-16,
    7.15999493483066421560e-16, 7.37242430179879890722e-16,
    7.65893637080557275482e-16, 8.11384933765648418565e-16]

# Taken from numpy
_ki_i64_data = [
    0x000EF33D8025EF6A, 0x0000000000000000, 0x000C08BE98FBC6A8,
    0x000DA354FABD8142, 0x000E51F67EC1EEEA, 0x000EB255E9D3F77E,
    0x000EEF4B817ECAB9, 0x000F19470AFA44AA, 0x000F37ED61FFCB18,
    0x000F4F469561255C, 0x000F61A5E41BA396, 0x000F707A755396A4,
    0x000F7CB2EC28449A, 0x000F86F10C6357D3, 0x000F8FA6578325DE,
    0x000F9724C74DD0DA, 0x000F9DA907DBF509, 0x000FA360F581FA74,
    0x000FA86FDE5B4BF8, 0x000FACF160D354DC, 0x000FB0FB6718B90F,
    0x000FB49F8D5374C6, 0x000FB7EC2366FE77, 0x000FBAECE9A1E50E,
    0x000FBDAB9D040BED, 0x000FC03060FF6C57, 0x000FC2821037A248,
    0x000FC4A67AE25BD1, 0x000FC6A2977AEE31, 0x000FC87AA92896A4,
    0x000FCA325E4BDE85, 0x000FCBCCE902231A, 0x000FCD4D12F839C4,
    0x000FCEB54D8FEC99, 0x000FD007BF1DC930, 0x000FD1464DD6C4E6,
    0x000FD272A8E2F450, 0x000FD38E4FF0C91E, 0x000FD49A9990B478,
    0x000FD598B8920F53, 0x000FD689C08E99EC, 0x000FD76EA9C8E832,
    0x000FD848547B08E8, 0x000FD9178BAD2C8C, 0x000FD9DD07A7ADD2,
    0x000FDA9970105E8C, 0x000FDB4D5DC02E20, 0x000FDBF95C5BFCD0,
    0x000FDC9DEBB99A7D, 0x000FDD3B8118729D, 0x000FDDD288342F90,
    0x000FDE6364369F64, 0x000FDEEE708D514E, 0x000FDF7401A6B42E,
    0x000FDFF46599ED40, 0x000FE06FE4BC24F2, 0x000FE0E6C225A258,
    0x000FE1593C28B84C, 0x000FE1C78CBC3F99, 0x000FE231E9DB1CAA,
    0x000FE29885DA1B91, 0x000FE2FB8FB54186, 0x000FE35B33558D4A,
    0x000FE3B799D0002A, 0x000FE410E99EAD7F, 0x000FE46746D47734,
    0x000FE4BAD34C095C, 0x000FE50BAED29524, 0x000FE559F74EBC78,
    0x000FE5A5C8E41212, 0x000FE5EF3E138689, 0x000FE6366FD91078,
    0x000FE67B75C6D578, 0x000FE6BE661E11AA, 0x000FE6FF55E5F4F2,
    0x000FE73E5900A702, 0x000FE77B823E9E39, 0x000FE7B6E37070A2,
    0x000FE7F08D774243, 0x000FE8289053F08C, 0x000FE85EFB35173A,
    0x000FE893DC840864, 0x000FE8C741F0CEBC, 0x000FE8F9387D4EF6,
    0x000FE929CC879B1D, 0x000FE95909D388EA, 0x000FE986FB939AA2,
    0x000FE9B3AC714866, 0x000FE9DF2694B6D5, 0x000FEA0973ABE67C,
    0x000FEA329CF166A4, 0x000FEA5AAB32952C, 0x000FEA81A6D5741A,
    0x000FEAA797DE1CF0, 0x000FEACC85F3D920, 0x000FEAF07865E63C,
    0x000FEB13762FEC13, 0x000FEB3585FE2A4A, 0x000FEB56AE3162B4,
    0x000FEB76F4E284FA, 0x000FEB965FE62014, 0x000FEBB4F4CF9D7C,
    0x000FEBD2B8F449D0, 0x000FEBEFB16E2E3E, 0x000FEC0BE31EBDE8,
    0x000FEC2752B15A15, 0x000FEC42049DAFD3, 0x000FEC5BFD29F196,
    0x000FEC75406CEEF4, 0x000FEC8DD2500CB4, 0x000FECA5B6911F12,
    0x000FECBCF0C427FE, 0x000FECD38454FB15, 0x000FECE97488C8B3,
    0x000FECFEC47F91B7, 0x000FED1377358528, 0x000FED278F844903,
    0x000FED3B10242F4C, 0x000FED4DFBAD586E, 0x000FED605498C3DD,
    0x000FED721D414FE8, 0x000FED8357E4A982, 0x000FED9406A42CC8,
    0x000FEDA42B85B704, 0x000FEDB3C8746AB4, 0x000FEDC2DF416652,
    0x000FEDD171A46E52, 0x000FEDDF813C8AD3, 0x000FEDED0F909980,
    0x000FEDFA1E0FD414, 0x000FEE06AE124BC4, 0x000FEE12C0D95A06,
    0x000FEE1E579006E0, 0x000FEE29734B6524, 0x000FEE34150AE4BC,
    0x000FEE3E3DB89B3C, 0x000FEE47EE2982F4, 0x000FEE51271DB086,
    0x000FEE59E9407F41, 0x000FEE623528B42E, 0x000FEE6A0B5897F1,
    0x000FEE716C3E077A, 0x000FEE7858327B82, 0x000FEE7ECF7B06BA,
    0x000FEE84D2484AB2, 0x000FEE8A60B66343, 0x000FEE8F7ACCC851,
    0x000FEE94207E25DA, 0x000FEE9851A829EA, 0x000FEE9C0E13485C,
    0x000FEE9F557273F4, 0x000FEEA22762CCAE, 0x000FEEA4836B42AC,
    0x000FEEA668FC2D71, 0x000FEEA7D76ED6FA, 0x000FEEA8CE04FA0A,
    0x000FEEA94BE8333B, 0x000FEEA950296410, 0x000FEEA8D9C0075E,
    0x000FEEA7E7897654, 0x000FEEA678481D24, 0x000FEEA48AA29E83,
    0x000FEEA21D22E4DA, 0x000FEE9F2E352024, 0x000FEE9BBC26AF2E,
    0x000FEE97C524F2E4, 0x000FEE93473C0A3A, 0x000FEE8E40557516,
    0x000FEE88AE369C7A, 0x000FEE828E7F3DFD, 0x000FEE7BDEA7B888,
    0x000FEE749BFF37FF, 0x000FEE6CC3A9BD5E, 0x000FEE64529E007E,
    0x000FEE5B45A32888, 0x000FEE51994E57B6, 0x000FEE474A0006CF,
    0x000FEE3C53E12C50, 0x000FEE30B2E02AD8, 0x000FEE2462AD8205,
    0x000FEE175EB83C5A, 0x000FEE09A22A1447, 0x000FEDFB27E349CC,
    0x000FEDEBEA76216C, 0x000FEDDBE422047E, 0x000FEDCB0ECE39D3,
    0x000FEDB964042CF4, 0x000FEDA6DCE938C9, 0x000FED937237E98D,
    0x000FED7F1C38A836, 0x000FED69D2B9C02B, 0x000FED538D06AE00,
    0x000FED3C41DEA422, 0x000FED23E76A2FD8, 0x000FED0A732FE644,
    0x000FECEFDA07FE34, 0x000FECD4100EB7B8, 0x000FECB708956EB4,
    0x000FEC98B61230C1, 0x000FEC790A0DA978, 0x000FEC57F50F31FE,
    0x000FEC356686C962, 0x000FEC114CB4B335, 0x000FEBEB948E6FD0,
    0x000FEBC429A0B692, 0x000FEB9AF5EE0CDC, 0x000FEB6FE1C98542,
    0x000FEB42D3AD1F9E, 0x000FEB13B00B2D4B, 0x000FEAE2591A02E9,
    0x000FEAAEAE992257, 0x000FEA788D8EE326, 0x000FEA3FCFFD73E5,
    0x000FEA044C8DD9F6, 0x000FE9C5D62F563B, 0x000FE9843BA947A4,
    0x000FE93F471D4728, 0x000FE8F6BD76C5D6, 0x000FE8AA5DC4E8E6,
    0x000FE859E07AB1EA, 0x000FE804F690A940, 0x000FE7AB488233C0,
    0x000FE74C751F6AA5, 0x000FE6E8102AA202, 0x000FE67DA0B6ABD8,
    0x000FE60C9F38307E, 0x000FE5947338F742, 0x000FE51470977280,
    0x000FE48BD436F458, 0x000FE3F9BFFD1E37, 0x000FE35D35EEB19C,
    0x000FE2B5122FE4FE, 0x000FE20003995557, 0x000FE13C82788314,
    0x000FE068C4EE67B0, 0x000FDF82B02B71AA, 0x000FDE87C57EFEAA,
    0x000FDD7509C63BFD, 0x000FDC46E529BF13, 0x000FDAF8F82E0282,
    0x000FD985E1B2BA75, 0x000FD7E6EF48CF04, 0x000FD613ADBD650B,
    0x000FD40149E2F012, 0x000FD1A1A7B4C7AC, 0x000FCEE204761F9E,
    0x000FCBA8D85E11B2, 0x000FC7D26ECD2D22, 0x000FC32B2F1E22ED,
    0x000FBD6581C0B83A, 0x000FB606C4005434, 0x000FAC40582A2874,
    0x000F9E971E014598, 0x000F89FA48A41DFC, 0x000F66C5F7F0302C,
    0x000F1A5A4B331C4A]

# Taken from numpy
_fi_double_data = [
    1.00000000000000000000e+00, 9.77101701267671596263e-01,
    9.59879091800106665211e-01, 9.45198953442299649730e-01,
    9.32060075959230460718e-01, 9.19991505039347012840e-01,
    9.08726440052130879366e-01, 8.98095921898343418910e-01,
    8.87984660755833377088e-01, 8.78309655808917399966e-01,
    8.69008688036857046555e-01, 8.60033621196331532488e-01,
    8.51346258458677951353e-01, 8.42915653112204177333e-01,
    8.34716292986883434679e-01, 8.26726833946221373317e-01,
    8.18929191603702366642e-01, 8.11307874312656274185e-01,
    8.03849483170964274059e-01, 7.96542330422958966274e-01,
    7.89376143566024590648e-01, 7.82341832654802504798e-01,
    7.75431304981187174974e-01, 7.68637315798486264740e-01,
    7.61953346836795386565e-01, 7.55373506507096115214e-01,
    7.48892447219156820459e-01, 7.42505296340151055290e-01,
    7.36207598126862650112e-01, 7.29995264561476231435e-01,
    7.23864533468630222401e-01, 7.17811932630721960535e-01,
    7.11834248878248421200e-01, 7.05928501332754310127e-01,
    7.00091918136511615067e-01, 6.94321916126116711609e-01,
    6.88616083004671808432e-01, 6.82972161644994857355e-01,
    6.77388036218773526009e-01, 6.71861719897082099173e-01,
    6.66391343908750100056e-01, 6.60975147776663107813e-01,
    6.55611470579697264149e-01, 6.50298743110816701574e-01,
    6.45035480820822293424e-01, 6.39820277453056585060e-01,
    6.34651799287623608059e-01, 6.29528779924836690007e-01,
    6.24450015547026504592e-01, 6.19414360605834324325e-01,
    6.14420723888913888899e-01, 6.09468064925773433949e-01,
    6.04555390697467776029e-01, 5.99681752619125263415e-01,
    5.94846243767987448159e-01, 5.90047996332826008015e-01,
    5.85286179263371453274e-01, 5.80559996100790898232e-01,
    5.75868682972353718164e-01, 5.71211506735253227163e-01,
    5.66587763256164445025e-01, 5.61996775814524340831e-01,
    5.57437893618765945014e-01, 5.52910490425832290562e-01,
    5.48413963255265812791e-01, 5.43947731190026262382e-01,
    5.39511234256952132426e-01, 5.35103932380457614215e-01,
    5.30725304403662057062e-01, 5.26374847171684479008e-01,
    5.22052074672321841931e-01, 5.17756517229756352272e-01,
    5.13487720747326958914e-01, 5.09245245995747941592e-01,
    5.05028667943468123624e-01, 5.00837575126148681903e-01,
    4.96671569052489714213e-01, 4.92530263643868537748e-01,
    4.88413284705458028423e-01, 4.84320269426683325253e-01,
    4.80250865909046753544e-01, 4.76204732719505863248e-01,
    4.72181538467730199660e-01, 4.68180961405693596422e-01,
    4.64202689048174355069e-01, 4.60246417812842867345e-01,
    4.56311852678716434184e-01, 4.52398706861848520777e-01,
    4.48506701507203064949e-01, 4.44635565395739396077e-01,
    4.40785034665803987508e-01, 4.36954852547985550526e-01,
    4.33144769112652261445e-01, 4.29354541029441427735e-01,
    4.25583931338021970170e-01, 4.21832709229495894654e-01,
    4.18100649837848226120e-01, 4.14387534040891125642e-01,
    4.10693148270188157500e-01, 4.07017284329473372217e-01,
    4.03359739221114510510e-01, 3.99720314980197222177e-01,
    3.96098818515832451492e-01, 3.92495061459315619512e-01,
    3.88908860018788715696e-01, 3.85340034840077283462e-01,
    3.81788410873393657674e-01, 3.78253817245619183840e-01,
    3.74736087137891138443e-01, 3.71235057668239498696e-01,
    3.67750569779032587814e-01, 3.64282468129004055601e-01,
    3.60830600989648031529e-01, 3.57394820145780500731e-01,
    3.53974980800076777232e-01, 3.50570941481406106455e-01,
    3.47182563956793643900e-01, 3.43809713146850715049e-01,
    3.40452257044521866547e-01, 3.37110066637006045021e-01,
    3.33783015830718454708e-01, 3.30470981379163586400e-01,
    3.27173842813601400970e-01, 3.23891482376391093290e-01,
    3.20623784956905355514e-01, 3.17370638029913609834e-01,
    3.14131931596337177215e-01, 3.10907558126286509559e-01,
    3.07697412504292056035e-01, 3.04501391976649993243e-01,
    3.01319396100803049698e-01, 2.98151326696685481377e-01,
    2.94997087799961810184e-01, 2.91856585617095209972e-01,
    2.88729728482182923521e-01, 2.85616426815501756042e-01,
    2.82516593083707578948e-01, 2.79430141761637940157e-01,
    2.76356989295668320494e-01, 2.73297054068577072172e-01,
    2.70250256365875463072e-01, 2.67216518343561471038e-01,
    2.64195763997261190426e-01, 2.61187919132721213522e-01,
    2.58192911337619235290e-01, 2.55210669954661961700e-01,
    2.52241126055942177508e-01, 2.49284212418528522415e-01,
    2.46339863501263828249e-01, 2.43408015422750312329e-01,
    2.40488605940500588254e-01, 2.37581574431238090606e-01,
    2.34686861872330010392e-01, 2.31804410824338724684e-01,
    2.28934165414680340644e-01, 2.26076071322380278694e-01,
    2.23230075763917484855e-01, 2.20396127480151998723e-01,
    2.17574176724331130872e-01, 2.14764175251173583536e-01,
    2.11966076307030182324e-01, 2.09179834621125076977e-01,
    2.06405406397880797353e-01, 2.03642749310334908452e-01,
    2.00891822494656591136e-01, 1.98152586545775138971e-01,
    1.95425003514134304483e-01, 1.92709036903589175926e-01,
    1.90004651670464985713e-01, 1.87311814223800304768e-01,
    1.84630492426799269756e-01, 1.81960655599522513892e-01,
    1.79302274522847582272e-01, 1.76655321443734858455e-01,
    1.74019770081838553999e-01, 1.71395595637505754327e-01,
    1.68782774801211288285e-01, 1.66181285764481906364e-01,
    1.63591108232365584074e-01, 1.61012223437511009516e-01,
    1.58444614155924284882e-01, 1.55888264724479197465e-01,
    1.53343161060262855866e-01, 1.50809290681845675763e-01,
    1.48286642732574552861e-01, 1.45775208005994028060e-01,
    1.43274978973513461566e-01, 1.40785949814444699690e-01,
    1.38308116448550733057e-01, 1.35841476571253755301e-01,
    1.33386029691669155683e-01, 1.30941777173644358090e-01,
    1.28508722279999570981e-01, 1.26086870220185887081e-01,
    1.23676228201596571932e-01, 1.21276805484790306533e-01,
    1.18888613442910059947e-01, 1.16511665625610869035e-01,
    1.14145977827838487895e-01, 1.11791568163838089811e-01,
    1.09448457146811797824e-01, 1.07116667774683801961e-01,
    1.04796225622487068629e-01, 1.02487158941935246892e-01,
    1.00189498768810017482e-01, 9.79032790388624646338e-02,
    9.56285367130089991594e-02, 9.33653119126910124859e-02,
    9.11136480663737591268e-02, 8.88735920682758862021e-02,
    8.66451944505580717859e-02, 8.44285095703534715916e-02,
    8.22235958132029043366e-02, 8.00305158146630696292e-02,
    7.78493367020961224423e-02, 7.56801303589271778804e-02,
    7.35229737139813238622e-02, 7.13779490588904025339e-02,
    6.92451443970067553879e-02, 6.71246538277884968737e-02,
    6.50165779712428976156e-02, 6.29210244377581412456e-02,
    6.08381083495398780614e-02, 5.87679529209337372930e-02,
    5.67106901062029017391e-02, 5.46664613248889208474e-02,
    5.26354182767921896513e-02, 5.06177238609477817000e-02,
    4.86135532158685421122e-02, 4.66230949019303814174e-02,
    4.46465522512944634759e-02, 4.26841449164744590750e-02,
    4.07361106559409394401e-02, 3.88027074045261474722e-02,
    3.68842156885673053135e-02, 3.49809414617161251737e-02,
    3.30932194585785779961e-02, 3.12214171919203004046e-02,
    2.93659397581333588001e-02, 2.75272356696031131329e-02,
    2.57058040085489103443e-02, 2.39022033057958785407e-02,
    2.21170627073088502113e-02, 2.03510962300445102935e-02,
    1.86051212757246224594e-02, 1.68800831525431419000e-02,
    1.51770883079353092332e-02, 1.34974506017398673818e-02,
    1.18427578579078790488e-02, 1.02149714397014590439e-02,
    8.61658276939872638800e-03, 7.05087547137322242369e-03,
    5.52240329925099155545e-03, 4.03797259336302356153e-03,
    2.60907274610215926189e-03, 1.26028593049859797236e-03]

# Taken from numpy
_ki_i32_data = [
    0x007799EC, 0x00000000, 0x006045F5, 0x006D1AA8, 0x00728FB4,
    0x007592AF, 0x00777A5C, 0x0078CA38, 0x0079BF6B, 0x007A7A35,
    0x007B0D2F, 0x007B83D4, 0x007BE597, 0x007C3788, 0x007C7D33,
    0x007CB926, 0x007CED48, 0x007D1B08, 0x007D437F, 0x007D678B,
    0x007D87DB, 0x007DA4FC, 0x007DBF61, 0x007DD767, 0x007DED5D,
    0x007E0183, 0x007E1411, 0x007E2534, 0x007E3515, 0x007E43D5,
    0x007E5193, 0x007E5E67, 0x007E6A69, 0x007E75AA, 0x007E803E,
    0x007E8A32, 0x007E9395, 0x007E9C72, 0x007EA4D5, 0x007EACC6,
    0x007EB44E, 0x007EBB75, 0x007EC243, 0x007EC8BC, 0x007ECEE8,
    0x007ED4CC, 0x007EDA6B, 0x007EDFCB, 0x007EE4EF, 0x007EE9DC,
    0x007EEE94, 0x007EF31B, 0x007EF774, 0x007EFBA0, 0x007EFFA3,
    0x007F037F, 0x007F0736, 0x007F0ACA, 0x007F0E3C, 0x007F118F,
    0x007F14C4, 0x007F17DC, 0x007F1ADA, 0x007F1DBD, 0x007F2087,
    0x007F233A, 0x007F25D7, 0x007F285D, 0x007F2AD0, 0x007F2D2E,
    0x007F2F7A, 0x007F31B3, 0x007F33DC, 0x007F35F3, 0x007F37FB,
    0x007F39F3, 0x007F3BDC, 0x007F3DB7, 0x007F3F84, 0x007F4145,
    0x007F42F8, 0x007F449F, 0x007F463A, 0x007F47CA, 0x007F494E,
    0x007F4AC8, 0x007F4C38, 0x007F4D9D, 0x007F4EF9, 0x007F504C,
    0x007F5195, 0x007F52D5, 0x007F540D, 0x007F553D, 0x007F5664,
    0x007F5784, 0x007F589C, 0x007F59AC, 0x007F5AB5, 0x007F5BB8,
    0x007F5CB3, 0x007F5DA8, 0x007F5E96, 0x007F5F7E, 0x007F605F,
    0x007F613B, 0x007F6210, 0x007F62E0, 0x007F63AA, 0x007F646F,
    0x007F652E, 0x007F65E8, 0x007F669C, 0x007F674C, 0x007F67F6,
    0x007F689C, 0x007F693C, 0x007F69D9, 0x007F6A70, 0x007F6B03,
    0x007F6B91, 0x007F6C1B, 0x007F6CA0, 0x007F6D21, 0x007F6D9E,
    0x007F6E17, 0x007F6E8C, 0x007F6EFC, 0x007F6F68, 0x007F6FD1,
    0x007F7035, 0x007F7096, 0x007F70F3, 0x007F714C, 0x007F71A1,
    0x007F71F2, 0x007F723F, 0x007F7289, 0x007F72CF, 0x007F7312,
    0x007F7350, 0x007F738B, 0x007F73C3, 0x007F73F6, 0x007F7427,
    0x007F7453, 0x007F747C, 0x007F74A1, 0x007F74C3, 0x007F74E0,
    0x007F74FB, 0x007F7511, 0x007F7524, 0x007F7533, 0x007F753F,
    0x007F7546, 0x007F754A, 0x007F754B, 0x007F7547, 0x007F753F,
    0x007F7534, 0x007F7524, 0x007F7511, 0x007F74F9, 0x007F74DE,
    0x007F74BE, 0x007F749A, 0x007F7472, 0x007F7445, 0x007F7414,
    0x007F73DF, 0x007F73A5, 0x007F7366, 0x007F7323, 0x007F72DA,
    0x007F728D, 0x007F723A, 0x007F71E3, 0x007F7186, 0x007F7123,
    0x007F70BB, 0x007F704D, 0x007F6FD9, 0x007F6F5F, 0x007F6EDF,
    0x007F6E58, 0x007F6DCB, 0x007F6D37, 0x007F6C9C, 0x007F6BF9,
    0x007F6B4F, 0x007F6A9C, 0x007F69E2, 0x007F691F, 0x007F6854,
    0x007F677F, 0x007F66A1, 0x007F65B8, 0x007F64C6, 0x007F63C8,
    0x007F62C0, 0x007F61AB, 0x007F608A, 0x007F5F5D, 0x007F5E21,
    0x007F5CD8, 0x007F5B7F, 0x007F5A17, 0x007F589E, 0x007F5713,
    0x007F5575, 0x007F53C4, 0x007F51FE, 0x007F5022, 0x007F4E2F,
    0x007F4C22, 0x007F49FA, 0x007F47B6, 0x007F4553, 0x007F42CF,
    0x007F4028, 0x007F3D5A, 0x007F3A64, 0x007F3741, 0x007F33ED,
    0x007F3065, 0x007F2CA4, 0x007F28A4, 0x007F245F, 0x007F1FCE,
    0x007F1AEA, 0x007F15A9, 0x007F1000, 0x007F09E4, 0x007F0346,
    0x007EFC16, 0x007EF43E, 0x007EEBA8, 0x007EE237, 0x007ED7C8,
    0x007ECC2F, 0x007EBF37, 0x007EB09D, 0x007EA00A, 0x007E8D0D,
    0x007E7710, 0x007E5D47, 0x007E3E93, 0x007E1959, 0x007DEB2C,
    0x007DB036, 0x007D6203, 0x007CF4B9, 0x007C4FD2, 0x007B3630,
    0x0078D2D2]

# Taken from numpy
_wi_float_data = [
    4.66198677960027669255e-07, 2.56588335019207033255e-08,
    3.41146697750176784592e-08, 4.00230311410932959821e-08,
    4.47179475877737745459e-08, 4.86837785973537366722e-08,
    5.21562578925932412861e-08, 5.52695199001886257153e-08,
    5.81078488992733116465e-08, 6.07279932024587421409e-08,
    6.31701613261172047795e-08, 6.54639842900233842742e-08,
    6.76319905583641815324e-08, 6.96917493470166688656e-08,
    7.16572544283857476692e-08, 7.35398519048393832969e-08,
    7.53488822443557479279e-08, 7.70921367281667127885e-08,
    7.87761895947956022626e-08, 8.04066446825615346857e-08,
    8.19883218760237408659e-08, 8.35254002936857088917e-08,
    8.50215298165053411740e-08, 8.64799190652369040985e-08,
    8.79034055989140110861e-08, 8.92945125124233511541e-08,
    9.06554945027956262312e-08, 9.19883756905278607229e-08,
    9.32949809202232869780e-08, 9.45769618559625849039e-08,
    9.58358188855612866442e-08, 9.70729196232813152662e-08,
    9.82895146313061088986e-08, 9.94867508514382224721e-08,
    1.00665683139461669691e-07, 1.01827284217853923044e-07,
    1.02972453302539369464e-07, 1.04102023612124921572e-07,
    1.05216768930574060431e-07, 1.06317409364335657741e-07,
    1.07404616410877866490e-07, 1.08479017436113134283e-07,
    1.09541199642370962438e-07, 1.10591713595628691212e-07,
    1.11631076370069356306e-07, 1.12659774359245895023e-07,
    1.13678265795837113569e-07, 1.14686983015899673063e-07,
    1.15686334498432158725e-07, 1.16676706706789039179e-07,
    1.17658465754873988919e-07, 1.18631958917986203582e-07,
    1.19597516005596215528e-07, 1.20555450611113917226e-07,
    1.21506061251817163689e-07, 1.22449632410483948386e-07,
    1.23386435488872536840e-07, 1.24316729681986364321e-07,
    1.25240762781015530062e-07, 1.26158771911939892267e-07,
    1.27070984215989333455e-07, 1.27977617477468922011e-07,
    1.28878880703854958297e-07, 1.29774974662539874521e-07,
    1.30666092378141980504e-07, 1.31552419593887221722e-07,
    1.32434135200211397569e-07, 1.33311411633413359243e-07,
    1.34184415246907777059e-07, 1.35053306657377859830e-07,
    1.35918241067904315860e-07, 1.36779368569952053923e-07,
    1.37636834425917531047e-07, 1.38490779333783508675e-07,
    1.39341339675287344817e-07, 1.40188647748881762555e-07,
    1.41032831988654882776e-07, 1.41874017170273235693e-07,
    1.42712324604921442006e-07, 1.43547872322127921816e-07,
    1.44380775242292721080e-07, 1.45211145339665544509e-07,
    1.46039091796461362146e-07, 1.46864721148745476208e-07,
    1.47688137424670065700e-07, 1.48509442275598857119e-07,
    1.49328735100614641423e-07, 1.50146113164867617390e-07,
    1.50961671712187416111e-07, 1.51775504072350982845e-07,
    1.52587701763369746341e-07, 1.53398354589133671168e-07,
    1.54207550732725568797e-07, 1.55015376845697999657e-07,
    1.55821918133584372604e-07, 1.56627258437898192833e-07,
    1.57431480314857468671e-07, 1.58234665111056041043e-07,
    1.59036893036289199880e-07, 1.59838243233728855017e-07,
    1.60638793847630850137e-07, 1.61438622088746393909e-07,
    1.62237804297600106296e-07, 1.63036416005787357730e-07,
    1.63834531995435479082e-07, 1.64632226356965902954e-07,
    1.65429572545287097020e-07, 1.66226643434541294491e-07,
    1.67023511371523209274e-07, 1.67820248227882200051e-07,
    1.68616925451215588827e-07, 1.69413614115155757272e-07,
    1.70210384968549673733e-07, 1.71007308483826142122e-07,
    1.71804454904642543391e-07, 1.72601894292900061024e-07,
    1.73399696575213681990e-07, 1.74197931588920988271e-07,
    1.74996669127712165834e-07, 1.75795978986961275677e-07,
    1.76595931008838063924e-07, 1.77396595127278238022e-07,
    1.78198041412889183130e-07, 1.79000340117867431104e-07,
    1.79803561721004406185e-07, 1.80607776972855859813e-07,
    1.81413056941151359868e-07, 1.82219473056520464354e-07,
    1.83027097158612474240e-07, 1.83836001542687613069e-07,
    1.84646259006759307383e-07, 1.85457942899367347876e-07,
    1.86271127168064649331e-07, 1.87085886408701333260e-07,
    1.87902295915592424729e-07, 1.88720431732658022414e-07,
    1.89540370705627262627e-07, 1.90362190535400839128e-07,
    1.91185969832669990437e-07, 1.92011788173893651535e-07,
    1.92839726158739913768e-07, 1.93669865469102145482e-07,
    1.94502288929804890433e-07, 1.95337080571120616772e-07,
    1.96174325693223683314e-07, 1.97014110932714374919e-07,
    1.97856524331352952716e-07, 1.98701655407150388211e-07,
    1.99549595227971635348e-07, 2.00400436487814600236e-07,
    2.01254273585938820883e-07, 2.02111202709026498408e-07,
    2.02971321916571014951e-07, 2.03834731229698846698e-07,
    2.04701532723644121196e-07, 2.05571830624108885378e-07,
    2.06445731407757185541e-07, 2.07323343907107312957e-07,
    2.08204779420104330037e-07, 2.09090151824673600213e-07,
    2.09979577698577670508e-07, 2.10873176444920111011e-07,
    2.11771070423665379388e-07, 2.12673385089569268965e-07,
    2.13580249136944118603e-07, 2.14491794651713402832e-07,
    2.15408157271244625533e-07, 2.16329476352486921685e-07,
    2.17255895148978920488e-07, 2.18187560997337924713e-07,
    2.19124625513888206785e-07, 2.20067244802139479285e-07,
    2.21015579671883851683e-07, 2.21969795870742159701e-07,
    2.22930064329060010376e-07, 2.23896561419128954210e-07,
    2.24869469229791575583e-07, 2.25848975857580322189e-07,
    2.26835275715640744118e-07, 2.27828569861799901001e-07,
    2.28829066347263833069e-07, 2.29836980587561823183e-07,
    2.30852535757505260518e-07, 2.31875963212094114516e-07,
    2.32907502935486642699e-07, 2.33947404020352726160e-07,
    2.34995925180156140289e-07, 2.36053335297164516378e-07,
    2.37119914009265667728e-07, 2.38195952338983970691e-07,
    2.39281753368440712742e-07, 2.40377632964396957621e-07,
    2.41483920557958384709e-07, 2.42600959984018662258e-07,
    2.43729110386077326413e-07, 2.44868747192698939290e-07,
    2.46020263172594533433e-07, 2.47184069576113545901e-07,
    2.48360597371852893654e-07, 2.49550298588131851232e-07,
    2.50753647770270890721e-07, 2.51971143565970967140e-07,
    2.53203310452642767375e-07, 2.54450700622322097890e-07,
    2.55713896041856770961e-07, 2.56993510708419870887e-07,
    2.58290193123138874550e-07, 2.59604629008804833146e-07,
    2.60937544301314385690e-07, 2.62289708448800566945e-07,
    2.63661938057441759882e-07, 2.65055100928844238758e-07,
    2.66470120540847889467e-07, 2.67907981031821866252e-07,
    2.69369732758258246335e-07, 2.70856498507068313229e-07,
    2.72369480457841388042e-07, 2.73909968006952220135e-07,
    2.75479346585437289399e-07, 2.77079107626811561009e-07,
    2.78710859870496796972e-07, 2.80376342222588603820e-07,
    2.82077438439999912690e-07, 2.83816193958769527230e-07,
    2.85594835255375795814e-07, 2.87415792215003905739e-07,
    2.89281724087851835900e-07, 2.91195549750371467233e-07,
    2.93160483161771875581e-07, 2.95180075129332912389e-07,
    2.97258262785797916083e-07, 2.99399428561531794298e-07,
    3.01608470935804138388e-07, 3.03890889921758510417e-07,
    3.06252891144972267537e-07, 3.08701513613258141075e-07,
    3.11244787989714509378e-07, 3.13891934589336184321e-07,
    3.16653613755314681314e-07, 3.19542246256559459667e-07,
    3.22572428717978242099e-07, 3.25761480217458181578e-07,
    3.29130173358915628534e-07, 3.32703730345002116955e-07,
    3.36513208964639108346e-07, 3.40597478255417943913e-07,
    3.45006114675213401550e-07, 3.49803789521323211592e-07,
    3.55077180848341416206e-07, 3.60946392031859609868e-07,
    3.67584959507244041831e-07, 3.75257645787954431030e-07,
    3.84399301057791926300e-07, 3.95804015855768440983e-07,
    4.11186015434435801956e-07, 4.35608969373823260746e-07]

# Taken from numpy
_fi_float_data = [
    1.00000000000000000000e+00, 9.77101701267671596263e-01,
    9.59879091800106665211e-01, 9.45198953442299649730e-01,
    9.32060075959230460718e-01, 9.19991505039347012840e-01,
    9.08726440052130879366e-01, 8.98095921898343418910e-01,
    8.87984660755833377088e-01, 8.78309655808917399966e-01,
    8.69008688036857046555e-01, 8.60033621196331532488e-01,
    8.51346258458677951353e-01, 8.42915653112204177333e-01,
    8.34716292986883434679e-01, 8.26726833946221373317e-01,
    8.18929191603702366642e-01, 8.11307874312656274185e-01,
    8.03849483170964274059e-01, 7.96542330422958966274e-01,
    7.89376143566024590648e-01, 7.82341832654802504798e-01,
    7.75431304981187174974e-01, 7.68637315798486264740e-01,
    7.61953346836795386565e-01, 7.55373506507096115214e-01,
    7.48892447219156820459e-01, 7.42505296340151055290e-01,
    7.36207598126862650112e-01, 7.29995264561476231435e-01,
    7.23864533468630222401e-01, 7.17811932630721960535e-01,
    7.11834248878248421200e-01, 7.05928501332754310127e-01,
    7.00091918136511615067e-01, 6.94321916126116711609e-01,
    6.88616083004671808432e-01, 6.82972161644994857355e-01,
    6.77388036218773526009e-01, 6.71861719897082099173e-01,
    6.66391343908750100056e-01, 6.60975147776663107813e-01,
    6.55611470579697264149e-01, 6.50298743110816701574e-01,
    6.45035480820822293424e-01, 6.39820277453056585060e-01,
    6.34651799287623608059e-01, 6.29528779924836690007e-01,
    6.24450015547026504592e-01, 6.19414360605834324325e-01,
    6.14420723888913888899e-01, 6.09468064925773433949e-01,
    6.04555390697467776029e-01, 5.99681752619125263415e-01,
    5.94846243767987448159e-01, 5.90047996332826008015e-01,
    5.85286179263371453274e-01, 5.80559996100790898232e-01,
    5.75868682972353718164e-01, 5.71211506735253227163e-01,
    5.66587763256164445025e-01, 5.61996775814524340831e-01,
    5.57437893618765945014e-01, 5.52910490425832290562e-01,
    5.48413963255265812791e-01, 5.43947731190026262382e-01,
    5.39511234256952132426e-01, 5.35103932380457614215e-01,
    5.30725304403662057062e-01, 5.26374847171684479008e-01,
    5.22052074672321841931e-01, 5.17756517229756352272e-01,
    5.13487720747326958914e-01, 5.09245245995747941592e-01,
    5.05028667943468123624e-01, 5.00837575126148681903e-01,
    4.96671569052489714213e-01, 4.92530263643868537748e-01,
    4.88413284705458028423e-01, 4.84320269426683325253e-01,
    4.80250865909046753544e-01, 4.76204732719505863248e-01,
    4.72181538467730199660e-01, 4.68180961405693596422e-01,
    4.64202689048174355069e-01, 4.60246417812842867345e-01,
    4.56311852678716434184e-01, 4.52398706861848520777e-01,
    4.48506701507203064949e-01, 4.44635565395739396077e-01,
    4.40785034665803987508e-01, 4.36954852547985550526e-01,
    4.33144769112652261445e-01, 4.29354541029441427735e-01,
    4.25583931338021970170e-01, 4.21832709229495894654e-01,
    4.18100649837848226120e-01, 4.14387534040891125642e-01,
    4.10693148270188157500e-01, 4.07017284329473372217e-01,
    4.03359739221114510510e-01, 3.99720314980197222177e-01,
    3.96098818515832451492e-01, 3.92495061459315619512e-01,
    3.88908860018788715696e-01, 3.85340034840077283462e-01,
    3.81788410873393657674e-01, 3.78253817245619183840e-01,
    3.74736087137891138443e-01, 3.71235057668239498696e-01,
    3.67750569779032587814e-01, 3.64282468129004055601e-01,
    3.60830600989648031529e-01, 3.57394820145780500731e-01,
    3.53974980800076777232e-01, 3.50570941481406106455e-01,
    3.47182563956793643900e-01, 3.43809713146850715049e-01,
    3.40452257044521866547e-01, 3.37110066637006045021e-01,
    3.33783015830718454708e-01, 3.30470981379163586400e-01,
    3.27173842813601400970e-01, 3.23891482376391093290e-01,
    3.20623784956905355514e-01, 3.17370638029913609834e-01,
    3.14131931596337177215e-01, 3.10907558126286509559e-01,
    3.07697412504292056035e-01, 3.04501391976649993243e-01,
    3.01319396100803049698e-01, 2.98151326696685481377e-01,
    2.94997087799961810184e-01, 2.91856585617095209972e-01,
    2.88729728482182923521e-01, 2.85616426815501756042e-01,
    2.82516593083707578948e-01, 2.79430141761637940157e-01,
    2.76356989295668320494e-01, 2.73297054068577072172e-01,
    2.70250256365875463072e-01, 2.67216518343561471038e-01,
    2.64195763997261190426e-01, 2.61187919132721213522e-01,
    2.58192911337619235290e-01, 2.55210669954661961700e-01,
    2.52241126055942177508e-01, 2.49284212418528522415e-01,
    2.46339863501263828249e-01, 2.43408015422750312329e-01,
    2.40488605940500588254e-01, 2.37581574431238090606e-01,
    2.34686861872330010392e-01, 2.31804410824338724684e-01,
    2.28934165414680340644e-01, 2.26076071322380278694e-01,
    2.23230075763917484855e-01, 2.20396127480151998723e-01,
    2.17574176724331130872e-01, 2.14764175251173583536e-01,
    2.11966076307030182324e-01, 2.09179834621125076977e-01,
    2.06405406397880797353e-01, 2.03642749310334908452e-01,
    2.00891822494656591136e-01, 1.98152586545775138971e-01,
    1.95425003514134304483e-01, 1.92709036903589175926e-01,
    1.90004651670464985713e-01, 1.87311814223800304768e-01,
    1.84630492426799269756e-01, 1.81960655599522513892e-01,
    1.79302274522847582272e-01, 1.76655321443734858455e-01,
    1.74019770081838553999e-01, 1.71395595637505754327e-01,
    1.68782774801211288285e-01, 1.66181285764481906364e-01,
    1.63591108232365584074e-01, 1.61012223437511009516e-01,
    1.58444614155924284882e-01, 1.55888264724479197465e-01,
    1.53343161060262855866e-01, 1.50809290681845675763e-01,
    1.48286642732574552861e-01, 1.45775208005994028060e-01,
    1.43274978973513461566e-01, 1.40785949814444699690e-01,
    1.38308116448550733057e-01, 1.35841476571253755301e-01,
    1.33386029691669155683e-01, 1.30941777173644358090e-01,
    1.28508722279999570981e-01, 1.26086870220185887081e-01,
    1.23676228201596571932e-01, 1.21276805484790306533e-01,
    1.18888613442910059947e-01, 1.16511665625610869035e-01,
    1.14145977827838487895e-01, 1.11791568163838089811e-01,
    1.09448457146811797824e-01, 1.07116667774683801961e-01,
    1.04796225622487068629e-01, 1.02487158941935246892e-01,
    1.00189498768810017482e-01, 9.79032790388624646338e-02,
    9.56285367130089991594e-02, 9.33653119126910124859e-02,
    9.11136480663737591268e-02, 8.88735920682758862021e-02,
    8.66451944505580717859e-02, 8.44285095703534715916e-02,
    8.22235958132029043366e-02, 8.00305158146630696292e-02,
    7.78493367020961224423e-02, 7.56801303589271778804e-02,
    7.35229737139813238622e-02, 7.13779490588904025339e-02,
    6.92451443970067553879e-02, 6.71246538277884968737e-02,
    6.50165779712428976156e-02, 6.29210244377581412456e-02,
    6.08381083495398780614e-02, 5.87679529209337372930e-02,
    5.67106901062029017391e-02, 5.46664613248889208474e-02,
    5.26354182767921896513e-02, 5.06177238609477817000e-02,
    4.86135532158685421122e-02, 4.66230949019303814174e-02,
    4.46465522512944634759e-02, 4.26841449164744590750e-02,
    4.07361106559409394401e-02, 3.88027074045261474722e-02,
    3.68842156885673053135e-02, 3.49809414617161251737e-02,
    3.30932194585785779961e-02, 3.12214171919203004046e-02,
    2.93659397581333588001e-02, 2.75272356696031131329e-02,
    2.57058040085489103443e-02, 2.39022033057958785407e-02,
    2.21170627073088502113e-02, 2.03510962300445102935e-02,
    1.86051212757246224594e-02, 1.68800831525431419000e-02,
    1.51770883079353092332e-02, 1.34974506017398673818e-02,
    1.18427578579078790488e-02, 1.02149714397014590439e-02,
    8.61658276939872638800e-03, 7.05087547137322242369e-03,
    5.52240329925099155545e-03, 4.03797259336302356153e-03,
    2.60907274610215926189e-03, 1.26028593049859797236e-03]

def _load_wi(builder, idx, fptype, data):
    module = builder.function.module
    fmt_ty = ir.ArrayType(fptype, 256)
    global_wi = ir.GlobalVariable(module, fmt_ty,
                                  name="__pnl_builtin_wi_{}".format(fptype))
    global_wi.linkage = "internal"
    global_wi.global_constant = True
    global_wi.initializer = fmt_ty(data)

    ptr = builder.gep(global_wi, [idx.type(0), idx])
    return builder.load(ptr)


def _load_ki(builder, idx, itype, data):
    module = builder.function.module
    fmt_ty = ir.ArrayType(itype, 256)
    global_ki = ir.GlobalVariable(module, fmt_ty,
                                  name="__pnl_builtin_ki_{}".format(itype))
    global_ki.linkage = "internal"
    global_ki.global_constant = True
    global_ki.initializer = fmt_ty(data)

    ptr = builder.gep(global_ki, [idx.type(0), idx])
    return builder.load(ptr)


def _load_fi(builder, idx, fptype, data):
    module = builder.function.module
    fmt_ty = ir.ArrayType(fptype, 256)
    name = "__pnl_builtin_fi_{}".format(fptype)
    try:
        global_fi = module.get_global(name)
        assert global_fi.type.pointee == fmt_ty
    except KeyError:
        global_fi = ir.GlobalVariable(module, fmt_ty,
                                      name=name)
        global_fi.linkage = "internal"
        global_fi.global_constant = True
        global_fi.initializer = fmt_ty(data)

    ptr = builder.gep(global_fi, [idx.type(0), idx])
    return builder.load(ptr)


def _setup_philox_rand_normal(ctx, state_ty, gen_float, gen_int, wi_data, ki_data, fi_data):
    fptype = gen_float.args[1].type.pointee
    itype = gen_int.args[1].type.pointee
    if fptype != ctx.float_ty:
        # We don't have numeric halpers available for the desired type
        return
    builder = _setup_builtin_func_builder(ctx, "philox_rand_normal",
                                         (state_ty.as_pointer(), fptype.as_pointer()))
    state, out = builder.function.args

    loop_block = builder.append_basic_block("gen_loop_ziggurat")

    # Allocate storage for calling int/float PRNG
    # outside of the loop
    tmp_fptype = builder.alloca(fptype, name="tmp_fp")
    tmp_itype = builder.alloca(itype, name="tmp_int")

    # Enter the main generation loop
    builder.branch(loop_block)
    builder.position_at_end(loop_block)

    r_ptr = tmp_itype
    builder.call(gen_int, [state, r_ptr])
    r = builder.load(r_ptr)

    # This is only for 64 bit
    # Extract index to the global table
    idx = builder.and_(r, r.type(0xff))
    r = builder.lshr(r, r.type(8))

    # Extract sign
    sign = builder.and_(r, r.type(0x1))
    r = builder.lshr(r, r.type(1))

    # Extract abs
    MANTISSA = 0x000fffffffffffff if fptype is ir.DoubleType() else 0x007fffff
    rabs = builder.and_(r, r.type(MANTISSA))
    rabs_f = builder.uitofp(rabs, fptype)

    wi = _load_wi(builder, idx, fptype, wi_data)
    x = builder.fmul(rabs_f, wi)

    # Apply sign
    neg_x = helpers.fneg(builder, x)
    x = builder.select(builder.trunc(sign, ctx.bool_ty), neg_x, x)

    ki = _load_ki(builder, idx, itype, ki_data)
    is_lt_ki = builder.icmp_unsigned("<", rabs, ki)
    with builder.if_then(is_lt_ki, likely=True):
        builder.store(x, out)
        builder.ret_void()

    is_idx0 = builder.icmp_unsigned("==", idx.type(0), idx)
    with builder.if_then(is_idx0):
        inner_loop_block = builder.block

        ZIGGURAT_NOR_R = 3.6541528853610087963519472518
        ZIGGURAT_NOR_INV_R = 0.27366123732975827203338247596

        # xx = -ziggurat_nor_inv_r * npy_log1p(-next_double(bitgen_state));
        builder.call(gen_float, [state, tmp_fptype])
        xx = builder.load(tmp_fptype)
        xx = helpers.fneg(builder, xx)
        xx = helpers.log1p(ctx, builder, xx)
        xx = builder.fmul(xx.type(-ZIGGURAT_NOR_INV_R), xx)

        # yy = -npy_log1p(-next_double(bitgen_state));
        builder.call(gen_float, [state, tmp_fptype])
        yy = builder.load(tmp_fptype)
        yy = helpers.fneg(builder, yy)
        yy = helpers.log1p(ctx, builder, yy)
        yy = helpers.fneg(builder, yy)

        # if (yy + yy > xx * xx)
        lhs = builder.fadd(yy, yy)
        rhs = builder.fmul(xx, xx)
        cond = builder.fcmp_ordered(">", lhs, rhs)
        with builder.if_then(cond):
            # return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
            val = builder.fadd(xx.type(ZIGGURAT_NOR_R), xx)
            neg_val = helpers.fneg(builder, val)
            sign_cond = builder.lshr(rabs, rabs.type(8))
            sign_cond = builder.trunc(sign_cond, ctx.bool_ty)
            val = builder.select(sign_cond, neg_val, val)
            builder.store(val, out)
            builder.ret_void()

        builder.branch(inner_loop_block)

    # idx has to be > 0
    fi_idx = _load_fi(builder, idx, fptype, fi_data)
    fi_idxm1 = _load_fi(builder, builder.sub(idx, idx.type(1)), fptype, fi_data)
    x_sq = builder.fmul(x, x)
    x_sq_nh = builder.fmul(x_sq, x_sq.type(-0.5))

    exp_x_sqnh = helpers.exp(ctx, builder, x_sq_nh)

    # next uniform random number
    r_ptr = tmp_fptype
    builder.call(gen_float, [state, r_ptr])
    r = builder.load(r_ptr)

    # if (((fi_double[idx - 1] - fi_double[idx]) * next_double(bitgen_state) +
    #       fi_double[idx]) < exp(-0.5 * x * x))
    lhs = builder.fsub(fi_idxm1, fi_idx)
    lhs = builder.fmul(lhs, r)
    lhs = builder.fadd(lhs, fi_idx)

    should_ret = builder.fcmp_ordered("<", lhs, exp_x_sqnh)
    with builder.if_then(should_ret):
        builder.store(x, out)
        builder.ret_void()

    builder.branch(loop_block)


def get_philox_state_struct(ctx):
    int64_ty = ir.IntType(64)
    int16_ty = ir.IntType(16)
    return ir.LiteralStructType([
        ir.ArrayType(int64_ty, 4),  # counter
        ir.ArrayType(int64_ty, 2),  # key
        ir.ArrayType(int64_ty, _PHILOX_DEFAULT_BUFFER_SIZE),  #  pre-gen buffer
        ctx.int32_ty,  #  the other half of random 64 bit int
        int16_ty,      #  buffer pos
        ctx.bool_ty,   #  has uint buffered
        int64_ty])     #  seed


def setup_philox(ctx):
    state_ty = get_philox_state_struct(ctx)

    _setup_philox_rand_init(ctx, state_ty)

    gen_int64 = _setup_philox_rand_int64(ctx, state_ty)
    gen_double = _setup_philox_rand_double(ctx, state_ty, gen_int64)
    _setup_philox_rand_normal(ctx, state_ty, gen_double, gen_int64, _wi_double_data, _ki_i64_data, _fi_double_data)

    gen_int32 = _setup_philox_rand_int32(ctx, state_ty, gen_int64)
    gen_float = _setup_philox_rand_float(ctx, state_ty, gen_int32)
    _setup_philox_rand_normal(ctx, state_ty, gen_float, gen_int32, _wi_float_data, _ki_i32_data, _fi_float_data)
