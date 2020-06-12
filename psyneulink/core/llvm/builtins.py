# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM builtins **************************************************************

from llvmlite import ir


from . import debug
from . import helpers
from .builder_context import LLVMBuilderContext, _BUILTIN_PREFIX


debug_env = debug.debug_env


def _setup_builtin_func_builder(ctx, name, args):
    builder = ctx.create_llvm_function(args, None, _BUILTIN_PREFIX + name)

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


# Setup vector copy builtin
def setup_vec_copy(ctx):
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()

    # builtin vector copy func
    # param1: ptr to vector 1
    # param2: sizeof vector
    # param3: ptr to output vector (make sure this is same size as param3)
    builder = _setup_builtin_func_builder(ctx, "vec_copy", (double_ptr_ty, ctx.int32_ty, double_ptr_ty))
    u, x, o = builder.function.args

    # Copy
    with helpers.for_loop_zero_inc(builder, x, "copy") as (b1, index):
        u_ptr = b1.gep(u, [index])
        o_ptr = b1.gep(o, [index])
        u_val = b1.load(u_ptr)

        b1.store(u_val, o_ptr)

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
    builder.ret_void()

    return builder.function


def _setup_mt_rand_init(ctx, state_ty, init_scalar):
    seed_ty = state_ty.elements[0].element
    builder = _setup_builtin_func_builder(ctx, "mt_rand_init", (state_ty.as_pointer(), seed_ty))
    state, seed = builder.function.args

    default_seed = seed.type(19650218)
    builder.call(init_scalar, [state, default_seed])

    # python considers everything to be an array
    key_array = builder.alloca(ir.ArrayType(seed.type, 1))
    key_p = builder.gep(key_array, [ctx.int32_ty(0), ctx.int32_ty(0)])
    builder.store(seed, key_p)

    pi = builder.alloca(ctx.int32_ty)
    builder.store(ctx.int32_ty(1), pi)
    pj = builder.alloca(ctx.int32_ty)
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
        pmag01 = builder.alloca(mag01.type)
        builder.store(mag01, pmag01)

        with helpers.for_loop_zero_inc(builder,
                                       ctx.int32_ty(_MERSENNE_N - _MERSENNE_M),
                                       "first_half") as (b, kk):
            pkk = b.gep(array, [ctx.int32_ty(0), kk])
            pkk_1 = b.gep(array, [ctx.int32_ty(0), b.add(kk, ctx.int32_ty(1))])

            val_kk = b.and_(b.load(pkk), pkk.type.pointee(0x80000000))
            val_kk_1 = b.and_(b.load(pkk_1), pkk_1.type.pointee(0x7fffffff))
            val = b.or_(val_kk, val_kk_1)

            val_1 = b.and_(val, val.type(1))
            pval_mag = b.gep(pmag01, [ctx.int32_ty(0), val_1])
            val_mag = b.load(pval_mag)

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

            val_1 = b.and_(val, val.type(1))
            pval_mag = b.gep(pmag01, [ctx.int32_ty(0), val_1])
            val_mag = b.load(pval_mag)

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

    al = builder.alloca(gen_int.args[1].type.pointee)
    builder.call(gen_int, [state, al])

    bl = builder.alloca(gen_int.args[1].type.pointee)
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
    tmp = builder.alloca(out.type.pointee)

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
        ctx.float_ty])  # last_gauss


def setup_mersenne_twister(ctx):
    state_ty = get_mersenne_twister_state_struct(ctx)

    init_scalar = _setup_mt_rand_init_scalar(ctx, state_ty)
    _setup_mt_rand_init(ctx, state_ty, init_scalar)

    gen_int = _setup_mt_rand_integer(ctx, state_ty)
    gen_float = _setup_mt_rand_float(ctx, state_ty, gen_int)
    _setup_mt_rand_normal(ctx, state_ty, gen_float)
