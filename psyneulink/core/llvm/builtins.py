# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM builtins **************************************************************

from llvmlite import ir
from psyneulink.core.llvm import helpers
from psyneulink.core.llvm.builder_context import LLVMBuilderContext


def setup_vxm(ctx):
    module = ctx.module
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()
    func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))

    # Create function
    function = ir.Function(module, func_ty, name="__pnl_builtin_vxm")
    function.attributes.add('argmemonly')
    function.attributes.add('alwaysinline')

    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(function, None)
    v, m, x, y, o = function.args

    # Add function arg attributes
    for a in v, m, o:
        a.attributes.add('nonnull')
        a.attributes.add('noalias')

    index = None
    # zero the output array
    with helpers.for_loop_zero_inc(builder, y, "zero") as (builder, index):
        ptr = builder.gep(o, [index])
        builder.store(ctx.float_ty(0), ptr)

    # Multiplication

    # Initialize outer loop variable
    index_i_var = builder.alloca(ctx.int32_ty)
    builder.store(ctx.int32_ty(0), index_i_var)

    # Outer loop cond BB
    outer_cond_block = builder.append_basic_block("outer-cond")
    outer_body_block = builder.append_basic_block("outer-body")
    outer_out_block = builder.append_basic_block("outer-out")

    # Loop condition
    builder.branch(outer_cond_block)
    with builder.goto_block(outer_cond_block):
        tmp = builder.load(index_i_var)
        cond = builder.icmp_signed("<", tmp, x)
        builder.cbranch(cond, outer_body_block, outer_out_block).set_weights([99, 1])

    # Loop body
    with builder.goto_block(outer_body_block):
        index_i = builder.load(index_i_var)

        # Initialize outer loop variable
        index_j_var = builder.alloca(ctx.int32_ty)
        builder.store(ctx.int32_ty(0), index_j_var)

        # Outer loop cond BB
        inner_cond_block = builder.append_basic_block("inner-cond")
        inner_body_block = builder.append_basic_block("inner-body")
        inner_out_block = builder.append_basic_block("inner-out")

        # Loop condition
        builder.branch(inner_cond_block)
        with builder.goto_block(inner_cond_block):
            tmp = builder.load(index_j_var)
            cond = builder.icmp_signed("<", tmp, y)
            builder.cbranch(cond, inner_body_block, inner_out_block).set_weights([99, 1])

        # Loop body
        with builder.goto_block(inner_body_block):
            index_j = builder.load(index_j_var)

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

            next_index_j = builder.add(index_j, ctx.int32_ty(1))
            builder.store(next_index_j, index_j_var)
            builder.branch(inner_cond_block)

        with builder.goto_block(inner_out_block):
            next_index_i = builder.add(index_i, ctx.int32_ty(1))
            builder.store(next_index_i, index_i_var)
            builder.branch(outer_cond_block)

    # Return
    with builder.goto_block(outer_out_block):
        builder.ret_void()

def setup_pnl_intrinsics(ctx):
    module = ctx.module
    # Setup types
    single_intr_ty = ir.FunctionType(ctx.float_ty, [ctx.float_ty])
    double_intr_ty = ir.FunctionType(ctx.float_ty, (ctx.float_ty, ctx.float_ty))

    # Create function declarations
    ir.Function(module, single_intr_ty, name="__pnl_builtin_exp")
    ir.Function(module, single_intr_ty, name="__pnl_builtin_log")
    ir.Function(module, double_intr_ty, name="__pnl_builtin_pow")

def _generate_intrinsic_wrapper(module, name, ret, args):
    intrinsic = module.declare_intrinsic("llvm." + name, list(set(args)))

    func_ty = ir.FunctionType(ret, args)
    function = ir.Function(module, func_ty, name="__pnl_builtin_" + name)
    function.attributes.add('alwaysinline')
    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(function, None)
    builder.ret(builder.call(intrinsic, function.args))

def _generate_cpu_builtins_module(_float_ty):
    """ Generate function wrappers for log, exp, and pow intrinsics. """
    module = ir.Module(name="cpu_builtins")
    for intrinsic in ('exp', 'log'):
        _generate_intrinsic_wrapper(module, intrinsic, _float_ty, [_float_ty])

    _generate_intrinsic_wrapper(module, "pow", _float_ty, [_float_ty, _float_ty])
    return module

_MERSENNE_N = 624
_MERSENNE_M = 397

def _setup_mt_rand_init_scalar(ctx, state_ty):
    seed_ty = state_ty.elements[0].element
    init_ty = ir.FunctionType(ir.VoidType(), (state_ty.as_pointer(), seed_ty))
    # Create init function
    init_scalar = ir.Function(ctx.module, init_ty, name="__pnl_builtin_mt_rand_init_scalar")
    init_scalar.attributes.add('argmemonly')
    init_scalar.attributes.add('alwaysinline')

    block = init_scalar.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(init_scalar, None)
    state, seed = init_scalar.args

    # Add function arg attributes
    state.attributes.add('nonnull')
    state.attributes.add('noalias')

    array = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0)])

    # Store seed to the 0-th element
    a_0 = builder.gep(array, [ctx.int32_ty(0), ctx.int32_ty(0)])
    seed_lo = builder.and_(seed, seed.type(0xffffffff))
    seed_lo = builder.trunc(seed, a_0.type.pointee)
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
    return init_scalar

def _setup_mt_rand_init(ctx, state_ty, init_scalar):
    seed_ty = state_ty.elements[0].element
    init_ty = ir.FunctionType(ir.VoidType(), (state_ty.as_pointer(), seed_ty))
    # Create init_array function
    init = ir.Function(ctx.module, init_ty, name="__pnl_builtin_mt_rand_init")
    init.attributes.add('argmemonly')
    init.attributes.add('alwaysinline')

    block = init.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(init, None)
    state, seed = init.args

    # Add function arg attributes
    state.attributes.add('nonnull')
    state.attributes.add('noalias')

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
    with helpers.for_loop_zero_inc(builder,
                                   ctx.int32_ty(_MERSENNE_N),
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
    return init

def _setup_mt_rand_integer(ctx, state_ty):
    int64_ty = ir.IntType(64)
    # Generate random number generator function. It produces random 32bit numberin a 64bit word
    gen_ty = ir.FunctionType(ir.VoidType(), (state_ty.as_pointer(), int64_ty.as_pointer()))
    gen_int = ir.Function(ctx.module, gen_ty, name="__pnl_builtin_mt_rand_int32")
    gen_int.attributes.add('argmemonly')
    gen_int.attributes.add('alwaysinline')

    block = gen_int.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(gen_int, None)
    state, out = gen_int.args

    # Add function arg attributes
    for a in state, out:
        a.attributes.add('nonnull')
        a.attributes.add('noalias')

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
            is_last = b.icmp_unsigned( "==", kk, ctx.int32_ty(_MERSENNE_N - 1))
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
    return gen_int

def _setup_mt_rand_float(ctx, state_ty, gen_int):
    # Generate random float number generator function
    gen_ty = ir.FunctionType(ir.VoidType(), (state_ty.as_pointer(), ctx.float_ty.as_pointer()))
    gen_float = ir.Function(ctx.module, gen_ty, name="__pnl_builtin_mt_rand_double")
    gen_float.attributes.add('argmemonly')
    gen_float.attributes.add('alwaysinline')

    block = gen_float.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(gen_float, None)
    state, out = gen_float.args

    # Add function arg attributes
    for a in state, out:
        a.attributes.add('nonnull')
        a.attributes.add('noalias')

    al = builder.alloca(gen_int.args[1].type.pointee)
    builder.call(gen_int, [state, al])

    bl = builder.alloca(gen_int.args[1].type.pointee)
    builder.call(gen_int, [state, bl])

    a = builder.load(al)
    b = builder.load(bl)

    a = builder.lshr(a, a.type(5))
    b = builder.lshr(b, b.type(6))

    af = builder.uitofp(a, ctx.float_ty)
    bf = builder.uitofp(b, ctx.float_ty)

    val = builder.fmul(af, ctx.float_ty(67108864.0))
    val = builder.fadd(val, bf)
    val = builder.fdiv(val, ctx.float_ty(9007199254740992.0))

    builder.store(val, out)
    builder.ret_void()
    return gen_float

def _setup_mt_rand_normal(ctx, state_ty, gen_float):
    # Generate random float from Normal distribution generator
    gen_ty = ir.FunctionType(ir.VoidType(), (state_ty.as_pointer(), ctx.float_ty.as_pointer()))
    gen_normal = ir.Function(ctx.module, gen_ty, name="__pnl_builtin_mt_rand_normal")
    gen_normal.attributes.add('argmemonly')
    gen_normal.attributes.add('alwaysinline')

    block = gen_normal.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = LLVMBuilderContext.get_debug_location(gen_normal, None)
    state, out = gen_normal.args

    # Add function arg attributes
    for a in state, out:
        a.attributes.add('nonnull')
        a.attributes.add('noalias')

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

    # X1
    builder.call(gen_float, [state, tmp])
    x1 = builder.load(tmp)
    x1 = builder.fmul(x1, ctx.float_ty(2.0))
    x1 = builder.fsub(x1, ctx.float_ty(1.0))

    # x2
    builder.call(gen_float, [state, tmp])
    x2 = builder.load(tmp)
    x2 = builder.fmul(x2, ctx.float_ty(2.0))
    x2 = builder.fsub(x2, ctx.float_ty(1.0))

    # r2
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
        ir.ArrayType(ctx.int32_ty, _MERSENNE_N), # array
        ctx.int32_ty, #index
        ctx.int32_ty, #last_gauss available
        ctx.float_ty]) #last_gauss

def setup_mersenne_twister(ctx):
    state_ty = get_mersenne_twister_state_struct(ctx)

    init_scalar = _setup_mt_rand_init_scalar(ctx, state_ty)
    _setup_mt_rand_init(ctx, state_ty, init_scalar)

    gen_int = _setup_mt_rand_integer(ctx, state_ty)
    gen_float = _setup_mt_rand_float(ctx, state_ty, gen_int)
    _setup_mt_rand_normal(ctx, state_ty, gen_float)
