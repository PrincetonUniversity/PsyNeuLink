from psyneulink.core import llvm as pnlvm

__all__ = ["gen_inject_unary_function_call",
           "gen_inject_vec_copy",
           "gen_inject_vec_binop",
           "gen_inject_vec_add",
           "gen_inject_vec_sub",
           "gen_inject_vec_hadamard",
           "gen_inject_mat_binop",
           "gen_inject_mat_add",
           "gen_inject_mat_sub",
           "gen_inject_mat_hadamard",
           "gen_inject_mat_scalar_mult",
           "gen_inject_vxm",
           "gen_inject_vxm_transposed"]

def gen_inject_unary_function_call(ctx, builder, unary_func, vector, output_vec=None):
    dim = len(vector.type.pointee)
    if output_vec is None:
        output_vec = builder.alloca(pnlvm.ir.types.ArrayType(ctx.float_ty, dim))
    assert len(output_vec.type.pointee) == dim

    # Get the pointer to the first element of the array to convert from [? x double]* -> double*
    vec_in = builder.gep(vector, [ctx.int32_ty(0), ctx.int32_ty(0)])
    vec_out = builder.gep(output_vec, [ctx.int32_ty(0), ctx.int32_ty(0)])

    builder.call(unary_func, [vec_in, ctx.int32_ty(dim), vec_out])
    return output_vec

def gen_inject_vec_copy(ctx, builder, vector, output_vec=None):
    dim = len(vector.type.pointee)
    if output_vec is None:
        output_vec = builder.alloca(pnlvm.ir.types.ArrayType(ctx.float_ty, dim))
    assert len(output_vec.type.pointee) == dim

    # Get the pointer to the first element of the array to convert from [? x double]* -> double*
    vec_in = builder.gep(vector, [ctx.int32_ty(0), ctx.int32_ty(0)])
    vec_out = builder.gep(output_vec, [ctx.int32_ty(0), ctx.int32_ty(0)])

    builtin = ctx.import_llvm_function("__pnl_builtin_vec_copy")
    builder.call(builtin, [vec_in, ctx.int32_ty(dim), vec_out])
    return output_vec

def gen_inject_vec_binop(ctx, builder, op, u, v, output_vec=None):
    dim = len(u.type.pointee)
    assert len(v.type.pointee) == dim
    if output_vec is None:
        output_vec = builder.alloca(pnlvm.ir.types.ArrayType(ctx.float_ty, dim))
    assert len(output_vec.type.pointee) == dim

    # Get the pointer to the first element of the array to convert from [? x double]* -> double*
    vec_u = builder.gep(u, [ctx.int32_ty(0), ctx.int32_ty(0)])
    vec_v = builder.gep(v, [ctx.int32_ty(0), ctx.int32_ty(0)])
    vec_out = builder.gep(output_vec, [ctx.int32_ty(0), ctx.int32_ty(0)])

    builder.call(ctx.import_llvm_function(op), [vec_u, vec_v, ctx.int32_ty(dim), vec_out])
    return output_vec

def gen_inject_vec_add(ctx, builder, u, v, output_vec=None):
    return gen_inject_vec_binop(ctx, builder, "__pnl_builtin_vec_add", u, v, output_vec)

def gen_inject_vec_sub(ctx, builder, u, v, output_vec=None):
    return gen_inject_vec_binop(ctx, builder, "__pnl_builtin_vec_sub", u, v, output_vec)

def gen_inject_vec_hadamard(ctx, builder, u ,v, output_vec=None):
    return gen_inject_vec_binop(ctx, builder, "__pnl_builtin_vec_hadamard", u, v, output_vec)

def gen_inject_mat_binop(ctx, builder, op, m1, m2, output_mat=None):
    x = len(m1.type.pointee)
    y = len(m1.type.pointee.element)
    assert len(m2.type.pointee) == x and len(m2.type.pointee.element) == y

    if output_mat is None:
        output_mat = builder.alloca(
            pnlvm.ir.types.ArrayType(
                pnlvm.ir.types.ArrayType(ctx.float_ty, y), x))
    assert len(output_mat.type.pointee) == x
    assert len(output_mat.type.pointee.element) == y

    builtin = ctx.import_llvm_function(op)
    builder.call(builtin, [builder.bitcast(m1, ctx.float_ty.as_pointer()),
                            builder.bitcast(m2, ctx.float_ty.as_pointer()),
                            ctx.int32_ty(x), ctx.int32_ty(y),
                            builder.bitcast(output_mat, ctx.float_ty.as_pointer())])
    return output_mat

def gen_inject_mat_add(ctx, builder, m1, m2, output_mat=None):
    return gen_inject_mat_binop(ctx, builder, "__pnl_builtin_mat_add", m1, m2, output_mat)

def gen_inject_mat_sub(ctx, builder, m1, m2, output_mat=None):
    return gen_inject_mat_binop(ctx, builder, "__pnl_builtin_mat_sub", m1, m2, output_mat)

def gen_inject_mat_hadamard(ctx, builder, m1, m2, output_mat=None):
    return gen_inject_mat_binop(ctx, builder, "__pnl_builtin_mat_hadamard", m1, m2, output_mat)

def gen_inject_mat_scalar_mult(ctx, builder, m1, s, output_mat=None):
    x = len(m1.type.pointee)
    y = len(m1.type.pointee.element)
    if output_mat is None:
        output_mat = builder.alloca(
            pnlvm.ir.types.ArrayType(
                pnlvm.ir.types.ArrayType(ctx.float_ty, y), x))
    assert len(output_mat.type.pointee) == x
    assert len(output_mat.type.pointee.element) == y

    builtin = ctx.import_llvm_function("__pnl_builtin_mat_scalar_mult")
    builder.call(builtin, [builder.bitcast(m1, ctx.float_ty.as_pointer()),
                            s, ctx.int32_ty(x), ctx.int32_ty(y),
                            builder.bitcast(output_mat, ctx.float_ty.as_pointer())])
    return output_mat

def gen_inject_vxm(ctx, builder, m1, m2, output_vec=None):
    y = len(m2.type.pointee)
    z = len(m2.type.pointee.element)
    assert len(m1.type.pointee) == y
    # create output vec
    if output_vec is None:
        output_vec = builder.alloca(pnlvm.ir.types.ArrayType(ctx.float_ty, z))
    assert len(output_vec.type.pointee) == z

    # Get the pointer to the first element of the array to convert from [? x double]* -> double*
    v = builder.gep(m1, [ctx.int32_ty(0), ctx.int32_ty(0)])
    out = builder.gep(output_vec, [ctx.int32_ty(0), ctx.int32_ty(0)])

    builtin = ctx.import_llvm_function("__pnl_builtin_vxm")
    builder.call(builtin, [v, builder.bitcast(m2, ctx.float_ty.as_pointer()),
                            ctx.int32_ty(y), ctx.int32_ty(z), out])
    return output_vec

def gen_inject_vxm_transposed(ctx, builder, m1, m2, output_vec=None):
    y = len(m2.type.pointee)
    z = len(m2.type.pointee.element)
    assert len(m1.type.pointee) == z
    # create output vec
    if output_vec is None:
        output_vec = builder.alloca(pnlvm.ir.types.ArrayType(ctx.float_ty, y))
    assert len(output_vec.type.pointee) == y

    # Get the pointer to the first element of the array to convert from [? x double]* -> double*
    v = builder.gep(m1, [ctx.int32_ty(0), ctx.int32_ty(0)])
    out = builder.gep(output_vec, [ctx.int32_ty(0), ctx.int32_ty(0)])

    builtin = ctx.import_llvm_function("__pnl_builtin_vxm_transposed")
    builder.call(builtin, [v, builder.bitcast(m2, ctx.float_ty.as_pointer()),
                            ctx.int32_ty(y), ctx.int32_ty(z), out])
    return output_vec
