# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM helpers **************************************************************

import ast
from contextlib import contextmanager
import warnings

from llvmlite import ir

from .debug import debug_env


@contextmanager
def for_loop(builder, start, stop, inc, id):
    # Initialize index variable
    assert start.type is stop.type
    index_var = builder.alloca(stop.type, name=id + "_index_var_loc")
    builder.store(start, index_var)

    # basic blocks
    cond_block = builder.append_basic_block(id + "-cond-bb")
    out_block = None

    # Loop condition
    builder.branch(cond_block)
    with builder.goto_block(cond_block):
        tmp = builder.load(index_var, name=id + "_cond_index_var")
        cond = builder.icmp_signed("<", tmp, stop, name=id + "_loop_cond")

        # Loop body
        with builder.if_then(cond, likely=True):
            index = builder.load(index_var, name=id + "_loop_index_var")

            yield (builder, index)

            index = builder.add(index, inc, name=id + "_index_var_inc")
            builder.store(index, index_var)
            builder.branch(cond_block)

        out_block = builder.block

    builder.position_at_end(out_block)


def for_loop_zero_inc(builder, stop, id):
    start = stop.type(0)
    inc = stop.type(1)
    return for_loop(builder, start, stop, inc, id)


def array_ptr_loop(builder, array, id):
    # Assume we'll never have more than 4GB arrays
    stop = ir.IntType(32)(array.type.pointee.count)
    return for_loop_zero_inc(builder, stop, id)

def memcpy(builder, dst, src):

    bool_ty = ir.IntType(1)
    char_ptr_ty = ir.IntType(8).as_pointer()
    ptr_src = builder.bitcast(src, char_ptr_ty)
    ptr_dst = builder.bitcast(dst, char_ptr_ty)

    obj_size_ty = ir.FunctionType(ir.IntType(64), [char_ptr_ty, bool_ty, bool_ty, bool_ty])
    obj_size_f = builder.function.module.declare_intrinsic("llvm.objectsize.i64", [], obj_size_ty)
    # the params are: obj pointer, 0 on unknown size, NULL is unknown, size at runtime
    obj_size = builder.call(obj_size_f, [ptr_dst, bool_ty(1), bool_ty(0), bool_ty(0)])

    if "unaligned_copy" in debug_env:
        memcpy_ty = ir.FunctionType(ir.VoidType(), [char_ptr_ty, char_ptr_ty, obj_size.type, bool_ty])
        memcpy_f = builder.function.module.declare_intrinsic("llvm.memcpy", [], memcpy_ty)
        builder.call(memcpy_f, [ptr_dst, ptr_src, obj_size, bool_ty(0)])
    else:
        int_ty = ir.IntType(32)
        int_ptr_ty = int_ty.as_pointer()
        obj_size = builder.add(obj_size, obj_size.type((int_ty.width // 8) - 1))
        obj_size = builder.udiv(obj_size, obj_size.type(int_ty.width // 8))
        ptr_src = builder.bitcast(src, int_ptr_ty)
        ptr_dst = builder.bitcast(dst, int_ptr_ty)

        with for_loop_zero_inc(builder, obj_size, id="memcopy_loop") as (b, idx):
            src = b.gep(ptr_src, [idx])
            dst = b.gep(ptr_dst, [idx])
            b.store(b.load(src), dst)

    return builder


def fclamp(builder, val, min_val, max_val):
    min_val = min_val if isinstance(min_val, ir.Value) else val.type(min_val)
    max_val = max_val if isinstance(max_val, ir.Value) else val.type(max_val)

    cond = builder.fcmp_unordered("<", val, min_val)
    tmp = builder.select(cond, min_val, val)
    cond = builder.fcmp_unordered(">", tmp, max_val)
    return builder.select(cond, max_val, tmp)


def uint_min(builder, val, other):
    other = other if isinstance(other, ir.Value) else val.type(other)

    cond = builder.icmp_unsigned("<=", val, other)
    return builder.select(cond, val, other)


def get_param_ptr(builder, component, params_ptr, param_name):
    # check if the passed location matches expected size
    assert len(params_ptr.type.pointee) == len(component.llvm_param_ids)

    idx = ir.IntType(32)(component.llvm_param_ids.index(param_name))
    return builder.gep(params_ptr, [ir.IntType(32)(0), idx],
                       name="ptr_param_{}_{}".format(param_name, component.name))


def get_state_ptr(builder, component, state_ptr, stateful_name, hist_idx=0):
    # check if the passed location matches expected size
    assert len(state_ptr.type.pointee) == len(component.llvm_state_ids)

    idx = ir.IntType(32)(component.llvm_state_ids.index(stateful_name))
    ptr = builder.gep(state_ptr, [ir.IntType(32)(0), idx],
                      name="ptr_state_{}_{}".format(stateful_name,
                                                    component.name))
    # The first dimension of arrays is history
    if hist_idx is not None and isinstance(ptr.type.pointee, ir.ArrayType):
        assert len(ptr.type.pointee) > hist_idx, \
            "History not available: {} ({})".format(ptr.type.pointee, hist_idx)
        ptr = builder.gep(state_ptr, [ir.IntType(32)(0), idx,
                                      ir.IntType(32)(hist_idx)],
                          name="ptr_state_{}_{}_hist{}".format(stateful_name,
                                                               component.name,
                                                               hist_idx))
    return ptr


def get_state_space(builder, component, state_ptr, name):
    val_ptr = get_state_ptr(builder, component, state_ptr, name, None)
    for i in range(len(val_ptr.type.pointee) - 1, 0, -1):
        dest_ptr = get_state_ptr(builder, component, state_ptr, name, i)
        src_ptr = get_state_ptr(builder, component, state_ptr, name, i - 1)
        builder.store(builder.load(src_ptr), dest_ptr)

    return get_state_ptr(builder, component, state_ptr, name)


def unwrap_2d_array(builder, element):
    if isinstance(element.type.pointee, ir.ArrayType) and isinstance(element.type.pointee.element, ir.ArrayType):
        assert element.type.pointee.count == 1
        return builder.gep(element, [ir.IntType(32)(0), ir.IntType(32)(0)])
    return element


def load_extract_scalar_array_one(builder, ptr):
    val = builder.load(ptr)
    if isinstance(val.type, ir.ArrayType) and val.type.count == 1:
        val = builder.extract_value(val, [0])
    return val


def umul_lo_hi(builder, a, b):
    assert a.type.width == b.type.width

    a_val = builder.zext(a, ir.IntType(a.type.width * 2))
    b_val = builder.zext(b, ir.IntType(b.type.width * 2))
    res = builder.mul(a_val, b_val)

    lo = builder.trunc(res, a.type)
    hi = builder.lshr(res, res.type(a.type.width))
    hi = builder.trunc(hi, a.type)
    return lo, hi


def fneg(builder, val, name=""):
    return builder.fsub(val.type(-0.0), val, name)


def exp(ctx, builder, x):
    exp_f = ctx.get_builtin("exp", [x.type])
    return builder.call(exp_f, [x])

def log(ctx, builder, x):
    log_f = ctx.get_builtin("log", [x.type])
    return builder.call(log_f, [x])

def log1p(ctx, builder, x):
    log_f = ctx.get_builtin("log", [x.type])
    x1p = builder.fadd(x, x.type(1))
    return builder.call(log_f, [x1p])

def sqrt(ctx, builder, x):
    sqrt_f = ctx.get_builtin("sqrt", [x.type])
    return builder.call(sqrt_f, [x])

def tanh(ctx, builder, x):
    tanh_f = ctx.get_builtin("tanh", [x.type])
    return builder.call(tanh_f, [x])


def coth(ctx, builder, x):
    coth_f = ctx.get_builtin("coth", [x.type])
    return builder.call(coth_f, [x])


def csch(ctx, builder, x):
    csch_f = ctx.get_builtin("csch", [x.type])
    return builder.call(csch_f, [x])


def is_close(ctx, builder, val1, val2, rtol=1e-05, atol=1e-08):
    assert val1.type == val2.type
    is_close_f = ctx.get_builtin("is_close_{}".format(val1.type))
    rtol_val = val1.type(rtol)
    atol_val = val1.type(atol)
    return builder.call(is_close_f, [val1, val2, rtol_val, atol_val])


def all_close(ctx, builder, arr1, arr2, rtol=1e-05, atol=1e-08):
    assert arr1.type == arr2.type
    all_ptr = builder.alloca(ir.IntType(1), name="all_close_slot")
    builder.store(all_ptr.type.pointee(1), all_ptr)
    with array_ptr_loop(builder, arr1, "all_close") as (b1, idx):
        val1_ptr = b1.gep(arr1, [idx.type(0), idx])
        val2_ptr = b1.gep(arr2, [idx.type(0), idx])
        val1 = b1.load(val1_ptr)
        val2 = b1.load(val2_ptr)
        res_close = is_close(ctx, b1, val1, val2, rtol, atol)

        all_val = b1.load(all_ptr)
        all_val = b1.and_(all_val, res_close)
        b1.store(all_val, all_ptr)

    return builder.load(all_ptr)


def create_sample(builder, allocation, search_space, idx):
    # Construct allocation corresponding to this index
    for i in reversed(range(len(search_space.type.pointee))):
        slot_ptr = builder.gep(allocation, [idx.type(0), idx.type(i)])

        dim_ptr = builder.gep(search_space, [idx.type(0), idx.type(i)])
        # Iterators store {start, step, num}
        if isinstance(dim_ptr.type.pointee,  ir.LiteralStructType):
            iter_val = builder.load(dim_ptr)
            dim_start = builder.extract_value(iter_val, 0)
            dim_step = builder.extract_value(iter_val, 1)
            dim_size = builder.extract_value(iter_val, 2)
            dim_idx = builder.urem(idx, dim_size)
            val = builder.uitofp(dim_idx, dim_step.type)
            val = builder.fmul(val, dim_step)
            val = builder.fadd(val, dim_start)
        elif isinstance(dim_ptr.type.pointee,  ir.ArrayType):
            # Otherwise it's just an array
            dim_size = idx.type(len(dim_ptr.type.pointee))
            dim_idx = builder.urem(idx, dim_size)
            val_ptr = builder.gep(dim_ptr, [idx.type(0), dim_idx])
            val = builder.load(val_ptr)
        else:
            assert False, "Unknown dimension type: {}".format(dim_ptr.type)

        idx = builder.udiv(idx, dim_size)

        builder.store(val, slot_ptr)


def convert_type(builder, val, t):
    assert isinstance(t, ir.Type)
    if val.type == t:
        return val

    if is_floating_point(val) and is_boolean(t):
        # convert any scalar to bool by comparing to 0
        # Python converts both 0.0 and -0.0 to False
        return builder.fcmp_unordered("!=", val, val.type(0.0))

    if is_boolean(val) and is_floating_point(t):
        # float(True) == 1.0, float(False) == 0.0
        return builder.select(val, t(1.0), t(0.0))

    if is_integer(val) and is_integer(t):
        if val.type.width > t.width:
            return builder.trunc(val, t)
        elif val.type.width < t.width:
            # Python integers are signed
            return builder.zext(val, t)
        else:
            assert False, "Unknown integer conversion: {} -> {}".format(val.type, t)

    if is_integer(val) and is_floating_point(t):
        # Python integers are signed
        return builder.sitofp(val, t)

    if is_floating_point(val) and is_integer(t):
        # Python integers are signed
        return builder.fptosi(val, t)

    if is_floating_point(val) and is_floating_point(t):
        if isinstance(val.type, ir.HalfType) or isinstance(t, ir.DoubleType):
            return builder.fpext(val, t)
        elif isinstance(val.type, ir.DoubleType) or isinstance(t, ir.HalfType):
            # FIXME: Direct conversion from double to half needs a runtime
            #        function (__truncdfhf2). llvmlite MCJIT fails to provide
            #        it and instead generates invocation of a NULL pointer.
            #        Use double conversion (double->float->half) instead.
            #        Both steps can be done in one CPU instruction,
            #        but the result can be slightly different
            #        see: https://github.com/numba/llvmlite/issues/834
            if isinstance(val.type, ir.DoubleType) and isinstance(t, ir.HalfType):
                val = builder.fptrunc(val, ir.FloatType())
            return builder.fptrunc(val, t)
        else:
            assert False, "Unknown float conversion: {} -> {}".format(val.type, t)

    assert False, "Unknown type conversion: {} -> {}".format(val.type, t)


def is_pointer(x):
    type_t = getattr(x, "type", x)
    assert isinstance(type_t, ir.Type)
    return isinstance(type_t, ir.PointerType)

def is_floating_point(x):
    type_t = getattr(x, "type", x)
    # dereference pointer
    if is_pointer(type_t):
        type_t = type_t.pointee
    return isinstance(type_t, (ir.DoubleType, ir.FloatType, ir.HalfType))

def is_integer(x):
    type_t = getattr(x, "type", x)
    # dereference pointer
    if is_pointer(type_t):
        type_t = type_t.pointee
    return isinstance(type_t, ir.IntType)

def is_scalar(x):
    return is_integer(x) or is_floating_point(x)

def is_vector(x):
    type_t = getattr(x, "type", x)
    if is_pointer(type_t):
        type_t = type_t.pointee
    return isinstance(type_t, ir.ArrayType) and is_scalar(type_t.element)

def is_2d_matrix(x):
    type_t = getattr(x, "type", x)
    if is_pointer(type_t):
        type_t = type_t.pointee
    return isinstance(type_t, ir.ArrayType) and is_vector(type_t.element)

def is_boolean(x):
    type_t = getattr(x, "type", x)
    if is_pointer(type_t):
        type_t = type_t.pointee
    return isinstance(type_t, ir.IntType) and type_t.width == 1

def get_array_shape(x):
    x_ty = x.type
    if is_pointer(x):
        x_ty = x_ty.pointee

    assert isinstance(x_ty, ir.ArrayType), f"Tried to get shape of non-array type: {x_ty}"
    dimensions = []
    while hasattr(x_ty, "count"):
        dimensions.append(x_ty.count)
        x_ty = x_ty.element

    return dimensions

def array_from_shape(shape, element_ty):
    array_ty = element_ty
    for dim in reversed(shape):
        array_ty = ir.ArrayType(array_ty, dim)
    return array_ty

@contextmanager
def recursive_iterate_arrays(ctx, builder, *args, loop_id="recursive_iteration"):
    """Recursively iterates over all elements in scalar arrays of the same shape"""

    assert len(args) > 0, "Need at least one array to iterate over!"
    assert all(isinstance(arr.type.pointee, ir.ArrayType) for arr in args), "Can only iterate over arrays!"

    u = args[0]
    assert all(len(u.type.pointee) == len(v.type.pointee) for v in args), "Tried to iterate over differing lengths!"

    with array_ptr_loop(builder, u, loop_id) as (b, idx):
        arg_ptrs = tuple(b.gep(arr, [ctx.int32_ty(0), idx]) for arr in args)
        if is_scalar(arg_ptrs[0]):
            yield (b, *arg_ptrs)
        else:
            with recursive_iterate_arrays(ctx, b, *arg_ptrs) as (b, *nested_args):
                yield (b, *nested_args)

def printf(ctx, builder, fmt, *args, tags:set):

    tags = frozenset(tags)
    user_tags = frozenset(ast.literal_eval(debug_env.get("printf_tags", "[]")))
    if "all" not in user_tags and "always" not in tags and not tags.intersection(user_tags):
        return

    # Set up the formatting string as global symbol
    int8 = ir.IntType(8)
    fmt_data = bytearray((fmt + "\0").encode("utf8"))
    fmt_ty = ir.ArrayType(int8, len(fmt_data))

    ir_module = builder.function.module
    global_fmt = ir.GlobalVariable(ir_module,
                                   fmt_ty,
                                   name="printf_fmt_" + str(len(ir_module.globals)))
    global_fmt.linkage = "internal"
    global_fmt.global_constant = True
    global_fmt.initializer = fmt_ty(fmt_data)

    printf_ty = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True)
    get_printf_addr_f = ctx.get_builtin("get_printf_address", [])
    printf_address = builder.call(get_printf_addr_f, [])

    printf_is_not_null = builder.icmp_unsigned("!=", printf_address, printf_address.type(0))
    with builder.if_then(printf_is_not_null, likely=True):
        printf_f = builder.inttoptr(printf_address, printf_ty.as_pointer())

        fmt_ptr = builder.gep(global_fmt, [ir.IntType(32)(0), ir.IntType(32)(0)])
        conv_args = [builder.fpext(a, ir.DoubleType()) if is_floating_point(a) else a for a in args]
        builder.call(printf_f, [fmt_ptr] + conv_args)


def printf_float_array(ctx, builder, array, prefix="", suffix="\n", *, tags:set):
    printf(ctx, builder, prefix, tags=tags)

    with array_ptr_loop(builder, array, "print_array_loop") as (b1, i):
        printf(ctx, b1, "%lf ", b1.load(b1.gep(array, [i.type(0), i])), tags=tags)

    printf(ctx, builder, suffix, tags=tags)


def printf_float_matrix(ctx, builder, matrix, prefix="", suffix="\n", *, tags:set):
    printf(ctx, builder, prefix, tags=tags)
    with array_ptr_loop(builder, matrix, "print_row_loop") as (b1, i):
        row = b1.gep(matrix, [i.type(0), i])
        printf_float_array(ctx, b1, row, suffix="\n", tags=tags)

    printf(ctx, builder, suffix, tags=tags)
