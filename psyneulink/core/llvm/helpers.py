# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM helpers **************************************************************

from contextlib import contextmanager
from ctypes import util
import warnings
import sys

from llvmlite import ir
import llvmlite.binding as llvm


from .debug import debug_env
from psyneulink.core.scheduling.condition import All, AllHaveRun, Always, Any, AtPass, AtTrial, BeforeNCalls, AtNCalls, AfterNCalls, \
    EveryNCalls, Never, Not, WhenFinished, WhenFinishedAny, WhenFinishedAll, Threshold
from psyneulink.core.scheduling.time import TimeScale


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
            return builder.sext(val, t)
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
            assert val.type == t
            return val

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

def recursive_iterate_arrays(ctx, builder, u, *args):
    """Recursively iterates over all elements in scalar arrays of the same shape"""
    assert isinstance(u.type.pointee, ir.ArrayType), "Can only iterate over arrays!"
    assert all(len(u.type.pointee) == len(v.type.pointee) for v in args), "Tried to iterate over differing lengths!"
    with array_ptr_loop(builder, u, "recursive_iteration") as (b, idx):
        u_ptr = b.gep(u, [ctx.int32_ty(0), idx])
        arg_ptrs = (b.gep(v, [ctx.int32_ty(0), idx]) for v in args)
        if is_scalar(u_ptr):
            yield (u_ptr, *arg_ptrs)
        else:
            yield from recursive_iterate_arrays(ctx, b, u_ptr, *arg_ptrs)

# TODO: Remove this function. Can be replaced by `recursive_iterate_arrays`
def call_elementwise_operation(ctx, builder, x, operation, output_ptr):
    """Recurse through an array structure and call operation on each scalar element of the structure. Store result in output_ptr"""
    for (inp_ptr, out_ptr) in recursive_iterate_arrays(ctx, builder, x, output_ptr):
        builder.store(operation(ctx, builder, builder.load(inp_ptr)), out_ptr)

def printf(builder, fmt, *args, override_debug=False):
    if "print_values" not in debug_env and not override_debug:
        return

    #FIXME: Fix builtin printf and use that instead of this
    libc_name = "msvcrt" if sys.platform == "win32" else "c"
    libc = util.find_library(libc_name)
    if libc is None:
        warnings.warn("Standard libc library not found, 'printf' not available!")
        return

    llvm.load_library_permanently(libc)
    # Address will be none if the symbol is not found
    printf_address = llvm.address_of_symbol("printf")
    if printf_address is None:
        warnings.warn("'printf' symbol not found in libc, 'printf' not available!")
        return

    # Direct pointer constants don't work
    printf_ty = ir.FunctionType(ir.IntType(32), [ir.IntType(8).as_pointer()], var_arg=True)
    printf = builder.inttoptr(ir.IntType(64)(printf_address), printf_ty.as_pointer())
    ir_module = builder.function.module
    fmt += "\0"

    int8 = ir.IntType(8)
    fmt_data = bytearray(fmt.encode("utf8"))
    fmt_ty = ir.ArrayType(int8, len(fmt_data))
    global_fmt = ir.GlobalVariable(ir_module, fmt_ty,
                                   name="printf_fmt_" + str(len(ir_module.globals)))
    global_fmt.linkage = "internal"
    global_fmt.global_constant = True
    global_fmt.initializer = fmt_ty(fmt_data)

    fmt_ptr = builder.gep(global_fmt, [ir.IntType(32)(0), ir.IntType(32)(0)])
    builder.call(printf, [fmt_ptr] + list(args))


def printf_float_array(builder, array, prefix="", suffix="\n", override_debug=False):
    printf(builder, prefix, override_debug=override_debug)

    with array_ptr_loop(builder, array, "print_array_loop") as (b1, i):
        printf(b1, "%lf ", b1.load(b1.gep(array, [ir.IntType(32)(0), i])), override_debug=override_debug)

    printf(builder, suffix, override_debug=override_debug)


def printf_float_matrix(builder, matrix, prefix="", suffix="\n", override_debug=False):
    printf(builder, prefix, override_debug=override_debug)
    with array_ptr_loop(builder, matrix, "print_row_loop") as (b1, i):
        row = b1.gep(matrix, [ir.IntType(32)(0), i])
        printf_float_array(b1, row, suffix="\n", override_debug=override_debug)
    printf(builder, suffix, override_debug=override_debug)


class ConditionGenerator:
    def __init__(self, ctx, composition):
        self.ctx = ctx
        self.composition = composition
        self._zero = ctx.int32_ty(0) if ctx is not None else None

    def get_private_condition_struct_type(self, composition):
        time_stamp_struct = ir.LiteralStructType([self.ctx.int32_ty,   # Trial
                                                  self.ctx.int32_ty,   # Pass
                                                  self.ctx.int32_ty])  # Step

        status_struct = ir.LiteralStructType([
                    self.ctx.int32_ty,  # number of executions in this run
                    time_stamp_struct   # time stamp of last execution
                ])
        structure = ir.LiteralStructType([
            time_stamp_struct,  # current time stamp
            ir.ArrayType(status_struct, len(composition.nodes))  # for each node
        ])
        return structure

    def get_private_condition_initializer(self, composition):
        return ((0, 0, 0),
                tuple((0, (-1, -1, -1)) for _ in composition.nodes))

    def get_condition_struct_type(self, node=None):
        node = self.composition if node is None else node

        subnodes = getattr(node, 'nodes', [])
        structs = [self.get_condition_struct_type(n) for n in subnodes]
        if len(structs) != 0:
            structs.insert(0, self.get_private_condition_struct_type(node))

        return ir.LiteralStructType(structs)

    def get_condition_initializer(self, node=None):
        node = self.composition if node is None else node

        subnodes = getattr(node, 'nodes', [])
        data = [self.get_condition_initializer(n) for n in subnodes]
        if len(data) != 0:
            data.insert(0, self.get_private_condition_initializer(node))

        return tuple(data)

    def bump_ts(self, builder, cond_ptr, count=(0, 0, 1)):
        """
        Increments the time structure of the composition.
        Count should be a tuple where there is a number in only one spot, and zeroes elsewhere.
        Indices greater than that of the one are zeroed.
        """

        # Only one element should be non-zero
        assert count.count(0) == len(count) - 1

        # Get timestruct pointer
        ts_ptr = builder.gep(cond_ptr, [self._zero, self._zero, self._zero])
        ts = builder.load(ts_ptr)

        assert len(ts.type) == len(count)
        # Update run, pass, step of ts
        for idx in range(len(ts.type)):
            if all(v == 0 for v in count[:idx]):
                el = builder.extract_value(ts, idx)
                el = builder.add(el, self.ctx.int32_ty(count[idx]))
            else:
                el = self.ctx.int32_ty(0)
            ts = builder.insert_value(ts, el, idx)

        builder.store(ts, ts_ptr)
        return builder

    def ts_compare(self, builder, ts1, ts2, comp):
        assert comp == '<'

        # True if all elements to the left of the current one are equal
        prefix_eq = self.ctx.bool_ty(1)
        result = self.ctx.bool_ty(0)

        assert ts1.type == ts2.type
        for element in range(len(ts1.type)):
            a = builder.extract_value(ts1, element)
            b = builder.extract_value(ts2, element)

            # Use existing prefix_eq to construct expression
            # for the current element
            element_comp = builder.icmp_signed(comp, a, b)
            current_comp = builder.and_(prefix_eq, element_comp)
            result = builder.or_(result, current_comp)

            # Update prefix_eq
            element_eq = builder.icmp_signed('==', a, b)
            prefix_eq = builder.and_(prefix_eq, element_eq)

        return result

    def __get_node_status_ptr(self, builder, cond_ptr, node):
        node_idx = self.ctx.int32_ty(self.composition.nodes.index(node))
        return builder.gep(cond_ptr, [self._zero, self._zero, self.ctx.int32_ty(1), node_idx])

    def __get_node_ts(self, builder, cond_ptr, node):
        status_ptr = self.__get_node_status_ptr(builder, cond_ptr, node)
        ts_ptr = builder.gep(status_ptr, [self.ctx.int32_ty(0),
                                          self.ctx.int32_ty(1)])
        return builder.load(ts_ptr)

    def get_global_ts(self, builder, cond_ptr):
        ts_ptr = builder.gep(cond_ptr, [self._zero, self._zero, self._zero])
        return builder.load(ts_ptr)

    def generate_update_after_run(self, builder, cond_ptr, node):
        status_ptr = self.__get_node_status_ptr(builder, cond_ptr, node)
        status = builder.load(status_ptr)

        # Update number of runs
        runs = builder.extract_value(status, 0)
        runs = builder.add(runs, self.ctx.int32_ty(1))
        status = builder.insert_value(status, runs, 0)

        # Update time stamp
        ts = self.get_global_ts(builder, cond_ptr)
        status = builder.insert_value(status, ts, 1)

        builder.store(status, status_ptr)

    def generate_ran_this_pass(self, builder, cond_ptr, node):
        global_ts = self.get_global_ts(builder, cond_ptr)
        global_trial = builder.extract_value(global_ts, 0)
        global_pass = builder.extract_value(global_ts, 1)

        node_ts = self.__get_node_ts(builder, cond_ptr, node)
        node_trial = builder.extract_value(node_ts, 0)
        node_pass = builder.extract_value(node_ts, 1)

        pass_eq = builder.icmp_signed("==", node_pass, global_pass)
        trial_eq = builder.icmp_signed("==", node_trial, global_trial)
        return builder.and_(pass_eq, trial_eq)

    def generate_ran_this_trial(self, builder, cond_ptr, node):
        global_ts = self.get_global_ts(builder, cond_ptr)
        global_trial = builder.extract_value(global_ts, 0)

        node_ts = self.__get_node_ts(builder, cond_ptr, node)
        node_trial = builder.extract_value(node_ts, 0)

        return builder.icmp_signed("==", node_trial, global_trial)

    # TODO: replace num_exec_locs use with equivalent from nodes_states
    def generate_sched_condition(self, builder, condition, cond_ptr, node,
                                 is_finished_callbacks, num_exec_locs,
                                 nodes_states):


        if isinstance(condition, Always):
            return self.ctx.bool_ty(1)

        if isinstance(condition, Never):
            return self.ctx.bool_ty(0)

        elif isinstance(condition, Not):
            orig_condition = self.generate_sched_condition(builder, condition.condition, cond_ptr, node, is_finished_callbacks, num_exec_locs, nodes_states)
            return builder.not_(orig_condition)

        elif isinstance(condition, All):
            agg_cond = self.ctx.bool_ty(1)
            for cond in condition.args:
                cond_res = self.generate_sched_condition(builder, cond, cond_ptr, node, is_finished_callbacks, num_exec_locs, nodes_states)
                agg_cond = builder.and_(agg_cond, cond_res)
            return agg_cond

        elif isinstance(condition, AllHaveRun):
            # Extract dependencies
            dependencies = self.composition.nodes
            if len(condition.args) > 0:
                dependencies = condition.args

            run_cond = self.ctx.bool_ty(1)
            for node in dependencies:
                if condition.time_scale == TimeScale.TRIAL:
                    node_ran = self.generate_ran_this_trial(builder, cond_ptr, node)
                elif condition.time_scale == TimeScale.PASS:
                    node_ran = self.generate_ran_this_pass(builder, cond_ptr, node)
                else:
                    assert False, "Unsupported 'AllHaveRun' time scale: {}".format(condition.time_scale)
                run_cond = builder.and_(run_cond, node_ran)
            return run_cond

        elif isinstance(condition, Any):
            agg_cond = self.ctx.bool_ty(0)
            for cond in condition.args:
                cond_res = self.generate_sched_condition(builder, cond, cond_ptr, node, is_finished_callbacks, num_exec_locs, nodes_states)
                agg_cond = builder.or_(agg_cond, cond_res)
            return agg_cond

        elif isinstance(condition, AtTrial):
            trial_num = condition.args[0]
            global_ts = self.get_global_ts(builder, cond_ptr)
            trial = builder.extract_value(global_ts, 0)
            return builder.icmp_unsigned("==", trial, trial.type(trial_num))

        elif isinstance(condition, AtPass):
            pass_num = condition.args[0]
            global_ts = self.get_global_ts(builder, cond_ptr)
            current_pass = builder.extract_value(global_ts, 1)
            return builder.icmp_unsigned("==", current_pass,
                                         current_pass.type(pass_num))

        elif isinstance(condition, EveryNCalls):
            target, count = condition.args
            assert count == 1, "EveryNCalls isonly supprted with count == 1"

            target_ts = self.__get_node_ts(builder, cond_ptr, target)
            node_ts = self.__get_node_ts(builder, cond_ptr, node)

            # If target ran after node did its TS will be greater node's
            return self.ts_compare(builder, node_ts, target_ts, '<')

        elif isinstance(condition, BeforeNCalls):
            target, count = condition.args
            scale = condition.time_scale.value
            target_num_execs_in_scale = builder.gep(num_exec_locs[target],
                                                    [self.ctx.int32_ty(0),
                                                     self.ctx.int32_ty(scale)])
            num_execs = builder.load(target_num_execs_in_scale)

            return builder.icmp_unsigned('<', num_execs, self.ctx.int32_ty(count))

        elif isinstance(condition, AtNCalls):
            target, count = condition.args
            scale = condition.time_scale.value
            target_num_execs_in_scale = builder.gep(num_exec_locs[target],
                                                    [self.ctx.int32_ty(0),
                                                     self.ctx.int32_ty(scale)])
            num_execs = builder.load(target_num_execs_in_scale)
            return builder.icmp_unsigned('==', num_execs, self.ctx.int32_ty(count))

        elif isinstance(condition, AfterNCalls):
            target, count = condition.args
            scale = condition.time_scale.value
            target_num_execs_in_scale = builder.gep(num_exec_locs[target],
                                                    [self.ctx.int32_ty(0),
                                                     self.ctx.int32_ty(scale)])
            num_execs = builder.load(target_num_execs_in_scale)
            return builder.icmp_unsigned('>=', num_execs, self.ctx.int32_ty(count))

        elif isinstance(condition, WhenFinished):
            # The first argument is the target node
            assert len(condition.args) == 1
            target = is_finished_callbacks[condition.args[0]]
            is_finished_f = self.ctx.import_llvm_function(target[0], tags=frozenset({"is_finished", "node_wrapper"}))
            return builder.call(is_finished_f, target[1])

        elif isinstance(condition, WhenFinishedAny):
            assert len(condition.args) > 0

            run_cond = self.ctx.bool_ty(0)
            for node in condition.args:
                target = is_finished_callbacks[node]
                is_finished_f = self.ctx.import_llvm_function(target[0], tags=frozenset({"is_finished", "node_wrapper"}))
                node_is_finished = builder.call(is_finished_f, target[1])

                run_cond = builder.or_(run_cond, node_is_finished)

            return run_cond

        elif isinstance(condition, WhenFinishedAll):
            assert len(condition.args) > 0

            run_cond = self.ctx.bool_ty(1)
            for node in condition.args:
                target = is_finished_callbacks[node]
                is_finished_f = self.ctx.import_llvm_function(target[0], tags=frozenset({"is_finished", "node_wrapper"}))
                node_is_finished = builder.call(is_finished_f, target[1])

                run_cond = builder.and_(run_cond, node_is_finished)

            return run_cond

        elif isinstance(condition, Threshold):
            target = condition.dependency
            param = condition.parameter
            threshold = condition.threshold
            comparator = condition.comparator
            indices = condition.indices

            # Convert execution_count to  ('num_executions', TimeScale.LIFE).
            # These two are identical in compiled semantics.
            if param == 'execution_count':
                assert indices is None
                param = 'num_executions'
                indices = TimeScale.LIFE

            assert param in target.llvm_state_ids, (
                f"Threshold for {target} only supports items in llvm_state_ids"
                f" ({target.llvm_state_ids})"
            )

            node_idx = self.composition._get_node_index(target)
            node_state = builder.gep(nodes_states, [self.ctx.int32_ty(0), self.ctx.int32_ty(node_idx)])
            param_ptr = get_state_ptr(builder, target, node_state, param)

            if isinstance(param_ptr.type.pointee, ir.ArrayType):
                if indices is None:
                    indices = [0, 0]
                elif isinstance(indices, TimeScale):
                    indices = [indices.value]

                indices = [self.ctx.int32_ty(x) for x in [0] + list(indices)]
                param_ptr = builder.gep(param_ptr, indices)

            val = builder.load(param_ptr)
            val = convert_type(builder, val, ir.DoubleType())
            threshold = val.type(threshold)

            if comparator == '==':
                return is_close(self.ctx, builder, val, threshold, condition.rtol, condition.atol)
            elif comparator == '!=':
                return builder.not_(is_close(self.ctx, builder, val, threshold, condition.rtol, condition.atol))
            else:
                return builder.fcmp_ordered(comparator, val, threshold)

        assert False, "Unsupported scheduling condition: {}".format(condition)
