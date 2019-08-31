# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM helpers **************************************************************

from llvmlite import ir
from contextlib import contextmanager


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

# Helper that builds a for loop starting at 0, stopping at stop, and incremented by 1
def for_loop_zero_inc(builder, stop, id):
    start = stop.type(0)
    inc = stop.type(1)
    return for_loop(builder, start, stop, inc, id)


def array_ptr_loop(builder, array, id):
    # Assume we'll never have more than 4GB arrays
    stop = ir.IntType(32)(array.type.pointee.count)
    return for_loop_zero_inc(builder, stop, id)


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


def load_extract_scalar_array_one(builder, ptr):
    val = builder.load(ptr)
    if isinstance(val.type, ir.ArrayType) and val.type.count == 1:
        val = builder.extract_value(val, [0])
    return val


def fneg(builder, val, name=""):
    return builder.fsub(val.type(-0.0), val, name)


def is_close(builder, val1, val2, rtol=1e-05, atol=1e-08):
    diff = builder.fsub(val1, val2, "is_close_diff")
    diff_neg = fneg(builder, diff, "is_close_fneg_diff")
    ltz = builder.fcmp_ordered("<", diff, diff.type(0.0), "is_close_ltz")
    abs_diff = builder.select(ltz, diff_neg, diff, "is_close_abs")

    rev2 = fneg(builder, val2, "is_close_fneg2")
    ltz2 = builder.fcmp_ordered("<", val2, val2.type(0.0), "is_close_ltz2")
    abs2 = builder.select(ltz2, rev2, val2, "is_close_abs2")
    rtol = builder.fmul(abs2.type(rtol), abs2, "is_close_rtol")
    atol = builder.fadd(rtol, rtol.type(atol), "is_close_atol")
    return builder.fcmp_ordered("<=", abs_diff, atol, "is_close_cmp")


def all_close(builder, arr1, arr2, rtol=1e-05, atol=1e-08):
    assert arr1.type == arr2.type
    all_ptr = builder.alloca(ir.IntType(1))
    builder.store(all_ptr.type.pointee(1), all_ptr)
    with array_ptr_loop(builder, arr1, "all_close") as (b1, idx):
        val1_ptr = b1.gep(arr1, [idx.type(0), idx])
        val2_ptr = b1.gep(arr1, [idx.type(0), idx])
        val1 = b1.load(val1_ptr)
        val2 = b1.load(val2_ptr)
        res_close = is_close(b1, val1, val2, rtol, atol)

        all_val = b1.load(all_ptr)
        all_val = b1.and_(all_val, res_close)
        b1.store(all_val, all_ptr)

    return builder.load(all_ptr)


class ConditionGenerator:
    def __init__(self, ctx, composition):
        self.ctx = ctx
        self.composition = composition
        self._zero = ctx.int32_ty(0) if ctx is not None else None

    def get_private_condition_struct_type(self, composition):
        time_stamp_struct = ir.LiteralStructType([self.ctx.int32_ty,
                                                  self.ctx.int32_ty,
                                                  self.ctx.int32_ty])

        structure = ir.LiteralStructType([
            time_stamp_struct,  # current time stamp
            ir.ArrayType(       # for each node
                ir.LiteralStructType([
                    self.ctx.int32_ty,  # number of executions
                    time_stamp_struct   # time stamp of last execution
                ]), len(composition.nodes)
            )
        ])
        return structure

    def get_private_condition_initializer(self, composition):
        return ((0, 0, 0),
                tuple((0, (-1, -1, -1)) for _ in composition.nodes))

    def get_condition_struct_type(self, composition=None):
        composition = self.composition if composition is None else composition
        structs = [self.get_private_condition_struct_type(composition)]
        for node in composition.nodes:
            structs.append(self.get_condition_struct_type(node) if isinstance(node, type(self.composition)) else ir.LiteralStructType([]))
        return ir.LiteralStructType(structs)

    def get_condition_initializer(self, composition=None):
        composition = self.composition if composition is None else composition
        data = [self.get_private_condition_initializer(composition)]
        for node in composition.nodes:
            data.append(self.get_condition_initializer(node) if isinstance(node, type(self.composition)) else tuple())
        return tuple(data)

    def bump_ts(self, builder, cond_ptr, count=(0, 0, 1)):
        ts_ptr = builder.gep(cond_ptr, [self._zero, self._zero, self._zero])
        ts = builder.load(ts_ptr)

        # run, pass, step
        for idx in range(3):
            if idx == 0 or all(v == 0 for v in count[:idx]) == 0:
                el = builder.extract_value(ts, idx)
                el = builder.add(el, self.ctx.int32_ty(count[idx]))
            else:
                el = self.ctx.int32_ty(0)
            ts = builder.insert_value(ts, el, idx)

        builder.store(ts, ts_ptr)
        return builder

    def ts_compare(self, builder, ts1, ts2, comp):
        assert comp == '<'
        part_eq = []
        part_cmp = []

        for element in range(3):
            a = builder.extract_value(ts1, element)
            b = builder.extract_value(ts2, element)
            part_eq.append(builder.icmp_signed('==', a, b))
            part_cmp.append(builder.icmp_signed(comp, a, b))

        trial = builder.and_(builder.not_(part_eq[0]), part_cmp[0])
        run = builder.and_(part_eq[0],
                           builder.and_(builder.not_(part_eq[1]), part_cmp[1]))
        step = builder.and_(builder.and_(part_eq[0], part_eq[1]),
                            part_cmp[2])

        return builder.or_(trial, builder.or_(run, step))

    def __get_node_status_ptr(self, builder, cond_ptr, node):
        node_idx = self.ctx.int32_ty(self.composition.nodes.index(node))
        return builder.gep(cond_ptr, [self._zero, self._zero, self.ctx.int32_ty(1), node_idx])

    def __get_node_ts(self, builder, cond_ptr, node):
        status_ptr = self.__get_node_status_ptr(builder, cond_ptr, node)
        ts_ptr = builder.gep(status_ptr, [self.ctx.int32_ty(0),
                                          self.ctx.int32_ty(1)])
        return builder.load(ts_ptr)

    def generate_update_after_run(self, builder, cond_ptr, node):
        status_ptr = self.__get_node_status_ptr(builder, cond_ptr, node)
        status = builder.load(status_ptr)

        # Update number of runs
        runs = builder.extract_value(status, 0)
        runs = builder.add(runs, self.ctx.int32_ty(1))
        status = builder.insert_value(status, runs, 0)

        # Update time stamp
        ts = builder.gep(cond_ptr, [self._zero, self._zero, self._zero])
        ts = builder.load(ts)
        status = builder.insert_value(status, ts, 1)

        builder.store(status, status_ptr)

    def generate_ran_this_pass(self, builder, cond_ptr, node):
        global_ts = builder.load(builder.gep(cond_ptr, [self._zero, self._zero, self._zero]))
        global_pass = builder.extract_value(global_ts, 1)
        global_run = builder.extract_value(global_ts, 0)

        node_ts = self.__get_node_ts(builder, cond_ptr, node)
        node_pass = builder.extract_value(node_ts, 1)
        node_run = builder.extract_value(node_ts, 0)

        pass_eq = builder.icmp_signed("==", node_pass, global_pass)
        run_eq = builder.icmp_signed("==", node_run, global_run)
        return builder.and_(pass_eq, run_eq)

    def generate_ran_this_trial(self, builder, cond_ptr, node):
        global_ts = builder.load(builder.gep(cond_ptr, [self._zero, self._zero, self._zero]))
        global_run = builder.extract_value(global_ts, 0)

        node_ts = self.__get_node_ts(builder, cond_ptr, node)
        node_run = builder.extract_value(node_ts, 0)

        return builder.icmp_signed("==", node_run, global_run)

    def generate_sched_condition(self, builder, condition, cond_ptr, node):

        from psyneulink.core.scheduling.condition import All, AllHaveRun, Always, EveryNCalls
        if isinstance(condition, Always):
            return ir.IntType(1)(1)
        elif isinstance(condition, All):
            agg_cond = ir.IntType(1)(1)
            for cond in condition.args:
                cond_res = self.generate_sched_condition(builder, cond, cond_ptr, node)
                agg_cond = builder.and_(agg_cond, cond_res)
            return agg_cond
        elif isinstance(condition, AllHaveRun):
            run_cond = ir.IntType(1)(1)
            array_ptr = builder.gep(cond_ptr, [self._zero, self._zero, self.ctx.int32_ty(1)])
            for node in self.composition.nodes:
                node_ran = self.generate_ran_this_trial(builder, cond_ptr, node)
                run_cond = builder.and_(run_cond, node_ran)
            return run_cond
        elif isinstance(condition, EveryNCalls):
            target, count = condition.args

            target_idx = self.ctx.int32_ty(self.composition.nodes.index(target))

            array_ptr = builder.gep(cond_ptr, [self._zero, self._zero, self.ctx.int32_ty(1)])
            target_status = builder.load(builder.gep(array_ptr, [self._zero, target_idx]))

            # Check number of runs
            target_runs = builder.extract_value(target_status, 0, target.name + " runs")
            ran = builder.icmp_unsigned('>', target_runs, self._zero)
            remainder = builder.urem(target_runs, self.ctx.int32_ty(count))
            divisible = builder.icmp_unsigned('==', remainder, self._zero)
            completedNruns = builder.and_(ran, divisible)

            # Check that we have not run yet
            my_time_stamp = self.__get_node_ts(builder, cond_ptr, node)
            target_time_stamp = self.__get_node_ts(builder, cond_ptr, target)
            ran_after_me = self.ts_compare(builder, my_time_stamp, target_time_stamp, '<')

            # Return: target.calls % N == 0 AND me.last_time < target.last_time
            return builder.and_(completedNruns, ran_after_me)

        assert False, "Unsupported scheduling condition: {}".format(condition)
