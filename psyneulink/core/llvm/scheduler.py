from enum import Enum
from llvmlite import ir

from . import helpers
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.scheduling.condition import All, AllHaveRun, Always, Any, AtPass, AtTrial, BeforeNCalls, AtNCalls, AfterNCalls, \
    EveryNCalls, Never, Not, WhenFinished, WhenFinishedAny, WhenFinishedAll, Threshold

class ConditionGenerator:
    class TimeIndex(Enum):
        TRIAL = 0,
        PASS  = 1,
        STEP  = 2,

    def __init__(self, ctx, composition):
        self.ctx = ctx
        self.composition = composition
        self._zero = ctx.int32_ty(0) if ctx is not None else None

    def get_private_condition_struct_type(self, composition):
        time_stamp_struct = ir.LiteralStructType([self.ctx.int32_ty for _ in self.TimeIndex])
        nodes_time_stamps_array = ir.ArrayType(time_stamp_struct, len(composition.nodes))

        return ir.LiteralStructType((time_stamp_struct, nodes_time_stamps_array))

    def get_private_condition_initializer(self, composition):
        init_global = tuple(0 for _ in self.TimeIndex)
        init_node = tuple(-1 for _ in self.TimeIndex)

        return (init_global, tuple(init_node for _ in composition.nodes))

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
        Indices greater than the incremented one are zeroed.
        """

        # Only one element should be non-zero
        assert count.count(0) == len(count) - 1

        # Get timestruct pointer
        ts_ptr = self.__get_global_ts_ptr(builder, cond_ptr)
        ts = builder.load(ts_ptr)

        assert len(ts.type) == len(count)

        # Update run, pass, step of ts
        for idx in range(len(ts.type)):
            if all(v == 0 for v in count[:idx]):
                el = builder.extract_value(ts, idx)
                el = builder.add(el, el.type(count[idx]))
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

    def __get_global_ts_ptr(self, builder, cond_ptr):
        # derefence the structure, the first element (private structure),
        # and the first element of the private strucutre is the global ts.
        return builder.gep(cond_ptr, [self._zero, self._zero, self._zero])

    def __get_node_ts_ptr(self, builder, cond_ptr, node):
        node_idx = self.ctx.int32_ty(self.composition.nodes.index(node))

        # derefence the structure, the first element (private structure), the
        # second element is the node time stamp array, use index in the array
        return builder.gep(cond_ptr, [self._zero, self._zero, self.ctx.int32_ty(1), node_idx])

    def __get_node_ts(self, builder, cond_ptr, node):
        ts_ptr = self.__get_node_ts_ptr(builder, cond_ptr, node)
        return builder.load(ts_ptr)

    def get_global_ts(self, builder, cond_ptr):
        ts_ptr = builder.gep(cond_ptr, [self._zero, self._zero, self._zero])
        return builder.load(ts_ptr)

    def _extract_global_time(self, builder, cond_ptr, time_index):
        global_ts = self.get_global_ts(builder, cond_ptr)
        return builder.extract_value(global_ts, time_index.value)

    def get_global_trial(self, builder, cond_ptr):
        return self._extract_global_time(builder, cond_ptr, self.TimeIndex.TRIAL)

    def get_global_pass(self, builder, cond_ptr):
        return self._extract_global_time(builder, cond_ptr, self.TimeIndex.PASS)

    def get_global_step(self, builder, cond_ptr):
        return self._extract_global_time(builder, cond_ptr, self.TimeIndex.STEP)

    def generate_update_after_node_execution(self, builder, cond_ptr, node):
        # Update time stamp of the last execution
        global_ts_ptr = self.__get_global_ts_ptr(builder, cond_ptr)
        node_ts_ptr = self.__get_node_ts_ptr(builder, cond_ptr, node)

        global_ts = builder.load(global_ts_ptr)
        builder.store(global_ts, node_ts_ptr)

    def _node_executions_for_scale(self, builder, node, node_states, time_scale:TimeScale):
        node_idx = self.composition._get_node_index(node)
        node_state = builder.gep(node_states, [self._zero, self.ctx.int32_ty(node_idx)])
        num_exec_ptr = helpers.get_state_ptr(builder, node, node_state, "num_executions")

        count_ptr = builder.gep(num_exec_ptr, [self._zero, self.ctx.int32_ty(time_scale.value)])
        return builder.load(count_ptr)

    def generate_sched_condition(self, builder, condition, cond_ptr, self_node, is_finished_callbacks, nodes_states):

        if isinstance(condition, Always):
            return self.ctx.bool_ty(1)

        if isinstance(condition, Never):
            return self.ctx.bool_ty(0)

        elif isinstance(condition, Not):
            orig_condition = self.generate_sched_condition(builder, condition.condition, cond_ptr, self_node, is_finished_callbacks, nodes_states)
            return builder.not_(orig_condition)

        elif isinstance(condition, All):
            agg_cond = self.ctx.bool_ty(1)
            for cond in condition.args:
                cond_res = self.generate_sched_condition(builder, cond, cond_ptr, self_node, is_finished_callbacks, nodes_states)
                agg_cond = builder.and_(agg_cond, cond_res)
            return agg_cond

        elif isinstance(condition, AllHaveRun):
            # Extract dependencies
            dependencies = self.composition.nodes
            if len(condition.args) > 0:
                dependencies = condition.args

            run_cond = self.ctx.bool_ty(1)
            for node in dependencies:
                count = self._node_executions_for_scale(builder, node, nodes_states, condition.time_scale)

                node_ran = builder.icmp_unsigned(">", count, count.type(0))
                run_cond = builder.and_(run_cond, node_ran)

            return run_cond

        elif isinstance(condition, Any):
            agg_cond = self.ctx.bool_ty(0)
            for cond in condition.args:
                cond_res = self.generate_sched_condition(builder, cond, cond_ptr, self_node, is_finished_callbacks, nodes_states)
                agg_cond = builder.or_(agg_cond, cond_res)
            return agg_cond

        elif isinstance(condition, AtTrial):
            trial_num = condition.args[0]
            current_trial = self.get_global_trial(builder, cond_ptr)
            return builder.icmp_unsigned("==", current_trial, current_trial.type(trial_num))

        elif isinstance(condition, AtPass):
            pass_num = condition.args[0]
            current_pass = self.get_global_pass(builder, cond_ptr)
            return builder.icmp_unsigned("==", current_pass, current_pass.type(pass_num))

        elif isinstance(condition, EveryNCalls):
            target, count = condition.args
            assert count == 1, "EveryNCalls is only supported with count == 1 (count: {})".format(count)

            target_ts = self.__get_node_ts(builder, cond_ptr, target)
            node_ts = self.__get_node_ts(builder, cond_ptr, self_node)

            # If target ran after node did its TS will be greater node's
            return self.ts_compare(builder, node_ts, target_ts, '<')

        elif isinstance(condition, BeforeNCalls):
            node, count = condition.args
            num_execs = self._node_executions_for_scale(builder, node, nodes_states, condition.time_scale)

            return builder.icmp_unsigned('<', num_execs, num_execs.type(count))

        elif isinstance(condition, AtNCalls):
            node, count = condition.args
            num_execs = self._node_executions_for_scale(builder, node, nodes_states, condition.time_scale)

            return builder.icmp_unsigned('==', num_execs, num_execs.type(count))

        elif isinstance(condition, AfterNCalls):
            node, count = condition.args
            num_execs = self._node_executions_for_scale(builder, node, nodes_states, condition.time_scale)

            return builder.icmp_unsigned('>=', num_execs, num_execs.type(count))

        elif isinstance(condition, WhenFinished):
            # The first argument is the target node
            assert len(condition.args) == 1
            target = is_finished_callbacks[condition.args[0]]
            is_finished_f = self.ctx.import_llvm_function(target[0], tags=frozenset({"is_finished", "node_assembly"}))
            return builder.call(is_finished_f, target[1])

        elif isinstance(condition, WhenFinishedAny):
            assert len(condition.args) > 0

            run_cond = self.ctx.bool_ty(0)
            for node in condition.args:
                target = is_finished_callbacks[node]
                is_finished_f = self.ctx.import_llvm_function(target[0], tags=frozenset({"is_finished", "node_assembly"}))
                node_is_finished = builder.call(is_finished_f, target[1])

                run_cond = builder.or_(run_cond, node_is_finished)

            return run_cond

        elif isinstance(condition, WhenFinishedAll):
            assert len(condition.args) > 0

            run_cond = self.ctx.bool_ty(1)
            for node in condition.args:
                target = is_finished_callbacks[node]
                is_finished_f = self.ctx.import_llvm_function(target[0], tags=frozenset({"is_finished", "node_assembly"}))
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
            param_ptr = helpers.get_state_ptr(builder, target, node_state, param)

            # parameters in state include history of at least one element
            # so they are always arrays.
            assert isinstance(param_ptr.type.pointee, ir.ArrayType)

            if indices is None:
                indices = [0, 0]
            elif isinstance(indices, TimeScale):
                indices = [indices.value]

            param_ptr = builder.gep(param_ptr, [self.ctx.int32_ty(x) for x in [0] + list(indices)])

            val = builder.load(param_ptr)
            val = helpers.convert_type(builder, val, ir.DoubleType())
            threshold = val.type(threshold)

            if comparator == '==':
                return helpers.is_close(self.ctx, builder, val, threshold, condition.rtol, condition.atol)
            elif comparator == '!=':
                return builder.not_(helpers.is_close(self.ctx, builder, val, threshold, condition.rtol, condition.atol))
            else:
                return builder.fcmp_ordered(comparator, val, threshold)

        assert False, "Unsupported scheduling condition: {}".format(condition)
