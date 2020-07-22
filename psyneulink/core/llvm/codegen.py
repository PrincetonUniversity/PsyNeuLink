# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM IR Generation **************************************************************
import ast
import warnings


from llvmlite import ir
from contextlib import contextmanager


from psyneulink.core.globals.keywords import AFTER, BEFORE
from psyneulink.core.scheduling.condition import Never
from psyneulink.core.scheduling.time import TimeScale
from . import helpers
from .debug import debug_env


def gen_node_wrapper(ctx, composition, node, *, tags:frozenset):
    assert "node_wrapper" in tags
    func_tags = tags.difference({"node_wrapper"})

    node_function = ctx.import_llvm_function(node, tags=func_tags)
    # FIXME: This is a hack
    is_mech = hasattr(node, 'function')

    data_struct_ptr = ctx.get_data_struct_type(composition).as_pointer()
    args = [
        ctx.get_state_struct_type(composition).as_pointer(),
        ctx.get_param_struct_type(composition).as_pointer(),
        ctx.get_input_struct_type(composition).as_pointer(),
        data_struct_ptr, data_struct_ptr]

    if not is_mech and "reset" not in tags:
        # Add condition struct of the parent composition
        # This includes structures of all nested compositions
        cond_gen = helpers.ConditionGenerator(ctx, composition)
        cond_ty = cond_gen.get_condition_struct_type().as_pointer()
        args.append(cond_ty)

    builder = ctx.create_llvm_function(args, node, node_function.name, tags=tags,
                                       return_type=node_function.type.pointee.return_type)
    llvm_func = builder.function
    for a in llvm_func.args:
        a.attributes.add('nonnull')

    state, params, comp_in, data_in, data_out = llvm_func.args[:5]

    if node is composition.input_CIM:
        # if there are incoming modulatory projections,
        # the input structure is shared
        if composition.parameter_CIM.afferents:
            node_in = builder.gep(comp_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
        else:
            node_in = comp_in
        incoming_projections = []
    elif node is composition.parameter_CIM and node.afferents:
        # if parameter_CIM has afferent projections,
        # their values are in comp_in[1]
        node_in = builder.gep(comp_in, [ctx.int32_ty(0), ctx.int32_ty(1)])
        # And we run no further projection
        incoming_projections = []
    elif not is_mech:
        node_in = builder.alloca(node_function.args[2].type.pointee)
        incoming_projections = node.input_CIM.afferents + node.parameter_CIM.afferents
    else:
        # this path also handles parameter_CIM with no afferent
        # projections. 'comp_in' does not include any extra values,
        # and the entire call should be optimized out.
        node_in = builder.alloca(node_function.args[2].type.pointee)
        incoming_projections = node.afferents

    if "reset" in tags or "is_finished" in tags:
        incoming_projections = []

    # Execute all incoming projections
    inner_projections = list(composition._inner_projections)
    zero = ctx.int32_ty(0)
    projections_params = helpers.get_param_ptr(builder, composition,
                                               params, "projections")
    projections_states = helpers.get_state_ptr(builder, composition,
                                               state, "projections")
    for proj in incoming_projections:
        # Skip autoassociative projections.
        # Recurrent projections are executed as part of the mechanism to
        # make sure their value is up-to-date inside the 'is_finished' loop
        if proj.sender.owner is proj.receiver.owner:
            continue

        # Get location of projection input data
        par_mech = proj.sender.owner
        if par_mech in composition._all_nodes:
            parent_idx = composition._get_node_index(par_mech)
        else:
            assert par_mech is par_mech.composition.output_CIM
            parent_idx = composition.nodes.index(par_mech.composition)

        assert proj.sender in par_mech.output_ports
        output_port_idx = par_mech.output_ports.index(proj.sender)
        proj_in = builder.gep(data_in, [ctx.int32_ty(0),
                                        ctx.int32_ty(0),
                                        ctx.int32_ty(parent_idx),
                                        ctx.int32_ty(output_port_idx)])

        # Get location of projection output (in mechanism's input structure
        rec_port = proj.receiver
        assert rec_port.owner is node or rec_port.owner is node.input_CIM or rec_port.owner is node.parameter_CIM
        indices = [0]
        if proj in rec_port.owner.path_afferents:
            rec_port_idx = rec_port.owner.input_ports.index(rec_port)

            assert proj in rec_port.pathway_projections
            projection_idx = rec_port.pathway_projections.index(proj)

            if not is_mech and node.parameter_CIM.afferents:
                # If there are afferent projections to parameter_CIM
                # the input structure is split between input_CIM
                # and parameter_CIM
                if proj in node.parameter_CIM.afferents:
                    # modulatory projection
                    indices.append(1)
                else:
                    # pathway projection
                    indices.append(0)
            indices.extend([rec_port_idx, projection_idx])
        elif proj in rec_port.owner.mod_afferents:
            # Only mechanism ports list mod projections in mod_afferents
            assert is_mech
            projection_idx = rec_port.owner.mod_afferents.index(proj)
            indices.extend([len(rec_port.owner.input_ports), projection_idx])
        else:
            assert False, "Projection neither pathway nor modulatory"

        proj_out = builder.gep(node_in, [ctx.int32_ty(i) for i in indices])

        # Get projection parameters and state
        proj_idx = inner_projections.index(proj)
        proj_params = builder.gep(projections_params, [zero, ctx.int32_ty(proj_idx)])
        proj_state = builder.gep(projections_states, [zero, ctx.int32_ty(proj_idx)])
        proj_function = ctx.import_llvm_function(proj, tags=func_tags)

        if proj_out.type != proj_function.args[3].type:
            warnings.warn("Shape mismatch: Projection ({}) results does not match the receiver state({}) input: {} vs. {}".format(proj, proj.receiver, proj.defaults.value, proj.receiver.defaults.variable))
            proj_out = builder.bitcast(proj_out, proj_function.args[3].type)
        builder.call(proj_function, [proj_params, proj_state, proj_in, proj_out])


    node_idx = ctx.int32_ty(composition._get_node_index(node))
    nodes_params = helpers.get_param_ptr(builder, composition, params, "nodes")
    nodes_states = helpers.get_state_ptr(builder, composition, state, "nodes")
    node_params = builder.gep(nodes_params, [zero, node_idx])
    node_state = builder.gep(nodes_states, [zero, node_idx])
    node_out = builder.gep(data_out, [zero, zero, node_idx])
    if is_mech:
        call_args = [node_params, node_state, node_in, node_out]
        if len(node_function.args) > 4:
            assert node is composition.controller
            call_args += [params, state, data_in]
        ret = builder.call(node_function, call_args)
    elif "reset" not in tags:
        # FIXME: reinitialization of compositions is not supported
        # Condition and data structures includes parent first
        nested_idx = ctx.int32_ty(composition._get_node_index(node) + 1)
        node_data = builder.gep(data_in, [zero, nested_idx])
        node_cond = builder.gep(llvm_func.args[5], [zero, nested_idx])
        ret = builder.call(node_function, [node_state, node_params, node_in,
                                           node_data, node_cond])
        # Copy output of the nested composition to its output place
        output_idx = node._get_node_index(node.output_CIM)
        result = builder.gep(node_data, [zero, zero, ctx.int32_ty(output_idx)])
        builder.store(builder.load(result), node_out)
    else:
        # composition reset
        ret = None

    if ret is None or isinstance(ret.type, ir.VoidType):
        builder.ret_void()
    else:
        builder.ret(ret)

    return llvm_func


@contextmanager
def _gen_composition_exec_context(ctx, composition, *, tags:frozenset, suffix="", extra_args=[]):
    cond_gen = helpers.ConditionGenerator(ctx, composition)

    name = "_".join(("wrap_exec", *tags ,composition.name + suffix))
    args = [ctx.get_state_struct_type(composition).as_pointer(),
            ctx.get_param_struct_type(composition).as_pointer(),
            ctx.get_input_struct_type(composition).as_pointer(),
            ctx.get_data_struct_type(composition).as_pointer(),
            cond_gen.get_condition_struct_type().as_pointer()]
    builder = ctx.create_llvm_function(args + extra_args, composition, name)
    llvm_func = builder.function

    for a in llvm_func.args:
        a.attributes.add('noalias')

    state, params, comp_in, data_arg, cond, *_ = llvm_func.args
    if "const_params" in debug_env:
        const_params = params.type.pointee(composition._get_param_initializer(None))
        params = builder.alloca(const_params.type, name="const_params_loc")
        builder.store(const_params, params)

    if "alloca_data" in debug_env:
        data = builder.alloca(data_arg.type.pointee)
        data_vals = builder.load(data_arg)
        builder.store(data_vals, data)
    else:
        data = data_arg

    node_tags = tags.union({"node_wrapper"})
    # Call input CIM
    input_cim_w = ctx.get_node_wrapper(composition, composition.input_CIM)
    input_cim_f = ctx.import_llvm_function(input_cim_w, tags=node_tags)
    builder.call(input_cim_f, [state, params, comp_in, data, data])

    # Call parameter CIM
    param_cim_w = ctx.get_node_wrapper(composition, composition.parameter_CIM)
    param_cim_f = ctx.import_llvm_function(param_cim_w, tags=node_tags)
    builder.call(param_cim_f, [state, params, comp_in, data, data])

    yield builder, data, params, cond_gen

    if "alloca_data" in debug_env:
        data_vals = builder.load(data)
        builder.store(data_vals, data_arg)

    # Bump run counter
    cond_gen.bump_ts(builder, cond, (1, 0, 0))

    builder.ret_void()


def gen_composition_exec(ctx, composition, *, tags:frozenset):
    simulation = "simulation" in tags
    node_tags = tags.union({"node_wrapper"})

    with _gen_composition_exec_context(ctx, composition, tags=tags) as (builder, data, params, cond_gen):
        state, _, comp_in, _, cond = builder.function.args

        nodes_states = helpers.get_param_ptr(builder, composition, state, "nodes")

        num_exec_locs = {}
        for idx, node in enumerate(composition._all_nodes):
            #FIXME: This skips nested compositions
            from psyneulink import Composition
            if isinstance(node, Composition):
                continue
            node_state = builder.gep(nodes_states, [ctx.int32_ty(0),
                                                    ctx.int32_ty(idx)])
            num_exec_locs[node] = helpers.get_state_ptr(builder, node,
                                                        node_state,
                                                        "num_executions")

        # Reset internal TRIAL clock for each node
        for time_loc in num_exec_locs.values():
            num_exec_time_ptr = builder.gep(time_loc, [ctx.int32_ty(0),
                                                       ctx.int32_ty(TimeScale.TRIAL.value)])
            builder.store(num_exec_time_ptr.type.pointee(0), num_exec_time_ptr)

        # Check if there's anything to reset
        for node in composition._all_nodes:
            when = getattr(node, "reset_stateful_function_when", Never())
            # FIXME: This should not be necessary. The code gets DCE'd,
            # but there are still some problems with generation
            # 'reset' function
            if node is composition.controller:
                continue

            reinit_cond = cond_gen.generate_sched_condition(
                builder, when, cond, node, None)
            with builder.if_then(reinit_cond):
                node_w = ctx.get_node_wrapper(composition, node)
                node_reinit_f = ctx.import_llvm_function(node_w, tags=node_tags.union({"reset"}))
                builder.call(node_reinit_f, [state, params, comp_in, data, data])

        if simulation is False and composition.enable_controller and \
           composition.controller_mode == BEFORE:
            assert composition.controller is not None
            controller_w = ctx.get_node_wrapper(composition, composition.controller)
            controller_f = ctx.import_llvm_function(controller_w,
                                                     tags=node_tags)
            builder.call(controller_f, [state, params, comp_in, data, data])


        # Allocate run set structure
        run_set_type = ir.ArrayType(ir.IntType(1), len(composition.nodes))
        run_set_ptr = builder.alloca(run_set_type, name="run_set")
        builder.store(run_set_type(None), run_set_ptr)

        # Allocate temporary output storage
        output_storage = builder.alloca(data.type.pointee, name="output_storage")

        iter_ptr = builder.alloca(ctx.int32_ty, name="iter_counter")
        builder.store(ctx.int32_ty(0), iter_ptr)

        # Generate pointers to 'is_finished' callbacks
        is_finished_callbacks = {}
        for node in composition.nodes:
            args = [state, params, comp_in, data, output_storage]
            wrapper = ctx.get_node_wrapper(composition, node)
            is_finished_callbacks[node] = (wrapper, args)

        loop_condition = builder.append_basic_block(name="scheduling_loop_condition")
        builder.branch(loop_condition)

        # Generate a while not 'end condition' loop
        builder.position_at_end(loop_condition)

        run_cond = cond_gen.generate_sched_condition(
            builder, composition.termination_processing[TimeScale.TRIAL],
            cond, None, is_finished_callbacks)
        run_cond = builder.not_(run_cond, name="not_run_cond")

        loop_body = builder.append_basic_block(name="scheduling_loop_body")
        exit_block = builder.append_basic_block(name="exit")
        builder.cbranch(run_cond, loop_body, exit_block)

        # Generate loop body
        builder.position_at_end(loop_body)

        zero = ctx.int32_ty(0)
        any_cond = ir.IntType(1)(0)
        # Calculate execution set before running the mechanisms
        for idx, node in enumerate(composition.nodes):
            run_set_node_ptr = builder.gep(run_set_ptr,
                                           [zero, ctx.int32_ty(idx)],
                                           name="run_cond_ptr_" + node.name)
            node_cond = cond_gen.generate_sched_condition(
                builder, composition._get_processing_condition_set(node),
                cond, node, is_finished_callbacks)
            ran = cond_gen.generate_ran_this_pass(builder, cond, node)
            node_cond = builder.and_(node_cond, builder.not_(ran),
                                     name="run_cond_" + node.name)
            any_cond = builder.or_(any_cond, node_cond, name="any_ran_cond")
            builder.store(node_cond, run_set_node_ptr)

        # Reset internal TIME_STEP and PASS clock for each node
        for time_loc in num_exec_locs.values():
            num_exec_time_ptr = builder.gep(time_loc, [ctx.int32_ty(0),
                                                       ctx.int32_ty(TimeScale.TIME_STEP.value)])
            builder.store(num_exec_time_ptr.type.pointee(0), num_exec_time_ptr)
            # FIXME: Move pass reset to actual pass count
            num_exec_time_ptr = builder.gep(time_loc, [ctx.int32_ty(0),
                                                       ctx.int32_ty(TimeScale.PASS.value)])
            builder.store(num_exec_time_ptr.type.pointee(0), num_exec_time_ptr)

        for idx, node in enumerate(composition.nodes):

            run_set_node_ptr = builder.gep(run_set_ptr, [zero, ctx.int32_ty(idx)])
            node_cond = builder.load(run_set_node_ptr,
                                     name="node_" + node.name + "_should_run")
            with builder.if_then(node_cond):
                node_w = ctx.get_node_wrapper(composition, node)
                node_f = ctx.import_llvm_function(node_w, tags=node_tags)
                builder.block.name = "invoke_" + node_f.name
                # Wrappers do proper indexing of all structures
                # Mechanisms have only 5 args
                args = [state, params, comp_in, data, output_storage]
                if len(node_f.args) >= 6:  # Composition wrappers have 6 args
                    args.append(cond)
                builder.call(node_f, args)

                cond_gen.generate_update_after_run(builder, cond, node)
            builder.block.name = "post_invoke_" + node_f.name

        # Writeback results
        for idx, node in enumerate(composition.nodes):
            run_set_node_ptr = builder.gep(run_set_ptr, [zero, ctx.int32_ty(idx)])
            node_cond = builder.load(run_set_node_ptr, name="node_" + node.name + "_ran")
            with builder.if_then(node_cond):
                out_ptr = builder.gep(output_storage, [zero, zero,
                                                       ctx.int32_ty(idx)],
                                      name="result_ptr_" + node.name)
                data_ptr = builder.gep(data, [zero, zero, ctx.int32_ty(idx)],
                                       name="data_result_" + node.name)
                builder.store(builder.load(out_ptr), data_ptr)

        # Update step counter
        with builder.if_then(any_cond):
            builder.block.name = "inc_step"
            cond_gen.bump_ts(builder, cond)

        builder.block.name = "update_iter_count"
        # Increment number of iterations
        iters = builder.load(iter_ptr, name="iterw")
        iters = builder.add(iters, ctx.int32_ty(1), name="iterw_inc")
        builder.store(iters, iter_ptr)

        max_iters = len(composition.scheduler.consideration_queue)
        completed_pass = builder.icmp_unsigned("==", iters,
                                               ctx.int32_ty(max_iters),
                                               name="completed_pass")
        # Increment pass and reset time step
        with builder.if_then(completed_pass):
            builder.block.name = "inc_pass"
            builder.store(zero, iter_ptr)
            # Bumping automatically zeros lower elements
            cond_gen.bump_ts(builder, cond, (0, 1, 0))

        builder.branch(loop_condition)

        builder.position_at_end(exit_block)

        if simulation is False and composition.enable_controller and \
           composition.controller_mode == AFTER:
            assert composition.controller is not None
            controller_w = ctx.get_node_wrapper(composition, composition.controller)
            controller_f = ctx.import_llvm_function(controller_w, tags=node_tags)
            builder.call(controller_f, [state, params, comp_in, data, data])

        # Call output CIM
        output_cim_w = ctx.get_node_wrapper(composition, composition.output_CIM)
        output_cim_f = ctx.import_llvm_function(output_cim_w, tags=node_tags)
        builder.block.name = "invoke_" + output_cim_f.name
        builder.call(output_cim_f, [state, params, comp_in, data, data])

    return builder.function


def gen_composition_run(ctx, composition, *, tags:frozenset):
    assert "run" in tags
    simulation = "simulation" in tags
    name = "_".join(("wrap",  *tags, composition.name))
    args = [ctx.get_state_struct_type(composition).as_pointer(),
            ctx.get_param_struct_type(composition).as_pointer(),
            ctx.get_data_struct_type(composition).as_pointer(),
            ctx.get_input_struct_type(composition).as_pointer(),
            ctx.get_output_struct_type(composition).as_pointer(),
            ctx.int32_ty.as_pointer(),
            ctx.int32_ty.as_pointer()]
    builder = ctx.create_llvm_function(args, composition, name)
    llvm_func = builder.function
    for a in llvm_func.args:
        a.attributes.add('noalias')

    state, params, data, data_in, data_out, runs_ptr, inputs_ptr = llvm_func.args
    # simulation does not care about the output
    # it extracts results of the controller objective mechanism
    if simulation:
        data_out.attributes.remove('nonnull')

    if not simulation and "const_data" in debug_env:
        const_data = data.type.pointee(composition._get_data_initializer(None))
        data = builder.alloca(data.type.pointee)
        builder.store(const_data, data)

    # Hardcode stateful parameters if set in the environment
    if not simulation and "const_state" in debug_env:
        const_state = state.type.pointee(composition._get_state_initializer(None))
        state = builder.alloca(const_state.type, name="const_state_loc")
        builder.store(const_state, state)

    if not simulation and "const_input" in debug_env:
        if not debug_env["const_input"]:
            input_init = [[os.defaults.variable.tolist()] for os in composition.input_CIM.input_ports]
            print("Setting default input: ", input_init)
        else:
            input_init = ast.literal_eval(debug_env["const_input"])
            print("Setting user input: ", input_init)

        builder.store(data_in.type.pointee(input_init), data_in)
        builder.store(inputs_ptr.type.pointee(1), inputs_ptr)

    # Allocate and initialize condition structure
    cond_gen = helpers.ConditionGenerator(ctx, composition)
    cond_type = cond_gen.get_condition_struct_type()
    cond = builder.alloca(cond_type)
    cond_init = cond_type(cond_gen.get_condition_initializer())
    builder.store(cond_init, cond)

    runs = builder.load(runs_ptr, "runs")
    with helpers.for_loop_zero_inc(builder, runs, "run_loop") as (b, iters):
        # Get the right input stimulus
        input_idx = b.urem(iters, b.load(inputs_ptr))
        data_in_ptr = b.gep(data_in, [input_idx])

        # Reset internal clocks of each node
        for idx, node in enumerate(composition._all_nodes):
            #FIXME: This skips nested nodes
            from psyneulink import Composition
            if isinstance(node, Composition):
                continue
            node_state = builder.gep(state, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(idx)])
            num_executions_ptr = helpers.get_state_ptr(builder, node, node_state, "num_executions")
            num_exec_time_ptr = builder.gep(num_executions_ptr, [ctx.int32_ty(0), ctx.int32_ty(TimeScale.RUN.value)])
            builder.store(ctx.int32_ty(0), num_exec_time_ptr)

        # Call execution
        exec_tags = tags.difference({"run"})
        exec_f = ctx.import_llvm_function(composition, tags=exec_tags)
        b.call(exec_f, [state, params, data_in_ptr, data, cond])

        if not simulation:
            # Extract output_CIM result
            idx = composition._get_node_index(composition.output_CIM)
            result_ptr = b.gep(data, [ctx.int32_ty(0), ctx.int32_ty(0),
                                      ctx.int32_ty(idx)])
            output_ptr = b.gep(data_out, [iters])
            result = b.load(result_ptr)
            b.store(result, output_ptr)

    builder.ret_void()
    return llvm_func


def gen_multirun_wrapper(ctx, function: ir.Function) -> ir.Function:
    if function.module is not ctx.module:
        function = ir.Function(ctx.module, function.type.pointee, function.name)
        assert function.is_declaration

    args = [a.type for a in function.args]
    args.append(ctx.int32_ty.as_pointer())
    multirun_ty = ir.FunctionType(function.type.pointee.return_type, args)
    multirun_f = ir.Function(ctx.module, multirun_ty, function.name + "_multirun")
    block = multirun_f.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    multi_runs = builder.load(multirun_f.args[-1])
    # Runs need special handling. data_in and data_out are one dimensional,
    # but hold entries for all parallel invocations.
    is_comp_run = len(function.args) == 7
    if is_comp_run:
        runs_count = builder.load(multirun_f.args[5])
        input_count = builder.load(multirun_f.args[6])

    with helpers.for_loop_zero_inc(builder, multi_runs, "multi_run_loop") as (b, index):
        # Index all pointer arguments
        indexed_args = []
        for i, arg in enumerate(multirun_f.args[:-1]):
            # Don't adjust #inputs and #trials
            if isinstance(arg.type, ir.PointerType):
                offset = index
                # #runs and #trials needs to be the same for every invocation
                if is_comp_run and i >= 5:
                    offset = ctx.int32_ty(0)
                # data arrays need special handling
                elif is_comp_run and i == 4:  # data_out
                    offset = b.mul(index, runs_count)
                elif is_comp_run and i == 3:  # data_in
                    offset = b.mul(index, input_count)

                arg = b.gep(arg, [offset])

            indexed_args.append(arg)

        b.call(function, indexed_args)

    builder.ret_void()
    return multirun_f


def gen_autodiffcomp_exec(ctx, composition, *, tags:frozenset):
    """Creates llvm bin execute for autodiffcomp"""
    assert composition.controller is None
    composition._build_pytorch_representation(composition.default_execution_id)
    pytorch_model = composition.parameters.pytorch_representation.get(composition.default_execution_id)
    with _gen_composition_exec_context(ctx, composition, tags=tags) as (builder, data, params, cond_gen):
        state, _, comp_in, _, cond = builder.function.args

        pytorch_func = ctx.import_llvm_function(pytorch_model, tags=tags)
        builder.call(pytorch_func, [state, params, data])

        node_tags = tags.union({"node_wrapper"})
        # Call output CIM
        output_cim_w = ctx.get_node_wrapper(composition, composition.output_CIM)
        output_cim_f = ctx.import_llvm_function(output_cim_w, tags=node_tags)
        builder.call(output_cim_f, [state, params, comp_in, data, data])

        return builder.function
