# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import ast
import atexit
import ctypes
from contextlib import contextmanager
import functools
import inspect
from llvmlite import ir
import numpy as np
import os
import re
from typing import Set
import weakref
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

from psyneulink.core.scheduling.condition import Never
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.globals.keywords import AFTER, BEFORE

from psyneulink.core import llvm as pnlvm
from .debug import debug_env
from .helpers import ConditionGenerator

__all__ = ['LLVMBuilderContext', '_modules', '_find_llvm_function']


_modules: Set[ir.Module] = set()
_all_modules: Set[ir.Module] = set()
_struct_count = 0


@atexit.register
def module_count():
    if "stat" in debug_env:
        print("Total LLVM modules: ", len(_all_modules))
        print("Total structures generated: ", _struct_count)


_BUILTIN_PREFIX = "__pnl_builtin_"
_builtin_intrinsics = frozenset(('pow', 'log', 'exp'))


class LLVMBuilderContext:
    __global_context = None
    __uniq_counter = 0
    _llvm_generation = 0
    int32_ty = ir.IntType(32)
    float_ty = ir.DoubleType()

    def __init__(self):
        self._modules = []
        self._cache = weakref.WeakKeyDictionary()

    def __enter__(self):
        module = ir.Module(name="PsyNeuLinkModule-" + str(LLVMBuilderContext._llvm_generation))
        self._modules.append(module)
        LLVMBuilderContext._llvm_generation += 1
        return self

    def __exit__(self, e_type, e_value, e_traceback):
        assert len(self._modules) > 0
        module = self._modules.pop()
        _modules.add(module)
        _all_modules.add(module)

    @property
    def module(self):
        assert len(self._modules) > 0
        return self._modules[-1]

    @classmethod
    def get_global(cls):
        if cls.__global_context is None:
            cls.__global_context = LLVMBuilderContext()
        return cls.__global_context

    @classmethod
    def get_unique_name(cls, name: str):
        cls.__uniq_counter += 1
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return name + '_' + str(cls.__uniq_counter)

    def get_builtin(self, name: str, args=[], function_type=None):
        if name in _builtin_intrinsics:
            return self.import_llvm_function(_BUILTIN_PREFIX + name)
        if name in ('maxnum'):
            function_type = pnlvm.ir.FunctionType(args[0], [args[0], args[0]])
        return self.module.declare_intrinsic("llvm." + name, args, function_type)

    def create_llvm_function(self, args, component, name=None, *, return_type=ir.VoidType(), tags:frozenset=frozenset()):
        name = "_".join((str(component), *tags)) if name is None else name

        # Builtins are already unique and need to keep their special name
        func_name = name if name.startswith(_BUILTIN_PREFIX) else self.get_unique_name(name)
        func_ty = pnlvm.ir.FunctionType(return_type, args)
        llvm_func = pnlvm.ir.Function(self.module, func_ty, name=func_name)
        llvm_func.attributes.add('argmemonly')
        for a in llvm_func.args:
            if isinstance(a.type, ir.PointerType):
                a.attributes.add('nonnull')

        metadata = self.get_debug_location(llvm_func, component)
        if metadata is not None:
            scope = dict(metadata.operands)["scope"]
            llvm_func.set_metadata("dbg", scope)

        # Create entry block
        block = llvm_func.append_basic_block(name="entry")
        builder = pnlvm.ir.IRBuilder(block)
        builder.debug_metadata = metadata

        return builder

    def gen_llvm_function(self, obj, *, tags:frozenset) -> ir.Function:
        cache = self._cache
        if obj not in cache:
            cache[obj] = dict()
        cache_variants = cache[obj]

        if tags not in cache_variants:
            cache_variants[tags] = obj._gen_llvm_function(tags=tags)
        return cache_variants[tags]

    def import_llvm_function(self, fun, *, tags:frozenset=frozenset()) -> ir.Function:
        """
        Get function handle if function exists in current modele.
        Create function declaration if it exists in a older module.
        """
        if isinstance(fun, str):
            f = _find_llvm_function(fun, _all_modules | {self.module})
        else:
            f = self.gen_llvm_function(fun, tags=tags)

        # Add declaration to the current module
        if f.name not in self.module.globals:
            decl_f = ir.Function(self.module, f.type.pointee, f.name)
            assert decl_f.is_declaration
            return decl_f
        return f

    @staticmethod
    def get_debug_location(func: ir.Function, component):
        if "debug_info" not in debug_env:
            return

        mod = func.module
        path = inspect.getfile(component.__class__) if component is not None else "<pnl_builtin>"
        d_version = mod.add_metadata([ir.IntType(32)(2), "Dwarf Version", ir.IntType(32)(4)])
        di_version = mod.add_metadata([ir.IntType(32)(2), "Debug Info Version", ir.IntType(32)(3)])
        flags = mod.add_named_metadata("llvm.module.flags")
        if len(flags.operands) == 0:
            flags.add(d_version)
            flags.add(di_version)
        cu = mod.add_named_metadata("llvm.dbg.cu")
        di_file = mod.add_debug_info("DIFile", {
            "filename": os.path.basename(path),
            "directory": os.path.dirname(path),
        })
        di_func_type = mod.add_debug_info("DISubroutineType", {
            # None as `null`
            "types": mod.add_metadata([None]),
        })
        di_compileunit = mod.add_debug_info("DICompileUnit", {
            "language": ir.DIToken("DW_LANG_Python"),
            "file": di_file,
            "producer": "PsyNeuLink",
            "runtimeVersion": 0,
            "isOptimized": False,
        }, is_distinct=True)
        cu.add(di_compileunit)
        di_func = mod.add_debug_info("DISubprogram", {
            "name": func.name,
            "file": di_file,
            "line": 0,
            "type": di_func_type,
            "isLocal": False,
            "unit": di_compileunit,
        }, is_distinct=True)
        di_loc = mod.add_debug_info("DILocation", {
            "line": 0, "column": 0, "scope": di_func,
        })
        return di_loc

    def get_input_struct_type(self, component):
        if hasattr(component, '_get_input_struct_type'):
            return component._get_input_struct_type(self)

        default_var = component.defaults.variable
        return self.convert_python_struct_to_llvm_ir(default_var)

    def get_output_struct_type(self, component):
        if hasattr(component, '_get_output_struct_type'):
            return component._get_output_struct_type(self)

        default_val = component.defaults.value
        return self.convert_python_struct_to_llvm_ir(default_val)

    def get_param_struct_type(self, component):
        if hasattr(component, '_get_param_struct_type'):
            return component._get_param_struct_type(self)

        params = component._get_param_values()
        return self.convert_python_struct_to_llvm_ir(params)

    def get_state_struct_type(self, component):
        if hasattr(component, '_get_state_struct_type'):
            return component._get_state_struct_type(self)

        stateful = component._get_state_values()
        return self.convert_python_struct_to_llvm_ir(stateful)

    def get_data_struct_type(self, component):
        if hasattr(component, '_get_data_struct_type'):
            return component._get_data_struct_type(self)

        return ir.LiteralStructType([])

    def get_param_ptr(self, component, builder, params_ptr, param_name):
        idx = self.int32_ty(component._get_param_ids().index(param_name))
        return builder.gep(params_ptr, [self.int32_ty(0), idx],
                           name="ptr_param_{}_{}".format(param_name, component.name))

    def get_state_ptr(self, component, builder, state_ptr, stateful_name):
        idx = self.int32_ty(component._get_state_ids().index(stateful_name))
        return builder.gep(state_ptr, [self.int32_ty(0), idx],
                           name="ptr_state_{}_{}".format(stateful_name,
                                                         component.name))

    def unwrap_2d_array(self, builder, element):
        if isinstance(element.type.pointee, ir.ArrayType) and isinstance(element.type.pointee.element, ir.ArrayType):
            assert element.type.pointee.count == 1
            return builder.gep(element, [self.int32_ty(0), self.int32_ty(0)])
        return element

    @contextmanager
    def _gen_composition_exec_context(self, composition, *, tags:frozenset, suffix="", extra_args=[]):
        cond_gen = ConditionGenerator(self, composition)

        name = "_".join(("wrap_exec", *tags ,composition.name + suffix))
        args = [self.get_state_struct_type(composition).as_pointer(),
                self.get_param_struct_type(composition).as_pointer(),
                self.get_input_struct_type(composition).as_pointer(),
                self.get_data_struct_type(composition).as_pointer(),
                cond_gen.get_condition_struct_type().as_pointer()]
        builder = self.create_llvm_function(args + extra_args, composition, name)
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
        input_cim_f = self.import_llvm_function(composition.input_CIM, tags=node_tags)
        builder.call(input_cim_f, [state, params, comp_in, data, data])

        # Call parameter CIM
        param_cim_f = self.import_llvm_function(composition.parameter_CIM,
                                                tags=node_tags)
        builder.call(param_cim_f, [state, params, comp_in, data, data])

        yield builder, data, params, cond_gen

        if "alloca_data" in debug_env:
            data_vals = builder.load(data)
            builder.store(data_vals, data_arg)

        # Bump run counter
        cond_gen.bump_ts(builder, cond, (1, 0, 0))

        builder.ret_void()

    def gen_autodiffcomp_learning_exec(self, composition, *, tags:frozenset):
        composition._build_pytorch_representation(composition.default_execution_id)
        pytorch_model = composition.parameters.pytorch_representation.get(composition.default_execution_id)
        with self._gen_composition_exec_context(composition, tags=tags) as (builder, data, params, cond_gen):
            state, _, comp_in, data, cond, = builder.function.args
            pytorch_model._gen_llvm_training_function_body(self, builder, state,
                                                           params, data)
            node_tags = tags.union({"node_wrapper"})
            # # Call output CIM
            output_cim_f = self.import_llvm_function(composition.output_CIM,
                                                     tags=node_tags)
            builder.block.name = "invoke_" + output_cim_f.name
            builder.call(output_cim_f, [state, params, comp_in, data, data])

            return builder.function

    def gen_autodiffcomp_exec(self, composition, *, tags:frozenset):
        """Creates llvm bin execute for autodiffcomp"""
        assert composition.controller is None
        composition._build_pytorch_representation(composition.default_execution_id)
        pytorch_model = composition.parameters.pytorch_representation.get(composition.default_execution_id)
        with self._gen_composition_exec_context(composition, tags=tags) as (builder, data, params, cond_gen):
            state, _, comp_in, _, cond = builder.function.args

            pytorch_forward_func = self.import_llvm_function(pytorch_model, tags=tags)
            builder.call(pytorch_forward_func, [state, params, data])

            node_tags = tags.union({"node_wrapper"})
            # Call output CIM
            output_cim_f = self.import_llvm_function(composition.output_CIM,
                                                     tags=node_tags)
            builder.call(output_cim_f, [state, params, comp_in, data, data])

            return builder.function

    def gen_composition_exec(self, composition, *, tags:frozenset):
        simulation = "simulation" in tags
        node_tags = tags.union({"node_wrapper"})
        extra_args = []
        # If there is a node that needs learning input we need to export it
        for node in filter(lambda n: hasattr(n, 'learning_enabled') and "learning" in tags, composition.nodes):
            node_wrap = composition._get_node_wrapper(node)
            node_f = self.import_llvm_function(node_wrap, tags=node_tags)
            extra_args = [node_f.args[-1].type]


        with self._gen_composition_exec_context(composition, tags=tags, extra_args=extra_args) as (builder, data, params, cond_gen):
            state, _, comp_in, _, cond, *learning = builder.function.args

            # Check if there's anything to reinitialize
            for node in composition._all_nodes:
                when = getattr(node, "reinitialize_when", Never())
                # FIXME: This should not be necessary. The code gets DCE'd,
                # but there are still some problems with generation
                # 'reinitialize' function
                if node is composition.controller:
                    continue

                reinit_cond = cond_gen.generate_sched_condition(
                    builder, when, cond, node)
                with builder.if_then(reinit_cond):
                    node_w = composition._get_node_wrapper(node)
                    node_reinit_f = self.import_llvm_function(node_w, tags=node_tags.union({"reinitialize"}))
                    builder.call(node_reinit_f, [state, params, comp_in, data, data])

            if simulation is False and composition.enable_controller and \
               composition.controller_mode == BEFORE:
                assert composition.controller is not None
                controller_f = self.import_llvm_function(composition.controller,
                                                         tags=node_tags)
                builder.call(controller_f, [state, params, comp_in, data, data])


            # Allocate run set structure
            run_set_type = ir.ArrayType(ir.IntType(1), len(composition.nodes))
            run_set_ptr = builder.alloca(run_set_type, name="run_set")
            builder.store(run_set_type(None), run_set_ptr)

            # Allocate temporary output storage
            output_storage = builder.alloca(data.type.pointee, name="output_storage")

            iter_ptr = builder.alloca(self.int32_ty, name="iter_counter")
            builder.store(self.int32_ty(0), iter_ptr)

            loop_condition = builder.append_basic_block(name="scheduling_loop_condition")
            builder.branch(loop_condition)

            # Generate a while not 'end condition' loop
            builder.position_at_end(loop_condition)
            run_cond = cond_gen.generate_sched_condition(
                builder, composition.termination_processing[TimeScale.TRIAL],
                cond, None)
            run_cond = builder.not_(run_cond, name="not_run_cond")

            loop_body = builder.append_basic_block(name="scheduling_loop_body")
            exit_block = builder.append_basic_block(name="exit")
            builder.cbranch(run_cond, loop_body, exit_block)

            # Generate loop body
            builder.position_at_end(loop_body)

            zero = self.int32_ty(0)
            any_cond = ir.IntType(1)(0)

            # Calculate execution set before running the mechanisms
            for idx, node in enumerate(composition.nodes):
                run_set_node_ptr = builder.gep(run_set_ptr,
                                               [zero, self.int32_ty(idx)],
                                               name="run_cond_ptr_" + node.name)
                node_cond = cond_gen.generate_sched_condition(
                    builder, composition._get_processing_condition_set(node),
                    cond, node)
                ran = cond_gen.generate_ran_this_pass(builder, cond, node)
                node_cond = builder.and_(node_cond, builder.not_(ran),
                                         name="run_cond_" + node.name)
                any_cond = builder.or_(any_cond, node_cond, name="any_ran_cond")
                builder.store(node_cond, run_set_node_ptr)

            for idx, node in enumerate(composition.nodes):
                run_set_node_ptr = builder.gep(run_set_ptr, [zero, self.int32_ty(idx)])
                node_cond = builder.load(run_set_node_ptr, name="node_" + node.name + "_should_run")
                with builder.if_then(node_cond):
                    node_w = composition._get_node_wrapper(node)
                    node_f = self.import_llvm_function(node_w, tags=node_tags)
                    builder.block.name = "invoke_" + node_f.name
                    # Wrappers do proper indexing of all structures
                    # Mechanisms have only 5 args
                    args = [state, params, comp_in, data, output_storage]
                    if len(node_f.args) >= 6:  # Composition wrappers have 6 args
                        args.append(cond)
                    if len(node_f.args) == 7:  # Learning wrappers have 7 args
                        args.append(*learning)
                    builder.call(node_f, args)

                    cond_gen.generate_update_after_run(builder, cond, node)
                builder.block.name = "post_invoke_" + node_f.name

            # Writeback results
            for idx, node in enumerate(composition.nodes):
                run_set_node_ptr = builder.gep(run_set_ptr, [zero, self.int32_ty(idx)])
                node_cond = builder.load(run_set_node_ptr, name="node_" + node.name + "_ran")
                with builder.if_then(node_cond):
                    out_ptr = builder.gep(output_storage, [zero, zero, self.int32_ty(idx)], name="result_ptr_" + node.name)
                    data_ptr = builder.gep(data, [zero, zero, self.int32_ty(idx)],
                                           name="data_result_" + node.name)
                    builder.store(builder.load(out_ptr), data_ptr)

            # Update step counter
            with builder.if_then(any_cond):
                builder.block.name = "inc_step"
                cond_gen.bump_ts(builder, cond)

            builder.block.name = "update_iter_count"
            # Increment number of iterations
            iters = builder.load(iter_ptr, name="iterw")
            iters = builder.add(iters, self.int32_ty(1), name="iterw_inc")
            builder.store(iters, iter_ptr)

            max_iters = len(composition.scheduler.consideration_queue)
            completed_pass = builder.icmp_unsigned("==", iters,
                                                   self.int32_ty(max_iters),
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
                controller_f = self.import_llvm_function(composition.controller,
                                                         tags=node_tags)
                builder.call(controller_f, [state, params, comp_in, data, data])

            # Call output CIM
            output_cim_f = self.import_llvm_function(composition.output_CIM,
                                                     tags=node_tags)
            builder.block.name = "invoke_" + output_cim_f.name
            builder.call(output_cim_f, [state, params, comp_in, data, data])

        return builder.function

    def gen_composition_run(self, composition, *, tags:frozenset):
        assert "run" in tags
        simulation = "simulation" in tags
        name = "_".join(("wrap",  *tags, composition.name))
        args = [self.get_state_struct_type(composition).as_pointer(),
                self.get_param_struct_type(composition).as_pointer(),
                self.get_data_struct_type(composition).as_pointer(),
                self.get_input_struct_type(composition).as_pointer(),
                self.get_output_struct_type(composition).as_pointer(),
                self.int32_ty.as_pointer(),
                self.int32_ty.as_pointer()]
        builder = self.create_llvm_function(args, composition, name)
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
                input_init = pnlvm._tupleize([[os.defaults.variable] for os in composition.input_CIM.input_ports])
                print("Setting default input: ", input_init)
            else:
                input_init = ast.literal_eval(debug_env["const_input"])
                print("Setting user input: ", input_init)

            builder.store(data_in.type.pointee(input_init), data_in)
            builder.store(inputs_ptr.type.pointee(1), inputs_ptr)

        # Allocate and initialize condition structure
        cond_gen = ConditionGenerator(self, composition)
        cond_type = cond_gen.get_condition_struct_type()
        cond = builder.alloca(cond_type)
        cond_init = cond_type(cond_gen.get_condition_initializer())
        builder.store(cond_init, cond)

        runs = builder.load(runs_ptr, "runs")
        with pnlvm.helpers.for_loop_zero_inc(builder, runs, "run_loop") as (b, iters):
            # Get the right input stimulus
            input_idx = b.urem(iters, b.load(inputs_ptr))
            data_in_ptr = b.gep(data_in, [input_idx])

            # Call execution
            exec_tags = tags.difference({"run"})
            exec_f = self.import_llvm_function(composition, tags=exec_tags)
            b.call(exec_f, [state, params, data_in_ptr, data, cond])

            if not simulation:
                # Extract output_CIM result
                idx = composition._get_node_index(composition.output_CIM)
                result_ptr = b.gep(data, [self.int32_ty(0), self.int32_ty(0),
                                          self.int32_ty(idx)])
                output_ptr = b.gep(data_out, [iters])
                result = b.load(result_ptr)
                b.store(result, output_ptr)

        builder.ret_void()
        return llvm_func

    def gen_multirun_wrapper(self, function: ir.Function) -> ir.Function:
        if function.module is not self.module:
            function = ir.Function(self.module, function.type.pointee, function.name)
            assert function.is_declaration

        args = [a.type for a in function.args]
        args.append(self.int32_ty.as_pointer())
        multirun_ty = ir.FunctionType(function.type.pointee.return_type, args)
        multirun_f = ir.Function(self.module, multirun_ty, function.name + "_multirun")
        block = multirun_f.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)

        multi_runs = builder.load(multirun_f.args[-1])
        # Runs need special handling. data_in and data_out are one dimensional,
        # but hold entries for all parallel invocations.
        is_comp_run = len(function.args) == 7
        if is_comp_run:
            runs_count = builder.load(multirun_f.args[5])
            input_count = builder.load(multirun_f.args[6])

        with pnlvm.helpers.for_loop_zero_inc(builder, multi_runs, "multi_run_loop") as (b, index):
            # Index all pointer arguments
            indexed_args = []
            for i, arg in enumerate(multirun_f.args[:-1]):
                # Don't adjust #inputs and #trials
                if isinstance(arg.type, ir.PointerType):
                    offset = index
                    # #runs and #trials needs to be the same for every invocation
                    if is_comp_run and i >= 5:
                        offset = self.int32_ty(0)
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

    def convert_python_struct_to_llvm_ir(self, t):
        if type(t) is list:
            if len(t) == 0:
                return ir.LiteralStructType([])
            assert all(type(x) is type(t[0]) for x in t)
            elem_t = self.convert_python_struct_to_llvm_ir(t[0])
            return ir.ArrayType(elem_t, len(t))
        elif type(t) is tuple:
            elems_t = (self.convert_python_struct_to_llvm_ir(x) for x in t)
            return ir.LiteralStructType(elems_t)
        elif isinstance(t, (int, float)):
            return self.float_ty
        elif isinstance(t, np.ndarray):
            return self.convert_python_struct_to_llvm_ir(t.tolist())
        elif t is None:
            return ir.LiteralStructType([])
        elif isinstance(t, np.random.RandomState):
            return pnlvm.builtins.get_mersenne_twister_state_struct(self)
        elif torch_available and isinstance(t, torch.Tensor):
            return self.convert_python_struct_to_llvm_ir(t.numpy())
        assert False, "Don't know how to convert {}".format(type(t))


def _find_llvm_function(name: str, mods=_all_modules) -> ir.Function:
    f = None
    for m in mods:
        if name in m.globals:
            f = m.get_global(name)

    if not isinstance(f, ir.Function):
        raise ValueError("No such function: {}".format(name))
    return f


def _gen_cuda_kernel_wrapper_module(function):
    module = ir.Module(name="wrapper_" + function.name)

    decl_f = ir.Function(module, function.type.pointee, function.name)
    assert decl_f.is_declaration
    kernel_func = ir.Function(module, function.type.pointee, function.name + "_cuda_kernel")
    block = kernel_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Calculate global id of a thread in x dimension
    intrin_ty = ir.FunctionType(ir.IntType(32), [])
    tid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.tid.x")
    ntid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.ntid.x")
    ctaid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.ctaid.x")
    global_id = builder.mul(builder.call(ctaid_x_f, []), builder.call(ntid_x_f, []))
    global_id = builder.add(global_id, builder.call(tid_x_f, []))

    # Runs need special handling. data_in and data_out are one dimensional,
    # but hold entries for all parallel invocations.
    is_comp_run = len(kernel_func.args) == 7
    if is_comp_run:
        runs_count = kernel_func.args[5]
        input_count = kernel_func.args[6]

    # Index all pointer arguments
    indexed_args = []
    for i, arg in enumerate(kernel_func.args):
        # Don't adjust #inputs and #trials
        if isinstance(arg.type, ir.PointerType):
            offset = global_id
            # #runs and #trials needs to be the same
            if is_comp_run and i >= 5:
                offset = ir.IntType(32)(0)
            # data arrays need special handling
            elif is_comp_run and i == 4:  # data_out
                offset = builder.mul(global_id, builder.load(runs_count))
            elif is_comp_run and i == 3:  # data_in
                offset = builder.mul(global_id, builder.load(input_count))

            arg = builder.gep(arg, [offset])

        indexed_args.append(arg)
    builder.call(decl_f, indexed_args)
    builder.ret_void()

    # Add kernel mark metadata
    module.add_named_metadata("nvvm.annotations", [kernel_func, "kernel", ir.IntType(32)(1)])

    return module


@functools.lru_cache(maxsize=128)
def _convert_llvm_ir_to_ctype(t: ir.Type):
    type_t = type(t)

    if type_t is ir.VoidType:
        return None
    elif type_t is ir.IntType:
        if t.width == 32:
            return ctypes.c_int
        elif t.width == 64:
            return ctypes.c_longlong
        else:
            assert False, "Integer type too big!"
    elif type_t is ir.DoubleType:
        return ctypes.c_double
    elif type_t is ir.FloatType:
        return ctypes.c_float
    elif type_t is ir.PointerType:
        pointee = _convert_llvm_ir_to_ctype(t.pointee)
        ret_t = ctypes.POINTER(pointee)
    elif type_t is ir.ArrayType:
        element_type = _convert_llvm_ir_to_ctype(t.element)
        ret_t = element_type * len(t)
    elif type_t is ir.LiteralStructType:
        global _struct_count
        uniq_name = "struct_" + str(_struct_count)
        _struct_count += 1

        field_list = []
        for i, e in enumerate(t.elements):
            # llvmlite modules get _unique string only works for symbol names
            field_uniq_name = uniq_name + "field_" + str(i)
            field_list.append((field_uniq_name, _convert_llvm_ir_to_ctype(e)))

        ret_t = type(uniq_name, (ctypes.Structure,), {"__init__": ctypes.Structure.__init__})
        ret_t._fields_ = field_list
        assert len(ret_t._fields_) == len(t.elements)
    else:
        assert False, "Don't know how to convert LLVM type: {}".format(t)

    return ret_t
