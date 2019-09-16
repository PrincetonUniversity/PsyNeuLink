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
import functools
import inspect
from llvmlite import ir
import numpy as np
import os
import re
import weakref
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.globals.keywords import AFTER, BEFORE

from psyneulink.core import llvm as pnlvm
from .debug import debug_env
from .helpers import ConditionGenerator

__all__ = ['LLVMBuilderContext', '_modules', '_find_llvm_function']


_modules = set()
_all_modules = set()
_struct_count = 0


@atexit.register
def module_count():
    if "stat" in debug_env:
        print("Total LLVM modules: ", len(_all_modules))
        print("Total structures generated: ", _struct_count)


# TODO: Should this be selectable?
_int32_ty = ir.IntType(32)
_float_ty = ir.DoubleType()
_global_context = None

_BUILTIN_PREFIX = "__pnl_builtin_"
_builtin_intrinsics = frozenset(('pow', 'log', 'exp', 'printf'))

class LLVMBuilderContext:
    uniq_counter = 0
    _llvm_generation = 0

    def __init__(self):
        self.int32_ty = _int32_ty
        self.float_ty = _float_ty
        self._modules = []
        self._cache = weakref.WeakKeyDictionary()
        self._learningcache = weakref.WeakKeyDictionary()
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

    @staticmethod
    def get_global():
        global _global_context
        if _global_context is None:
            _global_context = LLVMBuilderContext()
        return _global_context

    def get_unique_name(self, name: str):
        LLVMBuilderContext.uniq_counter += 1
        name = re.sub(r"[- ()\[\]]", "_", name)
        return name + '_' + str(LLVMBuilderContext.uniq_counter)

    def get_builtin(self, name: str, args=[], function_type=None):
        if name in _builtin_intrinsics:
            return self.get_llvm_function(_BUILTIN_PREFIX + name)
        if name in ('maxnum'):
            function_type = pnlvm.ir.FunctionType(args[0], [args[0], args[0]])
        return self.module.declare_intrinsic("llvm." + name, args, function_type)

    def create_llvm_function(self, args, component, name=None, return_type=ir.VoidType()):
        name = str(component) if name is None else name

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

    def gen_llvm_function(self, obj):
        cache = self._cache
        try:
            # HACK: allows for learning bin func and non-learning to differ
            if obj.learning_enabled is True:
                cache = self._learningcache
        except AttributeError as e:
            pass
            
        if obj not in cache:
            cache[obj] = obj._gen_llvm_function()
        return cache[obj]

    def get_llvm_function(self, name):
        try:
            f = self.gen_llvm_function(name)
        except AttributeError:
            f = _find_llvm_function(name, _all_modules | {self.module})
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

    def get_state_ptr(self, component, builder, state_ptr, state_name):
        idx = self.int32_ty(component._get_state_ids().index(state_name))
        return builder.gep(state_ptr, [self.int32_ty(0), idx],
                           name="ptr_state_{}_{}".format(state_name, component.name))

    def unwrap_2d_array(self, builder, element):
        if isinstance(element.type.pointee, ir.ArrayType) and isinstance(element.type.pointee.element, ir.ArrayType):
            assert element.type.pointee.count == 1
            return builder.gep(element, [self.int32_ty(0), self.int32_ty(0)])
        return element

    def inject_printf(self, builder, fmt, *args, override_debug=False):
        if "print_values" not in debug_env and not override_debug:
            return
        fmt += "\0"

        int8 = ir.IntType(8)
        stack_save = self.get_builtin("stacksave", [],
                                      ir.FunctionType(int8.as_pointer(), []))
        stack_restore = self.get_builtin("stackrestore", [],
                                         ir.FunctionType(ir.VoidType(), [int8.as_pointer()]))

        old_stack = builder.call(stack_save, [])
        fmt_data = bytearray(fmt.encode("utf8"))

        # Allocate array to ease initialization
        fmt = builder.alloca(ir.ArrayType(int8, len(fmt_data)))
        builder.store(fmt.type.pointee(fmt_data), fmt)
        fmt_ptr = builder.gep(fmt, [self.int32_ty(0), self.int32_ty(0)])

        printf = self.get_builtin("printf")
        builder.call(printf, [fmt_ptr] + list(args))

        builder.call(stack_restore, [old_stack])


    def inject_printf_float_array(self, builder, array, prefix="", suffix="\n", override_debug=False):
        self.inject_printf(builder,prefix,override_debug=override_debug)

        with pnlvm.helpers.array_ptr_loop(builder, array, "print_array_loop") as (b1, i):
            self.inject_printf(b1, "%f ", b1.load(b1.gep(array, [self.int32_ty(0), i])), override_debug=override_debug)

        self.inject_printf(builder,suffix,override_debug=override_debug)

    def gen_autodiffcomp_learning_exec(self,composition,simulation=False):
        composition._build_pytorch_representation(composition.default_execution_id)
        pytorch_model = composition.parameters.pytorch_representation.get(composition.default_execution_id)
        cond_gen = ConditionGenerator(self, composition)
        
        name = 'exec_learning_sim_wrap_' if simulation else 'exec_learning_wrap_'
        name += composition.name
        args = [self.get_state_struct_type(composition).as_pointer(),
                self.get_param_struct_type(composition).as_pointer(),
                self.get_input_struct_type(composition).as_pointer(),
                self.get_data_struct_type(composition).as_pointer(),
                cond_gen.get_condition_struct_type().as_pointer()]
        builder = self.create_llvm_function(args, composition, name)
        llvm_func = builder.function
        
        for a in llvm_func.args:
            a.attributes.add('noalias')
        
        context, params, comp_in, data_arg, cond = llvm_func.args
        pytorch_model._gen_llvm_training_function_body(self, builder, context, params, comp_in, data_arg, cond)
        # Call output CIM

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

        output_cim_w = composition._get_node_wrapper(composition.output_CIM, simulation)
        output_cim_f = self.get_llvm_function(output_cim_w)
        builder.block.name = "invoke_" + output_cim_f.name
        builder.call(output_cim_f, [context, params, comp_in, data, data])

        if "alloca_data" in debug_env:
            data_vals = builder.load(data)
            builder.store(data_vals, data_arg)

        # Bump run counter
        cond_gen.bump_ts(builder, cond, (1, 0, 0))

        builder.ret_void()
        
        return llvm_func

    def gen_autodiffcomp_exec(self,composition,simulation=False):
        """Creates llvm bin execute for autodiffcomp"""
        assert composition.controller is None
       
        composition._build_pytorch_representation(composition.default_execution_id)
        pytorch_model = composition.parameters.pytorch_representation.get(composition.default_execution_id)
        cond_gen = ConditionGenerator(self, composition)
        
        name = 'exec_sim_wrap_' if simulation else 'exec_wrap_'
        name += composition.name
        args = [self.get_state_struct_type(composition).as_pointer(),
                self.get_param_struct_type(composition).as_pointer(),
                self.get_input_struct_type(composition).as_pointer(),
                self.get_data_struct_type(composition).as_pointer(),
                cond_gen.get_condition_struct_type().as_pointer()]
        builder = self.create_llvm_function(args, composition, name)
        llvm_func = builder.function
        
        for a in llvm_func.args:
            a.attributes.add('noalias')
        
        context, params, comp_in, data_arg, cond = llvm_func.args

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
        
        # Call input CIM
        input_cim_w = composition._get_node_wrapper(composition.input_CIM, simulation)
        input_cim_f = self.get_llvm_function(input_cim_w)

        builder.call(input_cim_f, [context, params, comp_in, data, data])

        # Call pytorch internal compiled llvm func
        pytorch_forward_func = self.get_llvm_function(self.gen_llvm_function(pytorch_model).name)
        input_cim_idx = composition._get_node_index(composition.input_CIM)

        model_context = context
        model_params = builder.gep(params,[self.int32_ty(0),
                                        self.int32_ty(2)])

        # Extract the input that should be inserted into the model
        model_input = builder.gep(data,[self.int32_ty(0),
                                        self.int32_ty(0),
                                        self.int32_ty(input_cim_idx)])
        model_output = builder.gep(data,[self.int32_ty(0),
                                         ])
        
        builder.call(pytorch_forward_func,[model_context,model_params,model_input,model_output])
        
        # Call output CIM
        output_cim_w = composition._get_node_wrapper(composition.output_CIM, simulation)
        output_cim_f = self.get_llvm_function(output_cim_w)
        builder.block.name = "invoke_" + output_cim_f.name
        builder.call(output_cim_f, [context, params, comp_in, data, data])

        if "alloca_data" in debug_env:
            data_vals = builder.load(data)
            builder.store(data_vals, data_arg)

        # Bump run counter
        cond_gen.bump_ts(builder, cond, (1, 0, 0))

        builder.ret_void()
        
        return llvm_func

    def gen_composition_exec(self, composition, simulation=False):
        # Create condition generator
        cond_gen = ConditionGenerator(self, composition)

        name = 'exec_sim_wrap_' if simulation else 'exec_wrap_'
        name += composition.name
        args = [self.get_state_struct_type(composition).as_pointer(),
                self.get_param_struct_type(composition).as_pointer(),
                self.get_input_struct_type(composition).as_pointer(),
                self.get_data_struct_type(composition).as_pointer(),
                cond_gen.get_condition_struct_type().as_pointer()]
                
        builder = self.create_llvm_function(args, composition, name)
        llvm_func = builder.function
        for a in llvm_func.args:
            a.attributes.add('noalias')

        state, params, comp_in, data_arg, cond = llvm_func.args

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

        # Call input CIM
        input_cim_w = composition._get_node_wrapper(composition.input_CIM, simulation)
        input_cim_f = self.get_llvm_function(input_cim_w)
        builder.call(input_cim_f, [state, params, comp_in, data, data])

        if simulation is False and composition.enable_controller and \
           composition.controller_mode == BEFORE:
            assert composition.controller is not None
            controller = composition._get_node_wrapper(composition.controller, simulation)
            controller_f = self.get_llvm_function(controller)
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
        run_cond = cond_gen.generate_sched_condition(builder,
                        composition.termination_processing[TimeScale.TRIAL],
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
        for idx, mech in enumerate(composition.nodes):
            run_set_mech_ptr = builder.gep(run_set_ptr,
                                           [zero, self.int32_ty(idx)],
                                           name="run_cond_ptr_" + mech.name)
            mech_cond = cond_gen.generate_sched_condition(builder,
                            composition._get_processing_condition_set(mech),
                            cond, mech)
            ran = cond_gen.generate_ran_this_pass(builder, cond, mech)
            mech_cond = builder.and_(mech_cond, builder.not_(ran),
                                     name="run_cond_" + mech.name)
            any_cond = builder.or_(any_cond, mech_cond, name="any_ran_cond")
            builder.store(mech_cond, run_set_mech_ptr)

        for idx, mech in enumerate(composition.nodes):
            run_set_mech_ptr = builder.gep(run_set_ptr, [zero, self.int32_ty(idx)])
            mech_cond = builder.load(run_set_mech_ptr, name="mech_" + mech.name + "_should_run")
            with builder.if_then(mech_cond):
                mech_w = composition._get_node_wrapper(mech, simulation)
                mech_f = self.get_llvm_function(mech_w)
                builder.block.name = "invoke_" + mech_f.name
                # Wrappers do proper indexing of all structures
                if len(mech_f.args) == 5:  # Mechanism wrappers have 5 inputs
                    builder.call(mech_f, [state, params, comp_in, data, output_storage])
                else:
                    builder.call(mech_f, [state, params, comp_in, data, output_storage, cond])

                cond_gen.generate_update_after_run(builder, cond, mech)
            builder.block.name = "post_invoke_" + mech_f.name

        # Writeback results
        for idx, mech in enumerate(composition.nodes):
            run_set_mech_ptr = builder.gep(run_set_ptr, [zero, self.int32_ty(idx)])
            mech_cond = builder.load(run_set_mech_ptr, name="mech_" + mech.name + "_ran")
            with builder.if_then(mech_cond):
                out_ptr = builder.gep(output_storage, [zero, zero, self.int32_ty(idx)], name="result_ptr_" + mech.name)
                data_ptr = builder.gep(data, [zero, zero, self.int32_ty(idx)],
                                       name="data_result_" + mech.name)
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

        max_iters = len(composition.scheduler_processing.consideration_queue)
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
            controller = composition._get_node_wrapper(composition.controller, simulation)
            controller_f = self.get_llvm_function(controller)
            builder.call(controller_f, [state, params, comp_in, data, data])

        # Call output CIM
        output_cim_w = composition._get_node_wrapper(composition.output_CIM, simulation)
        output_cim_f = self.get_llvm_function(output_cim_w)
        builder.block.name = "invoke_" + output_cim_f.name
        builder.call(output_cim_f, [state, params, comp_in, data, data])

        if "alloca_data" in debug_env:
            data_vals = builder.load(data)
            builder.store(data_vals, data_arg)

        # Bump run counter
        cond_gen.bump_ts(builder, cond, (1, 0, 0))

        builder.ret_void()

        return llvm_func

    def gen_composition_run(self, composition, simulation=False):
        name = 'run_sim_wrap_' if simulation else 'run_wrap_'
        name += composition.name
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
                input_init = pnlvm._tupleize([[os.defaults.variable] for os in composition.input_CIM.input_states])
                print("Setting default input: ", input_init)
            else:
                input_init = ast.literal_eval(debug_env["const_input"])
                print("Setting user input: ", input_init)

            builder.store(data_in.type.pointee(input_init), data_in)
            builder.store(inputs_ptr.type.pointee(1), inputs_ptr)

        if "force_runs" in debug_env:
            num = int(debug_env["force_runs"]) if debug_env["force_runs"] else 1
            print("Forcing number of runs to: ", num)
            runs_ptr = builder.alloca(runs_ptr.type.pointee)
            builder.store(runs_ptr.type.pointee(num), runs_ptr)

        # Allocate and initialize condition structure
        cond_gen = ConditionGenerator(self, composition)
        cond_type = cond_gen.get_condition_struct_type()
        cond = builder.alloca(cond_type)
        cond_init = cond_type(cond_gen.get_condition_initializer())
        builder.store(cond_init, cond)

        iter_ptr = builder.alloca(self.int32_ty, name="iter_counter")
        builder.store(self.int32_ty(0), iter_ptr)

        loop_condition = builder.append_basic_block(name="run_loop_condition")
        builder.branch(loop_condition)

        # Generate a "while < count" loop
        builder.position_at_end(loop_condition)
        count = builder.load(iter_ptr)
        runs = builder.load(runs_ptr)
        run_cond = builder.icmp_unsigned('<', count, runs)

        loop_body = builder.append_basic_block(name="run_loop_body")
        exit_block = builder.append_basic_block(name="exit")
        builder.cbranch(run_cond, loop_body, exit_block)

        # Generate loop body
        builder.position_at_end(loop_body)

        # Current iteration
        iters = builder.load(iter_ptr)

        # Get the right input stimulus
        input_idx = builder.urem(iters, builder.load(inputs_ptr))
        data_in_ptr = builder.gep(data_in, [input_idx])
        
        # Call execution
        if simulation:
            exec_f = self.get_llvm_function(composition._llvm_simulation.name)
        else:
            exec_f = self.get_llvm_function(composition)
        builder.call(exec_f, [state, params, data_in_ptr, data, cond])

        if not simulation:
            # Extract output_CIM result
            idx = composition._get_node_index(composition.output_CIM)
            result_ptr = builder.gep(data, [self.int32_ty(0), self.int32_ty(0), self.int32_ty(idx)])
            output_ptr = builder.gep(data_out, [iters])
            result = builder.load(result_ptr)
            builder.store(result, output_ptr)

        # Increment counter
        iters = builder.add(iters, self.int32_ty(1))
        builder.store(iters, iter_ptr)
        builder.branch(loop_condition)

        builder.position_at_end(exit_block)

        # Store the number of executed iterations
        builder.store(builder.load(iter_ptr), runs_ptr)

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
        cond_block = multirun_f.append_basic_block(name="loop_cond")
        body_block = multirun_f.append_basic_block(name="loop_body")
        exit_block = multirun_f.append_basic_block(name="exit_loop")

        builder = ir.IRBuilder(block)

        limit_ptr = multirun_f.args[-1]
        index_ptr = builder.alloca(self.int32_ty)
        builder.store(index_ptr.type.pointee(0), index_ptr)
        builder.branch(cond_block)

        with builder.goto_block(cond_block):
            index = builder.load(index_ptr)
            limit = builder.load(limit_ptr)
            cond = builder.icmp_unsigned("<", index, limit)
            builder.cbranch(cond, body_block, exit_block)

        with builder.goto_block(body_block):
            # Runs need special handling. data_in and data_out are one dimensional,
            # but hold entries for all parallel invocations.
            is_comp_run = len(function.args) == 7
            if is_comp_run:
                runs_count = multirun_f.args[5]
                input_count = multirun_f.args[6]

            # Index all pointer arguments
            index = builder.load(index_ptr)
            indexed_args = []
            for i, arg in enumerate(multirun_f.args[:-1]):
                # Don't adjust #inputs and #trials
                if isinstance(arg.type, ir.PointerType):
                    offset = index
                    # #runs and #trials needs to be the same
                    if is_comp_run and i >= 5:
                        offset = self.int32_ty(0)
                    # data arrays need special handling
                    elif is_comp_run and i == 4:  # data_out
                        offset = builder.mul(index, builder.load(runs_count))
                    elif is_comp_run and i == 3:  # data_in
                        offset = builder.mul(index, builder.load(input_count))

                    arg = builder.gep(arg, [offset])

                indexed_args.append(arg)

            builder.call(function, indexed_args)
            new_idx = builder.add(index, index.type(1))
            builder.store(new_idx, index_ptr)
            builder.branch(cond_block)

        with builder.goto_block(exit_block):
            builder.ret_void()

        return multirun_f

    def convert_python_struct_to_llvm_ir(self, t):
        if type(t) is list:
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
        elif torch_available and isinstance(t,torch.Tensor):
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
def _convert_llvm_ir_to_ctype(t):
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
        # FIXME: Can this handle void*? Do we care?
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
