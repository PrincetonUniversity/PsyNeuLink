# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Binary Execution Wrappers **************************************************************

import concurrent.futures
import copy
import ctypes
import numpy as np
from inspect import isgenerator
import os
import sys
import time
from typing import Callable, Optional
import weakref


from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.context import Context

from . import helpers, jit_engine, builder_context
from .debug import debug_env

__all__ = ['CompExecution', 'FuncExecution', 'MechExecution']


def _convert_ctype_to_python(x):
    if isinstance(x, ctypes.Structure):
        return [_convert_ctype_to_python(getattr(x, field_name)) for field_name, _ in x._fields_]
    if isinstance(x, ctypes.Array):
        return [_convert_ctype_to_python(el) for el in x]
    if isinstance(x, (ctypes.c_double, ctypes.c_float)):
        return x.value
    if isinstance(x, (float, int)):
        return x

    assert False, "Don't know how to convert: {}".format(x)


def _tupleize(x):
    try:
        return tuple(_tupleize(y) for y in x)
    except TypeError:
        return x if x is not None else tuple()

def _element_dtype(x):
    """
    Extract base builtin type from aggregate type.

    Throws assertion failure if the aggregate type includes more than one base type.
    The assumption is that array of builtin type has the same binary layout as
    the original aggregate and it's easier to construct
    """
    dt = np.dtype(x)
    while dt.subdtype is not None:
        dt = dt.subdtype[0]

    if not dt.isbuiltin:
        fdts = (_element_dtype(f[0]) for f in dt.fields.values())
        dt = next(fdts)
        assert all(dt == fdt for fdt in fdts)

    assert dt.isbuiltin, "Element type is not builtin: {} from {}".format(dt, np.dtype(x))
    return dt

def _pretty_size(size):
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']
    for u in units:
        if abs(size) > 1536.0:
            size /= 1024.0
        else:
            break

    return "{:.2f} {}".format(size, u)


class Execution:
    def __init__(self):
        self._debug_env = debug_env

    def _get_compilation_param(self, name, init_method, arg):
        saved = getattr(self, name, None)
        if saved is None:
            struct_ty = self._bin_func.byref_arg_types[arg]
            init_f = getattr(self._obj, init_method)
            if len(self._execution_contexts) > 1:
                struct_ty = struct_ty * len(self._execution_contexts)
                init_start = time.time()
                initializer = (init_f(ex) for ex in self._execution_contexts)
            else:
                init_start = time.time()
                initializer = init_f(self._execution_contexts[0])

            init_end = time.time()
            struct = struct_ty(*initializer)
            struct_end = time.time()

            # numpy "frombuffer" creates a shared memory view of the provided buffer
            numpy_struct = np.frombuffer(struct, dtype=self._bin_func.np_params[arg], count=len(self._execution_contexts))

            assert numpy_struct.nbytes == ctypes.sizeof(struct), \
                "Size mismatch ({}), numpy: {} vs. ctypes:{}".format(name, numpy_struct.nbytes, ctypes.sizeof(struct))

            saved = (struct, numpy_struct)
            setattr(self, name, saved)

            if "time_stat" in self._debug_env:
                print("Time to get initializer for struct:", name,
                      "for", self._obj.name, ":", init_end - init_start)
                print("Time to instantiate struct:", name,
                      "for", self._obj.name, ":", struct_end - init_end)

            if "stat" in self._debug_env:
                print("Instantiated struct:", name, "( size:" ,
                      _pretty_size(ctypes.sizeof(struct_ty)), ")",
                      "for", self._obj.name)

            if len(self._execution_contexts) == 1:

                numpy_struct.shape = ()

                if name == '_state':
                    self._copy_params_to_pnl(self._execution_contexts[0],
                                             self._obj,
                                             numpy_struct,
                                             "llvm_state_ids")

                elif name == '_param':
                    self._copy_params_to_pnl(self._execution_contexts[0],
                                             self._obj,
                                             numpy_struct,
                                             "llvm_param_ids")

        return saved

    def _copy_params_to_pnl(self, context, component, params, ids:str):

        assert len(params.dtype.names) == len(getattr(component, ids))

        for numpy_name, attribute in zip(params.dtype.names, getattr(component, ids)):

            numpy_field = params[numpy_name]
            assert numpy_field.base is params or numpy_field.base is params.base

            def _enumerate_recurse(elements):
                for numpy_element_name, element in zip(numpy_field.dtype.names, elements):
                    numpy_element = numpy_field[numpy_element_name]
                    assert numpy_element.base is numpy_field.base

                    self._copy_params_to_pnl(context=context,
                                             component=element,
                                             params=numpy_element,
                                             ids=ids)

            # Handle custom compiled-only structures by name
            if attribute == 'nodes':
                _enumerate_recurse(component._all_nodes)

            elif attribute == 'projections':
                _enumerate_recurse(component._inner_projections)

            elif attribute == '_parameter_ports':
                _enumerate_recurse(component._parameter_ports)

            else:
                # TODO: Reconstruct Python RandomState
                if attribute == "random_state":
                    continue

                # TODO: Reconstruct Python memory storage
                if attribute == "ring_memory":
                    continue

                # TODO: Reconstruct Time class
                if attribute == "num_executions":
                    continue

                # TODO: Add support for syncing optimizer state
                if attribute == "optimizer":
                    continue

                # "old_val" is a helper storage in compiled RecurrentTransferMechanism
                # to workaround the fact that compiled projections do no pull values
                # from their source output ports
                if attribute == "old_val":
                    continue

                # Handle PNL parameters
                pnl_param = getattr(component.parameters, attribute)

                # Use ._get to retrieve underlying numpy arrays
                # (.get will extract a scalar if originally set as a scalar)
                pnl_value = pnl_param._get(context=context)

                # Recurse if the value is a PNL object with its own parameters
                if hasattr(pnl_value, 'parameters'):
                    self._copy_params_to_pnl(context=context,
                                             component=pnl_value,
                                             params=numpy_field,
                                             ids=ids)

                elif attribute == "input_ports" or attribute == "output_ports":
                    _enumerate_recurse(pnl_value)

                # Writeback parameter value
                else:

                    # Replace empty structures with None
                    if numpy_field.nbytes == 0:
                        value = None
                    else:
                        value = numpy_field

                        # Stateful parameters include history, get the most recent value
                        if "state" in ids:
                            value = value[-1]

                        # Reshape to match the shape of the old value.
                        # Do not try to reshape ragged arrays.
                        if getattr(pnl_value, 'dtype', object) != object and pnl_value.shape != value.shape:

                            # Reshape to match numpy 0d arrays and "matrix"
                            # parameters that are flattened in compiled form
                            assert pnl_value.shape == () or pnl_param.name == "matrix", \
                                "{}: {} vs. {}".format(pnl_param.name, pnl_value.shape, value.shape)

                            # Use an assignment instead of reshape().
                            # The latter would silently create a copy if the shape
                            # could not be achieved in metadata (stride, type, ...)
                            value.shape = pnl_value.shape

                    pnl_param.set(value, context=context, override=True, compilation_sync=True)


class CUDAExecution(Execution):
    def __init__(self, buffers=['param_struct', 'state_struct', 'out']):
        super().__init__()
        self._gpu_buffers = {}
        for b in buffers:
            self._gpu_buffers["_" + b] = None

    @property
    def _bin_func_multirun(self):
        # CUDA uses the same function for single and multi run
        return self._bin_func

    def __get_cuda_arg(self, struct_name, arg_handler):
        gpu_buffer = self._gpu_buffers[struct_name]

        np_struct = getattr(self, struct_name)[1]

        # .array is a public member of pycuda's In/Out ArgumentHandler classes
        if gpu_buffer is None or gpu_buffer.array is not np_struct:

            # 0-sized structures fail to upload use a dummy numpy array isntead
            gpu_buffer = arg_handler(np_struct if np_struct.nbytes > 0 else np.zeros(2))

            self._gpu_buffers[struct_name] = gpu_buffer

        return gpu_buffer

    @property
    def _cuda_param_struct(self):
        return self.__get_cuda_arg("_param_struct", jit_engine.pycuda.driver.In)

    @property
    def _cuda_state_struct(self):
        return self.__get_cuda_arg("_state_struct", jit_engine.pycuda.driver.InOut)

    @property
    def _cuda_data_struct(self):
        return self.__get_cuda_arg("_data_struct", jit_engine.pycuda.driver.InOut)

    @property
    def _cuda_conditions(self):
        return self.__get_cuda_arg("_conditions", jit_engine.pycuda.driver.InOut)

    @property
    def _cuda_out(self):
        gpu_buffer = self._gpu_buffers["_out"]
        if gpu_buffer is None:
            gpu_buffer = jit_engine.pycuda.driver.Out(np.ctypeslib.as_array(self._ct_vo))
            self._gpu_buffers["_out"] = gpu_buffer

        return gpu_buffer

    def cuda_execute(self, variable):
        # Create input argument
        new_var = np.asfarray(variable, dtype=self._vi_dty)
        data_in = jit_engine.pycuda.driver.In(new_var)

        self._bin_func.cuda_call(self._cuda_param_struct,
                                 self._cuda_state_struct,
                                 data_in,
                                 self._cuda_out,
                                 threads=len(self._execution_contexts))

        return _convert_ctype_to_python(self._ct_vo)


class FuncExecution(CUDAExecution):

    def __init__(self, component, execution_ids=[None], *, tags=frozenset()):
        super().__init__()

        self._bin_func = pnlvm.LLVMBinaryFunction.from_obj(component, tags=tags, numpy_args=(0, 1))
        self._execution_contexts = [
            Context(execution_id=eid) for eid in execution_ids
        ]
        self._component = component

        _, _, vi_ty, vo_ty = self._bin_func.byref_arg_types

        if len(execution_ids) > 1:
            self._bin_multirun = self._bin_func.get_multi_run()
            self._ct_len = ctypes.c_int(len(execution_ids))
            vo_ty = vo_ty * len(execution_ids)
            vi_ty = vi_ty * len(execution_ids)

        self._ct_vo = vo_ty()
        self._vi_dty = _element_dtype(vi_ty)
        if "stat" in self._debug_env:
            print("Input struct size:", _pretty_size(ctypes.sizeof(vi_ty)),
                  "for", self._component.name)
            print("Output struct size:", _pretty_size(ctypes.sizeof(vo_ty)),
                  "for", self._component.name)

    @property
    def _obj(self):
        return self._component

    @property
    def _param_struct(self):
        return self._get_compilation_param('_param', '_get_param_initializer', 0)

    @property
    def _state_struct(self):
        return self._get_compilation_param('_state', '_get_state_initializer', 1)

    def execute(self, variable):
        # Make sure function inputs are 2d.
        # Mechanism inputs are already 3d so the first part is nop.
        new_variable = np.asfarray(np.atleast_2d(variable), dtype=self._vi_dty)

        ct_vi = np.ctypeslib.as_ctypes(new_variable)
        if len(self._execution_contexts) > 1:
            # wrap_call casts the arguments so we only need contiguous data layout
            self._bin_multirun.wrap_call(self._param_struct[0],
                                         self._state_struct[0],
                                         ct_vi,
                                         self._ct_vo,
                                         self._ct_len)
        else:
            self._bin_func(self._param_struct[1], self._state_struct[1], ct_vi, self._ct_vo)

        return _convert_ctype_to_python(self._ct_vo)


class MechExecution(FuncExecution):

    def execute(self, variable):
        # Convert to 3d. We always assume that:
        #   a) the input is vector of input ports
        #   b) input ports take vector of projection outputs
        #   c) projection output is a vector (even 1 element vector)
        new_var = np.atleast_3d(variable)
        new_var.shape = (len(self._component.input_ports), 1, -1)
        return super().execute(new_var)


class CompExecution(CUDAExecution):

    active_executions = weakref.WeakSet()

    def __init__(self, composition, execution_ids=[None], *, additional_tags=frozenset()):
        super().__init__(buffers=['state_struct', 'param_struct', 'data_struct', 'conditions'])
        self._composition = composition
        self._execution_contexts = [
            Context(execution_id=eid) for eid in execution_ids
        ]
        self.__bin_exec_func = None
        self.__bin_exec_multi_func = None
        self.__bin_func = None
        self.__bin_run_func = None
        self.__bin_run_multi_func = None
        self.__frozen_vals = None
        self.__tags = frozenset(additional_tags)

        self.__conds = None

        if len(execution_ids) > 1:
            self._ct_len = ctypes.c_int(len(execution_ids))

        self.active_executions.add(self)

    def __del__(self):
        self.active_executions.discard(self)

    @staticmethod
    def get(composition, context, additional_tags=frozenset()):
        executions = composition._compilation_data.execution._get(context)
        if executions is None:
            executions = dict()
            composition._compilation_data.execution._set(executions, context)

        execution = executions.get(additional_tags, None)
        if execution is None:
            execution = pnlvm.CompExecution(composition, [context.execution_id],
                                            additional_tags=additional_tags)
            executions[additional_tags] = execution

        return execution

    @property
    def _obj(self):
        return self._composition

    @property
    def _bin_func(self):
        if self.__bin_func is not None:
            assert len(self._execution_contexts) == 1
            return self.__bin_func
        if self.__bin_exec_func is not None:
            return self.__bin_exec_func
        if self.__bin_run_func is not None:
            return self.__bin_run_func

        assert False, "Binary function not set for execution!"

    @property
    def _bin_func_multirun(self):
        if self.__bin_exec_multi_func is not None:
            return self.__bin_exec_multi_func
        if self.__bin_run_multi_func is not None:
            return self.__bin_run_multi_func

        return super()._bin_func_multirun

    def _set_bin_node(self, node):
        assert node in self._composition._all_nodes
        wrapper = builder_context.LLVMBuilderContext.get_current().get_node_assembly(self._composition, node)
        self.__bin_func = pnlvm.LLVMBinaryFunction.from_obj(
            wrapper, tags=self.__tags.union({"node_assembly"}), numpy_args=(0, 1, 4))

    @property
    def _conditions(self):
        if self.__conds is None:
            gen = helpers.ConditionGenerator(None, self._composition)
            if len(self._execution_contexts) > 1:
                cond_ctype = self._bin_func_multirun.byref_arg_types[4] * len(self._execution_contexts)
                cond_initializer = (gen.get_condition_initializer() for _ in self._execution_contexts)
            else:
                cond_ctype = self._bin_func.byref_arg_types[4]
                cond_initializer = gen.get_condition_initializer()

            c_conds = cond_ctype(*cond_initializer)
            self.__conds = (c_conds, np.ctypeslib.as_array(c_conds))
            if "stat" in self._debug_env:
                print("Instantiated condition struct ( size:" ,
                      _pretty_size(ctypes.sizeof(cond_ctype)), ")",
                      "for", self._composition.name)

        return self.__conds

    @property
    def _param_struct(self):
        return self._get_compilation_param('_param', '_get_param_initializer', 1)

    @property
    def _state_struct(self):
        return self._get_compilation_param('_state', '_get_state_initializer', 0)

    @property
    def _data_struct(self):
        # Run wrapper changed argument order
        arg = 2 if self._bin_func is self.__bin_run_func else 3
        return self._get_compilation_param('_data', '_get_data_initializer', arg)

    @_data_struct.setter
    def _data_struct(self, data_struct):
        self._data = data_struct

    def _extract_node_struct(self, node, data):
        # context structure consists of a list of node contexts,
        #   followed by a list of projection contexts; get the first one
        # parameter structure consists of a list of node parameters,
        #   followed by a list of projection parameters; get the first one
        # output structure consists of a list of node outputs,
        #   followed by a list of nested data structures; get the first one
        field_name = data._fields_[0][0]
        res_struct = getattr(data, field_name)

        # Get the index into the array of all nodes
        index = self._composition._get_node_index(node)
        field_name = res_struct._fields_[index][0]
        res_struct = getattr(res_struct, field_name)

        return _convert_ctype_to_python(res_struct)

    def extract_node_struct(self, node, struct):
        if len(self._execution_contexts) > 1:
            return [self._extract_node_struct(node, struct[i]) for i, _ in enumerate(self._execution_contexts)]
        else:
            return self._extract_node_struct(node, struct)

    def extract_frozen_node_output(self, node):
        return self.extract_node_struct(node, self.__frozen_vals)

    def extract_node_output(self, node):
        return self.extract_node_struct(node, self._data_struct[0])

    def extract_node_state(self, node):
        return self.extract_node_struct(node, self._state_struct[0])

    def extract_node_params(self, node):
        return self.extract_node_struct(node, self._param_struct[0])

    def insert_node_output(self, node, data):
        my_field_name = self._data_struct[0]._fields_[0][0]
        my_res_struct = getattr(self._data_struct[0], my_field_name)
        index = self._composition._get_node_index(node)
        node_field_name = my_res_struct._fields_[index][0]
        setattr(my_res_struct, node_field_name, _tupleize(data))

    def _get_input_struct(self, inputs):
        # Either node or composition execute.
        # All execute functions expect inputs to be 3rd param.
        c_input_type = self._bin_func.byref_arg_types[2]

        # Read provided input data and parse into an array (generator)
        if len(self._execution_contexts) > 1:
            assert len(self._execution_contexts) == len(inputs)
            c_input_type = c_input_type * len(self._execution_contexts)
            input_data = (([x] for x in self._composition._build_variable_for_input_CIM(inp)) for inp in inputs)
        else:
            input_data = ([x] for x in self._composition._build_variable_for_input_CIM(inputs))

        if "stat" in self._debug_env:
            print("Input struct size:", _pretty_size(ctypes.sizeof(c_input_type)),
                  "for", self._composition.name)
        c_input = c_input_type(*_tupleize(input_data))
        return c_input, np.ctypeslib.as_array(c_input)

    def freeze_values(self):
        self.__frozen_vals = copy.deepcopy(self._data_struct[0])

    def execute_node(self, node, inputs=None, context=None):
        # We need to reconstruct the input dictionary here if it was not provided.
        # This happens during node execution of nested compositions.
        assert len(self._execution_contexts) == 1
        if inputs is None and node is self._composition.input_CIM:
            if context is None:
                context = self._execution_contexts[0]
            port_inputs = {origin_port:[proj.parameters.value._get(context) for proj in p[0].path_afferents] for (origin_port, p) in self._composition.input_CIM_ports.items()}
            inputs = {}
            for p, v in port_inputs.items():
                data = inputs.setdefault(p.owner, [0] * len(p.owner.input_ports))
                index = p.owner.input_ports.index(p)
                data[index] = v[0]


        # Set bin node to make sure self._*struct works as expected
        self._set_bin_node(node)
        if inputs is not None:
            inputs = self._get_input_struct(inputs)[0]

        assert inputs is not None or node is not self._composition.input_CIM

        # Freeze output values if this is the first time we need them
        if node is not self._composition.input_CIM and self.__frozen_vals is None:
            self.freeze_values()

        self._bin_func(self._state_struct[1],
                       self._param_struct[1],
                       inputs,
                       self.__frozen_vals,
                       self._data_struct[1])

        if "comp_node_debug" in self._debug_env:
            print("RAN: {}. State: {}".format(node, self.extract_node_state(node)))
            print("RAN: {}. Params: {}".format(node, self.extract_node_params(node)))
            print("RAN: {}. Results: {}".format(node, self.extract_node_output(node)))

        node._propagate_most_recent_context(context)

    @property
    def _bin_exec_func(self):
        if self.__bin_exec_func is None:
            self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(
                self._composition, tags=self.__tags, numpy_args=(0, 1, 3))

        return self.__bin_exec_func

    @property
    def _bin_exec_multi_func(self):
        if self.__bin_exec_multi_func is None:
            self.__bin_exec_multi_func = self._bin_exec_func.get_multi_run()

        return self.__bin_exec_multi_func

    def execute(self, inputs):
        # NOTE: Make sure that input struct generation is inlined.
        # We need the binary function to be setup for it to work correctly.
        if len(self._execution_contexts) > 1:
            self._bin_exec_multi_func.wrap_call(self._state_struct[0],
                                                self._param_struct[0],
                                                self._get_input_struct(inputs)[0],
                                                self._data_struct[0],
                                                self._conditions[0],
                                                self._ct_len)
        else:
            self._bin_exec_func(self._state_struct[1],
                                self._param_struct[1],
                                self._get_input_struct(inputs)[0],
                                self._data_struct[1],
                                self._conditions[0])

    def cuda_execute(self, inputs):
        # NOTE: Make sure that input struct generation is inlined.
        # We need the binary function to be setup for it to work correctly.
        self._bin_exec_func.cuda_call(self._cuda_state_struct,
                                      self._cuda_param_struct,
                                      jit_engine.pycuda.driver.In(self._get_input_struct(inputs)[1]),
                                      self._cuda_data_struct,
                                      self._cuda_conditions,
                                      threads=len(self._execution_contexts))

    # Methods used to accelerate "Run"
    def _get_run_input_struct(self, inputs, num_input_sets, arg=3):
        # Callers that override input arg, should ensure that _bin_func is not None
        bin_f = self._bin_run_func if arg == 3 else self._bin_func

        input_type = bin_f.byref_arg_types[arg]
        c_input_type = (input_type * num_input_sets) * len(self._execution_contexts)
        if len(self._execution_contexts) == 1:
            inputs = [inputs]

        assert len(inputs) == len(self._execution_contexts)
        # Extract input for each trial and execution id
        run_inputs = ((([x] for x in self._composition._build_variable_for_input_CIM({k:v[i] for k,v in inp.items()})) for i in range(num_input_sets)) for inp in inputs)
        c_inputs = c_input_type(*_tupleize(run_inputs))
        if "stat" in self._debug_env:
            print("Instantiated struct: input ( size:" ,
                  _pretty_size(ctypes.sizeof(c_inputs)),
                  ")",
                  "for",
                  self._obj.name)

        return c_inputs

    def _get_generator_run_input_struct(self, inputs, runs):
        assert len(self._execution_contexts) == 1
        # Extract input for each trial
        run_inputs = ((np.atleast_2d(x) for x in self._composition._build_variable_for_input_CIM({k:np.atleast_1d(v) for k,v in inp.items()})) for inp in inputs)
        run_inputs = _tupleize(run_inputs)
        num_input_sets = len(run_inputs)
        runs = num_input_sets if runs == 0 or runs == sys.maxsize else runs
        c_input_type = self._bin_run_func.byref_arg_types[3] * num_input_sets
        return c_input_type(*run_inputs), runs

    @property
    def _bin_run_func(self):
        if self.__bin_run_func is None:
            self.__bin_run_func = pnlvm.LLVMBinaryFunction.from_obj(
                self._composition, tags=self.__tags.union({"run"}), numpy_args=(0, 1, 2))

        return self.__bin_run_func

    @property
    def _bin_run_multi_func(self):
        if self.__bin_run_multi_func is None:
            self.__bin_run_multi_func = self._bin_run_func.get_multi_run()

        return self.__bin_run_multi_func

    def run(self, inputs, runs=0, num_input_sets=0):
        if isgenerator(inputs):
            inputs, runs = self._get_generator_run_input_struct(inputs, runs)
            assert num_input_sets == 0 or num_input_sets == sys.maxsize
            num_input_sets = len(inputs)
        else:
            inputs = self._get_run_input_struct(inputs, num_input_sets)

        ct_vo = self._bin_run_func.byref_arg_types[4] * runs
        if len(self._execution_contexts) > 1:
            ct_vo = ct_vo * len(self._execution_contexts)

        outputs = ct_vo()

        if "stat" in self._debug_env:
            print("Input struct size:", _pretty_size(ctypes.sizeof(inputs)),
                  "for", self._composition.name)
            print("Output struct size:", _pretty_size(ctypes.sizeof(outputs)),
                  "for", self._composition.name)

        runs_count = ctypes.c_uint(runs)
        input_count = ctypes.c_uint(num_input_sets)
        if len(self._execution_contexts) > 1:
            self._bin_run_multi_func.wrap_call(self._state_struct[0],
                                               self._param_struct[0],
                                               self._data_struct[0],
                                               inputs,
                                               outputs,
                                               runs_count,
                                               input_count,
                                               self._ct_len)

            return _convert_ctype_to_python(outputs)
        else:
            # This is only needed for non-generator inputs that are wrapped in an extra context dimension
            inputs = ctypes.cast(inputs, self._bin_run_func.c_func.argtypes[3])

            self._bin_run_func(self._state_struct[1],
                               self._param_struct[1],
                               self._data_struct[1],
                               inputs,
                               outputs,
                               runs_count,
                               input_count)

            # Extract only #trials elements in case the run exited early
            assert runs_count.value <= runs, "Composition ran more times than allowed!"
            return _convert_ctype_to_python(outputs)[0:runs_count.value]

    def cuda_run(self, inputs, runs, num_input_sets):
        # Create input buffer
        if isgenerator(inputs):
            ct_inputs, runs = self._get_generator_run_input_struct(inputs, runs)
            assert num_input_sets == 0 or num_input_sets == sys.maxsize
            num_input_sets = len(ct_inputs)
        else:
            ct_inputs = self._get_run_input_struct(inputs, num_input_sets)

        # Create output buffer
        output_type = (self._bin_run_func.byref_arg_types[4] * runs)
        if len(self._execution_contexts) > 1:
            output_type = output_type * len(self._execution_contexts)

        ct_out = output_type()

        # number of trials argument
        np_runs = np.full(len(self._execution_contexts), runs, dtype=np.int32)

        self._bin_run_func.cuda_call(self._cuda_state_struct,
                                     self._cuda_param_struct,
                                     self._cuda_data_struct,
                                     jit_engine.pycuda.driver.In(np.ctypeslib.as_array(ct_inputs)), # input
                                     jit_engine.pycuda.driver.Out(np.ctypeslib.as_array(ct_out)),   # output
                                     jit_engine.pycuda.driver.InOut(np_runs),                       # runs
                                     jit_engine.pycuda.driver.In(np.int32(num_input_sets)),         # number of inputs
                                     threads=len(self._execution_contexts))

        assert all(np_runs <= runs), "Composition ran more times than allowed: {}".format(runs)

        if len(self._execution_contexts) > 1:
            return _convert_ctype_to_python(ct_out)
        else:
            # Extract only #trials elements in case the run exited early
            return _convert_ctype_to_python(ct_out)[0:np_runs[0]]

    def _prepare_evaluate(self, inputs, num_input_sets, num_evaluations, all_results:bool):
        ocm = self._composition.controller
        assert len(self._execution_contexts) == 1

        eval_type = "evaluate_type_all_results" if all_results else "evaluate_type_objective"
        tags = {"evaluate", "alloc_range", eval_type}
        bin_func = pnlvm.LLVMBinaryFunction.from_obj(ocm, tags=frozenset(tags), numpy_args=(0, 1, 6))
        self.__bin_func = bin_func

        # There are 8 arguments to evaluate_alloc_range:
        # comp_param, comp_state, from, to, results, input, comp_data, num_inputs
        # all but #4 are shared
        assert len(bin_func.byref_arg_types) == 8

        # Directly initialized structures
        assert ocm.agent_rep is self._composition
        comp_params = self._get_compilation_param('_eval_param', '_get_param_initializer', 0)[1]
        comp_state = self._get_compilation_param('_eval_state', '_get_state_initializer', 1)[1]
        comp_data = self._get_compilation_param('_eval_data', '_get_data_initializer', 6)[1]

        # Construct input variable, the 5th parameter of the evaluate function
        ct_inputs = self._get_run_input_struct(inputs, num_input_sets, 5)

        # Output ctype
        out_el_ty = bin_func.byref_arg_types[4]
        if all_results:
            num_trials = ocm.parameters.num_trials_per_estimate.get(self._execution_contexts[0])
            if num_trials is None:
                num_trials = num_input_sets
            out_el_ty *= num_trials
        out_ty = out_el_ty * num_evaluations

        ct_num_inputs = bin_func.byref_arg_types[7](num_input_sets)
        if "stat" in self._debug_env:
            print("Evaluate result struct type size:",
                  _pretty_size(ctypes.sizeof(out_ty)),
                  "( evaluations:", num_evaluations, "element size:", ctypes.sizeof(out_el_ty), ")",
                  "for", self._obj.name)

        return comp_params, comp_state, comp_data, ct_inputs, out_ty, ct_num_inputs

    def cuda_evaluate(self, inputs, num_input_sets, num_evaluations, all_results:bool=False):
        comp_params, comp_state, comp_data, ct_inputs, out_ty, _ = \
            self._prepare_evaluate(inputs, num_input_sets, num_evaluations, all_results)

        ct_results = out_ty()

        cuda_args = (jit_engine.pycuda.driver.In(comp_params),
                     jit_engine.pycuda.driver.InOut(comp_state),
                     jit_engine.pycuda.driver.Out(np.ctypeslib.as_array(ct_results)),   # results
                     jit_engine.pycuda.driver.In(np.ctypeslib.as_array(ct_inputs)),     # inputs
                     jit_engine.pycuda.driver.InOut(comp_data),                         # composition data
                     jit_engine.pycuda.driver.In(np.int32(num_input_sets)),             # number of inputs
                    )

        self.__bin_func.cuda_call(*cuda_args, threads=int(num_evaluations))

        return ct_results

    def thread_evaluate(self, inputs, num_input_sets, num_evaluations, all_results:bool=False):
        comp_params, comp_state, comp_data, ct_inputs, out_ty, ct_num_inputs = \
            self._prepare_evaluate(inputs, num_input_sets, num_evaluations, all_results)

        ct_results = out_ty()
        jobs = min(os.cpu_count(), num_evaluations)
        evals_per_job = (num_evaluations + jobs - 1) // jobs

        parallel_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:

            # Create input and result typed casts once, they are the same
            # for every submitted job.
            input_arg = ctypes.cast(ct_inputs, self.__bin_func.c_func.argtypes[5])
            results_arg = ctypes.cast(ct_results, self.__bin_func.c_func.argtypes[4])

            # There are 7 arguments to evaluate_alloc_range:
            # comp_param, comp_state, from, to, results, input, comp_data
            results = [ex.submit(self.__bin_func,
                                 comp_params,
                                 comp_state,
                                 int(i * evals_per_job),
                                 min((i + 1) * evals_per_job, num_evaluations),
                                 results_arg,
                                 input_arg,
                                 comp_data,
                                 ct_num_inputs)
                       for i in range(jobs)]

        parallel_stop = time.time()
        if "time_stat" in self._debug_env:
            print("Time to run {} executions of '{}' in {} threads: {}".format(
                      num_evaluations, self.__bin_func.name, jobs,
                      parallel_stop - parallel_start))


        exceptions = [r.exception() for r in results]
        assert all(e is None for e in exceptions), "Not all jobs finished sucessfully: {}".format(exceptions)

        return ct_results
