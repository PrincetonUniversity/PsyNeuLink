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

from . import builder_context, jit_engine, scheduler
from .debug import debug_env

__all__ = ['CompExecution', 'FuncExecution', 'MechExecution']


def _tupleize(x):
    try:
        return tuple(_tupleize(y) for y in x)
    except TypeError:
        return x if x is not None else tuple()


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
            init_start = time.time()
            initializer = init_f(self._execution_context)

            init_end = time.time()
            struct = struct_ty(*initializer)
            struct_end = time.time()

            # numpy "frombuffer" creates a shared memory view of the provided buffer
            numpy_struct = np.frombuffer(struct, dtype=self._bin_func.np_arg_dtypes[arg], count=1)

            assert numpy_struct.nbytes == ctypes.sizeof(struct), \
                "Size mismatch ({}), numpy: {} vs. ctypes:{}".format(name, numpy_struct.nbytes, ctypes.sizeof(struct))

            saved = numpy_struct
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

            numpy_struct.shape = ()

            if name == '_state':
                self._copy_params_to_pnl(self._execution_context,
                                         self._obj,
                                         numpy_struct,
                                         "llvm_state_ids")

            elif name == '_param':
                self._copy_params_to_pnl(self._execution_context,
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

    def _get_indexable(self, np_array):
        # outputs in recarrays need to be converted to list/tuple to be indexable
        return np_array.tolist() if np_array.dtype.base.shape == () else np_array

class CUDAExecution(Execution):
    def __init__(self, buffers=['param_struct', 'state_struct']):
        super().__init__()

        # Initialize GPU buffer map
        self._gpu_buffers = {"_" + b: None for b in buffers}

    def __get_cuda_arg(self, struct_name, arg_handler):
        gpu_buffer = self._gpu_buffers[struct_name]

        np_struct = getattr(self, struct_name)

        # .array is a public member of pycuda's In/Out ArgumentHandler classes
        if gpu_buffer is None or gpu_buffer.array is not np_struct:

            # 0-sized structures fail to upload use a dummy numpy array instead
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


class FuncExecution(CUDAExecution):

    def __init__(self, component, execution_id=None, *, tags=frozenset()):
        super().__init__()

        self._bin_func = pnlvm.LLVMBinaryFunction.from_obj(component, tags=tags)
        self._execution_context = Context(execution_id=execution_id)
        self._component = component

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
        new_variable = np.asfarray(variable, dtype=self._bin_func.np_arg_dtypes[2].base)
        data_in = new_variable.reshape(self._bin_func.np_arg_dtypes[2].shape)

        data_out = self._bin_func.np_buffer_for_arg(3)

        self._bin_func(self._param_struct, self._state_struct, data_in, data_out)

        return self._get_indexable(data_out)

    def cuda_execute(self, variable):
        # Create input argument, PyCUDA doesn't care about shape
        data_in = np.asfarray(variable, dtype=self._bin_func.np_arg_dtypes[2].base)
        data_out = self._bin_func.np_buffer_for_arg(3)

        self._bin_func.cuda_call(self._cuda_param_struct,
                                 self._cuda_state_struct,
                                 jit_engine.pycuda.driver.In(data_in),
                                 jit_engine.pycuda.driver.Out(data_out))

        return self._get_indexable(data_out)


class MechExecution(FuncExecution):
    pass


class CompExecution(CUDAExecution):

    active_executions = weakref.WeakSet()

    def __init__(self, composition, context:Context, *, additional_tags=frozenset()):
        super().__init__(buffers=['state_struct', 'param_struct', 'data_struct', 'conditions'])
        self._composition = composition
        self._execution_context = context
        self.__bin_exec_func = None
        self.__bin_func = None
        self.__bin_run_func = None
        self.__frozen_values = None
        self.__tags = frozenset(additional_tags)

        # Scheduling conditions, only used by "execute"
        self.__conditions = None

        self.active_executions.add(self)

    def __del__(self):
        self.active_executions.discard(self)

    @staticmethod
    def get(composition, context:Context, additional_tags=frozenset()):
        executions = composition._compilation_data.execution._get(context)
        if executions is None:
            executions = dict()
            composition._compilation_data.execution._set(executions, context)

        execution = executions.get(additional_tags, None)
        if execution is None:
            execution = pnlvm.CompExecution(composition, context, additional_tags=additional_tags)
            executions[additional_tags] = execution

        return execution

    @property
    def _obj(self):
        return self._composition

    @property
    def _bin_func(self):
        if self.__bin_func is not None:
            return self.__bin_func
        if self.__bin_exec_func is not None:
            return self.__bin_exec_func
        if self.__bin_run_func is not None:
            return self.__bin_run_func

        assert False, "Binary function not set for execution!"

    def _set_bin_node(self, node):
        assert node in self._composition._all_nodes
        node_assembly = builder_context.LLVMBuilderContext.get_current().get_node_assembly(self._composition, node)
        self.__bin_func = pnlvm.LLVMBinaryFunction.from_obj(node_assembly, tags=self.__tags.union({"node_assembly"}))

    @property
    def _conditions(self):
        if self.__conditions is None:
            gen = scheduler.ConditionGenerator(None, self._composition)

            conditions_ctype = self._bin_func.byref_arg_types[4]
            conditions_initializer = gen.get_condition_initializer()

            ct_conditions = conditions_ctype(*conditions_initializer)
            np_conditions = np.frombuffer(ct_conditions, dtype=self._bin_func.np_arg_dtypes[4], count=1)

            np_conditions.shape = ()

            self.__conditions = np_conditions

            if "stat" in self._debug_env:
                print("Instantiated condition struct ( size:" ,
                      _pretty_size(np_conditions.nbytes), ")",
                      "for", self._composition.name)

        return self.__conditions

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

    def extract_node_struct(self, node, data):
        # state structure consists of a list of node states,
        #   followed by a list of projection contexts; get the first one
        # parameter structure consists of a list of node parameters,
        #   followed by a list of projection parameters; get the first one
        # output structure consists of a list of node outputs,
        #   followed by a list of nested data structures; get the first one
        all_nodes = data[data.dtype.names[0]]

        # Get the index into the array of all nodes
        index = self._composition._get_node_index(node)
        node_struct = all_nodes[all_nodes.dtype.names[index]]

        # Return copies of the extracted functions to avoid corrupting the
        # returned results in next execution
        return node_struct.copy().tolist() if node_struct.shape == () else node_struct.copy()

    def extract_frozen_node_output(self, node):
        return self.extract_node_struct(node, self.__frozen_values)

    def extract_node_output(self, node):
        return self.extract_node_struct(node, self._data_struct)

    def extract_node_state(self, node):
        return self.extract_node_struct(node, self._state_struct)

    def extract_node_params(self, node):
        return self.extract_node_struct(node, self._param_struct)

    def insert_node_output(self, node, data):
        # output structure consists of a list of node outputs,
        #   followed by a list of nested data structures; get the first one
        all_nodes = self._data_struct[self._data_struct.dtype.names[0]]

        # Get the index into the array of all nodes
        index = self._composition._get_node_index(node)
        value = all_nodes[all_nodes.dtype.names[index]]
        np.copyto(value, np.asarray(data, dtype=value.dtype))

    def _get_input_struct(self, inputs):
        # Either node or composition execute.

        # Read provided input data and parse into an array (generator)
        data = self._composition._build_variable_for_input_CIM(inputs)

        np_input = np.asarray(_tupleize(data), dtype=self._bin_func.np_arg_dtypes[2].base)
        np_input = np_input.reshape(self._bin_func.np_arg_dtypes[2].shape)

        if "stat" in self._debug_env:
            print("Input struct size:", _pretty_size(np_input.nbytes), "for", self._composition.name)

        return np_input

    def freeze_values(self):
        self.__frozen_values = self._data_struct.copy()

    def execute_node(self, node, inputs=None):
        # We need to reconstruct the input dictionary here if it was not provided.
        # This happens during node execution of nested compositions.
        context = self._execution_context

        if inputs is None and node is self._composition.input_CIM:

            port_inputs = {origin_port:[proj.parameters.value._get(context) for proj in p[0].path_afferents] for (origin_port, p) in self._composition.input_CIM_ports.items()}
            inputs = {}
            for p, v in port_inputs.items():
                data = inputs.setdefault(p.owner, [0] * len(p.owner.input_ports))
                index = p.owner.input_ports.index(p)
                data[index] = v[0]

        assert inputs is not None or node is not self._composition.input_CIM

        # Set bin node to make sure self._*struct works as expected
        self._set_bin_node(node)

        # Numpy doesn't allow to pass NULL to the called function.
        # Create and pass a dummy buffer filled with NaN instead.
        if inputs is not None:
            inputs = self._get_input_struct(inputs)
        else:
            inputs = self._bin_func.np_buffer_for_arg(2)

        # Nodes other than input_CIM/parameter_CIM take inputs from projections
        # and need frozen values available
        if node is not self._composition.input_CIM and node is not self._composition.parameter_CIM:
            assert self.__frozen_values is not None
            data_in = self.__frozen_values
        else:
            # The ndarray argument check doesn't allow None for null so just provide
            # the same structure as outputs.
            data_in = self._data_struct

        self._bin_func(self._state_struct, self._param_struct, inputs, data_in, self._data_struct)

        if "comp_node_debug" in self._debug_env:
            print("RAN: {}. State: {}".format(node, self.extract_node_state(node)))
            print("RAN: {}. Params: {}".format(node, self.extract_node_params(node)))
            print("RAN: {}. Results: {}".format(node, self.extract_node_output(node)))

        node._propagate_most_recent_context(context)

    @property
    def _bin_exec_func(self):
        if self.__bin_exec_func is None:
            self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(self._composition, tags=self.__tags)

        return self.__bin_exec_func

    def execute(self, inputs):
        # NOTE: Make sure that input struct generation is inlined.
        # We need the binary function to be setup for it to work correctly.
        self._bin_exec_func(self._state_struct,
                            self._param_struct,
                            self._get_input_struct(inputs),
                            self._data_struct,
                            self._conditions)

    # Methods used to accelerate "Run"
    def _get_run_input_struct(self, inputs, num_input_sets, arg=3):
        # Callers that override input arg, should ensure that _bin_func is not None
        bin_f = self._bin_run_func if arg == 3 else self._bin_func

        input_type = bin_f.byref_arg_types[arg]
        c_input_type = (input_type * num_input_sets)

        # Extract input for each trial and execution id
        run_inputs = (([x] for x in self._composition._build_variable_for_input_CIM({k:v[i] for k,v in inputs.items()})) for i in range(num_input_sets))
        c_inputs = c_input_type(*_tupleize(run_inputs))
        if "stat" in self._debug_env:
            print("Instantiated struct: input ( size:" ,
                  _pretty_size(ctypes.sizeof(c_inputs)),
                  ")",
                  "for",
                  self._obj.name)

        return c_inputs

    def _get_generator_run_input_struct(self, inputs, runs):
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
            self.__bin_run_func = pnlvm.LLVMBinaryFunction.from_obj(self._composition,
                                                                    tags=self.__tags.union({"run"}),
                                                                    ctype_ptr_args=(3,),
                                                                    dynamic_size_args=(4,))

        return self.__bin_run_func

    def _prepare_run(self, inputs, runs, num_input_sets):

        # Create input buffer
        if isgenerator(inputs):
            inputs, runs = self._get_generator_run_input_struct(inputs, runs)
            assert num_input_sets == 0 or num_input_sets == sys.maxsize
            num_input_sets = len(inputs)
        else:
            inputs = self._get_run_input_struct(inputs, num_input_sets)

        # Create output buffer
        outputs = self._bin_func.np_buffer_for_arg(4, extra_dimensions=(runs,))
        assert ctypes.sizeof(self._bin_run_func.byref_arg_types[4]) * runs == outputs.nbytes

        if "stat" in self._debug_env:
            print("Output struct size:", _pretty_size(outputs.nbytes), "for", self._composition.name)

        runs_count = np.asarray(runs, dtype=np.uint32).copy()
        input_count = np.asarray(num_input_sets, dtype=np.uint32)

        return inputs, outputs, runs_count, input_count

    def run(self, inputs, runs, num_input_sets):
        ct_inputs, outputs, runs_count, input_count = self._prepare_run(inputs, runs, num_input_sets)

        self._bin_run_func(self._state_struct,
                           self._param_struct,
                           self._data_struct,
                           ct_inputs,
                           outputs,
                           runs_count,
                           input_count)

        # Extract only #trials elements in case the run exited early
        assert runs_count <= runs, "Composition ran more times than allowed!"
        return self._get_indexable(outputs[0:runs_count])

    def cuda_run(self, inputs, runs, num_input_sets):
        ct_inputs, outputs, runs_count, input_count = self._prepare_run(inputs, runs, num_input_sets)

        self._bin_run_func.cuda_call(self._cuda_state_struct,
                                     self._cuda_param_struct,
                                     self._cuda_data_struct,
                                     jit_engine.pycuda.driver.In(np.ctypeslib.as_array(ct_inputs)),
                                     jit_engine.pycuda.driver.Out(outputs),
                                     jit_engine.pycuda.driver.InOut(runs_count),
                                     jit_engine.pycuda.driver.In(input_count))

        # Extract only #trials elements in case the run exited early
        assert runs_count <= runs, "Composition ran more times than allowed: {}".format(runs)
        return self._get_indexable(outputs[0:runs_count])

    def _prepare_evaluate(self, inputs, num_input_sets, num_evaluations, all_results:bool):
        ocm = self._composition.controller

        eval_type = "evaluate_type_all_results" if all_results else "evaluate_type_objective"
        tags = {"evaluate", "alloc_range", eval_type}
        bin_func = pnlvm.LLVMBinaryFunction.from_obj(ocm, tags=frozenset(tags), ctype_ptr_args=(5,), dynamic_size_args=(4,))
        self.__bin_func = bin_func

        # There are 8 arguments to evaluate_alloc_range:
        # comp_param, comp_state, from, to, results, input, comp_data, num_inputs
        # all but #4 are shared
        assert len(bin_func.byref_arg_types) == 8

        # Directly initialized structures
        assert ocm.agent_rep is self._composition
        comp_params = self._get_compilation_param('_eval_param', '_get_param_initializer', 0)
        comp_state = self._get_compilation_param('_eval_state', '_get_state_initializer', 1)
        comp_data = self._get_compilation_param('_eval_data', '_get_data_initializer', 6)

        # Construct input variable, the 5th parameter of the evaluate function
        ct_inputs = self._get_run_input_struct(inputs, num_input_sets, 5)

        # Output buffer
        extra_dims = (num_evaluations,)
        if all_results:
            num_trials = ocm.parameters.num_trials_per_estimate.get(self._execution_context)
            assert num_trials is not None
            extra_dims = extra_dims + (num_trials,)

        outputs = self._bin_func.np_buffer_for_arg(4, extra_dimensions=extra_dims)

        num_inputs = np.asarray(num_input_sets, dtype=np.uint32)
        if "stat" in self._debug_env:
            print("Evaluate result struct type size:",
                  _pretty_size(ctypes.sizeof(outputs.nbytes)),
                  "( evaluations:", num_evaluations, "element size:", ctypes.sizeof(out_el_ty), ")",
                  "for", self._obj.name)

        return comp_params, comp_state, comp_data, ct_inputs, outputs, num_inputs

    def cuda_evaluate(self, inputs, num_input_sets, num_evaluations, all_results:bool=False):
        comp_params, comp_state, comp_data, ct_inputs, results, num_inputs = \
            self._prepare_evaluate(inputs, num_input_sets, num_evaluations, all_results)

        cuda_args = (jit_engine.pycuda.driver.In(comp_params),
                     jit_engine.pycuda.driver.In(comp_state),
                     jit_engine.pycuda.driver.Out(results),                             # results
                     jit_engine.pycuda.driver.In(np.ctypeslib.as_array(ct_inputs)),     # inputs
                     jit_engine.pycuda.driver.In(comp_data),                            # composition data
                     jit_engine.pycuda.driver.In(num_inputs),                           # number of inputs
                    )

        self.__bin_func.cuda_call(*cuda_args, threads=int(num_evaluations))

        return results

    def thread_evaluate(self, inputs, num_input_sets, num_evaluations, all_results:bool=False):
        comp_params, comp_state, comp_data, ct_inputs, outputs, num_inputs = \
            self._prepare_evaluate(inputs, num_input_sets, num_evaluations, all_results)

        jobs = min(os.cpu_count(), num_evaluations)
        evals_per_job = (num_evaluations + jobs - 1) // jobs

        parallel_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:

            # Create input typed cast once, it is the same for every submitted job.
            input_arg = ctypes.cast(ct_inputs, self.__bin_func.c_func.argtypes[5])

            # numpy dynamic args expect only one extra dimension
            output_arg = outputs.reshape(-1, *self.__bin_func.np_arg_dtypes[4].shape)
            assert output_arg.base is outputs

            # There are 8 arguments to evaluate_alloc_range:
            # comp_param, comp_state, from, to, results, input, comp_data, input length
            results = [ex.submit(self.__bin_func,
                                 comp_params,
                                 comp_state,
                                 int(i * evals_per_job),
                                 min((i + 1) * evals_per_job, num_evaluations),
                                 output_arg,
                                 input_arg,
                                 comp_data,
                                 num_inputs)
                       for i in range(jobs)]

        parallel_stop = time.time()
        if "time_stat" in self._debug_env:
            print("Time to run {} executions of '{}' in {} threads: {}".format(
                      num_evaluations, self.__bin_func.name, jobs,
                      parallel_stop - parallel_start))


        exceptions = [r.exception() for r in results]
        assert all(e is None for e in exceptions), "Not all jobs finished sucessfully: {}".format(exceptions)

        return outputs
