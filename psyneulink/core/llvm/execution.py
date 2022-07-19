# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Binary Execution Wrappers **************************************************************

from psyneulink.core.globals.context import Context

from collections import Counter
import concurrent.futures
import copy
import ctypes
import numpy as np
from inspect import isgenerator
import os
import sys
import time


from psyneulink.core import llvm as pnlvm
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
        struct = getattr(self, name, None)
        if struct is None:
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


            if "time_stat" in self._debug_env:
                print("Time to get initializer for struct:", name,
                      "for", self._obj.name, ":", init_end - init_start)
                print("Time to instantiate struct:", name,
                      "for", self._obj.name, ":", struct_end - init_end)
            setattr(self, name, struct)
            if "stat" in self._debug_env:
                print("Instantiated struct:", name, "( size:" ,
                      _pretty_size(ctypes.sizeof(struct_ty)), ")",
                      "for", self._obj.name)

        return struct


class CUDAExecution(Execution):
    def __init__(self, buffers=['param_struct', 'state_struct', 'out']):
        super().__init__()
        for b in buffers:
            setattr(self, "_buffer_cuda_" + b, None)
        self._uploaded_bytes = Counter()
        self._downloaded_bytes = Counter()

    def __del__(self):
        if "stat" in self._debug_env:
            try:
                name = self._bin_func.name
            except AttributeError:
                name = self._composition.name

            for k, v in self._uploaded_bytes.items():
                print("{} CUDA uploaded `{}': {}".format(name, k, _pretty_size(v)))
            if len(self._uploaded_bytes) > 1:
                print("{} CUDA uploaded `total': {}".format(name, _pretty_size(sum(self._uploaded_bytes.values()))))
            for k, v in self._downloaded_bytes.items():
                print("{} CUDA downloaded `{}': {}".format(name, k, _pretty_size(v)))
            if len(self._downloaded_bytes) > 1:
                print("{} CUDA downloaded `total': {}".format(name, _pretty_size(sum(self._downloaded_bytes.values()))))

    @property
    def _bin_func_multirun(self):
        # CUDA uses the same function for single and multi run
        return self._bin_func

    def upload_ctype(self, data, name='other'):
        self._uploaded_bytes[name] += ctypes.sizeof(data)
        if ctypes.sizeof(data) == 0:
            # 0-sized structures fail to upload
            # provide a small device buffer instead
            return jit_engine.pycuda.driver.mem_alloc(4)
        return jit_engine.pycuda.driver.to_device(bytearray(data))

    def download_ctype(self, source, ty, name='other'):
        self._downloaded_bytes[name] += ctypes.sizeof(ty)
        out_buf = bytearray(ctypes.sizeof(ty))
        jit_engine.pycuda.driver.memcpy_dtoh(out_buf, source)
        return ty.from_buffer(out_buf)

    def __get_cuda_buffer(self, struct_name):
        private_attr_name = "_buffer_cuda" + struct_name
        private_attr = getattr(self, private_attr_name)
        if private_attr is None:
            # Set private attribute to a new buffer
            private_attr = self.upload_ctype(getattr(self, struct_name), struct_name)
            setattr(self, private_attr_name, private_attr)

        return private_attr

    @property
    def _cuda_param_struct(self):
        return self.__get_cuda_buffer("_param_struct")

    @property
    def _cuda_state_struct(self):
        return self.__get_cuda_buffer("_state_struct")

    @property
    def _cuda_data_struct(self):
        return self.__get_cuda_buffer("_data_struct")

    @property
    def _cuda_conditions(self):
        return self.__get_cuda_buffer("_conditions")

    @property
    def _cuda_out(self):
        if self._buffer_cuda_out is None:
            size = ctypes.sizeof(self._ct_vo)
            self._buffer_cuda_out = jit_engine.pycuda.driver.mem_alloc(size)
        return self._buffer_cuda_out

    def cuda_execute(self, variable):
        # Create input argument
        new_var = np.asfarray(variable, dtype=self._vi_dty)
        data_in = jit_engine.pycuda.driver.In(new_var)
        self._uploaded_bytes['input'] += new_var.nbytes

        self._bin_func.cuda_call(self._cuda_param_struct,
                                 self._cuda_state_struct,
                                 data_in, self._cuda_out,
                                 threads=len(self._execution_contexts))

        # Copy the result from the device
        ct_res = self.download_ctype(self._cuda_out, type(self._ct_vo), 'result')
        return _convert_ctype_to_python(ct_res)


class FuncExecution(CUDAExecution):

    def __init__(self, component, execution_ids=[None], *, tags=frozenset()):
        super().__init__()
        self._bin_func = pnlvm.LLVMBinaryFunction.from_obj(component, tags=tags)
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
        new_variable = np.asfarray(np.atleast_2d(variable),
                                   dtype=self._vi_dty)

        ct_vi = np.ctypeslib.as_ctypes(new_variable)
        if len(self._execution_contexts) > 1:
            # wrap_call casts the arguments so we only need contiguous data
            # layout
            self._bin_multirun.wrap_call(self._param_struct,
                                         self._state_struct,
                                         ct_vi, self._ct_vo, self._ct_len)
        else:
            self._bin_func(ctypes.byref(self._param_struct),
                           ctypes.byref(self._state_struct),
                           ct_vi, ctypes.byref(self._ct_vo))

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
        wrapper = builder_context.LLVMBuilderContext.get_current().get_node_wrapper(self._composition, node)
        self.__bin_func = pnlvm.LLVMBinaryFunction.from_obj(
            wrapper, tags=self.__tags.union({"node_wrapper"}))

    @property
    def _conditions(self):
        if self.__conds is None:
            gen = helpers.ConditionGenerator(None, self._composition)
            if len(self._execution_contexts) > 1:
                cond_type = self._bin_func_multirun.byref_arg_types[4] * len(self._execution_contexts)
                cond_initializer = (gen.get_condition_initializer() for _ in self._execution_contexts)
            else:
                cond_type = self._bin_func.byref_arg_types[4]
                cond_initializer = gen.get_condition_initializer()

            self.__conds = cond_type(*cond_initializer)
            if "stat" in self._debug_env:
                print("Instantiated condition struct ( size:" ,
                      _pretty_size(ctypes.sizeof(cond_type)), ")",
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

    def _copy_params_to_pnl(self, context=None, component=None, params=None):

        if component is None:
            component = self._composition

        if params is None:
            assert component == self._composition
            params = self._param_struct

        for idx, attribute in enumerate(component.llvm_param_ids):
            if attribute == 'nodes':
                params_node_list = getattr(params, params._fields_[idx][0])
                for node_id, node in enumerate(component._all_nodes):
                    node_params = getattr(params_node_list,
                                          params_node_list._fields_[node_id][0])
                    self._copy_params_to_pnl(context=context, component=node,
                                             params=node_params)
            elif attribute == 'projections':
                params_projection_list = getattr(params, params._fields_[idx][0])
                for proj_id, projection in enumerate(component._inner_projections):
                    projection_params = getattr(params_projection_list,
                                                params_projection_list._fields_[proj_id][0])
                    self._copy_params_to_pnl(context=context,
                                             component=projection,
                                             params=projection_params)
            elif attribute == 'function':
                function_params = getattr(params, params._fields_[idx][0])
                self._copy_params_to_pnl(context=context,
                                         component=component.function,
                                         params=function_params)
            elif attribute == 'matrix':
                pnl_param = component.parameters.matrix
                parameter_ctype = getattr(params, params._fields_[idx][0])
                value = _convert_ctype_to_python(parameter_ctype)
                # Unflatten the matrix
                # FIXME: this seems to break something when generalized for all attributes
                value = np.array(value).reshape(pnl_param._get(context).shape)
                pnl_param._set(value, context=context)

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
        return self.extract_node_struct(node, self._data_struct)

    def extract_node_state(self, node):
        return self.extract_node_struct(node, self._state_struct)

    def extract_node_params(self, node):
        return self.extract_node_struct(node, self._param_struct)

    def insert_node_output(self, node, data):
        my_field_name = self._data_struct._fields_[0][0]
        my_res_struct = getattr(self._data_struct, my_field_name)
        index = self._composition._get_node_index(node)
        node_field_name = my_res_struct._fields_[index][0]
        setattr(my_res_struct, node_field_name, _tupleize(data))

    def _get_input_struct(self, inputs):
        # Either node or composition execute.
        # All execute functions expect inputs to be 3rd param.
        c_input = self._bin_func.byref_arg_types[2]

        # Read provided input data and parse into an array (generator)
        if len(self._execution_contexts) > 1:
            assert len(self._execution_contexts) == len(inputs)
            c_input = c_input * len(self._execution_contexts)
            input_data = (([x] for x in self._composition._build_variable_for_input_CIM(inp)) for inp in inputs)
        else:
            input_data = ([x] for x in self._composition._build_variable_for_input_CIM(inputs))

        if "stat" in self._debug_env:
            print("Input struct size:", _pretty_size(ctypes.sizeof(c_input)),
                  "for", self._composition.name)
        return c_input(*_tupleize(input_data))

    def freeze_values(self):
        self.__frozen_vals = copy.deepcopy(self._data_struct)

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
            inputs = self._get_input_struct(inputs)

        assert inputs is not None or node is not self._composition.input_CIM

        # Freeze output values if this is the first time we need them
        if node is not self._composition.input_CIM and self.__frozen_vals is None:
            self.freeze_values()

        self._bin_func(self._state_struct, self._param_struct,
                       inputs, self.__frozen_vals, self._data_struct)

        if "comp_node_debug" in self._debug_env:
            print("RAN: {}. CTX: {}".format(node, self.extract_node_state(node)))
            print("RAN: {}. Params: {}".format(node, self.extract_node_params(node)))
            print("RAN: {}. Results: {}".format(node, self.extract_node_output(node)))

        node._propagate_most_recent_context(context)

    @property
    def _bin_exec_func(self):
        if self.__bin_exec_func is None:
            self.__bin_exec_func = pnlvm.LLVMBinaryFunction.from_obj(
                self._composition, tags=self.__tags)

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
            self._bin_exec_multi_func.wrap_call(self._state_struct,
                                                self._param_struct,
                                                self._get_input_struct(inputs),
                                                self._data_struct,
                                                self._conditions, self._ct_len)
        else:
            self._bin_exec_func(self._state_struct, self._param_struct,
                                self._get_input_struct(inputs),
                                self._data_struct, self._conditions)

    def cuda_execute(self, inputs):
        # NOTE: Make sure that input struct generation is inlined.
        # We need the binary function to be setup for it to work correctly.
        self._bin_exec_func.cuda_call(self._cuda_state_struct,
                                      self._cuda_param_struct,
                                      self.upload_ctype(self._get_input_struct(inputs), 'input'),
                                      self._cuda_data_struct,
                                      self._cuda_conditions,
                                      threads=len(self._execution_contexts))

        # Copy the data struct from the device
        self._data_struct = self.download_ctype(self._cuda_data_struct, type(self._data_struct), '_data_struct')

    # Methods used to accelerate "Run"

    def _get_run_input_struct(self, inputs, num_input_sets):
        input_type = self._bin_run_func.byref_arg_types[3]
        c_input = (input_type * num_input_sets) * len(self._execution_contexts)
        if len(self._execution_contexts) == 1:
            inputs = [inputs]

        assert len(inputs) == len(self._execution_contexts)
        # Extract input for each trial and execution id
        run_inputs = ((([x] for x in self._composition._build_variable_for_input_CIM({k:v[i] for k,v in inp.items()})) for i in range(num_input_sets)) for inp in inputs)
        return c_input(*_tupleize(run_inputs))

    def _get_generator_run_input_struct(self, inputs, runs):
        assert len(self._execution_contexts) == 1
        # Extract input for each trial
        run_inputs = ((np.atleast_2d(x) for x in self._composition._build_variable_for_input_CIM({k:np.atleast_1d(v) for k,v in inp.items()})) for inp in inputs)
        run_inputs = _tupleize(run_inputs)
        num_input_sets = len(run_inputs)
        runs = num_input_sets if runs == 0 or runs == sys.maxsize else runs
        c_input = self._bin_run_func.byref_arg_types[3] * num_input_sets
        return c_input(*run_inputs), runs

    @property
    def _bin_run_func(self):
        if self.__bin_run_func is None:
            self.__bin_run_func = pnlvm.LLVMBinaryFunction.from_obj(
                self._composition, tags=self.__tags.union({"run"}))

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

        runs_count = ctypes.c_int(runs)
        input_count = ctypes.c_int(num_input_sets)
        if len(self._execution_contexts) > 1:
            self._bin_run_multi_func.wrap_call(self._state_struct, self._param_struct,
                                               self._data_struct, inputs, outputs,
                                               runs_count, input_count, self._ct_len)
            return _convert_ctype_to_python(outputs)
        else:
            self._bin_run_func.wrap_call(self._state_struct, self._param_struct,
                                         self._data_struct, inputs, outputs,
                                         runs_count, input_count)

            # Extract only #trials elements in case the run exited early
            assert runs_count.value <= runs, "Composition ran more times than allowed!"
            return _convert_ctype_to_python(outputs)[0:runs_count.value]

    def cuda_run(self, inputs, runs, num_input_sets):
        # Create input buffer
        if isgenerator(inputs):
            inputs, runs = self._get_generator_run_input_struct(inputs, runs)
            assert num_input_sets == 0 or num_input_sets == sys.maxsize
            num_input_sets = len(inputs)
        else:
            inputs = self._get_run_input_struct(inputs, num_input_sets)
        data_in = self.upload_ctype(inputs, 'input')

        # Create output buffer
        output_type = (self._bin_run_func.byref_arg_types[4] * runs)
        if len(self._execution_contexts) > 1:
            output_type = output_type * len(self._execution_contexts)
        output_size = ctypes.sizeof(output_type)
        data_out = jit_engine.pycuda.driver.mem_alloc(output_size)

        # number of trials argument
        runs_np = np.full(len(self._execution_contexts), runs, dtype=np.int32)
        runs_count = jit_engine.pycuda.driver.InOut(runs_np)
        self._uploaded_bytes['input'] += runs_np.nbytes
        self._downloaded_bytes['input'] += runs_np.nbytes

        # input_count argument
        input_count = jit_engine.pycuda.driver.In(np.int32(num_input_sets))
        self._uploaded_bytes['input'] += 4

        self._bin_run_func.cuda_call(self._cuda_state_struct,
                                     self._cuda_param_struct,
                                     self._cuda_data_struct,
                                     data_in, data_out, runs_count, input_count,
                                     threads=len(self._execution_contexts))

        # Copy the data struct from the device
        ct_out = self.download_ctype(data_out, output_type, 'result')
        if len(self._execution_contexts) > 1:
            return _convert_ctype_to_python(ct_out)
        else:
            # Extract only #trials elements in case the run exited early
            assert runs_np[0] <= runs, "Composition ran more times than allowed!"
            return _convert_ctype_to_python(ct_out)[0:runs_np[0]]

    def _prepare_evaluate(self, variable, num_evaluations):
        ocm = self._composition.controller
        assert len(self._execution_contexts) == 1

        bin_func = pnlvm.LLVMBinaryFunction.from_obj(ocm, tags=frozenset({"evaluate", "alloc_range"}))
        self.__bin_func = bin_func

        # There are 7 arguments to evaluate_alloc_range:
        # comp_param, comp_state, from, to, results, input, comp_data
        # all but #4 are shared
        assert len(bin_func.byref_arg_types) == 7

        # Directly initialized structures
        assert ocm.agent_rep is self._composition
        ct_comp_param = self._get_compilation_param('_eval_param', '_get_param_initializer', 0)
        ct_comp_state = self._get_compilation_param('_eval_state', '_get_state_initializer', 1)
        ct_comp_data = self._get_compilation_param('_eval_data', '_get_data_initializer', 6)

        # Construct input variable
        var_dty = _element_dtype(bin_func.byref_arg_types[5])
        converted_variable = np.concatenate(variable, dtype=var_dty)

        # Output ctype
        out_ty = bin_func.byref_arg_types[4] * num_evaluations

        # return variable as numpy array. pycuda can use it directly
        return ct_comp_param, ct_comp_state, ct_comp_data, converted_variable, out_ty

    def cuda_evaluate(self, variable, num_evaluations):
        ct_comp_param, ct_comp_state, ct_comp_data, converted_variable, out_ty = \
            self._prepare_evaluate(variable, num_evaluations)
        self._uploaded_bytes['input'] += converted_variable.nbytes

        # Output is allocated on device, but we need the ctype (out_ty).
        cuda_args = (self.upload_ctype(ct_comp_param, 'params'),
                     self.upload_ctype(ct_comp_state, 'state'),
                     jit_engine.pycuda.driver.mem_alloc(ctypes.sizeof(out_ty)),
                     jit_engine.pycuda.driver.In(converted_variable),
                     self.upload_ctype(ct_comp_data, 'data'),
                    )

        self.__bin_func.cuda_call(*cuda_args, threads=int(num_evaluations))
        ct_results = self.download_ctype(cuda_args[2], out_ty, 'result')

        return ct_results

    def thread_evaluate(self, variable, num_evaluations):
        ct_param, ct_state, ct_data, converted_variale, out_ty = \
            self._prepare_evaluate(variable, num_evaluations)

        ct_results = out_ty()
        ct_variable = converted_variale.ctypes.data_as(self.__bin_func.c_func.argtypes[5])
        jobs = min(os.cpu_count(), num_evaluations)
        evals_per_job = (num_evaluations + jobs - 1) // jobs

        parallel_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            # There are 7 arguments to evaluate_alloc_range:
            # comp_param, comp_state, from, to, results, input, comp_data
            results = [ex.submit(self.__bin_func, ct_param, ct_state,
                                 int(i * evals_per_job),
                                 min((i + 1) * evals_per_job, num_evaluations),
                                 ct_results, ct_variable, ct_data)
                       for i in range(jobs)]

        parallel_stop = time.time()
        if "time_stat" in self._debug_env:
            print("Time to run {} executions of '{}' in {} threads: {}".format(
                      num_evaluations, self.__bin_func.name, jobs,
                      parallel_stop - parallel_start))


        exceptions = [r.exception() for r in results]
        assert all(e is None for e in exceptions), "Not all jobs finished sucessfully: {}".format(exceptions)

        return ct_results
