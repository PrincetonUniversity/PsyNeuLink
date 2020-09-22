# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Binary Execution Wrappers **************************************************************

from psyneulink.core.globals.context import Context

from collections import Counter
import copy
import ctypes
import numpy as np
from inspect import isgenerator
import itertools
import sys


from psyneulink.core import llvm as pnlvm
from . import helpers, jit_engine, builder_context
from .debug import debug_env

__all__ = ['CompExecution', 'FuncExecution', 'MechExecution']


def _convert_ctype_to_python(x):
    if isinstance(x, ctypes.Structure):
        return [_convert_ctype_to_python(getattr(x, field_name)) for field_name, _ in x._fields_]
    if isinstance(x, ctypes.Array):
        return [_convert_ctype_to_python(num) for num in x]
    if isinstance(x, ctypes.c_double):
        return x.value
    if isinstance(x, (float, int)):
        return x

    assert False, "Don't know how to convert: {}".format(x)


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


class CUDAExecution:
    def __init__(self, buffers=['param_struct', 'state_struct', 'out']):
        for b in buffers:
            setattr(self, "_buffer_cuda_" + b, None)
        self._uploaded_bytes = Counter()
        self._downloaded_bytes = Counter()
        self.__debug_env = debug_env

    def __del__(self):
        if "cuda_data" in self.__debug_env:
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

    def _get_ctype_bytes(self, data):
        # Return dummy buffer. CUDA does not handle 0 size well.
        if ctypes.sizeof(data) == 0:
            return bytearray(b'aaaa')
        return bytearray(data)

    def upload_ctype(self, data, name='other'):
        self._uploaded_bytes[name] += ctypes.sizeof(data)
        return jit_engine.pycuda.driver.to_device(self._get_ctype_bytes(data))

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
            size = ctypes.sizeof(self._vo_ty)
            self._buffer_cuda_out = jit_engine.pycuda.driver.mem_alloc(size)
        return self._buffer_cuda_out

    def cuda_execute(self, variable):
        # Create input parameter
        new_var = np.asfarray(variable)
        data_in = jit_engine.pycuda.driver.In(new_var)
        self._uploaded_bytes['input'] += new_var.nbytes

        self._bin_func.cuda_call(self._cuda_param_struct,
                                 self._cuda_state_struct,
                                 data_in, self._cuda_out,
                                 threads=len(self._execution_contexts))

        # Copy the result from the device
        ct_res = self.download_ctype(self._cuda_out, self._vo_ty, 'result')
        return _convert_ctype_to_python(ct_res)


class FuncExecution(CUDAExecution):

    def __init__(self, component, execution_ids=[None], *, tags=frozenset()):
        super().__init__()
        self._bin_func = pnlvm.LLVMBinaryFunction.from_obj(component, tags=tags)
        self._execution_contexts = [
            Context(execution_id=eid) for eid in execution_ids
        ]
        self._component = component

        par_struct_ty, ctx_struct_ty, vi_ty, vo_ty = self._bin_func.byref_arg_types

        if len(execution_ids) > 1:
            self._bin_multirun = self._bin_func.get_multi_run()
            self._ct_len = ctypes.c_int(len(execution_ids))
            vo_ty = vo_ty * len(execution_ids)
            vi_ty = vi_ty * len(execution_ids)

            self.__param_struct = None
            self.__state_struct = None

        self._vo_ty = vo_ty
        self._ct_vo = vo_ty()
        self._vi_ty = vi_ty

    def _get_compilation_param(self, name, initializer, arg, context):
        param = getattr(self._component._compilation_data, name)
        struct = param._get(context)
        if struct is None:
            initializer = getattr(self._component, initializer)(context)
            struct_ty = self._bin_func.byref_arg_types[arg]
            struct = struct_ty(*initializer)
            param._set(struct, context=context)

        return struct

    def _get_multirun_struct(self, arg, init):
        struct_ty = self._bin_multirun.byref_arg_types[arg] * len(self._execution_contexts)
        initializer = (getattr(self._component, init)(ex_id) for ex_id in self._execution_contexts)
        return struct_ty(*initializer)

    @property
    def _param_struct(self):
        if len(self._execution_contexts) > 1:
            if self.__param_struct is None:
                self.__param_struct = self._get_multirun_struct(0, '_get_param_initializer')
            return self.__param_struct

        return self._get_compilation_param('parameter_struct', '_get_param_initializer', 0, self._execution_contexts[0])

    @property
    def _state_struct(self):
        if len(self._execution_contexts) > 1:
            if self.__state_struct is None:
                self.__state_struct = self._get_multirun_struct(1, '_get_state_initializer')
            return self.__state_struct

        return self._get_compilation_param('state_struct', '_get_state_initializer', 1, self._execution_contexts[0])

    def execute(self, variable):
        new_variable = np.asfarray(variable)

        if len(self._execution_contexts) > 1:
            # wrap_call casts the arguments so we only need contiguous data
            # layout
            ct_vi = np.ctypeslib.as_ctypes(new_variable)
            self._bin_multirun.wrap_call(self._param_struct,
                                         self._state_struct,
                                         ct_vi, self._ct_vo, self._ct_len)
        else:
            ct_vi = new_variable.ctypes.data_as(ctypes.POINTER(self._vi_ty))
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
        new_var = [np.atleast_2d(x) for x in np.atleast_1d(variable)]
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
        self.__debug_env = debug_env
        self.__frozen_vals = None
        self.__tags = frozenset(additional_tags)

        # TODO: Consolidate these
        if len(execution_ids) > 1:
            self.__state_struct = None
            self.__param_struct = None
            self.__data_struct = None
            self.__conds = None
            self._ct_len = ctypes.c_int(len(execution_ids))

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
        wrapper = builder_context.LLVMBuilderContext.get_global().get_node_wrapper(self._composition, node)
        self.__bin_func = pnlvm.LLVMBinaryFunction.from_obj(
            wrapper, tags=self.__tags.union({"node_wrapper"}))

    @property
    def _conditions(self):
        if len(self._execution_contexts) > 1:
            if self.__conds is None:
                cond_type = self._bin_func_multirun.byref_arg_types[4] * len(self._execution_contexts)
                gen = helpers.ConditionGenerator(None, self._composition)
                cond_initializer = (gen.get_condition_initializer() for _ in self._execution_contexts)
                self.__conds = cond_type(*cond_initializer)
            return self.__conds

        conds = self._composition._compilation_data.scheduler_conditions._get(self._execution_contexts[0])
        if conds is None:
            cond_type = self._bin_func.byref_arg_types[4]
            gen = helpers.ConditionGenerator(None, self._composition)
            cond_initializer = gen.get_condition_initializer()
            conds = cond_type(*cond_initializer)
            self._composition._compilation_data.scheduler_conditions._set(conds, context=self._execution_contexts[0])
        return conds

    def _get_compilation_param(self, name, initializer, arg, context):
        param = getattr(self._composition._compilation_data, name)
        struct = param._get(context)
        if struct is None:
            initializer = getattr(self._composition, initializer)(context)
            struct_ty = self._bin_func.byref_arg_types[arg]
            struct = struct_ty(*initializer)
            param._set(struct, context=context)

        return struct

    def _get_multirun_struct(self, arg, init):
        struct_ty = self._bin_func_multirun.byref_arg_types[arg] * len(self._execution_contexts)
        initializer = (getattr(self._composition, init)(ex) for ex in self._execution_contexts)
        return struct_ty(*initializer)

    @property
    def _param_struct(self):
        if len(self._execution_contexts) > 1:
            if self.__param_struct is None:
                self.__param_struct = self._get_multirun_struct(1, '_get_param_initializer')
            return self.__param_struct

        return self._get_compilation_param('parameter_struct', '_get_param_initializer', 1, self._execution_contexts[0])

    def _copy_params_to_pnl(self, context=None, component=None, params=None):
        # need to special case compositions
        from psyneulink.core.compositions import Composition
        from psyneulink.core.components.projections.pathway import MappingProjection

        if component is None:
            component = self._composition

        if params is None:
            assert component == self._composition
            params = self._param_struct

        if isinstance(component, Composition):
            # first handle all inner projections
            params_projections_list = getattr(params, params._fields_[1][0])
            for idx, projection in enumerate(component._inner_projections):
                projection_params = getattr(params_projections_list, params_projections_list._fields_[idx][0])
                self._copy_params_to_pnl(context=context, component=projection, params=projection_params)

            # now recurse on all nodes
            params_node_list = getattr(params, params._fields_[0][0])
            for idx, node in enumerate(component._all_nodes):
                node_params = getattr(params_node_list, params_node_list._fields_[idx][0])
                self._copy_params_to_pnl(context=context, component=node, params=node_params)
        elif isinstance(component, MappingProjection):
            # we copy all ids back
            for idx, attribute in enumerate(component.llvm_param_ids):
                to_set = getattr(component.parameters, attribute)
                parameter_ctype = getattr(params, params._fields_[idx][0])
                value = _convert_ctype_to_python(parameter_ctype)
                if attribute == 'matrix':
                    # special case since we have to unflatten matrix
                    # FIXME: this seems to break something when generalized for all attributes
                    value = np.array(value).reshape(component.matrix.shape)
                    to_set._set(value, context=context)

    @property
    def _state_struct(self):
        if len(self._execution_contexts) > 1:
            if self.__state_struct is None:
                self.__state_struct = self._get_multirun_struct(0, '_get_state_initializer')
            return self.__state_struct

        return self._get_compilation_param('state_struct', '_get_state_initializer', 0, self._execution_contexts[0])

    @property
    def _data_struct(self):
        # Run wrapper changed argument order
        arg = 2 if self._bin_func is self.__bin_run_func else 3

        if len(self._execution_contexts) > 1:
            if self.__data_struct is None:
                self.__data_struct = self._get_multirun_struct(arg, '_get_data_initializer')
            return self.__data_struct

        return self._get_compilation_param('data_struct', '_get_data_initializer', arg, self._execution_contexts[0])

    @_data_struct.setter
    def _data_struct(self, data_struct):
        if len(self._execution_contexts) > 1:
            self.__data_struct = data_struct
        else:
            self._composition._compilation_data.data_struct._set(data_struct, context=self._execution_contexts[0])

    def _extract_node_struct(self, node, data):
        # context structure consists of a list of node contexts,
        #   followed by a list of projection contexts; get the first one
        # parameter structure consists of a list of node parameters,
        #   followed by a list of projection parameters; get the first one
        # output structure consists of a list of node outputs,
        #   followed by a list of nested data structures; get the first one
        field = data._fields_[0][0]
        res_struct = getattr(data, field)
        index = self._composition._get_node_index(node)
        field = res_struct._fields_[index][0]
        res_struct = getattr(res_struct, field)
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

        return c_input(*_tupleize(input_data))

    def freeze_values(self):
        self.__frozen_vals = copy.deepcopy(self._data_struct)

    def execute_node(self, node, inputs=None):
        # We need to reconstruct the input dictionary here if it was not provided.
        # This happens during node execution of nested compositions.
        assert len(self._execution_contexts) == 1
        if inputs is None and node is self._composition.input_CIM:
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

        if "comp_node_debug" in self.__debug_env:
            print("RAN: {}. CTX: {}".format(node, self.extract_node_state(node)))
            print("RAN: {}. Params: {}".format(node, self.extract_node_params(node)))
            print("RAN: {}. Results: {}".format(node, self.extract_node_output(node)))

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
        else:
            inputs = self._get_run_input_struct(inputs, num_input_sets)

        ct_vo = self._bin_run_func.byref_arg_types[4] * runs
        if len(self._execution_contexts) > 1:
            ct_vo = ct_vo * len(self._execution_contexts)
        outputs = ct_vo()
        runs_count = ctypes.c_int(runs)
        input_count = ctypes.c_int(num_input_sets)
        if len(self._execution_contexts) > 1:
            self._bin_run_multi_func.wrap_call(self._state_struct, self._param_struct,
                                               self._data_struct, inputs, outputs,
                                               runs_count, input_count, self._ct_len)
        else:
            self._bin_run_func.wrap_call(self._state_struct, self._param_struct,
                                         self._data_struct, inputs, outputs,
                                         runs_count, input_count)
        return _convert_ctype_to_python(outputs)

    def cuda_run(self, inputs, runs, num_input_sets):
        # Create input buffer
        if isgenerator(inputs):
            inputs, runs = self._get_generator_run_input_struct(inputs, runs)
        else:
            inputs = self._get_run_input_struct(inputs, num_input_sets)
        data_in = self.upload_ctype(inputs, 'input')

        # Create output buffer
        output_type = (self._bin_run_func.byref_arg_types[4] * runs)
        if len(self._execution_contexts) > 1:
            output_type = output_type * len(self._execution_contexts)
        output_size = ctypes.sizeof(output_type)
        data_out = jit_engine.pycuda.driver.mem_alloc(output_size)

        runs_count = jit_engine.pycuda.driver.In(np.int32(runs))
        input_count = jit_engine.pycuda.driver.In(np.int32(num_input_sets))
        # runs_count + input_count
        self._uploaded_bytes['input'] += 8

        self._bin_run_func.cuda_call(self._cuda_state_struct,
                                     self._cuda_param_struct,
                                     self._cuda_data_struct,
                                     data_in, data_out, runs_count, input_count,
                                     threads=len(self._execution_contexts))

        # Copy the data struct from the device
        ct_out = self.download_ctype(data_out, output_type, 'result')
        return _convert_ctype_to_python(ct_out)

    def cuda_evaluate(self, variable, search_space):
        ocm = self._composition.controller
        assert len(self._execution_contexts) == 1
        context = self._execution_contexts[0]

        bin_func = pnlvm.LLVMBinaryFunction.from_obj(ocm, tags=frozenset({"evaluate"}))
        self.__bin_func = bin_func
        assert len(bin_func.byref_arg_types) == 6

        # There are 6 arguments to evaluate:
        # comp_param, comp_state, allocations, results, output, input, comp_data
        # all but #2 and #3 are shared
        ct_comp_param = bin_func.byref_arg_types[0](*ocm.agent_rep._get_param_initializer(context))
        ct_comp_state = bin_func.byref_arg_types[1](*ocm.agent_rep._get_state_initializer(context))
        # Make sure the dtype matches _gen_llvm_evaluate_function
        allocations = np.asfarray(np.atleast_2d([*itertools.product(*search_space)]))
        ct_allocations = allocations.ctypes.data_as(ctypes.POINTER(bin_func.byref_arg_types[2] * len(allocations)))
        out_ty = bin_func.byref_arg_types[3] * len(allocations)
        ct_in = variable.ctypes.data_as(ctypes.POINTER(bin_func.byref_arg_types[4]))

        ct_comp_data = bin_func.byref_arg_types[5](*ocm.agent_rep._get_data_initializer(context))

        cuda_args = (self.upload_ctype(ct_comp_param, 'params'),
                     self.upload_ctype(ct_comp_state, 'state'),
                     self.upload_ctype(ct_allocations.contents, 'input'),
                     jit_engine.pycuda.driver.mem_alloc(ctypes.sizeof(out_ty)),
                     self.upload_ctype(ct_in.contents, 'input'),
                     self.upload_ctype(ct_comp_data, 'data'),
                    )

        bin_func.cuda_call(*cuda_args, threads=len(allocations))
        ct_results = self.download_ctype(cuda_args[3], out_ty, 'result')

        return ct_allocations.contents, ct_results
