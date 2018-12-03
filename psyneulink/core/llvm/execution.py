# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Binary Execution Wrappers **************************************************************

from psyneulink.core.globals.utilities import CNodeRole

import copy, ctypes
from collections import defaultdict
import numpy as np

from .builder_context import *
from . import helpers, jit_engine

__all__ = ['CompExecution', 'FuncExecution', 'MechExecution']

def _convert_ctype_to_python(x):
    if isinstance(x, ctypes.Structure):
        return [_convert_ctype_to_python(getattr(x, field_name)) for field_name, _ in x._fields_]
    if isinstance(x, ctypes.Array):
        return [_convert_ctype_to_python(num) for num in x]
    if isinstance(x, ctypes.c_double):
        return x.value
    if isinstance(x, float):
        return x

    print(x)
    assert False

def _tupleize(x):
    try:
        return tuple(_tupleize(y) for y in x)
    except TypeError:
        return x if x is not None else tuple()


class CUDAExecution:
    def __init__(self, buffers=['param_struct', 'context_struct']):
        for b in buffers:
            setattr(self, "_buffer_cuda_" + b, None)
        self.__cuda_out_buf = None

    def _get_buffer(self, data):
        # Return dummy buffer. CUDA does not handle 0 size well.
        if ctypes.sizeof(data) == 0:
            return bytearray(b'aaaa')
        return bytearray(data)

    def _get_device_buffer(self, data):
        return jit_engine.pycuda.driver.to_device(self._get_buffer(data))

    def __getattr__(self, attribute):
        if not attribute.startswith("_cuda"):
            return getattr(super(), attribute)

        private_attr = "_buffer" + attribute
        if getattr(self, private_attr) is None:
            new_buffer = self._get_device_buffer(getattr(self, attribute[5:]))
            setattr(self, private_attr, new_buffer)

        return getattr(self, private_attr)

    @property
    def _cuda_out_buf(self):
        if self.__cuda_out_buf is None:
            vo_ty = self._bin_func.byref_arg_types[3]
            size = ctypes.sizeof(vo_ty)
            self.__cuda_out_buf = jit_engine.pycuda.driver.mem_alloc(size)
        return self.__cuda_out_buf

    def cuda_execute(self, variable):
        # Create input parameter
        new_var = np.asfarray(variable)
        data_in = jit_engine.pycuda.driver.In(new_var)

        self._bin_func.cuda_call(self._cuda_param_struct,
                                 self._cuda_context_struct,
                                 data_in, self._cuda_out_buf)

        # Copy the result from the device
        vo_ty = self._bin_func.byref_arg_types[3]
        out_buf = bytearray(ctypes.sizeof(vo_ty))
        jit_engine.pycuda.driver.memcpy_dtoh(out_buf, self._cuda_out_buf)
        ct_res = vo_ty.from_buffer(out_buf)
        return _convert_ctype_to_python(ct_res)


class FuncExecution(CUDAExecution):

    def __init__(self, component, execution_ids=[None]):
        super().__init__()
        self._bin_func = component._llvmBinFunction
        execution_id = execution_ids[0]

        par_struct_ty, context_struct_ty, _, _ = self._bin_func.byref_arg_types

        self._param_struct = par_struct_ty(*component._get_param_initializer(execution_id))

        self._context_struct = context_struct_ty(*component._get_context_initializer(execution_id))

    def execute(self, variable):
        new_var = np.asfarray(variable)
        _, _ , vi_ty, vo_ty = self._bin_func.byref_arg_types
        ct_vi = new_var.ctypes.data_as(ctypes.POINTER(vi_ty))
        ct_vo = vo_ty()

        self._bin_func(ctypes.byref(self._param_struct),
                       ctypes.byref(self._context_struct),
                       ct_vi, ctypes.byref(ct_vo))

        return _convert_ctype_to_python(ct_vo)


class MechExecution(FuncExecution):

    def __init__(self, mechanism, execution_ids = [None]):
        self._mechanism = mechanism
        super().__init__(mechanism, execution_ids)

        if mechanism._nv_state is not None:
            # TODO: This should consider execution_id
            self._context_struct = mechanism._nv_state

    def execute(self, variable):
        # convert to 3d. we always assume that:
        # a) the input is vector of input states
        # b) input states take vector of projection outputs
        # c) projection output is a vector (even 1 element vector)
        new_var = np.asfarray([np.atleast_2d(x) for x in variable])
        res = super().execute(new_var)

        # store updated context
        # TODO: This should consider execution_id
        self._mechanism._nv_state = self._context_struct
        return res


class CompExecution(CUDAExecution):

    def __init__(self, composition, execution_id):
        super().__init__(buffers=['context_struct', 'param_struct', 'data_struct', 'conditions'])
        self._composition = composition
        self.__frozen_vals = None
        self.__conds = None

        # At least the input_CIM wrapper should be generated
        with LLVMBuilderContext() as ctx:
            input_cim_fn_name = composition._get_node_wrapper(composition.input_CIM)
            input_cim_fn = ctx.get_llvm_function(input_cim_fn_name)

        # Context
        c_context = _convert_llvm_ir_to_ctype(input_cim_fn.args[0].type.pointee)
        self._context_struct = c_context(*composition._get_context_initializer(execution_id))

        # Params
        c_param = _convert_llvm_ir_to_ctype(input_cim_fn.args[1].type.pointee)
        self._param_struct = c_param(*self._composition._get_param_initializer(execution_id))
        # Data
        c_data = _convert_llvm_ir_to_ctype(input_cim_fn.args[3].type.pointee)
        self._data_struct = c_data(*self._composition._get_data_initializer(execution_id))


    @property
    def _conditions(self):
        if self.__conds is None:
            bin_exec = self._composition._get_bin_execution()
            gen = helpers.ConditionGenerator(None, self._composition)
            self.__conds = bin_exec.byref_arg_types[4](*gen.get_condition_initializer())
        return self.__conds

    def extract_frozen_node_output(self, node):
        return self.extract_node_output(node, self.__frozen_vals)

    def extract_node_output(self, node, data = None):
        data = self._data_struct if data == None else data
        field = self._data_struct._fields_[0][0]
        res_struct = getattr(self._data_struct, field)
        index = self._composition._get_node_index(node)
        field = res_struct._fields_[index][0]
        res_struct = getattr(res_struct, field)
        return _convert_ctype_to_python(res_struct)

    def insert_node_output(self, node, data):
        my_field_name = self._data_struct._fields_[0][0]
        my_res_struct = getattr(self._data_struct, my_field_name)
        index = self._composition._get_node_index(node)
        node_field_name = my_res_struct._fields_[index][0]
        setattr(my_res_struct, node_field_name, _tupleize(data))

    def _get_input_struct(self, inputs):
        origins = self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN)
        # Read provided input data and separate each input state
        input_data = ([x] for m in origins for x in inputs[m])

        # Either node execute or composition execute, either way the
        # input_CIM should be ready
        bin_input_node = self._composition._get_bin_mechanism(self._composition.input_CIM)
        c_input = bin_input_node.byref_arg_types[2]
        return c_input(*_tupleize(input_data))

    def _get_run_input_struct(self, inputs, num_input_sets):
        origins = self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN)
        run_inputs = []
        # Extract inputs for each trial
        for i in range(num_input_sets):
            run_inputs.append([])
            for m in origins:
                run_inputs[i] += [[v] for v in inputs[m][i]]

        input_type = self._composition._get_bin_run().byref_arg_types[3]
        c_input = input_type * num_input_sets
        return c_input(*_tupleize(run_inputs))

    def freeze_values(self):
        self.__frozen_vals = copy.deepcopy(self._data_struct)

    def execute_node(self, node, inputs = None, execution_id=None):
        # We need to reconstruct the inputs here if they were not provided.
        # This happens during node execution of nested compositions.
        if inputs is None and node is self._composition.input_CIM:
            # This assumes origin mechanisms are in the same order as
            # CIM input states
            origins = (n for n in self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN) for istate in n.input_states)
            input_data = ([proj.parameters.value.get(execution_id) for proj in state.all_afferents] for state in node.input_states)
            inputs = defaultdict(list)
            for n, d in zip(origins, input_data):
                inputs[n].append(d[0])

        if inputs is not None:
            inputs = self._get_input_struct(inputs)

        assert inputs is not None or node is not self._composition.input_CIM

        assert node in self._composition._all_nodes
        bin_node = self._composition._get_bin_mechanism(node)
        bin_node.wrap_call(self._context_struct, self._param_struct,
                           inputs, self.__frozen_vals, self._data_struct)

    def execute(self, inputs):
        inputs = self._get_input_struct(inputs)
        bin_exec = self._composition._get_bin_execution()
        bin_exec.wrap_call(self._context_struct, self._param_struct,
                           inputs, self._data_struct, self._conditions)

    def run(self, inputs, runs, num_input_sets):
        bin_run = self._composition._get_bin_run()
        inputs = self._get_run_input_struct(inputs, num_input_sets)
        outputs = (bin_run.byref_arg_types[4] * runs)()
        runs_count = ctypes.c_int(runs)
        input_count = ctypes.c_int(num_input_sets)
        bin_run.wrap_call(self._context_struct, self._param_struct,
                          self._data_struct, inputs, outputs, runs_count,
                          input_count)
        return _convert_ctype_to_python(outputs)

    def cuda_execute(self, inputs):
        bin_exec = self._composition._get_bin_execution()
        # Create input buffer
        inputs = self._get_input_struct(inputs)
        input_data = bytearray(inputs)
        data_in = jit_engine.pycuda.driver.to_device(input_data)

        bin_exec.cuda_call(self._cuda_context_struct, self._cuda_param_struct,
                           data_in, self._cuda_data_struct, self._cuda_conditions)

        # Copy the data struct from the device
        vo_ty = bin_exec.byref_arg_types[3]
        out_buf = bytearray(ctypes.sizeof(vo_ty))
        jit_engine.pycuda.driver.memcpy_dtoh(out_buf, self._cuda_data_struct)
        self._data_struct = vo_ty.from_buffer(out_buf)

    def cuda_run(self, inputs, runs, num_input_sets):
        bin_run = self._composition._get_bin_run()
        # Create input buffer
        inputs = self._get_run_input_struct(inputs, num_input_sets)
        input_data = bytearray(inputs)
        data_in = jit_engine.pycuda.driver.to_device(input_data)

        # Create output buffer
        output_type = (bin_run.byref_arg_types[4] * runs)
        output_size = ctypes.sizeof(output_type)
        data_out  = jit_engine.pycuda.driver.mem_alloc(output_size)

        runs_count = jit_engine.pycuda.driver.In(np.int32(runs))
        input_count = jit_engine.pycuda.driver.In(np.int32(num_input_sets))

        bin_run.cuda_call(self._cuda_context_struct, self._cuda_param_struct,
                          self._cuda_data_struct, data_in, data_out, runs_count,
                          input_count)

        # Copy the data struct from the device
        out_buf = bytearray(output_size)
        jit_engine.pycuda.driver.memcpy_dtoh(out_buf, data_out)
        outputs = output_type.from_buffer(out_buf)
        return _convert_ctype_to_python(outputs)
