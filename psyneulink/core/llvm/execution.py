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
from .debug import debug_env

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
        self._uploaded_bytes = 0
        self._downloaded_bytes = 0
        self._debug_env = debug_env

    def __del__(self):
        if "cuda_data" in self._debug_env:
            try:
                name = self._bin_func.name
            except:
                name = self._composition.name
            print("{} CUDA uploaded: {}".format(name, self._uploaded_bytes))
            print("{} CUDA downloaded: {}".format(name, self._downloaded_bytes))

    def _get_ctype_bytes(self, data):
        # Return dummy buffer. CUDA does not handle 0 size well.
        if ctypes.sizeof(data) == 0:
            return bytearray(b'aaaa')
        return bytearray(data)

    def upload_ctype(self, data):
        self._uploaded_bytes += ctypes.sizeof(data)
        return jit_engine.pycuda.driver.to_device(self._get_ctype_bytes(data))

    def download_ctype(self, source, ty):
        self._downloaded_bytes += ctypes.sizeof(ty)
        out_buf = bytearray(ctypes.sizeof(ty))
        jit_engine.pycuda.driver.memcpy_dtoh(out_buf, source)
        return ty.from_buffer(out_buf)

    def __getattr__(self, attribute):
        if not attribute.startswith("_cuda"):
            return getattr(super(), attribute)

        private_attr = "_buffer" + attribute
        if getattr(self, private_attr) is None:
            new_buffer = self.upload_ctype(getattr(self, attribute[5:]))
            setattr(self, private_attr, new_buffer)

        return getattr(self, private_attr)

    @property
    def _cuda_out_buf(self):
        if self.__cuda_out_buf is None:
            vo_ty = self._bin_func.byref_arg_types[3] * len(self._execution_ids)
            size = ctypes.sizeof(vo_ty)
            self.__cuda_out_buf = jit_engine.pycuda.driver.mem_alloc(size)
        return self.__cuda_out_buf

    def cuda_execute(self, variable):
        # Create input parameter
        new_var = np.asfarray(variable)
        data_in = jit_engine.pycuda.driver.In(new_var)
        self._uploaded_bytes += new_var.nbytes

        self._bin_func.cuda_call(self._cuda_param_struct,
                                 self._cuda_context_struct,
                                 data_in, self._cuda_out_buf,
                                 threads=len(self._execution_ids))

        # Copy the result from the device
        vo_ty = self._bin_func.byref_arg_types[3]
        if len(self._execution_ids) > 1:
            vo_ty =  vo_ty * len(self._execution_ids)

        ct_res =  self.download_ctype(self._cuda_out_buf, vo_ty)
        return _convert_ctype_to_python(ct_res)


class FuncExecution(CUDAExecution):

    def __init__(self, component, execution_ids=[None]):
        super().__init__()
        self._bin_func = component._llvmBinFunction
        self._execution_ids = execution_ids
        self._component = component

        par_struct_ty, ctx_struct_ty, _, _ = self._bin_func.byref_arg_types

        if len(execution_ids) > 1:
            par_struct_ty = par_struct_ty * len(execution_ids)
            ctx_struct_ty = ctx_struct_ty * len(execution_ids)

            par_initializer = (component._get_param_initializer(ex_id) for ex_id in execution_ids)
            ctx_initializer = (component._get_context_initializer(ex_id) for ex_id in execution_ids)
            self.__param_struct = par_struct_ty(*par_initializer)
            self.__context_struct = ctx_struct_ty(*ctx_initializer)

    def _get_compilation_param(self, name, initializer, arg, execution_id):
        param = getattr(self._component._compilation_data, name)
        struct = param.get(execution_id)
        if struct is None:
            initializer = getattr(self._component, initializer)(execution_id)
            struct_ty = self._bin_func.byref_arg_types[arg]
            struct = struct_ty(*initializer)
            param.set(struct, execution_context=execution_id)

        return struct

    @property
    def _param_struct(self):
        if len(self._execution_ids) > 1:
            return self.__param_struct

        return self._get_compilation_param('parameter_struct', '_get_param_initializer', 0, self._execution_ids[0])

    @property
    def _context_struct(self):
        if len(self._execution_ids) > 1:
            return self.__context_struct

        return self._get_compilation_param('context_struct', '_get_context_initializer', 1, self._execution_ids[0])

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

    def execute(self, variable):
        # convert to 3d. we always assume that:
        # a) the input is vector of input states
        # b) input states take vector of projection outputs
        # c) projection output is a vector (even 1 element vector)
        new_var = np.asfarray([np.atleast_2d(x) for x in variable])
        return super().execute(new_var)


class CompExecution(CUDAExecution):

    def __init__(self, composition, execution_ids = [None]):
        super().__init__(buffers=['context_struct', 'param_struct', 'data_struct', 'conditions'])
        self._composition = composition
        self.__frozen_vals = None
        self._execution_ids = execution_ids
        self._bin_func = None

        # At least the input_CIM wrapper should be generated
        with LLVMBuilderContext() as ctx:
            input_cim_fn_name = composition._get_node_wrapper(composition.input_CIM)
            input_cim_fn = ctx.get_llvm_function(input_cim_fn_name)

        # TODO: Consolidate these
        if len(execution_ids) > 1:
            # Input structures
            # TODO: Use the compiled version to get these
            c_context = _convert_llvm_ir_to_ctype(input_cim_fn.args[0].type.pointee)
            c_param = _convert_llvm_ir_to_ctype(input_cim_fn.args[1].type.pointee)
            c_data = _convert_llvm_ir_to_ctype(input_cim_fn.args[3].type.pointee)

            c_context = c_context * len(execution_ids)
            c_param = c_param * len(execution_ids)
            c_data = c_data * len(execution_ids)

            ctx_initializer = (composition._get_context_initializer(ex_id) for ex_id in execution_ids)
            par_initializer = (composition._get_param_initializer(ex_id) for ex_id in execution_ids)
            data_initializer = (composition._get_data_initializer(ex_id) for ex_id in execution_ids)
            # Instantiate structures
            self.__context_struct = c_context(*ctx_initializer)
            self.__param_struct = c_param(*par_initializer)
            self.__data_struct = c_data(*data_initializer)
            self.__conds = None

    @property
    def _conditions(self):
        if len(self._execution_ids) > 1:
            if self.__conds is None:
                cond_type = self._bin_func.byref_arg_types[4] * len(self._execution_ids)
                gen = helpers.ConditionGenerator(None, self._composition)
                cond_initializer = (gen.get_condition_initializer() for _ in self._execution_ids)
                self.__conds = cond_type(*cond_initializer)
            return self.__conds

        conds = self._composition._compilation_data.scheduler_conditions.get(self._execution_ids[0])
        if conds is None:
            cond_type = self._bin_func.byref_arg_types[4]
            gen = helpers.ConditionGenerator(None, self._composition)
            cond_initializer = gen.get_condition_initializer()
            conds = cond_type(*cond_initializer)
            self._composition._compilation_data.scheduler_conditions.set(conds, execution_context=self._execution_ids[0])
        return conds

    def _get_compilation_param(self, name, initializer, arg, execution_id):
        param = getattr(self._composition._compilation_data, name)
        struct = param.get(execution_id)
        if struct is None:
            initializer = getattr(self._composition, initializer)(execution_id)
            struct_ty = self._bin_func.byref_arg_types[arg]
            struct = struct_ty(*initializer)
            param.set(struct, execution_context=execution_id)

        return struct

    @property
    def _param_struct(self):
        if len(self._execution_ids) > 1:
            return self.__param_struct

        return self._get_compilation_param('parameter_struct', '_get_param_initializer', 1, self._execution_ids[0])

    @property
    def _context_struct(self):
        if len(self._execution_ids) > 1:
            return self.__context_struct

        return self._get_compilation_param('context_struct', '_get_context_initializer', 0, self._execution_ids[0])

    @property
    def _data_struct(self):
        if len(self._execution_ids) > 1:
            return self.__data_struct

        # Run wrapper changed argument order
        arg = 2 if len(self._bin_func.byref_arg_types) > 5 else 3

        return self._get_compilation_param('data_struct', '_get_data_initializer', arg, self._execution_ids[0])

    @_data_struct.setter
    def _data_struct(self, data_struct):
        if len(self._execution_ids) > 1:
            self.__data_struct = data_struct
        else:
            self._composition._compilation_data.data_struct.set(data_struct, execution_context = self._execution_ids[0])

    def extract_frozen_node_output(self, node):
        return self._extract_node_output(node, self.__frozen_vals)

    def _extract_node_output(self, node, data = None):
        data = self._data_struct if data == None else data
        field = data._fields_[0][0]
        res_struct = getattr(data, field)
        index = self._composition._get_node_index(node)
        field = res_struct._fields_[index][0]
        res_struct = getattr(res_struct, field)
        return _convert_ctype_to_python(res_struct)

    def extract_node_output(self, node):
        if len(self._execution_ids) > 1:
            return [self._extract_node_output(node, self._data_struct[i]) for i, _ in enumerate(self._execution_ids)]
        else:
            return self._extract_node_output(node)

    def insert_node_output(self, node, data):
        my_field_name = self._data_struct._fields_[0][0]
        my_res_struct = getattr(self._data_struct, my_field_name)
        index = self._composition._get_node_index(node)
        node_field_name = my_res_struct._fields_[index][0]
        setattr(my_res_struct, node_field_name, _tupleize(data))

    def _get_input_struct(self, inputs):
        origins = self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN)
        # Either node execute or composition execute, either way the
        # input_CIM should be ready
        bin_input_node = self._composition._get_bin_mechanism(self._composition.input_CIM)
        c_input = bin_input_node.byref_arg_types[2]
        if len(self._execution_ids) > 1:
            c_input = c_input * len(self._execution_ids)

        # Read provided input data and separate each input state
        if len(self._execution_ids) > 1:
            input_data = []
            for inp in inputs:
                input_data.append(([x] for m in origins for x in inp[m]))
        else:
            input_data = ([x] for m in origins for x in inputs[m])

        return c_input(*_tupleize(input_data))

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
        self._bin_func = self._composition._get_bin_mechanism(node)
        self._bin_func.wrap_call(self._context_struct, self._param_struct,
                           inputs, self.__frozen_vals, self._data_struct)

    def execute(self, inputs):
        inputs = self._get_input_struct(inputs)
        self._bin_func = self._composition._get_bin_execution()
        self._bin_func.wrap_call(self._context_struct, self._param_struct,
                           inputs, self._data_struct, self._conditions)

    def cuda_execute(self, inputs):
        self._bin_func = self._composition._get_bin_execution()
        # Create input buffer
        inputs = self._get_input_struct(inputs)
        data_in = self.upload_ctype(inputs)

        self._bin_func.cuda_call(self._cuda_context_struct, self._cuda_param_struct,
                           data_in, self._cuda_data_struct, self._cuda_conditions,
                           threads=len(self._execution_ids))

        # Copy the data struct from the device
        vo_ty = self._bin_func.byref_arg_types[3]
        if len(self._execution_ids) > 1:
            vo_ty = vo_ty * len(self._execution_ids)
        self._data_struct = self.download_ctype(self._cuda_data_struct, vo_ty)

    # Methods used to accelerate "Run"

    def _get_run_input_struct(self, inputs, num_input_sets):
        origins = self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN)
        input_type = self._composition._get_bin_run().byref_arg_types[3]
        c_input = input_type * num_input_sets
        if len(self._execution_ids) > 1:
            c_input = c_input * len(self._execution_ids)
            run_inputs = []
            for inp in inputs:
                run_inps = []
                # Extract inputs for each trial
                for i in range(num_input_sets):
                    run_inps.append([])
                    for m in origins:
                        run_inps[i] += [[v] for v in inp[m][i]]
                run_inputs.append(run_inps)

        else:
            run_inputs = []
            # Extract inputs for each trial
            for i in range(num_input_sets):
                run_inputs.append([])
                for m in origins:
                    run_inputs[i] += [[v] for v in inputs[m][i]]

        return c_input(*_tupleize(run_inputs))

    def run(self, inputs, runs, num_input_sets):
        self._bin_func = self._composition._get_bin_run()
        inputs = self._get_run_input_struct(inputs, num_input_sets)
        outputs = (self._bin_func.byref_arg_types[4] * runs)()
        runs_count = ctypes.c_int(runs)
        input_count = ctypes.c_int(num_input_sets)
        self._bin_func.wrap_call(self._context_struct, self._param_struct,
                          self._data_struct, inputs, outputs, runs_count,
                          input_count)
        return _convert_ctype_to_python(outputs)

    def cuda_run(self, inputs, runs, num_input_sets):
        self._bin_func = self._composition._get_bin_run()
        # Create input buffer
        inputs = self._get_run_input_struct(inputs, num_input_sets)
        data_in = self.upload_ctype(inputs)

        # Create output buffer
        output_type = (self._bin_func.byref_arg_types[4] * runs)
        if len(self._execution_ids) > 1:
            output_type = output_type * len(self._execution_ids)
        output_size = ctypes.sizeof(output_type)
        data_out  = jit_engine.pycuda.driver.mem_alloc(output_size)

        runs_count = jit_engine.pycuda.driver.In(np.int32(runs))
        input_count = jit_engine.pycuda.driver.In(np.int32(num_input_sets))
        self._uploaded_bytes += 8 # runs_count + input_count

        self._bin_func.cuda_call(self._cuda_context_struct, self._cuda_param_struct,
                          self._cuda_data_struct, data_in, data_out, runs_count,
                          input_count, threads=len(self._execution_ids))

        # Copy the data struct from the device
        ct_out = self.download_ctype(data_out, output_type)
        return _convert_ctype_to_python(ct_out)
