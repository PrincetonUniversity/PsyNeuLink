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

from .builder_context import *
from . import helpers

__all__ = ['CompExecution']

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

class CompExecution:

    def __init__(self, composition):
        self._composition = composition
        self.__frozen_vals = None
        self.__conds = None

        # At least the input_CIM wrapper should be generated
        with LLVMBuilderContext() as ctx:
            input_cim_fn_name = composition._get_node_wrapper(composition.input_CIM)
            input_cim_fn = ctx.get_llvm_function(input_cim_fn_name)

        # Context
        c_context = _convert_llvm_ir_to_ctype(input_cim_fn.args[0].type.pointee)
        self.__context_struct = c_context(*composition.get_context_initializer())

        # Params
        c_param = _convert_llvm_ir_to_ctype(input_cim_fn.args[1].type.pointee)
        self.__param_struct = c_param(*self._composition.get_param_initializer())
        # Data
        c_data = _convert_llvm_ir_to_ctype(input_cim_fn.args[3].type.pointee)
        self.__data_struct = c_data(*self._composition._get_data_initializer())


    @property
    def __conditions(self):
        if self.__conds is None:
            bin_exec = self._composition._get_bin_execution()
            gen = helpers.ConditionGenerator(None, self._composition)
            self.__conds = bin_exec.byref_arg_types[4](*gen.get_condition_initializer())
        return self.__conds

    def extract_frozen_node_output(self, node):
        return self.extract_node_output(node, self.__frozen_vals)

    def extract_node_output(self, node, data = None):
        data = self.__data_struct if data == None else data
        field = self.__data_struct._fields_[0][0]
        res_struct = getattr(self.__data_struct, field)
        index = self._composition._get_node_index(node)
        field = res_struct._fields_[index][0]
        res_struct = getattr(res_struct, field)
        return _convert_ctype_to_python(res_struct)

    def insert_node_output(self, node, data):
        my_field_name = self.__data_struct._fields_[0][0]
        my_res_struct = getattr(self.__data_struct, my_field_name)
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
        self.__frozen_vals = copy.deepcopy(self.__data_struct)

    def execute_node(self, node, inputs = None):
        # We need to reconstruct the inputs here if they were not provided.
        # This happens during node execution of nested compositions.
        if inputs is None and node is self._composition.input_CIM:
            # This assumes origin mechanisms are in the same order as
            # CIM input states
            origins = (n for n in self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN) for istate in n.input_states)
            input_data = ([proj.value for proj in state.all_afferents] for state in node.input_states)
            inputs = defaultdict(list)
            for n, d in zip(origins, input_data):
                inputs[n].append(d[0])

        if inputs is not None:
            inputs = self._get_input_struct(inputs)

        assert inputs is not None or node is not self._composition.input_CIM

        assert node in self._composition._all_nodes
        bin_node = self._composition._get_bin_mechanism(node)
        bin_node.wrap_call(self.__context_struct, self.__param_struct,
                           inputs, self.__frozen_vals, self.__data_struct)

    def execute(self, inputs):
        inputs = self._get_input_struct(inputs)
        bin_exec = self._composition._get_bin_execution()
        bin_exec.wrap_call(self.__context_struct, self.__param_struct,
                           inputs, self.__data_struct, self.__conditions)

    def run(self, inputs, runs, num_input_sets):
        bin_run = self._composition._get_bin_run()
        inputs = self._get_run_input_struct(inputs, num_input_sets)
        outputs = (bin_run.byref_arg_types[4] * runs)()
        runs_count = ctypes.c_int(runs)
        input_count = ctypes.c_int(num_input_sets)
        bin_run.wrap_call(self.__context_struct, self.__param_struct,
                          self.__data_struct, inputs, outputs, runs_count,
                          input_count)
        return _convert_ctype_to_python(outputs)
