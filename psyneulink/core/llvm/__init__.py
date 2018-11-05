# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

from psyneulink.core.globals.utilities import CNodeRole

import copy, ctypes, os
import sys as _sys

from llvmlite import binding, ir

from . import builtins
from .builder_context import *
from .jit_engine import cpu_jit_engine

__all__ = ['LLVMBinaryFunction', 'LLVMBuilderContext', '_convert_llvm_ir_to_ctype', '_convert_ctype_to_python']

__dumpenv = os.environ.get("PNL_LLVM_DUMP")
_compiled_modules = set()
_binary_generation = 0

def _try_parse_module(module):
    if __dumpenv is not None and __dumpenv.find("llvm") != -1:
        print(module)

    # IR module is not the same as binding module.
    # "assembly" in this case is LLVM IR assembly.
    # This is intentional design decision to ease
    # compatibility between LLVM versions.
    try:
        mod = binding.parse_assembly(str(module))
        mod.verify()
    except Exception as e:
        print("ERROR: llvm parsing failed: {}".format(e))
        mod = None

    return mod

def _llvm_build():
    # Parse generated modules and link them
    mod_bundle = binding.parse_assembly("")
    for m in _modules:
        new_mod = _try_parse_module(m)
        if new_mod is not None:
            mod_bundle.link_in(new_mod)
            _compiled_modules.add(m)

    _modules.clear()

    # Add the new module to jit engine
    _cpu_engine.opt_and_append_bin_module(mod_bundle)

    global _binary_generation
    if __dumpenv is not None and __dumpenv.find("compile") != -1:
        global _binary_generation
        print("COMPILING GENERATION: {} -> {}".format(_binary_generation, LLVMBuilderContext._llvm_generation))

    # update binary generation
    _binary_generation = LLVMBuilderContext._llvm_generation


_binaries = {}

class LLVMBinaryFunction:
    def __init__(self, name):
        self.name = name
        self.__ptr = None

        # Function signature
        f = _find_llvm_function(self.name, _compiled_modules)
        assert(isinstance(f, ir.Function))

        return_type = _convert_llvm_ir_to_ctype(f.return_value.type)
        params = []
        self.__byref_arg_types = []
        for a in f.args:
            if type(a.type) is ir.PointerType:
                # remember pointee type for easier initialization
                byref_type = _convert_llvm_ir_to_ctype(a.type.pointee)
                param_type = ctypes.POINTER(byref_type)
            else:
                param_type = _convert_llvm_ir_to_ctype(a.type)
                byref_type = None

            self.__byref_arg_types.append(byref_type)
            params.append(param_type)
        self.__c_func_type = ctypes.CFUNCTYPE(return_type, *params)

        # Binary pointer.
        self.ptr = _cpu_engine._engine.get_function_address(name)

    def __call__(self, *args, **kwargs):
        return self.c_func(*args, **kwargs)

    def wrap_call(self, *pargs):
        cpargs = [ctypes.byref(p) if p is not None else None for p in pargs]
        args = zip(cpargs, self.byref_arg_types)
        cargs = [ctypes.cast(p, ctypes.POINTER(t)) for p, t in args]
        self(*tuple(cargs))

    # This will be useful for non-native targets
    @property
    def ptr(self):
        return self.__ptr

    @ptr.setter
    def ptr(self, ptr):
        if self.__ptr != ptr:
            self.__ptr = ptr
            self.__c_func = self.__c_func_type(self.__ptr)

    @property
    def byref_arg_types(self):
        return self.__byref_arg_types

    @property
    def c_func(self):
        return self.__c_func

    @staticmethod
    def get(name):
        if LLVMBuilderContext._llvm_generation > _binary_generation:
            _llvm_build()
        if name not in _binaries.keys():
            _binaries[name] = LLVMBinaryFunction(name)
        return _binaries[name]


def _updateNativeBinaries(module, buffer):
    to_delete = []
    # update all pointers that might have been modified
    for k, v in _binaries.items():
        # One reference is held by the _binaries dict, second is held
        # by the k, v tuple here, third by this function, and 4th is the
        # one passed to getrefcount function
        if _sys.getrefcount(v) == 4:
            to_delete.append(k)
        else:
            new_ptr = _cpu_engine._engine.get_function_address(k)
            v.ptr = new_ptr

    for d in to_delete:
        del _binaries[d]

_cpu_engine = cpu_jit_engine(_updateNativeBinaries)

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
    if hasattr(x, '__len__'):
        return tuple([_tupleize(y) for y in x])
    return x if x is not None else tuple()

class CompExecution:

    def __init__(self, composition):
        self._composition = composition
        self.__frozen_vals = None
        self.__conds = None

        #TODO: This should use compiled function
        with LLVMBuilderContext() as ctx:
            # Data
            c_data = _convert_llvm_ir_to_ctype(self._composition._get_data_struct_type(ctx))
            self.__data_struct = c_data(*self._composition._get_data_initializer())

            # Params
            c_param = _convert_llvm_ir_to_ctype(ctx.get_param_struct_type(self._composition))
            self.__param_struct = c_param(*self._composition.get_param_initializer())
            # Context
            c_context = _convert_llvm_ir_to_ctype(ctx.get_context_struct_type(self._composition))
            self.__context_struct = c_context(*self._composition.get_context_initializer())

    @property
    def __conditions(self):
        if self.__conds is None:
            bin_exec = self._composition._get_bin_execution()
            gen = helpers.ConditionGenerator(None, self._composition)
            self.__conds = bin_exec.byref_arg_types[4](*gen.get_condition_initializer())
        return self.__conds

    @property
    def __all_nodes(self):
        return self._composition.c_nodes + [self._composition.input_CIM, self._composition.output_CIM]

    def extract_frozen_node_output(self, node):
        return self.extract_node_output(node, self.__frozen_vals)

    def extract_node_output(self, node, data = None):
        data = self.__data_struct if data == None else data
        field = self.__data_struct._fields_[0][0]
        res_struct = getattr(self.__data_struct, field)
        index = self.__all_nodes.index(node)
        field = res_struct._fields_[index][0]
        res_struct = getattr(res_struct, field)
        return _convert_ctype_to_python(res_struct)

    def insert_node_output(self, node, data):
        my_field_name = self.__data_struct._fields_[0][0]
        my_res_struct = getattr(self.__data_struct, my_field_name)
        index = self.__all_nodes.index(node)
        node_field_name = my_res_struct._fields_[index][0]
        setattr(my_res_struct, node_field_name, _tupleize(data))

    def _get_input_struct(self, inputs):
        origins = self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN)
        # Read provided input data and separate each input state
        input_data = [[x] for m in origins for x in inputs[m]]

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
            origins = self._composition.get_c_nodes_by_role(CNodeRole.ORIGIN)
            input_data = [[proj.value for proj in state.all_afferents] for state in node.input_states]
            inputs = dict(zip(origins, input_data))

        if inputs is not None:
            inputs = self._get_input_struct(inputs)

        assert inputs is not None or node is not self._composition.input_CIM

        assert node in self.__all_nodes
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

# Initialize builtins
with LLVMBuilderContext() as ctx:
    builtins.setup_vxm(ctx)
