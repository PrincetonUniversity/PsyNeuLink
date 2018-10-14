# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import ctypes
import os
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
_cpu_engine = cpu_jit_engine

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
        args = zip(self.byref_arg_types, pargs)
        cargs = [ctypes.cast(ctypes.byref(p), ctypes.POINTER(t)) for t, p in args]
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

_cpu_engine._engine.set_object_cache(_updateNativeBinaries)

def _convert_ctype_to_python(x):
    if isinstance(x, ctypes.Structure):
        return [_convert_ctype_to_python(getattr(x, field_name)) for field_name, _ in x._fields_]
    if isinstance(x, ctypes.Array):
        return [num for num in x]
    if isinstance(x, ctypes.c_double):
        return x.value
    if isinstance(x, float):
        return x

    print(x)
    assert False

# Initialize builtins
with LLVMBuilderContext() as ctx:
    builtins.setup_vxm(ctx)
