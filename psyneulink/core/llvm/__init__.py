# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import ctypes, os, sys

from llvmlite import ir

from . import builtins
from .builder_context import *
from .execution import *
from .execution import _tupleize
from .jit_engine import *

__all__ = ['LLVMBinaryFunction', 'LLVMBuilderContext']

__dumpenv = os.environ.get("PNL_LLVM_DUMP")
_compiled_modules = set()
_binary_generation = 0


def _llvm_build():
    _cpu_engine.compile_modules(_modules, _compiled_modules)
    _modules.clear()

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

        self.__c_func = None
        self.__c_func_type = None
        self.__byref_arg_types = None

    def _init_host_func_type(self):
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

    def __call__(self, *args, **kwargs):
        return self.c_func(*args, **kwargs)

    def wrap_call(self, *pargs):
        cpargs = (ctypes.byref(p) if p is not None else None for p in pargs)
        args = zip(cpargs, self.byref_arg_types)
        cargs = (ctypes.cast(p, ctypes.POINTER(t)) for p, t in args)
        self(*tuple(cargs))

    # This will be useful for non-native targets
    @property
    def ptr(self):
        if self.__ptr is None:
            # Binary pointer and recreate ctype function
            self.ptr = _cpu_engine._engine.get_function_address(self.name)
        return self.__ptr

    @ptr.setter
    def ptr(self, ptr):
        if self.__ptr != ptr:
            self.__ptr = ptr
            self.__c_func = self._c_func_type(self.__ptr)

    @property
    def byref_arg_types(self):
        if self.__byref_arg_types is None:
            self._init_host_func_type()
        return self.__byref_arg_types

    @property
    def _c_func_type(self):
        if self.__c_func_type is None:
            self._init_host_func_type()
        return self.__c_func_type

    @property
    def c_func(self):
        if self.__c_func is None:
            self.__c_func = self._c_func_type(self.ptr)
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
        if sys.getrefcount(v) == 4:
            to_delete.append(k)
        else:
            new_ptr = _cpu_engine._engine.get_function_address(k)
            v.ptr = new_ptr

    for d in to_delete:
        del _binaries[d]

_cpu_engine = cpu_jit_engine(_updateNativeBinaries)

# Initialize builtins
with LLVMBuilderContext() as ctx:
    builtins.setup_vxm(ctx)
