# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import ctypes
import functools
import numpy as np

from llvmlite import ir

from . import builtins
from .builder_context import *
from .builder_context import _all_modules, _convert_llvm_ir_to_ctype
from .debug import debug_env
from .execution import *
from .execution import _tupleize
from .jit_engine import *

__all__ = ['LLVMBuilderContext']


_compiled_modules = set()
_binary_generation = 0


def _llvm_build(target_generation=_binary_generation + 1):
    global _binary_generation
    if target_generation <= _binary_generation:
        if "compile" in debug_env:
            print("SKIPPING COMPILATION: {} -> {}".format(_binary_generation, target_generation))
        return

    if "compile" in debug_env:
        print("COMPILING GENERATION: {} -> {}".format(_binary_generation, target_generation))

    _cpu_engine.compile_modules(_modules, _compiled_modules)
    if ptx_enabled:
        _ptx_engine.compile_modules(_modules, set())
    _modules.clear()

    # update binary generation
    _binary_generation = target_generation


class LLVMBinaryFunction:
    def __init__(self, name):
        self.name = name

        self.__c_func = None
        self.__c_func_type = None
        self.__byref_arg_types = None

        self.__cuda_kernel = None

    def _init_host_func_type(self):
        # Function signature
        f = _find_llvm_function(self.name, _compiled_modules)

        return_type = _convert_llvm_ir_to_ctype(f.return_value.type)
        params = []
        self.__byref_arg_types = []
        for a in f.args:
            param_type = _convert_llvm_ir_to_ctype(a.type)
            if type(a.type) is ir.PointerType:
                # remember pointee type for easier initialization
                byref_type = _convert_llvm_ir_to_ctype(a.type.pointee)
            else:
                byref_type = None

            self.__byref_arg_types.append(byref_type)
            params.append(param_type)
        self.__c_func_type = ctypes.CFUNCTYPE(return_type, *params)

    def __call__(self, *args, **kwargs):
        return self.c_func(*args, **kwargs)

    def wrap_call(self, *pargs):
        #print(pargs)
        cpargs = (ctypes.byref(p) if p is not None else None for p in pargs)
        args = zip(cpargs, self.byref_arg_types)
        cargs = (ctypes.cast(p, ctypes.POINTER(t)) for p, t in args)
        self(*tuple(cargs))

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
            ptr = _cpu_engine._engine.get_function_address(self.name)
            self.__c_func = self._c_func_type(ptr)
        return self.__c_func

    @property
    def _cuda_kernel(self):
        if self.__cuda_kernel is None:
            self.__cuda_kernel = _ptx_engine.get_kernel(self.name)
        return self.__cuda_kernel

    def cuda_call(self, *args, threads=1):
        self._cuda_kernel(*args, block=(1, 1, 1), grid=(threads, 1))

    def cuda_wrap_call(self, *args):
        wrap_args = (jit_engine.pycuda.driver.InOut(a) if isinstance(a, np.ndarray) else a for a in args)
        self.cuda_call(*wrap_args)

    @staticmethod
    def from_obj(obj):
        name = LLVMBuilderContext.get_global().gen_llvm_function(obj).name
        return LLVMBinaryFunction.get(name)

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def get(name):
        _llvm_build(LLVMBuilderContext._llvm_generation)
        return LLVMBinaryFunction(name)

    def get_multi_run(self):
        try:
            multirun_llvm = _find_llvm_function(self.name + "_multirun")
        except ValueError:
            function = _find_llvm_function(self.name)
            with LLVMBuilderContext.get_global() as ctx:
                multirun_llvm = ctx.gen_multirun_wrapper(function)

        return LLVMBinaryFunction.get(multirun_llvm.name)


_cpu_engine = cpu_jit_engine()
if ptx_enabled:
    _ptx_engine = ptx_jit_engine()


# Initialize builtins
def init_builtins():
    with LLVMBuilderContext.get_global() as ctx:
        builtins.setup_pnl_intrinsics(ctx)
        builtins.setup_vxm(ctx)
        builtins.setup_vxm_transposed(ctx)
        builtins.setup_mersenne_twister(ctx)
        builtins.setup_vec_add(ctx)
        builtins.setup_mat_add(ctx)
        builtins.setup_vec_sub(ctx)
        builtins.setup_mat_sub(ctx)
        builtins.setup_vec_copy(ctx)
        builtins.setup_vec_hadamard(ctx)
        builtins.setup_mat_hadamard(ctx)
        builtins.setup_vec_scalar_mult(ctx)
        builtins.setup_mat_scalar_mult(ctx)
        builtins.setup_mat_scalar_add(ctx)


def cleanup():
    _cpu_engine.clean_module()
    if ptx_enabled:
        _ptx_engine.clean_module()

    _modules.clear()
    _compiled_modules.clear()
    _all_modules.clear()
    LLVMBinaryFunction.get.cache_clear()
    init_builtins()


init_builtins()
