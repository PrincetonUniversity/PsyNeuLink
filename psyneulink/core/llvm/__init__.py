# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import ctypes
import enum
import functools
import numpy as np
import time
from math import ceil, log2
from typing import Set

from llvmlite import ir

from . import codegen
from .builder_context import *
from .builder_context import _all_modules, _convert_llvm_ir_to_ctype
from .debug import debug_env
from .execution import *
from .execution import _tupleize
from .jit_engine import *

__all__ = ['LLVMBuilderContext', 'ExecutionMode']

class ExecutionMode(enum.Flag):
    Python   = 0
    LLVM     = enum.auto()
    PTX      = enum.auto()
    _Run      = enum.auto()
    _Exec     = enum.auto()
    _Fallback = enum.auto()

    Auto = _Fallback | _Run | _Exec | LLVM
    LLVMRun = LLVM | _Run
    LLVMExec = LLVM | _Exec
    PTXRun = PTX | _Run
    PTXExec = PTX | _Exec


_binary_generation = 0


def _compiled_modules() -> Set[ir.Module]:
    return set().union(*(e.compiled_modules for e in _get_engines()))


def _staged_modules() -> Set[ir.Module]:
    return set().union(*(e.staged_modules for e in _get_engines()))


def _llvm_build(target_generation=_binary_generation + 1):
    global _binary_generation
    if target_generation <= _binary_generation:
        if "compile" in debug_env:
            print("SKIPPING COMPILATION: {} -> {}".format(_binary_generation, target_generation))
        return

    if "compile" in debug_env:
        print("STAGING GENERATION: {} -> {}".format(_binary_generation, target_generation))

    for e in _get_engines():
        e.stage_compilation(_modules)
    _modules.clear()

    # update binary generation
    _binary_generation = target_generation


class LLVMBinaryFunction:
    def __init__(self, name: str):
        self.name = name

        self.__c_func = None
        self.__cuda_kernel = None

        # Make sure builder context is initialized
        LLVMBuilderContext.get_current()

        # Compile any pending modules
        _llvm_build(LLVMBuilderContext._llvm_generation)

        # Function signature
        # We could skip compilation if the function is in _compiled_models,
        # but that happens rarely
        f = _find_llvm_function(self.name, _compiled_modules() | _staged_modules())

        # Create ctype function instance
        start = time.perf_counter()
        return_type = _convert_llvm_ir_to_ctype(f.return_value.type)
        params = [_convert_llvm_ir_to_ctype(a.type) for a in f.args]
        middle = time.perf_counter()
        self.__c_func_type = ctypes.CFUNCTYPE(return_type, *params)
        finish = time.perf_counter()

        if "time_stat" in debug_env:
            print("Time to create ctype function '{}': {} ({} to create types)".format(
                  name, finish - start, middle - start))

        self.byref_arg_types = [p._type_ for p in params]

    @property
    def c_func(self):
        if self.__c_func is None:
            # This assumes there are potential staged modules.
            # The engine had to be instantiated to have staged modules,
            # so it's safe to access it directly
            _cpu_engine.compile_staged()
            ptr = _cpu_engine._engine.get_function_address(self.name)
            self.__c_func = self.__c_func_type(ptr)
        return self.__c_func

    def __call__(self, *args, **kwargs):
        return self.c_func(*args, **kwargs)

    def wrap_call(self, *pargs):
        cpargs = (ctypes.byref(p) if p is not None else None for p in pargs)
        args = zip(cpargs, self.c_func.argtypes)
        self(*(ctypes.cast(p, t) for p, t in args))

    @property
    def _cuda_kernel(self):
        if self.__cuda_kernel is None:
            _ptx_engine.compile_staged()
            self.__cuda_kernel = _ptx_engine.get_kernel(self.name)
        return self.__cuda_kernel

    def cuda_max_block_size(self, override):
        if override is not None:
            return override

        kernel = self._cuda_kernel
        device = jit_engine.pycuda.autoinit.device

        if kernel.shared_size_bytes > 0:
            # we use shared memory, prefer big blocks.
            # Limited by reg usage
            rounded_regs = 2 ** ceil(log2(kernel.num_regs))
            block_size = device.get_attribute(jit_engine.pycuda.driver.device_attribute.MAX_REGISTERS_PER_BLOCK) // rounded_regs
        else:
            # Use smallest possible blocks
            block_size = device.get_attribute(jit_engine.pycuda.driver.device_attribute.WARP_SIZE)

        block_size = min(device.get_attribute(jit_engine.pycuda.driver.device_attribute.MAX_THREADS_PER_BLOCK), block_size)
        if "stat" in debug_env:
            print("kernel '", self.name, "' registers:", kernel.num_regs)
            print("kernel '", self.name, "' local memory size:", kernel.local_size_bytes)
            print("kernel '", self.name, "' shared memory size:", kernel.shared_size_bytes)
            print("kernel '", self.name, "' selected block size:", block_size)

        return block_size

    def cuda_call(self, *args, threads=1, block_size=None):
        block_size = self.cuda_max_block_size(block_size)
        grid = ((threads + block_size - 1) // block_size, 1)
        ktime = self._cuda_kernel(*args, np.int32(threads), time_kernel="time_stat" in debug_env,
                                  block=(block_size, 1, 1), grid=grid)
        if "time_stat" in debug_env:
            print("Time to run kernel '{}' using {} threads: {}".format(
                self.name, threads, ktime))

    def cuda_wrap_call(self, *args, **kwargs):
        wrap_args = (jit_engine.pycuda.driver.InOut(a) if isinstance(a, np.ndarray) else a for a in args)
        self.cuda_call(*wrap_args, **kwargs)

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def from_obj(obj, *, tags:frozenset=frozenset()):
        name = LLVMBuilderContext.get_current().gen_llvm_function(obj, tags=tags).name
        return LLVMBinaryFunction.get(name)

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def get(name: str):
        return LLVMBinaryFunction(name)

    def get_multi_run(self):
        try:
            multirun_llvm = _find_llvm_function(self.name + "_multirun")
        except ValueError:
            function = _find_llvm_function(self.name)
            with LLVMBuilderContext.get_current() as ctx:
                multirun_llvm = codegen.gen_multirun_wrapper(ctx, function)

        return LLVMBinaryFunction.get(multirun_llvm.name)


_cpu_engine = None
_ptx_engine = None

def _get_engines():
    global _cpu_engine
    if _cpu_engine is None:
        _cpu_engine = cpu_jit_engine()

    global _ptx_engine
    if ptx_enabled:
        if _ptx_engine is None:
            _ptx_engine = ptx_jit_engine()
        return [_cpu_engine, _ptx_engine]

    return [_cpu_engine]



def cleanup():
    global _cpu_engine
    _cpu_engine = None
    global _ptx_engine
    _ptx_engine = None

    _modules.clear()
    _all_modules.clear()

    LLVMBinaryFunction.get.cache_clear()
    LLVMBinaryFunction.from_obj.cache_clear()

    LLVMBuilderContext.clear_global()
