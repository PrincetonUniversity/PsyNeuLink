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
import gc
import inspect
import numpy as np
import time
from math import ceil, log2
from psyneulink._typing import Set
import weakref

from llvmlite import ir

from . import codegen
from .builder_context import *
from .builder_context import _all_modules, _convert_llvm_ir_to_ctype, _convert_llvm_ir_to_dtype
from .debug import debug_env
from .execution import *
from .execution import _tupleize
from .jit_engine import *
from .warnings import *


__all__ = ['LLVMBuilderContext', 'ExecutionMode']

class ExecutionMode(enum.Flag):
    """Specify execution a `Composition` in interpreted or one of ithe compiled modes.
    These are used to specify the **execution_mode** argument of a Composition's `execute <Composition.execute>`,
    `run <Composition.run>`, and `learn <Composition.learn>` methods.  See `Compiled Modes
    <Composition_Compilation_Modes>` under `Compilation <Composition_Compilation>` for additional details concerning
    use of each mode by a Composition.

    Attributes
    ----------

    Python
      Execute using the Python interpreter;  this is the default mode.

    LLVM
      compile and run Composition `Nodes <Composition_Nodes>` and `Projections <Projection>` individually.

    LLVMExec
      compile and run each `TRIAL <TimeScale.TRIAL>` individually.

    LLVMRun
      compile and run multiple `TRIAL <TimeScale.TRIAL>`\\s.

    Auto
      progressively attempt LLVMRun, LLVMexec. LLVM and then Python.

    PyTorch
      execute the `AutodiffComposition` `learn <AutodiffComposition.learn>` method using PyTorch, and its
      `run <AutodiffComposition.run>` method using the Python interpreter.

      .. warning::
         For clarity, this mode should only be used when executing an `AutodiffComposition`; using it
         with a standard `Composition` is possible, but it will **not** have the expected effect of executing
         its `learn <Composition.learn>` method using PyTorch.

    PTX
      compile and run Composition `Nodes <Composition_Nodes>` and `Projections <Projection>` using CUDA for GPU.

    PTXRun
      compile and run multiple `TRIAL <TimeScale.TRIAL>`\\s using CUDA for GPU.
   """

    Python   = 0
    PyTorch = enum.auto()
    LLVM     = enum.auto()
    PTX      = enum.auto()
    _Run      = enum.auto()
    _Exec     = enum.auto()
    _Fallback = enum.auto()

    Auto = _Fallback | _Run | _Exec | LLVM
    LLVMRun = LLVM | _Run
    LLVMExec = LLVM | _Exec
    PTXRun = PTX | _Run
    COMPILED = ~ (Python | PyTorch)


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
    def __init__(self, name: str, *, ctype_ptr_args:tuple=(), dynamic_size_args:tuple=()):
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

        self.np_arg_dtypes = [_convert_llvm_ir_to_dtype(getattr(a.type, "pointee", a.type)) for a in f.args]

        args = [_convert_llvm_ir_to_ctype(a.type) for a in f.args]

        # '_type_' special attribute stores pointee type for pointers
        # https://docs.python.org/3/library/ctypes.html#ctypes._Pointer._type_
        self.byref_arg_types = [a._type_ if hasattr(a, "contents") else None for a in args]

        for i, arg in enumerate(self.np_arg_dtypes):
            if i not in ctype_ptr_args and self.byref_arg_types[i] is not None:
                if i in dynamic_size_args:
                    args[i] = np.ctypeslib.ndpointer(dtype=arg.base, ndim=len(arg.shape) + 1, flags='C_CONTIGUOUS')
                else:
                    args[i] = np.ctypeslib.ndpointer(dtype=arg.base, shape=arg.shape, flags='C_CONTIGUOUS')

        middle = time.perf_counter()
        self.__c_func_type = ctypes.CFUNCTYPE(return_type, *args)
        finish = time.perf_counter()

        if "time_stat" in debug_env:
            print("Time to create ctype function '{}': {} ({} to create types)".format(
                  name, finish - start, middle - start))

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

    def np_buffer_for_arg(self, arg_num, *, extra_dimensions=(), fill_value=np.nan):

        out_base = self.np_arg_dtypes[arg_num].base
        out_shape = extra_dimensions + self.np_arg_dtypes[arg_num].shape

        # fill the buffer with NaN poison
        return np.full(out_shape, fill_value, dtype=out_base)

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def from_obj(obj, *, tags:frozenset=frozenset(), ctype_ptr_args:tuple=(), dynamic_size_args:tuple=()):
        name = LLVMBuilderContext.get_current().gen_llvm_function(obj, tags=tags).name
        return LLVMBinaryFunction.get(name, ctype_ptr_args=ctype_ptr_args, dynamic_size_args=dynamic_size_args)

    @staticmethod
    @functools.lru_cache(maxsize=32)
    def get(name: str, *, ctype_ptr_args:tuple=(), dynamic_size_args:tuple=()):
        return LLVMBinaryFunction(name, ctype_ptr_args=ctype_ptr_args, dynamic_size_args=dynamic_size_args)


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



def cleanup(check_leaks:bool=False):
    global _cpu_engine
    _cpu_engine = None
    global _ptx_engine
    _ptx_engine = None

    _modules.clear()
    _all_modules.clear()

    LLVMBinaryFunction.get.cache_clear()
    LLVMBinaryFunction.from_obj.cache_clear()

    if check_leaks and LLVMBuilderContext.is_active():
        old_context = LLVMBuilderContext.get_current()

        LLVMBuilderContext.clear_global()

        # check that WeakKeyDictionary is not keeping any references
        # Try first without calling the GC
        c = weakref.WeakSet(old_context._cache.keys())
        if len(c) > 0:
            gc.collect()

        assert len(c) == 0, list(c)
    else:
        LLVMBuilderContext.clear_global()

        # If not checking for leaks, there might be active compositions that
        # cache pointers to binary functions. Accessing those pointers would
        # cause segfault.
        # Extract the set of associated compositions. Both to avoid duplicate
        # clears for executions that belong to the same composition, and to
        # avoid modifying the container that is iterated over.
        for c in {e._composition for e in CompExecution.active_executions}:
            c._compilation_data.execution.values.clear()
            c._compilation_data.execution.history.clear()

        # The set of active executions should be empty
        for e in CompExecution.active_executions:
            assert any(inspect.isframe(r) for r in gc.get_referrers(e))

        CompExecution.active_executions.clear()
