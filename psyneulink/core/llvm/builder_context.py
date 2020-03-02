# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import atexit
import ctypes
import functools
import inspect
from llvmlite import ir
import numpy as np
import os
import re
from typing import Set
import weakref

from psyneulink.core import llvm as pnlvm
from . import codegen
from .debug import debug_env

__all__ = ['LLVMBuilderContext', '_modules', '_find_llvm_function']


_modules: Set[ir.Module] = set()
_all_modules: Set[ir.Module] = set()
_struct_count = 0


@atexit.register
def module_count():
    if "stat" in debug_env:
        print("Total LLVM modules: ", len(_all_modules))
        print("Total structures generated: ", _struct_count)


_BUILTIN_PREFIX = "__pnl_builtin_"
_builtin_intrinsics = frozenset(('pow', 'log', 'exp'))


class LLVMBuilderContext:
    __global_context = None
    __uniq_counter = 0
    _llvm_generation = 0
    int32_ty = ir.IntType(32)
    float_ty = ir.DoubleType()

    def __init__(self):
        self._modules = []
        self._cache = weakref.WeakKeyDictionary()

    def __enter__(self):
        module = ir.Module(name="PsyNeuLinkModule-" + str(LLVMBuilderContext._llvm_generation))
        self._modules.append(module)
        LLVMBuilderContext._llvm_generation += 1
        return self

    def __exit__(self, e_type, e_value, e_traceback):
        assert len(self._modules) > 0
        module = self._modules.pop()
        _modules.add(module)
        _all_modules.add(module)

    @property
    def module(self):
        assert len(self._modules) > 0
        return self._modules[-1]

    @classmethod
    def get_global(cls):
        if cls.__global_context is None:
            cls.__global_context = LLVMBuilderContext()
        return cls.__global_context

    @classmethod
    def get_unique_name(cls, name: str):
        cls.__uniq_counter += 1
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return name + '_' + str(cls.__uniq_counter)

    def get_builtin(self, name: str, args=[], function_type=None):
        if name in _builtin_intrinsics:
            return self.import_llvm_function(_BUILTIN_PREFIX + name)
        if name in ('maxnum'):
            function_type = pnlvm.ir.FunctionType(args[0], [args[0], args[0]])
        return self.module.declare_intrinsic("llvm." + name, args, function_type)

    def create_llvm_function(self, args, component, name=None, *, return_type=ir.VoidType(), tags:frozenset=frozenset()):
        name = "_".join((str(component), *tags)) if name is None else name

        # Builtins are already unique and need to keep their special name
        func_name = name if name.startswith(_BUILTIN_PREFIX) else self.get_unique_name(name)
        func_ty = pnlvm.ir.FunctionType(return_type, args)
        llvm_func = pnlvm.ir.Function(self.module, func_ty, name=func_name)
        llvm_func.attributes.add('argmemonly')
        for a in llvm_func.args:
            if isinstance(a.type, ir.PointerType):
                a.attributes.add('nonnull')

        metadata = self.get_debug_location(llvm_func, component)
        if metadata is not None:
            scope = dict(metadata.operands)["scope"]
            llvm_func.set_metadata("dbg", scope)

        # Create entry block
        block = llvm_func.append_basic_block(name="entry")
        builder = pnlvm.ir.IRBuilder(block)
        builder.debug_metadata = metadata

        return builder

    def gen_llvm_function(self, obj, *, tags:frozenset) -> ir.Function:
        cache_variants = self._cache.setdefault(obj, dict())

        if tags not in cache_variants:
            cache_variants[tags] = obj._gen_llvm_function(tags=tags)
        return cache_variants[tags]

    def import_llvm_function(self, fun, *, tags:frozenset=frozenset()) -> ir.Function:
        """
        Get function handle if function exists in current modele.
        Create function declaration if it exists in a older module.
        """
        if isinstance(fun, str):
            f = _find_llvm_function(fun, _all_modules | {self.module})
        else:
            f = self.gen_llvm_function(fun, tags=tags)

        # Add declaration to the current module
        if f.name not in self.module.globals:
            decl_f = ir.Function(self.module, f.type.pointee, f.name)
            assert decl_f.is_declaration
            return decl_f
        return f

    @staticmethod
    def get_debug_location(func: ir.Function, component):
        if "debug_info" not in debug_env:
            return

        mod = func.module
        path = inspect.getfile(component.__class__) if component is not None else "<pnl_builtin>"
        d_version = mod.add_metadata([ir.IntType(32)(2), "Dwarf Version", ir.IntType(32)(4)])
        di_version = mod.add_metadata([ir.IntType(32)(2), "Debug Info Version", ir.IntType(32)(3)])
        flags = mod.add_named_metadata("llvm.module.flags")
        if len(flags.operands) == 0:
            flags.add(d_version)
            flags.add(di_version)
        cu = mod.add_named_metadata("llvm.dbg.cu")
        di_file = mod.add_debug_info("DIFile", {
            "filename": os.path.basename(path),
            "directory": os.path.dirname(path),
        })
        di_func_type = mod.add_debug_info("DISubroutineType", {
            # None as `null`
            "types": mod.add_metadata([None]),
        })
        di_compileunit = mod.add_debug_info("DICompileUnit", {
            "language": ir.DIToken("DW_LANG_Python"),
            "file": di_file,
            "producer": "PsyNeuLink",
            "runtimeVersion": 0,
            "isOptimized": False,
        }, is_distinct=True)
        cu.add(di_compileunit)
        di_func = mod.add_debug_info("DISubprogram", {
            "name": func.name,
            "file": di_file,
            "line": 0,
            "type": di_func_type,
            "isLocal": False,
            "unit": di_compileunit,
        }, is_distinct=True)
        di_loc = mod.add_debug_info("DILocation", {
            "line": 0, "column": 0, "scope": di_func,
        })
        return di_loc

    def get_input_struct_type(self, component):
        if hasattr(component, '_get_input_struct_type'):
            return component._get_input_struct_type(self)

        default_var = component.defaults.variable
        return self.convert_python_struct_to_llvm_ir(default_var)

    def get_output_struct_type(self, component):
        if hasattr(component, '_get_output_struct_type'):
            return component._get_output_struct_type(self)

        default_val = component.defaults.value
        return self.convert_python_struct_to_llvm_ir(default_val)

    def get_param_struct_type(self, component):
        if hasattr(component, '_get_param_struct_type'):
            return component._get_param_struct_type(self)

        params = component._get_param_values()
        return self.convert_python_struct_to_llvm_ir(params)

    def get_state_struct_type(self, component):
        if hasattr(component, '_get_state_struct_type'):
            return component._get_state_struct_type(self)

        stateful = component._get_state_values()
        return self.convert_python_struct_to_llvm_ir(stateful)

    def get_data_struct_type(self, component):
        if hasattr(component, '_get_data_struct_type'):
            return component._get_data_struct_type(self)

        return ir.LiteralStructType([])

    def get_node_wrapper(self, composition, node):
        cache = self._cache.setdefault(composition, {})
        if node not in cache:
            class node_wrapper():
                def _gen_llvm_function(s, *, tags:frozenset):
                    with self as ctx:
                        return codegen.gen_node_wrapper(ctx, composition, node, tags=tags)
            w = node_wrapper()
            cache[node] = w
            return w

        return cache[node]

    def get_param_ptr(self, component, builder, params_ptr, param_name):
        idx = self.int32_ty(component._get_param_ids().index(param_name))
        return builder.gep(params_ptr, [self.int32_ty(0), idx],
                           name="ptr_param_{}_{}".format(param_name, component.name))

    def get_state_ptr(self, component, builder, state_ptr, stateful_name):
        idx = self.int32_ty(component._get_state_ids().index(stateful_name))
        return builder.gep(state_ptr, [self.int32_ty(0), idx],
                           name="ptr_state_{}_{}".format(stateful_name,
                                                         component.name))

    def convert_python_struct_to_llvm_ir(self, t):
        if type(t) is list:
            if len(t) == 0:
                return ir.LiteralStructType([])
            assert all(type(x) is type(t[0]) for x in t)
            elem_t = self.convert_python_struct_to_llvm_ir(t[0])
            return ir.ArrayType(elem_t, len(t))
        elif type(t) is tuple:
            elems_t = (self.convert_python_struct_to_llvm_ir(x) for x in t)
            return ir.LiteralStructType(elems_t)
        elif isinstance(t, (int, float)):
            return self.float_ty
        elif isinstance(t, np.ndarray):
            return self.convert_python_struct_to_llvm_ir(t.tolist())
        elif t is None:
            return ir.LiteralStructType([])
        elif isinstance(t, np.random.RandomState):
            return pnlvm.builtins.get_mersenne_twister_state_struct(self)
        assert False, "Don't know how to convert {}".format(type(t))


def _find_llvm_function(name: str, mods=_all_modules) -> ir.Function:
    f = None
    for m in mods:
        if name in m.globals:
            f = m.get_global(name)

    if not isinstance(f, ir.Function):
        raise ValueError("No such function: {}".format(name))
    return f


def _gen_cuda_kernel_wrapper_module(function):
    module = ir.Module(name="wrapper_" + function.name)

    decl_f = ir.Function(module, function.type.pointee, function.name)
    assert decl_f.is_declaration
    kernel_func = ir.Function(module, function.type.pointee, function.name + "_cuda_kernel")
    block = kernel_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Calculate global id of a thread in x dimension
    intrin_ty = ir.FunctionType(ir.IntType(32), [])
    tid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.tid.x")
    ntid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.ntid.x")
    ctaid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.ctaid.x")
    global_id = builder.mul(builder.call(ctaid_x_f, []), builder.call(ntid_x_f, []))
    global_id = builder.add(global_id, builder.call(tid_x_f, []))

    # Runs need special handling. data_in and data_out are one dimensional,
    # but hold entries for all parallel invocations.
    is_comp_run = len(kernel_func.args) == 7
    if is_comp_run:
        runs_count = kernel_func.args[5]
        input_count = kernel_func.args[6]

    # Index all pointer arguments
    indexed_args = []
    for i, arg in enumerate(kernel_func.args):
        # Don't adjust #inputs and #trials
        if isinstance(arg.type, ir.PointerType):
            offset = global_id
            # #runs and #trials needs to be the same
            if is_comp_run and i >= 5:
                offset = ir.IntType(32)(0)
            # data arrays need special handling
            elif is_comp_run and i == 4:  # data_out
                offset = builder.mul(global_id, builder.load(runs_count))
            elif is_comp_run and i == 3:  # data_in
                offset = builder.mul(global_id, builder.load(input_count))

            arg = builder.gep(arg, [offset])

        indexed_args.append(arg)
    builder.call(decl_f, indexed_args)
    builder.ret_void()

    # Add kernel mark metadata
    module.add_named_metadata("nvvm.annotations", [kernel_func, "kernel", ir.IntType(32)(1)])

    return module


@functools.lru_cache(maxsize=128)
def _convert_llvm_ir_to_ctype(t: ir.Type):
    type_t = type(t)

    if type_t is ir.VoidType:
        return None
    elif type_t is ir.IntType:
        if t.width == 32:
            return ctypes.c_int
        elif t.width == 64:
            return ctypes.c_longlong
        else:
            assert False, "Integer type too big!"
    elif type_t is ir.DoubleType:
        return ctypes.c_double
    elif type_t is ir.FloatType:
        return ctypes.c_float
    elif type_t is ir.PointerType:
        pointee = _convert_llvm_ir_to_ctype(t.pointee)
        ret_t = ctypes.POINTER(pointee)
    elif type_t is ir.ArrayType:
        element_type = _convert_llvm_ir_to_ctype(t.element)
        ret_t = element_type * len(t)
    elif type_t is ir.LiteralStructType:
        global _struct_count
        uniq_name = "struct_" + str(_struct_count)
        _struct_count += 1

        field_list = []
        for i, e in enumerate(t.elements):
            # llvmlite modules get _unique string only works for symbol names
            field_uniq_name = uniq_name + "field_" + str(i)
            field_list.append((field_uniq_name, _convert_llvm_ir_to_ctype(e)))

        ret_t = type(uniq_name, (ctypes.Structure,), {"__init__": ctypes.Structure.__init__})
        ret_t._fields_ = field_list
        assert len(ret_t._fields_) == len(t.elements)
    else:
        assert False, "Don't know how to convert LLVM type: {}".format(t)

    return ret_t
