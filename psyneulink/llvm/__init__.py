# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import numpy as np
from llvmlite import binding, ir
import ctypes
import os, sys

from psyneulink.llvm import builtins
from psyneulink.llvm.jit_engine import cpu_jit_engine

__dumpenv = os.environ.get("PNL_LLVM_DUMP")
_modules = set()
_compiled_modules = set()

# TODO: Should this be selectable?
_int32_ty = ir.IntType(32)
_float_ty = ir.DoubleType()
_llvm_generation = 0
_binary_generation = 0

def _find_llvm_function(name, mods = _modules | _compiled_modules):
    f = None
    for m in mods:
        if name in m.globals:
            f = m.get_global(name)

    if not isinstance(f, ir.Function):
        raise ValueError("No such function: {}".format(name))
    return f

class LLVMBuilderContext:
    module = None
    nest_level = 0
    uniq_counter = 0

    def __init__(self):
        self.int32_ty = _int32_ty
        self.float_ty = _float_ty

    def __enter__(self):
        if LLVMBuilderContext.nest_level == 0:
            assert LLVMBuilderContext.module is None
            LLVMBuilderContext.module = ir.Module(name="PsyNeuLinkModule-" + str(_llvm_generation))
        LLVMBuilderContext.nest_level += 1
        return self

    def __exit__(self, e_type, e_value, e_traceback):
        LLVMBuilderContext.nest_level -= 1
        if LLVMBuilderContext.nest_level == 0:
            assert LLVMBuilderContext.module is not None
            _modules.add(LLVMBuilderContext.module)
            LLVMBuilderContext.module = None

        global _llvm_generation
        _llvm_generation += 1

    def get_unique_name(self, name):
        LLVMBuilderContext.uniq_counter += 1
        return name + '-' + str(LLVMBuilderContext.uniq_counter)

    def get_llvm_function(self, name):
        f = _find_llvm_function(name, _compiled_modules | _modules | {LLVMBuilderContext.module})
        # Add declaration to the current module
        if f.name not in LLVMBuilderContext.module.globals:
            decl_f = ir.Function(LLVMBuilderContext.module, f.type.pointee, f.name)
            assert decl_f.is_declaration
            return decl_f
        return f

    def convert_python_struct_to_llvm_ir(self, t):
        if type(t) is list:
            assert all(type(x) == type(t[0]) for x in t)
            elem_t = self.convert_python_struct_to_llvm_ir(t[0])
            return ir.ArrayType(elem_t, len(t))
        elif type(t) is tuple:
            elems_t = [self.convert_python_struct_to_llvm_ir(x) for x in t]
            return ir.LiteralStructType(elems_t)
        elif isinstance(t, (int, float)):
            return self.float_ty
        elif isinstance(t, np.ndarray):
            return self.convert_python_struct_to_llvm_ir(t.tolist())
        elif t is None:
            return ir.LiteralStructType([])

        print(type(t))
        assert(False)

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
    global _modules
    for m in _modules:
        new_mod = _try_parse_module(m)
        if new_mod is not None:
            mod_bundle.link_in(new_mod)
            _compiled_modules.add(m)

    _modules.clear()

    # Add the new module to jit engine
    _cpu_engine.opt_and_append_bin_module(mod_bundle)

    if __dumpenv is not None and __dumpenv.find("compile") != -1:
        global _binary_generation
        print("COMPILING GENERATION: {} -> {}".format(_binary_generation, _llvm_generation))

    # update binary generation
    _binary_generation = _llvm_generation


_field_count = 0
_struct_count = 0


def _convert_llvm_ir_to_ctype(t):
    if type(t) is ir.VoidType:
        return None
    elif type(t) is ir.PointerType:
        # FIXME: Can this handle void*? Do we care?
        pointee = _convert_llvm_ir_to_ctype(t.pointee)
        return ctypes.POINTER(pointee)
    elif type(t) is ir.IntType:
        # FIXME: We should consider bitwidth here
        return ctypes.c_int
    elif type(t) is ir.DoubleType:
        return ctypes.c_double
    elif type(t) is ir.FloatType:
        return ctypes.c_float
    elif type(t) is ir.LiteralStructType:
        field_list = []
        for e in t.elements:
            # llvmlite modules get _unique string only works for symbol names
            global _field_count
            uniq_name = "field_" + str(_field_count)
            _field_count += 1

            field_list.append((uniq_name, _convert_llvm_ir_to_ctype(e)))

        global _struct_count
        uniq_name = "struct_" + str(_struct_count)
        _struct_count += 1

        def __init__(self, *args, **kwargs):
            ctypes.Structure.__init__(self, *args, **kwargs)

        new_type = type(uniq_name, (ctypes.Structure,), {"__init__": __init__})
        new_type.__name__ = uniq_name
        new_type._fields_ = field_list
        assert len(new_type._fields_) == len(t.elements)
        return new_type
    elif type(t) is ir.ArrayType:
        element_type = _convert_llvm_ir_to_ctype(t.element)
        return element_type * len(t)

    print(t)
    assert(False)


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


_binaries = {}
_cpu_engine = cpu_jit_engine

class LLVMBinaryFunction:
    def __init__(self, name):
        self.__name = name
        self.__ptr = None

        # Function signature
        f = _find_llvm_function(self.__name, _compiled_modules)
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
        if _llvm_generation > _binary_generation:
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


_cpu_engine._engine.set_object_cache(_updateNativeBinaries)

# Initialize builtins
with LLVMBuilderContext() as ctx:
    builtins.setup_vxm(ctx)
