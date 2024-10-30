# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import atexit
import ctypes
import enum
import functools
import inspect
from llvmlite import ir
import numpy as np
import os
import re
import time
import weakref

from psyneulink._typing import Set
from psyneulink.core.scheduling.time import Time, TimeScale
from psyneulink.core.globals.sampleiterator import SampleIterator
from psyneulink.core.globals.utilities import ContentAddressableList
from psyneulink.core import llvm as pnlvm

from . import codegen
from . import helpers
from .debug import debug_env

__all__ = ['LLVMBuilderContext', '_modules', '_find_llvm_function']


_modules: Set[ir.Module] = set()
_all_modules: Set[ir.Module] = set()
_struct_count = 0


@atexit.register
def module_count():
    if "stat" in debug_env:
        print("Active LLVM modules: ", len(_all_modules))
        print("Total ctype structures created: ", _struct_count)


_BUILTIN_PREFIX = "__pnl_builtin_"
_builtin_intrinsics = frozenset(('pow', 'log', 'exp', 'tanh', 'coth', 'csch', 'sin', 'cos',
                                 'is_close_float', 'is_close_double',
                                 'mt_rand_init', 'philox_rand_init',
                                 'get_printf_address'))


class _node_assembly():
    def __init__(self, composition, node):
        self._comp = weakref.proxy(composition)
        self._node = node

    def __repr__(self):
        return "Node wrapper for node '{}' in composition '{}'".format(self._node, self._comp)

    def _gen_llvm_function(self, *, ctx, tags:frozenset):
        return codegen.gen_node_assembly(ctx, self._comp, self._node, tags=tags)

def _comp_cached(func):
    @functools.wraps(func)
    def wrapper(bctx, obj):
        bctx._stats[func.__name__ + "_requests"] += 1
        try:
            obj_cache = bctx._cache.setdefault(obj, dict())
        except TypeError:  # 'super()' references can't be cached
            obj_cache = None
        else:
            if func in obj_cache:
                return obj_cache[func]

        bctx._stats[func.__name__ + "_misses"] += 1
        val = func(bctx, obj)
        if obj_cache is not None:
            obj_cache[func] = val
        return val

    return wrapper


class LLVMBuilderContext:
    __current_context = None
    __uniq_counter = 0
    _llvm_generation = 0
    int32_ty = ir.IntType(32)
    default_float_ty = ir.DoubleType()
    bool_ty = ir.IntType(1)

    def __init__(self, float_ty):
        assert LLVMBuilderContext.__current_context is None
        self._modules = []
        self._cache = weakref.WeakKeyDictionary()
        self._component_param_use = weakref.WeakKeyDictionary()
        self._component_state_use = weakref.WeakKeyDictionary()

        # Supported stats are listed explicitly to catch typos
        self._stats = { "function_cache_misses":0,
                        "function_cache_requests":0,
                        "types_converted":0,
                        "get_param_struct_type_misses":0,
                        "get_state_struct_type_misses":0,
                        "get_data_struct_type_misses":0,
                        "get_input_struct_type_misses":0,
                        "get_output_struct_type_misses":0,
                        "get_param_struct_type_requests":0,
                        "get_state_struct_type_requests":0,
                        "get_data_struct_type_requests":0,
                        "get_input_struct_type_requests":0,
                        "get_output_struct_type_requests":0,
                      }
        self.float_ty = float_ty
        self.init_builtins()
        LLVMBuilderContext.__current_context = self

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

    def print_stats(self):
        def _hit_rate(reqs, misses):
            return round((reqs - misses) / reqs * 100, 2)

        print("LLVM codegen stats for context:", hex(id(self)))
        print("  Last compiled generation: {}".format(self._llvm_generation))

        req_stat = self._stats["function_cache_requests"]
        miss_stat = self._stats["function_cache_misses"]
        print("  Object function cache: {} requests, {} misses (hr: {}%)".format(req_stat, miss_stat, _hit_rate(req_stat, miss_stat)))
        for stat in ("input", "output", "param", "state", "data"):
            req_stat = self._stats["get_{}_struct_type_requests".format(stat)]
            miss_stat = self._stats["get_{}_struct_type_misses".format(stat)]
            print("  Total {} struct types requested from global context: {}, generated: {} (hr: {}%)".format(
                  stat, req_stat, miss_stat, _hit_rate(req_stat, miss_stat)))
        print("  Total python types converted by global context: {}".format(self._stats["types_converted"]))


    def __del__(self):
        if "stat" in debug_env:
            self.print_stats()

    @classmethod
    def get_current(cls):
        if cls.__current_context is None:
            return LLVMBuilderContext(cls.default_float_ty)
        return cls.__current_context

    @classmethod
    def is_active(cls):
        return cls.__current_context is not None

    @classmethod
    def clear_global(cls):
        cls.__current_context = None

    @classmethod
    def get_unique_name(cls, name: str):
        cls.__uniq_counter += 1
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        return name + '_' + str(cls.__uniq_counter)

    def init_builtins(self):
        start = time.perf_counter()
        with self as ctx:
            # Numeric
            pnlvm.builtins.setup_pnl_intrinsics(ctx)
            pnlvm.builtins.setup_csch(ctx)
            pnlvm.builtins.setup_coth(ctx)
            pnlvm.builtins.setup_tanh(ctx)
            pnlvm.builtins.setup_is_close(ctx)

            # PRNG
            pnlvm.builtins.setup_mersenne_twister(ctx)
            pnlvm.builtins.setup_philox(ctx)

            # Matrix/Vector
            pnlvm.builtins.setup_vxm(ctx)
            pnlvm.builtins.setup_vxm_transposed(ctx)
            pnlvm.builtins.setup_vec_add(ctx)
            pnlvm.builtins.setup_vec_sum(ctx)
            pnlvm.builtins.setup_mat_add(ctx)
            pnlvm.builtins.setup_vec_sub(ctx)
            pnlvm.builtins.setup_mat_sub(ctx)
            pnlvm.builtins.setup_vec_hadamard(ctx)
            pnlvm.builtins.setup_mat_hadamard(ctx)
            pnlvm.builtins.setup_vec_scalar_mult(ctx)
            pnlvm.builtins.setup_mat_scalar_mult(ctx)
            pnlvm.builtins.setup_mat_scalar_add(ctx)

        finish = time.perf_counter()

        if "time_stat" in debug_env:
            print("Time to setup PNL builtins: {}".format(finish - start))

    def get_uniform_dist_function_by_state(self, state):
        if len(state.type.pointee) == 5:
            return self.import_llvm_function("__pnl_builtin_mt_rand_double")
        elif len(state.type.pointee) == 7:
            # we have different versions based on selected FP precision
            return self.import_llvm_function("__pnl_builtin_philox_rand_{}".format(str(self.float_ty)))
        else:
            assert False, "Unknown PRNG type!"

    def get_binomial_dist_function_by_state(self, state):
        if len(state.type.pointee) == 5:
            return self.import_llvm_function("__pnl_builtin_mt_rand_binomial")
        elif len(state.type.pointee) == 7:
            return self.import_llvm_function("__pnl_builtin_philox_rand_binomial")
        else:
            assert False, "Unknown PRNG type!"

    def get_normal_dist_function_by_state(self, state):
        if len(state.type.pointee) == 5:
            return self.import_llvm_function("__pnl_builtin_mt_rand_normal")
        elif len(state.type.pointee) == 7:
            # Normal exists only for self.float_ty
            return self.import_llvm_function("__pnl_builtin_philox_rand_normal")
        else:
            assert False, "Unknown PRNG type!"

    def get_builtin(self, name: str, args=[], function_type=None):
        if name in _builtin_intrinsics:
            return self.import_llvm_function(_BUILTIN_PREFIX + name)
        if name in ('maxnum'):
            function_type = ir.FunctionType(args[0], [args[0], args[0]])
        return self.module.declare_intrinsic("llvm." + name, args, function_type)

    def create_llvm_function(self, args, component, name=None, *, return_type=ir.VoidType(), tags:frozenset=frozenset()):
        name = "_".join((str(component), *tags)) if name is None else name

        # Builtins are already unique and need to keep their special name
        func_name = name if name.startswith(_BUILTIN_PREFIX) else self.get_unique_name(name)
        func_ty = ir.FunctionType(return_type, args)
        llvm_func = ir.Function(self.module, func_ty, name=func_name)
        llvm_func.attributes.add('argmemonly')
        for a in llvm_func.args:
            if isinstance(a.type, ir.PointerType):
                a.attributes.add('nonnull')

        metadata = self.get_debug_location(llvm_func, component)
        scope = dict(metadata.operands)["scope"]
        llvm_func.set_metadata("dbg", scope)

        # Create entry block
        block = llvm_func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.debug_metadata = metadata

        return builder

    def gen_llvm_function(self, obj, *, tags:frozenset) -> ir.Function:
        obj_cache = self._cache.setdefault(obj, dict())

        self._stats["function_cache_requests"] += 1
        if tags not in obj_cache:
            self._stats["function_cache_misses"] += 1
            with self:
                obj_cache[tags] = obj._gen_llvm_function(ctx=self, tags=tags)
                self.check_used_params(obj, tags=tags)

        return obj_cache[tags]

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

    def get_random_state_ptr(self, builder, component, state, params):
        random_state_ptr = self.get_param_or_state_ptr(builder, component, "random_state", state_struct_ptr=state)

        # Used seed is the last member of both MT state and Philox state
        seed_idx = len(random_state_ptr.type.pointee) - 1
        used_seed_ptr = builder.gep(random_state_ptr, [self.int32_ty(0), self.int32_ty(seed_idx)])
        used_seed = builder.load(used_seed_ptr)

        seed_ptr = self.get_param_or_state_ptr(builder, component, "seed", param_struct_ptr=params)
        new_seed = pnlvm.helpers.load_extract_scalar_array_one(builder, seed_ptr)
        # FIXME: The seed should ideally be integer already.
        #        However, it can be modulated and we don't support
        #        passing integer values as computed results.
        new_seed = builder.fptoui(new_seed, used_seed.type)

        seeds_cmp = builder.icmp_unsigned("!=", used_seed, new_seed)
        with builder.if_then(seeds_cmp, likely=False):
            if seed_idx == 4:
                reseed_f = self.get_builtin("mt_rand_init")
            elif seed_idx == 6:
                reseed_f = self.get_builtin("philox_rand_init")
            else:
                assert False, "Unknown PRNG type!"

            builder.call(reseed_f, [random_state_ptr, new_seed])

        return random_state_ptr

    def get_param_or_state_ptr(self, builder, component, param, *, param_struct_ptr=None, state_struct_ptr=None, history=0):
        param_name = getattr(param, "name", param)
        param = None
        state = None

        if param_name in component.llvm_param_ids:
            assert param_struct_ptr is not None, "Can't get param ptr for: {}".format(param_name)
            self._component_param_use.setdefault(component, set()).add(param_name)
            param = helpers.get_param_ptr(builder, component, param_struct_ptr, param_name)

        if param_name in component.llvm_state_ids:
            assert state_struct_ptr is not None, "Can't get state ptr for: {}".format(param_name)
            self._component_state_use.setdefault(component, set()).add(param_name)
            state = helpers.get_state_ptr(builder, component, state_struct_ptr, param_name, history)

        if param is not None and state is not None:
            return (param, state)

        return param or state

    def get_state_space(self, builder, component, state_ptr, param):
        param_name = getattr(param, "name", param)
        self._component_state_use.setdefault(component, set()).add(param_name)
        return helpers.get_state_space(builder, component, state_ptr, param_name)

    def check_used_params(self, component, *, tags:frozenset):
        """
        This function checks that parameters included in the compiled structures are used in compiled code.

        If the assertion in this function triggers the parameter name should be added to the parameter
        block list in the Component class.
        """

        # Skip the check if the parameter use is not tracked. Some components (like node wrappers)
        # don't even have parameters.
        if component not in self._component_state_use and component not in self._component_param_use:
            return

        # Skip the check for variant functions
        if len(tags) != 0:
            return

        component_param_ids = set(component.llvm_param_ids)
        component_state_ids = set(component.llvm_state_ids)

        used_param_ids = self._component_param_use.get(component, set())
        used_state_ids = self._component_state_use.get(component, set())

        # initializers are  only used in "reset" variants
        initializers = {p.initializer for p in component.parameters}

        # has_initializers is only used in "reset" variants
        initializers.add('has_initializers')

        # 'termination_measure" is only used in "is_finished" variant
        used_param_ids.add('termination_measure')
        used_state_ids.add('termination_measure')

        # 'num_trials_per_estimate' is only used in "evaluate" variants
        if hasattr(component, 'evaluate_agent_rep'):
            used_param_ids.add('num_trials_per_estimate')

        unused_param_ids = component_param_ids - used_param_ids - initializers
        unused_state_ids = component_state_ids - used_state_ids

        assert len(unused_param_ids) == 0 and len(unused_state_ids) == 0, \
            "Compiled component '{}'(tags: {}) unused parameters: {}, state: {}".format(component, list(tags), unused_param_ids, unused_state_ids)

    @staticmethod
    def get_debug_location(func: ir.Function, component):
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

    @staticmethod
    def update_debug_loc_position(di_loc: ir.DIValue, line:int, column:int):
        di_func = dict(di_loc.operands)["scope"]

        return di_loc.parent.add_debug_info("DILocation", {
            "line": line, "column": column, "scope": di_func,
        })

    @_comp_cached
    def get_input_struct_type(self, component):
        if hasattr(component, '_get_input_struct_type'):
            return component._get_input_struct_type(self)

        default_var = component.defaults.variable
        return self.convert_python_struct_to_llvm_ir(default_var)

    @_comp_cached
    def get_output_struct_type(self, component):
        if hasattr(component, '_get_output_struct_type'):
            return component._get_output_struct_type(self)

        default_val = component.defaults.value
        return self.convert_python_struct_to_llvm_ir(default_val)

    @_comp_cached
    def get_param_struct_type(self, component):
        if hasattr(component, '_get_param_struct_type'):
            return component._get_param_struct_type(self)

        def _param_struct(p):
            val = p.get(None)   # this should use defaults
            if hasattr(val, "_get_compilation_params") or \
               hasattr(val, "_get_param_struct_type"):
                return self.get_param_struct_type(val)
            if isinstance(val, ContentAddressableList):
                return ir.LiteralStructType(self.get_param_struct_type(x) for x in val)
            elif p.name == 'matrix':   # Flatten matrix
                val = np.asfarray(val).flatten()
            elif p.name == 'num_trials_per_estimate':  # Should always be int
                val = np.int32(0) if val is None else np.int32(val)
            elif np.ndim(val) == 0 and component._is_param_modulated(p):
                val = [val]   # modulation adds array wrap
            return self.convert_python_struct_to_llvm_ir(val)

        elements = map(_param_struct, component._get_compilation_params())
        return ir.LiteralStructType(elements)

    @_comp_cached
    def get_state_struct_type(self, component):
        if hasattr(component, '_get_state_struct_type'):
            return component._get_state_struct_type(self)

        def _state_struct(p):
            val = p.get(None)   # this should use defaults
            if hasattr(val, "_get_compilation_state") or \
               hasattr(val, "_get_state_struct_type"):
                return self.get_state_struct_type(val)
            if isinstance(val, ContentAddressableList):
                return ir.LiteralStructType(self.get_state_struct_type(x) for x in val)
            if p.name == 'matrix':   # Flatten matrix
                val = np.asfarray(val).flatten()
            struct = self.convert_python_struct_to_llvm_ir(val)
            return ir.ArrayType(struct, p.history_min_length + 1)

        elements = map(_state_struct, component._get_compilation_state())
        return ir.LiteralStructType(elements)

    @_comp_cached
    def get_data_struct_type(self, component):
        if hasattr(component, '_get_data_struct_type'):
            return component._get_data_struct_type(self)

        return ir.LiteralStructType([])

    def get_node_assembly(self, composition, node):
        cache = getattr(composition, '_node_assemblies', None)
        if cache is None:
            cache = weakref.WeakKeyDictionary()
            setattr(composition, '_node_assemblies', cache)
        return cache.setdefault(node, _node_assembly(composition, node))

    def convert_python_struct_to_llvm_ir(self, t):
        self._stats["types_converted"] += 1
        if t is None:
            return ir.LiteralStructType([])

        elif isinstance(t, (list, tuple)):
            elems_t = [self.convert_python_struct_to_llvm_ir(x) for x in t]
            if len(elems_t) > 0 and all(x == elems_t[0] for x in elems_t):
                return ir.ArrayType(elems_t[0], len(elems_t))

            return ir.LiteralStructType(elems_t)

        elif isinstance(t, enum.Enum):
            # FIXME: Consider enums of non-int type
            assert all(round(x.value) == x.value for x in type(t))
            return self.int32_ty

        elif isinstance(t, (int, float, np.floating)):
            return self.float_ty

        elif isinstance(t, np.integer):
            # Python 'int' is handled above as it is the default type for '0'
            return ir.IntType(t.nbytes * 8)

        elif isinstance(t, np.ndarray):
            # 0d uint32 values were likely created from enums (above) and are
            # observed here after compilation sync.
            # Avoid silent promotion to float (via Python's builtin int-type)
            if t.ndim == 0 and t.dtype == np.uint32:
                return self.convert_python_struct_to_llvm_ir(t.reshape(1)[0])
            return self.convert_python_struct_to_llvm_ir(t.tolist())

        elif isinstance(t, np.random.RandomState):
            return pnlvm.builtins.get_mersenne_twister_state_struct(self)

        elif isinstance(t, np.random.Generator):
            assert isinstance(t.bit_generator, np.random.Philox)
            return pnlvm.builtins.get_philox_state_struct(self)

        elif isinstance(t, Time):
            return ir.ArrayType(self.int32_ty, len(TimeScale))

        elif isinstance(t, SampleIterator):
            if isinstance(t.generator, list):
                return ir.ArrayType(self.float_ty, len(t.generator))

            # Generic iterator is {start, increment, count}
            return ir.LiteralStructType((self.float_ty, self.float_ty, self.int32_ty))

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
    orig_args = function.type.pointee.args

    # remove indices if this is grid_evaluate_ranged
    is_grid_ranged = len(orig_args) == 8 and isinstance(orig_args[2], ir.IntType)
    if is_grid_ranged:
        orig_args = orig_args[:2] + orig_args[4:]

    wrapper_type = ir.FunctionType(ir.VoidType(), [*orig_args, ir.IntType(32)])
    kernel_func = ir.Function(module, wrapper_type, function.name + "_cuda_kernel")
    # Add kernel mark metadata
    module.add_named_metadata("nvvm.annotations", [kernel_func, "kernel", ir.IntType(32)(1)])

    # Start the function
    block = kernel_func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Calculate global id of a thread in x dimension
    intrin_ty = ir.FunctionType(ir.IntType(32), [])
    tid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.tid.x")
    ntid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.ntid.x")
    ctaid_x_f = ir.Function(module, intrin_ty, "llvm.nvvm.read.ptx.sreg.ctaid.x")

    # Number of threads per block
    ntid = builder.call(ntid_x_f, [])
    # Thread ID in block
    tid = builder.call(tid_x_f, [])

    global_id = builder.mul(builder.call(ctaid_x_f, []), ntid)
    global_id = builder.add(global_id, tid)

    # Index all pointer arguments. Ignore the thread count argument
    args = list(kernel_func.args)[:-1]
    indexed_args = []

    # pointer args do not alias
    for a in args:
        if isinstance(a.type, ir.PointerType):
            a.attributes.add('noalias')

    def _upload_to_shared(b, ptr, length, name):
        shared = ir.GlobalVariable(module, ptr.type.pointee,
                                   name=function.name + "_shared_" + name,
                                   addrspace=3)
        shared.alignment = 128
        shared.linkage = "internal"
        shared_ptr = b.addrspacecast(shared, shared.type.pointee.as_pointer())

        char_ptr_ty = ir.IntType(8).as_pointer()
        bool_ty = ir.IntType(1)

        ptr_src = b.bitcast(ptr, char_ptr_ty)
        ptr_dst = b.bitcast(shared_ptr, char_ptr_ty)

        # the params for objectsize are:
        # * obj pointer,
        # * 0 on unknown size instead of -1,
        # * NULL ptr is unknown size
        # * evaluate size at runtime
        obj_size_ty = ir.FunctionType(ir.IntType(32), [char_ptr_ty, bool_ty, bool_ty, bool_ty])
        obj_size_f = module.declare_intrinsic("llvm.objectsize.i32", [], obj_size_ty)
        obj_size = b.call(obj_size_f, [ptr_dst, bool_ty(1), bool_ty(0), bool_ty(0)])
        obj_size = b.mul(obj_size, length)

        if "unaligned_copy" not in debug_env:
            copy_ty = ir.IntType(32)
            copy_ptr_ty = copy_ty.as_pointer()
            copy_bytes = copy_ty.width // 8
            obj_size = builder.add(obj_size, obj_size.type(copy_bytes - 1))
            obj_size = builder.udiv(obj_size, obj_size.type(copy_bytes))
            ptr_src = builder.bitcast(ptr, copy_ptr_ty)
            ptr_dst = builder.bitcast(shared_ptr, copy_ptr_ty)


        # copy data using as many threads as available in thread group
        with helpers.for_loop(b, tid, obj_size, ntid, id="copy_" + name) as (b1, i):
            src = b1.gep(ptr_src, [i])
            dst = b1.gep(ptr_dst, [i])
            b1.store(b1.load(src), dst)

        sync_threads_ty = ir.FunctionType(ir.VoidType(), [])
        sync_threads = module.declare_intrinsic("llvm.nvvm.barrier0", [], sync_threads_ty)
        builder.call(sync_threads, [])

        return b, shared_ptr

    if is_grid_ranged and "cuda_no_shared" not in debug_env:
        one = ir.IntType(32)(1)

        # Upload static RO structures
        builder, args[0] = _upload_to_shared(builder, args[0], one, "params")
        builder, args[1] = _upload_to_shared(builder, args[1], one, "state")
        builder, args[4] = _upload_to_shared(builder, args[4], one, "data")

        # TODO: Investigate benefit of uplaoding dynamic RO structures to
        #       shared memory (e.g. inputs)

    # Check global id and exit if we're over
    should_quit = builder.icmp_unsigned(">=", global_id, kernel_func.args[-1])
    with builder.if_then(should_quit):
        builder.ret_void()

    # If we're calling ranged search there are no offsets
    if is_grid_ranged:
        next_id = builder.add(global_id, global_id.type(1))
        call_args = args[:2] + [global_id, next_id] + args[2:]
        builder.call(decl_f, call_args)
        builder.ret_void()
        return module

    # Runs need special handling. data_in and data_out are one dimensional,
    # but hold entries for all parallel invocations.
    # comp_state, comp_params, comp_data, comp_in, comp_out, #trials, #inputs
    is_comp_run = len(args) == 7
    if is_comp_run:
        runs_count = builder.load(args[5])
        input_count = builder.load(args[6])

    for i, arg in enumerate(args):
        if isinstance(arg.type, ir.PointerType):
            offset = global_id
            if is_comp_run:
                # #inputs needs to be the same for comp run
                if i == 6:
                    offset = ir.IntType(32)(0)
                # data arrays need special handling
                elif i == 4:  # data_out
                    offset = builder.mul(global_id, runs_count)
                elif i == 3:  # data_in
                    offset = builder.mul(global_id, input_count)

            arg = builder.gep(arg, [offset])

        indexed_args.append(arg)

    builder.call(decl_f, indexed_args)
    builder.ret_void()

    return module


@functools.lru_cache(maxsize=128)
def _convert_llvm_ir_to_ctype(t: ir.Type):
    type_t = type(t)

    if type_t is ir.VoidType:
        return None
    elif type_t is ir.IntType:
        if t.width == 1:
            return ctypes.c_bool
        elif t.width == 8:
            return ctypes.c_uint8
        elif t.width == 16:
            return ctypes.c_uint16
        elif t.width == 32:
            return ctypes.c_uint32
        elif t.width == 64:
            return ctypes.c_uint64
        else:
            assert False, "Unknown integer type: {}".format(type_t)
    elif type_t is ir.DoubleType:
        return ctypes.c_double
    elif type_t is ir.FloatType:
        return ctypes.c_float
    elif type_t is ir.HalfType:
        # There's no half type in ctypes. Use uint16 instead.
        # User will need to do the necessary casting.
        return ctypes.c_uint16
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

@functools.lru_cache(maxsize=16)
def _convert_llvm_ir_to_dtype(t: ir.Type):

    if isinstance(t, ir.IntType):
        if t.width == 8:
            return np.uint8().dtype

        elif t.width == 16:
            return np.uint16().dtype

        elif t.width == 32:
            return np.uint32().dtype

        elif t.width == 64:
            return np.uint64().dtype

        else:
            assert False, "Unsupported integer type: {}".format(type(t))

    elif isinstance(t, ir.DoubleType):
        return np.float64().dtype

    elif isinstance(t, ir.FloatType):
        return np.float32().dtype

    elif isinstance(t, ir.HalfType):
        return np.float16().dtype

    elif isinstance(t, ir.ArrayType):
        element_type = _convert_llvm_ir_to_dtype(t.element)

        # Create multidimensional array instead of nesting
        if element_type.subdtype is not None:
            element_type, shape = element_type.subdtype
        else:
            shape = ()

        ret_t = np.dtype((element_type, (len(t),) + shape))

    elif isinstance(t, ir.LiteralStructType):
        field_list = []
        for i, e in enumerate(t.elements):
            field_list.append(("field_" + str(i), _convert_llvm_ir_to_dtype(e)))

        ret_t = np.dtype(field_list, align=True)
    else:
        assert False, "Don't know how to convert LLVM type to dtype: {}".format(t)

    return ret_t
