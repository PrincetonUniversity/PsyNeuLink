# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

from llvmlite import binding

from .builder_context import LLVMBuilderContext, _find_llvm_function, _gen_cuda_kernel_wrapper_module
from .builtins import _generate_cpu_builtins_module
from .debug import debug_env

try:
    if "cuda" in debug_env:
        import pycuda
        # Do not continue if the version is too old
        if pycuda.VERSION[0] >= 2018:
            import pycuda.driver
            # pyCUDA needs to be built against 5.5+ to enable Linker
            if pycuda.driver.get_version()[0] > 5:
                from pycuda import autoinit as pycuda_default
                import pycuda.compiler
                ptx_enabled = True
            else:
                raise UserWarning("CUDA driver too old (need 6+): " + str(pycuda.driver.get_version()))
        else:
            raise UserWarning("pycuda too old (need 2018+): " + str(pycuda.VERSION))
    else:
        ptx_enabled = False
except Exception as e:
    print("WARNING: Failed to enable CUDA/PTX:", e)
    ptx_enabled = False


__all__ = ['cpu_jit_engine', 'ptx_enabled']

if ptx_enabled:
    __all__.append('ptx_jit_engine')


# Compiler binding
__initialized = False


def _binding_initialize():
    global __initialized
    if not __initialized:
        binding.initialize()
        if not ptx_enabled:
            # native == currently running CPU. ASM printer includes opcode emission
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
        else:
            binding.initialize_all_targets()
            binding.initialize_all_asmprinters()

        __initialized = True


def _cpu_jit_constructor():
    _binding_initialize()

    # PassManagerBuilder can be shared
    __pass_manager_builder = binding.PassManagerBuilder()
    __pass_manager_builder.loop_vectorize = False
    __pass_manager_builder.slp_vectorize = False
    __pass_manager_builder.opt_level = 2

    __cpu_features = binding.get_host_cpu_features().flatten()
    __cpu_name = binding.get_host_cpu_name()

    # Create compilation target, use default triple
    __cpu_target = binding.Target.from_default_triple()
    # FIXME: reloc='static' is needed to avoid crashes on win64
    # see: https://github.com/numba/llvmlite/issues/457
    __cpu_target_machine = __cpu_target.create_target_machine(cpu=__cpu_name, features=__cpu_features, opt=2, reloc='static')

    __cpu_pass_manager = binding.ModulePassManager()
    __cpu_target_machine.add_analysis_passes(__cpu_pass_manager)
    __pass_manager_builder.populate(__cpu_pass_manager)

    # And an execution engine with a builtins backing module
    builtins_module = _generate_cpu_builtins_module(LLVMBuilderContext.float_ty)
    if "llvm" in debug_env:
        with open(builtins_module.name + '.parse.ll', 'w') as dump_file:
            dump_file.write(str(builtins_module))

    __backing_mod = binding.parse_assembly(str(builtins_module))

    __cpu_jit_engine = binding.create_mcjit_compiler(__backing_mod, __cpu_target_machine)
    return __cpu_jit_engine, __cpu_pass_manager, __cpu_target_machine


def _ptx_jit_constructor():
    _binding_initialize()

    # PassManagerBuilder can be shared
    __pass_manager_builder = binding.PassManagerBuilder()
    __pass_manager_builder.inlining_threshold = 99999  # Inline all function calls
    __pass_manager_builder.loop_vectorize = True
    __pass_manager_builder.slp_vectorize = True
    __pass_manager_builder.opt_level = 3  # Most aggressive optimizations

    # Use default device
    # TODO: Add support for multiple devices
    __compute_capability = pycuda_default.device.compute_capability()
    __ptx_sm = "sm_{}{}".format(__compute_capability[0], __compute_capability[1])
    # Create compilation target, use 64bit triple
    __ptx_target = binding.Target.from_triple("nvptx64-nvidia-cuda")
    __ptx_target_machine = __ptx_target.create_target_machine(cpu=__ptx_sm, opt=3, codemodel='small')

    __ptx_pass_manager = binding.ModulePassManager()
    __ptx_target_machine.add_analysis_passes(__ptx_pass_manager)
#    __pass_manager_builder.populate(__ptx_pass_manager)

    return __ptx_pass_manager, __ptx_target_machine


def _try_parse_module(module):
    if "llvm" in debug_env:
        with open(module.name + '.parse.ll', 'w') as dump_file:
            dump_file.write(str(module))

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


class jit_engine:
    def __init__(self):
        self._jit_engine = None
        self._jit_pass_manager = None
        self._target_machine = None
        self.__mod = None
        # Add an extra reference to make sure it's not destroyed before
        # instances of jit_engine
        self.__debug_env = debug_env

        # Track few statistics:
        self.__optimized_modules = 0
        self.__linked_modules = 0
        self.__parsed_modules = 0

    def __del__(self):
        if "stat" in self.__debug_env:
            s = type(self).__name__
            print("Total optimized modules in '{}': {}".format(s, self.__optimized_modules))
            print("Total linked modules in '{}': {}".format(s, self.__linked_modules))
            print("Total parsed modules in '{}': {}".format(s, self.__parsed_modules))

    def opt_and_add_bin_module(self, module):
        self._pass_manager.run(module)
        if "opt" in self.__debug_env:
            with open(self.__class__.__name__ + '-' + str(self.__optimized_modules) + '.opt.ll', 'w') as dump_file:
                dump_file.write(str(module))

        # This prints generated x86 assembly
        if "isa" in self.__debug_env:
            with open(self.__class__.__name__ + '-' + str(self.__optimized_modules) + '.S', 'w') as dump_file:
                dump_file.write(self._target_machine.emit_assembly(module))

        self._engine.add_module(module)
        self._engine.finalize_object()
        self.__optimized_modules += 1

    def _remove_bin_module(self, module):
        if module is not None:
            self._engine.remove_module(module)

    def opt_and_append_bin_module(self, module):
        mod_name = module.name
        if self.__mod is None:
            self.__mod = module
        else:
            self._remove_bin_module(self.__mod)
            # Linking here invalidates 'module'
            self.__mod.link_in(module)
            self.__linked_modules += 1

        if "llvm" in debug_env:
            with open(mod_name + '.linked.ll', 'w') as dump_file:
                dump_file.write(str(self.__mod))

        self.opt_and_add_bin_module(self.__mod)

    def clean_module(self):
        self._remove_bin_module(self.__mod)
        self.__mod = None

    @property
    def _engine(self):
        if self._jit_engine is None:
            self._init()

        return self._jit_engine

    @property
    def _pass_manager(self):
        if self._jit_pass_manager is None:
            self._init()

        return self._jit_pass_manager

    # Unfortunately, this needs to be done for every jit_engine.
    # Liking step in opt_and_add_bin_module invalidates 'mod_bundle',
    # so it can't be linked mutliple times (in multiple engines).
    def compile_modules(self, modules, compiled_modules):
        # Parse generated modules and link them
        mod_bundle = binding.parse_assembly("")
        for m in modules:
            new_mod = _try_parse_module(m)
            self.__parsed_modules += 1
            if new_mod is not None:
                mod_bundle.link_in(new_mod)
                mod_bundle.name = m.name  # Set the name of the last module
                compiled_modules.add(m)

        self.opt_and_append_bin_module(mod_bundle)


class cpu_jit_engine(jit_engine):

    def __init__(self, object_cache=None):
        super().__init__()
        self._object_cache = object_cache

    def _init(self):
        assert self._jit_engine is None
        assert self._jit_pass_manager is None
        assert self._target_machine is None

        self._jit_engine, self._jit_pass_manager, self._target_machine = _cpu_jit_constructor()
        if self._object_cache is not None:
            self._jit_engine.set_object_cache(self._object_cache)


_ptx_builtin_source = """
__device__ {type} __pnl_builtin_log({type} a) {{ return log(a); }}
__device__ {type} __pnl_builtin_exp({type} a) {{ return exp(a); }}
__device__ {type} __pnl_builtin_pow({type} a, {type} b) {{ return pow(a, b); }}
"""


class ptx_jit_engine(jit_engine):
    class cuda_engine():
        def __init__(self, tm):
            self._modules = {}
            self._target_machine = tm

            # -dc option tells the compiler that the code will be used for linking
            self._generated_builtins = pycuda.compiler.compile(_ptx_builtin_source.format(type=str(LLVMBuilderContext.float_ty)), target='cubin', options=['-dc'])

        def set_object_cache(cache):
            pass

        def add_module(self, module):
            try:
                # LLVM can't produce CUBIN for some reason
                ptx = self._target_machine.emit_assembly(module)
                mod = pycuda.compiler.DynamicModule()
                mod.add_data(self._generated_builtins, pycuda.driver.jit_input_type.CUBIN, "builtins.cubin")
                mod.add_data(ptx.encode(), pycuda.driver.jit_input_type.PTX, module.name + ".ptx")
                ptx_mod = mod.link()

            except Exception as e:
                print("FAILED to generate PTX module:", e)
                print(ptx)
                return None

            self._modules[module] = ptx_mod

        def finalize_object(self):
            pass

        def remove_module(self, module):
            self._modules.pop(module, None)

        def _find_kernel(self, name):
            function = None
            for m in self._modules.values():
                try:
                    function = m.get_function(name)
                except pycuda._driver.LogicError:
                    pass
            return function

    def __init__(self, object_cache=None):
        super().__init__()
        self._object_cache = object_cache

    def _init(self):
        assert self._jit_engine is None
        assert self._jit_pass_manager is None
        assert self._target_machine is None

        self._jit_pass_manager, self._target_machine = _ptx_jit_constructor()
        self._jit_engine = ptx_jit_engine.cuda_engine(self._target_machine)

    def get_kernel(self, name):
        kernel = self._engine._find_kernel(name + "_cuda_kernel")
        if kernel is None:
            function = _find_llvm_function(name)
            wrapper_mod = _gen_cuda_kernel_wrapper_module(function)
            self.compile_modules([wrapper_mod], set())
            kernel = self._engine._find_kernel(name + "_cuda_kernel")

        return kernel
