# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

import llvmlite
import llvmlite.binding as binding
import packaging.version as version
import sys
import time
import warnings

from .builder_context import LLVMBuilderContext, _find_llvm_function, _gen_cuda_kernel_wrapper_module
from .builtins import _generate_cpu_builtins_module
from .debug import debug_env

try:
    import pycuda
    # Do not continue if the version is too old
    if pycuda.VERSION[0] < 2018:
        raise UserWarning("pycuda too old (need 2018+): " + str(pycuda.VERSION))
    import pycuda.driver
    # pyCUDA needs to be built against 6+ to enable Linker
    if pycuda.driver.get_version()[0] < 6:
        raise UserWarning("CUDA driver too old (need 6+): " + str(pycuda.driver.get_version()))

    from pycuda import autoinit as pycuda_default
    import pycuda.compiler
    assert pycuda_default.context is not None
    pycuda_default.context.set_cache_config(pycuda.driver.func_cache.PREFER_L1)
    ptx_enabled = True
    if "cuda-check" in debug_env:
        print("PsyNeuLink: CUDA backend enabled!")
except Exception as e:
    if "cuda-check" in debug_env:
        warnings.warn("Failed to enable CUDA/PTX: {}".format(e))
    ptx_enabled = False


__all__ = ['cpu_jit_engine', 'ptx_enabled']

if ptx_enabled:
    __all__.append('ptx_jit_engine')


# Compiler binding
__initialized = False

# llvmlite>=0.44 introduced new pass manager, but it was broken on windows[1].
# The new pass manager is, however, required starting llvmlite-0.45.0.[2]
# [1] https://github.com/numba/llvmlite/issues/1078
# [2] https://github.com/numba/llvmlite/pull/1092
__required_version_for_new_pass_manager = '0.45.0' if sys.platform == "win32" else '0.44.0'
__cpu_use_new_pass_manager = version.parse(llvmlite.__version__) >= version.parse(__required_version_for_new_pass_manager)
__gpu_use_new_pass_manager = version.parse(llvmlite.__version__) >= version.parse('0.44.0')

def _binding_initialize():
    global __initialized
    if not __initialized:
        # calling binding.initalize() is not needed in later versions
        # of llvmlite, and throws exception in >=0.45.0
        if version.parse(llvmlite.__version__) < version.parse('0.45.0'):
            binding.initialize()

        if not ptx_enabled:
            # native == currently running CPU. ASM printer includes opcode emission
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
        else:
            binding.initialize_all_targets()
            binding.initialize_all_asmprinters()

        __initialized = True


def _new_pass_builder(target_machine, opt_level, extra_opts = dict()):
    pto = binding.create_pipeline_tuning_options(speed_level=opt_level)
    pto.loop_vectorization = opt_level != 0
    pto.slp_vectorization = opt_level != 0

    assert set(extra_opts.keys()).issubset(dir(pto)), dir(pto)

    for key, value in extra_opts.items():
        setattr(pto, key, value)

    pass_builder = binding.create_pass_builder(target_machine, pto)

    return pass_builder

def _old_pass_manager(target_machine, opt_level, extra_opts = dict()):
    pass_manager_builder = binding.PassManagerBuilder()
    pass_manager_builder.loop_vectorize = opt_level != 0
    pass_manager_builder.slp_vectorize = opt_level != 0
    pass_manager_builder.opt_level = opt_level

    assert set(extra_opts.keys()).issubset(dir(pass_manager_builder))

    for key, value in extra_opts.items():
        setattr(pass_manager_builder, key, value)

    # Create module pass manager and populate it with analysis and opt passes
    pass_manager = binding.ModulePassManager()
    target_machine.add_analysis_passes(pass_manager)
    pass_manager_builder.populate(pass_manager)

    return pass_manager

def _cpu_jit_constructor():
    _binding_initialize()

    opt_level = int(debug_env.get('opt', 2))

    # Create compilation target, use triple from current process
    # FIXME: reloc='static' is needed to avoid crashes on win64
    # see: https://github.com/numba/llvmlite/issues/457
    cpu_target = binding.Target.from_triple(binding.get_process_triple())
    cpu_target_machine = cpu_target.create_target_machine(cpu=binding.get_host_cpu_name(),
                                                          features=binding.get_host_cpu_features().flatten(),
                                                          opt=opt_level,
                                                          reloc='static')

    if __cpu_use_new_pass_manager:
        pass_manager = None
        pass_builder = _new_pass_builder(cpu_target_machine, opt_level)
    else:
        pass_manager = _old_pass_manager(cpu_target_machine, opt_level)
        pass_builder = None

    # And an execution engine with a builtins backing module
    builtins_module = _generate_cpu_builtins_module(LLVMBuilderContext.get_current().float_ty)

    backing_mod = binding.parse_assembly(str(builtins_module))
    backing_mod.verify()

    if "dump-llvm-gen" in debug_env:
        with open(builtins_module.name + '.generated.ll', 'w') as dump_file:
            dump_file.write(str(backing_mod))

    cpu_jit_engine = binding.create_mcjit_compiler(backing_mod, cpu_target_machine)

    return cpu_jit_engine, cpu_target_machine, pass_manager, pass_builder


def _ptx_jit_constructor():
    _binding_initialize()

    opt_level = int(debug_env.get('opt', 2))

    # Use default device
    # TODO: Add support for multiple devices
    compute_capability = pycuda_default.device.compute_capability()
    ptx_sm = "sm_{}{}".format(compute_capability[0], compute_capability[1])

    # Create compilation target, use 64bit triple
    ptx_target = binding.Target.from_triple("nvptx64-nvidia-cuda")
    ptx_target_machine = ptx_target.create_target_machine(cpu=ptx_sm, opt=opt_level)

    # The threshold of '64' is empirically selected on GF 3050
    extra_opts = {'size_level' : 1, 'inlining_threshold': 64}

    if __gpu_use_new_pass_manager:
        # Inlining threshold is not supported until llvmlite-0.45.0
        # [1] https://github.com/numba/llvmlite/commit/ccfbf78bd838fef886a1ec9fc4a353ec952fa035
        if version.parse(llvmlite.__version__) < version.parse('0.45.0'):
            extra_opts.pop('inlining_threshold', None)

        # size_level check is mismatched between Python and C++ until 0.46 [1]
        # even then size_level is only allowed for opt_level==2
        # [1] https://github.com/numba/llvmlite/issues/1306
        if version.parse(llvmlite.__version__) < version.parse('0.46.0') or opt_level != 2:
            extra_opts.pop('size_level', None)

        ptx_pass_builder = _new_pass_builder(ptx_target_machine, opt_level, extra_opts)
        ptx_pass_manager = None
    else:
        ptx_pass_manager = _old_pass_manager(ptx_target_machine, opt_level, extra_opts)
        ptx_pass_builder = None

    return ptx_target_machine, ptx_pass_manager, ptx_pass_builder


def _try_parse_module(module):
    module_text_ir = str(module)

    if "dump-llvm-gen" in debug_env:
        with open(module.name + '.generated.ll', 'w') as dump_file:
            dump_file.write(module_text_ir)

    # IR module is not the same as binding module.
    # "assembly" in this case is LLVM IR assembly.
    # This is intentional design decision to ease
    # compatibility between LLVM versions.
    try:
        mod = binding.parse_assembly(module_text_ir)
        mod.verify()
    except Exception as e:
        print("ERROR: llvm parsing failed: {}".format(e))
        mod = None

    return mod


class jit_engine:
    def __init__(self):
        self._jit_engine = None
        self._jit_pass_manager = None
        self._jit_pass_builder = None
        self._jit_target_machine = None
        self.__mod = None

        # Add an extra reference to make sure it's not destroyed before
        # all instances of jit_engine
        self.__debug_env = debug_env

        self.staged_modules = set()
        self.compiled_modules = set()

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
        start = time.perf_counter()
        self._pass_manager.run(module, self._pass_builder)
        finish = time.perf_counter()

        if "time_stat" in debug_env:
            print("Time to optimize LLVM module bundle '{}': {}".format(module.name, finish - start))

        if "dump-llvm-opt" in self.__debug_env:
            with open(self.__class__.__name__ + '-' + str(self.__optimized_modules) + '.opt.ll', 'w') as dump_file:
                dump_file.write(str(module))

        # This prints generated x86 assembly
        if "dump-asm" in self.__debug_env:
            with open(self.__class__.__name__ + '-' + str(self.__optimized_modules) + '.S', 'w') as dump_file:
                dump_file.write(self._target_machine.emit_assembly(module))

        start = time.perf_counter()
        self._engine.add_module(module)
        self._engine.finalize_object()
        finish = time.perf_counter()

        if "time_stat" in debug_env:
            print("Time to finalize LLVM module bundle '{}': {}".format(module.name, finish - start))

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

        if "dump-llvm-gen" in debug_env:
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
    def _target_machine(self):
        if self._jit_target_machine is None:
            self._init()

        return self._jit_target_machine

    @property
    def _pass_manager(self):
        # use new pass manager
        if self._pass_builder is not None:
            return self._pass_builder.getModulePassManager()

        # use old pass manager
        if self._jit_pass_manager is None:
            self._init()

        return self._jit_pass_manager

    @property
    def _pass_builder(self):
        if self._jit_pass_builder is None and self._jit_pass_manager is None:
            self._init()

        return self._jit_pass_builder

    def stage_compilation(self, modules):
        self.staged_modules |= modules

    # Unfortunately, this needs to be done for every jit_engine.
    # Linking step in opt_and_add_bin_module invalidates 'mod_bundle',
    # so it can't be linked multiple times (in multiple engines).
    # These modules are still using 'unknown-unknown-unknown' triple
    def compile_staged(self):
        # Parse generated modules and link them
        mod_bundle = binding.parse_assembly("")

        while self.staged_modules:
            m = self.staged_modules.pop()

            start = time.perf_counter()
            new_mod = _try_parse_module(m)
            finish = time.perf_counter()

            if "time_stat" in debug_env:
                print("Time to parse LLVM modules '{}': {}".format(m.name, finish - start))

            self.__parsed_modules += 1
            if new_mod is not None:
                mod_bundle.link_in(new_mod)
                mod_bundle.name = m.name  # Set the name of the last module
                self.compiled_modules.add(m)

        self.opt_and_append_bin_module(mod_bundle)


class cpu_jit_engine(jit_engine):

    def __init__(self, object_cache=None):
        super().__init__()
        self._object_cache = object_cache

    def _init(self):
        assert self._jit_engine is None
        assert self._jit_pass_manager is None
        assert self._jit_pass_builder is None
        assert self._jit_target_machine is None

        self._jit_engine, self._jit_target_machine, self._jit_pass_manager, self._jit_pass_builder = _cpu_jit_constructor()
        if self._object_cache is not None:
            self._jit_engine.set_object_cache(self._object_cache)


# FIXME: Get device side printf pointer
_ptx_builtin_source = """
__device__ {type} __pnl_builtin_sin({type} a) {{ return sin(a); }}
__device__ {type} __pnl_builtin_cos({type} a) {{ return cos(a); }}
__device__ {type} __pnl_builtin_log({type} a) {{ return log(a); }}
__device__ {type} __pnl_builtin_exp({type} a) {{ return exp(a); }}
__device__ {type} __pnl_builtin_pow({type} a, {type} b) {{ return pow(a, b); }}
__device__ int64_t __pnl_builtin_get_printf_address() {{ return 0; }}
"""


class ptx_jit_engine(jit_engine):
    class cuda_engine():
        def __init__(self, tm):
            self._modules = {}
            self._target_machine = tm

            # -dc option tells the compiler that the code will be used for linking
            self._generated_builtins = pycuda.compiler.compile(_ptx_builtin_source.format(type=str(LLVMBuilderContext.get_current().float_ty)), target='cubin', options=['-dc'])

        def set_object_cache(cache):
            pass

        def add_module(self, module):
            max_regs = int(debug_env.get("cuda_max_regs", 256))
            try:
                # LLVM can't produce CUBIN for some reason
                start_time = time.perf_counter()
                ptx = self._target_machine.emit_assembly(module)
                ptx_time = time.perf_counter()
                mod = pycuda.compiler.DynamicModule(link_options=[(pycuda.driver.jit_option.MAX_REGISTERS, max_regs)])
                mod.add_data(self._generated_builtins, pycuda.driver.jit_input_type.CUBIN, "builtins.cubin")
                mod.add_data(ptx.encode(), pycuda.driver.jit_input_type.PTX, module.name + ".ptx")
                module_time = time.perf_counter()
                ptx_mod = mod.link()
                finish_time = time.perf_counter()
                if "time_stat" in debug_env:
                    print("Time to emit PTX module bundle '{}'({} lines): {}".format(module.name, len(ptx.splitlines()), ptx_time - start_time))
                    print("Time to add PTX module bundle '{}': {}".format(module.name, module_time - ptx_time))
                    print("Time to link PTX module bundle '{}': {}".format(module.name, finish_time - module_time))
                    print("Total time to process PTX module bundle '{}': {}".format(module.name, finish_time - start_time))

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
        assert self._jit_pass_builder is None
        assert self._jit_target_machine is None

        self._jit_target_machine, self._jit_pass_manager, self._jit_pass_builder = _ptx_jit_constructor()
        self._jit_engine = ptx_jit_engine.cuda_engine(self._jit_target_machine)

    def get_kernel(self, name):
        kernel = self._engine._find_kernel(name + "_cuda_kernel")
        if kernel is None:
            function = _find_llvm_function(name)
            wrapper_mod = _gen_cuda_kernel_wrapper_module(function)
            self.stage_compilation({wrapper_mod})
            self.compile_staged()
            kernel = self._engine._find_kernel(name + "_cuda_kernel")
#            kernel.set_cache_config(pycuda.driver.func_cache.PREFER_L1)

        return kernel
