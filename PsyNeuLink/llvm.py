# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* LLVM bindings **************************************************************

from llvmlite import binding,ir
import ctypes
import functools
import uuid
import os

__module = ir.Module(name="PsyNeuLinkMmodule")
__int32_ty = ir.IntType(32)
__double_ty = ir.DoubleType()

def __get_id(suffix=""):
    return uuid.uuid4().hex + suffix

def __set_array_body(builder, index, array, value):
    ptr = builder.gep(array, [index])
    builder.store(value, ptr)
    

def __for_loop(builder, start, stop, inc, body_func, id = __get_id()):
    # Initialize index variable
    index_var = builder.alloca(__int32_ty)
    builder.store(start, index_var)

    # basic blocks
    cond_block = builder.append_basic_block(id + "cond")
    out_block = None

    # Loop condition
    builder.branch(cond_block)
    with builder.goto_block(cond_block):
        tmp = builder.load(index_var);
        cond = builder.icmp_signed("<", tmp, stop)

        # Loop body
        with builder.if_then(cond, likely=True):
            index = builder.load(index_var)
            if (body_func is not None):
                body_func(builder, index)
            index = builder.add(index, inc)
            builder.store(index, index_var)
            builder.branch(cond_block)

        out_block = builder.block

    return ir.IRBuilder(out_block)

def setup_mxv():
    # Setup types
    double_ptr_ty = __double_ty.as_pointer()
    func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, __int32_ty, __int32_ty, double_ptr_ty))

    # Create function
    function = ir.Function(__module, func_ty, name="mxv")
    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    v, m, x, y, o = function.args

    kwargs = {"array": o, "value": __double_ty(.0)}

    # zero the output array
    zero_array = functools.partial(__set_array_body, **kwargs)
    builder = __for_loop(builder, __int32_ty(0), x, __int32_ty(1), zero_array, "zero-")

    # Multiplication

    # Initialize outer loop variable
    index_i_var = builder.alloca(__int32_ty)
    builder.store(__int32_ty(0), index_i_var)

    # Outer loop cond BB
    outer_cond_block = builder.append_basic_block("outer-cond")
    outer_body_block = builder.append_basic_block("outer-body")
    outer_out_block = builder.append_basic_block("outer-out")

    # Loop condition
    builder.branch(outer_cond_block)
    with builder.goto_block(outer_cond_block):
        tmp = builder.load(index_i_var);
        cond = builder.icmp_signed("<", tmp, y)
        builder.cbranch(cond, outer_body_block, outer_out_block)

    # Loop body
    with builder.goto_block(outer_body_block):
        index_i = builder.load(index_i_var)

        # Initialize outer loop variable
        index_j_var = builder.alloca(__int32_ty)
        builder.store(__int32_ty(0), index_j_var)

        # Outer loop cond BB
        inner_cond_block = builder.append_basic_block("inner-cond")
        inner_body_block = builder.append_basic_block("inner-body")
        inner_out_block = builder.append_basic_block("inner-out")
        
        # Loop condition
        builder.branch(inner_cond_block)
        with builder.goto_block(inner_cond_block):
            tmp = builder.load(index_j_var);
            cond = builder.icmp_signed("<", tmp, y)
            builder.cbranch(cond, inner_body_block, inner_out_block)

        # Loop body
        with builder.goto_block(inner_body_block):
            index_j = builder.load(index_j_var)

            # Multiplication and accumulation
            vector_ptr = builder.gep(v, [index_i])
            matrix_index = builder.mul(index_i, y)
            matrix_index = builder.add(matrix_index, index_j)
            matrix_ptr = builder.gep(m, [matrix_index])
            out_ptr = builder.gep(o, [index_j])

            vector_el = builder.load(vector_ptr)
            matrix_el = builder.load(matrix_ptr)
            out_el = builder.load(out_ptr)

            new_el = builder.fmul(vector_el, matrix_el)
            new_el = builder.fadd(new_el, out_el)

            builder.store(new_el, out_ptr)

            next_index_j = builder.add(index_j, __int32_ty(1))
            builder.store(next_index_j, index_j_var)
            builder.branch(inner_cond_block)
    

        with builder.goto_block(inner_out_block):
            next_index_i = builder.add(index_i, __int32_ty(1))
            builder.store(next_index_i, index_i_var)
            builder.branch(outer_cond_block)

    # Return
    with builder.goto_block(outer_out_block):
        builder.ret_void()


setup_mxv()

__dumpenv = os.environ.get("PNL_LLVM_DUMP")
if __dumpenv is not None and __dumpenv.find("llvm") != -1:
    print(__module)

# Compiler binding
binding.initialize()
# native == currently running CPU
binding.initialize_native_target()

# TODO: This prevents 'LLVM ERROR: Target does not support MC emission!',
# but why?
binding.initialize_native_asmprinter()

__features = binding.get_host_cpu_features().flatten()
__cpu_name = binding.get_host_cpu_name()

# Create compilation target, use default triple
__target = binding.Target.from_default_triple()
__target_machine = __target.create_target_machine(cpu = __cpu_name, features = __features, opt = 3)

# And an execution engine with an empty backing module
# TODO: why is empty backing mod necessary?
__backing_mod = binding.parse_assembly("")

# There are other engines beside MCJIT
# MCJIT makes it easier to run the compiled function right away.
__engine = binding.create_mcjit_compiler(__backing_mod, __target_machine)

# IR module is not the same as binding module.
# "assembly" in this case is LLVM IR assembly
# TODO is there a better way to convert this?
__mod = binding.parse_assembly(str(__module))
__mod.verify()

__pass_manager_builder = binding.PassManagerBuilder()
__pass_manager_builder.inlining_threshold = 99999 # Inline all function calls
__pass_manager_builder.loop_vectorize = True
__pass_manager_builder.slp_vectorize = True
__pass_manager_builder.opt_level = 3 # Most aggressive optimizations

__pass_manager = binding.ModulePassManager()

__pass_manager_builder.populate(__pass_manager);

__pass_manager.run(__mod)

# Now add the module and make sure it is ready for execution
__engine.add_module(__mod)
__engine.finalize_object()

#This prints generated x86 assembly
if __dumpenv is not None and __dumpenv.find("isa") != -1:
    print("ISA assembly:")
    print(__target_machine.emit_assembly(__mod))

def get_mxv():
    func_ptr = __engine.get_function_address('mxv');
    cfunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))(func_ptr)

    return cfunc
