# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM builtins **************************************************************

from PsyNeuLink.llvm import helpers
from llvmlite import ir
import functools

def __set_array_body(builder, index, array, value):
    ptr = builder.gep(array, [index])
    builder.store(value, ptr)

def setup_vxm(ctx):
    module = ctx.module
    # Setup types
    double_ptr_ty = ctx.float_ty.as_pointer()
    func_ty = ir.FunctionType(ir.VoidType(), (double_ptr_ty, double_ptr_ty, ctx.int32_ty, ctx.int32_ty, double_ptr_ty))

    # Create function
    function = ir.Function(module, func_ty, name="__pnl_builtin_vxm")
    function.attributes.add('argmemonly')
    function.attributes.add('alwaysinline')

    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)
    v, m, x, y, o = function.args

    # Add function arg attributes
    for a in v,m,o:
        a.attributes.add('nonnull')
        a.attributes.add('noalias')

    kwargs = {"array": o, "value": ctx.float_ty(.0)}

    # zero the output array
    zero_array = functools.partial(__set_array_body, **kwargs)
    builder = helpers.for_loop_zero_inc(builder, y, zero_array, "zero")

    # Multiplication

    # Initialize outer loop variable
    index_i_var = builder.alloca(ctx.int32_ty)
    builder.store(ctx.int32_ty(0), index_i_var)

    # Outer loop cond BB
    outer_cond_block = builder.append_basic_block("outer-cond")
    outer_body_block = builder.append_basic_block("outer-body")
    outer_out_block = builder.append_basic_block("outer-out")

    # Loop condition
    builder.branch(outer_cond_block)
    with builder.goto_block(outer_cond_block):
        tmp = builder.load(index_i_var);
        cond = builder.icmp_signed("<", tmp, x)
        builder.cbranch(cond, outer_body_block, outer_out_block).set_weights([99,1])

    # Loop body
    with builder.goto_block(outer_body_block):
        index_i = builder.load(index_i_var)

        # Initialize outer loop variable
        index_j_var = builder.alloca(ctx.int32_ty)
        builder.store(ctx.int32_ty(0), index_j_var)

        # Outer loop cond BB
        inner_cond_block = builder.append_basic_block("inner-cond")
        inner_body_block = builder.append_basic_block("inner-body")
        inner_out_block = builder.append_basic_block("inner-out")
        
        # Loop condition
        builder.branch(inner_cond_block)
        with builder.goto_block(inner_cond_block):
            tmp = builder.load(index_j_var);
            cond = builder.icmp_signed("<", tmp, y)
            builder.cbranch(cond, inner_body_block, inner_out_block).set_weights([99,1])

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

            next_index_j = builder.add(index_j, ctx.int32_ty(1))
            builder.store(next_index_j, index_j_var)
            builder.branch(inner_cond_block)
    

        with builder.goto_block(inner_out_block):
            next_index_i = builder.add(index_i, ctx.int32_ty(1))
            builder.store(next_index_i, index_i_var)
            builder.branch(outer_cond_block)

    # Return
    with builder.goto_block(outer_out_block):
        builder.ret_void()
