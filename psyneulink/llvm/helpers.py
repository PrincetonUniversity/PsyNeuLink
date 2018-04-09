# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* PNL LLVM helpers **************************************************************

from llvmlite import ir

def for_loop(builder, start, stop, inc, body_func, id):
    # Initialize index variable
    assert(start.type is stop.type)
    index_var = builder.alloca(stop.type)
    builder.store(start, index_var)

    # basic blocks
    cond_block = builder.append_basic_block(id + "-cond")
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

def for_loop_zero_inc(builder, stop, body_func, id):
    start = stop.type(0)
    inc = stop.type(1)
    return for_loop(builder, start, stop, inc, body_func, id)

def fclamp(builder, val, min_val, max_val):
    cond = builder.fcmp_unordered("<", val, min_val)
    tmp = builder.select(cond, min_val, val)
    cond = builder.fcmp_unordered(">", tmp, max_val)
    return builder.select(cond, max_val, tmp)

def fclamp_const(builder, val, min_val, max_val):
    minval = val.type(min_val)
    maxval = val.type(max_val)
    return fclamp(builder, val, minval, maxval)

def llvm_function_head(pnl_object, ctx, name=None):
    name = name if name is not None else pnl_object.name
    func_name = ctx.module.get_unique_name(name)
    func_ty = ir.FunctionType(ir.VoidType(),
        (pnl_object.get_param_struct_type().as_pointer(),
         pnl_object.get_context_struct_type().as_pointer(),
         pnl_object.get_input_struct_type().as_pointer(),
         pnl_object.get_output_struct_type().as_pointer()))

    llvm_func = ir.Function(ctx.module, func_ty, name=func_name)
    llvm_func.attributes.add('argmemonly')
    llvm_func.attributes.add('alwaysinline')
    for p in llvm_func.args:
        p.attributes.add('nonnull')
        p.attributes.add('noalias')

    # Create entry block
    block = llvm_func.append_basic_block(name="entry")
    return ir.IRBuilder(block)
