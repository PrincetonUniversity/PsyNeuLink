#!/bin/env python3

# Short example/demonstrator how to create and compile native code using
# LLVM MCJIT just-in-time compiler

from llvmlite import binding,ir
import ctypes

# Create some useful types
double = ir.DoubleType()
fnty = ir.FunctionType(double, (double, double))

# Create an empty module...
module = ir.Module(name=__file__)
# and declare a function named "fpadd" inside it
func = ir.Function(module, fnty, name="fpadd")

# Now implement basic addition
# basic blocks are sequences of instructions that have exactly one
# entry point and one exit point (no control flow)
# We only need one in this case
# See available operations at:
# http://llvmlite.readthedocs.io/en/latest/ir/builder.html#instruction-building
block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
a, b = func.args
result = builder.fadd(a, b, name="res")
builder.ret(result)

# Uncomment to print the module IR. This prints LLVM IR assembly.
# print("LLVM IR:")
# print(module)

binding.initialize()

# native == currently running CPU
binding.initialize_native_target()

# TODO: This prevents 'LLVM ERROR: Target does not support MC emission!',
# but why?
binding.initialize_native_asmprinter()

# Create compilation target, use default triple
target = binding.Target.from_default_triple()
target_machine = target.create_target_machine()

# And an execution engine with an empty backing module
# TODO: why is empty backing mod necessary?
backing_mod = binding.parse_assembly("")

# There are other engines beside MCJIT
# MCJIT makes it easier to run the compiled function right away.
engine = binding.create_mcjit_compiler(backing_mod, target_machine)

# IR module is not the same as binding module.
# "assembly" in this case is LLVM IR assembly
# TODO is there a better way to convert this?
mod = binding.parse_assembly(str(module))
mod.verify()

# Now add the module and make sure it is ready for execution
engine.add_module(mod)
engine.finalize_object()

# Uncomment to print generated x86 assembly
#print("x86 assembly:")
#print(target_machine.emit_assembly(mod))

# Look up the function pointer (a Python int)
# func_ptr is now an address to a compiled function
func_ptr = engine.get_function_address("fpadd")

# Run the function via ctypes
a = 10.0
b = 3.5
cfunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(func_ptr)
res = cfunc(10.0, 3.5)
if res != (a + b):
    print("TEST FAILED! {} instead of {}".format(res, a + b))
else:
    print("TEST PASSED! {} == {}".format(res, a + b))
    

engine.remove_module(mod)
# TODO: shutdown cleanly
# we need to do something extra before shutdown
#binding.shutdown()
