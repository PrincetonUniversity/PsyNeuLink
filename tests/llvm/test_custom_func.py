import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm


@pytest.mark.llvm
@pytest.mark.parametrize('mode', ['CPU',
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('val', [np.int8(0x7e),
                                 np.int16(0x7eec),
                                 np.int32(0x7eedbeee),
                                 np.int64(0x7eedcafedeadbeee)
                                ], ids=lambda x: str(x.dtype))
def test_integer_broadcast(mode, val):
    custom_name = None
    with pnlvm.LLVMBuilderContext.get_current() as ctx:
        custom_name = ctx.get_unique_name("broadcast")
        int_ty = ctx.convert_python_struct_to_llvm_ir(val)
        int_array_ty = pnlvm.ir.ArrayType(int_ty, 8)
        func_ty = pnlvm.ir.FunctionType(pnlvm.ir.VoidType(),
                                        (int_ty.as_pointer(),
                                         int_array_ty.as_pointer()))
        function = pnlvm.ir.Function(ctx.module, func_ty, name=custom_name)

        i, o = function.args
        block = function.append_basic_block(name="entry")
        builder = pnlvm.ir.IRBuilder(block)
        ival = builder.load(i)
        ival = builder.add(ival, ival.type(1))
        with pnlvm.helpers.array_ptr_loop(builder, o, "broadcast") as (b, i):
            out_ptr = builder.gep(o, [ctx.int32_ty(0), i])
            builder.store(ival, out_ptr)
        builder.ret_void()

    binf = pnlvm.LLVMBinaryFunction.get(custom_name)
    res = np.zeros(8, dtype=val.dtype)

    if mode == 'CPU':
        ct_res = np.ctypeslib.as_ctypes(res)
        ct_in = np.ctypeslib.as_ctypes(val)

        binf(ctypes.byref(ct_in), ctypes.byref(ct_res))
    else:
        binf.cuda_wrap_call(np.asarray(val), res)

    assert all(res == np.broadcast_to(val + 1, 8))
