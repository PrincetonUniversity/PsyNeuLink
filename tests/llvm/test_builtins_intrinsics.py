import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

x = np.random.rand()
y = np.random.rand()


@pytest.mark.benchmark(group="Builtins")
@pytest.mark.parametrize("op, args, builtin, result", [
                         (np.exp, (x,), "__pnl_builtin_exp", np.exp(x)),
                         #~900 is the limit after which exp returns inf
                         (np.exp, (900.0,), "__pnl_builtin_exp", np.exp(900.0)),
                         (np.log, (x,), "__pnl_builtin_log", np.log(x)),
                         (np.power, (x,y), "__pnl_builtin_pow", np.power(x, y)),
                         (np.tanh, (x,), "__pnl_builtin_tanh", np.tanh(x)),
                         #~450 is the limit after which exp(2x) used in tanh formula returns inf
                         (np.tanh, (450.0,), "__pnl_builtin_tanh", np.tanh(450)),
                         (lambda x: 1.0 / np.tanh(x), (x,), "__pnl_builtin_coth", 1 / np.tanh(x)),
                         #~450 is the limit after which exp(2x) used in coth formula returns inf
                         (lambda x: 1.0 / np.tanh(x), (450,), "__pnl_builtin_coth", 1 / np.tanh(450)),
                         (lambda x: 1.0 / np.sinh(x), (x,), "__pnl_builtin_csch", 1 / np.sinh(x)),
                         #~450 is the limit after which exp(2x) used in csch formula returns inf
                         (lambda x: 1.0 / np.sinh(x), (450,), "__pnl_builtin_csch", 1 / np.sinh(450)),
                         #~900 is the limit after which exp(x) used in csch formula returns inf
                         (lambda x: 1.0 / np.sinh(x), (900,), "__pnl_builtin_csch", 1 / np.sinh(900)),
                         ], ids=["EXP", "Large EXP", "LOG", "POW", "TANH", "Large TANH", "COTH", "Large COTH",
                                "CSCH", "Large CSCH", "xLarge CSCH"])
def test_builtin_op(benchmark, op, args, builtin, result, func_mode):
    if func_mode == 'Python':
        f = op
    elif func_mode == 'LLVM':
        f = pnlvm.LLVMBinaryFunction.get(builtin)
    elif func_mode == 'PTX':
        wrap_name = builtin + "_test_wrapper"
        with pnlvm.LLVMBuilderContext.get_current() as ctx:
            intrin = ctx.import_llvm_function(builtin)
            wrap_args = (*intrin.type.pointee.args,
                          intrin.type.pointee.return_type.as_pointer())
            builder = ctx.create_llvm_function(wrap_args, None, wrap_name)
            intrin_args = builder.function.args[:-1]
            ret = builder.call(intrin, intrin_args)
            builder.store(ret, builder.function.args[-1])
            builder.ret_void()

        bin_f = pnlvm.LLVMBinaryFunction.get(wrap_name)
        dty = np.dtype(bin_f.byref_arg_types[0])
        ptx_res = np.empty_like(result, dtype=dty)
        ptx_res_arg = pnlvm.jit_engine.pycuda.driver.Out(ptx_res)
        def f(*a):
            bin_f.cuda_call(*(dty.type(p) for p in a), ptx_res_arg)
            return ptx_res
    res = benchmark(f, *args)

    if pytest.helpers.llvm_current_fp_precision() == 'fp32':
        assert np.allclose(res, result)
    else:
        np.testing.assert_allclose(res, result)
