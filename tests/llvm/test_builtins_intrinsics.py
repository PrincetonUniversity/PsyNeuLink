import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

x = np.random.rand()
y = np.random.rand()


exp_res = np.exp(x)
log_res = np.log(x)
pow_res = np.power(x, y)

@pytest.mark.benchmark(group="Builtins")
@pytest.mark.parametrize("op, args, builtin, result", [
                         (np.exp, (x,), "__pnl_builtin_exp", exp_res),
                         (np.log, (x,), "__pnl_builtin_log", log_res),
                         (np.power, (x,y), "__pnl_builtin_pow", pow_res),
                         ], ids=["EXP", "LOG", "POW"])
@pytest.mark.parametrize("mode", ['Python',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
def test_builtin_op(benchmark, op, args, builtin, result, mode):
    if mode == 'Python':
        f = op
    elif mode == 'LLVM':
        f = pnlvm.LLVMBinaryFunction.get(builtin)
    elif mode == 'PTX':
        wrap_name = builtin + "_test_wrapper"
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            intrin = ctx.import_llvm_function(builtin)
            wrap_args = (*intrin.type.pointee.args,
                          intrin.type.pointee.return_type.as_pointer())
            builder = ctx.create_llvm_function(wrap_args, None, wrap_name)
            intrin_args = builder.function.args[:-1]
            ret = builder.call(intrin, intrin_args)
            builder.store(ret, builder.function.args[-1])
            builder.ret_void()

        bin_f = pnlvm.LLVMBinaryFunction.get(wrap_name)
        ptx_res = np.asarray(type(result)(0))
        ptx_res_arg = pnlvm.jit_engine.pycuda.driver.Out(ptx_res)
        def f(*a):
            bin_f.cuda_call(*(np.double(p) for p in a), ptx_res_arg)
            return ptx_res
    res = benchmark(f, *args)
    assert res == result
