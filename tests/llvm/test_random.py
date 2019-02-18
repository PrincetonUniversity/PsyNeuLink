#!/usr/bin/python3

import ctypes
import numpy as np
import pytest
import random

from psyneulink.core import llvm as pnlvm

SEED = 0

@pytest.mark.llvm
@pytest.mark.benchmark
@pytest.mark.parametrize('mode', ['Python', 'numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_random(benchmark, mode):
    if mode == 'Python':
        # Python treats every seed as array
        state = random.Random(SEED)
        res = state.random()
        benchmark(state.random)
    elif mode == 'numpy':
        # Python treats every seed as array, and numpy promotes elements to int64
        state = np.random.RandomState(np.asarray([SEED]))
        res = state.random_sample()
        benchmark(state.random_sample)
    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.byref_arg_types[0]()
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_double')
        out = ctypes.c_double()
        gen_fun(state, out)
        res = out.value
        benchmark(gen_fun, state, out)
    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.byref_arg_types[0]()
        gpu_state = pnlvm.jit_engine.pycuda.driver.to_device(bytearray(state))
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_double')
        out = np.asfarray(0.0, dtype=np.float64)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)
        gen_fun.cuda_call(gpu_state, gpu_out)
        res = out
        benchmark(gen_fun.cuda_call, gpu_state, gpu_out)

    assert np.allclose(res, 0.8444218515250481)
