import numpy as np
import pytest
import random

from psyneulink.core import llvm as pnlvm

SEED = 0

@pytest.mark.benchmark(group="Mersenne Twister bounded integer PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize("bounds, expected",
    [((0xffffffff,), [3626764237, 1654615998, 3255389356, 3823568514, 1806341205]),
     ((14,), [13, 12,  2,  5,  4]),
     ((0,14), [13, 12,  2,  5,  4]),
     ((5,0xffff), [2002, 28611, 19633,  1671, 37978]),
    ], ids=lambda x: str(x) if len(x) != 5 else "")
# Python uses sampling of upper bits (vs. lower bits in Numpy). Skip it in this test.
def test_random_int32_bounded(benchmark, mode, bounds, expected):

    if mode == 'numpy':
        # Numpy promotes elements to int64
        state = np.random.RandomState([SEED])

        def f():
            return state.randint(*bounds, dtype=np.uint32)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.np_buffer_for_arg(0)

        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_int32_bounded')

        def f():
            lower, upper = bounds if len(bounds) == 2 else (0, bounds[0])
            out = gen_fun.np_buffer_for_arg(3)
            gen_fun(state, lower, upper, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')

        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)

        init_fun.cuda_call(gpu_state, np.int32(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_int32_bounded')
        out = gen_fun.np_buffer_for_arg(3)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            lower, upper = bounds if len(bounds) == 2 else (0, bounds[0])
            gen_fun.cuda_call(gpu_state, np.uint32(lower), np.uint32(upper), gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f(), f(), f(), f(), f()]
    np.testing.assert_array_equal(res, expected)
    benchmark(f)

@pytest.mark.benchmark(group="Mersenne Twister integer PRNG")
@pytest.mark.parametrize('mode', ['Python', 'numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
def test_random_int32(benchmark, mode):

    if mode == 'Python':
        state = random.Random(SEED)

        def f():
            return state.randrange(0xffffffff)

    elif mode == 'numpy':
        # Numpy promotes elements to int64
        state = np.random.RandomState([SEED])

        def f():
            return state.randint(0xffffffff, dtype=np.uint32)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.np_buffer_for_arg(0)

        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_int32')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')

        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)

        init_fun.cuda_call(gpu_state, np.int32(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_int32')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f(), f(), f(), f(), f()]
    np.testing.assert_array_equal(res, [3626764237, 1654615998, 3255389356, 3823568514, 1806341205])
    benchmark(f)


@pytest.mark.benchmark(group="Mersenne Twister floating point PRNG")
@pytest.mark.parametrize('mode', ['Python', 'numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
def test_random_float(benchmark, mode):

    if mode == 'Python':
        # Python treats every seed as array
        state = random.Random(SEED)

        def f():
            return state.random()

    elif mode == 'numpy':
        # numpy promotes elements to int64
        state = np.random.RandomState([SEED])

        def f():
            return state.random_sample()

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_double')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')

        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)

        init_fun.cuda_call(gpu_state, np.int32(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_double')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f(), f()]
    np.testing.assert_allclose(res, [0.8444218515250481, 0.7579544029403025])
    benchmark(f)


@pytest.mark.benchmark(group="Marsenne Twister Normal distribution")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
# Python uses different algorithm so skip it in this test
def test_random_normal(benchmark, mode):

    if mode == 'numpy':
        # numpy promotes elements to int64
        state = np.random.RandomState([SEED])

        def f():
            return state.normal()

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_normal')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')

        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)

        init_fun.cuda_call(gpu_state, np.int32(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_normal')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f(), f()]
    np.testing.assert_allclose(res, [0.4644982638709743, 0.6202001216069017], rtol=1e-5, atol=1e-8)
    benchmark(f)

@pytest.mark.benchmark(group="Marsenne Twister Binomial distribution")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('n', [1])
@pytest.mark.parametrize('p, exp', [
    (0, [0]),
    (0.1, [0x20d00c]),
    (0.33, [0xc224f70d]),
    (0.5, [0xca76f71d]),
    (0.66, [0x3ddb08f2]),
    (0.95, [0xffffbffb]),
    (1, [0xffffffff]),
    ])
# Python uses different algorithm so skip it in this test
def test_random_binomial(benchmark, mode, n, p, exp):
    if mode == 'numpy':
        # numpy promotes elements to int64
        state = np.random.RandomState([SEED])

        def f():
            return state.binomial(n, p)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_binomial')
        n = np.asarray(n, dtype=gen_fun.np_arg_dtypes[1])
        p = np.asarray(p, dtype=gen_fun.np_arg_dtypes[2])

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, n, p, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_init')

        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)

        init_fun.cuda_call(gpu_state, np.int32(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_mt_rand_binomial')

        gpu_n = pnlvm.jit_engine.pycuda.driver.In(np.asarray(n, dtype=gen_fun.np_arg_dtypes[1]))
        gpu_p = pnlvm.jit_engine.pycuda.driver.In(np.asarray(p, dtype=gen_fun.np_arg_dtypes[2]))

        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_n, gpu_p, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f() for _ in range(32)]
    res = int(''.join(str(x) for x in res), 2)
    assert res == exp[n - 1]
    benchmark(f)
