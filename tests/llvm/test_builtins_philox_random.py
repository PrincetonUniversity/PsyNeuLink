import ctypes
import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

SEED = 0

@pytest.mark.benchmark(group="Philox integer PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_random_int64(benchmark, mode):
    res = []
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)
        def f():
            # Get uint range [0, MAX] to avoid any intermediate caching of random bits
            return prng.integers(0xffffffffffffffff, dtype=np.uint64, endpoint=True)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.byref_arg_types[0]()
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int64')
        out = ctypes.c_longlong()
        def f():
            gen_fun(state, out)
            return np.uint64(out.value)
    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = ctypes.sizeof(init_fun.byref_arg_types[0])
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int64')
        out = np.asarray([0], dtype=np.uint64)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)
        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out[0]

    # Get >4 samples to force regeneration of Philox buffer
    res = [f(), f(), f(), f(), f(), f()]
    assert np.allclose(res, [259491006799949737,  4754966410622352325,  8698845897610382596,
                             1686395276220330909, 18061843536446043542, 4723914225006068263])
    benchmark(f)


@pytest.mark.benchmark(group="Philox integer PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_random_int32(benchmark, mode):
    res = []
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)
        def f():
            # Get uint range [0, MAX] to avoid any intermediate caching of random bits
            return prng.integers(0xffffffff, dtype=np.uint32, endpoint=True)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.byref_arg_types[0]()
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int32')
        out = ctypes.c_int()
        def f():
            gen_fun(state, out)
            return np.uint32(out.value)
    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = ctypes.sizeof(init_fun.byref_arg_types[0])
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int32')
        out = np.asarray([0], dtype=np.uint32)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)
        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out[0]

    # Get >4 samples to force regeneration of Philox buffer
    res = [f(), f(), f(), f(), f(), f()]
    assert np.allclose(res, [582496169, 60417458, 4027530181, 1107101889, 1659784452, 2025357889])
    benchmark(f)


@pytest.mark.benchmark(group="Philox floating point PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_random_double(benchmark, mode):
    res = []
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)
        def f():
            return prng.random(dtype=np.float64)
    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.byref_arg_types[0]()
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_double')
        out = ctypes.c_double()
        def f():
            gen_fun(state, out)
            return out.value
    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = ctypes.sizeof(init_fun.byref_arg_types[0])
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_double')
        out = np.asfarray([0.0], dtype=np.float64)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)
        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out[0]

    res = [f(), f()]
    assert np.allclose(res, [0.014067035665647709, 0.2577672456246177])
    benchmark(f)


@pytest.mark.benchmark(group="Philox floating point PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
def test_random_float(benchmark, mode):
    res = []
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)
        def f():
            return prng.random(dtype=np.float32)
    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.byref_arg_types[0]()
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_float')
        out = ctypes.c_float()
        def f():
            gen_fun(state, out)
            return out.value
    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = ctypes.sizeof(init_fun.byref_arg_types[0])
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_float')
        out = np.asfarray([0.0], dtype=np.float32)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)
        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out[0]

    res = [f(), f()]
    assert np.allclose(res, [0.13562285900115967, 0.014066934585571289])
    benchmark(f)


@pytest.mark.benchmark(group="Philox Normal distribution")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.param('PTX', marks=pytest.mark.cuda)])
@pytest.mark.skip
def test_random_normal(benchmark, mode):
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)
        def f():
            return prng.standard_normal(dtype=np.float64)
    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.byref_arg_types[0]()
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_normal')
        out = ctypes.c_double()
        def f():
            gen_fun(state, out)
            return out.value
    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = ctypes.sizeof(init_fun.byref_arg_types[0])
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int32(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_normal')
        out = np.asfarray([0.0], dtype=np.float64)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)
        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out[0]

    res = [f(), f()]
    assert np.allclose(res, [-0.2059740286292238, -0.12884495093462758])
    benchmark(f)
