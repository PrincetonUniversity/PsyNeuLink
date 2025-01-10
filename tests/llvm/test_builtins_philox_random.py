import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm

SEED = 0

@pytest.mark.benchmark(group="Philox integer PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('seed, expected', [
    (0, [259491006799949737,  4754966410622352325,  8698845897610382596, 1686395276220330909, 18061843536446043542, 4723914225006068263]),
    (-5, [4936860362606747269, 11611290354192475889, 2015254117581537576, 4620074701282684350, 9574602527017877750, 2811009141214824706]),
    (15, [322160557315224026, 10187298772616605914, 11130303561932346278, 3540317624683947565, 245468466731153020, 17669502083357198575]),
    (0xfeedcafe, [14360762734736817955, 5188080951818105836, 1417692977344505657, 15919241602363537044, 11006348070701344872, 12539562470140893435]),
])
def test_random_int64(benchmark, mode, seed, expected):
    if mode == 'numpy':
        state = np.random.Philox([np.int64(seed).astype(np.uint64)])
        prng = np.random.Generator(state)
        def f():
            # Get uint range [0, MAX] to avoid any intermediate caching of random bits
            return prng.integers(0xffffffffffffffff, dtype=np.uint64, endpoint=True)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, seed)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int64')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)

        init_fun.cuda_call(gpu_state, np.int64(seed))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int64')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    # Get >4 samples to force regeneration of Philox buffer
    res = [f(), f(), f(), f(), f(), f()]
    np.testing.assert_array_equal(res, expected)
    benchmark(f)


@pytest.mark.benchmark(group="Philox integer PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize("bounds, expected",
    [((0xffffffff,), [582496168, 60417457, 4027530180, 1107101888, 1659784451, 2025357888]),
     ((15,), [2, 0, 14, 3, 5, 7]),
     ((0,15), [2, 0, 14, 3, 5, 7]),
     ((5,0xffff), [8892, 926, 61454, 16896, 25328, 30906]),
    ], ids=lambda x: str(x) if len(x) != 6 else "")
def test_random_int32_bounded(benchmark, mode, bounds, expected):
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)

        def f():
            return prng.integers(*bounds, dtype=np.uint32, endpoint=False)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int32_bounded')

        def f():
            lower, upper = bounds if len(bounds) == 2 else (0, bounds[0])
            out = gen_fun.np_buffer_for_arg(3)
            gen_fun(state, lower, upper, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int32_bounded')
        out = gen_fun.np_buffer_for_arg(3)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            lower, upper = bounds if len(bounds) == 2 else (0, bounds[0])
            gen_fun.cuda_call(gpu_state, np.uint32(lower), np.uint32(upper), gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    # Get >4 samples to force regeneration of Philox buffer
    res = [f(), f(), f(), f(), f(), f()]
    np.testing.assert_array_equal(res, expected)
    benchmark(f)

@pytest.mark.benchmark(group="Philox integer PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
def test_random_int32(benchmark, mode):
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)\

        def f():
            # Get uint range [0, MAX] to avoid any intermediate caching of random bits
            return prng.integers(0xffffffff, dtype=np.uint32, endpoint=True)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int32')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_int32')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    # Get >4 samples to force regeneration of Philox buffer
    res = [f(), f(), f(), f(), f(), f()]
    np.testing.assert_array_equal(res, [582496169, 60417458, 4027530181, 1107101889, 1659784452, 2025357889])
    benchmark(f)


@pytest.mark.benchmark(group="Philox floating point PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
def test_random_double(benchmark, mode):
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)

        def f():
            return prng.random(dtype=np.float64)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_double')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_double')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f(), f()]
    np.testing.assert_allclose(res, [0.014067035665647709, 0.2577672456246177])
    benchmark(f)


@pytest.mark.benchmark(group="Philox floating point PRNG")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
def test_random_float(benchmark, mode):
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)

        def f():
            return prng.random(dtype=np.float32)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_float')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_float')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f(), f()]
    np.testing.assert_allclose(res, [0.13562285900115967, 0.014066934585571289], rtol=1e-5, atol=1e-8)
    benchmark(f)


@pytest.mark.benchmark(group="Philox Normal distribution")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('fp_type', [pnlvm.ir.DoubleType(), pnlvm.ir.FloatType()], ids=str)
def test_random_normal(benchmark, mode, fp_type):
    if mode != 'numpy':
        # Instantiate builder context with the desired type
        pnlvm.LLVMBuilderContext(fp_type)

    dtype = np.float64 if fp_type is pnlvm.ir.DoubleType() else np.float32
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)

        def f():
            return prng.standard_normal(dtype=dtype)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_normal')

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_normal')
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f() for i in range(191000)]
    if fp_type is pnlvm.ir.DoubleType():
        np.testing.assert_allclose(res[0:2], [-0.2059740286292238, -0.12884495093462758])
        # 208 doesn't take the fast path but wraps around the main loop
        np.testing.assert_allclose(res[207:211],
                                   [-0.768690647997579, 0.4301874289485477, -0.7803640491708955, -1.146089287628737])
        # 450 doesn't take the fast path or wrap around the main loop,
        # but takes the special condition at the end of the loop
        np.testing.assert_allclose(res[449:453],
                                   [-0.7713655663874537, -0.5638348710823825, -0.9415838853097869, 0.6212784278881248])
        # 2013 takes the rare secondary loop and exists in the first iteration
        # taking the positive value
        np.testing.assert_allclose(res[2011:2015],
                                   [0.4201922976982861, 2.7021541445373916, 3.7809967764329375, 0.19919094793393655])
        # 5136 takes the rare secondary loop and exists in the first iteration
        # taking the negative value
        np.testing.assert_allclose(res[5134:5138],
                                   [0.12317411414687844, -0.17846827974421134, -3.6579887696059714, 0.2501530374224693])
        # 190855 takes the rare secondary loop and needs more than one iteration
        np.testing.assert_allclose(res[190853:190857],
                                   [-0.26418319904491194, 0.35889007879353746, -3.843811523424439, -1.5256469840469997])

    elif fp_type is pnlvm.ir.FloatType():
        # The indices are taken from above and don't have special meaning.
        np.testing.assert_allclose(res[0:2], [-0.24822916090488434, -0.02676701545715332])
        np.testing.assert_allclose(res[207:211],
                                   [-0.33086925745010376, -1.024695873260498, -0.5162619352340698, -0.15033885836601257])
        np.testing.assert_allclose(res[449:453],
                                   [-0.2223609834909439, 0.16769859194755554, -0.7806711196899414, 0.5867824554443359])
        np.testing.assert_allclose(res[2011:2015],
                                   [0.1979091316461563, -0.23467595875263214, 1.1458240747451782, -1.0285860300064087])
        np.testing.assert_allclose(res[5134:5138],
                                   [-1.0523858070373535, -3.007537603378296, -0.4331461489200592, -0.8841480612754822])
        np.testing.assert_allclose(res[190853:190857],
                                   [-0.8958197236061096, 0.10532315075397491, 2.000257730484009, -1.129721999168396])

    assert not any(np.isnan(res)), list(np.isnan(res)).index(True)
    benchmark(f)

@pytest.mark.benchmark(group="Philox Binomial distribution")
@pytest.mark.parametrize('mode', ['numpy',
                                  pytest.param('LLVM', marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param('PTX')])
@pytest.mark.parametrize('fp_type', [pnlvm.ir.DoubleType(), pnlvm.ir.FloatType()],
                         ids=str)
@pytest.mark.parametrize('n', [1])
@pytest.mark.parametrize('p, exp_64, exp_32', [
    (0, [0], [0]),
    (0.1, [0xa0c0100], [0x20440250]),
    (0.33, [0xa2c8186], [0x20440650]),
    (0.5, [0xa2c81c6], [0x226c8650]),
    (0.66, [0xf5d37e79], [0xdfbbf9af]),
    (0.95, [0xf7f3ffff], [0xffbffdaf]),
    (1, [0xffffffff], [0xffffffff]),
    ])
def test_random_binomial(benchmark, mode, fp_type, n, p, exp_64, exp_32):
    if mode != 'numpy':
        # Instantiate builder context with the desired type
        pnlvm.LLVMBuilderContext(fp_type)

    # numpy always uses fp64 uniform sampling
    exp = exp_64 if fp_type is pnlvm.ir.DoubleType() or mode == 'numpy' else exp_32
    if mode == 'numpy':
        state = np.random.Philox([SEED])
        prng = np.random.Generator(state)

        def f():
            return prng.binomial(n, p)

    elif mode == 'LLVM':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state = init_fun.np_buffer_for_arg(0)
        init_fun(state, SEED)

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_binomial')
        n = np.asarray(n, dtype=gen_fun.np_arg_dtypes[1])
        p = np.asarray(p, dtype=gen_fun.np_arg_dtypes[2])

        def f():
            out = gen_fun.np_buffer_for_arg(1)
            gen_fun(state, n, p, out)
            return out

    elif mode == 'PTX':
        init_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_init')
        state_size = init_fun.np_buffer_for_arg(0).nbytes
        gpu_state = pnlvm.jit_engine.pycuda.driver.mem_alloc(state_size)
        init_fun.cuda_call(gpu_state, np.int64(SEED))

        gen_fun = pnlvm.LLVMBinaryFunction.get('__pnl_builtin_philox_rand_binomial')
        gpu_n = pnlvm.jit_engine.pycuda.driver.In(np.asarray(n, dtype=gen_fun.np_arg_dtypes[1]))
        gpu_p = pnlvm.jit_engine.pycuda.driver.In(np.asarray(p, dtype=gen_fun.np_arg_dtypes[2]))
        out = gen_fun.np_buffer_for_arg(1)
        gpu_out = pnlvm.jit_engine.pycuda.driver.Out(out)

        def f():
            gen_fun.cuda_call(gpu_state, gpu_n, gpu_p, gpu_out)
            return out.copy()

    else:
        assert False, "Unknown mode: {}".format(mode)

    res = [f() for i in range(32)]
    res = int(''.join(str(x) for x in res), 2)
    assert res == exp[n - 1]

    benchmark(f)
