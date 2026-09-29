"""CUDA filtering operations must preserve the Torch reference weights/states."""

import numpy as np
import pytest

from psyneulink.core.batched.compiler import _systematic_resample
from psyneulink.core.batched.likelihood import _HistogramObservationWeights

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.batched, pytest.mark.triton_gpu]


@pytest.mark.parametrize("bins,sigma,alpha", [(1, 0., 0.), (5, .7, .3), (100, .5, 1.e-4)])
@pytest.mark.parametrize("categorical", [[], [0], [0, 2]])
def test_fused_weights_match_reference_at_edges_and_for_multiple_outputs(bins, sigma, alpha, categorical):
    observed = np.array([[0., .1, 1.], [1., .9, 0.], [0., 0., 1.]])
    continuous = [d for d in range(3) if d not in categorical]
    options = dict(categorical_dims=categorical, bins=bins, bin_range=[(0., 1.)] * len(continuous),
                   smoothing_sigma=sigma, pseudocount=alpha, categorical_cardinalities=[2] * len(categorical),
                   dtype=torch.float32, device="cuda", strict_observations=True, source_normalized=True)
    reference = _HistogramObservationWeights(observed, **options)
    fused = _HistogramObservationWeights(observed, fused=True, **options)
    simulated = torch.rand((2, 3, 1031, 3), device="cuda")
    for d in categorical:
        simulated[..., d] = (simulated[..., d] > .5).float()
    # Cover exact FP32 edges, adjacent floats, overflow, and nonfinite values.
    for d, edge in zip(continuous, reference.edges):
        values = torch.cat((edge, torch.nextafter(edge, torch.full_like(edge, -torch.inf)),
                            torch.nextafter(edge, torch.full_like(edge, torch.inf)),
                            torch.tensor([-torch.inf, torch.inf, torch.nan], device="cuda")))
        simulated[:, :, :len(values), d] = values
    for trial in range(len(observed)):
        for values in (simulated, simulated[..., ::2, :]):
            actual = fused(values, trial)
            expected = reference(values, trial)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0., atol=0.)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("estimates,width", [(1, 1), (37, 5), (10000, 47), (100000, 47), (137, 128)])
def test_fused_systematic_state_gather_preserves_reference_ancestors(estimates, width, shared):
    from psyneulink.core.batched.backend.triton.conditioned_ops import SystematicStateResampler

    generator = torch.Generator().manual_seed(81)
    weights = torch.rand((3, 2, estimates), generator=generator).cuda()
    weights = torch.where(weights > .93, weights, 1.e-5)
    # Uniform and point-mass filters coexist with the difficult sparse weights.
    weights[0, 0] = 1.
    weights[0, 1] = 0.
    weights[0, 1, estimates // 2] = 1.
    # The caller rejects zero support after the loop; gathering must remain
    # defined while the asynchronous error check is pending.
    weights[1, 1] = 0.
    terminal = torch.arange(3 * 2 * estimates * width, dtype=torch.float32, device="cuda").reshape(3, 2, estimates, width)
    ref_rng = torch.Generator(device="cuda").manual_seed(93)
    fast_rng = torch.Generator(device="cuda").manual_seed(93)
    resample = SystematicStateResampler(estimates, weights.device)
    previous_pointer = None
    for _ in range(2):
        ancestors = _systematic_resample(weights, generator=ref_rng, shared_first_axis=shared)
        expected = torch.gather(terminal, -2, ancestors[..., None].expand_as(terminal))
        actual = resample(weights, terminal, generator=fast_rng, shared_first_axis=shared)
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        if previous_pointer is not None:
            assert actual.data_ptr() == previous_pointer
        previous_pointer = actual.data_ptr()
    torch.testing.assert_close(ref_rng.get_state(), fast_rng.get_state(), rtol=0., atol=0.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_weights_categorical_only_and_exact_category_tolerance(dtype):
    options = dict(categorical_dims=[0], categorical_cardinalities=[2], bins=100,
                   dtype=dtype, device="cuda", strict_observations=True,
                   source_normalized=True, pseudocount=.1)
    observed = np.array([[0.], [1.]])
    simulated = torch.tensor([0., 1.e-6, 1.00001e-6, 1., 1.000001, float("nan")],
                             dtype=dtype, device="cuda")[None, :, None]
    reference = _HistogramObservationWeights(observed, **options)
    fused = _HistogramObservationWeights(observed, fused=True, **options)
    assert (fused.fused is None) == (dtype != torch.float32)
    for t in range(2):
        for actual, expected in zip(fused(simulated, t), reference(simulated, t)):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
