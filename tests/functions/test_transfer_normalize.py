import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.library.compositions.autodiffcomposition import torch_available


def numpy_reference_normalize(x, eps, per_item):
    x = np.asarray(x, dtype=float)

    if per_item and x.ndim > 1:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
    else:
        norm = np.linalg.norm(x)

    denom = np.maximum(norm, eps)
    return x / denom


@pytest.mark.function
@pytest.mark.transfer_function
class TestNormalizePython:
    @pytest.mark.parametrize(
        "x, eps, per_item",
        [
            pytest.param(np.array([3.0, 4.0]), 1e-12, True, id="vector_basic"),
            pytest.param(np.array([0.0, 0.0]), 1e-12, True, id="vector_zero"),
            pytest.param(np.array([3e-4, 4e-4]), 1e-3, True, id="vector_clamped"),
            pytest.param(np.array([[3.0, 4.0], [6.0, 8.0]]), 1e-12, True, id="matrix_rows_per_item"),
            pytest.param(np.array([[0.0, 0.0], [3e-4, 4e-4]]), 1e-3, True, id="matrix_mixed_per_item"),
            pytest.param(np.array([[3.0, 4.0], [6.0, 8.0]]), 1e-12, False, id="matrix_global"),
            pytest.param(np.array([[0.0, 0.0], [3e-4, 4e-4]]), 1e-3, False, id="matrix_mixed_global"),
        ],
    )
    def test_python_matches_reference(self, x, eps, per_item):
        f = pnl.Normalize(default_variable=x, eps=eps, per_item=per_item)
        result = np.asarray(f(x))
        expected = numpy_reference_normalize(x, eps, per_item)
        np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize(
        "x, eps, per_item",
        [
            pytest.param(np.array([3.0, 4.0]), 1e-12, True, id="unclamped_vector"),
            pytest.param(np.array([0.0, 0.0]), 1e-12, True, id="zero_vector"),
            pytest.param(np.array([3e-4, 4e-4]), 1e-3, True, id="clamped_vector"),
            pytest.param(np.array([[3.0, 4.0], [3e-4, 4e-4]]), 1e-3, True, id="mixed_rows_per_item"),
            pytest.param(np.array([[3.0, 4.0], [3e-4, 4e-4]]), 1e-3, False, id="mixed_rows_global"),
        ],
    )
    def test_invariants(self, x, eps, per_item):
        f = pnl.Normalize(default_variable=x, eps=eps, per_item=per_item)
        y = np.asarray(f(x), dtype=float)
        x = np.asarray(x, dtype=float)

        assert y.shape == x.shape

        expected = numpy_reference_normalize(x, eps, per_item)
        np.testing.assert_allclose(y, expected, rtol=1e-7, atol=1e-9)

        if x.ndim == 1 or not per_item:
            x_norm = np.linalg.norm(x)
            y_norm = np.linalg.norm(y)

            if x_norm > eps:
                np.testing.assert_allclose(y_norm, 1.0, rtol=1e-7, atol=1e-9)
        else:
            x_norm = np.linalg.norm(x, axis=-1)
            y_norm = np.linalg.norm(y, axis=-1)

            unclamped = x_norm > eps
            if np.any(unclamped):
                np.testing.assert_allclose(y_norm[unclamped], 1.0, rtol=1e-7, atol=1e-9)


if torch_available:
    import torch

    def torch_reference_normalize(x, eps, per_item):
        if per_item and x.ndim > 1:
            norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        else:
            norm = torch.linalg.norm(x, ord=2)

        denom = torch.clamp(norm, min=eps)
        return x / denom

    @pytest.mark.pytorch
    @pytest.mark.transfer_function
    class TestNormalizeTorch:
        @pytest.mark.parametrize(
            "x, eps, per_item",
            [
                pytest.param(np.array([3.0, 4.0]), 1e-12, True, id="vector_basic"),
                pytest.param(np.array([0.0, 0.0]), 1e-12, True, id="vector_zero"),
                pytest.param(np.array([3e-4, 4e-4]), 1e-3, True, id="vector_clamped"),
                pytest.param(np.array([[3.0, 4.0], [6.0, 8.0]]), 1e-12, True, id="matrix_rows_per_item"),
                pytest.param(np.array([[0.0, 0.0], [3e-4, 4e-4]]), 1e-3, True, id="matrix_mixed_per_item"),
                pytest.param(np.array([[3.0, 4.0], [6.0, 8.0]]), 1e-12, False, id="matrix_global"),
                pytest.param(np.array([[0.0, 0.0], [3e-4, 4e-4]]), 1e-3, False, id="matrix_mixed_global"),
            ],
        )
        def test_generated_pytorch_matches_reference(self, x, eps, per_item):
            f = pnl.Normalize(default_variable=x, eps=eps, per_item=per_item)
            context = pnl.Context(execution_id=None)
            pnl_torch_f = f._gen_pytorch_fct("cpu", context=context)

            x_torch = torch.tensor(x, dtype=torch.double)
            pnl_out = pnl_torch_f(x_torch).detach().cpu().numpy()
            ref_out = torch_reference_normalize(x_torch, eps, per_item).detach().cpu().numpy()

            np.testing.assert_allclose(pnl_out, ref_out, rtol=1e-7, atol=1e-9)

        @pytest.mark.parametrize(
            "x_init, target, eps, per_item",
            [
                pytest.param([3.0, 4.0], [1.0, -1.0], 1e-12, True, id="vector_unclamped"),
                pytest.param([3e-4, 4e-4], [1.0, -1.0], 1e-3, True, id="vector_clamped"),
                pytest.param([0.0, 0.0], [1.0, -1.0], 1e-3, True, id="vector_zero"),
                pytest.param([[3.0, 4.0], [3e-4, 4e-4]], [[1.0, -1.0], [0.5, 0.5]], 1e-3, True, id="matrix_per_item"),
                pytest.param([[3.0, 4.0], [3e-4, 4e-4]], [[1.0, -1.0], [0.5, 0.5]], 1e-3, False, id="matrix_global"),
            ],
        )
        def test_backward_matches_reference(self, x_init, target, eps, per_item):
            x_ref = torch.tensor(x_init, dtype=torch.double, requires_grad=True)
            x_pnl = torch.tensor(x_init, dtype=torch.double, requires_grad=True)
            target = torch.tensor(target, dtype=torch.double)

            f = pnl.Normalize(default_variable=x_init, eps=eps, per_item=per_item)
            context = pnl.Context(execution_id=None)
            pnl_torch_f = f._gen_pytorch_fct("cpu", context=context)

            y_ref = torch_reference_normalize(x_ref, eps, per_item)
            loss_ref = torch.sum((y_ref - target) ** 2)
            loss_ref.backward()

            y_pnl = pnl_torch_f(x_pnl)
            loss_pnl = torch.sum((y_pnl - target) ** 2)
            loss_pnl.backward()

            torch.testing.assert_close(y_pnl.detach(), y_ref.detach(), rtol=1e-7, atol=1e-9)
            torch.testing.assert_close(x_pnl.grad, x_ref.grad, rtol=1e-7, atol=1e-9)

        @pytest.mark.parametrize(
            "x, eps, per_item",
            [
                pytest.param([3e-4, 4e-4], 1e-3, True, id="vector_clamped"),
                pytest.param([[3.0, 4.0], [3e-4, 4e-4]], 1e-3, True, id="matrix_per_item"),
                pytest.param([[3.0, 4.0], [3e-4, 4e-4]], 1e-3, False, id="matrix_global"),
            ],
        )
        def test_autograd_jacobian_matches_reference(self, x, eps, per_item):
            x = torch.tensor(x, dtype=torch.double, requires_grad=True)

            f = pnl.Normalize(default_variable=np.array(x.detach().cpu().numpy()), eps=eps, per_item=per_item)
            context = pnl.Context(execution_id=None)
            pnl_torch_f = f._gen_pytorch_fct("cpu", context=context)

            jac_ref = torch.autograd.functional.jacobian(
                lambda z: torch_reference_normalize(z, eps, per_item), x
            )
            jac_pnl = torch.autograd.functional.jacobian(
                pnl_torch_f, x
            )

            torch.testing.assert_close(jac_pnl, jac_ref, rtol=1e-7, atol=1e-9)

        def test_autodiff_composition_learn_normalize_matches_torch(self):
            eps = 1e-3
            lr = 0.1

            seed = 0
            np.random.seed(seed)
            torch.manual_seed(seed)

            def torch_reference_normalize(x, eps=1e-3, per_item=True):
                if per_item and x.ndim > 1:
                    norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
                else:
                    norm = torch.linalg.norm(x, ord=2)

                denom = torch.clamp(norm, min=eps)
                return x / denom

            class SimpleForward(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fwd = torch.nn.Linear(2, 2, bias=False, dtype=torch.double)
                    with torch.no_grad():
                        self.fwd.weight.copy_(torch.eye(2))


                def forward(self, x):
                    x = torch_reference_normalize(x, eps=eps, per_item=True)
                    return self.fwd(x)

            x = torch.tensor(
                [[3.0, 4.0],
                 [3e-4, 4e-4]],
                dtype=torch.double,
            )
            target = torch.tensor(
                [[0.0, 1.0],
                 [1.0, 0.0]],
                dtype=torch.double,
            )

            # Torch: sequential updates to match trial-wise learning
            torch_model = SimpleForward()
            torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)
            torch_loss_fn = torch.nn.MSELoss()

            for xi, yi in zip(x, target):
                xi = xi.unsqueeze(0)
                yi = yi.unsqueeze(0)
                torch_optimizer.zero_grad()
                loss = torch_loss_fn(torch_model(xi), yi)
                loss.backward()
                torch_optimizer.step()

            with torch.no_grad():
                torch_after = torch_model(x).detach().cpu().numpy()

            np.random.seed(seed)
            torch.manual_seed(seed)

            input_mech = pnl.TransferMechanism(
                input_shapes=2,
                function=pnl.Normalize(
                    eps=eps,
                    per_item=True,
                ),
                name="INPUT",
            )

            mapping = pnl.MappingProjection(
                matrix=np.eye(2),
                name="MAP",
            )

            output_mech = pnl.TransferMechanism(
                input_shapes=2,
                name="OUTPUT",
            )

            comp = pnl.AutodiffComposition(
                learning_rate=lr,
                pathways=[[input_mech, mapping, output_mech]],
                loss_spec=pnl.Loss.MSE,
            )

            comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)

            comp.learn(
                inputs={input_mech: x.detach().cpu().numpy()},
                targets={output_mech: target.detach().cpu().numpy()},
                epochs=1,
                execution_mode=pnl.ExecutionMode.PyTorch,
            )

            comp.run(
                inputs={input_mech: x.detach().cpu().numpy()},
                execution_mode=pnl.ExecutionMode.PyTorch,
            )

            pnl_after = np.asarray(comp.results)[-2:, 0, :]

            np.testing.assert_allclose(
                pnl_after,
                torch_after,
                atol=1e-12,
                rtol=1e-12,
            )
