#!/usr/bin/env python3

"""
Minimal Torch vs PsyNeuLink training debug script.

This script tests the simplest possible case:
- one linear layer
- no bias
- no activation
- one sample
- one target
- one SGD step

It compares:
1. initial forward output
2. output after one learning step
3. final weights
4. weight update delta

Important:
PsyNeuLink parameter values are context-dependent. Reading
proj.parameters.matrix.get(None) may show the default matrix,
not the updated matrix used by the current execution context.

Use:
    proj.parameters.matrix.get(model.most_recent_context)
instead.
"""

import numpy as np
import torch
import torch.nn as nn
import psyneulink as pnl


# ----------------------------
# Torch model
# ----------------------------

class TorchLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, weight_matrix_in_out: np.ndarray):
        """
        weight_matrix_in_out: shape [in_dim, out_dim]
        Torch Linear stores weights as [out_dim, in_dim], so we transpose.
        """
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        with torch.no_grad():
            self.fc.weight.copy_(
                torch.tensor(weight_matrix_in_out.T, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


# ----------------------------
# PNL model
# ----------------------------

def create_pnl_linear(
    weight_matrix_in_out: np.ndarray,
    learning_rate: float = 0.01,
):
    """
    Create a one-layer linear AutodiffComposition.

    weight_matrix_in_out: shape [in_dim, out_dim]
    """
    in_dim, out_dim = weight_matrix_in_out.shape

    input_mech = pnl.TransferMechanism(
        name="input",
        input_shapes=in_dim,
        function=pnl.Linear(),
    )

    output_mech = pnl.TransferMechanism(
        name="output",
        input_shapes=out_dim,
        function=pnl.ReLU(),
    )

    proj = pnl.MappingProjection(
        name="input_to_output",
        sender=input_mech,
        receiver=output_mech,
        matrix=weight_matrix_in_out,
        learnable=True,
    )

    comp = pnl.AutodiffComposition(
        name="LinearComp",
        pathways=[[input_mech, proj, output_mech]],
        learning_rate=learning_rate,
        loss_spec=pnl.Loss.MSE,
        device=pnl.CPU,
    )

    return comp, input_mech, output_mech, proj


def pnl_run(model, input_mech, x_numpy: np.ndarray) -> np.ndarray:
    """
    Run forward once and return only the output of this call.
    """
    result = model.run(
        inputs={input_mech: x_numpy},
        execution_mode=pnl.ExecutionMode.PyTorch,
        # synch_projection_matrices_with_torch=pnl.RUN,
        # synch_node_values_with_torch=pnl.RUN,
        # synch_results_with_torch=pnl.RUN,
    )
    return np.array(result, dtype=np.float32)


def pnl_learn(
    model,
    input_mech,
    output_mech,
    x_numpy: np.ndarray,
    target_numpy: np.ndarray,
    learning_rate: float = 0.01,
) -> np.ndarray:
    """
    Run one learning step and return the result of that call.
    """
    result = model.learn(
        inputs={input_mech: x_numpy},
        targets={output_mech: target_numpy},
        learning_rate=learning_rate,
        execution_mode=pnl.ExecutionMode.PyTorch,
        synch_projection_matrices_with_torch=pnl.RUN,
        synch_node_values_with_torch=pnl.RUN,
        synch_results_with_torch=pnl.RUN,
    )
    return np.array(result, dtype=np.float32)


def get_pnl_matrix(proj, model) -> np.ndarray:
    """
    Read the projection matrix from the model's current execution context.
    """
    ctx = model.most_recent_context
    return np.array(proj.parameters.matrix.get(ctx), dtype=np.float32).copy()


# ----------------------------
# Debug test
# ----------------------------

def main():
    seed = 42
    lr = 0.01
    in_dim = 4
    out_dim = 3

    np.random.seed(seed)
    torch.manual_seed(seed)

    # One sample, explicit batch dimension
    x = np.random.randn(1, in_dim).astype(np.float32)
    target = np.random.randn(1, out_dim).astype(np.float32)

    # Shared initial weight matrix in [in, out] format
    W_init = np.random.randn(in_dim, out_dim).astype(np.float32)

    print("=== INPUT ===")
    print(x)
    print("\n=== TARGET ===")
    print(target)
    print("\n=== INITIAL WEIGHTS [in, out] ===")
    print(W_init)

    # ----------------------------
    # Torch side
    # ----------------------------
    torch_model = TorchLinear(in_dim, out_dim, W_init)
    x_t = torch.tensor(x, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.float32)

    optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

    torch_out_0 = torch_model(x_t).detach().cpu().numpy()
    torch_W_0 = torch_model.fc.weight.detach().cpu().numpy().T.copy()

    optimizer.zero_grad()
    torch_out_for_loss = torch_model(x_t)
    torch_loss = ((torch_out_for_loss - target_t) ** 2).mean()
    torch_loss.backward()
    optimizer.step()

    torch_out_1 = torch_model(x_t).detach().cpu().numpy()
    torch_W_1 = torch_model.fc.weight.detach().cpu().numpy().T.copy()

    # ----------------------------
    # PNL side
    # ----------------------------
    pnl_model, pnl_input, pnl_output, pnl_proj = create_pnl_linear(
        W_init,
        learning_rate=lr,
    )

    pnl_out_0 = pnl_run(pnl_model, pnl_input, x)
    pnl_W_0 = get_pnl_matrix(pnl_proj, pnl_model)

    _ = pnl_learn(
        pnl_model,
        pnl_input,
        pnl_output,
        x,
        target,
        learning_rate=lr,
    )

    pnl_out_1 = pnl_run(pnl_model, pnl_input, x)
    pnl_W_1 = get_pnl_matrix(pnl_proj, pnl_model)

    # ----------------------------
    # Diagnostics
    # ----------------------------
    torch_delta = torch_W_1 - torch_W_0
    pnl_delta = pnl_W_1 - pnl_W_0

    print("\n\n=== INITIAL OUTPUTS ===")
    print("torch_out_0:")
    print(torch_out_0)
    print("pnl_out_0:")
    print(pnl_out_0)
    print("max initial output diff:", np.max(np.abs(torch_out_0 - pnl_out_0)))

    print("\n=== UPDATED OUTPUTS ===")
    print("torch_out_1:")
    print(torch_out_1)
    print("pnl_out_1:")
    print(pnl_out_1)
    print("max updated output diff:", np.max(np.abs(torch_out_1 - pnl_out_1)))

    print("\n=== FINAL WEIGHTS ===")
    print("torch_W_1:")
    print(torch_W_1)
    print("pnl_W_1:")
    print(pnl_W_1)
    print("max final weight diff:", np.max(np.abs(torch_W_1 - pnl_W_1)))

    print("\n=== WEIGHT DELTAS ===")
    print("torch_delta:")
    print(torch_delta)
    print("pnl_delta:")
    print(pnl_delta)
    print("max delta diff:", np.max(np.abs(torch_delta - pnl_delta)))

    print("\n=== PNL MATRIX READOUT DEBUG ===")
    print("pnl matrix default context (get(None)):")
    print(np.array(pnl_proj.parameters.matrix.get(None), dtype=np.float32))
    print("pnl matrix current execution context:")
    print(get_pnl_matrix(pnl_proj, pnl_model))

    # ----------------------------
    # Assertions
    # ----------------------------
    np.testing.assert_allclose(torch_W_0, W_init, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(torch_out_0, pnl_out_0, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(torch_W_1, pnl_W_1, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(torch_out_1, pnl_out_1, atol=1e-6, rtol=1e-6)

    print("\nPASS: Torch and PNL match for the one-layer linear learning test.")


if __name__ == "__main__":
    main()