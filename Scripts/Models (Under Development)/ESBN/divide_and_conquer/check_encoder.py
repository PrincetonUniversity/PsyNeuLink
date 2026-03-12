import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim




def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b))


def mean_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.mean(np.abs(a - b))


def print_comparison(name, a, b):
    print(f"{name}:")
    print(f"  shape A:        {np.asarray(a).shape}")
    print(f"  shape B:        {np.asarray(b).shape}")
    print(f"  max abs diff:   {max_abs_diff(a, b):.10f}")
    print(f"  mean abs diff:  {mean_abs_diff(a, b):.10f}")
    print()


def torch_forward(torch_model, x_numpy):
    """
    Run a pure forward pass of the Torch encoder.

    Parameters
    ----------
    torch_model : torch.nn.Module
        Encoder-only Torch model
    x_numpy : np.ndarray
        Shape [batch, 1, 32, 32]
    seed : int
        Seed set before forward for consistency

    Returns
    -------
    z_numpy : np.ndarray
        Encoder output, shape [batch, 128]
    """

    x_torch = torch.tensor(x_numpy, dtype=torch.float32)

    with torch.no_grad():
        z_torch = torch_model(x_torch)

    return z_torch.detach().cpu().numpy()


def main():
    from encoder.torch_model import Model
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_model = Model(encoder_type='mlp')
    x_numpy = np.random.normal(size=(1,  32, 32))
    print(x_numpy[0][0][0])
    t_out = torch_forward(torch_model=torch_model, x_numpy=x_numpy.copy())

    #print("torch output:", t_out)

    torch.manual_seed(seed)
    np.random.seed(seed)

    from encoder.pnl import mlp
    pnl_model, _input = mlp.create_model()

    x_flattened = x_numpy.flatten()
    print(x_flattened[0])

    pnl_out = mlp.run(pnl_model,_input, x_numpy=x_flattened)[0]

    #print("pnl output:", pnl_out)

    print(t_out.shape, pnl_out.shape)
    print('*** COMPARISON ***')
    print(t_out)
    print(pnl_out)







def torch_training(torch_model, x_numpy, z_target_numpy, lr=1e-3, steps=1, seed=0):
    """
    Train the Torch encoder directly on embedding targets using MSE:
        loss = MSE(z_pred, z_target)

    Parameters
    ----------
    torch_model : torch.nn.Module
        Encoder-only Torch model
    x_numpy : np.ndarray
        Shape [batch, 1, 32, 32]
    z_target_numpy : np.ndarray
        Shape [batch, 128]
    lr : float
        Learning rate
    steps : int
        Number of SGD updates
    seed : int
        Seed set before training

    Returns
    -------
    result : dict
        {
            "pre_z": ...,
            "post_z": ...,
            "losses": [...],
            "state_dict": {...}
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = copy.deepcopy(torch_model)
    model.train()

    x_torch = torch.tensor(x_numpy, dtype=torch.float32)
    z_target_torch = torch.tensor(z_target_numpy, dtype=torch.float32)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        pre_z = model(x_torch).detach().cpu().numpy()

    losses = []

    for _ in range(steps):
        optimizer.zero_grad()
        z_pred = model(x_torch)
        loss = loss_fn(z_pred, z_target_torch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        post_z = model(x_torch).detach().cpu().numpy()

    state_dict_numpy = {
        name: param.detach().cpu().numpy().copy()
        for name, param in model.state_dict().items()
    }

    return {
        "pre_z": pre_z,
        "post_z": post_z,
        "losses": losses,
        "state_dict": state_dict_numpy,
    }


def run_forward_comparison(
    torch_model,
    pnl_forward,
    batch_size=4,
    seed=0,
):
    """
    Compare Torch forward vs PNL forward on the same random input.

    Assumes you implement:
        pnl_forward(x_numpy, seed=...) -> z_numpy
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_numpy = np.random.randn(batch_size, 1, 32, 32).astype(np.float32)

    z_torch = torch_forward(torch_model=torch_model, x_numpy=x_numpy, seed=seed)
    z_pnl = pnl_forward(x_numpy=x_numpy, seed=seed)

    print("=" * 80)
    print("FORWARD COMPARISON")
    print("=" * 80)
    print_comparison("encoder output", z_torch, z_pnl)

    return {
        "x": x_numpy,
        "z_torch": z_torch,
        "z_pnl": z_pnl,
    }


def run_training_comparison(
    torch_model,
    pnl_training,
    batch_size=8,
    lr=1e-3,
    steps=1,
    seed=0,
):
    """
    Compare Torch training vs PNL training on the same input and same target embedding.

    Assumes you implement:
        pnl_training(x_numpy, z_target_numpy, lr=..., steps=..., seed=...) -> {
            "pre_z": ...,
            "post_z": ...,
            "losses": [...],
            "state_dict": {...},   # optional but recommended
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_numpy = np.random.randn(batch_size, 1, 32, 32).astype(np.float32)
    z_target_numpy = np.random.randn(batch_size, 128).astype(np.float32)

    torch_result = torch_training(
        torch_model=torch_model,
        x_numpy=x_numpy,
        z_target_numpy=z_target_numpy,
        lr=lr,
        steps=steps,
        seed=seed,
    )

    pnl_result = pnl_training(
        x_numpy=x_numpy,
        z_target_numpy=z_target_numpy,
        lr=lr,
        steps=steps,
        seed=seed,
    )

    print("=" * 80)
    print("TRAINING COMPARISON")
    print("=" * 80)

    print_comparison("pre-update encoder output", torch_result["pre_z"], pnl_result["pre_z"])
    print_comparison("post-update encoder output", torch_result["post_z"], pnl_result["post_z"])

    print("losses:")
    print("  torch:", torch_result["losses"])
    print("  pnl:  ", pnl_result["losses"])
    print()

    if "state_dict" in pnl_result:
        print("parameter comparisons:")
        for name, torch_param in torch_result["state_dict"].items():
            if name not in pnl_result["state_dict"]:
                print(f"  missing in pnl: {name}")
                continue

            pnl_param = pnl_result["state_dict"][name]
            print(
                f"  {name}: "
                f"max_abs_diff={max_abs_diff(torch_param, pnl_param):.10f}, "
                f"mean_abs_diff={mean_abs_diff(torch_param, pnl_param):.10f}"
            )
        print()

    return {
        "x": x_numpy,
        "z_target": z_target_numpy,
        "torch_result": torch_result,
        "pnl_result": pnl_result,
    }

if __name__ == "__main__":
    main()

