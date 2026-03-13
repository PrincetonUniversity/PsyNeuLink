import numpy as np
import torch


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
    x_torch = torch.tensor(x_numpy, dtype=torch.float32)
    with torch.no_grad():
        z_torch = torch_model(x_torch)
    return z_torch.detach().cpu().numpy()


def main_backward_mlp_final_forward_check():
    from encoder.torch_model import Model
    from encoder.pnl import mlp

    seed = 42
    n_steps = 2
    lr = 0.01

    torch.manual_seed(seed)
    np.random.seed(seed)

    # -------------------------
    # initialize torch model
    # -------------------------
    torch_model = Model(encoder_type="mlp")

    x_numpy = np.random.normal(size=(1, 32, 32)).astype(np.float32)
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)

    # extract initial weights for PNL
    matrix_in_hidden_1 = torch_model.encoder.fc1.weight.T.detach().cpu().numpy().copy()
    matrix_hidden_1_hidden_2 = torch_model.encoder.fc2.weight.T.detach().cpu().numpy().copy()
    matrix_hidden_2_out = torch_model.encoder.fc3.weight.T.detach().cpu().numpy().copy()

    # initial torch output
    with torch.no_grad():
        torch_initial_out = torch_model(x_tensor).detach().cpu().numpy()

    target_numpy = np.random.normal(size=torch_initial_out.shape).astype(np.float32)
    target_tensor = torch.tensor(target_numpy, dtype=torch.float32)

    # -------------------------
    # torch training history
    # -------------------------
    torch_outputs = []
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=lr)

    for _ in range(n_steps):
        out = torch_model(x_tensor)
        torch_outputs.append(out.detach().cpu().numpy())

        optimizer.zero_grad()
        loss = loss_fn(out, target_tensor)
        loss.backward()
        optimizer.step()

    torch_outputs = np.array(torch_outputs)

    # final torch forward after training
    with torch.no_grad():
        torch_final_out = torch_model(x_tensor).detach().cpu().numpy()

    torch_w1 = torch_model.encoder.fc1.weight.detach().cpu().numpy().T.copy()
    torch_w2 = torch_model.encoder.fc2.weight.detach().cpu().numpy().T.copy()
    torch_w3 = torch_model.encoder.fc3.weight.detach().cpu().numpy().T.copy()

    # -------------------------
    # initialize pnl model
    # -------------------------
    (
        pnl_model,
        _input,
        _output,
        input_hidden_1_mp,
        hidden_1_hidden_2_mp,
        hidden_2_output_mp,
    ) = mlp.create_model(
        matrix_in_hidden_1=matrix_in_hidden_1,
        matrix_hidden_1_hidden_2=matrix_hidden_1_hidden_2,
        matrix_hidden_2_out=matrix_hidden_2_out,
        learning_rate=lr,
    )

    # same shape convention as your working forward test
    x_trial = x_numpy.flatten()               # (1024,)
    target_trial = target_numpy.squeeze()     # (128,)

    # initial pnl forward
    pnl_initial_out = np.array(mlp.run(pnl_model, _input, x_numpy=x_trial)[0])

    # single-call learn history
    input_list = [x_trial.copy() for _ in range(n_steps)]
    target_list = [target_trial.copy() for _ in range(n_steps)]

    pnl_history = mlp.learn(
        pnl_model,
        _input,
        _output,
        x_numpy=input_list,
        target_numpy=target_list,
        learning_rate=lr,
    )
    pnl_history = np.array(pnl_history)

    # final pnl forward after training
    pnl_final_out = np.array(mlp.run(pnl_model, _input, x_numpy=x_trial)[-1])

    pnl_w1 = mlp.get_matrix(input_hidden_1_mp, pnl_model)
    pnl_w2 = mlp.get_matrix(hidden_1_hidden_2_mp, pnl_model)
    pnl_w3 = mlp.get_matrix(hidden_2_output_mp, pnl_model)

    # -------------------------
    # print diagnostics
    # -------------------------
    print("torch_outputs.shape:", torch_outputs.shape)
    print("pnl_history.shape:  ", pnl_history.shape)
    print()

    print("*** TORCH TRAINING HISTORY (manual pre-update outputs) ***")
    print(torch_outputs)
    print()

    print("*** PNL LEARN HISTORY (model.results from learn call) ***")
    print(pnl_history)
    print()

    print("*** INITIAL OUTPUTS ***")
    print("torch_initial_out:")
    print(torch_initial_out)
    print("pnl_initial_out:")
    print(pnl_initial_out)
    print()
    print_comparison("initial output", torch_initial_out, pnl_initial_out)

    print("*** FINAL OUTPUTS AFTER TRAINING ***")
    print("torch_final_out:")
    print(torch_final_out)
    print("pnl_final_out:")
    print(pnl_final_out)
    print()
    print_comparison("final output", torch_final_out, pnl_final_out)

    print("*** FINAL MATRICES ***")
    print_comparison("fc1 matrix", torch_w1, pnl_w1)
    print_comparison("fc2 matrix", torch_w2, pnl_w2)
    print_comparison("fc3 matrix", torch_w3, pnl_w3)

    # -------------------------
    # assertions that actually test the claim
    # -------------------------
    # 1. known-good starting point
    np.testing.assert_allclose(
        torch_initial_out,
        pnl_initial_out,
        atol=1e-6,
        rtol=1e-6,
    )

    # 2. final trained model should be close
    np.testing.assert_allclose(
        torch_final_out,
        pnl_final_out,
        atol=5e-3,
        rtol=5e-3,
    )

    # 3. trained weights should be close
    np.testing.assert_allclose(torch_w1, pnl_w1, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(torch_w2, pnl_w2, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(torch_w3, pnl_w3, atol=5e-3, rtol=5e-3)

    print("PASS: final trained state matches closely; learn-history outputs are not the right thing to compare directly.")


if __name__ == "__main__":
    main_backward_mlp_final_forward_check()