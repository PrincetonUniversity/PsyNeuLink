import psyneulink as pnl
import torch
import torch.nn as nn
import numpy as np
import random

torch.set_default_dtype(torch.float64)

DIM = 3
LR = 1
N = 10

def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

############################
# PyTorch single-layer model
############################
class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        return self.w(x)

############################
# PNL single MappingProjection
############################
def make_pnl():
    inp = pnl.ProcessingMechanism(input_shapes=DIM)
    out = pnl.ProcessingMechanism(input_shapes=DIM)

    proj = pnl.MappingProjection(
        sender=inp,
        receiver=out,
        matrix=np.eye(DIM),
        learnable=True,
        learning_rate=LR,
    )

    comp = pnl.AutodiffComposition(
        pathways=[inp, proj, out],
        loss_spec=pnl.Loss.MSE,
    )
    return comp, inp, out, proj

############################
# Run comparison
############################
def run(seed=42):
    set_seed(seed)

    # Training data
    X = torch.rand(N, DIM)
    Y = 2 * X

    ########### TORCH ###########
    t = TorchModel()
    t.w.weight.data[:] = torch.eye(DIM)

    opt = torch.optim.SGD(t.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for x, y in zip(X, Y):
        opt.zero_grad()
        loss = loss_fn(t(x), y)
        loss.backward()
        opt.step()

    torch_W = t.w.weight.detach().numpy().copy()

    ########### PNL #############
    set_seed(seed)
    comp, inp, out, proj = make_pnl()

    for x, y in zip(X, Y):
        comp.learn(
            inputs={inp: [x.detach().numpy()]},
            targets={out: [y.detach().numpy()]},
            execution_mode=pnl.ExecutionMode.PyTorch,
        )

    pnl_W = proj.parameters.matrix.get(comp)

    print("\nTORCH updated W:\n", torch_W)
    print("\nPNL updated W:\n", pnl_W.T)  # seems to be a transpose ?

    print("\nallclose? ", np.allclose(torch_W, pnl_W.T, atol=1e-12))

if __name__ == "__main__":
    run()