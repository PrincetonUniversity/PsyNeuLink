import psyneulink as pnl

import random

import torch
import torch.nn as nn

import numpy as np

torch.set_default_dtype(torch.float64)

DIM = 3
LEARNING_RATE = .1
TRAINING_EXAMPLES = 10

# Script Control
RUN_TORCH = True
RUN_PNL = True
SHOW_PNL = True


#######################
# *** TORCH MODEL *** #
#######################
class SingleLayerPerceptronT(nn.Module):
    def __init__(self, matrix=None):
        super().__init__()
        self.layer = nn.Linear(DIM, DIM, bias=False)
        w = torch.eye(DIM) if matrix is None else matrix
        with torch.no_grad():
            self.layer.weight.copy_(w)

    def forward(self, x):
        return self.layer(x)


############################
# *** PSYNEULINK MODEL *** #
############################
def gen_single_layer_mlp_psyneulink(
        matrix=None,
        loss_spec=pnl.Loss.MSE
):
    # ** I/0 Nodes ** #
    inputs = pnl.ProcessingMechanism(name='INPUTS', input_shapes=DIM)
    outputs = pnl.ProcessingMechanism(name='OUTPUTS', input_shapes=DIM)

    # * Pathways * #
    w = pnl.IDENTITY_MATRIX if matrix is None else matrix
    pw_inputs_outputs = [
        inputs,
        pnl.MappingProjection(
            inputs, outputs,
            matrix=w, learnable=True, learning_rate=LEARNING_RATE,
        ),
        outputs,
    ]

    # ** Composition ** #
    comp = pnl.AutodiffComposition(
        pathways=[
            pw_inputs_outputs,
        ],
        loss_spec=loss_spec,
    )
    return comp, inputs, outputs


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


######################
# *** RUN SCRIPT *** #
######################
def run(seed=None):
    # Setup
    set_seed(seed)

    # get training data
    train = torch.rand(TRAINING_EXAMPLES, DIM)
    targets = 2 * train

    test_targets = torch.rand(5, DIM)
    out_torch = None
    out_pnl = None

    #####################
    # ** Torch Model ** #
    #####################
    if RUN_TORCH:
        set_seed(seed)
        model_t = SingleLayerPerceptronT()

        optimizer = torch.optim.SGD(model_t.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        for _tr, _ta in zip(train, targets):
            out = model_t(_tr)
            loss = loss_fn(out, _ta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        out_torch = model_t(test_targets)

    if RUN_PNL:
        set_seed(seed)
        model_pnl, inputs, outputs = gen_single_layer_mlp_psyneulink()
        if SHOW_PNL:
            model_pnl.show_graph(show_learning=True)
        for _tr, _ta in zip(train, targets):
            np_tr = _tr.detach().numpy().astype(np.float64)
            np_ta = _ta.detach().numpy().astype(np.float64)
            model_pnl.learn(
                inputs={inputs: np_tr},
                targets={outputs: np_ta},
                execution_mode=pnl.ExecutionMode.PyTorch,
            )

        model_pnl.run(inputs={inputs: np.array(test_targets)})
        out_pnl = model_pnl.results[-5:]

    if out_torch is not None and out_pnl is not None:
        for res in zip(out_torch, out_pnl):
            tr = np.array(res[0].detach(), dtype=np.float64)
            pl = np.array(res[1], dtype=np.float64)

            assert np.allclose(tr, pl), f"Not close enough {tr}, {pl}"

    # print('ALL CLOSE')


if __name__ == "__main__":
    seed = 42  # random.randint(0, int(1e12))
    run(seed)
