import psyneulink as pnl

import random

import torch
import torch.nn as nn

import numpy as np

torch.set_default_dtype(torch.float64)

# Static Constants (don't change)
DIM = 3

# Testable Params
LEARNING_RATE = .1
TRAINING_EXAMPLES = 20

# Script Control
RUN_TORCH = True
RUN_PNL = True
SHOW_PNL = True


#######################
# *** TORCH MODEL *** #
#######################
class SingleLayerMLPTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = nn.Linear(DIM, DIM, bias=False)

    def set_weights(self,
                    outputs=torch.eye(DIM),
                    ) -> None:
        with torch.no_grad():
            self.outputs.weight.copy_(outputs)

    def forward(self, x):
        out = self.outputs(x)
        return out


############################
# *** PSYNEULINK MODEL *** #
############################
def gen_single_layer_mlp_psyneulink_lr_on_projection():
    # ** I/0 Nodes ** #
    inputs = pnl.ProcessingMechanism(name='INPUTS', input_shapes=DIM)
    outputs = pnl.ProcessingMechanism(name='OUTPUTS', input_shapes=DIM)

    # * Pathways * #
    pw_inputs_outputs = [
        inputs,
        pnl.MappingProjection(
            inputs, outputs,
            name='io',
            matrix=pnl.IDENTITY_MATRIX, learnable=True, learning_rate=LEARNING_RATE,
        ),
        outputs,
    ]

    # ** Composition ** #
    comp = pnl.AutodiffComposition(
        pathways=[
            pw_inputs_outputs,
        ],
        loss_spec=pnl.Loss.MSE
    )
    return comp, inputs, outputs

def gen_single_layer_mlp_psyneulink_lr_on_autodiff():
    # ** I/0 Nodes ** #
    inputs = pnl.ProcessingMechanism(name='INPUTS', input_shapes=DIM)
    outputs = pnl.ProcessingMechanism(name='OUTPUTS', input_shapes=DIM)

    # * Pathways * #
    pw_inputs_outputs = [
        inputs,
        pnl.MappingProjection(
            inputs, outputs,
            name='io',
            matrix=pnl.IDENTITY_MATRIX, learnable=True
        ),
        outputs,
    ]

    # ** Composition ** #
    comp = pnl.AutodiffComposition(
        pathways=[
            pw_inputs_outputs,
        ],
        loss_spec=pnl.Loss.MSE,
        learning_rate=LEARNING_RATE,
    )
    return comp, inputs, outputs


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


######################
# *** RUN SCRIPT *** #
######################
def run(seed=None):
    # Setup
    set_random_seed(seed)

    # get training data
    train = torch.rand(TRAINING_EXAMPLES, 3)
    targets = 2 * train

    test_targets = torch.rand(5, 3)
    out_torch = None
    out_pnl = None
    #####################
    # ** Torch Model ** #
    #####################
    if RUN_TORCH:
        set_random_seed(seed)
        model_t = SingleLayerMLPTorch()
        model_t.set_weights()

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
        set_random_seed(seed)
        model_pnl_pr, inputs_pr, outputs_pr = gen_single_layer_mlp_psyneulink_lr_on_projection()
        if SHOW_PNL:
            model_pnl_pr.show_graph(show_learning=True)
        for _tr, _ta in zip(train, targets):
            np_tr = _tr.detach().numpy().astype(np.float64)
            np_ta = _ta.detach().numpy().astype(np.float64)
            model_pnl_pr.learn(
                inputs={inputs_pr: [np_tr]},
                targets={outputs_pr: [np_ta]},
                execution_mode=pnl.ExecutionMode.PyTorch,
            )

        model_pnl_pr.run(inputs={inputs_pr: np.array(test_targets)})
        out_pnl_pr = model_pnl_pr.results[-5:]

        set_random_seed(seed)
        model_pnl_ad, inputs_ad, outputs_ad = gen_single_layer_mlp_psyneulink_lr_on_autodiff()
        if SHOW_PNL:
            model_pnl_ad.show_graph(show_learning=True)
        for _tr, _ta in zip(train, targets):
            np_tr = _tr.detach().numpy().astype(np.float64)
            np_ta = _ta.detach().numpy().astype(np.float64)
            model_pnl_ad.learn(
                inputs={inputs_ad: [np_tr]},
                targets={outputs_ad: [np_ta]},
                execution_mode=pnl.ExecutionMode.PyTorch,
            )

        model_pnl_ad.run(inputs={inputs_ad: np.array(test_targets)})
        out_pnl_ad = model_pnl_pr.results[-5:]

    if out_torch is not None and out_pnl is not None:
        for res in zip(out_torch, out_pnl_pr, out_pnl_ad):
            tr = np.array(res[0].detach(), dtype=np.float64)
            pl_pr = np.array(res[1], dtype=np.float64)
            pl_ad = np.array(res[2], dtype=np.float64)


            assert np.allclose(tr, pl_pr), f"Not close enough {tr}, {pl_pr}"
            assert np.allclose(tr, pl_ad), f"Not close enough {tr}, {pl_ad}"

    # print('ALL CLOSE')


if __name__ == "__main__":
    seed = 42  # random.randint(0, int(1e12))
    run(seed)