import psyneulink as pnl
import torch
import torch.nn as nn
import numpy as np
import random

# =====================
# GLOBAL CONFIG
# =====================
DIM = 3
LR = 1e-3
TRAINING_LOOPS = 1
TRAINING_EXAMPLES = 20
TEST_SAMPLES = 5
SEED = 42

torch.set_default_dtype(torch.float64)


# =====================
# SEED
# =====================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# =====================
# ROTATIONS
# =====================
def r_x(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float64)

def r_y(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float64)

def r_z(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float64)

def get_rotations():
    a = torch.rand(3) * 2 * torch.pi
    R = r_x(a[0]) @ r_y(a[1]) @ r_z(a[2])
    R_inv = r_z(-a[2]) @ r_y(-a[1]) @ r_x(-a[0])
    assert torch.allclose(R @ R_inv, torch.eye(3), atol=1e-6)
    return R, R_inv


# =====================
# TORCH MODEL
# =====================
class TorchTeacherStudent(nn.Module):
    def __init__(self, R, R_inv):
        super().__init__()

        self.l1 = nn.Linear(DIM, DIM, bias=False)
        self.l2 = nn.Linear(DIM, DIM, bias=False)
        self.student = nn.Linear(DIM, DIM, bias=False)
        self.out = nn.Linear(DIM, DIM, bias=False)

        self.teacher_hidden = nn.Linear(DIM, DIM, bias=False)
        self.teacher = nn.Linear(DIM, DIM, bias=False)

        with torch.no_grad():
            self.l1.weight.copy_(torch.eye(DIM))
            self.l2.weight.copy_(torch.eye(DIM))
            self.student.weight.copy_(torch.eye(DIM))
            self.out.weight.copy_(R_inv)

            self.teacher_hidden.weight.copy_(R)
            self.teacher.weight.copy_(torch.eye(DIM))

        for p in self.out.parameters():
            p.requires_grad = False
        for p in self.teacher_hidden.parameters():
            p.requires_grad = False
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        s = self.student(self.l1(x) + self.l2(x))
        t = self.teacher(self.teacher_hidden(x))
        out = self.out(s)
        return s, t, out


def run_torch(R, R_inv, train, test):
    model = TorchTeacherStudent(R, R_inv)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.MSELoss()

    for _ in range(TRAINING_LOOPS):
        s, t, _ = model(train)
        loss = loss_fn(s, t.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, _, out = model(test)
    return out.numpy()


# =====================
# PNL MODEL
# =====================
def build_pnl(R, R_inv):

    inp = pnl.ProcessingMechanism(input_shapes=DIM)
    l1 = pnl.ProcessingMechanism(input_shapes=DIM)
    l2 = pnl.ProcessingMechanism(input_shapes=DIM)
    student = pnl.ProcessingMechanism(input_shapes=DIM)
    out = pnl.ProcessingMechanism(input_shapes=DIM)

    th = pnl.ProcessingMechanism(input_shapes=DIM)
    teacher = pnl.ProcessingMechanism(input_shapes=DIM)

    comp = pnl.AutodiffComposition(
        pathways=[
            [inp, pnl.MappingProjection(matrix=np.eye(DIM), learnable=True, learning_rate=LR), l1,
             pnl.MappingProjection(matrix=np.eye(DIM), learnable=True, learning_rate=LR), student,
             pnl.MappingProjection(matrix=R_inv.numpy(), learnable=False), out],

            [inp, pnl.MappingProjection(matrix=np.eye(DIM), learnable=True, learning_rate=LR), l2, student],

            [inp, pnl.MappingProjection(matrix=R.numpy(), learnable=False), th,
             pnl.MappingProjection(matrix=np.eye(DIM), learnable=False), teacher]
        ],
        loss_spec=pnl.Loss.MSE,
        targets={student: teacher},
    )

    return comp, inp, out


def run_pnl(R, R_inv, train, test):
    comp, inp, out = build_pnl(R, R_inv)

    for _ in range(TRAINING_LOOPS):
        comp.learn(
            inputs={inp: train.numpy()},
            execution_mode=pnl.ExecutionMode.PyTorch
        )

    comp.run(
        inputs={inp: test.numpy()},
        execution_mode=pnl.ExecutionMode.PyTorch
    )


    return np.array(comp.results[-TEST_SAMPLES:])


# =====================
# MAIN TEST
# =====================
if __name__ == "__main__":

    set_seed(SEED)

    R, R_inv = get_rotations()

    train = torch.rand(TRAINING_EXAMPLES, DIM)
    test = torch.rand(TEST_SAMPLES, DIM)

    out_torch = run_torch(R, R_inv, train, test)
    out_pnl = run_pnl(R, R_inv, train, test)

    print("\nTorch output:\n", out_torch)
    print("\nPNL output:\n", out_pnl)
    print("\nInput:\n", test.numpy())

    # assert np.allclose(out_torch, test.numpy(), atol=1e-3), "Torch identity failed"
    # assert np.allclose(out_pnl, test.numpy(), atol=1e-3), "PNL identity failed"
    assert np.allclose(out_torch, out_pnl)

    print("\n✓ Torch and PNL both passed identity test.")
