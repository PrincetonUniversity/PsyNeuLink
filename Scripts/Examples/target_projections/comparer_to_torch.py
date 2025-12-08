"""
Model
=====

            +-------------+
            | learn_net_1 |
inputs ---> |             + --> student --> outputs
     \      | learn_net_2 |      ^
      \     +-------------+     '
       \                       '
        \-> teach_hidden -> teacher (teacher for student node)


Learnable paths
---------------
- input -> learn_net_1 -> student
- input -> learn_net_2 -> student


Fixed paths
-----------
- input -> teach_hidden -> teacher
- student -> outputs


Test Procedure
==============

Setup
-----
input -> teach_hidden -> teacher:
This is the "teacher function" since the teacher value will be used as target for the student.
In our example this implements a random rotation in 3D space (ROT)

student -> outputs:
We fix this to be the inverse rotation to `input -> teach_hidden -> teacher` (ROT_INV)


Expectation
-----------
If `input -> ... -> student` really learns ROT, then the full `input -> ... -> student -> outputs` should
be the identity since it now implements input -> ROT (learned) -> ROT_INV (fixed) -> output
"""

import psyneulink as pnl

import torch
import torch.nn as nn

import numpy as np

# Static Constants (don't change)
DIM = 3

# Testable Params
LEARNING_RATE = .01
TRAINING_EXAMPLES = 1_000

# Script Control
RUN_TORCH = True
RUN_PNL = True
SHOW_PNL = False


#######################
# *** TORCH MODEL *** #
#######################
class LearnRotationsTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # ** Student Branch ** #
        self.learn_net_1 = nn.Linear(DIM, DIM, bias=False)
        self.learn_net_2 = nn.Linear(DIM, DIM, bias=False)

        # add [learn_net_1, learn_net_2] and project to the student
        self.student = nn.Linear(DIM, DIM, bias=False)

        # final output from student
        self.outputs = nn.Linear(DIM, DIM, bias=False)

        # ** Teacher Branch ** #
        self.teacher_hidden = nn.Linear(DIM, DIM, bias=False)
        self.teacher = nn.Linear(DIM, DIM, bias=False)

    def set_weights(self,
                    learn_net=torch.eye(DIM),
                    student=torch.eye(DIM),
                    outputs=torch.eye(DIM),
                    teacher_hidden=torch.eye(DIM),
                    teacher=torch.eye(DIM)) -> None:
        with torch.no_grad():
            self.learn_net_1.weight.copy_(learn_net)
            self.learn_net_2.weight.copy_(learn_net)
            self.student.weight.copy_(student)
            self.outputs.weight.copy_(outputs)
            self.teacher_hidden.weight.copy_(teacher_hidden)
            self.teacher.weight.copy_(teacher)

    def forward(self, x):
        # ** Student Branch ** #
        # Parallel learn nets, both taking the original input
        l1 = self.learn_net_1(x)
        l2 = self.learn_net_2(x)

        l_sum = .5 * l1 + .5 * l2

        # Learned student mapping (this is the one that should learn the teacher rotation)
        student = self.student(l_sum)

        # Fixed inverse rotation: student -> outputs
        out = self.outputs(student)  # weights of .outputs should be set to R^{-1} and frozen

        # ** Teacher Branch ** #
        t_hidden = self.teacher_hidden(x)
        teacher = self.teacher(t_hidden)

        return student, teacher, out


############################
# *** PSYNEULINK MODEL *** #
############################
def gen_learn_rotation_psyneulink(
        rot,
        rot_inv,
):
    # ** I/0 Nodes ** #
    inputs = pnl.ProcessingMechanism(name='INPUTS', input_shapes=DIM)
    outputs = pnl.ProcessingMechanism(name='OUTPUTS', input_shapes=DIM)

    # ** Student Branch ** #
    learn_net_1 = pnl.ProcessingMechanism(name='LEARN NET 1', input_shapes=DIM)
    learn_net_2 = pnl.ProcessingMechanism(name='LEARN NET 2', input_shapes=DIM)
    out_learn = pnl.ProcessingMechanism(name='OUT LEARN', input_shapes=DIM)
    student = pnl.ProcessingMechanism(name='STUDENT', input_shapes=DIM)

    # Learn Net
    pw_learn_net_1_out_learn = [
        learn_net_1,
        pnl.MappingProjection(
            learn_net_1, out_learn,
            matrix=pnl.IDENTITY_MATRIX, learnable=False,
        ),
        out_learn
    ]
    pw_learn_net_2_out_learn = [
        learn_net_2,
        pnl.MappingProjection(
            learn_net_2, out_learn,
            matrix=pnl.IDENTITY_MATRIX, learnable=False,
        ),
        out_learn
    ]

    learn_net = pnl.AutodiffComposition(pathways=[pw_learn_net_1_out_learn, pw_learn_net_2_out_learn], name='LEARN NET')

    # * Pathways * #
    # inputs -> learn_net (identity, not learnable)
    pw_inputs_learn_net_1 = [
        inputs,
        pnl.MappingProjection(
            inputs, learn_net_1,
            matrix=pnl.IDENTITY_MATRIX, learnable=True
        ),
        learn_net
    ]
    pw_inputs_learn_net_2 = [
        inputs,
        pnl.MappingProjection(
            inputs, learn_net_2,
            matrix=pnl.IDENTITY_MATRIX, learnable=True
        ),
        learn_net
    ]

    # learn_net -> student (identity, learnable should resemble rot after learning)
    pw_learn_net_student = [
        learn_net,
        pnl.MappingProjection(
            out_learn, student,
            matrix=pnl.IDENTITY_MATRIX, learnable=True, learning_rate=LEARNING_RATE
        ),
        student
    ]

    # student -> outputs (rot_inv, not learnable)
    pw_student_outputs = [
        student,
        pnl.MappingProjection(
            student, outputs,
            matrix=np.array(rot_inv), learnable=False
        ),
        outputs
    ]

    # ** Teacher Branch ** #
    teach_hidden = pnl.ProcessingMechanism(name='TEACH HIDDEN', input_shapes=DIM)
    teacher = pnl.ProcessingMechanism(name='TEACHER', input_shapes=DIM)

    # * Pathways * #

    pw_inputs_teacher_hidden = [
        inputs,
        pnl.MappingProjection(
            inputs, teach_hidden,
            matrix=pnl.IDENTITY_MATRIX, learnable=False
        ),
        teach_hidden
    ]
    pw_teach_hidden_teacher = [
        teach_hidden,
        pnl.MappingProjection(
            teach_hidden, teacher,
            matrix=np.array(rot), learnable=False
        ),
        teacher
    ]

    # ** Composition ** #
    comp = pnl.AutodiffComposition(
        pathways=[
            pw_inputs_learn_net_1,
            pw_inputs_learn_net_2,
            pw_learn_net_student,
            pw_student_outputs,
            pw_inputs_teacher_hidden,
            pw_teach_hidden_teacher,
        ],
        target=(teacher, student)
    )

    return comp, inputs


######################
# *** RUN SCRIPT *** #
######################
def run(seed=None):
    # Setup

    # get random angles
    psi = torch.rand(1) * 2 * torch.pi
    theta = torch.rand(1) * 2 * torch.pi
    phi = torch.rand(1) * 2 * torch.pi

    # get rotation matrices
    rot, rot_inv = get_rotation_matrices(psi, theta, phi)

    # rot = torch.eye(3)
    # rot_inv = torch.eye(3)

    # get training data
    train = torch.rand(TRAINING_EXAMPLES, 3)

    test_targets = torch.rand(5, 3)
    out_torch = None
    out_pnl = None
    #####################
    # ** Torch Model ** #
    #####################
    if RUN_TORCH:
        torch.manual_seed(seed)
        model_t = LearnRotationsTorch()

        # set the teacher hidden to be a rotation and the
        # output mapping to be the inverse
        # since the student network should learn the teacher
        # function, the full input -> output path should be
        # an identity after learning
        model_t.set_weights(teacher_hidden=rot, outputs=rot_inv)

        optimizer = torch.optim.Adam(model_t.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()

        student, teacher, out = model_t(train)
        # detach from teacher since we don't want to train the teacher net
        target_value = teacher.detach()
        loss = loss_fn(student, target_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        student_, teacher_, out_torch = model_t(test_targets)

    if RUN_PNL:
        torch.manual_seed(seed)
        model_pnl, inputs = gen_learn_rotation_psyneulink(rot, rot_inv)
        if SHOW_PNL:
            model_pnl.show_graph(show_learning=True)
        model_pnl.learn(inputs={
            inputs: np.array(train)
        })

        model_pnl.run(inputs={inputs: test_targets})

        out_pnl = [r[1] for r in model_pnl.results[-5:]]

    print(test_targets)
    if out_torch is not None and out_pnl is not None:
        for res in zip(out_torch, out_pnl):
            print(res)


##########################
# ** Helper Functions ** #
##########################
def get_rotation_matrices(
        psi,
        theta,
        phi
):
    """
    Get two fixed rotation matrices:

    - The teacher matrix is a rotation around psi, theta, and phi
    - The student to output matrix is the inverse of the teacher matrix

    -> If learn_net learns the correct transformation, the output should be equal to the input
    """
    # ensure angles are in [0, 2π)
    assert 0 <= psi < 2 * torch.pi
    assert 0 <= theta < 2 * torch.pi
    assert 0 <= phi < 2 * torch.pi

    # teacher rotation
    rotation = _r_x(psi) @ _r_y(theta) @ _r_z(phi)

    # correct inverse
    rotation_inv = _r_z(2 * torch.pi - phi) @ \
                   _r_y(2 * torch.pi - theta) @ \
                   _r_x(2 * torch.pi - psi)

    # numerical identity check
    assert torch.allclose(rotation @ rotation_inv, torch.eye(3), atol=1e-6)

    return rotation, rotation_inv


def _r_x(psi):
    """
    roll (rotation about x-axis)
    """
    c = torch.cos(psi)
    s = torch.sin(psi)
    return torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=torch.float32)


def _r_y(theta):
    """
    pitch (rotation about y-axis)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def _r_z(phi):
    """
    yaw (rotation about z-axis)
    """
    c = torch.cos(phi)
    s = torch.sin(phi)
    return torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


if __name__ == "__main__":
    import random

    run(random.randint(0, int(1e12)))
