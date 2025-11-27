"""
Architecture
============

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
--------------

Setup
*****
input -> teach_hidden -> teacher:
This is the "teacher function" since the teacher value will be used as target for the student.
In our example this implements a random rotation in 3D space (ROT)

student -> outputs:
We fix this to be the inverse rotation to `input -> teach_hidden -> teacher` (ROT_INV)

Expectation
************

If `input -> ... -> student` really learns ROT, then the full `input -> ... -> student -> outputs` should
be the identity since it now implements input -> ROT (learned) -> ROT_INV (fixed) -> output
"""


import torch
import torch.nn as nn


DIM = 3

TRAINING_LOOPS = 10000
TRAINING_EXAMPLES = 10000
LEARNING_RATE = 1e-3



class InnerTarget(nn.Module):
    """
    Architecture:

                +-------------+
                | learn_net_1 |
    inputs ---> |             + --> student --> outputs
         \      | learn_net_2 |      ^
          \     +-------------+     '
           \                       '
            \-> teach_hidden -> teacher (teacher for student node)
    """

    def __init__(
            self,
            dim_inputs: int = DIM,
            dim_learn_net: int = DIM,
            dim_teach_hidden: int = DIM,
            dim_student: int = DIM,
            dim_teacher: int = DIM,
            dim_out: int = DIM,
            activation_learn_net: nn.Module = nn.Identity(),
            activation_teach_net: nn.Module = nn.Identity(),
            _activation_other: nn.Module = nn.Identity(),

    ) -> None:
        super().__init__()

        # ** Student Branch ** #
        # inputs -> [learn_net] -> student -> outputs

        # learn net
        self.learn_net_1 = nn.Linear(dim_inputs, dim_learn_net, bias=False)
        self.learn_net_2 = nn.Linear(dim_inputs, dim_learn_net, bias=False)

        # add [learn_net_1, learn_net_2] and project to the student
        self.student = nn.Linear(dim_learn_net, dim_student, bias=False)

        # final output from student
        self.outputs = nn.Linear(dim_learn_net, dim_out, bias=False)

        # ** Teacher Branch ** #
        self.teacher_hidden = nn.Linear(dim_inputs, dim_teach_hidden, bias=False)
        self.teacher = nn.Linear(dim_teach_hidden, dim_teacher, bias=False)

        self.activation_learn_net = activation_learn_net
        self.activation_teach_net = activation_teach_net
        self._activation_other = _activation_other

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
        l1 = self.activation_learn_net(self.learn_net_1(x))
        l2 = self.activation_learn_net(self.learn_net_2(x))

        l_sum = l1 + l2

        # Learned student mapping (this is the one that should learn the teacher rotation)
        student_pre = self.student(l_sum)
        student = self._activation_other(student_pre)

        # Fixed inverse rotation: student -> outputs
        out_pre = self.outputs(student)  # weights of .outputs should be set to R^{-1} and frozen
        out = self._activation_other(out_pre)

        # ** Teacher Branch ** #
        t_hidden = self.activation_teach_net(self.teacher_hidden(x))
        t_pre = self.teacher(t_hidden)
        teacher = self._activation_other(t_pre)

        return student, teacher, out


def run(seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = InnerTarget()

    psi = torch.rand(1) * 2 * torch.pi
    theta = torch.rand(1) * 2 * torch.pi
    phi = torch.rand(1) * 2 * torch.pi

    rot, rot_inv = get_rotation_matrices(psi, theta, phi)

    # Keep copies for tests later
    original_teacher_weights = rot.clone()
    original_output_weights = rot_inv.clone()

    # set the teacher hidden to be a rotation and the
    # output mapping to be the inverse
    # since the student network should learn the teacher
    # function, the full input -> output path should be
    # an identity after learning
    model.set_weights(teacher_hidden=rot, outputs=rot_inv)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train = torch.rand(TRAINING_EXAMPLES, 3)

    for step in range(TRAINING_LOOPS):
        student, teacher, out = model(train)

        # detach from teacher since we don't want to train the teacher net
        target_value = teacher.detach()
        loss = loss_fn(student, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # *** Test *** #
    # (1) Teacher shouldn't learn
    assert torch.allclose(model.teacher_hidden.weight, original_teacher_weights), "Teacher weights do not match"

    # (2) Student to output shouldn't learn
    assert torch.allclose(model.outputs.weight, original_output_weights), "Output weights do not match"

    test_targets = torch.rand(5, 3)
    student_, teacher_, out_ = model(test_targets)

    print("\n=== FINAL TEST INPUTS ===")
    print(test_targets)
    print("\n=== FINAL TEST OUTPUTS ===")
    print(out_)

    # ---- Identity Test ----
    assert torch.allclose(out_, test_targets, atol=1e-3), \
        f"Identity test failed.\nExpected:\n{test_targets}\nGot:\n{out_}"

    print("\n✓ Identity test passed!  (out ≈ x)")



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
    run()