"""
Model
=====

inputs ---> student --> outputs
     \         ^
      \        '
       \ -> teacher (teacher for student node)                     '


Learnable paths
---------------
- inputs  -> student


Fixed paths
-----------
- inputs  -> teacher
- students -> outputs

"""

import psyneulink as pnl

import torch
import torch.nn as nn

import numpy as np

# Static Constants (don't change)
DIM = 3

# Testable Params
LEARNING_RATE = .01
TRAINING_EXAMPLES = 1

# Script Control
RUN_TORCH = True
RUN_PNL = True
SHOW_PNL = True

torch.set_default_dtype(torch.float64)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#######################
# *** TORCH MODEL *** #
#######################
class TeacherStudentNet(nn.Module):
    def __init__(self,
                 teacher_matrix,
                 student_matrix=None,
                 student_output_matrix=None) -> None:
        super().__init__()

        # ** Student Branch ** #
        self.student = nn.Linear(DIM, DIM, bias=False)

        # final output from student
        self.outputs = nn.Linear(DIM, DIM, bias=False)

        # ** Teacher Branch ** #
        self.teacher = nn.Linear(DIM, DIM, bias=False)

        # Set initial matrices
        if student_matrix is None:
            student_matrix = torch.eye(DIM)
        if student_output_matrix is None:
            student_output_matrix = torch.eye(DIM)

        with torch.no_grad():
            self.teacher.weight.copy_(teacher_matrix)
            self.student.weight.copy_(student_matrix)
            self.outputs.weight.copy_(student_output_matrix)

    def forward(self, x):
        # ** Student Branch ** #
        # Learned student mapping (this is the one that should learn the teacher rotation)
        student = self.student(x)

        # Fixed identity: student -> outputs
        out = self.outputs(student)  # weights of .outputs should be set to R^{-1} and frozen

        # ** Teacher Branch ** #
        teacher = self.teacher(x)

        return student, teacher, out


############################
# *** PSYNEULINK MODEL *** #
############################
def gen_teacher_student_model(
        teacher_matrix,
        student_matrix=None,
        student_output_matrix=None):
    # ** I/0 Nodes ** #
    inputs = pnl.ProcessingMechanism(name='INPUTS', input_shapes=DIM)
    outputs = pnl.ProcessingMechanism(name='OUTPUTS', input_shapes=DIM)

    # ** Student Branch ** #
    student = pnl.ProcessingMechanism(name='STUDENT', input_shapes=DIM)

    # * Pathways * #
    if student_matrix is None:
        student_matrix = pnl.IDENTITY_MATRIX
    if student_output_matrix is None:
        student_output_matrix = pnl.IDENTITY_MATRIX

    # inputs -> learn_net (identity, not learnable)
    pw_inputs_student = [
        inputs,
        pnl.MappingProjection(
            inputs, student,
            name='inputs_student',
            matrix=student_matrix, learnable=True, learning_rate=LEARNING_RATE
        ),
        student,
    ]

    # learn_net -> student (identity, learnable should resemble rot after learning)
    pw_student_outputs = [
        student,
        pnl.MappingProjection(
            student, outputs,
            name='student_outputs',
            matrix=student_output_matrix, learnable=False,
        ),
        outputs
    ]

    # ** Teacher Branch ** #
    teacher = pnl.ProcessingMechanism(name='TEACHER', input_shapes=DIM)

    # * Pathways * #
    pw_inputs_teacher = [
        inputs,
        pnl.MappingProjection(
            inputs, teacher,
            name='inputs_teacher',
            matrix=np.array(teacher_matrix), learnable=False
        ),
        teacher
    ]

    # ** Composition ** #
    comp = pnl.AutodiffComposition(
        pathways=[
            pw_inputs_student,
            pw_student_outputs,
            pw_inputs_teacher,
        ],
        targets=[(student, teacher)],
        loss_spec=pnl.Loss.MSE

    )
    return comp, inputs


def get_pnl_matrices(model):
    pytorch_rep = model.pytorch_representation

    student_proj = model.projections['inputs_student']
    outputs_proj = model.projections['student_outputs']
    teacher_proj = model.projections['inputs_teacher']

    student_m = pytorch_rep.get_torch_param_for_projection(student_proj).detach().clone().numpy()
    outputs_m = pytorch_rep.get_torch_param_for_projection(outputs_proj).detach().clone().numpy()
    teacher_t = pytorch_rep.get_torch_param_for_projection(teacher_proj).detach().clone().numpy()

    return student_m, outputs_m, teacher_t


######################
# *** RUN SCRIPT *** #
######################
def run(seed=None):
    # Setup
    set_random_seed(seed)

    # get random DIM x DIM teacher matrix
    teacher_matrix = torch.randn(DIM, DIM)
    # teacher_matrix = torch.eye(DIM)

    # get random training data
    train = torch.rand(TRAINING_EXAMPLES, 3)
    # train = torch.tensor(np.arange(DIM))

    test_targets = torch.rand(5, 3)
    # test_targets = torch.eye(5,3)
    # out_torch = None
    out_pnl = None
    ######################################
    # ** Initialization of the Models ** #
    ######################################
    if RUN_TORCH:
        model_t = TeacherStudentNet(teacher_matrix=teacher_matrix)

        orig_student_torch = model_t.student.weight.detach().clone().numpy()
        orig_outputs_torch = model_t.outputs.weight.detach().clone().numpy()
        orig_teacher_torch = model_t.teacher.weight.detach().clone().numpy()

    if RUN_PNL:
        model_pnl, inputs = gen_teacher_student_model(teacher_matrix=teacher_matrix)
        model_pnl._build_pytorch_representation()

        orig_student_pnl, orig_outputs_pnl, orig_teacher_pnl = get_pnl_matrices(model_pnl)

    if RUN_TORCH and RUN_PNL:
        # assert matrices in pnl and torch are the same
        assert np.allclose(
            orig_student_torch,
            orig_student_pnl, atol=1e-24), '[TORCH != PNL] Initial student matrices are not the same'

        assert np.allclose(
            orig_outputs_torch,
            orig_outputs_pnl, atol=1e-24), '[TORCH != PNL] Initial output matrices are not the same'

        assert np.allclose(
            orig_teacher_torch,
            orig_teacher_pnl, atol=1e-24), '[TORCH != PNL] Initial teacher matrces are not the same'

    ########################
    # ** Run the Models ** #
    ########################
    if RUN_TORCH:
        set_random_seed(seed)

        optimizer = torch.optim.SGD(model_t.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        for t in train:
            student, teacher, out = model_t(t)
            # detach from teacher since we don't want to train the teacher net
            target_value = teacher.detach()
            loss = loss_fn(student, target_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        after_student_torch = model_t.student.weight.detach().clone().numpy()
        after_outputs_torch = model_t.outputs.weight.detach().clone().numpy()
        after_teacher_torch = model_t.teacher.weight.detach().clone().numpy()

        # Inputs -> Student should change if Learning rate is > 0.
        if LEARNING_RATE > 0.0:
            assert not np.allclose(
                after_student_torch,
                orig_student_torch, atol=0.0), "[TORCH] Student matrix did not change"

        # Student -> outputs must not change
        assert np.allclose(
            after_outputs_torch,
            orig_outputs_torch,
            atol=0.0), "[TORCH] Output matrix changed!"

        # Inputs -> Teacher must not change
        assert np.allclose(
            after_teacher_torch,
            orig_teacher_torch,
            atol=0.0), "[TORCH] Teacher matrix changed!"

        student_, teacher_, out_torch = model_t(test_targets)

    if RUN_PNL:
        set_random_seed(seed)
        if SHOW_PNL:
            model_pnl.show_graph(show_pytorch=True)

        for t in train:
            model_pnl.learn(inputs={inputs: np.array(t)},
                            execution_mode=pnl.ExecutionMode.PyTorch
                            )
            assert True

        after_student_pnl, after_outputs_pnl, after_teacher_pnl = get_pnl_matrices(model_pnl)
        # Test matrices that shouldn't change
        print('*** PNL ***')
        print('* Student *')
        print(f'Before (identity):\n{orig_student_pnl}')
        print(f'After:\n{after_student_pnl}')
        print()

        print('* Outputs *')
        print(f'Before (identity):\n{orig_outputs_pnl}')
        print(f'After (identity):\n{after_outputs_pnl}')
        print()

        print('*** Teacher ***')
        print(f'Before:\n{orig_teacher_pnl}')
        print(f'After (same as before):\n{after_teacher_pnl}')
        print(f'Should be:\n{teacher_matrix.detach().clone().numpy()}')

        # Student input projection MUST change (sanity check)
        if LEARNING_RATE > 0:
            assert not np.allclose(
                after_student_pnl,
                orig_student_pnl,
            ), "[PNL] Student matrix did not change!"

        # Student -> outputs projection must NOT change
        assert np.allclose(
            after_outputs_pnl,
            orig_outputs_pnl,
        ), "[PNL] Output matrix changed!"

        # Teacher projection must NOT change
        assert np.allclose(
            orig_teacher_pnl,
            after_teacher_pnl
        ), "[PNL] Teacher matrix changed!"

        model_pnl.run(inputs={inputs: test_targets})
        out_pnl = [r[0] for r in model_pnl.results[-5:]]

    if out_torch is not None and out_pnl is not None:

        assert np.allclose(
            after_teacher_pnl,
            after_teacher_torch,
        ), f'[TORCH != PNL] Teacher not the same anymore'
        assert np.allclose(
            after_outputs_torch,
            after_outputs_pnl,
        ), f'[TORCH != PNL] Outputs not the same anymore'
        assert np.allclose(after_student_torch,
                           after_student_pnl,
                           ), f'[TORCH != PNL] Student Matrices are not the same\n{after_student_torch} !=\n{after_student_pnl}'
        for res in zip(out_torch, out_pnl):
            tr = np.array(res[0].detach(), dtype=float)
            pl = np.array(res[1], dtype=float)

            assert np.allclose(tr, pl), \
                (f"Not close enough {tr}, {pl}\n")

    print('ALL CLOSE')


if __name__ == "__main__":
    import random

    seed = 42  # random.randint(0, int(1e12))
    run(seed)
