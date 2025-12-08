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
TRAINING_EXAMPLES = 10

# Script Control
RUN_TORCH = True
RUN_PNL = True
SHOW_PNL = True


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
        learning_rate=LEARNING_RATE,
    )
    return comp, inputs


######################
# *** RUN SCRIPT *** #
######################
def run(seed=None):
    # Setup

    # get random DIM x DIM teacher matrix
    teacher_matrix = torch.randn(DIM, DIM)

    # get random training data
    train = torch.rand(TRAINING_EXAMPLES, 3)

    test_targets = torch.rand(5, 3)
    out_torch = None
    out_pnl = None
    #####################
    # ** Torch Model ** #
    #####################
    if RUN_TORCH:
        torch.manual_seed(seed)
        model_t = TeacherStudentNet(teacher_matrix=teacher_matrix)

        orig_teacher = model_t.teacher.weight.detach().clone()
        orig_outputs = model_t.outputs.weight.detach().clone()
        orig_student = model_t.student.weight.detach().clone()

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

        # Test matrices that shouldn't change
        assert torch.allclose(
            model_t.teacher.weight,
            orig_teacher,
            atol=0.0
        ), "Teacher matrix changed!"

        # (2) Student → outputs must not change
        assert torch.allclose(
            model_t.outputs.weight,
            orig_outputs,
            atol=0.0
        ), "Output matrix changed!"

        # Test learning happened
        if LEARNING_RATE > 0:
            assert not torch.allclose(
                model_t.student.weight,
                orig_student
            ), "Student matrix did not change — learning failed"

        student_, teacher_, out_torch = model_t(test_targets)

    if RUN_PNL:
        torch.manual_seed(seed)
        model_pnl, inputs = gen_teacher_student_model(teacher_matrix=teacher_matrix)

        orig_student_matrix = model_pnl.projections['inputs_student'].matrix.base.copy()
        orig_outputs_matrix = model_pnl.projections['student_outputs'].matrix.base.copy()
        orig_teacher_matrix = model_pnl.projections['inputs_teacher'].matrix.base.copy()

        if SHOW_PNL:
            model_pnl.show_graph(show_pytorch=True)

        pytorch_rep = model_pnl._build_pytorch_representation()
        # pytorch_rep = model_pnl.infer_backpropagation_pathways()

        n = 0
        for t in train:
            n+=1
            result = model_pnl.learn(inputs={inputs: np.array([t])},
                            execution_mode = pnl.ExecutionMode.PyTorch
                            )
            pytorch_rep = model_pnl.pytorch_representation
            student_proj = model_pnl.projections['inputs_student']
            teacher_proj = model_pnl.projections['inputs_teacher']
            student_torch_param = pytorch_rep.get_torch_param_for_projection(student_proj)
            teacher_torch_param = pytorch_rep.get_torch_param_for_projection(teacher_proj)
            print(f'\n\nTrial {n}:------------------\n')
            print(f'\t{t}: {result}')
            params = pytorch_rep.optimizer.param_groups[0]['params'][0]
            print(f'torch params[0] (only one with lr!=False: {params}')
            print(f'\nSTUDENT param: {student_torch_param}')
            print(f'\nTEACHER param: {teacher_torch_param}')
        # Test matrices that shouldn't change
        after_student_matrix = model_pnl.projections['inputs_student'].matrix.base.copy()

        after_outputs_matrix = model_pnl.projections['student_outputs'].matrix.base.copy()
        after_teacher_matrix = model_pnl.projections['inputs_teacher'].matrix.base.copy()

        # (1) Teacher projection must NOT change
        assert np.allclose(
            after_teacher_matrix,
            orig_teacher_matrix
        ), "Teacher projection matrix changed!"

        # (2) Student → outputs projection must NOT change
        assert np.allclose(
            after_outputs_matrix,
            orig_outputs_matrix
        ), "student -> outputs projection matrix changed!"

        # (3) Student input projection MUST change (sanity check)
        if LEARNING_RATE > 0:
            assert not np.allclose(
                after_student_matrix,
                orig_student_matrix
            ), "Student projection matrix did not change!"

        model_pnl.run(inputs={inputs: test_targets})
        out_pnl = [r[0] for r in model_pnl.results[-5:]]

    if out_torch is not None and out_pnl is not None:
        for res in zip(out_torch, out_pnl):
            tr = np.array(res[0].detach(), dtype=float)
            pl = np.array(res[1], dtype=float)

            assert np.allclose(tr, pl), \
                (f"Not close enough {tr}, {pl}\n"
                 f"learned matrices:\n"
                 f"{model_t.student.weight}, {after_student_matrix}")

    print('ALL CLOSE')


if __name__ == "__main__":
    import random

    seed = 42  # random.randint(0, int(1e12))
    run(seed)
