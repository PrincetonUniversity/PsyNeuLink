import argparse
import os

import numpy as np
import psyneulink as pnl

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "_images"),
    help="Path to store generated animations",
)
args = parser.parse_args()


# Based on tests/composition/test_learning.py::TestLearningPathwayMethods::test_run_no_targets
def composition_xor_animation():
    in_to_hidden_matrix = np.random.rand(2, 10)
    hidden_to_out_matrix = np.random.rand(10, 1)

    inp = pnl.TransferMechanism(name="Input", default_variable=np.zeros(2))

    hidden = pnl.TransferMechanism(
        name="Hidden", default_variable=np.zeros(10), function=pnl.Logistic()
    )

    output = pnl.TransferMechanism(
        name="Output", default_variable=np.zeros(1), function=pnl.Logistic()
    )

    in_to_hidden = pnl.MappingProjection(
        name="Input Weights",
        matrix=in_to_hidden_matrix.copy(),
        sender=inp,
        receiver=hidden,
    )

    hidden_to_out = pnl.MappingProjection(
        name="Output Weights",
        matrix=hidden_to_out_matrix.copy(),
        sender=hidden,
        receiver=output,
    )

    xor_comp = pnl.Composition()

    xor_comp.add_backpropagation_learning_pathway(
        [inp, in_to_hidden, hidden, hidden_to_out, output],
        learning_rate=10,
    )
    xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_comp.learn(
        inputs={inp: xor_inputs},
        animate={
            pnl.SHOW_LEARNING: True,
            pnl.MOVIE_DIR: args.directory,
            pnl.MOVIE_NAME: "Composition_XOR_animation",
        },
    )


composition_xor_animation()
