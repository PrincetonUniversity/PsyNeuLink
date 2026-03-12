import psyneulink as pnl
import numpy as np


def create_model():
    # Nodes
    input_ = pnl.TransferMechanism(
        name="input",
        input_shapes=32 * 32,
    )

    hidden_1 = pnl.TransferMechanism(
        name="hidden 1",
        input_shapes=512,
        function=pnl.ReLU()
    )

    hidden_2 = pnl.TransferMechanism(
        name="hidden 2",
        input_shapes=256,
        function=pnl.ReLU()
    )

    output_ = pnl.TransferMechanism(
        name="output",
        input_shapes=128,
        function=pnl.ReLU()
    )

    # Biases

    hidden_1_bias = pnl.TransferMechanism(
        name="hidden_1_bias",
        input_shapes=1,
        initial_value=1
    )

    hidden_2_bias = pnl.TransferMechanism(
        name="hidden_2_bias",
        input_shapes=1,
        initial_value=1
    )

    output_bias = pnl.TransferMechanism(
        name="output_bias",
        input_shapes=1,
        initial_value=1
    )

    # Mapping (nodes to nodes)

    # input -> hidden 1 mapping initialized using Kaiming normal distribution
    input_hidden_1_mp = pnl.MappingProjection(
        name="input_hidden_1_mp",
        sender=input_,
        receiver=hidden_1,
        matrix=np.random.randn(32*32, 512).astype(np.float32) * np.sqrt(2.0 / (32 * 32))
    )

    # hidden 1 -> hidden 2 mapping initialized using Kaiming normal distribution
    hidden_1_hidden_2_mp = pnl.MappingProjection(
        name="hidden_1_hidden_2_mp",
        sender=hidden_1,
        receiver=hidden_2,
        matrix=np.random.randn(512, 256).astype(np.float32) * np.sqrt(2.0 / 512)
    )

    # hidden 2 -> output mapping initialized using Kaiming normal distribution
    hidden_2_output_mp = pnl.MappingProjection(
        name="hidden_2_output_mp",
        sender=hidden_2,
        receiver=output_,
        matrix=np.random.randn(256, 128).astype(np.float32) * np.sqrt(2.0 / 256)
    )

    # Mapping (bias to nodes initialized to 0)
    hidden_1_bias_mp = pnl.MappingProjection(
        name="hidden_1_bias_mp",
        sender=hidden_1_bias,
        receiver=hidden_1,
        matrix=np.zeros((1, 512))
    )

    # Hidden bias mapping initialized to 0
    hidden_2_bias_mp = pnl.MappingProjection(
        name="hidden_2_bias_mp",
        sender=hidden_2_bias,
        receiver=hidden_2,
        matrix=np.zeros((1, 256))
    )

    # Output bias mapping initialized to 0
    output_bias_mp = pnl.MappingProjection(
        name="output_bias_mp",
        sender=output_bias,
        receiver=output_,
        matrix=np.zeros((1, 128))
    )

    pathways = [
        [input_, input_hidden_1_mp, hidden_1],
        #[hidden_1_bias, hidden_1_bias_mp, hidden_1],

        [hidden_1, hidden_1_hidden_2_mp, hidden_2],
        #[hidden_2_bias, hidden_2_bias_mp, hidden_2],

        [hidden_2, hidden_2_output_mp, output_],
        #[output_bias, output_bias_mp, output_],
    ]

    comp = pnl.AutodiffComposition(
        name="MLP Encoder",
        pathways=pathways,
    )

    return comp, input_


def run(model, input_, x_numpy):
    model.run(
        inputs={input_: x_numpy}
    )

    return model.results
