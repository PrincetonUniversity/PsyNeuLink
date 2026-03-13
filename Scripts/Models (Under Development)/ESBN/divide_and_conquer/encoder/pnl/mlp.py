import numpy as np
import psyneulink as pnl


def create_model(
        matrix_in_hidden_1,
        matrix_hidden_1_hidden_2,
        matrix_hidden_2_out,
        learning_rate=0.01,
):
    input_ = pnl.TransferMechanism(
        name="input",
        input_shapes=32 * 32,
    )

    hidden_1 = pnl.TransferMechanism(
        name="hidden_1",
        input_shapes=512,
        function=pnl.ReLU(),
    )

    hidden_2 = pnl.TransferMechanism(
        name="hidden_2",
        input_shapes=256,
        function=pnl.ReLU(),
    )

    output_ = pnl.TransferMechanism(
        name="output",
        input_shapes=128,
        function=pnl.ReLU(),
    )

    input_hidden_1_mp = pnl.MappingProjection(
        name="input_hidden_1_mp",
        sender=input_,
        receiver=hidden_1,
        matrix=matrix_in_hidden_1,
    )

    hidden_1_hidden_2_mp = pnl.MappingProjection(
        name="hidden_1_hidden_2_mp",
        sender=hidden_1,
        receiver=hidden_2,
        matrix=matrix_hidden_1_hidden_2,
    )

    hidden_2_output_mp = pnl.MappingProjection(
        name="hidden_2_output_mp",
        sender=hidden_2,
        receiver=output_,
        matrix=matrix_hidden_2_out,
    )

    # Bias (nodes)

    hidden_1_bias = pnl.TransferMechanism(
        name="hidden_1_bias",
        default_variable=[1.]
    )
    hidden_2_bias = pnl.TransferMechanism(
        name="hidden_2_bias",
        default_variable=[1.]
    )
    output_bias = pnl.TransferMechanism(
        name="output_bias",
        default_variable=[1.]
    )

    # Bias (projections)
    hidden_1_bias_mp = pnl.MappingProjection(
        name="hidden_1_bias_mp",
        sender=hidden_1_bias,
        receiver=hidden_1,
        matrix=np.zeros((1, 512)),
    )
    hidden_2_bias_mp = pnl.MappingProjection(
        name="hidden_2_bias_mp",
        sender=hidden_2_bias,
        receiver=hidden_2,
        matrix=np.zeros((1, 256)),
    )
    output_bias_mp = pnl.MappingProjection(
        name="output_bias_mp",
        sender=output_bias,
        receiver=output_,
        matrix=np.zeros((1, 128)),
    )

    pathways = [
        [input_, input_hidden_1_mp, hidden_1],
        [hidden_1, hidden_1_hidden_2_mp, hidden_2],
        [hidden_2, hidden_2_output_mp, output_],
        # bias
        [hidden_1_bias, hidden_1_bias_mp, hidden_1],
        [hidden_2_bias, hidden_2_bias_mp, hidden_2],
        [output_bias, output_bias_mp, output_],
    ]

    comp = pnl.AutodiffComposition(
        name="MLP Encoder",
        pathways=pathways,
        learning_rate=learning_rate,
        device=pnl.CPU,
        loss_spec=pnl.Loss.MSE,
    )

    return (
        comp,
        input_,
        output_,
        input_hidden_1_mp,
        hidden_1_hidden_2_mp,
        hidden_2_output_mp,
    )


def run(model, input_, x_numpy):
    model.run(
        inputs={input_: x_numpy}
    )
    return model.results


def learn(model, input_, output_, x_numpy, target_numpy, learning_rate=0.01):
    model.learn(
        inputs={input_: x_numpy},
        targets={output_: target_numpy},
        learning_rate=learning_rate,
        execution_mode=pnl.ExecutionMode.PyTorch,
        synch_projection_matrices_with_torch=pnl.RUN,
        synch_node_values_with_torch=pnl.RUN,
        synch_results_with_torch=pnl.RUN,
    )
    return model.results


def get_matrix(projection, model):
    ctx = model.most_recent_context
    return np.array(projection.parameters.matrix.get(ctx)).copy()
