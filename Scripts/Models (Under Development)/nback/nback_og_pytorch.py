import numpy as np
import psyneulink as pnl

# Input Ports exposed to user, with default values
hid1_layer_weight = np.zeros((90, 90), dtype="float32")
input1 = np.zeros((90,), dtype="float32")
embed_bias_weight = np.zeros((10, 90), dtype="float32")
hid2_layer_weight = np.zeros((90, 80), dtype="float32")
out_layer_weight = np.zeros((80, 2), dtype="float32")

FFWMGraph = pnl.Composition(name="FFWMGraph")

Add_10 = pnl.ProcessingMechanism(
    name="Add_10",
    function=pnl.Identity(default_variable=np.zeros((2, 90), dtype="float32")),
    default_variable=np.zeros((2, 90), dtype="float32"),
)
Dropout_13_14 = pnl.ProcessingMechanism(
    name="Dropout_13_14",
    function=pnl.Dropout(
        p=0.05000000074505806, default_variable=np.zeros((90,), dtype="float32")
    ),
    default_variable=np.zeros((90,), dtype="float32"),
)
Gather_9 = pnl.ProcessingMechanism(
    name="Gather_9",
    function=pnl.Identity(default_variable=np.zeros((10, 90), dtype="float32")),
    default_variable=np.zeros((10, 90), dtype="float32"),
)
MatMul_19_passthrough_terminal = pnl.ProcessingMechanism(
    name="MatMul_19_passthrough_terminal",
    function=pnl.Identity(default_variable=[0.0, 0.0]),
    default_variable=[0.0, 0.0],
)
MatMul_6_passthrough_origin = pnl.ProcessingMechanism(
    name="MatMul_6_passthrough_origin",
    function=pnl.Identity(default_variable=np.zeros((90,), dtype="float32")),
    default_variable=np.zeros((90,), dtype="float32"),
)
Relu_17 = pnl.ProcessingMechanism(
    name="Relu_17",
    function=pnl.ReLU(default_variable=np.zeros((80,), dtype="float32")),
    default_variable=np.zeros((80,), dtype="float32"),
)
Relu_7 = pnl.ProcessingMechanism(
    name="Relu_7",
    function=pnl.ReLU(default_variable=np.zeros((90,), dtype="float32")),
    default_variable=np.zeros((90,), dtype="float32"),
)

FFWMGraph.add_node(Add_10)
FFWMGraph.add_node(Dropout_13_14)
FFWMGraph.add_node(Gather_9)
FFWMGraph.add_node(MatMul_19_passthrough_terminal)
FFWMGraph.add_node(MatMul_6_passthrough_origin)
FFWMGraph.add_node(Relu_17)
FFWMGraph.add_node(Relu_7)

FFWMGraph.add_projection(
    projection=pnl.MappingProjection(name="Relu_7_Add_10"),
    sender=Relu_7,
    receiver=Add_10,
)
FFWMGraph.add_projection(
    projection=pnl.MappingProjection(name="Gather_9_Add_10"),
    sender=Gather_9,
    receiver=Add_10,
)
FFWMGraph.add_projection(
    projection=pnl.MappingProjection(name="Add_10_Dropout_13_14"),
    sender=Add_10,
    receiver=Dropout_13_14,
)
FFWMGraph.add_projection(
    projection=pnl.MappingProjection(
        name="MatMul_6_as_edge",
        function=pnl.MatrixTransform(
            default_variable=np.zeros((90,), dtype="float32"), matrix=hid1_layer_weight
        ),
    ),
    sender=MatMul_6_passthrough_origin,
    receiver=Relu_7,
)
FFWMGraph.add_projection(
    projection=pnl.MappingProjection(
        name="MatMul_16_as_edge",
        function=pnl.MatrixTransform(
            default_variable=np.zeros((90,), dtype="float32"), matrix=hid2_layer_weight
        ),
    ),
    sender=Dropout_13_14,
    receiver=Relu_17,
)
FFWMGraph.add_projection(
    projection=pnl.MappingProjection(
        name="MatMul_19_as_edge",
        function=pnl.MatrixTransform(
            default_variable=np.zeros((80,), dtype="float32"), matrix=out_layer_weight
        ),
    ),
    sender=Relu_17,
    receiver=MatMul_19_passthrough_terminal,
)