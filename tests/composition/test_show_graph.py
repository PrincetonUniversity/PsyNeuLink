from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.functions.learningfunctions import BackPropagation
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.globals.keywords import ALL
from psyneulink.library.components.mechanisms.modulatory.control.agt.lccontrolmechanism import LCControlMechanism


class TestSimpleCompositions:
    def test_process(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")

        comp = Composition()
        comp.add_linear_processing_pathway([a, b])

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label

    def test_diverging_pathways(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c", default_variable=[0, 0, 0, 0, 0])
        comp = Composition()
        comp.add_linear_processing_pathway([a, b])
        comp.add_linear_processing_pathway([a, c])

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)
        c_label = comp._get_graph_node_label(c, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label
        assert "out (5)" in c_label and "in (5)" in c_label

    def test_converging_pathways(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c", default_variable=[0, 0, 0, 0, 0])
        comp = Composition()
        comp.add_linear_processing_pathway([a, c])
        comp.add_linear_processing_pathway([b, c])

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)
        c_label = comp._get_graph_node_label(c, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label
        assert "out (5)" in c_label and "in (5)" in c_label


class TestLearning:
    def test_process(self):
        a = TransferMechanism(name="a-sg", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b-sg")

        comp = Composition()
        comp.add_linear_learning_pathway(
            [a, b], learning_function=BackPropagation
        )

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label

    def test_diverging_pathways(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c", default_variable=[0, 0, 0, 0, 0])
        comp = Composition()
        comp.add_linear_learning_pathway(
            [a, b], learning_function=BackPropagation
        )
        comp.add_linear_learning_pathway(
            [a, c], learning_function=BackPropagation
        )

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)
        c_label = comp._get_graph_node_label(c, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label
        assert "out (5)" in c_label and "in (5)" in c_label

    def test_converging_pathways(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c", default_variable=[0, 0, 0, 0, 0])
        comp = Composition()
        comp.add_linear_learning_pathway(
            [a, c], learning_function=BackPropagation
        )
        comp.add_linear_learning_pathway(
            [b, c], learning_function=BackPropagation
        )

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)
        c_label = comp._get_graph_node_label(c, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label
        assert "out (5)" in c_label and "in (5)" in c_label


class TestControl:
    def test_process(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        LC = LCControlMechanism(
            modulated_mechanisms=[a, b],
            objective_mechanism=ObjectiveMechanism(
                function=Linear, monitor=[b], name="lc_om"
            ),
            name="lc",
        )
        comp = Composition()
        comp.add_linear_processing_pathway([a, b])

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label

    def test_diverging_pathways(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c", default_variable=[0, 0, 0, 0, 0])
        LC = LCControlMechanism(
            modulated_mechanisms=[a, b],
            objective_mechanism=ObjectiveMechanism(
                function=Linear, monitor=[b], name="lc_om"
            ),
            name="lc",
        )
        comp = Composition()
        comp.add_linear_processing_pathway([a, b])
        comp.add_linear_processing_pathway([a, c])

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)
        c_label = comp._get_graph_node_label(c, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label
        assert "out (5)" in c_label and "in (5)" in c_label

    def test_converging_pathways(self):
        a = TransferMechanism(name="a", default_variable=[0, 0, 0])
        b = TransferMechanism(name="b")
        c = TransferMechanism(name="c", default_variable=[0, 0, 0, 0, 0])
        LC = LCControlMechanism(
            modulated_mechanisms=[a, b],
            objective_mechanism=ObjectiveMechanism(
                function=Linear, monitor=[b], name="lc_om"
            ),
            name="lc",
        )
        comp = Composition()
        comp.add_linear_processing_pathway([a, c])
        comp.add_linear_processing_pathway([b, c])

        a_label = comp._get_graph_node_label(a, show_dimensions=ALL)
        b_label = comp._get_graph_node_label(b, show_dimensions=ALL)
        c_label = comp._get_graph_node_label(c, show_dimensions=ALL)

        assert "out (3)" in a_label and "in (3)" in a_label
        assert "out (1)" in b_label and "in (1)" in b_label
        assert "out (5)" in c_label and "in (5)" in c_label
