import psyneulink as pnl
import pytest
import numpy as np

class TestCompositionMethods:

    def test_get_output_values_prop(self):
        A = pnl.ProcessingMechanism()
        c = pnl.Composition()
        c.add_node(A)
        result = c.run(inputs={A: [1]}, num_trials=2)
        assert result == c.output_values == [np.array([1])]

    def test_add_pathway_methods_return_pathway(self):
        c = pnl.Composition()
        p = c.add_linear_processing_pathway(pathway=[pnl.ProcessingMechanism(), pnl.ProcessingMechanism()])
        assert isinstance(p, pnl.Pathway)

        c = pnl.Composition()
        p = c.add_linear_learning_pathway(pathway=[pnl.ProcessingMechanism(), pnl.ProcessingMechanism()],
                                          learning_function=pnl.BackPropagation)
        assert isinstance(p, pnl.Pathway)
