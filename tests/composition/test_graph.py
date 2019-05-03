import numpy as np
import pytest

from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition, Graph, Vertex


class TestGraph:

    class DummyComponent:

        def __init__(self):
            pass

    def test_copy(self):

        g1 = Graph()
        vertices = [Vertex(TestGraph.DummyComponent()) for i in range(5)]

        for i in range(len(vertices)):
            g1.add_vertex(vertices[i])
            # each vertex has previous vertex as parent and next vertex as child
            g1.connect_vertices(vertices[(i - 1) % len(vertices)], vertices[i])

        g2 = g1.copy()

        assert len(g1.vertices) == len(g2.vertices)
        assert len(g1.comp_to_vertex) == len(g2.comp_to_vertex)

        for i in range(len(g2.vertices)):
            assert g2.vertices[i].parents == [g2.vertices[(i - 1) % len(g2.vertices)]]
            assert g2.vertices[i].children == [g2.vertices[(i + 1) % len(g2.vertices)]]

            assert g1.vertices[i] != g2.vertices[i]
            assert g1.vertices[i].component == g2.vertices[i].component
