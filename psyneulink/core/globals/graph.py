import collections
import enum
import logging
import typing

import networkx

from psyneulink._typing import Union
from psyneulink.core.globals.keywords import MAYBE

__all__ = [
    'EdgeType', 'GraphError'
]


logger = logging.getLogger(__name__)


class GraphError(Exception):
    pass


class EdgeType(enum.Enum):
    """
        Attributes:
            NON_FEEDBACK
                A standard edge that if it exists in a cycle will only be flattened, not pruned

            FEEDBACK
                A "feedbacK" edge that will be immediately pruned to create an acyclic graph

            FLEXIBLE
                An edge that will be pruned only if it exists in a cycle
    """
    NON_FEEDBACK = False
    FEEDBACK = True
    FLEXIBLE = MAYBE

    @classmethod
    def from_any(cls, value) -> 'EdgeType':
        """
        Returns:
            EdgeType: an `EdgeType` corresponding to **value** if it
            exists
        """
        try:
            value = value.upper()
        except AttributeError:
            # not a string
            pass

        try:
            return cls[value]
        except KeyError:
            # allow ValueError to raise
            return cls(value)

    @classmethod
    def has(cls, value) -> bool:
        """
        Returns:
            bool: True if **value** corresponds to an `EdgeType`, or
            False otherwise
        """
        try:
            cls.from_any(value)
        except ValueError:
            return False
        else:
            return True


class Vertex(object):
    """
        Stores a Component for use with a :py:class:`Graph`

        Arguments
        ---------

        component : Component
            the `Component <Component>` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`

        Attributes
        ----------

        component : Component
            the `Component <Component>` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`
    """

    def __init__(self, component, parents=None, children=None, feedback=None):
        self.component = component
        if parents is not None:
            self.parents = parents
        else:
            self.parents = []
        if children is not None:
            self.children = children
        else:
            self.children = []

        self.feedback = feedback

        # when pruning a vertex for a processing graph, we store the connection type (the vertex.feedback)
        # to the new child or parent here
        # self.source_types = collections.defaultdict(EdgeType.NORMAL)
        self.source_types = {}

    def __repr__(self):
        return '(Vertex {0} {1})'.format(id(self), self.component)

    @property
    def feedback(self):
        return self._feedback

    @feedback.setter
    def feedback(self, value: Union[bool, EdgeType]):
        if value is None:
            self._feedback = None
        else:
            self._feedback = EdgeType.from_any(value)


class Graph(object):
    """A Graph of vertices and edges.

    Attributes
    ----------

    comp_to_vertex : Dict[`Component <Component>` : `Vertex`]
        maps `Component` in the graph to the `Vertices <Vertex>` that represent them.

    vertices : List[Vertex]
        the `Vertices <Vertex>` contained in this Graph;  each can be a `Node <Composition_Nodes>` or a
        `Projection <Component_Projections>`.

    dependency_dict : Dict[`Component` : Set(`Component`)]
        maps each of the graph's Components to the others from which it receives input
        (i.e., their `value <Component.value>`).  For a `Node <Components_Nodes>`, this is one or more
        `Projections <Projection>`;  for a Projection, it is a single Node.

    """

    def __init__(self):
        self.comp_to_vertex = collections.OrderedDict()  # Translate from PNL Mech, Comp or Proj to corresponding vertex
        self.vertices = []  # List of vertices within graph

        self.cycle_vertices = []

    def copy(self):
        """
            Returns
            -------

            A copy of the Graph. `Vertices <Vertex>` are distinct from their originals, and point to the same
            `Component <Component>` object : :py:class:`Graph`
        """
        g = Graph()

        for vertex in self.vertices:
            g.add_vertex(Vertex(vertex.component, feedback=vertex.feedback))

        for i in range(len(self.vertices)):
            g.vertices[i].parents = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in
                                     self.vertices[i].parents]
            g.vertices[i].children = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in
                                      self.vertices[i].children]

        return g

    def add_component(self, component, feedback=False):
        if component in [vertex.component for vertex in self.vertices]:
            logger.info('Component {1} is already in graph {0}'.format(component, self))
        else:
            vertex = Vertex(component, feedback=feedback)
            self.comp_to_vertex[component] = vertex
            self.add_vertex(vertex)

    def add_vertex(self, vertex):
        if vertex in self.vertices:
            logger.info('Vertex {1} is already in graph {0}'.format(vertex, self))
        else:
            self.vertices.append(vertex)
            self.comp_to_vertex[vertex.component] = vertex

    def remove_component(self, component):
        try:
            self.remove_vertex(self.comp_to_vertex[component])
        except KeyError as e:
            raise GraphError('Component {1} not found in graph {2}: {0}'.format(e, component, self))

    def remove_vertex(self, vertex):
        try:
            for parent in vertex.parents:
                parent.children.remove(vertex)
            for child in vertex.children:
                child.parents.remove(vertex)

            self.vertices.remove(vertex)
            del self.comp_to_vertex[vertex.component]
            # TODO:
            #   check if this removal puts the graph in an inconsistent state
        except ValueError as e:
            raise GraphError('Vertex {1} not found in graph {2}: {0}'.format(e, vertex, self))

    def connect_components(self, parent, child):
        try:
            self.connect_vertices(self.comp_to_vertex[parent], self.comp_to_vertex[child])
        except KeyError as e:
            if parent not in self.comp_to_vertex:
                raise GraphError(
                    "Sender ({}) of Projection ({}) not (yet) assigned".format(
                        repr(parent.name), repr(child.name)
                    )
                )
            elif child not in self.comp_to_vertex:
                raise GraphError(
                    "Projection ({}) to {} not (yet) assigned".format(
                        repr(parent.name), repr(child.name)
                    )
                )
            else:
                raise KeyError(e)

    def connect_vertices(self, parent, child):
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def get_parents_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose parents will be returned

            Returns
            -------

            list[`Vertex`] :
              list of the parent `Vertices <Vertex>` of the Vertex associated with **component**.
        """
        return self.comp_to_vertex[component].parents

    def get_children_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            list[`Vertex`] :
                list of the child `Vertices <Vertex>` of the Vertex associated with **component**.
        """
        return self.comp_to_vertex[component].children

    def prune_feedback_edges(self):
        """
            Produces an acyclic graph from this Graph. `Feedback <EdgeType.FEEDBACK>` edges are pruned, as well as
            any edges that are `potentially feedback <EdgeType.FLEXIBLE>` that are in cycles. After these edges are
            removed, if cycles still remain, they are "flattened." That is, each edge in the cycle is pruned, and
            each the dependencies of each Node in the cycle are set to the pre-flattened union of all cyclic nodes'
            parents that are themselves not in a cycle.

            Returns:
                a tuple containing
                - the acyclic dependency dictionary produced from this
                Graph
                - a dependency dictionary containing only the edges
                removed to create the acyclic graph
                - the unmodified cyclic dependency dictionary of this
                Graph
        """

        # stores a modified version of the self in which cycles are "flattened"
        execution_dependencies = self.dependency_dict
        # stores the original unmodified dependencies
        structural_dependencies = self.dependency_dict
        # wipe and reconstruct list of vertices in cycles
        self.cycle_vertices = []
        flexible_edges = set()

        for node in execution_dependencies:
            # prune recurrent edges
            try:
                execution_dependencies[node].remove(node)
                self.cycle_vertices.append([node])
            except KeyError:
                pass

            for dep in tuple(execution_dependencies[node]):
                vert = self.comp_to_vertex[node]
                dep_vert = self.comp_to_vertex[dep]

                if dep_vert in vert.source_types:
                    # prune standard edges labeled as feedback
                    if vert.source_types[dep_vert] is EdgeType.FEEDBACK:
                        execution_dependencies[node].remove(dep)
                    # store flexible edges for potential pruning later
                    elif vert.source_types[dep_vert] is EdgeType.FLEXIBLE:
                        flexible_edges.add((dep, node))

        # construct a parallel networkx graph to use its cycle algorithms
        nx_graph = self._generate_networkx_graph(execution_dependencies)
        connected_components = list(networkx.strongly_connected_components(nx_graph))

        # prune only one flexible edge per attempt, to remove as few
        # edges as possible
        # For now, just prune the first flexible edge each time. Maybe
        # look for "best" edges to prune in future by frequency in
        # cycles, if that occurs
        try:
            flexible_edges_iter = sorted(flexible_edges)
        except TypeError:
            flexible_edges_iter = flexible_edges
        for parent, child in flexible_edges_iter:
            cycles = [c for c in connected_components if len(c) > 1]

            if len(cycles) == 0:
                break

            if any((parent in c and child in c) for c in cycles):
                # prune
                execution_dependencies[child].remove(parent)
                self.comp_to_vertex[child].source_types[self.comp_to_vertex[parent]] = EdgeType.FEEDBACK
                nx_graph.remove_edge(parent, child)
                # recompute cycles after each prune
                connected_components = list(networkx.strongly_connected_components(nx_graph))

        # find all the parent nodes for each node in a cycle, excluding
        # parents that are part of the cycle
        for cycle in [c for c in connected_components if len(c) > 1]:
            acyclic_dependencies = set()

            for node in cycle:
                acyclic_dependencies = acyclic_dependencies.union({
                    parent for parent in execution_dependencies[node]
                    if parent not in cycle
                })

            # replace the dependencies of each node in the cycle with
            # each of the above parents outside of the cycle. This
            # ensures that they all share the same parents and will then
            # exist in the same consideration set

            # NOTE: it is unnecessary to change any childrens'
            # dependencies because any child dependent on a node n_i in
            # a cycle will still depend on n_i when it is part of a
            # flattened cycle. The flattened cycle will simply add more
            # nodes to the consideration set in which n_i exists
            cycle_verts = []
            for child in cycle:
                cycle_verts.append(child)
                execution_dependencies[child] = acyclic_dependencies
            self.cycle_vertices.append(cycle_verts)

        return (
            execution_dependencies,
            {
                node: structural_dependencies[node] - execution_dependencies[node]
                for node in execution_dependencies
            },
            structural_dependencies
        )

    def get_strongly_connected_components(
        self,
        nx_graph: typing.Optional[networkx.DiGraph] = None
    ):
        if nx_graph is None:
            nx_graph = self._generate_networkx_graph()

        return list(networkx.strongly_connected_components(nx_graph))

    def _generate_networkx_graph(self, dependency_dict=None):
        if dependency_dict is None:
            dependency_dict = self.dependency_dict

        nx_graph = networkx.DiGraph()
        nx_graph.add_nodes_from(list(dependency_dict.keys()))
        for child in dependency_dict:
            for parent in dependency_dict[child]:
                nx_graph.add_edge(parent, child)

        return nx_graph

    @property
    def dependency_dict(self):
        return dict((v.component, set(d.component for d in v.parents)) for v in self.vertices)
