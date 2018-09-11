# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Composition ***************************************************************

"""
..
    Sections:
      * `Composition_Overview`

.. _Composition_Overview:

Overview
--------

Composition is the base class for objects that combine PsyNeuLink `Components <Component>` into an executable model.
It defines a common set of attributes possessed, and methods used by all Composition objects.

.. _Composition_Creation:

Creating a Composition
----------------------

A generic Composition can be created by calling the constructor, and then adding `Components <Component>` using the
Composition's add methods.  However, more commonly, a Composition is created using the constructor for one of its
subclasses:  `System` or `Process`.  These automatically create Compositions from lists of Components.  Once created,
Components can be added or removed from an existing Composition using its add and/or remove methods.

.. _Composition_Execution:

Execution
---------

See `System <System_Execution>` or `Process <Process_Execution>` for documentation concerning execution of the
corresponding subclass.

.. _Composition_Class_Reference:

Class Reference
---------------

"""

import collections
from collections import Iterable, OrderedDict
from enum import Enum
import logging
import numpy as np
import uuid

from psyneulink.components.component import function_type
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.components.shellclasses import Mechanism, Projection
from psyneulink.components.states.outputstate import OutputState
from psyneulink.components.functions.function import InterfaceStateMap
from psyneulink.components.states.inputstate import InputState
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import MATRIX_KEYWORD_VALUES, OWNER_VALUE, HARD_CLAMP, IDENTITY_MATRIX, NO_CLAMP, PULSE_CLAMP, SOFT_CLAMP
from psyneulink.scheduling.condition import Always
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.time import TimeScale

__all__ = [
    'Composition', 'CompositionError', 'CNodeRole',
]

logger = logging.getLogger(__name__)

class CNodeRole(Enum):
    """

    - ORIGIN
        A `ProcessingMechanism <ProcessingMechanism>` that is the first Mechanism of a `Process` and/or `System`, and
        that receives the input to the Process or System when it is :ref:`executed or run <Run>`.  A Process may have
        only one `ORIGIN` Mechanism, but a System may have many.  Note that the `ORIGIN` Mechanism of a Process is not
        necessarily an `ORIGIN` of the System to which it belongs, as it may receive `Projections <Projection>` from
        other Processes in the System (see `example <LearningProjection_Target_vs_Terminal_Figure>`). The `ORIGIN`
        Mechanisms of a Process or System are listed in its :keyword:`origin_nodes` attribute, and can be displayed
        using its :keyword:`show` method.  For additional details about `ORIGIN` Mechanisms in Processes, see `Process
        Mechanisms <Process_Mechanisms>` and `Process Input and Output <Process_Input_And_Output>`; and for Systems see
        `System Mechanisms <System_Mechanisms>` and `System Input and Initialization
        <System_Execution_Input_And_Initialization>`.

    - INTERNAL
        A `ProcessingMechanism <ProcessingMechanism>` that is not designated as having any other status.

    - CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is *not* an `ORIGIN` Mechanism, and receives a `Projection
        <Projection>` that closes a recurrent loop in a `Process` and/or `System`.  If it is an `ORIGIN` Mechanism, then
        it is simply designated as such (since it will be assigned input and therefore be initialized in any event).

    - INITIALIZE_CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is the `sender <Projection_Base.sender>` of a
        `Projection <Projection>` that closes a loop in a `Process` or `System`, and that is not an `ORIGIN` Mechanism
        (since in that case it will be initialized in any event). An `initial value  <Run_InitialValues>` can be
        assigned to such Mechanisms, that will be used to initialize the Process or System when it is first run.  For
        additional information, see `Run <Run_Initial_Values>`, `System Mechanisms <System_Mechanisms>` and
        `System Input and Initialization <System_Execution_Input_And_Initialization>`.

    - TERMINAL
        A `ProcessingMechanism <ProcessingMechanism>` that is the last Mechanism of a `Process` and/or `System`, and
        that provides the output to the Process or System when it is `executed or run <Run>`.  A Process may
        have only one `TERMINAL` Mechanism, but a System may have many.  Note that the `TERMINAL`
        Mechanism of a Process is not necessarily a `TERMINAL` Mechanism of the System to which it belongs,
        as it may send Projections to other Processes in the System (see `example
        <LearningProjection_Target_vs_Terminal_Figure>`).  The `TERMINAL` Mechanisms of a Process or System are listed in
        its :keyword:`terminalMechanisms` attribute, and can be displayed using its :keyword:`show` method.  For
        additional details about `TERMINAL` Mechanisms in Processes, see `Process_Mechanisms` and
        `Process_Input_And_Output`; and for Systems see `System_Mechanisms`.

    - SINGLETON
        A `ProcessingMechanism <ProcessingMechanism>` that is the only Mechanism in a `Process` and/or `System`.
        It can serve the functions of an `ORIGIN` and/or a `TERMINAL` Mechanism.

    - MONITORED
        .

    - LEARNING
        A `LearningMechanism <LearningMechanism>` in a `Process` and/or `System`.

    - TARGET
        A `ComparatorMechanism` of a `Process` and/or `System` configured for learning that receives a target value
        from its `execute <ComparatorMechanism.ComparatorMechanism.execute>` or
        `run <ComparatorMechanism.ComparatorMechanism.execute>` method.  It must be associated with the `TERMINAL`
        Mechanism of the Process or System. The `TARGET` Mechanisms of a Process or System are listed in its
        :keyword:`target_nodes` attribute, and can be displayed using its :keyword:`show` method.  For additional
        details, see `TARGET Mechanisms <LearningMechanism_Targets>`, `learning sequence <Process_Learning_Sequence>`,
        and specifying `target values <Run_Targets>`.

    - RECURRENT_INIT
        .


    """
    ORIGIN = 0
    INTERNAL = 1
    CYCLE = 2
    INITIALIZE_CYCLE = 3
    TERMINAL = 4
    SINGLETON = 5
    MONITORED = 6
    LEARNING = 7
    TARGET = 8
    RECURRENT_INIT = 9
    OBJECTIVE = 10

class CompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class RunError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class Vertex(object):
    '''
        Stores a Component for use with a `Graph`

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
    '''

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
        self.backward_sources = set()

    def __repr__(self):
        return '(Vertex {0} {1})'.format(id(self), self.component)

class Graph(object):
    '''
        A Graph of vertices and edges/

        Attributes
        ----------

        comp_to_vertex : Dict[`Component <Component>` : `Vertex`]
            maps `Component` in the graph to the `Vertices <Vertex>` that represent them.

        vertices : List[Vertex]
            the `Vertices <Vertex>` contained in this Graph.

    '''

    def __init__(self):
        self.comp_to_vertex = collections.OrderedDict()  # Translate from mechanisms to related vertex
        self.vertices = []  # List of vertices within graph

    def copy(self):
        '''
            Returns
            -------

            A copy of the Graph. `Vertices <Vertex>` are distinct from their originals, and point to the same
            `Component <Component>` object : `Graph`
        '''
        g = Graph()

        for vertex in self.vertices:
            g.add_vertex(Vertex(vertex.component, feedback=vertex.feedback))

        for i in range(len(self.vertices)):
            g.vertices[i].parents = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in self.vertices[i].parents]
            g.vertices[i].children = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in self.vertices[i].children]

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
            self.remove_vertex(self.comp_to_vertex(component))
        except KeyError as e:
            raise CompositionError('Component {1} not found in graph {2}: {0}'.format(e, component, self))

    def remove_vertex(self, vertex):
        try:
            self.vertices.remove(vertex)
            del self.comp_to_vertex[vertex.component]
            # TODO:
            #   check if this removal puts the graph in an inconsistent state
        except ValueError as e:
            raise CompositionError('Vertex {1} not found in graph {2}: {0}'.format(e, vertex, self))

    def connect_components(self, parent, child):
        self.connect_vertices(self.comp_to_vertex[parent], self.comp_to_vertex[child])

    def connect_vertices(self, parent, child):
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def get_parents_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose parents will be returned

            Returns
            -------

            A list[Vertex] of the parent `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        return self.comp_to_vertex[component].parents

    def get_children_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        return self.comp_to_vertex[component].children

    def get_forward_children_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        forward_children = []
        for child in self.comp_to_vertex[component].children:
            if component not in self.comp_to_vertex[child.component].backward_sources:
                forward_children.append(child)
        return forward_children

    def get_forward_parents_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        forward_parents = []
        for parent in self.comp_to_vertex[component].parents:
            if parent.component not in self.comp_to_vertex[component].backward_sources:
                forward_parents.append(parent)
        return forward_parents

    def get_backward_children_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''
        backward_children = []
        for child in self.comp_to_vertex[component].children:
            if component in self.comp_to_vertex[child.component].backward_sources:
                backward_children.append(child)
        return backward_children

    def get_backward_parents_from_component(self, component):
        '''
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        '''

        return list(self.comp_to_vertex[component].backward_sources)

class Composition(object):
    '''
        Composition

        Arguments
        ---------

        Attributes
        ----------

        graph : `Graph`
            The full `Graph` associated with this Composition. Contains both Nodes (`Mechanisms <Mechanism>` or
            `Compositions <Composition>` and `Projections <Projection>` used in processing or learning.

        c_nodes : `list[Mechanisms and Compositions]`
            A list of all Composition Nodes (`Mechanisms <Mechanism>` and `Compositions <Composition>`) contained in
            this Composition

        COMMENT:
        name : str
            see `name <Composition_Name>`

        prefs : PreferenceSet
            see `prefs <Composition_Prefs>`
        COMMENT

    '''

    def __init__(self, 
                 name=None,
                 controller=None,
                 enable_controller=None):
        # core attributes
        if name is None:
            name = "composition"
        self.name = name
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.c_nodes = []
        self.required_c_node_roles = []
        self.input_CIM = CompositionInterfaceMechanism(name=self.name + " Input_CIM",
                                                       composition=self)
        self.input_CIM_states = {}
        self.output_CIM = CompositionInterfaceMechanism(name=self.name + " Output_CIM",
                                                        composition=self)
        self.output_CIM_states = {}
        self.enable_controller = enable_controller
        self.execution_ids = []
        self.controller = controller

        self._scheduler_processing = None
        self._scheduler_learning = None

        # status attributes
        self.graph_consistent = True  # Tracks if the Composition is in a state that can be run (i.e. no dangling projections, (what else?))
        self.needs_update_graph = True   # Tracks if the Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True   # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler_processing = True  # Tracks if the processing scheduler needs to be regenerated
        self.needs_update_scheduler_learning = True  # Tracks if the learning scheduler needs to be regenerated (mechanisms/projections added/removed etc)

        self.c_nodes_to_roles = OrderedDict()

        # Create lists to track certain categories of Composition Nodes:
        # TBI???
        self.explicit_input_nodes = []  # Need to track to know which to leave untouched
        self.all_input_nodes = []
        self.explicit_output_nodes = []  # Need to track to know which to leave untouched
        self.all_output_nodes = []
        self.target_nodes = []  # Do not need to track explicit as they must be explicit

        # Reporting
        self.results = []

        # TBI: update self.sched whenever something is added to the composition
        self.sched = Scheduler(composition=self)


    def __repr__(self):
        return '({0} {1})'.format(type(self).__name__, self.name)

    @property
    def graph_processing(self):
        '''
            The Composition's processing graph (contains only `Mechanisms <Mechanism>`, excluding those
            used in learning).

            :getter: Returns the processing graph, and builds the graph if it needs updating
            since the last access.
        '''
        if self.needs_update_graph_processing or self._graph_processing is None:
            self._update_processing_graph()

        return self._graph_processing

    @property
    def scheduler_processing(self):
        '''
            A default `Scheduler` automatically generated by the Composition, used for the
            (`processing <System_Execution_Processing>` phase of execution.

            :getter: Returns the default processing scheduler, and builds it if it needs updating since the last access.
        '''
        if self.needs_update_scheduler_processing or self._scheduler_processing is None:
            old_scheduler = self._scheduler_processing
            self._scheduler_processing = Scheduler(graph=self.graph_processing)

            if old_scheduler is not None:
                self._scheduler_processing.add_condition_set(old_scheduler.condition_set)

            self.needs_update_scheduler_processing = False

        return self._scheduler_processing

    @property
    def scheduler_learning(self):
        '''
            A default `Scheduler` automatically generated by the Composition, used for the
            `learning <System_Execution_Learning>` phase of execution.

            :getter: Returns the default learning scheduler, and builds it if it needs updating since the last access.
        '''
        if self.needs_update_scheduler_learning or self._scheduler_learning is None:
            old_scheduler = self._scheduler_learning
            # self._scheduler_learning = Scheduler(graph=self.graph)

            # if old_scheduler is not None:
            #     self._scheduler_learning.add_condition_set(old_scheduler.condition_set)
            #
            # self.needs_update_scheduler_learning = False

        return self._scheduler_learning

    @property
    def termination_processing(self):
        return self.scheduler_processing.termination_conds

    @termination_processing.setter
    def termination_processing(self, termination_conds):
        self.scheduler_processing.termination_conds = termination_conds

    def _get_unique_id(self):
        return uuid.uuid4()

    def add_c_node(self, node):
        '''
            Adds a Composition Node (`Mechanism` or `Composition`) to the Composition, if it is not already added

            Arguments
            ---------

            node : `Mechanism` or `Composition`
                the node to add
        '''

        if node not in [vertex.component for vertex in self.graph.vertices]:  # Only add if it doesn't already exist in graph
            node.is_processing = True
            self.graph.add_component(node)  # Set incoming edge list of node to empty
            self.c_nodes.append(node)
            self.c_nodes_to_roles[node] = set()

            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler_processing = True
            self.needs_update_scheduler_learning = True

        if isinstance(node, ControlMechanism):
            self.add_control_mechanism(node)

    def add_controller(self, node):
        self.controller = node
        # self.add_c_node(node)

    def add_control_mechanism(self, control_mechanism):

        if not isinstance(control_mechanism, ControlMechanism):
            raise CompositionError("{} is not a ControlMechanism.".format(control_mechanism.name))
        for input_state in control_mechanism._objective_mechanism.input_states:
            input_state.internal_only = True
        objective_node = control_mechanism._objective_mechanism
        self.add_c_node(objective_node)
        self.add_projection(objective_node.path_afferents[0])
        self.add_projection(objective_node.efferents[0])
        self._add_c_node_role(objective_node, CNodeRole.OBJECTIVE)
        self.add_required_c_node_role(objective_node, CNodeRole.OBJECTIVE)

    def add_projection(self, projection=None, sender=None, receiver=None, feedback=False):
        '''

            Adds a projection to the Composition, if it is not already added.

            If a *projection* is not specified, then a default MappingProjection is created.

            The sender and receiver of a particular Projection vertex within the Composition (the *sender* and
            *receiver* arguments of add_projection) must match the `sender <Projection.sender>` and `receiver
            <Projection.receiver>` specified on the Projection object itself.

                - If the *sender* and/or *receiver* arguments are not specified, then the `sender <Projection.sender>`
                  and/or `receiver <Projection.receiver>` attributes of the Projection object set the missing value(s).
                - If the `sender <Projection.sender>` and/or `receiver <Projection.receiver>` attributes of the
                  Projection object are not specified, then the *sender* and/or *receiver* arguments set the missing
                  value(s).

            Arguments
            ---------

            sender : Mechanism, Composition, or OutputState
                the sender of **projection**

            projection : Projection, matrix
                the projection to add

            receiver : Mechanism, Composition, or OutputState
                the receiver of **projection**

            feedback : Boolean
                if False, any cycles containing this projection will be
        '''

        if isinstance(projection, (np.ndarray, np.matrix, list)):
            projection = MappingProjection(matrix=projection)
        elif isinstance(projection, str):
            if projection in MATRIX_KEYWORD_VALUES:
                projection = MappingProjection(matrix=projection)
            else:
                raise CompositionError("Invalid projection ({}) specified for {}.".format(projection, self.name))
        elif projection is None:
            projection = MappingProjection()
        elif not isinstance(projection, Projection):
            raise CompositionError("Invalid projection ({}) specified for {}. Must be a Projection."
                                   .format(projection, self.name))

        if sender is None:
            if hasattr(projection, "sender"):
                sender = projection.sender.owner
            else:
                raise CompositionError("For a Projection to be added to a Composition, a sender must be specified, "
                                       "either on the Projection or in the call to Composition.add_projection(). {}"
                                       " is missing a sender specification. ".format(projection.name))

        sender_mechanism = sender
        graph_sender = sender
        if isinstance(sender, OutputState):
            sender_mechanism = sender.owner
            graph_sender = sender.owner
        elif isinstance(sender, Composition):
            sender_mechanism = sender.output_CIM

        if hasattr(projection, "sender"):
            if projection.sender.owner != sender and \
               projection.sender.owner != graph_sender and \
               projection.sender.owner != sender_mechanism:
                raise CompositionError("The position of {} in {} conflicts with its sender attribute."
                                       .format(projection.name, self.name))
        if receiver is None:
            if hasattr(projection, "receiver"):
                receiver = projection.receiver.owner
            else:
                raise CompositionError("For a Projection to be added to a Composition, a receiver must be specified, "
                                       "either on the Projection or in the call to Composition.add_projection(). {}"
                                       " is missing a receiver specification. ".format(projection.name))

        receiver_mechanism = receiver
        graph_receiver = receiver
        if isinstance(receiver, InputState):
            receiver_mechanism = receiver.owner
            graph_receiver = receiver.owner
        elif isinstance(receiver, Composition):
            receiver_mechanism = receiver.input_CIM

        if projection not in [vertex.component for vertex in self.graph.vertices]:

            projection.is_processing = False
            projection.name = '{0} to {1}'.format(sender, receiver)
            self.graph.add_component(projection, feedback=feedback)

            self.graph.connect_components(graph_sender, projection)
            self.graph.connect_components(projection, graph_receiver)
            self._validate_projection(projection, sender, receiver, sender_mechanism, receiver_mechanism)

            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler_processing = True
            self.needs_update_scheduler_learning = True

        else:
            raise CompositionError("Cannot add Projection: {}. This Projection is already in the Compositon."
                                   .format(projection.name))
        return projection

    def add_pathway(self, path):
        '''
            Adds an existing Pathway to the current Composition

            Arguments
            ---------

            path: the Pathway (Composition) to be added

        '''

        # identify nodes and projections
        c_nodes, projections = [], []
        for c in path.graph.vertices:
            if isinstance(c.component, Mechanism):
                c_nodes.append(c.component)
            elif isinstance(c.component, Composition):
                c_nodes.append(c.component)
            elif isinstance(c.component, Projection):
                projections.append(c.component)

        # add all c_nodes first
        for node in c_nodes:
            self.add_c_node(node)

        # then projections
        for p in projections:
            self.add_projection(p, p.sender.owner, p.receiver.owner)

        self._analyze_graph()

    def add_linear_processing_pathway(self, pathway, feedback=False):
        # First, verify that the pathway begins with a node
        if isinstance(pathway[0], (Mechanism, Composition)):
            self.add_c_node(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg
            raise CompositionError("The first item in a linear processing pathway must be a Node (Mechanism or "
                                   "Composition).")
        # Then, add all of the remaining nodes in the pathway
        for c in range(1, len(pathway)):
            # if the current item is a mechanism, add it
            if isinstance(pathway[c], Mechanism):
                self.add_c_node(pathway[c])

        # Then, loop through and validate that the mechanism-projection relationships make sense
        # and add MappingProjections where needed
        for c in range(1, len(pathway)):
            # if the current item is a Node
            if isinstance(pathway[c], (Mechanism, Composition)):
                if isinstance(pathway[c - 1], (Mechanism, Composition)):
                    # if the previous item was also a Composition Node, add a mapping projection between them
                    self.add_projection(MappingProjection(sender=pathway[c - 1],
                                                          receiver=pathway[c]),
                                        pathway[c - 1],
                                        pathway[c],
                                        feedback=feedback)
            # if the current item is a Projection
            elif isinstance(pathway[c], (Projection, np.ndarray, np.matrix, str, list)):
                if c == len(pathway) - 1:
                    raise CompositionError("{} is the last item in the pathway. A projection cannot be the last item in"
                                           " a linear processing pathway.".format(pathway[c]))
                # confirm that it is between two nodes, then add the projection
                if isinstance(pathway[c - 1], (Mechanism, Composition)) \
                        and isinstance(pathway[c + 1], (Mechanism, Composition)):
                    proj = pathway[c]
                    if isinstance(pathway[c], (np.ndarray, np.matrix, list)):
                        proj = MappingProjection(sender=pathway[c - 1],
                                                 matrix=pathway[c],
                                                 receiver=pathway[c + 1])
                    self.add_projection(proj, pathway[c - 1], pathway[c + 1], feedback=feedback)
                else:
                    raise CompositionError(
                        "{} is not between two Composition Nodes. A Projection in a linear processing pathway must be "
                        "preceded by a Composition Node (Mechanism or Composition) and followed by a Composition Node"
                        .format(pathway[c]))
            else:
                raise CompositionError("{} is not a Projection or a Composition node (Mechanism or Composition). A "
                                       "linear processing pathway must be made up of Projections and Composition Nodes."
                                       .format(pathway[c]))

    def _validate_projection(self,
                             projection,
                             sender, receiver,
                             graph_sender,
                             graph_receiver,
                             ):

        if not hasattr(projection, "sender") or not hasattr(projection, "receiver"):
            projection.init_args['sender'] = graph_sender
            projection.init_args['receiver'] = graph_receiver
            projection.context.initialization_status = ContextFlags.DEFERRED_INIT
            projection._deferred_init(context=" INITIALIZING ")

        if projection.sender.owner != graph_sender:
            raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                   "Components in the Composition.".format(projection, sender))
        if projection.receiver.owner != graph_receiver:
            raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                   "Components in the Composition.".format(projection, receiver))

    def _analyze_graph(self, graph=None):
        ########
        # Determines identity of significant nodes of the graph
        # Each node falls into one or more of the following categories
        # - Origin: Origin nodes are those which do not receive any projections.
        # - Terminal: Terminal nodes provide the output of the composition. By
        #   default, those which do not send any projections, but they may also be
        #   specified explicitly.
        # - Recurrent_init: Recurrent_init nodes send projections that close recurrent
        #   loops in the composition (or projections that are explicitly specified as
        #   recurrent). They need an initial value so that their receiving nodes
        #   have input.
        # - Cycle: Cycle nodes receive projections from Recurrent_init nodes. They
        #   can be viewed as the starting points of recurrent loops.
        # The following categories can be explicitly set by the user in which case their
        # values are not changed based on the graph analysis. Additional nodes may
        # be automatically added besides those specified by the user.
        # - Input: Input nodes accept inputs from the input_dict of the composition.
        #   All Origin nodes are added to this category automatically.
        # - Output: Output nodes provide their values as outputs of the composition.
        #   All Terminal nodes are added to this category automatically.
        # - Target: Target nodes receive target values for the composition to be
        #   used by learning and control. They are usually Comparator nodes that
        #   compare the target value to the output of another node in the composition.
        # - Monitored: Monitored nodes send projections to Target nodes.
        ########
        if graph is None:
            graph = self.graph_processing

        # Clear old information
        self.c_nodes_to_roles.update({k: set() for k in self.c_nodes_to_roles})

        if len(self.scheduler_processing.consideration_queue) > 0:
            for node in self.scheduler_processing.consideration_queue[0]:
                self._add_c_node_role(node, CNodeRole.ORIGIN)
        if len(self.scheduler_processing.consideration_queue) > 0:
            for node in self.scheduler_processing.consideration_queue[-1]:
                self._add_c_node_role(node, CNodeRole.TERMINAL)
        # Identify Origin nodes
        for node in self.c_nodes:
            if graph.get_parents_from_component(node) == []:
                self._add_c_node_role(node, CNodeRole.ORIGIN)
        # Identify Terminal nodes
            if graph.get_children_from_component(node) == []:
                self._add_c_node_role(node, CNodeRole.TERMINAL)
        # Identify Recurrent_init and Cycle nodes
        visited = []  # Keep track of all nodes that have been visited
        for origin_node in self.get_c_nodes_by_role(CNodeRole.ORIGIN):  # Cycle through origin nodes first
            visited_current_path = []  # Track all nodes visited from the current origin
            next_visit_stack = []  # Keep a stack of nodes to be visited next
            next_visit_stack.append(origin_node)
            for node in next_visit_stack:  # While the stack isn't empty
                visited.append(node)  # Mark the node as visited
                visited_current_path.append(node)  # And visited during the current path
                children = [vertex.component for vertex in graph.get_children_from_component(node)]
                for child in children:
                    # If the child has been visited this path and is not already initialized
                    if child in visited_current_path:
                        self._add_c_node_role(node, CNodeRole.RECURRENT_INIT)
                        self._add_c_node_role(child, CNodeRole.CYCLE)
                    elif child not in visited:  # Else if the child has not been explored
                        next_visit_stack.append(child)  # Add it to the visit stack
        for node in self.c_nodes:
            if node not in visited:  # Check the rest of the nodes
                visited_current_path = []
                next_visit_stack = []
                next_visit_stack.append(node)
                for remaining_node in next_visit_stack:
                    visited.append(remaining_node)
                    visited_current_path.append(remaining_node)
                    children = [vertex.component for vertex in graph.get_children_from_component(remaining_node)]
                    for child in children:
                        if child in visited_current_path:
                            self._add_c_node_role(remaining_node, CNodeRole.RECURRENT_INIT)
                            self._add_c_node_role(child, CNodeRole.CYCLE)
                        elif child not in visited:
                            next_visit_stack.append(child)

        # toposorted_graph = self.scheduler_processing._call_toposort(graph)[0]
        # if len(toposorted_graph) > 0:
        #     for node in toposorted_graph[-1]:
        #         self._add_c_node_role(node, CNodeRole.TERMINAL)
        for node_role_pair in self.required_c_node_roles:
            self._add_c_node_role(node_role_pair[0], node_role_pair[1])

        self._create_CIM_states()

        self.needs_update_graph = False

    def _update_processing_graph(self):
        '''
        Constructs the processing graph (the graph that contains only non-learning nodes as vertices)
        from the composition's full graph
        '''
        logger.debug('Updating processing graph')

        self._graph_processing = self.graph.copy()

        visited_vertices = set()
        next_vertices = []  # a queue

        unvisited_vertices = True

        while unvisited_vertices:
            for vertex in self._graph_processing.vertices:
                if vertex not in visited_vertices:
                    next_vertices.append(vertex)
                    break
            else:
                unvisited_vertices = False

            logger.debug('processing graph vertices: {0}'.format(self._graph_processing.vertices))
            while len(next_vertices) > 0:
                cur_vertex = next_vertices.pop(0)
                logger.debug('Examining vertex {0}'.format(cur_vertex))

                # must check that cur_vertex is not already visited because in cycles, some nodes may be added to next_vertices twice
                if cur_vertex not in visited_vertices and not cur_vertex.component.is_processing:
                    for parent in cur_vertex.parents:
                        parent.children.remove(cur_vertex)
                        for child in cur_vertex.children:
                            child.parents.remove(cur_vertex)
                            if cur_vertex.feedback:
                                child.backward_sources.add(parent.component)
                            self._graph_processing.connect_vertices(parent, child)

                    for node in cur_vertex.parents + cur_vertex.children:
                        logger.debug('New parents for vertex {0}: \n\t{1}\nchildren: \n\t{2}'.format(node, node.parents, node.children))
                    logger.debug('Removing vertex {0}'.format(cur_vertex))

                    self._graph_processing.remove_vertex(cur_vertex)

                visited_vertices.add(cur_vertex)
                # add to next_vertices (frontier) any parents and children of cur_vertex that have not been visited yet
                next_vertices.extend([vertex for vertex in cur_vertex.parents + cur_vertex.children if vertex not in visited_vertices])

        self.needs_update_graph_processing = False

    def get_c_nodes_by_role(self, role):
        '''
            Returns a set of Composition Nodes in this Composition that have the role `role`

            Arguments
            _________

            role : CNodeRole
                the set of nodes having this role to return

            Returns
            -------

            set of Compositon Nodes with `CNodeRole` `role` : set(`Mechanisms <Mechanism>` and
            `Compositions <Composition>`)
        '''
        if role not in CNodeRole:
            raise CompositionError('Invalid CNodeRole: {0}'.format(role))

        try:
            return [node for node in self.c_nodes if role in self.c_nodes_to_roles[node]]

        except KeyError as e:
            raise CompositionError('Node missing from {0}.c_nodes_to_roles: {1}'.format(self, e))

    def get_roles_by_c_node(self, c_node):
        try:
            return self.c_nodes_to_roles[c_node]
        except KeyError:
            raise CompositionError('Node {0} not found in {1}.c_nodes_to_roles'.format(c_node, self))

    def _set_c_node_roles(self, c_node, roles):
        self._clear_c_node_roles(c_node)
        for role in roles:
            self._add_c_node_role(role)

    def _clear_c_node_roles(self, c_node):
        if c_node in self.c_nodes_to_roles:
            self.c_nodes_to_roles[c_node] = set()

    def _add_c_node_role(self, c_node, role):
        if role not in CNodeRole:
            raise CompositionError('Invalid CNodeRole: {0}'.format(role))

        self.c_nodes_to_roles[c_node].add(role)

    def _remove_c_node_role(self, c_node, role):
        if role not in CNodeRole:
            raise CompositionError('Invalid CNodeRole: {0}'.format(role))

        self.c_nodes_to_roles[c_node].remove(role)

    def add_required_c_node_role(self, c_node, role):
        if role not in CNodeRole:
            raise CompositionError('Invalid CNodeRole: {0}'.format(role))

        node_role_pair = (c_node, role)
        if node_role_pair not in self.required_c_node_roles:
            self.required_c_node_roles.append(node_role_pair)

    def remove_required_c_node_role(self, c_node, role):
        if role not in CNodeRole:
            raise CompositionError('Invalid CNodeRole: {0}'.format(role))

        node_role_pair = (c_node, role)
        if node_role_pair in self.required_c_node_roles:
            self.required_c_node_roles.remove(node_role_pair)

    # mech_type specifies a type of mechanism, mech_type_list contains all of the mechanisms of that type
    # feed_dict is a dictionary of the input states of each mechanism of the specified type
    # def _validate_feed_dict(self, feed_dict, mech_type_list, mech_type):
    #     for mech in feed_dict.keys():  # For each mechanism given an input
    #         if mech not in mech_type_list:  # Check that it is the right kind of mechanism in the composition
    #             if mech_type[0] in ['a', 'e', 'i', 'o', 'u']:  # Check for grammar
    #                 article = "an"
    #             else:
    #                 article = "a"
    #             # Throw an error informing the user that the mechanism was not found in the mech type list
    #             raise ValueError("The Mechanism \"{}\" is not {} {} of the composition".format(mech.name, article, mech_type))
    #         for i, timestep in enumerate(feed_dict[mech]):  # If mechanism is correct type, iterate over timesteps
    #             # Check if there are multiple input states specified
    #             try:
    #                 timestep[0]
    #             except TypeError:
    #                 raise TypeError("The Mechanism  \"{}\" is incorrectly formatted at time step {!s}. "
    #                                 "Likely missing set of brackets.".format(mech.name, i))
    #             if not isinstance(timestep[0], Iterable) or isinstance(timestep[0], str):  # Iterable imported from collections
    #                 # If not, embellish the formatting to match the verbose case
    #                 timestep = [timestep]
    #             # Then, check that each input_state is receiving the right size of input
    #             for i, value in enumerate(timestep):
    #                 val_length = len(value)
    #                 state_length = len(mech.input_state.instance_defaults.variable)
    #                 if val_length != state_length:
    #                     raise ValueError("The value provided for InputState {!s} of the Mechanism \"{}\" has length "
    #                                      "{!s} where the InputState takes values of length {!s}".
    #                                      format(i, mech.name, val_length, state_length))


    def _validate_feed_dict(self, feed_dict, mech_type_list, mech_type):
        for mech in feed_dict.keys():  # For each mechanism given an input
            if mech not in mech_type_list:  # Check that it is the right kind of mechanism in the composition
                if mech_type[0] in ['a', 'e', 'i', 'o', 'u']:  # Check for grammar
                    article = "an"
                else:
                    article = "a"
                # Throw an error informing the user that the mechanism was not found in the mech type list
                raise ValueError("The Mechanism \"{}\" is not {} {} of the composition".format(mech.name, article, mech_type))
            for i, timestep in enumerate(feed_dict[mech]):  # If mechanism is correct type, iterate over timesteps
                # Check if there are multiple input states specified
                try:
                    timestep[0]
                except TypeError:
                    raise TypeError("The Mechanism  \"{}\" is incorrectly formatted at time step {!s}. "
                                    "Likely missing set of brackets.".format(mech.name, i))
                if not isinstance(timestep[0], collections.Iterable) or isinstance(timestep[0], str):  # Iterable imported from collections
                    # If not, embellish the formatting to match the verbose case
                    timestep = [timestep]
                # Then, check that each input_state is receiving the right size of input
                for i, value in enumerate(timestep):
                    val_length = len(value)
                    state_length = len(mech.input_state.instance_defaults.value)
                    if val_length != state_length:
                        raise ValueError("The value provided for InputState {!s} of the Mechanism \"{}\" has length "
                                         "{!s} where the InputState takes values of length {!s}".
                                         format(i, mech.name, val_length, state_length))

    def _create_CIM_states(self):
        '''
            - remove the default InputState and OutputState from the CIMs if this is the first time that real
              InputStates and OutputStates are being added to the CIMs

            - create a corresponding InputState and OutputState on the Input CompositionInterfaceMechanism for each
              InputState of each origin node, and a Projection between the newly created InputCIM OutputState and the
              origin InputState

            - create a corresponding InputState and OutputState on the Output CompositionInterfaceMechanism for each
              OutputState of each terminal node, and a Projection between the terminal OutputState and the newly created
              OutputCIM InputState

            - build two dictionaries:

                (1) input_CIM_states = { Origin Node InputState: (InputCIM InputState, InputCIM OutputState) }

                (2) output_CIM_states = { Terminal Node OutputState: (OutputCIM InputState, OutputCIM OutputState) }

            - delete all of the above for any node States which were previously, but are no longer, classified as
              Origin/Terminal

        '''

        if not self.input_CIM.connected_to_composition:
            self.input_CIM.input_states.remove(self.input_CIM.input_state)
            self.input_CIM.output_states.remove(self.input_CIM.output_state)
            self.input_CIM.connected_to_composition = True

        if not self.output_CIM.connected_to_composition:
            self.output_CIM.input_states.remove(self.output_CIM.input_state)
            self.output_CIM.output_states.remove(self.output_CIM.output_state)
            self.output_CIM.connected_to_composition = True

        current_origin_input_states = set()

        #  INPUT CIMS
        # loop over all origin nodes

        for node in self.get_c_nodes_by_role(CNodeRole.ORIGIN):

            for input_state in node.external_input_states:
                # add it to our set of current input states
                current_origin_input_states.add(input_state)

                # if there is not a corresponding CIM output state, add one
                if input_state not in set(self.input_CIM_states.keys()):

                    interface_input_state = InputState(owner=self.input_CIM,
                                                       variable=input_state.value,
                                                       reference_value=input_state.value,
                                                       name="INPUT_CIM_" + node.name + "_" + input_state.name)

                    interface_output_state = OutputState(owner=self.input_CIM,
                                                         variable=OWNER_VALUE,
                                                         default_variable=self.input_CIM.variable,
                                                         function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                                                         name="INPUT_CIM_" + node.name + "_" + OutputState.__name__)

                    self.input_CIM_states[input_state] = [interface_input_state, interface_output_state]

                    MappingProjection(sender=interface_output_state,
                                      receiver=input_state,
                                      matrix= IDENTITY_MATRIX,
                                      name="("+interface_output_state.name + ") to ("
                                           + input_state.owner.name + "-" + input_state.name+")")

        sends_to_input_states = set(self.input_CIM_states.keys())

        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for input_state in sends_to_input_states.difference(current_origin_input_states):
            for projection in input_state.path_afferents:
                if projection.sender == self.input_CIM_states[input_state][1]:
                    # remove the corresponding projection from the ORIGIN node's path afferents
                    input_state.path_afferents.remove(projection)

                    # projection.receiver.efferents.remove(projection)
                    # Bug? ^^ projection is not in receiver.efferents??

            # remove the CIM input and output states associated with this Origin node input state
            self.input_CIM.input_states.remove(self.input_CIM_states[input_state][0])
            self.input_CIM.output_states.remove(self.input_CIM_states[input_state][1])

            # and from the dictionary of CIM output state/input state pairs
            del self.input_CIM_states[input_state]

        # OUTPUT CIMS
        # loop over all terminal nodes

        current_terminal_output_states = set()
        for node in self.get_c_nodes_by_role(CNodeRole.TERMINAL):
            for output_state in node.output_states:
                current_terminal_output_states.add(output_state)
                # if there is not a corresponding CIM output state, add one
                if output_state not in set(self.output_CIM_states.keys()):

                    interface_input_state = InputState(owner=self.output_CIM,
                                                       variable=output_state.instance_defaults.value,
                                                       reference_value=output_state.instance_defaults.value,
                                                       name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    interface_output_state = OutputState(
                        owner=self.output_CIM,
                        variable=OWNER_VALUE,
                        function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                        reference_value=output_state.instance_defaults.value,
                        name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    self.output_CIM_states[output_state] = [interface_input_state, interface_output_state]

                    proj_name = "(" + output_state.name + ") to (" + interface_input_state.name + ")"

                    MappingProjection(sender=output_state,
                                      receiver=interface_input_state,
                                      matrix=IDENTITY_MATRIX,
                                      name=proj_name)

        previous_terminal_output_states = set(self.output_CIM_states.keys())
        for output_state in previous_terminal_output_states.difference(current_terminal_output_states):
            # remove the CIM input and output states associated with this Terminal Node output state
            self.output_CIM.remove_states(self.output_CIM_states[output_state][0])
            self.output_CIM.remove_states(self.output_CIM_states[output_state][1])
            del self.output_CIM_states[output_state]

    def _assign_values_to_input_CIM(self, inputs):
        """
            Assign values from input dictionary to the InputStates of the Input CIM, then execute the Input CIM

        """

        build_CIM_input = []

        for input_state in self.input_CIM.input_states:
            # "input_state" is an InputState on the input CIM

            for key in self.input_CIM_states:
                # "key" is an InputState on an origin Node of the Composition
                if self.input_CIM_states[key][0] == input_state:
                    origin_input_state = key
                    origin_node = key.owner
                    index = origin_node.input_states.index(origin_input_state)

                    if isinstance(origin_node, CompositionInterfaceMechanism):
                        index = origin_node.input_states.index(origin_input_state)
                        origin_node = origin_node.composition

                    if origin_node in inputs:
                        value = inputs[origin_node][index]

                    else:
                        value = origin_node.instance_defaults.variable[index]

            build_CIM_input.append(value)

        self.input_CIM.execute(build_CIM_input)

    def _assign_execution_ids(self, execution_id=None):
        '''
            assigns the same uuid to each Node in the composition's processing graph as well as the CIMs. The uuid is
            either specified in the user's call to run(), or generated randomly at run time.
        '''

        # Traverse processing graph and assign one uuid to all of its nodes
        if execution_id is None:
            execution_id = self._get_unique_id()

        if execution_id not in self.execution_ids:
            self.execution_ids.append(execution_id)

        for v in self._graph_processing.vertices:
            v.component._execution_id = execution_id

        self.input_CIM._execution_id = execution_id
        self.output_CIM._execution_id = execution_id
        # self.target_CIM._execution_id = execution_id

        self._execution_id = execution_id
        return execution_id

    def _identify_clamp_inputs(self, list_type, input_type, origins):
        # clamp type of this list is same as the one the user set for the whole composition; return all nodes
        if list_type == input_type:
            return origins
        # the user specified different types of clamps for each origin node; generate a list accordingly
        elif isinstance(input_type, dict):
            return [k for k, v in input_type.items() if list_type == v]
        # clamp type of this list is NOT same as the one the user set for the whole composition; return empty list
        else:
            return []

    def _parse_runtime_params(self, runtime_params):
        if runtime_params is None:
            return {}
        for c_node in runtime_params:
            for param in runtime_params[c_node]:
                if isinstance(runtime_params[c_node][param], tuple):
                    if len(runtime_params[c_node][param]) == 1:
                        runtime_params[c_node][param] = (runtime_params[c_node][param], Always())
                    elif len(runtime_params[c_node][param]) != 2:
                        raise SystemError("Invalid runtime parameter specification ({}) for {}'s {} parameter in {}. "
                                          "Must be a tuple of the form (parameter value, condition), or simply the "
                                          "parameter value. ".format(runtime_params[c_node][param],
                                                                     c_node.name,
                                                                     param,
                                                                     self.name))
                else:
                    runtime_params[c_node][param] = (runtime_params[c_node][param], Always())
        return runtime_params


    def execute(
        self,
        inputs=None,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        call_before_time_step=None,
        call_before_pass=None,
        call_after_time_step=None,
        call_after_pass=None,
        execution_id=None,
        clamp_input=SOFT_CLAMP,
        targets=None,
        runtime_params=None,
        context=None
    ):
        '''
            Passes inputs to any Nodes receiving inputs directly from the user (via the "inputs" argument) then
            coordinates with the Scheduler to receive and execute sets of nodes that are eligible to run until
            termination conditions are met.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` or `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each node in the composition that receives inputs from
                the user. For each pair, the key is the node (Mechanism or Composition) and the value is an input,
                the shape of which must match the node's default variable.

            scheduler_processing : Scheduler
                the scheduler object that owns the conditions that will instruct the non-learning execution of this Composition. \
                If not specified, the Composition will use its automatically generated scheduler

            scheduler_learning : Scheduler
                the scheduler object that owns the conditions that will instruct the Learning execution of this Composition. \
                If not specified, the Composition will use its automatically generated scheduler

            execution_id : UUID
                execution_id will typically be set to none and assigned randomly at runtime

            call_before_time_step : callable
                will be called before each `TIME_STEP` is executed

            call_after_time_step : callable
                will be called after each `TIME_STEP` is executed

            call_before_pass : callable
                will be called before each `PASS` is executed

            call_after_pass : callable
                will be called after each `PASS` is executed

            Returns
            ---------

            output value of the final Mechanism executed in the Composition : various
        '''

        nested = False
        if len(self.input_CIM.path_afferents) > 0:
            nested = True

        runtime_params = self._parse_runtime_params(runtime_params)

        if targets is None:
            targets = {}
        execution_id = self._assign_execution_ids(execution_id)
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        if scheduler_learning is None:
            scheduler_learning = self.scheduler_learning

        if nested:
            self.execution_id = self.input_CIM.path_afferents[0].sender.owner.composition._execution_id
            self.input_CIM.context.execution_phase = ContextFlags.PROCESSING
            self.input_CIM.execute(context=ContextFlags.PROCESSING)

        else:
            inputs = self._adjust_execution_stimuli(inputs)
            self._assign_values_to_input_CIM(inputs)

        if termination_processing is None:
            termination_processing = self.termination_processing

        next_pass_before = 1
        next_pass_after = 1
        if clamp_input:
            soft_clamp_inputs = self._identify_clamp_inputs(SOFT_CLAMP, clamp_input, origin_nodes)
            hard_clamp_inputs = self._identify_clamp_inputs(HARD_CLAMP, clamp_input, origin_nodes)
            pulse_clamp_inputs = self._identify_clamp_inputs(PULSE_CLAMP, clamp_input, origin_nodes)
            no_clamp_inputs = self._identify_clamp_inputs(NO_CLAMP, clamp_input, origin_nodes)
        # run scheduler to receive sets of nodes that may be executed at this time step in any order
        execution_scheduler = scheduler_processing

        if call_before_pass:
            call_before_pass()

        for next_execution_set in execution_scheduler.run(termination_conds=termination_processing, execution_id=execution_id):
            if call_after_pass:
                if next_pass_after == execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL):
                    logger.debug('next_pass_after {0}\tscheduler pass {1}'.format(next_pass_after, execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL)))
                    call_after_pass()
                    next_pass_after += 1

            if call_before_pass:
                if next_pass_before == execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL):
                    call_before_pass()
                    logger.debug('next_pass_before {0}\tscheduler pass {1}'.format(next_pass_before, execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL)))
                    next_pass_before += 1

            if call_before_time_step:
                call_before_time_step()

            frozen_values = {}
            new_values = {}
            # execute each node with EXECUTING in context
            for node in next_execution_set:
                frozen_values[node] = node.output_values
                if node in origin_nodes:
                    # KAM 8/28 commenting out the below code because it's not necessarily how we want to handle
                    # a recurrent projection on the first time step (meaning, before its node has executed)
                    # FIX: determine the correct behavior for this case & document it

                    # if (
                    #     scheduler_processing.times[execution_id][TimeScale.TRIAL][TimeScale.TIME_STEP] == 0
                    #     and hasattr(node, "recurrent_projection")
                    # ):
                    #     node.recurrent_projection.sender.value = [0.0]
                    if clamp_input:
                        if node in hard_clamp_inputs:
                            # clamp = HARD_CLAMP --> "turn off" recurrent projection
                            if hasattr(node, "recurrent_projection"):
                                node.recurrent_projection.sender.value = [0.0]
                        elif node in no_clamp_inputs:
                            for input_state in node.input_states:
                                self.input_CIM_states[input_state][1].value = 0.0

                if isinstance(node, Mechanism):

                    execution_runtime_params = {}

                    if node in runtime_params:
                        for param in runtime_params[node]:
                            if runtime_params[node][param][1].is_satisfied(scheduler=execution_scheduler,
                                                                           # KAM 5/15/18 - not sure if this will always be the correct execution id:
                                                                                execution_id=self._execution_id):
                                execution_runtime_params[param] = runtime_params[node][param][0]

                    node.context.execution_phase = ContextFlags.PROCESSING
                    if not (CNodeRole.OBJECTIVE in self.get_roles_by_c_node(node) and not node is self.controller):

                        node.execute(runtime_params=execution_runtime_params,
                                     context=ContextFlags.COMPOSITION)


                    for key in node._runtime_params_reset:
                        node._set_parameter_value(key, node._runtime_params_reset[key])
                    node._runtime_params_reset = {}

                    for key in node.function_object._runtime_params_reset:
                        node.function_object._set_parameter_value(key,
                                                                  node.function_object._runtime_params_reset[
                                                                           key])
                    node.function_object._runtime_params_reset = {}
                    node.context.execution_phase = ContextFlags.IDLE
                elif isinstance(node, Composition):
                    node.execute(execution_id=self._execution_id)
                if node in origin_nodes:
                    if clamp_input:
                        if node in pulse_clamp_inputs:
                            for input_state in node.input_states:
                            # clamp = None --> "turn off" input node
                                self.input_CIM_states[input_state][1].value = 0
                new_values[node] = node.output_values

                for i in range(len(node.output_states)):
                    node.output_states[i].set_value_without_logging(frozen_values[node][i])

            for node in next_execution_set:

                for i in range(len(node.output_states)):
                    node.output_states[i].set_value_without_logging(new_values[node][i])

            if call_after_time_step:
                call_after_time_step()

        if call_after_pass:
            call_after_pass()

        self.output_CIM.context.execution_phase = ContextFlags.PROCESSING
        self.output_CIM.execute(context=ContextFlags.PROCESSING)

        output_values = []
        for i in range(0, len(self.output_CIM.output_states)):
            output_values.append(self.output_CIM.output_states[i].value)

        # TBI control phase

        return output_values

    def run(
        self,
        inputs=None,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        execution_id=None,
        num_trials=None,
        call_before_time_step=None,
        call_after_time_step=None,
        call_before_pass=None,
        call_after_pass=None,
        call_before_trial=None,
        call_after_trial=None,
        clamp_input=SOFT_CLAMP,
        targets=None,
        initial_values=None,
        runtime_params=None
    ):
        '''
            Passes inputs to compositions, then executes
            to receive and execute sets of nodes that are eligible to run until termination conditions are met.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` : list } or { `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each Node in the composition that receives inputs from
                the user. For each pair, the key is the Node and the value is a list of inputs. Each input in the
                list corresponds to a certain `TRIAL`.

            scheduler_processing : Scheduler
                the scheduler object that owns the conditions that will instruct the non-learning execution of
                this Composition. If not specified, the Composition will use its automatically generated scheduler.

            scheduler_learning : Scheduler
                the scheduler object that owns the conditions that will instruct the Learning execution of
                this Composition. If not specified, the Composition will use its automatically generated scheduler.

            execution_id : UUID
                execution_id will typically be set to none and assigned randomly at runtime.

            num_trials : int
                typically, the composition will infer the number of trials from the length of its input specification.
                To reuse the same inputs across many trials, you may specify an input dictionary with lists of length 1,
                or use default inputs, and select a number of trials with num_trials.

            call_before_time_step : callable
                will be called before each `TIME_STEP` is executed.

            call_after_time_step : callable
                will be called after each `TIME_STEP` is executed.

            call_before_pass : callable
                will be called before each `PASS` is executed.

            call_after_pass : callable
                will be called after each `PASS` is executed.

            call_before_trial : callable
                will be called before each `TRIAL` is executed.

            call_after_trial : callable
                will be called after each `TRIAL` is executed.

            initial_values : Dict[Node: Node Value]
                sets the values of nodes before the start of the run. This is useful in cases where a node's value is
                used before that node executes for the first time (usually due to recurrence or control).

            runtime_params : Dict[Node: Dict[Param: Tuple(Value, Condition)]]
                nested dictionary of (value, `Condition`) tuples for parameters of Nodes (`Mechanisms <Mechanism>` or
                `Compositions <Composition>` of the Composition; specifies alternate parameter values to be used only
                during this `Run` when the specified `Condition` is met.

                Outer dictionary:
                    - *key* - Node
                    - *value* - Runtime Parameter Specification Dictionary

                Runtime Parameter Specification Dictionary:
                    - *key* - keyword corresponding to a parameter of the Node
                    - *value* - tuple in which the index 0 item is the runtime parameter value, and the index 1 item is
                      a `Condition`

                See `Run_Runtime_Parameters` for more details and examples of valid dictionaries.

            Returns
            ---------

            output value of the final Node executed in the composition : various
        '''

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        # TBI: Learning
        if scheduler_learning is None:
            scheduler_learning = self.scheduler_learning

        if termination_processing is None:
            termination_processing = self.termination_processing

        if initial_values is not None:
            for node in initial_values:
                if node not in self.c_nodes:
                    raise CompositionError("{} (entry in initial_values arg) is not a node in \'{}\'".
                                      format(node.name, self.name))


        self._analyze_graph()

        execution_id = self._assign_execution_ids(execution_id)

        scheduler_processing._init_counts(execution_id=execution_id)
        # scheduler_learning._init_counts(execution_id=execution_id)

        scheduler_processing.update_termination_conditions(termination_processing)
        # scheduler_learning.update_termination_conditions(termination_learning)

        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)

        # if there is only one origin mechanism, allow inputs to be specified in a list
        if isinstance(inputs, (list, np.ndarray)):
            if len(origin_nodes) == 1:
                inputs = {next(iter(origin_nodes)): inputs}
            else:
                raise CompositionError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                               "nodes.".format(self.name, len(origin_nodes)))
        elif not isinstance(inputs, dict):
            if len(origin_nodes) == 1:
                raise CompositionError(
                    "Inputs to {} must be specified in a list or in a dictionary with the origin mechanism({}) "
                    "as its only key".format(self.name, next(iter(origin_nodes)).name))
            else:
                raise CompositionError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                               "nodes.".format(self.name, len(origin_nodes)))

        inputs, num_inputs_sets = self._adjust_stimulus_dict(inputs)

        if num_trials is not None:
            num_trials = num_trials
        else:
            num_trials = num_inputs_sets

        if targets is None:
            targets = {}

        scheduler_processing._reset_counts_total(TimeScale.RUN, execution_id)

        result = None

        # --- RESET FOR NEXT TRIAL ---
        # by looping over the length of the list of inputs - each input represents a TRIAL
        for trial_num in range(num_trials):
            # Execute call before trial "hook" (user defined function)
            if call_before_trial:
                call_before_trial()

            if termination_processing[TimeScale.RUN].is_satisfied(scheduler=scheduler_processing,
                                                                  execution_id=execution_id):
                break

        # PROCESSING ------------------------------------------------------------------------

            # Prepare stimuli from the outside world  -- collect the inputs for this TRIAL and store them in a dict
            execution_stimuli = {}
            stimulus_index = trial_num % num_inputs_sets
            for node in inputs:
                execution_stimuli[node] = inputs[node][stimulus_index]
            # execute processing
            # pass along the stimuli for this trial
            trial_output = self.execute(inputs=execution_stimuli,
                                        scheduler_processing=scheduler_processing,
                                        scheduler_learning=scheduler_learning,
                                        termination_processing=termination_processing,
                                        termination_learning=termination_learning,
                                        call_before_time_step=call_before_time_step,
                                        call_before_pass=call_before_pass,
                                        call_after_time_step=call_after_time_step,
                                        call_after_pass=call_after_pass,
                                        execution_id=execution_id,
                                        clamp_input=clamp_input,
                                        runtime_params=runtime_params)

        # ---------------------------------------------------------------------------------
            # store the result of this execute in case it will be the final result

            # terminal_mechanisms = self.get_c_nodes_by_role(CNodeRole.TERMINAL)
            # for terminal_mechanism in terminal_mechanisms:
            #     for terminal_output_state in terminal_mechanism.output_states:
            #         CIM_output_state = self.output_CIM_states[terminal_output_state]
            #         CIM_output_state.value = terminal_output_state.value

            # object.results.append(result)
            if isinstance(trial_output, Iterable):
                result_copy = trial_output.copy()
            else:
                result_copy = trial_output
            self.results.append(result_copy)

            if trial_output is not None:
                result = trial_output

        # LEARNING ------------------------------------------------------------------------
            # Prepare targets from the outside world  -- collect the targets for this TRIAL and store them in a dict
            execution_targets = {}
            target_index = trial_num % num_inputs_sets
            # Assign targets:
            if targets is not None:

                if isinstance(targets, function_type):
                    self.target = targets
                else:
                    for node in targets:
                        if callable(targets[node]):
                            execution_targets[node] = targets[node]
                        else:
                            execution_targets[node] = targets[node][target_index]

                    # devel needs the lines below because target and current_targets are attrs of system
                    # self.target = execution_targets
                    # self.current_targets = execution_targets

            # TBI execute learning
            # pass along the targets for this trial
            # self.learning_composition.execute(execution_targets,
            #                                   scheduler_processing,
            #                                   scheduler_learning,
            #                                   call_before_time_step,
            #                                   call_before_pass,
            #                                   call_after_time_step,
            #                                   call_after_pass,
            #                                   execution_id,
            #                                   clamp_input,
            #                                   )

            if call_after_trial:
                call_after_trial()

        scheduler_processing.clocks[execution_id]._increment_time(TimeScale.RUN)

        return self.results

    def run_simulation(self):
        print("simulation runs now")
    def _input_matches_variable(self, input_value, var):
        # input_value states are uniform
        if np.shape(np.atleast_2d(input_value)) == np.shape(var):
            return "homogeneous"
        # input_value states have different lengths
        elif len(np.shape(var)) == 1 and isinstance(var[0], (list, np.ndarray)):
            for i in range(len(input_value)):
                if len(input_value[i]) != len(var[i]):
                    return False
            return "heterogeneous"
        return False

    def _adjust_stimulus_dict(self, stimuli):

        # STEP 1: validate that there is a one-to-one mapping of input entries to origin nodes


        # Check that all of the nodes listed in the inputs dict are ORIGIN nodes in the self
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        for node in stimuli.keys():
            if not node in origin_nodes:
                raise CompositionError("{} in inputs dict for {} is not one of its ORIGIN nodes".
                               format(node.name, self.name))
        # Check that all of the ORIGIN nodes are represented - if not, use default_variable
        for node in origin_nodes:
            if not node in stimuli:
                # Change error below to warning??
                # raise RunError("Entry for ORIGIN Node {} is missing from the inputs dict for {}".
                #                format(node.name, self.name))
                stimuli[node] = node.default_external_input_values

        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

        # (1) Replace any user provided convenience notations with values that match the following specs:
        # a - all dictionary values are lists containing and input value on each trial (even if only one trial)
        # b - each input value is a 2d array that matches variable
        # example: { Mech1: [Fully_specified_input_for_mech1_on_trial_1, Fully_specified_input_for_mech1_on_trial_2 … ],
        #            Mech2: [Fully_specified_input_for_mech2_on_trial_1, Fully_specified_input_for_mech2_on_trial_2 … ]}
        # (2) Verify that all mechanism values provide the same number of inputs (check length of each dictionary value)

        adjusted_stimuli = {}
        num_input_sets = -1

        for node, stim_list in stimuli.items():
            if isinstance(node, Composition):
                if isinstance(stim_list, dict):

                    adjusted_stimulus_dict, num_trials = node._adjust_stimulus_dict(stim_list)
                    translated_stimulus_dict = {}

                    # first time through the stimulus dictionary, assemble a dictionary in which the keys are input CIM
                    # InputStates and the values are lists containing the first input value
                    for nested_origin_node, values in adjusted_stimulus_dict.items():
                        first_value = values[0]
                        for i in range(len(first_value)):
                            input_state = nested_origin_node.external_input_states[i]
                            input_cim_input_state = node.input_CIM_states[input_state][0]
                            translated_stimulus_dict[input_cim_input_state] = [first_value[i]]
                            # then loop through the stimulus dictionary again for each remaining trial
                            for trial in range(1, num_trials):
                                translated_stimulus_dict[input_cim_input_state].append(values[trial][i])

                    adjusted_stimulus_list = []
                    for trial in range(num_trials):
                        trial_adjusted_stimulus_list = []
                        for state in node.external_input_states:
                            trial_adjusted_stimulus_list.append(translated_stimulus_dict[state][trial])
                        adjusted_stimulus_list.append(trial_adjusted_stimulus_list)
                    stimuli[node] = adjusted_stimulus_list

            # excludes any input states marked "internal_only" (usually recurrent)
            input_must_match = node.external_input_values

            if input_must_match == []:
                # all input states are internal_only
                continue

            check_spec_type = self._input_matches_variable(stim_list, input_must_match)
            # If a node provided a single input, wrap it in one more list in order to represent trials
            if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
                if check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input states specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[node] = [np.atleast_2d(stim_list)]
                else:
                    adjusted_stimuli[node] = [stim_list]

                # verify that all nodes have provided the same number of inputs
                if num_input_sets == -1:
                    num_input_sets = 1
                elif num_input_sets != 1:
                    raise RunError("Input specification for {} is not valid. The number of inputs (1) provided for {}"
                                   "conflicts with at least one other node's input specification.".format(self.name,
                                                                                                               node.name))
            else:
                adjusted_stimuli[node] = []
                for stim in stimuli[node]:
                    check_spec_type = self._input_matches_variable(stim, input_must_match)
                    # loop over each input to verify that it matches variable
                    if check_spec_type == False:
                        err_msg = "Input stimulus ({}) for {} is incompatible with its external_input_values ({}).".\
                            format(stim, node.name, input_must_match)
                        # 8/3/17 CW: I admit the error message implementation here is very hacky; but it's at least not a hack
                        # for "functionality" but rather a hack for user clarity
                        if "KWTA" in str(type(node)):
                            err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other values)" \
                                                " to represent the outside stimulus for the inhibition input state, and " \
                                                "for systems, put your inputs"
                        raise RunError(err_msg)
                    elif check_spec_type == "homogeneous":
                        # np.atleast_2d will catch any single-input states specified without an outer list
                        # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                        adjusted_stimuli[node].append(np.atleast_2d(stim))
                    else:
                        adjusted_stimuli[node].append(stim)

                # verify that all nodes have provided the same number of inputs
                if num_input_sets == -1:
                    num_input_sets = len(stimuli[node])
                elif num_input_sets != len(stimuli[node]):
                    raise RunError("Input specification for {} is not valid. The number of inputs ({}) provided for {}"
                                   "conflicts with at least one other node's input specification."
                                   .format(self.name, (stimuli[node]), node.name))

        return adjusted_stimuli, num_input_sets

    def _adjust_execution_stimuli(self, stimuli):
        adjusted_stimuli = {}
        for node, stimulus in stimuli.items():
            if isinstance(node, Composition):
                input_must_match = node.input_values
                if isinstance(stimulus, dict):
                    adjusted_stimulus_dict = node._adjust_stimulus_dict(stimulus)
                    adjusted_stimuli[node] = adjusted_stimulus_dict
                    continue
            else:
                input_must_match = node.instance_defaults.variable


            check_spec_type = self._input_matches_variable(stimulus, input_must_match)
            # If a node provided a single input, wrap it in one more list in order to represent trials
            if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
                if check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input states specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[node] = np.atleast_2d(stimulus)
                else:
                    adjusted_stimuli[node] = stimulus

            else:
                raise CompositionError("Input stimulus ({}) for {} is incompatible with its variable ({})."
                                       .format(stimulus, node.name, input_must_match))
        return adjusted_stimuli

    @property
    def input_states(self):
        """Returns all InputStates that belong to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_states

    @property
    def output_states(self):
        """Returns all OutputStates that belong to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_states

    @property
    def output_values(self):
        """Returns values of all OutputStates that belong to the Output CompositionInterfaceMechanism"""
        output_values = []
        for state in self.output_CIM.output_states:
            output_values.append(state.value)
        return output_values

    @property
    def input_state(self):
        """Returns the index 0 InputState that belongs to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_states[0]

    @property
    def input_values(self):
        """Returns values of all InputStates that belong to the Input CompositionInterfaceMechanism"""
        input_values = []
        for state in self.input_CIM.input_states:
            input_values.append(state.value)
        return input_values

    #  For now, external_input_states == input_states and external_input_values == input_values
    #  They could be different in the future depending on new features (ex. if we introduce recurrent compositions)
    #  Useful to have this property for treating Compositions the same as Mechanisms in run & execute
    @property
    def external_input_states(self):
        """Returns all external InputStates that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_state for input_state in self.input_CIM.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_values(self):
        """Returns values of all external InputStates that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_state.value for input_state in self.input_CIM.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_values(self):
        """Returns the default values of all external InputStates that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_state.instance_defaults.value for input_state in self.input_CIM.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None


    @property
    def output_state(self):
        """Returns the index 0 OutputState that belongs to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_states[0]

