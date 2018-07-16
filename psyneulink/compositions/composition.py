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
import enum
import logging
import numpy as np
import uuid

from psyneulink.components.component import function_type
from psyneulink.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.shellclasses import Mechanism, Projection
from psyneulink.components.states.outputstate import OutputState
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import HARD_CLAMP, IDENTITY_MATRIX, NO_CLAMP, PULSE_CLAMP, SOFT_CLAMP
from psyneulink.scheduling.condition import Always
from psyneulink.scheduling.scheduler import Scheduler
from psyneulink.scheduling.time import TimeScale

__all__ = [
    'Composition', 'CompositionError', 'MechanismRole',
]

logger = logging.getLogger(__name__)


class MechanismRole(enum.Enum):
    """

    - ORIGIN
        A `ProcessingMechanism <ProcessingMechanism>` that is the first Mechanism of a `Process` and/or `System`, and
        that receives the input to the Process or System when it is :ref:`executed or run <Run>`.  A Process may have
        only one `ORIGIN` Mechanism, but a System may have many.  Note that the `ORIGIN` Mechanism of a Process is not
        necessarily an `ORIGIN` of the System to which it belongs, as it may receive `Projections <Projection>` from
        other Processes in the System (see `example <LearningProjection_Target_vs_Terminal_Figure>`). The `ORIGIN`
        Mechanisms of a Process or System are listed in its :keyword:`origin_mechanisms` attribute, and can be displayed
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
        :keyword:`target_mechanisms` attribute, and can be displayed using its :keyword:`show` method.  For additional
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

    def __init__(self, component, parents=None, children=None):
        self.component = component
        if parents is not None:
            self.parents = parents
        else:
            self.parents = []
        if children is not None:
            self.children = children
        else:
            self.children = []

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
            g.add_vertex(Vertex(vertex.component))

        for i in range(len(self.vertices)):
            g.vertices[i].parents = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in self.vertices[i].parents]
            g.vertices[i].children = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in self.vertices[i].children]

        return g

    def add_component(self, component):
        if component in [vertex.component for vertex in self.vertices]:
            logger.info('Component {1} is already in graph {0}'.format(component, self))
        else:
            vertex = Vertex(component)
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

class Composition(object):
    '''
        Composition

        Arguments
        ---------

        Attributes
        ----------

        graph : `Graph`
            The full `Graph` associated with this Composition. Contains both `Mechanisms <Mechanism>` and `Projections
            <Projection>` used in processing or learning.

        mechanisms : `list[Mechanism]`
            A list of all `Mechanisms <Mechanism>` contained in this Composition

        COMMENT:
        name : str
            see `name <Composition_Name>`

        prefs : PreferenceSet
            see `prefs <Composition_Prefs>`
        COMMENT

    '''

    def __init__(self):
        # core attributes
        self.name = "Composition-TestName"
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.mechanisms = []
        self.input_CIM = CompositionInterfaceMechanism(name="Stimulus_CIM")
        self.input_CIM_output_states = {}
        self.output_CIM = CompositionInterfaceMechanism(name="Output_CIM")
        self.output_CIM_output_states = {}
        self.execution_ids = []

        self._scheduler_processing = None
        self._scheduler_learning = None

        # status attributes
        self.graph_consistent = True  # Tracks if the Composition is in a state that can be run (i.e. no dangling projections, (what else?))
        self.needs_update_graph = True   # Tracks if the Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True   # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler_processing = True  # Tracks if the processing scheduler needs to be regenerated
        self.needs_update_scheduler_learning = True  # Tracks if the learning scheduler needs to be regenerated (mechanisms/projections added/removed etc)

        # helper attributes
        self.mechanisms_to_roles = collections.OrderedDict()

        # TBI: update self.sched whenever something is added to the composition
        self.sched = Scheduler(composition=self)

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
            self._scheduler_learning = Scheduler(graph=self.graph)

            if old_scheduler is not None:
                self._scheduler_learning.add_condition_set(old_scheduler.condition_set)

            self.needs_update_scheduler_learning = False

        return self._scheduler_learning

    @property
    def termination_processing(self):
        return self.scheduler_processing.termination_conds

    @termination_processing.setter
    def termination_processing(self, termination_conds):
        self.scheduler_processing.termination_conds = termination_conds

    def _get_unique_id(self):
        return uuid.uuid4()

    def add_mechanism(self, mech):
        '''
            Adds a Mechanism to the Composition, if it is not already added

            Arguments
            ---------

            mech : Mechanism
                the Mechanism to add
        '''
        if mech not in [vertex.component for vertex in self.graph.vertices]:  # Only add if it doesn't already exist in graph
            mech.is_processing = True
            self.graph.add_component(mech)  # Set incoming edge list of mech to empty
            self.mechanisms.append(mech)
            self.mechanisms_to_roles[mech] = set()

            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler_processing = True
            self.needs_update_scheduler_learning = True

    def add_projection(self, sender, projection, receiver):
        '''
            Adds a projection to the Composition, if it is not already added

            Arguments
            ---------

            sender : Mechanism
                the sender of **projection**

            projection : Projection
                the projection to add

            receiver : Mechanism
                the receiver of **projection**
        '''
        if projection not in [vertex.component for vertex in self.graph.vertices]:
            projection.is_processing = False
            projection.name = '{0} to {1}'.format(sender, receiver)
            self.graph.add_component(projection)

            # Add connections between mechanisms and the projection
            self.graph.connect_components(sender, projection)
            self.graph.connect_components(projection, receiver)
            self._validate_projection(sender, projection, receiver)

            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler_processing = True
            self.needs_update_scheduler_learning = True

    def add_pathway(self, path):
        '''
            Adds an existing Pathway to the current Composition

            Arguments
            ---------

            path: the Pathway (Composition) to be added

        '''

        # identify mechanisms and projections
        mechanisms, projections = [], []
        for c in path.graph.vertices:
            if isinstance(c.component, Mechanism):
                mechanisms.append(c.component)
            elif isinstance(c.component, Projection):
                projections.append(c.component)

        # add all mechanisms first
        for m in mechanisms:
            self.add_mechanism(m)

        # then projections
        for p in projections:
            self.add_projection(p.sender.owner, p, p.receiver.owner)

        self._analyze_graph()

    def add_linear_processing_pathway(self, pathway):
        # First, verify that the pathway begins with a mechanism
        if isinstance(pathway[0], Mechanism):
            self.add_mechanism(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg
            raise CompositionError("The first item in a linear processing pathway must be a Mechanism.")
        # Then, add all of the remaining mechanisms in the pathway
        for c in range(1, len(pathway)):
            # if the current item is a mechanism, add it
            if isinstance(pathway[c], Mechanism):
                self.add_mechanism(pathway[c])

        # Then, loop through and validate that the mechanism-projection relationships make sense
        # and add MappingProjections where needed
        for c in range(1, len(pathway)):
            if isinstance(pathway[c], Mechanism):
                if isinstance(pathway[c - 1], Mechanism):
                    # if the previous item was also a mechanism, add a mapping projection between them
                    self.add_projection(
                        pathway[c - 1],
                        MappingProjection(
                            sender=pathway[c - 1],
                            receiver=pathway[c]
                        ),
                        pathway[c]
                    )
            # if the current item is a projection
            elif isinstance(pathway[c], Projection):
                if c == len(pathway) - 1:
                    raise CompositionError("{} is the last item in the pathway. A projection cannot be the last item in"
                                           " a linear processing pathway.".format(pathway[c]))
                # confirm that it is between two mechanisms, then add the projection
                if isinstance(pathway[c - 1], Mechanism) and isinstance(pathway[c + 1], Mechanism):
                    self.add_projection(pathway[c - 1], pathway[c], pathway[c + 1])
                else:
                    raise CompositionError(
                        "{} is not between two mechanisms. A Projection in a linear processing pathway must be preceded"
                        " by a Mechanism and followed by a Mechanism".format(pathway[c]))
            else:
                raise CompositionError("{} is not a Projection or Mechanism. A linear processing pathway must be made "
                                       "up of Projections and Mechanisms.".format(pathway[c]))

    def _validate_projection(self, sender, projection, receiver):

        if hasattr(projection, "sender") and hasattr(projection, "receiver"):
            # the sender and receiver were passed directly to the Projection object AND to compositions'
            # add_projection() method -- confirm that these are consistent

            if projection.sender.owner != sender:
                raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                       "Components in their Composition.".format(projection, sender))

            if projection.receiver.owner != receiver:
                raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                       "Components in their Composition.".format(projection, receiver))
        else:
            # sender and receiver were NOT passed directly to the Projection object
            # assign them based on the sender and receiver passed into add_projection()
            projection.init_args['sender'] = sender
            projection.init_args['receiver'] = receiver
            projection.context.initialization_status = ContextFlags.DEFERRED_INIT
            projection._deferred_init(context=" INITIALIZING ")

        if projection.sender.owner != sender:
            raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                   "Components in the Composition.".format(projection, sender))
        if projection.receiver.owner != receiver:
            raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                   "Components in the Composition.".format(projection, receiver))

    def _analyze_graph(self, graph=None):
        ########
        # Determines identity of significant nodes of the graph
        # Each node falls into one or more of the following categories
        # - Origin: Origin mechanisms are those which do not receive any projections.
        # - Terminal: Terminal mechanisms provide the output of the composition. By
        #   default, those which do not send any projections, but they may also be
        #   specified explicitly.
        # - Recurrent_init: Recurrent_init mechanisms send projections that close recurrent
        #   loops in the composition (or projections that are explicitly specified as
        #   recurrent). They need an initial value so that their receiving mechanisms
        #   have input.
        # - Cycle: Cycle mechanisms receive projections from Recurrent_init mechanisms. They
        #   can be viewd as the starting points of recurrent loops.
        # The following categories can be explicitly set by the user in which case their
        # values are not changed based on the graph analysis. Additional mechanisms may
        # be automatically added besides those specified by the user.
        # - Input: Input mechanisms accept inputs from the input_dict of the composition.
        #   All Origin mechanisms are added to this category automatically.
        # - Output: Output mechanisms provide their values as outputs of the composition.
        #   All Terminal mechanisms are added to this category automatically.
        # - Target: Target mechanisms receive target values for the composition to be
        #   used by learning and control. They are usually Comparator mechanisms that
        #   compare the target value to the output of another mechanism in the composition.
        # - Monitored: Monitored mechanisms send projections to Target mechanisms.
        ########
        if graph is None:
            graph = self.graph_processing

        # Clear old information
        self.mechanisms_to_roles.update({k: set() for k in self.mechanisms_to_roles})

        # Identify Origin mechanisms
        for mech in self.mechanisms:
            if graph.get_parents_from_component(mech) == []:
                self._add_mechanism_role(mech, MechanismRole.ORIGIN)
        # Identify Terminal mechanisms
            if graph.get_children_from_component(mech) == []:
                self._add_mechanism_role(mech, MechanismRole.TERMINAL)
        # Identify Recurrent_init and Cycle mechanisms
        visited = []  # Keep track of all mechanisms that have been visited
        for origin_mech in self.get_mechanisms_by_role(MechanismRole.ORIGIN):  # Cycle through origin mechanisms first
            visited_current_path = []  # Track all mechanisms visited from the current origin
            next_visit_stack = []  # Keep a stack of mechanisms to be visited next
            next_visit_stack.append(origin_mech)
            for mech in next_visit_stack:  # While the stack isn't empty
                visited.append(mech)  # Mark the mech as visited
                visited_current_path.append(mech)  # And visited during the current path
                children = [vertex.component for vertex in graph.get_children_from_component(mech)]  # Get the children of that mechanism
                for child in children:
                    # If the child has been visited this path and is not already initialized
                    if child in visited_current_path:
                        self._add_mechanism_role(mech, MechanismRole.RECURRENT_INIT)
                        self._add_mechanism_role(child, MechanismRole.CYCLE)
                    elif child not in visited:  # Else if the child has not been explored
                        next_visit_stack.append(child)  # Add it to the visit stack
        for mech in self.mechanisms:
            if mech not in visited:  # Check the rest of the mechanisms
                visited_current_path = []
                next_visit_stack = []
                next_visit_stack.append(mech)
                for remaining_mech in next_visit_stack:
                    visited.append(remaining_mech)
                    visited_current_path.append(remaining_mech)
                    children = [vertex.component for vertex in graph.get_children_from_component(remaining_mech)]
                    for child in children:
                        if child in visited_current_path:
                            self._add_mechanism_role(remaining_mech, MechanismRole.RECURRENT_INIT)
                            self._add_mechanism_role(child, MechanismRole.CYCLE)
                        elif child not in visited:
                            next_visit_stack.append(child)

        self._create_CIM_output_states()

        self.needs_update_graph = False

    def _update_processing_graph(self):
        '''
        Constructs the processing graph (the graph that contains only non-learning mechanisms as vertices)
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
                            self._graph_processing.connect_vertices(parent, child)

                    for node in cur_vertex.parents + cur_vertex.children:
                        logger.debug('New parents for vertex {0}: \n\t{1}\nchildren: \n\t{2}'.format(node, node.parents, node.children))
                    logger.debug('Removing vertex {0}'.format(cur_vertex))
                    self._graph_processing.remove_vertex(cur_vertex)

                visited_vertices.add(cur_vertex)
                # add to next_vertices (frontier) any parents and children of cur_vertex that have not been visited yet
                next_vertices.extend([vertex for vertex in cur_vertex.parents + cur_vertex.children if vertex not in visited_vertices])

        self.needs_update_graph_processing = False

    def get_mechanisms_by_role(self, role):
        '''
            Returns a set of mechanisms in this Composition that have the role `role`

            Arguments
            _________

            role : MechanismRole
                the set of mechanisms having this role to return

            Returns
            -------

            set of Mechanisms with `MechanismRole` `role` : set(`Mechanism <Mechanism>`)
        '''
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        try:
            return [mech for mech in self.mechanisms if role in self.mechanisms_to_roles[mech]]
        except KeyError as e:
            raise CompositionError('Mechanism not assigned to role in mechanisms_to_roles: {0}'.format(e))

    def get_roles_by_mechanism(self, mechanism):
        try:
            return self.mechanisms_to_roles[mechanism]
        except KeyError:
            raise CompositionError('Mechanism {0} not found in {1}.mechanisms_to_roles'.format(mechanism, self))

    def _set_mechanism_roles(self, mech, roles):
        self.clear_mechanism_role(mech)
        for role in roles:
            self._add_mechanism_role(role)

    def _clear_mechanism_roles(self, mech):
        if mech in self.mechanisms_to_roles:
            self.mechanisms_to_roles[mech] = set()

    def _add_mechanism_role(self, mech, role):
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        self.mechanisms_to_roles[mech].add(role)

    def _remove_mechanism_role(self, mech, role):
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        self.mechanisms_to_roles[mech].remove(role)

    # mech_type specifies a type of mechanism, mech_type_list contains all of the mechanisms of that type
    # feed_dict is a dictionary of the input states of each mechanism of the specified type
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

    def _create_CIM_output_states(self):
        '''
            builds a dictionary of { Mechanism : OutputState } pairs where each origin mechanism has at least one
            corresponding OutputState on the CompositionInterfaceMechanism
        '''
        # FIX BUG: stimulus CIM output states are not properly destroyed when analyze graph is run multiple times
        # (extra mechanisms are marked as CIMs when graph is analyzed too early, so they create CIM output states)

        #  INPUT CIMS
        # loop over all origin mechanisms
        current_origin_input_states = set()
        for mech in self.get_mechanisms_by_role(MechanismRole.ORIGIN):

            for input_state in mech.input_states:
                # add it to our set of current input states
                current_origin_input_states.add(input_state)

                # if there is not a corresponding CIM output state, add one
                if input_state not in set(self.input_CIM_output_states.keys()):
                    interface_output_state = OutputState(owner=self.input_CIM,
                                                         variable=input_state.value,
                                                         reference_value= input_state.value,
                                                         name="STIMULUS_CIM_" + mech.name + "_" + input_state.name)
                    # self.input_CIM.add_states(interface_output_state)
                    self.input_CIM_output_states[input_state] = interface_output_state
                    MappingProjection(sender=interface_output_state,
                                      receiver=input_state,
                                      matrix= IDENTITY_MATRIX,
                                      name="("+interface_output_state.name + ") to ("
                                           + input_state.owner.name + "-" + input_state.name+")")

        sends_to_input_states = set(self.input_CIM_output_states.keys())
        # For any output state still registered on the CIM that does not map to a corresponding ORIGIN mech I.S.:
        for input_state in sends_to_input_states.difference(current_origin_input_states):
            for projection in input_state.path_afferents:
                if projection.sender == self.input_CIM_output_states[input_state]:
                    # remove the corresponding projection from the ORIGIN mechanism's path afferents
                    input_state.path_afferents.remove(projection)
                    projection = None

            # remove the output state associated with this input state (this iteration) from the CIM output states
            self.input_CIM.output_states.remove(self.input_CIM_output_states[input_state])

            # and from the dictionary of CIM output state/input state pairs
            del self.input_CIM_output_states[input_state]

        # OUTPUT CIMS
        # loop over all terminal mechanisms
        current_terminal_output_states = set()
        for mech in self.get_mechanisms_by_role(MechanismRole.TERMINAL):
            for output_state in mech.output_states:
                current_terminal_output_states.add(output_state)
                # if there is not a corresponding CIM output state, add one
                if output_state not in set(self.output_CIM_output_states.keys()):
                    interface_output_state = OutputState(owner=self.output_CIM,
                                                         variable=output_state.value,
                                                         reference_value=output_state.value,
                                                         name="OUTPUT_CIM_" + mech.name + "_" + output_state.name)

                    self.output_CIM_output_states[output_state] = interface_output_state
                    # MappingProjection(sender=interface_output_state,
                    #                   receiver=output_state,
                    #                   matrix= IDENTITY_MATRIX,
                    #                   name="("+interface_output_state.name + ") to ("
                    #                        + output_state.owner.name + "-" + output_state.name+")")

        previous_terminal_output_states = set(self.output_CIM_output_states.keys())
        for output_state in previous_terminal_output_states.difference(current_terminal_output_states):
            self.output_CIM.output_states.remove(self.output_CIM_output_states[output_state])
            del self.output_CIM_output_states[output_state]

    def _assign_values_to_CIM_output_states(self, inputs):
        current_mechanisms = set()
        for key in inputs:
            if isinstance(inputs[key], (float, int)):
                inputs[key] = np.atleast_2d(inputs[key])
            for i in range(len(inputs[key])):
                self.input_CIM_output_states[key.input_states[i]].value = inputs[key][i]
            current_mechanisms.add(key)

        origins = self.get_mechanisms_by_role(MechanismRole.ORIGIN)

        # NOTE: This may need to change from default_variable to wherever a default value of the mechanism's variable
        # is stored -- the point is that if an input is not supplied for an origin mechanism, the mechanism should use
        # its default variable value
        for mech in set(origins).difference(set(current_mechanisms)):
            self.input_CIM_output_states[mech.input_state].value = mech.instance_defaults.value


    def _assign_execution_ids(self, execution_id=None):
        '''
            assigns the same uuid to each Mechanism in the composition's processing graph as well as all input
            mechanisms for this composition. The uuid is either specified in the user's call to run(), or generated
            randomly at run time.
        '''

        # Traverse processing graph and assign one uuid to all of its mechanisms
        if execution_id is None:
            execution_id = self._get_unique_id()

        if execution_id not in self.execution_ids:
            self.execution_ids.append(execution_id)

        for v in self._graph_processing.vertices:
            v.component._execution_id = execution_id

        # Assign the uuid to all input mechanisms
        # for k in self.input_mechanisms.keys():
        #     self.input_mechanisms[k]._execution_id = execution_id

        self.input_CIM._execution_id = execution_id
        # self.target_CIM._execution_id = execution_id

        self._execution_id = execution_id
        return execution_id

    def _identify_clamp_inputs(self, list_type, input_type, origins):
        # clamp type of this list is same as the one the user set for the whole composition; return all mechanisms
        if list_type == input_type:
            return origins
        # the user specified different types of clamps for each origin mechanism; generate a list accordingly
        elif isinstance(input_type, dict):
            return [k for k, v in input_type.items() if list_type == v]
        # clamp type of this list is NOT same as the one the user set for the whole composition; return empty list
        else:
            return []

    def _parse_runtime_params(self, runtime_params):
        if runtime_params is None:
            return {}
        for mechanism in runtime_params:
            for param in runtime_params[mechanism]:
                if isinstance(runtime_params[mechanism][param], tuple):
                    if len(runtime_params[mechanism][param]) == 1:
                        runtime_params[mechanism][param] = (runtime_params[mechanism][param], Always())
                    elif len(runtime_params[mechanism][param]) != 2:
                        raise SystemError("Invalid runtime parameter specification ({}) for {}'s {} parameter in {}. "
                                          "Must be a tuple of the form (parameter value, condition), or simply the "
                                          "parameter value. ".format(runtime_params[mechanism][param],
                                                                     mechanism.name,
                                                                     param,
                                                                     self.name))
                else:
                    runtime_params[mechanism][param] = (runtime_params[mechanism][param], Always())
        return runtime_params


    def execute(
        self,
        inputs,
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
    ):
        '''
            Passes inputs to any Mechanisms receiving inputs directly from the user, then coordinates with the Scheduler
            to receive and execute sets of Mechanisms that are eligible to run until termination conditions are met.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` : list }
                a dictionary containing a key-value pair for each Mechanism in the composition that receives inputs from
                the user. For each pair, the key is the Mechanism and the value is a list of inputs.

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

        runtime_params = self._parse_runtime_params(runtime_params)

        if targets is None:
            targets = {}
        execution_id = self._assign_execution_ids(execution_id)
        origin_mechanisms = self.get_mechanisms_by_role(MechanismRole.ORIGIN)

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        if scheduler_learning is None:
            scheduler_learning = self.scheduler_learning

        if termination_processing is None:
            termination_processing = self.termination_processing

        self._assign_values_to_CIM_output_states(inputs)
        # self._assign_values_to_target_CIM_output_states(targets)
        execution_id = self._assign_execution_ids(execution_id)
        next_pass_before = 1
        next_pass_after = 1
        if clamp_input:
            soft_clamp_inputs = self._identify_clamp_inputs(SOFT_CLAMP, clamp_input, origin_mechanisms)
            hard_clamp_inputs = self._identify_clamp_inputs(HARD_CLAMP, clamp_input, origin_mechanisms)
            pulse_clamp_inputs = self._identify_clamp_inputs(PULSE_CLAMP, clamp_input, origin_mechanisms)
            no_clamp_inputs = self._identify_clamp_inputs(NO_CLAMP, clamp_input, origin_mechanisms)
        # run scheduler to receive sets of mechanisms that may be executed at this time step in any order
        execution_scheduler = scheduler_processing
        execution_scheduler._init_counts(execution_id=execution_id)
        num = None

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
            # execute each mechanism with EXECUTING in context
            for mechanism in next_execution_set:

                if mechanism in origin_mechanisms:
                    # KAM 8/28 commenting out the below code because it's not necessarily how we want to handle
                    # a recurrent projection on the first time step (meaning, before its mechanism has executed)
                    # FIX: determine the correct behavior for this case & document it

                    # if (
                    #     scheduler_processing.times[execution_id][TimeScale.TRIAL][TimeScale.TIME_STEP] == 0
                    #     and hasattr(mechanism, "recurrent_projection")
                    # ):
                    #     mechanism.recurrent_projection.sender.value = [0.0]
                    if clamp_input:
                        if mechanism in hard_clamp_inputs:
                            # clamp = HARD_CLAMP --> "turn off" recurrent projection
                            if hasattr(mechanism, "recurrent_projection"):
                                mechanism.recurrent_projection.sender.value = [0.0]
                        elif mechanism in no_clamp_inputs:
                            for input_state in mechanism.input_states:
                                self.input_CIM_output_states[input_state].value = 0.0
                            # self.input_mechanisms[mechanism]._output_states[0].value = 0.0

                if isinstance(mechanism, Mechanism):

                    execution_runtime_params = {}

                    if mechanism in runtime_params:
                        for param in runtime_params[mechanism]:
                            if runtime_params[mechanism][param][1].is_satisfied(scheduler=execution_scheduler,
                                               # KAM 5/15/18 - not sure if this will always be the correct execution id:
                                                                                execution_id=self._execution_id):
                                execution_runtime_params[param] = runtime_params[mechanism][param][0]

                    mechanism.context.execution_phase = ContextFlags.PROCESSING
                    num = mechanism.execute(runtime_params=execution_runtime_params,
                                            context=ContextFlags.COMPOSITION)

                    for key in mechanism._runtime_params_reset:
                        mechanism._set_parameter_value(key, mechanism._runtime_params_reset[key])
                    mechanism._runtime_params_reset = {}

                    for key in mechanism.function_object._runtime_params_reset:
                        mechanism.function_object._set_parameter_value(key,
                                                                       mechanism.function_object._runtime_params_reset[
                                                                           key])
                    mechanism.function_object._runtime_params_reset = {}
                    mechanism.context.execution_phase = ContextFlags.IDLE

                if mechanism in origin_mechanisms:
                    if clamp_input:
                        if mechanism in pulse_clamp_inputs:
                            for input_state in mechanism.input_states:
                            # clamp = None --> "turn off" input mechanism
                            # self.input_mechanisms[mechanism]._output_states[0].value = 0
                                self.input_CIM_output_states[input_state].value = 0

            if call_after_time_step:
                call_after_time_step()

        if call_after_pass:
            call_after_pass()

        return num

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
        runtime_params=None
    ):
        '''
            Passes inputs to any mechanisms receiving inputs directly from the user, then coordinates with the scheduler
            to receive and execute sets of mechanisms that are eligible to run until termination conditions are met.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` : list }
                a dictionary containing a key-value pair for each Mechanism in the composition that receives inputs from
                the user. For each pair, the key is the Mechanism and the value is a list of inputs. Each input in the
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

            runtime_params : Dict[Mechanism: Dict[Param: Tuple(Value, Condition)]]
                nested dictionary of (value, `Condition`) tuples for parameters of Mechanisms of the Composition; specifies
                alternate parameter values to be used only during this `Run` when the specified `Condition` is met.

                Outer dictionary:
                    - *key* - Mechanism
                    - *value* - Runtime Parameter Specification Dictionary

                Runtime Parameter Specification Dictionary:
                    - *key* - keyword corresponding to a parameter of the Mechanism
                    - *value* - tuple in which the index 0 item is the runtime parameter value, and the index 1 item is a
                      `Condition`

                See `Run_Runtime_Parameters` for more details and examples of valid dictionaries.

            Returns
            ---------

            output value of the final Mechanism executed in the composition : various
        '''

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        # TBI: Learning
        if scheduler_learning is None:
            scheduler_learning = self.scheduler_learning

        if termination_processing is None:
            termination_processing = self.termination_processing

        self._analyze_graph()

        execution_id = self._assign_execution_ids(execution_id)

        scheduler_processing._init_counts(execution_id=execution_id)
        scheduler_learning._init_counts(execution_id=execution_id)

        scheduler_processing.update_termination_conditions(termination_processing)
        scheduler_learning.update_termination_conditions(termination_learning)

        origin_mechanisms = self.get_mechanisms_by_role(MechanismRole.ORIGIN)

        # if there is only one origin mechanism, allow inputs to be specified in a list
        if isinstance(inputs, (list, np.ndarray)):
            if len(origin_mechanisms) == 1:
                inputs = {next(iter(origin_mechanisms)): inputs}
            else:
                raise CompositionError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                               "mechanisms.".format(self.name, len(origin_mechanisms)))
        elif not isinstance(inputs, dict):
            if len(origin_mechanisms) == 1:
                raise CompositionError(
                    "Inputs to {} must be specified in a list or in a dictionary with the origin mechanism({}) "
                    "as its only key".format(self.name, next(iter(origin_mechanisms)).name))
            else:
                raise CompositionError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                               "mechanisms.".format(self.name, len(origin_mechanisms)))

        inputs, num_inputs_sets = self._adjust_stimulus_dict(inputs)

        if num_trials is not None:
            num_trials = num_trials
        else:
            num_trials = num_inputs_sets

        if targets is None:
            targets = {}

        scheduler_processing._reset_counts_total(TimeScale.RUN, execution_id)

        # TBI: Handle runtime params?
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
            for mech in inputs:
                execution_stimuli[mech] = inputs[mech][stimulus_index]
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
                    for mech in targets:
                        if callable(targets[mech]):
                            execution_targets[mech] = targets[mech]
                        else:
                            execution_targets[mech] = targets[mech][target_index]

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
        terminal_mechanisms = self.get_mechanisms_by_role(MechanismRole.TERMINAL)

        for terminal_mechanism in terminal_mechanisms:
            for terminal_output_state in terminal_mechanism.output_states:
                CIM_output_state = self.output_CIM_output_states[terminal_output_state]
                CIM_output_state.value = terminal_output_state.value

        # return the output of the LAST mechanism executed in the composition
        return result

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

        # STEP 1: validate that there is a one-to-one mapping of input entries to origin mechanisms


        # Check that all of the mechanisms listed in the inputs dict are ORIGIN mechanisms in the self
        origin_mechanisms = self.get_mechanisms_by_role(MechanismRole.ORIGIN)
        for mech in stimuli.keys():
            if not mech in origin_mechanisms:
                raise CompositionError("{} in inputs dict for {} is not one of its ORIGIN mechanisms".
                               format(mech.name, self.name))
        # Check that all of the ORIGIN mechanisms in the self are represented by entries in the inputs dict
        for mech in origin_mechanisms:
            if not mech in stimuli:
                raise RunError("Entry for ORIGIN Mechanism {} is missing from the inputs dict for {}".
                               format(mech.name, self.name))

        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

        # (1) Replace any user provided convenience notations with values that match the following specs:
        # a - all dictionary values are lists containing and input value on each trial (even if only one trial)
        # b - each input value is a 2d array that matches variable
        # example: { Mech1: [Fully_specified_input_for_mech1_on_trial_1, Fully_specified_input_for_mech1_on_trial_2 … ],
        #            Mech2: [Fully_specified_input_for_mech2_on_trial_1, Fully_specified_input_for_mech2_on_trial_2 … ]}
        # (2) Verify that all mechanism values provide the same number of inputs (check length of each dictionary value)

        adjusted_stimuli = {}
        num_input_sets = -1

        for mech, stim_list in stimuli.items():

            check_spec_type = self._input_matches_variable(stim_list, mech.instance_defaults.value)
            # If a mechanism provided a single input, wrap it in one more list in order to represent trials
            if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
                if check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input states specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[mech] = [np.atleast_2d(stim_list)]
                else:
                    adjusted_stimuli[mech] = [stim_list]

                # verify that all mechanisms have provided the same number of inputs
                if num_input_sets == -1:
                    num_input_sets = 1
                elif num_input_sets != 1:
                    raise RunError("Input specification for {} is not valid. The number of inputs (1) provided for {}"
                                   "conflicts with at least one other mechanism's input specification.".format(self.name,
                                                                                                               mech.name))
            else:
                adjusted_stimuli[mech] = []
                for stim in stimuli[mech]:
                    check_spec_type = self._input_matches_variable(stim, mech.instance_defaults.value)
                    # loop over each input to verify that it matches variable
                    if check_spec_type == False:
                        err_msg = "Input stimulus ({}) for {} is incompatible with its variable ({}).".\
                            format(stim, mech.name, mech.instance_defaults.value)
                        # 8/3/17 CW: I admit the error message implementation here is very hacky; but it's at least not a hack
                        # for "functionality" but rather a hack for user clarity
                        if "KWTA" in str(type(mech)):
                            err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other values)" \
                                                " to represent the outside stimulus for the inhibition input state, and " \
                                                "for systems, put your inputs"
                        raise RunError(err_msg)
                    elif check_spec_type == "homogeneous":
                        # np.atleast_2d will catch any single-input states specified without an outer list
                        # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                        adjusted_stimuli[mech].append(np.atleast_2d(stim))
                    else:
                        adjusted_stimuli[mech].append(stim)

                # verify that all mechanisms have provided the same number of inputs
                if num_input_sets == -1:
                    num_input_sets = len(stimuli[mech])
                elif num_input_sets != len(stimuli[mech]):
                    raise RunError("Input specification for {} is not valid. The number of inputs ({}) provided for {}"
                                   "conflicts with at least one other mechanism's input specification."
                                   .format(self.name, (stimuli[mech]), mech.name))

        return adjusted_stimuli, num_input_sets
