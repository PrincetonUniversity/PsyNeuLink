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



"""

import logging
import uuid

from collections import Iterable, OrderedDict
from enum import Enum
import uuid
import numpy as np

from PsyNeuLink.Components.Functions.Function import Linear
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.Projection import Projection
from PsyNeuLink.Globals.Keywords import EXECUTING, RESULT
from PsyNeuLink.Globals.TimeScale import TimeScale
from PsyNeuLink.Scheduling.Scheduler import Scheduler


logger = logging.getLogger(__name__)


class MechanismRole(Enum):
    """

    - ORIGIN
        A `ProcessingMechanism <ProcessingMechanism>` that is the first Mechanism of a `Process` and/or `System`,
        and that receives the input to the Process or System when it is :ref:`executed or run <Run>`.  A Process may
        have only one `ORIGIN` Mechanism, but a System may have many.  Note that the `ORIGIN`
        Mechanism of a Process is not necessarily an `ORIGIN` of the System to which it belongs, as it may receiver
        `Projections <Projection>` from other Processes in the System. The `ORIGIN` Mechanisms of a Process or
        System are listed in its :keyword:`origin_mechanisms` attribute, and can be displayed using its :keyword:`show`
        method.  For additional details about `ORIGIN` Mechanisms in Processes, see
        `Process Mechanisms <Process_Mechanisms>` and `Process Input and Output <Process_Input_And_Output>`;
        and for Systems see `System Mechanisms <System_Mechanisms>` and
        `System Input and Initialization <System_Execution_Input_And_Initialization>`.

    - INTERNAL
        A `ProcessingMechanism <ProcessingMechanism>` that is not designated as having any other status.

    - CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is *not* an `ORIGIN` Mechanism, and receives a `Projection`
        that closes a recurrent loop in a `Process` and/or `System`.  If it is an `ORIGIN` Mechanism, then it is simply
        designated as such (since it will be assigned input and therefore be initialized in any event).

    - INITIALIZE_CYCLE
        A `ProcessingMechanism <ProcessingMechanism>` that is the `sender <Projection.Projection.sender>` of a
        `Projection` that closes a loop in a `Process` or `System`, and that is not an `ORIGIN` Mechanism (since in
        that case it will be initialized in any event). An `initial value  <Run_InitialValues>` can be assigned to such
        Mechanisms, that will be used to initialize the Process or System when it is first run.  For additional
        information, see `Run <Run_Initial_Values>`, `System Mechanisms <System_Mechanisms>` and
        `System Input and Initialization <System_Execution_Input_And_Initialization>`.

    - TERMINAL
        A `ProcessingMechanism <ProcessingMechanism>` that is the last Mechanism of a `Process` and/or `System`, and
        that provides the output to the Process or System when it is `executed or run <Run>`.  A Process may
        have only one `TERMINAL` mechanism, but a system may have many.  Note that the `TERMINAL`
        mechanism of a process is not necessarily a `TERMINAL` mechanism of the system to which it belongs,
        as it may send projections to other processes in the system.  The `TERMINAL` mechanisms of a process
        or system are listed in its :keyword:`terminalMechanisms` attribute, and can be displayed using its
        :keyword:`show` method.  For additional details about `TERMINAL` mechanisms in processes, see
        `Process_Mechanisms` and `Process_Input_And_Output`; and for systems see `System_Mechanisms`.

    - SINGLETON
        A `ProcessingMechanism` that is the only Mechanism in a `Process` and/or `System`.  It can serve the
        functions of an `ORIGIN` and/or a `TERMINAL` Mechanism.

    - MONITORED

    - LEARNING
        A `LearningMechanism <LearningMechanism>` in a `Process` and/or `System`.

    - TARGET
        A `ComparatorMechanism` of a `Process` and/or `System` configured for learning that receives a target value
        from its `execute <ComparatorMechanism.ComparatorMechanism.execute>` or
        `run <ComparatorMechanism.ComparatorMechanism.execute>` method.  It must be associated with the `TERMINAL`
        Mechanism of the Process or System. The `TARGET` Mechanisms of a Process or System are listed in its
        :keyword:`target_mechanisms` attribute, and can be displayed using its :keyword:`show` method.  For additional
        details, see `TARGET mechanisms <LearningProjection_Targets>` and specifying `target values <Run_Targets>`.

    - RECURRENT_INIT


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


class Vertex(object):
    '''
        Stores a Component for use with a `Graph`

        Arguments
        ---------

        component : Component
            the `Component` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`

        Attributes
        ----------

        component : Component
            the `Component` represented by this Vertex

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
        A Graph of vertices and edges

        Attributes
        ----------

        comp_to_vertex : dict{Component : Vertex}
            maps `Component`\ s in the graph to the `Vertices <Vertex>` that represent them

        vertices : list[Vertex]
            the `Vertices <Vertex>` contained in this Graph

    '''

    def __init__(self):
        self.comp_to_vertex = OrderedDict()  # Translate from mechanisms to related vertex
        self.vertices = []  # List of vertices within graph

    def copy(self):
        '''
            Returns
            -------

            A copy of the Graph. `Vertices <Vertex>` are distinct from their originals, and point to the same `Component` object
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
            Returns
            -------

            A list[Vertex] of the parent `Vertices <Vertex>` of the Vertex associated with **component**
        '''
        return self.comp_to_vertex[component].parents

    def get_children_from_component(self, component):
        '''
            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component**
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
            The full `Graph` associated with this Composition. Contains both `Mechanism`\ s and `Projection`\ s used in processing \
            or learning.

        graph_processing : `Graph`
            The `Graph` that contains only `Mechanisms`, excluding those used in learning

        mechanisms : `list[Mechanism]`
            A list of all `Mechanism`\ s contained in this Composition

        scheduler_processing : Scheduler
            The default `Scheduler` automatically generated by the Composition, used for the processing `phase of execution <execution>`

        scheduler_learning : Scheduler
            The default `Scheduler` automatically generated by the Composition, used for the learning `phase of execution <execution>`
    '''

    def __init__(self):
        # core attributes
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.mechanisms = []
        self.input_mechanisms = {}

        self._scheduler_processing = None
        self._scheduler_learning = None

        # status attributes
        self.graph_consistent = True  # Tracks if the Composition is in a state that can be run (i.e. no dangling projections, (what else?))
        self.needs_update_graph = True   # Tracks if the Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True   # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler_processing = True  # Tracks if the processing scheduler needs to be regenerated
        self.needs_update_scheduler_learning = True  # Tracks if the learning scheduler needs to be regenerated (mechanisms/projections added/removed etc)

        # helper attributes
        self.mechanisms_to_roles = OrderedDict()

        # Create lists to track identity of certain mechanism classes within the
        # composition.
        # Explicit classes:
        self.explicit_input_mechanisms = []  # Need to track to know which to leave untouched
        self.all_input_mechanisms = []
        self.explicit_output_mechanisms = []  # Need to track to know which to leave untouched
        self.all_output_mechanisms = []
        self.target_mechanisms = []  # Do not need to track explicit as they mush be explicit

        # TBI: update self.sched whenever something is added to the composition
        self.sched = Scheduler(composition=self)

    @property
    def graph_processing(self):
        '''
            Returns the Composition's processing graph. Builds the graph if it needs updating
            since the last access.
        '''
        if self.needs_update_graph_processing or self._graph_processing is None:
            self._update_processing_graph()

        return self._graph_processing

    @property
    def scheduler_processing(self):
        '''
            Returns the Composition's processing scheduler. Builds the scheduler if it needs updating
            since the last access.
        '''
        if self.needs_update_scheduler_processing or self._scheduler_processing is None:
            self._scheduler_processing = Scheduler(graph=self.graph_processing)
            self.needs_update_scheduler_processing = False

        return self._scheduler_processing

    @property
    def scheduler_learning(self):
        '''
            Returns the Composition's learning scheduler. Builds the scheduler if it needs updating
            since the last access.
        '''
        if self.needs_update_scheduler_learning or self._scheduler_learning is None:
            self._scheduler_learning = Scheduler(graph=self.graph)
            self.needs_update_scheduler_learning = False

        return self._scheduler_learning

    def _get_unique_id(self):
        return uuid.uuid4()

    def add_mechanism(self, mech):
        '''
            Adds a mechanism to the Composition, if it is not already added

            Arguments
            ---------

            mech : Mechanism
                the mechanism to add
        '''
        if mech not in [vertex.component for vertex in self.graph.vertices]:  # Only add if it doesn't already exist in graph
            mech.is_processing = True
            self.graph.add_component(mech)  # Set incoming edge list of mech to empty
            self.mechanisms.append(mech)
            self.mechanisms_to_roles[mech] = set()

            self.needs_update_graph = True
            self.needs_update_graph_processing = True

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

    def add_linear_processing_pathway(self, pathway):
        # First, verify that the pathway begins with a mechanism
        if isinstance(pathway[0], Mechanism):
            self.add_mechanism(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg 
            raise CompositionError("The first item in a linear processing pathway must be a "
                                   "mechanism.")
        # Then, add all of the remaining mechanisms in the pathway
        for c in range(1, len(pathway)):
            # if the current item is a mechanism, add it
            if isinstance(pathway[c], Mechanism):
                self.add_mechanism(pathway[c])

        # Then, loop through and validate that the mechanism-projection relationships make sense
        # and add MappingProjections where needed
        for c in range(1, len(pathway)):
            if isinstance(pathway[c], Mechanism):
                if isinstance(pathway[c-1], Mechanism):
                    # if the previous item was also a mechanism, add a mapping projection between them
                    self.add_projection(pathway[c-1],
                                        MappingProjection(sender=pathway[c-1],
                                                          receiver=pathway[c]),
                                        pathway[c])
            # if the current item is a projection
            elif isinstance(pathway[c], Projection):
                if c == len(pathway) - 1:
                    raise CompositionError("{} is the last item in the pathway. A projection cannot be the last item in"
                                           " a linear processing pathway.".format(pathway[c]))
                # confirm that it is between two mechanisms, then add the projection
                if isinstance(pathway[c - 1], Mechanism) and isinstance(pathway[c+1], Mechanism):
                    self.add_projection(pathway[c-1], pathway[c], pathway[c+1])
                else:
                    raise CompositionError(
                        "{} is not between two mechanisms. A projection in a linear processing pathway must be preceded"
                        " by a mechanism and followed by a mechanism".format(pathway[c]))
            else:
                raise CompositionError("{} is not a projection or mechanism. A linear processing pathway must be made "
                                       "up of projections and mechanisms.".format(pathway[c]))



    def _validate_projection(self, sender, projection, receiver):

        if hasattr(projection, "sender") and hasattr(projection, "receiver"):
            # the sender and receiver were passed directly to the Projection object AND to compositions'
            # add_projection() method -- confirm that these are consistent

            if projection.sender.owner != sender:
                raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                       "components in their composition.".format(projection, sender))

            if projection.receiver.owner != receiver:
                raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                       "components in their composition.".format(projection, receiver))
        else:
            # sender and receiver were NOT passed directly to the Projection object
            # assign them based on the sender and receiver passed into add_projection()
            projection.init_args['sender'] = sender
            projection.init_args['receiver'] = receiver
            projection._deferred_init(context=" INITIALIZING ")

        if projection.sender.owner != sender:
            raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                   "components in the composition.".format(projection, sender))
        if projection.receiver.owner != receiver:
            raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                   "components in the composition.".format(projection, receiver))

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

            set of Mechanisms with `MechanismRole` `role`
        '''
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        try:
            return set([mech for mech in self.mechanisms if role in self.mechanisms_to_roles[mech]])
        except KeyError as e:
            raise CompositionError('Mechanism not assigned to role in mechanisms_to_roles: {0}'.format(e))

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
                raise ValueError("The mechanism \"{}\" is not {} {} of the composition".format(mech.name, article, mech_type))
            for i, timestep in enumerate(feed_dict[mech]):  # If mechanism is correct type, iterate over timesteps
                # Check if there are multiple input states specified
                try:
                    timestep[0]
                except TypeError:
                    raise TypeError("The mechanism  \"{}\" is incorrectly formatted at time step {!s}. "
                                    "Likely missing set of brackets.".format(mech.name, i))
                if not isinstance(timestep[0], Iterable) or isinstance(timestep[0], str):  # Iterable imported from collections
                    # If not, embellish the formatting to match the verbose case
                    timestep = [timestep]
                # Then, check that each input_state is receiving the right size of input
                for i, value in enumerate(timestep):
                    val_length = len(value)
                    state_length = len(mech.input_state.variable)
                    if val_length != state_length:
                        raise ValueError("The value provided for input state {!s} of the mechanism \"{}\" has length {!s} \
                            where the input state takes values of length {!s}".format(i, mech.name, val_length, state_length))

    def _create_input_mechanisms(self):
        '''
            builds a dictionary of { Mechanism : InputMechanism } pairs where each origin mechanism has a corresponding
            InputMechanism
        '''
        is_origin = self.get_mechanisms_by_role(MechanismRole.ORIGIN)
        has_input_mechanism = self.input_mechanisms.keys()

        # consider all of the mechanisms that are only origins OR have input mechanisms
        for mech in is_origin.difference(has_input_mechanism):

            # If mech IS AN ORIGIN mechanism but it doesn't have an input mechanism, ADD input mechanism
            if mech not in has_input_mechanism:
                new_input_mech = TransferMechanism()
                self.input_mechanisms[mech] = new_input_mech
                MappingProjection(sender=new_input_mech, receiver=mech)

            # If mech HAS AN INPUT mechanism but isn't an origin, REMOVE the input mechanism
            else:
                del self.input_mechanisms[mech]

    def _assign_values_to_input_mechanisms(self, input_dict):
        '''
            loops over the input values in the inputs dictionary and assigns each value directly to the output state of
            its corresponding input mechanism
        '''
        for mech in self.input_mechanisms.keys():
            if mech in input_dict.keys():
                self.input_mechanisms[mech]._output_states[0].value = np.array(input_dict[mech])
            else:
                self.input_mechanisms[mech]._output_states[0].value = np.array(mech.variable)


    def _assign_execution_ids(self, execution_id):
        '''
            assigns the same uuid to each mechanism in the composition's processing graph as well as all input
            mechanisms for this composition. The uuid is either specified in the user's call to run(), or generated
            randomly at run time.
        '''

        # Traverse processing graph and assign one uuid to all of its mechanisms
        self._execution_id = execution_id or self._get_unique_id()
        for v in self._graph_processing.vertices:
            v.component._execution_id = self._execution_id
        # Assign the uuid to all input mechanisms
        for k in self.input_mechanisms.keys():
            self.input_mechanisms[k]._execution_id = self._execution_id

    def execute(
            self,
            inputs,
            scheduler_processing=None,
            scheduler_learning=None,
            call_before_pass=None,
            call_before_timestep=None,
            execution_id = None):
        '''
            Passes inputs to any mechanisms receiving inputs directly from the user, then coordinates with the scheduler
            to receive and execute sets of mechanisms that are eligible to run until termination conditions are met.

            Arguments
            ---------
            scheduler_processing : Scheduler
                the scheduler object which owns the conditions that will instruct the non-learning execution of this Composition. \
                If not specified, the Composition will use its automatically generated scheduler

            scheduler_learning : Scheduler
                the scheduler object which owns the conditions that will instruct the Learning execution of this Composition. \
                If not specified, the Composition will use its automatically generated scheduler

            inputs: { Mechanism : list }
                a dictionary containing a key-value pair for each mechanism in the composition that receives inputs from
                the user. For each pair, the key is the Mechanism and the value is a list of inputs.

            execution_id : UUID
                execution_id will typically be set to none and assigned randomly at runtime

            Returns
            ---------
            output value of the final mechanism executed in the composition
        '''

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        if scheduler_learning is None:
            scheduler_learning = self.scheduler_learning

        self._create_input_mechanisms()
        self._assign_values_to_input_mechanisms(inputs)
        self._assign_execution_ids(execution_id)

        # run scheduler to receive sets of mechanisms that may be executed at this time step in any order
        execution_scheduler = scheduler_processing
        for next_execution_set in execution_scheduler.run():
            if call_before_timestep:
                call_before_timestep()
            # execute each mechanism with EXECUTING in context
            for mechanism in next_execution_set:
                if isinstance(mechanism, Mechanism):
                    num = mechanism.execute(context=EXECUTING + "composition")
                    print(" -------------- EXECUTING ", mechanism.name, " -------------- ")
                    print("result = ", num)
                    print()
                    print()
        return num

    def run(
        self,
        scheduler_processing=None,
        scheduler_learning=None,
        inputs=None,
        execution_id=None,
        num_trials=None,
        call_before_trial = None,
        call_before_pass = None,
        call_before_timestep = None
    ):
        '''
            Passes inputs to any mechanisms receiving inputs directly from the user, then coordinates with the scheduler
            to receive and execute sets of mechanisms that are eligible to run until termination conditions are met.

            Arguments
            ---------

            scheduler_processing : Scheduler
                the scheduler object which owns the conditions that will instruct the non-learning execution of this Composition. \
                If not specified, the Composition will use its automatically generated scheduler

            scheduler_learning : Scheduler
                the scheduler object which owns the conditions that will instruct the Learning execution of this Composition. \
                If not specified, the Composition will use its automatically generated scheduler

            inputs: { Mechanism : list }
                a dictionary containing a key-value pair for each mechanism in the composition that receives inputs from
                the user. For each pair, the key is the Mechanism and the value is a list of inputs. Each input in the list \
                corresponds to a certain `TRIAL`

            execution_id : UUID
                execution_id will typically be set to none and assigned randomly at runtime

            num_trials : int
                typically, the composition will infer the number of trials from the length of its input specification.
                To reuse the same inputs across many trials, you may specify an input dictionary with lists of length 1,
                or use default inputs, and select a number of trials with num_trials.

            Returns
            ---------

            output value of the final mechanism executed in the composition
        '''
        reuse_inputs = False

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        if scheduler_learning is None:
            scheduler_learning = self.scheduler_learning
        if inputs is None:
            inputs = {}
            len_inputs = 1
        else:

            len_inputs = len(list(inputs.values())[0])

        # check whether the num trials given in the input dict matches the num_trials param
        if num_trials:
            if len_inputs != num_trials:
                # if one set of inputs was provided for many trials, set 'reuse_inputs' flag
                if len_inputs == 1:
                    reuse_inputs = True
                # otherwise, warn user that there is something wrong with their input specification
                else:
                    raise CompositionError("The number of trials [{}] specified for the composition [{}] does not match the "
                                           "length [{}] of the inputs specified in the inputs dictionary [{}]. "
                                           .format(num_trials, self, len_inputs, inputs))

        input_indices = range(len_inputs)

        # TBI: Handle learning graph

        # TBI: Handle runtime params?

        # loop over the length of the list of inputs (# of trials)
        for input_index in input_indices:

            if call_before_trial:
                call_before_trial()

            execution_inputs = {}

            # loop over all mechanisms that receive inputs from the outside world
            for mech in inputs.keys():
                execution_inputs[mech] = inputs[mech][0 if reuse_inputs else input_index]

            num = self.execute(execution_inputs, scheduler_processing, execution_id, call_before_timestep, call_before_pass)

        # return the output of the LAST mechanism executed in the composition
        return num
