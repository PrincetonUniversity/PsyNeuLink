import logging

from collections import Iterable, OrderedDict
from enum import Enum

from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
from PsyNeuLink.Globals.Keywords import EXECUTING, CENTRAL_CLOCK
from PsyNeuLink.Globals.TimeScale import TimeScale, CurrentTime, CentralClock
from PsyNeuLink.Components.Projections.Projection import _add_projection_to, _add_projection_from
import uuid

logger = logging.getLogger(__name__)


class MechanismRole(Enum):
    ORIGIN = 0
    INTERNAL = 1
    CYCLE = 2
    INITIALIZE_CYCLE = 3
    TERMINAL = 4
    SINGLETON = 5
    MONITORED = 6
    TARGET = 7
    RECURRENT_INIT = 8


class CompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class Vertex(object):
    ########
    # Helper class for Compositions.
    # Serves as vertex for composition graph.
    # Contains lists of incoming edges and outgoing edges.
    ########

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
    ########
    # Helper class for Compositions.
    # Serves to organize mechanisms.
    # Contains a list of vertices.
    ########

    def __init__(self):
        self.comp_to_vertex = OrderedDict()  # Translate from mechanisms to related vertex
        self.vertices = []  # List of vertices within graph

    def copy(self):
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
        return self.comp_to_vertex[component].parents

    def get_children_from_component(self, component):
        return self.comp_to_vertex[component].children


class Composition(object):

    def __init__(self):
        ########
        # Constructor for Compositions.
        # Creates an empty Composition which has the following elements:
        # - self.G is an OrderedDict that represents the Composition's graph.
        #   Keys are mechanisms and values are lists of Connections that
        #   terminate on that mechanism.
        # - self.scheduler is a Scheduler object (see PsyNeuLink.scheduler)
        #   that manages the order of mechanisms that fire on a given trial.
        # - self.graph_analyzed is a Boolean that keeps track of whether
        #   self.graph is up to date. This allows for analysis of the graph
        #   only when needed for running the Composition for efficiency.
        # - self.*_mechanisms is a list of Mechanisms that are of a certain
        #   class within the composition graph.
        ########

        # core attributes
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.mechanisms = []

        # status attributes
        # Needs to be created still| self.scheduler = Scheduler()
        self.needs_update_graph = True   # Tracks if the Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True   # Tracks if the processing graph is current with the full graph
        self.graph_consistent = True  # Tracks if the Composition is in a state that can be run (i.e. no dangling projections, (what else?))

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
        self.sched = Scheduler(composition=self)

    @property
    def graph_processing(self):
        if self.needs_update_graph_processing or self._graph_processing is None:
            self._update_processing_graph()

        return self._graph_processing

    def _get_unique_id(self):
        return uuid.uuid4()

    def add_mechanism(self, mech):
        ########
        # Adds a new Mechanism to the Composition.
        # If the mechanism has already been added, passes.
        ########
        if mech not in [vertex.component for vertex in self.graph.vertices]:  # Only add if it doesn't already exist in graph
            mech.is_processing = True
            self.graph.add_component(mech)  # Set incoming edge list of mech to empty
            self.mechanisms.append(mech)
            self.mechanisms_to_roles[mech] = set()

            self.needs_update_graph = True
            self.needs_update_graph_processing = True

    def add_projection(self, sender, projection, receiver):
        ########
        # Adds a new Projection to the Composition.
        # If the projection has already been added, passes.
        ########
        if projection not in [vertex.component for vertex in self.graph.vertices]:
            projection.is_processing = False
            projection.name = '{0} to {1}'.format(sender, receiver)
            self.graph.add_component(projection)

            # Add connections between mechanisms and the projection
            self.graph.connect_components(sender, projection)
            self.graph.connect_components(projection, receiver)
            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.validate_projection(sender, projection, receiver)

    def validate_projection(self, sender, projection, receiver):
        print(projection.sender.owner)
        print(projection.receiver.owner)
        print(sender)
        print(receiver)

        if hasattr(projection, "sender") and hasattr(projection, "receiver"):

            if projection.sender.owner != sender:
                raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                       "components in their composition.".format(projection, sender))

            if projection.receiver.owner != receiver:
                raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                       "components in their composition.".format(projection, receiver))
        else:
            print("REASSIGNED")
            projection.sender = sender
            projection.receiver = receiver
            projection._deferred_init(context="deferred init")

        if projection.sender.owner != sender:
            raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                   "components in the composition.".format(projection, sender))
        if projection.receiver.owner != receiver:
            raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                   "components in the composition.".format(projection, receiver))



    def analyze_graph(self, graph=None):
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
                self.add_mechanism_role(mech, MechanismRole.ORIGIN)
        # Identify Terminal mechanisms
            if graph.get_children_from_component(mech) == []:
                self.add_mechanism_role(mech, MechanismRole.TERMINAL)
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
                        self.add_mechanism_role(mech, MechanismRole.RECURRENT_INIT)
                        self.add_mechanism_role(child, MechanismRole.CYCLE)
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
                            self.add_mechanism_role(remaining_mech, MechanismRole.RECURRENT_INIT)
                            self.add_mechanism_role(child, MechanismRole.CYCLE)
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
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        try:
            return set([mech for mech in self.mechanisms if role in self.mechanisms_to_roles[mech]])
        except KeyError as e:
            raise CompositionError('Mechanism not assigned to role in mechanisms_to_roles: {0}'.format(e))

    def set_mechanism_roles(self, mech, roles):
        self.clear_mechanism_role(mech)
        for role in roles:
            self.add_mechanism_role(role)

    def clear_mechanism_roles(self, mech):
        if mech in self.mechanisms_to_roles:
            self.mechanisms_to_roles[mech] = set()

    def add_mechanism_role(self, mech, role):
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        self.mechanisms_to_roles[mech].add(role)

    def remove_mechanism_role(self, mech, role):
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        self.mechanisms_to_roles[mech].remove(role)

    # mech_type specifies a type of mechanism, mech_type_list contains all of the mechanisms of that type
    # feed_dict is a dictionary of the input states of each mechanism of the specified type
    def validate_feed_dict(self, feed_dict, mech_type_list, mech_type):
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

    def run(self, scheduler, inputs={}, targets=None, recurrent_init=None, execution_id = None):
        # all origin mechanisms
        is_origin = self.get_mechanisms_by_role(MechanismRole.ORIGIN)
        # all mechanisms with inputs specified in the inputs dictionary
        has_inputs = inputs.keys()

        if inputs != {}:
            len_inputs = len(list(inputs.values())[0])
        else:
            len_inputs = 1

        input_indices = range(len_inputs)

        # if inputs:
        #      self.validate_feed_dict(inputs, is_origin, "Inputs")
        # if targets:
        #     self.validate_feed_dict(targets, self.target_mechanisms, "Targets")
        # if recurrent_init:
        #     self.validate_feed_dict(recurrent_init, self.recurrent_init_mechanisms, "Recurrent Init")

        # Traverse processing graph and assign one uuid to all of its mechanisms
        self._execution_id = execution_id or self._get_unique_id()
        for v in self._graph_processing.vertices:
            v.component._execution_id = self._execution_id

        # TBI: Do the same for learning graph?

        # TBI: Handle runtime params?

        for input_index in input_indices:
            # print("TRIAL ", TimeScale.TRIAL)
            # print("RUN ", TimeScale.RUN)
            # print("TIME_STEP", TimeScale.TIME_STEP)

            # reset inputs to each mechanism, variables, previous values, etc.
            self.sched._reset_counts_total(time_scale=TimeScale.TRIAL)
            # self.sched._reset_time(TimeScale.RUN)

            # run scheduler to receive sets of mechanisms that may be executed at this time step in any order
            for next_execution_set in scheduler.run():

                # execute each mechanism with context = EXECUTING and the appropriate input
                for mechanism in next_execution_set:
                    if isinstance(mechanism, Mechanism):

                        # if mechanism is_origin and is featured in the inputs dictionary -- use specified input
                        if (mechanism in is_origin) and (mechanism in has_inputs):
                            print()
                            num = mechanism.execute(input=inputs[mechanism][input_index], context=EXECUTING + "composition")
                            print(" -------------- EXECUTING ", mechanism.name, " -------------- ")
                            print("result = ", num)
                            print()
                            print()

                        # otherwise, mechanism will use its default input OR whatever it received from its projection(s)
                        else:
                            num = mechanism.execute(context=EXECUTING+ "composition")
                            print(" -------------- EXECUTING ", mechanism.name, " -------------- ")
                            print("result = ", num)
                            print()
                            print()

        # return the output of the LAST mechanism executed in the composition
        return num
