import logging

from collections import Iterable, OrderedDict
from enum import Enum

from PsyNeuLink.scheduling.Scheduler import Scheduler

logger = logging.getLogger(__name__)


class MechanismRole(Enum):
    ORIGIN = 0
    INTERNAL = 1
    CYCLE = 2
    INITIALIZE_CYCLE = 3
    TERMINAL = 4
    SINGLETON = 5
    MONITORING = 6
    TARGET = 7


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
        # Analyzed classes:
        self.origin_mechanisms = []
        self.terminal_mechanisms = []
        self.monitored_mechanisms = []
        self.recurrent_init_mechanisms = []
        self.cycle_mechanisms = []
        # Explicit classes:
        self.explicit_input_mechanisms = []  # Need to track to know which to leave untouched
        self.all_input_mechanisms = []
        self.explicit_output_mechanisms = []  # Need to track to know which to leave untouched
        self.all_output_mechanisms = []
        self.target_mechanisms = []  # Do not need to track explicit as they mush be explicit
        self.sched = Scheduler(self)

    @property
    def graph_processing(self):
        if self.needs_update_graph_processing or self._graph_processing is None:
            self._update_processing_graph()

        return self._graph_processing

    def add_mechanism(self, mech):
        ########
        # Adds a new Mechanism to the Composition.
        # If the mechanism has already been added, passes.
        ########
        if mech not in [vertex.component for vertex in self.graph.vertices]:  # Only add if it doesn't already exist in graph
            mech.is_processing = True
            self.graph.add_component(mech)  # Set incoming edge list of mech to empty
            self.mechanisms.append(mech)

            self.needs_update_graph = True
            self.needs_update_graph_processing = True

    def add_projection(self, sender, projection, receiver):
        ########
        # Adds a new Projection to the Composition.
        # If the projection has already been added, passes.
        ########
        if projection not in [vertex.component for vertex in self.graph.vertices]:
            projection.is_processing = False
            self.graph.add_component(projection)

            # Add connections between mechanisms and the projection
            self.graph.connect_components(sender, projection)
            self.graph.connect_components(projection, receiver)
            self.needs_update_graph = True
            self.needs_update_graph_processing = True

    def analyze_graph(self):
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

        # Clear old information
        for mech in self.origin_mechanisms:
            self.remove_origin(mech)
        for mech in self.terminal_mechanisms:
            self.remove_terminal(mech)
        for mech in self.recurrent_init_mechanisms:
            self.remove_recurrent_init(mech)
        for mech in self.cycle_mechanisms:
            self.remove_cycle(mech)

        # Identify Origin mechanisms
        for mech in self.graph.mechanisms:
            if self.graph.get_incoming(mech) == []:
                self.set_origin(mech)
        # Identify Terminal mechanisms
            if self.graph.get_outgoing(mech) == []:
                self.set_terminal(mech)
        # Identify Recurrent_init and Cycle mechanisms
        visited = []  # Keep track of all mechanisms that have been visited
        for origin_mech in self.origin_mechanisms:  # Cycle through origin mechanisms first
            visited_current_path = []  # Track all mechanisms visited from the current origin
            next_visit_stack = []  # Keep a stack of mechanisms to be visited next
            next_visit_stack.append(origin_mech)
            for mech in next_visit_stack:  # While the stack isn't empty
                visited.append(mech)  # Mark the mech as visited
                visited_current_path.append(mech)  # And visited during the current path
                children = self.graph.get_children(mech)  # Get the children of that mechanism
                for child in children:
                    # If the child has been visited this path and is not already initialized
                    if child in visited_current_path:
                        if mech not in self.recurrent_init_mechanisms:
                            self.set_recurrent_init(mech)  # Set the parent as Recurrent_init
                        if child not in self.cycle_mechanisms:
                            self.set_cycle(child)  # And the child as Cycle
                    elif child not in visited:  # Else if the child has not been explored
                        next_visit_stack.append(child)  # Add it to the visit stack
        for mech in self.graph.mechanisms:
            if mech not in visited:  # Check the rest of the mechanisms
                visited_current_path = []
                next_visit_stack = []
                next_visit_stack.append(mech)
                for remaining_mech in next_visit_stack:
                    visited.append(remaining_mech)
                    visited_current_path.append(remaining_mech)
                    children = self.graph.get_children(remaining_mech)
                    for child in children:
                        if child in visited_current_path:
                            if remaining_mech not in self.recurrent_init_mechanisms:
                                self.set_recurrent_init(remaining_mech)
                            if child not in self.cycle_mechanisms:
                                self.set_cycle(child)
                        elif child not in visited:
                            next_visit_stack.append(child)

        self.needs_update_graph = False

    def _update_processing_graph(self):
        logger.debug('Updating processing graph')
        self._graph_processing = self.graph.copy()
        visited_vertices = set()
        next_vertices = []  # a queue

        while len(self.graph.vertices) > len(visited_vertices):
            for vertex in self.graph.vertices:
                if vertex not in visited_vertices:
                    next_vertices.append(vertex)
                    break

            while len(next_vertices) > 0:
                cur_vertex = next_vertices.pop(0)
                logger.debug('Examining vertex {0}'.format(cur_vertex))

                if not cur_vertex.component.is_processing:
                    for parent in cur_vertex.parents:
                        parent.children.remove(cur_vertex)
                        for child in cur_vertex.children:
                            child.parents.remove(cur_vertex)
                            self._graph_processing.connect_vertices(parent, child)

                    logger.debug('Removing vertex {0}'.format(cur_vertex))
                    self._graph_processing.remove_vertex(cur_vertex)

                visited_vertices.add(cur_vertex)
                # add to frontier any parents and children of cur_vertex that have not been visited yet
                next_vertices.extend([vertex for vertex in cur_vertex.parents + cur_vertex.children if vertex not in visited_vertices])

        self.needs_update_graph_processing = False

    def get_mechanisms_by_role(self, role):
        if role not in MechanismRole:
            raise CompositionError('Invalid MechanismRole: {0}'.format(role))

        try:
            return set([mech for mech in self.mechanisms if self.mechanisms_to_roles[mech] == role])
        except KeyError as e:
            raise CompositionError('Mechanism not assigned to role in mechanisms_to_roles: {0}'.format(e))

    def set_origin(self, mech):
        if mech not in self.origin_mechanisms:  # If mechanism isn't in Origin list already
            self.origin_mechanisms.append(mech)  # Add to Origin list

    def remove_origin(self, mech):
        if mech in self.origin_mechanisms:  # If mechanism is in Origin list
            self.origin_mechanisms.remove(mech)  # Remove from Origin list

    def set_terminal(self, mech):
        if mech not in self.terminal_mechanisms:  # If mechanism isn't in Terminal list already
            self.terminal_mechanisms.append(mech)  # Add to Terminal list

    def remove_terminal(self, mech):
        if mech in self.terminal_mechanisms:  # If mechanism is in Terminal list
            self.terminal_mechanisms.remove(mech)  # Remove from Terminal list

    def set_recurrent_init(self, mech):
        if mech not in self.recurrent_init_mechanisms:  # If mechanism isn't in Recurrent_init list already
            self.recurrent_init_mechanisms.append(mech)  # Add to Recurrent_init list

    def remove_recurrent_init(self, mech):
        if mech in self.recurrent_init_mechanisms:  # If mechanism is in Recurrent_init list
            self.recurrent_init_mechanisms.remove(mech)  # Remove from Recurrent_init list

    def set_cycle(self, mech):
        if mech not in self.cycle_mechanisms:  # If mechanism isn't in Cycle list already
            self.cycle_mechanisms.append(mech)  # Add to Cycle list

    def remove_cycle(self, mech):
        if mech in self.cycle_mechanisms:  # If mechanism is in Cycle list
            self.cycle_mechanisms.remove(mech)  # Remove from Cycle list

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

    def run(self, scheduler, inputs=None, targets=None, recurrent_init=None):

        if inputs:
            self.validate_feed_dict(inputs, self.origin_mechanisms, "Inputs")
        if targets:
            self.validate_feed_dict(targets, self.target_mechanisms, "Targets")
        if recurrent_init:
            self.validate_feed_dict(recurrent_init, self.recurrent_init_mechanisms, "Recurrent Init")

        '''
        for current_component in scheduler.run_trial():
            if current_component.name != "Clock":
                # print("NAME: ",current_component.name)
                current_vertex = self.graph.mech_to_vertex[current_component]
                # print("INCOMING PROJECTION: ", current_vertex.incoming)
                # print("OUTGOING PROJECTION: ", current_vertex.outgoing)
                if current_component in inputs.keys():
                    # print(current_component.name, " was found in inputs")
                    new_value = current_component.execute(inputs[current_component])
                    # for edge in current_vertex.outgoing:
                    #     edge.projection.execute(new_value)

                else:
                    current_component.execute()
                # print(current_component.value)
            else:
                current_component.execute()
                # print(current_component.value)
        '''
