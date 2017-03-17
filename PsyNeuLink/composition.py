from collections import OrderedDict, Iterable
import itertools
from PsyNeuLink.scheduling.constraint_scheduler import Scheduler

class Edge(object):
    ########
    # Helper class for Compositions.
    # Contains a sender, projection, and receiver.
    # Performs no operations outside getting and setting these attributes.
    ########
    def __init__(self, sender, projection, receiver):
        self.sender = sender # Mechanism that feeds into edge
        self.projection = projection # Projection that constitutes edge
        self.receiver = receiver # Mechanism that projection ends at

class Vertex(object):
    ########
    # Helper class for Compositions.
    # Serves as vertex for composition graph.
    # Contains lists of incoming edges and outgoing edges.
    ########
    def __init__(self, mechanism, incoming = None, outgoing = None):
        self.mechanism = mechanism
        if incoming is not None:
            self.incoming = incoming
        else: self.incoming = []
        if outgoing is not None:
            self.outgoing = outgoing
        else: self.outgoing = []

class Graph(object):
    ########
    # Helper class for Compositions.
    # Serves to organize mechanisms.
    # Contains a list of vertices.
    ########
    def __init__(self):
        self.mech_to_vertex = OrderedDict() # Translate from mechanisms to related vertex
        self.vertices = [] # List of vertices within graph
        self.mechanisms = [] # List of mechanisms associated with graph vertices

    def add_vertex(self, mechanism):
        if mechanism not in self.mechanisms:
            vertex = Vertex(mechanism)
            self.mech_to_vertex[mechanism] = vertex
            self.vertices.append(vertex)
            self.mechanisms.append(mechanism)

    def add_edge(self, sender, projection, receiver):
        sender_vertex = self.mech_to_vertex[sender]
        if projection not in sender_vertex.outgoing:
            sender_vertex.outgoing.append(Edge(sender, projection, receiver))
        receiver_vertex = self.mech_to_vertex[receiver]
        if projection not in receiver_vertex.incoming:
            receiver_vertex.incoming.append(Edge(sender, projection, receiver))

    def get_incoming(self, mechanism):
        return self.mech_to_vertex[mechanism].incoming

    def get_outgoing(self, mechanism):
        return self.mech_to_vertex[mechanism].outgoing

    def get_children(self, mechanism):
        return [edge.receiver for edge in self.mech_to_vertex[mechanism].outgoing]

    def get_parents(self, mechanism):
        return [edge.sender for edge in self.mech_to_vertex[mechanism].incoming]



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
        self.graph = Graph() # Graph of the Composition
        # Needs to be created still| self.scheduler = Scheduler()
        self.graph_analyzed = False # Tracks if the Composition is ready to run

        # Create lists to track identity of certain mechanism classes within the
        # composition.
        # Analyzed classes:
        self.origin_mechanisms = []
        self.terminal_mechanisms = []
        self.monitored_mechanisms = []
        self.recurrent_init_mechanisms = []
        self.cycle_mechanisms = []
        # Explicit classes:
        self.explicit_input_mechanisms = [] # Need to track to know which to leave untouched
        self.all_input_mechanisms = []
        self.explicit_output_mechanisms = [] # Need to track to know which to leave untouched
        self.all_output_mechanisms = []
        self.target_mechanisms = [] # Do not need to track explicit as they mush be explicit
        self.sched = Scheduler()

    def add_mechanism(self, mech):
        ########
        # Adds a new Mechanism to the Composition.
        # If the mechanism has already been added, passes.
        ########
        if mech not in self.graph.mechanisms: # Only add if it doesn't already exist in graph
            self.graph.add_vertex(mech) # Set incoming edge list of mech to empty
            self.graph_analyzed = False # Added mech so must re-analyze graph

    def add_projection(self, sender, projection, receiver):
        ########
        # Adds a new Projection to the Composition.
        # If the projection has already been added, passes.
        ########
        # Ensures that the projection does not already exist in the graph at this place
        if any([edge.projection is projection for edge in self.graph.get_incoming(receiver)]):
            return
        # Add edge to graph
        self.graph.add_edge(sender, projection, receiver)
        self.graph_analyzed = False # Added projection so must re-analyze graph

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
        visited = [] # Keep track of all mechanisms that have been visited
        for origin_mech in self.origin_mechanisms: # Cycle through origin mechanisms first
            visited_current_path = [] # Track all mechanisms visited from the current origin
            next_visit_stack = [] # Keep a stack of mechanisms to be visited next
            next_visit_stack.append(origin_mech)
            for mech in next_visit_stack: # While the stack isn't empty
                visited.append(mech) # Mark the mech as visited
                visited_current_path.append(mech) # And visited during the current path
                children = self.graph.get_children(mech) # Get the children of that mechanism
                for child in children:
                    # If the child has been visited this path and is not already initialized
                    if child in visited_current_path:
                        if mech not in self.recurrent_init_mechanisms:
                            self.set_recurrent_init(mech) # Set the parent as Recurrent_init
                        if child not in self.cycle_mechanisms:
                            self.set_cycle(child) # And the child as Cycle
                    elif child not in visited: # Else if the child has not been explored
                        next_visit_stack.append(child) # Add it to the visit stack
        for mech in self.graph.mechanisms:
            if mech not in visited: # Check the rest of the mechanisms
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
        return

    def set_origin(self, mech):
        if mech not in self.origin_mechanisms: # If mechanism isn't in Origin list already
            self.origin_mechanisms.append(mech) # Add to Origin list

    def remove_origin(self, mech):
        if mech in self.origin_mechanisms: # If mechanism is in Origin list
            self.origin_mechanisms.remove(mech) # Remove from Origin list

    def set_terminal(self, mech):
        if mech not in self.terminal_mechanisms: # If mechanism isn't in Terminal list already
            self.terminal_mechanisms.append(mech) # Add to Terminal list

    def remove_terminal(self, mech):
        if mech in self.terminal_mechanisms: # If mechanism is in Terminal list
            self.terminal_mechanisms.remove(mech) # Remove from Terminal list

    def set_recurrent_init(self, mech):
        if mech not in self.recurrent_init_mechanisms: # If mechanism isn't in Recurrent_init list already
            self.recurrent_init_mechanisms.append(mech) # Add to Recurrent_init list

    def remove_recurrent_init(self, mech):
        if mech in self.recurrent_init_mechanisms: # If mechanism is in Recurrent_init list
            self.recurrent_init_mechanisms.remove(mech) # Remove from Recurrent_init list

    def set_cycle(self, mech):
        if mech not in self.cycle_mechanisms: # If mechanism isn't in Cycle list already
            self.cycle_mechanisms.append(mech) # Add to Cycle list

    def remove_cycle(self, mech):
        if mech in self.cycle_mechanisms: # If mechanism is in Cycle list
            self.cycle_mechanisms.remove(mech) # Remove from Cycle list

    # mech_type specifies a type of mechanism, mech_type_list contains all of the mechanisms of that type
    # feed_dict is a dictionary of the input states of each mechanism of the specified type
    def validate_feed_dict(self, feed_dict, mech_type_list, mech_type):
        for mech in feed_dict.keys(): # For each mechanism given an input
            if mech not in mech_type_list: # Check that it is the right kind of mechanism in the composition
                if mech_type[0] in ['a', 'e', 'i', 'o', 'u']: # Check for grammar
                    article = "an"
                else: article = "a"
                # Throw an error informing the user that the mechanism was not found in the mech type list
                raise ValueError("The mechanism \"{}\" is not {} {} of the composition".format(mech.name, article, mech_type))
            for i, timestep in enumerate(feed_dict[mech]): # If mechanism is correct type, iterate over timesteps
                # Check if there are multiple input states specified
                try:
                    timestep[0]
                except TypeError:
                    raise TypeError("The mechanism  \"{}\" is incorrectly formatted at time step {!s}. "
                                    "Likely missing set of brackets.".format(mech.name,i))
                if not isinstance(timestep[0], Iterable) or isinstance(timestep[0], str): # Iterable imported from collections
                    # If not, embellish the formatting to match the verbose case
                    timestep = [timestep]
                # Then, check that each input_state is receiving the right size of input
                for i, value in enumerate(timestep):
                    val_length = len(value)
                    state_length = len(list(mech.inputStates.items())[0][1].variable)
                    if val_length != state_length:
                        raise ValueError("The value provided for input state {!s} of the mechanism \"{}\" has length {!s} \
                            where the input state takes values of length {!s}".format(i, mech.name, val_length, state_length))

    def run(self, scheduler, inputs = None, targets = None, recurrent_init = None):

        if inputs:
            self.validate_feed_dict(inputs, self.origin_mechanisms, "Inputs")
        if targets:
            self.validate_feed_dict(targets, self.target_mechanisms, "Targets")
        if recurrent_init:
            self.validate_feed_dict(recurrent_init, self.recurrent_init_mechanisms, "Recurrent Init")

        print(self.origin_mechanisms)
        print(self.target_mechanisms)
        print(self.recurrent_init_mechanisms)

        for current_component in scheduler.run_trial():
            if current_component in inputs.keys():
                print(current_component.name, " was found in inputs")
                current_component.execute(inputs[current_component])
            else:
                current_component.execute()
            print(current_component.value)
