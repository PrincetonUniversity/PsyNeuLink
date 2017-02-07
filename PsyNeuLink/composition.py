from collections import OrderedDict
import itertools
# Needs to be created still| from PsyNeuLink.scheduling import Scheduler

class Binding(object):
    ########
    # Helper class for Compositions.
    # Contains a sender, projection, and receiver
    # Performs no operations outside getting and setting these attributes
    ########
    def __init__(self, sender, projection, receiver):
        self.sender = sender
        self.projection = projection
        self.receiver = receiver


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
        #   self.???? is up to date. This allows for analysis of the graph
        #   only when needed for running the Composition for efficiency.
        # - self.*_mechanisms is a list of Mechanisms that are of a certain
        #   class within the composition graph.
        ########
        self.graph = OrderedDict() # Graph of the Composition
        # Needs to be created still| self.scheduler = Scheduler()
        self.graph_analyzed = False # Tracks if the Composition is ready to run

        # Create lists to track identity of certain mechanism classes within the
        # composition.
        # Analyzed classes:
        self.origin_mechanisms = []
        self.terminal_mechanisms = []
        self.monitored_mechanisms = []
        self.init_cycle_mechanisms = []
        self.cycle_mechanisms = []
        # Explicit classes:
        self.explicit_input_mechanisms = [] # Need to track to know which to leave untouched
        self.all_input_mechanisms = []
        self.explicit_output_mechanisms = [] # Need to track to know which to leave untouched
        self.target_mechanisms = [] # Do not need to track explicit as they mush be explicit

    def add_mechanism(self, mech):
        ########
        # Adds a new Mechanism to the Composition.
        # If the mechanism has already been added, passes.
        ########
        if mech not in self.graph.keys(): # Only add if it doesn't already exist in graph
            self.graph[mech] = [] # Set Connection list of mech to empty
            self.graph_analyzed = False # Added mech so must re-analyze graph

    def add_projection(self, sender, projection, receiver):
        ########
        # Adds a new Projection to the Composition.
        # If the projection has allready been added, passes.
        ########
        # Ensures that the projection does not already exist at this place in the graph
        if any([binding.projection is projection for binding in self.graph[receiver]]):
            return
        # Add connection to graph
        self.graph[receiver].append(Binding(sender, projection, receiver))
        self.graph_analyzed = False # Added projection so must re-analyze graph


    def set_origin(self, mech):
        if mech not in self.origin_mechanisms:
            self.origin_mechanisms.append(mech)

    def remove_origin(self, mech):
        if mech in self.origin_mechanisms:
            self.origin_mechanisms.remove(mech)

    def analyze_graph(self):
        ########
        # Determines identity of significant nodes of the graph
        # Each node falls into one or more of the following categories
        # - Origin: Origin mechanisms are those which do not receive any projections.
        # - Terminal: Terminal mechanisms provide the output of the composition. By
        #   default, those which do not send any projections, but they may also be
        #   specified explicitly.
        # - Monitored: Monitored mechanisms send projections to Target mechanisms.
        # - Init_cycle: Init_cycle mechanisms send projections that close recurrent
        #   loops in the composition (or projections that are explicitly specified as
        #   recurrent). They need an initial value so that their receiving mechanisms
        #   have input.
        # - Cycle: Cycle mechanisms receive projections from Init_cycle mechanisms. They
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
        #   compare the target value to the output of another mechanism in the composition
        # XXX Singleton: Singleton mechanisms are mechanisms that are both origin and terminal
        # XXX in the composition. ##NOT IMPLEMENTED
        ########

        # Clear old information
        self.origin_mechanisms = []
        self.terminal_mechanisms = []
        self.monitored_mechanisms = []
        self.init_cycle_mechanisms = []
        self.cycle_mechanisms = []

        # Get list of mechanisms that send projections to identify Terminals
        senders = [binding.sender for binding in itertools.chain.from_interable(self.graph.keys())]
        # Identify Origin Mechanisms
        for mech in self.graph.keys(): # Iterate through mechanisms
            if self.graph[mech] is []: # If there are no incoming bindings
                self.origin_mechanisms.append(mech) # Add as Origin
        # Identify Terminal Mechanisms
            if mech not in senders: # If the mech does not send any projections
                self.terminal_mechanisms.append(mech) # Add as Terminal
        


        return