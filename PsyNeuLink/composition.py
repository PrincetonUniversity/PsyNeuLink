from collections import OrderedDict
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
        ########
        self.G = OrderedDict() # Graph of the Composition
        # Needs to be created still| self.scheduler = Scheduler()
        self.graph_analyzed = False # Tracks if the Composition is ready to run

    def add_mechanism(self, mech):
        ########
        # Adds a new Mechanism to the Composition.
        # If the mechanism has already been added, passes.
        ########
        if mech not in self.G.keys(): # Only add if it doesn't already exist in graph
            self.G[mech] = [] # Set Connection list of mech to empty
            self.graph_analyzed = False # Added mech so must re-analyze graph

    def add_projection(self, sender, projection, receiver):
        ########
        # Adds a new Projection to the Composition.
        # If the projection has allready been added, passes.
        ########
        # Ensures that the projection does not already exist at this place in the graph
        if any([binding.projection is projection for binding in self.G[receiver]]):
            return
        # Add connection to graph
        self.G[receiver].append(Binding(sender, projection, receiver))

    def analyze_graph(self):
        ########
        # Determines identity of significant nodes of the graph
        # Each node falls into one or more of the following categories
        # Origin: 
        return