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
import inspect
import itertools
import logging

import numpy as np
import typecheck as tc
import uuid

from PIL import Image

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component, ComponentsMeta, function_type
from psyneulink.core.components.functions.interfacefunctions import InterfaceStateMap
from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import DEFAULT_MONITORED_STATE_EXPONENT, DEFAULT_MONITORED_STATE_MATRIX, DEFAULT_MONITORED_STATE_WEIGHT, ObjectiveMechanism
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.shellclasses import Composition_Base
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import ALL, BOLD, CONTROL, FUNCTIONS, HARD_CLAMP, IDENTITY_MATRIX, LABELS, MATRIX_KEYWORD_VALUES, MONITOR_FOR_CONTROL, NO_CLAMP, OWNER_VALUE, PROJECTIONS, PULSE_CLAMP, ROLES, SOFT_CLAMP, VALUES
from psyneulink.core.globals.parameters import Defaults, Parameter, ParametersBase
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import AutoNumber, CNodeRole, call_with_pruned_args
from psyneulink.core.scheduling.condition import All, Always, EveryNCalls
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

__all__ = [

    'Composition', 'CompositionError', 'CompositionRegistry'
]

logger = logging.getLogger(__name__)

CompositionRegistry = {}


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
class MonitoredOutputStatesOption(AutoNumber):
    """Specifies OutputStates to be monitored by a `ControlMechanism <ControlMechanism>`
    (see `ObjectiveMechanism_Monitored_Output_States` for a more complete description of their meanings."""
    ONLY_SPECIFIED_OUTPUT_STATES = ()
    """Only monitor explicitly specified Outputstates."""
    PRIMARY_OUTPUT_STATES = ()
    """Monitor only the `primary OutputState <OutputState_Primary>` of a Mechanism."""
    ALL_OUTPUT_STATES = ()
    """Monitor all OutputStates <Mechanism_Base.output_states>` of a Mechanism."""
    NUM_MONITOR_STATES_OPTIONS = ()

# Indices for items in tuple format used for specifying monitored_output_states using weights and exponents
OUTPUT_STATE_INDEX = 0
WEIGHT_INDEX = 1
EXPONENT_INDEX = 2
MATRIX_INDEX = 3
MonitoredOutputStateTuple = collections.namedtuple("MonitoredOutputStateTuple", "output_state weight exponent matrix")


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


class Composition(Composition_Base, metaclass=ComponentsMeta):
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

        external_input_sources : 'Dict`
            A dictionary in which the keys are Composition Nodes and the values are specifications of the node's
            external input source. Origin nodes are expected to receive an external input for each of their
            InputStates, via the input_CIM. If an origin node is specified as a key node in external_input_sources, it
            may borrow input_CIM output states from one or more other origin nodes, but still must receive an external
            input for each InputState. If a non-origin node is specified as a key node in external_input_sources, the
            external inputs are in addition to any other inputs the key node may receive, so there is not a requirement
            that all key node InputStates receive an input.

            Below are the options for specifying an external input source:

            - an origin node
                projections are created from the origin node's corresponding input_CIM OutputState(s) to the key node's
                InputState(s).

            - a list of origin nodes and origin node InputStates
                projections are created from each origin node InputState's correpsonding input_CIM OutputState to the
                key node's InputStates. If an origin node is included in the list, it is used as a short hand for all
                of its InputStates. If the key node is an origin node, then the number of InputStates represented in
                the list must exactly match the number of InputStates on the key node. Otherwise, the list may have the
                same number or fewer InputStates as the key node, and None may be used to "skip" over key node
                InputStates to which there should not be an external input.

            - `ALL`
                projections are created from each origin node's corresponding input_CIM OutputState(s) to the key node's
                InputState(s). (Excludes origin nodes that are already borrowing inputs from another origin node).

        default_execution_id
            if no *execution_id* is specified in a call to run, this *execution_id* will be used.

            :default value: the Composition's name

        COMMENT:
        name : str
            see `name <Composition_Name>`

        prefs : PreferenceSet
            see `prefs <Composition_Prefs>`
        COMMENT

    '''
    # Composition now inherits from Component, so registry inherits name None
    componentType = 'Composition'

    class Parameters(ParametersBase):
        """
            Attributes
            ----------

                results
                    see `results <Composition.results>`

                    :default value: []
                    :type: list

                simulation_results
                    see `simulation_results <Composition.simulation_results>`

                    :default value: []
                    :type: list

        """
        results = Parameter([], loggable=False)
        simulation_results = Parameter([], loggable=False)

    class _CompilationData(ParametersBase):
        ptx_execution = None
        parameter_struct = None
        context_struct = None
        data_struct = None
        scheduler_conditions = None

    def __init__(
        self,
        name=None,
        model_based_optimizer=None,
        enable_model_based_optimizer=None,
        external_input_sources=None,
        **param_defaults
    ):
        # also sets name
        register_category(
            entry=self,
            base_class=Composition,
            registry=CompositionRegistry,
            name=name,
        )

        # core attribute
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.c_nodes = []
        self.required_c_node_roles = []
        self.input_CIM = CompositionInterfaceMechanism(name=self.name + " Input_CIM",
                                                       composition=self)
        self.external_input_sources = external_input_sources
        if external_input_sources is None:
            self.external_input_sources = {}
        self.output_CIM = CompositionInterfaceMechanism(name=self.name + " Output_CIM",
                                                        composition=self)
        self.input_CIM_states = {}
        self.output_CIM_states = {}

        self.enable_model_based_optimizer = enable_model_based_optimizer
        self.default_execution_id = self.name
        self.execution_ids = [self.default_execution_id]
        self.model_based_optimizer = model_based_optimizer

        self.projections = []
        self.shadow_projections = {}

        self._scheduler_processing = None
        self._scheduler_learning = None

        # status attributes
        self.graph_consistent = True  # Tracks if the Composition is in a state that can be run (i.e. no dangling projections, (what else?))
        self.needs_update_graph = True   # Tracks if the Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True   # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler_processing = True  # Tracks if the processing scheduler needs to be regenerated
        self.needs_update_scheduler_learning = True  # Tracks if the learning scheduler needs to be regenerated (mechanisms/projections added/removed etc)

        self.c_nodes_to_roles = collections.OrderedDict()

        # Create lists to track certain categories of Composition Nodes:
        # TBI???
        self.explicit_input_nodes = []  # Need to track to know which to leave untouched
        self.all_input_nodes = []
        self.explicit_output_nodes = []  # Need to track to know which to leave untouched
        self.all_output_nodes = []
        self.target_nodes = []  # Do not need to track explicit as they must be explicit

        # TBI: update self.sched whenever something is added to the composition
        self.sched = Scheduler(composition=self, execution_id=self.default_execution_id)

        self.parameters = self.Parameters(owner=self, parent=self.class_parameters)
        self.defaults = Defaults(owner=self, **{k: v for (k, v) in param_defaults.items() if hasattr(self.parameters, k)})
        self._initialize_parameters()

        # Compiled resources
        self.__generated_wrappers = {}
        self.__compiled_mech = {}
        self.__generated_execution = None
        self.__compiled_execution = None
        self.__compiled_run = None

        self._compilation_data = self._CompilationData(owner=self)

    # def _instantiate_prediction_mechanisms(self, context=None):
    #
    #     if self.model_based_optimizer:
    #         if hasattr(self, "prediction_mechanisms"):
    #             for mechanism in self.prediction_mechanisms:
    #                 del mechanism
    #         self.prediction_mechanisms = []
    #         self.prediction_projections = {}
    #         self.prediction_origin_pairs = {}
    #         self.origin_prediction_pairs = {}
    #         for node in self.get_c_nodes_by_role(CNodeRole.ORIGIN):
    #             new_prediction_mechanism = IntegratorMechanism(name=node.name + " " + PREDICTION_MECHANISM,
    #                                                            default_variable=node.external_input_values)
    #             self.prediction_mechanisms.append(new_prediction_mechanism)
    #             self.prediction_origin_pairs[new_prediction_mechanism] = node
    #             self.origin_prediction_pairs[node] = new_prediction_mechanism
    #
    #             for i in range(len(node.external_input_states)):
    #                 input_state = node.external_input_states[i]
    #                 input_cim_output_state = self.input_CIM_states[input_state][1]
    #                 new_projection = MappingProjection(sender=input_cim_output_state,
    #                                                    receiver=new_prediction_mechanism._input_states[i])
    #
    #                 self.prediction_projections[node] = new_projection

    # def _execute_prediction_mechanisms(self, context=None):
    #
    #     for prediction_mechanism in self.prediction_mechanisms:
    #         for proj in prediction_mechanism.path_afferents:
    #             proj.context.execution_phase = ContextFlags.PROCESSING
    #         prediction_mechanism.context.execution_phase = ContextFlags.PROCESSING
    #         prediction_mechanism.execute(context=context)
    #
    #     self.__compiled_execution = None

    def __repr__(self):
        return '({0} {1})'.format(type(self).__name__, self.name)

    @property
    def graph_processing(self):
        '''
            The Composition's processing graph (contains only `Mechanisms <Mechanism>`, excluding those
            used in learning).

            :getter: Returns the processing graph, and builds the graph if it needs updating since the last access.
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
            self._scheduler_processing = Scheduler(graph=self.graph_processing, execution_id=self.default_execution_id)

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
            # self._scheduler_learning = Scheduler(graph=self.graph, execution_id=self.default_execution_id)

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

    def add_c_node(self, node, required_roles=None, external_input_source=None):
        '''
            Adds a Composition Node (`Mechanism` or `Composition`) to the Composition, if it is not already added

            Arguments
            ---------

            node : `Mechanism` or `Composition`
                the node to be added to the Composition

            required_roles : psyneulink.core.globals.utilities.CNodeRole or list of CNodeRoles
                any CNodeRoles roles that this node should have in addition to those determined by analyze graph.
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

            try:
                # activate any projections the node requires
                node._activate_projections_for_compositions(self)
            except AttributeError:
                pass

        if hasattr(node, "aux_components"):

            projections = []
            # Add all "c_nodes" to the composition first (in case projections reference them)
            for component in node.aux_components:
                if isinstance(component, (Mechanism, Composition)):
                    self.add_c_node(component)
                elif isinstance(component, Projection):
                    projections.append((component, False))
                elif isinstance(component, tuple):
                    if isinstance(component[0], Projection):
                        if isinstance(component[1], bool):
                            projections.append(component)
                        else:
                            raise CompositionError("Invalid component specification ({}) in {}'s aux_components. If a "
                                                   "tuple is used to specify a Projection, then the index 0 item must "
                                                   "be the Projection, and the index 1 item must be the feedback "
                                                   "specification (True or False).".format(component, node.name))
                    elif isinstance(component[0], (Mechanism, Composition)):
                        if isinstance(component[1], CNodeRole):
                            self.add_c_node(node=component[0], required_roles=component[1])
                        elif isinstance(component[1], list):
                            if isinstance(component[1][0], CNodeRole):
                                self.add_c_node(node=component[0], required_roles=component[1])
                            else:
                                raise CompositionError("Invalid component specification ({}) in {}'s aux_components. "
                                                       "If a tuple is used to specify a Mechanism or Composition, then "
                                                       "the index 0 item must be the node, and the index 1 item must "
                                                       "be the required_roles".format(component, node.name))

                        else:
                            raise CompositionError("Invalid component specification ({}) in {}'s aux_components. If a "
                                                   "tuple is used to specify a Mechanism or Composition, then the "
                                                   "index 0 item must be the node, and the index 1 item must be the "
                                                   "required_roles".format(component, node.name))
                    else:
                        raise CompositionError("Invalid component specification ({}) in {}'s aux_components. If a tuple"
                                               " is specified, then the index 0 item must be a Projection, Mechanism, "
                                               "or Composition.".format(component, node.name))
                else:
                    raise CompositionError("Invalid component ({}) in {}'s aux_components. Must be a Mechanism, "
                                           "Composition, Projection, or tuple."
                                           .format(component.name, node.name))

            # Add all projections to the composition
            for proj_spec in projections:
                self.add_projection(projection=proj_spec[0], feedback=proj_spec[1])
        if required_roles:
            if not isinstance(required_roles, list):
                required_roles = [required_roles]
            for required_role in required_roles:
                self.add_required_c_node_role(node, required_role)

        if external_input_source:
            self.external_input_sources[node] = external_input_source
        if hasattr(node, "shadow_external_inputs"):
            self.external_input_sources[node] = node.shadow_external_inputs

    def add_model_based_optimizer(self, optimizer):
        """
        Adds a `ModelBasedOptimizationControlMechanism` as the `model_based_optimizer
        <Composition.model_based_optimizer>` of the Composition, which gives the Mechanism access to the
        `Composition`'s `evaluate <Composition.evaluate>` method. This allows the
        `ModelBasedOptimizationControlMechanism` to use simulations to determine an optimal Control policy.
        """

        self.model_based_optimizer = optimizer
        self.model_based_optimizer.composition = self
        # if monitor_for_control:
        #     self.model_based_optimizer.objective_mechanism.add_monitored_output_states(monitor_for_control)
        self.add_c_node(self.model_based_optimizer.objective_mechanism)
        if hasattr(self.model_based_optimizer, "shadow_external_inputs"):
            self.external_input_sources[self.model_based_optimizer] = self.model_based_optimizer.shadow_external_inputs
        for proj in self.model_based_optimizer.objective_mechanism.path_afferents:
            self.add_projection(proj)

        optimizer._activate_projections_for_compositions(self)
        self._analyze_graph()

    def _get_control_signals_for_system(self, control_signals=None):
        """Generate and return a list of control_signal_specs for System

        Generate list from:
           ControlSignal specifications passed in from the **control_signals** argument.
           ParameterStates of the System's Mechanisms that have been assigned ControlProjections with deferred_init();
               Note: this includes any for which a ControlSignal rather than a ControlProjection
                     was used to specify control for a parameter (e.g., in a 2-item tuple specification for the
                     parameter); the initialization of the ControlProjection and, if specified, the ControlSignal
                     are completed in the call to _instantiate_control_signal() by the ControlMechanism.
        """

        control_signal_specs = control_signals or []
        for node in self.c_nodes:
            for parameter_state in node._parameter_states:
                for projection in parameter_state.mod_afferents:
                    # If Projection was deferred for init, instantiate its ControlSignal and then initialize it
                    if projection.context.initialization_status == ContextFlags.DEFERRED_INIT:
                        proj_control_signal_specs = projection.control_signal_params or {}
                        proj_control_signal_specs.update({PROJECTIONS: [projection]})
                        control_signal_specs.append(proj_control_signal_specs)
        return control_signal_specs

    def _get_monitored_output_states_for_system(self, model_based_optimizer=None):
        """
        Parse a list of OutputState specifications for System, model_based_optimizer, Mechanisms and/or their OutputStates:
            - if specification in output_state is None:
                 do NOT monitor this state (this overrides any other specifications)
            - if an OutputState is specified in *any* MONITOR_FOR_CONTROL, monitor it (this overrides any other specs)
            - if a Mechanism is terminal and/or specified in the System or `model_based_optimizer <Systsem_Base.model_based_optimizer>`:
                if MonitoredOutputStatesOptions is PRIMARY_OUTPUT_STATES:  monitor only its primary (first) OutputState
                if MonitoredOutputStatesOptions is ALL_OUTPUT_STATES:  monitor all of its OutputStates
            Note: precedence is given to MonitoredOutputStatesOptions specification in Mechanism > model_based_optimizer > System

        Notes:
        * MonitoredOutputStatesOption is an AutoNumbered Enum declared in ControlMechanism
            - it specifies options for assigning OutputStates of TERMINAL Mechanisms in the System
                to model_based_optimizer.monitored_output_states;  the options are:
                + PRIMARY_OUTPUT_STATES: assign only the `primary OutputState <OutputState_Primary>` for each
                  TERMINAL Mechanism
                + ALL_OUTPUT_STATES: assign all of the outputStates of each terminal Mechanism
            - precedence is given to MonitoredOutputStatesOptions specification in Mechanism > model_based_optimizer > System
        * model_based_optimizer.monitored_output_states is a list, each item of which is an OutputState from which a Projection
            will be instantiated to a corresponding InputState of the ControlMechanism
        * model_based_optimizer.input_states is the usual ordered dict of states,
            each of which receives a Projection from a corresponding OutputState in model_based_optimizer.monitored_output_states

        Returns list of MonitoredOutputStateTuples: (OutputState, weight, exponent, matrix)

        """
        # PARSE SPECS

        # Get OutputStates already being -- or specified to be -- monitored by model_based_optimizer
        if model_based_optimizer is not None and not inspect.isclass(model_based_optimizer):
            try:
                # Get from monitored_output_states attribute if model_based_optimizer is already implemented
                monitored_output_states = model_based_optimizer.monitored_output_states.copy() or []
                # Convert them to MonitoredOutputStateTuple specifications (for treatment below)
                monitored_output_state_specs = []
                for monitored_output_state, input_state in zip(monitored_output_states,
                                                               model_based_optimizer.objective_mechanism.input_states):
                    projection = input_state.path_afferents[0]
                    if not projection.sender is monitored_output_state:
                        raise SystemError("PROGRAM ERROR: Problem identifying projection ({}) for "
                                          "monitored_output_state ({}) specified for {} ({}) assigned to {}".
                                          format(projection.name,
                                                 monitored_output_state.name,
                                                 ControlMechanism.__name__,
                                                 model_based_optimizer.name,
                                                 self.name))
                    monitored_output_state_specs.append(MonitoredOutputStateTuple(monitored_output_state,
                                                                                  projection.weight,
                                                                                  projection.exponent,
                                                                                  projection.matrix))

                model_based_optimizer_specs = monitored_output_state_specs
            except AttributeError:
                # If model_based_optimizer has no monitored_output_states attribute, it has not yet been fully instantiated
                #    (i.e., the call to this method is part of its instantiation by a System)
                #    so, get specification from the **object_mechanism** argument
                if isinstance(model_based_optimizer.objective_mechanism, list):
                    # **objective_mechanism** argument was specified as a list
                    model_based_optimizer_specs = model_based_optimizer.objective_mechanism.copy() or []
                elif isinstance(model_based_optimizer.objective_mechanism, ObjectiveMechanism):
                    # **objective_mechanism** argument was specified as an ObjectiveMechanism, which has presumably
                    # already been instantiated, so use its monitored_output_states attribute
                    model_based_optimizer_specs = model_based_optimizer.objective_mechanism.monitored_output_states
        else:
            model_based_optimizer_specs = []

        # Get system's MONITOR_FOR_CONTROL specifications (specified in paramClassDefaults, so must be there)
        system_specs = self.monitor_for_control.copy()

        # If model_based_optimizer_specs has a MonitoredOutputStatesOption specification, remove any such spec from system specs
        if model_based_optimizer_specs:
            if (any(isinstance(item, MonitoredOutputStatesOption) for item in model_based_optimizer_specs)):
                option_item = next((item for item in system_specs if isinstance(item,MonitoredOutputStatesOption)),None)
                if option_item is not None:
                    del system_specs[option_item]
            for item in model_based_optimizer_specs:
                if item in system_specs:
                    del system_specs[system_specs.index(item)]

        # Combine model_based_optimizer and system specs
        # If there are none, assign PRIMARY_OUTPUT_STATES as default
        all_specs = model_based_optimizer_specs + system_specs or [MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]

        # Convert references to Mechanisms and/or OutputStates in all_specs to MonitoredOutputStateTuples;
        # Each spec to be converted should be one of the following:
        #    - a MonitoredOutputStatesOption (parsed below);
        #    - a MonitoredOutputStatesTuple (returned by _get_monitored_states_for_system when
        #          specs were initially processed by the System to parse its *monitor_for_control* argument;
        #    - a specification for an existing Mechanism or OutputStates from the *monitor_for_control* arg of System.
        all_specs_extracted_from_tuples=[]
        all_specs_parsed=[]
        for i, spec in enumerate(all_specs):

            # Leave MonitoredOutputStatesOption and MonitoredOutputStatesTuple spec in place;
            #    these are parsed later on
            if isinstance(spec, MonitoredOutputStatesOption):
                all_specs_extracted_from_tuples.append(spec)
                all_specs_parsed.append(spec)
                continue
            if isinstance(spec, MonitoredOutputStateTuple):
                all_specs_extracted_from_tuples.append(spec.output_state)
                all_specs_parsed.append(spec)
                continue

            # spec is from *monitor_for_control* arg, so convert/parse into MonitoredOutputStateTuple(s)
            # Note:  assign parsed spec(s) to a list, as there may be more than one (that will be added to all_specs)
            monitored_output_state_tuples = []

            weight=DEFAULT_MONITORED_STATE_WEIGHT
            exponent=DEFAULT_MONITORED_STATE_EXPONENT
            matrix=DEFAULT_MONITORED_STATE_MATRIX

            # spec is a tuple
            # - put OutputState(s) in spec
            # - assign any weight, exponent, and/or matrix specified
            if isinstance(spec, tuple):
                # 2-item tuple (<OutputState(s) name(s)>, <Mechanism>)
                if len(spec) == 2:
                    # FIX: DO ERROR CHECK ON THE FOLLOWING / ALLOW LIST OF STATES
                    spec = spec[1].output_states[spec[0]]
                # 3-item tuple (<OutputState(s) spec>, weight, exponent)
                elif len(spec) == 3:
                    spec, weight, exponent = spec
                # 4-item tuple (<OutputState(s) spec>, weight, exponent, matrix)
                elif len(spec) == 4:
                    spec, weight, exponent, matrix = spec

            if not isinstance(spec, list):
                spec_list = [spec]

            for spec in spec_list:
                # spec is an OutputState or Mechanism
                if isinstance(spec, (OutputState, Mechanism)):
                    # spec is an OutputState, so use it
                    if isinstance(spec, OutputState):
                        output_states = [spec]
                    # spec is Mechanism, so use the State's owner, and get the relevant OutputState(s)
                    elif isinstance(spec, Mechanism):
                        if (MONITOR_FOR_CONTROL in spec.params
                            and spec.params[MONITOR_FOR_CONTROL] is MonitoredOutputStatesOption.ALL_OUTPUT_STATES):
                            output_states = spec.output_states
                        else:
                            output_states = [spec.output_state]
                    for output_state in output_states:
                        monitored_output_state_tuples.extend(
                                [MonitoredOutputStateTuple(output_state=output_state,
                                                           weight=weight,
                                                           exponent=exponent,
                                                           matrix=matrix)])
                # spec is a string
                elif isinstance(spec, str):
                    # Search System for Mechanisms with OutputStates with the string as their name
                    for node in self.c_nodes:
                        for output_state in node.output_states:
                            if output_state.name == spec:
                                monitored_output_state_tuples.extend(
                                        [MonitoredOutputStateTuple(output_state=output_state,
                                                                   weight=weight,
                                                                   exponent=exponent,
                                                                   matrix=matrix)])

                else:
                    raise SystemError("Specification of item in \'{}\' arg in constructor for {} ({}) "
                                      "is not a recognized specification for an {}".
                                      format(MONITOR_FOR_CONTROL, self.name, spec, OutputState.__name__))

                all_specs_parsed.extend(monitored_output_state_tuples)
                all_specs_extracted_from_tuples.extend([item.output_state for item in monitored_output_state_tuples])

        all_specs = all_specs_parsed

        try:
            all (isinstance(item, (OutputState, MonitoredOutputStatesOption))
                 for item in all_specs_extracted_from_tuples)
        except:
            raise SystemError("PROGRAM ERROR: Fail to parse items of \'{}\' arg ({}) in constructor for {}".
                              format(MONITOR_FOR_CONTROL, self.name, spec, OutputState.__name__))

        # Get MonitoredOutputStatesOptions if specified for model_based_optimizer or System, and make sure there is only one:
        option_specs = [item for item in all_specs_extracted_from_tuples
                        if isinstance(item, MonitoredOutputStatesOption)]
        if not option_specs:
            ctlr_or_sys_option_spec = None
        elif len(option_specs) == 1:
            ctlr_or_sys_option_spec = option_specs[0]
        else:
            raise SystemError("PROGRAM ERROR: More than one MonitoredOutputStatesOption specified "
                              "for OutputStates to be monitored in {}: {}".
                           format(self.name, option_specs))

        # Get MONITOR_FOR_CONTROL specifications for each Mechanism and OutputState in the System
        # Assign OutputStates to monitored_output_states
        monitored_output_states = []

        # Notes:
        # * Use all_specs to accumulate specs from all mechanisms and their outputStates
        #     (for use in generating exponents and weights below)
        # * Use local_specs to combine *only current* Mechanism's specs with those from model_based_optimizer and system specs;
        #     this allows the specs for each Mechanism and its OutputStates to be evaluated independently of any others
        model_based_optimizer_and_system_specs = all_specs_extracted_from_tuples.copy()

        for node in self.c_nodes:

            # For each Mechanism:
            # - add its specifications to all_specs (for use below in generating exponents and weights)
            # - extract references to Mechanisms and outputStates from any tuples, and add specs to local_specs
            # - assign MonitoredOutputStatesOptions (if any) to option_spec, (overrides one from model_based_optimizer or system)
            # - use local_specs (which now has this Mechanism's specs with those from model_based_optimizer and system specs)
            #     to assign outputStates to monitored_output_states

            local_specs = model_based_optimizer_and_system_specs.copy()
            option_spec = ctlr_or_sys_option_spec

            # PARSE MECHANISM'S SPECS

            # Get MONITOR_FOR_CONTROL specification from Mechanism
            try:
                node_specs = node.paramsCurrent[MONITOR_FOR_CONTROL]

                if node_specs is NotImplemented:
                    raise AttributeError

                # Setting MONITOR_FOR_CONTROL to None specifies Mechanism's OutputState(s) should NOT be monitored
                if node_specs is None:
                    raise ValueError

            # Mechanism's MONITOR_FOR_CONTROL is absent or NotImplemented, so proceed to parse OutputState(s) specs
            except (KeyError, AttributeError):
                pass

            # Mechanism's MONITOR_FOR_CONTROL is set to None, so do NOT monitor any of its outputStates
            except ValueError:
                continue

            # Parse specs in Mechanism's MONITOR_FOR_CONTROL
            else:

                # Add mech_specs to all_specs
                all_specs.extend(node_specs)

                # Extract refs from tuples and add to local_specs
                for item in node_specs:
                    if isinstance(item, tuple):
                        local_specs.append(item[OUTPUT_STATE_INDEX])
                        continue
                    local_specs.append(item)

                # Get MonitoredOutputStatesOptions if specified for Mechanism, and make sure there is only one:
                #    if there is one, use it in place of any specified for model_based_optimizer or system
                option_specs = [item for item in node_specs if isinstance(item, MonitoredOutputStatesOption)]
                if not option_specs:
                    option_spec = ctlr_or_sys_option_spec
                elif option_specs and len(option_specs) == 1:
                    option_spec = option_specs[0]
                else:
                    raise SystemError("PROGRAM ERROR: More than one MonitoredOutputStatesOption specified in {}: {}".
                                   format(node.name, option_specs))

            # PARSE OutputState'S SPECS

            for output_state in node.output_states:

                # Get MONITOR_FOR_CONTROL specification from OutputState
                try:
                    output_state_specs = output_state.paramsCurrent[MONITOR_FOR_CONTROL]
                    if output_state_specs is NotImplemented:
                        raise AttributeError

                    # Setting MONITOR_FOR_CONTROL to None specifies OutputState should NOT be monitored
                    if output_state_specs is None:
                        raise ValueError

                # OutputState's MONITOR_FOR_CONTROL is absent or NotImplemented, so ignore
                except (KeyError, AttributeError):
                    pass

                # OutputState's MONITOR_FOR_CONTROL is set to None, so do NOT monitor it
                except ValueError:
                    continue

                # Parse specs in OutputState's MONITOR_FOR_CONTROL
                else:

                    # Note: no need to look for MonitoredOutputStatesOption as it has no meaning
                    #       as a specification for an OutputState

                    # Add OutputState specs to all_specs and local_specs
                    all_specs.extend(output_state_specs)

                    # Extract refs from tuples and add to local_specs
                    for item in output_state_specs:
                        if isinstance(item, tuple):
                            local_specs.append(item[OUTPUT_STATE_INDEX])
                            continue
                        local_specs.append(item)

            # Ignore MonitoredOutputStatesOption if any outputStates are explicitly specified for the Mechanism
            for output_state in node.output_states:
                if (output_state in local_specs or output_state.name in local_specs):
                    option_spec = None


            # ASSIGN SPECIFIED OUTPUT STATES FOR MECHANISM TO monitored_output_states

            for output_state in node.output_states:

                # If OutputState is named or referenced anywhere, include it
                if (output_state in local_specs or output_state.name in local_specs):
                    monitored_output_states.append(output_state)
                    continue

    # FIX: NEED TO DEAL WITH SITUATION IN WHICH MonitoredOutputStatesOptions IS SPECIFIED, BUT MECHANISM IS NEITHER IN
    # THE LIST NOR IS IT A TERMINAL MECHANISM

                # If:
                #   Mechanism is named or referenced in any specification
                #   or a MonitoredOutputStatesOptions value is in local_specs (i.e., was specified for a Mechanism)
                #   or it is a terminal Mechanism
                elif (node.name in local_specs or node in local_specs or
                              any(isinstance(spec, MonitoredOutputStatesOption) for spec in local_specs) or
                              node in self.get_c_nodes_by_role(CNodeRole.TERMINAL)):
                    #
                    if (not (node.name in local_specs or node in local_specs) and
                            not node in self.get_c_nodes_by_role(CNodeRole.TERMINAL)):
                        continue

                    # If MonitoredOutputStatesOption is PRIMARY_OUTPUT_STATES and OutputState is primary, include it
                    if option_spec is MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES:
                        if output_state is node.output_state:
                            monitored_output_states.append(output_state)
                            continue
                    # If MonitoredOutputStatesOption is ALL_OUTPUT_STATES, include it
                    elif option_spec is MonitoredOutputStatesOption.ALL_OUTPUT_STATES:
                        monitored_output_states.append(output_state)
                    elif node.name in local_specs or node in local_specs:
                        if output_state is node.output_state:
                            monitored_output_states.append(output_state)
                            continue
                    elif option_spec is None:
                        continue
                    else:
                        raise SystemError("PROGRAM ERROR: unrecognized specification of MONITOR_FOR_CONTROL for "
                                       "{0} of {1}".
                                       format(output_state.name, node.name))


        # ASSIGN EXPONENTS, WEIGHTS and MATRICES

        # Get and assign specification of weights, exponents and matrices
        #    for Mechanisms or OutputStates specified in tuples
        output_state_tuples = [MonitoredOutputStateTuple(output_state=item, weight=None, exponent=None, matrix=None)
                               for item in monitored_output_states]
        for spec in all_specs:
            if isinstance(spec, MonitoredOutputStateTuple):
                object_spec = spec.output_state
                # For each OutputState in monitored_output_states
                for i, output_state_tuple in enumerate(output_state_tuples):
                    output_state = output_state_tuple.output_state
                    # If either that OutputState or its owner is the object specified in the tuple
                    if (output_state is object_spec
                        or output_state.name is object_spec
                        or output_state.owner is object_spec):
                        # Assign the weight, exponent and matrix specified in the spec to the output_state_tuple
                        # (can't just assign spec, as its output_state entry may be an unparsed string rather than
                        #  an actual OutputState)
                        output_state_tuples[i] = MonitoredOutputStateTuple(output_state=output_state,
                                                                           weight=spec.weight,
                                                                           exponent=spec.exponent,
                                                                           matrix=spec.matrix)
        return output_state_tuples

    def add_projection(self, projection=None, sender=None, receiver=None, feedback=False, name=None):
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
            projection = MappingProjection(matrix=projection, name=name)
        elif isinstance(projection, str):
            if projection in MATRIX_KEYWORD_VALUES:
                projection = MappingProjection(matrix=projection, name=name)
            else:
                raise CompositionError("Invalid projection ({}) specified for {}.".format(projection, self.name))
        elif isinstance(projection, ModulatoryProjection_Base):
            pass
        elif projection is None:
            projection = MappingProjection(name=name)
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

        subcompositions = []

        sender_mechanism = sender
        graph_sender = sender
        if isinstance(sender, OutputState):
            sender_mechanism = sender.owner
            graph_sender = sender.owner
        elif isinstance(sender, Composition):
            sender_mechanism = sender.output_CIM
            subcompositions.append(sender)

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
            subcompositions.append(receiver)

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
            self.projections.append(projection)

            projection._activate_for_compositions(self)
            for comp in subcompositions:
                projection._activate_for_compositions(comp)

        return projection

    def _add_projection(self, projection):
        self.projections.append(projection)

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

    def _analyze_graph(self, graph=None, context=None):
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

        for node_role_pair in self.required_c_node_roles:
            self._add_c_node_role(node_role_pair[0], node_role_pair[1])

        # TEMPORARY? Disallowing objective mechanisms from having ORIGIN or TERMINAL role in a composition
        if len(self.scheduler_processing.consideration_queue) > 0:
            for node in self.scheduler_processing.consideration_queue[0]:
                if node not in self.get_c_nodes_by_role(CNodeRole.OBJECTIVE):
                    self._add_c_node_role(node, CNodeRole.ORIGIN)
        if len(self.scheduler_processing.consideration_queue) > 0:
            for node in self.scheduler_processing.consideration_queue[-1]:

                if self.model_based_optimizer:
                    if node == self.model_based_optimizer.objective_mechanism:
                        for vertex in graph.get_parents_from_component(node):
                            self._add_c_node_role(vertex.component, CNodeRole.TERMINAL)
                else:
                    self._add_c_node_role(node, CNodeRole.TERMINAL)
        # Identify Origin nodes
        for node in self.c_nodes:
            if graph.get_parents_from_component(node) == []:
                if not isinstance(node, ObjectiveMechanism):
                    self._add_c_node_role(node, CNodeRole.ORIGIN)
            # Identify Terminal nodes
            if graph.get_children_from_component(node) == []:

                if self.model_based_optimizer:
                    if node == self.model_based_optimizer.objective_mechanism:
                        for vertex in graph.get_parents_from_component(node):
                            self._add_c_node_role(vertex.component, CNodeRole.TERMINAL)
                else:
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
            Returns a List of Composition Nodes in this Composition that have the role `role`

            Arguments
            _________

            role : CNodeRole
                the List of nodes having this role to return

            Returns
            -------

            List of Composition Nodes with `CNodeRole` `role` : List(`Mechanisms <Mechanism>` and
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
    #             if not isinstance(timestep[0], collections.Iterable) or isinstance(timestep[0], str):  # Iterable imported from collections
    #                 # If not, embellish the formatting to match the verbose case
    #                 timestep = [timestep]
    #             # Then, check that each input_state is receiving the right size of input
    #             for i, value in enumerate(timestep):
    #                 val_length = len(value)
    #                 state_length = len(mech.input_state.defaults.variable)
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
                    state_length = len(mech.input_state.defaults.value)
                    if val_length != state_length:
                        raise ValueError("The value provided for InputState {!s} of the Mechanism \"{}\" has length "
                                         "{!s} where the InputState takes values of length {!s}".
                                         format(i, mech.name, val_length, state_length))

    # NOTE (CW 9/28): this is mirrored in autodiffcomposition.py, so any changes made here should also be made there
    def _create_CIM_states(self, context=None):

        '''
            - remove the default InputState and OutputState from the CIMs if this is the first time that real
              InputStates and OutputStates are being added to the CIMs

            - for each origin node:
                - if the origin node's external_input_sources specification is True or not listed, create a corresponding
                  InputState and OutputState on the Input CompositionInterfaceMechanism for each "external" InputState
                  of each origin node, and a Projection between the newly created InputCIM OutputState and the origin
                  InputState
                - if the origin node's external_input_sources specification is another origin node, create projections
                  from that other origin node's corresponding InputCIM OutputStates to the current origin node's
                  InputStates. The two nodes must have the same shape.
                - if the origin node's external_input_sources specification is a list, the list must contain only origin
                  nodes and/or InputStates of origin nodes. In this case, an origin node is shorthand for all of the
                  InputStates of that origin node. The concatenation of the values of all of the origin nodes specified
                  in the list must match the shape of the node whose external_input_sources is being specified.

            - create a corresponding InputState and OutputState on the Output CompositionInterfaceMechanism for each
              OutputState of each terminal node, and a Projection between the terminal OutputState and the newly created
              OutputCIM InputState

            - build two dictionaries:

                (1) input_CIM_states = { Origin Node InputState: (InputCIM InputState, InputCIM OutputState) }

                (2) output_CIM_states = { Terminal Node OutputState: (OutputCIM InputState, OutputCIM OutputState) }

            - delete all of the above for any node States which were previously, but are no longer, classified as
              Origin/Terminal

        '''
        origins_changed = False

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
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)

        redirected_inputs = set()
        origin_node_pairs = {}
        for node in origin_nodes:

            if node in self.external_input_sources:
                if self.external_input_sources[node] == True:
                    pass
                elif self.external_input_sources[node] in origin_nodes:
                    redirected_inputs.add(node)
                    continue
                elif isinstance(self.external_input_sources[node], list):
                    valid_spec = True
                    for source in self.external_input_sources[node]:
                        if isinstance(source, (Composition, Mechanism)):
                            if source not in origin_nodes:
                                valid_spec = False
                            elif source in self.external_input_sources:
                                if self.external_input_sources[source] != True:
                                    valid_spec = False
                        elif isinstance(source, InputState):
                            if source.owner not in origin_nodes:
                                valid_spec = False
                            elif source.owner in self.external_input_sources:
                                if self.external_input_sources[source.owner] != True:
                                    valid_spec = False
                    if valid_spec:
                        redirected_inputs.add(node)
                        continue
                    raise CompositionError("External input source ({0}) specified for {1} is not valid. It contains "
                                           "either (1) a source which is not an origin node or an InputState of an "
                                           "origin node, or (2) source which is an origin node (or origin node "
                                           "InputState), but is already borrowing input from yet another origin node."
                                           .format(self.external_input_sources[node], node.name))

                elif self.external_input_sources[node] == ALL:
                    redirected_inputs.add(node)
                    continue
                else:
                    raise CompositionError("External input source ({0}) specified for {1} is not valid. Must be (1) True "
                                           "[the key node is represented on the input_CIM by one or more pairs of "
                                           "states that pass its input value], (2) another origin node [the key node "
                                           "gets its input from another origin node's input_CIM representation], or (3)"
                                           " a list of origin nodes and/or origin node InputStates [the key node gets "
                                           "its input from a mix of other origin nodes' input_CIM representations."
                                           .format(self.external_input_sources[node], node.name))

            for input_state in node.external_input_states:
                # add it to our set of current input states
                current_origin_input_states.add(input_state)

                # if there is not a corresponding CIM output state, add one
                if input_state not in set(self.input_CIM_states.keys()):
                    origins_changed = True
                    interface_input_state = InputState(owner=self.input_CIM,
                                                       variable=input_state.defaults.value,
                                                       reference_value=input_state.defaults.value,
                                                       name="INPUT_CIM_" + node.name + "_" + input_state.name)

                    interface_output_state = OutputState(owner=self.input_CIM,
                                                         variable=OWNER_VALUE,
                                                         default_variable=self.input_CIM.defaults.variable,
                                                         function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                                                         name="INPUT_CIM_" + node.name + "_" + OutputState.__name__)

                    self.input_CIM_states[input_state] = [interface_input_state, interface_output_state]

                    projection = MappingProjection(sender=interface_output_state,
                                      receiver=input_state,
                                      matrix= IDENTITY_MATRIX,
                                      name="("+interface_output_state.name + ") to ("
                                           + input_state.owner.name + "-" + input_state.name+")")
                    projection._activate_for_compositions(self)
                    if isinstance(node, Composition):
                        projection._activate_for_compositions(node)

        new_shadow_projections = {}
        # allow projections from CIM to ANY node listed in external_input_sources
        for node in self.external_input_sources:

            if node not in origin_nodes or node in redirected_inputs:

                cim_rep = self.external_input_sources[node]
                expanded_cim_rep = []
                if isinstance(cim_rep, (Mechanism, Composition)):
                    for state in cim_rep.external_input_states:
                        expanded_cim_rep.append(state)
                elif isinstance(cim_rep, list):
                    for rep in cim_rep:
                        if isinstance(rep, (Mechanism, Composition)):
                            for rep_state in rep.external_input_states:
                                expanded_cim_rep.append(rep_state)
                        elif isinstance(rep, (InputState)):
                            expanded_cim_rep.append(rep)
                        elif rep is None:
                            if node not in redirected_inputs:
                                expanded_cim_rep.append(rep)
                        else:
                            raise CompositionError("Invalid item: {} in external_input_sources specified for {}: {}. Each "
                                                   "items in list must be a Mechanism, Composition, InputStates, or "
                                                   "None.".format(rep, node.name, cim_rep))
                elif cim_rep is ALL:
                    for origin_node in origin_nodes:
                        if origin_node not in redirected_inputs:
                            for state in origin_node.external_input_states:
                                expanded_cim_rep.append(state)
                else:
                    raise CompositionError("Invalid external_input_sources specified for {}: {}. Must be a Mechanism, "
                                           "Composition, or a List of Mechanisms, Compositions, and/or InputStates."
                                           .format(node.name, cim_rep))

                if node in redirected_inputs:
                    # each node in redirected inputs requires an input for each of its input states
                    if len(node.external_input_states) != len(expanded_cim_rep) and len(expanded_cim_rep) > 0:
                        raise CompositionError("The origin source specification of {0} ({1}) has an "
                                               "incompatible number of external input states. {0} has {2} external "
                                               "input states while the origin source specification has a total of {3}."
                                               .format(node.name,
                                                       cim_rep,
                                                           len(node.external_input_states),
                                                           len(expanded_cim_rep)))

                # for non-origin nodes, too few origin sources specified is ok, but too many is not
                if len(expanded_cim_rep) > len(node.external_input_states):
                    raise CompositionError("The origin source specification of {0} ({1}) has too many external input "
                                           "states. {0} has {2} external input states while the origin source "
                                           "specification has a total of {3}."
                                           .format(node.name,
                                                   cim_rep,
                                                   len(node.external_input_states),
                                                   len(expanded_cim_rep)))

                for i in range(len(expanded_cim_rep)):
                    if expanded_cim_rep[i]:
                        new_shadow_projections[(self.input_CIM_states[expanded_cim_rep[i]][1],
                                                node.external_input_states[i])] = None

        for shadow_projection_pair in self.shadow_projections:
            # if this projection is also in new projections, move it to new projections
            if shadow_projection_pair in new_shadow_projections:
                new_shadow_projections[shadow_projection_pair] = self.shadow_projections[shadow_projection_pair]
            # otherwise, it's no longer valid and should be removed from self.projections and self.shadow_projections
            else:
                self.projections.remove(self.shadow_projections[shadow_projection_pair])
                del self.shadow_projections[shadow_projection_pair]

        # for any entirely new shadow_projections, create a MappingProjection object and add to projections
        for output_state, input_state in new_shadow_projections:
            if new_shadow_projections[(output_state, input_state)] is None:
                shadow_projection = MappingProjection(sender=output_state,
                                                      receiver=input_state,
                                                      name="(" + output_state.name + ") to ("
                                                           + input_state.owner.name + "-" + input_state.name + ")")
                self.shadow_projections[(output_state, input_state)] = shadow_projection
                self.projections.append(shadow_projection)
                shadow_projection._activate_for_compositions(self)

        sends_to_input_states = set(self.input_CIM_states.keys())

        # For any states still registered on the CIM that does not map to a corresponding ORIGIN node I.S.:
        for input_state in sends_to_input_states.difference(current_origin_input_states):
            origins_changed = True
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
                                                       variable=output_state.defaults.value,
                                                       reference_value=output_state.defaults.value,
                                                       name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    interface_output_state = OutputState(
                        owner=self.output_CIM,
                        variable=OWNER_VALUE,
                        function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                        reference_value=output_state.defaults.value,
                        name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    self.output_CIM_states[output_state] = [interface_input_state, interface_output_state]

                    proj_name = "(" + output_state.name + ") to (" + interface_input_state.name + ")"

                    proj = MappingProjection(
                        sender=output_state,
                        receiver=interface_input_state,
                        matrix=IDENTITY_MATRIX,
                        name=proj_name
                    )
                    proj._activate_for_compositions(self)
                    self._add_projection(proj)
                    if isinstance(node, Composition):
                        projection._activate_for_compositions(node)

        previous_terminal_output_states = set(self.output_CIM_states.keys())
        for output_state in previous_terminal_output_states.difference(current_terminal_output_states):
            # remove the CIM input and output states associated with this Terminal Node output state
            self.output_CIM.remove_states(self.output_CIM_states[output_state][0])
            self.output_CIM.remove_states(self.output_CIM_states[output_state][1])
            del self.output_CIM_states[output_state]

        # if origins_changed:     # only update prediction mechanisms if the origin node(s) changed
        #     self._instantiate_prediction_mechanisms(context=context)

    def _assign_values_to_input_CIM(self, inputs, execution_id=None):
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
                        value = origin_node.defaults.variable[index]

            build_CIM_input.append(value)

        self.input_CIM.execute(build_CIM_input, execution_id=execution_id)

    def _assign_execution_ids(self, execution_id=None):
        '''
            assigns the same uuid to each Node in the composition's processing graph as well as the CIMs. The uuid is
            either specified in the user's call to run(), or from
            the Composition's **default_execution_id**
        '''

        # Traverse processing graph and assign one uuid to all of its nodes
        if execution_id is None:
            execution_id = self.default_execution_id

        if execution_id not in self.execution_ids:
            self.execution_ids.append(execution_id)

        self._assign_context_values(execution_id=None, composition=self)
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
                        raise CompositionError("Invalid runtime parameter specification ({}) for {}'s {} parameter in {}. "
                                          "Must be a tuple of the form (parameter value, condition), or simply the "
                                          "parameter value. ".format(runtime_params[c_node][param],
                                                                     c_node.name,
                                                                     param,
                                                                     self.name))
                else:
                    runtime_params[c_node][param] = (runtime_params[c_node][param], Always())
        return runtime_params

    def _get_graph_node_label(self, item, show_dimensions=None, show_role=None):

        # For Mechanisms, show length of each InputState and OutputState
        if isinstance(item, (Mechanism, Composition)):
            if show_role:
                try:
                    role = item.systems[self]
                    role = role or ""
                except KeyError:
                    if isinstance(item, ControlMechanism) and hasattr(item, 'system'):
                        role = 'MODEL_BASED_OPTIMIZATION_CONTROL_MECHANISM'
                    else:
                        role = ""
                name = "{}\n[{}]".format(item.name, role)
            else:
                name = item.name
            # TBI Show Dimensions
            # if show_dimensions in {ALL, MECHANISMS}:
            #     input_str = "in ({})".format(",".join(str(input_state.socket_width)
            #                                           for input_state in item.input_states))
            #     output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_state.value)))
            #                                             for output_state in item.output_states))
            #     return "{}\n{}\n{}".format(output_str, name, input_str)
            # else:
            return name

        # TBI: Show projections as nodes
        # For Projection, show dimensions of matrix
        elif isinstance(item, Projection):
            return item.name
        #     if show_dimensions in {ALL, PROJECTIONS}:
        #         # MappingProjections use matrix
        #         if isinstance(item, MappingProjection):
        #             value = np.array(item.matrix)
        #             dim_string = "({})".format("x".join([str(i) for i in value.shape]))
        #             return "{}\n{}".format(item.name, dim_string)
        #         # ModulatoryProjections use value
        #         else:
        #             value = np.array(item.value)
        #             dim_string = "({})".format(len(value))
        #             return "{}\n{}".format(item.name, dim_string)
        #     else:
        #         return item.name

        else:
            raise CompositionError("Unrecognized node type ({}) in graph for {}".format(item, self.name))

    def show_graph(self,
                   show_processes = False,
                   show_learning = False,
                   show_control = False,
                   show_roles = False,
                   show_dimensions = False,
                   show_mechanism_structure=False,
                   show_headers=True,
                   show_projection_labels=False,
                   direction = 'BT',
                   active_items = None,
                   active_color = BOLD,
                   origin_color = 'green',
                   terminal_color = 'red',
                   origin_and_terminal_color = 'brown',
                   learning_color = 'orange',
                   control_color='blue',
                   prediction_mechanism_color='pink',
                   system_color = 'purple',
                   output_fmt='pdf',
                   execution_id=NotImplemented,
                   ):
        """Generate a display of the graph structure of Mechanisms and Projections in the System.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        Displays a graph showing the structure of the System (based on the `System's graph <System.graph>`).
        By default, only the primary processing Components are shown, and Mechanisms are displayed as simple nodes.
        However, the **show_mechanism_structure** argument can be used to display more detailed information about
        each Mechanism, including its States and, optionally, the `function <Component.function>` and `value
        <Component.value>` of the Mechanism and each of its States (using the **show_functions** and **show_values**
        arguments, respectively).  The **show_dimension** argument can be used to display the dimensions of each
        Mechanism and Projection.  The **show_processes** argument arranges Mechanisms and Projections into the
        Processes to which they belong. The **show_learning** and **show_control** arguments can be used to
        show the Components associated with `learning <LearningMechanism>` and those associated with the
        System's `model_based_optimizer <System_Control>`.

        `Mechanisms <Mechanism>` are always displayed as nodes.  If **show_mechanism_structure** is `True`,
        Mechanism nodes are subdivided into sections for its States with information about each determined by the
        **show_values** and **show_functions** specifications.  Otherwise, Mechanism nodes are simple ovals.
        `ORIGIN` and  `TERMINAL` Mechanisms of the System are displayed with thicker borders in a colors specified
        for each. `Projections <Projection>` are displayed as labelled arrows, unless **show_learning** is specified,
        in which case `MappingProjections <MappingProjection> are displayed as diamond-shaped nodes, and any
        `LearningProjections <LearningProjecction>` as labelled arrows that point to them.

        COMMENT:
        node shapes: https://graphviz.gitlab.io/_pages/doc/info/shapes.html
        arrow shapes: https://graphviz.gitlab.io/_pages/doc/info/arrows.html
        colors: https://graphviz.gitlab.io/_pages/doc/info/colors.html
        COMMENT

        .. _System_Projection_Arrow_Corruption:

        .. note::
           There are two unresolved anomalies associated with show_graph (it is uncertain whether they are bugs in
           PsyNeuLink, Graphviz, or an interaction between the two):

           1) When both **show_mechanism_structure** and **show_processes** are specified together with
              **show_learning** and/or **show_control**, under some arcane conditions Projection arrows can be
              distorted and/or orphaned.  We have confirmed that this does not reflect a corruption of the underlying
              graph structure, and the System should execute normally.

           2) Specifying **show_processes** but not setting **show_headers** to `False` raises a GraphViz exception;
              to deal with this, if **show_processes** is specified, **show_headers** is automatically set to `False`.

           COMMENT:
               See IMPLEMENTATION NOTE under _assign_control_components() for description of the problem
           COMMENT

        Examples
        --------

        The figure below shows different renderings of the following System that can be generated using its
        show_graph method::

            import psyneulink as pnl
            mech_1 = pnl.TransferMechanism(name='Mech 1', size=3, output_states=[pnl.RESULTS, pnl.OUTPUT_MEAN])
            mech_2 = pnl.TransferMechanism(name='Mech 2', size=5)
            mech_3 = pnl.TransferMechanism(name='Mech 3', size=2, function=pnl.Logistic(gain=pnl.CONTROL))
            my_process_A = pnl.Process(pathway=[mech_1, mech_3], learning=pnl.ENABLED)
            my_process_B = pnl.Process(pathway=[mech_2, mech_3])
            my_system = pnl.System(processes=[my_process_A, my_process_B],
                                   model_based_optimizer=pnl.ControlMechanism(name='my_system ModelBasedOptimizationControlMechanism'),
                                   monitor_for_control=[(pnl.OUTPUT_MEAN, mech_1)],
                                   enable_model_based_optimizer=True)

        .. _System_show_graph_figure:

        **Output of show_graph using different options**

        .. figure:: _static/show_graph_figure.svg
           :alt: System graph examples
           :scale: 150 %

           Examples of renderings generated by the show_graph method with different options specified, and the call
           to the show_graph method used to generate each rendering shown below each example. **Panel A** shows the
           simplest rendering, with just Processing Components displayed; `ORIGIN` Mechanisms are shown in red,
           and the `TERMINAL` Mechanism in green.  **Panel B** shows the same graph with `MappingProjection` names
           and Component dimensions displayed.  **Panel C** shows the learning Components of the System displayed (in
           orange).  **Panel D** shows the control Components of the System displayed (in blue).  **Panel E** shows
           both learning and control Components;  the learning components are shown with all `LearningProjections
           <LearningProjection>` shown (by specifying show_learning=pnl.ALL).  **Panel F** shows a detailed view of
           the Processing Components, using the show_mechanism_structure option, that includes Component labels and
           values.  **Panel G** show a simpler rendering using the show_mechanism_structure, that only shows
           Component names, but includes the control Components (using the show_control option).


        Arguments
        ---------

        show_processes : bool : False
            specifies whether to organize the `ProcessingMechanisms <ProcessMechanism>` into the `Processes <Process>`
            to which they belong, with each Process shown in its own box.  If a Component belongs to more than one
            Process, it is shown in a separate box along with any others that belong to the same combination of
            Processes;  these represent intersections of Processes within the System.

        show_mechanism_structure : bool, VALUES, FUNCTIONS or ALL : default False
            specifies whether or not to show a detailed representation of each `Mechanism` in the graph, including its
            `States`;  can have the following settings:

            * `True` -- shows States of Mechanism, but not information about the `value
              <Component.value>` or `function <Component.function>` of the Mechanism or its States.

            * *VALUES* -- shows the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <State_Base.value>` of each of its States.

            * *LABELS* -- shows the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <State_Base.value>` of each of its States, using any labels for the values of InputStates and
              OutputStates specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
              `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

            * *FUNCTIONS* -- shows the `function <Mechanism_Base.function>` of the Mechanism and the `function
              <State_Base.function>` of its InputStates and OutputStates.

            * *ROLES* -- shows the `role <System_Mechanisms>` of the Mechanism in the System in square brackets
              (but not any of the other information;  use *ALL* to show ROLES with other information).

            * *ALL* -- shows both `value <Component.value>` and `function <Component.function>` of the Mechanism and
              its States (using labels for the values, if specified;  see above).

            Any combination of the settings above can also be specified in a list that is assigned to
            show_mechanism_structure

        COMMENT:
             and, optionally, the `function <Component.function>` and `value <Component.value>` of each
            (these can be specified using the **show_functions** and **show_values** arguments.  If this option
            is specified, Projections are connected to and from the State that is the `sender <Projection.sender>` or
            `receiver <Projection.receiver>` of each.
        COMMENT

        show_headers : bool : default False
            specifies whether or not to show headers in the subfields of a Mechanism's node;  only takes effect if
            **show_mechanism_structure** is specified (see above).

        COMMENT:
        show_functions : bool : default False
            specifies whether or not to show `function <Component.function>` of Mechanisms and their States in the
            graph (enclosed by parentheses);  this requires **show_mechanism_structure** to be specified as `True`
            to take effect.

        show_values : bool : default False
            specifies whether or not to show `value <Component.value>` of Mechanisms and their States in the graph
            (prefixed by "=");  this requires **show_mechanism_structure** to be specified as `True` to take effect.
        COMMENT

        show_projection_labels : bool : default False
            specifies whether or not to show names of projections.

        show_learning : bool or ALL : default False
            specifies whether or not to show the learning components of the system;
            they will all be displayed in the color specified for **learning_color**.
            Projections that receive a `LearningProjection` will be shown as a diamond-shaped node.
            if set to *ALL*, all Projections associated with learning will be shown:  the LearningProjections
            as well as from `ProcessingMechanisms <ProcessingMechanism>` to `LearningMechanisms <LearningMechanism>`
            that convey error and activation information;  if set to `True`, only the LearningPojections are shown.

        show_control :  bool : default False
            specifies whether or not to show the control components of the system;
            they will all be displayed in the color specified for **control_color**.

        show_roles : bool : default False
            specifies whether or not to include the `role <System_Mechanisms>` that each Mechanism plays in the System
            (enclosed by square brackets); 'ORIGIN' and 'TERMINAL' Mechanisms are also displayed in a color specified
            by the **origin_color**, **terminal_color** and **origin_and_terminal_color** arguments (see below).

        show_dimensions : bool, MECHANISMS, PROJECTIONS or ALL : default False
            specifies whether or not to show dimensions of Mechanisms (and/or MappingProjections when show_learning
            is `True`);  can have the following settings:

            * *MECHANISMS* -- shows `Mechanism` input and output dimensions.  Input dimensions are shown in parentheses
              below the name of the Mechanism; each number represents the dimension of the `variable
              <InputState.variable>` for each `InputState` of the Mechanism; Output dimensions are shown above
              the name of the Mechanism; each number represents the dimension for `value <OutputState.value>` of each
              of `OutputState` of the Mechanism.

            * *PROJECTIONS* -- shows `MappingProjection` `matrix <MappingProjection.matrix>` dimensions.  Each is
              shown in (<dim>x<dim>...) format;  for standard 2x2 "weight" matrix, the first entry is the number of
              rows (input dimension) and the second the number of columns (output dimension).

            * *ALL* -- eqivalent to `True`; shows dimensions for both Mechanisms and Projections (see above for
              formats).

        direction : keyword : default 'BT'
            'BT': bottom to top; 'TB': top to bottom; 'LR': left to right; and 'RL`: right to left.

        active_items : List[Component] : default None
            specifies one or more items in the graph to display in the color specified by *active_color**.

        active_color : keyword : default 'yellow'
            specifies how to highlight the item(s) specified in *active_items**:  either a color recognized
            by GraphViz, or the keyword *BOLD*.

        origin_color : keyword : default 'green',
            specifies the color in which the `ORIGIN` Mechanisms of the System are displayed.

        terminal_color : keyword : default 'red',
            specifies the color in which the `TERMINAL` Mechanisms of the System are displayed.

        origin_and_terminal_color : keyword : default 'brown'
            specifies the color in which Mechanisms that are both
            an `ORIGIN` and a `TERMINAL` of the System are displayed.

        learning_color : keyword : default `green`
            specifies the color in which the learning components are displayed.

        control_color : keyword : default `blue`
            specifies the color in which the learning components are displayed (note: if the System's
            `model_based_optimizer <System.model_based_optimizer>` is an `EVCControlMechanism`, then a link is shown in pink from the
            `prediction Mechanisms <EVCControlMechanism_Prediction_Mechanisms>` it creates to the corresponding
            `ORIGIN` Mechanisms of the System, to indicate that although no projection are created for these,
            the prediction Mechanisms determine the input to the `ORIGIN` Mechanisms when the EVCControlMechanism
            `simulates execution <EVCControlMechanism_Execution>` of the System).

        prediction_mechanism_color : keyword : default `pink`
            specifies the color in which the `prediction_mechanisms
            <EVCControlMechanism.prediction_mechanisms>` are displayed for a System using an `EVCControlMechanism`

        system_color : keyword : default `purple`
            specifies the color in which the node representing input from the System is displayed.

        output_fmt : keyword : default 'pdf'
            'pdf': generate and open a pdf with the visualization;
            'jupyter': return the object (ideal for working in jupyter/ipython notebooks).

        Returns
        -------

        display of system : `pdf` or Graphviz graph object
            'pdf' (placed in current directory) if :keyword:`output_fmt` arg is 'pdf';
            Graphviz graph object if :keyword:`output_fmt` arg is 'jupyter'.

        """

        INITIAL_FRAME = "INITIAL_FRAME"
        ALL = "ALL"

        if execution_id is NotImplemented:
            execution_id = self.default_execution_id

        # if active_item and self.scheduler_processing.clock.time.trial >= self._animate_num_trials:
        #     return

        # IMPLEMENTATION NOTE:
        #    The helper methods below (_assign_XXX__components) all take the main graph *and* subgraph as arguments:
        #        - the main graph (G) is used to assign edges
        #        - the subgraph (sg) is used to assign nodes to Processes if **show_processes** is specified
        #          (otherwise, it should simply be passed G)

        # HELPER METHODS

        tc.typecheck
        def _assign_processing_components(G, sg, rcvr,
                                          processes:tc.optional(list)=None,
                                          subgraphs:tc.optional(dict)=None):
            '''Assign nodes to graph, or subgraph for rcvr in any of the specified **processes** '''

            rcvr_rank = 'same'
            # Set rcvr color and penwidth info
            if rcvr in self.get_c_nodes_by_role(CNodeRole.ORIGIN) and \
                    rcvr in self.get_c_nodes_by_role(CNodeRole.TERMINAL):
                if rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = origin_and_terminal_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    rcvr_color = origin_and_terminal_color
                    rcvr_penwidth = str(bold_width)
            elif rcvr in self.get_c_nodes_by_role(CNodeRole.ORIGIN):
                if rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = origin_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    rcvr_color = origin_color
                    rcvr_penwidth = str(bold_width)
                rcvr_rank = origin_rank
            elif rcvr in self.get_c_nodes_by_role(CNodeRole.TERMINAL):
                if rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = terminal_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    rcvr_color = terminal_color
                    rcvr_penwidth = str(bold_width)
                rcvr_rank = terminal_rank
            elif rcvr in active_items:
                if active_color is BOLD:

                    rcvr_color = default_node_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(default_width + active_thicker_by)
                self.active_item_rendered = True

            else:
                rcvr_color = default_node_color
                rcvr_penwidth = str(default_width)

            # Implement rcvr node
            rcvr_label=self._get_graph_node_label(rcvr, show_dimensions, show_roles)

            if show_mechanism_structure:
                sg.node(rcvr_label,
                        rcvr.show_structure(**mech_struct_args),
                        color=rcvr_color,
                        rank=rcvr_rank,
                        penwidth=rcvr_penwidth)
            else:
                sg.node(rcvr_label,
                        shape=mechanism_shape,
                        color=rcvr_color,
                        rank=rcvr_rank,
                        penwidth=rcvr_penwidth)

            # handle auto-recurrent projections
            for input_state in rcvr.input_states:
                for proj in input_state.path_afferents:
                    if proj.sender.owner is not rcvr:
                        continue
                    if show_mechanism_structure:
                        sndr_proj_label = '{}:{}-{}'.format(rcvr_label, OutputState.__name__, proj.sender.name)
                        proc_mech_rcvr_label = '{}:{}-{}'.format(rcvr_label, InputState.__name__, proj.receiver.name)
                    else:
                        sndr_proj_label = proc_mech_rcvr_label = rcvr_label
                    if show_projection_labels:
                        edge_label = self._get_graph_node_label(proj, show_dimensions, show_roles)
                    else:
                        edge_label = ''
                    try:
                        has_learning = proj.has_learning_projection is not None
                    except AttributeError:
                        has_learning = None

                    # Handle learning components for AutoassociativeProjection
                    #  calls _assign_learning_components,
                    #  but need to manage it from here since MappingProjection needs be shown as node rather than edge

                    # show projection as edge
                    if proj.sender in active_items:
                        if active_color is BOLD:
                            proj_color = default_node_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        proj_color = default_node_color
                        proj_width = str(default_width)
                    G.edge(sndr_proj_label, proc_mech_rcvr_label, label=edge_label,
                           color=proj_color, penwidth=proj_width)

            # # if recvr is ObjectiveMechanism for System's model_based_optimizer, break, as those handled below
            # if isinstance(rcvr, ObjectiveMechanism) and rcvr.for_model_based_optimizer is True:
            #     return

            # loop through senders to implement edges
            sndrs = processing_graph[rcvr]

            for sndr in sndrs:
                if not processes or any(p in processes for p in sndr.processes.keys()):

                    # Set sndr info

                    sndr_label = self._get_graph_node_label(sndr, show_dimensions, show_roles)

                    # find edge name
                    for output_state in sndr.output_states:
                        projs = output_state.efferents
                        for proj in projs:
                            # if proj.receiver.owner == rcvr:
                            if show_mechanism_structure:
                                sndr_proj_label = '{}:{}-{}'.\
                                    format(sndr_label, OutputState.__name__, proj.sender.name)
                                proc_mech_rcvr_label = '{}:{}-{}'.\
                                    format(rcvr_label, proj.receiver.__class__.__name__, proj.receiver.name)
                                    # format(rcvr_label, InputState.__name__, proj.receiver.name)
                            else:
                                sndr_proj_label = sndr_label
                                proc_mech_rcvr_label = rcvr_label
                            # edge_name = self._get_graph_node_label(proj, show_dimensions, show_roles)
                            # edge_shape = proj.matrix.shape
                            try:
                                has_learning = proj.has_learning_projection is not None
                            except AttributeError:
                                has_learning = None
                            selected_proj = proj
                    edge_label = self._get_graph_node_label(proj, show_dimensions, show_roles)

                    # Render projections
                    if any(item in active_items for item in {selected_proj, selected_proj.receiver.owner}):
                        if active_color is BOLD:

                            proj_color = default_node_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True

                    else:
                        proj_color = default_node_color
                        proj_width = str(default_width)
                    proc_mech_label = edge_label

                    # Render Projection normally (as edge)
                    if show_projection_labels:
                        label = proc_mech_label
                    else:
                        label = ''
                    G.edge(sndr_proj_label, proc_mech_rcvr_label, label=label,
                           color=proj_color, penwidth=proj_width)

        def _assign_control_components(G, sg):
            '''Assign control nodes and edges to graph, or subgraph for rcvr in any of the specified **processes** '''

            model_based_optimizer = self.model_based_optimizer
            if model_based_optimizer in active_items:
                if active_color is BOLD:
                    ctlr_color = control_color
                else:
                    ctlr_color = active_color
                ctlr_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                ctlr_color = control_color
                ctlr_width = str(default_width)

            if model_based_optimizer is None:
                print ("\nWARNING: {} has not been assigned a \'model_based_optimizer\', so \'show_control\' option "
                       "can't be used in its show_graph() method\n".format(self.name))
                return

            # get projection from ObjectiveMechanism to ControlMechanism
            objmech_ctlr_proj = model_based_optimizer.input_state.path_afferents[0]
            if model_based_optimizer in active_items:
                if active_color is BOLD:
                    objmech_ctlr_proj_color = control_color
                else:
                    objmech_ctlr_proj_color = active_color
                objmech_ctlr_proj_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                objmech_ctlr_proj_color = control_color
                objmech_ctlr_proj_width = str(default_width)

            # get ObjectiveMechanism
            objmech = objmech_ctlr_proj.sender.owner
            if objmech in active_items:
                if active_color is BOLD:
                    objmech_color = control_color
                else:
                    objmech_color = active_color
                objmech_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                objmech_color = control_color
                objmech_width = str(default_width)

            ctlr_label = self._get_graph_node_label(model_based_optimizer, show_dimensions, show_roles)
            objmech_label = self._get_graph_node_label(objmech, show_dimensions, show_roles)
            if show_mechanism_structure:
                sg.node(ctlr_label,
                        model_based_optimizer.show_structure(**mech_struct_args),
                        color=ctlr_color,
                        penwidth=ctlr_width,
                        rank = control_rank
                       )
                sg.node(objmech_label,
                        objmech.show_structure(**mech_struct_args),
                        color=objmech_color,
                        penwidth=ctlr_width,
                        rank = control_rank
                        )
            else:
                sg.node(ctlr_label,
                        color=ctlr_color, penwidth=ctlr_width, shape=mechanism_shape,
                        rank=control_rank)
                sg.node(objmech_label,
                        color=objmech_color, penwidth=objmech_width, shape=mechanism_shape,
                        rank=control_rank)

            # objmech to model_based_optimizer edge
            if show_projection_labels:
                edge_label = objmech_ctlr_proj.name
            else:
                edge_label = ''
            if show_mechanism_structure:
                obj_to_ctrl_label = objmech_label + ':' + OutputState.__name__ + '-' + objmech_ctlr_proj.sender.name
                ctlr_from_obj_label = ctlr_label + ':' + InputState.__name__ + '-' + objmech_ctlr_proj.receiver.name
            else:
                obj_to_ctrl_label = objmech_label
                ctlr_from_obj_label = ctlr_label
            G.edge(obj_to_ctrl_label, ctlr_from_obj_label, label=edge_label,
                   color=objmech_ctlr_proj_color, penwidth=objmech_ctlr_proj_width)

            # IMPLEMENTATION NOTE:
            #   When two (or more?) Processes (e.g., A and B) have homologous constructions, and a ControlProjection is
            #   assigned to a ProcessingMechanism in one Process (e.g., the 1st one in Process A) and a
            #   ProcessingMechanism in the other Process corresponding to the next in the sequence (e.g., the 2nd one
            #   in Process B) the Projection arrow for the first one get corrupted and sometimes one or more of the
            #   following warning/error messages appear in the console:
            # Warning: Arrow type "arial" unknown - ignoring
            # Warning: Unable to reclaim box space in spline routing for edge "ProcessingMechanism4 ComparatorMechanism
            # [LEARNING]" -> "LearningMechanism for MappingProjection from ProcessingMechanism3 to ProcessingMechanism4
            # [LEARNING]". Something is probably seriously wrong.
            # These do not appear to reflect corruptions of the graph structure and/or execution.

            # outgoing edges (from model_based_optimizer to ProcessingMechanisms)
            for control_signal in model_based_optimizer.control_signals:
                for ctl_proj in control_signal.efferents:
                    proc_mech_label = self._get_graph_node_label(ctl_proj.receiver.owner, show_dimensions, show_roles)
                    if model_based_optimizer in active_items:
                        if active_color is BOLD:
                            ctl_proj_color = control_color
                        else:
                            ctl_proj_color = active_color
                        ctl_proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        ctl_proj_color = control_color
                        ctl_proj_width = str(default_width)
                    if show_projection_labels:
                        edge_label = ctl_proj.name
                    else:
                        edge_label = ''
                    if show_mechanism_structure:
                        ctl_sndr_label = ctlr_label + ':' + OutputState.__name__ + '-' + control_signal.name
                        proc_mech_rcvr_label = \
                            proc_mech_label + ':' + ParameterState.__name__ + '-' + ctl_proj.receiver.name
                    else:
                        ctl_sndr_label = ctlr_label
                        proc_mech_rcvr_label = proc_mech_label
                    G.edge(ctl_sndr_label,
                           proc_mech_rcvr_label,
                           label=edge_label,
                           color=ctl_proj_color,
                           penwidth=ctl_proj_width
                           )

            # incoming edges (from monitored mechs to objective mechanism)
            for input_state in objmech.input_states:
                for projection in input_state.path_afferents:
                    if objmech in active_items:
                        if active_color is BOLD:
                            proj_color = control_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        proj_color = control_color
                        proj_width = str(default_width)
                    if show_mechanism_structure:
                        sndr_proj_label = self._get_graph_node_label(projection.sender.owner, show_dimensions, show_roles) +\
                                          ':' + OutputState.__name__ + '-' + projection.sender.name
                        objmech_proj_label = objmech_label + ':' + InputState.__name__ + '-' + input_state.name
                    else:
                        sndr_proj_label = self._get_graph_node_label(projection.sender.owner, show_dimensions, show_roles)
                        objmech_proj_label = self._get_graph_node_label(objmech, show_dimensions, show_roles)
                    if show_projection_labels:
                        edge_label = projection.name
                    else:
                        edge_label = ''
                    G.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                           color=proj_color, penwidth=proj_width)

            # prediction mechanisms
            for mech in self.execution_list:
                if mech in active_items:
                    if active_color is BOLD:
                        pred_mech_color = prediction_mechanism_color
                    else:
                        pred_mech_color = active_color
                    pred_mech_width = str(default_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    pred_mech_color = prediction_mechanism_color
                    pred_mech_width = str(default_width)
                if mech._role is CONTROL and hasattr(mech, 'origin_mech'):
                    recvr = mech.origin_mech
                    recvr_label = self._get_graph_node_label(recvr, show_dimensions, show_roles)
                    # IMPLEMENTATION NOTE:
                    #     THIS IS HERE FOR FUTURE COMPATIBILITY WITH FULL IMPLEMENTATION OF PredictionMechanisms
                    if show_mechanism_structure and False:
                        proj = mech.output_state.efferents[0]
                        if proj in active_items:
                            if active_color is BOLD:
                                pred_proj_color = prediction_mechanism_color
                            else:
                                pred_proj_color = active_color
                            pred_proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            pred_proj_color = prediction_mechanism_color
                            pred_proj_width = str(default_width)
                        sg.node(mech.name,
                                shape=mech.show_structure(**mech_struct_args),
                                color=pred_mech_color,
                                penwidth=pred_mech_width)

                        G.edge(mech.name + ':' + OutputState.__name__ + '-' + mech.output_state.name,
                               recvr_label + ':' + InputState.__name__ + '-' + proj.receiver.name,
                               label=' prediction assignment',
                               color=pred_proj_color, penwidth=pred_proj_width)
                    else:
                        sg.node(self._get_graph_node_label(mech, show_dimensions, show_roles),
                                color=pred_mech_color, shape=mechanism_shape, penwidth=pred_mech_width)
                        G.edge(self._get_graph_node_label(mech, show_dimensions, show_roles),
                               recvr_label,
                               label=' prediction assignment',
                               color=prediction_mechanism_color)

        # MAIN BODY OF METHOD:

        import graphviz as gv

        self._analyze_graph()

        if show_dimensions == True:
            show_dimensions = ALL
        if show_processes:
            show_headers = False

        if not active_items:
            active_items = []
        elif active_items is INITIAL_FRAME:
            active_items = [INITIAL_FRAME]
        elif not isinstance(active_items, collections.Iterable):
            active_items = [active_items]
        elif not isinstance(active_items, list):
            active_items = list(active_items)
        for item in active_items:
            if not isinstance(item, Component) and item is not INITIAL_FRAME:
                raise CompositionError("PROGRAM ERROR: Item ({}) specified in {} argument for {} method of {} is not a {}".
                                  format(item, repr('active_items'), repr('show_graph'), self.name, Component.__name__))

        self.active_item_rendered = False

        # Argument values used to call Mechanism.show_structure()
        if isinstance(show_mechanism_structure, (list, tuple, set)):
            mech_struct_args = {'system':self,
                                'show_role':any(key in show_mechanism_structure for key in {ROLES, ALL}),
                                'show_functions':any(key in show_mechanism_structure for key in {FUNCTIONS, ALL}),
                                'show_values':any(key in show_mechanism_structure for key in {VALUES, ALL}),
                                'use_labels':any(key in show_mechanism_structure for key in {LABELS, ALL}),
                                'show_headers':show_headers,
                                'output_fmt':'struct'}
        else:
            mech_struct_args = {'system':self,
                                'show_role':show_mechanism_structure in {ROLES, ALL},
                                'show_functions':show_mechanism_structure in {FUNCTIONS, ALL},
                                'show_values':show_mechanism_structure in {VALUES, LABELS, ALL},
                                'use_labels':show_mechanism_structure in {LABELS, ALL},
                                'show_headers':show_headers,
                                'output_fmt':'struct'}

        default_node_color = 'black'
        mechanism_shape = 'oval'
        projection_shape = 'diamond'
        # projection_shape = 'point'
        # projection_shape = 'Mdiamond'
        # projection_shape = 'hexagon'

        bold_width = 3
        default_width = 1
        active_thicker_by = 2

        pos = None

        origin_rank = 'source'
        control_rank = 'min'
        obj_mech_rank = 'sink'
        terminal_rank = 'max'
        learning_rank = 'sink'

        # build graph and configure visualisation settings
        G = gv.Digraph(
                name = self.name,
                engine = "dot",
                # engine = "fdp",
                # engine = "neato",
                # engine = "circo",
                node_attr  = {
                    'fontsize':'12',
                    'fontname':'arial',
                    # 'shape':mechanism_shape,
                    'shape':'record',
                    'color':default_node_color,
                    'penwidth':str(default_width)
                },
                edge_attr  = {
                    # 'arrowhead':'halfopen',
                    'fontsize': '10',
                    'fontname': 'arial'
                },
                graph_attr = {
                    "rankdir" : direction,
                    'overlap' : "False"
                },
        )
        # G.attr(compound = 'True')

        processing_graph = self.scheduler_processing.dependency_sets
        # get System's ProcessingMechanisms
        rcvrs = list(processing_graph.keys())

        # if show_processes is specified, create subgraphs for each Process
        if show_processes:

            # Manage Processes
            process_intersections = {}
            subgraphs = {}  # Entries: Process:sg
            for process in self.processes:
                subgraph_name = 'cluster_'+process.name
                subgraph_label = process.name
                with G.subgraph(name=subgraph_name) as sg:
                    subgraphs[process.name]=sg
                    sg.attr(label=subgraph_label)
                    sg.attr(rank = 'same')
                    # sg.attr(style='filled')
                    # sg.attr(color='lightgrey')

                    # loop through receivers and assign to the subgraph any that belong to the current Process
                    for r in rcvrs:
                        intersection = [p for p in self.processes if p in r.processes]
                        # If the rcvr is in only one Process, add it to the subgraph for that Process
                        if len(intersection)==1:
                            # If the rcvr is in the current Process, assign it to the subgraph
                            if process in intersection:
                                _assign_processing_components(G, sg, r, [process])
                        # Otherwise, assign rcvr to entry in dict for process intersection (subgraph is created below)
                        else:
                            intersection_name = ' and '.join([p.name for p in intersection])
                            if not intersection_name in process_intersections:
                                process_intersections[intersection_name] = [r]
                            else:
                                if r not in process_intersections[intersection_name]:
                                    process_intersections[intersection_name].append(r)

            # Create a process for each unique intersection and assign rcvrs to that
            for intersection_name, mech_list in process_intersections.items():
                with G.subgraph(name='cluster_'+intersection_name) as sg:
                    sg.attr(label=intersection_name)
                    # get list of processes in the intersection (to pass to _assign_processing_components)
                    processes = [p for p in self.processes if p.name in intersection_name]
                    # loop through receivers and assign to the subgraph any that belong to the current Process
                    for r in mech_list:
                        if r in self.graph:
                            _assign_processing_components(G, sg, r, processes, subgraphs)
                        else:
                            raise CompositionError("PROGRAM ERROR: Component in interaction process ({}) is not in "
                                              "{}'s graph or learningGraph".format(r.name, self.name))

        else:
            for r in rcvrs:
                _assign_processing_components(G, G, r)

        # Add control-related Components to graph if show_control
        if show_control:
            if show_processes:
                with G.subgraph(name='cluster_MODEL_BASED_OPTIMIZATION_CONTROL_MECHANISM') as sg:
                    sg.attr(label='MODEL_BASED_OPTIMIZATION_CONTROL_MECHANISM')
                    sg.attr(rank='top')
                    # sg.attr(style='filled')
                    # sg.attr(color='lightgrey')
                    _assign_control_components(G, sg)
            else:
                _assign_control_components(G, G)

        # GENERATE OUTPUT

        # Show as pdf
        if output_fmt == 'pdf':
            # G.format = 'svg'
            G.view(self.name.replace(" ", "-"), cleanup=True, directory='show_graph OUTPUT/PDFS')

        # Generate images for animation
        elif output_fmt == 'gif':
            if self.active_item_rendered or INITIAL_FRAME in active_items:
                G.format = 'gif'
                execution_phase = self.parameters.context.get(execution_id).execution_phase
                if INITIAL_FRAME in active_items:
                    time_string = ''
                    phase_string = ''
                elif execution_phase == ContextFlags.PROCESSING:
                    # time_string = repr(self.scheduler_processing.clock.simple_time)
                    time = self.scheduler_processing.get_clock(execution_id).time
                    time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}".\
                        format(time.run, time.trial, time.pass_, time.time_step)
                    phase_string = 'Processing Phase - '
                elif execution_phase == ContextFlags.LEARNING:
                    time = self.scheduler_learning.get_clock(execution_id).time
                    time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}".\
                        format(time.run, time.trial, time.pass_, time.time_step)
                    phase_string = 'Learning Phase - '
                elif execution_phase == ContextFlags.CONTROL:
                    time_string = ''
                    phase_string = 'Control phase'
                else:
                    raise CompositionError("PROGRAM ERROR:  Unrecognized phase during execution of {}".format(self.name))
                label = '\n{}\n{}{}\n'.format(self.name, phase_string, time_string)
                G.attr(label=label)
                G.attr(labelloc='b')
                G.attr(fontname='Helvetica')
                G.attr(fontsize='14')
                if INITIAL_FRAME in active_items:
                    index = '-'
                else:
                    index = repr(self._component_execution_count)
                image_filename = repr(self.scheduler_processing.clock.simple_time.trial) + '-' + index + '-'
                image_file = self._animate_directory + '/' + image_filename + '.gif'
                G.render(filename = image_filename,
                         directory=self._animate_directory,
                         cleanup=True,
                         # view=True
                         )
                # Append gif to self._animation
                image = Image.open(image_file)
                if not self._save_images:
                    remove(image_file)
                if not hasattr(self, '_animation'):
                    self._animation = [image]
                else:
                    self._animation.append(image)

        # Return graph to show in jupyter
        elif output_fmt == 'jupyter':
            return G

    def execute(
        self,
        inputs=None,
        autodiff_stimuli=None,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        call_before_time_step=None,
        call_before_pass=None,
        call_after_time_step=None,
        call_after_pass=None,
        execution_id=None,
        base_execution_id=None,
        clamp_input=SOFT_CLAMP,
        targets=None,
        runtime_params=None,
        bin_execute=False,
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

            execution_id
                execution_id will be set to self.default_execution_id if unspecified

            base_execution_id
                the execution_id corresponding to the execution context from which this execution will be initialized, if
                values currently do not exist for **execution_id**

            call_before_time_step : callable
                will be called before each `TIME_STEP` is executed
                will be passed the current *execution_id* (but it is not necessary for your callable to take)

            call_after_time_step : callable
                will be called after each `TIME_STEP` is executed
                will be passed the current *execution_id* (but it is not necessary for your callable to take)

            call_before_pass : callable
                will be called before each `PASS` is executed
                will be passed the current *execution_id* (but it is not necessary for your callable to take)

            call_after_pass : callable
                will be called after each `PASS` is executed
                will be passed the current *execution_id* (but it is not necessary for your callable to take)

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

        # initialize from base context but don't overwrite any values already set for this execution_id
        if not nested:
            self._initialize_from_context(execution_id, base_execution_id, override=False)

        self._assign_context_values(execution_id, composition=self)

        if nested:
            self.input_CIM.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING
            self.input_CIM.execute(execution_id=execution_id, context=ContextFlags.PROCESSING)

        else:
            inputs = self._adjust_execution_stimuli(inputs)
            self._assign_values_to_input_CIM(inputs, execution_id=execution_id)

        if termination_processing is None:
            termination_processing = self.termination_processing

        # if hasattr(self, "prediction_mechanisms"):
        #     self._execute_prediction_mechanisms(context=context)

        next_pass_before = 1
        next_pass_after = 1
        if clamp_input:
            soft_clamp_inputs = self._identify_clamp_inputs(SOFT_CLAMP, clamp_input, origin_nodes)
            hard_clamp_inputs = self._identify_clamp_inputs(HARD_CLAMP, clamp_input, origin_nodes)
            pulse_clamp_inputs = self._identify_clamp_inputs(PULSE_CLAMP, clamp_input, origin_nodes)
            no_clamp_inputs = self._identify_clamp_inputs(NO_CLAMP, clamp_input, origin_nodes)
        # run scheduler to receive sets of nodes that may be executed at this time step in any order
        execution_scheduler = scheduler_processing

        if bin_execute == 'Python':
            bin_execute = False

        if bin_execute:
            try:
                if str(bin_execute).endswith('Exec'):
                    if bin_execute.startswith('LLVM'):
                        _comp_ex = pnlvm.CompExecution(self, [execution_id])
                        _comp_ex.execute(inputs)
                        return _comp_ex.extract_node_output(self.output_CIM)
                    elif bin_execute.startswith('PTX'):
                        self.__ptx_initialize(execution_id)
                        __execution = self._compilation_data.ptx_execution.get(execution_id)
                        __execution.cuda_execute(inputs)
                        return __execution.extract_node_output(self.output_CIM)

                # Filter out mechanisms. Nested nodes are not executed in this mode
                mechanisms = [n for n in self._all_nodes if isinstance(n, Mechanism)]
                # Generate all mechanism wrappers
                for m in mechanisms:
                    self._get_node_wrapper(m)
                # Compile all mechanism wrappers
                for m in mechanisms:
                    self._get_bin_mechanism(m)

                bin_execute = True
            except Exception as e:
                if bin_execute[:4] == 'LLVM':
                    raise e

                string = "Failed to compile wrapper for `{}' in `{}': {}".format(m.name, self.name, str(e))
                print("WARNING: {}".format(string))
                bin_execute = False

        if bin_execute:
            _comp_ex = pnlvm.CompExecution(self, [execution_id])
            _comp_ex.execute_node(self.input_CIM, inputs)

        if call_before_pass:
            call_with_pruned_args(call_before_pass, execution_context=execution_id)

        for next_execution_set in execution_scheduler.run(termination_conds=termination_processing, execution_id=execution_id):
            if call_after_pass:
                if next_pass_after == execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL):
                    logger.debug('next_pass_after {0}\tscheduler pass {1}'.format(next_pass_after, execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL)))
                    call_with_pruned_args(call_after_pass, execution_context=execution_id)
                    next_pass_after += 1

            if call_before_pass:
                if next_pass_before == execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL):
                    call_with_pruned_args(call_before_pass, execution_context=execution_id)
                    logger.debug('next_pass_before {0}\tscheduler pass {1}'.format(next_pass_before, execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS, TimeScale.TRIAL)))
                    next_pass_before += 1

            if call_before_time_step:
                call_with_pruned_args(call_before_time_step, execution_context=execution_id)

            frozen_values = {}
            new_values = {}
            if bin_execute:
                _comp_ex.freeze_values()

            # execute each node with EXECUTING in context
            for node in next_execution_set:
                frozen_values[node] = node.get_output_values(execution_id)
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
                                node.recurrent_projection.sender.parameters.value.set([0.0], execution_id, override=True)
                        elif node in no_clamp_inputs:
                            for input_state in node.input_states:
                                self.input_CIM_states[input_state][1].parameters.value.set(0.0, execution_id, override=True)
                            # self.input_mechanisms[mechanism]._output_states[0].value = 0.0

                if isinstance(node, Mechanism):

                    execution_runtime_params = {}

                    if node in runtime_params:
                        for param in runtime_params[node]:
                            if runtime_params[node][param][1].is_satisfied(scheduler=execution_scheduler,
                                                                           # KAM 5/15/18 - not sure if this will always be the correct execution id:
                                                                                execution_context=execution_id):
                                execution_runtime_params[param] = runtime_params[node][param][0]

                    node.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING

                    if bin_execute:
                        _comp_ex.execute_node(node)
                        node.parameters.context.get(execution_id).execution_phase = ContextFlags.IDLE
                    else:
                        if node is not self.model_based_optimizer:
                            node.execute(
                                execution_id=execution_id,
                                runtime_params=execution_runtime_params,
                                context=ContextFlags.COMPOSITION
                            )

                        if execution_id in node._runtime_params_reset:
                            for key in node._runtime_params_reset[execution_id]:
                                node._set_parameter_value(key, node._runtime_params_reset[execution_id][key], execution_id)
                        node._runtime_params_reset[execution_id] = {}

                        if execution_id in node.function._runtime_params_reset:
                            for key in node.function._runtime_params_reset[execution_id]:
                                node.function._set_parameter_value(
                                    key,
                                    node.function._runtime_params_reset[execution_id][key],
                                    execution_id
                                )
                        node.function._runtime_params_reset[execution_id] = {}
                        node.parameters.context.get(execution_id).execution_phase = ContextFlags.IDLE

                elif isinstance(node, Composition):
                    if bin_execute:
                        # Values of node with compiled wrappers are
                        # in binary data structure
                        srcs = (proj.sender.owner for proj in node.input_CIM.afferents if proj.sender.owner in self.__generated_wrappers)
                        for src_node in srcs:
                            assert src_node in self.c_nodes or src_node is self.input_CIM
                            data = _comp_ex.extract_frozen_node_output(src_node)
                            for i, v in enumerate(data):
                                #This sets frozen values
                                src_node.output_states[i].parameters.value.set(v, execution_id, skip_history=True, skip_log=True, override=True)

                    node._assign_context_values(execution_id, composition=node)

                    # autodiff compositions must be passed extra inputs
                    learning_enabled = False
                    if hasattr(node, "pytorch_representation"):
                        if node.learning_enabled:
                            learning_enabled = True
                    if learning_enabled:
                        ret = node.execute(inputs=autodiff_stimuli[node],
                                     execution_id=execution_id,
                                     context=ContextFlags.COMPOSITION)
                    else:
                        ret = node.execute(execution_id=execution_id,
                                     context=ContextFlags.COMPOSITION)
                    if bin_execute:
                        # Update result in binary data structure
                        _comp_ex.insert_node_output(node, ret)
                        for i, v in enumerate(ret):
                            # Set current output. This will be stored to "new_values" below
                            node.output_CIM.output_states[i].parameters.value.set(v, execution_id, skip_history=True, skip_log=True, override=True)

                if node in origin_nodes:
                    if clamp_input:
                        if node in pulse_clamp_inputs:
                            for input_state in node.input_states:
                            # clamp = None --> "turn off" input node
                                self.input_CIM_states[input_state][1].parameters.value.set(0, execution_id, override=True)
                new_values[node] = node.get_output_values(execution_id)

                for i in range(len(node.output_states)):
                    node.output_states[i].parameters.value.set(frozen_values[node][i], execution_id, override=True, skip_history=True, skip_log=True)

            for node in next_execution_set:

                for i in range(len(node.output_states)):
                    node.output_states[i].parameters.value.set(new_values[node][i], execution_id, override=True, skip_history=True, skip_log=True)

            if call_after_time_step:
                call_with_pruned_args(call_after_time_step, execution_context=execution_id)

        if call_after_pass:
            call_with_pruned_args(call_after_pass, execution_context=execution_id)

        # extract result here
        if bin_execute:
            _comp_ex.freeze_values()
            _comp_ex.execute_node(self.output_CIM)
            return _comp_ex.extract_node_output(self.output_CIM)

        # control phase
        execution_phase = self.parameters.context.get(execution_id).execution_phase
        if (
            execution_phase != ContextFlags.INITIALIZING
            and execution_phase != ContextFlags.SIMULATION
            and self.enable_model_based_optimizer
        ):
            if self.model_based_optimizer:
                self.model_based_optimizer.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING
                control_allocation = self.model_based_optimizer.execute(execution_id=execution_id, context=context)
                self.model_based_optimizer.apply_control_allocation(control_allocation, execution_id=execution_id, runtime_params=runtime_params, context=context)

        self.output_CIM.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING
        self.output_CIM.execute(execution_id=execution_id, context=ContextFlags.PROCESSING)

        output_values = []
        for i in range(0, len(self.output_CIM.output_states)):
            output_values.append(self.output_CIM.output_states[i].parameters.value.get(execution_id))

        return output_values

    def reinitialize(self, values, execution_context=NotImplemented):
        if execution_context is NotImplemented:
            execution_context = self.default_execution_id

        for i in range(self.stateful_nodes):
            self.stateful_nodes[i].reinitialize(values[i], execution_context=execution_context)

    def run(
        self,
        inputs=None,
        scheduler_processing=None,
        scheduler_learning=None,
        termination_processing=None,
        termination_learning=None,
        execution_id=None,
        base_execution_id=None,
        num_trials=None,
        call_before_time_step=None,
        call_after_time_step=None,
        call_before_pass=None,
        call_after_pass=None,
        call_before_trial=None,
        call_after_trial=None,
        clamp_input=SOFT_CLAMP,
        targets=None,
        bin_execute=False,
        initial_values=None,
        reinitialize_values=None,
        runtime_params=None,
        context=None
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

            execution_id
                execution_id will be set to self.default_execution_id if unspecified

            base_execution_id
                the execution_id corresponding to the execution context from which this execution will be initialized, if
                values currently do not exist for **execution_id**

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

            runtime_params : Dict[Node: Dict[Parameter: Tuple(Value, Condition)]]
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

        if reinitialize_values is None:
            reinitialize_values = {}

        for node in reinitialize_values:
            node.reinitialize(*reinitialize_values[node], execution_context=execution_id)

        try:
            if self.parameters.context.get(execution_id).execution_phase != ContextFlags.SIMULATION:
                self._analyze_graph()
        except AttributeError:
            # if context is None, it has not been created for this execution_id yet, so it is not
            # in a simulation
            self._analyze_graph()

        results = []

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

        inputs, num_inputs_sets, autodiff_stimuli = self._adjust_stimulus_dict(inputs)

        if num_trials is not None:
            num_trials = num_trials
        else:
            num_trials = num_inputs_sets

        if targets is None:
            targets = {}

        scheduler_processing._reset_counts_total(TimeScale.RUN, execution_id)

        if str(bin_execute).endswith('Run'):
            # initialize from base context but don't overwrite any values already set for this execution_id
            self._initialize_from_context(execution_id, base_execution_id, override=False)
            self._assign_context_values(execution_id, composition=self)

            if bin_execute.startswith('LLVM'):
                _comp_ex = pnlvm.CompExecution(self, [execution_id])
                results += _comp_ex.run(inputs, num_trials, num_inputs_sets)
            elif bin_execute.startswith('PTX'):
                self.__ptx_initialize(execution_id)
                EX = self._compilation_data.ptx_execution.get(execution_id)
                results += EX.cuda_run(inputs, num_trials, num_inputs_sets)

            full_results = self.parameters.results.get(execution_id)
            if full_results is None:
                full_results = results
            else:
                full_results.extend(results)

            self.parameters.results.set(full_results, execution_id)
            # KAM added the [-1] index after changing Composition run()
            # behavior to return only last trial of run (11/7/18)
            self.most_recent_execution_context = execution_id
            return full_results[-1]

        # --- RESET FOR NEXT TRIAL ---
        # by looping over the length of the list of inputs - each input represents a TRIAL
        for trial_num in range(num_trials):
            # Execute call before trial "hook" (user defined function)
            if call_before_trial:
                call_with_pruned_args(call_before_trial, execution_context=execution_id)

            if termination_processing[TimeScale.RUN].is_satisfied(scheduler=scheduler_processing,
                                                                  execution_context=execution_id):
                break
        # PROCESSING ------------------------------------------------------------------------

            # Prepare stimuli from the outside world  -- collect the inputs for this TRIAL and store them in a dict
            execution_stimuli = {}
            stimulus_index = trial_num % num_inputs_sets
            for node in inputs:
                if len(inputs[node]) == 1:
                    execution_stimuli[node] = inputs[node][0]
                    continue
                execution_stimuli[node] = inputs[node][stimulus_index]

            execution_autodiff_stimuli = {}
            for node in autodiff_stimuli:
                if isinstance(autodiff_stimuli[node], list):
                    execution_autodiff_stimuli[node] = autodiff_stimuli[node][stimulus_index]
                else:
                    execution_autodiff_stimuli[node] = autodiff_stimuli[node]

            # execute processing
            # pass along the stimuli for this trial
            trial_output = self.execute(inputs=execution_stimuli,
                                        autodiff_stimuli=execution_autodiff_stimuli,
                                        scheduler_processing=scheduler_processing,
                                        scheduler_learning=scheduler_learning,
                                        termination_processing=termination_processing,
                                        termination_learning=termination_learning,
                                        call_before_time_step=call_before_time_step,
                                        call_before_pass=call_before_pass,
                                        call_after_time_step=call_after_time_step,
                                        call_after_pass=call_after_pass,
                                        execution_id=execution_id,
                                        base_execution_id=base_execution_id,
                                        clamp_input=clamp_input,
                                        runtime_params=runtime_params,
                                        bin_execute=bin_execute)

        # ---------------------------------------------------------------------------------
            # store the result of this execute in case it will be the final result

            # terminal_mechanisms = self.get_c_nodes_by_role(CNodeRole.TERMINAL)
            # for terminal_mechanism in terminal_mechanisms:
            #     for terminal_output_state in terminal_mechanism.output_states:
            #         CIM_output_state = self.output_CIM_states[terminal_output_state]
            #         CIM_output_state.value = terminal_output_state.value

            # object.results.append(result)
            if isinstance(trial_output, collections.Iterable):
                result_copy = trial_output.copy()
            else:
                result_copy = trial_output

            if self.parameters.context.get(execution_id).execution_phase != ContextFlags.SIMULATION:
                results.append(result_copy)

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

            if call_after_trial:
                call_with_pruned_args(call_after_trial, execution_context=execution_id)

        scheduler_processing.clocks[execution_id]._increment_time(TimeScale.RUN)

        full_results = self.parameters.results.get(execution_id)
        if full_results is None:
            full_results = results
        else:
            full_results.extend(results)

        self.parameters.results.set(full_results, execution_id)

        self.most_recent_execution_context = execution_id
        return trial_output

    # def save_state(self):
    #     saved_state = {}
    #     for node in self.stateful_nodes:
    #         # "save" the current state of each stateful mechanism by storing the values of each of its stateful
    #         # attributes in the reinitialization_values dictionary; this gets passed into run and used to call
    #         # the reinitialize method on each stateful mechanism.
    #         reinitialization_value = []
    #
    #         if isinstance(node, Composition):
    #             # TBI: Store state for a Composition, Reinitialize Composition
    #             pass
    #         elif isinstance(node, Mechanism):
    #             if isinstance(node.function, IntegratorFunction):
    #                 for attr in node.function.stateful_attributes:
    #                     reinitialization_value.append(getattr(node.function, attr))
    #             elif hasattr(node, "integrator_function"):
    #                 if isinstance(node.integrator_function, IntegratorFunction):
    #                     for attr in node.integrator_function.stateful_attributes:
    #                         reinitialization_value.append(getattr(node.integrator_function, attr))
    #
    #         saved_state[node] = reinitialization_value
    #
    #     node_values = {}
    #     for node in self.c_nodes:
    #         node_values[node] = (node.value, node.output_values)
    #
    #     self.sim_reinitialize_values, self.sim_node_values = saved_state, node_values
    #     return saved_state, node_values

    # def _get_predicted_input(self, execution_id=None, context=None):
    #     """
    #     Called by the `model_based_optimizer <Composition.model_based_optimizer>` of the `Composition` before any
    #     simulations are run in order to (1) generate predicted inputs, (2) store current values that must be reinstated
    #     after all simulations are complete, and (3) set the number of trials of simulations.
    #     """
    #
    #     predicted_input = self._update_predicted_input(execution_id=execution_id)
    #
    #     return predicted_input

    def _after_agent_rep_execution(self, context=None):
        pass

    @property
    def _all_nodes(self):
        for n in self.c_nodes:
            yield n
        yield self.input_CIM
        yield self.output_CIM

    def _get_param_struct_type(self, ctx):
        mech_param_type_list = (ctx.get_param_struct_type(m) for m in self._all_nodes)
        proj_param_type_list = (ctx.get_param_struct_type(p) for p in self.projections)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(mech_param_type_list),
            pnlvm.ir.LiteralStructType(proj_param_type_list)))

    def _get_context_struct_type(self, ctx):
        mech_ctx_type_list = (ctx.get_context_struct_type(m) for m in self._all_nodes)
        proj_ctx_type_list = (ctx.get_context_struct_type(p) for p in self.projections)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(mech_ctx_type_list),
            pnlvm.ir.LiteralStructType(proj_ctx_type_list)))

    def _get_input_struct_type(self, ctx):
        return ctx.get_input_struct_type(self.input_CIM)

    def _get_output_struct_type(self, ctx):
        return ctx.get_output_struct_type(self.output_CIM)

    def _get_data_struct_type(self, ctx):
        output_type_list = (ctx.get_output_struct_type(m) for m in self._all_nodes)

        data = [pnlvm.ir.LiteralStructType(output_type_list)]
        for node in self.c_nodes:
            nested_data = ctx.get_data_struct_type(node)
            data.append(nested_data)
        return pnlvm.ir.LiteralStructType(data)

    def _get_context_initializer(self, execution_id=None):
        mech_contexts = (tuple(m._get_context_initializer(execution_id=execution_id)) for m in self._all_nodes)
        proj_contexts = (tuple(p._get_context_initializer(execution_id=execution_id)) for p in self.projections)
        return (tuple(mech_contexts), tuple(proj_contexts))

    def _get_param_initializer(self, execution_id):
        mech_params = (tuple(m._get_param_initializer(execution_id)) for m in self._all_nodes)
        proj_params = (tuple(p._get_param_initializer(execution_id)) for p in self.projections)
        return (tuple(mech_params), tuple(proj_params))

    def _get_data_initializer(self, execution_id=None):
        output = ((os.parameters.value.get(execution_id) for os in m.output_states) for m in self._all_nodes)
        data = [output]
        for node in self.c_nodes:
            nested_data = node._get_data_initializer(execution_id=execution_id) if hasattr(node, '_get_data_initializer') else []
            data.append(nested_data)
        return pnlvm._tupleize(data)

    def _get_node_index(self, node):
        return list(self._all_nodes).index(node)

    def _get_node_wrapper(self, node):
        if node not in self.__generated_wrappers:
            wrapper = self.__gen_node_wrapper(node)
            self.__generated_wrappers[node] = wrapper
            return wrapper

        return self.__generated_wrappers[node]

    def _get_bin_mechanism(self, mechanism):
        if mechanism not in self.__compiled_mech:
            wrapper = self._get_node_wrapper(mechanism)
            bin_f = pnlvm.LLVMBinaryFunction.get(wrapper)
            self.__compiled_mech[mechanism] = bin_f
            return bin_f

        return self.__compiled_mech[mechanism]

    def _get_execution_wrapper(self):
        if self.__generated_execution is None:
            self.__generated_execution = self.__gen_exec_wrapper()

        return self.__generated_execution

    def _get_bin_execution(self):
        if self.__compiled_execution is None:
            wrapper = self._get_execution_wrapper()
            bin_f = pnlvm.LLVMBinaryFunction.get(wrapper)
            self.__compiled_execution = bin_f

        return self.__compiled_execution

    def _get_bin_run(self):
        if self.__compiled_run is None:
            wrapper = self.__gen_run_wrapper()
            bin_f = pnlvm.LLVMBinaryFunction.get(wrapper)
            self.__compiled_run = bin_f

        return self.__compiled_run

    def reinitialize(self, execution_context=NotImplemented):
        if execution_context is NotImplemented:
            execution_context = self.default_execution_id

        self._compilation_data.ptx_execution.set(None, execution_context)
        self._compilation_data.parameter_struct.set(None, execution_context)
        self._compilation_data.context_struct.set(None, execution_context)
        self._compilation_data.data_struct.set(None, execution_context)
        self._compilation_data.scheduler_conditions.set(None, execution_context)

    def __ptx_initialize(self, execution_id=None):
        if self._compilation_data.ptx_execution.get(execution_id) is None:
            self._compilation_data.ptx_execution.set(pnlvm.CompExecution(self, [execution_id]), execution_id)

    def __gen_node_wrapper(self, node):
        is_mech = isinstance(node, Mechanism)

        func_name = None
        with pnlvm.LLVMBuilderContext() as ctx:
            func_name = ctx.get_unique_name("comp_wrap_" + node.name)
            data_struct_ptr = ctx.get_data_struct_type(self).as_pointer()
            args = [
                ctx.get_context_struct_type(self).as_pointer(),
                ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                data_struct_ptr, data_struct_ptr]

            if not is_mech:
                # Add condition struct
                cond_gen = pnlvm.helpers.ConditionGenerator(ctx, self)
                cond_ty = cond_gen.get_condition_struct_type().as_pointer()
                args.append(cond_ty)

            func_ty = pnlvm.ir.FunctionType(pnlvm.ir.VoidType(), tuple(args))
            llvm_func = pnlvm.ir.Function(ctx.module, func_ty, name=func_name)
            llvm_func.attributes.add('argmemonly')
            context, params, comp_in, data_in, data_out = llvm_func.args[:5]
            cond_ptr = llvm_func.args[-1]


            for a in llvm_func.args:
                a.attributes.add('nonnull')
                a.attributes.add('noalias')

            # Create entry block
            block = llvm_func.append_basic_block(name="entry")
            builder = pnlvm.ir.IRBuilder(block)

            if is_mech:
                m_function = ctx.get_llvm_function(node)
            else:
                m_func_name = node._get_execution_wrapper()
                m_function = ctx.get_llvm_function(m_func_name)

            if node is self.input_CIM:
                m_in = comp_in
                incoming_projections = []
            elif not is_mech:
                m_in = builder.alloca(m_function.args[2].type.pointee)
                incoming_projections = node.input_CIM.afferents
            else:
                m_in = builder.alloca(m_function.args[2].type.pointee)
                incoming_projections = node.afferents

            # Run all incoming projections
            #TODO: This should filter out projections with different execution ID

            for par_proj in incoming_projections:
                # Skip autoassociative projections
                if par_proj.sender.owner is par_proj.receiver.owner:
                    continue

                proj_idx = self.projections.index(par_proj)

                # Get parent mechanism
                par_mech = par_proj.sender.owner

                proj_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(proj_idx)])
                proj_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(proj_idx)])
                proj_function = ctx.get_llvm_function(par_proj)

                output_s = par_proj.sender
                assert output_s in par_mech.output_states
                if par_mech in self._all_nodes:
                    par_idx = self._get_node_index(par_mech)
                else:
                    comp = par_mech.composition
                    assert par_mech is comp.output_CIM
                    par_idx = self.c_nodes.index(comp)
                output_state_idx = par_mech.output_states.index(output_s)
                proj_in = builder.gep(data_in, [ctx.int32_ty(0),
                                                ctx.int32_ty(0),
                                                ctx.int32_ty(par_idx),
                                                ctx.int32_ty(output_state_idx)])

                state = par_proj.receiver
                assert state.owner is node or state.owner is node.input_CIM
                if state in state.owner.input_states:
                    state_idx = state.owner.input_states.index(state)

                    assert par_proj in state.pathway_projections
                    projection_idx = state.pathway_projections.index(par_proj)

                    # Adjust for AutoAssociative projections
                    for i in range(projection_idx):
                        if isinstance(state.pathway_projections[i], AutoAssociativeProjection):
                            projection_idx -= 1
                elif state in state.owner.parameter_states:
                    state_idx = state.owner.parameter_states.index(state) + len(state.owner.input_states)

                    assert par_proj in state.mod_afferents
                    projection_idx = state.mod_afferents.index(par_proj)
                else:
                    # Unknown state
                    assert False

                assert state_idx < len(m_in.type.pointee)
                assert projection_idx < len(m_in.type.pointee.elements[state_idx])
                proj_out = builder.gep(m_in, [ctx.int32_ty(0),
                                              ctx.int32_ty(state_idx),
                                              ctx.int32_ty(projection_idx)])

                if proj_in.type != proj_function.args[2].type:
                    assert node is self.output_CIM
                    proj_in = builder.bitcast(proj_in, proj_function.args[2].type)
                builder.call(proj_function, [proj_params, proj_context, proj_in, proj_out])


            idx = ctx.int32_ty(self._get_node_index(node))
            zero = ctx.int32_ty(0)
            m_params = builder.gep(params, [zero, zero, idx])
            m_context = builder.gep(context, [zero, zero, idx])
            m_out = builder.gep(data_out, [zero, zero, idx])
            if is_mech:
                builder.call(m_function, [m_params, m_context, m_in, m_out])
            else:
                # Condition and data structures includes parent first
                nested_idx = ctx.int32_ty(self._get_node_index(node) + 1)
                m_data = builder.gep(data_in, [zero, nested_idx])
                m_cond = builder.gep(cond_ptr, [zero, nested_idx])
                builder.call(m_function, [m_context, m_params, m_in, m_data, m_cond])
                # Copy output of the nested composition to its output place
                output_idx = node._get_node_index(node.output_CIM)
                result = builder.gep(m_data, [zero, zero, ctx.int32_ty(output_idx)])
                builder.store(builder.load(result), m_out)

            builder.ret_void()

        return func_name

    def __get_processing_condition_set(self, node):
        dep_group = []
        for group in self.scheduler_processing.consideration_queue:
            if node in group:
                break
            dep_group = group

        # NOTE: This is not ideal we don't need to depend on
        # the entire previous group. Only our dependencies
        cond = [EveryNCalls(dep, 1) for dep in dep_group]
        if node not in self.scheduler_processing.condition_set.conditions:
            cond.append(Always())
        else:
            cond += self.scheduler_processing.condition_set.conditions[node]

        return All(*cond)

    def __gen_exec_wrapper(self):
        func_name = None
        llvm_func = None
        with pnlvm.LLVMBuilderContext() as ctx:
            # Create condition generator
            cond_gen = pnlvm.helpers.ConditionGenerator(ctx, self)

            func_name = ctx.get_unique_name('exec_wrap_' + self.name)
            func_ty = pnlvm.ir.FunctionType(pnlvm.ir.VoidType(), (
                ctx.get_context_struct_type(self).as_pointer(),
                ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                ctx.get_data_struct_type(self).as_pointer(),
                cond_gen.get_condition_struct_type().as_pointer()))
            llvm_func = pnlvm.ir.Function(ctx.module, func_ty, name=func_name)
            llvm_func.attributes.add('argmemonly')
            context, params, comp_in, data, cond = llvm_func.args
            for a in llvm_func.args:
                a.attributes.add('nonnull')
                a.attributes.add('noalias')

            # Create entry block
            entry_block = llvm_func.append_basic_block(name="entry")
            builder = pnlvm.ir.IRBuilder(entry_block)

            # Call input CIM
            input_cim_name = self._get_node_wrapper(self.input_CIM);
            input_cim_f = ctx.get_llvm_function(input_cim_name)
            builder.call(input_cim_f, [context, params, comp_in, data, data])

            # Allocate run set structure
            run_set_type = pnlvm.ir.ArrayType(pnlvm.ir.IntType(1), len(self.c_nodes))
            run_set_ptr = builder.alloca(run_set_type, name="run_set")

            # Allocate temporary output storage
            output_storage = builder.alloca(data.type.pointee, name="output_storage")

            iter_ptr = builder.alloca(ctx.int32_ty, name="iter_counter")
            builder.store(ctx.int32_ty(0), iter_ptr)

            loop_condition = builder.append_basic_block(name="scheduling_loop_condition")
            builder.branch(loop_condition)

            # Generate a while not 'end condition' loop
            builder.position_at_end(loop_condition)
            run_cond = cond_gen.generate_sched_condition(builder,
                            self.termination_processing[TimeScale.TRIAL],
                            cond, None)
            run_cond = builder.not_(run_cond, name="not_run_cond")

            loop_body = builder.append_basic_block(name="scheduling_loop_body")
            exit_block = builder.append_basic_block(name="exit")
            builder.cbranch(run_cond, loop_body, exit_block)


            # Generate loop body
            builder.position_at_end(loop_body)

            zero = ctx.int32_ty(0)
            any_cond = pnlvm.ir.IntType(1)(0)

            # Calculate execution set before running the mechanisms
            for idx, mech in enumerate(self.c_nodes):
                run_set_mech_ptr = builder.gep(run_set_ptr,
                                               [zero, ctx.int32_ty(idx)],
                                               name="run_cond_ptr_" + mech.name)
                mech_cond = cond_gen.generate_sched_condition(builder,
                                self.__get_processing_condition_set(mech),
                                cond, mech)
                ran = cond_gen.generate_ran_this_pass(builder, cond, mech)
                mech_cond = builder.and_(mech_cond, builder.not_(ran),
                                         name="run_cond_" + mech.name)
                any_cond = builder.or_(any_cond, mech_cond, name="any_ran_cond")
                builder.store(mech_cond, run_set_mech_ptr)

            for idx, mech in enumerate(self.c_nodes):
                run_set_mech_ptr = builder.gep(run_set_ptr, [zero, ctx.int32_ty(idx)])
                mech_cond = builder.load(run_set_mech_ptr, name="mech_" + mech.name + "_should_run")
                with builder.if_then(mech_cond):
                    mech_name = self._get_node_wrapper(mech);
                    mech_f = ctx.get_llvm_function(mech_name)
                    if isinstance(mech, Mechanism):
                        builder.call(mech_f, [context, params, comp_in, data, output_storage])
                    else:
                        builder.call(mech_f, [context, params, comp_in, data, output_storage, cond])

                    cond_gen.generate_update_after_run(builder, cond, mech)

            # Writeback results
            for idx, mech in enumerate(self.c_nodes):
                run_set_mech_ptr = builder.gep(run_set_ptr, [zero, ctx.int32_ty(idx)])
                mech_cond = builder.load(run_set_mech_ptr, name="mech_" + mech.name + "_ran")
                with builder.if_then(mech_cond):
                    out_ptr = builder.gep(output_storage, [zero, zero, ctx.int32_ty(idx)], name="result_ptr_" + mech.name)
                    data_ptr = builder.gep(data, [zero, zero, ctx.int32_ty(idx)],
                                           name="data_result_" + mech.name)
                    builder.store(builder.load(out_ptr), data_ptr)

            # Update step counter
            with builder.if_then(any_cond):
                cond_gen.bump_ts(builder, cond)

            # Increment number of iterations
            iters = builder.load(iter_ptr, name="iterw")
            iters = builder.add(iters, ctx.int32_ty(1), name="iterw_inc")
            builder.store(iters, iter_ptr)

            max_iters = len(self.scheduler_processing.consideration_queue)
            completed_pass = builder.icmp_unsigned("==", iters,
                                                   ctx.int32_ty(max_iters),
                                                   name="completed_pass")
            # Increment pass and reset time step
            with builder.if_then(completed_pass):
                builder.store(zero, iter_ptr)
                # Bumping automatically zeros lower elements
                cond_gen.bump_ts(builder, cond, (0, 1, 0))

            builder.branch(loop_condition)

            builder.position_at_end(exit_block)
            # Call output CIM
            output_cim_name = self._get_node_wrapper(self.output_CIM);
            output_cim_f = ctx.get_llvm_function(output_cim_name)
            builder.call(output_cim_f, [context, params, comp_in, data, data])

            # Bump run counter
            cond_gen.bump_ts(builder, cond, (1, 0, 0))

            builder.ret_void()
        return func_name

    def __gen_run_wrapper(self):
        func_name = None
        with pnlvm.LLVMBuilderContext() as ctx:
            func_name = ctx.get_unique_name('run_wrap_' + self.name)
            func_ty = pnlvm.ir.FunctionType(pnlvm.ir.VoidType(), (
                ctx.get_context_struct_type(self).as_pointer(),
                ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_data_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                ctx.get_output_struct_type(self).as_pointer(),
                ctx.int32_ty.as_pointer(),
                ctx.int32_ty.as_pointer()))
            llvm_func = pnlvm.ir.Function(ctx.module, func_ty, name=func_name)
            llvm_func.attributes.add('argmemonly')
            context, params, data, data_in, data_out, runs_ptr, inputs_ptr = llvm_func.args
            for a in llvm_func.args:
                a.attributes.add('nonnull')
                a.attributes.add('noalias')

            # Create entry block
            entry_block = llvm_func.append_basic_block(name="entry")
            builder = pnlvm.ir.IRBuilder(entry_block)

            # Allocate and initialize condition structure
            cond_gen = pnlvm.helpers.ConditionGenerator(ctx, self)
            cond_type = cond_gen.get_condition_struct_type()
            cond = builder.alloca(cond_type)
            cond_init = cond_type(cond_gen.get_condition_initializer())
            builder.store(cond_init, cond)

            iter_ptr = builder.alloca(ctx.int32_ty, name="iter_counter")
            builder.store(ctx.int32_ty(0), iter_ptr)

            loop_condition = builder.append_basic_block(name="run_loop_condition")
            builder.branch(loop_condition)

            # Generate a while not 'end condition' loop
            builder.position_at_end(loop_condition)
            count = builder.load(iter_ptr)
            runs = builder.load(runs_ptr)
            run_cond = builder.icmp_unsigned('<', count, runs)

            loop_body = builder.append_basic_block(name="run_loop_body")
            exit_block = builder.append_basic_block(name="exit")
            builder.cbranch(run_cond, loop_body, exit_block)

            # Generate loop body
            builder.position_at_end(loop_body)

            # Current iteration
            iters = builder.load(iter_ptr);

            # Get the right input stimulus
            input_idx = builder.urem(iters, builder.load(inputs_ptr))
            data_in_ptr = builder.gep(data_in, [input_idx])

            # Call execution
            exec_f_name = self._get_execution_wrapper()
            exec_f = ctx.get_llvm_function(exec_f_name)
            builder.call(exec_f, [context, params, data_in_ptr, data, cond])

            # Extract output_CIM result
            idx = self._get_node_index(self.output_CIM)
            result_ptr = builder.gep(data, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(idx)])
            output_ptr = builder.gep(data_out, [iters])
            result = builder.load(result_ptr)
            builder.store(result, output_ptr)

            # increment counter
            iters = builder.add(iters, ctx.int32_ty(1))
            builder.store(iters, iter_ptr)
            builder.branch(loop_condition)

            builder.position_at_end(exit_block)

            builder.store(builder.load(iter_ptr), runs_ptr)

            builder.ret_void()

        return func_name

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

        autodiff_stimuli = {}
        all_stimuli_keys = list(stimuli.keys())
        for node in all_stimuli_keys:
            if hasattr(node, "pytorch_representation"):
                if node.learning_enabled:
                    autodiff_stimuli[node] = stimuli[node]
                    del stimuli[node]

        # STEP 1A: Check that all of the nodes listed in the inputs dict are ORIGIN nodes in the self
        origin_nodes = self.get_c_nodes_by_role(CNodeRole.ORIGIN)
        for node in stimuli.keys():
            if not node in origin_nodes:
                raise CompositionError("{} in inputs dict for {} is not one of its ORIGIN nodes".
                               format(node.name, self.name))

        # STEP 1B: Check that all of the ORIGIN nodes are represented - if not, use default_external_input_values
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
        nums_input_sets = set()
        for node, stim_list in stimuli.items():
            if isinstance(node, Composition):
                if isinstance(stim_list, dict):

                    adjusted_stimulus_dict, num_trials, autodiff_stimuli = node._adjust_stimulus_dict(stim_list)
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
                    stim_list = adjusted_stimulus_list  # ADDED CW 12/21/18: This line fixed a bug, but it might be a hack

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
                nums_input_sets.add(1)

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
                nums_input_sets.add(len(stimuli[node]))
        if len(nums_input_sets) > 1:
            if 1 in nums_input_sets:
                nums_input_sets.remove(1)
                if len(nums_input_sets) > 1:
                    raise CompositionError("The input dictionary for {} contains input specifications of different "
                                           "lengths ({}). The same number of inputs must be provided for each node "
                                           "in a Composition.".format(self.name, nums_input_sets))
            else:
                raise CompositionError("The input dictionary for {} contains input specifications of different "
                                       "lengths ({}). The same number of inputs must be provided for each node "
                                       "in a Composition.".format(self.name, nums_input_sets))
        num_input_sets = nums_input_sets.pop()
        return adjusted_stimuli, num_input_sets, autodiff_stimuli

    def _adjust_execution_stimuli(self, stimuli):
        adjusted_stimuli = {}
        for node, stimulus in stimuli.items():
            if isinstance(node, Composition):
                input_must_match = node.external_input_values
                if isinstance(stimulus, dict):
                    adjusted_stimulus_dict = node._adjust_stimulus_dict(stimulus)
                    adjusted_stimuli[node] = adjusted_stimulus_dict
                    continue
            else:
                input_must_match = node.default_external_input_values


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

    def evaluate(
        self,
        predicted_input=None,
        control_allocation=None,
        num_trials=1,
        runtime_params=None,
        base_execution_id=None,
        execution_id=None,
        context=None
    ):
        '''Runs a simulation of the `Composition`, with the specified control_allocation, excluding its
           `model_based_optimizer <Composition.model_based_optimizer>` in order to return the
           `net_outcome <ModelBasedOptimizationControlMechanism.net_outcome>` of the Composition, according to its
           `model_based_optimizer <Composition.model_based_optimizer>` under that control_allocation. All values are
           reset to pre-simulation values at the end of the simulation. '''

        # These attrs are set during composition.before_simulation
        # reinitialize_values = self.sim_reinitialize_values

        # FIX: DOES THIS TREAT THE ControlSignals AS STATEFUL W/IN THE SIMULATION?
        #      (i.e., DOES IT ASSIGN THE SAME CONTROLSIGNAL VALUES FOR ALL SIMULATIONS?)
        # (NECESSARY, SINCE adjustment_cost (?AND duration_cost) DEPEND ON PREVIOUS VALUE OF ControlSignal,
        #  AND ALL NEED TO BE WITH RESPECT TO THE *SAME* PREVIOUS VALUE
        # Assign control_allocation current being sampled
        if control_allocation is not None:
            self.model_based_optimizer.apply_control_allocation(control_allocation,
                                                                execution_id=execution_id,
                                                                runtime_params=runtime_params,
                                                                context=context)

        net_control_allocation_outcomes = []

        for i in range(num_trials):
            inputs = {}
            for j in range(len(self.model_based_optimizer.shadow_external_inputs)):
                inputs[self.model_based_optimizer.shadow_external_inputs[j]] = predicted_input[j][i]

            self.parameters.context.get(execution_id).execution_phase = ContextFlags.SIMULATION
            for output_state in self.output_states:
                for proj in output_state.efferents:
                    proj.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING

            self.run(inputs=inputs,
                     # reinitialize_values=reinitialize_values,
                     execution_id=execution_id,
                     runtime_params=runtime_params,
                     context=context)

            # KAM Note: Need to manage execution_id here in order to report simulation results on "outer" comp
            if context.initialization_status != ContextFlags.INITIALIZING:
                try:
                    self.parameters.simulation_results.get(base_execution_id).append(self.get_output_values(execution_id))
                except AttributeError:
                    self.parameters.simulation_results.set([self.get_output_values(execution_id)], base_execution_id)

            self.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING
            # need to update input states in order to get correct value for "outcome" (from objective mech)
            self.model_based_optimizer._update_input_states(execution_id, runtime_params, context.flags_string)

            outcome = self.model_based_optimizer.input_state.parameters.value.get(execution_id)
            all_costs = self.model_based_optimizer.parameters.costs.get(execution_id)
            combined_costs = self.model_based_optimizer.combine_costs(all_costs)
            # KAM Modified 12/5/18 to use OCM's compute_net_outcome fn rather than hard-coded difference
            net_outcome = self.model_based_optimizer.compute_net_outcome(outcome, combined_costs)
            net_control_allocation_outcomes.append(net_outcome)

        return net_control_allocation_outcomes

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
        return self.get_output_values()

    def get_output_values(self, execution_context=None):
        return [output_state.parameters.value.get(execution_context) for output_state in self.output_CIM.output_states]

    @property
    def input_state(self):
        """Returns the index 0 InputState that belongs to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_states[0]

    @property
    def input_values(self):
        """Returns values of all InputStates that belong to the Input CompositionInterfaceMechanism"""
        return self.get_input_values()

    def get_input_values(self, execution_context=None):
        return [input_state.parameters.value.get(execution_context) for input_state in self.input_CIM.input_states]

    @property
    def runs_simulations(self):
        return True

    @property
    def simulation_results(self):
        return self.parameters.simulation_results.get(self.default_execution_id)

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
            return [input_state.defaults.value for input_state in self.input_CIM.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def stateful_nodes(self):
        """
        List of all nodes in the system that are currently marked as stateful. For Mechanisms, statefulness is
        determined by checking whether node.has_initializers is True. For Compositions, statefulness is determined
        by checking whether the

        Returns
        -------
        all stateful nodes in the system : List[Node]

        """

        stateful_nodes = []
        for node in self.c_nodes:
            if isinstance(node, Composition):
                if len(node.stateful_nodes) > 0:
                    stateful_nodes.append(node)
            elif node.has_initializers:
                stateful_nodes.append(node)

        return stateful_nodes

    @property
    def output_state(self):
        """Returns the index 0 OutputState that belongs to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_states[0]

    @property
    def class_parameters(self):
        return self.__class__.parameters

    @property
    def stateful_parameters(self):
        return [param for param in self.parameters if param.stateful]

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.c_nodes,
            self.projections,
            [self.input_CIM, self.output_CIM],
            self.input_CIM.efferents,
            self.output_CIM.afferents,
            [self.model_based_optimizer] if self.model_based_optimizer is not None else []
        ))
