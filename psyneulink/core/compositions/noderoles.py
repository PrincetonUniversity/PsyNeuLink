# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************** Pathway **************************************************************

"""

Related
-------

* `Composition`

Contents
--------

  * `NodeRole`
  * `NodesRoleManager`

.. _Node_Role_Overview:

Overview
--------

NodeRoles specify the structural position and/or functional role(s) that a `Node <Composition_Nodes>` plays in a
`Composition` or the representation of one (e.g., the `pytorch_representation
<AutodiffComposition.pytorch_representation>` of an `AutodiffCompostion`.  The set of possible NodRoles are defined
by the `NodeRole` enum, and the data structures and methods used to associate Nodes with NodeRoles are defined in the
`NodeRoleManager` class.

Class Reference
---------------

"""
import warnings
import enum
from collections import defaultdict, OrderedDict

from psyneulink._typing import Literal
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.globals.graph import EdgeType

__all__ = [
    'NodeRole', 'NodeRoleManager'
]

class NodeRoleError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class NodeRole(enum.Enum):
    """Roles assigned to `Nodes <Composition_Nodes>` of a `Composition`.

    Attributes
    ----------

    ORIGIN
        A `Node <Composition_Nodes>` that has no scheduling dependencies
        on any other Nodes within its own `Composition`. Typically,
        an `ORIGIN` Node also do not receive any `Projections <Projection>` from
        any other Nodes
        within its own `Composition`, though if it is in a `nested Composition <Composition_Nested>` it may
        receive Projections from the outer Composition.  `Execution of a `Composition <Composition_Execution>`
        always begins with an `ORIGIN` Node.  A Composition may have many `ORIGIN` Nodes.  This role cannot be
        modified programmatically.

    INPUT
        A `Node <Composition_Nodes>` that receives input from outside its `Composition`, either from the Composition's
        `run <Compositions.run>` method or, if it is in a `nested Composition <Composition_Nested>`, from the outer
        Composition.  By default, the `ORIGIN` Nodes of a Composition are also its `INPUT` Nodes; however this can be
        modified by `assigning specified NodeRoles <Composition_Node_Role_Assignment>` to Nodes.  A Composition can
        have many `INPUT` Nodes.  Note that any Node that `shadows <InputPort_Shadow_Inputs>` an `INPUT` Node is itself
        also assigned the role of `INPUT` Node.

    PROBE
        A `Node <Composition_Nodes>` that is neither `ORIGIN` nor `TERMINAL` but that is treated as an

    SINGLETON
        A `Node <Composition_Nodes>` that is both an `ORIGIN` and a `TERMINAL`.  This role cannot be modified
        programmatically.

    BIAS
        A `Node <Composition_Nodes>` for which all of its `InputPorts <InputPort>` are assigned *DEFAULT_VARIABLE*
        as their `default_input <InputPort.default_input>` (which provides a pre-specified input to each InputPort
        that is constant across executions). Such a Node is always also an `ORIGIN` Node (since it does not receive
        Projections from any other Node) and never an `INPUT` Node (since it does not receive external input).

    INTERNAL
        A `Node <Composition_Nodes>` that is neither `INPUT` nor `OUTPUT`.  Note that it *can* also be `ORIGIN`,
        `TERMINAL` or `SINGLETON`, if it has no `afferent <Mechanism_Base.afferents>` or `efferent
        <Mechanism_Base.efferents>` Projections or neither, respectively. This role cannot be modified programmatically.

    CYCLE
        A `Node <Composition_Nodes>` that belongs to a cycle. This role cannot be modified programmatically.

    FEEDBACK_SENDER
        A `Node <Composition_Nodes>` with one or more efferent `Projections <Projection>` designated as `feedback
        <Composition_Feedback_Designation>` in the Composition.  This means that the Node executes last in the
        sequence of Nodes that would otherwise form a `cycle <Composition_Cycle_Structure>`. This role cannot be
        modified directly, but is modified if the feedback status of the Projection is `explicitly specified
        <Composition_Feedback_Designation>`.

    FEEDBACK_RECEIVER
        A `Node <Composition_Nodes>` with one or more afferent `Projections <Projection>` designated as `feedback
        <Composition_Feedback_Designation>` in the Composition. This means that the Node executes first in the
        sequence of Nodes that would otherwise form a `cycle <Composition_Cycle_Structure>`. This role cannot be
        modified directly, but is modified if the feedback status of the Projection is `explicitly specified
        <Composition_Feedback_Designation>`.

    CONTROL_OBJECTIVE
        A `Node <Composition_Nodes>` that is an `ObjectiveMechanism` associated with a `ControlMechanism` other
        than the Composition's `controller <Composition.controller>` (if it has one).

    CONTROLLER
        A `Node <Composition_Nodes>` that is the `controller <Composition.controller>` of a Composition.
        This role cannot be modified programmatically.

    CONTROLLER_OBJECTIVE
        A `Node <Composition_Nodes>` that is an `ObjectiveMechanism` associated with a Composition's `controller
        <Composition.controller>`.

    LEARNING
        A `Node <Composition_Nodes>` that is only executed when learning is enabled;  if it is not also assigned
        `TARGET` or `LEARNING_OBJECTIVE`, then it is a `LearningMechanism`. This role can, but generally should not be
        modified programmatically.

    COMMENT:
    LEARNING_OUTPUT
        A `Node <Composition_Nodes>` that is last one in a `learning Pathway <Composition_Learning_Pathway>`,
        the desired `value <Mechanism_Base.value>` of which is provided as input to the `TARGET_MECHANISM
        <Composition_Learning_Components>` for that pathway (see `OUTPUT_MECHANISM <OUTPUT_MECHANISM>`.
        This role can, but generally should not be modified programmatically.
    COMMENT

    TARGET
        A `Node <Composition_Nodes>` that receives the target for a `learning pathway <Composition_Learning_Pathway>`
        specifying the desired output of the `OUTPUT_MECHANISM <OUTPUT_MECHANISM>` for that pathway
        (see `TARGET_MECHANISM <Composition_Learning_Components>`). This role can, but generally should not
        be modified programmatically.

    LEARNING_OBJECTIVE
        A `Node <Composition_Nodes>` that is the `ObjectiveMechanism` of a `learning Pathway
        <Composition_Learning_Pathway>`; usually a `ComparatorMechanism` (see `OBJECTIVE_MECHANISM`). This role can,
        but generally should not be modified programmatically.

    PROBE
        An `INTERNAL` `Node <Composition_Nodes>` that is permitted to have Projections from it to the Composition's
        `output_CIM <Composition.output_CIM>`, but -- unlike an `OUTPUT` Node -- the `output_values
        <Mechanism_Base.output_values>` of which are *not* included in the Composition's `results
        <Composition.results>` attribute (see `allow_probes <OptimizationContorlMechanism.allow_probes>` for an
        example.

    OUTPUT
        A `Node <Composition_Nodes>` the `output_values <Mechanism_Base.output_values>` of which are included in
        the Composition's `results <Composition.results>` attribute.  By default, the `TERMINAL` Nodes of a
        Composition are also its `OUTPUT` Nodes; however this can be modified by `assigning specified NodeRoles
        <Composition_Node_Role_Assignment>` to Nodes.  A Composition can have many `OUTPUT` Nodes.

    TERMINAL
        A `Node <Composition_Nodes>` on which no other Nodes have
        scheduling dependencies within its own `Composition`, excluding
        `ObjectiveMechanism`. Typically, a `TERMINAL` Node does not send
        any `Projections <Projection>` to any other Nodes within
        its own `Composition`, though if it is in a `nested Composition <Composition_Nested>` it may send Projections
        to the outer Composition. A Composition may have many `TERMINAL` Nodes. The `ObjectiveMechanism` associated
        with the Composition's `controller <Composition.controller>` (assigned the role `CONTROLLER_OBJECTIVE`)
        cannot be a `TERMINAL` Node of a Composition.  `Execution of a Composition <Composition_Execution>` itself
        always ends with a `TERMINAL` Node, although the `controller <Composition.controller>` and its associated
        `ObjectiveMechanism` may execute after that; some `TERMINAL` Nodes may also execute earlier (i.e., if they
        belong to a `Pathway` that is shorter than the longest one in the Composition).
        Nodes in a flattened cycle will be either all TERMINAL or all
        not TERMINAL.
        This role cannot be modified programmatically.

    """
    ORIGIN = enum.auto()
    INPUT = enum.auto()
    SINGLETON = enum.auto()
    BIAS = enum.auto()
    INTERNAL = enum.auto()
    CYCLE = enum.auto()
    FEEDBACK_SENDER = enum.auto()
    FEEDBACK_RECEIVER = enum.auto()
    CONTROL_OBJECTIVE = enum.auto()
    CONTROLLER = enum.auto()
    CONTROLLER_OBJECTIVE = enum.auto()
    LEARNING = enum.auto()
    TARGET = enum.auto()
    LEARNING_OBJECTIVE = enum.auto()
    PROBE = enum.auto()
    OUTPUT = enum.auto()
    TERMINAL = enum.auto()


unmodifiable_node_roles = {NodeRole.ORIGIN,
                           NodeRole.INTERNAL,
                           NodeRole.SINGLETON,
                           NodeRole.TERMINAL,
                           NodeRole.CYCLE}


class NodeRoleManager(object):
    """Manage association of nodes with roles
    IMPLEMENTATION NOTE:
      - this (the base class) assumes that the owner is a Composition; however, subclasses can modify this
      - non-underscore methods are meant to be user accessible, and accordingly have "shell" versions on owner
      - underscore methods are only on NodeRoleManager, and meant only only for internal use
    """
    def __init__(self, owner, graph):
        self.owner = owner
        self.graph = graph
        self.nodes = owner.nodes # TEACHER_TARGET BREADCRUMB: CAN'T THIS JUST BE DEFINED BY THE KEYS AND VALUES OF THE GRAPH?
        self.nodes_to_roles = OrderedDict()
        self.required_node_roles = []
        self.excluded_node_roles = []

    def _determine_node_roles(self):
        """Assign NodeRoles to Nodes of owner

        .. note::
           Assignments are **not** subject to user-modification (i.e., "programmatic assignment")
           unless otherwise noted.

        Assignment criteria:

        ORIGIN:
          - all Nodes that are in first consideration_set (i.e., self.scheduler.consideration_queue[0]).
          .. note::
             - this takes account of any Projections designated as feedback by graph_processing
               (i.e., self.graph.comp_to_vertex[efferent].feedback == EdgeType.FEEDBACK)
             - these will all be assigned afferent Projections from Composition.input_CIM

        INPUT:
          - all Nodes excluding BIAS Nodes that have no incoming edges in this composition,
            or that are in a cycle with no external incoming edges, for
            which INPUT has not been removed and/or excluded using exclude_node_roles();
          - all Nodes for which INPUT has been assigned as a required_node_role by user
            (i.e., in self.required_node_roles[NodeRole.INPUT].

        SINGLETON:
          - all Nodes that are *both* ORIGIN and TERMINAL

        BIAS:
          - all Nodes that have one or more InputPorts for which default_input == DEFAULT_VARIABLE

        INTERNAL:
            A `Node <Composition_Nodes>` that is neither `INPUT` nor
            `OUTPUT`.  Note that it *can* also be `ORIGIN`, `TERMINAL`
            or `SINGLETON`, if it has no `afferent
            <Mechanism_Base.afferents>` or `efferent
            <Mechanism_Base.efferents>` Projections or neither,
            respectively. This role cannot be modified programmatically.

        CYCLE:
          - all Nodes that identified as being in a cycle by self.graph_processing
            (i.e., in a cycle in self.graph_processing.cycle_vertices)

        FEEDBACK_SENDER:
          - all Nodes that send a Projection designated as feedback by self.graph_processing OR
            specified as feedback by user

        FEEDBACK_RECEIVER:
          - all Nodes that receive a Projection designated as feedback by self.graph_processing OR
            specified as feedback by user

        CONTROL_OBJECTIVE
          - ObjectiveMechanism assigned CONTROL_OBJECTIVE as a required_node_role in ControlMechanism's
            _instantiate_objective_mechanism()
          .. note::
             - *not the same as* CONTROLLER_OBJECTIVE
             - all project to a ControlMechanism

        CONTROLLER_OBJECTIVE
          - ObjectiveMechanism assigned CONTROLLER_OBJECTIVE as a required_node_role in add_controller()
          .. note::
             - also assigned CONTROL_OBJECTIVE
             - *not the same as* CONTROL_OBJECTIVE

        LEARNING
          - all Nodes for which LEARNING is assigned as a required_noded_role in
            add_linear_learning_pathway() or _create_terminal_backprop_learning_components()

        TARGET
          - all Nodes for which TARGET has been assigned as a required_noded_role in
            add_linear_learning_pathway() or _create_terminal_backprop_learning_components()
          .. note::
             - receive a Projection from input_CIM, and project to LEARNING_OBJECTIVE and output_CIM
             - also assigned ORIGIN, INPUT, LEARNING, OUTPUT, and TERMINAL

        LEARNING_OBJECTIVE
          - all Nodes for which LEARNING_OBJECTIVE is assigned required_noded_role in
            add_linear_learning_pathway(), _create_non_terminal_backprop_learning_components,
            or _create_terminal_backprop_learning_components()
          .. note::
             - also assigned LEARNING
             - must project to a LearningMechanism

        OUTPUT:
          - all Nodes that have no outgoing edges in this compositions
            *unless* they are to:
            - a ModulatoryMechanism (i.e., ControlMechanism or LearningMechanism)
            - an ObjectiveMechanisms associated with ModulatoryMechanism (including for Learning)
          - all Nodes that project only to:
            - a ModulatoryMechanism
            - an ObjectiveMechanism designated CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
            ? unless it is the ??TARGET_MECHANISM for a 'learning pathway <Composition_Learning_Pathway>`
              this is currently the case, but is inconsistent with the analog in Control,
              where monitored Mechanisms *are* allowed to be OUTPUT;
              therefore, might be worth allowing TARGET_MECHANISM to be assigned as OUTPUT
          - all Nodes for which OUTPUT has been assigned as a required_node_role, including by user
            (i.e., in self.required_node_roles[NodeRole.OUTPUT]

        TERMINAL:
          - all Nodes that
            - are not an ObjectiveMechanism assigned the role CONTROLLER_OBJECTIVE
            - or have *no* efferent projections OR
            - or for which any efferent projections are either:
                - to output_CIM OR
                - assigned as feedback (i.e., self.graph.comp_to_vertex[efferent].feedback == EdgeType.FEEDBACK
          .. note::
             - this insures that for cases in which there are nested CYCLES
               (e.g., LearningMechanisms for a `learning Pathway <Composition.Learning_Pathway>`),
               only the Node in the *outermost* CYCLE that is specified as a FEEDBACK_SENDER
               is assigned as a TERMINAL Node
               (i.e., the LearningMechanism responsible for the *first* `learned Projection;
               <Composition_Learning_Components>` in the `learning Pathway  <Composition.Learning_Pathway>`)
             - an ObjectiveMechanism assigned CONTROLLER_OBJECTIVE is prohibited since it and the Composition's
               `controller <Composition.controller>` are executed outside of (either before or after)
               all of the other Components of the Composition, as managed directly by the scheduler;
             - `Execution of a `Composition <Composition_Execution>` always ends with a `TERMINAL` Node,
               although some `TERMINAL` Nodes may execute earlier (i.e., if they belong to a `Pathway` that
               is shorter than the longest one in the Composition).

        Arguments
        ---------

        processing_graph : dict : default self.graph_processing
           if provided, it is used for dependencies that determine roles in place of self.graph_processing
           (used to construct externally-used representations, such as AutodiffComposition.pytorch_representation)

        node_roles_map : dict : default self.nodes_to_roles
           if provided, it is where role assignments are stored
           (used to construct externally-used role assignments, such as AutodiffComposition.pytorch_representation)
       """

        from psyneulink.core.compositions.composition import Composition
        # TEACHER_TARGET BREADCRUMB: REPLACE "flatten" and "scope" WITH TEST OR FLAG FOR COMPOSITION

        # Clear old roles
        # # MODIFIED TEACHER_TARGET OLD:
        # self.nodes_to_roles.update({k: set() for k in self.nodes_to_roles})
        # MODIFIED TEACHER_TARGET NEW:
        self.nodes_to_roles = OrderedDict()
        # MODIFIED TEACHER_TARGET END

        # Assign required_node_roles
        for node_role_pair in self.required_node_roles:
            self._add_node_role(node_role_pair[0], node_role_pair[1])

        # Get ORIGIN and TERMINAL Nodes using self.scheduler.consideration_queue
        if self.scheduler.consideration_queue:
            self._determine_origin_and_terminal_nodes_from_consideration_queue()

        # MODIFIED TEACHER_TARGET OLD:
        # comp_graph_dependencies = (processing_graph if flatten
        #                            # If using the composition graph structure with conditions,
        #                            # the scheduler graph may be different than the composition graph itself.
        #                            # BREADCRUMB: WHICH SHOULD NodeRle assignments reflect?
        #                            else self.graph_processing.prune_feedback_edges()[0])
        # MODIFIED TEACHER_TARGET END
        # BREADCRUMB ----------------------------------------------------------------------

        #region INPUT

        # Start with all nodes from processing graph with no incoming edges
        input_nodes = {n for n in self.graph if len(self.graph[n]) == 0}

        try:
            # IMPLEMENTATION NOTE: Specific to Composition
            # an entire cycle that has no node with any incoming edge other
            # than from other nodes in the cycle is treated as INPUT
            # ex: tests/composition/test_composition.py::TestNodeRoles::test_BIAS
            #      Processing of cycles not currently supported for flattened processing_graph
            if self.graph.cycle_vertices:
                for cycle in self.owner.graph_processing.cycle_vertices:
                    for i, node in enumerate(cycle):
                        prev = cycle[(i - 1) % len(cycle)]
                        if self.graph[node] != {prev}:
                            break
                    else:
                        input_nodes = input_nodes.union(cycle)
        except AttributeError:
            # Ensure execution of above for Composition
            assert not isinstance(self.owner, Composition), \
                (f"PROGRAM ERROR: Failure to find 'cycle_vertices' attribute of graph "
                 f"used by NodeRolesManager for '{self.owner.name}.")

        for node in self.nodes:
            # Check all remaining ORIGIN Nodes
            if node in input_nodes:
                # Don't allow INTERNAL Nodes to be INPUTS
                if NodeRole.INTERNAL in self.get_roles_by_node(node):
                    continue
                self._add_node_role(node, NodeRole.INPUT)

                if isinstance(node, ControlMechanism):
                    try:
                        # IMPLEMENTATION NOTE: Specific to Composition:
                        # special case, ControlMechanisms create MappingProjections
                        # to inner composition parameter CIMs, which may or may not
                        # create scheduler dependencies (determined by user action).
                        # If an inner composition is not ORIGIN because of this
                        # condition, add it as INPUT anyway.
                        for child in self.owner.graph_processing.comp_to_vertex[node].children:
                            for parent in child.parents:
                                # MappingProjections from non-ControlMechanisms
                                # always obey standard scheduling behavior
                                if (
                                    not isinstance(parent.component, ControlMechanism)
                                    or parent.component not in input_nodes
                                ):
                                    continue

                                for proj in child.component.get_afferents(parent.component):
                                    if (
                                        isinstance(proj, PathwayProjection_Base)
                                        and proj._creates_scheduling_dependency
                                    ):
                                        self._add_node_role(child.component, NodeRole.INPUT)
                                        break
                    except AttributeError:
                        # Ensure execution of above for Composition
                        assert not isinstance(self.owner, Composition), \
                            (f"PROGRAM ERROR: Failure to find 'comp_to_vertex[node].children' attribute "
                             f"of graph used by NodeRolesManager for '{self.owner.name}.")

            # Node does not receive any path_afferents (except possibly from input_CIM)
            elif (not isinstance(node, (Composition, ModulatoryMechanism_Base))
                  and (not node.path_afferents
                       or all(p.sender.owner is self.input_CIM for p in node.path_afferents))):
                self._add_node_role(node, NodeRole.INPUT)

            if isinstance(node, Composition):
                if not node.get_nodes_by_role(NodeRole.INPUT):
                    # If a nested Composition has no INPUTS, remove it as an INPUT of the outer Composition
                    self._remove_node_role(node, NodeRole.INPUT)
        #endregion INPUT

        #region BIAS
        for node in self.nodes:
            if (isinstance(node, Mechanism)
                    and all(input_port.default_input == DEFAULT_VARIABLE for input_port in node.input_ports)):
                self._add_node_role(node, NodeRole.BIAS)
                # BIAS Nodes should never be included as INPUT Nodes:
                self._remove_node_role(node, NodeRole.INPUT)
                # BIAS Nodes should not be included as OUTPUT Nodes
                self._remove_node_role(node, NodeRole.OUTPUT)
                # FIX: Can above with below once nested BIAS Node is allowed to project to Node in outer Composition
                # #   *unless* they are in a nested Composition and project to a Node in an outer one
                # if not any(isinstance(p.receiver.owner, CompositionInterfaceMechanism) for p in node.efferents):
                #     self._remove_node_role(node, NodeRole.OUTPUT)
        #endregion BIAS

        try:
        #region CYCLE
            # IMPLEMENTATION NOTE: Specific to Composition
            for cycle in self.owner.graph_processing.cycle_vertices:
                for node in cycle:
                    self._add_node_role(node, NodeRole.CYCLE)
        #endregion CYCLE
        #region FEEDBACK_SENDER and FEEDBACK_RECEIVER
            # IMPLEMENTATION NOTE: Specific to Composition
            for receiver in self.graph_processing.vertices:
                for sender, typ in receiver.source_types.items():
                    if typ is EdgeType.FEEDBACK:
                        self._add_node_role(sender.component, NodeRole.FEEDBACK_SENDER)
                        self._add_node_role(receiver.component, NodeRole.FEEDBACK_RECEIVER)
        #endregion FEEDBACK_SENDER and FEEDBACK_RECEIVER
        except AttributeError:
            # Ensure execution of above for Composition
            assert not isinstance(self.owner, Composition), \
                (f"PROGRAM ERROR: Failure to find either 'vertices' or 'cycle_vertices' attribute "
                 f"of graph used by NodeRolesManager for '{self.owner.name}.")

        # FIX 4/25/20 [JDC]:  NEED TO AVOID AUTOMATICALLY (RE-)ASSIGNING ONES REMOVED BY exclude_node_roles
        #     - Simply exclude any LEARNING_OBJECTIVE and CONTROL_OBJECTIVE that project only to ModulatoryMechanism
        #     - NOTE IN PROGRAM ERROR FAILURE TO ASSIGN CONTROL_OBJECTIVE

        #region OUTPUT
        # Note: "TERMINAL" referenced below is in respect to the
        # the composition graph, not the scheduler graph, because OUTPUT
        # is determined by composition structure, not scheduling order.
        try:
            # Specific to Composition
            output_nodes = self._get_terminal_nodes(self.graph)
        except IndexError:
            output_nodes = []

        for node in self.nodes:
            # Assign OUTPUT if node is TERMINAL...
            if node in output_nodes:
                # unless it is a ModulatoryMechanism or a BIAS Node in nested Composition that projects to an outer one
                if (isinstance(node, ModulatoryMechanism_Base)
                        or (NodeRole.BIAS in self.get_roles_by_node(node)
                            and not any(isinstance(p.receiver.owner, CompositionInterfaceMechanism)
                                        for p in node.efferents))):
                    continue
                else:
                    self._add_node_role(node, NodeRole.OUTPUT)

            # Assign OUTPUT to any relevant non-TERMINAL Nodes
            else:

                # Assign CONTROL_OBJECTIVE to any ObjectiveMechanism that projects to a ControlMechanism
                #     and is not already so designated (needed for user-specified ObjectiveMechanisms
                if (isinstance(node, ObjectiveMechanism)
                        and NodeRole.CONTROL_OBJECTIVE not in self.get_roles_by_node(node)):
                    ctl_mech = next((p.receiver.owner for p in node.efferents
                                     if isinstance(p.receiver.owner, ControlMechanism)), None)
                    if ctl_mech:
                        node.control_mechanism = ctl_mech
                        self._add_required_node_role(node, NodeRole.CONTROL_OBJECTIVE)

                # IMPLEMENTATION NOTE:
                #   This version allows LEARNING_OBJECTIVE to be assigned as OUTPUT
                #   The alternate version below restricts OUTPUT only to RecurrentTransferMechasnism
                # # Assign OUTPUT if node projects only to itself and/or a LearningMechanism
                # #     (i.e., it is either a RecurrentTransferMechanism configured for learning
                # #      or the LEARNING_OBJECTIVE of a `learning pathway <Composition_Learning_Pathway>`
                # if all(p.receiver.owner is node or isinstance(p.receiver.owner, LearningMechanism)
                #        for p in node.efferents):
                #     self._add_node_role(node, NodeRole.OUTPUT)
                #     continue

                # Assign OUTPUT if it is a `RecurrentTransferMechanism` configured for learning
                #    and doesn't project to any Nodes other than its `AutoassociativeLearningMechanism`
                #    (this is not picked up as a `TERMINAL` since it projects to the `AutoassociativeLearningMechanism`)
                #    but can (or already does) project to an output_CIM
                if all((p.receiver.owner is node # <- recurrence
                        or isinstance(p.receiver.owner, AutoAssociativeLearningMechanism)
                        or p.receiver.owner is self.output_CIM) # <- already projects to an output_CIM
                       for p in node.efferents):
                    self._add_node_role(node, NodeRole.OUTPUT)
                    continue

                # Assign OUTPUT for all members of a CYCLE if they *all* project only to members of the CYCLE or:
                #  - an ObjectiveMechanism designated as CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
                #  - and/or directly to a ControlMechanism but is not an ObjectiveMechanism
                #  - and/or projects to another node in a CYCLE but otherwise meets the above criteria
                def _is_output_node(node, allow_cycle=False)->bool:
                    try:
                        return all((any(p.receiver.owner in self.get_nodes_by_role(role)
                                        for role in {NodeRole.CONTROL_OBJECTIVE,
                                                     NodeRole.CONTROLLER_OBJECTIVE,
                                                     NodeRole.LEARNING_OBJECTIVE})
                                    # or p.receiver.owner is node
                                    or p.receiver.owner is self.output_CIM
                                    or (isinstance(p.receiver.owner, ControlMechanism)
                                        and not isinstance(node, ObjectiveMechanism))
                                    or (allow_cycle and p.receiver.owner in self.get_nodes_by_role(NodeRole.CYCLE))
                                   for p in node.efferents))
                    except AttributeError:
                        # Ensure execution of above for Composition
                        assert not isinstance(self.owner, Composition),\
                            f"PROGRAM ERROR: Failure to find 'output_cim' as attribute of '{self.owner.name}."

                # Assign OUTPUT only if the node is not:
                #  - the TARGET_MECHANISM of a `learning Pathway <Composition_Learning_Pathway>`
                #  - a ModulatoryMechanism
                # and the node projects only to:
                #  - an ObjectiveMechanism designated as CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
                #  - and/or directly to a ControlMechanism but is not an ObjectiveMechanism
                #  - and/or (already projects) to output_CIM
                if NodeRole.TARGET in self.get_roles_by_node(node):
                    continue
                if isinstance(node, ModulatoryMechanism_Base):
                    continue
                if _is_output_node(node, allow_cycle=False):
                    self._add_node_role(node, NodeRole.OUTPUT)
                # Check for OUTPUT CYCLE (i.e., one in which all Nodes are OUTPUTS)
                # Note:  assign OUTPUT to all members of CYCLE once detected, to avoid re-checking for each
                elif (node in self.get_nodes_by_role(NodeRole.CYCLE)
                      and node not in self.get_nodes_by_role(NodeRole.OUTPUT)):
                    # # Get Nodes in the CYCLE:
                    cycle_nodes = [node]
                    queue = collections.deque([node])
                    i = 0
                    while queue:
                        curr_node = queue.popleft()
                        for next_node in [proj.receiver.owner for proj in curr_node.efferents
                                          if proj.receiver.owner in self.get_nodes_by_role(NodeRole.CYCLE)]:
                            if next_node in cycle_nodes: # Cycle closed
                                continue
                            cycle_nodes.append(next_node)
                            queue.append(next_node)
                        i += 1
                        assert i < 1000, f"PROGRAM ERROR: CYCLE DETECTION FAILED FOR {node} IN {self.name}."
                    # Ensure they are all satisfy criterial for OUTPUT Node
                    if (all(_is_output_node(cycle_node, allow_cycle=True)
                            # and cycle_node not in self.get_nodes_by_role(NodeRole.OUTPUT)
                            and not any(role in self.get_roles_by_node(cycle_node)
                                        for role in {NodeRole.OUTPUT,NodeRole.TERMINAL})
                            for cycle_node in cycle_nodes)):
                        for cycle_node in cycle_nodes:
                            self._add_node_role(cycle_node, NodeRole.OUTPUT)

                # If node is a Composition and its output_CIM has OutputPorts that either have no Projections
                #     or projections to self.output_CIM, then assign as OUTPUT Node
                # Note: this ensures that if a nested Comp has both Nodes that project to others in the outer
                #       Composition *and* legit OUTPUT Nodes (i.e., ones that project only to the outer Composition's
                #       output_CIM), the latter qualify to still make the nested Comp an OUTPUT Node
                elif isinstance(node, Composition):
                    if any(not port.efferents or
                           any(proj.receiver.owner is self.output_CIM for proj in port.efferents)
                           for port in node.output_CIM.output_ports):
                        self._add_node_role(node, NodeRole.OUTPUT)

                # MODIFIED TEACHER_TARGET NEW: BREADCRUMB UNCOMMENT THIS IF NOT HANDLED IN pytorchshowgraph
                # Assign OUTPUT if node is not LEARNING_OBJECTIVE and no other nodes are dependent on it
                elif (processing_graph
                      and not NodeRole.LEARNING_OBJECTIVE in self.get_roles_by_node(node)
                      and not any(node in [n for n in v
                                           if not (NodeRole.LEARNING_OBJECTIVE in self.get_roles_by_node(k))]
                                  for k, v in processing_graph.items())):
                    self._add_node_role(node, NodeRole.OUTPUT)
                # MODIFIED TEACHER_TARGET END
        #endregion OUTPUT

        #region Assign SINGLETON and INTERNAL nodes
        for node in self.nodes:
            # # MODIFIED TEACHER_TARGET OLD:
            # if all(n in node_roles_map[node] for n in {NodeRole.ORIGIN, NodeRole.TERMINAL}):
            #     self._add_node_role(node, NodeRole.SINGLETON)
            # if not any(n in node_roles_map[node] for n in {NodeRole.INPUT, NodeRole.OUTPUT}):
            #     self._add_node_role(node, NodeRole.INTERNAL)
            # MODIFIED TEACHER_TARGET NEW:
            if all(n in self.get_roles_by_node(node) for n in {NodeRole.ORIGIN, NodeRole.TERMINAL}):
                self._add_node_role(node, NodeRole.SINGLETON)
            # if all(n in self.get_roles_by_node(node) for n in {NodeRole.INPUT, NodeRole.OUTPUT}):
            #     self._add_node_role(node, NodeRole.INTERNAL, scope)
            if not any(n in self.get_roles_by_node(node) for n in {NodeRole.INPUT, NodeRole.OUTPUT}):
                self._add_node_role(node, NodeRole.INTERNAL)
            # MODIFIED TEACHER_TARGET END
        #endregion Assign SINGLETON and INTERNAL nodes

        # Finally, remove any NodeRole assignments specified in excluded_node_roles
        for node, role in self.excluded_node_roles:
            if role in self.get_roles_by_node(node):
                self._remove_node_role(node, role)
                self._get_nested_nodes()

        # Manual override to avoid INPUT/OUTPUT setting, which would cause
        # CIMs to be created, which is not correct for controllers
        try:
            if self.owner.controller is not None:
                self.nodes_to_roles[self.controller] = {NodeRole.CONTROLLER}
        except AttributeError:
            # Ensure execution of above for Composition
            assert not isinstance(self.owner, Composition), \
                f"PROGRAM ERROR: Failure to find 'controller' attribute of '{self.owner.name}."

        self.needs_determine_node_roles = False

    def _add_node_role(self, node, role):
        if role not in NodeRole:
            raise NodeRoleError('Invalid NodeRole: {0}'.format(role))
        try:
            self.nodes_to_roles[node].add(role)
        except KeyError:
            raise NodeRoleError(f"Attempt to assign {role} to '{node.name}' that is not a Node in {self.name}.")

    def _remove_node_role(self, node, role):
        if role not in NodeRole:
            raise NodeRoleError('Invalid NodeRole: {0}'.format(role))
        try:
            self.nodes_to_roles[node].remove(role)
        except KeyError as e:
            pass
            # if e.args[0] is node:
            #     assert False, f"PROGRAM ERROR in _remove_node_role: {node} not found in {self.name}.nodes_to_role."
            # elif e.args[0] is role:
            #     assert False, f"PROGRAM ERROR in _remove_node_role: " \
            #                   f"{role} not found for {node} in {self.name}.nodes_to_role."
            # else:
            #     assert False, f"PROGRAM ERROR: unexpected problem in '_remove_node_role'."

    def require_node_roles(self, node, roles, context=None):
        """
            Assign the `NodeRole`\\(s) specified in **roles** to **node**.  Remove exclusion of those NodeRoles if
            it any had previously been specified in `exclude_node_roles <Composition.exclude_node_roles>`.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` to which **role** should be assigned.

            roles : `NodeRole` or list[`NodeRole`]
                `NodeRole`\\(s) to assign to **node**.

        """
        # TEACHER_TARGET BREADCRUMB: ADD SCOPE FOR NESTED COMPS
        roles = convert_to_list(roles)
        for role in roles:
            self._add_required_node_role(node, role, context)

    def _add_required_node_role(self, node, role, context=None):
        """
            Assign the `NodeRole` specified by **role** to **node**.  Remove exclusion of that `NodeRole` if
            it had previously been specified in `exclude_node_roles <Composition.exclude_node_roles>`.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` to which **role** should be assigned.

            role : `NodeRole`
                `NodeRole` to assign to **node**.

        """
        if role not in NodeRole:
            raise NodeRoleError('Invalid NodeRole: {0}'.format(role))

        # Disallow assignment of NodeRoles by user that are not programmitically modifiable:
        # FIX 4/25/20 [JDC] - CHECK IF ROLE OR EQUIVALENT STATUS HAS ALREADY BEEN ASSIGNED AND, IF SO, ISSUE WARNING
        if context.source == ContextFlags.COMMAND_LINE:
            if role in {NodeRole.CONTROL_OBJECTIVE, NodeRole.CONTROLLER_OBJECTIVE} and not node.control_mechanism:
                warnings.warn(f"{role} should be assigned with caution to {self.name}. "
                              f"{ObjectiveMechanism.__name__}s are generally constructed automatically by a "
                              f"{ControlMechanism.__name__}, or assigned to it in the '{OBJECTIVE_MECHANISM}' "
                              f"argument of its constructor.  Doing so otherwise may cause unexpected results.")
            elif role in {NodeRole.LEARNING, NodeRole.LEARNING_OBJECTIVE, NodeRole.TARGET}:
                warnings.warn(f"{role} should be assigned with caution to {self.name}. "
                              f"Learning Components are generally constructed automatically as part of "
                              f"a learning Pathway. Doing so otherwise may cause unexpected results.")
            elif role in {NodeRole.FEEDBACK_SENDER, NodeRole.FEEDBACK_RECEIVER}:
                to_from = 'from'
                if role is NodeRole.FEEDBACK_RECEIVER:
                    to_from = 'to'
                from psyneulink.core.components.projections.projection import Projection
                warnings.warn(f"{role} is not a role that can be assigned directly {to_from} {self.name}. "
                              f"The relevant {Projection.__name__} to it must be designated as 'feedback' "
                              f"where it is addd to the {self.name};  assignment will be ignored.")
            elif role is NodeRole.BIAS:
                if any(p.path_afferents for p in node.input_ports):
                    if all([isinstance(proj.sender.owner, CompositionInterfaceMechanism) for proj in p.path_afferents]
                           for p in node.input_ports):
                        # Was an INPUT Node with Projections from input_CIM,
                        #   so exclude as INPUT which should remove its afferent Projections
                        self.exclude_node_roles(node, NodeRole.INPUT, context=context)
                        self._analyze_graph(context=context)
                    else:
                        # Had afferent Projections other than from input_CIM
                        raise NodeRoleError(f"Attempt to assign 'NodeRole.BIAS' to a node ('{node.name}') "
                                               f"in '{self.name}' that already has input(s) assigned.")
                for input_port in node.input_ports:
                    input_port.parameters.default_input._set(DEFAULT_VARIABLE, context, override=True)
                    input_port.internal_only = True
                # BIAS Node should *never* be considered as an INPUT Node;  *can* be an OUTPUT Node
                #   if it is in an inner Composition and projects to an outer one (handed in _determine_node_roles)
                self.exclude_node_roles(node, NodeRole.INPUT)
                self.exclude_node_roles(node, NodeRole.OUTPUT)
                self.required_node_roles.append((node, NodeRole.BIAS))

            elif role is NodeRole.INPUT:
                if (node, NodeRole.BIAS) in self.required_node_roles:
                    raise NodeRoleError(f"A Node assigned NodeRole.BIAS ('{node.name}') cannot also be "
                                           f"assigned NodeRole.INPUT (since it does not receive any input).")

            elif role in unmodifiable_node_roles:
                raise NodeRoleError(f"A Node assigned NodeRole.BIAS ('{node.name}') cannot also be "
                                       f"assigned NodeRole.INPUT (since it does not receive any input).")

        node_role_pair = (node, role)
        if node_role_pair not in self.required_node_roles:
            self.required_node_roles.append(node_role_pair)
        node_role_pairs = [item for item in self.excluded_node_roles if item[0] is node and item[1] is role]
        for item in node_role_pairs:
            self.excluded_node_roles.remove(item)

    def exclude_node_roles(self, node:Mechanism_Base, roles:list, context=None)->list:
        """
            Exclude the `NodeRole`\\(s) specified in **roles** from being assigned to **node**.

            Remove specified roles if they had previously been assigned either by default as a `required_node_role
            <Composition_Node_Role_Assignment>` or using the `required_node_roles <Composition.required_node_roles>`
            method.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` from which **role** should be removed.

            roles : `NodeRole` or list[`NodeRole`]
                `NodeRole`\\(s) to remove and/or exclude from **node**.
        """
        roles = convert_to_list(roles)
        for role in roles:
            if role not in NodeRole:
                raise NodeRoleError(f"Invalid NodeRole specified for {node} in 'exclude_node_roles': {role}.")

            # Disallow assignment of NodeRoles by user that are not programmitically modifiable:
            if (context.source == ContextFlags.COMMAND_LINE and
                    role in {NodeRole.ORIGIN, NodeRole.INTERNAL, NodeRole.SINGLETON, NodeRole.TERMINAL,
                             NodeRole.CYCLE, NodeRole.FEEDBACK_SENDER, NodeRole.FEEDBACK_RECEIVER, NodeRole.LEARNING}):
                raise NodeRoleError(f"Attempt to exclude {role} (from {node} of {self.name})"
                                       f"that cannot be modified by user.")
            node_role_pair = (node, role)
            if node_role_pair not in self.excluded_node_roles:
                self.excluded_node_roles.append(node_role_pair)
            if node_role_pair in comp.required_node_roles:
                self.required_node_roles.remove(node_role_pair)
            self._remove_node_role(node, role)

    def get_nodes_by_role(self, role:NodeRole, scope:Optional[Literal[ALL]]=None)->List:
        """Return a list of `Nodes <Composition_Nodes>` assigned the `NodeRole`specified in **role**.
        If **scope** is not specified, searches for and returns only nodes at top level of the Composition.
        If **scope** is ALL, includes nodes in top level Composition and any nested within it.

        Arguments
        _________

        role : `NodeRole`
            role for which `Nodes <Composition_Nodes>` are desired.

        scope : `ALL` or None
            specifies whether nodes with the specified `NodeRole` are returned only if they are in the top level
            Composition, or any nessted within it.

        Returns
        -------

        list[`Mechanisms <Mechanism>` and/or `Compositions <Composition>`] :
            list of `Nodes <Composition_Nodes>` assigned the `NodeRole` specified in **role**,
            which is empty if there are none.

        """
        if role is None or role not in NodeRole:
            raise NodeRoleError('Invalid NodeRole: {0}'.format(role))

        try:
            if scope is ALL:
                return self.get_nested_nodes_by_roles_at_any_level(self, role)
            return [node for node in self.nodes_to_roles if role in self.nodes_to_roles[node]]

        except KeyError as e:
            raise NodeRoleError(f'Node missing from {self}.nodes_to_roles: {e}.')

    # TEACHER_TARGET BREADCRUMB: DEAL WITH SCOPE, SINCE IT MAY NOT APPLY TO PYTORCH REPS
    def get_roles_by_node(self,
                          node:Union[Mechanism_Base, Composition_Base],
                          scope:Optional[Literal[ALL, NESTED]]=None)->list:
        """Return a list of `NodeRoles <NodeRole>` assigned to **node**.
        If **scope** is not specified, returns roles for the node only if it is at the top level of the Composition.
        If **scope** is *ALL*, the node can be in the top level Compostion or any nested within it.
        If **scope** is *NESTED*, indicates call is recursive from outer Composition called with *ALL*

        Arguments
        _________

        node : `Node <Composition_Nodes>`
            `Node <Composition_Nodes>` for which assigned `NodeRoles <NodeRole>` are desired.


        scope : `ALL` or None
            specifies whether roles are returned from just the source Composition, or all nessted within it.

        Returns
        -------

        List[`Mechanisms <Mechanism>` and/or `Compositions <Composition>`] or None:
            list of `NodeRoles <NodeRole>` assigned to **node** if node is found and has roles,
                which is empty if there are no roles assigned to the node;
            KeyError if node not found
        """

        try:
            if node in self.nodes or scope is None:
                return self.nodes_to_roles[node]
            # Scope is ALL (outermost Composition) or NESTED (called recursively from outer Composition)
            for nested_comp in [comp for comp in self.nodes if isinstance(comp, Composition)]:
                roles = nested_comp.get_roles_by_node(node, scope=NESTED)
                if roles:
                    return roles
                elif scope is NESTED:
                    # Continue to check any other nested Compositions
                    continue
            if scope is NESTED:
                # Allow search to continue
                return None
            else:
                # Outermost composition, and no more nested Compositions, so node was not found
                raise KeyError()

        except KeyError:
            if node in self._get_all_nodes():
                # Node not found in self.nodes_to_roles[node] of Composition and, if scope, any nested within it
                return []
            if not scope and node not in self.nodes:
                # Node note found in outermost Composition
                raise NodeRoleError(f"Node ('{node.name}') for which roles were requested is not in '{self.name}'")
            # Node was not found anywhere
            raise NodeRoleError(f"Node ('{node.name}') for which roles were requested is not in '{self.name}' "
                                   f"or any Compositions nested within it.")

    # TEACHER_TARGET BREADCRUMB: DEAL WITH SCOPE, SINCE IT MAY NOT APPLY TO PYTORCH REPS
    def get_required_roles_by_node(self, node):
        """
            Return a list of `NodeRoles <NodeRole>` that have been user-assigned to a specified **node**.

            Arguments
            _________

            node : `Node <Composition_Nodes>`
                `Node <Composition_Nodes>` for which assigned `NodeRoles <NodeRole>` are desired.

            Returns
            -------

            List[`Mechanisms <Mechanism>` and/or `Compositions <Composition>`] :
                list of `NodeRoles <NodeRole>` assigned to **node**.
        """

        try:
            return [role for n, role in self.required_node_roles if n is node]
        except KeyError:
            raise NodeRoleError(f"Node {node} not found in {self.nodes_to_roles}.")


    # TEACHER_TARGET BREADCRUMB: REFACTOR THIS TO DEAL WITH REPS THAT DON'T SUPPORT NESTING?
    def get_nested_nodes_by_roles_at_any_level(self, comp, include_roles, exclude_roles=None)->list or None:
        """Return all Nodes from comp or any nested within it that have *include_roles* but not *exclude_roles*.
        Returns Nodes that have or don't have the specified roles in the Composition specified by **comp**
        or any Composition nested within it, irrespective of their status at other levels of nesting.
        To get nodes that are either INPUT or OUTPUT Nodes at *all* levels of nesting, use either
            get_nested_input_nodes_at_all_levels() or get_nested_output_nodes_at_all_levels()
        Note:  do this recursively, checking roles on the "way down," as a Node may have a role in a
               deeply nested Composition, but that Composition itself may not have the same role in the Composition
               within which *it* is nested (e.g., a Node might be an INPUT Node of a nested Composition, but that
               nested Composition may not be an INPUT Node of the Composition in which it is nested).
        Note: exclude_roles takes precedence, so that if a NodeRole is listed in both,
              nodes with that role will be *excluded*.
        IMPLEMENTATION NOTE:
            This method is only support for Compositions.  If the owner of the NodeRoleMgr is not a Composition
            (e.g, it is the pytorch_representation of an AutodiffComposition), an error is raised when called.
        """
        from psyneulink.core.compositions.composition import Composition
        assert isinstance(comp, Composition), (f"PROGRAM ERROR: get_nested_nodes_by_roles_at_any_level() was called"
                                               f"for a NodeRoleMgr the owner of which is not a Composition")
        nested_nodes = []
        include_roles = [] if include_roles is None else convert_to_list(include_roles)
        exclude_roles = [] if exclude_roles is None else convert_to_list(exclude_roles)
        if isinstance(comp, Composition):
            # Get all nested nodes in comp that have include_roles and not exclude_roles:
            for node in [n for n in comp.nodes
                         if (any(n in comp.get_nodes_by_role(include)
                                 for include in include_roles)
                               and not any(n in comp.get_nodes_by_role(exclude)
                                           for exclude in exclude_roles))]:
                if isinstance(node, Composition):
                    nested_nodes.extend(node.node_roles_mgr.get_nested_nodes_by_roles_at_any_level(node,
                                                                                                   include_roles,
                                                                                                   exclude_roles))
                else:
                    nested_nodes.append(node)
        return nested_nodes if any(nested_nodes) else []

