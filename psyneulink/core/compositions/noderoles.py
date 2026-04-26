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
`NodeRolesManager` class.

Class Reference
---------------

"""
import warnings
import enum
import toposort
from copy import copy
from collections import OrderedDict, deque
from typing import List, Union, Literal

from psyneulink._typing import Literal, Optional
from psyneulink.core.components.mechanisms.mechanism import Mechanism, Mechanism_Base
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.modulatory.modulatorymechanism import ModulatoryMechanism_Base
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.projections.projection import Projection
from psyneulink.core.components.projections.pathway.pathwayprojection import PathwayProjection_Base
from psyneulink.core.components.ports.port import Port
from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.library.components.mechanisms.modulatory.learning.autoassociativelearningmechanism import \
    AutoAssociativeLearningMechanism
from psyneulink.core.globals.keywords import ALL, DEFAULT_VARIABLE, NESTED, OBJECTIVE_MECHANISM
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.globals.graph import EdgeType
from psyneulink.core.globals.context import Context, ContextFlags

__all__ = ['NodeRole', 'NodeRolesManager']


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
        `TARGET_INTERNAL`, `TARGET_INPUT` or `LEARNING_OBJECTIVE`, then it is a `LearningMechanism`.
        This role can, but generally should not be modified programmatically.

    COMMENT:
    LEARNING_OUTPUT
        A `Node <Composition_Nodes>` that is last one in a `learning Pathway <Composition_Learning_Pathway>`,
        the desired `value <Mechanism_Base.value>` of which is provided as input to the `TARGET_MECHANISM
        <Composition_Learning_Components>` for that pathway (see `SAMPLE_MECHANISM <SAMPLE_MECHANISM>`.
        This role can, but generally should not be modified programmatically.
    COMMENT

    SAMPLE
        A `Node <Composition_Nodes>` that provides the sample value used to compute the error for a
        `learning pathway <Composition_Learning_Pathway>` (see `TARGET_MECHANISM <Composition_Learning_Components>`).
        Note that the specific value used as the sample may be from any one of a `SAMPLE_MECHANISM`\'s `Ports <Port>`;
        use the Composition's `get_sample_ports()` method to identify the specific one(s) that are used.
        This role cannot be modified programmatically.

    TARGET_INPUT
        A `Node <Composition_Nodes>` that receives an external value (i.e., one specified in the **targets**,
        or optionally the **inputs** argument, of a call to `learn() <Composition.learn>`; see
        `Composition_Target_Inputs`) that, in turn, is used as the target value in computing the error for a
        `learning pathway <Composition_Learning_Pathway>` (see `TARGET_MECHANISM <Composition_Learning_Components>`).
        This role cannot be modified programmatically.

    TARGET_INTERNAL
        An `INTENRAL` `Node <Composition_Nodes>` with an `OutputPort` the `value <OutputPort.value>` of which is
        is used as the target value in computing the error for a `learning pathway <Composition_Learning_Pathway>`
        (see `TARGET_MECHANISM <Composition_Learning_Components>`). This is used only by Composition subclasses
        (such as `AutodiffComposition`) that permit the explicit specification of `SAMPLE_MECHANISM`\\s.
        This role cannot be modified programmatically.

        .. note::
           This role is assigned only to `TARGET_MECHANISM`\\s that are `INPUT` Nodes (i.e., that receive an
           externally-provided target value; `TARGET_MECHANISM`\\s that are `INTERNAL` Nodes (i.e., that generate
           the target value on their own) are assigned the NodeRole `TARGET_INTERNAL`.

    LEARNING_OBJECTIVE
        A `Node <Composition_Nodes>` that is the `ObjectiveMechanism` of a `learning Pathway
        <Composition_Learning_Pathway>`; usually a `ComparatorMechanism` (see `OBJECTIVE_MECHANISM`). This role can
        not be modified programmatically.

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
        not TERMINAL. This role cannot be modified programmatically.

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
    SAMPLE = enum.auto()
    TARGET_INPUT = enum.auto()
    TARGET_INTERNAL = enum.auto()
    LEARNING_OBJECTIVE = enum.auto()
    PROBE = enum.auto()
    OUTPUT = enum.auto()
    TERMINAL = enum.auto()


unmodifiable_node_roles = {
    NodeRole.ORIGIN,
    NodeRole.INTERNAL,
    NodeRole.SINGLETON,
    NodeRole.TERMINAL,
    NodeRole.CYCLE,
    NodeRole.CONTROLLER,
    NodeRole.SAMPLE,
    NodeRole.TARGET_INPUT,
    NodeRole.TARGET_INTERNAL,
    NodeRole.LEARNING_OBJECTIVE
}

class NodeRolesManager(object):
    """Manage association of nodes with roles
    Used by graphs for different kinds of representations (e.g., graph_processing.dependency_dict for `Composition`
    and processing_graph for `AutodiffComposition.pytorch_representation`).
    """
    def __init__(self, owner):
        self.owner = owner
        self.name = f"NodeRolesManager for {owner.name}"
        self.nodes = owner.nodes
        self.graph = self.owner.processing_graph
        self.nodes_to_roles = OrderedDict()
        self.required_node_roles = []
        self.excluded_node_roles = []
        self._need_determine_node_roles = True

    def _determine_node_roles(self):
        """Assign NodeRoles to Nodes based on the position / use in self.graph
        Assignments are stored in self.nodes_to_roles
        Helper methods are used to evaluate Node and assign NodeRoles appropriately,
        some of which take **composition** as an argument, as they handle NodeRoles or assignments
           that are specific to Compositions;  if composition==None, those are ignored.
        """
        from psyneulink.core.compositions.composition import Composition

        # This ensures that NodeRoleManagers that are not for a Composition don't invoke Composition-specific handling
        composition = self.owner if isinstance(self.owner, Composition) else None

        if not self.nodes:
            return

        # Make NodeRole assignments
        self._set_up()
        self._determine_origin_and_terminal_nodes_from_consideration_queue(composition)
        self._INPUT_Nodes(self.nodes, composition)
        self._BIAS_Nodes(self.nodes)
        self._CYCLE_and_FEEDBACK_Nodes(composition)
        self._OUTPUT_Nodes(self.nodes, composition)
        self._SINGLETON_and_INTERNAL_Nodes(self.nodes)
        self._exclude_roles(self.nodes, composition)
        self._CONTROLLER_Node(composition)

        self._need_determine_node_roles = False

    # HELPER METHODS FOR _determine_node_roles
    # region

    def _set_up(self):
        """Assign graph and clear node roles"""
        self.graph = self.owner.processing_graph
        # Clear old roles
        self.nodes_to_roles.update({k: set() for k in self.graph})
        # Assign required_node_roles
        for node, role in self.required_node_roles:
            if node in self.graph:
                self._add_node_role(node, role)

    def _INPUT_Nodes(self, nodes, composition=None):
        """Assign NodeRole.INPUT to qualifying Nodes"""
        from psyneulink.core.compositions.composition import Composition

        #  Start with all nodes from processing graph with no incoming edges
        input_nodes = {n for n in self.graph if len(self.graph[n]) == 0}

        if composition:
            self._CYCLE_Nodes_as_INPUT(composition, input_nodes)

        for node in nodes:
            # Check all remaining ORIGIN Nodes
            if node in input_nodes:
                # Don't allow INTERNAL Nodes to be INPUTS
                if NodeRole.INTERNAL in self._get_roles_by_node(node):
                    continue
                self._add_node_role(node, NodeRole.INPUT)

                if composition and isinstance(node, ControlMechanism):
                    self._INPUTS_subject_to_control(composition, node, input_nodes)

            elif composition and self._no_path_afferents(composition, node):
                self._add_node_role(node, NodeRole.INPUT)

            if composition and isinstance(node, Composition):
                # If a nested Composition has no INPUTS, remove it as an INPUT of the outer Composition
                self._nested_composition_as_INPUT(node)

    def _INPUTS_subject_to_control(self, composition, node, input_nodes):
        """Assign NodeRole.INPUT to appropriate Nodes that receive ControlProjections
        ControlMechanisms create MappingProjections to inner composition parameter CIMs,
        which may or may not create scheduler dependencies (determined by user action).
        If an inner Composition is not ORIGIN because of this condition, add it as INPUT anyway.
        """
        for child in composition.graph_processing.comp_to_vertex[node].children:
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

    def _nested_composition_as_INPUT(self, node):
        """If a nested Composition has no INPUTS, remove it as an INPUT of the outer Composition"""
        from psyneulink.core.compositions.composition import Composition
        if isinstance(node, Composition):
            if not node.get_nodes_by_role(NodeRole.INPUT):
                self._remove_node_role(node, NodeRole.INPUT)

    def _no_path_afferents(self, composition, node)->bool:
        """Return True if Node does not receive any path_afferents (except possibly from input_CIM)"""
        from psyneulink.core.compositions.composition import Composition
        return (not isinstance(node, (Composition, ModulatoryMechanism_Base))
              and (not node.path_afferents
                   or all(p.sender.owner is composition.input_CIM for p in node.path_afferents)))

    def _BIAS_Nodes(self, nodes):
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

    def _CYCLE_Nodes_as_INPUT(self, composition, input_nodes:dict):
        """Assign list of input nodes all Nodes of a cycle if none receive any other inputs
        ex: tests/composition/test_composition.py::TestNodeRoles::test_BIAS
             Processing of cycles not currently supported for flattened processing_graph
        """
        if composition.graph_processing.cycle_vertices:
            for cycle in composition.graph_processing.cycle_vertices:
                for i, node in enumerate(cycle):
                    prev = cycle[(i - 1) % len(cycle)]
                    if self.graph[node] != {prev}:
                        break
                else:
                    input_nodes = input_nodes.union(cycle)

    def _CYCLE_and_FEEDBACK_Nodes(self, composition):
        """Assign NodeRole.CYCLE to any Nodes in a cycle and appropriate FEEDBACK NodeRoles"""
        if composition:
            for cycle in composition.graph_processing.cycle_vertices:
                for node in cycle:
                    self._add_node_role(node, NodeRole.CYCLE)

            for receiver in composition.graph_processing.vertices:
                for sender, typ in receiver.source_types.items():
                    if typ is EdgeType.FEEDBACK:
                        self._add_node_role(sender.component, NodeRole.FEEDBACK_SENDER)
                        self._add_node_role(receiver.component, NodeRole.FEEDBACK_RECEIVER)

    def _modulatory_or_bias_node_that_projects_out_of_a_nested_composition(self, node)->bool:
        """Return True for ModulatoryMechanism or a BIAS Node in nested Composition that projects to an outer one"""
        return (isinstance(node, ModulatoryMechanism_Base)
                or (NodeRole.BIAS in self._get_roles_by_node(node)
                    and not any(isinstance(p.receiver.owner, CompositionInterfaceMechanism)
                                for p in self._get_projections(node, 'efferents')
                                if p in self.owner.projections)))

    def _CONTROL_OBJECTIVE_Node(self, node)->bool:
        """Assign NodeRole.CONTROL_OBJECTIVE to any ObjectiveMechanism that projects to a ControlMechanism
        Assign only if not already so designated;  this is needed for user-specified ObjectiveMechanisms
        """
        if (isinstance(node, ObjectiveMechanism)
                and NodeRole.CONTROL_OBJECTIVE not in self._get_roles_by_node(node)):
            ctl_mech = next((p.receiver.owner for p in self._get_projections(node, 'efferents')
                             if isinstance(p.receiver.owner, ControlMechanism)), None)
            if ctl_mech:
                node.control_mechanism = ctl_mech
                self._add_required_node_role(node, NodeRole.CONTROL_OBJECTIVE)

    def _SINGLETON_and_INTERNAL_Nodes(self, nodes):
        for node in self.nodes:
            if (all(n in self._get_roles_by_node(node) for n in {NodeRole.ORIGIN, NodeRole.TERMINAL})):
                self._add_node_role(node, NodeRole.SINGLETON)
            if not any(n in self._get_roles_by_node(node) for n in {NodeRole.INPUT, NodeRole.OUTPUT}):
                self._add_node_role(node, NodeRole.INTERNAL)

    def _TERMINAL_Nodes(self, composition, graph)->list:
        """Return TERMINAL nodes from graph"""
        try:
            return self._get_TERMINAL_Nodes(composition, graph)
        except IndexError:
            return []

    def _get_TERMINAL_Nodes(self, composition, graph, toposorted_graph=None)->set:
        """Return a list of nodes in this Composition that are NodeRole.TERMINAL with respect to an acyclic **graph**
        The result can change depending on whether the scheduler or composition graph is used. The **graph** of the
        scheduler graph is the scheduler's consideration_queue.

        Includes all nodes that have no receivers in **graph**. The ObjectiveMechanism of a Composition's controller
        cannot be NodeRole.TERMINAL, so if the ObjectiveMechanism is the only node with no receivers in **graph**,
        then that node's senders are assigned NodeRole.TERMINAL instead.
        """
        terminal_nodes = set()

        senders = {n: set() for n in graph}
        for receiver in graph:
            # graph is {receiver:{senders}
            # invert it to {sender:{receivers}}, so that nodes (senders) without receivers can be found
            for sender in graph[receiver]:
                senders[sender].add(receiver)
        nodes_without_receivers = {sender for sender in graph if len(senders[sender]) == 0}

        # if a node is in a flattened cycle, all others in that cycle
        # must also have no receivers, or that node cannot be terminal
        if composition.graph_processing.cycle_vertices:
            for node in copy(nodes_without_receivers):
                for cycle in composition.graph_processing.cycle_vertices:
                    if (
                        node in cycle
                        and any(n not in nodes_without_receivers for n in cycle)
                    ):
                        nodes_without_receivers.remove(node)

        for node in nodes_without_receivers:
            if NodeRole.CONTROLLER_OBJECTIVE not in self._get_roles_by_node(node):
                terminal_nodes.add(node)
            # If CONTROLLER_OBJECTIVE node is only one without receivers, get its senders as TERMINAL Nodes
            elif len(nodes_without_receivers) < 2:
                if toposorted_graph is None:
                    toposorted_graph = list(toposort.toposort(graph))
                assert len(toposorted_graph) > 1 and node in toposorted_graph[-1], (
                    'CONTROLLER_OBJECTIVE node skipped as terminal, but'
                    ' consideration queue is not suitable for fallback'
                )
                for previous_node in toposorted_graph[-2]:
                    terminal_nodes.add(previous_node)

        return terminal_nodes

    def _determine_origin_and_terminal_nodes_from_consideration_queue(self, composition):
        """Determine ORIGIN and TERMINAL from scheduler
        Assign NodeRole.ORIGIN to all nodes in the first entry of the consideration queue and NodeRole.TERMINAL
        to all nodes in the last entry of the consideration queue. The ObjectiveMechanism of a Composition's
        controller may not be NodeRole.TERMINAL, so if the ObjectiveMechanism is the only node in the last entry
        of the consideration queue, then the second-to-last entry is NodeRole.TERMINAL instead.
        """
        if composition and composition.scheduler.consideration_queue:
            queue = composition.scheduler.consideration_queue
            for node in list(queue)[0]:
                self._add_node_role(node, NodeRole.ORIGIN)
            for node in self._get_TERMINAL_Nodes(composition, composition.scheduler.dependency_dict, queue):
                self._add_node_role(node, NodeRole.TERMINAL)

    def _OUTPUT_Nodes(self, nodes, composition=None):
        """Assign NodeRole.OUTPUT to qualifying Nodes
        # TEACHER_TARGET BREADCRUMB:  ??IS THE FOLLOWING TRUE FOR ALL NODES OR ONLY ONES IN CYCLES:??
        Assign OUTPUT only if the node is not:
         - the TARGET_MECHANISM of a `learning Pathway <Composition_Learning_Pathway>`
         - a ModulatoryMechanism
        and the node projects only to:
         - an ObjectiveMechanism designated as CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
         - and/or directly to a ControlMechanism but is not an ObjectiveMechanism
         - and/or (already projects) to output_CIM
        """
        from psyneulink.core.compositions.composition import Composition

        if composition:
            terminal_nodes = self._TERMINAL_Nodes(composition, self.graph)
            # "TERMINAL" referenced below is with respect to the Composition graph, not the scheduler graph,
            # because OUTPUT is determined by composition structure, not scheduling order.
        else:
            terminal_nodes = []

        for node in self.nodes:
            # Assign as OUTPUT if node is TERMINAL but not a modulatory or bias node
            if node in terminal_nodes:

                if self._modulatory_or_bias_node_that_projects_out_of_a_nested_composition(node):
                    continue
                self._add_node_role(node, NodeRole.OUTPUT)

            # Assign OUTPUT to any other relevant non-TERMINAL Nodes
            else:
                # Identify CONTROL_OBJECTIVE Nodes; needed for determinations of status as OUTPUT Node below
                self._CONTROL_OBJECTIVE_Node(node)

                # Assign any RecurrentTransferMechanisms that qualify as OUTPUT Nodes
                if self._RECURRENT_MECHANISM_as_OUTPUT(node, composition):
                    self._add_node_role(node, NodeRole.OUTPUT)
                    continue

                # Exclude TARGETS as OUTPUT Node
                if NodeRole.TARGET_INPUT in self._get_roles_by_node(node):
                    continue

                # Exclude ModulatoryMechanisms as OUTPUT Node
                if isinstance(node, ModulatoryMechanism_Base):
                    continue

                # Assign Node in a CYCLE as OUTPUT Node if it qualifies
                if self._CYCLE_NODE_as_OUTPUT(node, composition, allow_cycle=False):
                    self._add_node_role(node, NodeRole.OUTPUT)

                # Check for OUTPUT CYCLE (i.e., one in which all Nodes are OUTPUTS)
                # Note:  assign OUTPUT to all members of CYCLE once detected, to avoid re-checking for each
                elif (node in self._get_nodes_by_role(NodeRole.CYCLE)
                      and node not in self._get_nodes_by_role(NodeRole.OUTPUT)):
                    # # Get Nodes in the CYCLE:
                    cycle_nodes = [node]
                    queue = deque([node])
                    i = 0
                    while queue:
                        curr_node = queue.popleft()
                        for next_node in [proj.receiver.owner for proj in self._get_projections(curr_node, 'efferents')
                                          if proj.receiver.owner in self._get_nodes_by_role(NodeRole.CYCLE)]:
                            if next_node in cycle_nodes: # Cycle closed
                                continue
                            cycle_nodes.append(next_node)
                            queue.append(next_node)
                        i += 1
                        assert i < 1000, f"PROGRAM ERROR: CYCLE DETECTION FAILED FOR {node} IN {self.name}."
                    # Ensure they are all satisfy criteria for OUTPUT Node
                    if (all(self._CYCLE_NODE_as_OUTPUT(cycle_node, composition, allow_cycle=True)
                            and not any(role in self._get_roles_by_node(cycle_node)
                                        for role in {NodeRole.OUTPUT,NodeRole.TERMINAL})
                            for cycle_node in cycle_nodes)):
                        for cycle_node in cycle_nodes:
                            self._add_node_role(cycle_node, NodeRole.OUTPUT)

                # If node is a (nested) Composition and its output_CIM has any OutputPorts that either
                #    have no Projections or projections only to self.output_CIM, then assign it as an OUTPUT Node
                # Note: this ensures that if a nested Comp has both Nodes that project to others in the outer
                #       Composition *and* legit OUTPUT Nodes (i.e., ones that project *only* to outer Composition's
                #       output_CIM), the latter qualify to still make the nested Comp an OUTPUT Node
                elif isinstance(node, Composition):
                    # Assign NodeRole.OUTPUT to a nested Composition if it has any nodes with no projections to any
                    # other Nodes within its Composition or any outer one other than ones to the output_CIM of nested
                    # (and from there outer) Composition
                    has_nodes_with_limited_projections = \
                        any(not self._get_projections(inner_node, 'efferents') or
                            all(proj.receiver.owner is node.output_CIM
                                for proj in self._get_projections(inner_node,'efferents'))
                            for inner_node in node.nodes
                            if isinstance(inner_node, Mechanism))

                    if has_nodes_with_limited_projections:
                        if any(not self._get_projections(port, 'efferents') or
                               all(proj.receiver.owner is composition.output_CIM
                                   or isinstance(proj.receiver.owner, LossMechanism)
                                   for proj in self._get_projections(port, 'efferents'))
                               for port in node.output_CIM.output_ports):
                            self._add_node_role(node, NodeRole.OUTPUT)

                # Assign as OUTPUT if:
                #    - node is not CONTROL_OBJECTIVE or LEARNING_OBJECTIVE (e.g., LossMechanism)
                #    - no other nodes are dependent on it
                elif (not any(mod_objective in self._get_roles_by_node(node)
                              for mod_objective in {NodeRole.CONTROL_OBJECTIVE,
                                                    NodeRole.CONTROLLER_OBJECTIVE,
                                                    NodeRole.LEARNING_OBJECTIVE})
                      and not any(node in [n for n in v
                                           if not (NodeRole.LEARNING_OBJECTIVE in self._get_roles_by_node(k))]
                                  for k, v in self.graph.items())):
                    self._add_node_role(node, NodeRole.OUTPUT)

                # Assign as OUTPUT if it only projects to LossMechanisms or output_CIM
                elif all((isinstance(receiver, LossMechanism) and receiver.sample.owner is node)
                         or (isinstance(receiver, CompositionInterfaceMechanism)
                             and receiver is receiver.composition.output_CIM)
                         for receiver in [efferent.receiver.owner
                                          for efferent in self._get_projections(node, 'efferents')]):
                    self._add_node_role(node, NodeRole.OUTPUT)

    def _RECURRENT_MECHANISM_as_OUTPUT(self, node, composition)->bool:
        """Assign NodeRole.OUTPUT to RecurrentTransferMechanism
        If configured for learning, return True if it doesn't project to any Nodes other than its
        AutoassociativeLearningMechanism; this isn't picked up as `TERMINAL` since it projects to
        the AutoassociativeLearningMechanism but can (or already does) project to an output_CIM.
        """

        recurrent = True
        for p in self._get_projections(node, 'efferents'):
            if not (p.receiver.owner is node # <- recurrence
                    or isinstance(p.receiver.owner, AutoAssociativeLearningMechanism)
                    or (p.receiver.owner is composition.output_CIM  # <- already projects to an output_CIM
                    if composition else None)):
                recurrent = False
        return recurrent

            # IMPLEMENTATION NOTE:
            #   The following alternate version allows LEARNING_OBJECTIVE to be assigned as OUTPUT
            #   The version above restricts OUTPUT only to RecurrentTransferMechanism
            # # Assign OUTPUT if node projects only to itself and/or a LearningMechanism
            # #     (i.e., it is either a RecurrentTransferMechanism configured for learning
            # #      or the LEARNING_OBJECTIVE of a `learning pathway <Composition_Learning_Pathway>`
            # return all(p.receiver.owner is node or isinstance(p.receiver.owner, LearningMechanism)
            #        for p in node.efferents):
            #     self._add_node_role(node, NodeRole.OUTPUT)

    def _CYCLE_NODE_as_OUTPUT(self, node, composition, allow_cycle=False)->bool:
        """Return True if node projects only to other members of the cycle or:
        - ObjectiveMechanism designated as CONTROL_OBJECTIVE, CONTROLLER_OBJECTIVE or LEARNING_OBJECTIVE
        - and/or directly to a ControlMechanism but is not an ObjectiveMechanism
        - and/or projects to another node in a CYCLE but otherwise meets the above criteria
        """
        return all((any(p.receiver.owner in self._get_nodes_by_role(role)
                        for role in {NodeRole.CONTROL_OBJECTIVE,
                                     NodeRole.CONTROLLER_OBJECTIVE,
                                     NodeRole.LEARNING_OBJECTIVE})
                    # or p.receiver.owner is node
                    or (p.receiver.owner is self.owner.output_CIM if composition else None)
                    or (isinstance(p.receiver.owner, ControlMechanism)
                        and not isinstance(node, ObjectiveMechanism))
                    or (allow_cycle and p.receiver.owner in self._get_nodes_by_role(NodeRole.CYCLE))
                    for p in self._get_projections(node, 'efferents')))

    def _CONTROLLER_Node(self, composition=None):
        # Manual override to avoid INPUT/OUTPUT setting, which would cause
        # CIMs to be created, which is not correct for controllers
        if composition and composition.controller is not None:
            self.nodes_to_roles[composition.controller] = {NodeRole.CONTROLLER}

    def _get_projections(self, component, aff_or_eff: Union[Literal['afferents', 'efferents']]) -> List[Projection]:
        """Return afferents or efferents of Node filtered for those that are in current Composition"""
        from psyneulink.core.compositions.composition import Composition
        if not isinstance(component, (Port, Mechanism_Base, Composition)):
            assert False, f"PROGRAM ERROR: node in call to _get_projections must be Mechanism or Composition."
        if aff_or_eff == 'afferents':
            xferents = component.afferents
        elif aff_or_eff == 'efferents':
            xferents = component.efferents
        else:
            assert False, f"PROGRAM ERROR: call to _get_projections must specify 'afferents' or 'efferents'"
        # if isinstance(self.owner, Composition):
        #     if hasattr(self.owner, '_pytorch_projections'):
        #         projections = self.owner._pytorch_projections
        #     else:
        #         projections = self.owner._get_all_projections()
        # else:
        #     # composition = self.owner.composition
        #     # composition = self.owner.projections_map
        #     if hasattr(self.owner, '_pytorch_projections'):
        #         projections = self.owner.composition._pytorch_projections
        #     else:
        #         projections = self.owner.composition._get_all_projections()
        projections = self._get_all_projections()
        # assert not any(p not in projections for p in xferents)
        return [p for p in xferents if p in projections]

    def _get_all_projections(self)->list:
        """Return all Projections in Composition the receivers of which are in the Composition hierarchy.
        Unlike Composition._get_all_projections, this includes direct Projections between outer and nested
        Compositions that are needed for support of other representations (e.g.,
        AutodiffComposition.pytorch_representation).
        """
        from psyneulink.core.compositions.composition import Composition
        if isinstance(self.owner, Composition):
            composition = self.owner
        else:
            composition = self.owner.composition
        nodes = composition._get_all_nodes(include_cims=True, include_controller=True)
        projections = []
        for node in nodes:
            (projections
             .extend([p for p in node.efferents + node.afferents if p.receiver.owner in nodes]))
        return projections

    def _exclude_roles(self, nodes, composition=None):
        """Remove from nodes_to_roles all NodeRole assignments specified in excluded_node_roles"""
        for node, role in self.excluded_node_roles:
            if role in self._get_roles_by_node(node):
                self._remove_node_role(node, role)
        self._need_determine_node_roles = True

    def _add_node_role(self, node, role):
        if role not in NodeRole:
            raise NodeRoleError('Invalid NodeRole: {0}'.format(role))
        try:
            self.nodes_to_roles[node].add(role)
        except KeyError:
            raise NodeRoleError(f"Attempt to assign {role} to '{node.name}' that is not a Node in {self.owner.name}.")
        self._need_determine_node_roles = True

    def _add_required_node_role(self, node, role, context=Context()):
        """Assign the `NodeRole` specified by **role** to **node**.
        Remove exclusion of that `NodeRole` if it had previously been specified in `exclude_node_roles
        <Composition.exclude_node_roles>`.

        Arguments
        _________

        node : `Node <Composition_Nodes>`
            `Node <Composition_Nodes>` to which **role** should be assigned.

        role : `NodeRole`
            `NodeRole` to assign to **node**.
        """
        if role not in NodeRole:
            raise NodeRoleError('Invalid NodeRole: {0}'.format(role))

        # Disallow assignment of NodeRoles by user that are not programmatically modifiable:
        # FIX 4/25/20 [JDC] - CHECK IF ROLE OR EQUIVALENT STATUS HAS ALREADY BEEN ASSIGNED AND, IF SO, ISSUE WARNING
        if context.source == ContextFlags.COMMAND_LINE:
            if role in {NodeRole.CONTROL_OBJECTIVE, NodeRole.CONTROLLER_OBJECTIVE} and not node.control_mechanism:
                warnings.warn(f"{role} should be assigned with caution to {self.name}. "
                              f"{ObjectiveMechanism.__name__}s are generally constructed automatically by a "
                              f"{ControlMechanism.__name__}, or assigned to it in the '{OBJECTIVE_MECHANISM}' "
                              f"argument of its constructor.  Doing so otherwise may cause unexpected results.")
            elif role in {NodeRole.LEARNING, NodeRole.LEARNING_OBJECTIVE, NodeRole.TARGET_INPUT}:
                warnings.warn(f"{role} should be assigned with caution to {self.owner.name}. "
                              f"Learning Components are generally constructed automatically as part of "
                              f"a learning Pathway. Doing so otherwise may cause unexpected results.")
            elif role in {NodeRole.FEEDBACK_SENDER, NodeRole.FEEDBACK_RECEIVER}:
                to_from = 'from'
                if role is NodeRole.FEEDBACK_RECEIVER:
                    to_from = 'to'
                from psyneulink.core.components.projections.projection import Projection
                warnings.warn(f"{role} is not a role that can be assigned directly {to_from} {self.owner.name}. "
                              f"The relevant {Projection.__name__} to it must be designated as 'feedback' "
                              f"where it is added to the {self.owner.name};  assignment will be ignored.")
            elif role is NodeRole.BIAS:
                if any(p.path_afferents for p in node.input_ports):
                    if all([isinstance(proj.sender.owner, CompositionInterfaceMechanism) for proj in p.path_afferents]
                           for p in node.input_ports):
                        # Was an INPUT Node with Projections from input_CIM,
                        #   so exclude as INPUT which should remove its afferent Projections
                        self.exclude_node_roles(node, NodeRole.INPUT, context=context)
                        try:
                            self.owner._analyze_graph(context=context)
                        except AttributeError as e:
                            from psyneulink.core.compositions.composition import Composition
                            assert not isinstance(self.owner, Composition), \
                                (f"PROGRAM ERROR: Failure to find '_analyze_graph' method of '{self.owner.name} "
                                 f"needed by its NodeRolesManager")
                    else:
                        # Had afferent Projections other than from input_CIM
                        raise NodeRoleError(f"Attempt to assign 'NodeRole.BIAS' to a node ('{node.name}') "
                                               f"in '{self.owner.name}' that already has input(s) assigned.")
                for input_port in node.input_ports:
                    input_port.parameters.default_input._set(DEFAULT_VARIABLE, context, override=True)
                    input_port.internal_only = True
                # BIAS Node should *never* be considered as an INPUT Node;  *can* be an OUTPUT Node
                #   if it is in an inner Composition and projects to an outer one (handed in _determine_node_roles)
                self.exclude_node_roles(node, NodeRole.INPUT, context=context)
                self.exclude_node_roles(node, NodeRole.OUTPUT, context=context)
                self.required_node_roles.append((node, NodeRole.BIAS))

            elif role is NodeRole.INPUT:
                if (node, NodeRole.BIAS) in self.required_node_roles:
                    raise NodeRoleError(f"A Node assigned NodeRole.BIAS ('{node.name}') cannot also be "
                                           f"assigned NodeRole.INPUT (since it does not receive any input).")

            elif role in unmodifiable_node_roles:
                raise NodeRoleError(f"Attempt to assign '{node.name }' the NodeRole.{role.name} which is not allowed.")

        node_role_pair = (node, role)
        if node_role_pair not in self.required_node_roles:
            self.required_node_roles.append(node_role_pair)
        node_role_pairs = [item for item in self.excluded_node_roles if item[0] is node and item[1] is role]
        for item in node_role_pairs:
            self.excluded_node_roles.remove(item)
        self._need_determine_node_roles = True

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
        self._need_determine_node_roles = True

    # endregion HELPER METHODS FOR _determine_node_roles

    # USER-ACCESSIBLE ROLE-MANAGEMENT METHODS
    # region
    def _get_nodes_to_roles(self):
        if self._need_determine_node_roles:
            self._determine_node_roles()
        return self.nodes_to_roles

    def require_node_roles(self,
                           node:Mechanism_Base,
                           roles:Union[list, NodeRole],
                           scope:Optional[Literal[ALL]]=None,
                           context:Context=None) -> bool:
        """Assign **roles** to **node** (see _modify_node_roles() for details)"""
        return self._modify_node_roles(node, roles, 'require', scope, context)

    def exclude_node_roles(self,
                           node:Mechanism_Base,
                           roles:list,
                           scope:Optional[Literal[ALL]]=None,
                           context=None)->bool:
        """Exclude **roles** from being assigned to **node** (see _modify_node_roles() for details).
        """
        return self._modify_node_roles(node, roles, 'exclude', scope, context)

    def _modify_node_roles(self,
                           node:Mechanism_Base,
                           roles:list,
                           modification:Literal['require','exclude'],
                           scope:Optional[Literal[ALL, NESTED]]=None,
                           context=None)->bool:
        """Modify NodeRole assignments

        Arguments
        _________

        node : Node
            Node for which **roles** should be modified

        roles : NodeRole or list[NodeRole]
            NodeRole(s) to modify for **node**.

        modification : Literal['require','exclude']:
            *require*:
              assign **roles** to **node** and remove exclusion if it any had previously been excluded using the
              exclude_node_roles() method.
            *exclude*:
              remove **roles** if any previously been required either by default (as a required_node_role)
              or using the required_node_roles() method.

        scope : `ALL` or None : default None
            If None (default) **roles** are modified for **node** only if it is at the top level of the Composition;
            if *ALL*, then they will be applied to any node within the Composition or one nested with in it.

        context : Context : default None

        """
        def _assign_required_roles(node, node_role_mgr, roles):
            for role in roles:
                node_role_mgr._add_required_node_role(node, role, context)
            node_role_mgr._need_determine_node_roles = True

        def _exclude_roles(node, node_roles_mgr, roles):
            for role in roles:
                node_role_pair = (node, role)
                if node_role_pair not in node_roles_mgr.excluded_node_roles:
                    node_roles_mgr.excluded_node_roles.append(node_role_pair)
                if node_role_pair in self.required_node_roles:
                    node_roles_mgr.required_node_roles.remove(node_role_pair)
                node_roles_mgr._remove_node_role(node, role)
            node_roles_mgr._need_determine_node_roles = True

        roles = convert_to_list(roles)
        require_or_exclude = _assign_required_roles if modification == 'require' else _exclude_roles

        bad_roles = [f"'{role.name}'" for role in roles if role not in NodeRole]
        plural = len(bad_roles) > 1
        s = 's' if plural else ''
        are_is = 'are' if plural else 'is'
        if bad_roles:
            raise NodeRoleError(f"The following invalid NodeRole{s} {are_is} specified for '{node.name}' in the "
                                f"{modification}_node_roles() of '{self.owner.name}': {','.join(bad_roles)}.")

        prohibited_roles = [f"'{role.name}'" for role in roles if
                            (context.source == ContextFlags.COMMAND_LINE and
                             role in {NodeRole.ORIGIN, NodeRole.INTERNAL, NodeRole.SINGLETON,
                                      NodeRole.TERMINAL, NodeRole.CYCLE, NodeRole.FEEDBACK_SENDER,
                                      NodeRole.FEEDBACK_RECEIVER, NodeRole.LEARNING})]
        if prohibited_roles:
            plural = len(prohibited_roles) > 1
            s = 's' if plural else ''
            are_is = 'are' if plural else 'is'
            raise NodeRoleError(f"Attempt to {modification} the following NodeRole{s} for '{node.name}' (in "
                                f"'{self.name}') that cannot be modified by user: {', '.join(prohibited_roles)}.")

        # Look for node in Composition (at top level only if there are nested Compositions)
        if node in self.owner.nodes:
            # node is in top level of Composition
            require_or_exclude(node, self, roles)
            return True

        if scope in {ALL, NESTED}:
            # Search for node in nested Compositions if there are any, and node was not found at the top level
            for comp in self.owner._get_nested_compositions():
                method = comp.require_node_roles if modification == 'require' else comp.exclude_node_roles
                if method(node, roles, NESTED, context):
                    return True
            if scope == NESTED:
                # If nested, don't bail (with error) yet, as Node might be in other nested Comps of outer Comp
                return False
            # If NOT nested, bail with error as the node wasn't found anywhere
            comp_str = f"the Composition or any nested within it"

        else:
            # node not found in Composition (and not searched in nested Compositions if there are any)
            comp_str = f"the Composition"

        raise NodeRoleError(f"The call to the {modification}_node_roles() method of '{self.owner.name}' contained a "
                            f"node '{node.name}' that is not in {comp_str}.")


    def get_nodes_by_role(self, *args, **kwargs):
        """Interface to get_nodes_by_role that enforces prior call to _determine_node_roles
        This should be used in general,
           but avoided in cases where it is called from under _determine_node_roles itself to avoid recursion.
        """
        if self._need_determine_node_roles:
            self._determine_node_roles()
        return self._get_nodes_by_role(*args, **kwargs)

    def _get_nodes_by_role(self, role:NodeRole, scope:Optional[Literal[ALL]]=None)->list:
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
            raise NodeRoleError(f'Invalid NodeRole: {role}.')

        try:
            if scope is ALL:
                return self.get_nested_nodes_by_roles_at_any_level(self.owner, role)
            return [node for node in self.nodes_to_roles if role in self.nodes_to_roles[node]]

        except KeyError as e:
            raise NodeRoleError(f'Node missing from {self}.nodes_to_roles: {e}.')

    def get_roles_by_node(self, *args, **kwargs):
        """Interface to _get_roles_by_node that enforces prior call to _determine_node_roles
        This should be used in general,
           but avoided in cases where it is called from under _determine_node_roles itself to avoid recursion.
        """

        if self._need_determine_node_roles:
            self._determine_node_roles()
        return self._get_roles_by_node(*args, **kwargs)

    def _get_roles_by_node(self, node, scope: Optional[Literal[ALL, NESTED]] = None) -> List[NodeRole]:
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
            from psyneulink.core.compositions.composition import Composition
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
            if node in self.owner._get_all_nodes():
                # Node not found in self.nodes_to_roles[node] of Composition and, if scope, any nested within it
                return []
            if not scope and node not in self.nodes:
                # Node note found in outermost Composition
                raise NodeRoleError(f"Node ('{node.name}') for which roles were requested is not in '{self.name}'")
            # Node was not found anywhere
            raise NodeRoleError(f"Node ('{node.name}') for which roles were requested is not in '{self.name}' "
                                   f"or any Compositions nested within it.")

    def get_required_roles_by_node(self, node) -> List[NodeRole]:
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
        assert isinstance(comp, Composition), (f"PROGRAM ERROR: get_nested_nodes_by_roles_at_any_level() was called "
                                               f"for a NodeRoleManager the owner of which is not a Composition.")
        if self._need_determine_node_roles:
            self._determine_node_roles()

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
    #endregion
