# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* show_graph *************************************************************

import enum
import inspect
import numpy as np
import typecheck as tc

from psyneulink.core.compositions.composition import Composition, NodeRole, CompositionError
from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    OptimizationControlMechanism, AGENT_REP
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection, MappingError
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.utilities import convert_to_list
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ALL, BOLD, BOTH, COMPONENT, CONDITIONS, FUNCTIONS, INSET, LABELS, MECHANISM, MECHANISMS, NESTED, \
    PROJECTION, PROJECTIONS, ROLES, SIMULATIONS, VALUES

__all__ = ['MECH_FUNCTION_PARAMS', 'PORT_FUNCTION_PARAMS', 'show_graph']

# Options for show_node_structure argument of show_graph()
MECH_FUNCTION_PARAMS = "MECHANISM_FUNCTION_PARAMS"
PORT_FUNCTION_PARAMS = "PORT_FUNCTION_PARAMS"

ENCLOSING_G = 'enclosing_g'
NESTING_LEVEL = 'nesting_level'
NUM_NESTING_LEVELS = 'num_nesting_levels'


@tc.typecheck
@handle_external_context(execution_id=NotImplemented, source=ContextFlags.COMPOSITION)
def show_graph(self,
               show_node_structure:tc.any(bool, tc.enum(VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS,
                                                        PORT_FUNCTION_PARAMS, ROLES, ALL))=False,
               show_nested:tc.optional(tc.any(bool,int,dict,tc.enum(NESTED, INSET)))=NESTED,
               show_nested_args:tc.optional(tc.any(bool,dict,tc.enum(ALL)))=ALL,
               show_cim:bool=False,
               show_controller:tc.any(bool, tc.enum(AGENT_REP))=True,
               show_learning:bool=False,
               show_headers:bool=True,
               show_types:bool=False,
               show_dimensions:bool=False,
               show_projection_labels:bool=False,
               direction:tc.enum('BT', 'TB', 'LR', 'RL')='BT',
               # active_items:tc.optional(list)=None,
               active_items=None,
               active_color=BOLD,
               input_color='green',
               output_color='red',
               input_and_output_color='brown',
               # feedback_color='yellow',
               controller_color='blue',
               learning_color='orange',
               composition_color='pink',
               control_projection_arrow='box',
               feedback_shape = 'septagon',
               cycle_shape = 'doublecircle',
               cim_shape='square',
               output_fmt:tc.optional(tc.enum('pdf','gv','jupyter','gif'))='pdf',
               context=None,
               **kwargs):
    """
    show_graph(                           \
       show_node_structure=False,         \
       show_nested=NESTED,                \
       show_nested_args=ALL,              \
       show_cim=False,                    \
       show_controller=True,              \
       show_learning=False,               \
       show_headers=True,                 \
       show_types=False,                  \
       show_dimensions=False,             \
       show_projection_labels=False,      \
       direction='BT',                    \
       active_items=None,                 \
       active_color=BOLD,                 \
       input_color='green',               \
       output_color='red',                \
       input_and_output_color='brown',    \
       controller_color='blue',           \
       composition_color='pink',          \
       feedback_shape = 'septagon',       \
       cycle_shape = 'doublecircle',      \
       cim_shape='square',                \
       output_fmt='pdf',                  \
       context=None)

    Show graphical display of Components in a Composition's graph.

    .. note::
       This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
       (standard with PsyNeuLink pip install)

    See `Visualizing a Composition <Composition_Visualization>` for details and examples.

    Arguments
    ---------

    show_node_structure : bool, VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS, PORT_FUNCTION_PARAMS, ROLES, \
    or ALL : default False
        show a detailed representation of each `Mechanism <Mechanism>` in the graph, including its `Ports <Port>`;
        can have any of the following settings alone or in a list:

        * `True` -- show Ports of Mechanism, but not information about the `value
          <Component.value>` or `function <Component.function>` of the Mechanism or its Ports.

        * *VALUES* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
          <Port_Base.value>` of each of its Ports.

        * *LABELS* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
          <Port_Base.value>` of each of its Ports, using any labels for the values of InputPorts and
          OutputPorts specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
          `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

        * *FUNCTIONS* -- show the `function <Mechanism_Base.function>` of the Mechanism and the `function
          <Port_Base.function>` of its InputPorts and OutputPorts.

        * *MECH_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
          Mechanism in the Composition (only applies if *FUNCTIONS* is True).

        * *PORT_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
          Port of each Mechanism in the Composition (only applies if *FUNCTIONS* is True).

        * *ROLES* -- show the `role <NodeRole>` of the Mechanism in the Composition
          (but not any of the other information;  use *ALL* to show ROLES with other information).

        * *ALL* -- shows the role, `function <Component.function>`, and `value <Component.value>` of the
          Mechanisms in the `Composition` and their `Ports <Port>` (using labels for
          the values, if specified -- see above), including parameters for all functions.

    show_nested : bool | int | NESTED | INSET : default NESTED
        specifies whether or not to show `nested Compositions <Composition_Nested>` and, if so, how many
        levels of nesting to show (*NESTED*, True or int) -- with Projections shown directly from Components
        in an enclosing Composition to and from ones in the nested Composition; or each nested Composition as
        a separate inset (*INSET*).  *NESTED* specifies all levels of nesting shown; 0 specifies none (same as
        False), and a non-zero integer species that number of nested levels to shown.  Compsitions nested at the
        specified level are shown as a node (pink box by default). and ones below the specified level are not
        shown at all.

     show_nested_args : bool | dict : default ALL
        specifies arguments in call to show_graph passed to `nested Composition(s) <Composition_Nested>` if
        **show_nested** is specified.  A dict can be used to specify any of the arguments allowed for
        show_graph to be used for the nested Composition(s);  *ALL* passes all arguments specified for the main
        Composition to the nested one(s);  True uses the default values of show_graph args for the nested
        Composition(s).

    show_cim : bool : default False
        specifies whether or not to show the Composition's `input_CIM <Composition.input_CIM>`, `parameter_CIM
        <Composition.parameter_CIM>`, and `output_CIM <Composition.output_CIM>` `CompositionInterfaceMechanisms
        <CompositionInterfaceMechanism>` (CIMs).

    show_controller :  bool or AGENT_REP : default True
        specifies whether or not to show the Composition's `controller <Composition.controller>` and associated
        `objective_mechanism <ControlMechanism.objective_mechanism>` if it has one.  If the controller is an
        OptimizationControlMechanism and it has an `agent_rep <OptimizationControlMechanism>`, then specifying
        *AGENT_REP* will also show that.  All control-related items are displayed in the color specified for
        **controller_color**.

    show_learning : bool or ALL : default False
        specifies whether or not to show the `learning components <Composition_Learning_Components>` of the
        `Composition`; they will all be displayed in the color specified for **learning_color**.
        Projections that receive a `LearningProjection` will be shown as a diamond-shaped node.
        If set to *ALL*, all Projections associated with learning will be shown:  the LearningProjections
        as well as from `ProcessingMechanisms <ProcessingMechanism>` to `LearningMechanisms <LearningMechanism>`
        that convey error and activation information;  if set to `True`, only the LearningPojections are shown.

    show_projection_labels : bool : default False
        specifies whether or not to show names of projections.

    show_headers : bool : default True
        specifies whether or not to show headers in the subfields of a Mechanism's node;  only takes effect if
        **show_node_structure** is specified (see above).

    show_types : bool : default False
        specifies whether or not to show type (class) of `Mechanism <Mechanism>` in each node label.

    show_dimensions : bool : default False
        specifies whether or not to show dimensions for the `variable <Component.variable>` and `value
        <Component.value>` of each Component in the graph (and/or MappingProjections when show_learning
        is `True`);  can have the following settings:

        * *MECHANISMS* -- shows `Mechanism <Mechanism>` input and output dimensions.  Input dimensions are shown
          in parentheses below the name of the Mechanism; each number represents the dimension of the `variable
          <InputPort.variable>` for each `InputPort` of the Mechanism; Output dimensions are shown above
          the name of the Mechanism; each number represents the dimension for `value <OutputPort.value>` of each
          of `OutputPort` of the Mechanism.

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

    input_color : keyword : default 'green',
        specifies the display color for `INPUT <NodeRole.INPUT>` Nodes in the Composition

    output_color : keyword : default 'red',
        specifies the display color for `OUTPUT` Nodes in the Composition

    input_and_output_color : keyword : default 'brown'
        specifies the display color of nodes that are both an `INPUT <NodeRole.INPUT>` and an `OUTPUT
        <NodeRole.OUTPUT>` Node in the Composition

    COMMENT:
    feedback_color : keyword : default 'yellow'
        specifies the display color of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.
    COMMENT

    controller_color : keyword : default 'blue'
        specifies the color in which the controller components are displayed

    learning_color : keyword : default 'orange'
        specifies the color in which the learning components are displayed

    composition_color : keyword : default 'brown'
        specifies the display color of nodes that represent nested Compositions.

    feedback_shape : keyword : default 'septagon'
        specifies the display shape of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.

    cycle_shape : keyword : default 'doublecircle'
        specifies the display shape of nodes that are assigned the `NodeRole` `CYCLE`.

    cim_shape : default 'square'
        specifies the display color input_CIM and output_CIM nodes

    output_fmt : keyword or None : default 'pdf'
        'pdf': generate and open a pdf with the visualization;
        'jupyter': return the object (for working in jupyter/ipython notebooks);
        'gv': return graphviz object
        'gif': return gif used for animation
        None : return None

    Returns
    -------

    `pdf` or Graphviz graph object :
        PDF: (placed in current directory) if :keyword:`output_fmt` arg is 'pdf';
        Graphviz graph object if :keyword:`output_fmt` arg is 'gv' or 'jupyter';
        gif if :keyword:`output_fmt` arg is 'gif'.

    """

    # HELPER METHODS ----------------------------------------------------------------------

    tc.typecheck
    _locals = locals().copy()

    def _assign_processing_components(g, rcvr, show_nested, show_nested_args, enclosing_g, nesting_level):
        """Assign nodes to graph"""

        from psyneulink.core.compositions.composition import Composition, NodeRole
        # DEAL WITH NESTED COMPOSITION

        # User passed args for nested Composition
        if isinstance(rcvr, Composition):
            if show_nested:
                output_fmt_arg = {'output_fmt':'gv'}
                if isinstance(show_nested_args, dict):
                    args = show_nested_args
                    args.update(output_fmt_arg)
                elif show_nested_args == ALL:
                    # Pass args from main call to show_graph to call for nested Composition
                    args = dict({k:_locals[k] for k in list(inspect.signature(show_graph).parameters)})
                    args.update(output_fmt_arg)
                    if kwargs:
                        args['kwargs'] = kwargs
                    else:
                        del  args['kwargs']
                else:
                    # Use default args for nested Composition
                    args = output_fmt_arg
                args.update({'self': rcvr,
                             ENCLOSING_G:g,
                             NESTING_LEVEL:nesting_level + 1,
                             NUM_NESTING_LEVELS:num_nesting_levels})

                # Get subgraph for nested Composition
                nested_comp_graph = show_graph(**args)

                nested_comp_graph.name = "cluster_" + rcvr.name
                rcvr_label = rcvr.name
                # if rcvr in self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
                #     nested_comp_graph.attr(color=feedback_color)
                if rcvr in self.get_nodes_by_role(NodeRole.INPUT) and \
                        rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
                    nested_comp_graph.attr(color=input_and_output_color)
                elif rcvr in self.get_nodes_by_role(NodeRole.INPUT):
                    nested_comp_graph.attr(color=input_color)
                elif rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
                    nested_comp_graph.attr(color=output_color)
                nested_comp_graph.attr(label=rcvr_label)
                g.subgraph(nested_comp_graph)

                if show_nested is NESTED:
                    return

        # DEAL WITH LEARNING
        # If rcvr is a learning component and not an INPUT node,
        #    break and handle in _assign_learning_components()
        #    (node: this allows TARGET node for learning to remain marked as an INPUT node)
        if (NodeRole.LEARNING in self.nodes_to_roles[rcvr]
                and not NodeRole.INPUT in self.nodes_to_roles[rcvr]):
            return

        # DEAL WITH CONTROLLER's OBJECTIVEMECHANIMS
        # If rcvr is ObjectiveMechanism for Composition's controller,
        #    break and handle in _assign_controller_components()
        if (isinstance(rcvr, ObjectiveMechanism)
                and self.controller
                and rcvr is self.controller.objective_mechanism):
            return

        # IMPLEMENT RECEIVER NODE:
        #    set rcvr shape, color, and penwidth based on node type
        rcvr_rank = 'same'

        # SET SPECIAL SHAPES

        # Cycle or Feedback Node
        if isinstance(rcvr, Composition):
            node_shape = composition_shape
        elif rcvr in self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
            node_shape = feedback_shape
        elif rcvr in self.get_nodes_by_role(NodeRole.CYCLE):
            node_shape = cycle_shape
        else:
            node_shape = mechanism_shape

        # SET STROKE AND COLOR
        #    Based on Input, Output, Composition and/or Active

        # Get condition if any associated with rcvr
        if rcvr in self.scheduler.conditions:
            condition = self.scheduler.conditions[rcvr]
        else:
            condition = None

        # INPUT and OUTPUT Node
        if rcvr in self.get_nodes_by_role(NodeRole.INPUT) and \
                rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
            if rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = input_and_output_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(bold_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                rcvr_color = input_and_output_color
                rcvr_penwidth = str(bold_width)

        # INPUT Node
        elif rcvr in self.get_nodes_by_role(NodeRole.INPUT):
            if rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = input_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(bold_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                rcvr_color = input_color
                rcvr_penwidth = str(bold_width)
            rcvr_rank = input_rank

        # OUTPUT Node
        elif rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
            if rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = output_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(bold_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                rcvr_color = output_color
                rcvr_penwidth = str(bold_width)
            rcvr_rank = output_rank

        # Composition that is neither an INPUT Node nor an OUTPUT Node
        elif isinstance(rcvr, Composition) and show_nested is not NESTED:
            if rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = composition_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(bold_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                rcvr_color = composition_color
                rcvr_penwidth = str(bold_width)

        # Active Node that is none of the above
        elif rcvr in active_items:
            if active_color == BOLD:
                rcvr_color = default_node_color
            else:
                rcvr_color = active_color
            rcvr_penwidth = str(default_width + active_thicker_by)
            self.active_item_rendered = True

        # Inactive Node that is none of the above
        else:
            rcvr_color = default_node_color
            rcvr_penwidth = str(default_width)

        # Implement rcvr node
        rcvr_label = _get_graph_node_label(self,
                                           rcvr,
                                           show_types,
                                           show_dimensions)

        if show_node_structure and isinstance(rcvr, Mechanism):
            g.node(rcvr_label,
                   rcvr._show_structure(**node_struct_args, node_border=rcvr_penwidth, condition=condition),
                   shape=struct_shape,
                   color=rcvr_color,
                   rank=rcvr_rank,
                   penwidth=rcvr_penwidth)
        else:
            g.node(rcvr_label,
                   shape=node_shape,
                   color=rcvr_color,
                   rank=rcvr_rank,
                   penwidth=rcvr_penwidth)

        # Implement sender edges from Nodes within Composition
        sndrs = processing_graph[rcvr]
        _assign_incoming_edges(g, rcvr, rcvr_label, sndrs, enclosing_g=enclosing_g)

    def _assign_cim_components(g, cims, enclosing_g):

        cim_rank = 'same'

        for cim in cims:

            # Skip cim if it is not doing anything
            if not (cim.afferents or cim.efferents):
                continue

            # ASSIGN CIM NODE ****************************************************************

            # Assign Node attributes

            # Also take opportunity to validate that cim is input_CIM, parameter_CIM or output_CIM
            if cim is self.input_CIM:
                cim_type_color = input_color
            elif cim is self.parameter_CIM:
                cim_type_color = controller_color
            elif cim is self.output_CIM:
                cim_type_color = output_color
            else:
                assert False, \
                    f'PROGRAM ERROR: _assign_cim_components called with node ' \
                    f'that is not input_CIM, parameter_CIM, or output_CIM'
            cim_penwidth = str(default_width)
            if cim in active_items:
                if active_color == BOLD:
                    cim_color = cim_type_color
                else:
                    cim_color = active_color
                cim_penwidth = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                cim_color = cim_type_color

            compact_cim = not cim.afferents or show_nested is not NESTED

            # Create CIM node
            cim_label = _get_graph_node_label(self, cim, show_types, show_dimensions)
            if show_node_structure:
                g.node(cim_label,
                       cim._show_structure(**node_struct_args,
                                           node_border=cim_penwidth,
                                           compact_cim=compact_cim),
                       shape=struct_shape,
                       color=cim_color,
                       rank=cim_rank,
                       penwidth=cim_penwidth)

            else:
                g.node(cim_label,
                       shape=cim_shape,
                       color=cim_color,
                       rank=cim_rank,
                       penwidth=cim_penwidth)

            # FIX 6/2/20:  THIS CAN BE CONDENSED (ABSTACTED INTO GENERIC FUNCTION TAKING cim-SPECIFIC PARAMETERS)
            # ASSIGN CIM PROJECTIONS ****************************************************************

            # INPUT_CIM -----------------------------------------------------------------------------

            if cim is self.input_CIM:

                # Projections from Node(s) in enclosing Composition to input_CIM
                for input_port in self.input_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:

                        # Get label for Node that sends the input (sndr_label)
                        sndr_node_output_port = proj.sender
                        # Skip if sender is a CIM (handled by enclosing Composition's call to this method)
                        if isinstance(sndr_node_output_port.owner, CompositionInterfaceMechanism):
                            continue
                        # Skip if there is no outer Composition (enclosing_g),
                        #    or Projections between Compositions are not being shown (show_nested=INSET)
                        if not enclosing_g or show_nested is INSET:
                            continue
                        sndr_node_output_port_owner = sndr_node_output_port.owner

                        sndr_label = _get_graph_node_label(self,
                                                           sndr_node_output_port_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's receiver
                            rcvr_cim_proj_label = f"{cim_label}:{InputPort.__name__}-{proj.receiver.name}"
                            if (isinstance(sndr_node_output_port_owner, Composition)
                                    and show_nested is not NESTED):
                                sndr_output_node_proj_label = sndr_label
                            else:
                                # Need to use direct reference to proj.sender rather than snder_input_node
                                #    since could be Composition, which does not have a get_port_name attribute
                                sndr_output_node_proj_label = \
                                    f"{sndr_label}:{OutputPort.__name__}-{proj.sender.name}"
                                # rcvr_input_node_proj_label = \
                                #     f"{rcvr_label}:" \
                                #     f"{rcvr_input_node_proj_owner._get_port_name(rcvr_input_node_proj)}"
                        else:
                            rcvr_cim_proj_label = cim_label
                            sndr_output_node_proj_label = sndr_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.sender.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        else:
                            label = ''

                        enclosing_g.edge(sndr_output_node_proj_label, rcvr_cim_proj_label, label=label,
                                         color=proj_color, penwidth=proj_width)

                # Projections from input_CIM to INPUT nodes
                for output_port in self.input_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:

                        # Get label for Node that receives the input (rcvr_label)
                        rcvr_input_node_proj = proj.receiver
                        if (isinstance(rcvr_input_node_proj.owner, CompositionInterfaceMechanism)
                                and not show_nested is NESTED):
                            rcvr_input_node_proj_owner = rcvr_input_node_proj.owner.composition
                        else:
                            rcvr_input_node_proj_owner = rcvr_input_node_proj.owner

                        if rcvr_input_node_proj_owner is self.controller:
                            # Projections to contoller are handled under _assign_controller_components
                            continue

                        # Validate the Projection is to an INPUT node or a node that is shadowing one
                        if ((rcvr_input_node_proj_owner in self.nodes_to_roles and
                             not NodeRole.INPUT in self.nodes_to_roles[rcvr_input_node_proj_owner])
                                and (proj.receiver.shadow_inputs in self.nodes_to_roles and
                                     not NodeRole.INPUT in self.nodes_to_roles[proj.receiver.shadow_inputs])):
                            raise CompositionError(f"Projection from input_CIM of {self.name} to node "
                                                   f"{rcvr_input_node_proj_owner} that is not an "
                                                   f"{NodeRole.INPUT.name} node or shadowing its "
                                                   f"{NodeRole.INPUT.name.lower()}.")
                        rcvr_label = _get_graph_node_label(self,
                                                           rcvr_input_node_proj_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's sender
                            sndr_cim_proj_label = f"{cim_label}:{OutputPort.__name__}-{proj.sender.name}"
                            if (isinstance(rcvr_input_node_proj_owner, Composition)
                                    and show_nested is not NESTED):
                                rcvr_input_node_proj_label = rcvr_label
                            else:
                                # Need to use direct reference to proj.receiver rather than rcvr_input_node_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                rcvr_input_node_proj_label = \
                                    f"{rcvr_label}:{InputPort.__name__}-{proj.receiver.name}"
                                # rcvr_input_node_proj_label = \
                                #     f"{rcvr_label}:" \
                                #     f"{rcvr_input_node_proj_owner._get_port_name(rcvr_input_node_proj)}"
                        else:
                            sndr_cim_proj_label = cim_label
                            rcvr_input_node_proj_label = rcvr_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        else:
                            label = ''
                        g.edge(sndr_cim_proj_label, rcvr_input_node_proj_label, label=label,
                           color=proj_color, penwidth=proj_width)

            # PARAMETER_CIM -------------------------------------------------------------------------

            if cim is self.parameter_CIM:

                # Projections from ControlMechanism(s) in enclosing Composition to parameter_CIM
                # (other than from controller;  that is handled in _assign_controller_compoents)
                for input_port in self.parameter_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:

                        # Get label for Node that sends the ControlProjection (sndr label)
                        ctl_mech_output_port = proj.sender
                        # Skip if sender is cim (handled by enclosing Composition's call to this method)
                        #   or Projections to cim aren't being shown (not NESTED)
                        if (isinstance(ctl_mech_output_port.owner, CompositionInterfaceMechanism)
                                or show_nested is not NESTED):
                            continue
                        else:
                            ctl_mech_output_port_owner = ctl_mech_output_port.owner
                        assert isinstance(ctl_mech_output_port_owner, ControlMechanism), \
                            f"PROGRAM ERROR: parameter_CIM of {self.name} recieves a Projection " \
                            f"from a Node from other than a {ControlMechanism.__name__}."
                        # Skip Projections from controller (handled in _assign_controller_components)
                        if ctl_mech_output_port_owner.composition:
                            continue
                        # Skip if there is no outer Composition (enclosing_g),
                        #    or Projections acorss nested Compositions are not being shown (show_nested=INSET)
                        if not enclosing_g or show_nested is INSET:
                            continue
                        sndr_label = _get_graph_node_label(self,
                                                           ctl_mech_output_port_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's receiver
                            rcvr_cim_proj_label = f"{cim_label}:{InputPort.__name__}-{proj.receiver.name}"
                            # Need to use direct reference to proj.sender rather than snder_input_node
                            #    since could be Composition, which does not have a get_port_name attribute
                            # sndr_output_node_proj_label = \
                            #     f"{sndr_label}:{OutputPort.__name__}-{proj.sender.name}"
                            rcvr_input_node_proj_label = \
                                f"{sndr_label}:" \
                                f"{ctl_mech_output_port_owner._get_port_name(ctl_mech_output_port)}"
                        else:
                            rcvr_cim_proj_label = cim_label
                            sndr_output_node_proj_label = sndr_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.sender.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        else:
                            label = ''
                        enclosing_g.edge(sndr_output_node_proj_label, rcvr_cim_proj_label, label=label,
                                         color=proj_color, penwidth=proj_width)

                # Projections from parameter_CIM to Nodes that are being modulated
                for output_port in self.parameter_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:

                        # Get label for Node that receives modulation (modulated_mech_label)
                        rcvr_modulated_mech_proj = proj.receiver
                        if (isinstance(rcvr_modulated_mech_proj.owner, CompositionInterfaceMechanism)
                                and not show_nested is NESTED):
                            rcvr_modulated_mech_proj_owner = rcvr_modulated_mech_proj.owner.composition
                        else:
                            rcvr_modulated_mech_proj_owner = rcvr_modulated_mech_proj.owner

                        if rcvr_modulated_mech_proj_owner is self.controller:
                            # Projections to contoller are handled under _assign_controller_components
                            # Note: at present controllers are not modulable; here for possible future condition(s)
                            continue
                        rcvr_label = _get_graph_node_label(self,
                                                           rcvr_modulated_mech_proj_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label for CIM's port as edge's sender
                            sndr_cim_proj_label = f"{cim_label}:{OutputPort.__name__}-{proj.sender.name}"
                            if (isinstance(rcvr_modulated_mech_proj_owner, Composition)
                                    and not show_nested is not NESTED):
                                rcvr_modulated_mec_proj_label = rcvr_label
                            else:
                                # Need to use direct reference to proj.receiver rather than rcvr_modulated_mec_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                rcvr_modulated_mec_proj_label = \
                                    f"{rcvr_label}:{ParameterPort.__name__}-{proj.receiver.name}"
                                # rcvr_modulated_mec_proj_label = \
                                #     f"{rcvr_label}:" \
                                #     f"{rcvr_input_node_proj_owner._get_port_name(rcvr_modulated_mec_proj)}"
                        else:
                            sndr_cim_proj_label = cim_label
                            rcvr_modulated_mec_proj_label = rcvr_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                proj_color = controller_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = controller_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        else:
                            label = ''
                        g.edge(sndr_cim_proj_label, rcvr_modulated_mec_proj_label, label=label,
                               color=proj_color, penwidth=proj_width)

            # OUTPUT_CIM ----------------------------------------------------------------------------

            if cim is self.output_CIM:

                # Projections from OUTPUT nodes to output_CIM
                for input_port in self.output_CIM.input_ports:
                    projs = input_port.path_afferents
                    for proj in projs:

                        sndr_output_node_proj = proj.sender
                        if (isinstance(sndr_output_node_proj.owner, CompositionInterfaceMechanism)
                                and not show_nested is NESTED):
                            sndr_output_node_proj_owner = sndr_output_node_proj.owner.composition
                        else:
                            sndr_output_node_proj_owner = sndr_output_node_proj.owner

                        # Validate the Projection is from an OUTPUT node
                        if ((sndr_output_node_proj_owner in self.nodes_to_roles and
                             not NodeRole.OUTPUT in self.nodes_to_roles[sndr_output_node_proj_owner])):
                            raise CompositionError(f"Projection to output_CIM of {self.name} "
                                                   f"from node {sndr_output_node_proj_owner} that is not "
                                                   f"an {NodeRole.OUTPUT} node.")

                        sndr_label = _get_graph_node_label(self,
                                                           sndr_output_node_proj_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label of CIM's port as edge's receiver
                            rcvr_cim_proj_label = f"{cim_label}:{InputPort.__name__}-{proj.receiver.name}"
                            if (isinstance(sndr_output_node_proj_owner, Composition)
                                    and show_nested is not NESTED):
                                sndr_output_node_proj_label = sndr_label
                            else:
                                # Need to use direct reference to proj.sender rather than sndr_output_node_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                sndr_output_node_proj_label = \
                                    f"{sndr_label}:{OutputPort.__name__}-{proj.sender.name}"
                                # sndr_output_node_proj_label = \
                                #     f"{sndr_label}:" \
                                #     f"{sndr_output_node_proj_owner._get_port_name(sndr_output_node_proj)}"
                        else:
                            sndr_output_node_proj_label = sndr_label
                            rcvr_cim_proj_label = cim_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        else:
                            label = ''
                        g.edge(sndr_output_node_proj_label, rcvr_cim_proj_label, label=label,
                               color=proj_color, penwidth=proj_width)

                # Projections from output_CIM to Node(s) in enclosing Composition
                for output_port in self.output_CIM.output_ports:
                    projs = output_port.efferents
                    for proj in projs:

                        rcvr_node_input_port = proj.receiver
                        # Skip if receiver is cim (handled by enclosing Composition's call to this method)
                        if isinstance(rcvr_node_input_port.owner, CompositionInterfaceMechanism):
                            continue
                        # Skip if there is no inner Composition (show_nested!=NESTED) or
                        #   or Projections across nested Compositions are not being shown (show_nested=INSET)
                        if not enclosing_g or show_nested is INSET:
                            continue
                        rcvr_node_input_port_owner = rcvr_node_input_port.owner

                        rcvr_label = _get_graph_node_label(self,
                                                           rcvr_node_input_port_owner,
                                                           show_types, show_dimensions)
                        # Construct edge name
                        if show_node_structure:
                            # Get label of CIM's port as edge's receiver
                            sndr_cim_proj_label = f"{cim_label}:{OutputPort.__name__}-{proj.sender.name}"
                            if (isinstance(rcvr_node_input_port_owner, Composition)
                                    and show_nested is not NESTED):
                                rcvr_input_node_proj_label = rcvr_label
                            else:
                                # Need to use direct reference to proj.sender rather than sndr_output_node_proj
                                #    since could be Composition, which does not have a get_port_name attribute
                                rcvr_input_node_proj_label = \
                                    f"{rcvr_label}:{InputPort.__name__}-{proj.receiver.name}"
                                # rcvr_input_node_proj_label = \
                                #     f"{sndr_label}:" \
                                #     f"{sndr_output_node_proj_owner._get_port_name(sndr_output_node_proj)}"
                        else:
                            rcvr_input_node_proj_label = rcvr_label
                            sndr_cim_proj_label = cim_label

                        # Render Projection
                        if any(item in active_items for item in {proj, proj.sender.owner}):
                            if active_color == BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        if show_projection_labels:
                            label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        else:
                            label = ''
                        enclosing_g.edge(sndr_cim_proj_label, rcvr_input_node_proj_label, label=label,
                                         color=proj_color, penwidth=proj_width)

    def _assign_controller_components(g):
        """Assign control nodes and edges to graph"""

        controller = self.controller
        if controller is None:
            # Only warn if there is no controller *and* no ControlProjections from an outer Composition
            if not self.parameter_CIM.output_ports:
                warnings.warn(f"{self.name} has not been assigned a \'controller\', "
                              f"so \'show_controller\' option in call to its show_graph() method will be ignored.")
            return

        if controller in active_items:
            if active_color == BOLD:
                ctlr_color = controller_color
            else:
                ctlr_color = active_color
            ctlr_width = str(default_width + active_thicker_by)
            self.active_item_rendered = True
        else:
            ctlr_color = controller_color
            ctlr_width = str(default_width)

        # Assign controller node
        node_shape = mechanism_shape
        ctlr_label = _get_graph_node_label(self, controller, show_types, show_dimensions)
        if show_node_structure:
            g.node(ctlr_label,
                   controller._show_structure(**node_struct_args, node_border=ctlr_width,
                                             condition=self.controller_condition),
                   shape=struct_shape,
                   color=ctlr_color,
                   penwidth=ctlr_width,
                   rank=control_rank
                   )
        else:
            g.node(ctlr_label,
                    color=ctlr_color, penwidth=ctlr_width, shape=node_shape,
                    rank=control_rank)

        # outgoing edges (from controller to ProcessingMechanisms)
        for control_signal in controller.control_signals:
            for ctl_proj in control_signal.efferents:

                # Skip ControlProjections not in the Composition
                if ctl_proj not in self.projections:
                    continue

                # Construct edge name  ---------------------------------------------------

                # Get receiver label for ControlProjection as base for edge's receiver label
                # First get label for receiver's owner node (Mechanism or nested Composition), used below
                ctl_proj_rcvr = ctl_proj.receiver
                # If receiver is a parameter_CIM
                if isinstance(ctl_proj_rcvr.owner, CompositionInterfaceMechanism):
                    # PATCH 6/7/20 to deal with ControlProjections across more than one level of nesting:
                    rcvr_comp = ctl_proj_rcvr.owner.composition
                    def find_rcvr_comp(r, c, l):
                        """Find deepest enclosing composition within range of num_nesting_levels"""
                        if (num_nesting_levels is not None and l > num_nesting_levels):
                            return c, l
                        elif r in c.nodes:
                            return r, l
                        l+=1
                        for nested_c in [nc for nc in c.nodes if isinstance(nc, Composition)]:
                            return find_rcvr_comp(r, nested_c, l)
                        return None
                    project_to_node = False
                    try:
                        enclosing_comp, l = find_rcvr_comp(rcvr_comp, self, 0)
                    except TypeError:
                        raise CompositionError(f"ControlProjection not found from {controller} in "
                                               f"{self.name} to {rcvr_comp}")
                    if show_nested is NESTED:
                        # Node that receives ControlProjection is within num_nesting_levels, so show it
                        if num_nesting_levels is None or l < num_nesting_levels:
                            project_to_node = True
                        # Node is not within range, but its Composition is,
                        #     so leave rcvr_comp assigned to that, and don't project_to_node
                        elif l == num_nesting_levels:
                            pass
                        # Receiver's Composition is not within num_nesting_levels, so use closest one that encloses it
                        else:
                            rcvr_comp = enclosing_comp
                    else:
                        rcvr_comp = enclosing_comp
                    # PATCH 6/6/20 END

                    # PATCH 6/6/20:
                    # if show_cim and show_nested is NESTED:
                    if show_cim and project_to_node:
                    # PATCH 6/6/20 END
                        # Use Composition's parameter_CIM port
                        ctl_proj_rcvr_owner = ctl_proj_rcvr.owner
                    # PATCH 6/6/20:
                    # elif show_nested is NESTED:
                    elif project_to_node:
                    # PATCH 6/6/20 END
                        # Use ParameterPort of modulated Mechanism in nested Composition
                        parameter_port_map = ctl_proj_rcvr.owner.composition.parameter_CIM_ports
                        ctl_proj_rcvr = next((k for k,v in parameter_port_map.items()
                                                    if parameter_port_map[k][0] is ctl_proj_rcvr), None)
                        ctl_proj_rcvr_owner = ctl_proj_rcvr.owner
                    else:
                        # Use Composition if show_cim is False
                        # PATCH 6/6/20:
                        # ctl_proj_rcvr_owner = ctl_proj_rcvr.owner.composition
                        ctl_proj_rcvr_owner = rcvr_comp
                        # PATCH 6/6/20 END
                # In all other cases, use Port (either ParameterPort of a Mech, or parameter_CIM for nested comp)
                else:
                    ctl_proj_rcvr_owner = ctl_proj_rcvr.owner
                rcvr_label = _get_graph_node_label(self, ctl_proj_rcvr_owner, show_types, show_dimensions)

                # Get sender and receiver labels for edge
                if show_node_structure:
                    # Get label for controller's port as edge's sender
                    ctl_proj_sndr_label = ctlr_label + ':' + controller._get_port_name(control_signal)
                    # Get label for edge's receiver as owner Mechanism:
                    if (isinstance(ctl_proj_rcvr.owner, CompositionInterfaceMechanism) and show_nested is not NESTED):
                        ctl_proj_rcvr_label = rcvr_label
                    # Get label for edge's receiver as Port:
                    else:
                        ctl_proj_rcvr_label = rcvr_label + ':' + ctl_proj_rcvr_owner._get_port_name(ctl_proj_rcvr)
                else:
                    ctl_proj_sndr_label = ctlr_label
                    ctl_proj_rcvr_label = rcvr_label

                # Assign colors, penwidth and label displayed for ControlProjection ---------------------
                if controller in active_items:
                    if active_color == BOLD:
                        ctl_proj_color = controller_color
                    else:
                        ctl_proj_color = active_color
                    ctl_proj_width = str(default_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    ctl_proj_color = controller_color
                    ctl_proj_width = str(default_width)
                if show_projection_labels:
                    edge_label = ctl_proj.name
                else:
                    edge_label = ''

                # Construct edge -----------------------------------------------------------------------
                g.edge(ctl_proj_sndr_label,
                       ctl_proj_rcvr_label,
                       label=edge_label,
                       color=ctl_proj_color,
                       penwidth=ctl_proj_width
                       )

        # If controller has objective_mechanism, assign its node and Projections
        if controller.objective_mechanism:
            # get projection from ObjectiveMechanism to ControlMechanism
            objmech_ctlr_proj = controller.input_port.path_afferents[0]
            if controller in active_items:
                if active_color == BOLD:
                    objmech_ctlr_proj_color = controller_color
                else:
                    objmech_ctlr_proj_color = active_color
                objmech_ctlr_proj_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                objmech_ctlr_proj_color = controller_color
                objmech_ctlr_proj_width = str(default_width)

            # get ObjectiveMechanism
            objmech = objmech_ctlr_proj.sender.owner
            if objmech in active_items:
                if active_color == BOLD:
                    objmech_color = controller_color
                else:
                    objmech_color = active_color
                objmech_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                objmech_color = controller_color
                objmech_width = str(default_width)

            objmech_label = _get_graph_node_label(self, objmech, show_types, show_dimensions)
            if show_node_structure:
                if objmech in self.scheduler.conditions:
                    condition = self.scheduler.conditions[objmech]
                else:
                    condition = None
                g.node(objmech_label,
                       objmech._show_structure(**node_struct_args, node_border=ctlr_width, condition=condition),
                       shape=struct_shape,
                       color=objmech_color,
                       penwidth=ctlr_width,
                       rank=control_rank
                       )
            else:
                g.node(objmech_label,
                        color=objmech_color, penwidth=objmech_width, shape=node_shape,
                        rank=control_rank)

            # objmech to controller edge
            if show_projection_labels:
                edge_label = objmech_ctlr_proj.name
            else:
                edge_label = ''
            if show_node_structure:
                obj_to_ctrl_label = objmech_label + ':' + objmech._get_port_name(objmech_ctlr_proj.sender)
                ctlr_from_obj_label = ctlr_label + ':' + objmech._get_port_name(objmech_ctlr_proj.receiver)
            else:
                obj_to_ctrl_label = objmech_label
                ctlr_from_obj_label = ctlr_label
            g.edge(obj_to_ctrl_label, ctlr_from_obj_label, label=edge_label,
                   color=objmech_ctlr_proj_color, penwidth=objmech_ctlr_proj_width)

            # incoming edges (from monitored mechs to objective mechanism)
            for input_port in objmech.input_ports:
                for projection in input_port.path_afferents:
                    if objmech in active_items:
                        if active_color == BOLD:
                            proj_color = controller_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        proj_color = controller_color
                        proj_width = str(default_width)
                    if show_node_structure:
                        sndr_proj_label = _get_graph_node_label(self,
                                                                projection.sender.owner,
                                                                show_types,
                                                                show_dimensions) + \
                                          ':' + objmech._get_port_name(projection.sender)
                        objmech_proj_label = objmech_label + ':' + objmech._get_port_name(input_port)
                    else:
                        sndr_proj_label = _get_graph_node_label(self,
                                                                projection.sender.owner,
                                                                show_types,
                                                                show_dimensions)
                        objmech_proj_label = _get_graph_node_label(self,
                                                                   objmech,
                                                                   show_types,
                                                                   show_dimensions)
                    if show_projection_labels:
                        edge_label = projection.name
                    else:
                        edge_label = ''
                    g.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                           color=proj_color, penwidth=proj_width)

        # If controller has an agent_rep, assign its node and edges (not Projections per se)
        if hasattr(controller, 'agent_rep') and controller.agent_rep and show_controller==AGENT_REP :
            # get agent_rep
            agent_rep = controller.agent_rep
            # controller is active, treat
            if controller in active_items:
                if active_color == BOLD:
                    agent_rep_color = controller_color
                else:
                    agent_rep_color = active_color
                agent_rep_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                agent_rep_color = controller_color
                agent_rep_width = str(default_width)

            # agent_rep node
            agent_rep_label = _get_graph_node_label(self, agent_rep, show_types, show_dimensions)
            g.node(agent_rep_label,
                    color=agent_rep_color, penwidth=agent_rep_width, shape=agent_rep_shape,
                    rank=control_rank)

            # agent_rep <-> controller edges
            g.edge(agent_rep_label, ctlr_label, color=agent_rep_color, penwidth=agent_rep_width)
            g.edge(ctlr_label, agent_rep_label, color=agent_rep_color, penwidth=agent_rep_width)

        # get any other incoming edges to controller (i.e., other than from ObjectiveMechanism)
        senders = set()
        for i in controller.input_ports[1:]:
            for p in i.path_afferents:
                senders.add(p.sender.owner)
        _assign_incoming_edges(g, controller, ctlr_label, senders, proj_color=ctl_proj_color, enclosing_g=None)

    def _assign_learning_components(g):
        """Assign learning nodes and edges to graph"""

        # Get learning_components, with exception of INPUT (i.e. TARGET) nodes
        #    (i.e., allow TARGET node to continue to be marked as an INPUT node)
        learning_components = [node for node in self.learning_components
                               if not NodeRole.INPUT in self.nodes_to_roles[node]]
        # learning_components.extend([node for node in self.nodes if
        #                             NodeRole.AUTOASSOCIATIVE_LEARNING in
        #                             self.nodes_to_roles[node]])

        for rcvr in learning_components:
            # if rcvr is Projection, skip (handled in _assign_processing_components)
            if isinstance(rcvr, MappingProjection):
                return

            # Get rcvr info
            rcvr_label = _get_graph_node_label(self, rcvr, show_types, show_dimensions)
            if rcvr in active_items:
                if active_color == BOLD:
                    rcvr_color = learning_color
                else:
                    rcvr_color = active_color
                rcvr_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                rcvr_color = learning_color
                rcvr_width = str(default_width)

            # rcvr is a LearningMechanism or ObjectiveMechanism (ComparatorMechanism)
            # Implement node for Mechanism
            if show_node_structure:
                g.node(rcvr_label,
                        rcvr._show_structure(**node_struct_args),
                        rank=learning_rank, color=rcvr_color, penwidth=rcvr_width)
            else:
                g.node(rcvr_label,
                        color=rcvr_color, penwidth=rcvr_width,
                        rank=learning_rank, shape=mechanism_shape)

            # Implement sender edges
            sndrs = processing_graph[rcvr]
            _assign_incoming_edges(g, rcvr, rcvr_label, sndrs)

    def _render_projection_as_node(g, proj, label,
                                   proj_color, proj_width,
                                   sndr_label=None,
                                   rcvr_label=None):

        proj_receiver = proj.receiver.owner

        # Node for Projection
        g.node(label, shape=learning_projection_shape, color=proj_color, penwidth=proj_width)

        # FIX: ??
        if proj_receiver in active_items:
            # edge_color = proj_color
            # edge_width = str(proj_width)
            if active_color == BOLD:
                edge_color = proj_color
            else:
                edge_color = active_color
            edge_width = str(default_width + active_thicker_by)
        else:
            edge_color = default_node_color
            edge_width = str(default_width)

        # Edges to and from Projection node
        if sndr_label:
            G.edge(sndr_label, label, arrowhead='none',
                   color=edge_color, penwidth=edge_width)
        if rcvr_label:
            G.edge(label, rcvr_label,
                   color=edge_color, penwidth=edge_width)

        # LearningProjection(s) to node
        # if proj in active_items or (proj_learning_in_execution_phase and proj_receiver in active_items):
        if proj in active_items:
            if active_color == BOLD:
                learning_proj_color = learning_color
            else:
                learning_proj_color = active_color
            learning_proj_width = str(default_width + active_thicker_by)
            self.active_item_rendered = True
        else:
            learning_proj_color = learning_color
            learning_proj_width = str(default_width)
        sndrs = proj._parameter_ports['matrix'].mod_afferents # GET ALL LearningProjections to proj
        for sndr in sndrs:
            sndr_label = _get_graph_node_label(self, sndr.sender.owner, show_types, show_dimensions)
            rcvr_label = _get_graph_node_label(self, proj, show_types, show_dimensions)
            if show_projection_labels:
                edge_label = proj._parameter_ports['matrix'].mod_afferents[0].name
            else:
                edge_label = ''
            if show_node_structure:
                G.edge(sndr_label + ':' + OutputPort.__name__ + '-' + 'LearningSignal',
                       rcvr_label,
                       label=edge_label,
                       color=learning_proj_color, penwidth=learning_proj_width)
            else:
                G.edge(sndr_label, rcvr_label, label = edge_label,
                       color=learning_proj_color, penwidth=learning_proj_width)
        return True

    @tc.typecheck
    def _assign_incoming_edges(g, rcvr, rcvr_label, senders, proj_color=None, proj_arrow=None, enclosing_g=None):

        proj_color = proj_color or default_node_color
        proj_arrow = default_projection_arrow

        # Deal with Projections from outer (enclosing_g) and inner (nested) Compositions
        # If not showing CIMs, then set up to find node for sender in inner or outer Composition
        if not show_cim:
            # Get sender node from inner Composition
            if show_nested is NESTED:
                # Add output_CIMs for nested Comps to find sender nodes
                cims = set([proj.sender.owner for proj in rcvr.afferents
                            if (isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                                and (proj.sender.owner is proj.sender.owner.composition.output_CIM))])
                senders.update(cims)
            # Get sender Node from outer Composition (enclosing_g)
            if enclosing_g and show_nested is not INSET:
                # Add input_CIM for current Composition to find senders from enclosing_g
                cims = set([proj.sender.owner for proj in rcvr.afferents
                            if (isinstance(proj.sender.owner, CompositionInterfaceMechanism)
                                and proj.sender.owner in {self.input_CIM, self.parameter_CIM})])
                senders.update(cims)

        for sender in senders:

            # Remove any Compositions from sndrs if show_cim is False and show_nested is True
            #    (since in that case the nodes for Compositions are bypassed)
            if not show_cim and show_nested is NESTED and isinstance(sender, Composition):
                continue

            # Iterate through all Projections from all OutputPorts of sender
            for output_port in sender.output_ports:
                for proj in output_port.efferents:

                    # Skip Projections not in the Composition
                    if proj not in self.projections:
                        continue

                    assign_proj_to_enclosing_comp = False

                    # Skip if sender is Composition and Projections to and from cim are being shown
                    #    (show_cim and show_nested) -- handled by _assign_cim_components
                    if isinstance(sender, Composition) and show_cim and show_nested is NESTED:
                        continue

                    if isinstance(sender, CompositionInterfaceMechanism):
                        if sender in {self.input_CIM, self.parameter_CIM}:
                            # FIX 6/2/20:
                            #     DELETE ONCE FILTERED BASED ON nesting_level IS IMPLEMENTED BEFORE CALL TO METHOD
                            # If cim has no afferents, presumably it is for the outermost Composition,
                            #     and therefore is not passing an afferent Projection from that Composition
                            if not sender.afferents:
                                continue
                            # Insure relevant InputPort of cim has only one afferent
                            assert len(sender.port_map[proj.receiver][0].path_afferents)==1,\
                                f"PROGRAM ERROR: {sender} of {self.name} has more than one afferent Projection."
                            sndr = sender.port_map[proj.receiver][0].path_afferents[0].sender.owner
                            # Skip:
                            # - cims as sources (handled in _assign_cim_compmoents)
                            # - controller (handled in _assign_controller_components)
                            if (isinstance(sndr, CompositionInterfaceMechanism)
                                    or (isinstance(sndr, ControlMechanism) and sndr.composition)):
                                continue
                            assign_proj_to_enclosing_comp = True

                        # sender is output_CIM
                        else:
                            # FIX 6/2/20:
                            #     DELETE ONCE FILTERED BASED ON nesting_level IS IMPLEMENTED BEFORE CALL TO METHOD
                            if not sender.efferents:
                                continue
                            # Insure cim has only one afferent
                            assert len([k.owner for k,v in sender.port_map.items() if v[1] is proj.sender])==1, \
                                f"PROGRAM ERROR: {sender} of {self.name} has more than one efferent Projection."
                            # Get Node from nested Composition that projects to rcvr
                            sndr = [k.owner for k,v in sender.port_map.items() if v[1] is proj.sender][0]
                            # Skip:
                            # - cims as sources (handled in _assign_cim_compmoents)
                            # - controller (handled in _assign_controller_components)
                            if (isinstance(sndr, CompositionInterfaceMechanism)
                                    or (isinstance(sndr, ControlMechanism) and sndr.composition)):
                                continue
                    else:
                        sndr = sender

                    # Set sndr info
                    sndr_label = _get_graph_node_label(self, sndr, show_types, show_dimensions)


                    # Skip any projections to ObjectiveMechanism for controller
                    #   (those are handled in _assign_controller_components)
                    # FIX 6/1/20 MOVE TO BELOW FOLLOWING IF STATEMENT AND REPLACE proj.receiver.owner WITH rcvr?
                    if (self.controller and
                            proj.receiver.owner in {self.controller, self.controller.objective_mechanism}):
                        continue

                    # FIX 6/6/20: ADD HANDLING OF parameter_CIM HERE??
                    # Only consider Projections to the rcvr (or its CIM if rcvr is a Composition)
                    if ((isinstance(rcvr, (Mechanism, Projection)) and proj.receiver.owner == rcvr)
                            or (isinstance(rcvr, Composition)
                                and proj.receiver.owner in {rcvr.input_CIM,
                                                            # MODIFIED 6/6/20 NEW:
                                                            rcvr.parameter_CIM
                                                            # MODIFIED 6/6/20 END
                                                            })):
                        if show_node_structure and isinstance(sndr, Mechanism):
                            sndr_port = proj.sender
                            sndr_port_owner = sndr_port.owner
                            if isinstance(sndr_port_owner, CompositionInterfaceMechanism):
                                # Sender is input_CIM or parameter_CIM
                                if sndr_port_owner in {sndr_port_owner.composition.input_CIM,
                                                       # MODIFIED 6/6/20 NEW:
                                                       sndr_port_owner.composition.parameter_CIM
                                                       # MODIFIED 6/6/20 END
                                                       }:
                                    # Get port for node of outer Composition that projects to it
                                    sndr_port = [v[0] for k,v in sender.port_map.items()
                                                 if k is proj.receiver][0].path_afferents[0].sender
                                # Sender is output_CIM
                                else:
                                    # Get port for node of inner Composition that projects to it
                                    sndr_port = [k for k,v in sender.port_map.items() if v[1] is proj.sender][0]
                            else:
                                sndr_port = proj.sender
                            sndr_proj_label = f'{sndr_label}:{sndr._get_port_name(sndr_port)}'
                        else:
                            sndr_proj_label = sndr_label
                        if show_node_structure and isinstance(rcvr, Mechanism):
                            proc_mech_rcvr_label = f'{rcvr_label}:{rcvr._get_port_name(proj.receiver)}'
                        else:
                            proc_mech_rcvr_label = rcvr_label

                        try:
                            has_learning = proj.has_learning_projection is not None
                        except AttributeError:
                            has_learning = None

                        edge_label = _get_graph_node_label(self, proj, show_types, show_dimensions)
                        is_learning_component = rcvr in self.learning_components or sndr in self.learning_components

                        # Check if Projection or its receiver is active
                        if any(item in active_items for item in {proj, proj.receiver.owner}):
                            if active_color == BOLD:
                                # if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                                if is_learning_component:
                                    proj_color = learning_color
                                else:
                                    pass
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True

                        # Projection to or from a LearningMechanism
                        elif (NodeRole.LEARNING in self.nodes_to_roles[rcvr]):
                            proj_color = learning_color
                            proj_width = str(default_width)

                        else:
                            proj_width = str(default_width)
                        proc_mech_label = edge_label

                        # RENDER PROJECTION AS EDGE

                        if show_learning and has_learning:
                            # Render Projection as Node
                            #    (do it here rather than in _assign_learning_components,
                            #     as it needs afferent and efferent edges to other nodes)
                            # IMPLEMENTATION NOTE: Projections can't yet use structured nodes:
                            deferred = not _render_projection_as_node(g=g, proj=proj,
                                                                      label=proc_mech_label,
                                                                      rcvr_label=proc_mech_rcvr_label,
                                                                      sndr_label=sndr_proj_label,
                                                                      proj_color=proj_color,
                                                                      proj_width=proj_width)
                            # Deferred if it is the last Mechanism in a learning Pathway
                            # (see _render_projection_as_node)
                            if deferred:
                                continue

                        else:
                            # Render Projection as edge
                            from psyneulink.core.components.projections.modulatory.controlprojection \
                                import ControlProjection
                            if isinstance(proj, ControlProjection):
                                arrowhead=control_projection_arrow
                            else:
                                arrowhead=proj_arrow
                            if show_projection_labels:
                                label = proc_mech_label
                            else:
                                label = ''

                        if assign_proj_to_enclosing_comp:
                            graph = enclosing_g
                        else:
                            graph = g
                        graph.edge(sndr_proj_label, proc_mech_rcvr_label,
                                   label=label,
                                   color=proj_color,
                                   penwidth=proj_width,
                                   arrowhead=arrowhead)

    def _generate_output(G):
        # Sort nodes for display
        # FIX 5/28/20:  ADD HANDLING OF NEST COMP:  SEARCH FOR 'subgraph cluster_'
        def get_index_of_node_in_G_body(node, node_type:tc.enum(MECHANISM, PROJECTION, BOTH)):
            """Get index of node in G.body"""
            for i, item in enumerate(G.body):
                if node.name + ' ' in item:  # Space needed to filter out node.name that is a substring of another name
                    if node_type in {MECHANISM, BOTH}:
                        if not '->' in item:
                            return i
                    elif node_type in {PROJECTION, BOTH}:
                        if '->' in item:
                            return i
                    else:
                        assert False, f'PROGRAM ERROR: node_type not specified or illegal ({node_type})'

        for node in self.nodes:
            if isinstance(node, Composition):
                continue
            roles = self.get_roles_by_node(node)
            # Put INPUT node(s) first
            if NodeRole.INPUT in roles:
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(0,G.body.pop(i))
            # Put OUTPUT node(s) last (except for ControlMechanisms)
            if NodeRole.OUTPUT in roles:
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))
            # Put ControlMechanism(s) last
            if isinstance(node, ControlMechanism):
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))

        for proj in self.projections:
            # Put ControlProjection(s) last (along with ControlMechanis(s))
            if isinstance(proj, ControlProjection):
                i = get_index_of_node_in_G_body(node, PROJECTION)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))

        if self.controller and show_controller:
            i = get_index_of_node_in_G_body(self.controller, MECHANISM)
            G.body.insert(len(G.body),G.body.pop(i))

        # GENERATE OUTPUT ---------------------------------------------------------------------

        # Show as pdf
        try:
            if output_fmt == 'pdf':
                # G.format = 'svg'
                G.view(self.name.replace(" ", "-"), cleanup=True, directory='show_graph OUTPUT/PDFS')

            # Generate images for animation
            elif output_fmt == 'gif':
                if self.active_item_rendered or INITIAL_FRAME in active_items:
                    self._generate_gifs(G, active_items, context)

            # Return graph to show in jupyter
            elif output_fmt == 'jupyter':
                return G

            elif output_fmt == 'gv':
                return G

            elif output_fmt == 'source':
                return G.source

            elif not output_fmt:
                return None

            else:
                raise CompositionError(f"Bad arg in call to {self.name}.show_graph: '{output_fmt}'.")

        except CompositionError as e:
            raise CompositionError(str(e.error_value))

        except:
            raise CompositionError(f"Problem displaying graph for {self.name}")

    # SETUP AND CONSTANTS -----------------------------------------------------------------

    INITIAL_FRAME = "INITIAL_FRAME"

    if context.execution_id is NotImplemented:
        context.execution_id = self.default_execution_id

    enclosing_g = kwargs.pop(ENCLOSING_G,None)
    nesting_level = kwargs.pop(NESTING_LEVEL,None)
    num_nesting_levels= kwargs.pop(NUM_NESTING_LEVELS,None)
    if kwargs:
        raise CompositionError(f'Unrecognized argument(s) in call to show_graph method '
                               f'of {Composition.__name__} {repr(self.name)}: {", ".join(kwargs.keys())}')

    # Get show_nested based on arg and current_nesting_level
    if enclosing_g is None:
        nesting_level = 0
        # FIX 6/7/20: MOVE ASSIGNMENT OF num_nesting_levels to here
        #             IF show_nestes is not False and is not an int, then set to float("inf")
        # show_nested arg specified number of nested levels to show, so set current show_nested value based on that
        if type(show_nested) is int:
            num_nesting_levels = show_nested
        elif show_nested is False:
            num_nesting_levels = 0
        elif show_nested is NESTED:
            num_nesting_levels = float("inf")
    if num_nesting_levels is not None:
        if nesting_level < num_nesting_levels:
            show_nested = NESTED
        else:
            show_nested = False
    # Otherwise, use set show_nested as NESTED unless it was specified as INSET
    elif show_nested and show_nested != INSET:
        show_nested = NESTED

    if show_dimensions == True:
        show_dimensions = ALL

    active_items = active_items or []
    if active_items:
        active_items = convert_to_list(active_items)
        if (self.scheduler.get_clock(context).time.run >= self._animate_num_runs or
                self.scheduler.get_clock(context).time.trial >= self._animate_num_trials):
            return

        for item in active_items:
            if not isinstance(item, Component) and item is not INITIAL_FRAME:
                raise CompositionError(
                    "PROGRAM ERROR: Item ({}) specified in {} argument for {} method of {} is not a {}".
                    format(item, repr('active_items'), repr('show_graph'), self.name, Component.__name__))

    self.active_item_rendered = False

    # Argument values used to call Mechanism._show_structure()
    if isinstance(show_node_structure, (list, tuple, set)):
        node_struct_args = {'composition': self,
                            'show_roles': any(key in show_node_structure for key in {ROLES, ALL}),
                            'show_conditions': any(key in show_node_structure for key in {CONDITIONS, ALL}),
                            'show_functions': any(key in show_node_structure for key in {FUNCTIONS, ALL}),
                            'show_mech_function_params': any(key in show_node_structure
                                                             for key in {MECH_FUNCTION_PARAMS, ALL}),
                            'show_port_function_params': any(key in show_node_structure
                                                              for key in {PORT_FUNCTION_PARAMS, ALL}),
                            'show_values': any(key in show_node_structure for key in {VALUES, ALL}),
                            'use_labels': any(key in show_node_structure for key in {LABELS, ALL}),
                            'show_headers': show_headers,
                            'output_fmt': 'struct',
                            'context':context}
    else:
        node_struct_args = {'composition': self,
                            'show_roles': show_node_structure in {ROLES, ALL},
                            'show_conditions': show_node_structure in {CONDITIONS, ALL},
                            'show_functions': show_node_structure in {FUNCTIONS, ALL},
                            'show_mech_function_params': show_node_structure in {MECH_FUNCTION_PARAMS, ALL},
                            'show_port_function_params': show_node_structure in {PORT_FUNCTION_PARAMS, ALL},
                            'show_values': show_node_structure in {VALUES, LABELS, ALL},
                            'use_labels': show_node_structure in {LABELS, ALL},
                            'show_headers': show_headers,
                            'output_fmt': 'struct',
                            'context': context}

    # DEFAULT ATTRIBUTES ----------------------------------------------------------------

    default_node_color = 'black'
    mechanism_shape = 'oval'
    learning_projection_shape = 'diamond'
    struct_shape = 'plaintext' # assumes use of html
    cim_shape = 'rectangle'
    composition_shape = 'rectangle'
    agent_rep_shape = 'egg'
    default_projection_arrow = 'normal'

    bold_width = 3
    default_width = 1
    active_thicker_by = 2

    input_rank = 'source'
    control_rank = 'min'
    learning_rank = 'min'
    output_rank = 'max'

    # BUILD GRAPH ------------------------------------------------------------------------

    import graphviz as gv

    G = gv.Digraph(
        name=self.name,
        engine="dot",
        node_attr={
            'fontsize': '12',
            'fontname': 'arial',
            'shape': 'record',
            'color': default_node_color,
            'penwidth': str(default_width),
        },
        edge_attr={
            'fontsize': '10',
            'fontname': 'arial'
        },
        graph_attr={
            "rankdir": direction,
            'overlap': "False"
        },
    )

    # get all Nodes
    # FIX: call to _analyze_graph in nested calls to show_graph cause trouble
    if output_fmt != 'gv':
        self._analyze_graph(context=context)
    processing_graph = self.graph_processing.dependency_dict
    rcvrs = list(processing_graph.keys())

    for rcvr in rcvrs:
        _assign_processing_components(G, rcvr, show_nested, show_nested_args, enclosing_g, nesting_level)

    # Add cim Components to graph if show_cim
    if show_cim:
        _assign_cim_components(G, [self.input_CIM, self.parameter_CIM, self.output_CIM], enclosing_g)

    # Add controller-related Components to graph if show_controller
    if show_controller:
        _assign_controller_components(G)

    # Add learning-related Components to graph if show_learning
    if show_learning:
        _assign_learning_components(G)

    # FIX 5/28/20:  RELEGATE REMAINDER OF show_graph TO THIS METHOD:
    return _generate_output(G)

def _get_graph_node_label(composition, item, show_types=None, show_dimensions=None):

    if not isinstance(item, (Mechanism, Composition, Projection)):
        raise CompositionError("Unrecognized node type ({}) in graph for {}".format(item, composition.name))

    name = item.name

    if show_types:
        name = item.name + '\n(' + item.__class__.__name__ + ')'

    if show_dimensions in {ALL, MECHANISMS} and isinstance(item, Mechanism):
        input_str = "in ({})".format(",".join(str(input_port.socket_width)
                                              for input_port in item.input_ports))
        output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_port.value)))
                                                for output_port in item.output_ports))
        return f"{output_str}\n{name}\n{input_str}"
    if show_dimensions in {ALL, PROJECTIONS} and isinstance(item, Projection):
        # MappingProjections use matrix
        if isinstance(item, MappingProjection):
            value = np.array(item.matrix)
            dim_string = "({})".format("x".join([str(i) for i in value.shape]))
            return "{}\n{}".format(item.name, dim_string)
        # ModulatoryProjections use value
        else:
            value = np.array(item.value)
            dim_string = "({})".format(len(value))
            return "{}\n{}".format(item.name, dim_string)

    if isinstance(item, CompositionInterfaceMechanism):
        name = name.replace('Input_CIM','INPUT')
        name = name.replace('Parameter_CIM', 'CONTROL')
        name = name.replace('Output_CIM', 'OUTPUT')

    return name

def _set_up_animation(self, context):

    self._component_animation_execution_count = None

    if isinstance(self._animate, dict):
        # Assign directory for animation files
        from psyneulink._version import root_dir
        default_dir = root_dir + '/../show_graph output/GIFs/' + self.name # + " gifs"
        # try:
        #     rmtree(self._animate_directory)
        # except:
        #     pass
        self._animate_unit = self._animate.pop(UNIT, EXECUTION_SET)
        self._image_duration = self._animate.pop(DURATION, 0.75)
        self._animate_num_runs = self._animate.pop(NUM_RUNS, 1)
        self._animate_num_trials = self._animate.pop(NUM_TRIALS, 1)
        self._animate_simulations = self._animate.pop(SIMULATIONS, False)
        self._movie_filename = self._animate.pop(MOVIE_NAME, self.name + ' movie') + '.gif'
        self._animation_directory = self._animate.pop(MOVIE_DIR, default_dir)
        self._save_images = self._animate.pop(SAVE_IMAGES, False)
        self._show_animation = self._animate.pop(SHOW, False)
        if not self._animate_unit in {COMPONENT, EXECUTION_SET}:
            raise CompositionError(f"{repr(UNIT)} entry of {repr('animate')} argument for {self.name} method "
                                   f"of {repr('run')} ({self._animate_unit}) "
                                   f"must be {repr(COMPONENT)} or {repr(EXECUTION_SET)}.")
        if not isinstance(self._image_duration, (int, float)):
            raise CompositionError(f"{repr(DURATION)} entry of {repr('animate')} argument for {repr('run')} method "
                                   f"of {self.name} ({self._image_duration}) must be an int or a float.")
        if not isinstance(self._animate_num_runs, int):
            raise CompositionError(f"{repr(NUM_RUNS)} entry of {repr('animate')} argument for {repr('show_graph')} "
                                   f"method of {self.name} ({self._animate_num_runs}) must an integer.")
        if not isinstance(self._animate_num_trials, int):
            raise CompositionError(f"{repr(NUM_TRIALS)} entry of {repr('animate')} argument for "
                                   f"{repr('show_graph')} method of {self.name} ({self._animate_num_trials}) "
                                   f"must an integer.")
        if not isinstance(self._animate_simulations, bool):
            raise CompositionError(f"{repr(SIMULATIONS)} entry of {repr('animate')} argument for "
                                   f"{repr('show_graph')} method of {self.name} ({self._animate_num_trials}) "
                                   f"must a boolean.")
        if not isinstance(self._animation_directory, str):
            raise CompositionError(f"{repr(MOVIE_DIR)} entry of {repr('animate')} argument for {repr('run')} "
                                   f"method of {self.name} ({self._animation_directory}) must be a string.")
        if not isinstance(self._movie_filename, str):
            raise CompositionError(f"{repr(MOVIE_NAME)} entry of {repr('animate')} argument for {repr('run')} "
                                   f"method of {self.name} ({self._movie_filename}) must be a string.")
        if not isinstance(self._save_images, bool):
            raise CompositionError(f"{repr(SAVE_IMAGES)} entry of {repr('animate')} argument for {repr('run')}"
                                   f"method of {self.name} ({self._save_images}) must be a boolean")
        if not isinstance(self._show_animation, bool):
            raise CompositionError(f"{repr(SHOW)} entry of {repr('animate')} argument for {repr('run')} "
                                   f"method of {self.name} ({self._show_animation}) must be a boolean.")
    elif self._animate:
        # self._animate should now be False or a dict
        raise CompositionError("{} argument for {} method of {} ({}) must be a boolean or "
                               "a dictionary of argument specifications for its {} method".
                               format(repr('animate'), repr('run'), self.name, self._animate, repr('show_graph')))

def _animate_execution(self, active_items, context):
    if self._component_animation_execution_count is None:
        self._component_animation_execution_count = 0
    else:
        self._component_animation_execution_count += 1
    self.show_graph(active_items=active_items,
                    **self._animate,
                    output_fmt='gif',
                    context=context,
                    )

def _generate_gifs(self, G, active_items, context):

    def create_phase_string(phase):
        return f'%16s' % phase + ' - '

    def create_time_string(time, spec):
        if spec == 'TIME':
            r = time.run
            t = time.trial
            p = time.pass_
            ts = time.time_step
        else:
            r = t = p = ts = '__'
        return f"Time(run: %2s, " % r + f"trial: %2s, " % t + f"pass: %2s, " % p + f"time_step: %2s)" % ts

    G.format = 'gif'
    execution_phase = context.execution_phase
    time = self.scheduler.get_clock(context).time
    run_num = time.run
    trial_num = time.trial

    if INITIAL_FRAME in active_items:
        phase_string = create_phase_string('Initializing')
        time_string = create_time_string(time, 'BLANKS')

    elif ContextFlags.PROCESSING in execution_phase:
        phase_string = create_phase_string('Processing Phase')
        time_string = create_time_string(time, 'TIME')
    # elif ContextFlags.LEARNING in execution_phase:
    #     time = self.scheduler_learning.get_clock(context).time
    #     time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}". \
    #         format(run_num, time.trial, time.pass_, time.time_step)
    #     phase_string = 'Learning Phase - '

    elif ContextFlags.CONTROL in execution_phase:
        phase_string = create_phase_string('Control Phase')
        time_string = create_time_string(time, 'TIME')

    else:
        raise CompositionError(
            f"PROGRAM ERROR:  Unrecognized phase during execution of {self.name}: {execution_phase.name}")

    label = f'\n{self.name}\n{phase_string}{time_string}\n'
    G.attr(label=label)
    G.attr(labelloc='b')
    G.attr(fontname='Monaco')
    G.attr(fontsize='14')
    index = repr(self._component_animation_execution_count)
    image_filename = '-'.join([repr(run_num), repr(trial_num), index])
    image_file = self._animation_directory + '/' + image_filename + '.gif'
    G.render(filename=image_filename,
             directory=self._animation_directory,
             cleanup=True,
             # view=True
             )
    # Append gif to self._animation
    image = Image.open(image_file)
    # TBI?
    # if not self._save_images:
    #     remove(image_file)
    if not hasattr(self, '_animation'):
        self._animation = [image]
    else:
        self._animation.append(image)
