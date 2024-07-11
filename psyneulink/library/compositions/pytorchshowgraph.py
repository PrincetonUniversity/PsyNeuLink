# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************** PyTorch show_graph *********************************************************

from beartype import beartype

from psyneulink._typing import Optional, Union, Literal

from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.compositions import NodeRole
from psyneulink.core.compositions.showgraph import ShowGraph, SHOW_JUST_LEARNING_PROJECTIONS
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.projections.projection import Projection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.globals.keywords import BOLD, NESTED, INSET

__all__ = ['SHOW_PYTORCH']

SHOW_PYTORCH = 'show_pytorch'
EXCLUDE_FROM_GRADIENT_CALC_LINE_STYLE = 'exclude_from_gradient_calc_line_style'
EXCLUDE_FROM_GRADIENT_CALC_COLOR = 'exclude_from_gradient_calc_color'

class PytorchShowGraph(ShowGraph):
    """ShowGraph object with `show_graph <ShowGraph.show_graph>` method for displaying `Composition`.

    This is a subclass of the `ShowGraph` class that is used to display the graph of a `Composition` used for learning
    in `PyTorch mode <Composition_Learning_AutodiffComposition>` (also see `AutodiffComposition_PyTorch`).  In this mode,
    any `nested Compositions <AutodiffComposition_Nesting>` are "flattened" (i.e., incorporated into the outermost
    Composition); also, any `Nodes <Composition_Nodes>`` designated as `exclude_from_gradient_calc
    <PytorchMechanismWrapper.exclude_from_gradient_calc>` will be moved to the end of the graph (as they are executed
    after the gradient calculation), and any Projections designated as `exclude_from_autodiff
    <Projection.exclude_from_autodiff>` will not be shown as they are not used in the gradient calculations at all.

    Arguments
    ---------

    show_pytorch : keyword : default 'PYTORCH'
        specifies that the PyTorch version of the graph should be shown.

    """

    def __init__(self, *args, **kwargs):
        self.show_pytorch = kwargs.pop(SHOW_PYTORCH, False)
        super().__init__(*args, **kwargs)

    @beartype
    @handle_external_context(source=ContextFlags.COMPOSITION)
    def show_graph(self, *args, **kwargs):
        """Override of show_graph to check if show_pytorch==True and if so build pytorch rep of autofiffcomposition"""
        self.show_pytorch = kwargs.pop(SHOW_PYTORCH, self.show_pytorch)
        context = kwargs.get('context')
        if self.show_pytorch:
            self.pytorch_rep = self.composition._build_pytorch_representation(context)
        self.exclude_from_gradient_calc_line_style = kwargs.pop(EXCLUDE_FROM_GRADIENT_CALC_LINE_STYLE, 'dotted')
        self.exclude_from_gradient_calc_color = kwargs.pop(EXCLUDE_FROM_GRADIENT_CALC_COLOR, 'brown')
        return super().show_graph(*args, **kwargs)

    def _get_processing_graph(self, composition, context):
        """Helper method that creates dependencies graph for nodes of autodiffcomposition used in Pytorch mode"""
        if self.show_pytorch:
            processing_graph = {}
            projections = self._get_projections(composition, context)
            # 7/9/24 FIX: COULD DO THIS BY ITERATING OVER PROJECTIONS INSTEAD OF NODES
            for node in self._get_nodes(composition, context):
                dependencies = set()
                for projection in projections:
                    if node is projection.receiver.owner:
                        dependencies.add(projection.sender.owner)
                    # Add dependency of INPUT node of nested graph on node in outer graph that projects to it
                    elif (isinstance(projection.receiver.owner, CompositionInterfaceMechanism) and
                          projection.receiver.owner._get_destination_info_from_input_CIM(projection.receiver)[1]
                          is node):
                        dependencies.add(projection.sender.owner)
                processing_graph[node] = dependencies
            # Add TARGET nodes
            for node in self.composition.learning_components:
                processing_graph[node] = set([afferent.sender.owner for afferent in node.path_afferents])
            return processing_graph
        else:
            return super()._get_processing_graph(composition, context)

    def _get_nodes(self, composition, context):
        """Override to return nodes of PytorchCompositionWrapper rather than autodiffcomposition"""
        if self.show_pytorch:
            nodes = list(self.pytorch_rep.nodes_map.keys())
            return nodes
        else:
            return super()._get_nodes(composition, context)

    def _get_projections(self, composition, context):
        """Override to return nodes of Pytorch graph"""
        if self.show_pytorch:
            projections = list(self.pytorch_rep.projections_map.keys())
            # FIX: NEED TO ADD PROJECTIONS TO NESTED COMPS THAT ARE TO CIM
            # Add any Projections to TARGET nodes
            projections += [afferent
                            for node in self.composition.learning_components
                            for afferent in node.path_afferents
                            if not isinstance(afferent.sender.owner, CompositionInterfaceMechanism)]
            return projections
        else:
            return super()._get_projections(composition, context)

    def _proj_in_composition(self, proj, composition_projections, context)->bool:
        """Override to include direct Projections from outer to nested comps in Pytorch mode"""
        processing_graph = self._get_processing_graph(self.composition, context)
        if proj in composition_projections:
            return True
        # If proj is betw. a sender and receiver specified in the processing_graphl, then it is in the autodiffcomp
        elif (proj.receiver.owner in processing_graph
              and proj.sender.owner in processing_graph[proj.receiver.owner]):
            return True
        else:
            return False

    def _get_roles_by_node(self, composition, node, context):
        """Override in Pytorch mode to return NodeRole.INTERNAL for all nodes in nested compositions"""
        if self.show_pytorch:
            try:
                return composition.get_roles_by_node(node)
            except:
                return [NodeRole.INTERNAL]
        if self.show_pytorch and node not in self.composition.nodes:
                return [NodeRole.INTERNAL]
        else:
            return super()._get_roles_by_node(composition, node, context)

    def _get_nodes_by_role(self, composition, role, context):
        """Override in Pytorch mode to return all nodes in nested compositions as INTERNAL"""
        if self.show_pytorch and composition is not self.composition:
            return None
        else:
            return super()._get_nodes_by_role(composition, role, context)

    def _implement_graph_node(self, g, rcvr, context, *args, **kwargs):
        """Override to assign EXCLUDE_FROM_GRADIENT_CALC nodes their own style in Pytorch mode"""
        if self.pytorch_rep.nodes_map[rcvr].exclude_from_gradient_calc:
            kwargs['style'] = self.exclude_from_gradient_calc_line_style
            kwargs['color'] = self.exclude_from_gradient_calc_color
        g.node(*args, **kwargs)

    def _implement_graph_edge(self, graph, proj, context, *args, **kwargs):
        """Override to assign custom attributes to edges"""

        kwargs['color'] = self.default_node_color

        modulatory_node = None
        if proj.parameter_ports[0].mod_afferents:
            modulatory_node = self.pytorch_rep.nodes_map[proj.parameter_ports[0].mod_afferents[0].sender.owner]

        if proj in self.pytorch_rep.projections_map:

            # If Projection is a LearningProjection that is active, assign color and arrowhead of a LearningProjection
            if self.pytorch_rep.projections_map[proj].matrix.requires_grad:
                kwargs['color'] = self.learning_color

            # If Projection is from a ModulatoryMechanism that is excluded from gradient calculations, assign that style
            elif modulatory_node and modulatory_node.exclude_from_gradient_calc:
                kwargs['color'] = self.exclude_from_gradient_calc_color
                kwargs['style'] = self.exclude_from_gradient_calc_line_style

        graph.edge(*args, **kwargs)

    # def _assign_incoming_edges(self,
    #                            g,
    #                            rcvr,
    #                            rcvr_label,
    #                            senders,
    #                            active_items,
    #                            show_nested,
    #                            show_cim,
    #                            show_learning,
    #                            show_types,
    #                            show_dimensions,
    #                            show_node_structure,
    #                            show_projection_labels,
    #                            show_projections_not_in_composition,
    #                            proj_color=None,
    #                            proj_arrow=None,
    #                            enclosing_comp=None,
    #                            comp_hierarchy=None,
    #                            nesting_level=None,
    #                            context=None):
    #
    #     from psyneulink.core.compositions.composition import Composition, NodeRole
    #     composition = self.composition
    #     composition_projections = self._get_projections(composition, context)
    #     if nesting_level not in comp_hierarchy:
    #         comp_hierarchy[nesting_level] = composition
    #     enclosing_g = enclosing_comp._show_graph.G if enclosing_comp else None
    #
    #     proj_color_default = proj_color or self.default_node_color
    #     proj_arrow_default = proj_arrow or self.default_projection_arrow
    #
    #     # Deal with Projections from outer (enclosing_g) and inner (nested) Compositions
    #     # If not showing CIMs, then set up to find node for sender in inner or outer Composition
    #     if not show_cim:
    #         # Get sender node from inner Composition
    #         if show_nested is NESTED:
    #             # Add output_CIMs for nested Comps to find sender nodes
    #             cims = set([proj.sender.owner for proj in rcvr.afferents
    #                         if (proj in composition_projections
    #                             and isinstance(proj.sender.owner, CompositionInterfaceMechanism)
    #                             and (proj.sender.owner is proj.sender.owner.composition.output_CIM))])
    #             senders.update(cims)
    #         # Get sender Node from outer Composition (enclosing_g)
    #         if enclosing_g and show_nested is not INSET:
    #             # Add input_CIM for current Composition to find senders from enclosing_g
    #             cims = set([proj.sender.owner for proj in rcvr.afferents
    #                         if (proj in composition_projections
    #                             and isinstance(proj.sender.owner, CompositionInterfaceMechanism)
    #                             and proj.sender.owner in {composition.input_CIM, composition.parameter_CIM})])
    #             senders.update(cims)
    #         # HACK: FIX 6/13/20 - ADD USER-SPECIFIED TARGET NODE FOR INNER COMPOSITION (NOT IN processing_graph)
    #
    #     def assign_sender_edge(sndr:Union[Mechanism, Composition],
    #                            proj:Projection,
    #                            proj_color:str,
    #                            proj_arrowhead:str
    #                            ) -> None:
    #         """Assign edge from sender to rcvr"""
    #
    #         # Set sndr info
    #         sndr_label = self._get_graph_node_label(composition, sndr, show_types, show_dimensions)
    #
    #         # Skip any projections to ObjectiveMechanism for controller
    #         #   (those are handled in _assign_controller_components)
    #         if (composition.controller and
    #                 proj.receiver.owner in {composition.controller.objective_mechanism}):
    #             return
    #
    #         # Only consider Projections to the rcvr (or its CIM if rcvr is a Composition)
    #         if ((isinstance(rcvr, (Mechanism, Projection)) and proj.receiver.owner == rcvr)
    #                 or (isinstance(rcvr, Composition)
    #                     and proj.receiver.owner in {rcvr.input_CIM,
    #                                                 rcvr.parameter_CIM})):
    #             if show_node_structure and isinstance(sndr, Mechanism):
    #                 # If proj is a ControlProjection that comes from a parameter_CIM, get the port for the sender
    #                 if (isinstance(proj, ControlProjection) and
    #                         isinstance(proj.sender.owner, CompositionInterfaceMechanism)):
    #                     sndr_port = sndr.output_port
    #                 # Usual case: get port from Projection's sender
    #                 else:
    #                     sndr_port = proj.sender
    #                 sndr_port_owner = sndr_port.owner
    #                 if isinstance(sndr_port_owner, CompositionInterfaceMechanism) and rcvr is not composition.controller:
    #                     # Sender is input_CIM or parameter_CIM
    #                     if sndr_port_owner in {sndr_port_owner.composition.input_CIM,
    #                                            sndr_port_owner.composition.parameter_CIM}:
    #                         # Get port for node of outer Composition that projects to it
    #                         sndr_port = [v[0] for k,v in sender.port_map.items()
    #                                      if k is proj.receiver][0].path_afferents[0].sender
    #                     # Sender is output_CIM
    #                     else:
    #                         # Get port for node of inner Composition that projects to it
    #                         sndr_port = [k for k,v in sender.port_map.items() if v[1] is proj.sender][0]
    #                 sndr_proj_label = f'{sndr_label}:{sndr._get_port_name(sndr_port)}'
    #             else:
    #                 sndr_proj_label = sndr_label
    #             if show_node_structure and isinstance(rcvr, Mechanism):
    #                 proc_mech_rcvr_label = f'{rcvr_label}:{rcvr._get_port_name(proj.receiver)}'
    #             else:
    #                 proc_mech_rcvr_label = rcvr_label
    #
    #             try:
    #                 has_learning = proj.has_learning_projection is not None
    #             except AttributeError:
    #                 has_learning = None
    #
    #             edge_label = self._get_graph_node_label(composition, proj, show_types, show_dimensions)
    #             is_learning_component = (rcvr in composition.learning_components
    #                                      or sndr in composition.learning_components)
    #             if isinstance(sender, ControlMechanism):
    #                 proj_color = self.control_color
    #                 if (not isinstance(rcvr, Composition)
    #                         or (not show_cim and
    #                             (show_nested is not NESTED)
    #                             or (show_nested is False))):
    #                     # # Allow MappingProjections to iconified rep of nested Composition to show as ControlProjection
    #                     # if (isinstance(proj, ControlProjection)
    #                     #         or (isinstance(proj.receiver.owner, CompositionInterfaceMechanism)
    #                     #             and (not show_cim or not show_nested))):
    #                     #     proj_arrowhead = self.control_projection_arrow
    #                     # else:  # This is to expose an errant MappingProjection if one slips in
    #                     #     proj_arrowhead = self.default_projection_arrow
    #                     #     proj_color = self.default_node_color
    #                     proj_arrowhead = self.control_projection_arrow
    #             # Check if Projection or its receiver is active
    #             if any(item in active_items for item in {proj, proj.receiver.owner}):
    #                 if self.active_color == BOLD:
    #                     # if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
    #                     if is_learning_component:
    #                         proj_color = self.learning_color
    #                     else:
    #                         pass
    #                 else:
    #                     proj_color = self.active_color
    #                 proj_width = str(self.default_width + self.active_thicker_by)
    #                 composition.active_item_rendered = True
    #
    #             # Projection to or from a LearningMechanism
    #             elif (NodeRole.LEARNING in self._get_roles_by_node(composition, rcvr, context)):
    #                 proj_color = self.learning_color
    #                 proj_width = str(self.default_width)
    #
    #             else:
    #                 proj_width = str(self.default_width)
    #             proc_mech_label = edge_label
    #
    #             # RENDER PROJECTION AS EDGE
    #
    #             if show_learning and has_learning:
    #                 # Render Projection as Node
    #                 #    (do it here rather than in _assign_learning_components,
    #                 #     as it needs afferent and efferent edges to other nodes)
    #                 # IMPLEMENTATION NOTE: Projections can't yet use structured nodes:
    #                 deferred = not self._render_projection_as_node(g,
    #                                                                active_items,
    #                                                                show_node_structure,
    #                                                                show_learning,
    #                                                                show_types,
    #                                                                show_dimensions,
    #                                                                show_projection_labels,
    #                                                                show_projections_not_in_composition,
    #                                                                proj,
    #                                                                label=proc_mech_label,
    #                                                                rcvr_label=proc_mech_rcvr_label,
    #                                                                sndr_label=sndr_proj_label,
    #                                                                proj_color=proj_color,
    #                                                                proj_width=proj_width)
    #                 # Deferred if it is the last Mechanism in a learning Pathway
    #                 # (see _render_projection_as_node)
    #                 if deferred:
    #                     return
    #
    #             else:
    #                 # Render Projection as edge
    #                 if show_projection_labels:
    #                     label = proc_mech_label
    #                 else:
    #                     label = ''
    #
    #                 if assign_proj_to_enclosing_comp:
    #                     graph = enclosing_g
    #                 else:
    #                     graph = g
    #
    #                 self._implement_graph_edge(graph,
    #                                            proj,
    #                                            context,
    #                                            sndr_proj_label,
    #                                            proc_mech_rcvr_label,
    #                                            label=label,
    #                                            color=proj_color,
    #                                            penwidth=proj_width,
    #                                            arrowhead=proj_arrowhead)
    #
    #     # Sorted to insure consistency of ordering in g for testing
    #     for sender in sorted(senders):
    #
    #         # Remove any Compositions from sndrs if show_cim is False and show_nested is True
    #         #    (since in that case the nodes for Compositions are bypassed)
    #         if not show_cim and show_nested is NESTED and isinstance(sender, Composition):
    #             continue
    #
    #         # Iterate through all Projections from all OutputPorts of sender
    #         for output_port in sender.output_ports:
    #             for proj in output_port.efferents:
    #
    #                 proj_color = proj_color_default
    #                 proj_arrowhead = proj_arrow_default
    #
    #                 if proj not in composition_projections:
    #                     if not show_projections_not_in_composition:
    #                         continue
    #                     else:
    #                         proj_color=self.inactive_projection_color
    #
    #                 assign_proj_to_enclosing_comp = False
    #
    #                 # Skip if sender is Composition and Projections to and from cim are being shown
    #                 #    (show_cim and show_nested) -- handled by _assign_cim_components
    #                 if isinstance(sender, Composition) and show_cim and show_nested is NESTED:
    #                     continue
    #
    #                 if isinstance(sender, CompositionInterfaceMechanism):
    #
    #                     # sender is input_CIM or parameter_CIM
    #                     if sender in {composition.input_CIM, composition.parameter_CIM}:
    #                         # FIX 6/2/20:
    #                         #     DELETE ONCE FILTERED BASED ON nesting_level IS IMPLEMENTED BEFORE CALL TO METHOD
    #                         # If cim has no afferents, presumably it is for the outermost Composition,
    #                         #     and therefore is not passing an afferent Projection from that Composition
    #                         if not sender.afferents and rcvr is not composition.controller:
    #                             continue
    #                         # FIX: LOOP HERE OVER sndr_spec IF THERE ARE SEVERAL
    #                         # Get node(s) from enclosing Composition that is/are source(s) of sender(s)
    #                         sndrs_specs = self._trace_senders_for_original_sender_mechanism(proj, nesting_level)
    #                         if not sndrs_specs:
    #                             continue
    #                         for sndr_spec in sorted(sndrs_specs):
    #                             sndr, sndr_port, sndr_nesting_level = sndr_spec
    #                             # if original sender is more than one level above receiver, replace enclosing_g with
    #                             # the g of the original sender composition
    #                             enclosing_comp = comp_hierarchy[sndr_nesting_level]
    #                             enclosing_g = enclosing_comp._show_graph.G
    #                             # Skip:
    #                             # - cims as sources (handled in _assign_cim_components)
    #                             #    unless it is the input_CIM for the outermost Composition and show_cim is not true
    #                             # - controller (handled in _assign_controller_components)
    #                             if (isinstance(sndr, CompositionInterfaceMechanism) and
    #                                     rcvr is not enclosing_comp.controller
    #                                     and rcvr is not composition.controller
    #                                     and not sndr.afferents and show_cim
    #                                     or self._is_composition_controller(sndr, context, enclosing_comp)):
    #                                 continue
    #                             if sender is composition.parameter_CIM:
    #                                 # # Allow MappingProjections to iconified rep of nested Composition
    #                                 # # to show as ControlProjection
    #                                 # if (isinstance(proj, ControlProjection)
    #                                 #         or (isinstance(proj.receiver.owner, CompositionInterfaceMechanism)
    #                                 #             and (not show_cim or not show_nested))):
    #                                 #     proj_arrowhead = self.control_projection_arrow
    #                                 #     proj_color = self.control_color
    #                                 # else:   # This is to expose an errant MappingProjection if one slips in
    #                                 #     proj_arrowhead = self.default_projection_arrow
    #                                 #     proj_color = proj_color=self.default_node_color
    #                                 proj_color = self.control_color
    #                                 proj_arrowhead = self.control_projection_arrow
    #                             assign_proj_to_enclosing_comp = True
    #                             assign_sender_edge(sndr, proj, proj_color, proj_arrowhead)
    #                         continue
    #
    #                     # sender is output_CIM
    #                     # FIX: 4/5/21 IMPLEMENT LOOP HERE COMPARABLE TO ONE FOR input_CIM ABOVE
    #                     else:
    #                         # FIX 6/2/20:
    #                         #     DELETE ONCE FILTERED BASED ON nesting_level IS IMPLEMENTED BEFORE CALL TO METHOD
    #                         if not sender.efferents:
    #                             continue
    #                         # Insure cim has only one afferent
    #                         assert len([k.owner for k,v in sender.port_map.items() if v[1] is proj.sender])==1, \
    #                             f"PROGRAM ERROR: {sender} of {composition.name} has more than one efferent Projection."
    #                         # Get Node from nested Composition that projects to rcvr
    #                         sndr = [k.owner for k,v in sender.port_map.items() if v[1] is proj.sender][0]
    #                         # Skip:
    #                         # - cims as sources (handled in _assign_cim_components)
    #                         # - controller (handled in _assign_controller_components)
    #                         # NOTE 7/20/20: if receiver is a controller, then we need to skip this block or shadow inputs
    #                         # is not rendered -DS
    #                         if (rcvr is not composition.controller
    #                                 and isinstance(sndr, CompositionInterfaceMechanism)
    #                                 or (isinstance(sndr, ControlMechanism) and sndr.composition)):
    #                             continue
    #                 else:
    #                     sndr = sender
    #
    #                 assign_sender_edge(sndr, proj, proj_color, proj_arrowhead)
    #
