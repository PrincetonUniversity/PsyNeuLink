# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************** PyTorch show_graph *********************************************************

from beartype import beartype

from psyneulink._typing import Optional, Union, Literal

from psyneulink.core.compositions import Composition, NodeRole
from psyneulink.core.compositions.showgraph import ShowGraph, SHOW_JUST_LEARNING_PROJECTIONS, SHOW_LEARNING
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.library.components.projections.modulatory.lossprojection import LossProjection
from psyneulink.core.llvm import ExecutionMode
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import SAMPLE, SHOW_PYTORCH, TARGET, PNL

EXCLUDE_FROM_GRADIENT_CALC_LINE_STYLE = 'exclude_from_gradient_calc_line_style'
EXCLUDE_FROM_GRADIENT_CALC_COLOR = 'exclude_from_gradient_calc_color'

class PytorchShowGraph(ShowGraph):
    """ShowGraph object with `show_graph <ShowGraph.show_graph>` method for displaying `Composition`.

    This is a subclass of the `ShowGraph` class that is used to display the graph of a `Composition` used for learning
    in `PyTorch mode <Composition_Learning_AutodiffComposition>` (also see `AutodiffComposition_PyTorch`).  In this mode,
    any `nested Compositions <AutodiffComposition_Nesting>` are "flattened" (i.e., incorporated into the outermost
    Composition); also, any `Nodes <Composition_Nodes>`` designated as `exclude_from_gradient_calc
    <PytorchMechanismWrapper.exclude_from_gradient_calc>` are moved to the end of the graph (as they are executed
    after the gradient calculation), and any Projections designated as `exclude_in_autodiff
    <Projection.exclude_in_autodiff>` are not shown as they are not used in the gradient calculations at all.

    Arguments
    ---------

    show_pytorch : keyword : default 'PYTORCH'
        specifies that the PyTorch version of the graph should be shown.
    """

    def __init__(self, *args, **kwargs):
        self.show_pytorch = kwargs.pop('show_pytorch', False)
        super().__init__(*args, **kwargs)

    @beartype
    @handle_external_context(source=ContextFlags.COMPOSITION)
    def show_graph(self, *args, **kwargs):
        """Override of show_graph to check for autodiff-specific options
        If show_pytorch==True, build pytorch rep of autofiffcomposition
        If show_learning==PNL, infer backpropagation learning pathways for Python version of graph
        """
        if SHOW_LEARNING in kwargs and kwargs[SHOW_LEARNING] == PNL:
            self.composition.infer_backpropagation_learning_pathways(ExecutionMode.Python)
            kwargs[SHOW_LEARNING] = True
            return super().show_graph(*args, **kwargs)
        elif SHOW_LEARNING in kwargs:
            # Remove SHOW_LEARNING spec from kwargs to prevent double-handling in super().show_graph
            kwargs.pop(SHOW_LEARNING, None)
            if self.composition._warned_about_unecessary_show_learning_arg_in_call_to_show_graph is False:
                import warnings
                warnings.warn(f"'{SHOW_LEARNING}' argument in call to show_graph() for '{self.composition.name}' "
                              f"is unnecessary since learning components are shown when '{SHOW_PYTORCH}' is used.")
                self.composition._warned_about_unecessary_show_learning_arg_in_call_to_show_graph = True
        self.show_pytorch = kwargs.pop('show_pytorch', False)
        context = kwargs.get('context')
        if self.show_pytorch:
            self.composition.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
            self.pytorch_rep = (
                self.composition._build_pytorch_representation(
                    context=Context(source=ContextFlags.SHOW_GRAPH, execution_id=context.execution_id),
                    new=False))
        self.exclude_from_gradient_calc_line_style = kwargs.pop(EXCLUDE_FROM_GRADIENT_CALC_LINE_STYLE, 'dotted')
        self.exclude_from_gradient_calc_color = kwargs.pop(EXCLUDE_FROM_GRADIENT_CALC_COLOR, 'brown')
        return super().show_graph(*args, **kwargs)

    def _make_additional_assignments(self,
                                     g, processing_graph,
                                     composition, enclosing_comp, comp_hierarchy, nesting_level, active_items,
                                     show_nested, show_cim, show_learning, show_types, show_dimensions,
                                     show_node_structure, node_structure_args,
                                     show_projection_labels, show_projections_not_in_composition,
                                     context):
        """Override to add Loss components to graph
        Add LossMechanism to processing_graph, and implement LossProjection (from LossMechanism to SAMPLE)
        """
        if not self.show_pytorch:
            return super()._make_additional_assignments(
                g, processing_graph,
                composition, enclosing_comp, comp_hierarchy, nesting_level, active_items,
                show_nested, show_cim, show_learning, show_types, show_dimensions,
                show_node_structure, node_structure_args,
                show_projection_labels, show_projections_not_in_composition,
                context)

        # If a node projects to a LossMechanism as its SAMPLE, add LossMechanism as dependency
        #  so that a return exclude_from_gradient_calc arrow can added to show the dependencey for learning
        loss_mechs = [n for n in composition.nodes if isinstance(n, LossMechanism)]
        if loss_mechs:
            for node in [n for n in processing_graph if n not in composition.get_nodes_by_role(NodeRole.TARGET)]:
                for loss_mech in loss_mechs:
                    if node is loss_mech.sample.owner:
                        processing_graph[node].add(loss_mech)
                        self._implement_graph_edge(g,
                                                   loss_mech.loss_projection,
                                                   context,
                                                   loss_mech.name,
                                                   loss_mech.sample.owner.name,
                                                   color=self.exclude_from_gradient_calc_color,
                                                   penwidth=self.default_width,
                                                   style=self.exclude_from_gradient_calc_line_style)

    def _get_processing_graph(self, composition, context):
        """Helper method that creates dependencies graph for nodes of AutodiffComposition used in PyTorch mode
        IMPLEMENTATION NOTE:
            learning_components (LossMechanism(s) and TARGET nodes) are included
            since these are always part of the graph in PyTorch mode
        """
        if self.show_pytorch:
            processing_graph = {}
            projections = self._get_projections(composition, context)
            nodes = self._get_nodes(composition, context)
            for node in nodes:
                dependencies = set()
                for projection in projections:
                    sender = projection.sender.owner
                    receiver = projection.receiver.owner
                    if node is receiver:
                        dependencies.add(sender)
                    # Add dependency of INPUT node of nested graph on node in outer graph that projects to it
                    elif (isinstance(receiver, CompositionInterfaceMechanism) and
                          receiver._get_source_info_from_output_CIM(projection.receiver)[1] is node):
                        dependencies.add(sender)
                    else:
                        for proj in [proj for proj in node.afferents if proj.sender.owner in nodes]:
                            dependencies.add(proj.sender.owner)
                processing_graph[node] = dependencies
            return {k: processing_graph[k] for k in sorted(processing_graph.keys())}

        else:
            return super()._get_processing_graph(composition, context)

    def _get_nodes(self, composition, context):
        """Override to return nodes of PytorchCompositionWrapper rather than autodiffcomposition"""
        if self.show_pytorch:
            nodes = sorted([node for node in self.pytorch_rep.nodes_map
                            if SHOW_PYTORCH in self.pytorch_rep.nodes_map[node]._use])
            return nodes
        else:
            return super()._get_nodes(composition, context)

    def _get_projections(self, composition, context):
        """Override to return nodes of Pytorch graph"""
        if self.show_pytorch:
            projections = self.pytorch_rep.composition._pytorch_projections
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
        sndr = proj.sender.owner
        rcvr = proj.receiver.owner
        if self.show_pytorch:
            processing_graph = self._get_processing_graph(self.composition, context)
            if proj in composition_projections:
                return True
            # Include if proj is betw. a sender and receiver specified as dependent on it in processing_graph
            elif (rcvr in processing_graph and sndr in processing_graph[rcvr]):
                return True
            else:
                return False
        else:
            return super()._proj_in_composition(proj, composition_projections, context)

    def _get_roles_by_node(self, composition, node, context):
        """Override in Pytorch mode to return NodeRole.INTERNAL for all nodes in nested compositions"""
        if self.show_pytorch:
            try:
                return composition.get_roles_by_node_at_any_level(node)
                # TEACHER_TARGET BREADCRUMB:  CHECK NESTED COMPS?
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
        if self.show_pytorch:
            if hasattr(rcvr, 'exclude_from_show_graph'):
                # Exclude PsyNeuLink Nodes in AutodiffComposition marked for exclusion from Pytorch graph
                return
            if rcvr in self.pytorch_rep.nodes_map and self.pytorch_rep.nodes_map[rcvr].exclude_from_gradient_calc:
                kwargs['color'] = self.exclude_from_gradient_calc_color
                kwargs['style'] = self.exclude_from_gradient_calc_line_style
            # # BREADCRUMB: REPLACE BELOW WITH THIS WHEN AUTODIFF_LEARNING_COMPONENTS IS IMPLEMENTED
            # elif rcvr in self.composition.autodiff_learning_components:
            elif isinstance(rcvr, LossMechanism):
                kwargs['color'] = self.learning_color
            elif rcvr in self.composition.get_nodes_by_role(NodeRole.TARGET):
                kwargs['color'] = self.learning_color
                kwargs['penwidth'] = str(self.bold_width)

            elif rcvr not in self.composition.nodes:
                #  Assign style to nodes of nested Compositions that are INPUT or OUTPUT nodes of Pytorch graph
                #  (since they are not in the outermost Composition and are therefore ignored when it is flattened)
                dependencies = self._get_processing_graph(self.composition, context)
                receivers = dependencies.keys()
                senders = [sender for sender_list in dependencies.values() for sender in sender_list]
                if rcvr in receivers and rcvr not in senders:
                    kwargs['color'] = self.output_color
                    kwargs['penwidth'] = str(self.bold_width)
                elif rcvr in senders and rcvr not in receivers:
                    kwargs['color'] = self.input_color
                    kwargs['penwidth'] = str(self.bold_width)
            g.node(*args, **kwargs)
        else:
            return super()._implement_graph_node( g, rcvr, context, *args, **kwargs)

    def _implement_graph_edge(self, graph, proj, context, *args, **kwargs):
        """Override to assign pytroch-specific custom attributes to edges"""

        if self.show_pytorch:
            kwargs['color'] = self.default_node_color

            if isinstance(proj, LossProjection):
                kwargs['color'] = self.exclude_from_gradient_calc_color
                kwargs['style'] = self.exclude_from_gradient_calc_line_style
                kwargs['penwidth'] = str(self.default_width)
                graph.edge(*args, **kwargs)
                return

            elif isinstance(proj.sender.owner, CompositionInterfaceMechanism):
                # Exclude any edges from CompositionInterfaceMechanism since those are never relevant in Pytorch graph
                return

            modulatory_node = None

            if proj.parameter_ports[0].mod_afferents:
                modulatory_mech = proj.parameter_ports[0].mod_afferents[0].sender.owner
                try:
                    modulatory_node = self.pytorch_rep.nodes_map[modulatory_mech]
                except KeyError:
                    pass

            if proj in self.pytorch_rep.projections_map:
                # If Projection is a LearningProjection that is active, assign color and arrowhead of a LearningProjection
                if proj.learnable or self.pytorch_rep.projections_map[proj].matrix.requires_grad:
                    kwargs['color'] = self.learning_color

                # If Projection is from a ModulatoryMechanism that is excluded from gradient calculations, assign that style
                elif modulatory_node and modulatory_node.exclude_from_gradient_calc:
                    kwargs['color'] = self.exclude_from_gradient_calc_color
                    kwargs['style'] = self.exclude_from_gradient_calc_line_style

            elif self._proj_in_composition(proj, self.pytorch_rep.projections_map, context) and proj.learnable:
                kwargs['color'] = self.learning_color

            graph.edge(*args, **kwargs)

        else:
            return super()._implement_graph_edge(graph, proj, context, *args, **kwargs)
