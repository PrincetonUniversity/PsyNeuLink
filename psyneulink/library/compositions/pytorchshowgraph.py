# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************** PyTorch show_graph *********************************************************

from beartype import beartype

from psyneulink._typing import Optional, Union, Literal

from psyneulink.core.compositions import NodeRole
from psyneulink.core.compositions.showgraph import ShowGraph, SHOW_JUST_LEARNING_PROJECTIONS, SHOW_LEARNING
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.llvm import ExecutionMode
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import SHOW_PYTORCH, PNL

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
        self.show_pytorch = kwargs.pop('show_pytorch', False)
        context = kwargs.get('context')
        if self.show_pytorch:
            self.pytorch_rep = self.composition._build_pytorch_representation(context, refresh=False)
        self.exclude_from_gradient_calc_line_style = kwargs.pop(EXCLUDE_FROM_GRADIENT_CALC_LINE_STYLE, 'dotted')
        self.exclude_from_gradient_calc_color = kwargs.pop(EXCLUDE_FROM_GRADIENT_CALC_COLOR, 'brown')
        return super().show_graph(*args, **kwargs)

    def _get_processing_graph(self, composition, context):
        """Helper method that creates dependencies graph for nodes of autodiffcomposition used in Pytorch mode"""
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
                    # FIX: 3/9/25 - HANDLE NODE THAT PROJECTS TO OUTPUT_CIM IN SAME WAY:
                    # Add dependency of INPUT node of nested graph on node in outer graph that projects to it
                    elif (isinstance(receiver, CompositionInterfaceMechanism) and
                          # projection.receiver.owner._get_destination_info_from_input_CIM(projection.receiver)[1]
                          # FIX: SUPPOSED TO RETRIEVE GRU NODE HERE,
                          #      BUT NEED TO DEAL WITH INTERFERING PROJECTION FROM OUTPUT NODE
                          receiver._get_source_info_from_output_CIM(projection.receiver)[1] is node):
                        dependencies.add(sender)
                    else:
                        for proj in [proj for proj in node.afferents if proj.sender.owner in nodes]:
                            dependencies.add(proj.sender.owner)
                processing_graph[node] = dependencies

            # Add TARGET nodes
            for node in self.composition.learning_components:
                processing_graph[node] = set([afferent.sender.owner for afferent in node.path_afferents])
            return {k: processing_graph[k] for k in sorted(processing_graph.keys())}

        else:
            return super()._get_processing_graph(composition, context)

    def _get_nodes(self, composition, context):
        """Override to return nodes of PytorchCompositionWrapper rather than autodiffcomposition"""
        if self.show_pytorch:
            nodes = [node for node in self.pytorch_rep.nodes_map
                           if SHOW_PYTORCH in self.pytorch_rep.nodes_map[node]._use]
            return nodes
        else:
            return super()._get_nodes(composition, context)

    def _get_projections(self, composition, context):
        """Override to return nodes of Pytorch graph"""
        if self.show_pytorch:
            # projections = list(self.pytorch_rep.projections_map.keys())
            projections = [proj for proj in self.pytorch_rep.projections_map
                           if SHOW_PYTORCH in self.pytorch_rep.projections_map[proj]._use]
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
        sndr = proj.sender.owner
        rcvr = proj.receiver.owner
        # # MODIFIED 2/16/25 NEW:
        # if isinstance(rcvr, CompositionInterfaceMechanism):
        #     # If receiver is an input_CIM, get the node in the inner Composition to which it projects
        #     #   as it may be specified as dependent on the sender in the autodiff processing_graph
        #     rcvr = rcvr._get_destination_info_from_input_CIM(proj.receiver)[1]
        # MODIFIED 2/16/25 END
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
        if self.show_pytorch:
            if hasattr(rcvr, 'exclude_from_show_graph'):
                # Exclude PsyNeuLink Nodes in AutodiffComposition marked for exclusion from Pytorch graph
                return
            if rcvr in self.pytorch_rep.nodes_map and self.pytorch_rep.nodes_map[rcvr].exclude_from_gradient_calc:
                kwargs['style'] = self.exclude_from_gradient_calc_line_style
                kwargs['color'] = self.exclude_from_gradient_calc_color
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
        """Override to assign custom attributes to edges"""

        if self.show_pytorch:
            kwargs['color'] = self.default_node_color

            modulatory_node = None
            if proj.parameter_ports[0].mod_afferents:
                # MODIFIED 2/22/25 OLD:
                modulatory_node = self.pytorch_rep.nodes_map[proj.parameter_ports[0].mod_afferents[0].sender.owner]
                # # MODIFIED 2/22/25 NEW:
                # modulatory_node = self.nodes_map[proj.parameter_ports[0].mod_afferents[0].sender.owner]
                # # MODIFIED 2/22/25 END

            if proj in self.pytorch_rep.projections_map:
                # # MODIFIED 2/25/25 NEW:
                # if ((hasattr(proj, 'learnable') and proj.learnable)
                #         or (proj in self.pytorch_rep.projections_map and
                #             self.pytorch_rep.projections_map[proj].matrix.requires_grad)):
                #     proj_is_learnable = True
                # # MODIFIED 2/25/25 END

                # If Projection is a LearningProjection that is active, assign color and arrowhead of a LearningProjection
                # # MODIFIED 2/25/25 OLD:
                if proj.learnable or self.pytorch_rep.projections_map[proj].matrix.requires_grad:
                # # MODIFIED 2/25/25 NEW:
                # if proj_is_learnable:
                # # MODIFIED 2/25/25 END
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
