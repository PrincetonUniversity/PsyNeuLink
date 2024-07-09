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
from psyneulink.core.compositions.showgraph import ShowGraph
from psyneulink.core.llvm import ExecutionMode


__all__ = ['SHOW_PYTORCH']

SHOW_PYTORCH = 'show_pytorch'

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
        from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
        self.show_pytorch = kwargs.pop(SHOW_PYTORCH, self.show_pytorch)
        context = kwargs.get('context')
        if self.show_pytorch:
            self.pytorch_rep = self.composition._build_pytorch_representation(context)
        return super().show_graph(*args, **kwargs)

    def _get_processing_graph(self, composition, context):
        """Helper method that creates dependencies graph for nodes of autodiffcomposition used in Pytorch mode"""
        if self.show_pytorch:
            processing_graph = {}
            projections = self._get_projections(composition, context)
            for node in self._get_nodes(composition, context):
                dependencies = set()
                for projection in projections:
                    if node is projection.receiver.owner:
                        dependencies.add(projection.sender.owner)
                processing_graph[node] = dependencies
            return processing_graph
        else:
            return super()._get_nodes(composition, context)

    def _get_nodes(self, composition, context):
        """Override to return nodes of PytorchCompositionWrapper rather than autodiffcomposition"""
        if self.show_pytorch:
            nodes = list(self.pytorch_rep.nodes_map.keys())
            return nodes
        else:
            return super()._get_nodes(composition, context)

    def _get_projections(self, composition, context):
        """Override to return nodes of PytorchCompositionWrapper rather than autodiffcomposition"""
        if self.show_pytorch:
            nodes = list(self.pytorch_rep.projections_map.keys())
            return nodes
        else:
            return super()._get_projections(composition, context)

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
