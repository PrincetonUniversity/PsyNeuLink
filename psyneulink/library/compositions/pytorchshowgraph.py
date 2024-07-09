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
from psyneulink.core.compositions.showgraph import ShowGraph

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
    <Projection.exclude_from_autodiff>` will not be used in the gradient calculations at all.

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
        # FIX: PUT PYTORCH SPECIFIC HANDING HERE
        return super().show_graph(*args, **kwargs)


