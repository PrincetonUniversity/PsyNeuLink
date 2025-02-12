# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for GRUComposition"""

import torch
import numpy as np
from typing import Union, Optional

from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper
from psyneulink.core.globals.context import handle_external_context

__all__ = ['PytorchGRUCompositionWrapper']

class PytorchGRUCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for GRUComposition as a Pytorch Module
    Functions both as a:

    * PytorchCompositionWrapper, in managing the exchange of the Composition's Projection `Matrices
      <MappingProjection_Matrix>` and the Pytorch GRU Module's parameters, and returning an output value
      when used as a standalone Composition;

    * PytorchMechanismWrapper in directly managing its own forward method and execution, including calcuating
      the value of its afferents as the input to the Pytorch GRU Module.
    """

    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 h0:Union[list, np.ndarray, torch.Tensor],
                 bias:bool,
                 composition,
                 device,
                 outer_creator,
                 **kwargs):
        super().__init__(composition,device,outer_creator, **kwargs)
        assert len(h0) == hidden_size, f"PROGRAM ERROR: Length of h0 is not the same as hidden_size."
        self.hidden_state = torch.tensor(h0, device=device)
        self.torch_gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, bias=bias)

    @handle_external_context()
    def forward(self, inputs, optimization_num, context=None)->dict:
        """Forward method of the model for PyTorch modes
        Returns a dictionary {output_node:value} with the output value for the module in case it is run as a
        standalone Composition; otherwise, those will be ignored and the outputs will be used by the aggregate_afferents
        method(s) of the other node(s) to which it projects
        """
        output, self.hidden_state = self.torch_gru(inputs, self.hidden_state)
        return {self: output}

    def execute_node(self, node, variable, optimization_num, context):
        """Override to execute the Pytorch GRU Module directly as a node."""
