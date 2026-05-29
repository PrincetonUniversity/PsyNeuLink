# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for EMComposition"""

# import torch
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from typing import Optional
from collections import defaultdict

import numpy as np

from psyneulink.core.components.functions.nonstateful.transformfunctions import COMPUTE_SCORES
from psyneulink.library.compositions.pytorchwrappers import (
    PytorchCompositionWrapper, PytorchMechanismWrapper, PytorchLossMechanismWrapper)
from psyneulink.library.components.mechanisms.processing.objective.lossmechanism import LossMechanism
from psyneulink.library.components.mechanisms.processing.integrator.externalmemorymechanism import (
    ExternalMemoryMechanism)
from psyneulink.core.globals.keywords import ALL, FIRST, LAST, RETRIEVE, STORE

__all__ = ['PytorchEMCompositionWrapper2']

class PytorchEMCompositionWrapper2(PytorchCompositionWrapper):
    """Wrapper for EMComposition as a Pytorch Module"""

    def _pytorch_mechanism_wrapper_type(self, mech):
        return defaultdict(lambda: PytorchMechanismWrapper,
                           # return defaultdict(lambda: super(PytorchCompositionWrapper)._pytorch_mechanism_wrapper_type(mech),
                           {LossMechanism: PytorchLossMechanismWrapper,
                            ExternalMemoryMechanism: PytorchExternalMemoryMechanismWrapper
                            })[mech.__class__]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure that execution_sets for field_memory_nodes are in the correct places in the sequence
        field_memory_pytorch_nodes = [v for k,v in self.nodes_map.items() if k in self.composition.field_memory_nodes]
        set(field_memory_pytorch_nodes) == self.execution_sets[1]
        assert len(self.execution_sets) == 6 if self.composition.field_weight_nodes else 4
        self.execution_sets.append(self.execution_sets[1])
        field_memory_executions = [i+1 for i, exec_set in enumerate(self.execution_sets)
                                   if field_memory_pytorch_nodes[0] in exec_set]
        assert len(field_memory_executions) == 3
        self.field_memory_operations = {k:v for k,v in zip(field_memory_executions, [COMPUTE_SCORES, RETRIEVE, STORE])}

        # IMPLEMENTATION NOTE:
        #    This is needed for access by subcomponents to the PytorchEMCompositionWrapper when EMComposition is nested,
        #    and so _build_pytorch_representation is called on the outer Composition but not EMComposition itself;
        #    access must be provided via EMComposition's pytorch_representation, rather than directly assigning
        #    PytorchEMCompositionWrapper as an attribute on the subcomponents, since doing the latter introduces a
        #    recursion when torch.nn.module.state_dict() is called on any wrapper in the hierarchy.
        if self.composition.pytorch_representation is None:
            self.composition.pytorch_representation = self

    @property
    def memory(self)->Optional[torch.Tensor]:
        """Return list of memories in which rows (outer dimension) are memories for each field.
        These are derived from the matrix parameters of the afferent Projections to the retrieval_nodes
        """
        num_fields = len(self.storage_node.afferents)
        memory_matrices = [field.matrix for field in self.retrieve_projection_wrappers]
        memory_capacity = len(memory_matrices[0])
        return (None if not all(val for val in [num_fields, memory_matrices, memory_capacity])
                else torch.stack([torch.stack([memory_matrices[j][i]
                                               for j in range(num_fields)])
                                  for i in range(memory_capacity)]))


class PytorchExternalMemoryMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for ExternalMemoryMechanism as a Pytorch Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = torch.tensor(self.mechanism.memory)

    # EM2 BREADCRUMB: REFACTOR TO HANDLE COMPUTATIONS ON SPECIFIC OPTIMIZATIONS
    # def execute(self, variable, optimization_num, synch_with_pnl_options, sequence_lengths, context=None):
    #     """Override to handle storage of entry to memory_matrix by EMStorage Function"""
    #     if self.mechanism is self.composition.storage_node:
    #         # 8/20/25 BREADCRUMB: REFACTOR TO USE execution_in_additional_optimizations
    #         num_optimizations = self._context.composition.parameters.optimizations_per_minibatch._get(context)
    #         store_on_optimization = self.composition.parameters.store_on_optimization._get(context)
    #         if optimization_num == 0 and store_on_optimization == FIRST:
    #             store = True
    #         elif ((optimization_num + 1) == num_optimizations) and store_on_optimization == LAST:
    #             store = True
    #         elif store_on_optimization == ALL:
    #             store = True
    #         else:
    #             store = False
    #         if store:
    #             self.store_memory(variable, context)
    #
    #     else:
    #         super().execute(variable, optimization_num, synch_with_pnl_options, context)

    def execute_function(self, function, variable, fct_has_mult_args=False):
        pytorch_rep = self.composition.pytorch_representation
        execution_set_num = self.composition.pytorch_representation.outer_creator.execution_set_num
        if execution_set_num in pytorch_rep.field_memory_operations:
            operation = pytorch_rep.field_memory_operations[execution_set_num]
            variable.append(operation)
            fct_has_mult_args = True
        super().execute_function(function, variable, fct_has_mult_args)