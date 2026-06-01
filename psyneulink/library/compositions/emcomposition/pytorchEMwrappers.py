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
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import ALL, FIRST, LAST, RETRIEVE, STORE, SYNCH

__all__ = ['PytorchEMCompositionWrapper']

class PytorchEMCompositionWrapper(PytorchCompositionWrapper):
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
        memory_cycle_nodes = getattr(self.composition, 'memory_cycle_nodes', self.composition.field_memory_nodes)
        field_memory_pytorch_nodes = [v for k,v in self.nodes_map.items() if k in memory_cycle_nodes]
        set(field_memory_pytorch_nodes) == self.execution_sets[1]
        self.execution_sets.append(self.execution_sets[1])
        field_memory_executions = [i+1 for i, exec_set in enumerate(self.execution_sets)
                                   if field_memory_pytorch_nodes[0] in exec_set]
        assert len(field_memory_executions) == 3
        self.field_memory_operations = {k:v for k,v in zip(field_memory_executions, [COMPUTE_SCORES, RETRIEVE, STORE])}

        # IMPLEMENTATION NOTE:
        #    This is needed for access by subcomponents to PytorchEMCompositionWrapper when EMComposition is nested,
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
    """Wrapper for EMStorageMechanism as a Pytorch Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = None
        self._forced_operation = None
        self._deferred_store_pending = False
        self._refresh_memory_reference()

    def _refresh_memory_reference(self):
        pytorch_function = getattr(self.function, 'function', None)
        if hasattr(pytorch_function, 'get_memory'):
            self.memory = pytorch_function.get_memory()

    def _copy_pytorch_memory_to_pnl(self, context=None):
        self._refresh_memory_reference()
        if self.memory is None:
            return

        memory = self.memory.detach().cpu().numpy().copy()
        self.mechanism.function.parameters.memory._set(memory, context)
        self.mechanism.function.scores_function.parameters.matrix._set(memory.T, context)

    def set_pnl_variable_and_values(self, set_variable=False, set_value=True, context=None):
        if SYNCH not in self._use:
            return

        super().set_pnl_variable_and_values(
            set_variable=set_variable,
            set_value=set_value,
            context=context,
        )
        if set_value:
            self._copy_pytorch_memory_to_pnl(context)

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

    def _get_field_memory_operation(self):
        if self._forced_operation is not None:
            return self._forced_operation

        pytorch_rep = self.composition.pytorch_representation
        outer_creator = pytorch_rep.outer_creator or pytorch_rep
        execution_set_num = outer_creator.execution_set_num
        field_memory_executions = [
            i for i, exec_set in enumerate(outer_creator.execution_sets)
            if self in exec_set
        ]

        if execution_set_num in field_memory_executions:
            operation_idx = field_memory_executions.index(execution_set_num)
            return [COMPUTE_SCORES, RETRIEVE, STORE][operation_idx]

    def _should_defer_store(self, context):
        return context is not None and ContextFlags.LEARNING_MODE in context.runmode

    def _should_store_on_optimization(self, optimization_num, context):
        num_optimizations = self._context.composition.parameters.optimizations_per_minibatch._get(context)
        store_on_optimization = self.composition.parameters.store_on_optimization._get(context)
        if optimization_num == 0 and store_on_optimization == FIRST:
            return True
        if ((optimization_num + 1) == num_optimizations) and store_on_optimization == LAST:
            return True
        if store_on_optimization == ALL:
            return True
        return False

    def execute(self, variable, optimization_num, synch_with_pnl_options, sequence_lengths, context=None):
        pytorch_rep = self.composition.pytorch_representation
        outer_creator = pytorch_rep.outer_creator or pytorch_rep
        deferred_variable = outer_creator._nodes_to_execute_after_gradient_calc.get(self)

        if self._deferred_store_pending and deferred_variable is variable:
            if not self._should_store_on_optimization(optimization_num, context):
                self._deferred_store_pending = False
                return self.output
            self._forced_operation = STORE
            try:
                return super().execute(variable, optimization_num, synch_with_pnl_options, sequence_lengths, context)
            finally:
                self._forced_operation = None
                self._deferred_store_pending = False

        if self._get_field_memory_operation() == STORE and self._should_defer_store(context):
            outer_creator._nodes_to_execute_after_gradient_calc[self] = variable
            self._deferred_store_pending = True
            return self.output

        return super().execute(variable, optimization_num, synch_with_pnl_options, sequence_lengths, context)

    def execute_function(self, function, variable, fct_has_mult_args=False):
        operation = self._get_field_memory_operation()
        if operation is not None:
            variable.append(operation)
            fct_has_mult_args = True
        return super().execute_function(function, variable, fct_has_mult_args)
