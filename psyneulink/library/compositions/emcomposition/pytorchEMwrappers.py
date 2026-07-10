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

import warnings
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
        self._differentiable_storage_mode_warned = False
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

    def _is_learning_mode_store(self, context):
        return context is not None and ContextFlags.LEARNING_MODE in context.runmode

    def _is_differentiable_storage(self, context):
        return bool(self.composition.parameters.differentiable_storage._get(context))

    def _top_pytorch_rep(self):
        top_rep = self.composition.pytorch_representation
        while getattr(top_rep, 'outer_creator', None) is not None:
            top_rep = top_rep.outer_creator
        return top_rep

    def _is_start_of_forward_pass(self):
        """True if the current execution belongs to the first sequence element of the current forward pass
        (in non-sequence mode every forward pass is a single element, so this is True on every pass)."""
        return getattr(self._top_pytorch_rep(), '_current_seq_index', 0) == 0

    def _warn_if_differentiable_storage_has_no_effect(self):
        """Warn (once) if differentiable_storage is used outside of full_sequence_mode, where it cannot
        have any effect: gradients can only flow through entries stored and retrieved within the *same*
        forward pass, and outside of full_sequence_mode each store's graph is severed before the entry
        can ever be retrieved (each trial is its own forward/backward pass)."""
        if self._differentiable_storage_mode_warned:
            return
        if not getattr(self._top_pytorch_rep(), '_full_sequence_mode', False):
            warnings.warn(
                f"'differentiable_storage' is enabled for '{self.composition.name}', but it is being trained "
                f"without 'full_sequence_mode'; gradients can only flow through stored entries that are "
                f"retrieved within the same forward pass, so the option will have no effect. Set "
                f"'full_sequence_mode=True' on the outermost AutodiffComposition (and provide each sequence "
                f"as a single trial) for differentiable storage to be effective."
            )
        self._differentiable_storage_mode_warned = True

    def _detach_memory(self):
        """Cut the autograd graph carried by the memory buffer (used with `differentiable_storage
        <EMComposition.differentiable_storage>` at the start of each forward pass: entries stored during one
        forward pass must not carry their graph into the next one, whose backward pass would fail because the
        parameters producing them were modified in place by the intervening optimizer step)."""
        pytorch_function = getattr(self.function, 'function', None)
        if pytorch_function is not None and hasattr(pytorch_function, 'set_memory'):
            pytorch_function.set_memory(pytorch_function.get_memory().detach())

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
        operation = self._get_field_memory_operation()

        # At the start of each forward pass, cut any autograd graph carried by the memory buffer from a
        # previous forward pass (relevant with `differentiable_storage <EMComposition.differentiable_storage>`;
        # a no-op otherwise, since non-differentiable stores never attach a graph). Entries stored during one
        # forward pass may not carry their graph into the next: the intervening optimizer step modified the
        # parameters that produced them in place, so backpropagating through them again is invalid.
        if operation == COMPUTE_SCORES and self._is_start_of_forward_pass():
            self._detach_memory()

        # During learning, perform the STORE for each sequence element *within* the forward pass, rather than
        # deferring it until after the backward pass. The previous (deferred) behavior recorded the store in a
        # dict keyed by this node, so for a multi-element sequence only the *last* element's entry survived, and
        # it was applied only after the full sequence's backward pass. That prevented within-sequence
        # read-before-write retrieval in `full_sequence_mode <AutodiffComposition.full_sequence_mode>`: every
        # element retrieved against an empty/stale memory. Storing per element here gives each element's
        # retrieval access to the entries stored by preceding elements, and is safe and correct because:
        #   * by default the store runs under ``torch.no_grad()`` so stored entries are detached -- the episodic
        #     memory is a non-differentiable buffer and gradients flow only through the retrieval *query*
        #     (matching the reference EM implementations, e.g. Giallanza et al. (2024)); with
        #     `differentiable_storage <EMComposition.differentiable_storage>` the store is instead performed
        #     out-of-place with the graph intact, so gradients also flow through the stored entries themselves
        #     (ESBN-style; Webb et al., 2021); and
        #   * retrievals read a ``clone()`` of the memory (see ``_gen_pytorch_fct`` of the field-memory
        #     function), so the store does not modify any tensor that is part of the autograd graph.
        # ``store_on_optimization`` still selects which optimization step performs the store.
        if operation == STORE and self._is_learning_mode_store(context):
            if not self._should_store_on_optimization(optimization_num, context):
                return self.output
            pytorch_function = getattr(self.function, 'function', None)
            if self._is_differentiable_storage(context):
                self._warn_if_differentiable_storage_has_no_effect()
                if pytorch_function is not None and hasattr(pytorch_function, 'set_differentiable'):
                    pytorch_function.set_differentiable(True)
                return super().execute(variable, optimization_num, synch_with_pnl_options, sequence_lengths,
                                       context)
            if pytorch_function is not None and hasattr(pytorch_function, 'set_differentiable'):
                pytorch_function.set_differentiable(False)
            with torch.no_grad():
                return super().execute(variable, optimization_num, synch_with_pnl_options, sequence_lengths, context)

        return super().execute(variable, optimization_num, synch_with_pnl_options, sequence_lengths, context)

    def execute_function(self, function, variable, fct_has_mult_args=False):
        operation = self._get_field_memory_operation()
        if operation is not None:
            variable.append(operation)
            fct_has_mult_args = True
        return super().execute_function(function, variable, fct_has_mult_args)
