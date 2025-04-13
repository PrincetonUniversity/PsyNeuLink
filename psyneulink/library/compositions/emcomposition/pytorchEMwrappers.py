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

from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper, PytorchMechanismWrapper
from psyneulink.library.components.mechanisms.modulatory.learning.EMstoragemechanism import EMStorageMechanism
from psyneulink.core.globals.keywords import AFTER

__all__ = ['PytorchEMCompositionWrapper']

class PytorchEMCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for EMComposition as a Pytorch Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Assign storage_node (EMComposition's EMStorageMechanism) (assumes there is only one)
        self.storage_node = self.nodes_map[self.composition.storage_node]
        # Execute storage_node after gradient calculation,
        #     since it assigns weights manually which messes up PyTorch gradient tracking in forward() and backward()
        self.storage_node.exclude_from_gradient_calc = AFTER

        # Get PytorchProjectionWrappers for Projections to match and retrieve nodes;
        #   used by get_memory() to construct memory_matrix and store_memory() to store entry in it
        pnl_storage_mech = self.storage_node.mechanism

        num_fields = len(pnl_storage_mech.input_ports)
        num_learning_signals = len(pnl_storage_mech.learning_signals)
        num_match_fields = num_learning_signals - num_fields

        # ProjectionWrappers for match nodes
        learning_signals_for_match_nodes = pnl_storage_mech.learning_signals[:num_match_fields]
        pnl_match_projs = [match_node_learning_signal.efferents[0].receiver.owner
                           for match_node_learning_signal in learning_signals_for_match_nodes]
        self.match_projection_wrappers = [self.projections_map[pnl_match_proj]
                                          for pnl_match_proj in pnl_match_projs]

        # ProjectionWrappers for retrieve nodes
        learning_signals_for_retrieve_nodes = pnl_storage_mech.learning_signals[num_match_fields:]
        pnl_retrieve_projs = [retrieve_node_learning_signal.efferents[0].receiver.owner
                              for retrieve_node_learning_signal in learning_signals_for_retrieve_nodes]
        self.retrieve_projection_wrappers = [self.projections_map[pnl_retrieve_proj]
                                             for pnl_retrieve_proj in pnl_retrieve_projs]

        # IMPLEMENTATION NOTE:
        #    This is needed for access by subcomponents to the PytorchEMCompositionWrapper when EMComposition is nested,
        #    and so _build_pytorch_representation is called on the outer Composition but not EMComposition itelf;
        #    access must be provided via EMComposition's pytorch_representation, rather than directly assigning
        #    PytorchEMCompositionWrapper as an attribute on the subcomponents, since doing the latter introduces a
        #    recursion when torch.nn.module.state_dict() is called on any wrapper in the hiearchay.
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


class PytorchEMMechanismWrapper(PytorchMechanismWrapper):
    """Wrapper for EMStorageMechanism as a Pytorch Module"""

    def execute(self, variable, optimization_num, synch_with_pnl_options, context=None):
        """Override to handle storage of entry to memory_matrix by EMStorage Function"""
        if self.mechanism is self.composition.storage_node:
            # Only execute store after last optimization repetition for current mini-batch
            # 7/10/24:  FIX: MOVE PASSING OF THESE PARAMETERS TO context
            if not (optimization_num + 1) % context.composition.parameters.optimizations_per_minibatch.get(context):
                self.store_memory(variable, context)
        else:
            super().execute(variable, optimization_num, synch_with_pnl_options, context)

    # # MODIFIED 7/29/24 NEW: NEEDED FOR torch MPS SUPPORT
    # @torch.jit.script_method
    # MODIFIED 7/29/24 END
    def store_memory(self, memory_to_store, context):
        """Store variable in memory_matrix (parallel EMStorageMechanism._execute)

        For each node in query_input_nodes and value_input_nodes,
        assign its value to weights of corresponding afferents to corresponding match_node and/or retrieved_node.
        - memory = matrix of entries made up vectors for each field in each entry (row)
        - entry_to_store = query_input or value_input to store
        - field_projections = Projections the matrices of which comprise memory

        DIVISION OF LABOR between this method and function called by it
        store_memory (corresponds to EMStorageMechanism._execute)
         - compute norms to find weakest entry in memory
         - compute storage_prob to determine whether to store current entry in memory
         - call function with memory matrix for each field, to decay existing memory and assign input to weakest entry
        storage_node.function (corresponds to EMStorage._function):
         - decay existing memories
         - assign input to weakest entry (given index for passed from EMStorageMechanism)

        :return: List[2d tensor] updated memories
        """
        pytorch_rep = self.composition.pytorch_representation

        memory = pytorch_rep.memory
        assert memory is not None, f"PROGRAM ERROR: '{pytorch_rep.name}'.memory is None"

        # Get current parameter values from EMComposition's EMStorageMechanism
        mech = self.mechanism
        random_state = mech.function.parameters.random_state._get(context)
        decay_rate = mech.parameters.decay_rate._get(context)      # modulable, so use getter
        storage_prob = mech.parameters.storage_prob._get(context)  # modulable, so use getter
        field_weights = mech.parameters.field_weights.get(context) # modulable, so use getter
        concatenation_node = mech.concatenation_node
        # MODIFIED 7/29/24 OLD:
        num_match_fields = 1 if concatenation_node else len([i for i in mech.field_types if i==1])
        # # MODIFIED 7/29/24 NEW: NEEDED FOR torch MPS SUPPORT
        # if concatenation_node:
        #     num_match_fields = 1
        # else:
        #     num_match_fields = 0
        #     for i in mech.field_types:
        #         if i==1:
        #             num_match_fields += 1
        # MODIFIED 7/29/24 END

        # Find weakest memory (i.e., with lowest norm)
        field_norms = torch.linalg.norm(memory, dim=2)
        if field_weights is not None:
            field_norms *= field_weights
        row_norms = torch.sum(field_norms, axis=1)
        idx_of_weakest_memory = torch.argmin(row_norms)

        values = []
        for field_projection in pytorch_rep.match_projection_wrappers + pytorch_rep.retrieve_projection_wrappers:
            field_idx = pytorch_rep.composition._field_index_map[field_projection._pnl_proj]
            if field_projection in pytorch_rep.match_projection_wrappers:
                # For match projections:
                # - get entry to store from value of sender of Projection matrix (to accommodate concatenation_node)
                entry_to_store = field_projection.sender_wrapper.output

                # Retrieve the correct field (for each batch, batch is first dimension)
                memory_to_store_indexed = memory_to_store[:, field_idx, :]

                # - store in row
                axis = 0
                if concatenation_node is None:
                    # Double check that the memory passed in is the output of the projection for the correct field
                    assert (entry_to_store == memory_to_store_indexed).all(), \
                        (f"PROGRAM ERROR: misalignment between memory to be stored (input passed to store_memory) "
                         f"and value of projection to corresponding field.")
            else:
                # For retrieve projections:
                # - get entry to store from memory_to_store (which has inputs to all fields)
                entry_to_store = memory_to_store[:, field_idx, :]
                # - store in column
                axis = 1
            # Get matrix containing memories for the field from the Projection
            field_memory_matrix = field_projection.matrix

            field_projection.matrix = self.function(entry_to_store,
                                                    memory_matrix=field_memory_matrix,
                                                    axis=axis,
                                                    storage_location=idx_of_weakest_memory,
                                                    storage_prob=storage_prob,
                                                    decay_rate=decay_rate,
                                                    random_state=random_state)
            values.append(field_projection.matrix)

        self.value = values
        return values
