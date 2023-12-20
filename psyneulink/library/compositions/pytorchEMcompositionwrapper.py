# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for EMComposition"""

import torch
# try:
#     import torch
# except (ImportError, ModuleNotFoundError):
#     torch = None
from typing import Optional

from psyneulink.library.compositions.pytorchwrappers import PytorchCompositionWrapper
from psyneulink.library.components.mechanisms.modulatory.learning.EMstoragemechanism import EMStorageMechanism

__all__ = ['PytorchEMCompositionWrapper']

class PytorchEMCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for EMComposition as a Pytorch Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Assign storage_node (EMComposition's EMStorageMechanism)
        self.storage_node = [node for node in self.nodes_map.values()
                             if isinstance(node._mechanism, EMStorageMechanism)][0]

        # Get PytorchProjectionWrappers for Projections to match and retrieve nodes;
        #   used by get_memory() to construct memory_matrix and store_memory() to store entry in it
        pnl_storage_mech = self.storage_node._mechanism
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

    def execute_node(self, node, variable, context):
        """Override to handle storage of entry to memory_matrix by EMStorage Function"""
        if node is self.storage_node:
            self.store_memory(variable, context)
        else:
            super().execute_node(node, variable, context)

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

        memory = self.memory
        assert memory is not None, f"PROGRAM ERROR: '{self.name}'.memory is None"

        # Get current parameter values from EMComposition's EMStorageMechanism
        mech = self.storage_node._mechanism
        random_state = mech.function.parameters.random_state._get(context)
        decay_rate = mech.parameters.decay_rate._get(context)      # modulable, so use getter
        storage_prob = mech.parameters.storage_prob._get(context)  # modulable, so use getter
        field_weights = mech.parameters.field_weights.get(context) # modulable, so use getter
        concatenation_node = mech.concatenation_node
        num_match_fields = 1 if concatenation_node else len([i for i in mech.field_types if i==1])

        # Find weakest memory (i.e., with lowest norm)
        field_norms = torch.empty((len(memory),len(memory[0])))
        for row in range(len(memory)):
            for col in range(len(memory[0])):
                field_norms[row][col] = torch.linalg.norm(memory[row][col])
        if field_weights is not None:
            field_norms *= field_weights
        row_norms = torch.sum(field_norms, axis=1)
        idx_of_weakest_memory = torch.argmin(row_norms)

        values = []
        for i, field_projection in enumerate(self.match_projection_wrappers + self.retrieve_projection_wrappers):
            if i < num_match_fields:
                # For match projections, get entry to store from value of sender of Projection matrix
                #   (this is to accomodate concatenation_node)
                axis = 0
                entry_to_store = field_projection.sender.value
                if concatenation_node is None:
                    assert (entry_to_store == memory_to_store[i]).all(), \
                        f"PROGRAM ERROR: misalignment between inputs and fields for storing them"
            else:
                # For retrieve projections, get entry to store from memory_to_store (which has inputs to all fields)
                axis = 1
                entry_to_store = memory_to_store[i - num_match_fields]
            # Get matrix containing memories for the field from the Projection
            field_memory_matrix = field_projection.matrix

            field_projection.matrix = self.storage_node.function(entry_to_store,
                                                                 memory_matrix=field_memory_matrix,
                                                                 axis=axis,
                                                                 storage_location=idx_of_weakest_memory,
                                                                 storage_prob=storage_prob,
                                                                 decay_rate=decay_rate,
                                                                 random_state=random_state)
            values.append(field_projection.matrix)

        self.storage_node.value = values
        return values
