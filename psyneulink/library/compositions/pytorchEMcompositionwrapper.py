# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ********************************************* PytorchComponent *************************************************

"""PyTorch wrapper for EMComposition"""

import graph_scheduler
import torch
import numpy as np


from psyneulink.library.compositions.pytorchcomponents import PytorchCompositionWrapper
from psyneulink.library.components.mechanisms.modulatory.learning.EMstoragemechanism import EMStorageMechanism

__all__ = ['PytorchEMCompositionWrapper']

class PytorchEMCompositionWrapper(PytorchCompositionWrapper):
    """Wrapper for EMComposition as a Pytorch Module"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Assign storage_node (EMComposition's EMStorageMechanism)
        self.storage_node = [node for node in self.nodes_map.values()
                             if isinstance(node._mechanism, EMStorageMechanism)][0]

        # Assign retrieval_projection_wrappers (used by get_memory() to construct memory_matrix)
        pnl_storage_mech = self.storage_node._mechanism
        num_fields = len(pnl_storage_mech.input_ports)
        num_learning_signals = len(pnl_storage_mech.learning_signals)
        learning_signals_for_retrieved = pnl_storage_mech.learning_signals[num_learning_signals - num_fields:]
        pnl_retrieval_projs = [retrieved_learning_signal.efferents[0].receiver.owner
                               for retrieved_learning_signal in learning_signals_for_retrieved]
        self.retrieval_projection_wrappers = [self.projections_map[pnl_retrieval_proj]
                           for pnl_retrieval_proj in pnl_retrieval_projs]

    def execute_node(self, node, variable, context):
        """Override to handle storage of entry to memory_matrix by EMStorage Function"""
        if node is self.storage_node:
            self.store_memory(variable, context)
        else:
            super().execute_node(node, variable, context)

    @property
    def memory(self)->list:
        """Return list of memories in which rows (outer dimension) are memories for each field.
        These are derived from the matrix parameters of the afferent Projections to the retrieval_nodes
        """
        num_fields = len(self.storage_node.afferents)
        memory_matrices = [field.matrix for field in self.retrieval_projection_wrappers]
        memory_capacity = len(memory_matrices[0])
        return (None if not all(val for val in [num_fields, memory_matrices, memory_capacity])
                else[[memory_matrices[j][i] for j in range(num_fields)] for i in range(memory_capacity)])

    def store_memory(self, memory_to_store, context):
        """Store variable in memory_matrix

        For each node in query_input_nodes and value_input_nodes,
        assign its value to afferent weights of corresponding retrieved_node.
        - memory = matrix of entries made up vectors for each field in each entry (row)
        - memory_full_vectors = matrix of entries made up vectors concatentated across all fields (used for norm)
        - entry_to_store = query_input or value_input to store
        - field_memories = weights of Projections for each field

        DIVISION OF LABOR BETWEEN MECHANISM AND FUNCTION:
        EMStorageMechanism._execute:
         - compute norms to find weakest entry in memory
         - compute storage_prob to determine whether to store current entry in memory
         - call function for each LearningSignal to decay existing memory and assign input to weakest entry
        EMStorage function:
         - decay existing memories
         - assign input to weakest entry (given index for passed from EMStorageMechanism)

        :return: List[2d np.array] self.learning_signal
        """

        from psyneulink.library.compositions.pytorchcomponents import pytorch_function_creator
        memory = self.memory
        assert memory is not None, f"PROGRAM ERROR: '{self.name}'.memory is None"

        # Get updated parameter values for EMComposition's EMStorageMechanism
        mech = self.storage_node._mechanism
        decay_rate = mech.parameters.decay_rate._get(context)      # modulable, so use getter
        storage_prob = mech.parameters.storage_prob._get(context)  # modulable, so use getter
        field_weights = mech.parameters.field_weights.get(context) # modulable, so use getter
        concatenation_node = mech.concatenation_node
        num_match_fields = 1 if concatenation_node else len([i for i in mech.field_types if i==1])

        # FIX: FROM pytorch_function_creator (torch version of EMStorage)
        # def func(entry, memory_matrix):
        #     # if random_state.uniform(0, 1) < storage_prob:
        #     if torch.rand(1) < storage_prob:
        #         if decay_rate:
        #             memory_matrix *= decay_rate
        #         if storage_location is not None:
        #             idx_of_min = storage_location
        #         else:
        #             # Find weakest entry (i.e., with lowest norm) along specified axis of matrix
        #             idx_of_min = torch.argmin(torch.linalg.norm(memory_matrix, axis=axis))
        #         if axis == 0:
        #             memory_matrix[:,idx_of_min] = torch.tensor(entry)
        #         elif axis == 1:
        #             memory_matrix[idx_of_min,:] = torch.tensor(entry)
        #     return memory_matrix

        # # FIX: MAYBE NEEDED BELOW: -------------------------
        # memory_matrix = mech.function.parameters.memory_matrix._get(context)
        # axis = mech.function.parameters.axis._get(context)
        # storage_location = mech.function.parameters.storage_location._get(context)
        # # FIX: ------------------------------------------

        # FIX: MODIFIED FROM EMStorageMechanism._execute:

        field_norms = torch.tensor([torch.linalg.norm(torch.stack(field), dim=1) for field in [row for row in
                                                                                            self.memory]])
        if field_weights is not None:
            field_norms *= field_weights
        row_norms = torch.sum(field_norms, axis=1)
        idx_of_weakest_memory = torch.argmin(row_norms)

        value = []
        for i, field_projection in enumerate([learning_signal.efferents[0].receiver.owner
                                            for learning_signal in self.learning_signals]):
            if i < num_match_fields:
                # For match matrices,
                #   get entry to store from variable of Projection matrix (memory_field)
                #   to match_node in which memory will be store (this is to accomodate concatenation_node)
                axis = 0
                entry_to_store = field_projection.variable
                if concatenation_node is None:
                    assert np.all(entry_to_store == variable[i]),\
                        f"PROGRAM ERROR: misalignment between inputs and fields for storing them"
            else:
                # For retrieval matrices,
                #    get entry to store from variable (which has inputs to all fields)
                axis = 1
                entry_to_store = variable[i - num_match_fields]
            # Get matrix containing memories for the field from the Projection
            field_memory_matrix = field_projection.parameters.matrix.get(context)

            value.append(super(LearningMechanism, self)._execute(variable=entry_to_store,
                                                                 memory_matrix=field_memory_matrix,
                                                                 axis=axis,
                                                                 storage_location=idx_of_weakest_memory,
                                                                 storage_prob=storage_prob,
                                                                 decay_rate=decay_rate,
                                                                 context=context,
                                                                 runtime_params=runtime_params))
        self.parameters.value._set(value, context)
        return value

        # FIX: MAYBE:??
        PARAMETER = node.execute(variable, memory_matrix)

