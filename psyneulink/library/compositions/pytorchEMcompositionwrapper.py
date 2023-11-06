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

    # FIX: ?NEED TO CONTRUCT memory_matix FROM ??memory_template or pytorch?? IN __init__?

    def execute_node(self, node, variable):
        """Override to handle storage of entry to memory_matrix by EMStorage Function"""
        if isinstance(node._mechanism, EMStorageMechanism):
            self.store_memory(variable)
        else:
            super().execute_node(node, variable)


    def store_memory(self, memory_to_store):
        """Store variable in memory_matrix

        For each node in key_input_nodes and value_input_nodes,
        assign its value to afferent weights of corresponding retrieved_node.
        - memory = matrix of entries made up vectors for each field in each entry (row)
        - memory_full_vectors = matrix of entries made up vectors concatentated across all fields (used for norm)
        - entry_to_store = key_input or value_input to store
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

        memory_matrix = self.projections_map[node._mechanism.afferents[XXX]].matrix
        PARAMETER = node.execute(variable, memory_matrix)


        # FIX: FROM pytorch_function_creator:

        # memory_matrix = get_fct_param_value('memory_matrix')
        axis = get_fct_param_value('axis')
        storage_location = get_fct_param_value('storage_location')
        storage_prob = get_fct_param_value('storage_prob')
        decay_rate = get_fct_param_value('decay_rate')
        def func(entry, memory_matrix):
            # if random_state.uniform(0, 1) < storage_prob:
            if torch.rand(1) < storage_prob:
                if decay_rate:
                    memory_matrix *= decay_rate
                if storage_location is not None:
                    idx_of_min = storage_location
                else:
                    # Find weakest entry (i.e., with lowest norm) along specified axis of matrix
                    idx_of_min = torch.argmin(torch.linalg.norm(memory_matrix, axis=axis))
                if axis == 0:
                    memory_matrix[:,idx_of_min] = torch.tensor(entry)
                elif axis == 1:
                    memory_matrix[idx_of_min,:] = torch.tensor(entry)
            return memory_matrix


        # FIX: FROM EMStorageMechanism._execute:

        decay_rate = self.parameters.decay_rate._get(context)      # modulable, so use getter
        storage_prob = self.parameters.storage_prob._get(context)  # modulable, so use getter
        field_weights = self.parameters.field_weights.get(context) # modulable, so use getter
        concatenation_node = self.concatenation_node
        num_match_fields = 1 if concatenation_node else len([i for i in self.field_types if i==1])

        memory = self.parameters.memory_matrix._get(context)
        if memory is None or self.is_initializing:
            if self.is_initializing:
                # Return existing matrices for field_memories  # FIX: THE FOLLOWING DOESN'T TEST FUNCTION:
                return [learning_signal.receiver.path_afferents[0].parameters.matrix.get()
                        for learning_signal in self.learning_signals]
            # Raise exception if not initializing and memory is not specified
            else:
                owner_string = ""
                if self.owner:
                    owner_string = " of " + self.owner.name
                raise EMStorageMechanismError(f"Call to {self.__class__.__name__} function {owner_string} "
                                              f"must include '{MEMORY_MATRIX}' in params arg.")

        # Get least used slot (i.e., weakest memory = row of matrix with lowest weights) computed across all fields
        field_norms = np.array([np.linalg.norm(field, axis=1) for field in [row for row in memory]])
        if field_weights is not None:
            field_norms *= field_weights
        row_norms = np.sum(field_norms, axis=1)
        idx_of_weakest_memory = np.argmin(row_norms)

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
