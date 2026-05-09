# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  EpisodicMemoryMechanism ****************************************************


import copy
import warnings
from psyneulink._typing import Optional, Union

import numpy as np

from psyneulink.core.components.functions.function import _random_state_getter, _seed_setter, DEFAULT_SEED
from psyneulink.core.globals.keywords import (
    MULTIPLICATIVE_PARAM, NAME, OWNER_VALUE, VARIABLE)
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, is_numeric_scalar
from psyneulink.core.globals.context import Context
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemorymechanism import (
    EpisodicMemoryMechanism, EpisodicMemoryMechanismError)

__all__ = ['EpisodicMemoryFieldMechanism',
           'EpisodicMemoryFieldMechanismError',
           'QUERY', 'SCORES', 'COMBINED_SCORES', 'RETRIEVED']

QUERY = "QUERY"
SCORES = "SCORES"
COMBINED_SCORES = "COMBINED_SCORES"
RETRIEVED = "RETRIEVED"
DEFAULT_INPUT_PORT_NAME_PREFIX = 'FIELD_'
DEFAULT_INPUT_PORT_NAME_SUFFIX = '_INPUT'
DEFAULT_OUTPUT_PORT_PREFIX = 'RETRIEVED_'


def _normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


class EpisodicMemoryFieldMechanismError(EpisodicMemoryMechanismError):
    pass

class EpisodicMemoryFieldMechanism(EpisodicMemoryMechanism):
    """
    EpisodicMemoryFieldMechanism

    A field-local EpisodicMemoryMechanism used by EMComposition2, that stores each field's memory
    directly on the Mechanism as field_memory

    IMPLEMENTATION NOTE:  This is in distinction to the original EMComposition, in which each field's memory
                          was stored in Projection matrices managed by EMStorageMechanism.

    Ports
    -----
    input_port[QUERY]
        Current vector for this field.  Used for score computation and storage.

    output_port[SCORES]
        Match-weight vector between QUERY and every row in field_memory.

    input_port[COMBINED_SCORES]
        Combined retrieval weights, usually the softmax-normalized aggregate of
        all key-field SCORES.

    output_port[RETRIEVED]
        Dot product of COMBINED_SCORES with field_memory.

    """

    componentName = "EM_FIELD_MECHANISM"

    class Parameters(EpisodicMemoryMechanism.Parameters):
        field_memory = Parameter(
            None,
            stateful=True,
            loggable=True,
            constructor_argument="field_memory",
            # getter=_field_memory_getter
        )
        storage_prob = Parameter(
            1.0,
            modulable=True,
            aliases=[MULTIPLICATIVE_PARAM],
            stateful=True,
        )
        decay_rate = Parameter(0.0, modulable=True, stateful=True)
        normalize_memories = Parameter(True)
        purge_by_field_weight = Parameter(False)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies="seed")
        seed = Parameter(DEFAULT_SEED(), modulable=True, setter=_seed_setter)
        storage_condition = Parameter(None, stateful=False, loggable=False)

        def _validate_storage_prob(self, storage_prob):
            if not is_numeric_scalar(storage_prob) or not 0 <= storage_prob <= 1:
                return "must be a float in the interval [0, 1]."

        def _validate_decay_rate(self, decay_rate):
            if decay_rate is None:
                return None
            if not is_numeric_scalar(decay_rate) or not 0 <= decay_rate <= 1:
                return "must be a float in the interval [0, 1]."

    @check_user_specified
    def __init__(
        self,
        field_shape: int,
        field_memory: Union[list, np.ndarray],
        storage_prob: float = 1.0,
        decay_rate: float = 0.0,
        normalize_memories: bool = True,
        seed=None,
        params=None,
        name=None,
        prefs=None,
        **kwargs,
    ):
        self.field_shape = field_shape
        self.memory_capacity = len(field_memory)

        default_variable = np.array([
            np.zeros(field_shape),
            np.zeros(self.memory_capacity),
        ], dtype=object)

        super().__init__(
            default_variable=default_variable,
            input_ports=[
                {NAME: QUERY, VARIABLE: np.zeros(field_shape)},
                {NAME: COMBINED_SCORES, VARIABLE: np.zeros(self.memory_capacity)},
            ],
            output_ports=[
                {NAME: SCORES, VARIABLE: (self, 0)},
                {NAME: RETRIEVED, VARIABLE: (self, 1)},
            ],
            memory=field_memory,
            storage_prob=storage_prob,
            decay_rate=decay_rate,
            normalize_memories=normalize_memories,
            seed=seed,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs,
        )
        # BREADCRUMB -- OR SHOULD THIS BE:
        # self.parameters.field_memory._set(np.asarray(self.memory, dtype=float), override=True)
        # self.parameters.memory_matrix._set(np.asarray(field_memory, dtype=float), context=None)
        # self.parameters.field_memory._set(np.asarray(field_memory, dtype=float), Context(), override=True)


    def _handle_default_variable(self, default_variable=None, input_shapes=None, input_ports=None, function=None, params=None):
        return default_variable

    def _instantiate_input_ports(self, context=None):
        input_ports = [
            {NAME: QUERY, VARIABLE: np.zeros(self.field_shape)},
            {NAME: COMBINED_SCORES, VARIABLE: np.zeros(self.memory_capacity)},
        ]
        super(EpisodicMemoryMechanism, self)._instantiate_input_ports(input_ports=input_ports, context=context)

    def _instantiate_output_ports(self, context=None):
        output_ports = [
            {NAME: SCORES, VARIABLE: (OWNER_VALUE, 0)},
            {NAME: RETRIEVED, VARIABLE: (OWNER_VALUE, 1)},
        ]
        self.parameters.output_ports._set(output_ports, override=True, context=context)
        super()._instantiate_output_ports(context=context)

        for output_port in self.output_ports:
            output_port.parameters.require_projection_in_composition._set(False, override=True, context=context)

    def _validate_variable(self, variable, context=None):
        variable = np.asarray(variable, dtype=object)
        if len(variable) != 2:
            raise EpisodicMemoryFieldMechanismError(
                f"Variable for {self.name} must contain two items: QUERY and COMBINED_SCORES."
            )
        if len(variable[0]) != self.field_shape:
            raise EpisodicMemoryFieldMechanismError(
                f"QUERY input for {self.name} has length {len(variable[0])}; expected {self.field_shape}."
            )
        if len(variable[1]) != self.memory_capacity:
            raise EpisodicMemoryFieldMechanismError(
                f"COMBINED_SCORES input for {self.name} has length {len(variable[1])}; "
                f"expected {self.memory_capacity}."
            )
        return variable

    def _execute(self, variable=None, context=None, runtime_params=None):
        variable = self._validate_variable(variable, context=context)

        query = np.asarray(variable[0], dtype=float)
        combined_scores = np.asarray(variable[1], dtype=float)

        # MODIFIED EM2 OLD:
        # field_memory = copy.deepcopy(self.parameters.field_memory._get(context))
        # MODIFIED EM2 NEW:
        field_memory = copy.deepcopy(self.parameters.memory._get(context))
        # MODIFIED EM2 END
        if field_memory is None:
            field_memory = np.zeros((self.memory_capacity, self.field_shape))

        field_memory = np.asarray(field_memory, dtype=float)

        scores = self._compute_scores(query, field_memory, context).squeeze()
        retrieved = self._retrieve(combined_scores, field_memory).squeeze()

        # # MODIFIED EM2 OLD:
        # self._store(query, field_memory, context)
        # MODIFIED EM2 NEW:
        storage_condition = self.parameters.storage_condition._get(context)
        if storage_condition is None or storage_condition.is_satisfied(scheduler=context.composition.scheduler,
                                                                       context=context):
            self._store(query, field_memory, context)
        # MODIFIED EM2 END

        value = convert_all_elements_to_np_array([scores, retrieved])
        self.parameters.value._set(value, context=context)
        return value

    def _compute_scores(self, query, field_memory, context=None):
        normalize_memories = self.parameters.normalize_memories._get(context)
        # BREADCRUMB: THIS SHOULD USE EpisodicMemoryMechanism TO DETERMINE THE DISTANCE / SIMILARITY FUNCTION USED
        #             AND THAT SHOULD BE SET ON EMComposition2 CONSTRUCTOR, WITH ABILITY TO DO IT FIELD-WISE
        #             AND WARNINGS IF IT IS NOT DIFFERENTIABLE (E.G., USING ARGMAX)

        if normalize_memories:
            query_norm = np.linalg.norm(query)
            normalized_query = query / query_norm if query_norm != 0 else np.zeros_like(query)
            normalized_memory = _normalize_rows(field_memory)
            return normalized_memory @ normalized_query

        return field_memory @ query

    def _retrieve(self, combined_scores, field_memory):
        return combined_scores @ field_memory

    def _store(self, query, field_memory, context=None):
        storage_prob = float(self.parameters.storage_prob._get(context))
        decay_rate = self.parameters.decay_rate._get(context)
        random_state = self.parameters.random_state._get(context)

        if storage_prob <= 0:
            return

        if random_state.uniform(0, 1) >= storage_prob:
            return

        if decay_rate:
            field_memory *= decay_rate

        idx_of_weakest_memory = int(np.argmin(np.linalg.norm(field_memory, axis=1)))
        field_memory[idx_of_weakest_memory] = query

        # MODIFIED EM2 OLD:
        # self.parameters.field_memory._set(field_memory, context, override=True)
        # MODIFIED EM2 NEW:
        self.parameters.memory._set(field_memory, context, override=True)
        # MODIFIED EM2 END

    # @property
    # def memory(self):
    #     return self.parameters.field_memory.get(self.most_recent_context)


