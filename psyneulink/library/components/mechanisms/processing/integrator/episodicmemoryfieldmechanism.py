# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  EpisodicMemoryFieldMechanism ***********************************************


import copy
from psyneulink._typing import Callable, List, Literal, Optional, Union
from beartype import beartype
import numpy as np

from psyneulink.core.components.functions import Function_Base
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Distance
from psyneulink.core.components.functions.nonstateful.selectionfunctions import OneHot
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.globals.keywords import (
    MULTIPLICATIVE_PARAM, NAME, NEWEST, OLDEST, OVERWRITE, OWNER_VALUE, RANDOM, VARIABLE)
from psyneulink.core.globals.parameters import Parameter, FunctionParameter, check_user_specified
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, is_numeric_scalar
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemorymechanism import (
    EpisodicMemoryMechanism, EpisodicMemoryMechanismError)

__all__ = ['EpisodicMemoryFieldMechanism',
           'EpisodicMemoryFieldMechanismError',
           'QUERY', 'SCORES', 'COMBINED_SCORES', 'RETRIEVED',
           'DifferentiableContentAddressableMemory_FUNCTION']


DifferentiableContentAddressableMemory_FUNCTION = 'DifferentiableContentAddressableMemory Function'
QUERY = "QUERY"
SCORES = "SCORES"
COMBINED_SCORES = "COMBINED_SCORES"
RETRIEVED = "RETRIEVED"
DEFAULT_INPUT_PORT_NAME_PREFIX = 'FIELD_'
DEFAULT_INPUT_PORT_NAME_SUFFIX = '_INPUT'
DEFAULT_OUTPUT_PORT_PREFIX = 'RETRIEVED_'


class DifferentiableContentAddressableMemory(Function_Base): #
    """
    DifferentiableContentAddressableMemory(  \
        default_variable=None,               \
        initializer=None,                    \
        memory_capacity=None,                \
        decay_rate=None,                     \
        params=None,                         \
        owner=None,                          \
        prefs=None,                          \
        )

    Limited form of ContentAddressableMemory, for specific use by EMComposition2

    Use scores Parameter to compute retrieved value, which
      allows pre-assigned scores to be used for retrieval (e.g., to use COMBINED_SCORES in EMComposition2)
    Return scores based on current query (e.g., so it can be used to calculate COMBINED_SCORES in EMComposition2)

    IMPLEMENTATION NOTE:
      - scores/match-weights/distance vector is returned so it can be combined with other fields
      - external scores/match-weights/distance vector is needed to apply one that has been combined with other fields
      - API is same as ContentAddressableMemory_FUNCTION
          - use memory as alias for initializer parameter

    """
    componentName = DifferentiableContentAddressableMemory_FUNCTION

    class Parameters(Function_Base.Parameters):
        variable = Parameter([[0]], pnl_internal=True, constructor_argument='default_variable')
        memory = Parameter(None, pnl_internal=True, stateful=True)
        scores = Parameter([0], stateful=True)
        store = Parameter(False)
        decay_rate = Parameter(1.0, modulable=True)

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 memory=None,
                 decay_rate: Optional[Union[int, float, List, np.ndarray]]=None,
                 params:Optional[Union[List, np.ndarray]]=None,
                 owner=None,
                 prefs:Optional[ValidPrefSet] = None):

        # self._memory = []
        # self.field_width = initializer.shape[1]

        super().__init__(
            default_variable=default_variable,
            # initializer=initializer,
            memory=memory,
            decay_rate=decay_rate,
            # memory_capacity=memory_capacity,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    # def _initialize_previous_value(self, initializer, context=None):
    #     """Ensure that initializer is appropriate for assignment as memory attribute and assign as previous_value
    #     This must be done here rather than in validate_params as it is needed to initialize previous_value
    #     """
    #
    #     if initializer is None or convert_all_elements_to_np_array(initializer).size == 0:
    #         return None
    #
    #     initializer = np.asarray(initializer)
    #     assert initializer.ndim == 2
    #     assert initializer.shape[1] == self.memory_capacity
    #     self.field_width = initializer.shape[1]
    #
    #     # FIX: HOW DOES THIS RELATE TO WHAT IS DONE IN __init__()?
    #     self.parameters.previous_value.set(initializer, context, override=True)
    #
    #     previous_value = self._memory
    #     self.parameters.previous_value.set(previous_value, context, override=True)
    #     return previous_value

    def _function(self,
                 variable:Optional[Union[list, np.array]]=None,
                 context=None,
                 params=None,
                 ) -> (np.array, np.array):
        """Override to accept, use and return scores argument, and to store() when storage_condition is satisfied
        - Use scores Parameter (col) to generate weighted avg of entries in memory -> retrieved value
        - Compute dot product of query with entries (rows) -> scores
        - Call _store() to store query if store == True

        Return retrieved value, scores

        """
        # if self.is_initializing:
        #     assert len(variable[0]) == self.field_width, \
        #         (f"PROGRAM ERROR: 1st item of variable (query) for DifferentiableContentAddressableMemory Function of "
        #          f"'{self.owner.name}' should be len = {self.memory_field_width}; got len = {len(variable[0])}.")
        query = np.asarray(variable, dtype=float)
        scores = self._get_current_parameter_value('scores', context)

        # If this is an initialization run, just return query
        if self.is_initializing:
            return query, np.zeros(len(self.memory))

        # MODIFIED EM2 NEW:
        memory = copy.deepcopy(self.parameters.memory._get(context))
        if memory is None:
           memory = np.zeros((self.memory_capacity, self.field_width))

        memory = np.asarray(memory, dtype=float)

        retrieved = self._retrieve(scores, memory).squeeze()

        if self.parameters.store._get(context) == True:
            self._store(query, context)
        # MODIFIED EM2 END

        value = convert_all_elements_to_np_array([scores, retrieved])
        self.parameters.value._set(value, context=context)

        return retrieved, scores

    def _compute_scores(self, query, field_memory, context=None):
        normalize_memories = self.parameters.normalize_memories._get(context)
        # BREADCRUMB: THIS SHOULD USE EpisodicMemoryMechanism TO DETERMINE THE DISTANCE / SIMILARITY FUNCTION USED
        #             AND THAT SHOULD BE SET ON EMComposition2 CONSTRUCTOR, WITH ABILITY TO DO IT FIELD-WISE
        #             AND WARNINGS IF IT IS NOT DIFFERENTIABLE (E.G., USING ARGMAX)

        if normalize_memories:
            query_norm = np.linalg.norm(query)
            normalized_query = query / query_norm if query_norm != 0 else np.zeros_like(query)
            normalized_memory = self._normalize_rows(field_memory)
            return normalized_memory @ normalized_query

        return field_memory @ query

    def _retrieve(self, combined_scores, memory):
        return combined_scores @ memory

    def _store(self, query, context=None):
        decay_rate = self.parameters.decay_rate._get(context)
        memory = self.parameters.initializer._get(context)

        if decay_rate <= 1.0:
            memory *= decay_rate

        idx_of_weakest_memory = int(np.argmin(np.linalg.norm(memory, axis=1)))
        memory[idx_of_weakest_memory] = query

        self.parameters.initializer._set(memory, context, override=True)

    def _normalize_rows(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


class EpisodicMemoryFieldMechanismError(EpisodicMemoryMechanismError):
    pass


class EpisodicMemoryFieldMechanism(EpisodicMemoryMechanism):
    """
    EpisodicMemoryFieldMechanism

    A field-local EpisodicMemoryMechanism used by EMComposition2, that:
      - is restricted to use of DifferentiableContentAddressableMemory as it function
      - uses storage_condition to enforce that storage occurs after retrieval
      - has two InputPorts:
        - QUERY used to compute field-specific SCORES
        - COMBINED_SCORES used for retrieval from memory
      - has two OutputPorts:
        - SCORES reports field-specific match weights for QUERY against memory
        - RETRIEVED used to report COMBINED_SCORES-weighted retrieval from memory

    DifferentiableContentAddressableMemory:
      - supports only a single field
      - is used to store the Mechanisms' memory
      - returns scores for match of query to each entry in memory
      - takes **scores** argument (received on Mechanism's COMBINED_SCORES InputPort) as argument used for retrieval

    IMPLEMENTATION NOTE:  This is in distinction to the original EMComposition, in which each field's memory
                          was stored in Projection matrices managed by EMStorageMechanism.

    Ports
    -----
    input_port[QUERY]
        Current vector for this field.  Used for score computation and storage.

    output_port[SCORES]
        Match-weight vector between QUERY and every row in field_memory,
        passed to `ContentAddressableField` Function for retrieval

    input_port[COMBINED_SCORES]
        Combined retrieval weights, usually the softmax-normalized aggregate of
        all key-field SCORES.

    output_port[RETRIEVED]
        Dot product of COMBINED_SCORES with field_memory.

    """

    componentName = "EM_FIELD_MECHANISM"

    class Parameters(EpisodicMemoryMechanism.Parameters):

        # BREADCRUMB: MAKE SURE THESE ARE SHARED WITH THE FUNCTION (AS IS INTEGRATION RATE WITH TRANSFERMECHANISM)
        # Leave these as Parameters shared with Function so that they can be modulated
        variable = Parameter([[0,0]], pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(DifferentiableContentAddressableMemory, stateful=False, loggable=False)
        memory = FunctionParameter(None, function_parameter_name='initializer')
        decay_rate = Parameter(0.0, modulable=True, stateful=True)
        # Used by Mechanism._execute to ensure that storage occurs after retrieval
        storage_condition = Parameter(None, stateful=False, loggable=False)

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

        # MODIFIED EM2 NEW:
        # BREADCRUMB: MOVE THESE ALL TO EMComposition2.field AND PASS IN HERE TO **kwargs??
        # These are all used for construction of ContentAddressableMemory, and exposed as properties on Mechanism:
        normalize_memories: bool = True,
        decay_rate: Optional[Union[int, float, List, np.ndarray]]=None,  # -> rate on ContentAddressableMemory
        noise: Optional[Union[int, float, List, np.ndarray, Callable]]=None,
        distance_function:Optional[Union[Distance, Callable]]=None,
        selection_function:Optional[Union[OneHot, SoftMax, Callable]]=None,
        duplicate_entries_allowed:Optional[Union[str, bool, Literal[OVERWRITE]]]=None,
        duplicate_threshold:Optional[Union[int,float]]=None,
        equidistant_entries_select:Optional[Union[str, Literal[RANDOM, OLDEST, NEWEST]]]=None,
        seed:Optional[int]=None,
        # MODIFIED EM2 END

        params=None,
        name=None,
        prefs=None,
        **kwargs,
    ):
        self.field_shape = field_shape
        self.memory_capacity = len(field_memory)

        field_memory = np.asarray(field_memory, dtype=float)
        default_variable = np.array([
            np.zeros(field_shape),
            np.zeros(self.memory_capacity),
        ], dtype=object)

        function = DifferentiableContentAddressableMemory(default_variable=field_memory[0],
                                                          memory=field_memory,
                                                          decay_rate=decay_rate,
                                                          params=params,
                                                          owner=self,
                                                          prefs=prefs
                                                          )

        super().__init__(
            default_variable=default_variable,
            input_ports=[
                {NAME: QUERY, VARIABLE: np.zeros(field_shape)},
                {NAME: COMBINED_SCORES, VARIABLE: np.zeros(self.memory_capacity)},
            ],
            function=function,
            output_ports=[
                {NAME: RETRIEVED, VARIABLE: (OWNER_VALUE, 0),
                 NAME: SCORES, VARIABLE: (OWNER_VALUE, 1)},
            ],
            memory=field_memory,
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

    def _parse_function_variable(self, variable, context=None):
        # Pass only query to function; scores are accessed from Parameter
        return variable[0]

    def _execute(self, variable=None, context=None, runtime_params=None):
        variable = self._validate_variable(variable, context=context)

        # EM2 BREADCRUMB: ALTERNATIVE WOULD BE TO ASSIGN store AND memory TO runtime_params; MORE LLVM FRIENDLY?
        storage_condition = self.parameters.storage_condition._get(context)
        store = (storage_condition.is_satisfied(scheduler=context.composition.scheduler,
                                                       context=context)
                        if storage_condition is not None else False)
        self.function.parameters.store._set(store, context)

        return super()._execute(variable, context, runtime_params)

    # @property
    # def memory(self):
    #     return self.parameters.field_memory.get(self.most_recent_context)

