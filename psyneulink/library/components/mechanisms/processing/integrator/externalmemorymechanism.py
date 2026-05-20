# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# ****************************************  ExternalMemoryMechanism ***********************************************

"""
Subclass of EpisodicMemoryMechanism customized for EMComposition2.

It is a field-local that uses MatrixMemory as its function, which supports only a single field of memory.

It has QUERY and COMBINED_SCORES InputPorts
If it is constructed with FieldType.KEY:
  - it has RETRIEVED, SCORES and NORMS OutputPorts
  - SCORES and NORMS are used by combined_scores_node of emcomposition2
         to combine then for retrieval (SCORES) and storage (NORMS)
  - function computes match scores between query and each entry in memory, reported in its SCORES OutputPort
If it is constructed with FieldType.VALUE:
  - it has only RETRIEVED and NORMS OutputPorts
    - no SCORES OutputPort, since by definition those are not computed for VALUES
    - has NORMS, since the value is included in the NORMS calculation jused to determine where to store
  - it does not compute match scores, but does report retrieved value based on COMBINED_SCORES input

It has a storage_condition, assigned by emcomposition2, that is used to determine when to retrieve and when to store:
  a) if storage_condition is NOT satisfied:
     - _execute() is called with runtime_params[OPERATION: RETRIEVE]
  b) if storage_condition is satisfied:
     - _execute() is called with runtime_params[OPERATION: STORE]
     - output_ports are NOT updated; their values remain the ones assigned on the last retrieval
  - it is assumed that (a) always occurs before (b) in execution of emcompostion2

EM2 BREADCRUMB: GET DOCSTRING FROM COMBINATION OF EpisodicMemoryMechanism and EMStorageMechanism

IMPLEMENTATION NOTE:
    emcompositon2 uses one ExternalMemoryMechanism per memory field
    instead of using EMStorageMechanism to update MappingProjection matrices.

"""
from psyneulink._typing import Callable, List, Literal, Optional, Union
from beartype import beartype
import numpy as np

from psyneulink.core.components.functions import Function_Base
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Distance
from psyneulink.core.components.functions.nonstateful.selectionfunctions import OneHot
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.transformfunctions import MatrixMemory
from psyneulink.core.globals.keywords import (
    DEFAULT, DOT_PRODUCT, L0, MULTIPLICATIVE_PARAM, NAME, NEWEST,
    OLDEST, OPERATION, OVERWRITE, OWNER_VALUE, RANDOM, RETRIEVE, STORE, VARIABLE)
from psyneulink.core.globals.parameters import Parameter, FunctionParameter, check_user_specified
from psyneulink.core.globals.utilities import convert_all_elements_to_np_array, is_numeric_scalar
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemorymechanism import (
    EpisodicMemoryMechanism, EpisodicMemoryMechanismError)

__all__ = ['ExternalMemoryMechanism', 'ExternalMemoryMechanismError',
           'QUERY', 'SCORES', 'COMBINED_SCORES', 'COMBINED_NORMS', 'RETRIEVED']


DifferentiableContentAddressableMemory_FUNCTION = 'DifferentiableContentAddressableMemory Function'
QUERY = "QUERY"
SCORES = "SCORES"
NORMS = "NORMS"
COMBINED_SCORES = "COMBINED SCORES"
COMBINED_NORMS = "COMBINED NORMS"
RETRIEVED = "RETRIEVED"
DEFAULT_INPUT_PORT_NAME_PREFIX = 'FIELD_'
DEFAULT_INPUT_PORT_NAME_SUFFIX = '_INPUT'
DEFAULT_OUTPUT_PORT_PREFIX = 'RETRIEVED_'


# class DifferentiableContentAddressableMemory(Function_Base): #
#     """
#     DifferentiableContentAddressableMemory(        \
#         default_variable=None,                     \
#         initializer=None,                          \
#         memory_capacity=None,                      \
#         decay_rate=0,                              \
#         storage_prob=1.0,                          \
#         distance_function=Distance(metric=COSINE), \
#         params=None,                               \
#         owner=None,                                \
#         prefs=None,                                \
#         )
#
#     Limited form of ContentAddressableMemory, for specific use by EMComposition2
#
#     Use scores Parameter to compute retrieved value, which
#       allows pre-assigned scores to be used for retrieval (e.g., to use COMBINED_SCORES in EMComposition2)
#     Uses param[OPERATION] passed to _function() to determine whether that calls _retrieve() or _store()
#     Return value (retrieved of stored), scores and norms and  based on current query (e.g., so it can be used to
#     calculate
#     COMBINED_SCORES in
#     EMComposition2)
#
#     IMPLEMENTATION NOTE:
#       - scores/match-weights/distance vector is returned so it can be combined with other fields
#       - external scores/match-weights/distance vector is needed to apply one that has been combined with other fields
#       - API is same as ContentAddressableMemory_FUNCTION
#           - use memory as alias for initializer parameter
#
#     """
#     componentName = DifferentiableContentAddressableMemory_FUNCTION
#
#     class Parameters(Function_Base.Parameters):
#         variable = Parameter([[0]], pnl_internal=True, constructor_argument='default_variable')
#         memory = Parameter(None, pnl_internal=True, stateful=True)
#         scores = Parameter([0], stateful=True)
#         weakest_memory = Parameter(0, stateful=True)
#         store = Parameter(False)
#         normalize_memories = Parameter(True)
#         decay_rate = Parameter(1.0, modulable=True)
#         storage_prob = Parameter(1.0, modulable=True, stateful=True, aliases=[MULTIPLICATIVE_PARAM])
#         # distance_function = Parameter(Distance(metric=COSINE), stateful=False, loggable=False)
#         scoring_function = Parameter(MatrixTransform(operation=DOT_PRODUCT, normalize=True),
#                                      stateful=False, loggable=False)
#         random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
#         seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_value=DEFAULT, setter=_seed_setter)
#
#         def _validate_storage_prob(self, storage_prob):
#             storage_prob = float(storage_prob)
#             if not all_within_range(storage_prob, 0, 1):
#                 return f"must be a float in the interval [0,1]."
#
#     @check_user_specified
#     @beartype
#     def __init__(self,
#                  default_variable=None,
#                  memory=None,
#                  normalize_memories: bool = True,
#                  decay_rate: Optional[Union[int, float, List, np.ndarray]]=None,
#                  storage_prob: Optional[Union[int, float, np.ndarray]] = 1.0,
#                  distance_function:Optional[Union[Distance, Callable]]=None,
#                  params:Optional[Union[List, np.ndarray]]=None,
#                  owner=None,
#                  prefs:Optional[ValidPrefSet] = None):
#
#         super().__init__(
#             default_variable=default_variable,
#             memory=memory,
#             normalize_memories=normalize_memories,
#             decay_rate=decay_rate,
#             storage_prob=storage_prob,
#             distance_function=distance_function,
#             params=params,
#             owner=owner,
#             prefs=prefs,
#         )
#
#     def _parse_distance_function_variable(self, variable, context=None):
#         return convert_all_elements_to_np_array([variable, variable])
#
#     def _function(self,
#                  variable:Optional[Union[list, np.array]]=None,
#                  context=None,
#                  params=None,
#                  ) -> (np.array, np.array, np.array):
#         """Override to accept, use and return scores and norm arguments, and store() when storage_condition is satisfied
#         Use specification of OPERATION (STORE or RETRIEVE) in params to determine which to do
#         If RETRIEVE:
#             - Use scores Parameter (col) to generate weighted avg of entries in memory -> retrieved value
#             - Compute distance (dot product by default) between query each entry (row) -> scores
#             - Compute norm of each entry (row) -> norms
#         If STORE:
#             - Call _store_memory() to store query if store == True
#
#         Return retrieved value, scores
#
#         """
#         variable = np.asarray(variable, dtype=float)
#         operation_err_msg = (f"PROGRAM ERROR: 'operation'  was not specified in (runtime_)params "
#                              f"in call to DifferentiableContentAddressableMemory for '{self.owner.name}'.")
#         try:
#             operation = params[OPERATION]
#             assert operation in {STORE, RETRIEVE}
#         except:
#             if self.is_initializing:
#                 operation = RETRIEVE
#             else:
#                 raise ExternalMemoryMechanismError(operation_err_msg)
#
#         if operation == RETRIEVE:
#             retrieved, match_scores, norms = self._retrieve_memory(variable, context)
#             return retrieved, match_scores, norms
#
#         # Store memory in place of weakest one if condition is met and storage_prob > 0
#         elif operation == STORE:
#             self._store_memory(variable, context)
#             filler = np.zeros(len(self.parameters.memory._get(context)))
#             return variable, filler, filler # Return stored item and fillers for scores and norms
#
#         else:
#             raise ExternalMemoryMechanismError(operation_err_msg)
#
#     def _retrieve_memory(self, query, context):
#
#         # If this is an initialization run, just return query and zeros for score and norms
#         if self.is_initializing:
#             scores_template = norms_template = np.zeros(len(self.memory))
#             return query, scores_template, norms_template
#
#         memory = self.parameters.memory._get(context)
#         scores_for_retrieval = self.parameters.scores._get(context)
#         normalize_memories = self.parameters.normalize_memories._get(context)
#
#         # Retrieve memory weighted by scores_for_retrieval
#         retrieved = (scores_for_retrieval @ memory).squeeze()
#
#         # Compute match scores for query
#         # BREADCRUMB: THIS SHOULD USE EpisodicMemoryMechanism TO DETERMINE THE DISTANCE / SIMILARITY FUNCTION USED
#         #             AND THAT SHOULD BE SET ON EMComposition2 CONSTRUCTOR, WITH ABILITY TO DO IT FIELD-WISE
#         #             AND WARNINGS IF IT IS NOT DIFFERENTIABLE (E.G., USING ARGMAX)
#         if normalize_memories:
#             query_norm = np.linalg.norm(query)
#             normalized_query = query / query_norm if query_norm != 0 else np.zeros_like(query)
#             normalized_memory = self._normalize_rows(memory)
#             # EM2 BREADCRUMB: USE DistanceFunction here:
#             match_scores = normalized_memory @ normalized_query
#             match_scores = self.distance_function([normalized_memory, normalized_query])
#         else:
#             # EM2 BREADCRUMB: USE DistanceFunction here:
#             match_scores = memory @ query
#
#         # Compute norms for entries in memory (to determine weakest memory for storage)
#         norms = np.linalg.norm(memory, axis=1)
#         return retrieved, match_scores, norms
#
#     def _store_memory(self, item_to_store, context=None):
#         storage_prob = self._get_current_parameter_value('storage_prob', context)
#         random_state = self.parameters.random_state._get(context)
#         if random_state.uniform(0, 1) < storage_prob:
#             decay_rate = self.parameters.decay_rate._get(context)
#             memory = self.parameters.memory._get(context)
#             store_idx = int(self.parameters.weakest_memory._get(context))
#             if decay_rate >= 0.0:
#                 memory *= (1-decay_rate)
#             memory[store_idx] = item_to_store
#             self.parameters.memory._set(memory, context, override=True)
#
#     def _normalize_rows(self, matrix):
#         matrix = np.asarray(matrix, dtype=float)
#         norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#         return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


# class MatrixMemory(MatrixTransform): #
#     """
#     MatrixMemory(                                \
#         default_variable=None,                     \
#         initializer=None,                          \
#         memory_capacity=None,                      \
#         decay_rate=0,                              \
#         storage_prob=1.0,                          \
#         scores_operation,                          \
#         params=None,                               \
#         owner=None,                                \
#         prefs=None,                                \
#         )
#
#     Limited form of ContentAddressableMemory, for specific use by EMComposition2
#
#     Use scores Parameter to compute retrieved value, which
#       allows pre-assigned scores to be used for retrieval (e.g., to use COMBINED_SCORES in EMComposition2)
#     Uses param[OPERATION] passed to _function() to determine whether that calls _retrieve() or _store()
#     Return value (retrieved of stored), scores and norms and  based on current query (e.g., so it can be used to
#     calculate
#     COMBINED_SCORES in
#     EMComposition2)
#
#     IMPLEMENTATION NOTE:
#       - scores/match-weights/distance vector is returned so it can be combined with other fields
#       - external scores/match-weights/distance vector is needed to apply one that has been combined with other fields
#       - API is same as ContentAddressableMemory_FUNCTION
#           - use memory as alias for initializer parameter
#
#     """
#     componentName = MATRIX_MEMORY_FUNCTION
#
#     class Parameters(MatrixTransform.Parameters):
#         scores = Parameter([0], stateful=True)
#         weakest_memory = Parameter(0, stateful=True)
#         store = Parameter(False)
#         decay_rate = Parameter(1.0, modulable=True)
#         storage_prob = Parameter(1.0, modulable=True, stateful=True, aliases=[MULTIPLICATIVE_PARAM])
#         random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies='seed')
#         seed = Parameter(DEFAULT_SEED(), modulable=True, fallback_value=DEFAULT, setter=_seed_setter)
#
#         def _validate_storage_prob(self, storage_prob):
#             storage_prob = float(storage_prob)
#             if not all_within_range(storage_prob, 0, 1):
#                 return f"must be a float in the interval [0,1]."
#
#     @check_user_specified
#     @beartype
#     def __init__(self,
#                  default_variable=None,
#                  memory=None,
#                  normalize_memories: bool = True,
#                  decay_rate: Optional[Union[int, float, List, np.ndarray]]=None,
#                  storage_prob: Optional[Union[int, float, np.ndarray]] = None,
#                  scores_operation: Optional[Literal[L0, DOT_PRODUCT]] = None,
#                  params:Optional[Union[List, np.ndarray]]=None,
#                  owner=None,
#                  prefs:Optional[ValidPrefSet] = None):
#
#         super().__init__(
#             default_variable=default_variable,
#             matrix=memory,
#             operation=scores_operation,
#             normalize=normalize_memories,
#             decay_rate=decay_rate,
#             storage_prob=storage_prob,
#             params=params,
#             owner=owner,
#             prefs=prefs,
#         )
#
#     def _function(self,
#                  variable:Optional[Union[list, np.array]]=None,
#                  context=None,
#                  params=None,
#                  ) -> (np.array, np.array, np.array):
#         """Override to accept, use and return scores and norm arguments, and store() when storage_condition is satisfied
#         Use specification of OPERATION (STORE or RETRIEVE) in params to determine which to do
#         If RETRIEVE:
#             - Use scores Parameter (col) to generate weighted avg of entries in memory -> retrieved value
#             - Compute distance (dot product by default) between query each entry (row) -> scores
#             - Compute norm of each entry (row) -> norms
#         If STORE:
#             - Call _store_memory() to store query if store == True
#
#         Return retrieved value, scores
#
#         """
#         variable = np.asarray(variable, dtype=float)
#         operation_err_msg = (f"PROGRAM ERROR: 'operation'  was not specified in (runtime_)params "
#                              f"in call to MatrixMemory for '{self.owner.name}'.")
#         try:
#             operation = params[OPERATION]
#             assert operation in {STORE, RETRIEVE}
#         except:
#             if self.is_initializing:
#                 operation = RETRIEVE
#             else:
#                 raise ExternalMemoryMechanismError(operation_err_msg)
#
#         if operation == RETRIEVE:
#             retrieved, match_scores, norms = self._retrieve_memory(variable, context)
#             return retrieved, match_scores, norms
#
#         # Store memory in place of weakest one if condition is met and storage_prob > 0
#         elif operation == STORE:
#             self._store_memory(variable, context)
#             filler = np.zeros(len(self.parameters.memory._get(context)))
#             return variable, filler, filler # Return stored item and fillers for scores and norms
#
#         else:
#             raise ExternalMemoryMechanismError(operation_err_msg)
#
#     def _retrieve_memory(self, query, context):
#
#         # If this is an initialization run, just return query and zeros for score and norms
#         if self.is_initializing:
#             scores_template = norms_template = np.zeros(len(self.memory))
#             return query, scores_template, norms_template
#
#
#         # EM2 BREADCRUMB: CALL super()._function here to compute retrieved
#         #                 the computer and return match_scores and norms as per below
#
#         memory = self.parameters.matrix._get(context)
#         scores_for_retrieval = self.parameters.scores._get(context)
#         normalize_memories = self.parameters.normalize._get(context)
#
#         # Retrieve memory weighted by scores_for_retrieval
#         retrieved = self._function(query, context)
#
#         # Compute match scores for query
#         # BREADCRUMB: THIS SHOULD USE EpisodicMemoryMechanism TO DETERMINE THE DISTANCE / SIMILARITY FUNCTION USED
#         #             AND THAT SHOULD BE SET ON EMComposition2 CONSTRUCTOR, WITH ABILITY TO DO IT FIELD-WISE
#         #             AND WARNINGS IF IT IS NOT DIFFERENTIABLE (E.G., USING ARGMAX)
#         if normalize_memories:
#             query_norm = np.linalg.norm(query)
#             normalized_query = query / query_norm if query_norm != 0 else np.zeros_like(query)
#             normalized_memory = self._normalize_rows(memory)
#             # EM2 BREADCRUMB: USE DistanceFunction here:
#             match_scores = normalized_memory @ normalized_query
#         else:
#             # EM2 BREADCRUMB: USE DistanceFunction here:
#             match_scores = memory @ query
#
#         # Compute norms for entries in memory (to determine weakest memory for storage)
#         norms = np.linalg.norm(memory, axis=1)
#
#         return retrieved, match_scores, norms
#
#     def _store_memory(self, item_to_store, context=None):
#         storage_prob = self._get_current_parameter_value('storage_prob', context)
#         random_state = self.parameters.random_state._get(context)
#         if random_state.uniform(0, 1) < storage_prob:
#             decay_rate = self.parameters.decay_rate._get(context)
#             memory = self.parameters.memory._get(context)
#             store_idx = int(self.parameters.weakest_memory._get(context))
#             if decay_rate >= 0.0:
#                 memory *= (1-decay_rate)
#             memory[store_idx] = item_to_store
#             self.parameters.memory._set(memory, context, override=True)
#
#     def _normalize_rows(self, matrix):
#         matrix = np.asarray(matrix, dtype=float)
#         norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#         return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


class ExternalMemoryMechanismError(EpisodicMemoryMechanismError):
    pass


class ExternalMemoryMechanism(EpisodicMemoryMechanism):
    """
    ExternalMemoryMechanism

    EM2 BREADCRUMB: REVISE THE FOLLOWING TO BE CONSISTENT WITH UPDATES IN MODULE DOCSTRING

    A field-local EpisodicMemoryMechanism used by EMComposition2, that:
      - is restricted to use of MatrixMemory as it function
      - uses storage_condition to enforce that storage occurs after retrieval
      - has two InputPorts:
        - QUERY used to compute field-specific SCORES
        - COMBINED_SCORES used for retrieval from memory
      - has two OutputPorts:
        - SCORES reports field-specific match weights for QUERY against memory
        - RETRIEVED used to report COMBINED_SCORES-weighted retrieval from memory

    MatrixMemory:
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

    componentName = "EM_FIELD_MEMORY"

    class Parameters(EpisodicMemoryMechanism.Parameters):

        # BREADCRUMB: MAKE SURE THESE ARE SHARED WITH THE FUNCTION (AS IS INTEGRATION RATE WITH TRANSFERMECHANISM)
        # Leave these as Parameters shared with Function so that they can be modulated
        variable = Parameter([[0,0]], pnl_internal=True, constructor_argument='default_variable')
        # function = Parameter(DifferentiableContentAddressableMemory, stateful=False, loggable=False)
        function = Parameter(MatrixMemory, stateful=False, loggable=False)
        # distance_function = Parameter(Distance(metric=COSINE), stateful=False, loggable=False)
        memory = FunctionParameter(None, function_parameter_name='initializer')
        decay_rate = Parameter(0.0, modulable=True, stateful=True)
        storage_prob = FunctionParameter(1.0,
                                         function_name='function',
                                         function_parameter_name='storage_prob',
                                         primary=True,
                                         modulable=True,
                                         stateful=True)
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
        field_type,
        field_shape: int,
        field_memory: Union[list, np.ndarray],

        # MODIFIED EM2 NEW:
        # BREADCRUMB: MOVE THESE ALL TO EMComposition2.field AND PASS IN HERE TO **kwargs??
        # These are all used for construction of ContentAddressableMemory, and exposed as properties on Mechanism:
        decay_rate: Optional[Union[int, float, List, np.ndarray]]=None,  # -> rate on ContentAddressableMemory
        storage_prob: Optional[Union[int, float, np.ndarray]] = 1.0,
        normalize_memories: bool = True,
        scores_operation: Optional[Literal[L0, DOT_PRODUCT]]=None,
        seed:Optional[int]=None,
        # MODIFIED EM2 END

        params=None,
        name=None,
        prefs=None,
        **kwargs,
    ):

        from psyneulink.library.compositions.emcomposition.emcomposition2 import FieldType
        assert isinstance(field_type, FieldType), \
            (f"PROGRAM ERROR: ExternalMemoryMechanism requires specification of field_type "
             f"as FieldType.KEY or FieldType.VALUE; got {field_type}.")
        self.field_type = field_type
        if field_type is FieldType.VALUE:
            self.value_input_specified = True

        self.field_shape = field_shape
        self.memory_capacity = len(field_memory)

        field_memory = np.asarray(field_memory, dtype=float)
        default_variable = np.array([
            np.zeros(field_shape),
            np.zeros(self.memory_capacity),
            np.zeros(1),
        ], dtype=object)

        # function = DifferentiableContentAddressableMemory(default_variable=field_memory[0],
        #                                                   memory=field_memory,
        #                                                   decay_rate=decay_rate,
        #                                                   scoring_functon=scoring_function,
        #                                                   storage_prob=storage_prob,
        #                                                   params=params,
        #                                                   owner=self,
        #                                                   prefs=prefs
        #                                                   )
        function = MatrixMemory(default_variable=field_memory[0],
                                memory=field_memory,
                                normalize_memories=normalize_memories,
                                scores_operation=scores_operation,
                                decay_rate=decay_rate,
                                storage_prob=storage_prob,
                                params=params,
                                owner=self,
                                prefs=prefs
                                )

        # EM2 BREADCRUMB: MOVE THESE BACK INTO _instantiate_<input/output>_ports():
        input_ports = [{NAME: QUERY, VARIABLE: np.zeros(field_shape)},
                       {NAME: COMBINED_SCORES, VARIABLE: np.zeros(self.memory_capacity)},
                       {NAME: COMBINED_NORMS, VARIABLE: np.zeros(1)},]

        output_ports = [{NAME: RETRIEVED, VARIABLE: (OWNER_VALUE, 0)}]
        output_ports.append({NAME: NORMS, VARIABLE: (OWNER_VALUE, 2)})
        if field_type == FieldType.KEY:
            output_ports.insert(1, {NAME: SCORES, VARIABLE: (OWNER_VALUE, 1)})

        super().__init__(
            default_variable=default_variable,
            # EM2 BREADCRUMB: REMOVE THESE ONCE MOVED PER ABOVE:
            input_ports=input_ports,
            output_ports=output_ports,
            # EM2 BREADCRUMB END
            function=function,
            # distance_function=distance_function,
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
        # input_ports = [
        #     {NAME: QUERY, VARIABLE: np.zeros(self.field_shape)},
        #     {NAME: COMBINED_SCORES, VARIABLE: np.zeros(self.memory_capacity)},
        # ]
        super(EpisodicMemoryMechanism, self)._instantiate_input_ports(input_ports=self.input_ports, context=context)

    # def _instantiate_output_ports(self, context=None):
    #     # output_ports = [
    #     #     {NAME: SCORES, VARIABLE: (OWNER_VALUE, 0)},
    #     #     {NAME: RETRIEVED, VARIABLE: (OWNER_VALUE, 1)},
    #     # ]
    #     # self.parameters.output_ports._set(output_ports, override=True, context=context)
    #     super()._instantiate_output_ports(context=context)
    #     #
    #     for output_port in self.output_ports:
    #         output_port.parameters.require_projection_in_composition._set(False, override=True, context=context)

    def _validate_variable(self, variable, context=None):
        variable = np.asarray(variable, dtype=object)
        assert len(variable) == 3, (f"Variable for {self.name} must contain three items: "
                                    f"QUERY, COMBINED_SCORES and COMBINED_NORMS.")
        assert len(variable[0]) == self.field_shape, (f"QUERY input for {self.name} has length {len(variable[0])}; "
                                                      f"expected {self.field_shape}.")
        assert len(variable[1]) == self.memory_capacity,(f"COMBINED_SCORES input for {self.name} has length "
                                                         f"{len(variable[1])}; expected {self.memory_capacity}.")
        assert len(variable[2]) == 1, (f"COMBINED_NORMS input for {self.name} has length "
                                       f"{len(variable[2])}; expected 1.")
        return variable

    def _parse_function_variable(self, variable, context=None):
        # Pass only query to function; scores are accessed from Parameter
        return variable[0]

    def _execute(self, variable=None, context=None, runtime_params=None):
        variable = self._validate_variable(variable, context=context)
        scores = variable[1]
        weakest_memory = variable[2]

        storage_condition = self.parameters.storage_condition._get(context)
        self.store = (storage_condition.is_satisfied(scheduler=context.composition.scheduler, context=context)
                      if storage_condition is not None else False)
        runtime_params = {} if runtime_params is None else runtime_params
        if self.store:
            from psyneulink.library.compositions.emcomposition.emcomposition2 import FieldType
            if self.field_type == FieldType.VALUE and not self.value_input_specified:
                # Use last retrieved value as value to store
                # IMPLEMENTATION NOTE: this assumes that retrieval is always executed before storage
                variable = self.output_ports[RETRIEVED].parameters.value._get(context)
            runtime_params.update({OPERATION: STORE})
        else:
            runtime_params.update({OPERATION: RETRIEVE})

        # EM2 BREADCRUMB: MAKE THESE FUNCTION PARAMETERS (LIKE storage_prob) ONCE THAT IS SUPPORTED FOR PYTORCH
        self.function.parameters.scores._set(scores, context)
        self.function.parameters.weakest_memory._set(weakest_memory, context)

        # return super()._execute(variable, context, runtime_params)
        return super()._execute(variable, context, runtime_params)

    def _update_output_ports(self, runtime_output_port_params, context):
        """Override to suppress updating of OutputPorts on STORE; should only ever occur on RETRIEVE."""
        if self.store:
            return
        super()._update_output_ports(runtime_output_port_params, context)


    # @property
    # def memory(self):
    #     return self.parameters.field_memory.get(self.most_recent_context)

