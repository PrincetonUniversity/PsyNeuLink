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

It has an access_condition, assigned by emcomposition2, that is used to determine when to retrieve and when to store:
  a) if access_condition is NOT satisfied:
     - _execute() is called with runtime_params[OPERATION: COMPUTE_SCORES]
  b) if access_condition is satisfied:
     - _execute() is called with runtime_params[OPERATION: ACCESS_MEMORY]
  - it is assumed that (a) always occurs before (b) in execution of emcompostion2

EM2 BREADCRUMB: GET DOCSTRING FROM COMBINATION OF EpisodicMemoryMechanism and EMStorageMechanism

IMPLEMENTATION NOTE:
    emcompositon2 uses one ExternalMemoryMechanism per memory field
    instead of using EMStorageMechanism to update MappingProjection matrices.

"""
from psyneulink._typing import Callable, List, Literal, Optional, Union
from beartype import beartype
import numpy as np

from psyneulink.core.components.functions.nonstateful.transformfunctions import (
    MatrixMemory, ACCESS_MEMORY, COMPUTE_SCORES)
from psyneulink.core.globals.keywords import DOT_PRODUCT, L0, NAME, OPERATION, OWNER_VALUE, VARIABLE
from psyneulink.core.globals.parameters import Parameter, FunctionParameter, check_user_specified
from psyneulink.core.globals.utilities import is_numeric_scalar
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


class ExternalMemoryMechanismError(EpisodicMemoryMechanismError):
    pass


class ExternalMemoryMechanism(EpisodicMemoryMechanism):
    """
    ExternalMemoryMechanism

    EM2 BREADCRUMB: REVISE THE FOLLOWING TO BE CONSISTENT WITH UPDATES IN MODULE DOCSTRING

    A field-local EpisodicMemoryMechanism used by EMComposition2, that:
      - is restricted to use of MatrixMemory as it function
      - uses access_condition to enforce that storage occurs after retrieval
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

        variable = Parameter([[0,0]], pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(MatrixMemory, stateful=False, loggable=False)
        # EM2 BREADCRUMB: MAKE THESE SHARED PARAMETERS WITH function
        # memory = FunctionParameter(None, function_parameter_name='initializer')
        # scores_metric=scores_metric,
        # normalize_memories: bool = True,
        storage_prob = FunctionParameter(1.0,
                                         function_name='function',
                                         function_parameter_name='storage_prob',
                                         primary=True,
                                         modulable=True,
                                         stateful=True)
        decay_rate = Parameter(0.0, modulable=True, stateful=True)
        # Used by Mechanism._execute to ensure that storage occurs after retrieval
        access_condition = Parameter(None, stateful=False, loggable=False)

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
        decay_rate: Optional[Union[int, float, List, np.ndarray]]=None,  # -> rate on ContentAddressableMemory
        storage_prob: Optional[Union[int, float, np.ndarray]] = 1.0,
        normalize_memories: bool = True,
        scores_metric: Optional[Literal[L0, DOT_PRODUCT]]=None,
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

        function = MatrixMemory(default_variable=default_variable,
                                memory=field_memory,
                                normalize_memories=normalize_memories,
                                scores_metric=scores_metric,
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

    def _execute(self, variable=None, context=None, runtime_params=None):
        variable = self._validate_variable(variable, context=context)
        # scores = variable[1]
        # weakest_memory = variable[2]

        access_condition = self.parameters.access_condition._get(context)
        self.access = (access_condition.is_satisfied(scheduler=context.composition.scheduler, context=context)
                      if access_condition is not None else False)
        runtime_params = {} if runtime_params is None else runtime_params
        if self.access:
            from psyneulink.library.compositions.emcomposition.emcomposition2 import FieldType
            runtime_params.update({OPERATION: ACCESS_MEMORY})
        else:
            runtime_params.update({OPERATION: COMPUTE_SCORES})

        # EM2 BREADCRUMB: MAKE THESE FUNCTION PARAMETERS (LIKE storage_prob) ONCE THAT IS SUPPORTED FOR PYTORCH
        # self.function.parameters.scores._set(scores, context)
        # self.function.parameters.weakest_memory._set(weakest_memory, context)

        return super()._execute(variable, context, runtime_params)
