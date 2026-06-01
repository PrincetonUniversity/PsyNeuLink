# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition2 *************************************************

"""
Refactored EMComposition prototype.

This module introduces ExternalMemoryMechanism, a field-local episodic memory mechanism
that owns the memory matrix for a single field. For the moment, ExternalMemoryMechanism uses
only Matrix as its Function (defined in the ExternalMemoryMechanism module) that is limited to a single field in
memory, that uses its _compute_scores() method to determine scores for each
entry in memory, and an _access_memory method that retrieves the memory based on the combined scores over all fields,
and then stores the query input into memory with a probability specified by storage_prob (True or False) when
access_condition is satisfied

The refactored EMComposition uses one ExternalMemoryMechanism per memory field instead of using EMStorageMechanism
to update MappingProjection matrices.

- memory_decay_rate is applied as 1-memory_decay_rate multiplier (retention factor) to memory
- If a value is not provided as input to KEY Field, then the retrieved value is stored;
   need to deal with nested emcomposition2 in that case:
   - does it automatically get a default input from the input_CIM?
   - could it be detected structurally by no afferent input to the relevant input_CIM port?

High-level execution per field:

1. QUERY input is sent to ExternalMemoryMechanism.input_port[QUERY].
2. ExternalMemoryMechanism computes a match-weight vector over its memory rows and emits SCORES.
3. SCORES from key fields are weighted, combined and softmax-normalized by RETRIEVE.
4. The normalized combined scores are sent back to each ExternalMemoryMechanism.input_port[COMBINED_SCORES].
5. Each ExternalMemoryMechanism retrieves its field value and emits RETRIEVED.
6. Each ExternalMemoryMechanism stores its QUERY input into its own memory matrix when access_condition is True
"""

import copy
import warnings
from typing import Optional, Union

import numpy as np
import torch

import psyneulink.core.scheduling.condition as conditions

from psyneulink.core.components.functions.function import DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.transformfunctions import Concatenate, LinearCombination, MatrixTransform
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import GatingMechanism
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import CompositionError, NodeRole
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import (
    ADAPTIVE,
    ALL,
    ARG_MAX,
    ARG_MAX_INDICATOR,
    AUTO,
    CONTEXT,
    CONTROL,
    DEFAULT_INPUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VARIABLE,
    DOT_PRODUCT,
    EM_COMPOSITION,
    FIRST,
    FULL_CONNECTIVITY_MATRIX,
    GAIN,
    IDENTITY_MATRIX,
    INPUT_SHAPES,
    LAST,
    L0,
    MULTIPLICATIVE_PARAM,
    NAME,
    OWNER_VALUE,
    PARAMS,
    PROB_INDICATOR,
    PRODUCT,
    PROJECTIONS,
    RANDOM,
    VARIABLE,
)
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.utilities import (
    ContentAddressableList,
    convert_all_elements_to_np_array,
    is_iterable,
    is_numeric_scalar,
)
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.scheduling.condition import AfterNodes, All, Always, Any, BeforeNCalls, AfterNCalls
from psyneulink.core.llvm import ExecutionMode
from psyneulink.library.components.mechanisms.processing.integrator.externalmemorymechanism import (
    ExternalMemoryMechanism, NORMS, QUERY, SCORES, RETRIEVED, COMBINED_SCORES, COMBINED_NORMS)
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available


__all__ = [
    "EMComposition2",
    "EMComposition2Error",
    "FieldType",
    "KEY",
    "FIELD_MEMORY",
    "FIELD_WEIGHT",
    'CONCATENATE_QUERIES_NAME',
    "LEARN_FIELD_WEIGHT",
    "PROBABILISTIC",
    "TARGET_FIELD",
    "WEIGHTED_AVG",
    "WEIGHTED_SCORES"
]


KEY = "key"

# softmax_choice options:
STORAGE_PROB = "storage_prob"
WEIGHTED_AVG = ALL
PROBABILISTIC = PROB_INDICATOR

# specs for entry of fields specification dict
FIELD_WEIGHT = "field_weight"
LEARN_FIELD_WEIGHT = "learn_field_weight"
TARGET_FIELD = "target_field"

# Node names
QUERY_NODE_NAME = "QUERY"
QUERY_AFFIX = f" [{QUERY_NODE_NAME}]"
VALUE_NODE_NAME = "VALUE"
VALUE_AFFIX = f" [{VALUE_NODE_NAME}]"
FIELD_MEMORY = "FIELD_MEMORY"
FIELD_MEMORY_AFFIX = f" [{FIELD_MEMORY}]"
MATCH = "MATCH"
MATCH_AFFIX = f" [{MATCH}]"
WEIGHT = "WEIGHT"
WEIGHT_AFFIX = f" [{WEIGHT}]"
WEIGHTED_SCORES = "WEIGHTED SCORE"
WEIGHTED_SCORES_NODE_NAME = "WEIGHTED SCORES"
WEIGHTED_SCORES_AFFIX = f" [{WEIGHTED_SCORES_NODE_NAME}]"
CONCATENATE_QUERIES_NAME = "CONCATENATE QUERIES"
COMBINED_SCORES_NODE_NAME = "COMBINED SCORES"
RETRIEVED_NODE_NAME = "RETRIEVED"
RETRIEVED_AFFIX = " [RETRIEVED]"


def _memory_getter(owning_component=None, context=None):
    """Return EMComposition2 memory as a 3d object array: entries x fields x field_values.
    These are derived from the memory attribute of the field_memory_node of each field.
    """
    if owning_component is None or owning_component.is_initializing:
        return None

    field_memories = [
        np.asarray(field.memory_node.function.parameters.memory.get(context))
        for field in owning_component.fields
    ]

    memory_capacity = owning_component.memory_capacity or owning_component.defaults.memory_capacity
    return convert_all_elements_to_np_array([
        [field_memories[field_idx][entry_idx] for field_idx in range(owning_component.num_fields)]
        for entry_idx in range(memory_capacity)
    ])

def field_weights_setter(field_weights, owning_component=None, context=None):
    if (
        owning_component is None
        or not owning_component.parameters.field_weights._has_value(context)
        or owning_component.parameters.field_weights._get(context) is None
    ):
        return field_weights

    if len(field_weights) != len(owning_component.field_weights):
        raise EMComposition2Error(
            f"The number of field_weights ({len(field_weights)}) must match "
            f"the number of fields ({len(owning_component.field_weights)})."
        )

    field_weights = list(field_weights)
    for i, fw in enumerate(field_weights.copy()):
        field_weights[i] = None if fw is None else fw

    if owning_component.normalize_field_weights:
        denominator = np.sum([fw if fw is not None else 0 for fw in field_weights]) or 1
        field_weights = [fw / denominator if fw is not None else None for fw in field_weights]

    field_wt_node_idx = 0
    for i, field_weight in enumerate(field_weights):
        if owning_component.parameters.field_weights.default_value[i] is None:
            if field_weight:
                raise EMComposition2Error(
                    f"Field '{owning_component.field_names[i]}' of '{owning_component.name}' was originally assigned "
                    f"as a value node (i.e., with a field_weight = None); this cannot be changed after construction. "
                    f"If you want to change it to a key field, you must re-construct the EMComposition2 using a scalar "
                    f"for its field in the `field_weights` arg (which can be 0).")

            continue
        owning_component.field_weight_nodes[field_wt_node_idx].input_port.defaults.variable = field_weight
        owning_component.fields[i].weight = field_weight
        field_wt_node_idx += 1

    return np.array(field_weights, dtype=object)


def get_softmax_gain(v, scale=1, base=1, entropy_weighting=.1) -> float:
    v = np.squeeze(v)
    # # MODIFIED EM2 OLD:
    # gain = scale * (base +
    #                 (entropy_weighting *
    #                  np.log(
    #                      -1 * np.sum((1 / (1 + np.exp(-1 * v))) * np.log(1 / (1 + np.exp(-1 * v)))))))
    # return gain
    # MODIFIED EM2 NEW:
    logistic = 1 / (1 + np.exp(-1 * v))
    entropy = -1 * np.sum(logistic * np.log(logistic))
    return scale * (base + entropy_weighting * np.log(entropy))
    # MODIFIED EM2 END


from psyneulink.library.compositions.emcomposition2.emcomposition2 import FieldType


class Field:
    """Object that contains information about a field in an EMComposition2's memory."""

    name = None

    def __init__(
        self,
        name: str = None,
        index: int = None,
        type: FieldType = None,
        weight: float = None,
        learn_weight: bool = None,
        learning_rate: float = None,
        target: bool = None,
    ):
        self.name = name
        self.index = index
        self.type = type
        self.weight = weight
        self.learn_weight = learn_weight
        self.learning_rate = learning_rate
        self.target = target

        self.input_node = None
        self.memory_node = None
        self.weight_node = None
        self.weighted_scores_node = None
        self.retrieved_node = None

        self.query_projection = None
        self.concatenation_projection = None
        self.scores_projection = None
        self.norms_projection = None
        self.combined_scores_projection = None
        self.combined_norms_projection = None
        self.retrieved_projection = None
        self.weight_projection = None
        self.weighted_scores_projection = None
        self.weighted_norms_projection = None

        self.missing_value = False


    @property
    def nodes(self):
        return [
            node for node in [
                self.input_node,
                self.memory_node,
                self.weight_node,
                self.weighted_scores_node,
                self.retrieved_node,
            ]
            if node is not None
        ]

    @property
    def projections(self):
        return [
            proj for proj in [
                self.query_projection,
                self.concatenation_projection,
                self.scores_projection,
                self.norms_projection,
                self.combined_scores_projection,
                self.combined_norms_projection,
                self.retrieved_projection,
                self.weight_projection,
                self.weighted_scores_projection,
                self.weighted_norms_projection,
            ]
            if proj is not None
        ]

    @property
    def query(self):
        return self.input_node.variable

    @property
    def match(self):
        return self.memory_node.output_ports[SCORES].value

    @property
    def retrieved_memory(self):
        return self.memory_node.output_ports[RETRIEVED].value

    @property
    def memory(self):
        return self.memory_node.memory

    @property
    def memories(self):
        return self.memory_node.function.parameters.memory.get(None)


class EMComposition2Error(CompositionError):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EMComposition2(AutodiffComposition):
    """
        EMComposition(                      \
        memory_template=[[0],[0]],      \
        memory_fill=0,                  \
        memory_capacity=None,           \
        fields=None,                    \
        field_names=None,               \
        field_weights=None,             \
        learn_field_weights=False,      \
        learning_rate=True,             \
        normalize_field_weights=True,   \
        concatenate_queries=False,      \
        normalize_memories=True,        \
        softmax_gain=THRESHOLD,         \
        storage_prob=1.0,               \
        store_on_optimization=FIRST,    \
        memory_decay_rate=AUTO,         \
        enable_learning=True,           \
        target_fields=None,             \
        use_gating_for_weighting=False, \
        name="EM_Composition"           \
        )

    Refactored EMComposition.

    This version replaces:
      - match_nodes backed by memory Projection matrices
      - retrieved_nodes backed by memory Projection matrices
      - EMStorageMechanism

    with:
      - one ExternalMemoryMechanism per field, each owning its field memory matrix.
      - storage occurs in each memory_node based on access_condition an its storage_prob

    The externally visible structure is kept similar to the original EMComposition:
      - input_nodes
      - query_input_nodes
      - value_input_nodes
      - field_weight_nodes
      - weighted_scores_nodes
      - combined_scores_node
      - retrieved_nodes

    Internally, field.memory_node is now the memory owner for each field.
    """

    componentCategory = EM_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.emcomposition2.pytorchEMwrappers2 import (
            PytorchEMCompositionWrapper2, PytorchExternalMemoryMechanismWrapper,
        )
        pytorch_composition_wrapper_type = PytorchEMCompositionWrapper2
        pytorch_mechanism_wrapper_type = PytorchExternalMemoryMechanismWrapper

    class Parameters(AutodiffComposition.Parameters):
        """
            Attributes
            ----------

                concatenate_queries
                    see `concatenate_queries <EMComposition.concatenate_queries>`

                    :default value: False
                    :type: ``bool``

                field_names
                    see `field_names <EMComposition.field_names>`

                    :default value: None
                    :type: ``list``

                field_weights
                    see `field_weights <EMComposition.field_weights>`

                    :default value: None
                    :type: ``numpy.ndarray``

                learn_field_weights
                    see `learn_field_weights <EMComposition.learn_field_weights>`

                    :default value: True
                    :type: ``numpy.ndarray``

                learning_rate
                    see `learning_results <EMComposition.learning_rate>`

                    :default value: []
                    :type: ``list``

                memory
                    see `memory <EMComposition.memory>`

                    :default value: None
                    :type: ``numpy.ndarray``

                memory_capacity
                    see `memory_capacity <EMComposition.memory_capacity>`

                    :default value: 1000
                    :type: ``int``

                memory_decay_rate
                    see `memory_decay_rate <EMComposition.memory_decay_rate>`

                    :default value: 0.001
                    :type: ``float``

                memory_template
                    see `memory_template <EMComposition.memory_template>`

                    :default value: np.array([[0],[0]])
                    :type: ``np.ndarray``

                normalize_field_weights
                    see `normalize_field_weights <EMComposition.normalize_field_weights>`

                    :default value: True
                    :type: ``bool``

                normalize_memories
                    see `normalize_memories <EMComposition.normalize_memories>`

                    :default value: True
                    :type: ``bool``

                purge_by_field_weights
                    see `purge_by_field_weights <EMComposition.purge_by_field_weights>`

                    :default value: False
                    :type: ``bool``

                random_state
                    see `random_state <NormalDist.random_state>`

                    :default value: None
                    :type: ``numpy.random.RandomState``

                softmax_gain
                    see `softmax_gain <EMComposition.softmax_gain>`
                    :default value: 1.0
                    :type: ``float, ADAPTIVE or CONTROL``

                softmax_choice
                    see `softmax_choice <EMComposition.softmax_choice>`
                    :default value: WEIGHTED_AVG
                    :type: ``keyword``

                softmax_threshold
                    see `softmax_threshold <EMComposition.softmax_threshold>`
                    :default value: .001
                    :type: ``float``

                storage_prob
                    see `storage_prob <EMComposition.storage_prob>`

                    :default value: 1.0
                    :type: ``float``

                store_on_optimization
                    see `store_on_optimization <EMComposition.store_on_optimization>`

                    :default value: FIRST
                    :type: ``str``
        """
        memory = Parameter(None, loggable=True, getter=_memory_getter, read_only=True)
        memory_template = Parameter([[0], [0]], structural=True, valid_types=(tuple, list, np.ndarray), read_only=True)
        memory_capacity = Parameter(1000, structural=True)
        field_names = Parameter(None, structural=True)
        field_weights = Parameter([1], setter=field_weights_setter)
        learn_field_weights = Parameter(False, structural=True)
        normalize_field_weights = Parameter(True)
        concatenate_queries = Parameter(False, structural=True)
        normalize_memories = Parameter(True)
        softmax_gain = Parameter(1.0, modulable=True)
        softmax_threshold = Parameter(.001, modulable=True, specify_none=True)
        softmax_choice = Parameter(WEIGHTED_AVG, modulable=False, specify_none=True)
        storage_prob = Parameter(1.0, modulable=True)
        store_on_optimization = Parameter(FIRST)
        memory_decay_rate = Parameter(AUTO, modulable=True)
        purge_by_field_weights = Parameter(False, structural=True)
        target_fields = Parameter(None, read_only=True, structural=True)
        random_state = Parameter(None, loggable=False, getter=_random_state_getter, dependencies="seed")
        seed = Parameter(DEFAULT_SEED(), modulable=True, setter=_seed_setter)

        def _validate_memory_template(self, memory_template):
            if isinstance(memory_template, tuple):
                if len(memory_template) not in {2, 3}:
                    return "must be length either 2 or 3 if it is a tuple."
                if not all(isinstance(item, int) for item in memory_template):
                    return "must have only integers as entries."
            elif isinstance(memory_template, (list, np.ndarray)):
                memory_template = np.array(memory_template, dtype=object)
                if memory_template.ndim not in {1, 2, 3}:
                    return "must be either 1d, 2d, or 3d."
            else:
                return "must be tuple, list, or array."

        def _validate_field_weights(self, field_weights):
            if field_weights is not None:
                if not np.atleast_1d(field_weights).ndim == 1:
                    return "must be a scalar, list of scalars, or 1d array."
                if any([field_weight < 0 for field_weight in field_weights if field_weight is not None]):
                    return "must all be positive values."

        def _validate_learn_field_weights(self, learn_field_weights):
            if isinstance(learn_field_weights, (list, np.ndarray)):
                if not all(isinstance(item, (bool, int, float, type(None))) for item in learn_field_weights):
                    return "can only contain bools, ints, floats, or None."
            elif not isinstance(learn_field_weights, bool):
                return "must be a bool or list of bools, ints, floats, or None."

        def _validate_memory_decay_rate(self, memory_decay_rate):
            if memory_decay_rate is None or memory_decay_rate == AUTO:
                return
            if not is_numeric_scalar(memory_decay_rate) or not 0 <= memory_decay_rate <= 1:
                return "must be a float in the interval [0, 1]."

        def _validate_softmax_gain(self, softmax_gain):
            if not is_numeric_scalar(softmax_gain) and softmax_gain not in {ADAPTIVE, CONTROL}:
                return f"must be a scalar or one of '{ADAPTIVE}' or '{CONTROL}'."

        def _validate_softmax_threshold(self, softmax_threshold):
            if softmax_threshold is not None and (not is_numeric_scalar(softmax_threshold) or softmax_threshold <= 0):
                return "must be a scalar greater than 0."

        def _validate_storage_prob(self, storage_prob):
            if not is_numeric_scalar(storage_prob) or not 0 <= storage_prob <= 1:
                return "must be a float in the interval [0, 1]."

        def _validate_store_on_optimization(self, option):
            if option not in {FIRST, LAST, ALL}:
                return "must be one of FIRST, LAST, or ALL."

    @check_user_specified
    def __init__(
        self,
        memory_template: Union[tuple, list, np.ndarray] = [[0], [0]],
        memory_capacity: Optional[int] = None,
        memory_fill: Union[int, float, tuple, RANDOM] = 0,
        fields: Optional[dict] = None,
        field_names: Optional[list] = None,
        field_weights: Union[int, float, list, tuple] = None,
        learn_field_weights: Union[bool, list, tuple] = None,
        learning_rate: Union[float, bool, int, dict] = None,
        normalize_field_weights: bool = True,
        concatenate_queries: bool = False,
        normalize_memories: bool = True,
        softmax_gain: Union[float, ADAPTIVE, CONTROL] = 1.0,
        softmax_threshold: Optional[float] = .001,
        softmax_choice: Optional[Union[WEIGHTED_AVG, ARG_MAX, PROBABILISTIC]] = WEIGHTED_AVG,
        storage_prob: float = 1.0,
        store_on_optimization: Union[FIRST, LAST, ALL] = FIRST,
        memory_decay_rate: Union[float, AUTO] = AUTO,
        purge_by_field_weights: bool = False,
        enable_learning: bool = True,
        target_fields: Optional[Union[list, tuple, np.ndarray]] = None,
        use_gating_for_weighting: bool = False,
        random_state=None,
        seed=None,
        name="EM_Composition",
        **kwargs,
    ):
        memory_fill = memory_fill or 0

        self._validate_memory_specs(
            memory_template,
            memory_capacity,
            memory_fill,
            field_weights,
            field_names,
            name,
            learn_field_weights,
        )

        self._enable_learning_warning_flag = False
        self._use_gating_for_weighting = use_gating_for_weighting

        memory_template, memory_capacity = self._parse_memory_template(memory_template,
                                                                       memory_capacity,
                                                                       memory_fill)

        self.fields = ContentAddressableList(component_type=Field)
        self.entry_template = memory_template[0]
        self.concatenate_queries_node = None

        (field_names,
         field_weights,
         learn_field_weights,
         target_fields,
         concatenate_queries,
         ) = self._parse_fields(fields,
                                field_names,
                                field_weights,
                                learn_field_weights,
                                learning_rate,
                                normalize_field_weights,
                                concatenate_queries,
                                normalize_memories,
                                target_fields,
                                name)

        if softmax_gain == CONTROL:
            self.parameters.softmax_gain.modulable = False

        super().__init__(
            name=name,
            memory_template=memory_template,
            memory_capacity=memory_capacity,
            field_names=field_names,
            field_weights=field_weights,
            learn_field_weights=learn_field_weights,
            learning_rate=learning_rate,
            normalize_field_weights=normalize_field_weights,
            concatenate_queries=concatenate_queries,
            normalize_memories=normalize_memories,
            softmax_gain=softmax_gain,
            softmax_threshold=softmax_threshold,
            softmax_choice=softmax_choice,
            storage_prob=storage_prob,
            store_on_optimization=store_on_optimization,
            memory_decay_rate=memory_decay_rate,
            purge_by_field_weights=purge_by_field_weights,
            enable_learning=enable_learning,
            target_fields=target_fields,
            random_state=random_state,
            seed=seed,
            **kwargs,
        )

        self._validate_options_with_learning(
            use_gating_for_weighting,
            enable_learning,
            softmax_choice,
        )

        self._construct_pathways(
            memory_template=self.memory_template,
            memory_capacity=self.memory_capacity,
            normalize_memories=self.normalize_memories,
            softmax_gain=self.softmax_gain,
            softmax_threshold=self.softmax_threshold,
            softmax_choice=self.softmax_choice,
            storage_prob=self.storage_prob,
            memory_decay_rate=self.memory_decay_rate,
            learn_field_weights=self.learn_field_weights,
            enable_learning=self.enable_learning,
            use_gating_for_weighting=self._use_gating_for_weighting,
            context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition2"),
        )

        self._assign_learning_attributes()
        self._assign_conditions()
        self._assign_node_roles()
        self._assign_attributes_for_show_graph()

        memory = self.memory
        if memory is not None and not np.any([
            np.any([memory[i][j] for i in range(self.memory_capacity)])
            for j in range(self.num_keys)
        ]):
            warnings.warn(
                f"Memory initialized with at least one key field that has all zeros; "
                f"a divide by zero can occur if 'normalize_memories' is True. "
                f"Use 'memory_fill' with non-zero values to avoid this."
            )

    # *****************************************************************************************************************
    # *********************************** Memory Construction Methods **************************************************
    # *****************************************************************************************************************

    def _validate_memory_specs(
        self,
        memory_template,
        memory_capacity,
        memory_fill,
        field_weights,
        field_names,
        name,
        learn_field_weights,
    ):
        if isinstance(memory_template, tuple):
            num_fields = memory_template[1] if len(memory_template) == 3 else memory_template[0]
            num_entries = memory_template[0] if len(memory_template) == 3 else memory_capacity
        elif isinstance(memory_template, (list, np.ndarray)):
            num_entries, num_fields = self._parse_memory_shape(memory_template)
        else:
            raise EMComposition2Error(
                f"Unrecognized specification for the 'memory_template' arg ({memory_template}) of {name}."
            )

        if not isinstance(memory_template, tuple) and num_entries > 1:
            for entry in memory_template:
                if not (
                    len(entry) == num_fields
                    and np.all([len(entry[i]) == len(memory_template[0][i]) for i in range(num_fields)])
                ):
                    raise EMComposition2Error(
                        f"The 'memory_template' arg for {name} must have the same shape for all entries."
                    )

        if not (
            isinstance(memory_fill, (int, float))
            or (
                isinstance(memory_fill, tuple)
                and len(memory_fill) == 2
                and all(isinstance(item, (int, float)) for item in memory_fill)
            )
        ):
            raise EMComposition2Error(
                f"The 'memory_fill' arg ({memory_fill}) specified for {name} "
                f"must be a float, int, or length-2 tuple of numbers."
            )

        if isinstance(learn_field_weights, list) and len(learn_field_weights) != num_fields:
            raise EMComposition2Error(
                f"The number of items ({len(learn_field_weights)}) in the "
                f"'learn_field_weights' arg for {name} must match the number "
                f"of fields in memory ({num_fields})."
            )

        if field_weights is not None:
            field_weights = np.atleast_1d(field_weights)
            if len(field_weights) > 1 and len(field_weights) != num_fields:
                raise EMComposition2Error(
                    f"The number of items ({len(field_weights)}) in the 'field_weights' arg "
                    f"for {name} must match the number of fields in memory ({num_fields})."
                )
            if all([fw is None for fw in field_weights]):
                raise EMComposition2Error(
                    f"The entries in 'field_weights' arg for {name} can't all be 'None' "
                    f"since that will preclude the construction of any keys."
                )
            if not any(field_weights):
                warnings.warn(
                    f"All of the entries in the 'field_weights' arg for {name} "
                    f"are either None or set to 0; this will result in no retrievals "
                    f"unless/until one or more of them are changed to a positive value."
                )
            elif any([fw == 0 for fw in field_weights if fw is not None]):
                warnings.warn(
                    f"Some of the entries in the 'field_weights' arg for {name} "
                    f"are set to 0; those fields will be ignored during retrieval "
                    f"unless/until they are changed to a positive value."
                )

        if field_names and len(field_names) != num_fields:
            raise EMComposition2Error(
                f"The number of items ({len(field_names)}) in the 'field_names' arg for {name} "
                f"must match the number of fields ({num_fields})."
            )

    def _parse_memory_template(self, memory_template, memory_capacity, memory_fill):
        def _construct_entries(entry_template, num_entries, memory_fill=None):
            if isinstance(memory_fill, tuple):
                entries = [
                    [
                        np.full(
                            len(field),
                            np.random.uniform(memory_fill[1], memory_fill[0], len(field)),
                        ).tolist()
                        for field in entry_template
                    ]
                    for _ in range(num_entries)
                ]
            else:
                if memory_fill is None:
                    entry = entry_template
                else:
                    entry = [np.full(len(field), memory_fill).tolist() for field in entry_template]
                entries = [np.array(entry, dtype=object) for _ in range(num_entries)]

            return np.array(np.array(entries, dtype=object), dtype=object)

        if isinstance(memory_template, tuple):
            if len(memory_template) == 2:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                memory = _construct_entries(np.full(memory_template, 0), memory_capacity, memory_fill)
            else:
                if memory_capacity and memory_template[0] != memory_capacity:
                    raise EMComposition2Error(
                        f"The first item ({memory_template[0]}) of 'memory_template' does not match "
                        f"'memory_capacity' ({memory_capacity})."
                    )
                memory_capacity = memory_template[0]
                memory = _construct_entries(np.full(memory_template[1:], 0), memory_capacity, memory_fill)
        else:
            num_entries, _ = self._parse_memory_shape(memory_template)

            if num_entries == 1:
                memory_capacity = memory_capacity or self.defaults.memory_capacity
                if any([np.array(field).any() for field in memory_template]):
                    memory_fill = None
                memory = _construct_entries(memory_template, memory_capacity, memory_fill)
            else:
                if not any(list(np.array(memory_template, dtype=object).flat)):
                    memory = _construct_entries(memory_template[0], memory_capacity, memory_fill)
                else:
                    memory_capacity = memory_capacity or num_entries
                    if num_entries > memory_capacity:
                        raise EMComposition2Error(
                            f"The number of entries ({num_entries}) specified in 'memory_template' exceeds "
                            f"'memory_capacity' ({memory_capacity})."
                        )
                    num_entries_needed = memory_capacity - len(memory_template)
                    remaining_entries = _construct_entries(memory_template[0], num_entries_needed, memory_fill)
                    memory = (
                        np.concatenate((np.array(memory_template, dtype=object), remaining_entries))
                        if num_entries_needed
                        else np.array(memory_template, dtype=object)
                    )

        self.entry_template = memory[0]
        return memory, memory_capacity

    def _parse_fields(
        self,
        fields,
        field_names,
        field_weights,
        learn_field_weights,
        learning_rate,
        normalize_field_weights,
        concatenate_queries,
        normalize_memories,
        target_fields,
        name,
    ):
        def _parse_fields_dict(fields_dict, num_fields):
            if len(fields_dict) != num_fields:
                raise EMComposition2Error(
                    f"The number of entries ({len(fields_dict)}) in the dict specified in the 'fields' arg "
                    f"of '{name}' does not match the number of fields in its memory ({num_fields})."
                )

            parsed_names = [None] * num_fields
            parsed_weights = [None] * num_fields
            parsed_learn = [None] * num_fields
            parsed_targets = [None] * num_fields

            for i, field_name in enumerate(fields_dict):
                parsed_names[i] = field_name
                spec = fields_dict[field_name]
                if isinstance(spec, (tuple, list)):
                    parsed_weights[i], parsed_learn[i], parsed_targets[i] = spec
                elif isinstance(spec, dict):
                    parsed_weights[i] = spec[FIELD_WEIGHT]
                    parsed_learn[i] = spec[LEARN_FIELD_WEIGHT]
                    parsed_targets[i] = spec[TARGET_FIELD]
                else:
                    raise EMComposition2Error(
                        f"Unrecognized specification for field '{field_name}' in 'fields' for '{name}'."
                    )

            return parsed_names, parsed_weights, parsed_learn, parsed_targets

        self.num_fields = len(self.entry_template)

        if isinstance(learning_rate, dict):
            raise EMComposition2Error(
                f"The 'learning_rate' arg for '{name}' is specified as a dict, "
                f"which is not supported for an EMComposition2;  "
                f"use either its 'fields' arg or its 'learn_field_weights' arg instead."
            )

        if fields:
            if any([field_names, field_weights, learn_field_weights, target_fields]):
                warnings.warn(
                    f"The 'fields' arg for '{name}' was specified, so any of the "
                    f"'field_names', 'field_weights',  'learn_field_weights' or "
                    f"'target_fields' args will be ignored."
                )
            field_names, field_weights, learn_field_weights, target_fields = _parse_fields_dict(
                fields,
                self.num_fields,
            )

        if field_weights is None:
            if len(self.entry_template) == 1:
                field_weights = [1]
            else:
                field_weights = [1] * self.num_fields
                field_weights[-1] = None

        field_weights = np.atleast_1d(field_weights)

        if normalize_field_weights and not all([fw == 0 for fw in field_weights if fw is not None]):
            weights_for_sum = [fw if fw is not None else 0 for fw in field_weights]
            denominator = np.sum(weights_for_sum) or 1
            parsed_field_weights = [
                fw / denominator if fw is not None else None
                for fw in field_weights
            ]
        else:
            parsed_field_weights = field_weights

        if len(field_weights) == 1 and self.num_fields > 1:
            parsed_field_weights = np.repeat(parsed_field_weights, self.num_fields)

        individually_specified = True
        if not is_iterable(learn_field_weights) and learn_field_weights in {None, True, False}:
            learn_field_weights = [learn_field_weights] * len(parsed_field_weights)
            individually_specified = False

        if isinstance(learn_field_weights, (list, tuple, np.ndarray)):
            learn_field_weights = list(learn_field_weights)
            for i, (fw, lfw) in enumerate(zip(parsed_field_weights, learn_field_weights)):
                if fw is None:
                    if lfw and individually_specified:
                        warnings.warn(
                            f"A learning_rate was specified for field '{field_names[i] if field_names else i}' "
                            f"in the 'learn_field_weights' arg for '{name}', "
                            f"but it is not allowed for value fields; it will be ignored."
                        )
                    learn_field_weights[i] = False
                elif lfw in {None, True}:
                    learn_field_weights[i] = learning_rate or lfw
        else:
            raise EMComposition2Error(
                f"PROGRAM ERROR: learn_field_weights ({learn_field_weights}) is not a valid specification."
            )

        parsed_field_names = field_names.copy() if field_names is not None else None

        self.key_indices = [i for i, fw in enumerate(parsed_field_weights) if fw is not None]
        self.value_indices = [i for i, fw in enumerate(parsed_field_weights) if fw is None]
        self.num_keys = len(self.key_indices)
        self.num_values = len(self.value_indices)

        if parsed_field_names:
            self.key_names = [parsed_field_names[i] for i in self.key_indices]
            self.value_names = [parsed_field_names[i] for i in self.value_indices]
        else:
            self.key_names = [f"{i}" for i in range(self.num_keys)] if self.num_keys > 1 else ["KEY"]
            self.value_names = (
                [f"{i} [VALUE]" for i in range(self.num_values)]
                if self.num_values > 1
                else (["VALUE"] if self.num_values == 1 else [])
            )
            parsed_field_names = self.key_names + self.value_names

        user_specified_concatenate_queries = concatenate_queries or False
        key_weights = [weight for weight in parsed_field_weights if weight is not None]
        concatenate_queries = (
            user_specified_concatenate_queries
            and self.num_keys > 1
            and all(np.all(key_weight == key_weights[0]) for key_weight in key_weights)
            and normalize_memories
        )
        if user_specified_concatenate_queries and not concatenate_queries:
            if self.num_keys == 1:
                error_msg = "there is only one key"
                correction_msg = ""
            elif not all(np.all(key_weight == key_weights[0]) for key_weight in key_weights):
                error_msg = f"field weights ({field_weights}) are not all equal"
                correction_msg = " To use concatenation, remove `field_weights` specification or make them all the same."
            elif not normalize_memories:
                error_msg = "normalize_memories is False"
                correction_msg = " To use concatenation, set normalize_memories to True."
            else:
                error_msg = "it is not supported"
                correction_msg = ""
            warnings.warn(
                f"The 'concatenate_queries' arg for '{name}' is True but {error_msg}; "
                f"concatenation will be ignored.{correction_msg}"
            )

        if target_fields is None:
            target_fields = [True] * self.num_fields

        self.learning_rate = learning_rate

        for i, field_name, weight, learn_weight, target in zip(
            range(self.num_fields),
            parsed_field_names,
            parsed_field_weights,
            learn_field_weights,
            target_fields,
        ):
            self.fields.append(
                Field(
                    name=field_name,
                    index=i,
                    type=FieldType.KEY if weight is not None else FieldType.VALUE,
                    weight=weight,
                    learn_weight=learn_weight,
                    target=target,
                )
            )

        return (
            parsed_field_names,
            parsed_field_weights,
            learn_field_weights,
            target_fields,
            concatenate_queries,
        )

    def _parse_memory_shape(self, memory_template):
        memory_template_dim = np.array(memory_template, dtype=object).ndim

        if memory_template_dim == 1 or all(isinstance(item, (int, float)) for item in memory_template[0]):
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template)
        else:
            fields_equal_length = all(len(field) == len(memory_template[0]) for field in memory_template[0])

        single_entry = (
            ((memory_template_dim == 1) and not fields_equal_length)
            or ((memory_template_dim == 2) and fields_equal_length)
        )
        num_entries = 1 if single_entry else len(memory_template)
        num_fields = len(memory_template) if single_entry else len(memory_template[0])
        return num_entries, num_fields

    # *****************************************************************************************************************
    # *********************************** Nodes and Pathway Construction Methods ***************************************
    # *****************************************************************************************************************

    def _construct_pathways(
        self,
        memory_template,
        memory_capacity,
        normalize_memories,
        softmax_gain,
        softmax_threshold,
        softmax_choice,
        storage_prob,
        memory_decay_rate,
        learn_field_weights,
        enable_learning,
        use_gating_for_weighting,
        context,
    ):
        self._construct_input_nodes()
        self._construct_concatenate_queries_node()
        self._construct_field_memory_nodes(
            memory_template,
            memory_capacity,
            normalize_memories,
            storage_prob,
            memory_decay_rate,
        )
        self._construct_concatenated_memory_node(
            memory_template,
            normalize_memories,
            storage_prob,
            memory_decay_rate,
        )
        self._construct_field_weight_nodes()
        self._construct_weighted_scores_nodes()
        self._construct_combined_scores_node(memory_capacity, softmax_gain, softmax_threshold, softmax_choice)
        self._construct_softmax_gain_control_node(softmax_gain)
        self._construct_retrieved_nodes()

        self._field_index_map = {
            node: field.index
            for field in self.fields
            for node in field.nodes
        }
        self._field_index_map.update({
            proj: field.index
            for field in self.fields
            for proj in field.projections
        })

        # EM2 BREADCRUMB: THIS NEED TO DEAL WITH MULTIPLE PROJECTIONS BETWEEN MEMORY NODES AND COMBINED_SCORES NODE
        if not self.enable_learning:
            self.add_nodes(self.input_nodes, context=context)
            if self.concatenate_queries_node:
                self.add_node(self.concatenate_queries_node, context=context)
            self.add_nodes(self.field_memory_nodes, context=context)
            if self.concatenated_memory_node:
                self.add_node(self.concatenated_memory_node, context=context)
            self.add_nodes(self.field_weight_nodes + self.weighted_scores_nodes, context=context)
            self.add_nodes([self.combined_scores_node] + self.retrieved_nodes, context=context)
            if self.softmax_gain_control_node:
                self.add_node(self.softmax_gain_control_node, context=context)
            self._add_pathway_projections(context)
            return

        for field in self.fields:
            self.add_linear_processing_pathway([field.input_node,
                                                field.memory_node])

        if self.concatenate_queries:
            for field in self.key_fields:
                self.add_linear_processing_pathway([field.input_node,
                                                    self.concatenate_queries_node])
            self.add_linear_processing_pathway([self.concatenate_queries_node,
                                                self.concatenated_memory_node,
                                                self.combined_scores_node])

        elif self.num_keys == 1:
            self.add_linear_processing_pathway([self.key_fields[0].memory_node,
                                                self.combined_scores_node])
        else:
            for field in self.key_fields:
                pathway = [field.memory_node,
                           self.combined_scores_node]
                if field.weighted_scores_node:
                    pathway.insert(1, field.weighted_scores_node)
                self.add_linear_processing_pathway(pathway)

        for field in self.fields:
            self.add_linear_processing_pathway([self.combined_scores_node,
                                                field.memory_node,
                                                field.retrieved_node])
            # EM2 BREADCRUMB:
            # self.add_projections([self.combined_scores_node.efferents])

        if self.softmax_gain_control_node:
            self.add_node(self.softmax_gain_control_node, context=context)

        for field in self.key_fields:
            if field.weight_node and field.weighted_scores_node:
                self.add_linear_processing_pathway([
                    field.weight_node,
                    field.weighted_scores_node])

        # EM2 BREADCRUMB:
        #     HACK TO DEAL WITH FAILURE OF composition.add_projection() to handle multiple projections between mechs
        for proj in (self.combined_scores_node.path_afferents + self.combined_scores_node.efferents):
            if proj not in self.projections:
                self.add_projection(proj, context=context)
        self._add_pathway_projections(context)

    def _construct_input_nodes(self):
        for field in self.key_fields:
            field.input_node = ProcessingMechanism(name=f"{field.name} [QUERY]",
                                                   input_shapes=len(self.entry_template[field.index]))
            field.type = FieldType.KEY

        for field in self.value_fields:
            field.input_node = ProcessingMechanism(name=f"{field.name} [VALUE]",
                                                   input_shapes=len(self.entry_template[field.index]))
            field.type = FieldType.VALUE

    def _construct_concatenate_queries_node(self):
        if not self.concatenate_queries:
            self.concatenate_queries_node = None
            self.concatenated_memory_node = None
            return

        self.concatenate_queries_node = ProcessingMechanism(
            name=CONCATENATE_QUERIES_NAME,
            function=Concatenate,
            input_ports=[
                {
                    NAME: "CONCATENATE",
                    INPUT_SHAPES: len(field.input_node.output_port.value),
                    PROJECTIONS: MappingProjection(
                        name=f"{field.name} to CONCATENATE",
                        sender=field.input_node.output_port,
                        matrix=IDENTITY_MATRIX,
                    ),
                }
                for field in self.key_fields
            ],
        )
        for field, proj in zip(self.key_fields, self.concatenate_queries_node.path_afferents):
            field.concatenation_projection = proj

    def _construct_field_memory_nodes(
        self,
        memory_template,
        memory_capacity,
        normalize_memories,
        storage_prob,
        memory_decay_rate,
    ):

        for field in self.fields:
            key_len = 1 if is_numeric_scalar(field.query.squeeze()) else len(field.query.squeeze())
            field_memory = np.array(memory_template[:, field.index].tolist()).astype(float)

            field.memory_node = ExternalMemoryMechanism(
                field_type = field.type,
                field_shape = len(self.entry_template[field.index]),
                field_memory = field_memory,
                decay_rate = memory_decay_rate,
                storage_prob = storage_prob,
                scores_metric = L0 if key_len == 1 else DOT_PRODUCT,
                normalize_memories = True if key_len == 1 else normalize_memories,
                name=f"{field.name}{FIELD_MEMORY_AFFIX}",
            )

            field.query_projection = MappingProjection(
                sender=field.input_node,
                receiver=field.memory_node.input_ports[QUERY],
                matrix=IDENTITY_MATRIX,
                name=f"{field.name} QUERY to FIELD MEMORY",
            )

    def _construct_concatenated_memory_node(
        self,
        memory_template,
        normalize_memories,
        storage_prob,
        memory_decay_rate,
    ):
        if not self.concatenate_queries:
            self.concatenated_memory_node = None
            return

        concatenated_memory = np.array([
            np.concatenate([entry[field.index] for field in self.key_fields])
            for entry in memory_template
        ]).astype(float)
        key_len = len(self.entry_template[self.key_fields[0].index])

        self.concatenated_memory_node = ExternalMemoryMechanism(
            field_type=FieldType.KEY,
            field_shape=concatenated_memory.shape[1],
            field_memory=concatenated_memory,
            decay_rate=memory_decay_rate,
            storage_prob=storage_prob,
            scores_metric=L0 if key_len == 1 else DOT_PRODUCT,
            normalize_memories=True if key_len == 1 else normalize_memories,
            name=f"{MATCH}{FIELD_MEMORY_AFFIX}",
        )
        self.concatenated_query_projection = MappingProjection(
            sender=self.concatenate_queries_node,
            receiver=self.concatenated_memory_node.input_ports[QUERY],
            matrix=IDENTITY_MATRIX,
            name=f"{CONCATENATE_QUERIES_NAME} to {MATCH}{FIELD_MEMORY_AFFIX}",
        )

    def _construct_field_weight_nodes(self):
        if self.num_keys <= 1 or self.concatenate_queries:
            return

        for field in self.key_fields:
            name = f"{field.name}{WEIGHT_AFFIX}"
            variable = np.array(field.weight)
            params = {DEFAULT_INPUT: DEFAULT_VARIABLE}

            field.weight_node = ProcessingMechanism(
                name=name,
                input_ports={
                    NAME: "FIELD_WEIGHT",
                    VARIABLE: variable,
                    PARAMS: params,
                },
            )

    def _construct_weighted_scores_nodes(self):
        if self.num_keys <= 1 or self.concatenate_queries:
            return

        for field in self.key_fields:
            field.weighted_scores_node = ProcessingMechanism(
                name=field.name + WEIGHTED_SCORES_AFFIX,
                default_variable=[
                    np.zeros(self.memory_capacity),
                    np.zeros(self.memory_capacity),
                ],
                input_ports=[
                    {
                        PROJECTIONS: MappingProjection(
                            name=f"{field.name} {SCORES} to {WEIGHTED_SCORES_NODE_NAME}",
                            sender=field.memory_node.output_ports[SCORES],
                            matrix=IDENTITY_MATRIX,
                        )
                    },
                    {
                        PROJECTIONS: MappingProjection(
                            name=f"{field.name} {WEIGHT} to {WEIGHTED_SCORES_NODE_NAME}",
                            sender=field.weight_node,
                            matrix=FULL_CONNECTIVITY_MATRIX,
                        )
                    }
                ],
                output_ports={NAME: WEIGHTED_SCORES,
                              VARIABLE: (OWNER_VALUE,0)},
                function=LinearCombination(operation=PRODUCT),
            )
            field.scores_projection = field.weighted_scores_node.path_afferents[0]
            field.weight_projection = field.weighted_scores_node.path_afferents[1]

    def _construct_combined_scores_node(self, memory_capacity, softmax_gain, softmax_threshold,
                                                 softmax_choice):
        """Construct combined_scores_node
        This is constructed even if num_keys == 1, since it computes the softmax over the scores
        IMPLEMENTATION NOTE:  This plays the same role as the softmax_node in emcomposition.py
        """

        if softmax_choice == ARG_MAX:
            softmax_choice = ARG_MAX_INDICATOR
        initial_softmax_gain = 1.0 if softmax_gain == CONTROL else softmax_gain
        softmax_function = SoftMax(gain=initial_softmax_gain,
                                   mask_threshold=softmax_threshold,
                                   output=softmax_choice,
                                   adapt_entropy_weighting=.95)

        # Construct combined_scores_function
        def _combined_scores_function(variable, gain=initial_softmax_gain):
            """Return softmax over combined scores, and index of minimum norm over combined norms
            variable[0] = scores of memory Nodes combined by hadamard addition in the COMBINED_SCORES input_port
            variable[1] = norms of memory Nodes combined by hadamard addition in the COMBINED_NORMS input_port
            """
            assert len(variable) == 2, \
                (f"PROGRAM ERROR: expected variable with 2 items for combined_scores_function; got {len(variable)}")
            return softmax_function(variable[0], params={GAIN: gain}), int(np.argmin(variable[1]))

        def _gen_pytorch_fct(device, context):
            """Return pytorch version of function"""
            # EM2 BREADCRUMB: CONTEXT execution_id NEEDS TO BE SET TO None,
            #                 SINCE _gen_pytorch_fct IS CALLED IN execution context
            #                 BUT SoftMax Function WAS CONSTRUCTED DURING __init__
            #                 AND SO ITS PARAMETERS HAVE NO VALUES FOR execution_id
            #                 ?? COULD BE DUE TO ORDER OF CALLS TO _gen_pytorch_fct IN PytorchFunctionWrapper??
            #                 POTENTIAL PROBLEM: WHEN FUNCTION IS CALLED IN EXECUTION CONTEXT,
            #                    WILL SOFTMAX FUNCTION PARAMS HAVE VALUES FOR CURRENT CONTEXT OR JUST USE None?
            local_context = copy.copy(context)
            local_context.execution_id = None
            softmax_func = softmax_function._gen_pytorch_fct(device, local_context)
            def func(variable):
                scores = variable[:, :, 0, ...]
                norms = variable[:, :, 1, ...]
                softmax_scores = softmax_func(scores)
                weakest_memory_idx = torch.argmin(norms, dim=-1, keepdim=True).to(dtype=softmax_scores.dtype)
                return [[[softmax_scores[b, s, ...], weakest_memory_idx[b, s, ...]]
                         for s in range(softmax_scores.shape[1])]
                        for b in range(softmax_scores.shape[0])]
            return func

        combined_scores_function = UserDefinedFunction(_combined_scores_function,
                                                       default_variable=[np.zeros(memory_capacity),
                                                                         np.zeros(memory_capacity)],
                                                       pytorch_function_generator =_gen_pytorch_fct
                                                       )
        # combined_scores_function._gen_pytorch_fct = _gen_pytorch_fct

        field_weighting = self.num_keys > 1 and not self.concatenate_queries
        assert (self.weighted_scores_nodes and self.field_weight_nodes) if field_weighting else not field_weighting, \
            (f"PROGRAM ERROR: Mismatch between num_keys and presence of weighted_scores_nodes and/or field_weight_nodes")

        if self.concatenate_queries:
            scores_inputs = [self.concatenated_memory_node.output_ports[SCORES]]
            scores_input_names = [CONCATENATE_QUERIES_NAME]
        else:
            scores_inputs = [(field.weighted_scores_node.output_ports[WEIGHTED_SCORES] if field_weighting
                              else field.memory_node.output_ports[SCORES])
                              for field in self.key_fields]
            scores_input_names = [field.name for field in self.key_fields]
        # EM2 BREADCRUMB: THIS WEIGHTS THE NORMS, WHICH IS PROBABLY NOT CORRECT:
        # norms_inputs = [(field.weighted_scores_node if field.type == FieldType.KEY and field_weighting
        #                  else field.memory_node).output_ports[NORMS]
        #                 for field in self.fields]
        norms_inputs = [field.memory_node.output_ports[NORMS] for field in self.fields]
        self.combined_scores_node = ProcessingMechanism(
            name=COMBINED_SCORES_NODE_NAME,
            input_ports=[
                {NAME:SCORES,
                 INPUT_SHAPES: memory_capacity,
                 PROJECTIONS: [
                     MappingProjection(
                         sender=source,
                         matrix=IDENTITY_MATRIX,
                         name=f"{'WEIGHTED' if field_weighting else ''} {SCORES} for {scores_input_names[i]}")
                              # f" to {COMBINED_SCORES_NODE_NAME}")
                     for i, source in enumerate(scores_inputs)]},
                {NAME:NORMS,
                 INPUT_SHAPES: memory_capacity,
                 PROJECTIONS: [
                     MappingProjection(
                         sender=source,
                         matrix=IDENTITY_MATRIX,
                         name=f"{'WEIGHTED' if field_weighting else ''} {NORMS} for {self.fields[i].name}")
                              # f" to {COMBINED_SCORES_NODE_NAME}")
                     for i, source in enumerate(norms_inputs)]},
            ],
            output_ports=[{NAME:COMBINED_SCORES, VARIABLE: (OWNER_VALUE, 0)},
                          {NAME:COMBINED_NORMS, VARIABLE: (OWNER_VALUE, 1)}],
            function=combined_scores_function
        )

        # EM2 BREADCRUMB: MAKE THIS SPECIFIC TO SCORES, AND ADD SIMILAR LOOP FOR NORMS
        if self.concatenate_queries:
            self.concatenated_scores_projection = next(
                proj for proj in self.combined_scores_node.path_afferents
                if proj.sender is self.concatenated_memory_node.output_ports[SCORES]
            )
        for field in self.fields:
            # Assign Projections from memory_nodes to combined_scores nodes to relevant attributes of field
            if field.type == FieldType.KEY and not self.concatenate_queries:
                # EM2 BREADCRUMB: NEED TO GET AFFERENT FROM field_weighted_scores NODE IF field_weighting
                scores_proj = next(proj for proj in self.combined_scores_node.path_afferents
                                   if proj.sender is (field.weighted_scores_node.output_ports[WEIGHTED_SCORES]
                                                      if field_weighting else field.memory_node.output_ports[SCORES]))
                field.weighted_scores_projection = scores_proj
            norms_proj = next(proj for proj in self.combined_scores_node.path_afferents
                              if proj.sender is field.memory_node.output_ports[NORMS])
            field.weighted_norms_projection = norms_proj

            # EM2 BREADCRUMB: NEED TO EXPLICITLY ADD PROJECTIONS TO COMPOSITION,
            #     SINCE THE COMBINED_SCORES ONE DOES NOT SEEM TO BE GETTING ADDED (BLOCKED BY COMBINED_NORMS ONE?)
            # Assign Projections from combined_scores nodes back to COMBINED_SCORES input_ports of field_memory_nodes
            # Note: this has to be constructed here, as it depends on the combined_scores_node being constructed first
            field.combined_scores_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_SCORES],
                feedback=True,
                receiver=field.memory_node.input_ports[COMBINED_SCORES],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {field.name} COMBINED_SCORES",
            )
            # Assign Projections from combined_scores nodes back to COMBINED_NORMS input_ports of field_memory_nodes
            # Note: this has to be constructed here, as it depends on the combined_scores_node being constructed first
            field.combined_norms_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_NORMS],
                feedback=True,
                receiver=field.memory_node.input_ports[COMBINED_NORMS],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {field.name} COMBINED_NORMS",
            )

        if self.concatenate_queries:
            self.concatenated_combined_scores_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_SCORES],
                feedback=True,
                receiver=self.concatenated_memory_node.input_ports[COMBINED_SCORES],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {CONCATENATE_QUERIES_NAME} COMBINED_SCORES",
            )
            self.concatenated_combined_norms_projection = MappingProjection(
                sender=self.combined_scores_node.output_ports[COMBINED_NORMS],
                feedback=True,
                receiver=self.concatenated_memory_node.input_ports[COMBINED_NORMS],
                matrix=IDENTITY_MATRIX,
                name=f"{COMBINED_SCORES_NODE_NAME} to {CONCATENATE_QUERIES_NAME} COMBINED_NORMS",
            )


    def _construct_softmax_gain_control_node(self, softmax_gain):
        node = None
        if softmax_gain == CONTROL:
            node = ControlMechanism(
                name="SOFTMAX GAIN CONTROL",
                monitor_for_control=self.combined_scores_node or self.key_fields[0].memory_node,
                control_signals=[(GAIN, self.combined_scores_node)],
                function=get_softmax_gain,
            )
        self.softmax_gain_control_node = node

    def _construct_retrieved_nodes(self):
        for field in self.fields:
            field.retrieved_node = ProcessingMechanism(
                name=field.name + RETRIEVED_AFFIX,
                input_ports={
                    INPUT_SHAPES: len(field.input_node.variable[0]),
                    PROJECTIONS: MappingProjection(
                        sender=field.memory_node.output_ports[RETRIEVED],
                        matrix=IDENTITY_MATRIX,
                        name=f"{field.name} RETRIEVED to OUTPUT",
                    ),
                },
            )
            field.retrieved_projection = field.retrieved_node.path_afferents[0]

    def _assign_conditions(self):


        for field in self.fields:

            # Input and weight nodes should run only once, at the beginning of the trial
            # EM2 BREADCRUMB: DOES THIS CONDITION NEED A TimeScale SPECIFICATION (I.E., TRIAL)?
            self.scheduler.add_condition(field.input_node, BeforeNCalls(field.input_node, 1))
            if field.weight_node is not None:
                self.scheduler.add_condition(field.weight_node, BeforeNCalls(field.weight_node, 1))

            # Field-memory mechanisms must run once after inputs, then again after RETRIEVE.
            self.scheduler.add_condition(
                field.memory_node,
                Any(All(AfterNCalls(field.input_node, 1),
                        BeforeNCalls(self.combined_scores_node, 1)),
                    All(AfterNCalls(self.combined_scores_node, 1),
                        BeforeNCalls(field.retrieved_node, 1)))
            )

            # Storage should be after RETRIEVAL
            field.memory_node.parameters.access_condition.set(
                conditions.AfterNCalls(self.combined_scores_node, 1),
                context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition2 storage conditions"),
                override=True)


            # Retrieved nodes run only after both field-memory mechanisms have run twice.
            if self.concatenated_memory_node:
                self.scheduler.add_condition(
                    field.retrieved_node,
                    All(AfterNCalls(field.memory_node, 2),
                        AfterNCalls(self.concatenated_memory_node, 2))
                )
            else:
                self.scheduler.add_condition(field.retrieved_node, AfterNCalls(field.memory_node, 2))

        if self.concatenated_memory_node:
            self.scheduler.add_condition(
                self.concatenated_memory_node,
                Any(All(AfterNCalls(self.concatenate_queries_node, 1),
                        BeforeNCalls(self.combined_scores_node, 1)),
                    All(AfterNCalls(self.combined_scores_node, 1),
                        BeforeNCalls(self.retrieved_nodes[0], 1)))
            )
            self.concatenated_memory_node.parameters.access_condition.set(
                conditions.AfterNCalls(self.combined_scores_node, 1),
                context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition2 storage conditions"),
                override=True)

        # # RETRIEVE runs only after both field-memory mechanisms have run once.
        # args = ([AfterNCalls(node, 1) for node in self.field_memory_nodes]
        #         + [BeforeNCalls(node, 2) for node in self.field_memory_nodes])
        # self.scheduler.add_condition(self.combined_scores_node, All(*args))

        # # Storage should be after RETRIEVAL
        # for field_memory_node in self.field_memory_nodes:
        #     field_memory_node.parameters.access_condition.set(
        #         conditions.AfterNCalls(self.combined_scores_node, 1),
        #         context=Context(source=ContextFlags.COMMAND_LINE, string="FROM EMComposition2 storage conditions"),
        #         override=True,
        #     )

        # # BREADCRUMB: NECESSARY??
        # # End the trial after all retrieved nodes have executed once.
        # args = [AfterNCalls(node, 1) for node in self.retrieved_nodes]
        # self.scheduler.termination_conds[TimeScale.TRIAL] = (All(*args))


    def _assign_node_roles(self):
        for node in self.field_weight_nodes:
            self.exclude_node_roles(node, NodeRole.INPUT)
        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)
        if self.concatenate_queries_node:
            self.exclude_node_roles(self.concatenate_queries_node, NodeRole.OUTPUT)
        if self.concatenated_memory_node:
            self.exclude_node_roles(self.concatenated_memory_node, NodeRole.OUTPUT)
        self.exclude_node_roles(self.combined_scores_node, NodeRole.OUTPUT)


    def _assign_attributes_for_show_graph(self):
        for node in self.value_input_nodes:
            node.output_port.parameters.require_projection_in_composition.set(False, override=True)
        if self.concatenate_queries_node:
            self.concatenate_queries_node.output_port.parameters.require_projection_in_composition.set(False, override=True)
        if self.concatenated_memory_node:
            for output_port in self.concatenated_memory_node.output_ports:
                output_port.parameters.require_projection_in_composition.set(False, override=True)
        self.combined_scores_node.output_port.parameters.require_projection_in_composition.set(False, override=True)

    def _add_pathway_projections(self, context):
        projections = []
        for field in self.fields:
            projections.extend(field.projections)

        if self.concatenate_queries:
            projections.extend([
                self.concatenated_query_projection,
                self.concatenated_scores_projection,
                self.concatenated_combined_scores_projection,
                self.concatenated_combined_norms_projection,
            ])

        projections.extend(self.combined_scores_node.path_afferents)
        projections.extend(self.combined_scores_node.efferents)

        for proj in [proj for proj in projections if proj is not None]:
            if proj not in self.projections:
                self.add_projection(proj, context=context)

    def _assign_learning_attributes(self):
        self.execute_in_additional_optimizations = {}

        field_weight_projections = []
        for projection in self.projections:
            if projection.sender.owner in self.field_weight_nodes:
                field_weight_projections.append(projection)
            else:
                projection.learnable = False
                projection.learning_rate = False

        learn_field_weights = self.parameters.learn_field_weights.spec
        if not isinstance(learn_field_weights, (list, np.ndarray)):
            assert not self.enable_learning, (
                "PROGRAM ERROR: self.learn_field_weights is not a list, but should be by this point."
            )

        if (
            all(item is False for item in learn_field_weights)
            or len(self.query_input_nodes) == 1
        ):
            lr_dict = {}
            for projection in field_weight_projections:
                projection.learnable = False
                projection.learning_rate = False
                lr_dict[projection] = False
            self._enable_learning_warning_flag = True
        else:
            lr_dict = {}
            constructor_learning_rate = self.parameters.learning_rate.get(None)
            if not isinstance(constructor_learning_rate, dict):
                lr_dict[DEFAULT_LEARNING_RATE] = constructor_learning_rate

            for i, field in enumerate(self.fields):
                if field.type == FieldType.KEY and field.weight_node:
                    proj = field.weight_node.efferents[0]
                    if learn_field_weights[i] is False:
                        lr_dict[proj] = False
                        proj.learnable = False
                    elif is_numeric_scalar(learn_field_weights[i]):
                        lr_dict[proj] = learn_field_weights[i]
                    elif learn_field_weights[i] is None:
                        continue
                    else:
                        raise EMComposition2Error(
                            f"PROGRAM ERROR: learning_rate for {field.name} "
                            f"({learn_field_weights[i]}) is not valid."
                        )

        self.parameters.learning_rate._set(lr_dict, context=Context(execution_id=None))

    def _validate_options_with_learning(self, use_gating_for_weighting, enable_learning, softmax_choice):
        if use_gating_for_weighting and enable_learning:
            warnings.warn(
                f"The 'enable_learning' option for '{self.name}' cannot be used with "
                f"'use_gating_for_weighting=True'; this will generate an error if learn() is called."
            )

        if softmax_choice in {ARG_MAX, PROBABILISTIC} and enable_learning:
            warnings.warn(
                f"The 'softmax_choice' arg of '{self.name}' is set to '{softmax_choice}' with "
                f"'enable_learning' set to True; this will generate an error if its "
                f"'learn' method is called. Set 'softmax_choice' to WEIGHTED_AVG before learning."
            )

    # *****************************************************************************************************************
    # ***************************************** Execution Methods ******************************************************
    # *****************************************************************************************************************

    @handle_external_context(fallback_default=True)
    def learn(
        self,
        *args,
        context: Optional[Context] = None,
        base_context: Context = Context(execution_id=None),
        skip_initialization: bool = False,
        **kwargs,
    ) -> list:
        if (
            not skip_initialization
            and (
                context is None
                or ContextFlags.SIMULATION_MODE not in context.runmode
            )
        ):
            self._initialize_from_context(context, base_context, override=False)

        softmax_choice = self.parameters.softmax_choice.get(context)
        use_gating_for_weighting = self._use_gating_for_weighting
        enable_learning = self.parameters.enable_learning.get(context)

        if use_gating_for_weighting and enable_learning:
            raise EMComposition2Error(
                f"Field weights cannot be learned when 'use_gating_for_weighting' is True; "
                f"construct '{self.name}' with 'enable_learning=False'."
            )

        if self.concatenate_queries:
            raise EMComposition2Error(
                "EMComposition2 does not support learning with 'concatenate_queries'='True'."
            )

        if softmax_choice in {ARG_MAX, PROBABILISTIC}:
            raise EMComposition2Error(
                f"The ARG_MAX and PROBABILISTIC options for the 'softmax_choice' arg of '{self.name}' "
                f"cannot be used during learning; change to WEIGHTED_AVG."
            )

        if self._enable_learning_warning_flag and not self.is_nested:
            if len(self.query_input_nodes) == 1:
                warnings.warn(
                    f"The 'enable_learning' arg of '{self.name}' is True, but it has only one key, "
                    f"so field_weights and field-weight learning have no effect."
                )

        return super().learn(
            *args,
            context=context,
            base_context=base_context,
            skip_initialization=skip_initialization,
            **kwargs,
        )

    def _instantiate_input_dict(self, input_dict):
        """Override to determine — and respond appropriately -- if any KEY and/or VALUE fields are not specified.
        - If any KEY fields are missing, raise error
        - If any VALUE fields are missing, issue warning that the retrieved value will be stored with the specified KEY
        """
        if self.is_nested:
            # EM2 BREADCRUMB: NEED TESTS FOR THIS
            missing_query_nodes = [f"'{node.name}'" for node in self.query_input_nodes
                                   if self.input_CIM._get_source_node_for_input_CIM(node.input_port)]
            missing_value_nodes = [node for node in self.value_input_nodes
                                   if self.input_CIM._get_source_node_for_input_CIM(node.input_port)]
        else:
            missing_query_nodes = [f"'{node.name}'" for node in self.query_input_nodes if node not in input_dict]
            missing_value_nodes = [node for node in self.value_input_nodes if node not in input_dict]

        if missing_query_nodes:
            raise EMComposition2Error(
                f"'inputs' argument of call to learn() method for '{self.name}' is missing entries "
                f"for the following query_input_nodes: {', '.join(missing_query_nodes)}")

        if missing_value_nodes:
            for field in [f for f in self.fields if f.input_node in missing_value_nodes]:
                field.input_node.value_input_specified = False
            missing_value_nodes_str = [f"'{node.name}'" for node in missing_value_nodes]
            plural = len(missing_value_nodes) > 1
            query_str = 'queries' if plural else 'query'
            key_str = 'keys' if plural else 'key'
            s = "s" if plural else ""
            their_its = 'their' if plural else 'its'
            entries = "entries" if plural else "an entry"
            warnings.warn(f"'inputs' argument of call to learn() method for '{self.name}' is missing {entries} "
                          f"for the following value_input_node{s}, so the retrieved value{s} will be stored with "
                          f"the specified {query_str} as {their_its} {key_str}: {', '.join(missing_value_nodes_str)}.")

        return super()._instantiate_input_dict(input_dict)

    def _get_execution_mode(self, execution_mode):
        if execution_mode is None:
            if self._warned_about_default_execution_mode is False:
                warnings.warn(
                    f"The execution_mode argument was not specified in learn() for {self.name}; "
                    f"ExecutionMode.PyTorch will be used by default."
                )
                self._warned_about_default_execution_mode = True
            execution_mode = ExecutionMode.PyTorch
        return execution_mode

    def _identify_output_nodes(self, context) -> list:
        target_fields = self.target_fields

        if target_fields is False:
            if self.enable_learning:
                warnings.warn(
                    f"The 'enable_learning' arg for {self.name} is True but 'target_fields' is False, "
                    f"so enable_learning will have no effect."
                )
            target_nodes = []
        elif target_fields is True:
            target_nodes = [node for node in self.retrieved_nodes]
        elif isinstance(target_fields, (list, tuple, np.ndarray)):
            target_nodes = [
                node for node in self.retrieved_nodes
                if target_fields[self.retrieved_nodes.index(node)]
            ]
        else:
            assert False, (
                f"PROGRAM ERROR: target_fields arg for {self.name}: {target_fields} "
                f"is neither True, False, nor a list of bools."
            )

        super()._identify_output_nodes(context)
        return target_nodes

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None, base_context=None):
        return super().infer_backpropagation_learning_pathways(
            execution_mode,
            context=context,
            base_context=base_context,
        )

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        # EM storage is field-local and executed by ExternalMemoryMechanism after retrieval.
        # Field-weight learning can be restored by calling super() once the PyTorch wrapper
        # supports ExternalMemoryMechanism as a differentiable memory component.
        pass

    def add_node(self, node, required_roles=None, context=None):
        if context is None:
            raise EMComposition2Error(f"Nodes cannot be added to an {self.componentCategory}: ('{self.name}').")
        super().add_node(node, required_roles, context)

    def add_projection(self, *args, **kwargs):
        if CONTEXT not in kwargs or kwargs[CONTEXT] is None:
            raise EMComposition2Error(f"Projections cannot be added to an {self.componentCategory}: ('{self.name}').")
        return super().add_projection(*args, **kwargs)

    # *****************************************************************************************************************
    # ***************************************** Properties *************************************************************
    # *****************************************************************************************************************

    @property
    def key_fields(self):
        return [field for field in self.fields if field.type == FieldType.KEY]

    @property
    def value_fields(self):
        return [field for field in self.fields if field.type == FieldType.VALUE]

    @property
    def input_nodes(self):
        return [field.input_node for field in self.fields]

    @property
    def query_input_nodes(self):
        return [field.input_node for field in self.key_fields]

    @property
    def value_input_nodes(self):
        return [field.input_node for field in self.value_fields]

    @property
    def field_memory_nodes(self):
        return [field.memory_node for field in self.fields]

    @property
    def memory_cycle_nodes(self):
        nodes = list(self.field_memory_nodes)
        if getattr(self, "concatenated_memory_node", None) is not None:
            nodes.append(self.concatenated_memory_node)
        return nodes

    @property
    def match_nodes(self):
        # Compatibility alias: the old "match_nodes" are now the key ExternalMemoryMechanisms.
        if self.concatenate_queries and getattr(self, "concatenated_memory_node", None) is not None:
            return [self.concatenated_memory_node]
        return [field.memory_node for field in self.key_fields]

    @property
    def field_weight_nodes(self):
        return [
            field.weight_node
            for field in self.key_fields
            if field.weight_node is not None
        ]

    @property
    def weighted_scores_nodes(self):
        return [
            field.weighted_scores_node
            for field in self.key_fields
            if field.weighted_scores_node is not None
        ]

    @property
    def retrieved_nodes(self):
        return [field.retrieved_node for field in self.fields]
