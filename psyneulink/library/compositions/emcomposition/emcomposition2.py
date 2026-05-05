# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition2 *************************************************

# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* EMComposition2 *************************************************

"""
Refactored EMComposition prototype.

This module introduces EpisodicMemoryFieldMechanism, a field-local episodic memory mechanism
that owns the memory matrix for a single field.  The refactored EMComposition
uses one EpisodicMemoryFieldMechanism per memory field instead of using EMStorageMechanism
to update MappingProjection matrices.

High-level execution per field:

1. QUERY input is sent to EpisodicMemoryFieldMechanism.input_port[QUERY].
2. EpisodicMemoryFieldMechanism computes a match-weight vector over its memory rows and emits SCORES.
3. SCORES from key fields are weighted and combined by COMBINE MATCHES.
4. The combined match vector is softmax-normalized by RETRIEVE.
5. The normalized combined scores are sent back to each EpisodicMemoryFieldMechanism.input_port[COMBINED_SCORES].
6. Each EpisodicMemoryFieldMechanism retrieves its field value and emits RETRIEVED.
7. Each EpisodicMemoryFieldMechanism stores its QUERY input into its own memory matrix according to storage_prob.
"""

import copy
import warnings
from enum import Enum
from typing import Optional, Union

import numpy as np

import psyneulink.core.scheduling.condition as conditions

from psyneulink._typing import Literal
from psyneulink.core.components.functions.function import DEFAULT_SEED, _random_state_getter, _seed_setter
from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.transformfunctions import LinearCombination
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
    EM_COMPOSITION,
    FIRST,
    FULL_CONNECTIVITY_MATRIX,
    GAIN,
    IDENTITY_MATRIX,
    INPUT_SHAPES,
    LAST,
    MULTIPLICATIVE_PARAM,
    NAME,
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
from psyneulink.core.llvm import ExecutionMode
from psyneulink.library.components.mechanisms.processing.integrator.episodicmemoryfieldmechanism import (
    EpisodicMemoryFieldMechanism, QUERY, SCORES, COMBINED_SCORES, RETRIEVED)
from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition, torch_available


__all__ = [
    "EMComposition2",
    "EMComposition2Error",
    "FieldType",
    "FIELD_WEIGHT",
    "KEY",
    "LEARN_FIELD_WEIGHT",
    "PROBABILISTIC",
    "TARGET_FIELD",
    "WEIGHTED_AVG",
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
WEIGHTED_MATCH_NODE_NAME = "WEIGHTED MATCH"
WEIGHTED_MATCH_AFFIX = f" [{WEIGHTED_MATCH_NODE_NAME}]"
COMBINE_MATCHES_NODE_NAME = "COMBINE MATCHES"
SOFTMAX_NODE_NAME = "RETRIEVE"
RETRIEVED_NODE_NAME = "RETRIEVED"
RETRIEVED_AFFIX = " [RETRIEVED]"


class EMComposition2Error(CompositionError):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class FieldType(Enum):
    KEY = 0
    VALUE = 1


def _normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


# def _field_memory_getter(owning_component=None, context=None):
#     if owning_component is None or owning_component.is_initializing:
#         return None
#     return owning_component.parameters.field_memory._get(context)


def _memory_getter(owning_component=None, context=None):
    """Return EMComposition memory as a 3d object array: entries x fields x field_values."""
    if owning_component is None or owning_component.is_initializing:
        return None

    field_memories = [
        field.field_mechanism.parameters.field_memory.get(context)
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
                    f"Field '{owning_component.field_names[i]}' of '{owning_component.name}' "
                    f"was originally assigned as a value field; it cannot be changed to a key field "
                    f"after construction."
                )
            continue

        owning_component.field_weight_nodes[field_wt_node_idx].input_port.defaults.variable = field_weight
        owning_component.fields[i].weight = field_weight
        field_wt_node_idx += 1

    return np.array(field_weights, dtype=object)


def get_softmax_gain(v, scale=1, base=1, entropy_weighting=.1) -> float:
    v = np.squeeze(v)
    logistic = 1 / (1 + np.exp(-1 * v))
    entropy = -1 * np.sum(logistic * np.log(logistic))
    return scale * (base + entropy_weighting * np.log(entropy))


class Field:
    """Object that contains information about a field in an EMComposition's memory."""

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
        self.field_mechanism = None
        self.weight_node = None
        self.weighted_match_node = None
        self.retrieved_node = None

        self.query_projection = None
        self.scores_projection = None
        self.combined_scores_projection = None
        self.retrieved_projection = None
        self.weight_projection = None
        self.weighted_match_projection = None

    @property
    def nodes(self):
        return [
            node for node in [
                self.input_node,
                self.field_mechanism,
                self.weight_node,
                self.weighted_match_node,
                self.retrieved_node,
            ]
            if node is not None
        ]

    @property
    def projections(self):
        return [
            proj for proj in [
                self.query_projection,
                self.scores_projection,
                self.combined_scores_projection,
                self.retrieved_projection,
                self.weight_projection,
                self.weighted_match_projection,
            ]
            if proj is not None
        ]

    @property
    def query(self):
        return self.input_node.variable

    @property
    def match(self):
        return self.field_mechanism.output_ports[SCORES].value

    @property
    def retrieved_memory(self):
        return self.field_mechanism.output_ports[RETRIEVED].value

    @property
    def memories(self):
        return self.field_mechanism.memory


class EMComposition2(AutodiffComposition):
    """
    Refactored EMComposition.

    This version replaces:
      - match_nodes backed by memory Projection matrices
      - retrieved_nodes backed by memory Projection matrices
      - EMStorageMechanism

    with:
      - one EpisodicMemoryFieldMechanism per field, each owning its field memory matrix.

    The externally visible structure is kept similar to the original EMComposition:
      - input_nodes
      - query_input_nodes
      - value_input_nodes
      - field_weight_nodes
      - weighted_match_nodes
      - combined_matches_node
      - softmax_node
      - retrieved_nodes

    Internally, field.field_mechanism is now the memory owner for each field.
    """

    componentCategory = EM_COMPOSITION

    if torch_available:
        from psyneulink.library.compositions.emcomposition.pytorchEMwrappers import (
            PytorchEMCompositionWrapper,
            PytorchEMMechanismWrapper,
        )
        pytorch_composition_wrapper_type = PytorchEMCompositionWrapper
        pytorch_mechanism_wrapper_type = PytorchEMMechanismWrapper

    class Parameters(AutodiffComposition.Parameters):
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
        storage_prob = Parameter(1.0, modulable=True, aliases=[MULTIPLICATIVE_PARAM])
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

        memory_template, memory_capacity = self._parse_memory_template(
            memory_template,
            memory_capacity,
            memory_fill,
        )

        self.fields = ContentAddressableList(component_type=Field)
        self.entry_template = memory_template[0]

        (
            field_names,
            field_weights,
            learn_field_weights,
            target_fields,
            concatenate_queries,
        ) = self._parse_fields(
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
        )

        if memory_decay_rate is AUTO:
            memory_decay_rate = 1 / memory_capacity

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

        self._set_learning_attributes()

        for field_mechanism in self.field_mechanisms:
            self.scheduler.add_condition(field_mechanism, conditions.AfterNodes(self.softmax_node))

        for node in self.value_input_nodes:
            node.output_port.parameters.require_projection_in_composition.set(False, override=True)

        self.softmax_node.output_port.parameters.require_projection_in_composition.set(False, override=True)

        for node in self.field_weight_nodes:
            self.exclude_node_roles(node, NodeRole.INPUT)

        for node in self.value_input_nodes:
            self.exclude_node_roles(node, NodeRole.OUTPUT)

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
                    f"The entries in 'field_weights' arg for {name} can't all be None."
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
                            np.random.uniform(memory_fill[0], memory_fill[1], len(field)),
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
                    f"The number of entries ({len(fields_dict)}) in 'fields' for '{name}' "
                    f"does not match the number of fields ({num_fields})."
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
                f"The 'learning_rate' arg for '{name}' cannot be a dict; use 'fields' or "
                f"'learn_field_weights' instead."
            )

        if fields:
            if any([field_names, field_weights, learn_field_weights, target_fields]):
                warnings.warn(
                    f"The 'fields' arg for '{name}' was specified, so 'field_names', "
                    f"'field_weights', 'learn_field_weights', and 'target_fields' will be ignored."
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
                            f"A learning_rate was specified for value field '{field_names[i] if field_names else i}' "
                            f"in '{name}', but value fields do not have learnable field weights; ignored."
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

        if concatenate_queries:
            warnings.warn(
                f"The refactored EMComposition prototype does not yet support 'concatenate_queries=True'; "
                f"it will be ignored."
            )
            concatenate_queries = False

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
        self._construct_field_mechanisms(
            memory_template,
            memory_capacity,
            normalize_memories,
            storage_prob,
            memory_decay_rate,
        )
        self._construct_field_weight_nodes(use_gating_for_weighting)
        self._construct_weighted_match_nodes()
        self._construct_combined_matches_node(memory_capacity, use_gating_for_weighting)
        self._construct_softmax_node(memory_capacity, softmax_gain, softmax_threshold, softmax_choice)
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

        if not self.enable_learning:
            self.add_nodes(self.input_nodes, context=context)
            self.add_nodes(self.field_mechanisms, context=context)
            self.add_nodes(self.field_weight_nodes + self.weighted_match_nodes, context=context)
            if self.combined_matches_node:
                self.add_node(self.combined_matches_node, context=context)
            self.add_nodes([self.softmax_node] + self.retrieved_nodes, context=context)
            if self.softmax_gain_control_node:
                self.add_node(self.softmax_gain_control_node, context=context)
            return

        for field in self.fields:
            self.add_linear_processing_pathway([
                field.input_node,
                field.field_mechanism,
            ])

        if self.num_keys == 1:
            self.add_linear_processing_pathway([
                self.key_fields[0].field_mechanism,
                self.softmax_node,
            ])
        else:
            for field in self.key_fields:
                pathway = [
                    field.field_mechanism,
                    self.combined_matches_node,
                ]
                if field.weighted_match_node:
                    pathway.insert(1, field.weighted_match_node)
                self.add_linear_processing_pathway(pathway)

            self.add_linear_processing_pathway([
                self.combined_matches_node,
                self.softmax_node,
            ])

        for field in self.fields:
            self.add_linear_processing_pathway([
                self.softmax_node,
                field.field_mechanism,
                field.retrieved_node,
            ])

        if self.softmax_gain_control_node:
            self.add_node(self.softmax_gain_control_node, context=context)

        for field in self.key_fields:
            if field.weight_node and field.weighted_match_node:
                self.add_linear_processing_pathway([
                    field.weight_node,
                    field.weighted_match_node,
                ])

    def _construct_input_nodes(self):
        for field in self.key_fields:
            field.input_node = ProcessingMechanism(
                name=f"{field.name} [QUERY]",
                input_shapes=len(self.entry_template[field.index]),
            )
            field.type = FieldType.KEY

        for field in self.value_fields:
            field.input_node = ProcessingMechanism(
                name=f"{field.name} [VALUE]",
                input_shapes=len(self.entry_template[field.index]),
            )
            field.type = FieldType.VALUE

    def _construct_field_mechanisms(
        self,
        memory_template,
        memory_capacity,
        normalize_memories,
        storage_prob,
        memory_decay_rate,
    ):
        for field in self.fields:
            field_memory = np.array(memory_template[:, field.index].tolist()).astype(float)

            field.field_mechanism = EpisodicMemoryFieldMechanism(
                field_shape=len(self.entry_template[field.index]),
                field_memory=field_memory,
                storage_prob=storage_prob,
                decay_rate=memory_decay_rate,
                normalize_memories=normalize_memories,
                name=f"{field.name}{FIELD_MEMORY_AFFIX}",
            )

            field.query_projection = MappingProjection(
                sender=field.input_node,
                receiver=field.field_mechanism.input_ports[QUERY],
                matrix=IDENTITY_MATRIX,
                name=f"{field.name} QUERY to FIELD MEMORY",
            )

    def _construct_field_weight_nodes(self, use_gating_for_weighting):
        if self.num_keys <= 1:
            return

        for field in self.key_fields:
            name = f"{field.name}{WEIGHT_AFFIX}"
            variable = np.array(field.weight)
            params = {DEFAULT_INPUT: DEFAULT_VARIABLE}

            if use_gating_for_weighting:
                field.weight_node = GatingMechanism(
                    name=name,
                    input_ports={
                        NAME: "OUTCOME",
                        VARIABLE: variable,
                        PARAMS: params,
                    },
                    gate=field.field_mechanism.output_ports[SCORES],
                )
            else:
                field.weight_node = ProcessingMechanism(
                    name=name,
                    input_ports={
                        NAME: "FIELD_WEIGHT",
                        VARIABLE: variable,
                        PARAMS: params,
                    },
                )

    def _construct_weighted_match_nodes(self):
        if self.num_keys <= 1:
            return

        for field in self.key_fields:
            field.weighted_match_node = ProcessingMechanism(
                name=field.name + WEIGHTED_MATCH_AFFIX,
                default_variable=[
                    np.zeros(self.memory_capacity),
                    np.zeros(self.memory_capacity),
                ],
                input_ports=[
                    {
                        PROJECTIONS: MappingProjection(
                            name=f"{field.name} SCORES to WEIGHTED MATCH",
                            sender=field.field_mechanism.output_ports[SCORES],
                            matrix=IDENTITY_MATRIX,
                        )
                    },
                    {
                        PROJECTIONS: MappingProjection(
                            name=f"{field.name} WEIGHT to WEIGHTED MATCH",
                            sender=field.weight_node,
                            matrix=FULL_CONNECTIVITY_MATRIX,
                        )
                    },
                ],
                function=LinearCombination(operation=PRODUCT),
            )
            field.scores_projection = field.weighted_match_node.path_afferents[0]
            field.weight_projection = field.weighted_match_node.path_afferents[1]

    def _construct_combined_matches_node(self, memory_capacity, use_gating_for_weighting):
        if self.num_keys == 1:
            self.combined_matches_node = None
            return

        input_source = (
            [field.field_mechanism.output_ports[SCORES] for field in self.key_fields]
            if use_gating_for_weighting
            else self.weighted_match_nodes
        )

        self.combined_matches_node = ProcessingMechanism(
            name=COMBINE_MATCHES_NODE_NAME,
            input_ports=[
                {
                    INPUT_SHAPES: memory_capacity,
                    PROJECTIONS: [
                        MappingProjection(
                            sender=source,
                            matrix=IDENTITY_MATRIX,
                            name=f"{self.key_fields[i].name} to {COMBINE_MATCHES_NODE_NAME}",
                        )
                        for i, source in enumerate(input_source)
                    ],
                }
            ],
        )

        for i, proj in enumerate(self.combined_matches_node.path_afferents):
            self.key_fields[i].weighted_match_projection = proj

    def _construct_softmax_node(self, memory_capacity, softmax_gain, softmax_threshold, softmax_choice):
        if self.num_keys == 1:
            input_source = self.key_fields[0].field_mechanism.output_ports[SCORES]
            proj_name = f"{self.key_fields[0].name} SCORES to {SOFTMAX_NODE_NAME}"
        else:
            input_source = self.combined_matches_node
            proj_name = f"{COMBINE_MATCHES_NODE_NAME} to {SOFTMAX_NODE_NAME}"

        if softmax_choice == ARG_MAX:
            softmax_choice = ARG_MAX_INDICATOR

        self.softmax_node = ProcessingMechanism(
            name=SOFTMAX_NODE_NAME,
            input_ports={
                INPUT_SHAPES: memory_capacity,
                PROJECTIONS: MappingProjection(
                    sender=input_source,
                    matrix=IDENTITY_MATRIX,
                    name=proj_name,
                ),
            },
            function=SoftMax(
                gain=softmax_gain,
                mask_threshold=softmax_threshold,
                output=softmax_choice,
                adapt_entropy_weighting=.95,
            ),
        )

    def _construct_softmax_gain_control_node(self, softmax_gain):
        node = None
        if softmax_gain == CONTROL:
            node = ControlMechanism(
                name="SOFTMAX GAIN CONTROL",
                monitor_for_control=self.combined_matches_node or self.key_fields[0].field_mechanism,
                control_signals=[(GAIN, self.softmax_node)],
                function=get_softmax_gain,
            )
        self.softmax_gain_control_node = node

    def _construct_retrieved_nodes(self):
        for field in self.fields:
            field.combined_scores_projection = MappingProjection(
                sender=self.softmax_node,
                receiver=field.field_mechanism.input_ports[COMBINED_SCORES],
                matrix=IDENTITY_MATRIX,
                name=f"{SOFTMAX_NODE_NAME} to {field.name} COMBINED_SCORES",
            )

            field.retrieved_node = ProcessingMechanism(
                name=field.name + RETRIEVED_AFFIX,
                input_ports={
                    INPUT_SHAPES: len(field.input_node.variable[0]),
                    PROJECTIONS: MappingProjection(
                        sender=field.field_mechanism.output_ports[RETRIEVED],
                        matrix=IDENTITY_MATRIX,
                        name=f"{field.name} RETRIEVED to OUTPUT",
                    ),
                },
            )
            field.retrieved_projection = field.retrieved_node.path_afferents[0]

    def _set_learning_attributes(self):
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
                f"'enable_learning=True'; use WEIGHTED_AVG during learning."
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

        if softmax_choice in {ARG_MAX, PROBABILISTIC}:
            raise EMComposition2Error(
                f"The ARG_MAX and PROBABILISTIC options for 'softmax_choice' of '{self.name}' "
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

    def _identify_target_nodes(self, context) -> list:
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
        elif isinstance(target_fields, list):
            target_nodes = [
                node for node in self.retrieved_nodes
                if target_fields[self.retrieved_nodes.index(node)]
            ]
        else:
            assert False, (
                f"PROGRAM ERROR: target_fields arg for {self.name}: {target_fields} "
                f"is neither True, False, nor a list of bools."
            )

        super()._identify_target_nodes(context)
        return target_nodes

    def infer_backpropagation_learning_pathways(self, execution_mode, context=None):
        return super().infer_backpropagation_learning_pathways(execution_mode, context=context)

    def do_gradient_optimization(self, retain_in_pnl_options, context, optimization_num=None):
        # EM storage is field-local and executed by EpisodicMemoryFieldMechanism after retrieval.
        # Field-weight learning can be restored by calling super() once the PyTorch wrapper
        # supports EpisodicMemoryFieldMechanism as a differentiable memory component.
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
    def field_mechanisms(self):
        return [field.field_mechanism for field in self.fields]

    @property
    def match_nodes(self):
        # Compatibility alias: the old "match_nodes" are now the key EpisodicMemoryFieldMechanisms.
        return [field.field_mechanism for field in self.key_fields]

    @property
    def field_weight_nodes(self):
        return [
            field.weight_node
            for field in self.key_fields
            if field.weight_node is not None
        ]

    @property
    def weighted_match_nodes(self):
        return [
            field.weighted_match_node
            for field in self.key_fields
            if field.weighted_match_node is not None
        ]

    @property
    def retrieved_nodes(self):
        return [field.retrieved_node for field in self.fields]