# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************** KohoneMechanism ******************************************************

"""

Overview
--------

A KohonenMechanism is a subclass of `RecurrentTransferMechanism` that implements a `Kohonen network
<http://www.scholarpedia.org/article/Kohonen_network>`_ (`brief explanation
<https://www.cs.bham.ac.uk/~jlw/sem2a2/Web/Kohonen.htm>`_; `nice demo <https://www.youtube.com/watch?v=QvI6L-KqsT4>`_),
which is a particular form of `self-organized map (SOM) <https://en.wikipedia.org/wiki/Self-organizing_map>`_. By
default, a KohonenMechanism uses a `KohonenLearningMechanism` and the `Kohonen` `LearningFunction <LearningFunctions>`
to implement implement a form of unsupervised learning that produces the self-organized map.

.. _Kohonen_Creation:

Creating a KohonenMechanism
---------------------------

A KohonenMechanism can be created directly by calling its constructor.

.. _Kohonen_Structure:

Structure
---------

XXX

.. _Kohonen_Learning:


.. _Kohonen_Execution:

Execution
---------

XXX

.. _Kohonen_Reference:

Class Reference
---------------

"""

import itertools
import logging
import numbers
import warnings

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.learningfunctions import Kohonen
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.mechanisms.adaptive.learning.learningmechanism import ACTIVATION_INPUT, ACTIVATION_OUTPUT, LearningMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import DEFAULT_MATRIX, FUNCTION, GAUSSIAN, IDENTITY_MATRIX, INITIALIZING, KOHONEN_MECHANISM, LEARNED_PROJECTION, LEARNING_SIGNAL, MATRIX, MAX_INDICATOR, NAME, OWNER_VALUE, OWNER_VARIABLE, RESULT, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.utilities import is_numeric_or_none, parameter_spec
from psyneulink.library.components.mechanisms.adaptive.learning.kohonenlearningmechanism import KohonenLearningMechanism

__all__ = [
    'KohonenMechanism', 'KohonenError',
]

logger = logging.getLogger(__name__)


class KohonenError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


MAX_ACTIVITY_OUTPUT = 'MAX_ACTIVITY_OUTPUT'
INPUT_PATTERN = 'INPUT_PATTERN'


class KohonenMechanism(TransferMechanism):
    """
    KohonenMechanism(                                      \
    default_variable=None,                                 \
    size=None,                                             \
    function=Linear,                                       \
    matrix=None,                                           \
    integrator_function=AdaptiveIntegrator,                \
    initial_value=None,                                    \
    noise=0.0,                                             \
    integration_rate=1.0,                                  \
    clip=None,                                             \
    enable_learning=True,                                  \
    learning_function=Kohonen(distance_function=GAUSSIAN), \
    learning_rate=None,                                    \
    params=None,                                           \
    name=None,                                             \
    prefs=None)

    Subclass of `TransferMechanism` that learns a `self-organized <https://en.wikipedia.org/wiki/Self-organizing_map>`_
    map of its input.

    Arguments
    ---------

    default_variable : number, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the input to the mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` method;
        also serves as a template to specify the length of `variable <KohonenMechanism.variable>` for
        `function <KohonenMechanism.function>`, and the `primary OutputState <OutputState_Primary>`
        of the mechanism.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : TransferFunction : default Linear
        specifies the function used to transform the input.

    COMMENT:
    selection_function : SelectionFunction, function or method : default OneHot(mode=MAX_VAL)
        specifes the function used to select the element of the input used to train the `matrix
        <MappingProjection.matrix>` of afferent `MappingProjection` to the Mechanism.
    COMMENT

    integrator_function : IntegratorFunction : default AdaptiveIntegrator
        specifies `IntegratorFunction` to use in `integration_mode <KohonenMechanism.integration_mode>`.

    initial_value :  value, list or np.ndarray : default Transfer_DEFAULT_BIAS
        specifies the starting value for time-averaged input (only relevant if
        `integration_rate <KohonenMechanism.integration_rate>` is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        a value added to the result of the `function <KohonenMechanism.function>` or to the result of
        `integrator_function <KohonenMechanism.integrator_function>`, depending on whether `integrator_mode
        <KohonenMechanism.integrator_mode>` is True or False. See `noise <KohonenMechanism.noise>` for more details.

    integration_rate : float : default 0.5
        the smoothing factor for exponential time averaging of input when `integrator_mode
        <KohonenMechanism.integrator_mode>` is set to True ::

         result = (integration_rate * current input) +
         (1-integration_rate * result on previous time_step)

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <KohonenMechanism.function>` the item in index 0 specifies the
        minimum allowable value of the result, and the item in index 1 specifies the maximum allowable value; any
        element of the result that exceeds the specified minimum or maximum value is set to the value of
        `clip <KohonenMechanism.clip>` that it exceeds.

    enable_learning : boolean : default True
        specifies whether the Mechanism should be configured for learning;  if it is not (the default), then learning
        cannot be enabled until it is configured for learning by calling the Mechanism's `configure_learning
        <KohonenMechanism.configure_learning>` method.

    learning_rate : scalar, or list, 1d or 2d np.array, or np.matrix of numeric values: default False
        specifies the learning rate used by its `learning function <KohonenMechanism.learning_function>`.
        If it is `None`, the `default learning_rate for a LearningMechanism <LearningMechanism_Learning_Rate>` is
        used; if it is assigned a value, that is used as the learning_rate (see `learning_rate
        <KohonenMechanism.learning_rate>` for details).

    learning_function : LearningFunction, function or method : default Kohonen(distance_function=GUASSIAN)
        specifies function used by `learning_mechanism <KohonenMechanism.learning_mechanism>` to update `matrix
        <MappingProjection.matrix>` of `learned_projection <KohonenMechanism.learned_projection>.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <Kohonen Mechanism.name>`
        specifies the name of the Kohonen Mechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the Kohonen Mechanism; see `prefs <Kohonen Mechanism.prefs>` for details.

    context : str : default componentType+INITIALIZING
        string used for contextualization of instantiation, hierarchical calls, executions, etc.

    Attributes
    ----------

    variable : value
        the input to Mechanism's `function <KohonenMechanism.variable>`.

    function : Function
        the Function used to transform the input.

    COMMENT:
    selection_function : SelectionFunction, function or method : default OneHot(mode=MAX_VAL)
        determines the function used to select the element of the input used to train the `matrix
        <MappingProjection.matrix>` of the `learned_projection <KohonenMechanism.learned_projection>`.
    COMMENT

    distance_function : Function, function or method : default Gaussian
        determines the function used to evaluate the distance of each element from the most active one
        one identified by `selection_function <KohonenMechanism.selection_function>`

    matrix : 2d np.array
        `matrix <AutoAssociativeProjection.matrix>` parameter of the `learned_projection
        <KohonenMechanism.learned_projection>`.

    learning_enabled : bool
        indicates whether `learning is enabled <Kohonen_Learning>`;  see `learning_enabled
        <KohonenMechanism.learning_enabled>` for additional details.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        determines the learning rate used by the `learning_function <KohonenMechanism.learning_function>`
        of the `learning_mechanism <KohonenMechanism.learning_mechanism>` (see `learning_rate
        <KohonenLearningMechanism.learning_rate>` for details concerning specification and default value assignment).

    learned_projection : MappingProjection
        `MappingProjection` that projects to the Mechanism and is trained by its `learning_mechanism
        <KohonenMechanism.learning_mechanism>`.

    learning_function : LearningFunction, function or method
        function used by `learning_mechanism <KohonenMechanism.learning_mechanism>` to update `matrix
        <MappingProjection.matrix>` of `learned_projection <KohonenMechanism.learned_projection>.

    learning_mechanism : LearningMechanism
        created automatically if `learning is specified <KohonenMechanism_Learning>`, and used to train the
        `learned_projection <KohonenMechanism.learned_projection>`.

    integrator_function :  IntegratorFunction
        the `IntegratorFunction` used when `integrator_mode <KohonenMechanism.integrator_mode>` is set to
        `True` (see `integrator_mode <KohonenMechanism.integrator_mode>` for details).

        .. note::
            The KohonenMechanism's `integration_rate <KohonenMechanism.integration_rate>`, `noise
            <KohonenMechanism.noise>`, and `initial_value <KohonenMechanism.initial_value>` parameters
            specify the respective parameters of its `integrator_function` (with **initial_value** corresponding
            to `initializer <IntegratorFunction.initializer>` of integrator_function.

    initial_value :  value, list or np.ndarray : Transfer_DEFAULT_BIAS
        determines the starting value for time-averaged input
        (only relevant if `integration_rate <KohonenMechanism.integration_rate>` parameter is not 1.0).
        COMMENT:
            Transfer_DEFAULT_BIAS SHOULD RESOLVE TO A VALUE
        COMMENT

    noise : float or function : default 0.0
        When `integrator_mode <KohonenMechanism.integrator_mode>` is set to True, noise is passed into the `integrator_function
        <KohonenMechanism.integrator_function>`. Otherwise, noise is added to the output of the `function <KohonenMechanism.function>`.

        If noise is a list or array, it must be the same length as `variable <KohonenMechanism.default_variable>`.

        If noise is specified as a single float or function, while `variable <KohonenMechanism.variable>` is a list or array,
        noise will be applied to each variable element. In the case of a noise function, this means that the function
        will be executed separately for each variable element.

        .. note::
            In order to generate random noise, we recommend selecting a probability distribution function
            (see `Distribution Functions <DistributionFunction>` for details), which will generate a new noise value from
            its distribution on each execution. If noise is specified as a float or as a function with a fixed output, then
            the noise will simply be an offset that remains the same across all executions.

    integration_rate : float : default 0.5
        the smoothing factor for exponential time averaging of input when `integrator_mode <KohonenMechanism.integrator_mode>` is set
        to True::

          result = (integration_rate * current input) + (1-integration_rate * result on previous time_step)

    integrator_function:
        When *integrator_mode* is set to True, the KohonenMechanism executes its `integrator_function
        <KohonenMechanism.integrator_function>`, which is the `AdaptiveIntegrator`. See `AdaptiveIntegrator
        <AdaptiveIntegrator>` for more details on what it computes. Keep in mind that the `integration_rate
        <KohonenMechanism.integration_rate>` parameter of the KohonenMechanism corresponds to the `rate
        <IntegratorFunction.rate>` of the `integrator_function <KohonenMechanism.integrator_function>`.

    integrator_mode:
        **When integrator_mode is set to True:**

        the variable of the mechanism is first passed into the following equation:

        .. math::
            value = previous\\_value(1-smoothing\\_factor) + variable \\cdot smoothing\\_factor + noise

        The result of the integrator function above is then passed into the `mechanism's function <KohonenMechanism.function>`. Note that
        on the first execution, *initial_value* sets previous_value.

        **When integrator_mode is set to False:**

        The variable of the mechanism is passed into the `function of the mechanism <KohonenMechanism.function>`. The mechanism's
        `integrator_function <KohonenMechanism.integrator_function>` is skipped entirely, and all related arguments (*noise*, *leak*,
        *initial_value*, and *time_step_size*) are ignored.

    clip : list [float, float] : default None (Optional)
        specifies the allowable range for the result of `function <KohonenMechanism.function>`

        the item in index 0 specifies the minimum allowable value of the result, and the item in index 1 specifies the
        maximum allowable value; any element of the result that exceeds the specified minimum or maximum value is set to
         the value of `clip <KohonenMechanism.clip>` that it exceeds.

    previous_input : 1d np.array of floats
        the value of the input on the previous execution, including the value of `recurrent_projection`.

    value : 2d np.array [array(float64)]
        result of executing `function <KohonenMechanism.function>`; same value as first item of
        `output_values <KohonenMechanism.output_values>`.

    COMMENT:
        CORRECTED:
        value : 1d np.array
            the output of ``function``;  also assigned to ``value`` of the TRANSFER_RESULT OutputState
            and the first item of ``output_values``.
    COMMENT

    output_states : Dict[str, OutputState]
        an OrderedDict with the following `OutputStates <OutputState>`:

        * `TRANSFER_RESULT`, the `value <OutputState.value>` of which is the **result** of `function
          <KohonenMechanism.function>`;

        * `MOST_ACTIVE`, the `value <OutputState.value>` of which is a "one hot" encoding of the most active
          element of the Mechanism's `value <KohonenMechanism.value>` in the last execution (used by the
          `learning_mechanism <KohonenMechanisms.learning_mechanism>` to modify the `learned_projection
          <KohonenMechanism.learned_projection>`.

    output_values : List[array(float64), array(float64)]
        a list with the `value <OutputState.value>` of each of the Mechanism's `output_states
        <KohonenMechanism.output_states>`.

    name : str
        the name of the Kohonen Mechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Kohonen Mechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    Returns
    -------
    instance of KohonenMechanism : KohonenMechanism

    """

    componentType = KOHONEN_MECHANISM

    class Parameters(TransferMechanism.Parameters):
        """
            Attributes
            ----------

                enable_learning
                    see `enable_learning <KohonenMechanism.enable_learning>`

                    :default value: True
                    :type: bool

                learning_function
                    see `learning_function <KohonenMechanism.learning_function>`

                    :default value: `Kohonen`
                    :type: `Function`

                learning_rate
                    see `learning_rate <KohonenMechanism.learning_rate>`

                    :default value: None
                    :type:

                matrix
                    see `matrix <KohonenMechanism.matrix>`

                    :default value: `AUTO_ASSIGN_MATRIX`
                    :type: str

        """
        learning_function = Parameter(Kohonen(distance_function=GAUSSIAN), stateful=False, loggable=False)

        learning_rate = Parameter(None, modulable=True)

        enable_learning = True
        matrix = DEFAULT_MATRIX


    paramClassDefaults = TransferMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({'function': Linear})  # perhaps hacky? not sure (7/10/17 CW)

    standard_output_states = TransferMechanism.standard_output_states.copy()
    standard_output_states.extend([{NAME:MAX_ACTIVITY_OUTPUT,
                                    VARIABLE:(OWNER_VALUE,0),
                                    FUNCTION: OneHot(mode=MAX_INDICATOR)}
                                   ])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 function=Linear,
                 # selection_function=OneHot(mode=MAX_INDICATOR),  # RE-INSTATE WHEN IMPLEMENT NHot function
                 integrator_function=AdaptiveIntegrator,
                 initial_value=None,
                 noise: is_numeric_or_none = 0.0,
                 integration_rate: is_numeric_or_none = 0.5,
                 integrator_mode=False,
                 clip=None,
                 enable_learning=True,
                 learning_rate:tc.optional(tc.any(parameter_spec, bool))=None,
                 learning_function:is_function_type=Kohonen(distance_function=GAUSSIAN),
                 learned_projection:tc.optional(MappingProjection)=None,
                 additional_output_states:tc.optional(tc.any(str, Iterable))=None,
                 name=None,
                 prefs: is_pref_set = None,
                 **kwargs
                 ):
        # # Default output_states is specified in constructor as a string rather than a list
        # # to avoid "gotcha" associated with mutable default arguments
        # # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        # if output_states is None:
        #     output_states = [RESULT]

        output_states = [RESULT, {NAME: INPUT_PATTERN, VARIABLE: OWNER_VARIABLE}]
        if additional_output_states:
            if isinstance(additional_output_states, list):
                output_states += additional_output_states
            else:
                output_states.append(additional_output_states)

        self._learning_enabled = enable_learning
        self._learning_enable_deferred = False

        params = self._assign_args_to_param_dicts(
                integrator_mode=integrator_mode,
                learning_rate=learning_rate,
                learning_function=learning_function,
                learned_projection=learned_projection,
                enable_learning=enable_learning,
                output_states=output_states)

        super().__init__(default_variable=default_variable,
                         size=size,
                         function=function,
                         integrator_function=integrator_function,
                         integrator_mode=integrator_mode,
                         initial_value=initial_value,
                         noise=noise,
                         integration_rate=integration_rate,
                         clip=clip,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    def _validate_params(self, request_set, target_set=None, context=None):
        super()._validate_params(request_set, target_set, context)

        if LEARNED_PROJECTION in target_set and target_set[LEARNED_PROJECTION]:
            if target_set[LEARNED_PROJECTION].receiver.owner != self:
                raise KohonenError("{} specified in {} argument for {} projects to a different Mechanism ({})".
                                   format(MappingProjection.__name__,
                                          repr(LEARNED_PROJECTION), self.name,
                                          target_set[LEARNED_PROJECTION].receiver.owner.name))

    def _instantiate_attributes_after_function(self, context=None):
        super()._instantiate_attributes_after_function(context)
        if self.learning_enabled:
            self.configure_learning(context=context)

    # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
    @handle_external_context()
    def configure_learning(self,
                           learning_function:tc.optional(tc.any(is_function_type))=None,
                           learning_rate:tc.optional(tc.any(numbers.Number, list, np.ndarray, np.matrix))=None,
                           learned_projection:tc.optional(MappingProjection)=None,
                           context=None):
        """Provide user-accessible-interface to _instantiate_learning_mechanism

        Configure KohonenMechanism for learning. Creates the following Components:

        * a `LearningMechanism` -- if the **learning_function** and/or **learning_rate** arguments are
          specified, they are used to construct the LearningMechanism, otherwise the values specified in the
          KohonenMechanism's constructor are used;
        ..
        * a `MappingProjection` from the KohonenMechanism's `primary OutputState <OutputState_Primary>`
          to the LearningMechanism's *ACTIVATION_INPUT* InputState;
        ..
        * a `LearningProjection` from the LearningMechanism's *LEARNING_SIGNAL* OutputState to the learned_projection;
          by default this is the KohonenMechanism's `learned_projection <KohonenMechanism.learned_projection>`;
          however a different one can be specified.

        """
        # This insures that these are validated if the method is called from the command line (i.e., by the user)
        if learning_function:
            self.learning_function = learning_function
        if learning_rate:
            self.learning_rate = learning_rate
        if learned_projection:
            self.learned_projection = learned_projection

        # Assign learned_projection, using as default the first Projection to the Mechanism's primary InputState
        try:
            self.learned_projection = self.learned_projection or self.input_state.path_afferents[0]
        except:
            self.learned_projection = None
        if not self.learned_projection:
            # Mechanism already belongs to a Process or System, so should have a MappingProjection by now
            if (self.processes or self.systems):
                raise KohonenError("Configuring learning for {} requires that it receive a {} "
                                   "from another {} within a {} to which it belongs".
                                   format(self.name, MappingProjection.__name__, Mechanism.__name__, Process.__name__))
                                   # "receive at least one {} or that the {} be specified".
                                   # format(self.name, MappingProjection.__name__, repr(LEARNED_PROJECTION)))
            # Mechanism doesn't yet belong to a Process or System, so wait until then to configure learning
            #  (this method will be called again from _add_projection_to_mechanism if a Projection is added)
            else:
                self._learning_enable_deferred = True
                return

        self.matrix = self.learned_projection.parameter_states[MATRIX]

        self.learning_mechanism = self._instantiate_learning_mechanism(learning_function=self.learning_function,
                                                                       learning_rate=self.learning_rate,
                                                                       learned_projection=self.learned_projection,
                                                                       context=context)

        self.learning_projection = self.learning_mechanism.output_states[LEARNING_SIGNAL].efferents[0]

        if self.learning_mechanism is None:
            self.learning_enabled = False

    # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION WHEN THAT IS IMPLEMENTED
    def _instantiate_learning_mechanism(self,
                                        learning_function,
                                        learning_rate,
                                        learned_projection,
                                        context=None):

        learning_mechanism = KohonenLearningMechanism(default_variable=[self.learned_projection.sender.value,
                                                                        self.learned_projection.receiver.value],
                                                      matrix=self.matrix,
                                                      function=learning_function,
                                                      learning_rate=learning_rate,
                                                      # learning_signals=[self.matrix],
                                                      name="{} for {}".format(
                                                              LearningMechanism.className,
                                                              self.name))

        # KDM 10/22/18: should below be aux_components?

        # Instantiate Projection from learned_projection's sender to LearningMechanism
        MappingProjection(sender=self.learned_projection.sender,
                          receiver=learning_mechanism.input_states[ACTIVATION_INPUT],
                          matrix=IDENTITY_MATRIX,
                          name="Error Projection for {}".format(learning_mechanism.name))

        # Instantiate Projection from learned_projection's receiver (Mechanism's input) to LearningMechanism
        MappingProjection(sender=self.output_states[INPUT_PATTERN],
                          receiver=learning_mechanism.input_states[ACTIVATION_OUTPUT],
                          matrix=IDENTITY_MATRIX,
                          name="Error Projection for {}".format(learning_mechanism.name))

        # Instantiate Projection from LearningMechanism to learned_projection
        LearningProjection(sender=learning_mechanism.output_states[LEARNING_SIGNAL],
                           receiver=self.matrix,
                           name="{} for {}".format(LearningProjection.className, self.learned_projection.name))

        return learning_mechanism

    def _projection_added(self, projection, context=None):
        super()._projection_added(projection, context)
        if self._learning_enable_deferred:
            self.configure_learning(context=context)

    @property
    def learning_enabled(self):
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value:bool):

        self._learning_enabled = value
        # Enable learning for KohonenMechanism's learning_mechanism
        if hasattr(self, 'learning_mechanism'):
            self.learning_mechanism.learning_enabled = value
        # If KohonenMechanism has no LearningMechanism, warn and then ignore attempt to set learning_enabled
        elif value is True:
            warnings.warn("Learning cannot be enabled for {} because it has no {}".
                  format(self.name, LearningMechanism.__name__))
            return

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            [self.learning_mechanism],
            [self.learning_projection],
        ))
