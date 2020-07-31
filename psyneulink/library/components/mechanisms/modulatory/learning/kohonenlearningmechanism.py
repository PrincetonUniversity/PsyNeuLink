# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  KohonenLearningMechanism **********************************************

"""

Contents
--------

  * `KohonenLearningMechanism_Overview`
  * `KohonenLearningMechanism_Creation`
  * `KohonenLearningMechanism_Structure`
  * `KohonenLearningMechanism_Execution`
  * `KohonenLearningMechanism_Class_Reference`


.. _KohonenLearningMechanism_Overview:

Overview
--------

An KohonenLearningMechanism is a subclass of `LearningMechanism <LearningMechanism>`, modified for use with a
`KohonenMechanism`.  It implements a form of unsupervised learning used to train a `MappingProjection` to the
KohonenMechanism, to implement a `self-organized map (SOM) <https://en.wikipedia.org/wiki/Self-organizing_map>`_.

.. _KohonenLearningMechanism_Creation:

Creating a KohonenLearningMechanism
-----------------------------------

An KohonenLearningMechanism can be created directly by calling its constructor, but most commonly it is
created automatically when a KohonenMechanism is `configured for learning <KohonenMechanism_Learning>`.

.. _KohonenLearningMechanism_Structure:

Structure
---------

An KohonenLearningMechanism is identical to a `LearningMechanism` in all respects except the following:

  * its *ACTIVATION_INPUT* InputPort receives its `Projection` from the `sender <MappingProjection.sender>` of the
    `learned_projection <KohonenMechanism.learned_projection>` for the `KohonenMechanisms with which it is associated
    (identified by the `activity_source <KohonenLearningMechanism.activity_source>` attribute);  and its
    *ACTIVATION_OUTPUT* InputPort receives its input from the `receiver <MappingProjection.receiver>` of the
    `learned_projection <KohonenMechanism.learned_projection>`. It does not have a *LEARNING_SIGNAL* `InputPort`
    (since it implements a form of unsupervised learning).

  * it has a single *LEARNING_SIGNAL* `OutputPort` that sends a `LearningProjection` to the `matrix
    <MappingProjection.matrix>` parameter of the KohonenMechanism's `learned_projection
    <KohonenMechanism.learned_projection>`.

  * it has no `input_source <LearningMechanism.input_source>`, `output_source <LearningMechanism.output_source>`,
    or `error_source <LearningMechanism.error_source>` attributes;  instead, it has a single `matrix
    <KohonenLearningMechanism.matrix>` attribute that identifies the `matrix <MappingProjection.matrix>` parameter of
    the KohonenMechanism's `learned_projection <KohonenMechanism.learned_projection>`.

  * its `function <KohonenLearningMechanism.function>` takes as its `variable <Function_Base.variable>`
    a list containing two 1d arrays (the `value <InputPort.value>` of its *ACTIVATION_INPUT* and *ACTIVATION_OUTPUT*
    InputPorts) and a 2d array (the current weight matrix of its `matrix <KohonenLearningMechanism.matrix>`
    attribute), and it returns a `learning_signal <LearningMechanism.learning_signal>`
    (a weight change matrix assigned to the Mechanism's *LEARNING_SIGNAL* OutputPort), but not an `error_signal
    <LearningMechanism.error_signal>`.

  * its `learning_rate <KohonenLearningMechanism.learning_rate>` can be specified as a 1d or 2d array (or
    matrix) to scale the contribution made, respectively, by individual elements or connections among them,
    to the weight change matrix;  as with a standard `LearningMechanism`, a scalar can also be specified to scale
    the entire weight change matrix (see `learning_rate <KohonenLearningMechanism.learning_rate>` for
    additional details).

.. _KohonenLearningMechanism_Execution:

Execution
---------

An KohonenLearningMechanism executes in the same manner as standard `LearningMechanism`, with two exceptions:
* 1) its execution can be enabled or disabled by setting the `learning_enabled
  <KohonenMechanism.learning_enabled>` attribute of the `KohonenMechanism` with which it is
  associated (identified in its `activity_source <KohonenLearningMechanism.activity_source>` attribute).
* 2) it is executed during the `execution phase <System_Execution>` of the System's execution.  Note that this is
  different from the behavior of supervised learning algorithms (such as `Reinforcement` and `BackPropagation`),
  that are executed during the `learning phase <System_Execution>` of a System's execution


.. _KohonenLearningMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.learningfunctions import Hebbian
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    ACTIVATION_INPUT, ACTIVATION_OUTPUT, LearningMechanism, LearningTiming, LearningType
from psyneulink.core.components.projections.projection import Projection_Base, projection_keywords
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    ADDITIVE, CONTROL_PROJECTIONS, INPUT_PORTS, KOHONEN_LEARNING_MECHANISM, \
    LEARNING, LEARNING_PROJECTION, LEARNING_SIGNAL, NAME, OUTPUT_PORTS, OWNER_VALUE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_numeric, parameter_spec

__all__ = [
    'KohonenLearningMechanism', 'KohonenLearningMechanismError', 'input_port_names', 'output_port_names',
]

# Parameters:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

input_port_names = [ACTIVATION_INPUT, ACTIVATION_OUTPUT]
output_port_names = [LEARNING_SIGNAL]

# DefaultTrainingMechanism = ObjectiveMechanism


class KohonenLearningMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class KohonenLearningMechanism(LearningMechanism):
    """
    KohonenLearningMechanism(                \
        variable,                            \
        function=Hebbian,                    \
        matrix=None,                         \
        learning_rate=None,                  \
        learning_signals=LEARNING_SIGNAL,    \
        modulation=ADDITIVE,v                \
        params=None,                         \
        name=None,                           \
        prefs=None)

    Subclass of `LearningMechanism` that modifies the `matrix <MappingProjection.matrix>` parameter of the
    `learning_projection <KohonenMechanism.learning_projection` of the associated `KohonenMechanism`
    (`activity_source <KohonenLearningMechanism.activity_source>`).
    See `LearningMechanism <LearningMechanism_Class_Reference>` for additional arguments and attributes.


    Arguments
    ---------

    variable : List[1d array, 1d array] or 2d np.array : default None
        it must have a two items that corresponds to the value required by the KohonenLearningMechanism's
        `function <KohonenLearningMechanism.function>`;  it must each be compatible (in number and type)
        with the `value <InputPort.value>` of the Mechanism's `InputPort <LearningMechanism_InputPorts>` (see
        `variable <KohonenLearningMechanism.variable>` for additional details).

    learning_signals : List[parameter of Projection, ParameterPort, Projection, tuple[str, Projection] or dict] \
    : default None
        specifies the `matrix <AutoAssociativeProjection.matrix>` to be learned (see `learning_signals
        <LearningMechanism.learning_signals>` for details of specification).

    modulation : str : default ADDITIVE
        specifies the default form of modulation used by the KohonenLearningMechanism's LearningSignals,
        unless they are `individually specified <LearningSignal_Specification>`.

    function : LearningFunction or function : default Kohonen
        specifies the function used to calculate the KohonenLearningMechanism's `learning_signal
        <KohonenLearningMechanism.learning_signal>` attribute.  It must take as its **variable** argument a
        list of three items (two 1d arrays and one 2d array, all of numeric values) and return a list, 2d np.array or
        np.matrix that is a square matrix with the same dimensions as the third item of the **variable** arugment).

    learning_rate : float : default None
        specifies the learning rate for the KohonenLearningMechanism. (see `learning_rate
        <KohonenLearningMechanism.learning_rate>` for details).


    Attributes
    ----------

    COMMENT:
        componentType : LEARNING_MECHANISM
    COMMENT

    variable : List[1d array, 1d array]
        has two items, corresponding to the `value <InputPort.value>` of its *ACTIVATION_INPUT* and
        *ACTIVATION_OUTPUT* InputPorts.

    activity_source : KohonenMechanism
        the `KohonenMechanism` with which the KohonenLearningMechanism is associated.

    input_ports : ContentAddressableList[OutputPort]
        has a two items, that contains the KohonenLearningMechanism's *ACTIVATION_INPUT*  and *ACTIVATION_OUTPUT*
        `InputPorts`.

    learned_projection : MappingProjection
        the `learning_projection <KohonenMechanism.learning_projection>` of the `KohoneMechanism` with which the
        KohonenLearningMechanism is associated (same as its `primary_learned_projection
        <LearningMechanism.primary_learned_projection>`.

    function : LearningFunction or function : default Kohonen
        the function used to calculate the `learning_signal <KohonenLearningMechanism.learning_signal>`
        (assigned to the KohonenLearningMechanism's `LearningSignal(s) <LearningMechanism_LearningSignal>`).
        It's `variable <Function_Base.variable>` must be a list of three items (two 1d arrays and one 2d array, all of
        numeric values);  returns a list, 2d np.array or np.matrix that is a square matrix with the same dimensions
        as the third item of its `variable <Kohonen.variable>`).

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        determines the learning rate used by the KohonenLearningMechanism's `function
        <KohonenLearningMechanism.function>` to scale the weight change matrix it returns. If it is a scalar, it is
        used to multiply the weight change matrix;  if it is a 2d array or matrix,
        it is used to Hadamard (elementwise) multiply the weight matrix (allowing the contribution of individual
        *connections* to be scaled);  if it is a 1d np.array, it is used to Hadamard (elementwise) multiply the input
        to the `function <KohonenLearningMechanism.function>` (i.e., the `value <InputPort.value>` of the
        KohonenLearningMechanism's *ACTIVATION_INPUT* `InputPort <KohonenLearningMechanism_Structure>`,
        allowing the contribution of individual *units* to be scaled). If specified, the value supersedes the
        learning_rate assigned to any `Process` or `System` to which the KohonenLearningMechanism belongs.
        If it is `None`, then the `learning_rate <Process.learning_rate>` specified for the Process to which the
        KohonenLearningMechanism belongs belongs is used;  and, if that is `None`, then the `learning_rate
        <System.learning_rate>` for the System to which it belongs is used. If all are `None`, then the
        `default_learning_rate <LearningFunction.default_learning_rate>` for the `function
        <KohonenLearningMechanism.function>` is used (see `learning_rate <LearningMechanism_Learning_Rate>`
        for additional details).

    learning_signal : 2d ndarray or matrix of numeric values
        the value returned by `function <KohonenLearningMechanism.function>`, that specifies
        the changes to the weights of the `matrix <KohonenLearningMechanism.matrix>`.
        It is assigned as the `value <LearningSignal.value>` of the KohonenLearningMechanism's `LearningSignal(s)
        <LearningMechanism_LearningSignal>` and, in turn, its `LearningProjection(s) <LearningProjection>`.

    learning_signals : List[LearningSignal]
        list of all of the `LearningSignals <LearningSignal>` for the KohonenLearningMechanism, each of which
        sends one or more `LearningProjections <LearningProjection>` to the `ParameterPort(s) <ParameterPort>` for
        the `matrix <MappingProjection.matrix>` parameter of the `MappingProjection(s)
        <MappingProjection>` trained by the KohonenLearningMechanism.  Although in most instances a
        KohonenLearningMechanism is used to train a single MappingProjection, like a standard
        `LearningMechanism`, it can be assigned additional LearningSignals and/or LearningProjections to train
        additional ones;  in such cases, the `value <LearningSignal>` for all of the LearningSignals is the
        the same:  the KohonenLearningMechanism's `learning_signal
        <KohonenLearningMechanism.learning_signal>` attribute.  Since LearningSignals are `OutputPorts
        <OutputPort>`, they are also listed in the KohonenLearningMechanism's `output_ports
        <KohonenLearningMechanism.output_ports>` attribute.

    learning_projections : List[LearningProjection]
        list of all of the LearningProjections <LearningProjection>` from the KohonenLearningMechanism, listed
        in the order of the `LearningSignals <LearningSignal>` to which they belong (that is, in the order they are
        listed in the `learning_signals <KohonenLearningMechanism.learning_signals>` attribute).

    output_ports : ContentAddressableList[OutputPort]
        list of the KohonenLearningMechanism's `OutputPorts <OutputPort>`, beginning with its
        `learning_signals <KohonenLearningMechanism.learning_signals>`, and followed by any additional
        (user-specified) `OutputPorts <OutputPort>`.

    output_values : 2d np.array
        the first item is the `value <OutputPort.value>` of the LearningMechanism's `learning_signal
        <KohonenLearningMechanism.learning_signal>`.

    modulation : str
        the default form of modulation used by the KohonenLearningMechanism's `LearningSignal(s)
        <LearningMechanism_LearningSignal>`, unless they are `individually specified <LearningSignal_Specification>`.

    """

    componentType = KOHONEN_LEARNING_MECHANISM
    className = componentType
    suffix = " " + className

    classPreferenceLevel = PreferenceLevel.TYPE

    learning_type = LearningType.UNSUPERVISED
    learning_timing = LearningTiming.EXECUTION_PHASE

    class Parameters(LearningMechanism.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <KohonenLearningMechanism.function>`

                    :default value: `Hebbian`
                    :type: `Function`

                learning_rate
                    see `learning_rate <KohonenLearningMechanism.learning_rate>`

                    :default value: None
                    :type:

                learning_timing
                    see `learning_timing <KohonenLearningMechanism.learning_timing>`

                    :default value: LearningTiming.EXECUTION_PHASE
                    :type: `LearningTiming`

                learning_type
                    see `learning_type <KohonenLearningMechanism.learning_type>`

                    :default value: LearningType.UNSUPERVISED
                    :type: `LearningType`

                matrix
                    see `matrix <KohonenLearningMechanism.matrix>`

                    :default value: None
                    :type:
        """
        function = Parameter(Hebbian, stateful=False, loggable=False)

        matrix = Parameter(None, modulable=True)
        learning_rate = Parameter(None, modulable=True)

        learning_type = LearningType.UNSUPERVISED
        learning_timing = LearningTiming.EXECUTION_PHASE
        modulation = ADDITIVE

    @tc.typecheck
    def __init__(self,
                 default_variable:tc.any(list, np.ndarray),
                 size=None,
                 matrix:tc.optional(ParameterPort)=None,
                 function: tc.optional(is_function_type) = None,
                 learning_signals:tc.optional(tc.optional(list)) = None,
                 modulation:tc.optional(str)=None,
                 learning_rate:tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self._init_args = locals().copy()
        # self._init_args['context'] = self
        # self._init_args['name'] = name

        # # Flag for deferred initialization
        # self.initialization_status = ContextFlags.DEFERRED_INIT
        # self.initialization_status = ContextFlags.DEFERRED_INIT

        # self._learning_rate = learning_rate

        super().__init__(
            default_variable=default_variable,
            size=size,
            function=function,
            modulation=modulation,
            learning_rate=learning_rate,
            matrix=matrix,
            learning_signals=learning_signals,
            params=params,
            name=name,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        """Validate that variable has only one item: activation_input.
        """

        # Skip LearningMechanism._validate_variable in call to super(), as it requires variable to have 3 items
        variable = super(LearningMechanism, self)._validate_variable(variable, context)

        if np.array(variable).ndim != 2 or not is_numeric(variable):
            raise KohonenLearningMechanismError("Variable for {} ({}) must be a list with two items "
                                                "or a 2d np.array, all of which may contain only numbers".
                                                        format(self.name, variable))
        return variable

    def _parse_function_variable(self, variable, context=None):
        variable = variable.tolist()
        variable.append(self.matrix.parameters.value._get(context).tolist())
        return variable

    def _execute(self,
                 variable=None,
                 context=None,
                 runtime_params=None,

                 ):
        """Execute KohonenLearningMechanism. function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # COMPUTE LEARNING SIGNAL (note that function is assumed to return only one value)
        # IMPLEMENTATION NOTE:  skip LearningMechanism's implementation of _execute
        #                       as it assumes projections from other LearningMechanisms
        #                       which are not relevant to an autoassociative projection

        learning_signal = super(LearningMechanism, self)._execute(
            variable=variable,
            context=context,
            runtime_params=runtime_params,

        )

        if self.initialization_status != ContextFlags.INITIALIZING and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, learning_signal))

        return [learning_signal]

    def _update_output_ports(self, context=None, runtime_params=None):
        """Update the weights for the MappingProjection for which this is the KohonenLearningMechanism

        Must do this here, so it occurs after LearningMechanism's OutputPort has been updated.
        This insures that weights are updated within the same trial in which they have been learned
        """

        super()._update_output_ports(context, runtime_params)

        if context.composition is not None and ContextFlags.LEARNING_MODE in context.runmode:
            context.execution_phase = ContextFlags.LEARNING
            self.learned_projection.execute(context=context)
            context.remove_flag(ContextFlags.LEARNING)

    @property
    def learned_projection(self):
        return self.primary_learned_projection

    @property
    def activity_source(self):
        # return self.input_port.path_afferents[0].sender.owner
        return self.primary_learned_projection.sender.owner
