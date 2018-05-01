# Princeton University licenses this file to You under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# ***************************** PredictionErrorMechanism ***********************

"""
Overview
--------

A PredictionErrorMechanism is a subclass of `ComparatorMechanism` that receives
two inputs (a sample and a target), and calculates the temporal difference
prediction error as found in `Montague, Dayan, and Sejnowski (1996) <http://www.jneurosci.org/content/jneuro/16/5/1936.full.pdf>`_
using its `function <PredictionErrorMechanism.function>`, and places the delta
values (the difference between the actual and predicted reward) in its *OUTCOME*
`OutputState <PredictionErrorMechanism.output_state>`.

.. _PredictionErrorMechanism_Creation:

Creating a PredictionErrorMechanism
-----------------------------------

A PredictionErrorMechanism is usually created automatically when a `LearningMechanism`
`is created <LearningMechanism_Creation>` using the `TDLearning` function).
A PredictionErrorMechanism can also be created directly by calling its constructor.
Its **sample** and **target**  arguments are used to specify the OutputStates
that provide the sample and target inputs, respectively (see
`ObjectiveMechanism Monitored Output States <ObjectiveMechanism_Monitored_Output_States>`
for details). When the PredictionErrorMechanism is created, two InputStates are
created, one each for its sample and target inputs (and named, by default
*SAMPLE* and *TARGET*). Each is assigned a MappingProjection from the
corresponding OutputState specified in the **sample** and **target** arguments.

It is important to recognize that the value of the *SAMPLE* and *TARGET*
InputStates must have the same length and type, so that they can be compared
using the PredictionErrorMechanism's `function
<PredictionErrorMechanism.function>` By default, they use the format of the
OutputStates specified in the **sample** and **target** arguments, respectively,
and the `MappingProjection` to each uses an `IDENTITY_MATRIX`. Therefore, for
the default configuration, the OutputStates specified in the **sample** and
**target** arguments must have values of the same length and type. If these
differ, the **input_states** argument can be used to explicitly specify the
format of the PredictionErrorMechanism's *SAMPLE* and *TARGET* InputStates, to
insure they are compatible with one another (as well as to customize their
names, if desired). If the **input_states** argument is used, *both* the sample
and target InputStates must be specified. Any of the formats for `specifying
InputStates <InputState_Specification>` can be used in the argument. If values
are assigned for the InputStates, they must be of equal length and type. Their
types must also be compatible with the value of the OutputStates specified in
the **sample** and **target** arguments. However, the length specified for an
InputState can differ from its corresponding OutputState; in that case, by
default, the MappingProjection created uses a `FULL_CONNECTIVITY` matrix. Thus,
OutputStates of differing lengths can be mapped to the sample and target
InputStates of a PredictionErrorMechanism (see the `example
<PredictionErrorMechanism_Example>` below), so long as the latter of of the
same length. If a projection other than a `FULL_CONNECTIVITY` matrix is
needed, this can be specified using the *PROJECTION* entry of a `State
specification dictionary <State_Specification>` for the InputState in the
**input_states** argument.

.. _PredictionErrorMechanism_Structure:

Structure
---------

A PredictionErrorMechanism has two `input_states
<ComparatorMechanism.input_states>`, each of which receives a
`MappingProjection` from a corresponding OutputState specified in the
**sample** and **target** arguments of its constructor. The InputStates are
listed in the Mechanism's `input_states <ComparatorMechanism.input_states>`
attribute and named, respectively, *SAMPLE* and *TARGET*. The OutputStates
from which they receive their projections (specified in the the **sample** and
**target** arguments) are listed in the Mechanism's `sample
<ComparatorMechanism.sample>` and `target
<ComparatorMechanism.target>` attributes as well as in its
`monitored_output_states <ComparatorMechanism.monitored_output_states>`
attribute. The PredictionErrorMechanism's `function
<PredictionErrorMechanism.function>` calculates the difference between the
predicted reward and the true reward at each timestep in **SAMPLE**. By
default, it uses a `PredictionErrorDeltaFunction`. However, the
`function <PredictionErrorMechanism.function>` can be customized, so long as it
is replaced with one that takes two arrays with the same format as its inputs
and generates a similar array as its result. The result is assigned as the
value of the PredictionErrorMechanism's *OUTCOME* (`primary
<OutputState_Primary>`) OutputState.

.. _PredictionErrorMechanism_Function:

Execution
---------

When a PredictionErrorMechanism is executed, it updates its input_states with
the values of the OutputStates specified in its **sample** and **target**
arguments, and then uses its `function <PredictionErrorMechanism.function>` to
compare these. By default, the result is assigned to the `value
<PredictionErrorMechanism.value>` of its *OUTCOME* `output_state
<PredictionErrorMechanism.output_state>`, and as the first item of the
Mechanism's `output_values <PredictionErrorMechanism.output_values>` attribute.

.. _PredictionErrorMechanism_Example:

Example
-------

.. _PredictionErrorMechanism_Default_Input_Value_Example:

*Formatting InputState values*

The **default_variable** argument can be used to specify a particular format
for the SAMPLE and/or TARGET InputStates of a PredictionErrorMechanism. This
can be useful when one or both of these differ from the format of the
OutputState(s) specified in the **sample** and **target** arguments. For
example, for `Temporal Difference Learning <TDLearning>`, a
PredictionErrorMechanism is used to compare the predicted reward from the
sample with the true reward (the target). In the example below, the sample
Mechanism is a `TransferMechanism` that uses the `Linear` function to output
the sample values. Because the output is a vector, specifying it as the
PredictionErrorMechanism's **sample** argument will generate a corresponding
InputState with a vector as its value. This should match the the reward
signal specified in the PredictionErrorMechanism's **target** argument, the
value of which is a vector of the same length as the output of sample.

    >>> import psyneulink as pnl
    >>> sample_mech = pnl.TransferMechanism(size=5,
    ...                                     function=pnl.Linear())
    >>> reward_mech = pnl.TransferMechanism(size=5)
    >>> prediction_error_mech = pnl.PredictionErrorMechanism(sample=sample_mech,
    ...                                                      target=reward_mech)

Note that ``sample_mech`` is specified to take an array of length 5 as its
input, and therefore generate one of the same length as its `primary output
<OutputState_Primary>`. Since it is assigned as the **sample** of the
PredictionErrorMechanism, by default this will create a *SAMPLE* InputState of
length 5, that will match the length of the *TARGET* InputState.

Currently the default method of implementing temporal difference learning in
PsyNeuLink requires the values of *SAMPLE* and *TARGET* to be provided as an
array representing a full time series as an experiment. See
`MontagueDayanSejnowski.py` in the Scripts folder for an example.

.. _PredictionErrorMechanism_Class_Reference

Class Reference
---------------

"""
from typing import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import PredictionErrorDeltaFunction
from psyneulink.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.components.mechanisms.processing.objectivemechanism import OUTCOME
from psyneulink.components.states.outputstate import OutputState
from psyneulink.globals.keywords import INITIALIZING, PREDICTION_ERROR_MECHANISM, SAMPLE, TARGET
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel, kwPreferenceSetName
from psyneulink.globals.utilities import is_numeric
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism, ComparatorMechanismError

__all__ = [
    'PredictionErrorMechanism',
    'PredictionErrorMechanismError'
]


class PredictionErrorMechanismError(ComparatorMechanismError):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class PredictionErrorMechanism(ComparatorMechanism):
    """
    PredictionErrorMechanism(                                \
        sample,                                              \
        target,                                              \
        function=PredictionErrorDeltaFunction,               \
        output_states=[OUTCOME],                             \
        params=None,                                         \
        name=None,                                           \
        prefs=None)

    Calculates the prediction error between the predicted reward and the target

    Arguments
    ---------

    sample : OutputState, Mechanism_Base, dict, number, or str
        specifies the SAMPLE InputState, which will be evaluated by
        the function

    target : OutputState, Mechanism_Base, dict, number, or str
        specifies the TARGET InputState, which will be used by the function to
        evaluate the sample

    function : CombinationFunction, ObjectiveFunction, function, or method : default PredictionErrorDeltaFunction
        the function used to evaluate the sample and target inputs.

    output_states : str, Iterable : default OUTCOME
        by default, contains only the *OUTCOME* (`primary <OutputState_Primary>`)
        OutputState of the PredictionErrorMechanism.

    learning_rate : Number : default 0.3
        controls the weight of later timesteps compared to earlier ones. Higher
        rates weight later timesteps more heavily than previous ones.

    name : str
        the name of the PredictionErrorMechanism; if it is not specified in the
        **name** argument of the constructor, a default is assigned by
        MechanismRegistry (see `Naming` for conventions used for default and
        duplicate names).


    Attributes
    ----------

    sample : OutputState, Mechanism_Base, dict, number, or str
        specifies the SAMPLE InputState, which will be evaluated by
        the function

    target : OutputState, Mechanism_Base, dict, number, or str
        specifies the TARGET InputState, which will be used by the function to
        evaluate the sample

    function : CombinationFunction, ObjectiveFunction, Function, or method : default PredictionErrorDeltaFunction
        the function used to evaluate the sample and target inputs.

    output_states : str, Iterable : default OUTCOME
        by default, contains only the *OUTCOME* (`primary <OutputState_Primary>`)
        OutputState of the PredictionErrorMechanism.

    learning_rate : Number : default 0.3
        controls the weight of later timesteps compared to earlier ones. Higher
        rates weight later timesteps more heavily than previous ones.

    name : str
        the name of the PredictionErrorMechanism; if it is not specified in the
        **name** argument of the constructor, a default is assigned by
        MechanismRegistry (see `Naming` for conventions used for default and
        duplicate names).

    """
    componentType = PREDICTION_ERROR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    classPreferences = {
        kwPreferenceSetName: 'PredictionErrorMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    class ClassDefaults(ComparatorMechanism.ClassDefaults):
        variable = None

    paramClassDefaults = ComparatorMechanism.paramClassDefaults.copy()
    standard_output_states = ComparatorMechanism.standard_output_states.copy()

    @tc.typecheck
    def __init__(self,
                 sample: tc.optional(tc.any(OutputState, Mechanism_Base, dict,
                                            is_numeric,
                                            str)) = None,
                 target: tc.optional(tc.any(OutputState, Mechanism_Base, dict,
                                            is_numeric,
                                            str)) = None,
                 function=PredictionErrorDeltaFunction(),
                 output_states: tc.optional(tc.any(str, Iterable)) = OUTCOME,
                 learning_rate: is_numeric = 0.3,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None,
                 context=componentType + INITIALIZING):
        input_states = [sample, target]
        params = self._assign_args_to_param_dicts(sample=sample,
                                                  target=target,
                                                  function=function,
                                                  input_states=input_states,
                                                  output_states=output_states,
                                                  learning_rate=learning_rate,
                                                  params=params)

        super().__init__(sample=sample,
                         target=target,
                         input_states=input_states,
                         function=function,
                         output_states=output_states,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _parse_function_variable(self, variable):
        # TODO: update to take sample/reward from variable
        # sample = x(t) in Montague on first run, V(t) on subsequent runs
        sample = self.input_states[SAMPLE].value
        reward = self.input_states[TARGET].value

        return [sample, reward]

    def _execute(self, variable=None, function_variable=None, runtime_params=None, context=None):
        delta = super()._execute(variable=variable, function_variable=function_variable, runtime_params=runtime_params, context=context)
        delta = delta[1:]
        delta = np.append(delta, 0)

        return delta
