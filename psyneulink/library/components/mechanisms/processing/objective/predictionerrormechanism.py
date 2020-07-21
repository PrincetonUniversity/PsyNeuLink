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

Contents
--------

  * `PredictionErrorMechanism_Overview`
  * `PredictionErrorMechanism_Creation`
  * `PredictionErrorMechanism_Structure`
  * `PredictionErrorMechanism_Execution`
  * `PredictionErrorMechanism_Example`
  * `PredictionErrorMechanism_Class_Reference`


.. _PredictionErrorMechanism_Overview:

Overview
--------

A PredictionErrorMechanism is a subclass of `ComparatorMechanism` that receives two inputs (a sample and a target),
and calculates the temporal difference prediction error as found in `Montague, Dayan, and Sejnowski (1996)
<http://www.jneurosci.org/content/jneuro/16/5/1936.full.pdf>`_ using its `function
<PredictionErrorMechanism.function>`, and places the delta values (the difference between the actual and predicted
reward) in its *OUTCOME* `OutputPort <PredictionErrorMechanism.output_port>`.

.. _PredictionErrorMechanism_Creation:

Creating a PredictionErrorMechanism
-----------------------------------

A PredictionErrorMechanism is usually created automatically when a `LearningMechanism`
`is created <LearningMechanism_Creation>` using the `TDLearning` function).
A PredictionErrorMechanism can also be created directly by calling its constructor.
Its **sample** and **target**  arguments are used to specify the OutputPorts
that provide the sample and target inputs, respectively (see
`ObjectiveMechanism Monitored Output Ports <ObjectiveMechanism_Monitor>`
for details). When the PredictionErrorMechanism is created, two InputPorts are
created, one each for its sample and target inputs (and named, by default
*SAMPLE* and *TARGET*). Each is assigned a MappingProjection from the
corresponding OutputPort specified in the **sample** and **target** arguments.

It is important to recognize that the value of the *SAMPLE* and *TARGET*
InputPorts must have the same length and type, so that they can be compared
using the PredictionErrorMechanism's `function
<PredictionErrorMechanism.function>.` By default, they use the format of the
OutputPorts specified in the **sample** and **target** arguments, respectively,
and the `MappingProjection` to each uses an `IDENTITY_MATRIX`. Therefore, for
the default configuration, the OutputPorts specified in the **sample** and
**target** arguments must have values of the same length and type. If these
differ, the **input_ports** argument can be used to explicitly specify the
format of the PredictionErrorMechanism's *SAMPLE* and *TARGET* InputPorts, to
insure they are compatible with one another (as well as to customize their
names, if desired). If the **input_ports** argument is used, *both* the sample
and target InputPorts must be specified. Any of the formats for `specifying
InputPorts <InputPort_Specification>` can be used in the argument. If values
are assigned for the InputPorts, they must be of equal length and type. Their
types must also be compatible with the value of the OutputPorts specified in
the **sample** and **target** arguments. However, the length specified for an
InputPort can differ from its corresponding OutputPort; in that case, by
default, the MappingProjection created uses a `FULL_CONNECTIVITY` matrix. Thus,
OutputPorts of differing lengths can be mapped to the sample and target
InputPorts of a PredictionErrorMechanism (see the `example
<PredictionErrorMechanism_Example>` below), so long as the latter of of the
same length. If a projection other than a `FULL_CONNECTIVITY` matrix is
needed, this can be specified using the *PROJECTION* entry of a `Port
specification dictionary <Port_Specification>` for the InputPort in the
**input_ports** argument.

.. _PredictionErrorMechanism_Structure:

Structure
---------

A PredictionErrorMechanism has two `input_ports
<ComparatorMechanism.input_ports>`, each of which receives a
`MappingProjection` from a corresponding OutputPort specified in the
**sample** and **target** arguments of its constructor. The InputPorts are
listed in the Mechanism's `input_ports <ComparatorMechanism.input_ports>`
attribute and named, respectively, *SAMPLE* and *TARGET*. The OutputPorts
from which they receive their projections (specified in the **sample** and
**target** arguments) are listed in the Mechanism's `sample
<ComparatorMechanism.sample>` and `target
<ComparatorMechanism.target>` attributes as well as in its
`monitored_output_ports <ComparatorMechanism.monitored_output_ports>`
attribute. The PredictionErrorMechanism's `function
<PredictionErrorMechanism.function>` calculates the difference between the
predicted reward and the true reward at each timestep in **SAMPLE**. By
default, it uses a `PredictionErrorDeltaFunction`. However, the
`function <PredictionErrorMechanism.function>` can be customized, so long as it
is replaced with one that takes two arrays with the same format as its inputs
and generates a similar array as its result. The result is assigned as the
value of the PredictionErrorMechanism's *OUTCOME* (`primary
<OutputPort_Primary>`) OutputPort.

.. _PredictionErrorMechanism_Execution:

Execution
---------

When a PredictionErrorMechanism is executed, it updates its input_ports with
the values of the OutputPorts specified in its **sample** and **target**
arguments, and then uses its `function <PredictionErrorMechanism.function>` to
compare these. By default, the result is assigned to the `value
<PredictionErrorMechanism.value>` of its *OUTCOME* `output_port
<PredictionErrorMechanism.output_port>`, and as the first item of the
Mechanism's `output_values <PredictionErrorMechanism.output_values>` attribute.

.. _PredictionErrorMechanism_Example:

Example
-------

.. _PredictionErrorMechanism_Default_Input_Value_Example:

*Formatting InputPort values*

The **default_variable** argument can be used to specify a particular format
for the SAMPLE and/or TARGET InputPorts of a PredictionErrorMechanism. This
can be useful when one or both of these differ from the format of the
OutputPort(s) specified in the **sample** and **target** arguments. For
example, for `Temporal Difference Learning <TDLearning>`, a
PredictionErrorMechanism is used to compare the predicted reward from the
sample with the true reward (the target). In the example below, the sample
Mechanism is a `TransferMechanism` that uses the `Linear` function to output
the sample values. Because the output is a vector, specifying it as the
PredictionErrorMechanism's **sample** argument will generate a corresponding
InputPort with a vector as its value. This should match the reward
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
<OutputPort_Primary>`. Since it is assigned as the **sample** of the
PredictionErrorMechanism, by default this will create a *SAMPLE* InputPort of
length 5, that will match the length of the *TARGET* InputPort.

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

from psyneulink.core.components.functions.combinationfunctions import PredictionErrorDeltaFunction
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.keywords import INITIALIZING, OUTCOME, PREDICTION_ERROR_MECHANISM, SAMPLE, TARGET
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel, PREFERENCE_SET_NAME
from psyneulink.core.globals.utilities import is_numeric
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism, ComparatorMechanismError

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
        output_ports=[OUTCOME],                             \
        params=None,                                         \
        name=None,                                           \
        prefs=None)

    Subclass of ComparatorMechanism that calculates the prediction error between the predicted reward and the target.
    See `ComparatorMechanism <ComparatorMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    sample : OutputPort, Mechanism_Base, dict, number, or str
        specifies the *SAMPLE* InputPort, that is evaluated by the `function <PredictionErrorMechanism.function>`.

    target : OutputPort, Mechanism_Base, dict, number, or str
        specifies the *TARGET* InputPort used by the function to evaluate `sample<PredictionErrorMechanism.sample>`.

    function : CombinationFunction, ObjectiveFunction, function, or method : default PredictionErrorDeltaFunction
        the function used to evaluate the SAMPLE and TARGET inputs.

    learning_rate : Number : default 0.3
        controls the weight of later timesteps compared to earlier ones;  higher rates weight later timesteps more
        heavily than previous ones.

    Attributes
    ----------

    sample : OutputPort, Mechanism_Base, dict, number, or str
        the *SAMPLE* `InputPort`, the `value <InputPort.value>` of which will be evaluated by the function.

    target : OutputPort, Mechanism_Base, dict, number, or str
        the *TARGET* `InputPort`, the `value <InputPort.value>` of which will be used to evaluate `sample
        <PredictionErrorMechanism.sample>`.

    function : CombinationFunction, ObjectiveFunction, Function, or method : default PredictionErrorDeltaFunction
        the function used to evaluate the sample and target inputs.

    output_ports : str, Iterable : default OUTCOME
        by default, contains only the *OUTCOME* (`primary <OutputPort_Primary>`) `OutputPort` of the
        PredictionErrorMechanism.

    learning_rate : Number : default 0.3
        controls the weight of later timesteps compared to earlier ones; higher rates weight later timesteps more
        heavily than previous ones.

    """
    componentType = PREDICTION_ERROR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    classPreferences = {
        PREFERENCE_SET_NAME: 'PredictionErrorMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)
    }

    class Parameters(ComparatorMechanism.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <PredictionErrorMechanism.variable>`

                    :default value: None
                    :type:
                    :read only: True

                function
                    see `function <PredictionErrorMechanism.function>`

                    :default value: `PredictionErrorDeltaFunction`
                    :type: `Function`

                learning_rate
                    see `learning_rate <PredictionErrorMechanism.learning_rate>`

                    :default value: 0.3
                    :type: ``float``
        """

        variable = Parameter(None, read_only=True, pnl_internal=True, constructor_argument='default_variable')
        learning_rate = Parameter(0.3, modulable=True)
        function = Parameter(PredictionErrorDeltaFunction, stateful=False, loggable=False)
        sample = None
        target = None

    @tc.typecheck
    def __init__(self,
                 sample: tc.optional(tc.any(OutputPort, Mechanism_Base, dict,
                                            is_numeric,
                                            str)) = None,
                 target: tc.optional(tc.any(OutputPort, Mechanism_Base, dict,
                                            is_numeric,
                                            str)) = None,
                 function=None,
                 output_ports: tc.optional(tc.optional(tc.any(str, Iterable))) = None,
                 learning_rate: tc.optional(is_numeric) = None,
                 params=None,
                 name=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs
                 ):

        input_ports = [sample, target]
        super().__init__(
            sample=sample,
            target=target,
            input_ports=input_ports,
            function=function,
            output_ports=output_ports,
            learning_rate=learning_rate,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

    def _parse_function_variable(self, variable, context=None):
        # TODO: update to take sample/reward from variable
        # sample = x(t) in Montague on first run, V(t) on subsequent runs
        sample = self.input_ports[SAMPLE].parameters.value._get(context)
        reward = self.input_ports[TARGET].parameters.value._get(context)

        return np.array([sample, reward])

    def _execute(self, variable=None, context=None, runtime_params=None):
        delta = super()._execute(variable=variable, context=context, runtime_params=runtime_params)
        delta = delta[0][1:]
        delta = np.append(delta, 0)
        return delta
