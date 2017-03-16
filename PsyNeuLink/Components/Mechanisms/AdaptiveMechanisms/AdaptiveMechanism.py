# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  AdaptiveMechanism *****************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**
COMMENT:
  MOVE TO AdaptiveMechanisms overview:
  Different AdaptiveMechanisms transform their input in different ways, and some allow this to be customized
  by modifying their ``function`` parameter.  For example, a :doc:`TransferMechanism` can be configured to produce a
  linear, logistic, or exponential transform of its input.
COMMENT



Overview
--------

An AdaptiveMechanism monitors one or more `outputStates <OutputState>` of one or more
`ProcessingMechanisms <ProcessingMechanism>`, and uses that information to modify the parameters of another
PsyNeuLink component.  There are two types of AdaptiveMechanism: `LearningMechanisms <LearningMechanism>`, that modify
the parameters of `MappingProjections <MappingProjection>`; and `ControlMechanisms <ControlMechanism>` that modify the
parameters of other ProcessingMechanisms.  In general, an AdaptiveMechanism receives its input (i.e., monitors the
outputState) of an `ObjectiveMechanism`, however this need not be the case.  AdaptiveMechanisms are always executed
after all ProcessingMechanisms in the `process <Process>` or `system <System>` to which they belong have been
`executed <LINK>`, with all ControlMechanisms executed before all LearningMechanisms. Both types of
AdaptiveMechanisms are executed before the next `round of execution <LINK>`, so that the modifications
they make are available during that next round of execution of the process or system.

.. _AdaptiveMechanism_Creation:

Creating an AdaptiveMechanism
------------------------------

CAN BE DONE MANUALLY, BUT GENERALLY AUTOMATICALLY (SEE EACH SUBCLASS)

A ComparatorMechanism can be created directly by calling its constructor
COMMENT:
    , or using the
    `mechanism` function and specifying keyword:`ComparatorMechanism` as its :keyword:`mech_spec` argument.
COMMENT
. The type of comparison is specified in the `comparison_operation` argument, which can be `SUBTRACTION` or
`DIVISION`.  It can also be created by `in-context specification <Projection_Creation>` of a LearningProjection for a
projection to the `TERMINAL` mechanism of a process.  One or more ComparatorMechanisms are also created automatically
when learning is specified for a `process <Process_Learning>` or `system <System_Execution_Learning>`. Each
ComparatorMechanism is assigned a projection from a `TERMINAL` mechanism that receives a MappingProjection being
learned. A LearningProjection to that MappingProjection is also created (see `learning in a process <Process_Learning>`,
and `automatic creation of LearningSignals  <LearningProjection_Automatic_Creation>` for details).

.. _Comparator_Structure:

Structure
---------

A ComparatorMechanism has two `inputStates <InputState>`:

    * :keyword:`COMPARATOR_SAMPLE` inputState receives a MappingProjection
      from the `primary outputState <OutputState_Primary>` of a `TERMINAL` mechanism in a process;
    ..
    * `COMPARATOR_TARGET` inputState is assigned its value from the :keyword:`target` argument of a call to the
      `run <Run>` method of a process or system.  It has five outputStates, described under
      :ref:`Execution <Comparator_Execution>` below.


.. _Comparator_Execution:

Execution
---------

A ComparatorMechanism always executes after the mechanism it is monitoring.  The :keyword:`value` of the
`primary outputState <OutputState_Primary>` of the mechanism being monitored is assigned as the :keyword:`value` of the
ComparatorMechanism's :keyword:`COMPARATOR_SAMPLE` inputState;  the value of the :keyword:`COMPARATOR_TARGET`
inputState is received from the process (or system to which it belongs) when it is run (i.e., the input provided
 in the process' or system's :keyword:`execute` method or :keyword:`run` method). When the ComparatorMechanism
is executed, if `comparison_operation` is:

    * `SUBTRACTION`, its `function <ComparatorMechanism.function>` subtracts the  `COMPARATOR_SAMPLE` from the
      `COMPARATOR_TARGET`;
    ..
    * `DIVISION`, its `function <ComparatorMechanism.function>` divides the `COMPARATOR_TARGET`by the
      `COMPARATOR_SAMPLE`.

After each execution of the mechanism:

.. _Comparator_Results:

    * the **result** of the `function <ComparatorMechanism.function>` calculation is assigned to the mechanism's
      `value <ComparatorMechanism.value>` attribute, the value of its `COMPARISON_RESULT`
      outputState, and to the 1st item of its `outputValue <ComparatorMechanism.outputValue>` attribute;
    ..
    * the **mean** of the result is assigned to the :keyword:`value` of the mechanism's `COMPARISON_MEAN` outputState,
      and to the 2nd item of its `outputValue <ComparatorMechanism.outputValue>` attribute.
    ..

    * the **sum** of the result is assigned to the :keyword:`value` of the mechanism's `COMPARISON_SUM` outputState,
      and to the 3rd item of its `outputValue <ComparatorMechanism.outputValue>` attribute.
    ..

    * the **sum of squares** of the result is assigned to the :keyword:`value` of the mechanism's `COMPARISON_SSE`
      outputState, and to the 4th item of its `outputValue <ComparatorMechanism.outputValue>` attribute.
    ..

    * the **mean of the squares** of the result is assigned to the :keyword:`value` of the mechanism's
      :keyword:`COMPARISON_MSE` outputState, and to the 5th item of its `outputValue <ComparatorMechanism.outputValue>`
      attribute.

.. _Comparator_Class_Reference:

Class Reference
---------------


"""

from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.ShellClasses import *

# ControlMechanismRegistry = {}


class AdpativeMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class AdaptiveMechanism_Base(Mechanism_Base):
    # DOCUMENTATION: this is a TYPE and subclasses are SUBTYPES
    #                primary purpose is to implement TYPE level preferences for all adaptive mechanisms
    #                inherits all attributes and methods of Mechanism -- see Mechanism for documentation
    # IMPLEMENT: consider moving any properties of adaptive mechanisms not used by control mechanisms to here
    """Abstract class for AdaptiveMechanism subclasses
   """

    componentType = "AdaptiveMechanism"

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'AdaptiveMechanismClassPreferences',
    #     kp<pref>: <setting>...}

    # variableClassDefault = defaultControlAllocation
    # This must be a list, as there may be more than one (e.g., one per controlSignal)
    variableClassDefault = defaultControlAllocation

    def __init__(self,
                 variable=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Abstract class for AdaptiveMechanism

        :param variable: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        self.system = None

        super().__init__(variable=variable,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_inputs(self, inputs=None):
        # Let mechanism itself do validation of the input
        pass
