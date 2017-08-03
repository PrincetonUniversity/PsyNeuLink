# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  EVCMechanism ******************************************************

"""

Overview
--------

An EVCMechanism is a `ControlMechanism <ControlMechanism>` that manages a "portfolio" of
`ControlSignals <ControlSignal>` that regulate the performance of the System to which they belong. The
EVCMechanism is one of the most powerful, but also one of the most complex components in PsyNeuLink.  It is
designed to implement a form of the Expected Value of Control (EVC) Theory described in
`Shenhav et al. (2013) <https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_, which provides useful background concerning
the purpose and structure of the EVCMechanism.

An EVCMechanism belongs to a `System` specified in its `system <EVCMechanism.system>` attribute, and a `ControlSignal`
for each parameter of the Components in the `system <EVCMechanism.system>` that it controls.  Each ControlSignal is
associated with a `ControlProjection` that regulates the value of the parameter it controls, with the magnitude of
that regulation determined by the ControlSignal's `intensity`.  A particular combination of ControlSignal `intensity`
values is called an `allocation_policy`. When a `System` is executed that uses an EVCMechanism as its `controller
<System_Base.controller>`, it concludes by executing the EVCMechanism which determines its `allocation_policy` for the
next `TRIAL`.  That, in turn, determines the `intensity` for each of the ControlSignals, and therefore the values of
the parameters being controlled on the next `TRIAL`.


.. _EVCMechanism_EVC:

Expected Value of Control (EVC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EVCMechanism uses it `function <EVCMechanism.function>` to select an `allocation_policy` for its `system
<EVCMechanism.system>`.  In the `default configuration <EVC_Default_Configuration>`, an EVCMechanism carries out an
exhaustive evaluation of allocation policies, simulating its `system <EVCMechanism.system>` under each, and using an
`ObjectiveMechanism` and several `auxiliary functions <EVCMechanism_Auxiliary_Functions>` to calculate the **expected
value of control (EVC)** for each `allocation_policy`: a cost-benefit analysis that weighs the `cost
<ControlSignal.cost> of the ControlSignals against the outcome of the `system <EVCMechanism.system>` \s performance for
a given `allocation_policy`. The EVCMechanism selects the `allocation_policy` that generates the maximum EVC, and
implements that for the next `TRIAL`. Each step of this procedure can be modified, or replaced entirely, by assigning
custom functions to corresponding parameters of the EVCMechanism, as described `below <EVCMechanism_Functions>`.

.. _EVCMechanism_Creation:

Creating an EVCMechanism
------------------------

An EVCMechanism can be created in any of the ways used to `create Mechanisms <Mechanism_Creation>`;  it is also
created automatically when a `System` is created and the EVCMechanism class is specified in the **controller**
argument of the System's constructor (see `System_Creation`).

When an EVCMechanism is created explicitly (using its constructor), it creates:

* an `ObjectiveMechanism`, using the list of `OutputState` specifications in the **monitor_for_control** argument of
  the EVCMechanism's constructor to specify the ObjectiveMechanism's `monitored_values
  <ObjectiveMechanism.monitored_values>` attribute, and the function specified in the **outcome_function** argument
  of the EVCMechanism's constructor to specify the ObjectiveMechanism's `function <ObjectiveMechanism.function>`;
..
* a `MappingProjection` that projects from the ObjectiveMechanism's *ERROR_SIGNAL* `OutputState
  <ObjectiveMechanism_Structure>` to the EVCMechanism's `primary InputState <InputState_Primary>`.
..
* a `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` for each `ORIGIN` Mechanism in its `system
  <EVCMechanism.system>`, assigns a MappingProjection to each from the `system <EVCMechanism.system>`,
  and assigns these to its `prediction_mechanisms` attribute.

When an EVCMechanism is created automatically as part of a `System <System_Creation>`, the same set of Components are
created as described above, with the following modifications:

* the `OutputStates <OutputState>` specified in the System's `monitor_for_control <System_Base.monitor_for_control>`
  attribute are used to create the ObjectiveMechanism
..
* a `ControlSignal` is created and assigned to the EVCMechanisn's `control_signals <EVCMechanism.control_signals>`
  attribute for every parameter of any `Component` in the System that has been specified for control (that is,
  by including a `ControlProjection` or `ControlSignal` in a `tuple specification <>` for the parameter, or by
  specifying the parameter (or its associated `ParameterState`) in the **control_signals** argument of a
  `ControlMechanism <ControlMechanism_Control_Signals>`.

An EVCMechanism that has been constructed automatically can be customized by assigning values to its attributes (e.g.,
those described above, or its `function <EVCMechanism.function>` as described under `EVC_Default_Configuration `below).

.. _EVCMechanism_Structure:

Structure
---------

An EVCMechanism belongs to a `System` (identified in its `system <EVCMechanism.system>` attribute), and has a
specialized set of Components that support its operation.  It receives its input from the *ERROR_SIGNAL* `OutputState
<ObjectiveMechanism_Structure>` of an `ObjectiveMechanism` (identified in its `monitoring_mechanism
<EVCMechansm.monitoring_mechanism>` attribute), and has a specialized set of `functions <EVCMechanism_Functions>` and
`mechanisms <EVCMechanism_Prediction_Mechanisms>` that it can use to simulate and evaluate the performance of its
`system <EVCMechanism.system>` under the influence of different values of its `ControlSignals
<EVCMechanism_ControlSignals>`.  Each of these specialized Components is described below.


ObjectiveMechanism
~~~~~~~~~~~~~~~~~~

.. _EVCMechanism_ObjectiveMechanism:

When an EVCMechanism is created, it creates an `ObjectiveMechanism` that is assigned as its `monitoring_mechanism
<EVCMechanism.monitoring_mechanism>` attribute, and can be used to evaluate the performance of its `system
<EVCMechanism.system>`.  The `monitoring_mechanism <EVCMechanism.monitoring_mechanism>` receives a `MappingProjection`
from each of the `OutputStates <OutputState>` specified in the EVCMechanism's `monitored_output_states
<EVCMechanism.monitored_output_states>` attribute (also listed in the `monitored_values
<ObjectiveMechanism.monitored_values>` attribute of the `monitoring_mechanism <EVCMechanism.monitoring_mechanism>`),
and the EVCMechanism's `outcome_function <EVCMechanism.outcome_function>` is assinged as the `monitoring_mechanism
<EVCMechanism.monitoring_mechanism>` \'s `function <ObjectiveMechanism.function>`.  By default, this is a
`LinearCombination` function that calculates the product of the `value <OutputState.value>` of the OutputStates
specified in `monitored_output_states <EVCMechanism.monitored_output_states>`.  However, the contribution of individual
OutputStates can be specified in the **monitor_for_control** argument of the EVCMechanism's constructor, by using a
tuple that includes a weight and exponent along with the OutputState for its entry in the list (see
`MonitoredOutputState Tuple <ObjectiveMechanism_OutputState_Tuple>`).  The outcome calcuated by the
`monitoring_mechanism <EVCMechanism.monitoring_mechanism>` is conveyed to the EVCMechanism and assigned as the
`value <InputState.value>` of its `primary InputState <InputState_Primary>`.


.. _EVCMechanism_Prediction_Mechanisms:

Prediction Mechanisms
~~~~~~~~~~~~~~~~~~~~~

These are used to provide input to the `system <EVCMechanism.system>` when the EVCMechanism's default `function
<EVCMechanism.function>` (`ControlSignalGridSearch`) `simulates its execution <EVC_Default_Configuration>` to evaluate
the EVC for each `allocation_policy`.  When an EVCMechanism is created, a prediction Mechanism is created for each
`ORIGIN` Mechanism in its `system <EVCMechanism.system>`, and for each `Projection` received by an `ORIGIN` Mechanism,
a `MappingProjection` from the same source is created that projects to the corresponding prediction Mechanism. The type
of `Mechanism` used for the prediction Mechanisms is specified by the EVCMechanism's `prediction_mechanism_type`
attribute, and their parameters can be specified with the `prediction_mechanism_params` attribute. The default type is
an 'IntegratorMechanism`, that calculates an exponentially weighted time-average of its input. The prediction mechanisms
for an EVCMechanism are listed in its `prediction_mechanisms` attribute.


.. _EVCMechanism_Functions:

Function
~~~~~~~~

The `function <EVCMechanism.function>` of an EVCMechanism returns an `allocation_policy` -- that is, an `allocation
<ControlSignal.allocation>` for each `ControlSignal` listed in its `control_signals <EVCMechanism.control_signals>`
attribute -- that are used by the Components of the EVCMechanism's `system <EVCMechanism.system>` in the next `TRIAL`
of execution.  The default is `ControlSignalGridSearch` (see `EVC_Default_Configuration`), but any function can be used
that returns an appropriate value (i.e., that specifies an `allocation_policy` for the number of `ControlSignals
<EVCMechanism_ControlSignals>` in the EVCMechanism's `control_signals` attribute, using the correct format for the
`allocation <ControlSignal.allocation>` value of each ControlSignal).  In addition to its primary `function
<EVCMechanism.function>`, an EVCMechanism has a set of `auxiliary functions <EVCMechanism_Auxiliary_Functions>` that
evaluate the performance of its `system <EVCMechanism.system>` (`outcome_function <EVCMechanism.outcome_function>`),
the `cost <ControlSignal.cost>` associated with its ControlSignals (`cost_function <EVCMechanism.cost_function>`), and
combine these (combine_outcome_and_cost_function <EVCMechanism.combine_outcome_and_cost_function>`) to calculate the
`EVC <EVCMechanism_EVC>`.  These functions are used by the EVCMechanism's default function to select an
`allocation_policy` with the maximum EVC among a range of policies specified by its ControlSignals, as described below.

.. _EVCMechanism_Default_Configuration:

Default Configuration of EVC Function and Auxiliary Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In its default configuration, an EVCMechanism simulates and evaluates the performance of its `system
<EVCMechanism.system>` under a set of allocation_policies determined by the `allocation_samples
<ControlSignal.allocation_samples>` attributes of its `ControlSignals <EVCMechanism_ControlSignals>`, and implements
the one that generates the maximum `EVC <EVCMechanism_EVC>`.  This is carried out by the EVCMechanism's
default `function <EVCMechanism.function>` and four auxiliary functions, as described below.

The default `function <EVCMechanism.function>` of an EVCMechanism is `ControlSignalGridSearch`. It identifies the
`allocation_policy` with the maximum `EVC <EVCMechanism_EVC>` by a conducting an exhaustive search over every possible
`allocation_policy`— that is, all combinations of `allocation <ControlSignal.allication>` values for its `ControlSignal
<EVCMechanism_ControlSignals>`, where the `allocation <ControlSignal.allication>` values sampled for each ControlSignal
are determined by its `allocation_samples` attribute.  For each `allocation_policy`, the EVCMechanism executes the
`system <EVCMechanism.system>`, evaluates the `EVC <EVCMechanism_EVC>` for that policy, and returns the
`allocation_policy` that yields the greatest EVC value. The following steps are used to calculate the EVC in each
`allocation_policy`:

  * **Implement the policy and simulate the system** - assign the `allocation <ControlSignal.allocation>` that the
    selected `allocation_policy` specifies for each ControlSignal, and then simulate these system using the
    corresponding parameter values.

  * **Calculate outcome** - combine the `value <OutputState.value>` \s of the OutputStates listed in the
    EVCMechanism's `monitored_output_states <EVCMechanism.monitored_output_states>` attribute using the function
    specified by its `outcome_function <EVCMechanism.outcome_function>` attribute (this is done by the EVCMechanism's
    `monitoring_mechanism <EVCMechanism.monitoring_mechanism>`, and passed to the EVCMechanism's `primary InputState
    <InputState_Primary>`);  the default is take the product of all the values, but this can be configured
    to weight and/or exponentiate individual values (see  documentation for `outcome_function
    <EVCMechanism.outcome_function>`).
  ..
  * **Calculate EVC** - by calling the EVCMechanism's `value_function <EVCMechanism.value_function>` with the
    outcome (received from the `monitoring_mechanism`) and a list of the `costs <ControlSignal.cost>` \s of its
    `ControlSignals <EVCMechanism_ControlSignals>`; default `value_function <EVCMechanism.value_function>` calls
    to additional auxiliary functions:  first it calls the `cost_function <EVCMechanism.cost_function>` which sums
    the costs, though this can be configured to weight and/or exponentiate individual costs (see `cost_function
    <EVCMechanism.cost_function>` attribute);  it then calls the `combine_outcome_and_cost_function
    <EVCMechanism.combine_outcome_and_cost_function>`, which subtracts the sum of the costs from the outcome
    to generate the EVC, though this too can be configured (see documentation for `combine_outcome_and_cost_function
    <EVCMechanism.combine_outcome_and_cost_function>`).

In addition to modifying the default functions (as noted above), any or all of them can be replaced with
a custom function to modify how the `allocation_policy` is determined, so long as the custom function accepts
arguments and return values that are compatible with any that call that function (see note below).

.. _EVCMechanism_Calling_and_Assigning_Functions:

    .. note::
       The `EVCMechanism auxilliary functions <EVC_Auxiliary_Functions>` described above are all implemented as
       PsyNeuLink `Functions <Function>`.  Therefore, to call a function itself, it must be referenced as
       ``<EVCMechanism>.<function_attribute>.function``.  A custom function assigned to one of the auxiliary functions
       can be either a PsyNeuLink `Function <Function>`, or a generic python function or method (including a lambda
       function).  If it is one of the latter, it is automatically "wrapped" as a PsyNeuLink `Function <Function>`
       (specifically, it is assigned as the `function <UserDefinedFunction.function>` attribute of a
       `UserDefinedFunction` object), so that it can be referenced and called in the same manner as
       the default function assignment. Therefore, once assigned, it too must be referenced as
       ``<EVCMechanism>.<function_attribute>.function``.

.. _EVCMechanism_ControlSignals:

ControlSignals
~~~~~~~~~~~~~~

The OutputStates of an EVCMechanism (like any `ControlMechanism`) are a set of `ControlSignals <ControlSignal>`, that
are listed in its `control_signals <EVCMechanism.control_signals>` attribute (as well as its `output_states
<ControlMechanism.output_states>` attribute).  Each ControlSignal is assigned a  `ControlProjection` that projects to
the `ParameterState` for a parameter controlled by the EVCMechanism.  When an EVCMechanism is `created automatically
<EVCMechanism_Creation>`, it is assigned one ControlSignal for each of the parameters `specified for control
<ControlMechanism_Control_Signals>` in its `system <EVCMechanism.system>`; if it is created directly, then it creates
one ControlSignal for each of the parameters specified in the **control_signals** argument of its constructor.
ControlSignals can be added to an EVCMechanism using its `assign_params` method.  Each ControlSignal is assigned an
item of the EVCMechanism's `allocation_policy`, that determines its `allocation <ControlSignal.allocation>` for a given
`TRIAL` of execution.  The `allocation <ControlSignal.allocation>` is used by a ControlSignal to determine its
`intensity <ControlSignal.intensity>`, which is then assigned as the `value <ConrolProjection.value>` of the
ControlSignal's ControlProjection.   The `value <ControlProjection>` of the ControlProjection is used by the
`ParameterState` to which it projects to modify the value of the parameter (see `ControlSignal_Modulation` for
description of how a ControlSignal modulates the value of a parameter it controls).  A ControlSignal also calculates a
`cost <ControlSignal.cost>`, based on its `intensity <ControlSignal.intensity>` and/or its time course. The
`cost <ControlSignal.cost>` is included in the evaluation that the EVCMechanism carries out for a given
`allocation_policy`, and that it uses to adapt the ControlSignal's `allocation  <ControlSignal.allocation>` in the
future.  When the EVCMechanism chooses an `allocation_policy` to evaluate,  it selects an allocation value from the
ControlSignal's `allocation_samples <ControlSignal.allocation_samples>` attribute.


.. _EVCMechanism_Execution:

Execution
---------

An EVCMechanism, like any `ControlMechanism`, is always the last `Mechanism` to be executed in a `TRIAL` for its
`system <EVCMechanism.system>` (see `System Control <System_Execution_Control>` and `Execution <System_Execution>`).
When an EVCMechanism is executed, it updates the value of its `prediction_mechanisms` and `monitoring_mechanism`,
and then calls its `function <EVCMechanism.function>`, which determines and implements the `allocation_policy` for
the next `TRIAL` of its `system <EVCMechanism.system>` \s execution.  The default `function <EVCMechanism.function>`
executes the following steps (described in greater detailed `above <EVC_Default_Configuration>`):

* samples every allocation_policy (i.e., every combination of the `allocation` \s specified for the EVCMechanism's
  ControlSignals specified by their `allocation_samples` attributes);  for each `allocation_policy`, it:

  * Executes the EVCMechanism's `system <EVCMechanism.system>` with the parameter values specified by that
    `allocation_policy`;  this includes the EVCMechanism's `monitoring_mechanism`, which executes the function
    specified in the EVCMechanism's `outcome_function <EVCMechanism.outcome_function>`, and provides the result
    to the EVCMechanism.

  * Calls the EVCMechanism's `value_function <EVCMechanism.value_function>`, which in turn calls EVCMechanism's
    `cost_function <EVCMechanism.cost_function>` and `combine_outcome_and_cost_function
    <EVCMechanism.combine_outcome_and_cost_function>` to evaluate the EVC for that `allocation_policy`.

  * Selects and returns the `allocation_policy` that generates the maximum EVC value.

This procedure can be modified by specifying a custom function for any or all of the `functions
<EVC_Auxiliary_Functions>` referred to above.


.. _EVCMechanism_Examples:

Examples
--------

The following example implements a system with an EVCMechanism (and two processes not shown)::

    mySystem = system(processes=[myRewardProcess, myDecisionProcess],
                      controller=EVCMechanism,
                      monitor_for_control=[Reward, DDM_DECISION_VARIABLE,(RESPONSE_TIME, -1, 1)],

It uses the system's `monitor_for_control` argument to assign three outputStates to be monitored.  The first one
references the Reward mechanism (not shown);  its `primary outputState <OutputState_Primary>` will be used by default.
The second and third use keywords that are the names of outputStates of a  `DDM` mechanism (also not shown).
The last one (RESPONSE_TIME) is assigned an exponent of -1 and weight of 1. As a result, each calculation of the EVC
computation will multiply the value of the primary outputState of the Reward mechanism by the value of the
DDM_DECISION_VARIABLE outputState of the DDM mechanism, and then divide that by the value of the RESPONSE_TIME
outputState of the DDM mechanism.

COMMENT:
ADD: This example specifies the EVCMechanism on its own, and then uses it for a system.
COMMENT


.. _EVCMechanism_Class_Reference:

Class Reference
---------------

"""
import numbers

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import function_type
from PsyNeuLink.Components.Functions.Function import ModulationParam, _is_modulation_param
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCAuxiliary import ControlSignalGridSearch, ValueFunction
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismList
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.ShellClasses import Function
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import AUTO_ASSIGN_MATRIX, CONTROL, COST_FUNCTION, EVC_MECHANISM, EXPONENT, FUNCTION, INITIALIZING, INIT_FUNCTION_METHOD_ONLY, MAKE_DEFAULT_CONTROLLER, MONITOR_FOR_CONTROL, NAME, OUTCOME_FUNCTION, PARAMETER_STATES, PREDICTION_MECHANISM, PREDICTION_MECHANISM_PARAMS, PREDICTION_MECHANISM_TYPE, PRODUCT, SUM, WEIGHT
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import ContentAddressableList
from PsyNeuLink.Scheduling.TimeScale import CentralClock, Clock, TimeScale

OBJECT_INDEX = 0
WEIGHT_INDEX = 1
EXPONENT_INDEX = 2

# -------------------------------------------    KEY WORDS  -------------------------------------------------------

MONITORING_MECHANISM = 'monitoring_mechanism'
ALLOCATION_POLICY = 'allocation_policy'

# ControlSignal Costs
INTENSITY_COST = 'INTENSITY COST'
ADJUSTMENT_COST = 'ADJUSTMENT COST'
DURATION_COST = 'DURATION COST'

# ControlSignal Cost Function Names
INTENSITY_COST_FUNCTION = 'intensity_cost_function'
ADJUSTMENT_COST_FUNCTION = 'adjustment_cost_function'
DURATION_COST_FUNCTION = 'duration_cost_function'
COST_COMBINATION_FUNCTION = 'cost_combination_function'
costFunctionNames = [INTENSITY_COST_FUNCTION,
                     ADJUSTMENT_COST_FUNCTION,
                     DURATION_COST_FUNCTION,
                     COST_COMBINATION_FUNCTION]

# Attributes / KVO keypaths
# kpLog = "Control Signal Log"
kpAllocation = "Control Signal Allocation"
kpIntensity = "Control Signal Intensity"
kpCostRange = "Control Signal Cost Range"
kpIntensityCost = "Control Signal Intensity Cost"
kpAdjustmentCost = "Control Signal Adjustment Cost"
kpDurationCost = "Control Signal duration_cost"
kpCost = "Control Signal Cost"


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EVCMechanism(ControlMechanism_Base):
    """EVCMechanism(                                                   \
    prediction_mechanism_type=IntegratorMechanism,                     \
    prediction_mechanism_params=None,                                  \
    monitor_for_control=None,                                          \
    control_signals=None,                                              \
    function=ControlSignalGridSearch                                   \
    value_function=ValueFunction,                                      \
    outcome_function=LinearCombination(operation=PRODUCT),             \
    cost_function=LinearCombination(operation=SUM),                    \
    combine_outcome_and_cost_function=LinearCombination(operation=SUM) \
    save_all_values_and_policies:bool=:keyword:`False`,                \
    params=None,                                                       \
    name=None,                                                         \
    prefs=None)

    Subclass of `ControlMechanism` that optimizes the `ControlSignals <ControlSignal>` for a `System`.

    COMMENT:
        Class attributes:
            + componentType (str): System Default Mechanism
            + paramClassDefaults (dict):
                + SYSTEM (System)
                + MONITOR_FOR_CONTROL (list of Mechanisms and/or OutputStates)

        Class methods:
            None

       **********************************************************************************************

       PUT SOME OF THIS STUFF IN ATTRIBUTES, BUT USE DEFAULTS HERE

        # - specification of system:  required param: SYSTEM
        # - kwDefaultController:  True =>
        #         takes over all unassigned ControlProjections (i.e., without a sender) in its system;
        #         does not take monitored states (those are created de-novo)
        # TBI: - CONTROL_PROJECTIONS:
        #         list of projections to add (and for which outputStates should be added)

        # - input_states: one for each performance/environment variable monitiored

        ControlProjection Specification:
        #    - wherever a ControlProjection is specified, using kwEVC instead of CONTROL_PROJECTION
        #     this should override the default sender SYSTEM_DEFAULT_CONTROLLER in ControlProjection._instantiate_sender
        #    ? expclitly, in call to "EVC.monitor(input_state, parameter_state=NotImplemented) method

        # - specification of function: default is default allocation policy (BADGER/GUMBY)
        #   constraint:  if specified, number of items in variable must match number of input_states in INPUT_STATES
        #                  and names in list in kwMonitor must match those in INPUT_STATES

       **********************************************************************************************

       NOT CURRENTLY IN USE:

        system : System
            system for which the EVCMechanism is the controller;  this is a required parameter.

        default_variable : Optional[number, list or np.ndarray] : `defaultControlAllocation <LINK]>`

    COMMENT


    Arguments
    ---------

    system : System : default None
        specifies the `System` for which the EVCMechanism should serve as a `controller <System_Base.controller>`;
        the EVCMechanism will inherit any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <EVCMechanism.system>` \'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    prediction_mechanism_type : CombinationFunction: default IntegratorMechanism
        the `Mechanism` class used for `prediction mechanism(s) <EVCMechanism_Prediction_Mechanisms>`.
        Each instance is named using the name of the `ORIGIN` Mechanism + "PREDICTION_MECHANISM"
        and assigned an `OutputState` with a name based on the same.

    prediction_mechanism_params : Optional[Dict[param keyword, param value]] : default None
        a `parameter dictionary <ParameterState_Specification>` passed to the constructor for a Mechanism
        of `prediction_mechanism_type`. The same parameter dictionary is passed to all
        `prediction mechanisms <EVCMechanism_Prediction_Mechanisms>` created for the EVCMechanism.

    monitor_for_control : List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d np.array]] : \
    default MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES
        specifies set of `OutputStates <OutputState>` to monitor (see `ControlMechanism_Monitored_OutputStates` for
        specification options).

    control_signals : List[Attribute of Mechanism or its function, ParameterState, or tuple[str, Mechanism]
        specifies the parameters to be controlled by the EVCMechanism
        (see `control_signals <EVCMechanism.control_signals>` for details).

    function : function or method : ControlSignalGridSearch
        specifies the function used to determine the `allocation_policy` for the next execution of the
        EVCMechanism's `system <EVCMechanism.system>` (see `function <EVCMechanism.function>` for details).

    value_function : function or method : value_function
        specifies the function used to calculate the `EVC <EVCMechanism_EVC>` for the current `allocation_policy`
        (see `value_function <EVCMechanism.value_function>` for details).

    outcome_function : function or method : LinearCombination(operation=PRODUCT)
        specifies the function used to calculate the outcome associated with the current `allocation_policy`
        (see `outcome_function <EVCMechanism.outcome_function>) for details).

    cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to calculate the cost associated with the current `allocation_policy`
        (see `cost_function <EVCMechanism.cost_function>` for details).

    combine_outcome_and_cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to combine the outcome and cost associated with the current `allocation_policy`,
        to determine its value (see `combine_outcome_and_cost_function` for details).

    save_all_values_and_policies : bool : default False
        specifes whether to save every `allocation_policy` tested in `EVC_policies` and their values in `EVC_values`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the mechanism, its function, and/or a custom function and its parameters.  Values specified
        for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default EVCMechanism-<index>
        a string used for the name of the mechanism.
        If not is specified, a default is assigned by `MechanismRegistry`
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict] : default Process.classPreferences
        the `PreferenceSet` for the mechanism.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see `PreferenceSet <LINK>` for details).

    Attributes
    ----------

    make_default_controller : bool : default True
        if `True`, calls deferred_init() for each `ControlProjection` in its system without a sender,
        creates a `ControlSignal` for it, and assigns itself as its sender.

    system : System
        the `system <System>` for which EVCMechanism is the `controller <System_Base.controller>`.

    control_signals : ContentAddressableList[ControlSignal]
        list of the EVCMechanism's `ControlSignals <EVCMechanism_ControlSignals>`, including any that it inherited
        from its `system <EVCMechanism.system>`.

    prediction_mechanisms : List[ProcessingMechanism]
        list of `predictions mechanisms <EVCMechanism_Prediction_Mechanisms>` generated for the EVCMechanism's
        `system <EVCMechanism.system>` when the EVCMechanism is created, one for each `ORIGIN` Mechanism in the System.

    origin_prediction_mechanisms : Dict[ProcessingMechanism, ProcessingMechanism]
        dictionary of `prediction mechanisms <EVCMechanism_Prediction_Mechanisms>` added to the EVCMechanism's
        `system <EVCMechanism.system>`, one for each of its `ORIGIN` Mechanisms.  The key for each
        entry is an `ORIGIN` Mechanism of the System, and the value is the corresponding prediction mechanism.

    prediction_mechanism_type : ProcessingMechanism : default IntegratorMechanism
        the `ProcessingMechanism` class used for `prediction mechanism(s) <EVCMechanism_Prediction_Mechanisms>`.
        Each instance is named based on `ORIGIN` mechanism + "PREDICTION_MECHANISM", and assigned an `OutputState`
        with a name based on the same.

    prediction_mechanism_params : Dict[param key, param value] : default None
        a `parameter dictionary <ParameterState_Specification>` passed to `prediction_mechanism_type` when
        the `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` is created.  The same dictionary will be passed
        to all instances of `prediction_mechanism_type` created.

    predicted_input : Dict[ProcessingMechanism, value]
        dictionary with the `value <Mechanism.Mechanism_Base.value>` of each `prediction mechanism
        <EVCMechanism_Prediction_Mechanisms>` listed in `prediction_mechanisms` corresponding to each ORIGIN
        mechanism of the system. The key for each entry is the name of an ORIGIN mechanism, and its
        value the `value <Mechanism.Mechanism_Base.value>` of the corresponding prediction mechanism.

    monitoring_mechanism : ObjectiveMechanism
        the 'ObjectiveMechanism' used by the EVCMechanism to evaluate the performance of its `system
        <EVCMechanism.system>`.  The OutputStates specified in the **monitor_for_control** argument of the
        EVCMechanism's constructor are assigned as the `monitored_values <ObjectiveMechanism.monitored_values>`
        attribute for the `monitoring_mechanism <EVCMechanism.monitoring_mechanism>`, the EVCMechanism's
        `outcome_function <EVCMechanism.outcome_function>` is assigned as the `function <ObjectiveMechanism.function>`
        for the `monitoring_mechanism <EVCMechanism.monitoring_mechanism>`, and a MappingProjection is assigned from
        it to the EVCMechanism's `primary InputState <InputState_Primary>`.

    monitored_output_states : List[OutputState]
        list of the OutputStates specified in the **monitor_for_control** argument of the EVCMechanism's constructor,
        and assigned to the `monitoring_mechanism <EVCMechanism.monitoring_mechanism>` for use in evaluating the
        performance of the EVCMechanism's `system <EVCMechanism.system>`.

    COMMENT:
    [TBI]
        monitored_values : 3D np.array
            an array of values of the outputStates in `monitored_output_states` (equivalent to the values of
            the EVCMechanism's `input_states <EVCMechanism.input_states>`).
    COMMENT

    monitor_for_control_weights_and_exponents: List[Tuple[scalar, scalar]]
        a list of tuples, each of which contains the weight and exponent (in that order) for an outputState in
        `monitored_outputStates`, listed in the same order as the outputStates are listed in `monitored_outputStates`.

    function : function : default ControlSignalGridSearch
        determines the `allocation_policy <EVCMechanism.allocation_policy>` to use for the next round of the system's
        execution. The default function, `ControlSignalGridSearch`, conducts an exhaustive (*grid*) search of all
        combinations of the `allocation_samples` of its ControlSignals (and contained in its
        `control_signal_search_space` attribute), by executing the system (using `run_simulation`) for each
        combination, evaluating the result using `value_function`, and returning the `allocation_policy` that yielded
        the greatest `EVC <EVCMechanism_EVC>` value (see `EVCMechanism_Default_Configuration` for additional details).
        If a custom function is specified, it must accommodate a **controller** argument that specifies an EVCMechanism
        (and provides access to its attributes, including `control_signal_search_space`), and must return an array with
        the same format (number and type of elements) as the EVCMechanism's `allocation_policy` attribute.

    COMMENT:
        NOTES ON API FOR CUSTOM VERSIONS:
            Gets controller as argument (along with any standard params specified in call)
            Must include **kwargs to receive standard args (variable, params, time_scale, and context)
            Must return an allocation policy compatible with controller.allocation_policy:
                2d np.array with one array for each allocation value

            Following attributes are available:
            controller._get_simulation_system_inputs gets inputs for a simulated run (using predictionMechamisms)
            controller._assign_simulation_inputs assigns value of prediction_mechanisms to inputs of ORIGIN mechanisms
            controller.run will execute a specified number of trials with the simulation inputs
            controller.monitored_states is a list of the mechanism outputStates being monitored for outcome
            controller.input_value is a list of current outcome values (values for monitored_states)
            controller.monitor_for_control_weights_and_exponents is a list of parameterizations for outputStates
            controller.control_signals is a list of control_signal objects
            controller.control_signal_search_space is a list of all allocationPolicies specifed by allocation_samples
            control_signal.allocation_samples is the set of samples specified for that control_signal
            [TBI:] control_signal.allocation_range is the range that the control_signal value can take
            controller.allocation_policy - holds current allocation_policy
            controller.output_values is a list of current control_signal values
            controller.value_function - calls the three following functions (done explicitly, so each can be specified)
            controller.outcome_function - aggregates monitored outcomes (using specified weights and exponentiation)
            controller.cost_function - aggregate costs of control signals
            controller.combine_outcome_and_cost_function - combines outcoms and costs
    COMMENT

    allocation_policy : 2d np.array : defaultControlAllocation
        determines the value assigned as the `variable <ControlSignal.variable>` for each `ControlSignal` and its
        associated `ControlProjection`.  Each item of the array must be a 1d array (usually containing a scalar)
        that specifies an `allocation` for the corresponding ControlSignal, and the number of items must equal the
        number of ControlSignals in the EVCMechanism's `control_signals` attribute.

    value_function : function : default ValueFunction
        calculates the `EVC <EVCMechanism_EVC>` for a given `allocation_policy`.  It takes as its arguments an
        `EVCMechanism`, an **outcome** value and a list or ndarray of **costs**, uses these to calculate an EVC,
        and returns a three item tuple with the calculated EVC, and the outcome value and aggregated value of costs
        used to calculate the EVC.  The default, `ValueFunction`,  calls the EVCMechanism's `cost_function
        <EVCMechanism.cost_function>` to aggregate the value of the costs, and then calls its
        `combine_outcome_and_costs <EVCMechanism.combine_outcome_and_costs>` to calculate the EVC from the outcome
        and aggregated cost (see `EVCMechanism_Default_Configuration` for additional details).  A custom
        function can be assigned to `value_function` so long as it returns a tuple with three items: the calculated
        EVC (which must be a scalar value), and the outcome and cost from which it was calculated (these can be scalar
        values or `None`). If used with the EVCMechanism's default `function <EVCMechanism.function>`, a custom
        `value_function` must accommodate three arguments (passed by name): a **controller** argument that is the
        EVCMechanism for which it is carrying out the calculation; an **outcome** argument that is a value; and a
        `costs` argument that is a list or ndarray.  A custom function assigned to `value_function` can also call any
        of the `helper functions <EVCMechanism_Auxiliary_Functions>` that it calls (however, see `note
        <EVCMechanism_Calling_and_Assigning_Functions>` above).

    outcome_function : function : default LinearCombination(operation=PRODUCT)
        calculates a measure of performance of the EVCMechanism's `system <EVCMechanism.system>` for the current
        `allocation_policy`.

        .. note::

            This function is not called by the EVCMechanism directly; it is assigned to its
            `monitoring_mechanism` when that is created (or when a new assignment is made to its `outcome_function
            <EVCMechanism.outcome_function>` attribute), and then called when the `monitoring_mechanism
            <EVCMechanism.monitoring>` executes.

        The default function combines the values of the OutputStates listed in the EVCMechanism's
        `monitored_output_states <EVCMechanism.monitored_output_states>` attribute (and the `monitoring_mechanism
        <EVCMechanism.monitoring_mechanism>` \'s `monitored_values <ObjectiveMechanism.monitored_values>` attribute)
        by taking their elementwise (Hadamard) product.  Any weighs and exponents `specified in a tuple
        <ObjectiveMechanism_OutputState_Tuple>` for an OutputState in the **monitor_for_control** argument of
        the EVCMechanism's constructor are used as the `weights <LinearCombination.weights>` and
        `exponents <LinearCombination.exponents>` parameters of the `LinearCombination` (default) function. The
        default function can be replaced with any `custom function <EVCMechanism_Calling_and_Assigning_Functions>` that
        accepts an array of values as its input and returns a scalar value.

    cost_function : function : default LinearCombination(operation=SUM)
        calculates the cost of the `ControlSignals <ControlSignal>` for the current `allocation_policy`.  The default
        function sums the `cost <ControlSignal.cost>` of each of the EVCMechanism's `ControlSignals
        <EVCMechanism_ControlSignals>`.  The `weights <LinearCombination.weights>` and/or `exponents
        <LinearCombination.exponents>` parameters of the function can be used, respectively, to scale and/or
        exponentiate the contribution of each ControlSignal cost to the combined cost.  These must be specified as
        1d arrays in a *WEIGHTS* and/or *EXPONENTS* entry of a `parameter dictionary <ParameterState_Specification>`
        assigned to the **params** argument of the constructor of a `LinearCombination` function; the length of
        each array must equal the number of (and the values listed in the same order as) the ControlSignals in the
        EVCMechanism's `control_signals <EVCMechanism.control_signals>` attribute. The default function can also be
        replaced with any `custom function <EVCMechanism_Calling_and_Assigning_Functions>` that takes an array as
        input and returns a scalar value.  If used with the EVCMechanism's default `value_function
        <EVCMechanism.value_function>`, a custom `cost_function <EVCMechanism.cost_function>` must accommodate two
        arguments (passed by name): a **controller** argument that is the EVCMechanism itself;  and a **costs**
        argument that is a 1d array of scalar values specifying the `cost <ControlSignal.cost>` for each ControlSignal
        listed in the `control_signals` attribute of the ControlMechanism specified in the **controller** argument.

    combine_outcome_and_cost_function : function : default LinearCombination(operation=SUM)
        combines the outcome and cost for given `allocation_policy` to determine its `EVC <EVCMechanisms_EVC>`. The
        default function subtracts the cost from the outcome, and returns the difference.  This can be modified using
        the `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>` parameters of the
        function, as described for the `cost_function <EVCMechanisms.cost_function>`.  The default function can also be
        replaced with any `custom function <EVCMechanism_Calling_and_Assigning_Functions>` that returns a scalar value.  If used with the EVCMechanism's default `value_function`, a custom
        If used with the EVCMechanism's default `value_function`, a custom combine_outcome_and_cost_function must
        accomoudate three arguments (passed by name): a **controller** argument that is the EVCMechanism itself; an
        **outcome** argument that is a 1d array with the outcome of the current `allocation_policy`; and a **cost**
        argument that is 1d array with the cost of the current `allocation_policy`.

    control_signal_search_space : 2d np.array
        an array each item of which is an `allocation_policy`.  By default, it is assigned the set of all possible
        allocation policies, using np.meshgrid to construct all permutations of `ControlSignal` values from the set
        specified for each by its `allocation_samples <EVCMechanism.allocation_samples>` attribute.

    EVC_max : 1d np.array with single value
        the maximum `EVC <EVCMechanism_EVC>` value over all allocation policies in `control_signal_search_space`.

    EVC_max_state_values : 2d np.array
        an array of the values for the OutputStates in `monitored_output_states` using the `allocation_policy` that
        generated `EVC_max`.

    EVC_max_policy : 1d np.array
        an array of the ControlSignal `intensity <ControlSignal.intensity> values for the allocation policy that
        generated `EVC_max`.

    save_all_values_and_policies : bool : default False
        specifies whether or not to save every `allocation_policy and associated EVC value (in addition to the max).
        If it is specified, each `allocation_policy` tested in the `control_signal_search_space` is saved in
        `EVC_policies`, and their values are saved in `EVC_values`.

    EVC_policies : 2d np.array
        array with every `allocation_policy` tested in `control_signal_search_space`.  The `EVC <EVCMechanism_EVC>`
        value of each is stored in `EVC_values`.

    EVC_values :  1d np.array
        array of `EVC <EVCMechanism_EVC>` values, each of which corresponds to an `allocation_policy` in `EVC_policies`;

    """

    componentType = EVC_MECHANISM
    initMethod = INIT_FUNCTION_METHOD_ONLY


    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    # This must be a list, as there may be more than one (e.g., one per control_signal)
    variableClassDefault = defaultControlAllocation

    from PsyNeuLink.Components.Functions.Function import LinearCombination
    # from Components.__init__ import DefaultSystem
    paramClassDefaults = ControlMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({MAKE_DEFAULT_CONTROLLER: True,
                               ALLOCATION_POLICY: None,
                               PARAMETER_STATES: False})

    @tc.typecheck
    def __init__(self,
                 system=None,
                 # default_variable=None,
                 # size=None,
                 prediction_mechanism_type=IntegratorMechanism,
                 prediction_mechanism_params:tc.optional(dict)=None,
                 monitor_for_control:tc.optional(list)=None,
                 control_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 function=ControlSignalGridSearch,
                 value_function=ValueFunction,
                 outcome_function=LinearCombination(operation=PRODUCT),
                 cost_function=LinearCombination(operation=SUM,
                                                 context=componentType+COST_FUNCTION),
                 combine_outcome_and_cost_function=LinearCombination(operation=SUM,
                                                                     context=componentType+FUNCTION),
                 save_all_values_and_policies:bool=False,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=componentType+INITIALIZING):

        # This is done here to hide it from IDE (where it would show if default assignment for arg in constructor)
        prediction_mechanism_params = prediction_mechanism_params or {MONITOR_FOR_CONTROL:None}

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(system=system,
                                                  prediction_mechanism_type=prediction_mechanism_type,
                                                  prediction_mechanism_params=prediction_mechanism_params,
                                                  monitor_for_control=monitor_for_control,
                                                  control_signals=control_signals,
                                                  modulation=modulation,
                                                  function=function,
                                                  value_function=value_function,
                                                  outcome_function=outcome_function,
                                                  cost_function=cost_function,
                                                  combine_outcome_and_cost_function=combine_outcome_and_cost_function,
                                                  save_all_values_and_policies=save_all_values_and_policies,
                                                  params=params)

        super(EVCMechanism, self).__init__(# default_variable=default_variable,
                                           # size=size,
                                           monitor_for_control=monitor_for_control,
                                           control_signals=control_signals,
                                           function=function,
                                           params=params,
                                           name=name,
                                           prefs=prefs,
                                           context=self)

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

    def _instantiate_input_states(self, context=None):
        """Instantiate inputState and MappingProjections for list of Mechanisms and/or States to be monitored

        """
        super()._instantiate_input_states(context=context)

        self._instantiate_prediction_mechanisms(context=context)
        self._instantiate_monitoring_mechanism(context=context)

    def _instantiate_prediction_mechanisms(self, context=None):
        """Add prediction mechanism and associated process for each ORIGIN (input) mechanism in the system

        Instantiate prediction_mechanisms for ORIGIN mechanisms in self.system; these will now be TERMINAL mechanisms
            - if their associated input mechanisms were TERMINAL MECHANISMS, they will no longer be so
            - therefore if an associated input mechanism must be monitored by the EVCMechanism, it must be specified
                explicitly in an outputState, mechanism, controller or systsem MONITOR_FOR_CONTROL param (see below)

        For each ORIGIN mechanism in self.system:
            - instantiate a corresponding predictionMechanism
            - instantiate a Process, with a pathway that projects from the ORIGIN to the prediction mechanism
            - add the process to self.system.processes

        Instantiate self.predicted_input dict:
            - key for each entry is an ORIGIN mechanism of the system
            - value of each entry is the value of the corresponding predictionMechanism:
            -     each value is a 2d array, each item of which is the value of an inputState of the predictionMechanism

        Args:
            context:
        """

        # Dictionary of prediction_mechanisms, keyed by the ORIGIN mechanism to which they correspond
        self.origin_prediction_mechanisms = {}

        # self.predictionProcesses = []

        # List of prediction mechanism tuples (used by system to execute them)
        self.prediction_mechs = []

        # Get any params specified for predictionMechanism(s) by EVCMechanism
        try:
            prediction_mechanism_params = self.paramsCurrent[PREDICTION_MECHANISM_PARAMS]
        except KeyError:
            prediction_mechanism_params = {}


        for origin_mech in self.system.origin_mechanisms.mechanisms:

            # # IMPLEMENT THE FOLLOWING ONCE INPUT_STATES CAN BE SPECIFIED IN CONSTRUCTION OF ALL MECHANISMS
            # #           (AS THEY CAN CURRENTLY FOR ObjectiveMechanisms)
            # state_names = []
            # variables = []
            # for state_name in origin_mech.input_states.keys():
            #     state_names.append(state_name)
            #     variables.append(origin_mech_intputStates[state_name].variable)

            # Instantiate predictionMechanism
            prediction_mechanism = self.paramsCurrent[PREDICTION_MECHANISM_TYPE](
                                                            name=origin_mech.name + " " + PREDICTION_MECHANISM,
                                                            default_variable = origin_mech.input_state.variable,
                                                            # default_variable=variables,
                                                            # INPUT_STATES=state_names,
                                                            params = prediction_mechanism_params,
                                                            context=context)
            prediction_mechanism._role = CONTROL
            prediction_mechanism.origin_mech = origin_mech

            # Assign projections to prediction_mechanism that duplicate those received by origin_mech
            #    (this includes those from ProcessInputState, SystemInputState and/or recurrent ones
            for orig_input_state, prediction_input_state in zip(origin_mech.input_states,
                                                            prediction_mechanism.input_states):
                for projection in orig_input_state.path_afferents:
                    MappingProjection(sender=projection.sender,
                                      receiver=prediction_input_state,
                                      matrix=projection.matrix)

            # Assign list of processes for which prediction_mechanism will provide input during the simulation
            # - used in _get_simulation_system_inputs()
            # - assign copy,
            #       since don't want to include the prediction process itself assigned to origin_mech.processes below
            prediction_mechanism.use_for_processes = list(origin_mech.processes.copy())

            # # FIX: REPLACE REFERENCE TO THIS ELSEWHERE WITH REFERENCE TO MECH_TUPLES BELOW
            self.origin_prediction_mechanisms[origin_mech] = prediction_mechanism

            # Add to list of EVCMechanism's prediction_object_items
            # prediction_object_item = prediction_mechanism
            self.prediction_mechs.append(prediction_mechanism)

            # Add to system executionGraph and executionList
            self.system.executionGraph[prediction_mechanism] = set()
            self.system.executionList.append(prediction_mechanism)

        self.prediction_mechanisms = MechanismList(self, self.prediction_mechs)

        # Assign list of destinations for predicted_inputs:
        #    the variable of the ORIGIN mechanism for each process in the system
        self.predicted_input = {}
        for i, origin_mech in zip(range(len(self.system.origin_mechanisms)), self.system.origin_mechanisms):
            # self.predicted_input[origin_mech] = self.system.processes[i].origin_mechanisms[0].input_value
            self.predicted_input[origin_mech] = self.system.processes[i].origin_mechanisms[0].variable

    # IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
    # FIX: MOVE THIS TO ControlMechanism??
    def _instantiate_monitoring_mechanism(self, context=None):
        """
        Assign InputState to ControlMechanism for each OutputState to be monitored;
            uses _instantiate_monitoring_input_state and _instantiate_control_mechanism_input_state to do so.
            For each item in self.monitored_output_states:
            - if it is a OutputState, call _instantiate_monitoring_input_state()
            - if it is a Mechanism, call _instantiate_monitoring_input_state for relevant Mechanism.outputStates
                (determined by whether it is a terminal mechanism and/or MonitoredOutputStatesOption specification)
            - each inputState is assigned a name with the following format:
                '<name of mechanism that owns the monitoredOutputState>_<name of monitoredOutputState>_Monitor'

        Notes:
        * self.monitored_output_states is a list, each item of which is a Mechanism.outputState from which a projection
            will be instantiated to a corresponding inputState of the ControlMechanism
        * self.input_states is the usual ordered dict of states,
            each of which receives a projection from a corresponding outputState in self.monitored_output_states
        """

        self._get_monitored_states(context=context)

        monitoring_input_states = []
        for i, state in enumerate(self.monitored_output_states):
            self._validate_monitored_state_in_system(state)
            monitoring_input_states.append({NAME: state.name,
                                            WEIGHT: float(self.monitor_for_control_weights_and_exponents[i][0]),
                                            EXPONENT: float(self.monitor_for_control_weights_and_exponents[i][1])
                                            })

        # Note: weights and exponents are assigned as parameters of outcome_function in _get_monitored_states
        self.monitoring_mechanism = ObjectiveMechanism(monitored_values=self.monitored_output_states,
                                                       # input_states=monitoring_input_states,
                                                       function=self.outcome_function,
                                                       name=self.name + ' Monitoring Mechanism')

        if self.prefs.verbosePref:
            print ("{0} monitoring:".format(self.name))
            for state in self.monitored_output_states:
                weight = self.monitor_for_control_weights_and_exponents[self.monitored_output_states.index(state)][0]
                exponent = self.monitor_for_control_weights_and_exponents[self.monitored_output_states.index(state)][1]
                print ("\t{0} (exp: {1}; wt: {2})".format(state.name, weight, exponent))

        MappingProjection(sender=self.monitoring_mechanism,
                          receiver=self,
                          matrix=AUTO_ASSIGN_MATRIX,
                          name = self.system.name + ' outcome signal'
                          )

        self.system.executionList.append(self.monitoring_mechanism)
        self.system.executionGraph[self.monitoring_mechanism] = set(self.system.executionList[:-1])

    def _get_monitored_states(self, context=None):
        """
        Parse paramsCurent[MONITOR_FOR_CONTROL] for system, controller, mechanisms and/or their outputStates:
            - if specification in outputState is None:
                 do NOT monitor this state (this overrides any other specifications)
            - if an outputState is specified in *any* MONITOR_FOR_CONTROL, monitor it (this overrides any other specs)
            - if a mechanism is terminal and/or specified in the system or controller:
                if MonitoredOutputStatesOptions is PRIMARY_OUTPUT_STATES:  monitor only its primary (first) outputState
                if MonitoredOutputStatesOptions is ALL_OUTPUT_STATES:  monitor all of its outputStates
            Note: precedence is given to MonitoredOutputStatesOptions specification in mechanism > controller > system

        Notes:
        * MonitoredOutputStatesOption is an AutoNumbered Enum declared in ControlMechanism
            - it specifies options for assigning outputStates of terminal Mechanisms in the System
                to self.monitored_output_states;  the options are:
                + PRIMARY_OUTPUT_STATES: assign only the `primary outputState <OutputState_Primary>` for each
                  TERMINAL Mechanism
                + ALL_OUTPUT_STATES: assign all of the outputStates of each terminal Mechanism
            - precedence is given to MonitoredOutputStatesOptions specification in mechanism > controller > system
        * self.monitored_output_states is a list, each item of which is a Mechanism.outputState from which a projection
            will be instantiated to a corresponding inputState of the ControlMechanism
        * self.input_states is the usual ordered dict of states,
            each of which receives a projection from a corresponding outputState in self.monitored_output_states

        """

        from PsyNeuLink.Components.Mechanisms.Mechanism import MonitoredOutputStatesOption
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import _validate_monitored_value

        # PARSE SPECS

        # Get controller's MONITOR_FOR_CONTROL specifications (optional, so need to try)
        try:
            controller_specs = self.paramsCurrent[MONITOR_FOR_CONTROL].copy() or []
        except KeyError:
            controller_specs = []

        # Get system's MONITOR_FOR_CONTROL specifications (specified in paramClassDefaults, so must be there)
        system_specs = self.system.monitor_for_control.copy()

        # If the controller has a MonitoredOutputStatesOption specification, remove any such spec from system specs
        if controller_specs:
            if (any(isinstance(item, MonitoredOutputStatesOption) for item in controller_specs)):
                option_item = next((item for item in system_specs if isinstance(item, MonitoredOutputStatesOption)),None)
                if option_item is not None:
                    del system_specs[option_item]
            for item in controller_specs:
                if item in system_specs:
                    del system_specs[system_specs.index(item)]

        # Combine controller and system specs
        # If there are none, assign PRIMARY_OUTPUT_STATES as default
        all_specs = controller_specs + system_specs or [MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]

        # Extract references to mechanisms and/or outputStates from any tuples
        # Note: leave tuples in all_specs for use in generating weight and exponent arrays below
        all_specs_extracted_from_tuples = []
        for item in all_specs:
            # VALIDATE SPECIFICATION
            # Handle EVCMechanism's tuple format:
            # MODIFIED 2/22/17: [DEPRECATED -- weights and exponents should be specified as params of the function]
            if isinstance(item, tuple):
                if len(item) != 3:
                    raise EVCError("Specification of tuple ({0}) in MONITOR_FOR_CONTROL for {1} "
                                         "has {2} items;  it should be 3".
                                         format(item, self.name, len(item)))
                if not isinstance(item[1], numbers.Number):
                    raise EVCError("Specification of the exponent ({0}) for MONITOR_FOR_CONTROL of {1} "
                                         "must be a number".
                                         format(item[1], self.name))
                if not isinstance(item[2], numbers.Number):
                    raise EVCError("Specification of the weight ({0}) for MONITOR_FOR_CONTROL of {1} "
                                         "must be a number".
                                         format(item[0], self.name))
                # Set state_spec to the output_state item for validation below
                item = item[0]
            # MODIFIED 2/22/17 END
            # Validate by ObjectiveMechanism:
            _validate_monitored_value(self, item, context=context)
            # Extract references from specification tuples
            if isinstance(item, tuple):
                all_specs_extracted_from_tuples.append(item[OBJECT_INDEX])
            # Otherwise, add item as specified:
            else:
                all_specs_extracted_from_tuples.append(item)

        # Get MonitoredOutputStatesOptions if specified for controller or System, and make sure there is only one:
        option_specs = [item for item in all_specs if isinstance(item, MonitoredOutputStatesOption)]
        if not option_specs:
            ctlr_or_sys_option_spec = None
        elif len(option_specs) == 1:
            ctlr_or_sys_option_spec = option_specs[0]
        else:
            raise EVCError("PROGRAM ERROR: More than one MonitoredOutputStatesOption specified in {}: {}".
                           format(self.name, option_specs))

        # Get MONITOR_FOR_CONTROL specifications for each mechanism and outputState in the System
        # Assign outputStates to self.monitored_output_states
        self.monitored_output_states = []

        # Notes:
        # * Use all_specs to accumulate specs from all mechanisms and their outputStates
        #     (for use in generating exponents and weights below)
        # * Use local_specs to combine *only current* mechanism's specs with those from controller and system specs;
        #     this allows the specs for each mechanism and its outputStates to be evaluated independently of any others
        controller_and_system_specs = all_specs_extracted_from_tuples.copy()

        for mech in self.system.mechanisms:

            # For each mechanism:
            # - add its specifications to all_specs (for use below in generating exponents and weights)
            # - extract references to Mechanisms and outputStates from any tuples, and add specs to local_specs
            # - assign MonitoredOutputStatesOptions (if any) to option_spec, (overrides one from controller or system)
            # - use local_specs (which now has this mechanism's specs with those from controller and system specs)
            #     to assign outputStates to self.monitored_output_states

            mech_specs = []
            output_state_specs = []
            local_specs = controller_and_system_specs.copy()
            option_spec = ctlr_or_sys_option_spec

            # PARSE MECHANISM'S SPECS

            # Get MONITOR_FOR_CONTROL specification from mechanism
            try:
                mech_specs = mech.paramsCurrent[MONITOR_FOR_CONTROL]

                if mech_specs is NotImplemented:
                    raise AttributeError

                # Setting MONITOR_FOR_CONTROL to None specifies mechanism's outputState(s) should NOT be monitored
                if mech_specs is None:
                    raise ValueError

            # Mechanism's MONITOR_FOR_CONTROL is absent or NotImplemented, so proceed to parse outputState(s) specs
            except (KeyError, AttributeError):
                pass

            # Mechanism's MONITOR_FOR_CONTROL is set to None, so do NOT monitor any of its outputStates
            except ValueError:
                continue

            # Parse specs in mechanism's MONITOR_FOR_CONTROL
            else:

                # Add mech_specs to all_specs
                all_specs.extend(mech_specs)

                # Extract refs from tuples and add to local_specs
                for item in mech_specs:
                    if isinstance(item, tuple):
                        local_specs.append(item[OBJECT_INDEX])
                        continue
                    local_specs.append(item)

                # Get MonitoredOutputStatesOptions if specified for mechanism, and make sure there is only one:
                #    if there is one, use it in place of any specified for controller or system
                option_specs = [item for item in mech_specs if isinstance(item, MonitoredOutputStatesOption)]
                if not option_specs:
                    option_spec = ctlr_or_sys_option_spec
                elif option_specs and len(option_specs) == 1:
                    option_spec = option_specs[0]
                else:
                    raise EVCError("PROGRAM ERROR: More than one MonitoredOutputStatesOption specified in {}: {}".
                                   format(mech.name, option_specs))

            # PARSE OUTPUT STATE'S SPECS

            # for output_state_name, output_state in list(mech.outputStates.items()):
            for output_state in mech.output_states:

                # Get MONITOR_FOR_CONTROL specification from outputState
                try:
                    output_state_specs = output_state.paramsCurrent[MONITOR_FOR_CONTROL]
                    if output_state_specs is NotImplemented:
                        raise AttributeError

                    # Setting MONITOR_FOR_CONTROL to None specifies outputState should NOT be monitored
                    if output_state_specs is None:
                        raise ValueError

                # outputState's MONITOR_FOR_CONTROL is absent or NotImplemented, so ignore
                except (KeyError, AttributeError):
                    pass

                # outputState's MONITOR_FOR_CONTROL is set to None, so do NOT monitor it
                except ValueError:
                    continue

                # Parse specs in outputState's MONITOR_FOR_CONTROL
                else:

                    # Note: no need to look for MonitoredOutputStatesOption as it has no meaning
                    #       as a specification for an outputState

                    # Add outputState specs to all_specs and local_specs
                    all_specs.extend(output_state_specs)

                    # Extract refs from tuples and add to local_specs
                    for item in output_state_specs:
                        if isinstance(item, tuple):
                            local_specs.append(item[OBJECT_INDEX])
                            continue
                        local_specs.append(item)

            # Ignore MonitoredOutputStatesOption if any outputStates are explicitly specified for the mechanism
            for output_state in mech.output_states:
                if (output_state in local_specs or output_state.name in local_specs):
                    option_spec = None


            # ASSIGN SPECIFIED OUTPUT STATES FOR MECHANISM TO self.monitored_output_states

            for output_state in mech.output_states:

                # If outputState is named or referenced anywhere, include it
                if (output_state in local_specs or output_state.name in local_specs):
                    self.monitored_output_states.append(output_state)
                    continue

# FIX: NEED TO DEAL WITH SITUATION IN WHICH MonitoredOutputStatesOptions IS SPECIFIED, BUT MECHANISM IS NEITHER IN
# THE LIST NOR IS IT A TERMINAL MECHANISM

                # If:
                #   mechanism is named or referenced in any specification
                #   or a MonitoredOutputStatesOptions value is in local_specs (i.e., was specified for a mechanism)
                #   or it is a terminal mechanism
                elif (mech.name in local_specs or mech in local_specs or
                              any(isinstance(spec, MonitoredOutputStatesOption) for spec in local_specs) or
                              mech in self.system.terminalMechanisms.mechanisms):
                    #
                    if (not (mech.name in local_specs or mech in local_specs) and
                            not mech in self.system.terminalMechanisms.mechanisms):
                        continue

                    # If MonitoredOutputStatesOption is PRIMARY_OUTPUT_STATES and outputState is primary, include it
                    if option_spec is MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES:
                        if output_state is mech.output_state:
                            self.monitored_output_states.append(output_state)
                            continue
                    # If MonitoredOutputStatesOption is ALL_OUTPUT_STATES, include it
                    elif option_spec is MonitoredOutputStatesOption.ALL_OUTPUT_STATES:
                        self.monitored_output_states.append(output_state)
                    elif mech.name in local_specs or mech in local_specs:
                        if output_state is mech.output_state:
                            self.monitored_output_states.append(output_state)
                            continue
                    elif option_spec is None:
                        continue
                    else:
                        raise EVCError("PROGRAM ERROR: unrecognized specification of MONITOR_FOR_CONTROL for "
                                       "{0} of {1}".
                                       format(output_state.name, mech.name))


        # ASSIGN WEIGHTS AND EXPONENTS TO OUTCOME_FUNCTION

        # Note: these values will be superseded by any assigned as arguments to the outcome_function
        #       if it is specified in the constructor for the mechanism

        num_monitored_output_states = len(self.monitored_output_states)
        weights = np.ones((num_monitored_output_states,1))
        exponents = np.ones_like(weights)

        # Get and assign specification of weights and exponents for mechanisms or outputStates specified in tuples
        for spec in all_specs:
            if isinstance(spec, tuple):
                object_spec = spec[OBJECT_INDEX]
                # For each outputState in monitored_output_states
                for item in self.monitored_output_states:
                    # If either that outputState or its owner is the object specified in the tuple
                    if item is object_spec or item.name is object_spec or item.owner is object_spec:
                        # Assign the weight and exponent specified in the tuple to that outputState
                        i = self.monitored_output_states.index(item)
                        weights[i] = spec[WEIGHT_INDEX]
                        exponents[i] = spec[EXPONENT_INDEX]

        # Assign weights and exponents to corresponding attributes of default OUTCOME_FUNCTION
        # Note: done here (rather than in call to outcome_function in value_function) for efficiency
        self.outcome_function.weights = weights
        self.outcome_function.exponents = exponents

        # Assign weights and exponents to monitor_for_control_weights_and_exponents attribute
        #    (so that it is accessible to custom functions)
        self.monitor_for_control_weights_and_exponents = list(zip(weights, exponents))

    def _validate_monitored_state_in_system(self, state_spec, context=None):
        """Validate specified outputstate is for a mechanism in the controller's system

        Called by both self._instantiate_monitoring_mechanism() and self.add_monitored_value() (in ControlMechanism)
        """

        # Get outputState's owner
        from PsyNeuLink.Components.States.OutputState import OutputState
        if isinstance(state_spec, OutputState):
            state_spec = state_spec.owner

        # Confirm it is a mechanism in the system
        if not state_spec in self.system.mechanisms:
            raise EVCError("Request for controller in {0} to monitor the outputState(s) of "
                                              "a mechanism ({1}) that is not in {2}".
                                              format(self.system.name, state_spec.name, self.system.name))

        # Warn if it is not a terminalMechanism
        if not state_spec in self.system.terminalMechanisms.mechanisms:
            if self.prefs.verbosePref:
                print("Request for controller in {0} to monitor the outputState(s) of a mechanism ({1}) that is not"
                      " a terminal mechanism in {2}".format(self.system.name, state_spec.name, self.system.name))

    def _instantiate_attributes_after_function(self, context=None):

        super()._instantiate_attributes_after_function(context=context)

        if not self.system.enable_controller:
            return

        outcome_Function = self.outcome_function
        cost_Function = self.cost_function

        if isinstance(outcome_Function, Function):
            # Insure that length of the weights and/or exponents arguments for the outcome_function
            #    matches the number of monitored_output_states
            num_monitored_output_states = len(self.monitored_output_states)
            if outcome_Function.weights is not None:
                num_outcome_weights = len(outcome_Function.weights)
                if  num_outcome_weights != num_monitored_output_states:
                    raise EVCError("The length of the weights argument {} for the {} of {} "
                                   "must equal the number of its monitored_output_states {}".
                                   format(num_outcome_weights,
                                          outcome_Function,
                                          self.name,
                                          num_monitored_output_states))
            if outcome_Function.exponents is not None:
                num_outcome_exponents = len(outcome_Function.exponents)
                if  num_outcome_exponents != num_monitored_output_states:
                    raise EVCError("The length of the exponents argument {} for the {} of {} "
                                   "must equal the number of its control signals {}".
                                   format(num_outcome_exponents,
                                          OUTCOME_FUNCTION,
                                          self.name,
                                          num_monitored_output_states))

        if isinstance(cost_Function, Function):
            # Insure that length of the weights and/or exponents arguments for the cost_function
            #    matches the number of control signals
            num_control_projections = len(self.control_projections)
            if cost_Function.weights is not None:
                num_cost_weights = len(cost_Function.weights)
                if  num_cost_weights != num_control_projections:
                    raise EVCError("The length of the weights argument {} for the {} of {} "
                                   "must equal the number of its control signals {}".
                                   format(num_cost_weights,
                                          COST_FUNCTION,
                                          self.name,
                                          num_control_projections))
            if cost_Function.exponents is not None:
                num_cost_exponents = len(cost_Function.exponents)
                if  num_cost_exponents != num_control_projections:
                    raise EVCError("The length of the exponents argument {} for the {} of {} "
                                   "must equal the number of its control signals {}".
                                   format(num_cost_exponents,
                                          COST_FUNCTION,
                                          self.name,
                                          num_control_projections))

    def _execute(self,
                    variable=None,
                    runtime_params=None,
                    clock=CentralClock,
                    time_scale=TimeScale.TRIAL,
                    context=None):
        """Determine allocation_policy for next run of system

        Update prediction mechanisms
        Construct control_signal_search_space (from allocation_samples of each item in control_signals):
            * get `allocation_samples` for each ControlSignal in `control_signals`
            * construct `control_signal_search_space`: a 2D np.array of control allocation policies, each policy of which
              is a different combination of values, one from the `allocation_samples` of each ControlSignal.
        Call self.function -- default is ControlSignalGridSearch
        Return an allocation_policy
        """

        self._update_predicted_input()
        # self.system._cache_state()

        # CONSTRUCT SEARCH SPACE

        control_signal_sample_lists = []
        control_signals = self.control_signals

        # Get allocation_samples for all ControlSignals
        num_control_signals = len(control_signals)

        for control_signal in self.control_signals:
            control_signal_sample_lists.append(control_signal.allocation_samples)

        # Construct control_signal_search_space:  set of all permutations of ControlProjection allocations
        #                                     (one sample from the allocationSample of each ControlProjection)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.control_signal_search_space = \
            np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,num_control_signals)

        # EXECUTE SEARCH

        # IMPLEMENTATION NOTE:
        # self.system._store_system_state()

        allocation_policy = self.function(controller=self,
                                          variable=variable,
                                          runtime_params=runtime_params,
                                          time_scale=time_scale,
                                          context=context)
        # IMPLEMENTATION NOTE:
        # self.system._restore_system_state()

        return allocation_policy

    def _update_predicted_input(self):
        """Assign values of prediction mechanisms to predicted_input

        Assign value of each predictionMechanism.value to corresponding item of self.predictedIinput
        Note: must be assigned in order of self.system.processes

        """

        # Assign predicted_input for each process in system.processes

        # The number of ORIGIN mechanisms requiring input should = the number of prediction mechanisms
        num_origin_mechs = len(self.system.origin_mechanisms)
        num_prediction_mechs = len(self.origin_prediction_mechanisms)
        if num_origin_mechs != num_prediction_mechs:
            raise EVCError("PROGRAM ERROR:  The number of ORIGIN mechanisms ({}) does not equal"
                           "the number of prediction_predictions mechanisms ({}) for {}".
                           format(num_origin_mechs, num_prediction_mechs, self.system.name))
        for origin_mech in self.system.origin_mechanisms:
            # Get origin mechanism for each process
            # Assign value of predictionMechanism to the entry of predicted_input for the corresponding ORIGIN mechanism
            self.predicted_input[origin_mech] = self.origin_prediction_mechanisms[origin_mech].value
            # self.predicted_input[origin_mech] = self.origin_prediction_mechanisms[origin_mech].outputState.value

    def add_monitored_values(self, states_spec, context=None):
        """Validate and then instantiate outputStates to be monitored by EVC

        Use by other objects to add a state or list of states to be monitored by EVC
        states_spec can be a Mechanism, OutputState or list of either or both
        If item is a Mechanism, each of its outputStates will be used
        All of the outputStates specified must be for a Mechanism that is in self.System

        Args:
            states_spec (Mechanism, MechanimsOutputState or list of either or both:
            context:
        """
        states_spec = list(states_spec)
        self._validate_monitored_state_in_system(states_spec, context=context)
        self._instantiate_monitored_output_states(states_spec, context=context)

    def run_simulation(self,
                       inputs,
                       allocation_vector,
                       runtime_params=None,
                       time_scale=TimeScale.TRIAL,
                       context=None):
        """
        Run simulation of `system <System>` for which the EVCMechanism is the `controller`.

        Arguments
        ----------

        inputs : List[input] or ndarray(input) : default default_variable
            the inputs used for each in a sequence of executions of the mechanism in the `system <System>`.  This
            should be the `value <Mechanism.Mechanism_Base.value> for each
            `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` listed in the `prediction_mechanisms`
            attribute.  The inputs are available from the `predicted_input` attribute.

        allocation_vector : (1D np.array)
            the allocation policy to use in running the simulation, with one allocation value for each of the
            EVCMechanism's ControlSignals (listed in `control_signals`).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]
            a dictionary that can include any of the parameters used as arguments to instantiate the mechanisms,
            their functions, or projection(s) to any of their states.  See `Mechanism_Runtime_Parameters` for a full
            description.

        time_scale :  TimeScale : default TimeScale.TRIAL
            specifies whether the mechanism is executed on the TIME_STEP or TRIAL time scale.

        """

        if self.value is None:
            # Initialize value if it is None
            self.value = self.allocation_policy

        # Implement the current allocation_policy over ControlSignals (outputStates),
        #    by assigning allocation values to EVCMechanism.value, and then calling _update_output_states
        for i in range(len(self.control_signals)):
            # self.control_signals[list(self.control_signals.values())[i]].value = np.atleast_1d(allocation_vector[i])
            self.value[i] = np.atleast_1d(allocation_vector[i])
        self._update_output_states(runtime_params=runtime_params, time_scale=time_scale,context=context)

        # Execute simulation run of system for the current allocation_policy
        sim_clock = Clock('EVC SIMULATION CLOCK')

        self.system.run(inputs=inputs, clock=sim_clock, time_scale=time_scale, context=context)

        # Get outcomes for current allocation_policy
        #    = the values of the monitored output states (self.input_states)
        #    stored in self.input_value = list(self.variable)
        # self.monitoring_mechanism.execute(context=EVC_SIMULATION)
        self._update_input_states(runtime_params=runtime_params, time_scale=time_scale,context=context)

        for i in range(len(self.control_signals)):
            self.control_signal_costs[i] = self.control_signals[i].cost


    # The following implementation of function attributes as properties insures that even if user sets the value of a
    #    function directly (i.e., without using assign_params), it will still be wrapped as a UserDefinedFunction.
    # This is done to insure they can be called by value_function in the same way as the defaults
    #    (which are all Functions), and so that they can be passed a params dict.

    # def wrap_function(self, function):
    #     if isinstance(function, function_type):
    #         return ValueFunction(function)
    #     elif inspect.isclass(assignment) and issubclass(assignment, Function):
    #         self._value_function = ValueFunction()
    #     else:
    #         self._value_function = assignment

    @property
    def value_function(self):
        return self._value_function

    @value_function.setter
    def value_function(self, assignment):
        if isinstance(assignment, function_type):
            self._value_function = ValueFunction(function)
        elif assignment is ValueFunction:
            self._value_function = ValueFunction()
        else:
            self._value_function = assignment

    @property
    def outcome_function(self):
        # # MODIFIED 7/27/17 OLD:
        # return self._outcome_function
        # MODIFIED 7/27/17 NEW:
        # Get outcome_function from monitoring_mechanism if it has the attribute
        if hasattr(self, MONITORING_MECHANISM) and self.monitoring_mechanism:
            # # Check that outcome_function is same as function of monitoring_mechanism
            # if not self._outcome_function == self.monitoring_mechanism.function_object:
            #     raise EVCError("PROGRAM ERROR: outcome_function for {} ({}) is not same as "
            #                    "monitoring_mechanism.function_object ({})".
            #                    format(self.name, self._outcome_function, self.monitoring_mechanism.function_object))
            return self.monitoring_mechanism.function_object
        #     return self._outcome_function
        # else:
        return self._outcome_function
        # MODIFIED 7/27/17 END

    @outcome_function.setter
    def outcome_function(self, value):
        from PsyNeuLink.Components.Functions.Function import UserDefinedFunction
        if isinstance(value, function_type):
            udf = UserDefinedFunction(function=value)
            self._outcome_function = udf
        # elif inspect.isclass(value) and issubclass(value, Function):
        #     self._outcome_function = UserDefinedFunction()
        else:
            self._outcome_function = value

        # MODIFIED 7/27/17 NEW:
        # Assign outcome_function to monitoring_mechanism
        if hasattr(self, MONITORING_MECHANISM):
            self.monitoring_mechanism.assign_params({FUNCTION:self.outcome_function})
        # MODIFIED 7/27/17 END

    @property
    def cost_function(self):
        return self._cost_function

    @cost_function.setter
    def cost_function(self, value):
        from PsyNeuLink.Components.Functions.Function import UserDefinedFunction
        if isinstance(value, function_type):
            udf = UserDefinedFunction(function=value)
            self._cost_function = udf
        else:
            self._cost_function = value

    @property
    def combine_outcome_and_cost_function(self):
        return self._combine_outcome_and_cost_function

    @combine_outcome_and_cost_function.setter
    def combine_outcome_and_cost_function(self, value):
        from PsyNeuLink.Components.Functions.Function import UserDefinedFunction
        if isinstance(value, function_type):
            udf = UserDefinedFunction(function=value)
            self._combine_outcome_and_cost_function = udf
        else:
            self._combine_outcome_and_cost_function = value
