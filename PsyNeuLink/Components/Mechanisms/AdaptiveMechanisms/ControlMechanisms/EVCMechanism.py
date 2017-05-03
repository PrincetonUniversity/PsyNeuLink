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
`ControlSignals <ControlSignal>` that regulate the performance of the system to which they belong. The
EVCMechanism is one of the most powerful, but also one of the most complex components in PsyNeuLink.  It is
designed to implement a form of the Expected Value of Control (EVC) Theory described in
`Shenhav et al. (2013) <https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_, which provides useful background concerning
the purpose and structure of the EVCMechanism.

An EVCMechanism has one `ControlSignal` for each parameter of the mechanism or function that it controls.  Each
ControlSignal is associated with a `ControlProjection`.  The ControlProjection regulates the value of the parameter
it controls, with the magnitude of that regulation determined by the ControlSignal's `intensity`. A
particular combination of ControlSignal intensities is called an `allocation_policy`. When a system is executed,
it concludes by executing the EVCMechanism, which determines the `allocation_policy` (i.e., the ControlSignal
intensities, and thereby the values of the parameters being controlled) in the next round of execution.

.. _EVCMechanism_EVC:

The procedure by which the EVCMechanism selects an `allocation_policy` when it is executed is determined by its
`function <EVCMechanism.function>` attribute. By default, this evaluates the performance of the system under every
possible `allocation_policy`, and chooses the best one. It does this by simulating the system under each
`allocation_policy`, and evaluating the expected value of control (EVC): a cost-benefit analysis that weighs
the cost of the ControlSignals against the outcome of performance for the given policy.  The EVCMechanism then
selects the `allocation_policy` that generates the maximum EVC, and that allocation_policy is implemented for the
next round of execution. Each step of this procedure can be modified, or it can be replaced entirely, by assigning
custom functions to corresponding parameters of the EVCMechanism, as described under `EVC_Calculation` below.

.. _EVCMechanism_Creation:

Creating an EVCMechanism
------------------------

An EVCMechanism is generated automatically when a system is created and an EVCMechanism is specified as its
`controller` attribute (see `Controller <System_Execution_Control>`).  However, it can also be created using the
standard Python method of calling its constructor. An EVCMechanism that has been constructed automatically can be
customized by assigning values to its attributes (e.g., its functions, as described under `EVC_Calculation` below).

When an EVCMechanism is constructed automatically, it creates an `ObjectiveMechanism` (specified in its
`montioring_mechanism` attribute) that is used to monitor and evaluate the system's performance.  The
ObjectiveMechanism monitors each mechanism and/or outputState listed in the EVCMechanism's
'monitor_for_control <EVCMechanism.monitor_for_control>` attribute, and evaluates them using the function specified in
the EVCMechanism's `outcome_function` attribute. This information is used to set the `allocation` values for the
EVCMechanism's  `ControlSignals <ControlSignal>`.  Each ControlSignal is implemented as an `outputState
<OutputState>` of the EVCMechanism, that is assigned a  `ControlProjections <ControlProjection>` which projects to the
`parameterStates <ParameterState>` for the parameters of the mechanisms and/or functions controlled by that
ControlSignal.  In addition, a set of prediction mechanisms  are created that are used to keep a running average of
inputs to the system over the course of multiple executions.   These averages are used to generate input to the
system when the EVCMechanism simulates its execution. Each of these specialized components is described in the 
sections that follow.

.. _EVCMechanism_Structure:

Structure
---------

.. _EVCMechanism_InputStates:
.. _EVCMechanism_MonitoredOutputStates:

ObjectiveMechanism
~~~~~~~~~~~~~~~~~~

An EVCMechanism uses an `ObjectiveMechanism` to evaluate the performance of the system. The ObjectiveMechanism is
assigned a projection from each of the mechanisms and/or outputStates specified in EVCMechanism's
`monitor_for_control <EVCMechanism.monitor_for_control>` attribute, which it evaluates using the function specified in
the EVCMechanism's `outcome_function` attribute.  By default, the ObjectiveMechanism is assigned a projection from
the `primary outputState <OutputState_Primary>` of every `TERMINAL` mechanism in the system, and its function
calculates the product of their values.  However, the contribution of each item listed in
`monitor_for_control <EVCMechanism.monitor_for_control>` can be specified using the EVCMechanism's
`monitor_for_control_weights_and_exponents` attribute` (see `below <EVCMechanism_Examples>` for examples).
The outputStates of the system being monitored by an EVCMechanism are listed in its `monitored_output_states` attribute.

.. _EVC_Function

Function
~~~~~~~~

The `function <EVCMechanism.function>` of an EVCMechanism returns an `allocation_policy` -- that is, the `intensity` of
each of its `ControlSignals <ControlSignal>` -- that will be used in the next round of the system's execution.  Any
function can be used that returns an appropriate value (i.e., that specifies an `allocation_policy` for the exact
number of ControlSignals in the EVCMechanism's `controlSignals` attribute, using the correct format for the `allocation`
value of each ControlSignal). The default function is `ControlSignalGridSearch`, which evaluates the performance of the
system under a range of specified allocationPolicies, and returns the `allocation_policy` that generates the best
performance (the greatest EVC). This evaluation and selection procedure, including the four evaluation functions that it
uses (all of which are customizable), is described below.

.. _EVC_Calculation:

EVC Calculation
^^^^^^^^^^^^^^^

The default EVC `function <EVCMechanism.function>` calculates the expected value of control (EVC) by a conducting a
grid search over every possible `allocation_policy`.  The set of allocationPolicies sampled is determined by the
`allocation_samples` attribute of each `ControlSignal`. Each policy is constructed by drawing one value from the
`allocation_samples` attribute of each of the EVCMechanism's ControlSignals.  An `allocation_policy` is constructed
for every possible combination of values, and stored in the EVCMechanism's `controlSignalSearchSpace` attribute.  The
EVCMechanism's `run_simulation` method is then used to simulate the system under each `allocation_policy` in
`controlSignalSearchSpace`, calculate the EVC for each of those policies, and return the policy with the greatest EVC.
By default, only the maximum EVC is saved and returned.  However, by setting the `save_all_values_and_policies`
attribute to `True`, each policy and its EVC can be saved for each simulation run (in `EVC_policies` and `EVC_values`,
respectively). The EVC is calculated for each policy using the following four functions, each of which can be
customized by using the EVCMechanism's `assign_params` method to designate custom functions (the safest way),
or by assigning them directly to the corresponding attribute (see `note <EVCMechanism_Calling_and_Assigning_Functions>`
below):

COMMENT:
  [TBI:]  The ``controlSignalSearchSpace`` described above is constructed by default.  However, this can be customized
          by assigning either a 2d array or a function that returns a 2d array to the ``controlSignalSearchSpace``
          attribute.  The first dimension (or axis 0) of the 2d array must be an array of control allocation
          policies (of any length), each of which contains a value for each ControlProjection in the
          EVCMechanism, assigned in the same order they are listed in its ``controlProjections`` attribute.
COMMENT

.. _EVC_Auxiliary_Functions:

COMMENT:
    MENTION HIERARCHY
COMMENT

* `value_function` - this is an "orchestrating" function that calls the three subordinate functions described below,
  which do the actual work of evaluating the performance of the system and the `cost` of the ControlSignals under the
  current `allocation_policy`, and combining these to calculate the EVC.  This function can be replaced with a
  user-defined function to fully customize the calculation of the EVC, by assigning a custom function to the
  `value_function` attribute of the EVCMechanism (see `note <EVCMechanism_Calling_and_Assigning_Functions>` below).
..
* `outcome_function` - this combines the values of the outputStates in the EVCMechanism's `monitored_output_states`
  attribute to generate an aggregated outcome value for the current `allocation_policy`. The default is the
  `LinearCombination` function, which computes an elementwise (Hadamard) product of the outputState values, using any
  `weights and/or exponents specified for the outputStates <ObjectiveMechanism_OutputState_Tuple>` to scale and/or
  exponentiate the contribution that each makes to the aggregated outcome (see `examples <EVCMechanism_Examples>`).
  Evaluation of the system's performance can be further customized by specifying a custom function for the
  EVCMechanism's `outcome_function` attribute (see `note <EVCMechanism_Calling_and_Assigning_Functions>` below).
..
* `cost_function` - this combines the `cost` of the EVCMechanism's ControlSignals to generate an aggregated cost for
  the current `allocation_policy`.  The default is the `LinearCombination` function, which sums the costs.  The
  evaluation of cost can be further customized by specifying a custom function for the `cost_function` attribute.
..
* `combine_outcome_and_cost_function` - this combines the aggregated outcome and aggregated `cost` values for the
  current `allocation_policy`, to determine the EVC for that policy.  The default is the `LinearCombination`
  function, which subtracts the aggregated cost from the aggregated outcome. The way in which the outcome and cost
  are combined to determine the EVC can be customized by specifying a custom function for the
  `combine_outcome_and_cost_function` attribute (see `note <EVCMechanism_Calling_and_Assigning_Functions>` below).

.. _EVCMechanism_Calling_and_Assigning_Functions:

.. note::
   The EVCMechanism function attributes described above are all implemented as PsyNeuLink `Function <Function>` objects
   (so that, among other reasons, they can be parameterized using a `params` dictionary).  Therefore, to call the
   function itself, it must be referenced as ``<EVCMechanism>.<function_attribute>.function``.  A custom function
   assigned to one of the function attributes can be either a PsyNeuLink `Function <Function>`, or a generic python
   function or method (including a lambda function).  However, if it is one of the latter, it is automatically
   "wrapped" as a PsyNeuLink `Function <Function>` (specifically, it is assigned as the
   `function <UserDefinedFunction.function>` attribute of a `UserDefinedFunction` object), so that it can be called
   in the same manner as the default function assignment. Therefore, once assigned, it too must be referenced as
   ``<EVCMechanism>.<function_attribute>.function``.


.. _EVCMechanism_ControlSignal:

ControlSignals
~~~~~~~~~~~~~~

A `ControlSignal` is used to regulate the parameter of a mechanism or its function. An EVCMechanism has one
ControlSignal for each parameter that it controls.  One `outputState <OutputState>` of the EVCMechanism is dedicated to
each of its ControlSignals, and the value of that outputState is the ControlSignal's `intensity`.  When an EVCMechanism
is `created automatically <EVCMechanism_Creation>`, it creates a ControlSignal for each parameter that has been
specified for control in the system (a parameter is specified  for control by assigning it a ControlProjection;
see `Mechanism_Parameters`).  The ControlSignals of an EVCMechanism are listed in it `controlSignals`
attribute. Each ControlSignal is associated with a `ControlProjection` that projects to the
`parameterState <ParameterState>` for the parameter controlled by that ControlSignal. The EVCMechanism's
`function <EVCMechanism.function>` assigns an `allocation` value to each of its ControlSignals. The
`allocation` for a given ControlSignal determines that ControlSignal's `intensity`, which is then assigned as the
value of the ControlSignal's ControlProjection.  The value of the ControlProjection is then used by the parameterState
to which it projects to modify the value of the parameter for which it is responsible.  A ControlSignal also
calculates a `cost`, based on its `intensity` and/or its time course. The `cost` is included in the evaluation that the
EVCMechanism carries out for a given `allocation_policy`, and that it uses to adapt the ControlSignal's `allocation` in
the future.  When the EVCMechanism chooses an `allocation_policy` to evaluate, it selects an allocation value from the
ControlSignal's `allocation_samples` attribute.

.. _EVCMechanism_Prediction_Mechanisms:

Prediction Mechanisms
~~~~~~~~~~~~~~~~~~~~~

Each time the EVCMechanism is executed, it `simulates the execution <EVCMechanism_Execution>` of the system
in order to evaluate the system's performance.  To do so, it must provide an input to the system.  It uses its
prediction mechanisms to do this.  Each prediction mechanism provides an estimate of the input to an `ORIGIN`
mechanism in the system, based on a running average of inputs to that mechanism in previous rounds of execution.
The EVCMechanism uses these estimates to provide input to the system each time it simulates it to evaluate its
performance.  When an EVCMechanism is `created automatically <EVCMechanism_Creation>`, a prediction mechanism is
created for each `ORIGIN` mechanism in the system. For each projection received by the `ORIGIN` mechanism,   
a `MappingProjection` from the same source is created that projects to the prediction mechanism.  The type of 
mechanism used for the prediction mechanisms can be specified using the EVCMechanism's
`prediction_mechanism_type` attribute, and their parameters can be specified using the EVCMechanism's
`prediction_mechanism_params` attribute.  The default type is an 'IntegratorMechanism`, that calculates an
exponentially weighted time-average of its input.  The prediction mechanisms for an EVCMechanism are listed in its
`predictionMechanisms` attribute.

.. _EVCMechanism_Execution:

Execution
---------

When an EVCMechanism is executed, it updates the value of its `predictionMechanisms` and `monitoring_mechanism`,
and then calls its `function <EVCMechanism.function>`, which determines and implements the `allocation_policy` for
the next round of the system's execution.  By default, the EVCMechanism identifies and implements the
`allocation_policy` that maximizes the EVC evaluated for the outputStates it is monitoring, as described below.
However, this procedure can be modified by specifying a custom function for any or all of the functions described below.

.. _EVCMechanism_Default_Function:

The default `function <EVCMechanism.function>` for an EVCMechanism selects an `allocation_policy` by assessing
the performance of the system under each of the policies in its `controlSignalSearchSpace`, and selecting the
one that yields the maximum EVC. The `controlSignalSearchSpace` is constructed by creating a set of
allocationPolicies that represent all permutations of the `allocation` values to be sampled for each ControlSignal.
Each `allocation_policy` in the set is constructed by drawing one value from the `allocation_samples` of each
ControlSignal, and the set contains all combinations of these values.  For each `allocation_policy`, the default
`function <EVCMechanism.function>` calls the EVCMechanism's `value_function` which, in turn, carries out the
following steps:

* **Implement the allocation_policy.** Assign the `allocation <ControlSignal.ControlSignal.allocation>` value specified
  for each ControlSignal.
..
* **Simulate performance.**  Execute the system under the current `allocation_policy` using the EVCMechanism's
  `run_simulation` method and the value of its `predicted_inputs` attribute as the input to the system (this uses
  the history of previous trials to generate an average expected input value).
..
* **Calculate the EVC for the allocation_policy.**  This uses three functions:

    * the `outcome_function` calculates the **outcome** for the allocation_policy by aggregating the value of the
      outputStates the EVCMechanism monitors (listed in its `monitored_output_states` attribute);
    ..
    * the `cost_function` calculates the **cost** of the allocation_policy by aggregating the `cost` of the
      EVCMechanism's ControlSignals;
    ..
    * the `combine_outcome_and_cost_function` calculates the **value** (EVC) of the allocation_policy by subtracting the
      aggregated cost from the aggregated outcome.

If the `save_all_values_and_policies` attribute is `True`, the allocation policy is saved in the
EVCMechanism's `EVC_policies` attribute, and its value is saved in the `EVC_values` attribute. The
`function <EVCMechanism.function>` returns the allocation_policy that yielded the maximum EVC. This is then
implemented by assigning the `allocation` specified for each ControlSignal by the designated allocation_policy.
These allocations determine the value of the parameters being controlled in the next round of the system's execution.

This procedure can be modified by assigning custom functions to any or all of the ones described above, including the
EVCMechanism's `function <EVCMechanism.function>` itself.  The requirements for each are described in the function
attribute entries below.

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
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCAuxiliary import \
    ControlSignalGridSearch, ValueFunction

from PsyNeuLink.Components.Process import Process_Base
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismList, MechanismTuple
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlMechanism import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection

OBJECT = 0
WEIGHT = 1
EXPONENT = 2

# # Default control allocation mode values:
# class DefaultControlAllocationMode(Enum):
#     GUMBY_MODE = 0.0
#     BADGER_MODE = 1.0
#     TEST_MODE = 240
# defaultControlAllocation = DefaultControlAllocationMode.BADGER_MODE.value
DEFAULT_ALLOCATION_SAMPLES = np.arange(0.1, 1.01, 0.3)

class ControlSignalCostOptions(IntEnum):
    NONE               = 0
    INTENSITY_COST     = 1 << 1
    ADJUSTMENT_COST    = 1 << 2
    DURATION_COST      = 1 << 3
    ALL                = INTENSITY_COST | ADJUSTMENT_COST | DURATION_COST
    DEFAULTS           = INTENSITY_COST

# -------------------------------------------    KEY WORDS  -------------------------------------------------------

ALLOCATION_POLICY = 'allocation_policy'
CONTROL_SIGNAL_COST_OPTIONS = 'controlSignalCostOptions'

# ControlProjection Function Names
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
    function=ControlSignalGridSearch                                   \
    value_function=ValueFunction,                                      \
    outcome_function=LinearCombination(operation=PRODUCT),             \
    cost_function=LinearCombination(operation=SUM),                    \
    combine_outcome_and_cost_function=LinearCombination(operation=SUM) \
    save_all_values_and_policies:bool=:keyword:`False`,                \
    params=None,                                                       \
    name=None,                                                         \
    prefs=None)

    Optimizes the `ControlSignals <ControlSignal>` for a sysem <System>`.

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
        #         takes over all projections from default Controller;
        #         does not take monitored states (those are created de-novo)
        # TBI: - CONTROL_PROJECTIONS:
        #         list of projections to add (and for which outputStates should be added)

        # - inputStates: one for each performance/environment variable monitiored

        ControlProjection Specification:
        #    - wherever a ControlProjection is specified, using kwEVC instead of CONTROL_PROJECTION
        #     this should override the default sender SYSTEM_DEFAULT_CONTROLLER in ControlProjection._instantiate_sender
        #    ? expclitly, in call to "EVC.monitor(input_state, parameter_state=NotImplemented) method

        # - specification of function: default is default allocation policy (BADGER/GUMBY)
        #   constraint:  if specified, number of items in variable must match number of inputStates in INPUT_STATES
        #                  and names in list in kwMonitor must match those in INPUT_STATES

       **********************************************************************************************

       NOT CURRENTLY IN USE:

        system : System
            system for which the EVCMechanism is the controller;  this is a required parameter.

        default_input_value : Optional[number, list or np.ndarray] : `defaultControlAllocation <LINK]>`

    COMMENT


    Arguments
    ---------

    prediction_mechanism_type : CombinationFunction: default IntegratorMechanism
        the mechanism class used for `prediction mechanism(s) <EVCMechanism_Prediction_Mechanisms>`.
        Each instance is named using the name of the `ORIGIN` mechanism + PREDICTION_MECHANISM
        and assigned an `outputState <OutputState>` with a name based on the same.

    prediction_mechanism_params : Optional[Dict[param keyword, param value]] : default None
        a `parameter dictionary <ParameterState_Specifying_Parameters>` passed to the constructor for the
        `prediction_mechanism_type` mechanism. The same one is passed to all
        `prediction mechanisms <EVCMechanism_Prediction_Mechanisms>` created for the EVCMechanism.

    monitor_for_control : List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d np.array]] : \
    default MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES
        specifies set of outputState values to monitor (see `ControlMechanism_Monitored_OutputStates` for
        specification options).

    function : function or method : ControlSignalGridSearch
        specifies the function used to determine the `allocation_policy` for the next execution of the system
        (see `function <EVCMechanism.function>` attribute for a description of the default function).

    value_function : function or method : value_function
        specifies the function used to calculate the value of the current `allocation_policy`
        (see `value_function` attribute for additional details).

    outcome_function : function or method : LinearCombination(operation=PRODUCT)
        specifies the function used to calculate the outcome associated with the current `allocation_policy`
        (see `outcome_function` attribute for additional details).

    cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to calculate the cost associated with the current `allocation_policy`
        (see `cost_function` attribute for additional details).

    combine_outcome_and_cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to combine the outcome and cost associated with the current `allocation_policy`,
        to determine its value (see `combine_outcome_and_cost_function` attribute for additional details).

    save_all_values_and_policies : bool : default False
        when it is :keyword:`True`, saves all of the control allocation policies tested in `EVC_policies` and their
        values in `EVC_values`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
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
        if `True`, calls deferred_init() for each ControlProjection in its system without a sender,
        creates a ControlSignal for it, and assigns itself as its sender.

    system : System
        the `system <System>` for which EVCMechanism is the `controller`.

    controlSignals : OrderedDict[str, ControlSignal]
        list of `outputStates <OutputState>` for the EVCMechanism, each of which corresponds to one of its
        ControlSignals.

    predictionMechanisms : MechanismList
        a list of `prediction mechanisms <EVCMechanism_Prediction_Mechanisms>` added to the system, along with any 
        `runtime_params <Mechanism.runtime_params>` and the `phase <Mechanism.phase>` in which they execute.

    origin_prediction_mechanisms : Dict[ProcessingMechanism, ProcessingMechanism]
        dictionary of `prediction mechanisms <EVCMechanism_Prediction_Mechanisms>` added to the `system <System>` for
        which the EVCMechanism is the `controller`, one for each of its `ORIGIN` mechanisms.  The key for each
        entry is an `ORIGIN` mechanism of the system, and the value is the corresponding prediction mechanism.

    COMMENT:
        predictionProcesses : List[Process]
            a list of prediction processes added to the system, each comprised of one of its `ORIGIN` mechanisms
            and the associated `prediction mechanism <EVCMechanism_Prediction_Mechanisms>`.
    COMMENT

    prediction_mechanism_type : ProcessingMechanism : default IntegratorMechanism
        the `ProcessingMechanism` class used for `prediction mechanism(s) <EVCMechanism_Prediction_Mechanisms>`.
        Each instance is named based on `ORIGIN` mechanism + PREDICTION_MECHANISM,
        and assigned an `outputState <OutputState>` with a name based on the same

    prediction_mechanism_params : Dict[param key, param value] : default None
        a `parameter dictionary <ParameterState_Specifying_Parameters>` passed to `prediction_mechanism_type` when
        the `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` is created.  The same dictionary will be passed
        to all instances of `prediction_mechanism_type` created.

    COMMENT:
        OLD
        predictedInput : 3d np.array
            array with the `value <Mechanism.Mechanism_Base.value>` of each
            `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` listed in `prediction_mechanisms`.  Each item of
            axis 0 corresponds to the `value <Mechanism.Mechanism_Base.value>` of a prediction mechanism,
            axis 1 an `inputState <InputState>` of that prediction mechanism, and
            axis 2 the elements of the input for that inputState.
    COMMENT

    predictedInput : dict
        dictionary with the `value <Mechanism.Mechanism_Base.value>` of each
        `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` listed in `prediction_mechanisms` corresponding
        to each ORIGIN mechanism of the system. The key for each entry is the name of an ORIGIN mechanism, and its
        value the `value <Mechanism.Mechanism_Base.value>` of the corresponding prediction mechanism.

    monitoring_mechanism : ObjectiveMechanism
        the 'ObjectiveMechanism' that monitors the mechanisms and/or outputStates used by the EVCMechanism to evaluate
        the system's performance.  Each mechanism and/or outputState listed in the EVCMechanism's
        `monitored_output_states` attribute projects to an inputState of the `monitoring_mechanism`.  The EVCMechanism's
        `outcome_function` is assiged as the `function <ObjectiveMechanism.function>` for `monitoring_mechanism`.
        Its result is provided by a projection from the `monitoring_mechanism` to the EVCMechanism.

    monitored_output_states : List[OutputState]
        each item is an outputState of a mechanism in the system that has been assigned a projection to a corresponding
        inputState of the EVCMechanism.

    COMMENT:
    [TBI]
        monitored_values : 3D np.array
            an array of values of the outputStates in `monitored_output_states` (equivalent to the values of
            the EVCMechanism's `inputStates <EVCMechanism.inputStates>`).
    COMMENT

    monitor_for_control_weights_and_exponents: List[Tuple[scalar, scalar]]
        a list of tuples, each of which contains the weight and exponent (in that order) for an outputState in
        `monitored_outputStates`, listed in the same order as the outputStates are listed in `monitored_outputStates`.

    function : function : default ControlSignalGridSearch
        determines the `allocation_policy <EVCMechanism.allocation_policy>` to use for the next round of the system's
        execution. The default function, `ControlSignalGridSearch`, conducts an exhaustive (*grid*) search of all
        combinations of the `allocation_samples` of its ControlSignals (and contained in its
        `controlSignalSearchSpace` attribute), by executing the system (using `run_simulation`) for each
        combination, evaluating the result using `value_function`, and returning the allocation_policy that generated
        the highest value.  If a custom function is specified, it must accommodate a :keyword:`controller` argument that
        specifies an EVCMechanism (and provides access to its attributes, including `controlSignalSearchSpace`),
        and must return an array with the same format (number and type of elements) as the EVCMechanism's
        `allocation_policy` attribute.

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
            controller.inputValue is a list of current outcome values (values for monitored_states)
            controller.monitor_for_control_weights_and_exponents is a list of parameterizations for outputStates
            controller.controlSignals is a list of controlSignal objects
            controller.controlSignalSearchSpace is a list of all allocationPolicies specifed by allocation_samples
            controlSignal.allocation_samples is the set of samples specified for that controlSignal
            [TBI:] controlSignal.allocation_range is the range that the controlSignal value can take
            controller.allocation_policy - holds current allocation_policy
            controller.outputValue is a list of current controlSignal values
            controller.value_function - calls the three following functions (done explicitly, so each can be specified)
            controller.outcome_function - aggregates monitored outcomes (using specified weights and exponentiation)
            controller.cost_function - aggregate costs of control signals
            controller.combine_outcome_and_cost_function - combines outcoms and costs
    COMMENT

    allocation_policy : 2d np.array : defaultControlAllocation
        determines the value assigned as the `variable <ControlSignal.variable>` for each `ControlSignal` and its
        associated `ControlProjection`.  Each item of the array must be a 1d array (usually containing a scalar)
        that specifies an `allocation` for the corresponding ControlSignal, and the number of items must equal the
        number of ControlSignals in the EVCMechanism's `controlSignals` attribute.

    value_function : function : default value_function()
        calculates the value for a given `allocation_policy`.  The default uses `outcome_function` to determine the
        outcome of the policy, `cost_function` to determine its cost, combines these using
        `combine_outcome_and_cost_function`, and returns the result as the first item of a three-item tuple, the second
        and third of which are the outcome and cost used to determine the result.  The default function can be
        replaced by any function that returns a tuple with three items: the calculated EVC (which must be a scalar
        value), and the outcome and cost from which it was calculated (these can be scalar values or `None`).
        If used with the EVCMechanism's default `function <EVCMechanism.function>`, a custom `value_function` must
        accommodate three arguments (passed by name): a :keyword:`controller` argument that is the EVCMechanism for
        which it is carrying out the calculation; an :keyword:`outcome` argument that is a scalar value that reflects
        the outcome of the function of the ObjectiveMechanism (based on the value of the outputStates being monitored
        (and specified in the EVCMechanism's `monitored_output_states` attribute;
        and a :keyword:`costs` argument that is a 2d array of costs, each item of which is the `cost` of a
        ControlSignal in the EVCMechanism's `controlSignals` attribute.  A custom function assigned to
        `value_function` can also call any of the other EVCMechanism functions described below (however,
        see `note <EVCMechanism_Calling_and_Assigning_Functions>` above).

    outcome_function : function : default LinearCombination(operation=PRODUCT)
        calculates the outcome for a given `allocation_policy`.  The default combines the values of the outputStates in
        `monitored_output_states` by taking their product, using the `LinearCombination` function.  The
        `weights and/or exponents specified for the outputStates <ControlMechanism_OutputState_Tuple>` (see
        examples <EVCMechanism_Examples>`) are used as the `weights` and `exponents` parameters of the
        `LinearCombination` function, respectively. If the default `outcome_function` is called by a custom
        `value_function`, the weights and/or exponents can be specified as 1d arrays in a `WEIGHTS` and/or `EXPONENTS`
        entry of a `parameter dictionary <ParameterState_Specifying_Parameters>` specified for the `params` argument of
        the `LinearCombination` function. The length of each array must equal the number of (and values be listed in
        the same order as) the outputStates in the EVCMechanism's `monitored_output_states` attribute.  These
        specifications will supercede any made for individual outputStates in the `monitor_for_control` argument or
        `MONITOR_FOR_CONTROL <monitor_for_control>` entry of a parameter specification dictionary for the
        EVCMechanism (see `ControlMechanism_Monitored_OutputStates`).  The default function can also be replaced
        with any `custom function <EVCMechanism_Calling_and_Assigning_Functions>` that returns a scalar value.  If
        used with the EVCMechanism's default `value_function`, a custom outcome_function must accommodate two
        arguments (passed by name): a :keyword:`controller` argument that is the EVCMechanism itself (and can be used
        access to its attributes, including the `monitor_for_control_weights_and_exponents` attribute that lists the
        weights and exponents assigned to each outputState being monitored);  and an :keyword:`outcome` argument,
        that is a scalar value specifying the result of the ObjectiveMechanism's function (based on the outputStates
        listed in the `monitored_output_states` attribute of the :keyword:`controller` argument).

    cost_function : function : default LinearCombination(operation=SUM)
        calculates the cost for a given `allocation_policy`.  The default combines the `cost` of each ControlSignals in
        `controlSignals` by summing them using the `LinearCombination` function. If the default `cost_function` is
        called by a custom `value_function`, the weights and/or exponents parameters of the function can be used,
        respectively, to scale and/or exponentiate the contribution of each ControlSignal's cost to the aggregated
        value.  These must be specified as 1d arrays in a `WEIGHTS` and/or `EXPONENTS` entry of a
        `parameter dictionary <ParameterState_Specifying_Parameters>` specified for the `params` argument of the
        `LinearCombination` function; the length of each array must equal the number of (and the values listed in the
        same order as) the ControlSignals in the EVCMechanism's `controlSignals` attribute, and be in the same order.
        The default function can also be replaced with any
        `custom function <EVCMechanism_Calling_and_Assigning_Functions>` that returns a scalar value.  If used with
        the EVCMechanism's default `value_function`, a custom cost_function must accommodate two arguments (passed by
        name): a :keyword:`controller` argument that is the EVCMechanism itself;  and a
        :keyword:`costs` argument, that is 1d array of scalar values specifying the `cost` for each ControlSignal listed
        in the `controlSignals` attribute of the :keyword:`controller` argument.

    combine_outcome_and_cost_function : function : default LinearCombination(operation=SUM)
        combines the outcome and cost for given `allocation_policy` to determine its value.  The default uses the
        `LinearCombination` function to subtract the cost from the outcome, and returns the difference.  If the
        default `combine_outcome_and_cost_function` is called by a custom `value_function`, the weights and/or
        exponents parameters of the `LinearCombination` function can be used, respectively, to scale and/or exponentiate
        the contribution of the outcome and/or cost to the result.  These must be specified as 1d arrays in a `WEIGHTS`
        and/or EXPONENTS entry of a  `parameter specifiction dictionary <ParameterState_Specifying_Parameters>` 
        assigned to the function's `params` argument; each array must have two elements, the first for the outcome 
        and second for the cost. The default function can also be replaced with any
        `custom function <EVCMechanism_Calling_and_Assigning_Functions>` that returns a scalar value.  If used with
        the EVCMechanism's default `value_function`, a custom combine_outcome_and_cost_function must accomoudate three
        arguments (passed by name): a :keyword:`controller` argument that is the EVCMechanism itself; an
        :keyword:`outcome` argument that is a 1d array with the outcome of the current `allocation_policy`; and a
        :keyword:`cost` argument that is 1d array with the cost of the current `allocation_policy`.

    controlSignalSearchSpace : 2d np.array
        an array that contains arrays of allocation policies.  Each allocation policy contains one value for each of
        the mechanism's ControlSignals.  By default, it is assigned a set of all possible allocation policies
        (using np.meshgrid to construct all permutations of ControlSignal values).

    EVC_max : 1d np.array with single value
        the maximum EVC value over all allocation policies in `controlSignalSearchSpace`.

    EVC_max_state_values : 2d np.array
        an array of the values for the outputStates in `monitored_output_states` using the allocation policy that
        generated `EVC_max`.

    EVC_max_policy : 1d np.array
        an array of the ControlSignal intensity values for the allocation policy that generated `EVC_max`.

    save_all_values_and_policies : bool : default False
        specifies whether or not to save all allocation policies and associated EVC values (in addition to the max).
        If it is specified, each policy tested in the `controlSignalSearchSpace` is saved in `EVC_policies` and their
        values are saved in `EVC_values`.

    EVC_policies : 2d np.array
        array of allocation policies tested in `controlSignalSearchSpace`.  The values of each are stored in
        `EVC_values`.

    EVC_values :  1d np.array
        array of EVC values corresponding to the policies in `EVC_policies`.

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

    # This must be a list, as there may be more than one (e.g., one per controlSignal)
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
                 # default_input_value=None,
                 prediction_mechanism_type=IntegratorMechanism,
                 prediction_mechanism_params:tc.optional(dict)=None,
                 monitor_for_control:tc.optional(list)=None,
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
                                                  function=function,
                                                  value_function=value_function,
                                                  outcome_function=outcome_function,
                                                  cost_function=cost_function,
                                                  combine_outcome_and_cost_function=combine_outcome_and_cost_function,
                                                  save_all_values_and_policies=save_all_values_and_policies,
                                                  params=params)

        super(EVCMechanism, self).__init__(# default_input_value=default_input_value,
                                           monitor_for_control=monitor_for_control,
                                           function=function,
                                           params=params,
                                           name=name,
                                           prefs=prefs,
                                           context=self)
        TEST = True

    def _validate_params(self, request_set, target_set=None, context=None):

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

    def _instantiate_input_states(self, context=None):
        """Instantiate inputState and MappingProjections for list of Mechanisms and/or States to be monitored

        """
        super()._instantiate_input_states(context=context)

        self._instantiate_prediction_mechanisms(context=context)
        self._instantiate_monitoring_mechanism(context=context)

        # # MODIFIED 2/9/17 NEW:
        # # Re-instantiate system with predictionMechanism Process(es) and monitoringMechanism added
        # self.system._instantiate_processes(input=self.system.variable, context=context)
        # self.system._instantiate_graph(context=context)
        # # MODIFIED 2/9/17 END

    def _instantiate_prediction_mechanisms(self, context=None):
        """Add prediction mechanism and associated process for each ORIGIN (input) mechanism in the system

        Instantiate PredictionMechanisms for ORIGIN mechanisms in self.system; these will now be TERMINAL mechanisms
            - if their associated input mechanisms were TERMINAL MECHANISMS, they will no longer be so
            - therefore if an associated input mechanism must be monitored by the EVCMechanism, it must be specified
                explicitly in an outputState, mechanism, controller or systsem MONITOR_FOR_CONTROL param (see below)

        For each ORIGIN mechanism in self.system:
            - instantiate a corresponding predictionMechanism
            - instantiate a Process, with a pathway that projects from the ORIGIN to the prediction mechanism
            - add the process to self.system.processes

        Instantiate self.predictedInput dict:
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
        self.prediction_mech_tuples = []

        # Get any params specified for predictionMechanism(s) by EVCMechanism
        try:
            prediction_mechanism_params = self.paramsCurrent[PREDICTION_MECHANISM_PARAMS]
        except KeyError:
            prediction_mechanism_params = {}


        for origin_mech in self.system.originMechanisms.mechanisms:

            # # IMPLEMENT THE FOLLOWING ONCE INPUT_STATES CAN BE SPECIFIED IN CONSTRUCTION OF ALL MECHANISMS
            # #           (AS THEY CAN CURRENTLY FOR ObjectiveMechanisms)
            # state_names = []
            # variables = []
            # for state_name in origin_mech.inputStates.keys():
            #     state_names.append(state_name)
            #     variables.append(origin_mech_intputStates[state_name].variable)

            # Instantiate predictionMechanism
            prediction_mechanism = self.paramsCurrent[PREDICTION_MECHANISM_TYPE](
                                                            name=origin_mech.name + " " + PREDICTION_MECHANISM,
                                                            default_input_value = origin_mech.inputState.variable,
                                                            # default_input_value=variables,
                                                            # INPUT_STATES=state_names,
                                                            params = prediction_mechanism_params,
                                                            context=context)
            prediction_mechanism._role = CONTROL
            prediction_mechanism.origin_mech = origin_mech

            # Assign projections to prediction_mechanism that duplicate those received by origin_mech
            #    (this includes those from ProcessInputState, SystemInputState and/or recurrent ones
            for orig_state_name, prediction_state_name in zip(origin_mech.inputStates.keys(),
                                                                prediction_mechanism.inputStates.keys()):
                for projection in origin_mech.inputStates[orig_state_name].receivesFromProjections:
                    MappingProjection(sender=projection.sender,
                                      receiver=prediction_mechanism.inputStates[prediction_state_name],
                                      matrix=projection.matrix)

            # Assign list of processes for which prediction_mechanism will provide input during the simulation
            # - used in _get_simulation_system_inputs()
            # - assign copy,
            #       since don't want to include the prediction process itself assigned to origin_mech.processes below
            prediction_mechanism.use_for_processes = list(origin_mech.processes.copy())

            # # FIX: REPLACE REFERENCE TO THIS ELSEWHERE WITH REFERENCE TO MECH_TUPLES BELOW
            self.origin_prediction_mechanisms[origin_mech] = prediction_mechanism

            # Add to list of EVCMechanism's prediction_mech_tuples
            prediction_mech_tuple = MechanismTuple(prediction_mechanism, None, origin_mech.phaseSpec)
            self.prediction_mech_tuples.append(prediction_mech_tuple)

            # Add to system executionGraph and executionList
            self.system.executionGraph[prediction_mech_tuple] = set()
            self.system.executionList.append(prediction_mech_tuple)

        self.predictionMechanisms = MechanismList(self, self.prediction_mech_tuples)

        # Assign list of destinations for predicted_inputs:
        #    the variable of the ORIGIN mechanism for each process in the system
        self.predictedInput = {}
        for i, origin_mech in zip(range(len(self.system.originMechanisms)), self.system.originMechanisms):
            # self.predictedInput[origin_mech] = self.system.processes[i].originMechanisms[0].inputValue
            self.predictedInput[origin_mech] = self.system.processes[i].originMechanisms[0].variable

    def _instantiate_monitoring_mechanism(self, context=None):
        """
        Assign inputState to controller for each state to be monitored;
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
        * self.inputStates is the usual ordered dict of states,
            each of which receives a projection from a corresponding outputState in self.monitored_output_states
        """

        self._get_monitored_states(context=context)

        for state in self.monitored_output_states:
            self._validate_monitored_state_in_system(state)

        # Note: weights and exponents are assigned as parameters of outcome_function in _get_monitored_states
        self.monitoring_mechanism = ObjectiveMechanism(monitored_values=self.monitored_output_states,
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

        self.system.executionList.append(MechanismTuple(self.monitoring_mechanism, None, self.system.numPhases-1))

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
        * self.inputStates is the usual ordered dict of states,
            each of which receives a projection from a corresponding outputState in self.monitored_output_states

        """

        from PsyNeuLink.Components.States.OutputState import OutputState
        from PsyNeuLink.Components.Mechanisms.Mechanism import MonitoredOutputStatesOption
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import validate_monitored_value

        # PARSE SPECS

        # Get controller's MONITOR_FOR_CONTROL specifications (optional, so need to try)
        try:
            controller_specs = self.paramsCurrent[MONITOR_FOR_CONTROL] or []
        except KeyError:
            controller_specs = []

        # Get system's MONITOR_FOR_CONTROL specifications (specified in paramClassDefaults, so must be there)
        system_specs = self.system.paramsCurrent[MONITOR_FOR_CONTROL]

        # If the controller has a MonitoredOutputStatesOption specification, remove any such spec from system specs
        if controller_specs:
            if (any(isinstance(item, MonitoredOutputStatesOption) for item in controller_specs)):
                option_item = next((item for item in system_specs if isinstance(item, MonitoredOutputStatesOption)),None)
                if option_item is not None:
                    del system_specs[option_item]

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
            validate_monitored_value(self, item, context=context)
            # Extract references from specification tuples
            if isinstance(item, tuple):
                all_specs_extracted_from_tuples.append(item[OBJECT])
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
                        local_specs.append(item[OBJECT])
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
            for output_state_name, output_state in mech.outputStates.items():

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
                            local_specs.append(item[OBJECT])
                            continue
                        local_specs.append(item)

            # Ignore MonitoredOutputStatesOption if any outputStates are explicitly specified for the mechanism
            for output_state_name, output_state in list(mech.outputStates.items()):
                if (output_state in local_specs or output_state.name in local_specs):
                    option_spec = None


            # ASSIGN SPECIFIED OUTPUT STATES FOR MECHANISM TO self.monitored_output_states

            for output_state_name, output_state in list(mech.outputStates.items()):

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
                        if output_state is mech.outputState:
                            self.monitored_output_states.append(output_state)
                            continue
                    # If MonitoredOutputStatesOption is ALL_OUTPUT_STATES, include it
                    elif option_spec is MonitoredOutputStatesOption.ALL_OUTPUT_STATES:
                        self.monitored_output_states.append(output_state)
                    elif mech.name in local_specs or mech in local_specs:
                        if output_state is mech.outputState:
                            self.monitored_output_states.append(output_state)
                            continue
                    elif option_spec is None:
                        continue
                    else:
                        raise EVCError("PROGRAM ERROR: unrecognized specification of MONITOR_FOR_CONTROL for "
                                       "{0} of {1}".
                                       format(output_state_name, mech.name))


        # ASSIGN WEIGHTS AND EXPONENTS TO OUTCOME_FUNCTION

        # Note: these values will be superceded by any assigned as arguments to the outcome_function
        #       if it is specified in the constructor for the mechanism

        num_monitored_output_states = len(self.monitored_output_states)
        weights = np.ones((num_monitored_output_states,1))
        exponents = np.ones_like(weights)

        # Get and assign specification of weights and exponents for mechanisms or outputStates specified in tuples
        for spec in all_specs:
            if isinstance(spec, tuple):
                object_spec = spec[OBJECT]
                # For each outputState in monitored_output_states
                for item in self.monitored_output_states:
                    # If either that outputState or its owner is the object specified in the tuple
                    if item is object_spec or item.name is object_spec or item.owner is object_spec:
                        # Assign the weight and exponent specified in the tuple to that outputState
                        i = self.monitored_output_states.index(item)
                        weights[i] = spec[WEIGHT]
                        exponents[i] = spec[EXPONENT]

        # Assign weights and exponents to corresponding attributes of default OUTCOME_FUNCTION
        # Note: done here (rather than in call to outcome_function in value_function) for efficiency
        self.paramsCurrent[OUTCOME_FUNCTION]._assign_params(request_set={WEIGHTS:weights,
                                                                         EXPONENTS:exponents},
                                                            context=context)

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

    def _instantiate_control_projection(self, projection, params=None, context=None):
        """
        """

        if self.allocation_policy is None:
            self.allocation_policy = np.array(defaultControlAllocation)
        else:
            self.allocation_policy = np.append(self.allocation_policy, defaultControlAllocation)

        # Call super to instantiate ControlSignal outputStates
        super()._instantiate_control_projection(projection=projection,
                                                params=params,
                                                context=context)

        # Assign controlSignals in the order they are stored of OutputStates
        self.controlSignals = [self.outputStates[state_name] for state_name in self.outputStates.keys()]

        # # TEST PRINT
        # print("\n{}.controlSignals: ".format(self.name))
        # for control_signal in self.controlSignals:
        #     print("{}".format(control_signal.name))

    def _instantiate_function(self, context=None):
        super()._instantiate_function(context=context)

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
            num_control_projections = len(self.controlProjections)
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
        Construct controlSignalSearchSpace (from allocation_samples of each item in controlSignals):
            * get `allocation_samples` for each ControlSignal in `controlSignals`
            * construct `controlSignalSearchSpace`: a 2D np.array of control allocation policies, each policy of which
              is a different combination of values, one from the `allocation_samples` of each ControlSignal.
        Call self.function -- default is ControlSignalGridSearch
        Return an allocation_policy
        """

        self._update_predicted_input()
        # self.system._cache_state()

        # CONSTRUCT SEARCH SPACE

        control_signal_sample_lists = []
        control_signals = self.controlSignals

        # Get allocation_samples for all ControlSignals
        num_control_signals = len(control_signals)

        for control_signal in self.controlSignals:
            control_signal_sample_lists.append(control_signal.allocation_samples)

        # Construct controlSignalSearchSpace:  set of all permutations of ControlProjection allocations
        #                                     (one sample from the allocationSample of each ControlProjection)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.controlSignalSearchSpace = \
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
        """Assign values of prediction mechanisms to predictedInput

        Assign value of each predictionMechanism.value to corresponding item of self.predictedIinput
        Note: must be assigned in order of self.system.processes

        """

        # Assign predictedInput for each process in system.processes

        # The number of ORIGIN mechanisms requiring input should = the number of prediction mechanisms
        num_origin_mechs = len(self.system.originMechanisms)
        num_prediction_mechs = len(self.origin_prediction_mechanisms)
        if num_origin_mechs != num_prediction_mechs:
            raise EVCError("PROGRAM ERROR:  The number of ORIGIN mechanisms ({}) does not equal"
                           "the number of prediction_predictions mechanisms ({}) for {}".
                           format(num_origin_mechs, num_prediction_mechs, self.system.name))
        for origin_mech in self.system.originMechanisms:
            # Get origin mechanism for each process
            # Assign value of predictionMechanism to the entry of predictedInput for the corresponding ORIGIN mechanism
            self.predictedInput[origin_mech] = self.origin_prediction_mechanisms[origin_mech].value
            # self.predictedInput[origin_mech] = self.origin_prediction_mechanisms[origin_mech].outputState.value

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

        inputs : List[input] or ndarray(input) : default default_input_value
            the inputs used for each in a sequence of executions of the mechanism in the `system <System>`.  This
            should be the `value <Mechanism.Mechanism_Base.value> for each
            `prediction mechanism <EVCMechanism_Prediction_Mechanisms>` listed in the `predictionMechanisms`
            attribute.  The inputs are available from the `predictedInput` attribute.

        allocation_vector : (1D np.array)
            the allocation policy to use in running the simulation, with one allocation value for each of the
            EVCMechanism's ControlSignals (listed in `controlSignals`).

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
        for i in range(len(self.controlSignals)):
            # self.controlSignals[list(self.controlSignals.values())[i]].value = np.atleast_1d(allocation_vector[i])
            self.value[i] = np.atleast_1d(allocation_vector[i])
        self._update_output_states(runtime_params=runtime_params, time_scale=time_scale,context=context)

        # Execute simulation run of system for the current allocation_policy
        sim_clock = Clock('EVC SIMULATION CLOCK')

        self.system.run(inputs=inputs, clock=sim_clock, time_scale=time_scale, context=context)

        # Get outcomes for current allocation_policy
        #    = the values of the monitored output states (self.inputStates)
        #    stored in self.inputValue = list(self.variable)
        # self.monitoring_mechanism.execute(context=EVC_SIMULATION)
        self._update_input_states(runtime_params=runtime_params, time_scale=time_scale,context=context)

        for i in range(len(self.controlSignals)):
            self.controlSignalCosts[i] = self.controlSignals[i].cost


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
        return self._outcome_function

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
