# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  LVOCControlMechanism ******************************************************

"""

Overview
--------

An LVOCControlMechanism is a `ControlMechanism <ControlMechanism>` that regulates it `ControlSignals <ControlSignal>` in
order to optimize the performance of the System to which it belongs.  LVOCControlMechanism is one of the most
powerful, but also one of the most complex components in PsyNeuLink.  It is designed to implement a form of the
Expected Value of Control (EVC) Theory described in `Shenhav et al. (2013)
<https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_, which provides useful background concerning the purpose and
structure of the LVOCControlMechanism.

An LVOCControlMechanism is similar to a standard `ControlMechanism`, with the following exceptions:

  * it can only be assigned to a System as its `controller <System.controller>`, and not in any other capacity
    (see `ControlMechanism_System_Controller`);
  ..
  * it has several specialized functions that are used to search over the `allocations <ControlSignal.allocations>`\\s
    of its its `ControlSignals <ControlSignal>`, and evaluate the performance of its `system
    <LVOCControlMechanism.system>`; by default, it simulates its `system <LVOCControlMechanism.system>` and evaluates
    its performance under all combinations of ControlSignal values to find the one that optimizes the `Expected
    Value of Control <LVOCControlMechanism_EVC>`, however its functions can be customized or replaced to implement
    other optimization procedures.
  ..
  * it creates a specialized set of `prediction Mechanisms` LVOCControlMechanism_Prediction_Mechanisms` that are used to
    simulate the performnace of its `system <LVOCControlMechanism.system>`.

.. _LVOCControlMechanism_EVC:

*Expected Value of Control (EVC)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LVOCControlMechanism uses it `function <LVOCControlMechanism.function>` to select an `allocation_policy` for its
`system <LVOCControlMechanism.system>`.  In the `default configuration <LVOCControlMechanism_Default_Configuration>`,
an LVOCControlMechanism carries out an exhaustive evaluation of allocation policies, simulating its `system
<LVOCControlMechanism.system>` under each, and using an `ObjectiveMechanism` and several `auxiliary functions
<LVOCControlMechanism_Functions>` to calculate the **expected value of control (EVC)** for each `allocation_policy`:
a cost-benefit analysis that weighs the `cost <ControlSignal.cost> of the ControlSignals against the outcome of the
`system <LVOCControlMechanism.system>` \\s performance for a given `allocation_policy`. The LVOCControlMechanism
selects the `allocation_policy` that generates the maximum EVC, and implements that for the next `TRIAL`. Each step
of this procedure can be modified, or replaced entirely, by assigning custom functions to corresponding parameters of
the LVOCControlMechanism, as described `below <LVOCControlMechanism_Functions>`.

.. _LVOCControlMechanism_Creation:

Creating an LVOCControlMechanism
------------------------

An LVOCControlMechanism can be created in any of the ways used to `create a ControlMechanism
<ControlMechanism_Creation>`; it is also created automatically when a `System` is created and the LVOCControlMechanism
class is specified in the **controller** argument of the System's constructor (see `System_Creation`).  The
ObjectiveMechanism, the OutputStates it monitors and evaluates, and the parameters controlled by an
LVOCControlMechanism can be specified in the standard way for a ControlMechanism (see
`ControlMechanism_ObjectiveMechanism` and `ControlMechanism_Control_Signals`, respectively).

.. note::
   Although an LVOCControlMechanism can be created on its own, it can only be assigned to, and executed within a `System`
   as the System's `controller <System.controller>`.

When an LVOCControlMechanism is assigned to, or created by a System, it is assigned the OutputStates to be monitored and
parameters to be controlled specified for that System (see `System_Control`), and a `prediction Mechanism
<LVOCControlMechanism_Prediction_Mechanisms>` is created for each `ORIGIN` Mechanism in the `system
<LVOCControlMechanism.system>`. The prediction Mechanisms are assigned to the LVOCControlMechanism's
`prediction_mechanisms` attribute. The OutputStates used to determine an LVOCControlMechanism’s allocation_policy and
the parameters it controls can be listed using its show method. The LVOCControlMechanism and the Components
associated with it in its `system <LVOCControlMechanism.system>` can be displayed using the System's
`System.show_graph` method with its **show_control** argument assigned as `True`

An LVOCControlMechanism that has been constructed automatically can be customized by assigning values to its
attributes (e.g., those described above, or its `function <LVOCControlMechanism.function>` as described under
`LVOCControlMechanism_Default_Configuration `below).


.. _LVOCControlMechanism_Structure:

Structure
---------

An LVOCControlMechanism must belong to a `System` (identified in its `system <LVOCControlMechanism.system>` attribute).
In addition to the standard Components of a `ControlMechanism`, has a specialized set of `prediction mechanisms
<LVOCControlMechanism_Prediction_Mechanisms>` and `functions <LVOCControlMechanism_Functions>` that it uses to simulate
and evaluate the performance of its `system <LVOCControlMechanism.system>` under the influence of different values of
its `ControlSignals <LVOCControlMechanism_ControlSignals>`.  Each of these specialized Components is described below.

.. _LVOCControlMechanism_Input:

*Input*
~~~~~~~

.. _LVOCControlMechanism_ObjectiveMechanism:

ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

Like any ControlMechanism, an LVOCControlMechanism receives its input from the *OUTCOME* `OutputState
<ObjectiveMechanism_Output>` of an `ObjectiveMechanism`, via a MappingProjection to its `primary InputState
<InputState_Primary>`.  The ObjectiveFunction is listed in the LVOCControlMechanism's `objective_mechanism
<LVOCControlMechanism.objective_mechanism>` attribute.  By default, the ObjectiveMechanism's function is a
`LinearCombination` function with its `operation <LinearCombination.operation>` attribute assigned as *PRODUCT*;
this takes the product of the `value <OutputState.value>`\\s of the OutputStates that it monitors (listed in its
`monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute.  However, this can be customized
in a variety of ways:

    * by specifying a different `function <ObjectiveMechanism.function>` for the ObjectiveMechanism
      (see `Objective Mechanism Examples <ObjectiveMechanism_Weights_and_Exponents_Example>` for an example);
    ..
    * using a list to specify the OutputStates to be monitored  (and the `tuples format
      <InputState_Tuple_Specification>` to specify weights and/or exponents for them) in the
      **objective_mechanism** argument of the LVOCControlMechanism's constructor;
    ..
    * using the  **monitored_output_states** argument for an ObjectiveMechanism specified in the `objective_mechanism
      <LVOCControlMechanism.objective_mechanism>` argument of the LVOCControlMechanism's constructor;
    ..
    * specifying a different `ObjectiveMechanism` in the **objective_mechanism** argument of the LVOCControlMechanism's
      constructor. The result of the `objective_mechanism <LVOCControlMechanism.objective_mechanism>`'s `function
      <ObjectiveMechanism.function>` is used as the outcome in the calculations described below.

    .. _LVOCControlMechanism_Objective_Mechanism_Function_Note:

    .. note::
       If a constructor for an `ObjectiveMechanism` is used for the **objective_mechanism** argument of the
       LVOCControlMechanism's constructor, then the default values of its attributes override any used by the
       LVOCControlMechanism for its `objective_mechanism <LVOCControlMechanism.objective_mechanism>`.  In particular,
       whereas an LVOCControlMechanism uses the same default `function <ObjectiveMechanism.function>` as an
       `ObjectiveMechanism` (`LinearCombination`), it uses *PRODUCT* rather than *SUM* as the default value of the
       `operation <LinearCombination.operation>` attribute of the function.  As a consequence, if the constructor for
       an ObjectiveMechanism is used to specify the LVOCControlMechanism's **objective_mechanism** argument,
       and the **operation** argument is not specified, *SUM* rather than *PRODUCT* will be used for the
       ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  To ensure that *PRODUCT* is used, it must be
       specified explicitly in the **operation** argument of the constructor for the ObjectiveMechanism (see 1st
       example under `System_Control_Examples`).

The result of the LVOCControlMechanism's `objective_mechanism <LVOCControlMechanism.objective_mechanism>` is used by
its `function <ObjectiveMechanism.function>` to evaluate the performance of its `system <LVOCControlMechanism.system>`
when computing the `EVC <LVOCControlMechanism_EVC>`.


.. _LVOCControlMechanism_Prediction_Mechanisms:

Prediction Mechanisms
^^^^^^^^^^^^^^^^^^^^^

These are used to provide input to the `system <LVOCControlMechanism.system>` if the LVOCControlMechanism's
`function <LVOCControlMechanism.function>` (`ControlSignalGridSearch`) `simulates its execution
<LVOCControlMechanism_Default_Configuration>` to evaluate the EVC for a given `allocation_policy`.  When an
LVOCControlMechanism is created, a prediction Mechanism is created for each `ORIGIN` Mechanism in its `system
<LVOCControlMechanism.system>`, and are listed in the LVOCControlMechanism's `prediction_mechanisms
<LVOCControlMechanism.prediction_mechanisms>` attribute in the same order as the `ORIGIN` Mechanisms are
listed in the `system <LVOCControlMechanism.system>`\\'s `origin_mechanisms <System.origin_mechanisms>` attribute.
For each `Projection <Projection>` received by an `ORIGIN` Mechanism, a `MappingProjection` from the same source is
created that projects to the corresponding prediction Mechanism.  By default, the `PredictionMechanism` subclass  is
used for all prediction mechanisms, which calculates an exponentially weighted time-average of its input over (
non-simuated) trials, that is provided as input to the corresponding `ORIGIN` Mechanism on each simulated trial.
However, any other type of Mechanism can be used as a prediction mechanism, so long as it has the same number of
`InputStates <InputState>` as the `ORIGIN` Mechanism to which it corresponds, and an `OutputState` corresponding to
each.  The default type is a `PredictionMechanism`, that calculates an exponentially weighted time-average of its
input. The prediction mechanisms can be customized using the *prediction_mechanisms* argument of the
LVOCControlMechanism's constructor, which can be specified using any of the following formats:

  * **Mechanism** -- convenience format for cases in which the LVOCControlMechanism's `system
    <LVOCControlMechanism.system>` has a single `ORIGIN` Mechanism;  the Mechanism must have the same number of
    `InputStates <InputState>` as the `system <LVOCControlMechanism.system>`\\'s `ORIGIN` Mechanism, and
    an `OutputState` for each.
  ..
  * **Mechanism subclass** -- used as the class for all prediction mechanisms; a default instance of that class
    is created for each prediction mechanism, with a number of InputStates and OutputStates equal to the number of
    InputStates of the `ORIGIN` Mechanism to which it corresponds.
  ..
  * **dict** -- a `parameter specification dictionary <ParameterState_Specification>` specifying the parameters to be
    assigned to all prediction mechanisms, all of which are instances of a `PredictionMechanism` (thus, the parameters
    specified must be appropriate for a PredictionMechanism).
  ..
  * **2-item tuple:** *(Mechanism subclass, dict)* -- the Mechanism subclass, and parameters specified in
    the `parameter specification dictionary <ParameterState_Specification>`, are used for all prediction mechanisms.
  ..
  * **list** -- its length must equal the number of `ORIGIN` Mechanisms in the LVOCControlMechanism's `system
    <LVOCControlMechanism.system>` each item must be a Mechanism, a subclass of one, or a 2-item tuple (see above),
    that is used as the specification for the prediction Mechanism for the corresponding `ORIGIN` Mechanism listed in
    the System's `origin_mechanisms <System.origin_mechanisms>` attribute.

The prediction mechanisms for an LVOCControlMechanism are listed in its `prediction_mechanisms` attribute.

.. _LVOCControlMechanism_Functions:

*Function*
~~~~~~~~~~

By default, the primary `function <LVOCControlMechanism.function>` is `ControlSignalGridSearch` (see
`LVOCControlMechanism_Default_Configuration`), that systematically evaluates the effects of its ControlSignals on the
performance of its `system <LVOCControlMechanism.system>` to identify an `allocation_policy
<LVOCControlMechanism.allocation_policy>` that yields the highest `EVC <LVOCControlMechanism_EVC>`.  However,
any function can be used that returns an appropriate value (i.e., that specifies an `allocation_policy` for the
number of `ControlSignals <LVOCControlMechanism_ControlSignals>` in the LVOCControlMechanism's `control_signals`
attribute, using the correct format for the `allocation <ControlSignal.allocation>` value of each ControlSignal).
In addition to its primary `function <LVOCControlMechanism.function>`, an LVOCControlMechanism has several auxiliary
functions, that can be used by its `function <LVOCControlMechanism.function>` to calculate the EVC to select an
`allocation_policy` with the maximum EVC among a range of policies specified by its ControlSignals.  The default
set of functions and their operation are described in the section that follows;  however, the LVOCControlMechanism's
`function <LVOCControlMechanism.function>` can call any other function to customize how the EVC is calcualted.

.. _LVOCControlMechanism_Default_Configuration:

Default Configuration of EVC Function and its Auxiliary Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In its default configuration, an LVOCControlMechanism simulates and evaluates the performance of its `system
<LVOCControlMechanism.system>` under a set of allocation_policies determined by the `allocation_samples
<ControlSignal.allocation_samples>` attributes of its `ControlSignals <LVOCControlMechanism_ControlSignals>`, and
implements (for the next `TRIAL` of execution) the one that generates the maximum `EVC <LVOCControlMechanism_EVC>`.
This is carried out by the LVOCControlMechanism's default `function <LVOCControlMechanism.function>` and three
auxiliary functions, as described below.

The default `function <LVOCControlMechanism.function>` of an LVOCControlMechanism is `ControlSignalGridSearch`. It
identifies the `allocation_policy` with the maximum `EVC <LVOCControlMechanism_EVC>` by a conducting an exhaustive
search over every possible `allocation_policy`— that is, all combinations of `allocation <ControlSignal.allocation>`
values for its `ControlSignals <LVOCControlMechanism_ControlSignals>`, where the `allocation
<ControlSignal.allocation>` values sampled for each ControlSignal are determined by its `allocation_samples`
attribute.  For each `allocation_policy`, the LVOCControlMechanism executes the `system <LVOCControlMechanism.system>`,
evaluates the `EVC <LVOCControlMechanism_EVC>` for that policy, and returns the `allocation_policy` that yields the
greatest EVC value. The following steps are used to calculate the EVC for each `allocation_policy`:

  * **Implement the policy and simulate the System** - assign the `allocation <ControlSignal.allocation>` that the
    selected `allocation_policy` specifies for each ControlSignal, and then simulate the `system
    <LVOCControlMechanism.system>` using the corresponding parameter values by calling the System's `run_simulation
    <System.run_simulation>` method; this uses the `value <PredictionMechanism.value>` of eacah of the
    LVOCControlMechanism's `prediction_mechanisms <LVOCControlMechanism.prediction_mechanisms>` as input to the
    corresponding `ORIGIN` Mechanisms of its `system <LVOCControlMechanism.system>` (see `PredictionMechanism`).  The
    values of all :ref:`stateful attributes` of 'Components` in the System are :ref:`re-initialized` to the same value
    prior to each simulation, so that the results for each `allocation_policy <LVOCControlMechanism.allocation_policy>`
    are based on the same initial conditions.  Each simulation includes execution of the LVOCControlMechanism's
    `objective_mechanism`, which provides the result to the LVOCControlMechanism.  If `system
    <LVOCControlMechanism.system>`\\.recordSimulationPref is `True`, the results of each simulation are appended to the
    `simulation_results <System.simulation_results>` attribute of `system <LVOCControlMechanism.system>`.
  |
  * **Evaluate the System's performance** - this is carried out by the LVOCControlMechanism's `objective_mechanism
    <LVOCControlMechanism.objective_mechanism>`, which is executed as part of the simulation of the System.  The
    `function <ObjectiveMechanism.function>` for a default ObjectiveMechanism is a `LinearCombination` Function that
    combines the `value <OutputState.value>`\\s of the OutputStates listed in the LVOCControlMechanism's
    `monitored_output_states <LVOCControlMechanism.monitored_output_states>` attribute (and the `objective_mechanism
    <LVOCControlMechanism.objective_mechanism>`'s `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
    attribute) by taking their elementwise (Hadamard) product.  However, this behavior can be customized in a variety
    of ways, as described `above <LVOCControlMechanism_ObjectiveMechanism>`.
  |
  * **Calculate EVC** - call the LVOCControlMechanism's `value_function <LVOCControlMechanism.value_function>` passing it
    the outcome (received from the `objective_mechanism`) and a list of the `costs <ControlSignal.cost>` \\s of its
    `ControlSignals <LVOCControlMechanism_ControlSignals>`.  The default `value_function
    <LVOCControlMechanism.value_function>` calls two additional auxiliary functions, in the following order:
    |
    - `cost_function <LVOCControlMechanism.cost_function>`, which sums the costs;  this can be configured to weight
      and/or exponentiate individual costs (see `cost_function <LVOCControlMechanism.cost_function>` attribute);
    |
    - `combine_outcome_and_cost_function <LVOCControlMechanism.combine_outcome_and_cost_function>`, which subtracts the
      sum of the costs from the outcome to generate the EVC;  this too can be configured (see
      `combine_outcome_and_cost_function <LVOCControlMechanism.combine_outcome_and_cost_function>`).

In addition to modifying the default functions (as noted above), any or all of them can be replaced with a custom
function to modify how the `allocation_policy <LVOCControlMechanism.allocation_policy>` is determined, so long as the
custom function accepts arguments and returns values that are compatible with any other functions that call that
function (see note below).

.. _LVOCControlMechanism_Calling_and_Assigning_Functions:

    .. note::
       The `LVOCControlMechanism auxiliary functions <LVOCControlMechanism_Functions>` described above are all
       implemented as PsyNeuLink `Functions <Function>`.  Therefore, to call a function itself, it must be referenced
       as ``<LVOCControlMechanism>.<function_attribute>.function``.  A custom function assigned to one of the auxiliary
       functions can be either a PsyNeuLink `Function <Function>`, or a generic python function or method (including
       a lambda function).  If it is one of the latter, it is automatically "wrapped" as a PsyNeuLink `Function
       <Function>` (specifically, it is assigned as the `function <UserDefinedFunction.function>` attribute of a
       `UserDefinedFunction` object), so that it can be referenced and called in the same manner as the default
       function assignment. Therefore, once assigned, it too must be referenced as
       ``<LVOCControlMechanism>.<function_attribute>.function``.

.. _LVOCControlMechanism_ControlSignals:

*ControlSignals*
~~~~~~~~~~~~~~~~

The OutputStates of an LVOCControlMechanism (like any `ControlMechanism`) are a set of `ControlSignals
<ControlSignal>`, that are listed in its `control_signals <LVOCControlMechanism.control_signals>` attribute (as well as
its `output_states <ControlMechanism.output_states>` attribute).  Each ControlSignal is assigned a  `ControlProjection`
that projects to the `ParameterState` for a parameter controlled by the LVOCControlMechanism.  Each ControlSignal is
assigned an item of the LVOCControlMechanism's `allocation_policy`, that determines its `allocation
<ControlSignal.allocation>` for a given `TRIAL` of execution.  The `allocation <ControlSignal.allocation>` is used by
a ControlSignal to determine its `intensity <ControlSignal.intensity>`, which is then assigned as the `value
<ControlProjection.value>` of the ControlSignal's ControlProjection.   The `value <ControlProjection>` of the
ControlProjection is used by the `ParameterState` to which it projects to modify the value of the parameter (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter it controls).
A ControlSignal also calculates a `cost <ControlSignal.cost>`, based on its `intensity <ControlSignal.intensity>`
and/or its time course. The `cost <ControlSignal.cost>` is included in the evaluation that the LVOCControlMechanism
carries out for a given `allocation_policy`, and that it uses to adapt the ControlSignal's `allocation
<ControlSignal.allocation>` in the future.  When the LVOCControlMechanism chooses an `allocation_policy` to evaluate,
it selects an allocation value from the ControlSignal's `allocation_samples <ControlSignal.allocation_samples>`
attribute.


.. _LVOCControlMechanism_Execution:

Execution
---------

An LVOCControlMechanism must be the `controller <System.controller>` of a System, and as a consequence it is always the
last `Mechanism <Mechanism>` to be executed in a `TRIAL` for its `system <LVOCControlMechanism.system>` (see `System
Control <System_Execution_Control>` and `Execution <System_Execution>`). When an LVOCControlMechanism is executed,
it updates the value of its `prediction_mechanisms` and `objective_mechanism`, and then calls its `function
<LVOCControlMechanism.function>`, which determines and implements the `allocation_policy` for the next `TRIAL` of its
`system <LVOCControlMechanism.system>`\\s execution.  The default `function <LVOCControlMechanism.function>` executes
the following steps (described in greater detailed `above <LVOCControlMechanism_Default_Configuration>`):

* Samples every allocation_policy (i.e., every combination of the `allocation` \\s specified for the
  LVOCControlMechanism's ControlSignals specified by their `allocation_samples` attributes);  for each
  `allocation_policy`, it:

  * Runs a simulation of the LVOCControlMechanism's `system <LVOCControlMechanism.system>` with the parameter values
    specified by that `allocation_policy`, by calling the system's  `run_simulation <System.run_simulation>` method;
    each simulation uses inputs provided by the LVOCControlMechanism's `prediction_mechanisms
    <LVOCControlMechanism.prediction_mechanisms>` and includes execution of its `objective_mechanism`, which provides
    its result to the LVOCControlMechanism.

  * Calls the LVOCControlMechanism's `value_function <LVOCControlMechanism.value_function>`, which in turn calls
    LVOCControlMechanism's `cost_function <LVOCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
    <LVOCControlMechanism.combine_outcome_and_cost_function>` to evaluate the EVC for that `allocation_policy`.

* Selects and returns the `allocation_policy` that generates the maximum EVC value.

This procedure can be modified by specifying a custom function for any or all of the `functions
<LVOCControlMechanism_Functions>` referred to above.


.. _LVOCControlMechanism_Examples:

Example
-------

The following example implements a System with an LVOCControlMechanism (and two processes not shown)::


    >>> import psyneulink as pnl                                                        #doctest: +SKIP
    >>> myRewardProcess = pnl.Process(...)                                              #doctest: +SKIP
    >>> myDecisionProcess = pnl.Process(...)                                            #doctest: +SKIP
    >>> mySystem = pnl.System(processes=[myRewardProcess, myDecisionProcess],           #doctest: +SKIP
    ...                       controller=pnl.LVOCControlMechanism,                       #doctest: +SKIP
    ...                       monitor_for_control=[Reward,                              #doctest: +SKIP
    ...                                            pnl.DDM_OUTPUT.DECISION_VARIABLE,    #doctest: +SKIP
    ...                                            (pnl.RESPONSE_TIME, 1, -1)],         #doctest: +SKIP

It uses the System's **monitor_for_control** argument to assign three OutputStates to be monitored.  The first one
references the Reward Mechanism (not shown);  its `primary OutputState <OutputState_Primary>` will be used by default.
The second and third use keywords that are the names of outputStates of a  `DDM` Mechanism (also not shown).
The last one (RESPONSE_TIME) is assigned a weight of 1 and an exponent of -1. As a result, each calculation of the EVC
computation will multiply the value of the primary OutputState of the Reward Mechanism by the value of the
*DDM_DECISION_VARIABLE* OutputState of the DDM Mechanism, and then divide that by the value of the *RESPONSE_TIME*
OutputState of the DDM Mechanism.

See `ObjectiveMechanism <ObjectiveMechanism_Monitored_Output_States_Examples>` for additional examples of how to specify it's
**monitored_output_states** argument, `ControlMechanism <ControlMechanism_Examples>` for additional examples of how to
specify ControlMechanisms, and `System <System_Examples>` for how to specify the `controller <System.controller>`
of a System.

.. _LVOCControlMechanism_Class_Reference:

Class Reference
---------------

"""
from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.functions.function import ModulationParam, _is_modulation_param, Buffer, Linear, BayesGLM
from psyneulink.components.mechanisms.mechanism import Mechanism
from psyneulink.components.mechanisms.adaptive.control.controlmechanism import ControlMechanism
from psyneulink.components.mechanisms.processing.objectivemechanism import OUTCOME
from psyneulink.components.states.inputstate import InputState
from psyneulink.components.states.outputstate import OutputState
from psyneulink.components.states.parameterstate import ParameterState
from psyneulink.components.states.modulatorysignals.controlsignal import ControlSignalCosts
from psyneulink.components.shellclasses import Composition_Base
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import \
    ALL, FUNCTION, INIT_FUNCTION_METHOD_ONLY, LVOC_MECHANISM, NAME, PARAMETER_STATES, PROJECTIONS, VARIABLE, \
    FUNCTION_PARAMS
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.globals.utilities import ContentAddressableList, is_iterable, is_numeric
from psyneulink.library.subsystems.lvoc.lvocauxiliary import ControlSignalGradientAscent

__all__ = [
    'LVOCControlMechanism', 'LVOCError', 'SHADOW_INPUTS',
]

SHADOW_INPUTS = 'SHADOW_INPUTS'
PREDICTION_WEIGHTS = 'PREDICTION_WEIGHTS'

class LVOCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class LVOCControlMechanism(ControlMechanism):
    """LVOCControlMechanism(                                           \
    system=True,                                                       \
    objective_mechanism=None,                                          \
    function=ControlSignalGradientAscent                               \
    update_function=UpdateWeights,                                     \
    cost_function=LinearCombination(operation=SUM),                    \
    combine_outcome_and_cost_function=LinearCombination(operation=SUM) \
    save_all_values_and_policies=:keyword:`False`,                     \
    control_signals=None,                                              \
    params=None,                                                       \
    name=None,                                                         \
    prefs=None)

    Subclass of `ControlMechanism <ControlMechanism>` that optimizes the `ControlSignals <ControlSignal>` for a
    `System`.

    COMMENT:
        Class attributes:
            + componentType (str): System Default Mechanism
            + paramClassDefaults (dict):
                + SYSTEM (System)
                + MONITORED_OUTPUT_STATES (list of Mechanisms and/or OutputStates)

        Class methods:
            None

       **********************************************************************************************

       PUT SOME OF THIS STUFF IN ATTRIBUTES, BUT USE DEFAULTS HERE

        # - specification of System:  required param: SYSTEM
        # - kwDefaultController:  True =>
        #         takes over all unassigned ControlProjections (i.e., without a sender) in its System;
        #         does not take monitored states (those are created de-novo)
        # TBI: - CONTROL_PROJECTIONS:
        #         list of projections to add (and for which outputStates should be added)

        # - input_states: one for each performance/environment variable monitored

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
            System for which the LVOCControlMechanism is the controller;  this is a required parameter.

        default_variable : Optional[number, list or np.ndarray] : `defaultControlAllocation <LINK]>`

    COMMENT


    Arguments
    ---------

    system : System : default None
        specifies the `System` for which the LVOCControlMechanism should serve as a `controller <System.controller>`;
        the LVOCControlMechanism will inherit any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <LVOCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    objective_mechanism : ObjectiveMechanism, List[OutputState or Tuple[OutputState, list or 1d np.array, list or 1d
    np.array]] : \
    default MonitoredOutputStatesOptions.PRIMARY_OUTPUT_STATES
        specifies either an `ObjectiveMechanism` to use for the LVOCControlMechanism or a list of the OutputStates it should
        monitor; if a list of `OutputState specifications <ObjectiveMechanism_Monitored_Output_States>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitored_output_states** argument.

    function : function or method : ControlSignalGradientAscent
        specifies the function used to determine the `allocation_policy` for the current execution of the
        LVOCControlMechanism's `composition <LVOCControlMechanism.composition>` (see `function
        <LVOCControlMechanism.function>` for details).

    value_function : function or method : value_function
        specifies the function used to calculate the `EVC <LVOCControlMechanism_EVC>` for the current `allocation_policy`
        (see `value_function <LVOCControlMechanism.value_function>` for details).

    cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to calculate the cost associated with the current `allocation_policy`
        (see `cost_function <LVOCControlMechanism.cost_function>` for details).

    combine_outcome_and_cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to combine the outcome and cost associated with the current `allocation_policy`,
        to determine its value (see `combine_outcome_and_cost_function` for details).

    save_all_values_and_policies : bool : default False
        specifes whether to save every `allocation_policy` tested in `EVC_policies` and their values in `EVC_values`.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the LVOCControlMechanism
        (see `ControlSignal_Specification` for details of specification).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function <LVOCControlMechanism.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <LVOCControlMechanism.name>`
        specifies the name of the LVOCControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the LVOCControlMechanism; see `prefs <LVOCControlMechanism.prefs>` for details.

    Attributes
    ----------

    system : System
        the `System` for which LVOCControlMechanism is the `controller <System.controller>`;
        the LVOCControlMechanism inherits any `OutputStates <OutputState>` specified in the **monitor_for_control**
        argument of the `system <LVOCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    objective_mechanism : ObjectiveMechanism
        the 'ObjectiveMechanism' used by the LVOCControlMechanism to evaluate the performance of its `system
        <LVOCControlMechanism.system>`.  If a list of OutputStates is specified in the **objective_mechanism** argument of the
        LVOCControlMechanism's constructor, they are assigned as the `monitored_output_states <ObjectiveMechanism.monitored_output_states>`
        attribute for the `objective_mechanism <LVOCControlMechanism.objective_mechanism>`.

    monitored_output_states : List[OutputState]
        list of the OutputStates monitored by `objective_mechanism <LVOCControlMechanism.objective_mechanism>` (and listed in
        its `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute), and used to evaluate the
        performance of the LVOCControlMechanism's `system <LVOCControlMechanism.system>`.

    COMMENT:
    [TBI]
        monitored_output_states : 3D np.array
            an array of values of the outputStates in `monitored_output_states` (equivalent to the values of
            the LVOCControlMechanism's `input_states <LVOCControlMechanism.input_states>`).
    COMMENT

    monitored_output_states_weights_and_exponents: List[Tuple[scalar, scalar]]
        a list of tuples, each of which contains the weight and exponent (in that order) for an OutputState in
        `monitored_outputStates`, listed in the same order as the outputStates are listed in `monitored_outputStates`.

    sampled_prediction_weights :

    weighted_predictor_values :

    function : function : default ControlSignalGridSearch
        determines the `allocation_policy` to use for the next round of the System's
        execution. The default function, `ControlSignalGridSearch`, conducts an exhaustive (*grid*) search of all
        combinations of the `allocation_samples` of its ControlSignals (and contained in its
        `control_signal_search_space` attribute), by executing the System (using `run_simulation`) for each
        combination, evaluating the result using `value_function`, and returning the `allocation_policy` that yielded
        the greatest `EVC <LVOCControlMechanism_EVC>` value (see `LVOCControlMechanism_Default_Configuration` for additional details).
        If a custom function is specified, it must accommodate a **controller** argument that specifies an LVOCControlMechanism
        (and provides access to its attributes, including `control_signal_search_space`), and must return an array with
        the same format (number and type of elements) as the LVOCControlMechanism's `allocation_policy` attribute.

    COMMENT:
        NOTES ON API FOR CUSTOM VERSIONS:
            Gets controller as argument (along with any standard params specified in call)
            Must include **kwargs to receive standard args (variable, params, and context)
            Must return an allocation policy compatible with controller.allocation_policy:
                2d np.array with one array for each allocation value

            Following attributes are available:
            controller._get_simulation_system_inputs gets inputs for a simulated run (using predictionMechanisms)
            controller.run will execute a specified number of trials with the simulation inputs
            controller.monitored_states is a list of the Mechanism OutputStates being monitored for outcome
            controller.input_value is a list of current outcome values (values for monitored_states)
            controller.monitored_output_states_weights_and_exponents is a list of parameterizations for OutputStates
            controller.control_signals is a list of control_signal objects
            controller.control_signal_search_space is a list of all allocationPolicies specifed by allocation_samples
            control_signal.allocation_samples is the set of samples specified for that control_signal
            [TBI:] control_signal.allocation_range is the range that the control_signal value can take
            controller.allocation_policy - holds current allocation_policy
            controller.output_values is a list of current control_signal values
            controller.value_function - calls the three following functions (done explicitly, so each can be specified)
            controller.cost_function - aggregate costs of control signals
            controller.combine_outcome_and_cost_function - combines outcomes and costs
    COMMENT

    update_function : function : default UpdateWeights
        TBW

    COMMENT:
    cost_function : function : default LinearCombination(operation=SUM)
        calculates the cost of the `ControlSignals <ControlSignal>` for the current `allocation_policy`.  The default
        function sums the `cost <ControlSignal.cost>` of each of the LVOCControlMechanism's `ControlSignals
        <LVOCControlMechanism_ControlSignals>`.  The `weights <LinearCombination.weights>` and/or `exponents
        <LinearCombination.exponents>` parameters of the function can be used, respectively, to scale and/or
        exponentiate the contribution of each ControlSignal cost to the combined cost.  These must be specified as
        1d arrays in a *WEIGHTS* and/or *EXPONENTS* entry of a `parameter dictionary <ParameterState_Specification>`
        assigned to the **params** argument of the constructor of a `LinearCombination` function; the length of
        each array must equal the number of (and the values listed in the same order as) the ControlSignals in the
        LVOCControlMechanism's `control_signals <LVOCControlMechanism.control_signals>` attribute. The default function can also be
        replaced with any `custom function <LVOCControlMechanism_Calling_and_Assigning_Functions>` that takes an array as
        input and returns a scalar value.  If used with the LVOCControlMechanism's default `value_function
        <LVOCControlMechanism.value_function>`, a custom `cost_function <LVOCControlMechanism.cost_function>` must accommodate two
        arguments (passed by name): a **controller** argument that is the LVOCControlMechanism itself;  and a **costs**
        argument that is a 1d array of scalar values specifying the `cost <ControlSignal.cost>` for each ControlSignal
        listed in the `control_signals` attribute of the ControlMechanism specified in the **controller** argument.

    combine_outcome_and_cost_function : function : default LinearCombination(operation=SUM)
        combines the outcome and cost for given `allocation_policy` to determine its `EVC <LVOCControlMechanisms_EVC>`. The
        default function subtracts the cost from the outcome, and returns the difference.  This can be modified using
        the `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>` parameters of the
        function, as described for the `cost_function <LVOCControlMechanisms.cost_function>`.  The default function can also be
        replaced with any `custom function <LVOCControlMechanism_Calling_and_Assigning_Functions>` that returns a scalar value.  If used with the LVOCControlMechanism's default `value_function`, a custom
        If used with the LVOCControlMechanism's default `value_function`, a custom combine_outcome_and_cost_function must
        accomoudate three arguments (passed by name): a **controller** argument that is the LVOCControlMechanism itself; an
        **outcome** argument that is a 1d array with the outcome of the current `allocation_policy`; and a **cost**
        argument that is 1d array with the cost of the current `allocation_policy`.

    control_signal_search_space : 2d np.array
        an array each item of which is an `allocation_policy`.  By default, it is assigned the set of all possible
        allocation policies, using np.meshgrid to construct all permutations of `ControlSignal` values from the set
        specified for each by its `allocation_samples <LVOCControlMechanism.allocation_samples>` attribute.
    COMMENT

    EVC_max : 1d np.array with single value
        the maximum `EVC <LVOCControlMechanism_EVC>` value over all allocation policies in `control_signal_search_space`.

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
        array with every `allocation_policy` tested in `control_signal_search_space`.  The `EVC <LVOCControlMechanism_EVC>`
        value of each is stored in `EVC_values`.

    EVC_values :  1d np.array
        array of `EVC <LVOCControlMechanism_EVC>` values, each of which corresponds to an `allocation_policy` in `EVC_policies`;

    allocation_policy : 2d np.array : defaultControlAllocation
        determines the value assigned as the `variable <ControlSignal.variable>` for each `ControlSignal` and its
        associated `ControlProjection`.  Each item of the array must be a 1d array (usually containing a scalar)
        that specifies an `allocation` for the corresponding ControlSignal, and the number of items must equal the
        number of ControlSignals in the LVOCControlMechanism's `control_signals` attribute.

    control_signals : ContentAddressableList[ControlSignal]
        list of the LVOCControlMechanism's `ControlSignals <LVOCControlMechanism_ControlSignals>`, including any that it inherited
        from its `system <LVOCControlMechanism.system>` (same as the LVOCControlMechanism's `output_states
        <Mechanism_Base.output_states>` attribute); each sends a `ControlProjection` to the `ParameterState` for the
        parameter it controls

    name : str
        the name of the LVOCControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the LVOCControlMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = LVOC_MECHANISM
    initMethod = INIT_FUNCTION_METHOD_ONLY


    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'DefaultControlMechanismCustomClassPreferences',
    #     kp<pref>: <setting>...}

    class ClassDefaults(ControlMechanism.ClassDefaults):
        function = ControlSignalGradientAscent

    from psyneulink.components.functions.function import LinearCombination
    paramClassDefaults = ControlMechanism.paramClassDefaults.copy()
    paramClassDefaults.update({PARAMETER_STATES: NotImplemented}) # This suppresses parameterStates

    @tc.typecheck
    def __init__(self,
                 composition:tc.optional(Composition_Base)=None,
                 input_states:tc.optional(tc.any(Iterable, Mechanism, OutputState, InputState))=SHADOW_INPUTS,
                 # objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputState))=None,
                 function=ControlSignalGradientAscent(),
                 # update_function=None,
                 # cost_function=LinearCombination(operation=SUM),
                 # combine_outcome_and_cost_function=LinearCombination(operation=SUM),
                 # save_all_values_and_policies:bool=False,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterState))=None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(composition=composition,
                                                  input_states=input_states,
                                                  # update_function=update_function,
                                                  # cost_function=cost_function,
                                                  # combine_outcome_and_cost_function=combine_outcome_and_cost_function,
                                                  # save_all_values_and_policies=save_all_values_and_policies,
                                                  params=params)

        super().__init__(system=None,
                         # objective_mechanism=ObjectiveMechanism(monitored_output_states=monitor_for_control,
                         #                                        function=BayesGLM(predictor_weight_priors,
                         #                                                          predictor_variance_priors)),
                         monitor_for_control=monitor_for_control,
                         function=function,
                         control_signals=control_signals,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs)

        # self.update_function = update_function or UpdateWeights(
        #     prediction_weights_priors=prediction_weights_priors,
        #     predictor_variance_priors=prediction_variances_priors,
        #     function=BayesGLM(num_predictors=self.num_predictors,
        #                       mu_prior=prediction_weights_priors,
        #                       sigma_prior=prediction_variances_priors)
        # )

    def _instantiate_input_states(self, context=None):
        """Instantiate PredictionMechanisms
        """

        # FIX:
        #  IF input_states ARGUMENT IS SPECIFIED, USE THOSE BY CALLING RELEVANT super()._instantiate_input_states
        #    this should allow the output of any other Mechanism in the Composition to be used as the source of
        #    signals LVOC uses in learning to predict control
        #  IF THE KEYWORD "SHADOW" APPEARS IN THE SPECIFICATION DICT
        #      IF System ARGUMENT IS SPECIFIED:
        #           CREATE:
        #             CIM OutputState -[MappingProjection]-> PredictionMechanism -[MappingProjection]-> LVOC InputState
        #           FOR ALL MECHANISMS LISTED IN SHADOW ENTRY
        #      IF System ARGUMENT IS NOT SPECIFIED, RAISE EXCEPTION
        #  IF input_states ARGUMENT IS NOT SPECIFIED:
        #      IF System ARGUMENT IS SPECIFIED, SHADOW ALL ORIGIN MECHANISMS [DEFAULT CASE] AS ABOVE
        #      IF System ARGUMENT IS NOT SPECIFIED, RAISE EXCEPTION

        # input_states CAN BE ANY SPECIFICATION FOR input_states (REF TO DOCS)
        #     THAT WILL ASSIGN AN input_state WITH A PROJECTION FROM THE SPECIFIED SOURCE
        # AND/OR DICT WITH KEYWORD "SHADOW_INPUTS" AS THE KEY FOR AN ENTRY AND EITHER OF THE FOLLOWING AS ITS VALUE:
        #     - KEYWORD "ORIGIN_MECHANISMS":  INPUT_STATES FOR AND POJECTIONS FROM ALL OUTPUT_STATES OF COMPOSITION
        #     - LIST OF ORIGIN MECHANISMS AND/OR THEIR INPUT_STATES
        # IF IT CONTAINS A DICT WITH A "SHADOW_INPUTS" ENTRY, IT CAN ALSO INCLUDE A FUNCTION:<Function> ENTRY
        #     THAT WILL BE USED AS THE FUNCTION OF THE input_states CREATED FOR THE LVOCControlMechanism

        # If input_states has SHADOW_INPUTS in any of its specifications, parse into input_states specifications
        if any(SHADOW_INPUTS in spec for spec in self.input_states):
            self.input_states = self._parse_predictor_specs(composition=self.composition,
                                                            predictors=self.input_states,
                                                            context=context)

        # Insert primary InputState for outcome from ObjectiveMechanism; assumes this will be a single scalar value
        self.input_states.insert(0, OUTCOME),

        # Configure default_variable to comport with full set of input_states
        self.instance_defaults.variable, ignore = self._handle_arg_input_states(self.input_states)

        super()._instantiate_input_states(context=context)

    tc.typecheck
    def add_predictors(self, predictors, composition:tc.optional(Composition_Base)=None):
        '''Add InputStates and Projections to LVOCControlMechanism for predictors used to predict allocation_policy

        **predictors** argument can use any of the forms of specification allowed
            for the **input_states** argument of the LVOCMechanism.
        '''

        if self.composition is None:
            self.composition = composition
        else:
            if not composition is self.composition:
                raise LVOCError("Specified composition ({}) conflicts with one to which {} is already assigned ({})".
                                format(composition.name, self.name, self.composition.name))
        features = self._parse_input_specs(composition=composition,
                                                 inputs=predictors,
                                                 context=ContextFlags.COMMAND_LINE)
        self.add_states(InputState, features)

    def _parse_predictor_specs(self, composition:Composition_Base, predictors=SHADOW_INPUTS, context=None):
        """Parse entries of _input_states list that specify shadowing of Mechanisms' or Composition's inputs

        Generate an InputState specification dictionary for each predictor specified in predictors argument
        If it is InputState specificaditon, use as is
        If it is a SHADOW_INPUT entry, generate a Projection from the OutputState that projects to the specified item

        Returns list of InputState specifications
        """

        composition = composition or self.composition
        if not composition:
            raise LVOCError("PROGRAM ERROR: A Composition must be specified in call to _instantiate_inputs")

        from psyneulink.compositions.composition import CNodeRole
        parsed_predictors = []

        for spec in predictors:
            if SHADOW_INPUTS in spec:
                # If spec is SHADOW_INPUTS keyword on its own, assume inputs to all ORIGIN Mechanisms
                if isinstance(spec, str):
                    spec = {SHADOW_INPUTS:ALL}
                spec = self._parse_shadow_input_spec(spec)
            else:
                spec = [spec] # (so that extend can be used below)
            parsed_predictors.extend(spec)

        return parsed_predictors

    @tc.typecheck
    def _parse_shadow_input_spec(self, spec:dict):
        ''' Return a list of InputState specifications for the inputs specified in value of dict

        If ALL is specified, specify an InputState for each ORIGIN Mechanism in the Composition
            with Projection from the OutputState of the Compoisitions Input CIM for that ORIGIN Mechanism
        For any other specification, specify an InputState with a Projection from the sender of any Projections
            that project to the specified item
        If FUNCTION entry, assign as Function for all InputStates
        '''

        input_state_specs = []

        shadow_spec = spec[SHADOW_INPUTS]

        if shadow_spec is ALL:
            # Generate list of InputState specification dictionaries,
            #    one for each input to the Composition
            # for composition_input in self.composition.input_CIM.output_states:
            #     input_state_specs.append(composition_input)
            input_state_specs.extend([{NAME:'INPUT OF ' + c.efferents[0].receiver.name +
                                            ' of ' + c.efferents[0].receiver.owner.name,
                                       PROJECTIONS:c}
                                      for c in self.composition.input_CIM.output_states])
        elif isinstance(shadow_spec, list):
            for item in shadow_spec:
                if isinstance(item, Mechanism):
                    # Shadow all of the InputStates for the Mechanism
                    input_states = item.input_states
                if isinstance(item, InputState):
                    # Place in a list for consistency of handling below
                    input_states = [item]
                # Shadow all of the Projections to each specified InputState
                input_state_specs.extend([{NAME:i.name + 'of' + i.owner.name,
                                           VARIABLE: i.variable,
                                           PROJECTIONS: i.path_afferents}
                                          for i in input_states])

        if FUNCTION in spec:
            for i in input_state_specs:
                i.update({FUNCTION:spec[FUNCTION]})

        return input_state_specs

    def _instantiate_attributes_after_function(self, context=None):
        '''Validate cost function, instantiate Projections to ObjectiveMechanism, and construct
        control_signal_search_space.

        Instantiate Projections to ObjectiveMechansm for worth and current weights

        Construct control_signal_search_space (from allocation_samples of each item in control_signals):
            * get `allocation_samples` for each ControlSignal in `control_signals`
            * construct `control_signal_search_space`: a 2D np.array of control allocation policies, each policy of
              which is a different combination of values, one from the `allocation_samples` of each ControlSignal.
        '''

        super()._instantiate_attributes_after_function(context=context)

        if self.composition is None:
            return

        # # Validate cost function
        # cost_Function = self.cost_function
        # if isinstance(cost_Function, Function):
        #     # Insure that length of the weights and/or exponents arguments for the cost_function
        #     #    matches the number of control signals
        #     num_control_projections = len(self.control_projections)
        #     if cost_Function.weights is not None:
        #         num_cost_weights = len(cost_Function.weights)
        #         if  num_cost_weights != num_control_projections:
        #             raise LVOCError("The length of the weights argument {} for the {} of {} "
        #                            "must equal the number of its control signals {}".
        #                            format(num_cost_weights,
        #                                   COST_FUNCTION,
        #                                   self.name,
        #                                   num_control_projections))
        #     if cost_Function.exponents is not None:
        #         num_cost_exponents = len(cost_Function.exponents)
        #         if  num_cost_exponents != num_control_projections:
        #             raise LVOCError("The length of the exponents argument {} for the {} of {} "
        #                            "must equal the number of its control signals {}".
        #                            format(num_cost_exponents,
        #                                   COST_FUNCTION,
        #                                   self.name,
        #                                   num_control_projections))
        #
        # # Construct control_signal_search_space
        # control_signal_sample_lists = []
        # control_signals = self.control_signals
        # # Get allocation_samples for all ControlSignals
        # num_control_signals = len(control_signals)
        #
        # for control_signal in self.control_signals:
        #     control_signal_sample_lists.append(control_signal.allocation_samples)
        #
        # # Construct control_signal_search_space:  set of all permutations of ControlProjection allocations
        # #                                     (one sample from the allocationSample of each ControlProjection)
        # # Reference for implementation below:
        # # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        # self.control_signal_search_space = \
        #     np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,num_control_signals)

    def _instantiate_control_signal(self, control_signal, context=None):
        '''Implement ControlSignalCosts.DEFAULTS as default for cost_option of ControlSignals
        EVCControlMechanism requires use of at least one of the cost options
        '''
        control_signal = super()._instantiate_control_signal(control_signal, context)

        if control_signal.cost_options is None:
            control_signal.cost_options = ControlSignalCosts.DEFAULTS
            control_signal._instantiate_cost_attributes()
        return control_signal

    def _execute(self, variable=None, runtime_params=None, context=None):
        """Determine `allocation_policy <LVOCControlMechanism.allocation_policy>` for current run of Composition

        Get sampled_prediction_weights by drawing a value for each using prediction_weights and predictor_variances
            received from BayesGLMObjectiveMechanism.
        Call self.function -- default: ControlSignalGradientDescent:
            does gradient descent on allocation_policy to fit within budget
            based on current_prediction_weights control_signal costs.
        Return an allocation_policy
        """

        # EXECUTE SEARCH

        # IMPLEMENTATION NOTE:
        # self.composition._store_system_state()

        # IMPLEMENTATION NOTE:
        # - skip ControlMechanism._execute since it is a stub method that returns input_values
        allocation_policy = super(ControlMechanism, self)._execute(controller=self,
                                                                   variable=variable,
                                                                   runtime_params=runtime_params,
                                                                   context=context
                                                                   )

        # IMPLEMENTATION NOTE:
        # self.composition._restore_system_state()

        return allocation_policy.reshape((len(allocation_policy),1))

    def _parse_function_variable(self, variable, context=None):
        '''Return array of current predictor values and last prediction weights received from LVOCObjectiveMechanism'''

        # This the value received from the ObjectiveMechanism:
        outcome = variable[0]

        # This is a vector of the concatenated values received from all of the other InputStates (i.e., variable[1:])
        self.predictor_values = np.array(variable[1:]).reshape(-1)

        return [self.predictor_values, outcome]

    # @property
    # def update_function(self):
    #     return self._update_function
    #
    # @update_function.setter
    # def update_function(self, assignment):
    #     if isinstance(assignment, function_type):
    #         self._update_function = UpdateWeights(assignment)
    #     elif assignment is UpdateWeights:
    #         self._update_function = UpdateWeights()
    #     else:
    #         self._update_function = assignment
    #
    # @property
    # def cost_function(self):
    #     return self._cost_function
    #
    # @cost_function.setter
    # def cost_function(self, value):
    #     from psyneulink.components.functions.function import UserDefinedFunction
    #     if isinstance(value, function_type):
    #         udf = UserDefinedFunction(function=value)
    #         self._cost_function = udf
    #     else:
    #         self._cost_function = value
    #
    # @property
    # def combine_outcome_and_cost_function(self):
    #     return self._combine_outcome_and_cost_function
    #
    # @combine_outcome_and_cost_function.setter
    # def combine_outcome_and_cost_function(self, value):
    #     from psyneulink.components.functions.function import UserDefinedFunction
    #     if isinstance(value, function_type):
    #         udf = UserDefinedFunction(function=value)
    #         self._combine_outcome_and_cost_function = udf
    #     else:
    #         self._combine_outcome_and_cost_function = value
