# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************************  EVCControlMechanism ***********************************************

"""

Overview
--------

An EVCControlMechanism is a `ControlMechanism <ControlMechanism>` that regulates it `ControlSignals <ControlSignal>` in
order to optimize the performance of the System to which it belongs.  EVCControlMechanism is one of the most
powerful, but also one of the most complex components in PsyNeuLink.  It is designed to implement a form of the
Expected Value of Control (EVC) Theory described in `Shenhav et al. (2013)
<https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_, which provides useful background concerning the purpose and
structure of the EVCControlMechanism.

An EVCControlMechanism is similar to a standard `ControlMechanism`, with the following exceptions:

  * it can only be assigned to a System as its `controller <System.controller>`, and not in any other capacity
    (see `ControlMechanism_Composition_Controller`);
  ..
  * it has several specialized functions that are used to search over the `allocations <ControlSignal.allocations>`\\s
    of its its `ControlSignals <ControlSignal>`, and evaluate the performance of its `system
    <EVCControlMechanism.system>`; by default, it simulates its `system <EVCControlMechanism.system>` and evaluates
    its performance under all combinations of ControlSignal values to find the one that optimizes the `Expected
    Value of Control <EVCControlMechanism_EVC>`, however its functions can be customized or replaced to implement
    other optimization procedures.
  ..
  * it creates a specialized set of `prediction Mechanisms` EVCControlMechanism_Prediction_Mechanisms` that are used to
    simulate the performnace of its `system <EVCControlMechanism.system>`.

.. _EVCControlMechanism_EVC:

*Expected Value of Control (EVC)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EVCControlMechanism uses it `function <EVCControlMechanism.function>` to select an `control_allocation` for its
`system <EVCControlMechanism.system>`.  In the `default configuration <EVCControlMechanism_Default_Configuration>`,
an EVCControlMechanism carries out an exhaustive evaluation of allocation policies, simulating its `system
<EVCControlMechanism.system>` under each, and using an `ObjectiveMechanism` and several `auxiliary functions
<EVCControlMechanism_Functions>` to calculate the **expected value of control (EVC)** for each `control_allocation`:
a cost-benefit analysis that weighs the `cost <ControlSignal.cost> of the ControlSignals against the outcome of the
`system <EVCControlMechanism.system>` \\s performance for a given `control_allocation`. The EVCControlMechanism
selects the `control_allocation` that generates the maximum EVC, and implements that for the next `TRIAL`. Each step
of this procedure can be modified, or replaced entirely, by assigning custom functions to corresponding parameters of
the EVCControlMechanism, as described `below <EVCControlMechanism_Functions>`.

.. _EVCControlMechanism_Creation:

Creating an EVCControlMechanism
------------------------

An EVCControlMechanism can be created in any of the ways used to `create a ControlMechanism
<ControlMechanism_Creation>`; it is also created automatically when a `System` is created and the EVCControlMechanism
class is specified in the **controller** argument of the System's constructor (see `System_Creation`).  The
ObjectiveMechanism, the OutputPorts it monitors and evaluates, and the parameters controlled by an
EVCControlMechanism can be specified in the standard way for a ControlMechanism (see
`ControlMechanism_ObjectiveMechanism` and `ControlMechanism_ControlSignals`, respectively).

.. note::
   Although an EVCControlMechanism can be created on its own, it can only be assigned to, and executed within a `System`
   as the System's `controller <System.controller>`.

When an EVCControlMechanism is assigned to, or created by a System, it is assigned the OutputPorts to be monitored and
parameters to be controlled specified for that System (see `System_Control`), and a `prediction Mechanism
<EVCControlMechanism_Prediction_Mechanisms>` is created for each `ORIGIN` Mechanism in the `system
<EVCControlMechanism.system>`. The prediction Mechanisms are assigned to the EVCControlMechanism's
`prediction_mechanisms` attribute. The OutputPorts used to determine an EVCControlMechanism’s control_allocation and
the parameters it controls can be listed using its show method. The EVCControlMechanism and the Components
associated with it in its `system <EVCControlMechanism.system>` can be displayed using the System's
`System.show_graph` method with its **show_control** argument assigned as `True`

An EVCControlMechanism that has been constructed automatically can be customized by assigning values to its
attributes (e.g., those described above, or its `function <EVCControlMechanism.function>` as described under
`EVCControlMechanism_Default_Configuration `below).


.. _EVCControlMechanism_Structure:

Structure
---------

An EVCControlMechanism must belong to a `System` (identified in its `system <EVCControlMechanism.system>` attribute).
In addition to the standard Components of a `ControlMechanism`, has a specialized set of `prediction mechanisms
<EVCControlMechanism_Prediction_Mechanisms>` and `functions <EVCControlMechanism_Functions>` that it uses to simulate
and evaluate the performance of its `system <EVCControlMechanism.system>` under the influence of different values of
its `ControlSignals <EVCControlMechanism_ControlSignals>`.  Each of these specialized Components is described below.

.. _EVCControlMechanism_Input:

*Input*
~~~~~~~

.. _EVCControlMechanism_ObjectiveMechanism:

ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

Like any ControlMechanism, an EVCControlMechanism receives its input from the *OUTCOME* `OutputPort
<ObjectiveMechanism_Output>` of an `ObjectiveMechanism`, via a MappingProjection to its `primary InputPort
<InputPort_Primary>`.  The ObjectiveMechanism is listed in the EVCControlMechanism's `objective_mechanism
<EVCControlMechanism.objective_mechanism>` attribute.  By default, the ObjectiveMechanism's function is a
`LinearCombination` function with its `operation <LinearCombination.operation>` attribute assigned as *PRODUCT*;
this takes the product of the `value <OutputPort.value>`\\s of the OutputPorts that it monitors (listed in its
`monitored_output_ports <ObjectiveMechanism.monitored_output_ports>` attribute.  However, this can be customized
in a variety of ways:

    * by specifying a different `function <ObjectiveMechanism.function>` for the ObjectiveMechanism
      (see `Objective Mechanism Examples <ObjectiveMechanism_Weights_and_Exponents_Example>` for an example);
    ..
    * using a list to specify the OutputPorts to be monitored  (and the `tuples format
      <InputPort_Tuple_Specification>` to specify weights and/or exponents for them) in either the
      **monitor_for_control** or **objective_mechanism** arguments of the EVCControlMechanism's constructor;
    ..
    * using the  **monitored_output_ports** argument for an ObjectiveMechanism specified in the `objective_mechanism
      <EVCControlMechanism.objective_mechanism>` argument of the EVCControlMechanism's constructor;
    ..
    * specifying a different `ObjectiveMechanism` in the **objective_mechanism** argument of the EVCControlMechanism's
      constructor. The result of the `objective_mechanism <EVCControlMechanism.objective_mechanism>`'s `function
      <ObjectiveMechanism.function>` is used as the outcome in the calculations described below.

    .. _EVCControlMechanism_Objective_Mechanism_Function_Note:

    .. note::
       If a constructor for an `ObjectiveMechanism` is used for the **objective_mechanism** argument of the
       EVCControlMechanism's constructor, then the default values of its attributes override any used by the
       EVCControlMechanism for its `objective_mechanism <EVCControlMechanism.objective_mechanism>`.  In particular,
       whereas an EVCControlMechanism uses the same default `function <ObjectiveMechanism.function>` as an
       `ObjectiveMechanism` (`LinearCombination`), it uses *PRODUCT* rather than *SUM* as the default value of the
       `operation <LinearCombination.operation>` attribute of the function.  As a consequence, if the constructor for
       an ObjectiveMechanism is used to specify the EVCControlMechanism's **objective_mechanism** argument,
       and the **operation** argument is not specified, *SUM* rather than *PRODUCT* will be used for the
       ObjectiveMechanism's `function <ObjectiveMechanism.function>`.  To ensure that *PRODUCT* is used, it must be
       specified explicitly in the **operation** argument of the constructor for the ObjectiveMechanism (see 1st
       example under `System_Control_Examples`).

The result of the EVCControlMechanism's `objective_mechanism <EVCControlMechanism.objective_mechanism>` is used by
its `function <EVCControlMechanism.function>` to evaluate the performance of its `System <EVCControlMechanism.system>`
when computing the `EVC <EVCControlMechanism_EVC>`.


.. _EVCControlMechanism_Prediction_Mechanisms:

Prediction Mechanisms
^^^^^^^^^^^^^^^^^^^^^

These are used to provide input to the `system <EVCControlMechanism.system>` if the EVCControlMechanism's
`function <EVCControlMechanism.function>` (`ControlSignalGridSearch`) `simulates its execution
<EVCControlMechanism_Default_Configuration>` to evaluate the EVC for a given `control_allocation`.  When an
EVCControlMechanism is created, a prediction Mechanism is created for each `ORIGIN` Mechanism in its `system
<EVCControlMechanism.system>`, and are listed in the EVCControlMechanism's `prediction_mechanisms
<EVCControlMechanism.prediction_mechanisms>` attribute in the same order as the `ORIGIN` Mechanisms are
listed in the `system <EVCControlMechanism.system>`\\'s `origin_mechanisms <System.origin_mechanisms>` attribute.
For each `Projection <Projection>` received by an `ORIGIN` Mechanism, a `MappingProjection` from the same source is
created that projects to the corresponding prediction Mechanism.  By default, the `PredictionMechanism` subclass  is
used for all prediction mechanisms, which calculates an exponentially weighted time-average of its input over (
non-simuated) trials, that is provided as input to the corresponding `ORIGIN` Mechanism on each simulated trial.
However, any other type of Mechanism can be used as a prediction mechanism, so long as it has the same number of
`InputPorts <InputPort>` as the `ORIGIN` Mechanism to which it corresponds, and an `OutputPort` corresponding to
each.  The default type is a `PredictionMechanism`, that calculates an exponentially weighted time-average of its
input. The prediction mechanisms can be customized using the *prediction_mechanisms* argument of the
EVCControlMechanism's constructor, which can be specified using any of the following formats:

  * **Mechanism** -- convenience format for cases in which the EVCControlMechanism's `system
    <EVCControlMechanism.system>` has a single `ORIGIN` Mechanism;  the Mechanism must have the same number of
    `InputPorts <InputPort>` as the `system <EVCControlMechanism.system>`\\'s `ORIGIN` Mechanism, and
    an `OutputPort` for each.
  ..
  * **Mechanism subclass** -- used as the class for all prediction mechanisms; a default instance of that class
    is created for each prediction mechanism, with a number of InputPorts and OutputPorts equal to the number of
    InputPorts of the `ORIGIN` Mechanism to which it corresponds.
  ..
  * **dict** -- a `parameter specification dictionary <ParameterPort_Specification>` specifying the parameters to be
    assigned to all prediction mechanisms, all of which are instances of a `PredictionMechanism` (thus, the parameters
    specified must be appropriate for a PredictionMechanism).
  ..
  * **2-item tuple:** *(Mechanism subclass, dict)* -- the Mechanism subclass, and parameters specified in
    the `parameter specification dictionary <ParameterPort_Specification>`, are used for all prediction mechanisms.
  ..
  * **list** -- its length must equal the number of `ORIGIN` Mechanisms in the EVCControlMechanism's `system
    <EVCControlMechanism.system>` each item must be a Mechanism, a subclass of one, or a 2-item tuple (see above),
    that is used as the specification for the prediction Mechanism for the corresponding `ORIGIN` Mechanism listed in
    the System's `origin_mechanisms <System.origin_mechanisms>` attribute.

The prediction mechanisms for an EVCControlMechanism are listed in its `prediction_mechanisms` attribute.

.. _EVCControlMechanism_Functions:

*Function*
~~~~~~~~~~

By default, the primary `function <EVCControlMechanism.function>` is `ControlSignalGridSearch` (see
`EVCControlMechanism_Default_Configuration`), that systematically evaluates the effects of its ControlSignals on the
performance of its `system <EVCControlMechanism.system>` to identify an `control_allocation
<EVCControlMechanism.control_allocation>` that yields the highest `EVC <EVCControlMechanism_EVC>`.  However,
any function can be used that returns an appropriate value (i.e., that specifies an `control_allocation` for the
number of `ControlSignals <EVCControlMechanism_ControlSignals>` in the EVCControlMechanism's `control_signals`
attribute, using the correct format for the `allocation <ControlSignal.allocation>` value of each ControlSignal).
In addition to its primary `function <EVCControlMechanism.function>`, an EVCControlMechanism has several auxiliary
functions, that can be used by its `function <EVCControlMechanism.function>` to calculate the EVC to select an
`control_allocation` with the maximum EVC among a range of policies specified by its ControlSignals.  The default
set of functions and their operation are described in the section that follows;  however, the EVCControlMechanism's
`function <EVCControlMechanism.function>` can call any other function to customize how the EVC is calcualted.

.. _EVCControlMechanism_Default_Configuration:

Default Configuration of EVC Function and its Auxiliary Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In its default configuration, an EVCControlMechanism simulates and evaluates the performance of its `system
<EVCControlMechanism.system>` under a set of allocation_policies determined by the `allocation_samples
<ControlSignal.allocation_samples>` attributes of its `ControlSignals <EVCControlMechanism_ControlSignals>`, and
implements (for the next `TRIAL` of execution) the one that generates the maximum `EVC <EVCControlMechanism_EVC>`.
This is carried out by the EVCControlMechanism's default `function <EVCControlMechanism.function>` and three
auxiliary functions, as described below.

The default `function <EVCControlMechanism.function>` of an EVCControlMechanism is `ControlSignalGridSearch`. It
identifies the `control_allocation` with the maximum `EVC <EVCControlMechanism_EVC>` by a conducting an exhaustive
search over every possible `control_allocation`— that is, all combinations of `allocation <ControlSignal.allocation>`
values for its `ControlSignals <EVCControlMechanism_ControlSignals>`, where the `allocation
<ControlSignal.allocation>` values sampled for each ControlSignal are determined by its `allocation_samples`
attribute.  For each `control_allocation`, the EVCControlMechanism executes the `system <EVCControlMechanism.system>`,
evaluates the `EVC <EVCControlMechanism_EVC>` for that policy, and returns the `control_allocation` that yields the
greatest EVC value. The following steps are used to calculate the EVC for each `control_allocation`:

  * **Implement the policy and simulate the System** - assign the `allocation <ControlSignal.allocation>` that the
    selected `control_allocation` specifies for each ControlSignal, and then simulate the `system
    <EVCControlMechanism.system>` using the corresponding parameter values by calling the System's `evaluate
    <System.evaluate>` method; this uses the `value <PredictionMechanism.value>` of eacah of the
    EVCControlMechanism's `prediction_mechanisms <EVCControlMechanism.prediction_mechanisms>` as input to the
    corresponding `ORIGIN` Mechanisms of its `system <EVCControlMechanism.system>` (see `PredictionMechanism`).  The
    values of all :ref:`stateful attributes` of 'Components` in the System are :ref:`re-initialized` to the same value
    prior to each simulation, so that the results for each `control_allocation <EVCControlMechanism.control_allocation>`
    are based on the same initial conditions.  Each simulation includes execution of the EVCControlMechanism's
    `objective_mechanism`, which provides the result to the EVCControlMechanism.  If `system
    <EVCControlMechanism.system>`\\.recordSimulationPref is `True`, the results of each simulation are appended to the
    `simulation_results <System.simulation_results>` attribute of `system <EVCControlMechanism.system>`.

  * **Evaluate the System's performance** - this is carried out by the EVCControlMechanism's `objective_mechanism
    <EVCControlMechanism.objective_mechanism>`, which is executed as part of the simulation of the System.  The
    `function <ObjectiveMechanism.function>` for a default ObjectiveMechanism is a `LinearCombination` Function that
    combines the `value <OutputPort.value>`\\s of the OutputPorts listed in the EVCControlMechanism's
    `monitored_output_ports <EVCControlMechanism.monitored_output_ports>` attribute (and the `objective_mechanism
    <EVCControlMechanism.objective_mechanism>`'s `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>`
    attribute) by taking their elementwise (Hadamard) product.  However, this behavior can be customized in a variety
    of ways, as described `above <EVCControlMechanism_ObjectiveMechanism>`.

  * **Calculate EVC** - call the EVCControlMechanism's `value_function <EVCControlMechanism.value_function>` passing it
    the outcome (received from the `objective_mechanism`) and a list of the `costs <ControlSignal.cost>` \\s of its
    `ControlSignals <EVCControlMechanism_ControlSignals>`.  The default `value_function
    <EVCControlMechanism.value_function>` calls two additional auxiliary functions, in the following order:

    - `cost_function <EVCControlMechanism.cost_function>`, which sums the costs;  this can be configured to weight
      and/or exponentiate individual costs (see `cost_function <EVCControlMechanism.cost_function>` attribute);

    - `combine_outcome_and_cost_function <EVCControlMechanism.combine_outcome_and_cost_function>`, which subtracts the
      sum of the costs from the outcome to generate the EVC;  this too can be configured (see
      `combine_outcome_and_cost_function <EVCControlMechanism.combine_outcome_and_cost_function>`).

In addition to modifying the default functions (as noted above), any or all of them can be replaced with a custom
function to modify how the `control_allocation <EVCControlMechanism.control_allocation>` is determined, so long as the
custom function accepts arguments and returns values that are compatible with any other functions that call that
function (see note below).

.. _EVCControlMechanism_Calling_and_Assigning_Functions:

    .. note::
       The `EVCControlMechanism auxiliary functions <EVCControlMechanism_Functions>` described above are all
       implemented as PsyNeuLink `Functions <Function>`.  Therefore, to call a function itself, it must be referenced
       as ``<EVCControlMechanism>.<function_attribute>.function``.  A custom function assigned to one of the auxiliary
       functions can be either a PsyNeuLink `Function <Function>`, or a generic python function or method (including
       a lambda function).  If it is one of the latter, it is automatically "wrapped" as a PsyNeuLink `Function
       <Function>` (specifically, it is assigned as the `function <UserDefinedFunction.function>` attribute of a
       `UserDefinedFunction` object), so that it can be referenced and called in the same manner as the default
       function assignment. Therefore, once assigned, it too must be referenced as
       ``<EVCControlMechanism>.<function_attribute>.function``.

.. _EVCControlMechanism_ControlSignals:

*ControlSignals*
~~~~~~~~~~~~~~~~

The OutputPorts of an EVCControlMechanism (like any `ControlMechanism`) are a set of `ControlSignals
<ControlSignal>`, that are listed in its `control_signals <EVCControlMechanism.control_signals>` attribute (as well as
its `output_ports <ControlMechanism.output_ports>` attribute).  Each ControlSignal is assigned a  `ControlProjection`
that projects to the `ParameterPort` for a parameter controlled by the EVCControlMechanism.  Each ControlSignal is
assigned an item of the EVCControlMechanism's `control_allocation`, that determines its `allocation
<ControlSignal.allocation>` for a given `TRIAL` of execution.  The `allocation <ControlSignal.allocation>` is used by
a ControlSignal to determine its `intensity <ControlSignal.intensity>`, which is then assigned as the `value
<ControlProjection.value>` of the ControlSignal's ControlProjection.   The `value <ControlProjection>` of the
ControlProjection is used by the `ParameterPort` to which it projects to modify the value of the parameter (see
`ControlSignal_Modulation` for description of how a ControlSignal modulates the value of a parameter it controls).
A ControlSignal also calculates a `cost <ControlSignal.cost>`, based on its `intensity <ControlSignal.intensity>`
and/or its time course. The `cost <ControlSignal.cost>` is included in the evaluation that the EVCControlMechanism
carries out for a given `control_allocation`, and that it uses to adapt the ControlSignal's `allocation
<ControlSignal.allocation>` in the future.  When the EVCControlMechanism chooses an `control_allocation` to evaluate,
it selects an allocation value from the ControlSignal's `allocation_samples <ControlSignal.allocation_samples>`
attribute.


.. _EVCControlMechanism_Execution:

Execution
---------

An EVCControlMechanism must be the `controller <System.controller>` of a System, and as a consequence it is always the
last `Mechanism <Mechanism>` to be executed in a `TRIAL` for its `system <EVCControlMechanism.system>` (see `System
Control <System_Execution_Control>` and `Execution <System_Execution>`). When an EVCControlMechanism is executed,
it updates the value of its `prediction_mechanisms` and `objective_mechanism`, and then calls its `function
<EVCControlMechanism.function>`, which determines and implements the `control_allocation` for the next `TRIAL` of its
`system <EVCControlMechanism.system>`\\s execution.  The default `function <EVCControlMechanism.function>` executes
the following steps (described in greater detailed `above <EVCControlMechanism_Default_Configuration>`):

* Samples every control_allocation (i.e., every combination of the `allocation` \\s specified for the
  EVCControlMechanism's ControlSignals specified by their `allocation_samples` attributes);  for each
  `control_allocation`, it:

  * Runs a simulation of the EVCControlMechanism's `system <EVCControlMechanism.system>` with the parameter values
    specified by that `control_allocation`, by calling the system's  `evaluate <System.evaluate>` method;
    each simulation uses inputs provided by the EVCControlMechanism's `prediction_mechanisms
    <EVCControlMechanism.prediction_mechanisms>` and includes execution of its `objective_mechanism`, which provides
    its result to the EVCControlMechanism.

  * Calls the EVCControlMechanism's `value_function <EVCControlMechanism.value_function>`, which in turn calls
    EVCControlMechanism's `cost_function <EVCControlMechanism.cost_function>` and `combine_outcome_and_cost_function
    <EVCControlMechanism.combine_outcome_and_cost_function>` to evaluate the EVC for that `control_allocation`.

* Selects and returns the `control_allocation` that generates the maximum EVC value.

This procedure can be modified by specifying a custom function for any or all of the `functions
<EVCControlMechanism_Functions>` referred to above.


.. _EVCControlMechanism_Examples:

Example
-------

The following example implements a System with an EVCControlMechanism (and two processes not shown)::


    >>> import psyneulink as pnl                                                        #doctest: +SKIP
    >>> myRewardProcess = pnl.Process(...)                                              #doctest: +SKIP
    >>> myDecisionProcess = pnl.Process(...)                                            #doctest: +SKIP
    >>> mySystem = pnl.System(processes=[myRewardProcess, myDecisionProcess],           #doctest: +SKIP
    ...                       controller=pnl.EVCControlMechanism,                       #doctest: +SKIP
    ...                       monitor_for_control=[Reward,                              #doctest: +SKIP
    ...                                            pnl.DECISION_VARIABLE,    #doctest: +SKIP
    ...                                            (pnl.RESPONSE_TIME, 1, -1)],         #doctest: +SKIP

It uses the System's **monitor_for_control** argument to assign three OutputPorts to be monitored.  The first one
references the Reward Mechanism (not shown);  its `primary OutputPort <OutputPort_Primary>` will be used by default.
The second and third use keywords that are the names of outputPorts of a  `DDM` Mechanism (also not shown).
The last one (RESPONSE_TIME) is assigned a weight of 1 and an exponent of -1. As a result, each calculation of the EVC
computation will multiply the value of the primary OutputPort of the Reward Mechanism by the value of the
*DDM_DECISION_VARIABLE* OutputPort of the DDM Mechanism, and then divide that by the value of the *RESPONSE_TIME*
OutputPort of the DDM Mechanism.

See `ObjectiveMechanism <ObjectiveMechanism_Monitor_Examples>` for additional examples of how to specify it's
**monitored_output_ports** argument, `ControlMechanism <ControlMechanism_Examples>` for additional examples of how to
specify ControlMechanisms, and `System <System_Examples>` for how to specify the `controller <System.controller>`
of a System.

.. _EVCControlMechanism_Class_Reference:

Class Reference
---------------

"""

import copy
import numpy as np
import typecheck as tc
import warnings

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismList
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.shellclasses import Function, System_Base
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    CONTROL, CONTROLLER, COST_FUNCTION, EVC_MECHANISM, INIT_FUNCTION_METHOD_ONLY, \
    MULTIPLICATIVE, PARAMETER_PORTS, PREDICTION_MECHANISM, PREDICTION_MECHANISMS, SUM
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable
from psyneulink.library.components.mechanisms.modulatory.control.evc.evcauxiliary import ControlSignalGridSearch, PredictionMechanism, ValueFunction

__all__ = [
    'EVCControlMechanism', 'EVCError',
]


class EVCError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class EVCControlMechanism(ControlMechanism):
    """EVCControlMechanism(                                            \
    system=True,                                                       \
    objective_mechanism=None,                                          \
    prediction_mechanisms=PredictionMechanism,                         \
    function=ControlSignalGridSearch                                   \
    value_function=ValueFunction,                                      \
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

        Class methods:
            None

       **********************************************************************************************

       PUT SOME OF THIS STUFF IN ATTRIBUTES, BUT USE DEFAULTS HERE

        # - specification of System:  required param: SYSTEM
        # - kwDefaultController:  True =>
        #         takes over all unassigned ControlProjections (i.e., without a sender) in its System;
        #         does not take monitored ports (those are created de-novo)
        # TBI: - CONTROL_PROJECTIONS:
        #         list of projections to add (and for which outputPorts should be added)

        # - input_ports: one for each performance/environment variable monitored

        ControlProjection Specification:
        #    - wherever a ControlProjection is specified, using kwEVC instead of CONTROL_PROJECTION
        #     this should override the default sender SYSTEM_DEFAULT_CONTROLLER in ControlProjection._instantiate_sender
        #    ? expclitly, in call to "EVC.monitor(input_port, parameter_port=NotImplemented) method

        # - specification of function: default is default allocation policy (BADGER/GUMBY)
        #   constraint:  if specified, number of items in variable must match number of input_ports in INPUT_PORTS
        #                  and names in list in kwMonitor must match those in INPUT_PORTS

       **********************************************************************************************

       NOT CURRENTLY IN USE:

        system : System
            System for which the EVCControlMechanism is the controller;  this is a required parameter.

        default_variable : Optional[number, list or np.ndarray] : `defaultControlAllocation <LINK]>`

    COMMENT


    Arguments
    ---------

    system : System : default None
        specifies the `System` for which the EVCControlMechanism should serve as a `controller <System.controller>`;
        the EVCControlMechanism will inherit any `OutputPorts <OutputPort>` specified in the **monitor_for_control**
        argument of the `system <EVCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    objective_mechanism : ObjectiveMechanism, List[OutputPort or Tuple[OutputPort, list or 1d np.array, list or 1d
    np.array]] : \
    default MonitoredOutputPortsOptions.PRIMARY_OUTPUT_PORTS
        specifies either an `ObjectiveMechanism` to use for the EVCControlMechanism or a list of the OutputPorts it
        should monitor; if a list of `OutputPort specifications <ObjectiveMechanism_Monitor>` is used,
        a default ObjectiveMechanism is created and the list is passed to its **monitored_output_ports** argument.

    prediction_mechanisms : Mechanism, Mechanism subclass, dict, (Mechanism subclass, dict) or list: \
    default PredictionMechanism
        the `Mechanism(s) <Mechanism>` or class(es) of Mechanisms  used for `prediction Mechanism(s)
        <EVCControlMechanism_Prediction_Mechanisms>` and, optionally, their parameters (specified in a `parameter
        specification dictionary <ParameterPort_Specification>`);  see `EVCControlMechanism_Prediction_Mechanisms`
        for details.

        COMMENT:
        the `Mechanism(s) <Mechanism>` or class(es) of Mechanisms  used for `prediction Mechanism(s)
        <EVCControlMechanism_Prediction_Mechanisms>`.  If a class, dict, or tuple is specified, it is used as the
        specification for all prediction Mechanisms.  A dict specified on its own is assumed to be a `parameter
        specification dictionary <ParameterPort_Specification>` for a `PredictionMechanism`; a dict specified
        in a tuple must be a `parameter specification dictionary <ParameterPort_Specification>` appropriate for the
        type of Mechanism specified as the first item of the tuple.  If a list is specified, its length must equal
        the number of `ORIGIN` Mechanisms in the System for which the EVCControlMechanism is the `controller
        <System.controller>`;  each item must be a Mechanism, subclass of one, or a tuple specifying a subclass and
        parameter specification dictionary, that is used as the specification for the prediction Mechanism for the
        corresponding item in list of Systems in `ORIGIN` Mechanism in its `origin_mechanisms
        <System.origin_mechanisms>` attribute.

        ..note::
            Specifying a single instantiated Mechanism (i.e., outside of a list) is a convenience notation, that
            assumes the System for which the EVCControlMechanism is the `controller <System.controller>` has a single
            `ORIGIN` Mechanism; this will cause an error if the System has more than one `ORIGIN` Mechanism;  in that
            case, one of the other forms of specification must be used.
        COMMENT

    function : function or method : ControlSignalGridSearch
        specifies the function used to determine the `control_allocation` for the next execution of the
        EVCControlMechanism's `system <EVCControlMechanism.system>` (see `function <EVCControlMechanism.function>`
        for details).

    value_function : function or method : value_function
        specifies the function used to calculate the `EVC <EVCControlMechanism_EVC>` for the current `control_allocation`
        (see `value_function <EVCControlMechanism.value_function>` for details).

    cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to calculate the cost associated with the current `control_allocation`
        (see `cost_function <EVCControlMechanism.cost_function>` for details).

    combine_outcome_and_cost_function : function or method : LinearCombination(operation=SUM)
        specifies the function used to combine the outcome and cost associated with the current `control_allocation`,
        to determine its value (see `combine_outcome_and_cost_function` for details).

    save_all_values_and_policies : bool : default False
        specifes whether to save every `control_allocation` tested in `EVC_policies` and their values in `EVC_values`.

    control_signals : ControlSignal specification or List[ControlSignal specification, ...]
        specifies the parameters to be controlled by the EVCControlMechanism
        (see `ControlSignal_Specification` for details of specification).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that can be used to specify the parameters for the
        Mechanism, its `function <EVCControlMechanism.function>`, and/or a custom function and its parameters.  Values
        specified for parameters in the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <EVCControlMechanism.name>`
        specifies the name of the EVCControlMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the EVCControlMechanism; see `prefs <EVCControlMechanism.prefs>` for details.

    Attributes
    ----------

    system : System
        the `System` for which EVCControlMechanism is the `controller <System.controller>`;
        the EVCControlMechanism inherits any `OutputPorts <OutputPort>` specified in the **monitor_for_control**
        argument of the `system <EVCControlMechanism.system>`'s constructor, and any `ControlSignals <ControlSignal>`
        specified in its **control_signals** argument.

    prediction_mechanisms : List[ProcessingMechanism]
        list of `predictions mechanisms <EVCControlMechanism_Prediction_Mechanisms>` generated for the
        EVCControlMechanism's `system <EVCControlMechanism.system>` when the EVCControlMechanism is created,
        one for each `ORIGIN` Mechanism in the `system <EVCControlMechanism.system>`.  Each prediction Mechanism is
        named using the name of the ` ORIGIN` Mechanism + "PREDICTION_MECHANISM" and assigned an `OutputPort` with
        a name based on the same.

    origin_prediction_mechanisms : Dict[ProcessingMechanism, ProcessingMechanism]
        dictionary of `prediction mechanisms <EVCControlMechanism_Prediction_Mechanisms>` added to the
        EVCControlMechanism's `system <EVCControlMechanism.system>`, one for each of its `ORIGIN` Mechanisms.  The
        key for each entry is an `ORIGIN` Mechanism of the System, and the value is the corresponding prediction
        Mechanism.

    predicted_input : Dict[ProcessingMechanism, value]
        dictionary with the `value <Mechanism_Base.value>` of each `prediction Mechanism
        <EVCControlMechanism_Prediction_Mechanisms>` listed in `prediction_mechanisms` corresponding to each `ORIGIN`
        Mechanism of the System. The key for each entry is the name of an `ORIGIN` Mechanism, and its
        value the `value <Mechanism_Base.value>` of the corresponding prediction Mechanism.

    objective_mechanism : ObjectiveMechanism
        the 'ObjectiveMechanism' used by the EVCControlMechanism to evaluate the performance of its `system
        <EVCControlMechanism.system>`.  If a list of OutputPorts is specified in the **objective_mechanism** argument
        of the EVCControlMechanism's constructor, they are assigned as the `monitored_output_ports
        <ObjectiveMechanism.monitored_output_ports>` attribute for the `objective_mechanism
        <EVCControlMechanism.objective_mechanism>`.

    monitored_output_ports : List[OutputPort]
        list of the OutputPorts monitored by `objective_mechanism <EVCControlMechanism.objective_mechanism>` (and
        listed in its `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>` attribute), and used to
        evaluate the performance of the EVCControlMechanism's `system <EVCControlMechanism.system>`.

    COMMENT:
    [TBI]
        monitored_output_ports : 3D np.array
            an array of values of the outputPorts in `monitored_output_ports` (equivalent to the values of
            the EVCControlMechanism's `input_ports <EVCControlMechanism.input_ports>`).
    COMMENT

    monitored_output_ports_weights_and_exponents: List[Tuple[scalar, scalar]]
        a list of tuples, each of which contains the weight and exponent (in that order) for an OutputPort in
        `monitored_outputPorts`, listed in the same order as the outputPorts are listed in `monitored_outputPorts`.

    function : function : default ControlSignalGridSearch
        determines the `control_allocation` to use for the next round of the System's
        execution. The default function, `ControlSignalGridSearch`, conducts an exhaustive (*grid*) search of all
        combinations of the `allocation_samples` of its ControlSignals (and contained in its
        `control_signal_search_space` attribute), by executing the System (using `evaluate`) for each
        combination, evaluating the result using `value_function`, and returning the `control_allocation` that yielded
        the greatest `EVC <EVCControlMechanism_EVC>` value (see `EVCControlMechanism_Default_Configuration` for
        additional details). If a custom function is specified, it must accommodate a **controller** argument that
        specifies an EVCControlMechanism (and provides access to its attributes, including
        `control_signal_search_space`), and must return an array with the same format (number and type of elements)
        as the EVCControlMechanism's `control_allocation` attribute.

    COMMENT:
        NOTES ON API FOR CUSTOM VERSIONS:
            Gets controller as argument (along with any standard params specified in call)
            Must include **kwargs to receive standard args (variable, params, and context)
            Must return an allocation policy compatible with controller.control_allocation:
                2d np.array with one array for each allocation value

            Following attributes are available:
            controller._get_simulation_system_inputs gets inputs for a simulated run (using predictionMechanisms)
            controller._assign_simulation_inputs assigns value of prediction_mechanisms to inputs of `ORIGIN` Mechanisms
            controller.run will execute a specified number of trials with the simulation inputs
            controller.monitored_ports is a list of the Mechanism OutputPorts being monitored for outcome
            controller.input_value is a list of current outcome values (values for monitored_ports)
            controller.monitored_output_ports_weights_and_exponents is a list of parameterizations for OutputPorts
            controller.control_signals is a list of control_signal objects
            controller.control_signal_search_space is a list of all allocationPolicies specifed by allocation_samples
            control_signal.allocation_samples is the set of samples specified for that control_signal
            [TBI:] control_signal.allocation_range is the range that the control_signal value can take
            controller.control_allocation - holds current control_allocation
            controller.output_values is a list of current control_signal values
            controller.value_function - calls the three following functions (done explicitly, so each can be specified)
            controller.cost_function - combines costs of control signals
            controller.combine_outcome_and_cost_function - combines outcomes and costs
    COMMENT

    value_function : function : default ValueFunction
        calculates the `EVC <EVCControlMechanism_EVC>` for a given `control_allocation`.  It takes as its arguments an
        `EVCControlMechanism`, an **outcome** value and a list or ndarray of **costs**, uses these to calculate an EVC,
        and returns a three item tuple with the calculated EVC, and the outcome value and combined value of costs
        used to calculate the EVC.  The default, `ValueFunction`,  calls the EVCControlMechanism's `cost_function
        <EVCControlMechanism.cost_function>` to combine the value of the costs, and then calls its
        `combine_outcome_and_costs <EVCControlMechanism.combine_outcome_and_costs>` to calculate the EVC from the
        outcome and combined cost (see `EVCControlMechanism_Default_Configuration` for additional details).  A custom
        function can be assigned to `value_function` so long as it returns a tuple with three items: the calculated
        EVC (which must be a scalar value), and the outcome and cost from which it was calculated (these can be scalar
        values or `None`). If used with the EVCControlMechanism's default `function <EVCControlMechanism.function>`, a
        custom `value_function` must accommodate three arguments (passed by name): a **controller** argument that is
        the EVCControlMechanism for which it is carrying out the calculation; an **outcome** argument that is a
        value; and a `costs` argument that is a list or ndarray.  A custom function assigned to `value_function` can
        also call any of the `helper functions <EVCControlMechanism_Functions>` that it calls (however, see `note
        <EVCControlMechanism_Calling_and_Assigning_Functions>` above).

    cost_function : function : default LinearCombination(operation=SUM)
        calculates the cost of the `ControlSignals <ControlSignal>` for the current `control_allocation`.  The default
        function sums the `cost <ControlSignal.cost>` of each of the EVCControlMechanism's `ControlSignals
        <EVCControlMechanism_ControlSignals>`.  The `weights <LinearCombination.weights>` and/or `exponents
        <LinearCombination.exponents>` parameters of the function can be used, respectively, to scale and/or
        exponentiate the contribution of each ControlSignal cost to the combined cost.  These must be specified as
        1d arrays in a *WEIGHTS* and/or *EXPONENTS* entry of a `parameter dictionary <ParameterPort_Specification>`
        assigned to the **params** argument of the constructor of a `LinearCombination` function; the length of
        each array must equal the number of (and the values listed in the same order as) the ControlSignals in the
        EVCControlMechanism's `control_signals <EVCControlMechanism.control_signals>` attribute. The default function
        can also be replaced with any `custom function <EVCControlMechanism_Calling_and_Assigning_Functions>` that
        takes an array as input and returns a scalar value.  If used with the EVCControlMechanism's default
        `value_function <EVCControlMechanism.value_function>`, a custom `cost_function
        <EVCControlMechanism.cost_function>` must accommodate two arguments (passed by name): a **controller**
        argument that is the EVCControlMechanism itself;  and a **costs** argument that is a 1d array of scalar
        values specifying the `cost <ControlSignal.cost>` for each ControlSignal listed in the `control_signals`
        attribute of the ControlMechanism specified in the **controller** argument.

    combine_outcome_and_cost_function : function : default LinearCombination(operation=SUM)
        combines the outcome and cost for given `control_allocation` to determine its `EVC <EVCControlMechanisms_EVC>`.
        The default function subtracts the cost from the outcome, and returns the difference.  This can be modified
        using the `weights <LinearCombination.weights>` and/or `exponents <LinearCombination.exponents>` parameters
        of the function, as described for the `cost_function <EVCControlMechanisms.cost_function>`.  The default
        function can also be replaced with any `custom function <EVCControlMechanism_Calling_and_Assigning_Functions>`
        that returns a scalar value.  If used with the EVCControlMechanism's default `value_function`, a custom
        If used with the EVCControlMechanism's default `value_function`, a custom combine_outcome_and_cost_function
        must accomodate three arguments (passed by name): a **controller** argument that is the EVCControlMechanism
        itself; an **outcome** argument that is a 1d array with the outcome of the current `control_allocation`;
        and a **cost** argument that is 1d array with the cost of the current `control_allocation`.

    control_signal_search_space : 2d np.array
        an array each item of which is an `control_allocation`.  By default, it is assigned the set of all possible
        allocation policies, using np.meshgrid to construct all permutations of `ControlSignal` values from the set
        specified for each by its `allocation_samples <EVCControlMechanism.allocation_samples>` attribute.

    EVC_max : 1d np.array with single value
        the maximum `EVC <EVCControlMechanism_EVC>` value over all allocation policies in `control_signal_search_space`.

    EVC_max_port_values : 2d np.array
        an array of the values for the OutputPorts in `monitored_output_ports` using the `control_allocation` that
        generated `EVC_max`.

    EVC_max_policy : 1d np.array
        an array of the ControlSignal `intensity <ControlSignal.intensity> values for the allocation policy that
        generated `EVC_max`.

    save_all_values_and_policies : bool : default False
        specifies whether or not to save every `control_allocation and associated EVC value (in addition to the max).
        If it is specified, each `control_allocation` tested in the `control_signal_search_space` is saved in
        `EVC_policies`, and their values are saved in `EVC_values`.

    EVC_policies : 2d np.array
        array with every `control_allocation` tested in `control_signal_search_space`.  The `EVC
        <EVCControlMechanism_EVC>` value of each is stored in `EVC_values`.

    EVC_values :  1d np.array
        array of `EVC <EVCControlMechanism_EVC>` values, each of which corresponds to an `control_allocation` in
        `EVC_policies`;

    control_allocation : 2d np.array : [[defaultControlAllocation]]
        determines the value assigned as the `variable <Projection_Base.variable>` for each `ControlSignal` and its
        associated `ControlProjection`.  Each item of the array must be a 1d array (usually containing a scalar)
        that specifies an `allocation` for the corresponding ControlSignal, and the number of items must equal the
        number of ControlSignals in the EVCControlMechanism's `control_signals` attribute.

    control_signals : ContentAddressableList[ControlSignal]
        list of the EVCControlMechanism's `ControlSignals <EVCControlMechanism_ControlSignals>`, including any that it
        inherited from its `system <EVCControlMechanism.system>` (same as the EVCControlMechanism's `output_ports
        <Mechanism_Base.output_ports>` attribute); each sends a `ControlProjection` to the `ParameterPort` for the
        parameter it controls

    name : str
        the name of the EVCControlMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the EVCControlMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = EVC_MECHANISM
    initMethod = INIT_FUNCTION_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE

    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'DefaultControlMechanismCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(ControlMechanism.Parameters):
        """
            Attributes
            ----------

                EVC_max
                    see `EVC_max <EVCControlMechanism.EVC_max>`

                    :default value: None
                    :type:
                    :read only: True

                EVC_max_policy
                    see `EVC_max_policy <EVCControlMechanism.EVC_max_policy>`

                    :default value: None
                    :type:
                    :read only: True

                EVC_max_port_values
                    see `EVC_max_port_values <EVCControlMechanism.EVC_max_port_values>`

                    :default value: None
                    :type:
                    :read only: True

                EVC_policies
                    see `EVC_policies <EVCControlMechanism.EVC_policies>`

                    :default value: []
                    :type: ``list``
                    :read only: True

                EVC_values
                    see `EVC_values <EVCControlMechanism.EVC_values>`

                    :default value: []
                    :type: ``list``
                    :read only: True

                combine_outcome_and_cost_function
                    see `combine_outcome_and_cost_function <EVCControlMechanism.combine_outcome_and_cost_function>`

                    :default value: `LinearCombination`
                    :type: `Function`

                control_signal_search_space
                    see `control_signal_search_space <EVCControlMechanism.control_signal_search_space>`

                    :default value: None
                    :type:
                    :read only: True

                cost_function
                    see `cost_function <EVCControlMechanism.cost_function>`

                    :default value: `LinearCombination`
                    :type: `Function`

                function
                    see `function <EVCControlMechanism.function>`

                    :default value: `ControlSignalGridSearch`
                    :type: `Function`

                modulation
                    see `modulation <EVCControlMechanism.modulation>`

                    :default value: `MULTIPLICATIVE_PARAM`
                    :type: ``str``

                origin_objective_mechanism
                    see `origin_objective_mechanism <EVCControlMechanism.origin_objective_mechanism>`

                    :default value: None
                    :type:

                predicted_input
                    see `predicted_input <EVCControlMechanism.predicted_input>`

                    :default value: None
                    :type:
                    :read only: True

                prediction_mechanisms
                    see `prediction_mechanisms <EVCControlMechanism_Prediction_Mechanisms>`

                    :default value: None
                    :type:

                save_all_values_and_policies
                    see `save_all_values_and_policies <EVCControlMechanism.save_all_values_and_policies>`

                    :default value: False
                    :type: ``bool``

                terminal_objective_mechanism
                    see `terminal_objective_mechanism <EVCControlMechanism.terminal_objective_mechanism>`

                    :default value: None
                    :type:

                value_function
                    see `value_function <EVCControlMechanism.value_function>`

                    :default value: `ValueFunction`
                    :type: `Function`
        """
        function = Parameter(ControlSignalGridSearch, stateful=False, loggable=False)
        value_function = Parameter(ValueFunction, stateful=False, loggable=False)
        cost_function = Parameter(LinearCombination, stateful=False, loggable=False)
        combine_outcome_and_cost_function = Parameter(LinearCombination, stateful=False, loggable=False)
        save_all_values_and_policies = Parameter(False, stateful=False, loggable=False)

        modulation = MULTIPLICATIVE

        EVC_max = Parameter(None, read_only=True)
        EVC_values = Parameter([], read_only=True)
        EVC_policies = Parameter([], read_only=True)
        EVC_max_port_values = Parameter(None, read_only=True)
        EVC_max_policy = Parameter(None, read_only=True)
        control_signal_search_space = Parameter(None, read_only=True)
        predicted_input = Parameter(None, read_only=True)

        prediction_mechanisms = None
        origin_objective_mechanism = None
        terminal_objective_mechanism = None
        system = None

    @tc.typecheck
    def __init__(self,
                 system:tc.optional(System_Base)=None,
                 prediction_mechanisms:tc.any(is_iterable, Mechanism, type)=PredictionMechanism,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 origin_objective_mechanism=False,
                 terminal_objective_mechanism=False,
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputPort))=None,
                 function=ControlSignalGridSearch,
                 value_function=ValueFunction,
                 cost_function=LinearCombination(operation=SUM),
                 combine_outcome_and_cost_function=LinearCombination(operation=SUM),
                 save_all_values_and_policies:bool=False,
                 control_signals:tc.optional(tc.any(is_iterable, ParameterPort, ControlSignal))=None,
                 modulation:tc.optional(str)=MULTIPLICATIVE,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        super().__init__(
            system=system,
            prediction_mechanisms=prediction_mechanisms,
            origin_objective_mechanism=origin_objective_mechanism,
            terminal_objective_mechanism=terminal_objective_mechanism,
            value_function=value_function,
            cost_function=cost_function,
            combine_outcome_and_cost_function=combine_outcome_and_cost_function,
            save_all_values_and_policies=save_all_values_and_policies,
            objective_mechanism=objective_mechanism,
            monitor_for_control=monitor_for_control,
            function=function,
            control_signals=control_signals,
            modulation=modulation,
            params=params,
            name=name,
            prefs=prefs
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate prediction_mechanisms"""

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if PREDICTION_MECHANISMS in target_set:
            prediction_mechanisms = target_set[PREDICTION_MECHANISMS]
            if isinstance(prediction_mechanisms, type) and not issubclass(prediction_mechanisms, Mechanism):
                raise EVCError("Class used to specify {} argument of {} ({}) must be a type of {}".
                               format(self.name,repr(PREDICTION_MECHANISMS),prediction_mechanisms,Mechanism.__name__))
            elif isinstance(prediction_mechanisms, list):
                for pm in prediction_mechanisms:
                    if not (isinstance(pm,Mechanism) or
                            (isinstance(pm,type) and issubclass(pm,Mechanism)) or
                            (isinstance(pm,tuple) and issubclass(pm[0],Mechanism) and isinstance(pm[1],dict))):
                        raise EVCError("Unrecognized item ({}) in the list specified for {} arg of constructor for {}; "
                                       "must be a Mechanism, a class of Mechanism, or a tuple with a Mechanism class "
                                       "and parameter specification dictionary".
                                       format(pm, repr(PREDICTION_MECHANISMS), self.name))

    def _instantiate_input_ports(self, context=None):
        """Instantiate PredictionMechanisms
        """
        if self.system is not None:
            self._instantiate_prediction_mechanisms(system=self.system, context=context)
        super()._instantiate_input_ports(context=context)

    def _instantiate_control_signals(self, context):
        """Size control_allocation and assign modulatory_signals
        Set size of control_allocadtion equal to number of modulatory_signals.
        Assign each modulatory_signal sequentially to corresponding item of control_allocation.
        """
        from psyneulink.core.globals.keywords import OWNER_VALUE
        for i, spec in list(enumerate(self.control)):
            control_signal = self._instantiate_control_signal(spec, context=context)
            control_signal._variable_spec = (OWNER_VALUE, i)
            self.control[i] = control_signal
        self.defaults.value = np.tile(control_signal.parameters.variable.default_value, (i + 1, 1))
        self.parameters.control_allocation._set(copy.deepcopy(self.defaults.value), context)

    def _instantiate_prediction_mechanisms(self, system:System_Base, context=None):
        """Add prediction Mechanism and associated process for each `ORIGIN` (input) Mechanism in system

        Instantiate prediction_mechanisms for `ORIGIN` Mechanisms in system; these will now be `TERMINAL` Mechanisms:
            - if their associated input mechanisms were TERMINAL MECHANISMS, they will no longer be so;  therefore...
            - if an associated input Mechanism must be monitored by the EVCControlMechanism, it must be specified
                explicitly in an OutputPort, Mechanism, controller or system OBJECTIVE_MECHANISM param (see below)

        For each `ORIGIN` Mechanism in system:
            - instantiate a corresponding PredictionMechanism
            - instantiate a MappingProjection to the PredictionMechanism
                that shadows the one from the SystemInputPort to the ORIGIN Mechanism

        Instantiate self.predicted_input dict:
            - key for each entry is an `ORIGIN` Mechanism of system
            - value of each entry is a list of the values of the corresponding predictionMechanism,
                one for each trial to be simulated; each value is a 2d array, each item of which is the value of an
                InputPort of the predictionMechanism

        Args:
            context:
        """

        from psyneulink.core.components.process import ProcessInputPort
        from psyneulink.core.components.system import SystemInputPort

        # FIX: 1/16/18 - Should should check for any new origin_mechs? What if origin_mech deleted?
        # If system's controller already has prediction_mechanisms, use those
        if hasattr(system, CONTROLLER) and hasattr(system.controller, PREDICTION_MECHANISMS):
            self.prediction_mechanisms = system.controller.prediction_mechanisms
            self.origin_prediction_mechanisms = system.controller.origin_prediction_mechanisms
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.predicted_input = system.controller.predicted_input
            return

        # Dictionary of prediction_mechanisms, keyed by the ORIGIN Mechanism to which they correspond
        self.origin_prediction_mechanisms = {}

        # List of prediction Mechanism tuples (used by System to execute them)
        self.prediction_mechs = []

        # IF IT IS A MECHANISM, PUT IT IN A LIST
        # IF IT IS A CLASS, PUT IT IN A TUPLE WITH NONE
        # NOW IF IT IS TUPLE,

        # self.prediction_mechanisms is:
        if isinstance(self.prediction_mechanisms, Mechanism):
            # a single Mechanism, so put it in a list
            prediction_mech_specs = [self.prediction_mechanisms]
        elif isinstance(self.prediction_mechanisms, type):
            # a class, so put it as 1st item in a 2-item tuple, with None as 2nd item
            prediction_mech_specs = (self.prediction_mechanisms, None)
        elif isinstance(self.prediction_mechanisms, dict):
            # a dict, so put it as 2nd item in a 2-item tuple, with PredictionMechanism as 1st item
            prediction_mech_specs = (PredictionMechanism, self.prediction_mechanisms)
        elif isinstance(self.prediction_mechanisms, tuple):
            # a tuple, so leave as is for now (put in list below)
            prediction_mech_specs = self.prediction_mechanisms

        if isinstance(prediction_mech_specs, tuple):
            # a tuple, so create a list with same length as self.system.origin_mechanisms, and tuple as each item
            prediction_mech_specs = [prediction_mech_specs] * len(system.origin_mechanisms)

        # Make sure prediction_mechanisms is the same length as self.system.origin_mechanisms
        from psyneulink.core.components.system import ORIGIN_MECHANISMS
        if len(prediction_mech_specs) != len(system.origin_mechanisms):
            raise EVCError("Number of PredictionMechanisms specified for {} ({}) "
                           "must equal the number of {} ({}) in the System it controls ({})".
                           format(self.name, len(prediction_mech_specs),
                           repr(ORIGIN_MECHANISMS), len(system.orign_mechanisms), self.system.name))

        for origin_mech, pm_spec in zip(system.origin_mechanisms.mechanisms, prediction_mech_specs):
            port_Names = []
            variable = []
            for port_Name in origin_mech.input_ports.names:
                port_Names.append(port_Name)
                # variable.append(origin_mech.input_ports[port_Name].defaults.variable)
                variable.append(origin_mech.input_ports[port_Name].value)

            # Instantiate PredictionMechanism
            if isinstance(pm_spec, Mechanism):
                prediction_mechanism=pm_spec
            elif isinstance(pm_spec, tuple):
                mech_class = pm_spec[0]
                mech_params = pm_spec[1] or {}
                prediction_mechanism = mech_class(
                        name=origin_mech.name + " " + PREDICTION_MECHANISM,
                        default_variable=variable,
                        input_ports=port_Names,
                        # params = mech_params
                        **mech_params,
                        context=context
                )
            else:
                raise EVCError("PROGRAM ERROR: Unexpected item ({}) in list for {} arg of constructor for {}".
                               format(pm_spec, repr(PREDICTION_MECHANISMS), self.name))

            prediction_mechanism._role = CONTROL
            prediction_mechanism.origin_mech = origin_mech

            # Assign projections to prediction_mechanism that duplicate those received by origin_mech
            #    (this includes those from ProcessInputPort, SystemInputPort and/or recurrent ones
            for orig_input_port, prediction_input_port in zip(origin_mech.input_ports,
                                                            prediction_mechanism.input_ports):
                for projection in orig_input_port.path_afferents:
                    proj = MappingProjection(sender=projection.sender,
                                      receiver=prediction_input_port,
                                      matrix=projection.matrix)

                    if isinstance(proj.sender, (ProcessInputPort, SystemInputPort)):
                        proj._activate_for_compositions(proj.sender.owner)
                    else:
                        proj._activate_for_compositions(system)

            # Assign list of processes for which prediction_mechanism will provide input during the simulation
            # - used in _get_simulation_system_inputs()
            # - assign copy,
            #       since don't want to include the prediction process itself assigned to origin_mech.processes below
            prediction_mechanism.use_for_processes = list(origin_mech.processes.copy())

            # # FIX: REPLACE REFERENCE TO THIS ELSEWHERE WITH REFERENCE TO MECH_TUPLES BELOW
            self.origin_prediction_mechanisms[origin_mech] = prediction_mechanism

            # Add to list of EVCControlMechanism's prediction_object_items
            # prediction_object_item = prediction_mechanism
            self.prediction_mechs.append(prediction_mechanism)

            # Add to system execution_graph and execution_list
            system.execution_graph[prediction_mechanism] = set()
            system.execution_list.append(prediction_mechanism)

        self.prediction_mechanisms = MechanismList(self, self.prediction_mechs)

        # Assign list of destinations for predicted_inputs:
        #    the variable of the ORIGIN Mechanism for each Process in the system
        predicted_input = {}
        for i, origin_mech in zip(range(len(system.origin_mechanisms)), system.origin_mechanisms):
            predicted_input[origin_mech] = system.processes[i].origin_mechanisms[0].defaults.variable
        self.parameters.predicted_input._set(predicted_input, context)

    def _instantiate_attributes_after_function(self, context=None):
        """Validate cost function"""

        super()._instantiate_attributes_after_function(context=context)

        if self.system is None or not self.system.enable_controller:
            return

        cost_Function = self.cost_function
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

    @handle_external_context()
    @tc.typecheck
    def assign_as_controller(self, system:System_Base, context=None):
        self._instantiate_prediction_mechanisms(system=system, context=context)
        super().assign_as_controller(system=system, context=context)

    def _execute(
        self,
        variable=None,
        context=None,
        runtime_params=None,

    ):
        """Determine `control_allocation <EVCControlMechanism.control_allocation>` for next run of System

        Update prediction mechanisms
        Construct control_signal_search_space (from allocation_samples of each item in control_signals):
            * get `allocation_samples` for each ControlSignal in `control_signals`
            * construct `control_signal_search_space`: a 2D np.array of control allocation policies, each policy of
              which is a different combination of values, one from the `allocation_samples` of each ControlSignal.
        Call self.function -- default is ControlSignalGridSearch
        Return an control_allocation
        """

        if context.source != ContextFlags.PROPERTY:
            self._update_predicted_input(context=context)
        # self.system._cache_state()

        # CONSTRUCT SEARCH SPACE

        control_signal_sample_lists = []
        control_signals = self.control_signals
        # Get allocation_samples for all ControlSignals
        num_control_signals = len(control_signals)


        for control_signal in self.control_signals:
            control_signal_sample_lists.append(control_signal.parameters.allocation_samples._get(context)())

        # Construct control_signal_search_space:  set of all permutations of ControlSignal allocations
        #                                     (one sample from the allocationSample of each ControlSignal)
        # Reference for implementation below:
        # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.parameters.control_signal_search_space._set(
            np.array(np.meshgrid(*control_signal_sample_lists)).T.reshape(-1,num_control_signals),
            context,
        )

        # EXECUTE SEARCH

        # IMPLEMENTATION NOTE:
        # self.system._store_system_state()

        # IMPLEMENTATION NOTE:  skip ControlMechanism._execute since it is a stub method that returns input_values
        control_allocation = super(ControlMechanism, self)._execute(
            controller=self,
            variable=variable,
            context=context,
            runtime_params=runtime_params,

        )

        # IMPLEMENTATION NOTE:
        # self.system._restore_system_state()
        return control_allocation

    def _update_predicted_input(self, context=None):
        """Assign values of prediction mechanisms to predicted_input

        Assign value of each predictionMechanism.value to corresponding item of self.predictedIinput
        Note: must be assigned in order of self.system.processes

        """

        # The number of ORIGIN mechanisms requiring input should = the number of prediction mechanisms
        num_origin_mechs = len(self.system.origin_mechanisms)
        num_prediction_mechs = len(self.origin_prediction_mechanisms)
        if num_origin_mechs != num_prediction_mechs:
            raise EVCError("PROGRAM ERROR:  The number of ORIGIN mechanisms ({}) does not equal"
                           "the number of prediction_predictions mechanisms ({}) for {}".
                           format(num_origin_mechs, num_prediction_mechs, self.system.name))

        # Assign predicted_input for each process in system.processes
        for origin_mech in self.system.origin_mechanisms:
            # Get origin Mechanism for each process
            # Assign value of predictionMechanism to the entry of predicted_input for the corresponding ORIGIN Mechanism
            self.parameters.predicted_input._get(context)[origin_mech] = self.origin_prediction_mechanisms[origin_mech].parameters.value._get(context)
            # self.predicted_input[origin_mech] = self.origin_prediction_mechanisms[origin_mech].output_port.value

    def evaluate(
        self,
        inputs,
        allocation_vector,
        context=None,
        runtime_params=None,
        reinitialize_values=None,

    ):
        """
        Run simulation of `System` for which the EVCControlMechanism is the `controller <System.controller>`.

        Arguments
        ----------

        inputs : List[input] or ndarray(input) : default default_variable
            the inputs provided to the ORIGIN Mechanisms of the `System` during each simulation.  This should be the
            `value <Mechanism_Base.value> for each `prediction Mechanism <EVCControlMechanism_Prediction_Mechanisms>`
            in the `prediction_mechanisms` attribute.  The inputs are available in the `predicted_input` attribute.

        allocation_vector : (1D np.array)
            the allocation policy to use in running the simulation, with one allocation value for each of the
            EVCControlMechanism's ControlSignals (listed in `control_signals`).

        runtime_params : Optional[Dict[str, Dict[str, Dict[str, value]]]]
            a dictionary that can include any of the parameters used as arguments to instantiate the mechanisms,
            their functions, or Projection(s) to any of their ports.  See `Mechanism_Runtime_Parameters` for a full
            description.

        """

        if self.parameters.value._get(context) is None:
            # Initialize value if it is None
            self.parameters.value._set(np.empty(len(self.control_signals)), context)

        # Implement the current control_allocation over ControlSignals (OutputPorts),
        #    by assigning allocation values to EVCControlMechanism.value, and then calling _update_output_ports
        for i in range(len(self.control_signals)):
            # MODIFIED 6/6/19 OLD:
            self.parameters.value._get(context)[i] = np.atleast_1d(allocation_vector[i])
            # # MODIFIED 6/6/19 NEW: [JDC]
            # self._apply_control_allocation(control_allocation=allocation_vector,
            #                                runtime_params=runtime_params,
            #                                context=context)
            # MODIFIED 6/6/19 END
        self._update_output_ports(context=context, runtime_params=runtime_params)

        # RUN SIMULATION

        # Buffer System attributes
        animate_buffer = self.system._animate

        # Run simulation
        context.execution_phase = ContextFlags.SIMULATION
        self.system.run(
            inputs=inputs,
            context=context,
            reinitialize_values=reinitialize_values,
            animate=False,

        )
        context.execution_phase = ContextFlags.CONTROL

        # Restore System attributes
        self.system._animate = animate_buffer

        # Get outcomes for current control_allocation
        #    = the values of the monitored output ports (self.input_ports)
        # self.objective_mechanism.execute(context=EVC_SIMULATION)
        monitored_ports = self._update_input_ports(context=context, runtime_params=runtime_params)

        # # MODIFIED 9/18/18 OLD:
        # for i in range(len(self.control_signals)):
        #     self.control_signal_costs[i] = self.control_signals[i].cost
        # # MODIFIED 9/18/18 NEW:
        # for i in range(len(self.control_signals)):
        #     if self.control_signal_costs[i].cost_options is not None:
        #         self.control_signal_costs[i] = self.control_signals[i].cost
        # MODIFIED 9/18/18 NEWER:
        control_signal_costs = self.parameters.control_signal_costs._get(context)
        for i, c in enumerate(self.control_signals):
            if c.parameters.cost_options._get(context) is not None:
                control_signal_costs[i] = c.parameters.cost._get(context)
        self.parameters.control_signal_costs._set(control_signal_costs, context)
        # MODIFIED 9/18/18 END

        return monitored_ports
