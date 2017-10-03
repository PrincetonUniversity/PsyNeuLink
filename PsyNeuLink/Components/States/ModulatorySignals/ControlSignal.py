# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  ControlSignal *****************************************************

"""
Overview
--------

A ControlSignal is a type of `ModulatorySignal <ModulatorySignal>` that is specialized for use with a `ControlMechanism
<ControlMechanism>` and one or more `ControlProjections <ControlProjection>`, to modify the parameter(s) of one or more
`Components <Component>`. A ControlSignal receives an `allocation <ControlSignal.allocation>` value from the
ControlMechanism to which it belongs, and uses that to compute an `intensity` (also referred to as a `control_signal`)
that is assigned as the `value <ControlProjection.ControlProjection.value>` of its ControlProjections. Each
ControlProjection conveys its value to the `ParameterState` for the parameter it controls, which uses that value to
`modulate <ModulatorySignal_Modulation>` the `value <ParameterState.value>` of the parameter.  A ControlSignal also
calculates a `cost`, based on its `intensity` and/or its time course, that may be used by the ControlMechanism to
adapt the ControlSignal's `allocation <ControlSignal.allocation>` in the future.

.. _ControlSignal_Creation:

Creating a ControlSignal
------------------------

A ControlSignal is created automatically whenever the parameter of a Mechanism or of its function is `specified for
control <ControlMechanism_Control_Signals>`.  ControlSignals can also be specified in the **control_signals** argument
of the constructor for a `ControlMechanism <ControlMechanism>`.  Although a ControlSignal can be created directly
using its constructor (or any of the other ways for `creating an outputState <OutputStates_Creation>`), this is usually
not necessary nor is it advisable, as a ControlSignal has dedicated components and requirements for configuration
that must be met for it to function properly.

.. _ControlSignal_Specification:

Specifying ControlSignals
~~~~~~~~~~~~~~~~~~~~~~~~~

When a ControlSignal is specified in the **control_signals** argument of the constructor for a `ControlMechanism
<ControlMechanism>`, the parameter to be controlled must be specified.  This can take any of the following forms:

  * a **ParameterState** of the Mechanism to which the parameter belongs;
  ..
  * a **tuple**, with the name of the parameter as its 1st item and the *Mechanism* to which it belongs as the 2nd;
    note that this is a convenience format, which is simpler to use than a specification dictionary (see below),
    but precludes specification of any `parameters <ControlSignal_Structure>` for the ControlSignal.
  ..
  * a **specification dictionary**, that must contain at least the following two entries:

    * *NAME*: str
        a string that is the name of the parameter to be controlled;

    * *MECHANISM*: Mechanism
        the Mechanism must be the one to the which the parameter belongs.
        (note: the Mechanism itself should be specified even if the parameter belongs to its function).

    The dictionary can also contain entries for any other ControlSignal attributes to be specified
    (e.g., a *MODULATION* and/or *ALLOCATION_SAMPLES* entry); see `below <ControlSignal_Structure>` for a
    description of ControlSignal attributes.

.. _ControlSignal_Structure:

Structure
---------

A ControlSignal is owned by an `ControlMechanism <ControlMechanism>`, and controls the parameters of one or more
Components by modulating the `function <ParameterState.function>` of the `ParameterState` that determines the value
of each of the parameters that it control.  Its operation is governed by several attributes of the ControlSignal,
that are described below.

.. _ControlSignal_Projections:

Projections
~~~~~~~~~~~

When a ControlSignal is created, it can be assigned one or more `ControlProjections <ControlProjection>`, using either
the **projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument
with the key *PROJECTIONS*.  These will be assigned to its `efferents  <ControlSignal.efferents>` attribute.  See
`State Projections <State_Projections>` for additional details concerning the specification of Projections when
creating a State.

.. note::
   Although a ControlSignal can be assigned more than one `ControlProjection`, all of those Projections will receive
   the same `value <ControlProjection.value>` (based on the `intensity` of that ControlSignal), and use the same
   form of `modulation <ControlSignal_Modulation>`.  Thus, for them to be meaningful, they should project to
   ParameterStates for parameters that are meaningfully related to one another (for example, the threshold parameter
   of multiple `DDM` Mechanisms).

.. _ControlSignal_Modulation:

Modulation
~~~~~~~~~~

A ControlSignal has a `modulation <GatingSignal.modulation>` attribute that determines how its ControlSignal's
`value <ControlSignal.value>` is used by the States to which it projects to modify their `value <State_Base.value>` \\s
(see `ModulatorySignal_Modulation` for an explanation of how the `modulation <ControlSignal.modulation>`  attribute is
specified and used to modulate the `value <State_Base.value>` of a State). The `modulation <ControlSignal.modulation>`
attribute can be specified in the **modulation** argument of the constructor for a ControlSignal, or in a specification
dictionary as described `above <ControlSignal_Specification>`. The value must be a value of `ModulationParam`;  if it
is not specified, its default is the value of the `modulation <ControlMechanism.modulation>` attribute of the
ControlMechanism to which the ControlSignal belongs (which is the same for all of the ControlSignals belonging to that
ControlMechanism).  The value of the `modulation <ControlSignal.modulation>` attribute of a ControlSignal is used by
all of the `ControlProjections <ControlProjection>` that project from that ControlSignal.

.. _ControlSignal_Allocation_and_Intensity

Allocation, Function and Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Allocation (variable)*. A ControlSignal is assigned an `allocation <ControlSignal>` by the ControlMechanism to
which it belongs. Some ControlMechanisms sample different allocation values for their ControlSignals to determine
which to use (such as the `EVCControlMechanism <EVC_Default_Configuration>`);  in those cases, they use each ControlSignal's
`allocation_samples <ControlSignal.allocation_samples>` attribute (specified in the **allocation_samples** argument
of the ControlSignal's constructor) to determine the allocation values to sample for that ControlSignal.  A
ControlSignal's `allocation <ControlSignal>` attribute reflects value assigned to it by the ControlMechanism
at the end of the previous `TRIAL` (i.e., when the ControlMechanism last executed --  see
`ControlMechanism Execution <ControlMechanism_Execution>`); its value from the previous `TRIAL` is assigned to the
`last_allocation` attribute.

*Function*. A ControlSignal's `allocation <ControlSignal.alloction>` serves as its`variable <ControlSignal.variable>`,
and is used by its `function <ControlSignal.function>` to generate an `intensity`. The default `function
<ControlSignal.function>` for a ControlSignal is an identity function (`Linear` with `slope <Linear.slope>` \\=1 and
`intercept <Linear.intercept>`\\=0), that simply assigns the `allocation <ControlSignal.allocation>` as the
ControlSignal's `intensity <ControlSignal.intensity>`. However, another `TransferFunction` can be assigned
(e.g., `Exponential`), or any other function that takes and returns a scalar value or 1d array.

*Intensity (value)*. The result of the function is assigned as the value of the ControlSignal's `intensity`
attribute, which serves as the ControlSignal's `value <ControlSignal.value>` (also referred to as `control_signal`).
The `intensity` is used by its `ControlProjection(s) <ControlProjection>` to modulate the parameter(s) for which the
ControlSignal is responsible. The ControlSignal's `intensity` attribute  reflects its value for the current `TRIAL`;
its value from the previous `TRIAL` is assigned to the `last_intensity` attribute.

.. _ControlSignal_Costs:

Costs and Cost Functions
~~~~~~~~~~~~~~~~~~~~~~~~

A ControlSignal has a `cost <ControlSignal.cost>` attribute that may be used by the ControlMechanism to which it
belongs to determine its future `allocation <ControlSignal.allocation>`.  The value of the `cost <ControlSignal.cost>`
is computed from the ControlSignal's `intensity` using one or more of three cost functions, each of which
computes a different component of the cost, and a function that combines them, as listed below:

    * `intensity_cost` - calculated by the `intensity_cost_function` based on the current `intensity` of the
      ControlSignal;
    ..
    * `adjustment_cost` - calculated by the `adjustment_cost_function` based on a change in the ControlSignal's
      `intensity` from its last value;
    ..
    * `duration_cost - calculated by the `duration_cost_function` based on an integral of the the ControlSignal's
      `cost <ControlSignal.cost>`;
    ..
    * `cost` - calculated by the `cost_combination_function` that combines the results of any cost functions that are
      enabled.

The components used to determine the ControlSignal's `cost <ControlSignal.cost>` can be specified in the
**costs_options** argument of its constructor, or using its `enable_costs`, `disable_costs` and `assign_costs`
methods.  All of these take one or more values of `ControlSignalCosts`, each of which specifies a cost component.
How the enabled components are combined is determined by the `cost_combination_function`.  By default, the values of
the enabled cost components are summed, however this can be modified by specifying the `cost_combination_function`.

    COMMENT:
    .. _ControlSignal_Toggle_Costs:

    *Enabling and Disabling Cost Functions*.  Any of the cost functions (except the `cost_combination_function`) can
    be enabled or disabled using the `toggle_cost_function` method to turn it `ON` or `OFF`. If it is disabled, that
    component of the cost is not included in the ControlSignal's `cost` attribute.  A cost function  can  also be
    permanently disabled for the ControlSignal by assigning it's attribute `None`.  If a cost function is permanently
    disabled for a ControlSignal, it cannot be re-enabled using `toggle_cost_function`.
    COMMENT

.. note:: The `index <OutputState.OutputState.index>` and `calculate <OutputState.OutputState.calculate>`
        attributes of a ControlSignal are automatically assigned and should not be modified.

.. _ControlSignal_Execution:

Execution
---------

A ControlSignal cannot be executed directly.  It is executed whenever the `ControlMechanism <ControlMechanism>` to
which it belongs is executed.  When this occurs, the ControlMechanism provides the ControlSignal with an `allocation
<ControlSignal.allocation>`, that is used by its `function <ControlSignal.function>` to compute its `intensity` for
that `TRIAL`.  The `intensity` is used by the ControlSignal's `ControlProjections <ControlProjection>` to set the
`value <ParameterState.value>` \\(s) of the `ParameterState(s) <ParameterState>` to which the ControlSignal projects.
Each ParameterState uses that value to modify the value(s) of the parameter(s) that the ControlSignal controls. See
`ModulatorySignal_Modulation` for a more detailed description of how modulation operates).  The ControlSignal's
`intensity` is also used  by its `cost functions <ControlSignal_Costs>` to compute its `cost` attribute. That is used
by some ControlMechanisms, along with the ControlSignal's `allocation_samples` attribute, to evaluate an
`allocation_policy <ControlMechanism.allocation_policy>`, and adjust the ControlSignal's `allocation
<ControlSignal.allocation>` for the next `TRIAL`.

.. note::
   The changes in a parameter in response to the execution of a ControlMechanism are not applied until the Mechanism
   with the parameter being controlled is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).

.. _ControlSignal_Examples:

Examples
~~~~~~~~

*Modulate the parameter of a Mechanism's function*.  The following example assigns a
ControlSignal to the `bias <Logistic.gain>` parameter of the `Logistic` Function used by a `TransferMechanism`::

    My_Mech = TransferMechanism(function=Logistic(bias=(1.0, ControlSignal)))

Note that the ControlSignal is specified by it class.  This will create a default ControlSignal,
with a ControlProjection that projects to the TransferMechanism's `ParameterState` for the `bias <Logistic.bias>`
parameter of its `Logistic` Function.  The default value of a ControlSignal's `modulation <ControlSignal.modulation>` attribute is Modulation.MULTIPLICATIVE, so that it will multiply the value of the `bias <Logistic.bias>` parameter. When the TransferMechanism executes, the Logistic Function will use the value of the ControlSignal as its
gain parameter.

*Specify attributes of a ControlSignal*.  Ordinarily, ControlSignals modify the *MULTIPLICATIVE_PARAM* of a
ParameterState's `function <ParameterState.function>` to modulate the parameter's value.
In the example below, this is changed by specifying the `modulation <ControlSignal.modulation>` attribute of a
`ControlSignal` for the `Logistic` Function of a `TransferMechanism`.  It is changed so that the value of the
ControlSignal adds to, rather than multiplies, the value of the `gain <Logistic.gain>` parameter of the Logistic
function::

    My_Mech = TransferMechanism(function=Logistic(gain=(1.0, ControlSignal(modulation=ModulationParam.ADDITIVE))))

Note that the `ModulationParam` specified for the `ControlSignal` pertains to the function of a *ParameterState*
for the *Logistic* Function (in this case, its `gain <Logistic.gain>` parameter), and *not* the Logistic function
itself -- that is, in this example, the value of the ControlSignal is added to the *gain parameter* of the Logistic
function, *not* its `variable <Logistic.variable>`).  If the value of the ControlSignal's **modulation** argument
had been ``ModulationParam.OVERRIDE``, then the ControlSignal's value would have been used as (i.e., replaced) the
value of the *Logistic* Function's `gain <Logistic.gain>` parameter, rather than added to it.

COMMENT:
    MOVE THIS EXAMPLE TO EVCControlMechanism

*Modulate the parameters of several Mechanisms by an EVCControlMechanism*.  This shows::

    My_Mech_A = TransferMechanism(function=Logistic)
    My_Mech_B = TransferMechanism(function=Linear,
                                 output_states=[RESULT, MEAN])

    Process_A = process(pathway=[My_Mech_A])
    Process_B = process(pathway=[My_Mech_B])
    My_System = system(processes=[Process_A, Process_B])

    My_EVC_Mechanism = EVCControlMechanism(system=My_System,
                                    monitor_for_control=[My_Mech_A.output_states[RESULT],
                                                         My_Mech_B.output_states[MEAN]],
                                    control_signals=[(GAIN, My_Mech_A),
                                                     {NAME: INTERCEPT,
                                                      MECHANISM: My_Mech_B,
                                                      MODULATION:ModulationParam.ADDITIVE}],
                                    name='My EVC Mechanism')
COMMENT

*Modulate the parameters of several Mechanisms in a System*.  The following example assigns ControlSignals to modulate
the `gain <Logistic.gain>` parameter of the `Logistic` function for ``My_Mech_A`` and the `intercept
<Logistic.intercept>` parameter of the `Linear` function for ``My_Mech_B``::

    My_Mech_A = TransferMechanism(function=Logistic)
    My_Mech_B = TransferMechanism(function=Linear,
                                 output_states=[RESULT, MEAN])
    Process_A = process(pathway=[My_Mech_A])
    Process_B = process(pathway=[My_Mech_B])

    My_System = system(processes=[Process_A, Process_B],
                                    monitor_for_control=[My_Mech_A.output_states[RESULT],
                                                         My_Mech_B.output_states[MEAN]],
                                    control_signals=[(GAIN, My_Mech_A),
                                                     {NAME: INTERCEPT,
                                                      MECHANISM: My_Mech_B,
                                                      MODULATION: ModulationParam.ADDITIVE}],
                       name='My Test System')


Class Reference
---------------

"""

import inspect
import warnings
from enum import IntEnum

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import InitStatus, function_type, method_type
# import Components
# FIX: EVCControlMechanism IS IMPORTED HERE TO DEAL WITH COST FUNCTIONS THAT ARE DEFINED IN EVCControlMechanism
#            SHOULD THEY BE LIMITED TO EVC??
from PsyNeuLink.Components.Functions.Function import CombinationFunction, Exponential, IntegratorFunction, Linear, \
    LinearCombination, Reduce, SimpleIntegrator, TransferFunction, _is_modulation_param, is_function_type
from PsyNeuLink.Components.ShellClasses import Function, Mechanism
from PsyNeuLink.Components.States.ModulatorySignals.ModulatorySignal import ModulatorySignal
from PsyNeuLink.Components.States.OutputState import PRIMARY_OUTPUT_STATE
from PsyNeuLink.Components.States.State import State_Base
from PsyNeuLink.Globals.Defaults import defaultControlAllocation
from PsyNeuLink.Globals.Keywords import ALLOCATION_SAMPLES, AUTO, CONTROLLED_PARAM, CONTROL_PROJECTION, EXECUTING, \
    FUNCTION, FUNCTION_PARAMS, INTERCEPT, OFF, ON, OUTPUT_STATES, OUTPUT_STATE_PARAMS, PROJECTION_TYPE, SEPARATOR_BAR, \
    SLOPE, SUM, kwAssign
from PsyNeuLink.Globals.Log import LogEntry, LogLevel
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import is_numeric, iscompatible, kwCompatibilityLength, kwCompatibilityNumeric, \
    kwCompatibilityType
from PsyNeuLink.Scheduling.TimeScale import CurrentTime, TimeScale

# class OutputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE

# # Default control allocation mode values:
# class DefaultControlAllocationMode(Enum):
#     GUMBY_MODE = 0.0
#     BADGER_MODE = 1.0
#     TEST_MODE = 240
# defaultControlAllocation = DefaultControlAllocationMode.BADGER_MODE.value
DEFAULT_ALLOCATION_SAMPLES = np.arange(0.1, 1.01, 0.3)

# -------------------------------------------    KEY WORDS  -------------------------------------------------------

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

COST_OPTIONS = 'cost_options'
class ControlSignalCosts(IntEnum):
    """Options for selecting `cost functions <ControlSignal_Costs>` to be used by a ControlSignal.

    These can be used alone or in combination with one another, by `enabling or disabling <_ControlSignal_Toggle_Costs>`
    each using the ControlSignal's `toggle_cost_function` method.

    Attributes
    ----------

    NONE
        ControlSignal's `cost` is not computed.

    INTENSITY_COST
        `intensity_cost_function` is used to calculate a contribution to the ControlSignal's `cost <ControlSignal.cost>`
        based its current `intensity` value.

    ADJUSTMENT_COST
        `adjustment_cost_function` is used to calculate a contribution to the `cost` based on the change in its
        `intensity` from its last value.

    DURATION_COST
        `duration_cost_function` is used to calculate a contribitution to the `cost` based on an integral of the
        ControlSignal's `cost <ControlSignal.cost>` (i.e., it accumulated value over multiple executions).

    ALL
        all of the `cost functions <ControlSignal_Costs> are used to calculate the ControlSignal's
        `cost <ControlSignal.cost>`.

    DEFAULTS
        assign default set of `cost functions <ControlSignal_Costs>` (currently set to `INTENSITY_COST`).

    """
    NONE               = 0
    INTENSITY_COST     = 1 << 1
    ADJUSTMENT_COST    = 1 << 2
    DURATION_COST      = 1 << 3
    ALL                = INTENSITY_COST | ADJUSTMENT_COST | DURATION_COST
    DEFAULTS           = INTENSITY_COST


class ControlSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


    def __str__(self):
        return repr(self.error_value)



class ControlSignal(ModulatorySignal):
    """
    ControlSignal(                                       \
        owner,                                           \
        function=LinearCombination(operation=SUM),       \
        costs_options=ControlSignalCosts.DEFAULTS,       \
        intensity_cost_function=Exponential,             \
        adjustment_cost_function=Linear,                 \
        duration_cost_function=Integrator,               \
        cost_combination_function=Reduce(operation=SUM), \
        allocation_samples=DEFAULT_ALLOCATION_SAMPLES,   \
        modulation=ModulationParam.MULTIPLICATIVE        \
        projections=None                                 \
        params=None,                                     \
        name=None,                                       \
        prefs=None)

    A subclass of `ModulatorySignal <ModulatorySignal>` used by a `ControlMechanism <ControlMechanism>` to
    modulate the parameter(s) of one or more other `Mechanisms <Mechanism>`.

    COMMENT:

        Description
        -----------
            The ControlSignal class is a subtype of the OutputState type in the State category of Component,
            It is used as the sender for ControlProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = CONTROL_SIGNAL
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS   (Operation.PRODUCT)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: Linear)

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    owner : ControlMechanism
        specifies the `ControlMechanism <ControlMechanism>` to which to assign the ControlSignal.

    function : Function or method : default Linear
        specifies the function used to determine the `intensity` of the ControlSignal from its `allocation`.

    cost_options : ControlSignalCosts or List[ControlSignalCosts] : ControlSignalsCosts.DEFAULTS
        specifies the cost components to include in the computation of the ControlSignal's `cost <ControlSignal.cost>`.

    intensity_cost_function : Optional[TransferFunction] : default Exponential
        specifies the function used to calculate the contribution of the ControlSignal's `intensity` to its
        `cost <ControlSignal.cost>`.

    adjustment_cost_function : Optional[TransferFunction] : default Linear
        specifies the function used to calculate the contribution of the change in the ControlSignal's `intensity`
        (from its `last_intensity` value) to its `cost <ControlSignal.cost>`.

    duration_cost_function : Optional[IntegratorFunction] : default Integrator
        specifies the function used to calculate the contribution of the ControlSignal's duration to its
        `cost <ControlSignal.cost>`.

    cost_combination_function : function : default `Reduce(operation=SUM) <Function.Reduce>`
        specifies the function used to combine the results of any cost functions that are enabled, the result of
        which is assigned as the ControlSignal's `cost <ControlSignal.cost>` attribute.

    allocation_samples : list : default range(0.1, 1, 0.1)
        specifies the values used by `ControlSignal's `ControlSignal.owner` to determine its
        `allocation_policy <ControlMechanism.allocation_policy>` (see `ControlSignal_Execution`).

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        specifies the way in which the `value <ControlSignal.value>` the ControlSignal is used to modify the value of
        the parameter(s) that it controls.

    projections : list of Projection specifications
        specifies the `ControlProjection(s) <ControlProjection>` to be assigned to the ControlSignal, and that will be
        listed in its `efferents <ControlSignal.efferents>` attribute (see `ControlSignal_Projections` for additional
        details).

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default OutputState-<index>
        a string used for the name of the OutputState.
        If not is specified, a default is assigned by the StateRegistry of the Mechanism to which the OutputState
        belongs (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the OutputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : ControlMechanism
        the `ControlMechanism <ControlMechanism>` to which the ControlSignal belongs.

    variable : number, list or np.ndarray
        same as `allocation <ControlSignal.allocation>`;  used by `function <ControlSignal.function>` to compute the
        ControlSignal's `intensity`.

    allocation : float : default: defaultControlAllocation
        value used as `variable <ControlSignal.variable>` for the ControlSignal's `function <ControlSignal.function>`
        to determine its `intensity`.

    last_allocation : float
        value of `allocation` in the previous execution of ControlSignal's `owner <ControlSignal.owner>`.

    allocation_samples : list : DEFAULT_SAMPLE_VALUES
        set of values to sample by the ControlSignal's `owner <ControlSignal.owner>` to determine its
        `allocation_policy <ControlMechanism.allocation_policy>`.

    function : TransferFunction :  default Linear(slope=1, intercept=0)
        converts `allocation` into the ControlSignal's `intensity`.  The default is the identity function, which
        assigns the ControlSignal's `allocation` as its `intensity`.

    value : number, list or np.ndarray
        result of the ControlSignal's `function <ControlSignal.function>`; same as `intensity` and `control_signal`.

    intensity : float
        result of the ControlSignal's `function <ControlSignal.function>`;
        assigned as the value of the ControlSignal's ControlProjection, and used to modify the value of the parameter
        to which the ControlSignal is assigned; same as `control_signal <ControlSignal.control_signal>`.

    last_intensity : float
        the `intensity` of the ControlSignal on the previous execution of its `owner <ControlSignal.owner>`.

    control_signal : float
        result of the ControlSignal's `function <ControlSignal.function>`; same as `intensity`.

    cost_options : int
        boolean combination of currently assigned ControlSignalCosts. Specified initially in **costs** argument of
        ControlSignal's constructor;  can be modified using the `assign_cost_options` method.

    intensity_cost_function : TransferFunction : default default Exponential
        calculates `intensity_cost` from the current value of `intensity`. It can be any `TransferFunction`, or any
        other function that takes and returns a scalar value. The default is `Exponential`.  It can be disabled
        permanently for the ControlSignal by assigning `None`.

    intensity_cost : float
        cost associated with the current `intensity`.

    adjustment_cost_function : TransferFunction : default Linear
        calculates `adjustment_cost` based on the change in `intensity` from  `last_intensity`.  It can be any
        `TransferFunction`, or any other function that takes and returns a scalar value. It can be disabled
        permanently for the ControlSignal by assigning `None`.

    adjustment_cost : float
        cost associated with last change to `intensity`.

    duration_cost_function : IntegratorFunction : default Linear
        calculates an integral of the ControlSignal's `cost`.  It can be any `IntegratorFunction`, or any other
        function that takes a list or array of two values and returns a scalar value. It can be disabled permanently
        for the ControlSignal by assigning `None`.

    duration_cost : float
        intregral of `cost`.

    cost_combination_function : function : default Reduce(operation=SUM)
        combines the results of all cost functions that are enabled, and assigns the result to `cost`.
        It can be any function that takes an array and returns a scalar value.

    cost : float
        combined result of all `cost functions <ControlSignal_Costs>` that are enabled.

    modulation : ModulationParam
        specifies the way in which the `value <ControlSignal.value>` the ControlSignal is used to modify the value of
        the parameter(s) that it controls.

    efferents : [List[ControlProjection]]
        a list of the `ControlProjections <ControlProjection>` assigned to (i.e., that project from) the ControlSignal.

    name : str : default <State subclass>-<index>
        name of the OutputState.
        Specified in the **name** argument of the constructor for the OutputState.  If not is specified, a default is
        assigned by the StateRegistry of the Mechanism to which the OutputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, state names are "scoped" within a Mechanism, meaning that states with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: states within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the OutputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = OUTPUT_STATES
    paramsType = OUTPUT_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_TYPE: CONTROL_PROJECTION,
        CONTROLLED_PARAM:None
    })
    #endregion


    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 variable=None,
                 size=None,
                 index=PRIMARY_OUTPUT_STATE,
                 calculate=Linear,
                 function=LinearCombination(operation=SUM),
                 cost_options:tc.any(ControlSignalCosts, list)=ControlSignalCosts.DEFAULTS,
                 intensity_cost_function:(is_function_type)=Exponential,
                 adjustment_cost_function:tc.optional(is_function_type)=Linear,
                 duration_cost_function:tc.optional(is_function_type)=SimpleIntegrator,
                 cost_combination_function:tc.optional(is_function_type)=Reduce(operation=SUM),
                 allocation_samples=DEFAULT_ALLOCATION_SAMPLES,
                 modulation:tc.optional(_is_modulation_param)=None,
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Note index and calculate are not used by ControlSignal, but included here for consistency with OutputState
        if params and ALLOCATION_SAMPLES in params and params[ALLOCATION_SAMPLES] is not None:
            allocation_samples =  params[ALLOCATION_SAMPLES]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  cost_options=cost_options,
                                                  intensity_cost_function=intensity_cost_function,
                                                  adjustment_cost_function=adjustment_cost_function,
                                                  duration_cost_function=duration_cost_function,
                                                  cost_combination_function=cost_combination_function,
                                                  allocation_samples=allocation_samples,
                                                  params=params)

        # FIX: 5/26/16
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_states here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputStates in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        super().__init__(owner=owner,
                         reference_value=reference_value,
                         variable=variable,
                         size=size,
                         modulation=modulation,
                         index=index,
                         calculate=calculate,
                         projections=projections,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate allocation_samples and control_signal cost functions

        Checks if:
        - cost functions are all appropriate
        - allocation_samples is a list with 2 numbers
        - all cost functions are references to valid ControlProjection costFunctions (listed in self.costFunctions)
        - IntensityFunction is identity function, in which case ignoreIntensityFunction flag is set (for efficiency)

        """

        # Validate cost functions in request_set
        #   This should be all of them if this is an initialization call;
        #   Otherwise, just those specified in assign_params
        for cost_function_name in [item for item in request_set if item in costFunctionNames]:
            cost_function = request_set[cost_function_name]

            # cost function assigned None: OK
            if not cost_function:
                continue

            # cost_function is Function class specification:
            #    instantiate it and test below
            if inspect.isclass(cost_function) and issubclass(cost_function, Function):
                cost_function = cost_function()

            # cost_function is Function object:
            #     COST_COMBINATION_FUNCTION must be CombinationFunction
            #     DURATION_COST_FUNCTION must be an IntegratorFunction
            #     others must be TransferFunction
            if isinstance(cost_function, Function):
                if cost_function_name == COST_COMBINATION_FUNCTION:
                    if not isinstance(cost_function, CombinationFunction):
                        raise ControlSignalError("Assignment of Function to {} ({}) must be a CombinationFunction".
                                                 format(COST_COMBINATION_FUNCTION, cost_function))
                elif cost_function_name == DURATION_COST_FUNCTION:
                    if not isinstance(cost_function, IntegratorFunction):
                        raise ControlSignalError("Assignment of Function to {} ({}) must be an IntegratorFunction".
                                                 format(DURATION_COST_FUNCTION, cost_function))
                elif not isinstance(cost_function, TransferFunction):
                    raise ControlSignalError("Assignment of Function to {} ({}) must be a TransferFunction".
                                             format(cost_function_name, cost_function))

            # cost_function is custom-specified function
            #     DURATION_COST_FUNCTION and COST_COMBINATION_FUNCTION must accept an array
            #     others must accept a scalar
            #     all must return a scalar
            elif isinstance(cost_function, (function_type, method_type)):
                if cost_function_name in COST_COMBINATION_FUNCTION:
                    test_value = [1, 1]
                else:
                    test_value = 1
                try:
                    result = cost_function(test_value)
                    if not (is_numeric(result) or is_numeric(np.asscalar(result))):
                        raise ControlSignalError("Function assigned to {} ({}) must return a scalar".
                                                 format(cost_function_name, cost_function))
                except:
                    raise ControlSignalError("Function assigned to {} ({}) must accept {}".
                                             format(cost_function_name, cost_function, type(test_value)))

            # Unrecognized function assignment
            else:
                raise ControlSignalError("Unrecognized function ({}) assigned to {}".
                                         format(cost_function, cost_function_name))

        # Validate allocation samples list:
        # - default is 1D np.array (defined by DEFAULT_ALLOCATION_SAMPLES)
        # - however, for convenience and compatibility, allow lists:
        #    check if it is a list of numbers, and if so convert to np.array
        if ALLOCATION_SAMPLES in request_set:
            allocation_samples = request_set[ALLOCATION_SAMPLES]
            if isinstance(allocation_samples, list):
                if iscompatible(allocation_samples, **{kwCompatibilityType: list,
                                                           kwCompatibilityNumeric: True,
                                                           kwCompatibilityLength: False,
                                                           }):
                    # Convert to np.array to be compatible with default value
                    request_set[ALLOCATION_SAMPLES] = np.array(allocation_samples)
            elif isinstance(allocation_samples, np.ndarray) and allocation_samples.ndim == 1:
                pass
            else:
                raise ControlSignalError("allocation_samples argument ({}) in {} must be "
                                             "a list or 1D np.array of numbers".
                                         format(allocation_samples, self.name))

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # ControlProjection Cost Functions
        for cost_function_name in [item for item in target_set if item in costFunctionNames]:
            cost_function = target_set[cost_function_name]
            if not cost_function:
                continue
            if ((not isinstance(cost_function, (Function, function_type, method_type)) and
                     not issubclass(cost_function, Function))):
                raise ControlSignalError("{0} not a valid Function".format(cost_function))

    def _instantiate_attributes_before_function(self, context=None):

        super()._instantiate_attributes_before_function(context=context)

        # Instantiate cost functions (if necessary) and assign to attributes
        for cost_function_name in costFunctionNames:
            cost_function = self.paramsCurrent[cost_function_name]
            # cost function assigned None
            if not cost_function:
                self.toggle_cost_function(cost_function_name, OFF)
                continue
            # cost_function is Function class specification
            if inspect.isclass(cost_function) and issubclass(cost_function, Function):
                cost_function = cost_function()
            # cost_function is Function object
            if isinstance(cost_function, Function):
                cost_function.owner = self
                cost_function = cost_function.function
            # cost_function is custom-specified function
            elif isinstance(cost_function, function_type):
                pass
            # safeguard/sanity check (should never happen if validation is working properly)
            else:
                raise ControlSignalError("{} is not a valid cost function for {}".
                                         format(cost_function, cost_function_name))

            self.paramsCurrent[cost_function_name] = cost_function

        # Assign instance attributes
        self.allocation_samples = self.paramsCurrent[ALLOCATION_SAMPLES]

        # Default intensity params
        self.default_allocation = defaultControlAllocation
        self.allocation = self.default_allocation  # Amount of control currently licensed to this signal
        self.last_allocation = self.allocation
        self.intensity = self.allocation

        # Default cost params
        self.intensity_cost = self.intensity_cost_function(self.intensity)
        self.adjustment_cost = 0
        self.duration_cost = 0
        self.last_duration_cost = self.duration_cost
        self.cost = self.intensity_cost
        self.last_cost = self.cost

        # If intensity function (self.function) is identity function, set ignoreIntensityFunction
        function = self.params[FUNCTION]
        function_params = self.params[FUNCTION_PARAMS]
        if ((isinstance(function, Linear) or (inspect.isclass(function) and issubclass(function, Linear)) and
                function_params[SLOPE] == 1 and
                function_params[INTERCEPT] == 0)):
            self.ignoreIntensityFunction = True
        else:
            self.ignoreIntensityFunction = False

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate calculate function
        """
        super()._instantiate_attributes_after_function(context=context)

        self.intensity = self.function(self.allocation)
        self.last_intensity = self.intensity

    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):
        """Adjust the control signal, based on the allocation value passed to it

        Computes new intensity and cost attributes from allocation

        Use self.function to assign intensity

            - if ignoreIntensityFunction is set (for efficiency, if the execute method it is the identity function):

                - ignore self.function
                - pass allocation (input to control_signal) along as its output
        Update cost.
        Assign intensity to value of ControlSignal (done in setter property for value)

        :parameter allocation: (single item list, [0-1])
        :return: (intensity)
        """


        # MODIFIED 4/15/17 OLD: [NOT SURE WHY, BUT THIS SKIPPED OutputState.update() WHICH CALLS self.calculate()
        # super(OutputState, self).update(params=params, time_scale=time_scale, context=context)
        # MODIFIED 4/15/17 NEW: [THIS GOES THROUGH OutputState.update() WHICH CALLS self.calculate()
        super().update(params=params, time_scale=time_scale, context=context)
        # MODIFIED 4/15/17 END

        # store previous state
        self.last_allocation = self.allocation
        self.last_intensity = self.intensity
        self.last_cost = self.cost
        self.last_duration_cost = self.duration_cost

        # update current intensity
        # FIX: INDEX MUST BE ASSIGNED WHEN OUTPUTSTATE IS CREATED FOR ControlMechanism (IN PLACE OF LIST OF PROJECTIONS)
        self.allocation = self.owner.value[self.index]
        # self.allocation = self.sender.value

        if self.ignoreIntensityFunction:
            # self.set_intensity(self.allocation)
            self.intensity = self.allocation
        else:
            self.intensity = self.function(self.allocation, params)
        intensity_change = self.intensity-self.last_intensity

        if self.prefs.verbosePref:
            intensity_change_string = "no change"
            if intensity_change < 0:
                intensity_change_string = str(intensity_change)
            elif intensity_change > 0:
                intensity_change_string = "+" + str(intensity_change)
            if self.prefs.verbosePref:
                warnings.warn("\nIntensity: {0} [{1}] (for allocation {2})".format(self.intensity,
                                                                                   intensity_change_string,
                                                                                   self.allocation))
                warnings.warn("[Intensity function {0}]".format(["ignored", "used"][self.ignoreIntensityFunction]))

        # compute cost(s)
        new_cost = intensity_cost = adjustment_cost = duration_cost = 0

        if self.cost_options & ControlSignalCosts.INTENSITY_COST:
            intensity_cost = self.intensity_cost = self.intensity_cost_function(self.intensity)
            if self.prefs.verbosePref:
                print("++ Used intensity cost")

        if self.cost_options & ControlSignalCosts.ADJUSTMENT_COST:
            adjustment_cost = self.adjustment_cost = self.adjustment_cost_function(intensity_change)
            if self.prefs.verbosePref:
                print("++ Used adjustment cost")

        if self.cost_options & ControlSignalCosts.DURATION_COST:
            duration_cost = self.duration_cost = self.duration_cost_function([self.last_duration_cost, new_cost])
            if self.prefs.verbosePref:
                print("++ Used duration cost")

        new_cost = self.cost_combination_function([float(intensity_cost), adjustment_cost, duration_cost])

        if new_cost < 0:
            new_cost = 0
        self.cost = new_cost


        # Report new values to stdio
        if self.prefs.verbosePref:
            cost_change = new_cost - self.last_cost
            cost_change_string = "no change"
            if cost_change < 0:
                cost_change_string = str(cost_change)
            elif cost_change > 0:
                cost_change_string = "+" + str(cost_change)
            print("Cost: {0} [{1}])".format(self.cost, cost_change_string))

        #region Record control_signal values in owner Mechanism's log
        # Notes:
        # * Log control_signals for ALL states of a given Mechanism in the Mechanism's log
        # * Log control_signals for EACH state in a separate entry of the Mechanism's log

        # Get receiver Mechanism and state
        controller = self.owner

        # Get logPref for Mechanism
        log_pref = controller.prefs.logPref

        # Get context
        if not context:
            context = controller.name + " " + self.name + kwAssign
        else:
            context = context + SEPARATOR_BAR + self.name + kwAssign

        # If context is consistent with log_pref:
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context))):
            # record info in log

# FIX: ENCODE ALL OF THIS AS 1D ARRAYS IN 2D PROJECTION VALUE, AND PASS TO .value FOR LOGGING
            controller.log.entries[self.name + " " +
                                      kpIntensity] = LogEntry(CurrentTime(), context, float(self.intensity))
            if not self.ignoreIntensityFunction:
                controller.log.entries[self.name + " " + kpAllocation] = LogEntry(CurrentTime(),
                                                                                  context,
                                                                                  float(self.allocation))
                controller.log.entries[self.name + " " + kpIntensityCost] =  LogEntry(CurrentTime(),
                                                                                      context,
                                                                                      float(self.intensity_cost))
                controller.log.entries[self.name + " " + kpAdjustmentCost] = LogEntry(CurrentTime(),
                                                                                      context,
                                                                                      float(self.adjustment_cost))
                controller.log.entries[self.name + " " + kpDurationCost] = LogEntry(CurrentTime(),
                                                                                    context,
                                                                                    float(self.duration_cost))
                controller.log.entries[self.name + " " + kpCost] = LogEntry(CurrentTime(),
                                                                            context,
                                                                            float(self.cost))
    #endregion



# MODIFIED 9/30/17 NEW:
    def _parse_state_specific_tuple(self, owner, state_specification_tuple):
        """Get ControlSignal specified for a parameter or in a 'control_signals' argument

        Tuple specification can be:
            (parameter_name, Mechanism)
            [TBI:] (parameter_name, Mechanism, weight, exponent, projection_specs)

        Returns params dict with CONNECTIONS entries if any of these was specified.

        """
        from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        from PsyNeuLink.Components.Projections.Projection import _parse_projection_specs
        from PsyNeuLink.Globals.Keywords import CONNECTIONS, PROJECTIONS

        params_dict = {}

        try:
            param_name, mech = state_specification_tuple
        except:
            raise ControlSignalError("Illegal {} specification tuple for {} ({});  "
                                     "it must contain two items: (<param_name>, <{}>)".
                                     format(ControlSignal.__name__, owner.name,
                                            state_specification_tuple, Mechanism.__name__))
        if not isinstance(mech, Mechanism):
            raise ControlSignalError("Second item of the {} specification tuple for {} ({}) must be a Mechanism".
                                     format(ControlSignal.__name__, owner.name, mech, mech.name))
        if not isinstance(param_name, str):
            raise ControlSignalError("First item of the {} specification tuple for {} ({}) must be a string "
                                     "that is the name of a parameter of its second item ({})".
                                     format(ControlSignal.__name__, owner.name, param_name, mech.name))
        try:
            parameter_state = mech._parameter_states[param_name]
        except KeyError:
            raise ControlSignalError("No {} found for {} param of {} in {} specificadtion tuple for {}".
                                     format(ParameterState.__name__, param_name, mech.name,
                                            ControlSignal.__name__, owner.name))
        except AttributeError:
            raise ControlSignalError("{} does not have any {} specified, so can't"
                                     "assign {} specified for {} ({})".
                                     format(mech.name, ParameterState.__name__, ControlSignal.__name__,
                                            owner.name, state_specification_tuple))

        # Assign connection specs to PROJECTIONS entry of params dict
        try:
            # params_dict[CONNECTIONS] = _parse_projection_specs(self.__class__,
            params_dict[PROJECTIONS] = _parse_projection_specs(self,
                                                               owner=owner,
                                                               connections=parameter_state)
        except ControlSignalError:
            raise ControlSignalError("Unable to parse {} specification dictionary for {} ({})".
                                        format(ControlSignal.__name__, owner.name, state_specification_tuple))

        return params_dict
# MODIFIED 9/30/17 END




    @property
    def allocation_samples(self):
        return self._allocation_samples

    @allocation_samples.setter
    def allocation_samples(self, samples):
        if isinstance(samples, (list, np.ndarray)):
            self._allocation_samples = list(samples)
            return
        if isinstance(samples, tuple):
            self._allocation_samples = samples
            sample_range = samples
        elif samples == AUTO:

            # (7/21/17 CW) Note that since the time of writing this "stub", the value of AUTO in Keywords.py has changed
            # from True to "auto" due to the addition of "auto" as a parameter for RecurrentTransferMechanisms! Just FYI

            # THIS IS A STUB, TO BE REPLACED BY AN ACTUAL COMPUTATION OF THE ALLOCATION RANGE
            raise ControlSignalError("AUTO not yet supported for {} param of ControlProjection; default will be used".
                                     format(ALLOCATION_SAMPLES))
        else:
            sample_range = DEFAULT_ALLOCATION_SAMPLES
        self._allocation_samples = []
        i = sample_range[0]
        while i < sample_range[1]:
            self._allocation_samples.append(i)
            i += sample_range[2]

    @property
    def intensity(self):
        # FIX: NEED TO DEAL WITH LOGGING HERE (AS PER @PROPERTY State.value)
        return self._intensity

    @intensity.setter
    def intensity(self, new_value):
        try:
            old_value = self._intensity
        except AttributeError:
            old_value = 0
        self._intensity = new_value
        # if len(self.observers[kpIntensity]):
        #     for observer in self.observers[kpIntensity]:
        #         observer.observe_value_at_keypath(kpIntensity, old_value, new_value)

    @property
    def control_signal(self):
        return self.value

    @tc.typecheck
    def assign_costs(self, costs:tc.any(ControlSignalCosts, list)):
        """assign_costs(costs)
        Assigns specified costs; all others are disabled.

        Arguments
        ---------
        costs: ControlSignalCost or List[ControlSignalCosts]
            cost or list of costs to be used;  all other will be disabled.
        Returns
        -------
        cost_options :  boolean combination of ControlSignalCosts
            current value of `cost_options`.

        """
        if isinstance(costs, ControlSignalCosts):
            costs = [costs]
        self.cost_options = ControlSignalCosts.NONE
        return self.enable_costs(costs)

    @tc.typecheck
    def enable_costs(self, costs:tc.any(ControlSignalCosts, list)):
        """enable_costs(costs)
        Enables specified costs; settings for all other costs are left intact.

        Arguments
        ---------
        costs: ControlSignalCost or List[ControlSignalCosts]
            cost or list of costs to be enabled, in addition to any that are already enabled.
        Returns
        -------
        cost_options :  boolean combination of ControlSignalCosts
            current value of `cost_options`.

        """
        if isinstance(costs, ControlSignalCosts):
            options = [costs]
        for cost in costs:
            self.cost_options |= cost
        return self.cost_options

    @tc.typecheck
    def disable_costs(self, costs:tc.any(ControlSignalCosts, list)):
        """disable_costs(costs)
        Disables specified costs; settings for all other costs are left intact.

        Arguments
        ---------
        costs: ControlSignalCost or List[ControlSignalCosts]
            cost or list of costs to be disabled.
        Returns
        -------
        cost_options :  boolean combination of ControlSignalCosts
            current value of `cost_options`.

        """
        if isinstance(costs, ControlSignalCosts):
            options = [costs]
        for cost in costs:
            self.cost_options &= ~cost
        return self.cost_options

    def get_cost_options(self):
        options = []
        if self.cost_options & ControlSignalCosts.INTENSITY_COST:
            options.append(INTENSITY_COST)
        if self.cost_options & ControlSignalCosts.ADJUSTMENT_COST:
            options.append(ADJUSTMENT_COST)
        if self.cost_options & ControlSignalCosts.DURATION_COST:
            options.append(DURATION_COST)
        return

    def toggle_cost_function(self, cost_function_name, assignment=ON):
        """Enables/disables use of a cost function.

        ``cost_function_name`` should be a keyword (list under :ref:`Structure <ControlProjection_Structure>`).
        """
        if cost_function_name == INTENSITY_COST_FUNCTION:
            cost_option = ControlSignalCosts.INTENSITY_COST
        elif cost_function_name == DURATION_COST_FUNCTION:
            cost_option = ControlSignalCosts.DURATION_COST
        elif cost_function_name == ADJUSTMENT_COST_FUNCTION:
            cost_option = ControlSignalCosts.ADJUSTMENT_COST
        elif cost_function_name == COST_COMBINATION_FUNCTION:
            raise ControlSignalError("{} cannot be disabled".format(COST_COMBINATION_FUNCTION))
        else:
            raise ControlSignalError("toggle_cost_function: unrecognized cost function: {}".format(cost_function_name))

        if assignment:
            if not self.paramsCurrent[cost_function_name]:
                raise ControlSignalError("Unable to toggle {} ON as function assignment is \'None\'".
                                         format(cost_function_name))
            self.cost_options |= cost_option
        else:
            self.cost_options &= ~cost_option

    # def set_intensity_cost(self, assignment=ON):
    #     if assignment:
    #         self.control_signal_cost_options |= ControlSignalCosts.INTENSITY_COST
    #     else:
    #         self.control_signal_cost_options &= ~ControlSignalCosts.INTENSITY_COST
    #
    # def set_adjustment_cost(self, assignment=ON):
    #     if assignment:
    #         self.control_signal_cost_options |= ControlSignalCosts.ADJUSTMENT_COST
    #     else:
    #         self.control_signal_cost_options &= ~ControlSignalCosts.ADJUSTMENT_COST
    #
    # def set_duration_cost(self, assignment=ON):
    #     if assignment:
    #         self.control_signal_cost_options |= ControlSignalCosts.DURATION_COST
    #     else:
    #         self.control_signal_cost_options &= ~ControlSignalCosts.DURATION_COST
    #
    def get_costs(self):
        """Return three-element list with the values of ``intensity_cost``, ``adjustment_cost`` and ``duration_cost``
        """
        return [self.intensity_cost, self.adjustment_cost, self.duration_cost]

    @property
    def variable(self):
        return self.allocation

    @variable.setter
    def variable(self, assignment):
        self.allocation = assignment

    @property
    def value(self):
        # In case the ControlSignal has not yet been assigned (and its value is INITIALIZING or DEFERRED_INITIALIZATION
        if self.init_status in {InitStatus.DEFERRED_INITIALIZATION, InitStatus.INITIALIZING}:
            return None
        else:
            # FIX: NEED TO DEAL WITH LOGGING HERE (AS PER @PROPERTY State.value)
            return self._intensity

    @value.setter
    def value(self, assignment):
        self._value = assignment

# FIX: Refactor to function like _parse_gating_signal_spec,
# FIX:     then combine the two into a single _parse_modulatory_signal_spec
def _parse_control_signal_spec(owner, control_signal_spec, context=None):
    """Take specifications for one or more parameters to be controlled and return ControlSignal specification dictionary

    THIS DOCUMENTATION IS TAKEN FROM _parse_gating_signal AS A TEMPLATE FOR HOW _parse_control_signal_spec
    SHOULD FUNCTION, BUT NOT ALL OF IT HAS BEEN IMPLEMENTED

    control_signal_spec can take any of the following forms:
        - an existing ControlSignal
        - an existing Parameter for a parameter of Mechanisms in self.system
        - a list of state specifications (see below)
        - a dictionary that contains either a:
            - single state specification:
                NAME:str - contains the name of a ParameterState belonging to MECHANISM
                MECHANISM:Mechanism - contains a reference to a Mechanism in self.system that owns NAME'd state
                <PARAM_KEYWORD>:<ControlSignal param value>
            - multiple state specification:
                NAME:str - used as name of ControlSignal
                STATES:List[tuple, dict] - each item must be state specification tuple or dict
                <PARAM_KEYWORD>:<ControlSignal param value>

    Each state specification must be a:
        - (str, Mechanism) tuple
        - {NAME:str, MECHANISM:Mechanism} dict
        where:
            str is the name of a ParameterState of the Mechanism,
            Mechanism is a reference to an existing Mechanism that belongs to self.system

    Checks for duplicate state specifications within state_spec or with any existing ControlSignal of the owner
        (i.e., states that will receive more than one ControlProjection from the owner)

    If state_spec is already a ControlSignal, it is returned (in the CONTROL_SIGNAL entry) along with its parsed
    elements

    Returns dictionary with the following entries:
        NAME:str - name of either the Parameter to be controlled (if there is only one) or the ControlSignal
        STATES:list - list of ParameterStates to be controlled
        CONTROL_SIGNAL:ControlSignal or None
        PARAMS:dict - params dict if any were included in the state_spec
    """
    from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
    from PsyNeuLink.Components.States.State import _parse_state_spec
    from PsyNeuLink.Components.States.ParameterState import ParameterState, _get_parameter_state
    from PsyNeuLink.Globals.Keywords import NAME, PARAMS, \
                                            CONTROL_SIGNAL, CONTROL_SIGNAL_SPECS, PARAMETER_STATE, \
                                            MECHANISM, PROJECTIONS, SENDER, RECEIVER

    mech = None
    param_name = None
    control_projection = None
    control_signal_params = None
    parameter_state = None
    control_signal = None

    control_signal_dict = _parse_state_spec(owner=owner,
                                            state_type=ControlSignal,
                                            state_spec=control_signal_spec,
                                            context=context)

    # Specification is a ParameterState
    if isinstance(control_signal_dict, ParameterState):
        mech = control_signal_spec.owner
        param_name = control_signal_spec.name
        parameter_state = _get_parameter_state(owner, CONTROL_SIGNAL, param_name, mech)

    # Specification was tuple or dict, now parsed into a dict
    elif isinstance(control_signal_dict, dict):
        param_name = control_signal_dict[NAME]
        control_signal_params = control_signal_dict[PARAMS]

        # control_signal was a specification dict, with MECHANISM as an entry (and parameter as NAME)
        if control_signal_params and MECHANISM in control_signal_params:
            mech = control_signal_params[MECHANISM]
            # Delete MECHANISM entry as it is not a parameter of ControlSignal
            #     (which will balk at it in ControlSignal._validate_params)
            del control_signal_params[MECHANISM]
            parameter_state = _get_parameter_state(owner, CONTROL_SIGNAL, param_name, mech)

        # Specification was originally a tuple, either in parameter specification or control_signal arg;
        #    1st item was either assigned to the NAME entry of the control_signal_spec dict
        #        (if tuple was a (param_name, Mechanism tuple) for control_signal arg;
        #        or used as param value, if it was a parameter specification tuple
        #    2nd item was placed in CONTROL_SIGNAL_PARAMS entry of params dict in control_signal_spec dict,
        #        so parse:
        elif (control_signal_params and
                any(kw in control_signal_dict[PARAMS] for kw in {CONTROL_SIGNAL_SPECS, PROJECTIONS})):

            # IMPLEMENTATION NOTE:
            #    CONTROL_SIGNAL_SPECS is used by _parse_control_signal_spec,
            #                         to pass specification from a parameter specification tuple
            #    PROJECTIONS is used by _parse_state_spec to place the 2nd item of any tuple in params dict;
            #                      here, the tuple comes from a (param, Mechanism) specification in control_signal arg
            #    Delete whichever one it was, as neither is a recognized ControlSignal param
            #        (which will balk at it in ControlSignal._validate_params)
            if CONTROL_SIGNAL_SPECS in control_signal_dict[PARAMS]:
                spec = control_signal_params[CONTROL_SIGNAL_SPECS]
                del control_signal_params[CONTROL_SIGNAL_SPECS]
            elif PROJECTIONS in control_signal_dict[PARAMS]:
                spec = control_signal_params[PROJECTIONS]
                del control_signal_params[PROJECTIONS]

            # ControlSignal
            if isinstance(spec, ControlSignal):
                # Note: don't specify mech since ControlSignal could project to more than one ParameterState
                control_signal_dict = spec

            else:
                # Mechanism
                # IMPLEMENTATION NOTE: Mechanism was placed in list in PROJECTIONS entry by _parse_state_spec
                if isinstance(spec, list) and isinstance(spec[0], Mechanism):
                    mech = spec[0]
                    parameter_state = _get_parameter_state(owner, CONTROL_SIGNAL, param_name, mech)

                # Projection (in a list)
                elif isinstance(spec, list):
                    control_projection = spec[0]
                    if not isinstance(control_projection, ControlProjection):
                        raise ControlSignalError("PROGRAM ERROR: list in {} entry of params dict for {} of {} "
                                                    "must contain a single ControlProjection".
                                                    format(CONTROL_SIGNAL_SPECS, CONTROL_SIGNAL, owner.name))
                    if len(spec)>1:
                        raise ControlSignalError("PROGRAM ERROR: Multiple ControlProjections are not "
                                                    "currently supported in specification of a ControlSignal")
                    # Get receiver mech
                    if control_projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
                        parameter_state = control_projection.init_args[RECEIVER]
                        # ControlProjection was created in response to specification of ControlSignal
                        #     (in a 2-item tuple where the parameter was specified),
                        #     so get ControlSignal spec
                        if SENDER in control_projection.init_args:
                            control_signal_spec = control_projection.init_args[SENDER]
                            if control_signal_spec and not isinstance(control_signal_spec, ControlSignal):
                                raise ControlSignalError("PROGRAM ERROR: "
                                                            "Sender of {} for {} {} of {} is not a {}".
                                                            format(CONTROL_PROJECTION,
                                                                   parameter_state.name,
                                                                   PARAMETER_STATE,
                                                                   parameter_state.owner.name,
                                                                   CONTROL_SIGNAL))
                    else:
                        parameter_state = control_projection.receiver
                    param_name = parameter_state.name
                    mech = parameter_state.owner

                else:
                    raise ControlSignalError("PROGRAM ERROR: failure to parse specification of {} for {}".
                                                format(CONTROL_SIGNAL, owner.name))
        else:
            raise ControlSignalError("PROGRAM ERROR: No entry found in params dict with specification of "
                                        "parameter's Mechanism or ControlProjection for {} of {}".
                                        format(CONTROL_SIGNAL, owner.name))

        if isinstance(control_signal_spec, ControlSignal):
            control_signal = control_signal_spec
        else:
            control_signal = None

    return {MECHANISM: mech,
            NAME: param_name,
            PARAMS: control_signal_params,
            PARAMETER_STATE: parameter_state,
            CONTROL_PROJECTION: control_projection,
            CONTROL_SIGNAL: control_signal}

