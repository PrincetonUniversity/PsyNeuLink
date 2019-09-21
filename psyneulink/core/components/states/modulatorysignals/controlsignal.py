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
adapt the ControlSignal's `allocation <ControlSignal.allocation>` in subsequent executions.

.. _ControlSignal_Creation:

Creating a ControlSignal
------------------------

A ControlSignal is created automatically whenever the parameter of a Mechanism or of its function is `specified for
control <ControlMechanism_Control_Signals>`.  ControlSignals can also be specified in the **control_signals** argument
of the constructor for a `ControlMechanism <ControlMechanism>`, as well as in the `specification of the parameter
<ParameterState_Specification>` that the ControlSignal is intended to modulate (also see `Modualory Specificadtion
<ParameterState_Modulatory_Specification>`.  Although a ControlSignal can also be  created on its own using its
constructor (or any of the other ways for `creating an OutputState <OutputStates_Creation>`), this is usually not
necessary nor is it advisable, as a ControlSignal has dedicated components and requirements for configuration that
must be met for it to function properly.

.. _ControlSignal_Specification:

*Specifying ControlSignals*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ControlSignal is specified in the **control_signals** argument of the constructor for a `ControlMechanism
<ControlMechanism>`, the parameter(s) to be controlled must be specified.  If other attributes of the ControlSignal
need to be specified (e.g., one or more of its `cost functions <ControlSignal_Costs>`), then the Constructor for the
ControlSignal can be used or a `state specification dictionary <State_Specification>`, in which the parameter(s) to be
controlled in the **projections** argument or *PROJECTIONS* entry, respectively, using any of the forms below.
For convenience, the parameters can also be specified on their own in the **control_signals** argument of the
ControlMechanism's constructor, in which case a default ControlSignal will be created for each.  In all cases, any
of the following can be use to specify the parameter(s) to be controlled:

  * **ParameterState** (or list of them) -- for the Mechanism(s) to which the parameter(s) belong;
  ..
  * **2-item tuple:** *(parameter name or list of them>, <Mechanism>)* -- the 1st item must be the name of the
    parameter (or list of parameter names), and the 2nd item the Mechanism to which it (they) belong(s); this is a
    convenience format, that is simpler to use than a specification dictionary (see above), but precludes
    specification of any `parameters <ControlSignal_Structure>` for the ControlSignal.

  * **specification dictionary** -- this is an abbreviated form of `state specification dictionary
    <State_Specification>`, in which the parameter(s) to be controlled can be specified in either of the two
    following ways:

    * for controlling a single parameter, the dictionary can have the following two entries:

        * *NAME*: str
            the string must be the name of the parameter to be controlled;

        * *MECHANISM*: Mechanism
            the Mechanism must be the one to the which the parameter to be controlled belongs.

    * for controlling multiple parameters, the dictionary can have the following entry:

        * <str>:list
            the string used as the key specifies the name to be used for the ControlSignal,
            and each item of the list must be a `specification of a parameter <ParameterState_Specification>` to be
            controlled by the ControlSignal (and that will receive a `ControlProjection` from it).
  ..

.. _ControlSignal_Structure:

Structure
---------

A ControlSignal is owned by an `ControlMechanism <ControlMechanism>`, and controls the parameters of one or more
Components by modulating the `function <ParameterState.function>` of the `ParameterState` that determines the value
of each of the parameters that it control.  Its operation is governed by several attributes of the ControlSignal,
that are described below.

.. _ControlSignal_Projections:

*Projections*
~~~~~~~~~~~~~

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

*Modulation*
~~~~~~~~~~~~

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

*Allocation, Function and Intensity*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Allocation (variable)*. A ControlSignal is assigned an `allocation <ControlSignal>` by the ControlMechanism to which
it belongs. Some ControlMechanisms sample different allocation values for their ControlSignals to determine which to
use (e.g., `OptimizationControlMechanism`);  in those cases, they use each ControlSignal's
`allocation_samples <ControlSignal.allocation_samples>` attribute (specified in the **allocation_samples** argument
of the ControlSignal's constructor) to determine the allocation values to sample for that ControlSignal.  A
ControlSignal's `allocation <ControlSignal>` attribute contains the value assigned to it by the ControlMechanism
at the end of the previous `TRIAL` (i.e., when the ControlMechanism last executed --  see
`ControlMechanism Execution <ControlMechanism_Execution>`); its value from the previous `TRIAL` is assigned to the
`last_intensity` attribute.

FIX: 8/30/19 -- ADD DESCRIPTION OF function AS ACTUALLY IMPLEMENTED AS TransferWithCosts
                AND MODIFY DOCUMENTATION OF COSTS AND COST FUNCTIONS BELOW, AND THEIR Attributes ENTRIES ACCORDINGLY:
                - cost functions can be specified, but attributes are pointers to function's cost functions
                - cost attributes get value of corresponding attributes of cost function
                - ?handling of cost_options
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

*Costs and Cost Functions*
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlSignal has a `cost <ControlSignal.cost>` attribute that may be used by the ControlMechanism to which it
belongs to determine its `allocation <ControlSignal.allocation>`.  The value of the `cost <ControlSignal.cost>`
is computed from the ControlSignal's `intensity` using one or more of three cost functions, each of which
computes a different component of the cost, and a function that combines them, as listed below:

    * `intensity_cost` - calculated by the `intensity_cost_function` based on the current `intensity` of the
      ControlSignal;
    ..
    * `adjustment_cost` - calculated by the `adjustment_cost_function` based on a change in the ControlSignal's
      `intensity` from its last value;
    ..
    * `duration_cost` - calculated by the `duration_cost_function` based on an integral of the ControlSignal's
      `cost <ControlSignal.cost>`;
    ..
    * `cost` - calculated by the `combine_costs_function` that combines the results of any cost functions that are
      enabled.

The components used to determine the ControlSignal's `cost <ControlSignal.cost>` can be specified in the
**costs_options** argument of its constructor, or using its `enable_costs`, `disable_costs` and `assign_costs`
methods.  All of these take one or more values of `CostFunctions`, each of which specifies a cost component.
How the enabled components are combined is determined by the `combine_costs_function`.  By default, the values of
the enabled cost components are summed, however this can be modified by specifying the `combine_costs_function`.

    COMMENT:
    .. _ControlSignal_Toggle_Costs:

    *Enabling and Disabling Cost Functions*.  Any of the cost functions (except the `combine_costs_function`) can
    be enabled or disabled using the `toggle_cost_function` method to turn it `ON` or `OFF`. If it is disabled, that
    component of the cost is not included in the ControlSignal's `cost` attribute.  A cost function  can  also be
    permanently disabled for the ControlSignal by assigning it's attribute `None`.  If a cost function is permanently
    disabled for a ControlSignal, it cannot be re-enabled using `toggle_cost_function`.
    COMMENT

.. note:: The `index <OutputState.OutputState.index>` and `assign <OutputState.OutputState.assign>`
        attributes of a ControlSignal are automatically assigned and should not be modified.

.. _ControlSignal_Execution:

Execution
---------

A ControlSignal cannot be executed directly.  It is executed whenever the `ControlMechanism <ControlMechanism>` to
which it belongs is executed.  When this occurs, the ControlMechanism provides the ControlSignal with an `allocation
<ControlSignal.allocation>`, that is used by its `function <ControlSignal.function>` to compute its `intensity` for
that `TRIAL`.  The `intensity` is used by the ControlSignal's `ControlProjections <ControlProjection>` to set the
`value <ParameterState.value>` \\(s) of the `ParameterState(s) <ParameterState>` to which the ControlSignal projects.

Recall that the ParameterState value is referenced anywhere that the controlled parameter is used in computation, and
that it does not update until the component to which the ParameterState belongs executes. If the distinction between the
base value stored in the parameter attribute (i.e. MyTransferMech.function.gain) and the value of the
ParameterState is unfamiliar, see `Parameter State documentation <ParameterState>` for more details, or see
`ModulatorySignal_Modulation` for a detailed description of how modulation operates.

The ControlSignal's `intensity` is also used  by its `cost functions <ControlSignal_Costs>` to compute its `cost`
attribute. That is used by some ControlMechanisms, along with the ControlSignal's `allocation_samples` attribute, to
evaluate a `control_allocation <ControlMechanism.control_allocation>`, and adjust the ControlSignal's `allocation
<ControlSignal.allocation>` for the next `TRIAL`.

.. note::
   The changes in a parameter in response to the execution of a ControlMechanism are not applied until the Mechanism
   with the parameter being controlled is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of
   "lazy" updating).

.. _ControlSignal_Examples:

Examples
--------

*Modulate the parameter of a Mechanism's function*.  The following example assigns a
ControlSignal to the `bias <Logistic.gain>` parameter of the `Logistic` Function used by a `TransferMechanism`::

    >>> from psyneulink import *
    >>> my_mech = TransferMechanism(function=Logistic(bias=(1.0, ControlSignal)))

Note that the ControlSignal is specified by it class.  This will create a default ControlSignal,
with a ControlProjection that projects to the TransferMechanism's `ParameterState` for the `bias <Logistic.bias>`
parameter of its `Logistic` Function.  The default value of a ControlSignal's `modulation <ControlSignal.modulation>`
attribute is *MULTIPLICATIVE*, so that it will multiply the value of the `bias <Logistic.bias>` parameter.
When the TransferMechanism executes, the Logistic Function will use the value of the ControlSignal as its
bias parameter.

*Specify attributes of a ControlSignal*.  Ordinarily, ControlSignals modify the *MULTIPLICATIVE_PARAM* of a
ParameterState's `function <ParameterState.function>` to modulate the parameter's value.
In the example below, this is changed by specifying the `modulation <ControlSignal.modulation>` attribute of a
`ControlSignal` for the `Logistic` Function of the `TransferMechanism`.  It is changed so that the value of the
ControlSignal adds to, rather than multiplies, the value of the `gain <Logistic.gain>` parameter of the Logistic
function::

    >>> my_mech = TransferMechanism(function=Logistic(gain=(1.0, ControlSignal(modulation=ADDITIVE))))

Note that the `ModulationParam` specified for the `ControlSignal` refers to how the parameter of the *Logistic*
Function (in this case, its `gain <Logistic.gain>` parameter) is modified, and not directly to input Logistic function;
that is, in this example, the value of the ControlSignal is added to the *gain parameter* of the Logistic
function, *not* its `variable <Logistic.variable>`.  If the value of the ControlSignal's **modulation** argument
had been ``ModulationParam.OVERRIDE``, then the ControlSignal's value would have been used as (i.e., it would have
replaced) the value of the *Logistic* Function's `gain <Logistic.gain>` parameter, rather than added to it.

COMMENT:
    MOVE THIS EXAMPLE TO EVCControlMechanism

*Modulate the parameters of several Mechanisms by an OptimizationControlMechanism*.  This shows::

    my_mech_A = TransferMechanism(function=Logistic)
    my_mech_B = TransferMechanism(function=Linear,
                                 output_states=[RESULT, OUTPUT_MEAN])

    my_ocm = OptimizationControlMechanism(monitor_for_control=[my_mech_A.output_states[RESULT],
                                                               my_mech_B.output_states[OUTPUT_MEAN]],
                                          control_signals=[(GAIN, my_mech_A),
                                                           {NAME: INTERCEPT,
                                                            MECHANISM: my_mech_B,
                                                            MODULATION:ADDITIVE}],
                                          name='my_ocm')

*Modulate the parameters of several Mechanisms in a System*.  The following example assigns ControlSignals to modulate
the `gain <Logistic.gain>` parameter of the `Logistic` function for ``my_mech_A`` and the `intercept
<Logistic.intercept>` parameter of the `Linear` function for ``my_mech_B``::

    >>> my_mech_A = TransferMechanism(function=Logistic)
    >>> my_mech_B = TransferMechanism(function=Linear,
    ...                                   output_states=[RESULT, OUTPUT_MEAN])

    >>> process_a = Process(pathway=[my_mech_A])
    >>> process_b = Process(pathway=[my_mech_B])

    >>> my_system = System(processes=[process_a, process_b],
    ...                        monitor_for_control=[my_mech_A.output_states[RESULTS],
    ...                                             my_mech_B.output_states[OUTPUT_MEAN]],
    ...                        control_signals=[(GAIN, my_mech_A),
    ...                                         {NAME: INTERCEPT,
    ...                                          MECHANISM: my_mech_B,
    ...                                          MODULATION: ADDITIVE}],
    ...                        name='My Test System')

COMMENT


Class Reference
---------------

"""

import inspect
import itertools
import warnings

from enum import IntEnum

import numpy as np
import typecheck as tc

# FIX: EVCControlMechanism IS IMPORTED HERE TO DEAL WITH COST FUNCTIONS THAT ARE DEFINED IN EVCControlMechanism
#            SHOULD THEY BE LIMITED TO EVC??
from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import SimpleIntegrator
from psyneulink.core.components.functions.transferfunctions import Exponential, Linear, CostFunctions
from psyneulink.core.components.states.modulatorysignals.modulatorysignal import ModulatorySignal
from psyneulink.core.components.states.outputstate import SEQUENTIAL, _output_state_variable_getter
from psyneulink.core.components.states.state import State_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    ALLOCATION_SAMPLES, CONTROLLED_PARAMS, CONTROL_PROJECTION, CONTROL_SIGNAL, \
    INPUT_STATE, INPUT_STATES, \
    OUTPUT_STATE, OUTPUT_STATES, OUTPUT_STATE_PARAMS, \
    PARAMETER_STATE, PARAMETER_STATES, \
    PROJECTION_TYPE, RECEIVER, SUM
from psyneulink.core.globals.parameters import Parameter, get_validator_by_function, get_validator_by_type_only
from psyneulink.core.globals.sampleiterator import is_sample_spec
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import \
    is_numeric, iscompatible, kwCompatibilityLength, kwCompatibilityNumeric, kwCompatibilityType
from psyneulink.core.globals.sampleiterator import SampleSpec, SampleIterator

__all__ = ['ControlSignal', 'ControlSignalError', 'COST_OPTIONS']

# class OutputStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


# -------------------------------------------    KEY WORDS  -------------------------------------------------------


# class CostFunctions(IntEnum):
#     """Options for selecting `cost functions <ControlSignal_Costs>` to be used by a ControlSignal.
#
#     These can be used alone or in combination with one another, by `enabling or disabling <ControlSignal_Toggle_Costs>`
#     each using the ControlSignal's `toggle_cost_function` method.
#
#     Attributes
#     ----------
#
#     NONE
#         ControlSignal's `cost` is not computed.
#
#     INTENSITY
#         `intensity_cost_function` is used to calculate a contribution to the ControlSignal's `cost
#         <ControlSignal.cost>` based its current `intensity` value.
#
#     ADJUSTMENT
#         `adjustment_cost_function` is used to calculate a contribution to the `cost` based on the change in its
#         `intensity` from its last value.
#
#     DURATION
#         `duration_cost_function` is used to calculate a contribitution to the `cost` based on an integral of the
#         ControlSignal's `cost <ControlSignal.cost>` (i.e., it accumulated value over multiple executions).
#
#     ALL
#         all of the `cost functions <ControlSignal_Costs> are used to calculate the ControlSignal's
#         `cost <ControlSignal.cost>`.
#
#     DEFAULTS
#         assign default set of `cost functions <ControlSignal_Costs>` as `INTENSITY`).
#
#     """
#     NONE          = 0
#     INTENSITY     = 1 << 1
#     ADJUSTMENT    = 1 << 2
#     DURATION      = 1 << 3
#     ALL           = INTENSITY | ADJUSTMENT | DURATION
#     DEFAULTS      = INTENSITY

# Getters for cost attributes (from TransferWithCosts function)

from psyneulink.core.components.functions.transferfunctions import \
    ENABLED_COST_FUNCTIONS, \
    INTENSITY_COST, INTENSITY_COST_FUNCTION, ADJUSTMENT_COST, ADJUSTMENT_COST_FUNCTION, \
    DURATION_COST, DURATION_COST_FUNCTION, COMBINED_COSTS, COMBINE_COSTS_FUNCTION, costFunctionNames

COST_OPTIONS = 'cost_options'


# # FIX: DOESN'T WORK SINCE DON'T HAVE ACCESS TO OTHER ARGS
# def _function_parser(function):
#     from psyneulink.core.components.functions.transferfunctions import TransferWithCosts
#     return TransferWithCosts(default_variable=self.defaults.variable,
#                              transfer_fct=function,
#                              enabled_cost_functions=cost_options,
#                              intensity_cost_fct=intensity_cost_function,
#                              adjustment_cost_fct=adjustment_cost_function,
#                              duration_cost_fct=duration_cost_function,
#                              combine_costs_fct=combine_costs_function)


def _cost_options_getter(owning_component=None, context=None):
    try:
        return getattr(owning_component.function.parameters, ENABLED_COST_FUNCTIONS)._get(context)
    except (TypeError, IndexError):
        return None


def _cost_options_setter(value, owning_component=None, context=None):
    if hasattr(owning_component, "function") and owning_component.function:
        if hasattr(owning_component.function.parameters, ENABLED_COST_FUNCTIONS):
            getattr(owning_component.function.parameters, ENABLED_COST_FUNCTIONS)._set(value, context)
    return value


def _intensity_cost_getter(owning_component=None, context=None):
    try:
        return getattr(owning_component.function.parameters, INTENSITY_COST)._get(context)
    except (TypeError, IndexError):
        return None


def _adjustment_cost_getter(owning_component=None, context=None):
    try:
        return getattr(owning_component.function.parameters, ADJUSTMENT_COST)._get(context)
    except (TypeError, IndexError):
        return None


def _duration_cost_getter(owning_component=None, context=None):
    try:
        return getattr(owning_component.function.parameters, DURATION_COST)._get(context)
    except (TypeError, IndexError):
        return None


def _cost_getter(owning_component=None, context=None):
    try:
        return getattr(owning_component.function.parameters, COMBINED_COSTS)._get(context)
    except (TypeError, IndexError):
        return None


class ControlSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


    def __str__(self):
        return repr(self.error_value)


class ControlSignal(ModulatorySignal):
    """
    ControlSignal(                                                 \
        owner,                                                     \
        default_allocation=defaultControlAllocation,               \
        index=SEQUENTIAL,                                          \
        function=TransferWithCosts,                                \
        costs_options=None,                                        \
        intensity_cost_function=Exponential,                       \
        adjustment_cost_function=Linear,                           \
        duration_cost_function=IntegratorFunction,                 \
        combine_costs_function=Reduce(operation=SUM),              \
        allocation_samples=self.class_defaults.allocation_samples, \
        modulation=MULTIPLICATIVE                                  \
        projections=None                                           \
        params=None,                                               \
        name=None,                                                 \
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
                + FUNCTION (Linear)
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

    default_allocation : scalar, list or np.ndarray : defaultControlAllocation
        specifies the template and default value used for `allocation <ControlSignal.allocation>`;  must match the
        shape of each item specified in `allocation_samples <ControlSignal.allocation_samples>`.

    index : int : default SEQUENTIAL
        specifies the item of the owner ControlMechanism's `control_allocation <ControlMechanism.control_allocation>`
        used as the ControlSignal's `value <ControlSignal.value>`.

    function : Function or method : default TransferWithCosts(transfer_fct=Linear(slope=1, intercept=0))
        specifies the function used to determine the `intensity` of the ControlSignal from its `allocation`;
        must be TransferWithCosts or a subclass of that, or one that meets the requirements described see `above
        <ControlSignal_Function>`);  see `function <ControlSignal.function>` for default behavior.

    cost_options : CostFunctions or List[CostsFunctions] : None
        specifies the cost components to include in the computation of the ControlSignal's `cost <ControlSignal.cost>`.

    intensity_cost_function : Optional[TransferFunction] : default Exponential
        specifies the function used to calculate the contribution of the ControlSignal's `intensity` to its
        `cost <ControlSignal.cost>`.

    adjustment_cost_function : Optional[TransferFunction] : default Linear
        specifies the function used to calculate the contribution of the change in the ControlSignal's `intensity`
        (from its `last_intensity` value) to its `cost <ControlSignal.cost>`.

    duration_cost_function : IntegratorFunction : default IntegratorFunction
        specifies the function used to calculate the contribution of the ControlSignal's duration to its
        `cost <ControlSignal.cost>`.

    combine_costs_function : function : default `Reduce(operation=SUM) <Function.Reduce>`
        specifies the function used to combine the results of any cost functions that are enabled, the result of
        which is assigned as the ControlSignal's `cost <ControlSignal.cost>` attribute.

    allocation_samples : list, 1d array, or SampleSpec : default SampleSpec(0.1, 1, 0.1)
        specifies the values used by the ControlSignal's `owner <ControlSignal.owner>` to determine its
        `control_allocation <ControlMechanism.control_allocation>` (see `ControlSignal_Execution`).

    modulation : ModulationParam : default ModulationParam.MULTIPLICATIVE
        specifies the way in which the `value <ControlSignal.value>` the ControlSignal is used to modify the value of
        the parameter(s) that it controls.

    modulates : list of Projection specifications
        specifies the `ControlProjection(s) <ControlProjection>` to be assigned to the ControlSignal, and that will be
        listed in its `efferents <ControlSignal.efferents>` attribute (see `ControlSignal_Projections` for additional
        details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default see ModulatorySignal `name <ModulatorySignal.name>`
        specifies the name of the ControlSignal; see ControlSignal `name <ModulatorySignal.name>` for additional
        details.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the ControlSignal; see `prefs <ControlSignal.prefs>` for details.


    Attributes
    ----------

    owner : ControlMechanism
        the `ControlMechanism <ControlMechanism>` to which the ControlSignal belongs.

    variable : scalar, list or np.ndarray
        same as `allocation <ControlSignal.allocation>`.

    allocation : float : default: defaultControlAllocation
        value assigned by the ControlSignal's `owner <ControlSignal.owner>`, and used as the `variable
        <ControlSignal.variable>` of its `function <ControlSignal.function>` to determine the ControlSignal's
        `ControlSignal.intensity`.
    COMMENT:
    FOR DEVELOPERS:  Implemented as an alias of the ControlSignal's variable Parameter
    COMMENT

    last_allocation : float
        value of `allocation` in the previous execution of ControlSignal's `owner <ControlSignal.owner>`.

    allocation_samples : SampleIterator
        `SampleIterator` created from **allocation_samples** specification and used to generate a set of values to
        sample by the ControlSignal's `owner <ControlSignal.owner>` when determining its `control_allocation
        <ControlMechanism.control_allocation>`.

    function : TransferWithCosts
        converts `allocation` into the ControlSignal's `intensity`.  The default is the identity function, which
        assigns the ControlSignal's `allocation` as its `intensity`, and does not calculate any costs.  See
        `ControlSignals_Function` for additional details.

    value : float
        result of the ControlSignal's `function <ControlSignal.function>`; same as `intensity` and `control_signal`.

    intensity : float
        result of the ControlSignal's `function <ControlSignal.function>`;
        assigned as the value of the ControlSignal's ControlProjection, and used to modify the value of the parameter
        to which the ControlSignal is assigned; same as `control_signal <ControlSignal.control_signal>`.

    last_intensity : float
        the `intensity` of the ControlSignal on the previous execution of its `owner <ControlSignal.owner>`.

    index : int
        the item of the owner ControlMechanism's `control_allocation <ControlMechanism.control_allocation>` used as the
        ControlSignal's `value <ControlSignal.value>`.

    control_signal : float
        result of the ControlSignal's `function <ControlSignal.function>`; same as `intensity`.

    cost_options : CostFunctions or None
        boolean combination of currently assigned `CostFunctions`. Specified initially in **costs** argument of
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

        .. note::

        A ControlSignal's `adjustment_cost`, and its `adjustment_cost_function` are distinct from the
        `reconfiguration_cost <ModulatoryMechanism.reconfiguration_cost>` and `compute_reconfiguration_cost
        <ModulatoryMechanism.compute_reconfiguration_cost` function of the `ModulatoryMechanism` to which the
        ControlSignal belongs (see `ModulatoryMechanism Reconfiguration Cost <ModulatoryMechanism_Reconfiguration_Cost>`
        for additional details).

    duration_cost_function : IntegratorFunction : default Linear
        calculates an integral of the ControlSignal's `cost`.  It can be any `IntegratorFunction`, or any other
        function that takes a list or array of two values and returns a scalar value. It can be disabled permanently
        for the ControlSignal by assigning `None`.

    duration_cost : float
        intregral of `cost`.

    combine_costs_function : function : default Reduce(operation=SUM)
        combines the results of all cost functions that are enabled, and assigns the result to `cost`.
        It can be any function that takes an array and returns a scalar value.

    cost : float
        combined result of all `cost functions <ControlSignal_Costs>` that are enabled.

    modulation : ModulationParam
        specifies the way in which the `value <ControlSignal.value>` the ControlSignal is used to modify the value of
        the parameter(s) that it controls.

    efferents : [List[ControlProjection]]
        a list of the `ControlProjections <ControlProjection>` assigned to (i.e., that project from) the ControlSignal.

    name : str
        name of the ControlSignal; if it is not specified in the **name** argument of its constructor, a default name
        is assigned (see `name <ModulatorySignal.name>`).

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ControlSignal; if it is not specified in the **prefs** argument of the constructor,
        a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet <LINK>` for
        details).

    """

    #region CLASS ATTRIBUTES

    componentType = CONTROL_SIGNAL
    paramsType = OUTPUT_STATE_PARAMS

    class Parameters(ModulatorySignal.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ControlSignal.variable>`

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray

                value
                    see `value <ControlSignal.value>`

                    :default value: numpy.array([1.])
                    :type: numpy.ndarray
                    :read only: True

                adjustment_cost
                    see `adjustment_cost <ControlSignal.adjustment_cost>`

                    :default value: 0
                    :type: int

                adjustment_cost_function
                    see `adjustment_cost_function <ControlSignal.adjustment_cost_function>`

                    :default value: `Linear`
                    :type: `Function`

                allocation_samples
                    see `allocation_samples <ControlSignal.allocation_samples>`

                    :default value: None
                    :type:

                combine_costs_function
                    see `combine_costs_function <ControlSignal.combine_costs_function>`

                    :default value: `Reduce`
                    :type: `Function`

                cost
                    see `cost <ControlSignal.cost>`

                    :default value: None
                    :type:

                cost_options
                    see `cost_options <ControlSignal.cost_options>`

                    :default value: CostFunctions.INTENSITY
                    :type: `CostFunctions`

                duration_cost
                    see `duration_cost <ControlSignal.duration_cost>`

                    :default value: 0
                    :type: int

                duration_cost_function
                    see `duration_cost_function <ControlSignal.duration_cost_function>`

                    :default value: `SimpleIntegrator`
                    :type: `Function`

                intensity_cost
                    see `intensity_cost <ControlSignal.intensity_cost>`

                    :default value: None
                    :type:

                intensity_cost_function
                    see `intensity_cost_function <ControlSignal.intensity_cost_function>`

                    :default value: `Exponential`
                    :type: `Function`

        """
        # FIX: if the specification of this getter is happening in several other classes, should consider
        #      refactoring Parameter to allow individual attributes to be inherited, othwerise, leaving this is an
        #      isolated case
        variable = Parameter(np.array([defaultControlAllocation]),
                             aliases='allocation',
                             getter=_output_state_variable_getter)

        # # FIX: DOESN'T WORK, SINCE DON'T HAVE ACCESS TO OTHER ARGS
        # function = Parameter(TransferWithCosts, stateful=False, loggable=False, )
        # _parse_function = _function_parser

        value = Parameter(np.array([defaultControlAllocation]), read_only=True, aliases=['intensity'],
                          history_min_length=1)
        allocation_samples = Parameter(None, modulable=True)

        cost_options = Parameter(CostFunctions.DEFAULTS, getter=_cost_options_getter, setter=_cost_options_setter)
        intensity_cost = Parameter(None, read_only=True, getter=_intensity_cost_getter)
        adjustment_cost = Parameter(0, read_only=True, getter=_adjustment_cost_getter)
        duration_cost = Parameter(0, read_only=True, getter=_duration_cost_getter)
        cost = Parameter(None, read_only=True, getter=_cost_getter)

        intensity_cost_function = Parameter(Exponential, stateful=False, loggable=False)
        adjustment_cost_function = Parameter(Linear, stateful=False, loggable=False)
        duration_cost_function = Parameter(SimpleIntegrator, stateful=False, loggable=False)
        combine_costs_function = Parameter(Reduce(operation=SUM), stateful=False, loggable=False)
        modulation = None
        _validate_cost_options = get_validator_by_type_only([CostFunctions, list])
        _validate_intensity_cost_function = get_validator_by_function(is_function_type)
        _validate_adjustment_cost_function = get_validator_by_function(is_function_type)
        _validate_duration_cost_function = get_validator_by_function(is_function_type)

        # below cannot validate because the default value is None and this is considered
        # invalid. Is it that tc.typecheck only runs if an argument is specified at
        # construction?
        # _validate_modulation = get_validator_by_function(_is_modulation_param)

    stateAttributes = ModulatorySignal.stateAttributes | {ALLOCATION_SAMPLES,
                                                          COST_OPTIONS,
                                                          INTENSITY_COST_FUNCTION,
                                                          ADJUSTMENT_COST_FUNCTION,
                                                          DURATION_COST_FUNCTION,
                                                          COMBINE_COSTS_FUNCTION}

    connectsWith = [PARAMETER_STATE, INPUT_STATE, OUTPUT_STATE]
    connectsWithAttribute = [PARAMETER_STATES, INPUT_STATES, OUTPUT_STATES]
    projectionSocket = RECEIVER
    modulators = []

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()
    # paramClassDefaults = OutputState.paramClassDefaults.copy()
    paramClassDefaults.update({
        PROJECTION_TYPE: CONTROL_PROJECTION,
        CONTROLLED_PARAMS:None
    })
    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 default_allocation=None,
                 size=None,
                 index=None,
                 function=Linear(),
                 cost_options:tc.optional(tc.any(CostFunctions, list))=None,
                 intensity_cost_function:(is_function_type)=Exponential,
                 adjustment_cost_function:tc.optional(is_function_type)=Linear,
                 duration_cost_function:tc.optional(is_function_type)=SimpleIntegrator,
                 combine_costs_function:tc.optional(is_function_type)=Reduce(operation=SUM),
                 allocation_samples=Parameters.allocation_samples.default_value,
                 modulation:tc.optional(str)=None,
                 modulates=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        from psyneulink.core.components.functions.transferfunctions import TransferWithCosts
        function = TransferWithCosts(default_variable=self.defaults.variable,
                                     transfer_fct=function,
                                     enabled_cost_functions=cost_options,
                                     intensity_cost_fct=intensity_cost_function,
                                     adjustment_cost_fct=adjustment_cost_function,
                                     duration_cost_fct=duration_cost_function,
                                     combine_costs_fct=combine_costs_function)

        # This is included in case ControlSignal was created by another Component (such as ControlProjection)
        #    that specified ALLOCATION_SAMPLES in params
        if params and ALLOCATION_SAMPLES in params and params[ALLOCATION_SAMPLES] is not None:
            allocation_samples =  params[ALLOCATION_SAMPLES]

        # Note index and assign are not used by ControlSignal, but included here for consistency with OutputState
        # If index has not been specified, but the owner has, control_allocation has been determined, so use that
        index = index or SEQUENTIAL

        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(function=function,
                                                  cost_options=cost_options,
                                                  intensity_cost_function=intensity_cost_function,
                                                  adjustment_cost_function=adjustment_cost_function,
                                                  duration_cost_function=duration_cost_function,
                                                  combine_costs_function=combine_costs_function,
                                                  allocation_samples=allocation_samples,
                                                  params=params)

        # FIX: ??MOVE THIS TO _validate_params OR ANOTHER _instantiate METHOD??
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_states here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputStates in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        super().__init__(owner=owner,
                         reference_value=reference_value,
                         default_allocation=default_allocation,
                         size=size,
                         index=index,
                         assign=None,
                         function=function,
                         modulation=modulation,
                         modulates=modulates,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate cost functions and allocation_samples

        Checks if:
        - cost functions are all appropriate
           (i.e., are references to valid ControlProjection costFunctions (listed in self.costFunctions)
        - allocation_samples is a list, array, range or SampleSpec

        """

        # FIX: REINSTATE IF WANT VALIDATION ON CONTROLSIGNAL RATHER THAN ITS FUNCTION (TO HIDE FROM USER)
        # # MODIFIED 8/30/19 OLD:
        # # Validate cost functions in request_set
        # #   This should be all of them if this is an initialization call;
        # #   Otherwise, just those specified in assign_params
        # for cost_function_name in [item for item in request_set if item in costFunctionNames]:
        #     cost_function = request_set[cost_function_name]
        #
        #     # cost function assigned None: OK
        #     if not cost_function:
        #         continue
        #
        #     # cost_function is Function class specification:
        #     #    instantiate it and test below
        #     if inspect.isclass(cost_function) and issubclass(cost_function, Function):
        #         cost_function = cost_function()
        #
        #     # cost_function is Function object:
        #     #     COMBINE_COSTS_FUNCTION must be CombinationFunction
        #     #     DURATION_COST_FUNCTION must be an IntegratorFunction
        #     #     others must be TransferFunction
        #     if isinstance(cost_function, Function):
        #         if cost_function_name == COMBINE_COSTS_FUNCTION:
        #             if not isinstance(cost_function, CombinationFunction):
        #                 raise ControlSignalError("Assignment of Function to {} ({}) must be a CombinationFunction".
        #                                          format(COMBINE_COSTS_FUNCTION, cost_function))
        #         elif cost_function_name == DURATION_COST_FUNCTION:
        #             if not isinstance(cost_function, IntegratorFunction):
        #                 raise ControlSignalError("Assignment of Function to {} ({}) must be an IntegratorFunction".
        #                                          format(DURATION_COST_FUNCTION, cost_function))
        #         elif not isinstance(cost_function, TransferFunction):
        #             raise ControlSignalError("Assignment of Function to {} ({}) must be a TransferFunction".
        #                                      format(cost_function_name, cost_function))
        #
        #     # cost_function is custom-specified function
        #     #     DURATION_COST_FUNCTION and COMBINE_COSTS_FUNCTION must accept an array
        #     #     others must accept a scalar
        #     #     all must return a scalar
        #     elif isinstance(cost_function, (function_type, method_type)):
        #         if cost_function_name in COMBINE_COSTS_FUNCTION:
        #             test_value = [1, 1]
        #         else:
        #             test_value = 1
        #         try:
        #             result = cost_function(test_value)
        #             if not (is_numeric(result) or is_numeric(np.asscalar(result))):
        #                 raise ControlSignalError("Function assigned to {} ({}) must return a scalar".
        #                                          format(cost_function_name, cost_function))
        #         except:
        #             raise ControlSignalError("Function assigned to {} ({}) must accept {}".
        #                                      format(cost_function_name, cost_function, type(test_value)))
        #
        #     # Unrecognized function assignment
        #     else:
        #         raise ControlSignalError("Unrecognized function ({}) assigned to {}".
        #                                  format(cost_function, cost_function_name))
        # MODIFIED 8/30/19 END

        # Validate allocation samples list:
        # - default is 1D np.array (defined by self.class_defaults.allocation_samples)
        # - however, for convenience and compatibility, allow lists:
        #    check if it is a list of numbers, and if so convert to np.array
        if ALLOCATION_SAMPLES in request_set:
            from psyneulink.core.globals.sampleiterator import allowable_specs
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
            # elif all(isinstance(allocation_samples, spec) for spec in allowable_specs):
            elif is_sample_spec(allocation_samples):
                pass
            else:
                raise ControlSignalError("allocation_samples argument ({}) in {} must be "
                                         "a list or 1D array of numbers, a range, or a {}".
                                         format(allocation_samples, self.name, SampleSpec.__name__))

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # FIX: ??REINSTATE IF ABOVE IS REINSTATED??
        # # MODIFIED 8/30/19 OLD:
        # # ControlProjection Cost Functions
        # for cost_function_name in [item for item in target_set if item in costFunctionNames]:
        #     cost_function = target_set[cost_function_name]
        #     if not cost_function:
        #         continue
        #     if ((not isinstance(cost_function, (Function, function_type, method_type)) and
        #              not issubclass(cost_function, Function))):
        #         raise ControlSignalError("{0} not a valid Function".format(cost_function))
        # MODIFIED 8/30/19 END

    def _instantiate_attributes_before_function(self, function=None, context=None):

        super()._instantiate_attributes_before_function(function=function, context=context)
        self._instantiate_cost_functions(context=context)
        self._instantiate_allocation_samples(context=context)

    def _instantiate_allocation_samples(self, context=None):
        """Assign specified `allocation_samples <ControlSignal.allocation_samples>` to a `SampleIterator`."""

        a = self.paramsCurrent[ALLOCATION_SAMPLES]

        if a is None:
            return

        # KDM 12/14/18: below is a temporary fix that exists to bypass a validation loop
        # resulting from the function_object->function refactor. When this validation/assign_params/etc.
        # is taken care of, this check can be removed
        if isinstance(a, SampleIterator):
            return

        if isinstance(a, (range, np.ndarray)):
            a = list(a)
        self.parameters.allocation_samples._set(SampleIterator(specification=a), context)

    def _instantiate_cost_attributes(self, context=None):

        if self.cost_options:
            # Default cost params
            if self.initialization_status != ContextFlags.DEFERRED_INIT:
                self.intensity_cost = self.intensity_cost_function(self.defaults.allocation)
            else:
                self.intensity_cost = self.intensity_cost_function(self.class_defaults.allocation)
            self.defaults.intensity_cost = self.intensity_cost
            self.adjustment_cost = 0
            self.duration_cost = 0
            self.cost = self.defaults.cost = self.intensity_cost

    def _instantiate_cost_functions(self, context=None):

        for cost_function_name in costFunctionNames:
            self.paramsCurrent[cost_function_name.replace('fct','function')] = getattr(self.function,cost_function_name)

    def _parse_state_specific_specs(self, owner, state_dict, state_specific_spec):
        """Get ControlSignal specified for a parameter or in a 'control_signals' argument

        Tuple specification can be:
            (parameter name, Mechanism)
            [TBI:] (Mechanism, parameter name, weight, exponent, projection_specs)

        Returns params dict with CONNECTIONS entries if any of these was specified.

        """
        from psyneulink.core.components.projections.projection import _parse_connection_specs
        from psyneulink.core.globals.keywords import PROJECTIONS

        params_dict = {}
        state_spec = state_specific_spec

        if isinstance(state_specific_spec, dict):
            return None, state_specific_spec

        elif isinstance(state_specific_spec, tuple):

            state_spec = None
            params_dict[PROJECTIONS] = _parse_connection_specs(connectee_state_type=self,
                                                               owner=owner,
                                                               connections=state_specific_spec)
        elif state_specific_spec is not None:
            raise ControlSignalError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, state_specific_spec))

        if params_dict[PROJECTIONS] is None:
            raise ControlSignalError("PROGRAM ERROR: No entry found in {} params dict for {} "
                                     "with specification of parameter's Mechanism or ControlProjection(s) to it".
                                        format(CONTROL_SIGNAL, owner.name))

        return state_spec, params_dict

    def _update(self, context=None, params=None):
        """Update value (intensity) and costs
        """
        super()._update(context=context, params=params)

        if self.parameters.cost_options._get(context):
            intensity = self.parameters.value._get(context)
            self.parameters.cost._set(self.compute_costs(intensity, context), context)

    def compute_costs(self, intensity, context=None):
        """Compute costs based on self.value (`intensity <ControlSignal.intensity>`)."""
        # FIX 8/30/19: NEED TO DEAL WITH DURATION_COST AS STATEFUL:  DON'T WANT TO MESS UP MAIN VALUE

        cost_options = self.parameters.cost_options._get(context)

        try:
            intensity_change = intensity - self.parameters.intensity.get_previous(context)
        except TypeError:
            intensity_change = [0]

        # COMPUTE COST(S)
        intensity_cost = adjustment_cost = duration_cost = 0

        if CostFunctions.INTENSITY & cost_options:
            intensity_cost = self.intensity_cost_function(intensity, context=context)
            self.parameters.intensity_cost._set(intensity_cost, context)

        if CostFunctions.ADJUSTMENT & cost_options:
            adjustment_cost = self.adjustment_cost_function(intensity_change, context=context)
            self.parameters.adjustment_cost._set(adjustment_cost, context)
        # COMPUTE COST(S)
        # Initialize as backups for cost function that are not enabled
        intensity_cost = adjustment_cost = duration_cost = 0

        if CostFunctions.INTENSITY & cost_options:
            intensity_cost = self.intensity_cost_function(intensity)
            self.parameters.intensity_cost._set(intensity_cost, context)

        if CostFunctions.ADJUSTMENT & cost_options:
            try:
                intensity_change = intensity - self.parameters.intensity.get_previous(context)
            except TypeError:
                intensity_change = [0]
            adjustment_cost = self.adjustment_cost_function(intensity_change)
            self.parameters.adjustment_cost._set(adjustment_cost, context)

        if CostFunctions.DURATION & cost_options:
            duration_cost = self.duration_cost_function(self.parameters.cost._get(context), context=context)
            self.parameters.duration_cost._set(duration_cost, context)

        return max(0.0,
                   self.combine_costs_function([intensity_cost,
                                                adjustment_cost,
                                                duration_cost],
                                               context=context))
