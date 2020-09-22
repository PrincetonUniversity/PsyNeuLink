# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  ControlSignal *****************************************************

"""

Contents:
---------

  * `Control_signal_Overview`
  * `ControlSignal_Creation`
      - `ControlSignal_Specification`
  * `ControlSignal_Structure`
      - `ControlSignal_Projections`
      - `ControlSignal_Modulation`
      - `ControlSignal_Allocation_and_Intensity`
      - `ControlSignal_Costs`
  * `ControlSignal_Execution`d
  * `ControlSignal_Examples`
  * `ControlSignal_Class_Reference`


.. _Control_Signal_Overview:

Overview
--------

A ControlSignal is a type of `ModulatorySignal <ModulatorySignal>` that is specialized for use with a `ControlMechanism
<ControlMechanism>` and one or more `ControlProjections <ControlProjection>`, to modify the parameter(s) of one or more
`Components <Component>`. A ControlSignal receives an `allocation <ControlSignal.allocation>` value from the
ControlMechanism to which it belongs, and uses that to compute an `intensity` (also referred to as a `control_signal`)
that is assigned as the `value <ControlProjection.ControlProjection.value>` of its ControlProjections. Each
ControlProjection conveys its value to the `ParameterPort` for the parameter it controls, which uses that value to
`modulate <ModulatorySignal_Modulation>` the `value <ParameterPort.value>` of the parameter.  A ControlSignal also
calculates a `cost`, based on its `intensity` and/or its time course, that may be used by the ControlMechanism to
adapt the ControlSignal's `allocation <ControlSignal.allocation>` in subsequent executions.

.. _ControlSignal_Creation:

Creating a ControlSignal
------------------------

A ControlSignal is created automatically whenever the parameter of a Mechanism or of its function is `specified for
control <ControlMechanism_ControlSignals>`.  ControlSignals can also be specified in the **control_signals** argument
of the constructor for a `ControlMechanism <ControlMechanism>`, as well as in the `specification of the parameter
<ParameterPort_Specification>` that the ControlSignal is intended to modulate (also see `Modualory Specification
<ParameterPort_Modulatory_Specification>`.  Although a ControlSignal can also be  created on its own using its
constructor (or any of the other ways for `creating an OutputPort <OutputPorts_Creation>`), this is usually not
necessary nor is it advisable, as a ControlSignal has dedicated components and requirements for configuration that
must be met for it to function properly.

.. _ControlSignal_Specification:

*Specifying ControlSignals*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ControlSignal is specified in the **control_signals** argument of the constructor for a `ControlMechanism
<ControlMechanism>`, the parameter(s) to be controlled must be specified.  If other attributes of the ControlSignal
need to be specified (e.g., one or more of its `cost functions <ControlSignal_Costs>`), then the Constructor for the
ControlSignal can be used or a `port specification dictionary <Port_Specification>`, in which the parameter(s) to be
controlled in the **projections** argument or *PROJECTIONS* entry, respectively, using any of the forms below.
For convenience, the parameters can also be specified on their own in the **control_signals** argument of the
ControlMechanism's constructor, in which case a default ControlSignal will be created for each.  In all cases, any
of the following can be use to specify the parameter(s) to be controlled:

  * **ParameterPort** (or list of them) -- for the Mechanism(s) to which the parameter(s) belong;

  * **2-item tuple:** *(parameter name or list of them>, <Mechanism>)* -- the 1st item must be the name of the
    parameter (or list of parameter names), and the 2nd item the Mechanism to which it (they) belong(s); this is a
    convenience format, that is simpler to use than a specification dictionary (see above), but precludes
    specification of any `parameters <ControlSignal_Structure>` for the ControlSignal.

  .. _ControlSignal_Specification_Dictionary:

  * **specification dictionary** -- this is an abbreviated form of `port specification dictionary
    <Port_Specification>`, in which the parameter(s) to be controlled can be specified in either of the two
    following ways:

    * for controlling a single parameter, the dictionary can have the following two entries:

        * *NAME*: str
            the string must be the name of the parameter to be controlled;

        * *MECHANISM*: Mechanism
            the Mechanism must be the one to the which the parameter to be controlled belongs.

    * for controlling multiple parameters, the dictionary can have the following entry:

        * <str>:list
            the string used as the key specifies the name to be used for the ControlSignal,
            and each item of the list must be a `specification of a parameter <ParameterPort_Specification>` to be
            controlled by the ControlSignal (and that will receive a `ControlProjection` from it).
  ..

.. _ControlSignal_Structure:

Structure
---------

A ControlSignal is owned by an `ControlMechanism <ControlMechanism>`, and controls the parameters of one or more
Components by modulating the `function <ParameterPort.function>` of the `ParameterPort` that determines the value
of each of the parameters that it control.  Its operation is governed by several attributes of the ControlSignal,
that are described below.

.. _ControlSignal_Projections:

*Projections*
~~~~~~~~~~~~~

When a ControlSignal is created, it can be assigned one or more `ControlProjections <ControlProjection>`, using either
the **projections** argument of its constructor, or in an entry of a dictionary assigned to the **params** argument
with the key *PROJECTIONS*.  These will be assigned to its `efferents  <ModulatorySignal.efferents>` attribute.  See
`Port Projections <Port_Projections>` for additional details concerning the specification of Projections when
creating a Port.

.. note::
   Although a ControlSignal can be assigned more than one `ControlProjection`, all of those Projections will receive
   the same `value <ControlProjection.value>` (based on the `intensity` of that ControlSignal), and use the same
   form of `modulation <ControlSignal_Modulation>`.  Thus, for them to be meaningful, they should project to
   ParameterPorts for parameters that are meaningfully related to one another (for example, the threshold parameter
   of multiple `DDM` Mechanisms).

.. _ControlSignal_Modulation:

*Modulation*
~~~~~~~~~~~~

A ControlSignal has a `modulation <ModulatorySignal.modulation>` attribute that determines how its ControlSignal's
`value <ControlSignal.value>` is used by the Ports to which it projects to modify their `value <Port_Base.value>` \\s
(see `ModulatorySignal_Modulation` for an explanation of how the `modulation <ModulatorySignal.modulation>` attribute is
specified and used to modulate the `value <Port_Base.value>` of a Port). The `modulation <ModulatorySignal.modulation>`
attribute can be specified in the **modulation** argument of the constructor for a ControlSignal, or in a specification
dictionary as described `above <ControlSignal_Specification>` (see `ModulatorySignal_Types` for forms of specification).
If it is not specified, its default is the value of the `modulation <ControlMechanism.modulation>` attribute of the
ControlMechanism to which the ControlSignal belongs (which is the same for all of the ControlSignals belonging to
that ControlMechanism).  The value of the `modulation <ModulatorySignal.modulation>` attribute of a ControlSignal is
used by all of the `ControlProjections <ControlProjection>` that project from that ControlSignal.

.. _ControlSignal_Allocation_and_Intensity:

*Allocation, Function and Intensity*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Allocation (variable)*. A ControlSignal is assigned an `allocation <ControlSignal>` by the ControlMechanism to which
it belongs. Some ControlMechanisms sample different allocation values for their ControlSignals to determine which to
use (e.g., `OptimizationControlMechanism`);  in those cases, they use each ControlSignal's
`allocation_samples <ControlSignal.allocation_samples>` attribute (specified in the **allocation_samples** argument
of the ControlSignal's constructor) to determine the allocation values to sample for that ControlSignal.  A
ControlSignal's `allocation <ControlSignal>` attribute contains the value assigned to it by the ControlMechanism
at the end of the previous `TRIAL <TimeScale.TRIAL>` (i.e., when the ControlMechanism last executed --  see
`ControlMechanism Execution <ControlMechanism_Execution>`); its value from the previous `TRIAL <TimeScale.TRIAL>` is
assigned to the `last_intensity` attribute.

*Function*. A ControlSignal's `allocation <ControlSignal.allocation>` serves as its `variable
<ModulatorySignal.variable>`, and is used by its `function <ControlSignal.function>` to generate an `intensity`.
The default `function <ControlSignal.function>` for a ControlSignal is `TransferWithCosts`.  This is a
`Function` that supplements its core `TransferFunction` (specified by its `transfer_fct
<TransferWithCosts.transfer_fct>` with a set of cost functions that can be used to compute the ControlSignal's `cost
attributes <ControlSignal_Costs>`.  The default `transfer_fct <TransferWithCosts.transfer_fct>`> for TransferWithCosts
is an identity function (`Linear` with `slope <Linear.slope>` \\=1 and `intercept <Linear.intercept>`\\=0), that simply
assigns the ControlSignal's `allocation <ControlSignal.allocation>` as its `intensity <ControlSignal.intensity>`.
However, the TransferWithCosts function can be specified as **function** argument in the ControlSignal's constructor,
with a different function specified for the **transfer_fct** argument of the TransferWithCosts's constructor (e.g.,
`Exponential`, or any other function that takes and returns a scalar value or 1d array).  The TransferWithCosts' `cost
functions <TransferWithCosts_Cost_Functions>` can also be assigned using its own constructor.  A function other than
TransferWithCosts can also be assigned as the ControlSignal's `function <ControlSignal.function>`, however in that
case, the ControlSignal's costs can't be computed and will all be assigned None.

*Intensity (value)*. The result of the function is assigned as the value of the ControlSignal's `intensity`
attribute, which serves as the ControlSignal's `value <ControlSignal.value>` (also referred to as `control_signal`).
The `intensity` is used by its `ControlProjection(s) <ControlProjection>` to modulate the parameter(s) for which the
ControlSignal is responsible. The ControlSignal's `intensity` attribute  reflects its value for the current `TRIAL
<TimeScale.TRIAL>`; its value from the previous `TRIAL <TimeScale.TRIAL>` is assigned to the `last_intensity` attribute.

.. _ControlSignal_Costs:

*Costs and Cost Functions*
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ControlSignal has a `cost <ControlSignal.cost>` attribute that may be used by the ControlMechanism to which it
belongs to determine its `allocation <ControlSignal.allocation>`.  The value of the `cost <ControlSignal.cost>`
is computed from the ControlSignal's `intensity` using one or more of four cost functions.  These are only
available if the ControlSignal's `function <ControlSignal.function>` is `TransferWithCosts` (which it is by default),
and are actually functions that belong to the `TransferWithCosts` Function (see `TransferWithCosts_Cost_Functions`).
Three of these compute different types of cost, and a fourth combines them, the result of which is assigned as the
ControlSignal's `cost <ControlSignal.cost>` attribute.  The ControlSignal has attributes that reference each of the
TransferWithCosts' cost functions and their attributesas listed below:

    * `intensity_cost` - calculated by the `intensity_cost_fct <TransferWithCosts.intensity_cost_fct>` based on the
      current `intensity` of the ControlSignal, and can be referenced by the ControlSignal's `intensity_cost_function
      <ControlSignal.intensity_cost_function>` attribute.
    ..
    * `adjustment_cost` - calculated by the `adjustment_cost_fct <TransferWithCosts.adjustment_cost_fct>` based on
      a change in the ControlSignal's `intensity` from its last value, and can be referenced by the ControlSignal's
      `adjustment_cost_function <ControlSignal.adjustment_cost_function>` attribute.
    ..
    * `duration_cost` - calculated by the `duration_cost_fct <TransferWithCosts.duration_cost_fct>` based on an
      integral of the ControlSignal's `cost <ControlSignal.cost>`, and can be referenced by the ControlSignal's
      `duration_cost_function <ControlSignal.duration_cost_function>` attribute.
    ..
    * `cost` - calculated by the `combine_costs_fct <TransferWithCosts.combine_costs_fct>` that combines the results
      of any cost functions that are enabled, and can be referenced by the ControlSignal's `duration_cost_function
      <ControlSignal.combine_costs_function>` attribute.

Which of the cost functions are used can be specified in the **costs_options** argument ControlSignal's constructor,
which is passed to the corresponding argument of the TransferWithCosts constructor (where it can also be specified).
The costs functions can also be enabled and disabled using the TransferWithCosts function's `enable_costs
<TransferWithCosts.enable_costs>`,  `disable_costs <TransferWithCosts.disable_costs>`, `toggle_cost
<TransferWithCosts.toggle_cost>` and `assign_costs <TransferWithCosts.assign_costs>` methods.  All of these
take one or more values of `CostFunctions`.  The `combine_costs_function <ControlSignal.combine_costs_function>` is used
to combine the values generated by the enabled cost functions, and the result is assigned as the ControlSignal's `cost
<ControlSignal.cost>` attribute.  By default, the `combine_costs_function <ControlSignal.combine_costs_function>` sums
the results of the enabled cost functions. However, this can be modified by specifying the `combine_costs_function
<ControlSignal.combine_costs_function>` by specifying the ControlSignal's  `function <ControlSignal.function>` using
the constructor for the TransferWithCosts function. The parameters of the ControlSignal's cost functions can also be
modulated by another `ControlMechanism` (see `example <ControlSignal_Example_Modulate_Costs>` below).

    COMMENT:
    .. _ControlSignal_Toggle_Costs:

    *Enabling and Disabling Cost Functions*.  Any of the cost functions (except the `combine_costs_function`) can
    be enabled or disabled using the `toggle_cost` method to turn it `ON` or `OFF`. If it is disabled, that
    component of the cost is not included in the ControlSignal's `cost` attribute.  A cost function  can  also be
    permanently disabled for the ControlSignal by assigning it's attribute `None`.  If a cost function is permanently
    disabled for a ControlSignal, it cannot be re-enabled using `toggle_cost`.
    COMMENT

.. _ControlSignal_Execution:

Execution
---------

A ControlSignal cannot be executed directly.  It is executed whenever the `ControlMechanism <ControlMechanism>` to
which it belongs is executed.  When this occurs, the ControlMechanism provides the ControlSignal with an `allocation
<ControlSignal.allocation>`, that is used by its `function <ControlSignal.function>` to compute its `intensity` for that
`TRIAL <TimeScale.TRIAL>`.  The `intensity` is used by the ControlSignal's `ControlProjections <ControlProjection>` to
set the `value <ParameterPort.value>`\\(s) of the `ParameterPort(s) <ParameterPort>` to which the ControlSignalprojects.

Recall that the ParameterPort value is referenced anywhere that the controlled parameter is used in computation, and
that it does not update until the component to which the ParameterPort belongs executes. If the distinction between the
base value stored in the parameter attribute (i.e. MyTransferMech.function.gain) and the value of the
ParameterPort is unfamiliar, see `Parameter Port documentation <ParameterPort>` for more details, or see
`ModulatorySignal_Modulation` for a detailed description of how modulation operates.

The ControlSignal's `intensity` is also used  by its `cost functions <ControlSignal_Costs>` to compute its `cost`
attribute. That is used by some ControlMechanisms, along with the ControlSignal's `allocation_samples` attribute, to
evaluate a `control_allocation <ControlMechanism.control_allocation>`, and adjust the ControlSignal's `allocation
<ControlSignal.allocation>` for the next `TRIAL <TimeScale.TRIAL>`.

.. note::
   The changes in a parameter in response to the execution of a ControlMechanism are not applied until the Mechanism
   with the parameter being controlled is next executed; see `Lazy Evaluation <Component_Lazy_Updating>` for an
   explanation of "lazy" updating).

.. _ControlSignal_Examples:

Examples
--------

As explained `above <ControlSignal_Creation>` (and in `ControlMechanism_ControlSignals`), a ControlSignal can be
specified either where the parameter it controls is specified, or in the constructor of the ControlMechanism to
which the ControlSignal belongs (i.e., that is responsible for the control).  Which of these to use is largely an
aesthetic matter -- that is, where you wish the specification of control to appear. Examples of each are provided below.

.. note::
  If a ControlSignal is specified where the parameter it controls is specified, the ControlSignal is
  not implemented until the `Component` to which the parameter belongs is assigned to a `Composition`
  (see `ControlMechanism_ControlSignals`).

.. _ControlSignal_Example_Specify_with_Parameter:

*Specify a ControlSignal where a parameter is specified*

The simplest way to do this is to specify the ControlSignal by class, or using the keyword *CONTROL*, in a
`tuple specification <ParameterPort_Tuple_Specification>` for the parameter, as in the following examples:

    >>> from psyneulink import *
    >>> my_mech = ProcessingMechanism(function=Logistic(bias=(1.0, ControlMechanism))) #doctest: +SKIP
    or
    >>> my_mech = ProcessingMechanism(function=Logistic(bias=(1.0, CONTROL)))

Both of these assign a ControlSignal to the `bias <Logistic.gain>` parameter of the `Logistic` Function used by a
`ProcessingMechanism`. This creates a default ControlSignal, with a `ControlProjection` that projects to the
ProcessingMechanism's `ParameterPort` for the `bias <Logistic.bias>` parameter of its `Logistic` Function.  The
default value of a ControlSignal's `modulation <ModulatorySignal.modulation>` attribute is *MULTIPLICATIVE*,
so that it will multiply the value of the `bias <Logistic.bias>` parameter. When the ProcessingMechanism executes,
the Logistic Function will use the value of the ControlSignal as its bias parameter.

.. _ControlSignal_Example_Modulation:

If attributes of the ControlSignal must be specified, then its constructor can be used.  For example, ordinarily a
ControlSignal modulates the `MULTIPLICATIVE_PARAM <ModulatorySignal_Types>` of a `Port's <Port>` `function
<Port_Base.function>`.  However, this can be changed by using the ControlSignal's constructor to specify its
**modulation** argument, as follows::

    >>> my_mech = ProcessingMechanism(function=Logistic(gain=(1.0, ControlSignal(modulation=ADDITIVE))))

This specifies that the value of the ControlSignal should be added to, rather than multiply the value of the `gain
<Logistic.gain>` parameter of the Logistic function.  Note that the **modulation** argument determines how to modify a
*parameter* of the *Logistic* Function (in this case, its `gain <Logistic.gain>` parameter), and its input directly;
that is, in this example, the value of the ControlSignal is added to the *gain parameter* of the Logistic function,
*not* to its `variable <Lo˚gistic.variable>`.  If the value of the ControlSignal's **modulation** argument had been
*OVERRIDE*, then the ControlSignal's value would have been used as (i.e., it would have replaced) the ˚value of the
*Logistic* Function's `gain <Logistic.gain>` parameter, rather than being added to it.

*Specify ControlSignals in a ControlMechanism's constructor*

Parameters can also be specified for control in constructor of the `ControlMechanism <ControlMechanism>` that controls
them.  This is done in the **control_signals** argument of the ControlMechanism's constructor, by specifying a
ControlSignal for each parameter to be controlled.  The following example shows several ways in which the ControlSignal
can be specified (see `ControlSignal_Specification` for additional details)::

    >>> mech_A = ProcessingMechanism()
    >>> mech_B = ProcessingMechanism(function=Logistic)
    >>> ctl_mech = ControlMechanism(control_signals=[(INTERCEPT, mech_A),
    ...                                              ControlSignal(modulates=(GAIN, mech_B),
    ...                                                            cost_options=CostFunctions.INTENSITY),
    ...                                              {
    ...                                                  NAME: BIAS,
    ...                                                  MECHANISM: mech_B,
    ...                                                  MODULATION: ADDITIVE
    ...                                              }])

Here, ``ctl_mech`` modulates the `intercept <Linear.intercept>` parameter of ``mech_A``'s `Linear` Function
(the default `function <Mechanism_Base.function>` for a `ProcessingMechanism`), and the `bias <Logistic.bias>`
and `gain <Logistic.gain>` parameters of ``mech_B``'s `Logistic` Function.  The first ControlSignal is specified
using a 2-item tuple (the simplest way to do so); the second uses the ControlSignal's constructor (allowing another
parameter to be specified -- here, its `cost_options <ControlSignal_Costs>`); and the third uses a specification
dictionary (which supports additional options; see `above <ControlSignal_Specification_Dictionary>`).

.. _ControlSignal_Example_Modulate_Costs:

*Modulate the parameters of a ControlSignal's cost function*.

ControlSignals can be used to modulate the parameters of other ControlSignals.  This is done using the **modulation**
argument, but instead of using the keyword for a `generic form of modulation <ModulatorySignal_Types>` (as in the
example `above <ControlSignal_Example_Modulation>`), a parameter of the ControlSignal's `function
<ControlSignal.function>` is used.  For example, the following shows how the parameters of a ControlSignal's `cost
function <ControlSignal_Costs>` can be modulated by another ControlSignal::

  >>> from psyneulink import *
  >>> mech = ProcessingMechanism(name='my_mech')
  >>> ctl_mech_A = ControlMechanism(monitor_for_control=mech,
  ...                               control_signals=ControlSignal(modulates=(INTERCEPT,mech),
  ...                                                             cost_options=CostFunctions.INTENSITY))
  >>> ctl_mech_B = ControlMechanism(monitor_for_control=mech,
  ...                               control_signals=ControlSignal(modulates=ctl_mech_A.control_signals[0],
  ...                                                             modulation=INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM))
  >>> comp = Composition()
  >>> pway = comp.add_linear_processing_pathway(pathway=[mech,
  ...                                              ctl_mech_A,
  ...                                              ctl_mech_B
  ...                                              ])
  >>> pway.pathway
  [(ProcessingMechanism my_mech)]

Here, the `ControlSignal` of ``ctl_mech_A`` is configured to monitor the output of ``mech``, modulate the
the `intercept <Linear.intercept>` parameter of its `function <Mechanism_Base.function>` (which is a `Linear` by
default), and to implement its `intensity_cost_fct <TransferWithCosts.intensity_cost_fct>` (using the
**cost_options** argument of the ControlSignal's constructor).  ``ctl_mech_B`` is configured to also monitor
``mech``, but to modulate the `multiplicative_param <Function_Modulatory_Params>` of the `intensity_cost_fct
<TransferWithCosts.intensity_cost_fct>` of ``ctl_mech_A``\\s ControlSignal.  The default for the `intensity_cost_fct
<TransferWithCosts.intensity_cost_fct>` is `Exponential`, the `multiplicative_param <Function_Modulatory_Params>`
of which is `rate <Exponential>`.  (Note that the pathway returned from the call to `add_linear_processing_pathway
<Composition.add_linear_processing_pathway>` contains only ``my_mech``, since that is the only `ProcessingMechanism`
in the `Pathway`. When the ``comp`` is run with an input of ``3``, since the default `function
<Mechanism_Base.function>` for ``mech`` is `Linear` with a `slope <Linear.slope>` of 1 and an `intercept <Linear>`
of 0, its output is also ``3``, which is used by both ``ctl_mech_A`` and ``ctl_mech_B`` as their `allocation
<ControlSignal.allocation>`.  Since the ControlSignals of both use their default `function <ControlSignal>` ——
TransferWithCosts with an identity function as its `transfer_fct <TransferWithCosts.transfer_fct>` -- the
`intensity <ControlMechanism.intensity>` of both is the same as their ``allocation`` (i.e., ``3``).  This is used
as the input to the `Exponential` function used as `intensity_cost_function <ControlSignal.intensity_cost_function>`
of ``ctl_mech_A``.  However, since the `rate <Exponential.rate>` of that function is modulated by ``ctl_mech_B``,
the `intensity <ControlSignal.intensity>` of which is also ``3``, the value of the `intensity_cost
<ControlSignal.intensity_cost>` for ``ctl_mech_A`` is e ^ (allocation (3) * value of ctl_mech_B (also 3)) = e^9,
as shown below::

    >>> comp.run(inputs={mech:[3]}, num_trials=2)
    [array([3.])]
    >>> ctl_mech_A.control_signals[0].intensity_cost
    array([8103.08392758])

The Composition must be executed for 2 trials to see this, since the `value <ControlProjection.value>` computed
by ``ctl_mech_B`` must be computed on the first trial before it can have its effect on ``ctl_mech_A`` on the next
(i.e., second) trial (see noted under `ControlSignal_Execution`).

.. _ControlSignal_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

# FIX: EVCControlMechanism IS IMPORTED HERE TO DEAL WITH COST FUNCTIONS THAT ARE DEFINED IN EVCControlMechanism
#            SHOULD THEY BE LIMITED TO EVC??
from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import SimpleIntegrator
from psyneulink.core.components.functions.transferfunctions import Exponential, Linear, CostFunctions, TransferWithCosts
from psyneulink.core.components.ports.modulatorysignals.modulatorysignal import ModulatorySignal
from psyneulink.core.components.ports.outputport import SEQUENTIAL, _output_port_variable_getter
from psyneulink.core.components.ports.port import Port_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    ALLOCATION_SAMPLES, CONTROLLED_PARAMS, CONTROL_PROJECTION, CONTROL_SIGNAL, \
    INPUT_PORT, INPUT_PORTS, \
    OUTPUT_PORT, OUTPUT_PORTS, OUTPUT_PORT_PARAMS, \
    PARAMETER_PORT, PARAMETER_PORTS, \
    PROJECTION_TYPE, RECEIVER, SUM, FUNCTION
from psyneulink.core.globals.parameters import FunctionParameter, Parameter, get_validator_by_function
from psyneulink.core.globals.sampleiterator import is_sample_spec
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import \
    is_numeric, iscompatible, kwCompatibilityLength, kwCompatibilityNumeric, kwCompatibilityType, convert_all_elements_to_np_array
from psyneulink.core.globals.sampleiterator import SampleSpec, SampleIterator

__all__ = ['ControlSignal', 'ControlSignalError', 'COST_OPTIONS']

from psyneulink.core.components.functions.transferfunctions import \
    ENABLED_COST_FUNCTIONS, \
    INTENSITY_COST_FUNCTION, ADJUSTMENT_COST_FUNCTION, \
    DURATION_COST_FUNCTION, COMBINED_COSTS, COMBINE_COSTS_FUNCTION

COST_OPTIONS = 'cost_options'


class ControlSignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


    def __str__(self):
        return repr(self.error_value)


class ControlSignal(ModulatorySignal):
    """
    ControlSignal(                                                 \
        default_allocation=defaultControlAllocation,               \
        function=TransferWithCosts,                                \
        costs_options=None,                                        \
        intensity_cost_function=Exponential,                       \
        adjustment_cost_function=Linear,                           \
        duration_cost_function=IntegratorFunction,                 \
        combine_costs_function=Reduce(operation=SUM),              \
        allocation_samples=self.class_defaults.allocation_samples, \
        modulates=None,                                            \
        projections=None)

    A subclass of `ModulatorySignal <ModulatorySignal>` used by a `ControlMechanism <ControlMechanism>` to
    modulate the parameter(s) of one or more other `Mechanisms <Mechanism>`.  See `ModulatorySignal
    <ModulatorySignal_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    default_allocation : scalar, list or np.ndarray : defaultControlAllocation
        specifies the template and default value used for `allocation <ControlSignal.allocation>`;  must match the
        shape of each item specified in `allocation_samples <ControlSignal.allocation_samples>`.

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
        specifies the values used by the ControlSignal's `owner <ModulatorySignal.owner>` to determine its
        `control_allocation <ControlMechanism.control_allocation>` (see `ControlSignal_Execution`).

    modulates : list of Projection specifications
        specifies the `ControlProjection(s) <ControlProjection>` to be assigned to the ControlSignal, and that will be
        listed in its `efferents <ModulatorySignal.efferents>` attribute (see `ControlSignal_Projections` for additional
        details).

    Attributes
    ----------

    allocation : float : default: defaultControlAllocation
        value assigned by the ControlSignal's `owner <ModulatorySignal.owner>`, and used as the `variable
        <Projection_Base.variable>` of its `function <ControlSignal.function>` to determine the ControlSignal's
        `ControlSignal.intensity`.
    COMMENT:
    FOR DEVELOPERS:  Implemented as an alias of the ControlSignal's variable Parameter
    COMMENT

    last_allocation : float
        value of `allocation` in the previous execution of ControlSignal's `owner <ModulatorySignal.owner>`.

    allocation_samples : SampleIterator
        `SampleIterator` created from **allocation_samples** specification and used to generate a set of values to
        sample by the ControlSignal's `owner <ModulatorySignal.owner>` when determining its `control_allocation
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
        the `intensity` of the ControlSignal on the previous execution of its `owner <ModulatorySignal.owner>`.

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
        `reconfiguration_cost <ControlMechanism.reconfiguration_cost>` and `compute_reconfiguration_cost
        <ControlMechanism.compute_reconfiguration_cost` function of the `ControlMechanism` to which the
        ControlSignal belongs (see `ControlMechanism Reconfiguration Cost <ControlMechanism_Reconfiguration_Cost>`
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

    efferents : [List[ControlProjection]]
        a list of the `ControlProjections <ControlProjection>` assigned to (i.e., that project from) the ControlSignal.

    """

    #region CLASS ATTRIBUTES

    componentType = CONTROL_SIGNAL
    paramsType = OUTPUT_PORT_PARAMS

    class Parameters(ModulatorySignal.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ControlSignal.variable>`

                    :default value: numpy.array([1.])
                    :type: ``numpy.ndarray``

                value
                    see `value <ControlSignal.value>`

                    :default value: numpy.array([1.])
                    :type: ``numpy.ndarray``
                    :read only: True

                adjustment_cost
                    see `adjustment_cost <ControlSignal.adjustment_cost>`

                    :default value: 0
                    :type: ``int``
                    :read only: True

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
                    :read only: True

                cost_options
                    see `cost_options <ControlSignal.cost_options>`

                    :default value: CostFunctions.INTENSITY
                    :type: `CostFunctions`

                duration_cost
                    see `duration_cost <ControlSignal.duration_cost>`

                    :default value: 0
                    :type: ``int``
                    :read only: True

                duration_cost_function
                    see `duration_cost_function <ControlSignal.duration_cost_function>`

                    :default value: `SimpleIntegrator`
                    :type: `Function`

                function
                    see `function <ControlSignal.function>`

                    :default value: `TransferWithCosts`
                    :type: `Function`

                intensity_cost
                    see `intensity_cost <ControlSignal.intensity_cost>`

                    :default value: None
                    :type:
                    :read only: True

                intensity_cost_function
                    see `intensity_cost_function <ControlSignal.intensity_cost_function>`

                    :default value: `Exponential`
                    :type: `Function`

                transfer_function
                    see `transfer_function <ControlSignal.transfer_function>`

                    :default value: `Linear`
                    :type: `Function`
        """
        # FIX: if the specification of this getter is happening in several other classes, should consider
        #      refactoring Parameter to allow individual attributes to be inherited, othwerise, leaving this is an
        #      isolated case
        variable = Parameter(
            np.array([defaultControlAllocation]),
            aliases='allocation',
            getter=_output_port_variable_getter,
            pnl_internal=True, constructor_argument='default_variable'
        )

        function = Parameter(TransferWithCosts, stateful=False, loggable=False)
        transfer_function = FunctionParameter(
            Linear,
            function_parameter_name='transfer_fct',
            primary=False
        )

        value = Parameter(
            np.array([defaultControlAllocation]),
            read_only=True,
            aliases=['intensity'],
            pnl_internal=True,
            history_min_length=1
        )
        allocation_samples = Parameter(None, modulable=True)

        cost_options = FunctionParameter(
            CostFunctions.DEFAULTS,
            function_parameter_name=ENABLED_COST_FUNCTIONS,
            valid_types=(CostFunctions, list)
        )
        intensity_cost = FunctionParameter(None)
        adjustment_cost = FunctionParameter(0)
        duration_cost = FunctionParameter(0)

        cost = FunctionParameter(None, function_parameter_name=COMBINED_COSTS)

        intensity_cost_function = FunctionParameter(
            Exponential,
            function_parameter_name='intensity_cost_fct'
        )
        adjustment_cost_function = FunctionParameter(
            Linear,
            function_parameter_name='adjustment_cost_fct'
        )
        duration_cost_function = FunctionParameter(
            SimpleIntegrator,
            function_parameter_name='duration_cost_fct'
        )
        combine_costs_function = FunctionParameter(
            Reduce,
            function_parameter_name='combine_costs_fct'
        )
        _validate_intensity_cost_function = get_validator_by_function(is_function_type)
        _validate_adjustment_cost_function = get_validator_by_function(is_function_type)
        _validate_duration_cost_function = get_validator_by_function(is_function_type)

        # below cannot validate because the default value is None and this is considered
        # invalid. Is it that tc.typecheck only runs if an argument is specified at
        # construction?
        # _validate_modulation = get_validator_by_function(_is_modulation_param)

        def _validate_allocation_samples(self, allocation_samples):
            try:
                samples_as_array = convert_all_elements_to_np_array(allocation_samples)
                first_item_type = type(allocation_samples[0])
                first_item_shape = samples_as_array[0].shape

                for i in range(1, len(allocation_samples)):
                    if not isinstance(allocation_samples[i], first_item_type):
                        return 'all items must have the same type'
                    if not samples_as_array[i].shape == first_item_shape:
                        return 'all items must have the same shape'
            except (TypeError, IndexError):
                # not iterable, so assume single value
                pass

    portAttributes = ModulatorySignal.portAttributes | {ALLOCATION_SAMPLES,
                                                          COST_OPTIONS,
                                                          INTENSITY_COST_FUNCTION,
                                                          ADJUSTMENT_COST_FUNCTION,
                                                          DURATION_COST_FUNCTION,
                                                          COMBINE_COSTS_FUNCTION}

    connectsWith = [PARAMETER_PORT, INPUT_PORT, OUTPUT_PORT]
    connectsWithAttribute = [PARAMETER_PORTS, INPUT_PORTS, OUTPUT_PORTS]
    projectionSocket = RECEIVER
    modulators = []
    projection_type = CONTROL_PROJECTION

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'OutputPortCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    #endregion

    @tc.typecheck
    def __init__(self,
                 owner=None,
                 reference_value=None,
                 default_allocation=None,
                 size=None,
                 transfer_function=None,
                 cost_options:tc.optional(tc.any(CostFunctions, list))=None,
                 intensity_cost_function:tc.optional(is_function_type)=None,
                 adjustment_cost_function:tc.optional(is_function_type)=None,
                 duration_cost_function:tc.optional(is_function_type)=None,
                 combine_costs_function:tc.optional(is_function_type)=None,
                 allocation_samples=None,
                 modulation:tc.optional(str)=None,
                 modulates=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):

        try:
            if kwargs[FUNCTION] is not None:
                raise TypeError(
                    f'{self.__class__.__name__} automatically creates a '
                    'TransferWithCosts function, and does not accept override. '
                    'TransferWithCosts uses the transfer_function parameter.'
                )
        except KeyError:
            pass

        # This is included in case ControlSignal was created by another Component (such as ControlProjection)
        #    that specified ALLOCATION_SAMPLES in params
        if params and ALLOCATION_SAMPLES in params and params[ALLOCATION_SAMPLES] is not None:
            allocation_samples = params[ALLOCATION_SAMPLES]

        # FIX: ??MOVE THIS TO _validate_params OR ANOTHER _instantiate METHOD??
        # IMPLEMENTATION NOTE:
        # Consider adding self to owner.output_ports here (and removing from ControlProjection._instantiate_sender)
        #  (test for it, and create if necessary, as per OutputPorts in ControlProjection._instantiate_sender),

        # Validate sender (as variable) and params
        super().__init__(
            owner=owner,
            reference_value=reference_value,
            default_allocation=default_allocation,
            size=size,
            transfer_function=transfer_function,
            modulation=modulation,
            modulates=modulates,
            cost_options=cost_options,
            intensity_cost_function=intensity_cost_function,
            adjustment_cost_function=adjustment_cost_function,
            duration_cost_function=duration_cost_function,
            combine_costs_function=combine_costs_function,
            allocation_samples=allocation_samples,
            params=params,
            name=name,
            prefs=prefs,
            **kwargs
        )

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
        #     elif isinstance(cost_function, (types.FunctionType, types.MethodType)):
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
        #     if ((not isinstance(cost_function, (Function, types.FunctionType, types.MethodType)) and
        #              not issubclass(cost_function, Function))):
        #         raise ControlSignalError("{0} not a valid Function".format(cost_function))
        # MODIFIED 8/30/19 END

    def _instantiate_attributes_before_function(self, function=None, context=None):

        super()._instantiate_attributes_before_function(function=function, context=context)
        self._instantiate_allocation_samples(context=context)

    def _instantiate_function(self, function, function_params=None, context=None):

        from psyneulink.core.components.functions.transferfunctions import \
            TRANSFER_FCT, INTENSITY_COST_FCT, ADJUSTMENT_COST_FCT, DURATION_COST_FCT, COMBINE_COSTS_FCT

        fcts = {
            TRANSFER_FCT:self.defaults.transfer_function,
            INTENSITY_COST_FCT:self.defaults.intensity_cost_function,
            ADJUSTMENT_COST_FCT:self.defaults.adjustment_cost_function,
            DURATION_COST_FCT:self.defaults.duration_cost_function,
            COMBINE_COSTS_FCT:self.defaults.combine_costs_function,
        }
        function = TransferWithCosts(default_variable=self.defaults.variable,
                                     enabled_cost_functions=self.defaults.cost_options,
                                     **fcts)

        super()._instantiate_function(function, function_params, context)

    def _instantiate_allocation_samples(self, context=None):
        """Assign specified `allocation_samples <ControlSignal.allocation_samples>` to a `SampleIterator`."""

        a = self.parameters.allocation_samples._get(context)

        if a is None:
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

    def _parse_port_specific_specs(self, owner, port_dict, port_specific_spec):
        """Get ControlSignal specified for a parameter or in a 'control_signals' argument

        Tuple specification can be:
            (parameter name, Mechanism)
            [TBI:] (Mechanism, parameter name, weight, exponent, projection_specs)

        Returns params dict with CONNECTIONS entries if any of these was specified.

        """
        from psyneulink.core.components.projections.projection import _parse_connection_specs
        from psyneulink.core.globals.keywords import PROJECTIONS

        params_dict = {}
        port_spec = port_specific_spec

        if isinstance(port_specific_spec, dict):
            return None, port_specific_spec

        elif isinstance(port_specific_spec, tuple):

            port_spec = None
            params_dict[PROJECTIONS] = _parse_connection_specs(connectee_port_type=self,
                                                               owner=owner,
                                                               connections=port_specific_spec)
        elif port_specific_spec is not None:
            raise ControlSignalError("PROGRAM ERROR: Expected tuple or dict for {}-specific params but, got: {}".
                                  format(self.__class__.__name__, port_specific_spec))

        if params_dict[PROJECTIONS] is None:
            raise ControlSignalError("PROGRAM ERROR: No entry found in {} params dict for {} "
                                     "with specification of parameter's Mechanism or ControlProjection(s) to it".
                                        format(CONTROL_SIGNAL, owner.name))

        return port_spec, params_dict

    def _update(self, params=None, context=None):
        """Update value (intensity) and costs
        """
        super()._update(params=params, context=context)

        if self.parameters.cost_options._get(context):
            intensity = self.parameters.value._get(context)
            self.parameters.cost._set(self.compute_costs(intensity, context), context)

    def compute_costs(self, intensity, context=None):
        """Compute costs based on self.value (`intensity <ControlSignal.intensity>`)."""
        # FIX 8/30/19: NEED TO DEAL WITH DURATION_COST AS STATEFUL:  DON'T WANT TO MESS UP MAIN VALUE

        cost_options = self.parameters.cost_options._get(context)

        # try:
        #     intensity_change = intensity - self.parameters.intensity.get_previous(context)
        # except TypeError:
        #     intensity_change = [0]

        # COMPUTE COST(S)
        # Initialize as backups for cost function that are not enabled
        intensity_cost = adjustment_cost = duration_cost = 0

        if CostFunctions.INTENSITY & cost_options:
            intensity_cost = self.intensity_cost_function(intensity, context)
            self.parameters.intensity_cost._set(intensity_cost, context)

        if CostFunctions.ADJUSTMENT & cost_options:
            try:
                intensity_change = intensity - self.parameters.intensity.get_previous(context)
            except TypeError:
                intensity_change = [0]
            adjustment_cost = self.adjustment_cost_function(intensity_change, context)
            self.parameters.adjustment_cost._set(adjustment_cost, context)

        if CostFunctions.DURATION & cost_options:
            duration_cost = self.duration_cost_function(self.parameters.cost._get(context), context=context)
            self.parameters.duration_cost._set(duration_cost, context)

        return max(0.0,
                   self.combine_costs_function([intensity_cost,
                                                adjustment_cost,
                                                duration_cost],
                                               context=context))
