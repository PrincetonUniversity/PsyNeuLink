# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  LCControlMechanism ************************************************

"""

Contents
--------

  * `LCControlMechanism_Overview`
  * `LCControlMechanism_Creation`
      - `LCControlMechanism_ObjectiveMechanism_Creation`
      - `LCControlMechanism_Modulated_Mechanisms`
  * `LCControlMechanism_Structure`
      - `LCControlMechanism_Input`
          • `LCControlMechanism_ObjectiveMechanism`
      - `LCControlMechanism_Function`
          • `LCControlMechanism_Modes_Of_Operation`
      - `LCControlMechanism_Output`
  * `LCControlMechanism_Execution`
  * `LCControlMechanism_Examples`
  * `LCControlMechanism_Class_Reference`


.. _LCControlMechanism_Overview:

Overview
--------

An LCControlMechanism is a `ControlMechanism <ControlMechanism>` that multiplicatively modulates the `function
<Mechanism_Base.function>` of one or more `Mechanisms <Mechanism>` (usually `TransferMechanisms <TransferMechanism>`).
It implements an abstract model of the `locus coeruleus (LC)  <https://www.ncbi.nlm.nih.gov/pubmed/12371518>`_ that
uses an `FitzHughNagumoIntegrator` Function to generate its output.  This is modulated by a `mode
<LCControlMechanism.mode_FitzHughNagumo>` parameter that regulates its function between `"tonic" and "phasic" modes of
operation <LCControlMechanism_Modes_Of_Operation>`.  The Mechanisms modulated by an LCControlMechanism can be listed
using its `show <LCControlMechanism.show>` method.  When used with an `AGTControlMechanism` to regulate the `mode
<FitzHughNagumoIntegrator.mode>` parameter of its `FitzHughNagumoIntegrator` Function, it implements a form of the
`Adaptive Gain Theory <http://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.28.061604.135709>`_ of the locus
coeruleus-norepinephrine (LC-NE) system.

.. _LCControlMechanism_Creation:

Creating an LCControlMechanism
------------------------------

An LCControlMechanism can be created in any of the ways used to `create a ControlMechanism <ControlMechanism_Creation>`.
The following sections describe how to specify the inputs that drive the LCControlMechanism's response, and the
Mechanisms that it controls.

.. _LCControlMechanism_ObjectiveMechanism_Creation:

*ObjectiveMechanism and Monitored OutputPorts*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the **objective_mechanism** argument is specified then, as with a standard ControlMechanism, the specified
`ObjectiveMechanism` is assigned to its `objective_mechanism <ControlMechanism.objective_mechanism>` attribute. The
`value <OutputPort.value>` of the ObjectiveMechanism's *OUTCOME* `OutputPort` must be a scalar (that is used as the
input to the LCControlMechanism's `function <LCControlMechanism.function>` to drive its `phasic response
<LCControlMechanism_Modes_Of_Operation>`.  An ObjectiveMechanism can also be constructed automatically, by specifying
**objective_mechanism** as True; that is assigned a `CombineMeans` Function  as its `function
<ObjectiveMechanism.function>` (see `LCControlMechanism_ObjectiveMechanism`).

If an ObjectiveMechanism is assigned to the LCControlMechanism (whether by specifying one explicitly or that it be
created automatically), the LCControlMechanism receives its input from that ObjectiveMechanism, which receives its
input from any `OutputPorts <OutputPort>` specified in **monitor_for_control** argument of the constructor for
LCControlMechanism.


.. _LCControlMechanism_Modulated_Mechanisms:

*Mechanisms to Modulate*
~~~~~~~~~~~~~~~~~~~~~~~~

The Mechanisms to be modulated by an LCControlMechanism are specified in the **modulated_mechanisms** argument of its
constructor. An LCControlMechanism controls a `Mechanism <Mechanism>` by modifying the `multiplicative_param
<Function_Modulatory_Params>` of the Mechanism's `function <Mechanism_Base.function>`.  Therefore, any Mechanism
specified for control by an LCControlMechanism must be either a `ProcessingMechanism`, or a Mechanism that uses as its
`function <Mechanism_Base.function>` a class of `Function <Function>` that implements a `multiplicative_param
<Function_Modulatory_Params>`.  The **modulate_mechanisms** argument must be either a list of such Mechanisms, or
a `Composition` (to modulate all of the `ProcessingMechanisms <ProcessingMechanism>` in a Composition -- see below).
see below). If a Mechanism specified in the **modulated_mechanisms** argument does not implement a multiplicative_param,
it is ignored. A `ControlProjection` is automatically created that projects from the LCControlMechanism to the
`ParameterPort` for the `multiplicative_param <Function_Modulatory_Params>` of every Mechanism specified in the
**modulated_mechanisms** argument.  The Mechanisms modulated by an LCControlMechanism are listed in its
`modulated_mechanisms <LCControlMechanism.modulated_mechanisms>` attribute).

If `Composition` is assigned as the value of **modulate_mechanisms**, then the LCControlMechanism will modulate all
of the `ProcessingMechanisms` in that Composition, with the exception of any `ObjectiveMechanism`\\s that are assigned
a the `objective_mechanism <ControlMechanism.objective_mechanism>` of another `ControlMechanism`.  Note that only the
Mechanisms that already belong to that Composition are included at the time the LCControlMechanism is constructed.
Therefore, to include *all* Mechanisms in the Composition at the time it is run, the LCControlMechanism should be
constructed and `added to the Composition using the Composition's `add_node <Composition.add_node>` method) after all
of the other Mechanisms have been added.

.. _LCControlMechanism_Structure:

Structure
---------

.. _LCControlMechanism_Input:

*Input*
~~~~~~~

An LCControlMechanism has a single (primary) `InputPort <InputPort_Primary>`, the `value <InputPort.value>` of
which is a scalar that is provided by a `MappingProjection` from the *OUTCOME* `OutputPort <ObjectiveMechanism_Output>`
of the LCControlMechanism's `ObjectiveMechanism`.  That value is used as the input to the LCControlMechanism's
`function <LCControlMechanism.function>`, which drives its `phasic response <LCControlMechanism_Modes_Of_Operation>`.


.. _LCControlMechanism_ObjectiveMechanism:

ObjectiveMechanism
^^^^^^^^^^^^^^^^^^

If an ObjectiveMechanism is `automatically created <LCControlMechanism_ObjectiveMechanism_Creation> for an
LCControlMechanism, it receives its inputs from the `OutputPort(s) <OutputPort>` specified the
**monitor_for_control** argument of the LCControlMechanism constructor, or the **montiored_output_ports** argument
of the LCControlMechanism's `ObjectiveMechanism <ControlMechanism_ObjectiveMechanism>`.  By default, the
ObjectiveMechanism is assigned a `CombineMeans` Function with a default `operation <LinearCombination.operation>` of
*SUM*; this takes the mean of each array that the ObjectiveMechanism receives from the `value <OutputPort.value>` of
each of the OutputPorts that it monitors, and returns the sum of these means.  The `value <OutputPort.value>` of
each OutputPort can be weighted (multiplicatively and/or exponentially), by specifying this in the
**monitor_for_control** argument of the LCControlMechanism (see `ControlMechanism_Monitor_for_Control` for details).
As with any ControlMechanism, its ObjectiveMechanism can be explicitly specified to customize its `function
<ObjectiveMechanism.function>` or any of its other parameters, by specifyihng it in the **objective_mechanism**
argument of the LCControlMechanism's constructor.

.. _LCControlMechanism_Objective_Mechanism_Function_Note:

.. note::
   If an `ObjectiveMechanism` is specified in the **objective_mechanism** argument of the LCControlMechanism's
   constructor, then its attribute values (including any defaults) override those used by a LCControlMechanism for
   creating its `objective_mechanism <LCControlMechanism.objective_mechanism>`.  In particular, whereas an
   ObjectiveMechanism uses `LinearCombination` as the default for its `function <ObjectiveMechanism.function>`,
   an LCControlMechanism uses `CombineMeans` as the `function <ObjectiveMechanism.function>` of its `objective_mechanism
   <LCControlMechanism.objective_mechanism>`.  As a consequence, if an ObjectiveMechanism is explicitly specified in
   the LCControlMechanism's **objective_mechanism** argument, and its **function** argument is not also
   explicitly specified as `CombineMeans`, then `LinearCombination` will be used for the ObjectiveMechanism's `function
   <ObjectiveMechanism.function>`.  To insure that `CombineMeans` is used, it must be specified explicitly in the
   **function** argument of the constructor for the ObjectiveMechanism (for an example of a similar condition
   see example under `ControlMechanism_ObjectiveMechanism_Function`).

The ObjectiveFunction is listed in the LCControlMechanism's `objective_mechanism
<LCControlMechanism.objective_mechanism>` attribute.  The OutputPorts it monitors are listed in the
ObjectiveMechanism's `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>` attribute) as well as the
LCControlMechanism's `monitor_for_control <LCControlMechanism.monitor_for_control>` attribute.  These can be
displayed using the LCControlMechanism's `show <LCControlMechanism.show>` method.

.. _LCControlMechanism_Function:

*Function*
~~~~~~~~~~

An LCControlMechanism uses the `FitzHughNagumoIntegrator` as its `function <LCControlMechanism.function>`; this
implements a `FitzHugh-Nagumo model <https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model>`_ often used to describe
the spiking of a neuron, but in this case the population activity of the LC (see `Gilzenrat et al., 2002
<http://www.sciencedirect.com/science/article/pii/S0893608002000552?via%3Dihub>`_). The `FitzHughNagumoIntegrator`
Function of an LCControlMechanism takes a scalar as its `variable <FitzHughNagumoIntegrator.variable>`, received from
the `input <LCControlMechanism_Input>` to the LCControlMechanism, and the result serves as the `control_allocation
<LCControlMechanism.control_allocation>` for the LCControlMechanism. All of the parameters of the
`FitzHughNagumoIntegrator` function are accessible as attributes of the LCControlMechanism.

.. _LCControlMechanism_Modes_Of_Operation:

LC Modes of Operation
^^^^^^^^^^^^^^^^^^^^^

The `mode <FitzHughNagumoIntegrator.mode>` parameter of the LCControlMechanism's `FitzHughNagumoIntegrator` Function
regulates its operation between `"tonic" and "phasic" modes <https://www.ncbi.nlm.nih.gov/pubmed/8027789>`_:

  * in the *tonic mode* (low value of `mode <FitzHughNagumoIntegrator.mode>`), the output of the LCControlMechanism is
    moderately low and constant; that is, it is relatively unaffected by its `input <LCControlMechanism_Input`.
    This blunts the response of the Mechanisms that the LCControlMechanism controls to their inputs.

  * in the *phasic mode* (high value of `mode <FitzHughNagumoIntegrator.mode>`), when the `input to the
    LCControlMechanism <LCControlMechanism_Input>` is low, its `output <LCControlMechanism_Output>` is even lower
    than when it is in the tonic regime, and thus the response of the Mechanisms it controls to their outputs is even
    more blunted.  However, when the LCControlMechanism's input rises above a certain value (determined by the
    `threshold <LCControlMechanism.threshold>` parameter), its output rises sharply generating a "phasic response",
    and inducing a much sharper response of the Mechanisms it controls to their inputs.

.. _LCControlMechanism_Output:

*Output*
~~~~~~~~

An LCControlMechanism has a single `ControlSignal`, that uses its `control_allocation
<LCControlMechanism.control_allocation>` (the scalar value generated by its `function <LCControlMechanism.function>`)
to modulate the function of the Mechanism(s) it controls.  The ControlSignal is assigned a `ControlProjection` to the
`ParameterPort` for the `multiplicative_param <Function_Modulatory_Params>` of the `function
<Mechanism_Base.function>` for each of those Mechanisms.  The Mechanisms modulated by an LCControlMechanism are listed
in its `modulated_mechanisms <LCControlMechanism.modulated_mechanisms>` attribute) and can be displayed using its
:func:`show <LCControlMechanism.show>` method.

COMMENT:
VERSION FOR MULTIPLE CONTROL SIGNALS
An LCControlMechanism has a `ControlSignal` for each Mechanism listed in its `modulated_mechanisms
<LCControlMechanism.modulated_mechanisms>` attribute.  All of its ControlSignals are assigned the same value:  the
result of the LCControlMechanism's `function <LCControlMechanism.function>`.  Each ControlSignal is assigned a
`ControlProjection` to the `ParameterPort` for the  `multiplicative_param <Function_Modulatory_Params>` of `function
<Mechanism_Base.function>` for the Mechanism in `modulated_mechanisms <LCControlMechanism.modulate_mechanisms>` to
which it corresponds. The Mechanisms modulated by an LCControlMechanism can be displayed using its :func:`show
<LCControlMechanism.show>` method.
COMMENT

.. _LCControlMechanism_Execution:

Execution
---------

An LCControlMechanism executes within a `Composition` at a point specified in the Composition's `Scheduler` or, if it
is the `controller <Composition.controller>` for a `Composition`, after all of the other Mechanisms in the Composition
have `executed <Composition_Execution>` in a `TRIAL <TimeScale.TRIAL>`. It's `function <LCControlMechanism.function>`
takes the `value <InputPort.value>` of the LCControlMechanism's `primary InputPort <InputPort_Primary>` as its input,
and generates a response -- under the influence of its `mode <FitzHughNagumoIntegrator.mode>` parameter -- that is
assigned as the `allocation <LCControlSignal.allocation>` of its `ControlSignals <ControlSignal>`.  The latter are
used by its `ControlProjections <ControlProjection>` to modulate the response -- in the next `TRIAL <TimeScale.TRIAL>`
of execution -- of the Mechanisms the LCControlMechanism controls.

.. note::
   A `ParameterPort` that receives a `ControlProjection` does not update its value until its owner Mechanism executes
   (see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of "lazy" updating).  This means that even if a
   LCControlMechanism has executed, the `multiplicative_param <Function_Modulatory_Params>` parameter of the `function
   <Mechanism_Base.function>` of a Mechanism that it controls will not assume its new value until that Mechanism has
   executed.

.. _LCControlMechanism_Examples:

Examples
--------

The following example generates an LCControlMechanism that modulates the function of two TransferMechanisms, one that
uses a `Linear` function and the other a `Logistic` function::

    >>> import psyneulink as pnl
    >>> my_mech_1 = pnl.TransferMechanism(function=pnl.Linear,
    ...                                   name='my_linear_mechanism')
    >>> my_mech_2 = pnl.TransferMechanism(function=pnl.Logistic,
    ...                                   name='my_logistic_mechanism')

    >>> LC = LCControlMechanism(modulated_mechanisms=[my_mech_1, my_mech_2],
    ...                         name='my_LC')

COMMENT:
# Calling `LC.show()` generates the following report::
#
#     >>> LC.show()
#     <BLANKLINE>
#     ---------------------------------------------------------
#     <BLANKLINE>
#     my_LC
#     <BLANKLINE>
#       Monitoring the following Mechanism OutputPorts:
#     <BLANKLINE>
#       Modulating the following parameters:
#         my_logistic_mechanism: gain
#         my_linear_mechanism: slope
#     <BLANKLINE>
#     ---------------------------------------------------------
COMMENT

Calling `LC.show()` generates the following report::

    my_LC

      Monitoring the following Mechanism OutputPorts:

      Modulating the following parameters:
        my_logistic_mechanism: gain
        my_linear_mechanism: slope

Note that the LCControlMechanism controls the `multiplicative_param <Function_Modulatory_Params>` of the `function
<Mechanism_Base.function>` of each Mechanism:  the `gain <Logistic.gain>` parameter for ``my_mech_1``, since it uses
a `Logistic` Function; and the `slope <Linear.slope>` parameter for ``my_mech_2``, since it uses a `Linear` Function.

COMMENT:
ADDITIONAL EXAMPLES HERE OF THE DIFFERENT FORMS OF SPECIFICATION FOR
**monitor_for_control** and **modulated_mechanisms**

STRUCTURE:
MODE INPUT_PORT <- NAMED ONE, LAST?
SIGNAL INPUT_PORT(S) <- PRIMARY;  MUST BE FROM PROCESSING MECHANISMS
CONTROL SIGNALS
COMMENT

.. _LCControlMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import FitzHughNagumoIntegrator
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.keywords import \
    ALL, CONTROL, CONTROL_PROJECTIONS, FUNCTION, INIT_EXECUTE_METHOD_ONLY, \
    MULTIPLICATIVE, MULTIPLICATIVE_PARAM, PROJECTIONS
from psyneulink.core.globals.parameters import Parameter, ParameterAlias
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_iterable, convert_to_list

__all__ = [
    'CONTROL_SIGNAL_NAME', 'ControlMechanismRegistry', 'LCControlMechanism', 'LCControlMechanismError',
    'MODULATED_MECHANISMS',
]

MODULATED_MECHANISMS = 'modulated_mechanisms'
CONTROL_SIGNAL_NAME = 'LCControlMechanism_ControlSignal'

ControlMechanismRegistry = {}

class LCControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


class LCControlMechanism(ControlMechanism):
    """
    LCControlMechanism(                           \
        modulated_mechanisms=None,                \
        initial_w_FitzHughNagumo=0.0,             \
        initial_v_FitzHughNagumo=0.0,             \
        time_step_size_FitzHughNagumo=0.05,       \
        t_0_FitzHughNagumo=0.0,                   \
        a_v_FitzHughNagumo=-1/3,                  \
        b_v_FitzHughNagumo=0.0,                   \
        c_v_FitzHughNagumo=1.0,                   \
        d_v_FitzHughNagumo=0.0,                   \
        e_v_FitzHughNagumo=-1.0,                  \
        f_v_FitzHughNagumo=1.0,                   \
        threshold_FitzHughNagumo=-1.0             \
        time_constant_v_FitzHughNagumo=1.0,       \
        a_w_FitzHughNagumo=1.0,                   \
        b_w_FitzHughNagumo=-0.8,                  \
        c_w_FitzHughNagumo=0.7,                   \
        mode_FitzHughNagumo=1.0,                  \
        uncorrelated_activity_FitzHughNagumo=0.0  \
        time_constant_w_FitzHughNagumo = 12.5,    \
        integration_method="RK4"                  \
        base_level_gain=0.5,                      \
        scaling_factor_gain=3.0)

    Subclass of `ControlMechanism` that modulates the `multiplicative_param <Function_Modulatory_Params>` of the
    `function <Mechanism_Base.function>` of one or more `Mechanisms <Mechanism>`.
    See `ControlMechanism <ControlMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    modulated_mechanisms : List[`Mechanism <Mechanism>`] or *ALL*
        specifies the Mechanisms to be modulated by the LCControlMechanism. If it is a list, every item must be a
        Mechanism with a `function <Mechanism_Base.function>` that implements a `multiplicative_param
        <Function_Modulatory_Params>`;  alternatively the keyword *ALL* can be used to specify all of the
        `ProcessingMechanisms <ProcessingMechanism>` in the Composition(s) to which the LCControlMechanism  belongs.

    initial_w_FitzHughNagumo : float : default 0.0
        sets `initial_w <initial_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    initial_v_FitzHughNagumo : float : default 0.0
        sets `initial_v <initial_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    time_step_size_FitzHughNagumo : float : default 0.0
        sets `time_step_size <time_step_size.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    t_0_FitzHughNagumo : float : default 0.0
        sets `t_0 <t_0.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    a_v_FitzHughNagumo : float : default -1/3
        sets `a_v <a_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    b_v_FitzHughNagumo : float : default 0.0
        sets `b_v <b_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    c_v_FitzHughNagumo : float : default 1.0
        sets `c_v <c_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    d_v_FitzHughNagumo : float : default 0.0
        sets `d_v <d_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    e_v_FitzHughNagumo : float : default -1.0
        sets `e_v <e_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    f_v_FitzHughNagumo : float : default 1.0
        sets `f_v <f_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    threshold_FitzHughNagumo : float : default -1.0
        sets `threshold <threshold.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    time_constant_v_FitzHughNagumo : float : default 1.0
        sets `time_constant_w <time_constant_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    a_w_FitzHughNagumo : float : default 1.0
        sets `a_w <a_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    b_w_FitzHughNagumo : float : default -0.8,
        sets `b_w <b_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    c_w_FitzHughNagumo : float : default 0.7
        sets `c_w <c_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    mode_FitzHughNagumo : float : default 1.0
        sets `mode <mode.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    uncorrelated_activity_FitzHughNagumo : float : default 0.0
        sets `uncorrelated_activity <uncorrelated_activity.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    time_constant_w_FitzHughNagumo  : float : default  12.5
        sets `time_constant_w <time_constant_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    integration_method : float : default "RK4"
        sets `integration_method <integration_method.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    base_level_gain : float : default 0.5
        sets the base value in the equation used to compute the time-dependent gain value that the LCControl applies
        to each of the mechanisms it modulates

        .. math::

            g(t) = G + k w(t)

        base_level_gain = G

    scaling_factor_gain : float : default 3.0
        sets the scaling factor in the equation used to compute the time-dependent gain value that the LCControl
        applies to each of the mechanisms it modulates

        .. math::

            g(t) = G + k w(t)

        scaling_factor_gain = k


    Attributes
    ----------

    monitor_for_control : List[OutputPort]
        list of the `OutputPorts <OutputPort>` that project to `objective_mechanism
        <LCControlMechanism.objective_mechanism>` (and also listed in the ObjectiveMechanism's `monitored_output_ports
        <ObjectiveMechanism.monitored_output_ports>` attribute);  these are used by the ObjectiveMechanism to
        generate the ControlMechanism's `input <ControlMechanism_Input>`, which drives the `phasic response
        <LCControlMechanism_Modes_Of_Operation>` of its `function <LControlMechanism.function>`.

    monitored_output_ports_weights_and_exponents : List[Tuple(float, float)]
        each tuple in the list contains the weight and exponent associated with a corresponding item of
        `monitored_output_ports <LCControlMechanism.monitored_output_ports>`;  these are the same as those in
        the `monitored_output_ports_weights_and_exponents
        <ObjectiveMechanism.monitored_output_ports_weights_and_exponents>` attribute of the `objective_mechanism
        <LCControlMechanism.objective_mechanism>`, and are used by the ObjectiveMechanism's `function
        <ObjectiveMechanism.function>` to parametrize the contribution made to its output by each of the values that
        it monitors (see `ObjectiveMechanism Function <ObjectiveMechanism_Function>`).

    function : FitzHughNagumoIntegrator
        takes the LCControlMechanism's `input <LCControlMechanism_Input>` and generates its response
        <LCControlMechanism_Output>` under
        the influence of the `FitzHughNagumoIntegrator` Function's `mode <FitzHughNagumoIntegrator.mode>` attribute
        (see `LCControlMechanism_Function` for additional details).

    control_allocation : 2d np.array
        contains a single item — the result of the LCControlMechanism's `function <LCControlMechanism.function>` —
        that is assigned as the `allocation <ControlSignal.allocation>` for the LCControlMechanism's single
        `ControlSignal`, listed in its `control_signals` attribute;  the control_allocation is the same as the
        ControlMechanism's `value <Mechanism_Base.value>` attribute).

    control_signals : List[ControlSignal]
        contains the LCControlMechanism's single `ControlSignal`, which sends `ControlProjections
        <ControlProjection>` to the `multiplicative_param <Function_Modulatory_Params>` of each of the Mechanisms
        listed in the LCControlMechanism's `modulated_mechanisms <LCControlMechanism.modulated_mechanisms>`
        attribute.

    control_projections : List[ControlProjection]
        list of `ControlProjections <ControlProjection>` sent by the LCControlMechanism's `ControlSignal`, each of
        which projects to the `ParameterPort` for the `multiplicative_param <Function_Modulatory_Params>` of the
        `function <Mechanism_Base.function>` of one of the Mechanisms listed in `modulated_mechanisms
        <LCControlMechanism.modulated_mechanisms>` attribute.

    modulated_mechanisms : List[Mechanism]
        list of `Mechanisms <Mechanism>` modulated by the LCControlMechanism.

        initial_w_FitzHughNagumo : float : default 0.0
        sets `initial_w <initial_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    initial_v_FitzHughNagumo : float : default 0.0
        sets `initial_v <initial_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    time_step_size_FitzHughNagumo : float : default 0.0
        sets `time_step_size <time_step_size.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator <FitzHughNagumoIntegrator>` function

    t_0_FitzHughNagumo : float : default 0.0
        sets `t_0 <t_0.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    a_v_FitzHughNagumo : float : default -1/3
        sets `a_v <a_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    b_v_FitzHughNagumo : float : default 0.0
        sets `b_v <b_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    c_v_FitzHughNagumo : float : default 1.0
        sets `c_v <c_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    d_v_FitzHughNagumo : float : default 0.0
        sets `d_v <d_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    e_v_FitzHughNagumo : float : default -1.0
        sets `e_v <e_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    f_v_FitzHughNagumo : float : default 1.0
        sets `f_v <f_v.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    threshold_FitzHughNagumo : float : default -1.0
        sets `threshold <threshold.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    time_constant_v_FitzHughNagumo : float : default 1.0
        sets `time_constant_w <time_constant_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    a_w_FitzHughNagumo : float : default 1.0
        sets `a_w <a_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    b_w_FitzHughNagumo : float : default -0.8,
        sets `b_w <b_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    c_w_FitzHughNagumo : float : default 0.7
        sets `c_w <c_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    mode_FitzHughNagumo : float : default 1.0
        sets `mode <mode.FitzHughNagumoIntegrator>` on the LCControlMechanism's `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    uncorrelated_activity_FitzHughNagumo : float : default 0.0
        sets `uncorrelated_activity <uncorrelated_activity.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    time_constant_w_FitzHughNagumo  : float : default  12.5
        sets `time_constant_w <time_constant_w.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    integration_method : float : default "RK4"
        sets `integration_method <integration_method.FitzHughNagumoIntegrator>` on the LCControlMechanism's
        `FitzHughNagumoIntegrator
        <FitzHughNagumoIntegrator>` function

    base_level_gain : float : default 0.5
        sets the base value in the equation used to compute the time-dependent gain value that the LCControl applies
        to each of the mechanisms it modulates

        .. math::

            g(t) = G + k w(t)

        base_level_gain = G

    scaling_factor_gain : float : default 3.0
        sets the scaling factor in the equation used to compute the time-dependent gain value that the LCControl
        applies to each of the mechanisms it modulates

        .. math::

            g(t) = G + k w(t)

        scaling_factor_gain = k

    """

    componentType = "LCControlMechanism"

    initMethod = INIT_EXECUTE_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ControlMechanismClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(ControlMechanism.Parameters):
        """
            Attributes
            ----------

                base_level_gain
                    see `base_level_gain <LCControlMechanism.base_level_gain>`

                    :default value: 0.5
                    :type: ``float``

                function
                    see `function <LCControlMechanism_Function>`

                    :default value: `FitzHughNagumoIntegrator`
                    :type: `Function`

                modulated_mechanisms
                    see `modulated_mechanisms <LCControlMechanism_Modulated_Mechanisms>`

                    :default value: None
                    :type:

                scaling_factor_gain
                    see `scaling_factor_gain <LCControlMechanism.scaling_factor_gain>`

                    :default value: 3.0
                    :type: ``float``
        """
        function = Parameter(FitzHughNagumoIntegrator, stateful=False, loggable=False)

        base_level_gain = Parameter(0.5, modulable=True)
        scaling_factor_gain = Parameter(3.0, modulable=True)

        modulated_mechanisms = Parameter(None, stateful=False, loggable=False)

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 objective_mechanism:tc.optional(tc.any(ObjectiveMechanism, list))=None,
                 monitor_for_control:tc.optional(tc.any(is_iterable, Mechanism, OutputPort))=None,
                 # modulated_mechanisms:tc.optional(tc.optional(tc.any(list,str))) = None,
                 modulated_mechanisms=None,
                 modulation:tc.optional(str)=None,
                 integration_method="RK4",
                 initial_w_FitzHughNagumo=0.0,
                 initial_v_FitzHughNagumo=0.0,
                 time_step_size_FitzHughNagumo=0.05,
                 t_0_FitzHughNagumo=0.0,
                 a_v_FitzHughNagumo=-1 / 3,
                 b_v_FitzHughNagumo=0.0,
                 c_v_FitzHughNagumo=1.0,
                 d_v_FitzHughNagumo=0.0,
                 e_v_FitzHughNagumo=-1.0,
                 f_v_FitzHughNagumo=1.0,
                 time_constant_v_FitzHughNagumo=1.0,
                 a_w_FitzHughNagumo=1.0,
                 b_w_FitzHughNagumo=-0.8,
                 c_w_FitzHughNagumo=0.7,
                 threshold_FitzHughNagumo=-1.0,
                 time_constant_w_FitzHughNagumo=12.5,
                 mode_FitzHughNagumo=1.0,
                 uncorrelated_activity_FitzHughNagumo=0.0,
                 base_level_gain=None,
                 scaling_factor_gain=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None
                 ):

        super().__init__(
            default_variable=default_variable,
            objective_mechanism=objective_mechanism,
            monitor_for_control=monitor_for_control,
            function=FitzHughNagumoIntegrator(
                integration_method=integration_method,
                initial_v=initial_v_FitzHughNagumo,
                initial_w=initial_w_FitzHughNagumo,
                time_step_size=time_step_size_FitzHughNagumo,
                t_0=t_0_FitzHughNagumo,
                a_v=a_v_FitzHughNagumo,
                b_v=b_v_FitzHughNagumo,
                c_v=c_v_FitzHughNagumo,
                d_v=d_v_FitzHughNagumo,
                e_v=e_v_FitzHughNagumo,
                f_v=f_v_FitzHughNagumo,
                time_constant_v=time_constant_v_FitzHughNagumo,
                a_w=a_w_FitzHughNagumo,
                b_w=b_w_FitzHughNagumo,
                c_w=c_w_FitzHughNagumo,
                threshold=threshold_FitzHughNagumo,
                mode=mode_FitzHughNagumo,
                uncorrelated_activity=uncorrelated_activity_FitzHughNagumo,
                time_constant_w=time_constant_w_FitzHughNagumo,
            ),
            modulated_mechanisms=modulated_mechanisms,
            modulation=modulation,
            base_level_gain=base_level_gain,
            scaling_factor_gain=scaling_factor_gain,
            params=params,
            name=name,
            prefs=prefs
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate modulated_mechanisms argument.

        Validate that **modulated_mechanisms** is either a Composition or a list of eligible Mechanisms .
        Eligible Mechanisms are ones with a `function <Mechanism_Base>` that has a multiplicative_param.

        """

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if MODULATED_MECHANISMS in target_set and target_set[MODULATED_MECHANISMS]:
            spec = target_set[MODULATED_MECHANISMS]

            from psyneulink.core.compositions.composition import Composition
            if isinstance(spec, Composition):
                pass
            else:
                if not isinstance(spec, list):
                    spec = [spec]
                    for mech in spec:
                        if not isinstance(mech, Mechanism):
                            raise LCControlMechanismError("The specification of the {} argument for {} "
                                                          "contained an item ({}) that is not a Mechanism.".
                                                          format(repr(MODULATED_MECHANISMS), self.name, mech))
                        elif not hasattr(mech.function, MULTIPLICATIVE_PARAM):
                            raise LCControlMechanismError(f"The specification of the {repr(MODULATED_MECHANISMS)} "
                                                          f"argument for {self.name} contained a Mechanism ({mech}) "
                                                          f"that does not have a {repr(MULTIPLICATIVE_PARAM)}.")

    def _instantiate_output_ports(self, context=None):
        """Instantiate ControlSignals and assign ControlProjections to Mechanisms in self.modulated_mechanisms

        If **modulated_mechanisms** argument of constructor was specified as *ALL*, assign all ProcessingMechanisms
           in Compositions to which LCControlMechanism belongs to self.modulated_mechanisms.
        Instantiate ControlSignal with Projection to the ParameterPort for the multiplicative_param of every
           Mechanism listed in self.modulated_mechanisms.
        """
        from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base

        # A Composition is specified for modulated_mechanisms, so assign all Processing Mechanisms in composition
        #     to its modulated_mechanisms attribute
        from psyneulink.core.compositions.composition import Composition, NodeRole
        if isinstance(self.modulated_mechanisms, Composition):
            comp = self.modulated_mechanisms
            self.modulated_mechanisms = []
            for mech in [m for m in comp.nodes if (isinstance(m, ProcessingMechanism_Base) and
                                                   not (isinstance(m, ObjectiveMechanism)
                                                        and comp.get_roles_for_node(m) != NodeRole.CONTROL)
                                                   and hasattr(m.function, MULTIPLICATIVE_PARAM))]:
                self.modulated_mechanisms.append(mech)

        # Get the name of the multiplicative_param of each Mechanism in self.modulated_mechanisms
        if self.modulated_mechanisms:
            # Create (param_name, Mechanism) specification for **control_signals** argument of ControlSignal constructor
            self.modulated_mechanisms = convert_to_list(self.modulated_mechanisms)
            multiplicative_param_names = []
            for mech in self.modulated_mechanisms:
                if isinstance(mech.function.parameters.multiplicative_param, ParameterAlias):
                    multiplicative_param_names.append(mech.function.parameters.multiplicative_param.source.name)
                else:
                    multiplicative_param_names.append(mech.function.parameters.multiplicative_param.name)
            ctl_sig_projs = []
            for mech, mult_param_name in zip(self.modulated_mechanisms, multiplicative_param_names):
                ctl_sig_projs.append((mult_param_name, mech))
            self.parameters.control._set([{PROJECTIONS: ctl_sig_projs}], context)
            self.parameters.control_allocation.default_value = self.value[0]

        super()._instantiate_output_ports(context=context)
        self.aux_components.extend(self.control_projections)

    def _check_for_composition(self, context=None):
        # FIX 5/17/20:
        #  IN _instantiate_output_ports:
        #     BREAK OUT NEW METHOD:  _add_modulated_mechanisms
        #     IF modulated_mechanisms IS SPECIFICED AS A COMPOSITION, SET FLAG;
        #  HERE:
        #     IF FLAG IS FOUND HERE, GO THROUGH ALL MECHANISMS OF COMP AGAIN AND ADD ANY THAT ARE NOT ARLREADY ASSIGNED
        #     ONLY NOW, CALL COMPOSITION TO CREATE THE PROJECTIONS
        pass

    def _execute(
        self,
        variable=None,
        context=None,
        runtime_params=None,

    ):
        """Updates LCControlMechanism's ControlSignal based on input and mode parameter value
        """
        # IMPLEMENTATION NOTE:  skip ControlMechanism._execute since it is a stub method that returns input_values
        output_values = super(ControlMechanism, self)._execute(
            variable=variable,
            context=context,
            runtime_params=runtime_params,

        )

        gain_t = self.parameters.scaling_factor_gain._get(context) * output_values[1] \
                 + self.parameters.base_level_gain._get(context)

        return gain_t, output_values[0], output_values[1], output_values[2]

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, state, variable, *, tags:frozenset):
        assert function is self.function
        mf_out, builder = super()._gen_llvm_invoke_function(ctx, builder, function, params, state, variable, tags=tags)

        # prepend gain type (matches output[1] type)
        gain_ty = mf_out.type.pointee.elements[1]
        elements = gain_ty, *mf_out.type.pointee.elements
        elements_ty = pnlvm.ir.LiteralStructType(elements)

        # allocate new output type
        new_out = builder.alloca(elements_ty)

        # Load mechanism parameters
        params = builder.function.args[0]
        scaling_factor_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                         "scaling_factor_gain")
        base_factor_ptr = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                      "base_level_gain")
        # If modulated, scaling factor is a single element array
        if isinstance(scaling_factor_ptr.type.pointee, pnlvm.ir.ArrayType):
            assert len(scaling_factor_ptr.type.pointee) == 1
            scaling_factor_ptr = builder.gep(scaling_factor_ptr,
                                             [ctx.int32_ty(0), ctx.int32_ty(0)])
        # If modulated, base factor is a single element array
        if isinstance(base_factor_ptr.type.pointee, pnlvm.ir.ArrayType):
            assert len(base_factor_ptr.type.pointee) == 1
            base_factor_ptr = builder.gep(base_factor_ptr,
                                          [ctx.int32_ty(0), ctx.int32_ty(0)])
        scaling_factor = builder.load(scaling_factor_ptr)
        base_factor = builder.load(base_factor_ptr)

        # Apply to the entire vector
        vi = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(1)])
        vo = builder.gep(new_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        with pnlvm.helpers.array_ptr_loop(builder, vi, "LC_gain") as (b1, index):
            in_ptr = b1.gep(vi, [ctx.int32_ty(0), index])
            val = b1.load(in_ptr)
            val = b1.fmul(val, scaling_factor)
            val = b1.fadd(val, base_factor)

            out_ptr = b1.gep(vo, [ctx.int32_ty(0), index])
            b1.store(val, out_ptr)

        # copy the main function return value
        for i, _ in enumerate(mf_out.type.pointee.elements):
            ptr = builder.gep(mf_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
            out_ptr = builder.gep(new_out, [ctx.int32_ty(0), ctx.int32_ty(i + 1)])
            val = builder.load(ptr)
            builder.store(val, out_ptr)

        return new_out, builder

    # 5/8/20: ELIMINATE SYSTEM
    # SEEMS TO STILL BE USED BY SOME MODELS;  DELETE WHEN THOSE ARE UPDATED
    # @tc.typecheck
    # def _add_system(self, system, role:str):
    #     super()._add_system(system, role)
    #     if isinstance(self.modulated_mechanisms, str) and self.modulated_mechanisms == ALL:
    #         # Call with ContextFlags.COMPONENT so that OutputPorts are replaced rather than added
    #         self._instantiate_output_ports(context=Context(source=ContextFlags.COMPONENT))

    @tc.typecheck
    def add_modulated_mechanisms(self, mechanisms:list):
        """Add ControlProjections to the specified Mechanisms.
        """

        request_set = {MODULATED_MECHANISMS:mechanisms}
        target_set = {}
        self._validate_params(request_set=request_set, target_set=target_set)

        # Assign ControlProjection from the LCControlMechanism's ControlSignal
        #    to the ParameterPort for the multiplicative_param of each Mechanism in mechanisms
        for mech in mechanisms:
            self.modulated_mechanisms.append(mech)
            parameter_port = mech._parameter_ports[mech.multiplicative_param]
            ControlProjection(sender=self.control_signals[0],
                              receiver=parameter_port)
            # self.aux_components.append(ControlProjection(sender=self.control_signals[0],
            #                                              receiver=parameter_port))

    @tc.typecheck
    def remove_modulated_mechanisms(self, mechanisms:list):
        """Remove the ControlProjections to the specified Mechanisms.
        """

        for mech in mechanisms:
            if mech not in self.modulated_mechanisms:
                continue

            parameter_port = mech._parameter_ports[mech.multiplicative_param]

            # Get ControlProjection
            for projection in parameter_port.mod_afferents:
                if projection.sender.owner is self:
                    control_projection = projection
                    break

            # Delete ControlProjection ControlSignal's list of efferents
            index = self.control_signals[0].efferents[control_projection]
            del(self.control_signals[0].efferents[index])

            # Delete ControlProjection from recipient ParameterPort
            index = parameter_port.mod_afferents[control_projection]
            del(parameter_port.mod_afferents[index])

            # Delete Mechanism from self.modulated_mechanisms
            index = self.modulated_mechanisms.index(mech)
            del(self.modulated_mechanisms[index])

    def show(self):
        """Display the `OutputPorts <OutputPort>` monitored by the LCControlMechanism's `objective_mechanism`
        and the `multiplicative_params <Function_Modulatory_Params>` modulated by the LCControlMechanism.
        """

        print("\n---------------------------------------------------------")

        print("\n{0}".format(self.name))
        print("\n\tMonitoring the following Mechanism OutputPorts:")
        if self.objective_mechanism is None:
            print("\t\tNone")
        else:
            for port in self.objective_mechanism.input_ports:
                for projection in port.path_afferents:
                    monitored_port = projection.sender
                    monitored_port_Mech = projection.sender.owner
                    monitored_port_index = self.monitored_output_ports.index(monitored_port)

                    weight = self.monitored_output_ports_weights_and_exponents[monitored_port_index][0]
                    exponent = self.monitored_output_ports_weights_and_exponents[monitored_port_index][1]

                    print("\t\t{0}: {1} (exp: {2}; wt: {3})".
                          format(monitored_port_Mech.name, monitored_port.name, weight, exponent))

        print("\n\tModulating the following parameters:".format(self.name))
        # Sort for consistency of output:
        port_Names_sorted = sorted(self.output_ports.names)
        for port_Name in port_Names_sorted:
            for projection in self.output_ports[port_Name].efferents:
                print("\t\t{0}: {1}".format(projection.receiver.owner.name, projection.receiver.name))

        print("\n---------------------------------------------------------")
