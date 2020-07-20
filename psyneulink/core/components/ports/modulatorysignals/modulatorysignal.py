# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ModulatorySignal *******************************************************

"""

Contents
--------
  * `ModulatorySignal_Overview`
  * `ModulatorySignal_Creation`
  * `ModulatorySignal_Structure`
      - `ModulatorySignal_Projections`
      - `ModulatorySignal_Modulation`
          • `ModulatorySignal_Uses`
          • `ModulatorySignal_Types`
          • `ModulatorySignal_Implementation`
  * `ModulatorySignal_Execution`
  COMMENT:
  * `ModulatorySignal_Examples`
  COMMENT
  * `ModulatorySignal_Class_Reference`


.. _ModulatorySignal_Overview:

Overview
--------

A ModulatorySignal is a subclas of `OutputPort` that belongs to a `ModulatoryMechanism <ModulatoryMechanism>`, and is
used to `modulate <ModulatorySignal_Modulation>` the `value <Port_Base.value>` of one or more `Ports <Port>` by way of
one or more `ModulatoryProjctions <ModulatoryProjection>`. A ModulatorySignal modulates the value of a Port by modifying
a  parameter of that Port's `function <Port_Base.function>`.  There are three types of ModulatorySignals, each of which
is  associated wth a particular type of `ModulatoryMechanism <ModulatoryMechanism>` and `ModulatoryProjection
<ModulatoryProjection>`, and modifies the value of different types of Ports, as summarized `below:

* `ControlSignal`
    takes the `allocation <ControlSignal.allocation>` assigned to it by the `function <ControlMechanism.function>`
    of the `ControlMechanism <ControlMechanism>` to which it belongs, and uses it to modulate the parameter of a
    `Mechanism <Mechanism>` or its `function <Mechanism_Base.function>` (and thereby the `value
    <Mechanism_Base.value>` of that Mechanism), or a parameter of the `function <Port_Base.function>` one of the
    Mechanism's `InputPorts <InputPort>` or `OutputPorts <OutputPort>` (and thereby the `value <Port_Base.value>`
    of the corresponding Port).
..
* `GatingSignal` takes the `allocation <GatingSignal.allocation>` assigned to it by the `function
    <GatingMechanism.function>` of the `GatingMechanism` to which it belongs, and uses it to modulate the parameter
    of the `function <Port_Base.function>` of an `InputPort` or `OutputPort` (and hence that Port's `value
    <Port_Base.value>`).  A GatingMechanism and GatingSignal can be thought of as implementing a form of control
    specialized for gating the input to and/or output of a Mechanism.
..
* `LearningSignal`
    takes the `learning_signal <LearningMechanism>` calculated by the `function <LearningMechanism.function>` of the
    `LearningMechanism` to which it belongs, and uses it to modulate the `matrix <MappingProjection.matrix>` parameter
    of a `MappingProjection`.

These are shown in a `figure <ModulatorySignal_Anatomy_Figure>` below, and are descdribed in greater detail in
in the sections under `ModulatorySignal_Structure`.

See `ModulatoryMechanism <ModulatoryMechanism_Naming>` for conventions used for the names of Modulatory components.

.. _ModulatorySignal_Creation:

Creating a ModulatorySignal
---------------------------

ModulatorySignal is a base class, and cannot be instantiated directly.  However, the three types of ModulatorySignals
listed above can be created directly, by calling the constructor for the desired type.  More commonly, however,
ModulatorySignals are created automatically by the `ModulatoryMechanism <ModulatoryMechanism>` to which they belong, or
by specifying them in the constructor for a `ModulatoryMechanism <ModulatoryMechanism>` (the details of which are
described in the documentation for each type of ModulatorySignal).  If a ModulatorySignal is constructed explicitly,
the type of modulation it uses is specifed in the **modulation** argument of its constructor, using a 2-item tuple
that contains the Port to be modulated as the first item, and either the name of the parameter of the Port's `function
<Port_Base.function>` to be modulated, or a keyword specifying the type of modulation, as the second item (see
`ModulatorySignal_Types` for additional details).

.. _ModulatorySignal_Structure:

Structure
---------

A ModulatorySignal is always assigned to a `ModulatoryMechanism <ModulatoryMechanism>`, and must be assigned to an
ModulatoryMechanism of the appropriate type (see `Types of ModulatoryMechanism <ModulatoryMechanism_Types>`).  The
ModulatorySignal receives an `allocation <ModulatorySignal.allocation>` from the ModulatoryMechanism to which it is
assigned, that it uses as the `variable <Function_Base.variable>` for its `function <Mechanism_Base.function>`, the
result of which is the modulatory `value <ModulatorySignal.value>` of the ModulatorySignal.  A ModulatorySignal is
associated with one or more `ModulatoryProjections <ModulatoryProjection>` of the corresponding type, that receive
the ModulatorySignal's `value <ModulatorySignal.value>`, and use this to modulate the Port(s) to which they project.
All of the ModulatoryProjections from a given ModulatorySignal are assigned the same modulatory `value
<ModulatorySignal.value>` (see `ModulatorySignal_Projections` below) and use the same `type of modulation
<ModulatorySignal_Types>` specified by the the ModulatorySignal's `modulation <ModulatorySignal.modulation>` attribute.
The ModulatoryProjections received by a `Port <Port>` are listed in the Port's `mod_afferents <Port_Base.mod_afferents>`
attribute.

The section on `Modulation` below provides a comparison of ModulatorySignal `subclasses and their uses
<ModulatorySignal_Uses>` (summarized in an accompanying `table <ModulatorySignal_Table>` and `figure
<ModulatorySignal_Anatomy_Figure>`), as well as a description of the different `types of modulation
<ModulatorySignal_Types>` and a more detailed description of `how modulation is implemented
<ModulatorySignal_Implementation>`.

.. _ModulatorySignal_Projections:

*ModulatoryProjections*
~~~~~~~~~~~~~~~~~~~~~~~

A ModulatorySignal can be assigned one or more `ModulatoryProjections <ModulatoryProjection>`,
using either the **projections** argument of its constructor, or in an entry of a dictionary assigned to the
**params** argument with the key *PROJECTIONS*.  These are assigned to its `efferents  <ModulatorySignal.efferents>`
attribute.  See `Port Projections <Port_Projections>` for additional details concerning the specification of
Projections when creating a Port.

Although a ModulatorySignal can be assigned more than one `ModulatoryProjection <ModulatoryProjection>`,
all of those Projections receive and convey the same modulatory `value <ModulatorySignal.value>` from the
ModulatorySignal, and use the same form of `modulation <ModulatorySignal_Modulation>`.  This is a common use for some
ModulatorySignals (e.g., the use of a single `GatingSignal` to gate multiple `InputPort(s) <InputPort>` or
`OutputPort(s) <OutputPort>`), but requires more specialized circumstances for others (e.g., the use of a single
`LearningSignal` for more than one `MappingProjection`, or a single `ControlSignal` for the parameters of more than
one Mechanism or function).

.. _ModulatorySignal_Modulation:

*Modulation*
~~~~~~~~~~~~

A ModulatorySignal modulates the value of a `Port <Port>` either by modifying a parameter of the Port's `function
<Port_Base.function>` (which determines the Port's `value <Port_Base.value>`), or by  assigning a value to the Port
directly.  The `type of modulation <ModulatorySignal_Types>` is determined by the ModulatorySignal's
`modulation <ModulatorySignal.modulation>` attribute, which can be specified in the **modulation** argument of its
ModulatorySignal's constructor, or in a *MODULATION* entry of a `Port specification dictionary
<Port_Specification>` used to create the ModulatorySignal (see `Type of Modualtion <ModulatorySignal_Types>` and
`figure <ModulatorySignal_Detail_Figure>` below for details). If the type of `modulation <ModulatorySignal.modulation>`
is not specified when a ModulatorySignal is created, it is assigned the value of the `modulation
<ModulatoryMechanism_Base.modulation>` attribute for the `ModulatoryMechanism <ModulatoryMechanism>` to which it
belongs.

.. _ModulatorySignal_Uses:

Uses of Modulation
^^^^^^^^^^^^^^^^^^

There are three broad types of modulation that serve different purposes, and differ according to the
ModulatorySignals used and the type of Port modulated;  these are modulation of a:

  * `Mechanism's  <Mechanism>` `function <Mechanism_Base.function>`
        a `ControlSignal` must be used; this modulates the `value <ParameterPort.value>` of `ParameterPort` for a
        parameter of the Mechanism's `function <Mechanism_Base.function>` which, in turn, determines how it computes
        the Mechanism's `value <Mechanism_Base.value>`;

  * `Mechanism's <Mechanism>` input or output
        a `GatingSignal` is specialized for this purpose, though a `ControlSignal` can also be used.  These modulate
        either the value <InputPort.value>` an `InputPort` of the Mechanism, that determines the Mechanism's `variable
        <Mechanism_Base.variable>` used as the input to its `function <Mechanism_Base.function>`; or the `value
        <OutputPort.value>` of an `OutputPort` of the Mechanism, that determines how the `value <Mechanism_Base.value>`
        of the Mechanism (i.e., the result of its `function <Mechanism_Base.function>`) is used to generate the output
        from that Mechanism.

  * `MappingProjection`
        a `LearningSignal` must be used; this modulates the `ParameterPort` for the `matrix <MappingProjection.matrix>`
        parameter of a MappingProjection's `function <Projection_Base.function>` which, in turn, determines how it
        computes the MappingProjection's `value <Projection_Base.value>`.

The following table summarizes the three uses of modulation, the ModulatorySignals for each, and the Ports they
modulate.  The mechanics of modulation are described in greater detail in `ModulatorySignal_Implementation`,
and shown in the  `figure below <ModulatorySignal_Anatomy_Figure>`.

.. _ModulatorySignal_Table:

**ModulatorySignals and Ports they Modulate**  (colors listed are those used in the `figure <ModulatorySignal_Anatomy_Figure>` below)

.. table::
  :align: left

+------------------------------------+------------------------+------------------------------+---------------------------------------+----------------------------+
|                                    |                        |Default type of `modulation   |                                       |Default Function (mod param)|
|             Purpose                |  ModulatorySignal      |<ModulatorySignal.modulation>`|           Recipient Port              |for Recipient Port          |
+====================================+========================+==============================+=======================================+============================+
| Modulate the parameter of a        |                        |                              | Mechanism `ParameterPort` (by default)|                            |
| Mechanism's `function              | `ControlSignal` (blue) |     *MULTIPLICATIVE*         | but can also be an                    |     `Linear` (`slope`)     |
| <Mechanism_Base.function>`         |                        |                              | `InputPort` or `OutputPort`           |                            |
+------------------------------------+------------------------+------------------------------+---------------------------------------+----------------------------+
| Modulate the input or output of    |                        |                              |                                       |                            |
| a Mechanism's `function            | `GatingSignal` (brown) |     *MULTIPLICATIVE*         |  Mechanism `InputPort`/`OutputPort`   |    `Linear` (`slope`)      |
| <Mechanism_Base.function>`         |                        |                              |                                       |                            |
+------------------------------------+------------------------+------------------------------+---------------------------------------+----------------------------+
| Modulate a MappingProjection's     |                        |                              |                                       |   `AccumulatorIntegrator`  |
| `matrix <MappingProjection.matrix>`|`LearningSignal` (green)|        *ADDITIVE*            |  MappingProjection `ParameterPort`    |   (`increment`)            |
| parameter                          |                        |                              |                                       |                            |
+------------------------------------+------------------------+------------------------------+---------------------------------------+----------------------------+

It is important to emphasize that, although the purpose of a ModulatorySignal is to modify the functioning of a
`Mechanism <Mechanism>` or a `MappingProjection`, it does this indirectly by modifying a Port that determines the input
or output of a Mechanism, or the parameters of a Mechanism or Projection's `function`, rather than directly modifying
the function of the Mechanism or Projection itself.  This is shown in the following figure, and described in greater
detail under `ModulatorySignal_Implementation`.

.. _ModulatorySignal_Anatomy_Figure:

**Anatomy of Modulation**

.. figure:: _static/Modulation_Anatomy_fig.svg
   :alt: Modulation
   :scale: 150 %

   **Three types of Modulatory Components and the Ports they modulate**. The default `type of modulation
   <ModulatorySignal_Types>` for each type of ModulatorySignal, and the default Function and modulated parameter of
   its recipient Port are listed in the `table <ModulatorySignal_Table>` above. Note that the `ControlMechanism`
   and `ControlSignal <ControlSignal>` are shown in the figure modulating the `ParameterPort` of a Mechanism;
   however, like Gating components, they can also be used to modulate `InputPorts <InputPort>` and `OutputPorts
   <OutputPort>`. The `figure <ModulatorySignal_Detail_Figure>` below shows a detailed view of how ModulatorySignals
   modulate the parameters of a Port's `function <Port_Base.function>`.


.. _ModulatorySignal_Types:

Types of Modulation
^^^^^^^^^^^^^^^^^^^

The `modulation <ModulatorySignal.modulation>` attribute of a ModulatorySignal determines the way in which it
modulates the `value <Port_Base.value>` of a `Port <Port>`, by specifying which parameter of the Port's `function
<Port_Base.function>` that it modifies (see `figure <ModulatorySignal_Detail_Figure>` below).  This is specified
in a tuple containing the Port and the name of the parameter to be modified (see `example
<ControlSignal_Example_Modulate_Costs>`). Alternatively, a keyword can be used in place of the parameter's name.
For some `Functions <Function>`, keywords can be used to specify function-specific forms of modulation (e.g., see
`TransferWithCosts Function <TransferWithCosts_Modulation_of_Cost_Params>` for an example).  In addition, there are
four keywords that can be used to specify generic forms of modulation supported by most `Functions <Function>`:

  * *MULTPLICATIVE_PARAM* - assign the `value <ModulatorySignal.value>` of the ModulatorySignal to the parameter of
    the Port's `function <Port_Base.function>` specified as its `multiplicative_param <Function_Modulatory_Params>`.
    For example, if the Port's `function <Port_Base.function>` is `Linear` (the default for most Ports), then
    the ModulatorySignal's `value <ModulatorySignal.value>` is assigned to the function's `slope <Linear.slope>`
    parameter (it's multiplicative_param), thus multiplying the Port's `variable <Port_Base.variable>` by that
    amount each time the Port is executed, and assigning the result as the Port's `value <Port_Base.value>`.

  * *ADDITIVE_PARAM* - assign the `value <ModulatorySignal.value>` of the ModulatorySignal to the parameter of the
    Port's `function <Port_Base.function>` specified as its `additive_param <Function_Modulatory_Params>`. For
    example, if the Port's `function <Port_Base.function>` is `Linear` (the default for most Ports), then the
    ModulatorySignal's `value <ModulatorySignal.value>` is assigned to the function's `intercept <Linear.intercept>`
    parameter (it's additive_param), thus adding that value to the Port's `variable <Port_Base.variable>` each
    time the Port is executed, and assigning the result as the Port's `value <Port_Base.value>`.

  * *OVERRIDE* - assign the `value <ModulatorySignal.value>` of the ModulatorySignal directly to the Port's
    `value <Port_Base.value>`; in effect, this bypasses the Port's `function <Port_Base.function>`. Note that
    this can be specified for **only one** `ModulatorySignal` that modulates a given Port (see `below
    <ModulatorySignal_Multiple>` for additional details).

  * *DISABLE* - suppresses the modulatory effect of the ModulatorySignal;  the Port's `function <Port_Base.function>`
    will operate as if it did not receive a `ModulatoryProjection <ModulatoryProjection>` from that ModulatorySignal.

   .. note::

      the *MULTPLICATIVE_PARAM* and *ADDITIVE_PARAM* keywords can be used only with `Functions <Function>` that
      specify a `multiplicative_param and/or additive_param <Function_Modulatory_Params>`, respectively.

COMMENT:
FOR DEVELOPERS:  the MULTPLICATIVE_PARAM and ADDITIVE_PARAM options above are keywords for aliases to the relevant
parameters of a given Function, declared in its Parameters subclass declaration of the Function's declaration.
COMMENT

The default type of modulation for `ControlSignals <ControlSignal>` and `GatingSignals <GatingSignal>` is
*MULTIPLICATIVE*.  The default for `LearningSignals <LearningSignal>` is *ADDITIVE* (which additively modifies the
`value <LearningSignal.value>` of the LearningSignal (i.e., the weight changes computed by the `LearningMechanism`)
to the Port's `variable <Port_Base.variable>` (i.e., the current weight `matrix <MappingProjection.matrix>` for
the `MappingProjection` being learned).


.. _ModulatorySignal_Implementation:

Implementation of Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the `purpose of a ModulatorySignal <ModulatorySignal_Uses>` is to modify the operation of a `Mechanism
<Mechanism>` or `MappingProjection`, it does not do this *directly*; rather, it does it by way of a `Port` that
modulates the input, function parameter, or output of the Mechanism or MappingProjeciton to be regulated. More
specifically, a ModulatorySignal modulates the `function <Port_Base.function>` of a *Port* responsible for
generating those values, by modifying a parameter of that Port's `function <Port_Base.function>`.  This is shown
in the figure below:

.. _ModulatorySignal_Detail_Figure:

**Detailed View of Modulation**

.. figure:: _static/Modulation_Detail_fig.svg
   :alt: Modulation_Detail
   :scale: 150 %

   A ModulatorySignal modulates the `value <Port_Base.value>` of a Port either by modifying a parameter of the
   Port's `function <Port_Base.function>`, or assigining the `value <Port_Base.value>` of the Port directly.
   This is determined by the ModulatorySignal's `modulation <ModulatorySignal.modulation>` attribute.  That can be
   assigned either the name of a parameter of the Port's `function <Port_Base.function>`, or a keyword that
   specifies a standard form of modulation.  The keywords *MULTIPLICATIVE* and *ADDITIVE* specify that the `value
   <ModulatorySignal.value>` of the ModulatorySignal be assigned to the `multiplicative_param or `additive_param
   <Function_Modulatory_Params>` of the Port's function, respectively;  *OVERRIDE* specifies that the
   ModulatorySignal's `value <ModulatorySignal.value>` be assigned directly as the Port's `value
   <Port_Base.value>`, in effect bypassing the Port's `function <Port_Base.function>` (see
   `ModulatorySignal_Types` for additional details).

Though this implementaton of modulation is indirect, it provides a standard for all forms of modulation, as well as
considerable flexibility in the modulatory regulation of Components within a `Composition` (see
`ModulatorySignal_Types` below).

The types of Ports modulated by each type of ModulatorySignal are summarized in `ModulatorySignal_Uses`,
and the accompanying `table <ModulatorySignal_Table>` and `figure <ModulatorySignal_Anatomy_Figure>`.

.. _ModulatorySignal_Multiple:

Any `modulable <Parameter.modulable>` parameter of a Port's `function <Port_Base.function>` can be modulated,
and different parameters of the same `function <Port_Base.function>` of a Port can be modulated by different
ModulatorySignals. The same parameter can also be modulated by more than on ModulatorySignal. If more than one
ModulatorySignal modulates the same parameter of a Port's `function <Port_Base.function>`, then that parameter's
`modulation_combine_function <Parameter.modulation_combine_function>` attribute determines how the `value
<ModulatorySignal.value>`\\s of the different ModulatorySignals are combined.  By default, the product of their
vaues is used.  However, if *OVERRIDE* is specfied as the type of `modulation <ModulatorySignal.modulation>` for one
of them, then that ModulatorySignal's  `value <ModulatorySignal.value>` is assigned directly as the Port's `value
<Port_Base.value>`, and the others are all ignored.  Only one ModulatorySignal specified as *OVERRIDE* can modulate
a given parameter;  if there is more than, then an error is generated.

.. _ModulatorySignal_Execution:

Execution
---------

ModulatorySignals cannot be executed directly.  This done when the `ModulatoryMechanism <ModulatoryMechanism>` to
which they belong is executed. When a ModulatorySignal is executed, it calculates its `value <ModulatorySignal.value>`,
which is then assigned as the `variable <Projection_Base.variable>` of the `ModulatoryProjections
<ModulatoryProjection>` listed in its `efferents <ModulatorySignal.efferents>` attribute.
When those Projections execute, they convey the ModulatorySignal's `value <ModulatorySignal.value>` to the `function
<Port_Base.function>` of the `Port <Port>` to which they project.  The Port's `function <Port_Base.function>`
then uses that value in determining value of the parameter designated by the `modulation <ModulatorySignal.modulation>`
attribute of the ModulatorySignal when the Port's `value <Port_Base.value>` is updated.

COMMENT:

# FIX: 9/3/19 -- REWORK AND ADD EXAMPLE HERE

For example, consider a `ControlSignal` that modulates the `bias <Logistic.bias>` parameter of a `Logistic` Function
used by a `TransferMechanism`, and assume that the `ParameterPort` for the bias parameter (to which the ControlSignal
projects) uses a `Linear` function (the default for a ParameterPort) to set the `value <ParameterPort.value>` of
that parameter. If the `modulation  <ModulatorySignal.modulation>` attribute of the `ControlSignal` is *MULTIPLICATIVE*
then, when the TransferMechanism's `Logistic` `function <Mechanism_Base.function>` is executed, the `function
<ParameterPort.function>` of the ParameterPort that sets the value of the `Logistic` Function's `bias <Logistic.bias>`
parameter is executed;  that is a `Linear` Function, that uses the ControlSignal's `value <ControlSignal.value>` as
its `slope <Linear.slope>` parameter.  Thus, the effect is that the ControlSignal's `value <ControlSignal.value>` is
multiplied by the base value of the `bias <Logistic.bias>` parameter, before that is used by the TransferMechanism's
`Logistic` Function.  Thus, the `value <ControlSignal.value>` of the ControlSignal modulates the `bias
<Logistic.bias>` parameter of the `Logistic` Function when the TransferMechanism's `function
<Mechanism_Base.function>` is executed (see `Port Execution <State_Execution>` for additional details).

COMMENT

# FIX 5/8/20 -- REWORK TO BE ALIGNED WITH ModulatoryMechanism

.. note::
   The change in the value of a `Port <Port>` in response to a ModulatorySignal does not occur until the Mechanism to
   which the port belongs is next executed; see `Lazy Evaluation <Component_Lazy_Updating>` for an explanation of
   "lazy" updating).

COMMENT:

.. _ModulatorySignal_Examples:

Examples
--------

MOVE THESE TO SPECIFIC SUBCLASSES? AND REFERENCE THOSE HERE
FIX: EXAMPLE OF FULL SPECIFIATION (BY PORT AND PORT'S FUCNTION'S PARAMETER NAME)
The following example uses a parameter's name to specify
    >>> my_mech = ProcessingMechanism(function=Logistic)
    >>> ctl_mech = ControlMechanism(monitor_for_control=my_mech,
    ...                             control_signals=ControlSignal(modulates=my_mech.parameter_ports[GAIN],
    ...                                                           modulation=SLOPE))

FIX: EXAMPLE OF SPECIFIATION OF CONTROLSIGNAL WITH MECHANISM AND PORT'S PARAMETER NAME

FIX: EXAMPLE OF SPECIFIATION BY CONTROLSIGNAL WITH MECHANISM AND MECHANISM'S PARAMETER NAME

MENTION PORT-SPECIFIC CONVENIENCE METHODS

FIX: EXAMPLE OF CONTROL SIGNAL MODULATION OF INPUTPORT


.  For
example, the `TransferWithCosts` Function defines keywords for `modulating the parameters of its cost functions
<TransferWithCosts_Modulation_of_Cost_Params>`.
A ControlMechanism can even modulate the parameters of another
ControlMechanism, or its ControlSignals.  For example, in the following, ``ctl_mech_A`` modulates the `intensity_cost
<ControlSignal.intensity_cost>` parameter of ``ctl_mech``\\'s ControlSignal::

    >>> my_mech = ProcessingMechanism()
    >>> ctl_mech_A = ControlMechanism(monitor_for_control=my_mech,
    ...                               control_signals=ControlSignal(modulates=(SLOPE,my_mech),
    >>>                                                              cost_options = CostFunctions.INTENSITY))
    >>> ctl_mech_B = ControlMechanism(monitor_for_control=my_mech,
    ...                               control_signals=ControlSignal(modulates=ctl_mech_A.control_signals[0],
    ...                                                             modulation=INTENSITY_COST_FCT_MULTIPLICATIVE_PARAM))

FIX: EXAMPLE HERE
     THEN, AFTER EXAMPLE: MODULATORYSIGNAL'S VALUE IS ASSIGNED TO THE SPECIFIED PARAMETER.
COMMENT

.. _ModulatorySignal_Class_Reference:

Class Reference
---------------
"""

from psyneulink.core.components.component import component_keywords
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, DISABLE, MAYBE, MECHANISM, MODULATION, MODULATORY_SIGNAL, MULTIPLICATIVE_PARAM, \
    OVERRIDE, PROJECTIONS, VARIABLE

from psyneulink.core.globals.defaults import defaultModulatoryAllocation
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'modulatory_signal_keywords', 'ModulatorySignal', 'ModulatorySignalError',
]


def _is_modulatory_spec(spec, include_matrix_spec=True):
    from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import _is_learning_spec
    from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import _is_control_spec
    from psyneulink.core.components.mechanisms.modulatory.control.gating.gatingmechanism import _is_gating_spec

    if (_is_learning_spec(spec, include_matrix_spec=include_matrix_spec)
        or _is_control_spec(spec)
        or _is_gating_spec(spec)
        ):
        return True
    else:
        return False

modulatory_signal_keywords = {MECHANISM, MODULATION}
modulatory_signal_keywords.update(component_keywords)
modulation_type_keywords = [MULTIPLICATIVE_PARAM, ADDITIVE_PARAM, OVERRIDE, DISABLE]


class ModulatorySignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ModulatorySignal(OutputPort):
    """Subclass of `OutputPort` used by a `ModulatoryMechanism <ModulatoryMechanism>` to modulate the value
    of one more `Ports <Port>`.  See `OutputPort <OutputPort_Class_Reference>` and subclasses for additional
    arguments and attributes.

    .. note::
       ModulatorySignal is an abstract class and should *never* be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <ModulatorySignal_Subtypes>`.

    COMMENT:
    PortRegistry
    -------------
        All OutputPorts are registered in PortRegistry, which maintains an entry for the subclass,
        a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    default_allocation : scalar : defaultModulatoryAllocation
        specifies the default template and value used for `variable <ModulatorySignal.variable>`.

    modulation : str : default MULTIPLICATIVE
        specifies the type of modulation the ModulatorySignal uses to determine the value of the Port(s) it modulates;
        must be either a keyword defined by the `Function` of the parameter to be modulated, or one of the following
        generic keywords -- *MULTIPLICATIVE*, *ADDITIVE*, *OVERRIDE* or *DISABLE* (see `ModulatorySignal_Types` for
        additional details).

    Attributes
    ----------

    variable : scalar, list or np.ndarray
        same as `allocation <ModulatorySignal.allocation>`.

    allocation : float
        value assigned by the ModulatorySignal's `owner <ModulatorySignal.owner>`, and used as the `variable
        <ModulatorySignal.variable>` of its `function <ModulatorySignal.function>` to determine the ModulatorySignal's
        `ModulatorySignal.value`.
    COMMENT:
    FOR DEVELOPERS:  Implemented as an alias of the ModulatorySignal's variable Parameter
    COMMENT

    function : TransferFunction
        used to transform the ModulatorySignal's `allocation <ModulatorySignal.allocation>` into its `value
        <ModulatorySignal.value>`;  default is the `Identity` Function that simply assigns `allocation
        <ModulatorySignal.allocation>` to `value <ModulatorySignal.value>`.

    value : number, list or np.ndarray
        result of `function <ModulatorySignal.function>`, used to determine the `value <Port_Base.value>` of the Port(s)
        being modulated.

    modulation : str
        determines how the `value <ModulatorySignal.value>` of the ModulatorySignal is used to modulate the value of
        the port(s) being modulated (see `ModulatorySignal_Types` for additional details).

    efferents : [List[GatingProjection]]
        a list of the `ModulatoryProjections <ModulatoryProjection>` assigned to the ModulatorySignal.

    name : str
        the name of the ModulatorySignal. If the ModulatorySignal's `initialization has been deferred
        <Port_Deferred_Initialization>`, it is assigned a temporary name (indicating its deferred initialization
        status) until initialization is completed, at which time it is assigned its designated name.  If that is the
        name of an existing ModulatorySignal, it is appended with an indexed suffix, incremented for each Port with
        the same base name (see `Registry_Naming`). If the name is not  specified in the **name** argument of its
        constructor, a default name is assigned as follows; if the ModulatorySignal has:

        * no projections (which are used to name it) -- the name of its class is used, with an index that is
          incremented for each ModulatorySignal with a default named assigned to its `owner <ModulatorySignal.owner>`;

        * one `ModulatoryProjection <ModulatoryProjction>` -- the following template is used:
          "<target Mechanism name> <target Port name> <ModulatorySignal type name>"
          (for example, ``'Decision[drift_rate] ControlSignal'``, or ``'Input Layer[InputPort-0] GatingSignal'``);

        * multiple ModulatoryProjections, all to Ports of the same Mechanism -- the following template is used:
          "<target Mechanism name> (<target Port name>,...) <ModulatorySignal type name>"
          (for example, ``Decision (drift_rate, threshold) ControlSignal``, or
          ``'Input Layer[InputPort-0, InputPort-1] GatingSignal'``);

        * multiple ModulatoryProjections to Ports of different Mechanisms -- the following template is used:
          "<owner Mechanism's name> divergent <ModulatorySignal type name>"
          (for example, ``'ControlMechanism divergent ControlSignal'`` or ``'GatingMechanism divergent GatingSignal'``).

        .. note::
            Unlike other PsyNeuLink components, Port names are "scoped" within a Mechanism, meaning that Ports with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: Ports within a Mechanism with the same base name are appended an index in the order of their
            creation.

    """

    componentType = MODULATORY_SIGNAL
    # paramsType = OUTPUT_PORT_PARAMS

    class Parameters(OutputPort.Parameters):
        """
            Attributes
            ----------

                modulation
                    see `modulation <ModulatorySignal_Modulation>`

                    :default value: None
                    :type:
        """
        modulation = None

    portAttributes = OutputPort.portAttributes | {MODULATION}

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'OutputPortCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    def __init__(self,
                 owner=None,
                 size=None,
                 reference_value=None,
                 default_allocation=defaultModulatoryAllocation,
                 function=None,
                 modulates=None,
                 modulation=None,
                 index=None,
                 assign=None,
                 params=None,
                 name=None,
                 prefs=None,
                 **kwargs):

        if kwargs:
            if VARIABLE in kwargs:
                default_allocation = kwargs.pop(VARIABLE, default_allocation)
            if PROJECTIONS in kwargs:
                modulates = kwargs.pop(PROJECTIONS, modulates)

        if params is not None:
            if PROJECTIONS in params:
                modulates = params.pop(PROJECTIONS, modulates)

        # Deferred initialization
        # if self.initialization_status & (ContextFlags.DEFERRED_INIT | ContextFlags.INITIALIZING):
        if self.initialization_status & ContextFlags.DEFERRED_INIT:
            # If init was deferred, it may have been because owner was not yet known (see OutputPort.__init__),
            #   and so modulation hasn't had a chance to be assigned to the owner's value
            #   (i.e., if it was not specified in the constructor), so do it now;
            #   however modulation has already been assigned to params, so need to assign it there
            modulation = modulation or owner.modulation

        if modulates is not None and not isinstance(modulates, list):
            modulates = [modulates]

        super().__init__(owner=owner,
                         reference_value=reference_value,
                         variable=default_allocation,
                         size=size,
                         projections=modulates,
                         index=index,
                         assign=assign,
                         function=function,
                         modulation=modulation,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

        if self.initialization_status == ContextFlags.INITIALIZED:
            self._assign_default_port_Name()

    def _instantiate_attributes_after_function(self, context=None):
        # If owner is specified but modulation has not been specified, assign to owner's value

        super()._instantiate_attributes_after_function(context=context)
        if self.owner and self.modulation is None:
            self.modulation = self.owner.modulation
        if self.modulation is not None:
            if self.modulation not in modulation_type_keywords:
                try:
                    getattr(self.function.parameters, self.modulation)
                except:
                    raise ModulatorySignalError(f"The {MODULATION} arg for {self.name} of {self.owner.name} must be "
                                                f"the name of a modulable parameter of its function "
                                                f"({self.function.__class__.__name__}) or a {MODULATION} keyword "
                                                f"(MULTIPLICATIVE, ADDITIVE, OVERRIDE, DISABLE).")

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of Port's constructor

        Specification should be an existing ModulatoryProjection, or a receiver Mechanism or Port
        Disallow any other specifications (including PathwayProjections)
        Call _instantiate_projection_from_port to assign ModulatoryProjections to .efferents

        """
       # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
        for receiver_spec in projections:
            projection = self._instantiate_projection_from_port(projection_spec=type(self),
                                                                receiver=receiver_spec,
                                                                # MODIFIED 8/12/19 NEW: [JDC] - MODIFIED FEEDBACK
                                                                # feedback=True,
                                                                feedback=MAYBE,
                                                                # MODIFIED 8/12/19 END
                                                                context=context)
            # Projection might be None if it was duplicate
            if projection:
                projection._assign_default_projection_name(port=self)

    def _assign_default_port_Name(self, context=None):

        # If the name is not a default name for the class,
        #    or the ModulatorySignal has no projections (which are used to name it)
        #    then return
        if (
            (
                not (
                    self.name is self.__class__.__name__
                    or self.__class__.__name__ + '-' in self.name
                )
                or len(self.efferents) == 0
            )
            and self.name not in [p.receiver.name for p in self.efferents]
        ):
            return self.name

        # Construct default name
        receiver_names = []
        receiver_owner_names = []
        receiver_owner_receiver_names = []
        class_name = self.__class__.__name__

        for projection in self.efferents:
            receiver = projection.receiver
            receiver_name = receiver.name
            receiver_owner_name = receiver.owner.name
            receiver_names.append(receiver_name)
            receiver_owner_names.append(receiver_owner_name)
            receiver_owner_receiver_names.append("{}[{}]".format(receiver_owner_name, receiver_name))

        # Only one ModulatoryProjection: "<target mech> <Port.name> <ModulatorySignal>"
        # (e.g., "Decision drift_rate ControlSignal", or "Input Layer InputPort-0 GatingSignal")
        if len(receiver_owner_receiver_names) == 1:
            default_name = receiver_owner_receiver_names[0] + " " + class_name

        # Multiple ModulatoryProjections all for same mech: "<target mech> (<Port.name>,...) <ModulatorySignal>"
        # (e.g., "Decision (drift_rate, threshold) ControlSignal" or
        #        "InputLayer (InputPort-0, InputPort-0) ControlSignal")
        elif all(name is receiver_owner_names[0] for name in receiver_owner_names):
            default_name = "{}[{}] {}".format(receiver_owner_names[0], ", ".join(receiver_names), class_name)

        # Mult ModulatoryProjections for diff mechs: "<owner mech> divergent <ModulatorySignal>"
        # (e.g., "EVC divergent ControlSignal", or "GatingMechanism divergent GatingSignal")
        else:
            default_name = self.name + " divergent " + class_name

        self.name = default_name

        return self.name
