# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ModulatorySignal *******************************************************

"""
..
Sections
--------
  * `ModulatorySignal_Overview`
    - `Naming Conventions for ModulatorySignals <ModulatorySignal_Naming>`
  * `ModulatorySignal_Creation`
  * `ModulatorySignal_Structure`
    - `ModulatorySignal_Projections`
    - `ModulatorySignal_Modulation`
      • `ModulatorySignal_Uses`
      • `ModulatorySignal_Types`
      • `ModulatorySignal_Implementation`
  * `ModulatorySignal_Execution`
  * `ModulatorySignal_Class_Reference`
  

.. _ModulatorySignal_Overview:

Overview
--------

A ModulatorySignal is a subclas of `OutputState` that belongs to an `AdaptiveMechanism <AdaptiveMechanism>`, and is
used to `modulate <ModulatorySignal_Modulation>` the `value <State_Base.value>` of one or more `States <State>` by way
of one or more `ModulatoryProjctions <ModulatoryProjection>`. A ModulatorySignal modulates the value of a State by
modifying a  parameter of thatState's `function <State_Base.function>`.  There are three types of ModulatorySignals,
each of which is  associated wth a particular type of `AdaptiveMechanism <AdaptiveMechanism>` and `ModulatoryProjection
<ModulatoryProjection>`, and modifies the value of different types of States, as summarized `below:

* `ControlSignal`
    takes the `allocation <ControlSignal.allocation>` assigned to it by the `function <ControlMechanism.function>`
    of the `ControlMechanism <ControlMechanism>` to which it belongs, and uses it to modulate the parameter of a
    `Mechanism <Mechanism>` or its `function <Mechanism_Base.function>` (and thereby the `value
    <Mechanism_Base.value>` of that Mechanism), or a parameter of the `function <State_Base.function>` one of the
    Mechanism's `InputStates <InputState>` or `OutputStates <OutputState>` (and thereby the `value <State_Base.value>`
    of the corresponding State).
..
* `GatingSignal` takes the `allocation <GatingSignal.allocation>` assigned to it by the `function
    <GatingMechanism.function>` of the `GatingMechanism` to which it belongs, and uses it to modulate the parameter
    of the `function <State_Base.function>` of an `InputState` or `OutputState` (and hence that State's `value
    <State_Base.value>`).  A GatingMechanism and GatingSignal can be thought of as implementing a form of control
    specialized for gating the input to and/or output of a Mechanism.
..
* `LearningSignal`
    takes the `learning_signal <LearningMechanism>` calculated by the `function <LearningMechanism.function>` of the
    `LearningMechanism` to which it belongs, and uses it to modulate the `matrix <MappingProjection.matrix>` parameter
    of a `MappingProjection`.

These are shown in a `figure <ModulatorySignal_Anatomy_Figure>` below, and are descdribed in greater detail in
in the sections under `ModulatorySignal_Structure`.

.. _ModulatorySignal_Naming:

*Naming Conventions for ModulatorySignals*

Modulatory Components and their attributes are named according to the category of modulation:

    - AdaptiveMechanism name: <*Category*>Mechanism (e.g., ControlMechanism)
    - ModulatorySignal name: <*Category*>Signal (e.g., ControlSignal)
    - ModulatoryProjection name: <*Category*>Projection (e.g., ControlProjection)
    - List of an AdaptiveMechanism's ModulatorySignals: <*CategoryMechanism*>.<category>_signals
      (e.g., ControlMechanism.control_signals)
    - Value of a ModulatorySignal: <*CategorySignal*>.<category>_signal (e.g., ControlSignal.control_signal)

.. _ModulatorySignal_Creation:

Creating a ModulatorySignal
---------------------------

ModulatorySignal is a base class, and cannot be instantiated directly.  However, the three types of ModulatorySignals
listed above can be created directly, by calling the constructor for the desired type.  More commonly, however,
ModulatorySignals are created automatically by the `AdaptiveMechanism <AdaptiveMechanism>` to which they belong, or by
specifying them in the constructor for an `AdaptiveMechanism <AdaptiveMechanism>` (the details of which are described in
the documentation for each type of ModulatorySignal).  If a ModulatorySignal is constructed explicitly, the type of
modulation it uses is specifed in the **modulation** argument of its constructor, using a 2-item tuple that contains
the State to be modulated as the first item, and either the name of the parameter of the State's `function
<State_Base.function>` to be modulated, or a keyword specifying the type of modulation, as the second item (see
`ModulatorySignal_Types` for additional details).

.. _ModulatorySignal_Structure:

Structure
---------

A ModulatorySignal is always assigned to an `AdaptiveMechanism <AdaptiveMechanism>`, and must be assigned to an
AdaptiveMechanism of the appropriate type (`see types of AdaptiveMechanism <AdaptiveMechanism_Types>`).  The
ModulatorySignal receives a `modulatory_allocation` from the AdaptiveMechanism to which it is assigned, that it uses
as the `variable <Function_Base.variable>` for its `function <ModulatorySignal.function>`, the result of which is the
modulatory `value <ModulatorySignal.value>` of the ModulatorySignal.  A ModulatorySignal is associated with one or more
`ModulatoryProjections <ModulatoryProjection>` of the corresponding type, that that receive the ModulatorySignal's
`value <ModulatorySignal.value>`, and use this to modulate the State(s) to which they project.  All of the
ModulatoryProjections from a given ModulatorySignal are assigned the same modulatory `value <ModulatorySignal.value>`
(see `ModulatorySignal_Projections` below) and use the same `type of modulation <ModulatorySignal_Types>` specified
by the the ModulatorySignal's `modulation <ModulatorySignal.modulation>` attribute.  The ModulatoryProjections
received by a `State <State>` are listed in the State's `mod_afferents <State_Base.mod_afferents>` attribute.

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
attribute.  See `State Projections <State_Projections>` for additional details concerning the specification of
Projections when creating a State.

Although a ModulatorySignal can be assigned more than one `ModulatoryProjection <ModulatoryProjection>`,
all of those Projections receive and convey the same modulatory `value <ModulatorySignal.value>` from the
ModulatorySignal, and use the same form of `modulation <ModulatorySignal_Modulation>`.  This is a common use for some
ModulatorySignals (e.g., the use of a single `GatingSignal` to gate multiple `InputState(s) <InputState>` or
`OutputState(s) <OutputState>`), but requires more specialized circumstances for others (e.g., the use of a single
`LearningSignal` for more than one `MappingProjection`, or a single `ControlSignal` for the parameters of more than
one Mechanism or function).

.. _ModulatorySignal_Modulation:

*Modulation*
~~~~~~~~~~~~

A ModulatorySignal modulates the value of a `State <State>` either by modifying a parameter of the State's `function
<State_Base.function>` (which determines the State's `value <State_Base.value>`), or by  assigning a value to the State
directly.  The `type of modulation <ModulatorySignal_Types>` is determined by the ModulatorySignal's
`modulation <ModulatorySignal.modulation>` attribute, which can be specified in the **modulation** argument of its
ModulatorySignal's constructor, or in a *MODULATION* entry of a `State specification dictionary
<State_Specification>` used to create the ModulatorySignal (see `Type of Modualtion <ModulatorySignal_Types>` and
`figure <ModulatorySignal_Detail_Figure>` below for details). If the type of `modulation <ModulatorySignal.modulation>`
is not specified when a ModulatorySignal is created, it is assigned the value of the `modulation
<AdaptiveMechanism_Base.modulation>` attribute for the `AdaptiveMechanism <AdaptiveMechanism>` to which it belongs.

.. _ModulatorySignal_Uses:

Uses of Modulation
^^^^^^^^^^^^^^^^^^

There are three broad categories of modulation that serve different purposes, and differ according to the
ModulatorySignals used and the type of State modulated:

  * **modulation of a** `Mechanism`\\s <function <Mechanism_Base.function> -- a `ControlSignal` must be used; this
    modulates the `ParameterState` for a parameter of the Mechanism's `function <Mechanism_Base.function>` which,
    in turn, determines how it computes the Mechanism's `value <Mechanism_Base.value>`;

  * **modulation of a** `Mechanism`\\s input or output -- a `GatingSignal` is specialized for this purpose, though a
    `ControlSignal` can also be used;  these modulate an `InputState` of the Mechanism, that determines the
    Mechanism's `variable <Mechanism_Base.variable>` used as the input to its `function <Mechanism_Base.function>`,
    or an `OutputState` of the Mechanism, that determines how the `value <Mechanism_Base.value>` of the Mechanism
    (i.e., the result of its `function <Mechanism_Base.function>`) is used to generate the output of the Mechanism.

  * **modulation of a** `MappingProjection` -- a `LearningSignal` must be used; this modulates the `ParameterState` for
    the `matrix <MappingProjection.matrix>` parameter of a MappingProjection's `function  <MappingProjection.function>`
    which, in turn, determines how it computes the MappingProjection's `value <MappingProjection.value>`.

The following table summarizes the three uses of modulation, the ModulatorySignals for each, and the States they
modulate. The mechanics of modulation are described in greater detail in `ModulatorySignal_Implementation`,
and shown in the `figure below <ModulatorySignal_Anatomy_Figure>`.

.. _ModulatorySignal_Table:

.. table:: **ModulatorySignals and States they Modulate**
  :align: left

  +------------------------------------+------------------------+------------------------------+----------------------------------------+----------------------------+
  |                                    |                        |Default type of `modulation   |                                        |Default Function (mod param)|
  |             Purpose                |  ModulatorySignal      |<ModulatorySignal.modulation>`|           Recipient State              |for Recipient State         |
  +====================================+========================+==============================+========================================+============================+
  | Modulate the parameter of a        |                        |                              | Mechanism `ParameterState` (by default)|                            |
  | Mechanism's `function              | `ControlSignal` (blue) |     *MULTIPLICATIVE*         | but can also be an                     |     `Linear` (`slope`)     |
  | <Mechanism_Base.function>`         |                        |                              | `InputState` or `OutputState`          |                            |
  +------------------------------------+------------------------+------------------------------+----------------------------------------+----------------------------+
  | Modulate the input or output of    |                        |                              |                                        |                            |
  | a Mechanism's `function            | `GatingSignal` (brown) |     *MULTIPLICATIVE*         |  Mechanism `InputState`/`OutputState`  |     `Linear` (`slope`)     |
  | <Mechanism_Base.function>`         |                        |                              |                                        |                            |
  +------------------------------------+------------------------+------------------------------+----------------------------------------+----------------------------+
  | Modulate a MappingProjection's     |                        |                              |                                        |   `AccumulatorIntegrator`  |
  | `matrix <MappingProjection.matrix>`|`LearningSignal` (green)|        *ADDITIVE*            |  MappingProjection `ParameterState`    |   (`increment`)            |
  | parameter                          |                        |                              |                                        |                            |
  +------------------------------------+------------------------+------------------------------+----------------------------------------+----------------------------+

Colors listed are those used in the `figure <ModulatorySignal_Anatomy_Figure>` below.

It is important to emphasize that, although the purpose of a ModulatorySignal is to modify the functioning of a
`Mechanism` or a `MappingProjection`, it does this indirectly by modifying a State that determines the input or
output of a Mechanism, or the parameters of a Mechanism or Projection's `function`, rather than directly modifying
the function of the Mechanism or Projection itself.  This is shown in the following figure, and described in greater
detail under `ModulatorySignal_Implementation`.

.. _ModulatorySignal_Anatomy_Figure:

**Anatomy of Modulation**

.. figure:: _static/Modulation_Anatomy_fig.svg
   :alt: Modulation
   :scale: 150 %

   **Three types of Modulatory Components and the States they modulate**. The default `type of modulation
   <ModulatorySignal_Types>` for each type of ModulatorySignal, and the default Function and modulated parameter of
   its recipient State are listed in the `table <ModulatorySignal_Table>` above. Note that the `ControlMechanism`
   and `ControlSignal <ControlSignal>` are shown in the figure modulating the `ParameterState` of a Mechanism;
   however, like Gating components, they can also be used to modulate `InputStates <InputState>` and `OutputStates
   <OutputState>`. The `figure <ModulatorySignal_Detail_Figure>` below shows a detailed view of how ModulatorySignals
   modulate the parameters of a State's `function <State_Base.function>`.


.. _ModulatorySignal_Types:

Types of Modulation
^^^^^^^^^^^^^^^^^^^

The `modulation <ModulatorySignal.modulation>` attribute of a ModulatorySignal determines the way in which it
modulates the `value <State_Base.value>` of a `State`, by specifying which paramter of the State's `function
<State_Base.function>`that it modifies (see `figure <ModulatorySignal_Detail_Figure>` below).  This is specified
in a tuple containing the State and the name of the parameter to be modified (see `example <EXAMPLE??> below).
Alternatively, there are four keywords that can be used in place of the parameter's name, that specify the two most
commonly used types of modulation, and allow two other types:

  * *MULTPLICATIVE_PARAM* - assign the `value <ModulatorySignal.value>` of the ModulatorySignal to the parameter of
    the State's `function <State_Base.function>` specified as its `multiplicative_param <Function_Modulatory_Params>`.
    For example, if the State's `function <State_Base.function>` is `Linear` (the default for most States), then
    the ModulatorySignal's `value <ModulatorySignal.value>` is assigned to the function's `slope <Linear.slope>`
    parameter (it's multiplicative_param), thus multiplying the State's `variable <State_Base.variable>` by that
    amount each time the State is executed, and assigning the result as the State's `value <State_Base.value>`.

  * *ADDITIVE_PARAM* - assign the `value <ModulatorySignal.value>` of the ModulatorySignal to the parameter of the
    State's `function <State_Base.function>` specified as its `additive_param <Function_Modulatory_Params>`. For
    example, if the State's `function <State_Base.function>` is `Linear` (the default for most States), then the
    ModulatorySignal's `value <ModulatorySignal.value>` is assigned to the function's `intercept <Linear.intercept>`
    parameter (it's additive_param), thus adding that value to the State's `variable <State_Base.variable>` each
    time the State is executed, and assigning the result as the State's `value <State_Base.value>`.

  * *OVERRIDE* - assign the `value <ModulatorySignal.value>` of the ModulatorySignal directly to the State's
    `value <State_Base.value>`; in effect, this bypasses the State's `function <State_Base.function>`. Note that
    this can be specified for **only one** `ModulatorySignal` that modulates a given State (see `below
    <ModulatorySignal_Multiple>` for additional details).

  * *DISABLE* - suppresses the modulatory effect of the ModulatorySignal;  the State's `function <State_Base.function>`
    will operate as if it did not receive a `ModulatoryProjection <ModulatoryProjection>` from that ModulatorySignal.

   .. note:

      the *MULTPLICATIVE_PARAM* and *ADDITIVE_PARAM* keywords can be used only with `Functions <Function>` that
      specify a `multiplicative_param and/or additive_param <Function_Modulatory_Params>`, respectively.

COMMENT:
FOR DEVELOPERS:  the MULTPLICATIVE_PARAM and ADDITIVE_PARAM options above are keywords for aliases to the relevant
parameters of a given Function, declared in its Parameters subclass declaration of the Function's declaration.
COMMENT

The default type of modulation for `ControlSignals <ControlSignal>` and `GatingSignals <GatingSignal>` is
*MULTIPLICATIVE*.  The default for `LearningSignals <LearningSignal>` is *ADDITIVE* (which additively modifies the
`value <LearningSignal.value>` of the LearningSignal (i.e., the weight changes computed by the `LearningMechanism`)
to the State's `variable <State_Base.variable>` (i.e., the current weight `matrix <MappingProjection.matrix>` for
the `MappingProjection` being learned).


.. _ModulatorySignal_Implementation:

Implementation of Modulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the `purpose of a ModulatorySignal <ModulatorySignal_Uses>` is to modify the operation of
a `Mechanism` or `MappingProjection`, it does not do this *directly*; rather, it does it by way of a `State` that
modulates the input, function parameter, or output of the Mechanism or MappingProjeciton to be regulated. More
specifically, a ModulatorySignal modulates the `function <State_Base.function>` of a *State* responsible for
generating those values, by modifying a parameter of that State's `function <State_Base.function>`.  This is shown
in the figure below:

.. _ModulatorySignal_Detail_Figure:

**Detailed View of Modulation**

.. figure:: _static/Modulation_Detail_fig.svg
   :alt: Modulation_Detail
   :scale: 150 %

   A ModulatorySignal modulates the `value <State_Base.value>` of a State either by modifying a parameter of the
   State's `function <State_Base.function>`, or assigining the `value <State_Base.value>` of the State directly.
   This is determined by the ModulatorySignal's `modulation <ModulatorySignal.modulation>` attribute.  That can be
   assigned either the name of a parameter of the State's `function <State_Base.function>`, or a keyword that
   specifies a standard form of modulation.  The keywords *MULTIPLICATIVE* and *ADDITIVE* specify that the `value
   <ModulatorySignal.value>` of the ModulatorySignal be assigned to the `multiplicative_param or `additive_param
   <Function_Modulatory_Params>` of the State's function, respectively;  *OVERRIDE* specifies that the
   ModulatorySignal's `value <ModulatorySignal.value>` be assigned directly as the State's `value
   <State_Base.value>`, in effect bypassing the State's `function <State_Base.function>` (see
   `ModulatorySignal_Types` for additional details).

Though this implementaton of modulation is indirect, it provides a standard for all forms of modulation, as well as
considerable flexibility in the adaptive regulation of Components within a `Composition` (see
`ModulatorySignal_Types` below).

The types of States modulated by each type of ModulatorySignal are summarized in `ModulatorySignal_Uses`,
and the accompanying `table <ModulatorySignal_Table>` and `figure <ModulatorySignal_Anatomy_Figure>`.

.. _ModulatorySignal_Multiple:

Any `modulable <Parameter.modulable>` parameter of a State's `function <State_Base.function>` can modulated,
and different parameters of the same `function <State_Base.function>` of a State can be modulated by different
ModulatorySignals. The same parameter can also be modulated by more than on ModulatorySignal. If more than one
ModulatorySignal modulates the same parameter of a State's `function <State_Base.function>`, then that parameter's
`modulation_combine_function <Parameter.modulation_combine_function>` attribute determines how the `value
<ModulatorySignal.value>`\\s of the different ModulatorySignals are combined.  By default, the product of their
vaues is used.  However, if *OVERRIDE* is specfied as the type of `modulation <ModulatorySignal.modulation>` for one
of them, then that ModulatorySignal's  `value <ModulatorySignal.value>` is assigned directly as the State's `value
<State_Base.value>`, and the others are all ignored.  Only one ModulatorySignal specified as *OVERRIDE* can modulate
a given parameter;  if there is more than, then an error is generated.

.. _ModulatorySignal_Execution:

Execution
---------

ModulatorySignals cannot be executed directly.  This done when the `AdaptiveMechanism <AdaptiveMechanism>` to
which they belong is executed. When a ModulatorySignal is executed, it calculates its `value <ModulatorySignal.value>`,
which is then assigned as the `variable <ModulatoryProjection_Base.variable>` of the `ModulatoryProjections
<ModulatoryProjection>` listed in its `efferents <ModulatorySignal.efferents>` attribute.
When those Projections execute, they convey the ModulatorySignal's `value <ModulatorySignal.value>` to the `function
<State_Base.function>` of the `State <State>` to which they project.  The State's `function <State_Base.function>`
then uses that value in determining value of the parameter designated by the `modulation <ModulatorySignal.modulation>`
attribute of the ModulatorySignal when the State's `value <State_Baselvalue>` is updated.

COMMENT:

# FIX: 9/3/19 -- REWORK AND ADD EXAMPLE HERE

For example, consider a `ControlSignal` that modulates the `bias <Logistic.bias>` parameter of a `Logistic` Function
used by a `TransferMechanism`, and assume that the `ParameterState` for the bias parameter (to which the ControlSignal
projects) uses a `Linear` function (the default for a ParameterState) to set the `value <ParameterState.value>` of
that parameter. If the `modulation  <ModulatorySignal.modulation>` attribute of the `ControlSignal` is *MULTIPLICATIVE*
then, when the TransferMechanism's `Logistic` `function <TransferMechanism.function>` is executed, the `function
<ParameterState.function>` of the ParameterState that sets the value of the `Logistic` Function's `bias <Logistic.bias>`
parameter is executed;  that is a `Linear` Function, that uses the ControlSignal's `value <ControlSignal.value>` as
its `slope <Linear.slope>` parameter.  Thus, the effect is that the ControlSignal's `value <ControlSignal.value>` is
multiplied by the base value of the `bias <Logistic.bias>` parameter, before that is used by the TransferMechanism's
`Logistic` Function.  Thus, the `value <ControlSignal.value>` of the ControlSignal modulates the `bias
<Logistic.bias>` parameter of the `Logistic` Function when the TransferMechanism's `function
<TransferMechanism.function>` is executed (see `State Execution <State_Execution>` for additional details).

COMMENT

.. note::
   The change in the value of a `State <State>` in response to a ModulatorySignal does not occur until the Mechanism to
   which the state belongs is next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

.. _ModulatorySignal_Class_Reference:

COMMENT:

Examples
--------

MOVE THESE TO SPECIFIC SUBCLASSES? AND REFERENCE THOSE HERE
FIX: EXAMPLE OF FULL SPECIFIATION (BY STATE AND STATE'S FUCNTION'S PARAMETER NAME)
The following example uses a parameter's name to specify
    >>> my_mech = ProcessingMechanism(function=Logistic)
    >>> ctl_mech = ControlMechanism(monitor_for_control=my_mech,
    ...                             control_signals=ControlSignal(modulates=my_mech.parameter_states[GAIN],
    ...                                                           modulation=SLOPE))

FIX: EXAMPLE OF SPECIFIATION OF CONTROLSIGNAL WITH MECHANISM AND STATE'S PARAMETER NAME

FIX: EXAMPLE OF SPECIFIATION BY CONTROLSIGNAL WITH MECHANISM AND MECHANISM'S PARAMETER NAME

MENTION STATE-SPECIFIC CONVENIENCE METHODS

FIX: EXAMPLE OF CONTROL SIGNAL MODULATION OF INPUT STATE


.  For
example, the `TransferWithCosts` `Function` defines keywords for `modulating the parameters of its cost functions
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

FIX: PUT THIS IN ModulatoryMechanism AND CROSS REFERENCE HERE? OR VICE VERSA?
EXAMPLE OF MIXED MODULATORY SIGNALS IN MODULATORY MECHANISM

        m = ProcessingMechanism(function=Logistic)
        c = ModulatoryMechanism(
                modulatory_signals=[
                    ControlSignal(name="CS1", modulates=(GAIN, m)),
                    GatingSignal(name="GS", modulates=m),
                    ControlSignal(name="CS2", modulates=(BIAS, m)),
                ]
        )


COMMENT

Class Reference
---------------
"""

from psyneulink.core.components.component import component_keywords
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.states.state import State_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import MAYBE, MECHANISM, MODULATION, MODULATORY_SIGNAL, VARIABLE, PROJECTIONS
from psyneulink.core.globals.defaults import defaultModulatoryAllocation
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'modulatory_signal_keywords', 'ModulatorySignal', 'ModulatorySignalError',
]


def _is_modulatory_spec(spec, include_matrix_spec=True):
    from psyneulink.core.components.mechanisms.adaptive.learning.learningmechanism import _is_learning_spec
    from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import _is_control_spec
    from psyneulink.core.components.mechanisms.adaptive.gating.gatingmechanism import _is_gating_spec

    if (_is_learning_spec(spec, include_matrix_spec=include_matrix_spec)
        or _is_control_spec(spec)
        or _is_gating_spec(spec)
        ):
        return True
    else:
        return False


class ModulatorySignalError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

modulatory_signal_keywords = {MECHANISM, MODULATION}
modulatory_signal_keywords.update(component_keywords)


class ModulatorySignal(OutputState):
    """
    ModulatorySignal(                                  \
        owner,                                         \
        default_allocation=defaultModulatoryAllocation \
        function=LinearCombination(operation=SUM),     \
        modulation=MULTIPLICATIVE                      \
        projections=None,                              \
        params=None,                                   \
        name=None,                                     \
        prefs=None)

    Subclass of `OutputState` used by an `AdaptiveMechanism <AdaptiveMechanism>` to modulate the value
    of one more `States <State>`.

    .. note::
       ModulatorySignal is an abstract class and should NEVER be instantiated by a call to its constructor.
       It should be instantiated using the constructor for a `subclass <ModulatorySignal_Subtypes>`.

    COMMENT:

        Description
        -----------
            The ModulatorySignal class is a subtype of the OutputState class in the State category of Component,
            It is used primarily as the sender for GatingProjections
            Its FUNCTION updates its value:
                note:  currently, this is the identity function, that simply maps variable to self.value

        Class attributes:
            + componentType (str) = GATING_SIGNAL
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS (Modulation.MULTIPLY)

        Class methods:
            function (executes function specified in params[FUNCTION];  default: Linear

        StateRegistry
        -------------
            All OutputStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT


    Arguments
    ---------

    owner : ModulatoryMechanism
        specifies the `GatingMechanism` to which to assign the ModulatorySignal.

    default_allocation : scalar : defaultModulatoryAllocation
        specifies the default template and value used for `variable <ModulatorySignal.variable>`.

    function : Function or method : default Linear
        specifies the function used to determine the value of the ModulatorySignal from the value of its
        `owner <GatingMechanism.owner>`.

    modulation : ModulationParam : default MULTIPLICATIVE
        specifies the type of modulation the ModulatorySignal uses to determine the value of the State(s) it modulates.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the ControlSignal and/or a custom function and its parameters. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <ModulatorySignal.name>`
        specifies the name of the ModulatorySignal.

    prefs : PreferenceSet or specification dict : default State.classPreferences
        specifies the `PreferenceSet` for the LearningSignal; see `prefs <ControlSignal.prefs>` for details.


    Attributes
    ----------

    owner : AdaptiveMechanism
        the `AdaptiveMechanism <AdaptiveMechanism>` to which the ModulatorySignal belongs.

    variable : scalar, list or np.ndarray
        same as `allocation <ModulatorySignal.allocation>`.

    allocation : float
        value assigned by the ModulatorySignal's `owner <ControlSignal.owner>`, and used as the `variable
        <ModulatorySignal.variable>` of its `function <ModulatorySignal.function>` to determine the ModulatorySignal's
        `ModulatorySignal.intensity`.
    COMMENT:
    FOR DEVELOPERS:  Implemented as an alias of the ModulatorySignal's variable Parameter
    COMMENT

    function : TransferFunction
        provides the ModulatorySignal's `value <ModulatorySignal.value>`; the default is an identity function that
        assigns `variable <ModulatorySignal.variable>` as ModulatorySignal's `value <ModulatorySignal.value>`.

    value : number, list or np.ndarray
        result of `function <ModulatorySignal.function>`, used to determine the `value <State_Base.value>` of the
        State(s) being modulated.

    modulation : ModulationParam
        determines how the output of the ModulatorySignal is used to modulate the value of the state(s) being modulated.

    efferents : [List[GatingProjection]]
        a list of the `ModulatoryProjections <ModulatoryProjection>` assigned to the ModulatorySignal.

    name : str
        the name of the ModulatorySignal. If the ModulatorySignal's `initialization has been deferred
        <State_Deferred_Initialization>`, it is assigned a temporary name (indicating its deferred initialization
        status) until initialization is completed, at which time it is assigned its designated name.  If that is the
        name of an existing ModulatorySignal, it is appended with an indexed suffix, incremented for each State with
        the same base name (see `Naming`). If the name is not  specified in the **name** argument of its constructor,
        a default name is assigned as follows; if the ModulatorySignal has:

        * no projections (which are used to name it) -- the name of its class is used, with an index that is
          incremented for each ModulatorySignal with a default named assigned to its `owner <ModulatorySignal.owner>`;

        * one `ModulatoryProjection <ModulatoryProjction>` -- the following template is used:
          "<target Mechanism name> <target State name> <ModulatorySignal type name>"
          (for example, ``'Decision[drift_rate] ControlSignal'``, or ``'Input Layer[InputState-0] GatingSignal'``);

        * multiple ModulatoryProjections, all to States of the same Mechanism -- the following template is used:
          "<target Mechanism name> (<target State name>,...) <ModulatorySignal type name>"
          (for example, ``Decision (drift_rate, threshold) ControlSignal``, or
          ``'Input Layer[InputState-0, InputState-1] GatingSignal'``);

        * multiple ModulatoryProjections to States of different Mechanisms -- the following template is used:
          "<owner Mechanism's name> divergent <ModulatorySignal type name>"
          (for example, ``'ControlMechanism divergent ControlSignal'`` or ``'GatingMechanism divergent GatingSignal'``).

        .. note::
            Unlike other PsyNeuLink components, State names are "scoped" within a Mechanism, meaning that States with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: States within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ModulatorySignal; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = MODULATORY_SIGNAL
    # paramsType = OUTPUT_STATE_PARAMS

    class Parameters(OutputState.Parameters):
        """
            Attributes
            ----------

                modulation
                    see `modulation <ModulatorySignal.modulation>`

                    :default value: None
                    :type:

        """
        modulation = None

    stateAttributes =  OutputState.stateAttributes | {MODULATION}

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'OutputStateCustomClassPreferences',
    #     kp<pref>: <setting>...}

    paramClassDefaults = State_Base.paramClassDefaults.copy()

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

        # Deferred initialization
        # if self.initialization_status & (ContextFlags.DEFERRED_INIT | ContextFlags.INITIALIZING):
        if self.initialization_status & ContextFlags.DEFERRED_INIT:
            # If init was deferred, it may have been because owner was not yet known (see OutputState.__init__),
            #   and so modulation hasn't had a chance to be assigned to the owner's value
            #   (i.e., if it was not specified in the constructor), so do it now;
            #   however modulation has already been assigned to params, so need to assign it there
            params[MODULATION] = self.modulation or owner.modulation

        # Standard initialization
        else:
            # Assign args to params and functionParams dicts
            params = self._assign_args_to_param_dicts(params=params,
                                                      modulation=modulation)

        super().__init__(owner=owner,
                         reference_value=reference_value,
                         variable=default_allocation,
                         size=size,
                         projections=modulates,
                         index=index,
                         assign=assign,
                         function=function,
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs)

        if self.initialization_status == ContextFlags.INITIALIZED:
            self._assign_default_state_name()

    def _instantiate_attributes_after_function(self, context=None):
        # If owner is specified but modulation has not been specified, assign to owner's value

        super()._instantiate_attributes_after_function(context=context)
        if self.owner and self.modulation is None:
            self.modulation = self.owner.modulation


    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of State's constructor

        Specification should be an existing ModulatoryProjection, or a receiver Mechanism or State
        Disallow any other specifications (including PathwayProjections)
        Call _instantiate_projection_from_state to assign ModulatoryProjections to .efferents

        """
       # IMPLEMENTATION NOTE: THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
        for receiver_spec in projections:
            projection = self._instantiate_projection_from_state(projection_spec=type(self),
                                                                 receiver=receiver_spec,
                                                                 # MODIFIED 8/12/19 NEW: [JDC] - MODIFIED FEEDBACK
                                                                 # feedback=True,
                                                                 feedback=MAYBE,
                                                                 # MODIFIED 8/12/19 END
                                                                 context=context)
            # Projection might be None if it was duplicate
            if projection:
                projection._assign_default_projection_name(state=self)

    def _assign_default_state_name(self, context=None):

        # If the name is not a default name for the class,
        #    or the ModulatorySignal has no projections (which are used to name it)
        #    then return
        if (not (self.name is self.__class__.__name__
                 or self.__class__.__name__ + '-' in self.name) or
                    len(self.efferents)==0):
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

        # Only one ModulatoryProjection: "<target mech> <State.name> <ModulatorySignal>"
        # (e.g., "Decision drift_rate ControlSignal", or "Input Layer InputState-0 GatingSignal")
        if len(receiver_owner_receiver_names) == 1:
            default_name = receiver_owner_receiver_names[0] + " " + class_name

        # Multiple ModulatoryProjections all for same mech: "<target mech> (<State.name>,...) <ModulatorySignal>"
        # (e.g., "Decision (drift_rate, threshold) ControlSignal" or
        #        "InputLayer (InputState-0, InputState-0) ControlSignal")
        elif all(name is receiver_owner_names[0] for name in receiver_owner_names):
            default_name = "{}[{}] {}".format(receiver_owner_names[0], ", ".join(receiver_names), class_name)

        # Mult ModulatoryProjections for diff mechs: "<owner mech> divergent <ModulatorySignal>"
        # (e.g., "EVC divergent ControlSignal", or "GatingMechanism divergent GatingSignal")
        else:
            default_name = self.name + " divergent " + class_name

        self.name = default_name

        return self.name
