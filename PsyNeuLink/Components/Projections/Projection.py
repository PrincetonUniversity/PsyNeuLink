# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **********************************************  Projection ***********************************************************

"""
..
    Sections of this document:
      * :ref:`Projection_Overview`
      * :ref:`Projection_Creation`
      * :ref:`Projection_Structure`
         * :ref:`Projection_Sender`
         * :ref:`Projection_Receiver`
      * :ref:`Projection_Execution`
      * :ref:`Projection_Class_Reference`

.. _Projection_Overview:

Overview
--------

Projections allow information to be passed between `Mechanisms <Mechanism>`.  A Projection takes its input from
its `sender <Projection.sender>` and transmits that information to its `receiver <Projection.receiver>`.  The
`sender <Projection.sender>` and `receiver <Projection.receiver>` of a Projection are always `States <State>`:
the`sender <Projection.sender>` is always the `OutputState` of a `Mechanism <Mechanism>`; the `receiver
<Projection.receiver>` depends upon the type of Projection.  There are two broad categories of Projections,
each of which has subtypes that differ in the type of information they transmit, how they do this, and the type of
`State <State>` to which they project (i.e., of their `receiver <Projection.receiver>`):

* `PathwayProjection <PathwayProjection>`
    Used in conjunction with `ProcessingMechanisms <ProcessingMechanism>` to convey information along a processing
    `pathway <Process_Base.pathway`>.  There is currently one on type of PathwayProjection:

  * `MappingProjection`
      takes the `value <OutputState.value>` of an `OutputState` of a `ProcessingMechanism <ProcessingMechanism>`
      converts it by convolving it with the MappingProjection's `matrix <MappingProjection.MappingProjection.matrix>`
      parameter, and transmits the result to the `InputState` of another ProcessingMechanism.  Typically,
      MappingProjections are used to connect Mechanisms in the `pathway` of a `Process`, though they can be use for
      other purposes as well (for example, to convey the output of an `ObjectiveMechanism` to an `AdaptiveMechanism
      <AdaptiveMechanism>`).

* `ModulatoryProjection <ModulatoryProjection>`
    takes the `value <OutputState.value>` of a `ModulatorySignal <ModulatorySignal>` of an `AdaptiveMechanism
    <ProcessingMechanism>`, uses it to regulate modify the `value <State_Base.value>` of an `InputState,
    `ParameterState` or `OutputState` of another Component.  ModulatorySignals are specialized types of `OutputState`,
    that are used to specify how to modify the `value <State_Base.value>` of the `State <State>` to which a
    ModulatoryProjection projects. There are three types of ModulatoryProjections, corresponding to the three types
    of AdaptiveMechanisms (and corresponding ModulatorySignals; see `figure <ModulatorySignal_Anatomy_Figure>`),
    that project to different types of `States <State>`:

  * `LearningProjection`
      takes the `value <LearningSignal.value>` of a `LearningSignal` of a `LearningMechanism`, and transmits
      this to the `ParameterState` of a `MappingProjection` that uses this to modify its `matrix
      <MappingProjection.MappingProjection.matrix>` parameter. LearningProjections are used when learning has
      been specified for a `Process <Process_Learning_Sequence>` or `System <System_Execution_Learning>`.
  ..
  * `ControlProjection`
      takes the `value <ControlSignal.value>` of a `ControlSignal` of a `ControlMechanism <ControlMechanism>`, and
      transmit this to the `ParameterState of a `ProcessingMechanism <ProcessingMechanism>` that uses this to modify
      the parameter of the (or its `function <Mechanism_Base.function>`) for which it is responsible.
      ControlProjections are used when control has been used specified for a `System`.
  ..
  * `GatingProjection`
      takes the `value <GatingSignal.value>` of a `GatingSignal` of a `GatingMechanism`, and transmits this to
      the `InputState` or `OutputState` of a `ProcessingMechanism <ProcessingMechanism>` that uses this to modify the
      State's `value <State_Base.value>`

.. _Projection_Creation:

Creating a Projection
---------------------

A Projection can be created on its own, by calling the constructor for the desired type of Projection.  More
commonly, however, Projections are either specified `in context <Projection_In_Context_Specification>`, or
are `created automatically <Projection_Automatic_Creation>`, as described below.


.. _Projection_In_Context_Specification:

In Context Specification
~~~~~~~~~~~~~~~~~~~~~~~~

Projections can be specified in a number of places where they are required or permitted, for example in the
specification of a `pathway <Process_Base.pathway>` for a `Process`, where the value of a parameter is specified
(e.g., to assign a `ControlProjection`) or where a `MappingProjection` is specified  (to assign it a
`LearningProjection <MappingProjection_Tuple_Specification>`).  Any of the following can be used to specify a
Projection in context:

  * **Constructor**.  Used the same way in context as it is ordinarily.
  ..
  * **Projection reference**.  This must be a reference to a Projection that has already been created.
  ..
  * **Keyword**.  This creates a default instance of the specified type, and can be any of the following:

      * *MAPPING_PROJECTION* -- if the `sender <MappingProjection.sender>` and/or its `receiver
        <MappingProjection.receiver>` cannot be inferred from the context in which this specification occurs, then its
        `initialization is deferred <MappingProjection_Deferred_Initialization>` until both of those have been
        determined (e.g., it is used in the specification of a `pathway <Process_Base.pathway>` for a `Process`).
      |
      * *LEARNING_PROJECTION*  (or *LEARNING*) -- this can only be used in the specification of a `MappingProjection`
        (see `tuple <Mapping_Matrix_Specification>` format).  If the `receiver <MappingProjection.receiver>` of the
        MappingProjection projects to a `LearningMechanism` or a `ComparatorMechanism` that projects to one, then a
        `LearningSignal` is added to that LearningMechanism and assigned as the LearningProjection's `sender
        <LearningProjection.sender>`;  otherwise, a LearningMechanism is `automatically created
        <LearningMechanism_Creation>`, along with a LearningSignal that is assigned as the LearningProjection's `sender
        <LearningProjection.sender>`. See `LearningMechanism_Learning_Configurations` for additional details.
      |
      * *CONTROL_PROJECTION* (or *CONTROL*)-- this can be used when specifying a parameter using the `tuple format
        <ParameterState_Tuple_Specification>`, to create a default `ControlProjection` to the `ParameterState` for that
        parameter.  If the `Component <Component>` to which the parameter belongs is part of a `System`, then a
        `ControlSignal` is added to the System's `controller <System_Base.controller>` and assigned as the
        ControlProjection's `sender <ControlProjection.sender>`;  otherwise, the ControlProjection's `initialization
        is deferred <ControlProjection_Deferred_Initialization>` until the Mechanism is assigned to a System,
        at which time the ControlSignal is added to the System's `controller <System_Base.controller>` and assigned
        as its the ControlProjection's `sender <ControlProjection.sender>`.  See `ControlMechanism_Control_Signals` for
        additional details.
      |
      * *GATING_PROJECTION* (or *GATING*)-- this can be used when specifying an `InputState <InputState_Projections>`
        or an `OutputState <OutputState_Projections>`, to create a default `GatingProjection` to the `State <State>`.
        If the GatingProjection's `sender <GatingProjection.sender>` cannot be inferred from the context in which this
        specification occurs, then its `initialization is deferred <GatingProjection_Deferred_Initialization>` until
        it can be determined (e.g., a `GatingMechanism` or `GatingSignal` is created to which it is assigned).
  ..
  * **Projection type**.  This creates a default instance of the specified Projection subclass.  The assignment or
    creation of the Projection's `sender <Projection.sender>` is handled in the same manner as described above for the
    keyword specifications.
  ..
  * **Specification dictionary**.  This can contain an entry specifying the type of Projection, and/or entries
    specifying the value of parameters used to instantiate it. These should take the following form:

      * *PROJECTION_TYPE*: *<name of a Projection type>* --
        if this entry is absent, a default Projection will be created that is appropriate for the context
        (for example, a `MappingProjection` for an `InputState`, a `LearningProjection` for the `matrix
        <MappingProjection.matrix>` parameter of a `MappingProjection`, and a `ControlProjection` for any other
        type of parameter.
      |
      * *PROJECTION_PARAMS*: *Dict[Projection argument, argument value]* --
        the key for each entry of the dictionary must be the name of a Projection parameter, and its value the value
        of the parameter.  It can contain any of the standard parameters for instantiating a Projection (in particular
        its `sender <Projection_Sender>` and `receiver <Projection_Receiver>`, or ones specific to a particular type
        of Projection (see documentation for subclass).  If the `sender <Projection_Sender>` and/or
        `receiver <Projection_Receiver>` are not specified, their assignment and/or creation are handled in the same
        manner as described above for keyword specifications.

      COMMENT:
          WHAT ABOUT SPECIFICATION USING OutputState/ModulatorySignal OR Mechanism?
      COMMENT

      COMMENT:  ??IMPLEMENTED FOR PROJECTION PARAMS??
        Note that parameter
        values in the specification dictionary will be used to instantiate the Projection.  These can be overridden
        during execution by specifying `runtime parameters <Mechanism_Runtime_parameters>` for the Projection,
        either when calling the `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>`
        method for a Mechanism directly, or where it is specified in the `pathway` of a Process.
      COMMENT

.. _Projection_Automatic_Creation:

Automatic creation
~~~~~~~~~~~~~~~~~~

Under some circumstances Projections are created automatically. For example, a `Process` automatically creates a
`MappingProjection` between adjacent `ProcessingMechanisms <ProcessingMechanism>` in its `pathway
<Process_Base.pathway>` if none is specified; and `LearningProjections <LearningProjection>` are automatically created
when :keyword:`learning` is specified for a `Process <Process_Learning_Sequence>` or `System
<System_Execution_Learning>`).

.. _MappingProjection_Deferred_Initialization:

Deferred Initialization
~~~~~~~~~~~~~~~~~~~~~~~

When a Projection is created, its full initialization is `deferred <Component_Deferred_Init>` until its `sender
<ControlProjection.sender>` and `receiver <ControlProjection.receiver>` have been fully specified.  This allows
a Projection to be created before its `sender` and/or `receiver` have been created (e.g., before them in a script),
by calling its constructor without specifying its **sender** or **receiver** arguments. However, for the Projection
to be operational, initialization must be completed by calling its `_deferred_init` method.  Under most conditions
this occurs automatically (e.g., when the projection is assigned to a type of Component that expects to be the
`sender <Projection.sender>` or `receiver <Projection.receiver>` for that type of Projection); these conditions are
described in the section on *Deferred Initialization* for each type of Projection.  Otherwise, the Projection's
`_deferred_init` method must be called explicitly, once the missing attribute assignments have been made.


.. _Projection_Structure:

Structure
---------

In addition to its `function <Projection.function>`, a Projection has two primary attributes: a `sender
<Projection.sender>` and `receiver <Projection.receiver>`.  The types of `State(s) <State>` that can be
assigned to these, and the attributes of those States to which Projections of each type are assigned, are
summarized in the following table, and described in greater detail in the subsections below.

.. _Projection_Table:

+-----------------------------------------------------------------------------------------------------------------+
|            Sender, receiver and attribute assignments for different types of Projections                        |
+----------------------+---------------------------------------+--------------------------------------------------+
|     Projection       |   sender                              |  receiver                                        |
|                      |   (attribute)                         |  (attribute)                                     |
+======================+=======================================+==================================================+
| `MappingProjection`  | `OutputState`                         | `InputState`                                     |
|                      | (`efferents <OutputState.efferents>`) | (`path_afferents <InputState.path_afferents>`)   |
+----------------------+---------------------------------------+--------------------------------------------------+
| `LearningProjection` | `LearningSignal`                      | `ParameterState`                                 |
|                      | (`efferents <OutputState.efferents>`) | (`mod_afferents <ParameterState.mod_afferents>`) |
+----------------------+---------------------------------------+--------------------------------------------------+
| `ControlProjection`  | `ControlSignal`                       | `ParameterState`                                 |
|                      | (`efferents <OutputState.efferents>`) | (`mod_afferents <ParameterState.mod_afferents>`) |
+----------------------+---------------------------------------+--------------------------------------------------+
| `GatingProjection`   | `GatingSignal`                        | `InputState` or `OutputState`                    |
|                      | (`efferents <OutputState.efferents>`) | (`mod_afferents <State_Base.mod_afferents>`)     |
+----------------------+---------------------------------------+--------------------------------------------------+

.. _Projection_Sender:

Sender
~~~~~~

This must be an `OutputState` or a `ModulatorySignal <ModulatorySignal>` (a subclass of OutputState specialized for
`ModulatoryProjections <ModulatoryProjection>`).  The Projection is assigned to the OutputState or ModulatorySignal's
`efferents <State_Base.efferents>` list and, for ModulatoryProjections, to the list of ModulatorySignals specific to
the `AdaptiveMechanism <AdaptiveMechanism>` from which it projects.  The OutputState or ModulatorySignal's `value
<OutputState.value>` is used as the `variable <Function.variable>` for Projection's `function <Projection.function>`.

A sender can be specified as:

  * an **OutputState** or **ModulatorySignal**, as appropriate for the Projection's type, using any of the ways for
    `specifying an OutputState <OutputState_Specification>`.
  ..
  * a **Mechanism**;  for a `MappingProjection`, the Mechanism's `primary OutputState <OutputState_Primary>` is
    assigned as the `sender <Projection.sender>`; for a `ModulatoryProjection <ModulatoryProjection>`, a
    `ModulatorySignal <ModulatorySignal>` of the appropriate type is created and assigned to the Mechanism.

If the `sender <Projection.sender>` is not specified and it can't be determined from the context, or an OutputState
specification is not associated with a Mechanism that can be determined from context, then the initialization of the
Projection is `deferred <Projection_Deferred_Initialization>`.

.. _Projection_Receiver:

Receiver
~~~~~~~~

The `receiver <Projection.receiver>` required by a Projection depends on its type, as listed below:

    * MappingProjection: `InputState`
    * LearningProjection: `ParameterState` (for the `matrix <MappingProjection>` of a `MappingProjection`)
    * ControlProjection: `ParameterState`
    * GatingProjection: `InputState` or OutputState`

A `MappingProjection` (as a `PathwayProjection <PathwayProjection>`) is assigned to the `path_afferents
<State.path_afferents>` attribute of its `receiver <Projection.receiver>`.  The ModulatoryProjections are assigned to
the `mod_afferents <State.mod_afferents>` attribute of their `receiver <Projection.receiver>`.

A `receiver <Projection.receiver>` can be specified as:

  * an existing **State**;
  ..
  * an existing **Mechanism** or **Projection**; which of these is permissible, and how a state is assigned to it, is
    determined by the type of Projection — see subclasses for details).
  ..
  * a **specification dictionary** (see subclasses for details).

.. _Projection_Weight_Exponent:

Weight and Exponent
~~~~~~~~~~~~~~~~~~~

Every Projecton has a `weight <Projection.weight>` and `exponent <Projection.exponent>` attribute. These are applied
to its `value <Projection.value>` before combining it with other Projections that project to the same `State`.  If
both are specified, the `exponent <Projection.exponent>` is applied before the `weight <Projection.weight>`.  These
attributes determine both how the Projection's `value <Projection.value>` is combined with others to determine the
`variable <State.variable>` of the State to which they project.

.. note::
   The `weight <Projection.weight>` and `exponent <Projection.exponent>` attributes of a Projection are not
   normalized, and their aggregate effects contribute to the magnitude of the `variable <State.variable>` to which
   they project.


ParameterStates and Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ParameterStates <ParameterState>` provide the value for each parameter of a Projection and its `function
<Mechanism_Base.function>`.  ParameterStates and their associated parameters are handled in the same way by
Projections as they are for Mechanisms (see `Mechanism_ParameterStates` for details).  The ParameterStates for a
Projection are listed in its `parameter_states <Projection.parameter_states>` attribute.


.. _Projection_Execution:

Execution
---------

A Projection cannot be executed directly.  It is executed when the `State <State>` to which it projects (i.e., its
`receiver <Projection.receiver>`) is updated;  that occurs when the State's owner `Mechanism <Mechanism>` is executed.
When a Projection executes, it gets the value of its `sender <Projection.sender>`, assigns this as the `variable
<Projection.variable>` of its `function <Projection.function>`, calls the `function <Projection.function>`, and
provides the result as to its `receiver <Projection.receiver>`.  The `function <Projection.function>` of a Projection
converts the value received from its `sender <Projection.sender>` to a form suitable as input for its `receiver
<Projection.receiver>`.

.. _Projection_Class_Reference:

"""
import inspect
import typecheck as tc
import warnings

from PsyNeuLink.Components.Component import Component, InitStatus
from PsyNeuLink.Components.ShellClasses import Mechanism, Process, Projection, State
from PsyNeuLink.Globals.Keywords import CONTROL, CONTROL_PROJECTION, GATING, GATING_PROJECTION, INPUT_STATE, LEARNING, LEARNING_PROJECTION, MAPPING_PROJECTION, MATRIX_KEYWORD_SET, MECHANISM, OUTPUT_STATE, PARAMETER_STATE_PARAMS, PROJECTION, PROJECTION_SENDER, PROJECTION_TYPE, kwAddInputState, kwAddOutputState, kwProjectionComponentCategory
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Globals.Utilities import ContentAddressableList, iscompatible, type_match

ProjectionRegistry = {}

kpProjectionTimeScaleLogEntry = "Projection TimeScale"

projection_keywords = set()

PROJECTION_SPEC_KEYWORDS = {MAPPING_PROJECTION,
                            LEARNING, LEARNING_PROJECTION,
                            CONTROL, CONTROL_PROJECTION,
                            GATING, GATING_PROJECTION}

from collections import namedtuple
ConnectionTuple = namedtuple("ConnectionTuple", "state, weight, exponent, projection")


class ProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

# Projection factory method:
# def projection(name=NotImplemented, params=NotImplemented, context=None):
#         """Instantiates default or specified subclass of Projection
#
#         If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (ParameterState)
#         If called with a name string:
#             - if registered in ProjectionRegistry class dictionary as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in ProjectionRegistry:
#             return ProjectionRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Components.Defaults import DefaultProjection
#             return DefaultProjection(name, params)
#

class Projection_Base(Projection):
    """
    Projection_Base(  \
    receiver,         \
    sender=None,      \
    params=None,      \
    name=None,        \
    prefs=None)

    Base class for all Projections.

    .. note::
       Projection is an abstract class and should NEVER be instantiated by a direct call to its constructor.
       It should be created by calling the constructor for a subclass` or by using any of the other methods for
       `specifying a Projection <Projection_In_Context_Specification>`.


    COMMENT:
        Description
        -----------
            Projection category of Component class (default type:  MappingProjection)

        Gotchas
        -------
            When referring to the Mechanism that is a Projection's sender or receiver Mechanism, must add ".owner"

        Class attributes
        ----------------
            + componentCategory (str): kwProjectionFunctionCategory
            + className (str): kwProjectionFunctionCategory
            + suffix (str): " <className>"
            + registry (dict): ProjectionRegistry
            + classPreference (PreferenceSet): ProjectionPreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + ClassDefaults.variable (value): [0]
            + requiredParamClassDefaultTypes = {PROJECTION_SENDER: [str, Mechanism, State]}) # Default sender type
            + paramClassDefaults (dict)
            + FUNCTION (Function class or object, or method)

        Class methods
        -------------
            None

        ProjectionRegistry
        ------------------
            All Projections are registered in ProjectionRegistry, which maintains a dict for each subclass,
              a count for all instances of that type, and a dictionary of those instances
    COMMENT

    Attributes
    ----------

    variable : value
        input to Projection, received from OutputState.value of sender.

    sender : State
        State from which Projection receives its input.

    receiver : State
        State to which Projection sends its output.

    value : value
        Output of Projection, transmitted as variable to InputState of receiver.

    parameter_states : ContentAddressableList[str, ParameterState]
        a list of the Projection's `ParameterStates <Projection_ParameterStates>`, one for each of its specifiable
        parameters and those of its `function <Mechanism_Base.function>` (i.e., the ones for which there are
        arguments in their constructors).  The value of the parameters of the Projection are also accessible as
        attributes of the Projection (using the name of the parameter); the function parameters are listed in the
        Projection's `function_params <Projection.function_params>` attribute, and as attributes of the `Function`
        assigned to its `function_object <Component.function_object>` attribute.

    parameter_states : ContentAddressableList[str, ParameterState]
        a read-only list of the Projection's `ParameterStates <Mechanism_ParameterStates>`, one for each of its
        `configurable parameters <ParameterState_Configurable_Parameters>`, including those of its `function
        <Projection.function>`.  The value of the parameters of the Projection and its `function
        <Projection.function>` are also accessible as (and can be modified using) attributes of the Projection,
        in the same manner as they can for a `Mechanism <Mechanism_ParameterStates>`).

    weight : number
       multiplies `value <Projection.value>` of the Projection after applying `exponent <Projection.exponent>`,
       and before combining with any others that project to the same `State` to determine that State's `variable
       <State.variable>`.

    exponent : number
        exponentiates the `value <Projection.value>` of the Projection, before applying `weight <Projection.weight>`,
        and before combining it with any other Projections that project to the same `State` to determine that State's
        `variable <State.variable>`.

    COMMENT:
        projectionSender : Mechanism, State, or Object
            This is assigned by __init__.py with the default sender state for each subclass.
            It is used if sender arg is not specified in the constructor or when the Projection is assigned.
            If it is different than the default;  where it is used, it overrides the ``sender`` argument even if that is
            provided.

        projectionSender : 1d array
            Used to instantiate projectionSender
    COMMENT

    name : str : default <Projection subclass>-<index>
        the name of the Projection.
        Specified in the **name** argument of the constructor for the Projection;  if not is specified,
        a default is assigned by ProjectionRegistry based on the Projection's subclass
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection.classPreferences
        the `PreferenceSet` for the Projection.
        Specified in the **prefs** argument of the constructor for the Projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    color = 0

    componentCategory = kwProjectionComponentCategory
    className = componentCategory
    suffix = " " + className

    class ClassDefaults(Projection.ClassDefaults):
        variable = [0]

    registry = ProjectionRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    requiredParamClassDefaultTypes = Component.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({PROJECTION_SENDER: [str, Mechanism, State]}) # Default sender type

    def __init__(self,
                 receiver,
                 sender=None,
                 weight=None,
                 exponent=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None):
        """Assign sender, receiver, and execute method and register Mechanism with ProjectionRegistry

        This is an abstract class, and can only be called from a subclass;
           it must be called by the subclass with a context value

# DOCUMENT:  MOVE TO ABOVE, UNDER INSTANTIATION
        Initialization arguments:
            - sender (Mechanism, State or dict):
                specifies source of input to Projection (default: senderDefault)
            - receiver (Mechanism, State or dict)
                 destination of Projection (default: none)
            - params (dict) - dictionary of Projection params:
                + FUNCTION:<method>
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
            - prefs (PreferenceSet or specification dict):
                 if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
                 dict entries must have a preference keyPath as key, and a PreferenceEntry or setting as their value
                 (see Description under PreferenceSet for details)
            - context (str): must be a reference to a subclass, or an exception will be raised

        NOTES:
        * Receiver is required, since can't instantiate a Projection without a receiving State
        * If sender and/or receiver is a Mechanism, the appropriate State is inferred as follows:
            MappingProjection:
                sender = <Mechanism>.output_state
                receiver = <Mechanism>.input_state
            ControlProjection:
                sender = <Mechanism>.output_state
                receiver = <Mechanism>.paramsCurrent[<param>] IF AND ONLY IF there is a single one
                            that is a ParameterState;  otherwise, an exception is raised
        * _instantiate_sender, _instantiate_receiver must be called before _instantiate_function:
            - _validate_params must be called before _instantiate_sender, as it validates PROJECTION_SENDER
            - instantatiate_sender may alter self.instance_defaults.variable, so it must be called before _validate_function
            - instantatiate_receiver must be called before _validate_function,
                 as the latter evaluates receiver.value to determine whether to use self.function or FUNCTION
        * If variable is incompatible with sender's output, it is set to match that and revalidated (_instantiate_sender)
        * if FUNCTION is provided but its output is incompatible with receiver value, self.function is tried
        * registers Projection with ProjectionRegistry

        :param sender: (State or dict)
        :param receiver: (State or dict)
        :param param_defaults: (dict)
        :param name: (str)
        :param context: (str)
        :return: None
        """
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        from PsyNeuLink.Components.States.State import State_Base

        if not isinstance(context, Projection_Base):
            raise ProjectionError("Direct call to abstract class Projection() is not allowed; "
                                 "use projection() or one of the following subclasses: {0}".
                                 format(", ".join("{!s}".format(key) for (key) in ProjectionRegistry.keys())))

        # Register with ProjectionRegistry or create one
        register_category(entry=self,
                          base_class=Projection_Base,
                          name=name,
                          registry=ProjectionRegistry,
                          context=context)

        # # MODIFIED 9/11/16 NEW:
        # Create projection's _stateRegistry and ParameterState entry
        self._stateRegistry = {}

        register_category(entry=ParameterState,
                          base_class=State_Base,
                          registry=self._stateRegistry,
                          context=context)

        try:
            if self.init_status is InitStatus.DEFERRED_INITIALIZATION:
                self.init_args = locals().copy()
                self.init_args['context'] = self
                self.init_args['name'] = name

                # remove local imports
                del self.init_args['ParameterState']
                del self.init_args['State_Base']

                return
        except AttributeError:
            # if this Projection does not have an init_status attribute, we can guarantee that it's not in
            # deferred init state. It's tricky to ensure this attribute always exists due to the nature
            # of deferred init
            pass

# FIX: 6/23/16 NEEDS ATTENTION *******************************************************A
#      NOTE: SENDER IS NOT YET KNOWN FOR DEFAULT control_signal
#      WHY IS self.sender IMPLEMENTED WHEN sender IS NOT??

        self.sender = sender
        self.receiver = receiver

# MODIFIED 6/12/16:  VARIABLE & SENDER ASSIGNMENT MESS:
        # ADD _validate_variable, THAT CHECKS FOR SENDER?
        # WHERE DOES DEFAULT SENDER GET INSTANTIATED??
        # VARIABLE ASSIGNMENT SHOULD OCCUR AFTER THAT

# MODIFIED 6/12/16:  ADDED ASSIGNMENT HERE -- BUT SHOULD GET RID OF IT??
        # AS ASSIGNMENT SHOULD BE DONE IN _validate_variable, OR WHEREVER SENDER IS DETERMINED??
# FIX:  NEED TO KNOW HERE IF SENDER IS SPECIFIED AS A MECHANISM OR STATE
        try:
            variable = self._update_variable(sender.value)
        except:
            try:
                if self.receiver.prefs.verbosePref:
                    warnings.warn("Unable to get value of sender ({0}) for {1};  will assign default ({2})".
                                  format(sender, self.name, self.ClassDefaults.variable))
                variable = self._update_variable(None)
            except AttributeError:
                raise ProjectionError("{} has no receiver assigned".format(self.name))

        # MODIFIED 6/27/17 NEW: commented this out because this is throwing an error as follows: -Changyan
        # AttributeError: 'MappingProjection' object has no attribute '_prefs'
        # MODIFIED 4/21/17 NEW: [MOVED FROM MappingProjection._instantiate_receiver]
        # Assume that if receiver was specified as a Mechanism, it should be assigned to its (primary) InputState
        if isinstance(self.receiver, Mechanism):
            # if (len(self.receiver.input_states) > 1 and
            #         (self.prefs.verbosePref or self.receiver.prefs.verbosePref)):
            #     print("{0} has more than one InputState; {1} was assigned to the first one".
            #           format(self.receiver.owner.name, self.name))
            self.receiver = self.receiver.input_state
        # MODIFIED 4/21/17 END


# FIX: SHOULDN'T default_variable HERE BE sender.value ??  AT LEAST FOR MappingProjection?, WHAT ABOUT ControlProjection??
# FIX:  ?LEAVE IT TO _validate_variable, SINCE SENDER MAY NOT YET HAVE BEEN INSTANTIATED
# MODIFIED 6/12/16:  ADDED ASSIGNMENT ABOVE
#                   (TO HANDLE INSTANTIATION OF DEFAULT ControlProjection SENDER -- BUT WHY ISN'T VALUE ESTABLISHED YET?
        # Validate variable, function and params, and assign params to paramInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        super(Projection_Base, self).__init__(default_variable=variable,
                                              param_defaults=params,
                                              name=self.name,
                                              prefs=prefs,
                                              context=context.__class__.__name__)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate PROJECTION_SENDER and/or sender arg (current self.sender), and assign one of them as self.sender

        Check:
        - that PROJECTION_SENDER is a Mechanism or State
        - if it is different from paramClassDefaults[PROJECTION_SENDER], use it
        - if it is the same or is invalid, check if sender arg was provided to __init__ and is valid
        - if sender arg is valid use it (if PROJECTION_SENDER can't be used);
        - otherwise use paramClassDefaults[PROJECTION_SENDER]
        - when done, sender is assigned to self.sender

        Note: check here only for sender's type, NOT content (e.g., length, etc.); that is done in _instantiate_sender

        :param request_set:
        :param target_set:
        :param context:
        :return:
        """

        super(Projection, self)._validate_params(request_set, target_set, context)

        # try:
        #     sender_param = target_set[PROJECTION_SENDER]
        # except KeyError:
        #     # This should never happen, since PROJECTION_SENDER is a required param
        #     raise ProjectionError("Program error: required param \'{0}\' missing in {1}".
        #                           format(PROJECTION_SENDER, self.name))

        if PROJECTION_SENDER in target_set:
            sender_param = target_set[PROJECTION_SENDER]
            # PROJECTION_SENDER is either an instance or class of Mechanism or State:
            if (isinstance(sender_param, (Mechanism, State)) or
                    (inspect.isclass(sender_param) and issubclass(sender_param, (Mechanism, State)))):
                # it is NOT the same as the default, use it
                if sender_param is not self.paramClassDefaults[PROJECTION_SENDER]:
                    self.sender = sender_param
                # it IS the same as the default, but sender arg was not provided, so use it (= default):
                elif self.sender is None:
                    self.sender = sender_param
                    if self.prefs.verbosePref:
                        warnings.warn("Neither {0} nor sender arg was provided for {1} projection to {2}; "
                                      "default ({3}) will be used".format(PROJECTION_SENDER,
                                                                          self.name,
                                                                          self.receiver.owner.name,
                                                                          sender_param.__class__.__name__))
                # it IS the same as the default, so check if sender arg (self.sender) is valid
                elif not (isinstance(self.sender, (Mechanism, State, Process)) or
                              # # MODIFIED 12/1/16 OLD:
                              # (inspect.isclass(self.sender) and
                              #      (issubclass(self.sender, Mechanism) or issubclass(self.sender, State)))):
                              # MODIFIED 12/1/16 NEW:
                              (inspect.isclass(self.sender) and issubclass(self.sender, (Mechanism, State)))):
                              # MODIFIED 12/1/16 END
                    # sender arg (self.sender) is not valid, so use PROJECTION_SENDER (= default)
                    self.sender = sender_param
                    if self.prefs.verbosePref:
                        warnings.warn("{0} was not provided for {1} projection to {2}, "
                                      "and sender arg ({3}) is not valid; default ({4}) will be used".
                                      format(PROJECTION_SENDER,
                                             self.name,
                                             self.receiver.owner.name,
                                             self.sender,
                                             sender_param.__class__.__name__))

        # FIX: IF PROJECTION, PUT HACK HERE TO ACCEPT AND FORGO ANY FURTHER PROCESSING??
                # IS the same as the default, and sender arg was provided, so use sender arg
                else:
                    pass
            # PROJECTION_SENDER is not valid, and:
            else:
                # sender arg was not provided, use paramClassDefault
                if self.sender is None:
                    self.sender = self.paramClassDefaults[PROJECTION_SENDER]
                    if self.prefs.verbosePref:
                        warnings.warn("{0} ({1}) is invalid and sender arg ({2}) was not provided;"
                                      " default {3} will be used".
                                      format(PROJECTION_SENDER, sender_param, self.sender,
                                             self.paramClassDefaults[PROJECTION_SENDER]))
                # sender arg is also invalid, so use paramClassDefault
                elif not isinstance(self.sender, (Mechanism, State)):
                    self.sender = self.paramClassDefaults[PROJECTION_SENDER]
                    if self.prefs.verbosePref:
                        warnings.warn("Both {0} ({1}) and sender arg ({2}) are both invalid; default {3} will be used".
                                      format(PROJECTION_SENDER, sender_param, self.sender,
                                             self.paramClassDefaults[PROJECTION_SENDER]))
                else:
                    self.sender = self.paramClassDefaults[PROJECTION_SENDER]
                    if self.prefs.verbosePref:
                        warnings.warn("{0} ({1}) is invalid; sender arg ({2}) will be used".
                                      format(PROJECTION_SENDER, sender_param, self.sender))
                if not isinstance(self.paramClassDefaults[PROJECTION_SENDER], (Mechanism, State)):
                    raise ProjectionError("Program error: {0} ({1}) and sender arg ({2}) for {3} are both "
                                          "absent or invalid and default (paramClassDefault[{4}]) is also invalid".
                                          format(PROJECTION_SENDER,
                                                 # sender_param.__name__,
                                                 # self.sender.__name__,
                                                 # self.paramClassDefaults[PROJECTION_SENDER].__name__))
                                                 sender_param,
                                                 self.sender,
                                                 self.name,
                                                 self.paramClassDefaults[PROJECTION_SENDER]))

    def _instantiate_attributes_before_function(self, context=None):
        self._instantiate_sender(context=context)
        self._instantiate_parameter_states(context=context)

    def _instantiate_parameter_states(self, context=None):

        from PsyNeuLink.Components.States.ParameterState import _instantiate_parameter_states
        _instantiate_parameter_states(owner=self, context=context)


    def _instantiate_sender(self, context=None):
        """Assign self.sender to OutputState of sender and insure compatibility with self.instance_defaults.variable

        Assume self.sender has been assigned in _validate_params, from either sender arg or PROJECTION_SENDER
        Validate, set self.instance_defaults.variable, and assign projection to sender's efferents attribute

        If self.sender is a Mechanism, re-assign it to <Mechanism>.output_state
        If self.sender is a State class reference, validate that it is a OutputState
        Assign projection to sender's efferents attribute
        If self.value / self.instance_defaults.variable is None, set to sender.value
        """

        from PsyNeuLink.Components.States.OutputState import OutputState
        from PsyNeuLink.Components.States.ParameterState import ParameterState

        # If sender is specified as a Mechanism (rather than a State),
        #     get relevant OutputState and assign it to self.sender
        # IMPLEMENTATION NOTE: Assume that sender should be the primary OutputState; if that is not the case,
        #                      sender should either be explicitly assigned, or handled in an override of the
        #                      method by the relevant subclass prior to calling super
        if isinstance(self.sender, Mechanism):
            self.sender = self.sender.output_state

        # At this point, self.sender should be a OutputState
        if not isinstance(self.sender, OutputState):
            raise ProjectionError("Sender specified for {} ({}) must be a Mechanism or an OutputState".
                                  format(self.name, self.sender))

        # Assign projection to sender's efferents list attribute
        if not self in self.sender.efferents:
            self.sender.efferents.append(self)

        # Validate projection's variable (self.instance_defaults.variable) against sender.output_state.value
        if iscompatible(self.instance_defaults.variable, self.sender.value):
            # Is compatible, so assign sender.output_state.value to self.instance_defaults.variable
            self.instance_defaults.variable = self.sender.value

        else:
            # Not compatible, so:
            # - issue warning
            if self.prefs.verbosePref:
                warnings.warn(
                    "The variable ({0}) of {1} projection to {2} is not compatible with output ({3})"
                    " of function {4} for sender ({5}); it has been reassigned".format(
                        self.instance_defaults.variable,
                        self.name,
                        self.receiver.owner.name,
                        self.sender.value,
                        self.sender.function.__class__.__name__,
                        self.sender.owner.name
                    )
                )
            # - reassign self.instance_defaults.variable to sender.value
            self._instantiate_defaults(variable=self.sender.value, context=context)

    def _instantiate_attributes_after_function(self, context=None):
        self._instantiate_receiver(context=context)

    def _instantiate_receiver(self, context=None):
        """Call receiver's owner to add projection to its afferents list

        Notes:
        * Assume that subclasses implement this method in which they:
          - test whether self.receiver is a Mechanism and, if so, replace with State appropriate for projection
          - calls this method (as super) to assign projection to the Mechanism
        * Constraint that self.value is compatible with receiver.input_state.value
            is evaluated and enforced in _instantiate_function, since that may need to be modified (see below)
        * Verification that projection has not already been assigned to receiver is handled by _add_projection_to;
            if it has, a warning is issued and the assignment request is ignored

        :param context: (str)
        :return:
        """
        # IMPLEMENTATION NOTE: since projection is added using Mechanism.add_projection(projection, state) method,
        #                      could add state specification as arg here, and pass through to add_projection()
        #                      to request a particular state
        # IMPLEMENTATION NOTE: should check that projection isn't already received by receivers

        if isinstance(self.receiver, State):
            _add_projection_to(receiver=self.receiver.owner,
                               state=self.receiver,
                               projection_spec=self,
                               context=context)

        # This should be handled by implementation of _instantiate_receiver by projection's subclass
        elif isinstance(self.receiver, Mechanism):
            raise ProjectionError("PROGRAM ERROR: receiver for {0} was specified as a Mechanism ({1});"
                                  "this should have been handled by _instantiate_receiver for {2}".
                                  format(self.name, self.receiver.name, self.__class__.__name__))

        else:
            raise ProjectionError("Unrecognized receiver specification ({0}) for {1}".format(self.receiver, self.name))

    def _update_parameter_states(self, runtime_params=None, time_scale=None, context=None):
        for state in self._parameter_states:
            state_name = state.name
            state.update(params=runtime_params, time_scale=time_scale, context=context)

            # Assign ParameterState's value to parameter value in runtime_params
            if runtime_params and state_name in runtime_params[PARAMETER_STATE_PARAMS]:
                param = param_template = runtime_params
            # Otherwise use paramsCurrent
            else:
                param = param_template = self.paramsCurrent

            # Determine whether template (param to type-match) is at top level or in a function_params dictionary
            try:
                param_template[state_name]
            except KeyError:
                param_template = self.function_params

            # Get its type
            param_type = type(param_template[state_name])
            # If param is a tuple, get type of parameter itself (= 1st item;  2nd is projection or Modulation)
            if param_type is tuple:
                param_type = type(param_template[state_name][0])

            # Assign version of ParameterState.value matched to type of template
            #    to runtime param or paramsCurrent (per above)
            # FYI (7/18/17 CW) : in addition to the params and attribute being set, the state's variable is ALSO being
            # set by the statement below. For example, if state_name is 'matrix', the statement below sets
            # params['matrix'] to state.value, calls setattr(state.owner, 'matrix', state.value), which sets the
            # 'matrix' parameter state's variable to ALSO be equal to state.value! If this is unintended, please change.
            param[state_name] = type_match(state.value, param_type)

    def add_to(self, receiver, state, context=None):
        _add_projection_to(receiver=receiver, state=state, projection_spec=self, context=context)

    @property
    def parameter_states(self):
        return self._parameter_states

    @parameter_states.setter
    def parameter_states(self, value):
        # IMPLEMENTATION NOTE:
        # This keeps parameter_states property readonly,
        #    but averts exception when setting paramsCurrent in Component (around line 850)
        pass

def _is_projection_spec(spec, include_matrix_spec=True):
    """Evaluate whether spec is a valid Projection specification

    Return `True` if spec is any of the following:
    + Projection class (or keyword string constant for one):
    + Projection object:
    + 2-item tuple of which the second is a projection_spec (checked recursively with thi method):
    + specification dict containing:
        + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
    + valid matrix specification (if include_matrix_spec is set to `True`)

    Otherwise, return :keyword:`False`
    """

    if inspect.isclass(spec) and issubclass(spec, Projection):
        return True
    if isinstance(spec, Projection):
        return True
    if isinstance(spec, dict) and PROJECTION_TYPE in spec:
        return True
    if isinstance(spec, str) and spec in PROJECTION_SPEC_KEYWORDS:
        return True
    if include_matrix_spec:
        if isinstance(spec, str) and spec in MATRIX_KEYWORD_SET:
            return True
        from PsyNeuLink.Components.Functions.Function import get_matrix
        if get_matrix(spec) is not None:
            return True
    if isinstance(spec, tuple) and len(spec) == 2:
        # Call recursively on first item, which should be a standard projection spec
        if _is_projection_spec(spec[0]):
            # IMPLEMENTATION NOTE: keywords must be used to refer to subclass, to avoid import loop
            if _is_projection_subclass(spec[1], MAPPING_PROJECTION):
                return True
            if _is_projection_subclass(spec[1], LEARNING_PROJECTION):
                return True
            if _is_projection_subclass(spec[1], CONTROL_PROJECTION):
                return True
            if _is_projection_subclass(spec[1], GATING_PROJECTION):
                return True
    return False


def _is_projection_subclass(spec, keyword):
    """Evaluate whether spec is a valid specification of type

    keyword must specify a class registered in ProjectionRegistry

    Return true if spec ==
    + keyword
    + subclass of Projection associated with keyword (from ProjectionRegistry)
    + instance of the subclass
    + specification dict for instance of the subclass:
        keyword is a keyword for an entry in the spec dict
        keyword[spec] is a legal specification for the subclass

    Otherwise, return :keyword:`False`
    """
    if spec is keyword:
        return True
    # Get projection subclass specified by keyword
    try:
        type = ProjectionRegistry[keyword]
    except KeyError:
        pass
    else:
        # Check if spec is either the name of the subclass or an instance of it
        if inspect.isclass(spec) and issubclass(spec, type):
            return True
        if isinstance(spec, type):
            return True
    # spec is a specification dict for an instance of the projection subclass
    if isinstance(spec, dict) and keyword in spec:
        # Recursive call to determine that the entry of specification dict is a legal spec for the projection subclass
        if _is_projection_subclass(spec[keyword], keyword):
            return True
    return False


# MODIFIED 9/30/17 NEW:
# FIX: NEED TO ADD RECOGNITION OF PROJECTION AS THE STATE SPECIFICATION ITSELF (OR JUST USE PROJECTION SPEC)
# FIX: REPLACE "PROJECTIONS" WITH "CONNECTIONS"
# FIX: IN RECURSIVE CALLS TO _parse_state_spec, SPECIFY THAT IT HAS TO RETURN AN INSTANTIATED STATE
# FIX: MAKE SURE IT IS OK TO USE DICT PASSED IN (as params) AND NOT INADVERTENTLY OVERWRITING STUFF HERE
# FIX: ADD FACILITY TO SPECIFY WEIGHTS AND/OR EXPONENTS AND PROJECTION_SPEC FOR EACH ConnectWith ITEM:
#      CHANGE *PROJECTIONS* to *CONNECTS_WITH*
#      MAKE EACH ENTRY OF CONNECTS_WITH A DICT OR TUPLE:
#          DICT ENTRIES: *STATE*, *WEIGHT*, *EXPONENT*, *PROJECTION*
#          TUPLE: (State, weight, exponent, projection_spec)
#      PURPOSE:  Resolve to set of specs that can be handed to Composition to instantiate
#      PROJECT SHOULD BE USED TO INSTANTIATE THE PROJECTION TO/FROM THE SPECIFIED STATE
#      WEIGHTS AND EXPONENTS SHOULD BE USED BY THE InputState's LinearCombination Function
#          (AKIN TO HOW THE MECHANISM'S FUNCTION COMBINES InputState VALUES)
#          (NOTE: THESE ARE DISTINCT FROM THE WEIGHT AND EXPONENT FOR THE InputState ITSELF)
#      THIS WOULD ALLOW TWO LEVELS OF HIEARCHICAL NESTING OF ALGEBRAIC COMBINATIONS OF INPUT VALUES TO A MECHANISM
# @tc.typecheck
# def _parse_projection_specs(connectee_state_type:is_state_class,
def _parse_projection_specs(connectee_state_type,
                            owner,
                            # connections:tc.any(State, Mechanism, dict, tuple, ConnectionTuple)):
                            connections):
    """Parse specification(s) for States to/from which the connectee_state_type should be connected

    TERMINOLOGY NOTE:
        "CONNECTION" is used instead of "PROJECTION" because:
            - the method abstracts over type and direction of Projection, so it is ambiguous whether
                the projection involved is to or from connectee_state_type; however, can always say it "connects with"
            - specification is not always (in fact, usually is not) in the form of a Projection;
                usually it is a Mechanism or State to/from which the connectee_state_type should send/receive the Projection,
                so calling the method "_parse_projections" would be misleading.

    This method deals with CONNECTION specifications that are made in one of the following places/ways:
        - *CONNECTIONS* entry of a State specification dict [SYNONYM: *PROJECTIONS* - for backward compatiability];
        - last item of a State specification tuple.

    In both cases, the CONNECTION specification can be a single (stand-alone) item or a list of them.

    Each CONNECTION specification can, itself, be one of the following:
        * State - must be an instantiated State;
        * Mechanism - primary State is used, if applicable, otherwise an exception is generated;
        * dict - must have the first and can have any of the additional entries below:
            *STATE*:<state_spec> - required; must resolve to an instantiated state;  can use any of the following:
                                       State - the State is used;
                                       Mechanism - primary State will be used if appropriate,
                                                   otherwise generates an exception;
                                       {Mechanism:state_spec or [state_spec<, state_spec...>]} -
                                                   each state_spec must be for an instantiated State of the Mechanism,
                                                   referenced by its name or in a CONNECTION specification that uses
                                                   its name (or, for completeness, the State itself);
                                                   _parse_connections() is called recursively for each state_spec
                                                   (first replacing the name with the actual state);
                                                   and returns a list of ConnectionTuples; any weights, exponents,
                                                   or projections assigned in those tuples are left;  otherwise, any
                                                   values in the entries of the outer dict (below) are assigned;
                                                   note:  the dictionary can have multiple Mechanism entries
                                                          (which permits the same defaults to be assigned to all the
                                                          States for all of the Mechanisms)
                                                          or they can be assigned each to their own dictionary
                                                          (which permits different defaults to be assigned to the
                                                          States for each Mechanism);
            *WEIGHT*:<int> - optional; specifies weight given to projection by receiving InputState
            *EXPONENT:<int> - optional; specifies weight given to projection by receiving InputState
            *PROJECTION*:<projection_spec> - optional; specifies projection (instantiated or matrix) for connection
                                             default is PROJECTION_TYPE specified for STATE
        * tuple or list of tuples: (specification requirements same as for dict above);  each must be:
            (state_spec, projection_spec) or
            (state_spec, weight, exponent, projection_spec)

    **DEPRECATED** [SHOULD ONLY MATTER FOR OBJECTIVE MECHANISMS]:
        # If params is a dict:
        #     entry key can be any of the following, with the corresponding value:
        #         Mechanism:<connection_spec> or [connection_spec<, connection_spec..>]
        #            - generates projection for each specified ConnectWith State
        #         MECHANISMS:<Mechanism> or [Mechanism<, Mechanism>]
        #            - generates projection for primary ConnectWith State of each Mechanism
        #
        # If params is a tuple:
        #     - the first must be a BaseSpec specification (processed by _parse_state_spec, not here)
        #     - if it has two items, the second must resolve to a ConnectWith
        #         (parsed in a recursive call to _parse_state_specific_entries)
        #     - if it has three or four items:
        #         - the second is a weight specification
        #         - the third is an exponent specification
        #         - the fourth (optional) must resolve to an ConnectWith specification
        #           (parsed in a recursive call to _parse_state_specific_entries)

    Returns list of ConnectionTuples

    """

    # FIX: vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # FIX: MOVE HANDLING OF ALL THIS TO REGISTRY

    from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ProcessingMechanism import ProcessingMechanism_Base
    from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanism.LearningMechanism import LearningMechanism
    from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism import ControlMechanism
    from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanism.GatingMechanism import GatingMechanism
    from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
    from PsyNeuLink.Components.States.State import _get_existing_state
    from PsyNeuLink.Components.States.InputState import InputState
    from PsyNeuLink.Components.States.OutputState import OutputState
    from PsyNeuLink.Components.States.ParameterState import ParameterState
    from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
    from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
    from PsyNeuLink.Components.States.ModulatorySignals.GatingSignal import GatingSignal
    from PsyNeuLink.Globals.Keywords import SENDER, RECEIVER, INPUT_STATES, OUTPUT_STATES, \
                                            LEARNING_SIGNALS, CONTROL_SIGNALS, GATING_SIGNALS

    # BaseSpec = connectee_state_type

    # CONNECTION CHARACTERISTICS THAT MUST BE DECLARED BY EACH TYPE (SUBCLASS) OF State
    # ConnectWith : State
    #    - specifies the type (subclass) of State to which the connectee_state_type should be assigned projection(s)
    #    - [TBI] subclass' attribute: connect_with [??CURRENTLY:  PROJECTION_TYPE]
    # connect_with_attr : str
    #    - specifies the name of the attribute of the Mechanism that holds the states of the ConnectWith's type
    #    - [TBI] subclass' attribute: connect_with_attr
    # CONNECTIONS_KEYWORD : str
    #    - specifies the keyword used in State specification dictionary for entry specifying States to connect to
    #    - [TBI] subclass' attribute: connect_with_keyword
    # PROJECTION_SOCKET : [SENDER or RECEIVER]
    #    - specifies for this method whether to use a Projection's sender or receiver for the connection
    #    - [TBI] subclass' attribute: projection_socket
    # Modulator : ModulatorySignal
    #    -  class of ModulatorySignal that can send ModulatoryProjection to the connectee_state_type
    #    - [TBI] subclass' attribute: modulator
    # MOD_KEYWORD : str
    #    - specifies the keyword used in State specification dictionary for entry specifying ModulatorySignal
    #    - [TBI] subclass' attribute: mod_keyword

    if not inspect.isclass(connectee_state_type):
        raise ProjectionError("Called for {} with \'connectee_state_type\' arg ({}) that is not a class".
                         format(owner.name, connectee_state_type))
    else:
        BaseSpec = connectee_state_type

    # Request for afferent Projections (projection socket is SENDER)
    if isinstance(owner, Mechanism) and issubclass(connectee_state_type, InputState):
        ConnectWith = OutputState            # type of State to which the connectee connects
        connect_with_attr = 'output_states'  # attribute that holds the ConnectWith States
        CONNECTIONS_KEYWORD = OUTPUT_STATES  # keyword used in a State specification dictionary for connection specs
        PROJECTION_SOCKET = SENDER           # socket of the Projection that connects to the ConnectWith State
        Modulator = GatingSignal             # type of ModulatorySignal the connecteed can receiver
        # MOD_KEYWORD = GATING_SIGNALS         # keyword used in a State specification dictionary for Modulatory specs
    elif isinstance(owner, Mechanism) and issubclass(connectee_state_type, ParameterState):
        ConnectWith = ControlSignal
        connect_with_attr = 'control_signals'
        CONNECTIONS_KEYWORD = CONTROL_SIGNALS
        PROJECTION_SOCKET = SENDER
        Modulator = ControlSignal
        # MOD_KEYWORD = CONTROL_SIGNALS
    elif isinstance(owner, MappingProjection) and issubclass(connectee_state_type, ParameterState):
        ConnectWith = LearningSignal
        connect_with_attr = 'learning_signals'
        CONNECTIONS_KEYWORD = LEARNING_SIGNALS
        PROJECTION_SOCKET = SENDER
        Modulator = LearningSignal
        MOD_KEYWORD = LEARNING_SIGNALS

    # Request for efferent Projections (projection socket is RECEIVER)
    elif isinstance(owner, ProcessingMechanism_Base) and issubclass(connectee_state_type, OutputState):
        ConnectWith = InputState
        connect_with_attr = 'input_states'
        CONNECTIONS_KEYWORD = INPUT_STATES
        PROJECTION_SOCKET = RECEIVER
        Modulator = GatingSignal
        MOD_KEYWORD = GATING_SIGNALS
    elif isinstance(owner, ControlMechanism) and issubclass(connectee_state_type, ControlSignal):
        ConnectWith = ParameterState
        connect_with_attr = 'parameter_states'
        # CONNECTIONS_KEYWORD = CONTROLLED_PARAMS
        PROJECTION_SOCKET = RECEIVER
        Modulator = None
        MOD_KEYWORD = None
    elif isinstance(owner, LearningMechanism) and issubclass(connectee_state_type, LearningSignal):
        ConnectWith = ParameterState
        connect_with_attr = 'parameter_states'
        # CONNECTIONS_KEYWORD = LEARNED_PROJECTIONS
        PROJECTION_SOCKET = RECEIVER
        Modulator = None
        MOD_KEYWORD = None
    elif isinstance(owner, GatingMechanism) and issubclass(connectee_state_type, GatingSignal):
        # FIX:
        ConnectWith = InputState or OutputState
        # FIX:
        connect_with_attr = 'input_states' or 'output_states'
        # CONNECTIONS_KEYWORD = GATED_STATES
        PROJECTION_SOCKET = RECEIVER
        Modulator = None
        MOD_KEYWORD = None

    else:
        raise ProjectionError("Called for {} with unsupported owner type ({}), connectee_state_type ({}), "
                         "or combination of them".
                         format(owner.name, owner.__class__.__name__, connectee_state_type.__name__))

    # FIX: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


    from PsyNeuLink.Globals.Keywords import MECHANISMS

    DEFAULT_WEIGHT = None
    DEFAULT_EXPONENT = None
    # DEFAULT_PROJECTION = PROJECTION_TYPE
    DEFAULT_PROJECTION = None

    # Convert to list for subsequent processing
    if not isinstance(connections, list):
        connections = [connections]
    connect_with_states = []

    for connection in connections:

        # If a Mechanism, State, or str (name) is used to specify the connection on its own (i.e., w/o dict or tuple)
        #     put in tuple with default values of other specs, and call _parse_projection_specs recursively
        #     to validate the state spec and append ConnectionTuple to connect_with_states
        if isinstance(connection, (Mechanism, State)):
            connection_tuple =  (connection, DEFAULT_WEIGHT, DEFAULT_EXPONENT, DEFAULT_PROJECTION)
            connect_with_states.extend(_parse_projection_specs(connectee_state_type, owner, connection_tuple))

        # If a projection specification is used to specify the connection:
        #  assign the projection specification to the projection_specification item of the tuple,
        #  but also leave it is as the connection specification (it will get resolved to a State reference when the
        #    tuple is created in the recursive call to _parse_projection_specs below).
        if _is_projection_spec(connection, include_matrix_spec=False):
            projection_spec = connection
            connection_tuple =  (connection, DEFAULT_WEIGHT, DEFAULT_EXPONENT, projection_spec)
            connect_with_states.extend(_parse_projection_specs(connectee_state_type, owner, connection_tuple))

        # Dict of one or more Mechanism specifications, used to specify individual States of (each) Mechanism;
        #   convert all entries to tuples and call _parse_projection_specs recursively to generate ConnectionTuples;
        #   main purpose of this is to resolve any str references to name of state (using context of owner Mechanism)
        elif isinstance(connection, dict):

            # Check that dict has at least one entry with a Mechanism as the key
            if (not any(isinstance(spec, Mechanism) for spec in connection) and
                    not any(spec == STATES for spec in connection)):
                raise ProjectionError("There are no {} or {} entries in the connection specification dictionary for {}".
                                 format(Mechanism.__name__, STATES, owner.name))

            # Add default WEIGHT, EXPONENT, and/or PROJECTION specification for any that are not aleady in the dict
            #    (used as the default values for all the States of all Mechanisms specified for this dict;
            #    can use different dicts to implement different sets of defaults for the States of diff Mechanisms)
            if not WEIGHT in connection:
                connection[WEIGHT] = DEFAULT_WEIGHT
            if not EXPONENT in connection:
                connection[EXPONENT] = DEFAULT_EXPONENT
            if not PROJECTION in connection:
                connection[PROJECTION] = DEFAULT_PROJECTION

            # Now process each entry that has *STATES* or a Mechanism as its key
            for key, state_connect_specs in connection.items():

                # Convert state_connect_specs to a list for subsequent processing
                if not isinstance(state_connect_specs, list):
                    state_connect_specs = [state_connect_specs]

                for state_connect_spec in state_connect_specs:

                    # State, str (name) or Projection specification
                    if isinstance(state_connect_spec, (State, str, _is_projection_spec)):

                        # If state_connection_spec is a string (name), it has to be in a Mechanism entry
                        if isinstance(state_connect_spec, str) and isinstance(key, Mechanism):
                            mech = key
                        else:
                            raise ProjectionError("{} specified by name ({}) is not in a {} entry".
                                             format(State.__name__, state_connect_spec, Mechanism.__name__))

                        # Call _get_existing_state to parse if it is a str,
                        #    and in either case to make sure it belongs to mech
                        state = _get_existing_state(owner=owner,
                                                    state_spec=state_connect_spec,
                                                    state_type=connect_with_attr,
                                                    mech=mech,
                                                    projection_socket=PROJECTION_SOCKET)

                        # Assign state along with dict's default values to tuple
                        state_connect_spec = (state,
                                              connection[WEIGHT],
                                              connection[EXPONENT],
                                              connection[PROJECTION])

                    # Dict specification for state itself
                    elif isinstance(state_connect_spec, dict):
                        # Get STATE entry
                        state_spec = state_connect_spec[STATE]
                        # Parse it to get reference to actual State make sure it belongs to mech:
                        state = _get_existing_state(owner=owner,
                                                    state_spec=state_spec,
                                                    state_type=connect_with_attr,
                                                    mech=mech,
                                                    projection_socket=PROJECTION_SOCKET)
                        # Re-assign to STATE entry of dict (to preserve any other connection specifications in dict)
                        state_connect_spec[STATE] = state

                    # Tuple specification for State itself
                    elif isinstance(state_connect_spec, tuple):
                        # Get STATE entry
                        state_spec = state_connect_spec[0]
                        # Parse it to get reference to actual State make sure it belongs to mech:
                        state = _get_existing_state(owner=owner,
                                                    state_spec=state_spec,
                                                    state_type=connect_with_attr,
                                                    mech=mech,
                                                    projection_socket=PROJECTION_SOCKET)
                        # Replace parsed value in original tuple, but...
                        #    tuples are immutable, so have to create new one, with state_spec as (new) first item
                        # Get items from original tuple
                        state_connect_spec_tuple_items = [item for item in state_connect_spec]
                        # Replace state_spec
                        state_connect_spec_tuple_items[0] = state
                        # Reassign to new tuple
                        state_connect_spec = tuple(state_connect_spec_tuple_items)

                    # Recusively call _parse_projection_specs to get ConnectionTuple and append to connect_with_states
                    connect_with_states.extend(_parse_projection_specs(connectee_state_type, owner, state_connect_spec))

        # Process tuple, including final validation of State specification
        # Tuple could be:
        #     (state_spec, projection_spec) or
        #     (state_spec, weight, exponent, projection_spec)
        # Note:  this is NOT the same as the State specification tuple (which can have a similar format);
        #        the weights and exponents here specify *individual* Projections to a particular state,
        #            (vs. weights and exponents for an entire state (as for InputState);
        #        State specification tuple is handled in the _parse_state_specific_tuple() method of State subclasses

        elif isinstance(connection, tuple):
        # Notes:
        #    - first item is assumed to always be a specification for the State itself

            if len(connection) == 2:
                state_spec, projection_spec = connection
                weight = DEFAULT_WEIGHT
                exponent = DEFAULT_EXPONENT
            elif len(connection) == 4:
                state_spec, weight, exponent, projection_spec = connection
            else:
                # FIX: FINISH ERROR MESSAGE
                raise ProjectionError("{} specificaton tuple for {} ({}) must have either two or four items".
                                      format(connectee_state_type.__name__, owner.name, connection))

            # # Validate state specification and get actual state referenced
            state = _get_existing_state(owner=owner,
                                        state_spec=state_spec,
                                        state_type=ConnectWith,
                                        mech_state_attribute=connect_with_attr,
                                        projection_socket=PROJECTION_SOCKET)

            # Validate that the type of the State to be connected to is consistent with connectee's type:
            if inspect.isclass(state):
                state_type = state
            else:
                state_type = state.__class__
            if not issubclass(state_type, ConnectWith):
                raise ProjectionError("Connection was specified for a(n) {} of {} to a(n) {} of {} "
                                      "that is of the wrong type; should be {}".
                                      format(connectee_state_type.__name__,
                                             owner.name,
                                             state_type.__name__,
                                             state_spec.name,
                                             ConnectWith.__name__))

            # Validate projection specification
            if projection_spec is not None:
                if _is_projection_spec(projection_spec):
                    _validate_connection_request(owner, ConnectWith, projection_spec, PROJECTION_SOCKET, connectee_state_type)
                else:
                    raise ProjectionError("Invalid specification of {} ({}) for connection between {} and {} of {}.".
                                     format(Projection.__class__.__name__,
                                            projection_spec,
                                            state.name,
                                            connectee_state_type.__name__,
                                            owner.name))

            connect_with_states.extend([ConnectionTuple(state, weight, exponent, projection_spec)])

    if not all(isinstance(connection_tuple, ConnectionTuple) for connection_tuple in connect_with_states):
        raise ProjectionError("PROGRAM ERROR: Not all items are ConnectionTuples for {}".format(owner.name))

    return connect_with_states

@tc.typecheck
def _validate_connection_request(
        owner,                                   # Owner of State seeking connection
        connect_with_state:type,                 # State to which connection is being sought
        projection_spec:_is_projection_spec,     # projection specification
        projection_socket:str,                   # socket of Projection to be connected to target state
        connectee_state:tc.optional(type)=None): # State for which connection is being sought

    """Validate that a Projection specification is compatible with the State to which a connection is specified

    Carries out undirected validation (i.e., without knowing whether the connectee is the sender or receiver).
    Use _validate_receiver or ([TBI] validate_sender) for directed validation.
    Note: connectee_state is used only for name in errors

    If projection_spec is a Projection:
        - if it is instantiated, compare the projection_socket specified (sender or receiver) with connect_with_state
        - if it in deferred_init, check to see if the specified projection_socket has been specified in init_args;
            otherwise, use Projection's type
    If projection_spec is a class specification, use Projection's type
    If projection_spec is a dict:
        - check if there is an entry for the socket and if so, use that
        - otherwise, check to see if there is an entry for the Projection's type

    Returns:
        `True` if validation has been achieved to same level (though possibly with warnings);
        `False` if validation could not be done;
        raises an exception if an incompatibility is detected.
    """


    if connectee_state:
        connectee_str =  " {} of".format(connectee_state.__name__)
    else:
        connectee_str =  ""


    # Used below
    def _validate_projection_type(projection_class):
        # Validate that Projection's type can connect with the class of connect_with_state
        if connect_with_state.__name__ in getattr(projection_class.sockets, projection_socket):
            if owner.verbosePref:
                warnings.warn("{0} specified to be connected with{1} {2} is compatible with the {3} of the "
                              "specified {4} ({5}), but the initialization of the {4} is not yet complete so "
                              "compatibility can't be fully confirmed".
                              format(State.__name__, connectee_str, owner.name,
                                     projection_socket, Projection.__name__, projection_spec))

    # If it is an actual Projection
    if isinstance(projection_spec, Projection):

        # It is in deferred_init status
        if projection_spec.init_status is InitStatus.DEFERRED_INITIALIZATION:

            # Try to get the State to which the Projection will be connected when fully initialized
            #     as positive confirmation that it is the correct type for state_type
            try:
                projection_socket_state = projection_spec.init_args[projection_socket]
                # Projection's socket has been assigned to a State
                if projection_socket_state:
                    # Validate that the State is same class as connect_with_state
                    if issubclass(projection_socket_state, connect_with_state):
                        return True
                else:
                    _validate_projection_type(projection_spec.__class__)
                    return True
            # State for projection's socket couldn't be determined
            except KeyError:
                # Us Projection's type for validation
                # At least validate that Projection's type can connect with the class of connect_with_state
                    _validate_projection_type(projection_spec.__class__)
                    return True

        # Projection has been instantiated
        else:
            # Compare the State to which the Projection's socket has been assigned with connect_with_state
            projection_socket_state = getattr(projection_spec, projection_socket)
            if projection_socket_state is connect_with_state:
                return True

        # None of the above worked, so must be incompatible
        raise ProjectionError("{} specified to be connected with{} {} "
                              "is not compatible with the {} of the specified {} ({})".
                              format(State.__name__, connectee_str, owner.name,
                                     projection_socket, Projection.__name__, projection_spec))

    # Projection class
    elif inspect.isclass(projection_spec):
        _validate_projection_type(projection_spec)
        return True

    # Projection specification dictionary
    elif isinstance(projection_spec, dict):
        # Try to validate using entry for projection_socket
        if projection_socket in projection_spec and projection_spec[projection_socket] is not None:
            if projection_spec[projection_socket] is connect_with_state:
                return True
            else:
                raise ProjectionError("{} specified to be connected with{} {} is not compatible "
                                      "with the {} in the specification dict for the {} ({})".
                                      format(State.__name__, connectee_str, owner.name,
                                             projection_socket,
                                             Projection.__name__,
                                             projection_spec[projection_socket]))
        # Try to validate using entry for Projection' type
        elif PROJECTION_TYPE in projection_spec and projection_spec[PROJECTION_TYPE] is not None:
            _validate_projection_type(projection_spec[PROJECTION_TYPE])

    # Projection spec is too abstract to validate here
    #    (e.g., value or a name that will be used in context to instantiate it)
    if owner.verbosePref:
        warnings.warn("Specification of {} ({}) for connection between {} and{} {} "
                      "cannot be fully validated.".format(Projection.__class__.__name__,
                                                          projection_spec,
                                                          connect_with_state.name,
                                                          connectee_str,
                                                          owner.name))
    return False


# IMPLEMENTATION NOTE: MOVE THIS TO ModulatorySignals WHEN THAT IS IMPLEMENTED
@tc.typecheck
def _validate_receiver(sender_mech:Mechanism,
                       projection:Projection,
                       expected_owner_type:type,
                       spec_type=None,
                       context=None):
    """Check that Projection is to expected_receiver_type and in the same System as the sender_mech (if specified)

    expected_owner_type must be a Mechanism or a Projection
    spec_type should be LEARNING_SIGNAL, CONTROL_SIGNAL or GATING_SIGNAL

    Note:  this is a "directed" validation;
           for undirected validation of a Projection, use _validate_projection_specification

    """
    spec_type = " in the {} arg ".format(spec_type) or ""

    if projection.init_status is InitStatus.DEFERRED_INITIALIZATION:
        # receiver = projection.init_args['receiver'].owner
        state = projection.init_args['receiver']
        receiver = state.owner
    else:
        # receiver = projection.receiver.owner
        state = projection.receiver
        receiver = state.owner

    if isinstance(receiver, Mechanism):
        receiver_mech = receiver
    elif isinstance(receiver, Projection):
        receiver_mech = receiver.receiver.owner
    else:
        raise ProjectionError("receiver of projection ({}) must be a {} or {}".
                              format(projection.name, MECHANISM, PROJECTION))

    if not isinstance(receiver, expected_owner_type):
        raise ProjectionError("A {} specified {}for {} ({}) projects to a component other than the {} of a {}".
                                    format(projection.__class__.__name__,
                                           spec_type,
                                           sender_mech.name,
                                           receiver,
                                           state.__class__.__name__,
                                           expected_owner_type.__name__))

    # Check if receiver_mech is in the same system as sender_mech;
    #    if either has not been assigned a system, return

    # Check whether mech is in the same system as sender_mech
    receiver_systems = set()
    # receiver_mech is a ControlMechanism (which has a system but no systems attribute)
    if hasattr(receiver_mech, 'system') and receiver_mech.system:
        receiver_systems.update({receiver_mech.system})
    # receiver_mech is a ProcessingMechanism (which has a systems but system attribute is usually None)
    elif hasattr(receiver_mech, 'systems') and receiver_mech.systems:
        receiver_systems.update(set(receiver_mech.systems))
    else:
        return

    sender_systems = set()
    # sender_mech is a ControlMechanism (which has a system but no systems attribute)
    if hasattr(sender_mech, 'system') and sender_mech.system:
        sender_systems.update({sender_mech.system})
    # sender_mech is a ProcessingMechanism (which has a systems but system attribute is usually None)
    elif hasattr(sender_mech, 'systems')and sender_mech.systems:
        sender_systems.update(set(sender_mech.systems))
    else:
        return

    #  Check that projection is to a (projection to a) mechanisms in the same system as sender_mech
    if not receiver_systems & sender_systems:
        raise ProjectionError("A {} specified {}for {} projects to a Component that is not in the same System".
                                    format(projection.__class__.__name__,
                                           spec_type,
                                           sender_mech.name))


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _add_projection_to(receiver, state, projection_spec, context=None):
    """Assign an "incoming" Projection to a receiver InputState or ParameterState of a Component object

    Verify that projection has not already been assigned to receiver;
        if it has, issue a warning and ignore the assignment request.

    Requirements:
       * receiver must be an appropriate Component object (currently, a Mechanism or a Projection);
       * state must be a specification of an InputState or ParameterState;
       * specification of InputState can be any of the following:
                - INPUT_STATE - assigns projection_spec to (primary) InputState;
                - InputState object;
                - index for Mechanism.input_states;
                - name of an existing InputState (i.e., key for Mechanism.input_states);
                - the keyword kwAddInputState or the name for an InputState to be added;
       * specification of ParameterState must be a ParameterState object
       * projection_spec can be any valid specification of a projection_spec
           (see `State._instantiate_projections_to_state`).

    Args:
        receiver (Mechanism or Projection)
        state (State subclass)
        projection_spec: (Projection, dict, or str)
        context

    """
    # IMPLEMENTATION NOTE:  ADD FULL SET OF ParameterState SPECIFICATIONS
    #                       CURRENTLY, ASSUMES projection_spec IS AN ALREADY INSTANTIATED PROJECTION

    from PsyNeuLink.Components.States.State import _instantiate_state
    from PsyNeuLink.Components.States.State import State_Base
    from PsyNeuLink.Components.States.InputState import InputState
    from PsyNeuLink.Components.States.ParameterState import ParameterState

    if not isinstance(state, (int, str, InputState, ParameterState)):
        raise ProjectionError("State specification(s) for {0} (as receivers of {1}) contain(s) one or more items"
                             " that is not a name, reference to an InputState or ParameterState object, "
                             " or an index (for input_states)".
                             format(receiver.name, projection_spec.name))

    # state is State object, so use that
    if isinstance(state, State_Base):
        state._instantiate_projections_to_state(projections=projection_spec, context=context)
        return

    # Generic INPUT_STATE is specified, so use (primary) InputState
    elif state is INPUT_STATE:
        receiver.input_state._instantiate_projections_to_state(projections=projection_spec, context=context)
        return

    # input_state is index into input_states OrderedDict, so get corresponding key and assign to input_state
    elif isinstance(state, int):
        try:
            key = receiver.input_states[state]
        except IndexError:
            raise ProjectionError("Attempt to assign projection_spec ({0}) to InputState {1} of {2} "
                                 "but it has only {3} input_states".
                                 format(projection_spec.name, state, receiver.name, len(receiver.input_states)))
        else:
            input_state = key

    # input_state is string (possibly key retrieved above)
    #    so try as key in input_states OrderedDict (i.e., as name of an InputState)
    if isinstance(state, str):
        try:
            receiver.input_state[state]._instantiate_projections_to_state(projections=projection_spec, context=context)
        except KeyError:
            pass
        else:
            if receiver.prefs.verbosePref:
                warnings.warn("Projection_spec {0} added to {1} of {2}".
                              format(projection_spec.name, state, receiver.name))
            # return

    # input_state is either the name for a new InputState or kwAddNewInputState
    if not state is kwAddInputState:
        if receiver.prefs.verbosePref:
            reassign = input("\nAdd new InputState named {0} to {1} (as receiver for {2})? (y/n):".
                             format(input_state, receiver.name, projection_spec.name))
            while reassign != 'y' and reassign != 'n':
                reassign = input("\nAdd {0} to {1}? (y/n):".format(input_state, receiver.name))
            if reassign == 'n':
                raise ProjectionError("Unable to assign projection {0} to receiver {1}".
                                      format(projection_spec.name, receiver.name))

    # validate that projection has not already been assigned to receiver
    if receiver.verbosePref or projection_spec.sender.owner.verbosePref:
        if projection_spec in receiver.all_afferents:
            warnings.warn("Request to assign {} as projection to {} was ignored; it was already assigned".
                          format(projection_spec.name, receiver.owner.name))

    input_state = _instantiate_state(owner=receiver,
                                    state_type=InputState,
                                    state_name=input_state,
                                    state_spec=projection_spec.value,
                                    reference_value=projection_spec.value,
                                    reference_value_name='Projection_spec value for new InputState',
                                    context=context)

    #  Update InputState and input_states
    if receiver.input_states:
        receiver.input_states[input_state.name] = input_state

    # No InputState(s) yet, so create them
    else:
        receiver.input_states = ContentAddressableList(component_type=State_Base,
                                                       list=[input_state],
                                                       name=receiver.name+'.input_states')

    input_state._instantiate_projections_to_state(projections=projection_spec, context=context)


# IMPLEMENTATION NOTE:  THIS SHOULD BE MOVED TO COMPOSITION ONCE THAT IS IMPLEMENTED
def _add_projection_from(sender, state, projection_spec, receiver, context=None):
    """Assign an "outgoing" Projection from an OutputState of a sender Mechanism

    projection_spec can be any valid specification of a projection_spec (see State._instantiate_projections_to_state)
    state must be a specification of an OutputState
    Specification of OutputState can be any of the following:
            - OUTPUT_STATE - assigns projection_spec to (primary) OutputState
            - OutputState object
            - index for Mechanism OutputStates OrderedDict
            - name of OutputState (i.e., key for Mechanism.OutputStates OrderedDict))
            - the keyword kwAddOutputState or the name for an OutputState to be added

    Args:
        sender (Mechanism):
        projection_spec: (Projection, dict, or str)
        state (OutputState, str, or value):
        context:
    """


    from PsyNeuLink.Components.States.State import _instantiate_state
    from PsyNeuLink.Components.States.State import State_Base
    from PsyNeuLink.Components.States.OutputState import OutputState

    # Validate that projection is not already assigned to sender; if so, warn and ignore

    if isinstance(projection_spec, Projection):
        projection = projection_spec
        if ((isinstance(sender, OutputState) and projection.sender is sender) or
                (isinstance(sender, Mechanism) and projection.sender is sender.output_state)):
            if sender.verbosePref:
                warnings.warn("Request to assign {} as sender of {}, but it has already been assigned".
                              format(sender.name, projection.name))
                return

    if not isinstance(state, (int, str, OutputState)):
        raise ProjectionError("State specification for {0} (as sender of {1}) must be the name, reference to "
                              "or index of an OutputState of {0} )".format(sender.name, projection_spec))

    # state is State object, so use that
    if isinstance(state, State_Base):
        state._instantiate_projection_from_state(projection_spec=projection_spec, receiver=receiver, context=context)
        return

    # Generic OUTPUT_STATE is specified, so use (primary) OutputState
    elif state is OUTPUT_STATE:
        sender.output_state._instantiate_projections_to_state(projections=projection_spec, context=context)
        return

    # input_state is index into OutputStates OrderedDict, so get corresponding key and assign to output_state
    elif isinstance(state, int):
        try:
            key = list(sender.output_states.keys)[state]
        except IndexError:
            raise ProjectionError("Attempt to assign projection_spec ({0}) to OutputState {1} of {2} "
                                 "but it has only {3} OutputStates".
                                 format(projection_spec.name, state, sender.name, len(sender.output_states)))
        else:
            output_state = key

    # output_state is string (possibly key retrieved above)
    #    so try as key in output_states ContentAddressableList (i.e., as name of an OutputState)
    if isinstance(state, str):
        try:
            sender.output_state[state]._instantiate_projections_to_state(projections=projection_spec, context=context)
        except KeyError:
            pass
        else:
            if sender.prefs.verbosePref:
                warnings.warn("Projection_spec {0} added to {1} of {2}".
                              format(projection_spec.name, state, sender.name))
            # return

    # output_state is either the name for a new OutputState or kwAddNewOutputState
    if not state is kwAddOutputState:
        if sender.prefs.verbosePref:
            reassign = input("\nAdd new OutputState named {0} to {1} (as sender for {2})? (y/n):".
                             format(output_state, sender.name, projection_spec.name))
            while reassign != 'y' and reassign != 'n':
                reassign = input("\nAdd {0} to {1}? (y/n):".format(output_state, sender.name))
            if reassign == 'n':
                raise ProjectionError("Unable to assign projection {0} to sender {1}".
                                      format(projection_spec.name, sender.name))

    output_state = _instantiate_state(owner=sender,
                                     state_type=OutputState,
                                     state_name=output_state,
                                     state_spec=projection_spec.value,
                                     reference_value=projection_spec.value,
                                     reference_value_name='Projection_spec value for new InputState',
context=context)
    #  Update output_state and output_states
    try:
        sender.output_states[output_state.name] = output_state
    # No OutputState(s) yet, so create them
    except AttributeError:
        from PsyNeuLink.Components.States.State import State_Base
        sender.output_states = ContentAddressableList(component_type=State_Base,
                                                      list=[output_state],
                                                      name=sender.name+'.output_states')

    output_state._instantiate_projections_to_state(projections=projection_spec, context=context)
