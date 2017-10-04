# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ParameterState ******************************************************

"""

Overview
--------

ParameterStates belong to either a `Mechanism <Mechanism>` or a `Projection` and are used to represent, and possibly
modify the, values of all of the `configurable parameters <ParameterState_Configurable_Parameters>` of the `Component
<Component>` or those of its `function <Component_Function>`. A ParameterState can receive one or more
`ControlProjections  <ControlProjection>` and/or `LearningProjections <LearningProjection>` that modify the value of
the parameter. The Projections received by a ParameterState are listed in its `mod_afferents
<ParameterState.mod_afferents>` attribute.  The ParameterState's `function <ParameterState.function>` combines the
values of those Projections, and uses the result to modify the value of the parameter, that is then used by the
Component or its `function <Component.function>` when it executes.


.. _ParameterState_Creation:

Creating a ParameterState
-------------------------

ParameterStates are created automatically when the `Mechanism <Mechanism>` or `Projection <Projection>` to which they
belong is created.  The `owner <ParameterState.owner>` of a ParameterState must be a `Mechanism or `MappingProjection`.
One ParameterState is created for each configurable parameter of its owner, as well as for each configurable parameter
of the owner's `function <Component.function>` (the `configurable parameters <ParameterState_Configurable_Parameters>`
of a Component are listed in its `user_params <Component.user_params>` and function_params <Component.function_params>`
dictionaries. Each ParameterState is created using the value specified for the corresponding parameter, as described
below.  The ParameterStates for the parameters of a Mechanism or Projection are listed in its
:keyword:`parameter_states` attribute.

COMMENT:
    FOR DEVELOPERS: The instantiation of ParameterStates for all of the `user_params` of a Component can be
                    suppressed if a *PARAMETER_STATES* entry is included and set to `NotImplemented` in the
                    paramClassDefaults dictionary of its class definition;  the instantiation of a ParameterState
                    for an individual parameter in user_params can be suppressed by including it in
                    ClassDefaults.exclude_from_parameter_states for the class (or one of its parent classes)
                    (see LearningProjection and EVCControlMechanism for examples, and `note
                    <ParameterStates_Suppression>` below for additional information about how
                    to suppress creation of a ParameterState for individual parameters.  This should be done
                    for any parameter than can take a value or a string that is a keyword as its specification
                    (i.e., of the arg for the parameter in the Component's constructor) but should not have a
                    ParameterState (e.g., input_state and output_state), as otherwise the
                    specification will be interpreted as a numeric parameter (in the case of a value) or
                    a parameter of the keyword's type, a ParameterState will be created, and then it's value,
                    rather than the parameter's actual value, will be returned when the parameter is accessed
                    using "dot notation" (this is because the getter for an attribute's property first checks
                    to see if there is a ParameterState for that attribute and, if so, returns the value of the
                    ParameterState).
COMMENT

.. _ParameterState_Specification:

Specifying Parameters
~~~~~~~~~~~~~~~~~~~~~

Parameters can be specified in one of several places:

    * In the **argument** of the constructor for the `Component <Component>` to which the parameter belongs
      (see `Component_Configurable_Attributes` for additional details).
    ..
    * In a *parameter specification dictionary* assigned to the **params** argument in the constructor for the
      Component to which the parameter belongs. The entry for each parameter must use the name of the parameter
      (or a corresponding keyword) as its key, and the parameter's specification as its value (see
      `examples <ParameterState_Specification_Examples>` below). Parameters for a Component's
      `function <Component.function>` can be specified in an entry with the key *FUNCTION_PARAMS*, and a value that
      is itself a parameter specification dictionary containing an entry for each of the function's parameters to be
      specified.  When a value is assigned to a parameter in a specification dictionary, it overrides any value
      assigned to the argument for the parameter in the Component's constructor.
    ..
    * By direct assignment to the Component's attribute for the parameter
      (see `below <ParameterState_Configurable_Parameters>`).
    ..
    * In the `assign_params <Component.assign_params>` method for the Component.
    ..
    * In the **runtime_params** argument of a call to component's `execute <Mechanism_Base.execute>` method.

.. _ParameterState_Value_Specification:

The specification of the initial value of a parameter can take any of the following forms:

    .. _ParameterState_Value_Assignment:

    * **Value** -- this must be a valid value for the parameter. It creates a default ParameterState,
      assigns the parameter's default value as the ParameterState's `value <ParameterState.value>`,
      and assigns the parameter's name as the name of the ParameterState.
    ..
    * **ParameterState reference** -- this must refer to an existing **ParameterState** object; its name must be the
      name of a parameter of the owner or of the owner's `function <Component.function>`, and its value must be a valid
      one for the parameter.

      .. note::
          This capability is provided for generality and potential
          future use, but its current use is not advised.
    ..
    .. _ParameterState_Modulatory_Specification:

    * **Modulatory specification** -- this can be an existing `ControlSignal` or `ControlProjection`,
      a `LearningSignal` or `LearningProjection`, a constructor or the class name for any of these, or the
      keywords *CONTROL*, *CONTROL_PROJECTION*, *LEARNING*, or *LEARNING_PROJECTION.  Any of these create a default
      ParameterState, assign the parameter's default value as the ParameterState's `value <ParameterState.value>`,
      and assign the parameter's name as the name of the ParameterState.  They also create and/or assign the
      corresponding ModulatorySignal and ModulatoryProjection, and assign the ParameterState as the
      ModulatoryProjection's `receiver <Projection.Projection.receiver>`. If the ModulatorySignal and/or
      ModulatoryProjection already exist, their value(s) must be valid one(s) for the parameter.  Note that only
      Control and Learning Modulatory components can be assigned to a ParameterState (Gating components cannot be
      used -- they can only be assigned to `InputStates <InputState>` and `OutputStates <OutputState>`).
    ..
    .. _ParameterState_Tuple_Specification:

    * **Tuple** (value, Modulatory specification) -- this creates a default ParameterState, uses the value
      (1st) item of the tuple as parameter's `value assignment <ParameterState_Value_Assignment>`, and assigns the
      parameter's name as the name of the ParameterState.  The Modulatory (2nd) item of the tuple is used as the
      ParameterState's `modulatory assignment <ParameterState_Modulatory_Specification>`, and the ParameterState
      is assigned as the `receiver <Projection.Projection.receiver>` for the corresponding `ModulatoryProjection
      <ModulatoryProjection>`.

      .. note::
          Currently, the `function <Component.function>` of a Component, although it can be specified as a parameter
          value, cannot be assigned a `ModulatorySignal <ModulatorySignal>` or modified in the **runtime_params**
          argument of a call to a Mechanism's `execute <Mechanism_Base.execute>` method. This may change in the future.

The value specified for a parameter (either explicitly or by default) is assigned to an attribute of the Component or
of the Component's `function <Mechanism_Base.function>` to which the parameter belongs.  The attribute has the same
name as the parameter, and can be referenced using standard Python attribute ("dot") notation;  for example, the value
of a parameter named *param* is assigned to an attribute named ``param`` that can be referenced as
``my_component.param``). The parameter's value is assigned as the **default value** for the ParameterState.

.. _ParameterStates_Suppression:

.. note::
   If the value of a parameter is specified as `NotImplemented`, or any non-numeric value that is not one of those
   listed above, then no ParameterState is created and the parameter cannot be modified by a `ModulatorySignal
   <ModulatorySignal>` or in the **runtime_params** argument of a call to a Mechanism's `execute
   <Mechanism_Base.execute>` method.

.. _ParameterState_Specification_Examples:

Examples
~~~~~~~~

In the following example, a Mechanism is created by specifying two of its parameters, as well as its
`function <Component.function>` and two of that function's parameters, each using a different specification format::

    my_mechanism = RecurrentTransferMechanism(size=5,
                                              noise=ControlSignal),
                                              function=Logistic(gain=(0.5, ControlSignal),
                                                                bias=(1.0, ControlSignal(
                                                                              modulation=ModulationParam.ADDITIVE))))

The first argument of the constructor for the Mechanism specifies its `size <Component.size>` parameter by
directly assigning a value to it.  The second specifies the `noise <RecurrentTransferMechanism.noise>` parameter
by assigning a default `ControlSignal`;  this will use the default value of the
`noise <RecurrentTransferMechanism.noise>` attribute.  The **function** argument is specified using the constructor for
a `Logistic` function, that specifies two of its parameters.  The `gain <Logistic.gain>` parameter
is specified using a tuple, the first item of which is the value to be assigned, and the second specifies
a default `ControlSignal`.  The `bias <Logistic.bias>` parameter is also specified using a tuple,
in this case with a constructor for the ControlSignal that specifies its `modulation <ControlSignal.modulation>`
parameter.

In the following example, a `MappingProjection` is created, and its
`matrix <MappingProjection.MappingProjection.matrix>` parameter is assigned a random weight matrix (using a
`matrix keyword <Matrix_Keywords>`) and `LearningSignal`::

    my_mapping_projection = MappingProjection(sender=my_input_mechanism,
                                              receiver=my_output_mechanism,
                                              matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningSignal))

.. note::
   The `matrix <MappingProjection.MappingProjection.matrix>` parameter belongs to the MappingProjection's
   `function <MappingProjection.MappingProjection.function>`;  however, since it has only one standard function,
   its arguments are available in the constructor for the Projection (see
   `Component_Specifying_Functions_and_Parameters` for a more detailed explanation).

The example below shows how to specify the parameters in the first example using a parameter specification dictionary::

    my_mechanism = RecurrentTransferMechanism(
                              size=5
                              params={NOISE:5,
                                      'size':ControlSignal,
                                      FUNCTION:Logistic,
                                      FUNCTION_PARAMS:{GAIN:(0.5, ControlSignal),
                                                       BIAS:(1.0, ControlSignal(modulation=ModulationParam.ADDITIVE))))

There are several things to note here.  First, the parameter specification dictionary must be assigned to the
**params** argument of the constructor.  Second, both methods for specifying a parameter -- directly in an argument
for the parameter, or in an entry of a parameter specification dictionary -- can be used within the same constructor.
If a particular parameter is specified in both ways (as is the case for **size** in the example), the value in the
parameter specification dictionary takes priority (i.e., it is the value that will be assigned to the parameter).  If
the parameter is specified in a parameter specification dictionary, the key for the parameter must be a string that is
the same as the name of parameter (i.e., identical to how it appears as an arg in the constructor; as is shown
for **size** in the example), or using a keyword that resolves to such a string (as shown for *NOISE* in the
example).  Finally, the keyword *FUNCTION_PARAMS* can be used in a parameter specification dictionary to specify
parameters of the Component's `function <Component.function>`, as shown for the **gain** and **bias** parameters of
the Logistic function in the example.

.. _ParameterState_Structure:

Structure
---------

Every ParameterState is owned by a `Mechanism <Mechanism>` or `MappingProjection`. It can receive one or more
`ControlProjections <ControlProjection>` or `LearningProjections <LearningProjection>`, that are listed in its
`mod_afferents <ParameterState.mod_afferents>` attribute.  A ParameterState cannot receive
`PathwayProjections <PathwayProjection>` or `GatingProjections <GatingProjection>`.  When the ParameterState is
updated (i.e., its owner is executed), it uses the values of its ControlProjections and LearningProjections to
determine whether and how to modify its parameter's attribute value, which is then assigned as the ParameterState's
`value <ParameterState.value>` (see `ParameterState_Execution` for addition details). ParameterStates have the
following core attributes:

* `variable <ParameterState.variable>` - the parameter's attribute value; that is, the value assigned to the
  attribute for the parameter of the ParameterState's owner;  it can be thought of as the parameter's "base" value.
  It is used by its `function <ParameterState.function>` to determine the ParameterState's
  `value <ParameterState.value>`.  It must match the format (the number and type of elements) of the parameter's
  attribute value.

* `mod_afferents <ParameterState.mod_afferents>` - lists the `ModulatoryProjections <ModulationProjection>` received
  by the ParameterState.  These specify either modify the ParameterState's `function <ParameterState.function>`, or
  directly assign the `value <ParameterState.value>` of the ParameterState itself (see `ModulatorySignals_Modulation`).

* `function <ParameterState.function>` - takes the parameter's attribute value as its input, modifies it under the
  influence of any `ModulatoryProjections` it receives (listed in `mod_afferents <ParameterState.mod_afferents>`,
  and assigns the result as the ParameterState's `value <ParameterState.value>` which is used as the parameter's
  "actual" value.

* `value <ParameterState.value>` - the result of `function <ParameterState.function>`; used by the ParameterState's
  owner as the value of the parameter for which the the ParameterState is responsible.

.. _ParameterState_Configurable_Parameters:

All of the configurable parameters of a Component -- that is, for which it has ParameterStates -- are listed in its
`user_params <Component.user_params>` attribute, which is a read-only dictionary with an entry for each parameter.
The parameters for a Component's `function <Component.function>` are listed both in a *FUNCTION_PARAMS* entry of the
`user_params <Component.user_params>` dictionary, and in their own `function_params <Component.function_params>`
attribute, which is also a read-only dictionary (with an entry for each of its function's parameters).  The
ParameterStates for a Mechanism or Projection are listed in its :keyword:`parameter_states` attribute, which is also
read-only.

An initial value can be assigned to a parameter in the corresponding argument of the constructor for the Component
(see `above <ParameterState_Value_Specification>`.  Parameter values can also be modified by a assigning a value to
the corresponding attribute, or in groups using the Component's `assign_params <Component.assign_params>` method.
The parameters of a Component's function can be modified by assigning a value to the corresponding attribute of the
Component's `function_object <Component.function_object>` attribute (e.g., ``myMechanism.function_object.my_parameter``)
or in *FUNCTION_PARAMS* dict in a call to the Component's `assign_params <Component.assign_params>` method.
See `Mechanism_ParameterStates` for additional information.


.. _ParameterState_Execution:

Execution
---------

A ParameterState cannot be executed directly.  It is executed when the Component to which it belongs is executed.
When this occurs, the ParameterState executes any `ModulatoryProjections` it receives, the values of which
modulate parameters of the ParameterState's `function <ParameterState.function>`.  The ParameterState then calls
its `function <ParameterState.function>` and the result is assigned as its `value <ParameterState.value>`.  The
ParameterState's `value <ParameterState.value>` is used as the value of the corresponding parameter by the Component,
or by its own `function <Component.function>`.

.. note::
   It is important to note the distinction between the `function <ParameterState.function>` of a ParameterState,
   and the `function <Component.function>` of the Component to which it belongs. The former is used to determine the
   value of a parameter used by the latter (see `figure <ModulatorySignal_Anatomy_Figure>`, and `State_Execution` for
   additional details).

.. _ParameterState_Class_Reference:

Class Reference
---------------

"""

import inspect

import numpy as np
import typecheck as tc

from PsyNeuLink.Components.Component import Component, function_type, method_type, parameter_keywords
from PsyNeuLink.Components.Functions.Function import Linear, get_param_value_for_keyword
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.ShellClasses import Mechanism, Projection
from PsyNeuLink.Components.States.State import StateError, State_Base, _instantiate_state, state_type_keywords
from PsyNeuLink.Globals.Keywords import CONTROL_PROJECTION, FUNCTION, FUNCTION_PARAMS, MECHANISM, PARAMETER_STATE, PARAMETER_STATES, PARAMETER_STATE_PARAMS, PATHWAY_PROJECTION, PROJECTION, PROJECTION_TYPE, VALUE
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import is_pref_set
from PsyNeuLink.Globals.Preferences.PreferenceSet import PreferenceLevel
from PsyNeuLink.Globals.Utilities import ContentAddressableList, ReadOnlyOrderedDict, is_numeric, is_value_spec, iscompatible

state_type_keywords = state_type_keywords.update({PARAMETER_STATE})


class ParameterStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ParameterState(State_Base):
    """
    ParameterState(                                              \
    owner,                                                       \
    reference_value=None                                         \
    function=LinearCombination(operation=PRODUCT),               \
    variable=None,                                               \
    size=None,                                                   \
    parameter_modulation_operation=Modulation.MULTIPLY,          \
    params=None,                                                 \
    name=None,                                                   \
    prefs=None)

    Subclass of `State <State>` that represents and possibly modifies the parameter of
    a `Mechanism <Mechanism>`, `Projection <Projection>`, or its `Function`.


    COMMENT:

        Description
        -----------
            The ParameterState class is a componentType in the State category of Function,
            Its FUNCTION executes the Projections that it receives and updates the ParameterState's value

        Class attributes
        ----------------
            + componentType (str) = kwMechanisParameterState
            + classPreferences
            + classPreferenceLevel (PreferenceLevel.Type)
            + paramClassDefaults (dict)
                + FUNCTION (Linear)
                + PROJECTION_TYPE (CONTROL_PROJECTION)

        Class methods
        -------------
            _instantiate_function: insures that function is ARITHMETIC) (default: Operation.PRODUCT)
            update_state: updates self.value from Projections, base_value and runtime in PARAMETER_STATE_PARAMS

        StateRegistry
        -------------
            All ParameterStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT

    Arguments
    ---------

    owner : Mechanism or MappingProjection
        the `Mechanism <Mechanism>` or `MappingProjection` to which to which the ParameterState belongs; it must be
        specified or determinable from the context in which the ParameterState is created. The owner of a ParameterState
        for the parameter of a `function <Component.function>` should be specified as the Mechanism or Projection to
        which the function belongs.

    reference_value : number, list or np.ndarray
        specifies the default value of the parameter for which the ParameterState is responsible.

    variable : number, list or np.ndarray
        specifies the parameter's initial value and attribute value — that is, the value of the attribute of the
        ParameterState's owner or its `function <Component.function>` assigned to the parameter.

    size : int, list or np.ndarray of ints
        specifies variable as array(s) of zeros if **variable** is not passed as an argument;
        if **variable** is specified, it takes precedence over the specification of **size**.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to convert the parameter's attribute value (same as the ParameterState's
        `variable <ParameterState.variable>`) to the ParameterState's `value <ParameterState.value>`.

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the ParameterState or its function, and/or a custom function and its parameters.  Values specified for
        parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default InputState-<index>
        a string used for the name of the InputState.
        If not is specified, a default is assigned by StateRegistry of the Mechanism to which the InputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the InputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : Mechanism or MappingProjection
        the `Mechanism <Mechanism>` or `MappingProjection` to which the ParameterState belongs.

    mod_afferents : Optional[List[Projection]]
        a list of the `ModulatoryProjection <ModulatoryProjection>` that project to the ParameterState (i.e.,
        for which it is a `receiver <Projection.Projection.receiver>`); these can be
        `ControlProjection(s) <ControlProjection>` and/or `LearningProjection(s) <LearningProjection>`,
        but not `GatingProjection <GatingProjection>`.  The `value <ModulatoryProjection.value>` of each
        must match the format (number and types of elements) of the ParameterState's
        `variable <ParameterState.variable>`.

    variable : number, list or np.ndarray
        the parameter's attribute value — that is, the value of the attribute of the
        ParameterState's owner or its `function <Component.function>` assigned to the parameter.

    function : Function : default Linear
        converts the parameter's attribute value (same as the ParameterState's `variable <ParameterState.variable>`)
        to the ParameterState's `value <ParameterState.value>`, under the influence of any
        `ModulatoryProjections <ModulatoryProjection>` received by the ParameterState (and listed in its
        `mod_afferents <ParameterState.mod_afferents>` attribute.  The result is assigned as the ParameterState's
        `value <ParameterState>`.

    value : number, List[number] or np.ndarray
        the result returned by the ParameterState's `function <ParameterState.function>`, and used by the
        ParameterState's owner or its `function <Component.function>` as the value of the parameter for which the
        ParmeterState is responsible.  Note that this is not necessarily the same as the parameter's attribute value
        (that is, the value of the owner's attribute for the parameter), since the ParameterState's
        `function <ParameterState.function>` may modify the latter under the influence of its
        `mod_afferents <ParameterState.mod_afferents>`.

    name : str : default <State subclass>-<index>
        the name of the InputState.
        Specified in the **name** argument of the constructor for the OutputState.  If not is specified, a default is
        assigned by the StateRegistry of the Mechanism to which the OutputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, states' names are "scoped" within a Mechanism, meaning that states with
            the same name are permitted in different Mechanisms.  However, they are *not* permitted in the same
            Mechanism: states within a Mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the InputState.
        Specified in the **prefs** argument of the constructor for the Projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentType = PARAMETER_STATE
    paramsType = PARAMETER_STATE_PARAMS

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TypeDefaultPreferences
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     kwPreferenceSetName: 'ParameterStateCustomClassPreferences',
    #     kp<pref>: <setting>...}


    paramClassDefaults = State_Base.paramClassDefaults.copy()
    paramClassDefaults.update({PROJECTION_TYPE: CONTROL_PROJECTION})
    #endregion

    tc.typecheck
    def __init__(self,
                 owner,
                 reference_value=None,
                 variable=None,
                 size=None,
                 function=Linear(),
                 projections=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # FIX: UPDATED TO INCLUDE LEARNING [CHANGE THIS TO INTEGRATOR FUNCTION??]
        # # Reassign default for MATRIX param of MappingProjection
        # if isinstance(owner, MappingProjection) and name is MATRIX:
        #     function = LinearCombination(operation=SUM)

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramInstanceDefaults
        # Note: pass name of Mechanism (to override assignment of componentName in super.__init__)
        super(ParameterState, self).__init__(owner,
                                             variable=variable,
                                             size=size,
                                             projections=projections,
                                             params=params,
                                             name=name,
                                             prefs=prefs,
                                             context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that ParameterState (as identified by its name) is for a valid parameter of the owner

        Parameter can be either owner's, or owner's function_object
        """

        # If the parameter is not in either the owner's user_params dict or its function_params dict, throw exception
        if not self.name in self.owner.user_params.keys() and not self.name in self.owner.function_params.keys():
            raise ParameterStateError("Name of requested ParameterState ({}) does not refer to a valid parameter "
                                      "of the component ({}) or it function ({})".
                                      format(self.name,
                                             # self.owner.function_object.__class__.__name__,
                                             self.owner.name,
                                             self.owner.function.componentName))

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

    def _instantiate_function(self, context=None):
        """Insure function is LinearCombination and that its output is compatible with param with which it is associated

        Notes:
        * Relevant param should have been provided as reference_value arg in the call to InputState__init__()
        * Insures that self.value has been assigned (by call to super()._validate_function)
        * This method is called only if the parameterValidationPref is True

        :param context:
        :return:
        """
        super()._instantiate_function(context=context)

        # # Insure that output of function (self.value) is compatible with relevant parameter's reference_value
        if not iscompatible(self.value, self.reference_value):
            raise ParameterStateError("Value ({0}) of the {1} ParameterState for the {2} Mechanism is not compatible "
                                      "the type of value expected for that parameter ({3})".
                                           format(self.value,
                                                  self.name,
                                                  self.owner.name,
                                                  self.reference_value))

    def _instantiate_projections(self, projections, context=None):
        """Instantiate Projections specified in PROJECTIONS entry of params arg of State's constructor

        Disallow any PathwayProjections
        Call _instantiate_projections_to_state to assign ModulatoryProjections to .mod_afferents

        """

        # MODIFIED 7/8/17
        # FIX:  THIS SHOULD ALSO LOOK FOR OTHER FORMS OF SPECIFICATION
        # FIX:  OF A PathwayProjection (E.G., TARGET STATE OR MECHANISM)
        from PsyNeuLink.Components.Projections.PathwayProjections.PathwayProjection import PathwayProjection_Base
        pathway_projections = [proj for proj in projections if isinstance(proj, PathwayProjection_Base)]
        if pathway_projections:
            pathway_proj_names = []
            for proj in pathway_projections:
                pathway_proj_names.append(proj.name + ' ')
            raise StateError("{} not allowed for {}: {}".
                             format(PathwayProjection_Base.__self__.__name__,
                                    self.__class__.__name__,
                                    pathway_proj_names))

        self._instantiate_projections_to_state(projections=projections, context=context)

    def _execute(self, function_params, context):
        """Call self.function with current parameter value as the variable

        Get backingfield ("base") value of param of function of Mechanism to which the ParameterState belongs.
        Update its value in call to state's function.
        """

        # Most commonly, ParameterState is for the parameter of a function
        try:
            param_value = getattr(self.owner.function_object, '_'+ self.name)
            # param_value = self.owner.function_object.params[self.name]

       # Otherwise, should be for an attribute of the ParameterState's owner:
        except AttributeError:
            # param_value = self.owner.params[self.name]
            param_value = getattr(self.owner, '_'+ self.name)

        value = self.function(variable=param_value,
                              params=function_params,
                              context=context)

        # TEST PRINT
        # TEST DEBUG MULTILAYER
        # if MATRIX == self.name:
        #     print("\n{}\n@@@ WEIGHT CHANGES FOR {} TRIAL {}:\n{}".
        #           format(self.__class__.__name__.upper(), self.owner.name, CentralClock.trial, value))

        return value

# FIX: TUPLE MIGHT SPECIFY CONTROL SIGNAL? (I.E., WHEN A PARAMETER IS SPECIFIED AS A TUPLE)??
# FIX: CHECK THIS BY EXECUTING IN DEVEL AND BREAKING INSIDE TUPLE BRANCH OF _parse_state_spec

# MODIFIED 9/30/17 NEW:
    @tc.typecheck
    def _parse_state_specific_tuple(self, owner, state_specification_tuple):
        """Get connections specified in a ParameterState specification tuple

        Tuple specification can be:
            (state_spec, connections)

        Returns params dict with CONNECTIONS entries if any of these was specified.

        """
        from PsyNeuLink.Components.Projections.Projection import _parse_projection_specs
        from PsyNeuLink.Components.Projections.Projection import Projection
        from PsyNeuLink.Globals.Keywords import CONNECTIONS, PROJECTIONS

        params_dict = {}
        tuple_spec = state_specification_tuple

        # Note:  first item is assumed to be a specification for the InputState itself, handled in _parse_state_spec()

        # Get connection (afferent Projection(s)) specification from tuple
        CONNECTIONS_INDEX = len(tuple_spec)-1
        try:
            connections_spec = tuple_spec[CONNECTIONS_INDEX]
            # Recurisvely call _parse_state_specific_entries() to get OutputStates for afferent_source_spec
        except IndexError:
            connections_spec = None

        if connections_spec:
            try:
                # params_dict[CONNECTIONS] = _parse_projection_specs(self,
                params_dict[PROJECTIONS] = _parse_projection_specs(self,
                                                                   owner=owner,
                                                                   connections=connections_spec)
            except ParameterStateError:
                raise ParameterStateError("Item {} of tuple specification in {} specification dictionary "
                                      "for {} ({}) is not a recognized specification".
                                      format(CONNECTIONS_INDEX,
                                             ParameterState.__name__,
                                             owner.name,
                                             connections_spec))

        return params_dict
# MODIFIED 9/30/17 END

    @property
    def pathway_projections(self):
        raise ParameterStateError("PROGRAM ERROR: Attempt to access {} for {}; {}s do not have {}s".
                                  format(PATHWAY_PROJECTION, self.name, PARAMETER_STATE, PATHWAY_PROJECTION))

    @pathway_projections.setter
    def pathway_projections(self, value):
        raise ParameterStateError("PROGRAM ERROR: Attempt to assign {} to {}; {}s cannot accept {}s".
                                  format(PATHWAY_PROJECTION, self.name, PARAMETER_STATE, PATHWAY_PROJECTION))


def _instantiate_parameter_states(owner, context=None):
    """Call _instantiate_parameter_state for all params in user_params to instantiate ParameterStates for them

    If owner.params[PARAMETER_STATE] is None or False:
        - no ParameterStates will be instantiated.
    Otherwise, instantiate ParameterState for each allowable param in owner.user_params

    """

    # TBI / IMPLEMENT: use specs to implement ParameterStates below

    owner._parameter_states = ContentAddressableList(ParameterState, name=owner.name+'.parameter_states')

    # Check that all ParameterStates for owner have not been explicitly suppressed
    #    (by assigning `NotImplemented` to PARAMETER_STATES entry of paramClassDefaults)
    try:
        if owner.params[PARAMETER_STATES] is NotImplemented:
            return
    except KeyError:
        # PARAMETER_STATES not specified at all, so OK to continue and construct them
        pass

    try:
        owner.user_params
    except AttributeError:
        return
    # Instantiate ParameterState for each param in user_params (including all params in function_params dict),
    #     using its value as the state_spec
    # Exclude input_states and output_states which are also in user_params
    # IMPLEMENTATION NOTE:  Use user_params_for_instantiation since user_params may have been overwritten
    #                       when defaults were assigned to paramsCurrent in Component.__init__,
    #                       (since that will assign values to the properties of each param;
    #                       and that, in turn, will overwrite their current values with the defaults from paramsCurrent)
    for param_name, param_value in owner.user_params_for_instantiation.items():
        # Skip any parameter that has been specifically excluded by
        if param_name in owner.ClassDefaults.exclude_from_parameter_states:
            continue
        _instantiate_parameter_state(owner, param_name, param_value, context=context)


def _instantiate_parameter_state(owner, param_name, param_value, context):
    """Call _instantiate_state for allowable params, to instantiate a ParameterState for it

    Include ones in owner.user_params[FUNCTION_PARAMS] (nested iteration through that dict)
    Exclude if it is a:
        ParameterState that already exists (e.g., in case of a call from Component.assign_params)
        non-numeric value (including NotImplemented, False or True)
            unless it is:
                a tuple (could be on specifying ControlProjection, LearningProjection or Modulation)
                a dict with the name FUNCTION_PARAMS (otherwise exclude)
        function or method
            IMPLEMENTATION NOTE: FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED
            (this is because paramInstanceDefaults[FUNCTION] could be a class rather than an bound method;
            i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)
    # FIX: UPDATE WITH MODULATION_MODS
    # FIX:    CHANGE TO Integrator FUnction ONCE LearningProjection MODULATES ParameterState Function:
    If param_name is FUNCTION_PARAMS and param is a matrix (presumably for a MappingProjection)
        modify ParameterState's function to be LinearCombination (rather Linear which is the default)
    """


    # EXCLUSIONS:

    # # Skip if ParameterState already exists (e.g., in case of call from Component.assign_params)
    # if param_name in owner.ParameterStates:
    #     return

    if param_value is NotImplemented:
        return
    # Allow numerics but omit booleans (which are treated by is_numeric as numerical)
    if is_numeric(param_value) and not isinstance(param_value, bool):
        pass
    # Only allow a FUNCTION_PARAMS dict
    elif isinstance(param_value, ReadOnlyOrderedDict) and param_name is FUNCTION_PARAMS:
        pass
    # FIX: UPDATE WITH MODULATION_MODS
    # WHAT ABOUT GatingProjection??
    # Allow ControlProjection, LearningProjection
    elif isinstance(param_value, Projection):
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
        if isinstance(param_value, (ControlProjection, LearningProjection)):
            pass
        else:
            return
    # Allow Projection class
    elif inspect.isclass(param_value) and issubclass(param_value, Projection):
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
        if issubclass(param_value, (ControlProjection, LearningProjection)):
            pass
        else:
            return
    # Allow tuples (could be spec that includes a Projection or Modulation)
    elif isinstance(param_value, tuple):
        # # MODIFIED 4/18/17 NEW:
        # # FIX: EXTRACT VALUE HERE (AS IN Component.__init__?? [4/18/17]
        # param_value = owner._get_param_value_from_tuple(param_value)
        # # MODIFIED 4/18/17 END
        pass
    # Allow if it is a keyword for a parameter
    elif isinstance(param_value, str) and param_value in parameter_keywords:
        pass
    # Exclude function (see docstring above)
    elif param_name is FUNCTION:
        return
    # (7/19/17 CW) added this if statement below while adding `hetero` and `auto` and AutoAssociativeProjections: this
    # allows `hetero` to be specified as a matrix, while still generating a ParameterState
    elif isinstance(param_value, np.ndarray) or isinstance(param_value, np.matrix):
        pass
    # Exclude all others
    else:
        return

    # Assign parameterStates to component for parameters of its function (function_params), except for ones that are:
    #    - another component
    #    - a function or method
    #    - have a value of None (see IMPLEMENTATION_NOTE below)
    #    - they have the same name as another parameter of the component (raise exception for this)
    if param_name is FUNCTION_PARAMS:
        for function_param_name in param_value.keys():
            function_param_value = param_value[function_param_name]

            # IMPLEMENTATION NOTE:
            # The following is necessary since, if ANY parameters of a function are specified, entries are made
            #    in the FUNCTION_PARAMS dict of its owner for ALL of the function's params;  however, their values
            #    will be set to None (and there may not be any relevant paramClassDefaults or a way to determine a
            #    default; e.g., the length of the array for the weights or exponents params for LinearCombination).
            #    Therefore, None will be passed as the reference_value, which will cause validation of the
            #    ParameterState's function (in _instantiate_function()) to fail.
            #  Current solution is to simply not instantiate a ParameterState for any function_param that has
            #    not been explicitly specified
            if function_param_value is None:
                continue

            if not _is_legal_param_value(owner, function_param_value):
                continue

            # Raise exception if the function parameter's name is the same as one that already exists for its owner
            if function_param_name in owner.user_params:
                if inspect.isclass(owner.function):
                    function_name = owner.function.__name__
                else:
                    function_name= owner.function.name
                raise ParameterStateError("PROGRAM ERROR: the function ({}) of a component ({}) has a parameter ({}) "
                                          "with the same name as a parameter of the component itself".
                                          format(function_name, owner.name, function_param_name))

            # Use function_param_value as constraint
            # IMPLEMENTATION NOTE:  need to copy, since _instantiate_state() calls _parse_state_value()
            #                       for constraints before state_spec, which moves items to subdictionaries,
            #                       which would make them inaccessible to the subsequent parse of state_spec
            from copy import deepcopy
            reference_value = deepcopy(function_param_value)

            # Assign parameterState for function_param to the component
            state = _instantiate_state(owner=owner,
                                      state_type=ParameterState,
                                      name=function_param_name,
                                      # state_spec=function_param_value,
                                      reference_value=reference_value,
                                      reference_value_name=function_param_name,
                                      params=None,
                                      context=context)
            if state:
                owner._parameter_states[function_param_name] = state

    elif _is_legal_param_value(owner, param_value):
        state = _instantiate_state(owner=owner,
                                  state_type=ParameterState,
                                  name=param_name,
                                  reference_value=param_value,
                                  reference_value_name=param_name,
                                  params=None,
                                  context=context)
        if state:
            owner._parameter_states[param_name] = state


def _is_legal_param_value(owner, value):

    # LEGAL PARAMETER VALUES:

    # # lists, arrays or numeric values
    if is_value_spec(value):
        return True

    # tuple, first item of which is a legal parameter value
    #     note: this excludes (param_name, Mechanism) tuples used to specify a ParameterState
    #           (e.g., if specified for the control_signals param of ControlMechanism)
    if isinstance(value, tuple):
        if _is_legal_param_value(owner, value[0]):
            return True

    if isinstance(value, dict) and VALUE in value:
        return True

    # keyword that resolves to one of the above
    if get_param_value_for_keyword(owner, value) is not None:
        return True

    # Assignment of ParameterState for Component objects, function or method are not currently supported
    if isinstance(value, (function_type, method_type, Component)):
        return False


def _get_parameter_state(sender_owner, sender_type, param_name, component):
    """Return ParameterState for named parameter of a Mechanism requested by owner
    """

    # Validate that component is a Mechanism or Projection
    if not isinstance(component, (Mechanism, Projection)):
        raise ParameterStateError("Request for {} of a component ({}) that is not a {} or {}".
                                  format(PARAMETER_STATE, component, MECHANISM, PROJECTION))

    try:
        return component._parameter_states[param_name]
    except KeyError:
        # Check that param (named by str) is an attribute of the Mechanism
        if not (hasattr(component, param_name) or hasattr(component.function_object, param_name)):
            raise ParameterStateError("{} (in specification of {}  {}) is not an attribute "
                                        "of {} or its function"
                                        .format(param_name, sender_type, sender_owner.name, component))
        # Check that the Mechanism has a ParameterState for the param
        if not param_name in component._parameter_states.names:
            raise ParameterStateError("There is no ParameterState for the parameter ({}) of {} "
                                        "specified in {} for {}".
                                        format(param_name, component.name, sender_type, sender_owner.name))

