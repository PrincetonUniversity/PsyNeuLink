# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


#  *********************************************  State ********************************************************

"""

Overview
--------

A State provides an interface to one or more `projections <Projection>`, and receives the `value(s) <Projection>`
provide by them.  The value of a state can be modulated by a `ModulatoryProjection`. There are three primary types of
states, all of which are used by `Mechanisms <Mechanism>`, one of which is used by
`MappingProjections <MappingProjection>`, and all of which are subject to modulation by
`ModulatorySignals <ModulatorySignal>`, as summarized below:

* `InputState`:
    used by a mechanism to receive input from `MappingProjections <MappingProjection>`;
    its value can be modulated by a `GatingSignal`.

* `ParameterState`:
    * used by a mechanism to represent the value of one of its parameters, or a parameter of its
      `function <Mechanism.function>`, that can be modulated by a `ControlSignal`;
    * used by a `MappingProjection` to represent the value of its `matrix <MappingProjection.MappingProjection.matrix>`
      parameter, that can be modulated by a `LearningSignal`.

* `OutputState`:
    used by a mechanism to send its value to any efferent projections.  For
    `ProcessingMechanisms <ProcessingMechanism>` these are `PathwayProjections <PathwayProjection>`, most commonly
    `MappingProjection <MappingProjection>`.  For `ModulatoryMechanisms <ModulatoryMechanism>`, these
    `ModulatoryProjectios <ModulatoryProjection>` as described below. The `value <OutputState.value> of an
    OutputState can be modulated by a `GatingSignal`.

* `ModulatorySignal`:
    used by an `AdaptiveMechanism` to modulate the value of the primary types of states listed above.
    There are three types of ModulatorySignals:
    * `LearningSignal`, used by a `LearningMechanism` to modulate the *MATRIX* ParameterState of a `MappingProjection`;
    * `ControlSignal`, used by a `ControlMechanism` to modulate the `ParameterState` of a `Mechanism`;
    * `GatingSignal`, used by a `GatingMechanism` to modulate the `InputState` or `OutputState` of a `Mechanism`.
    Modulation is discussed further `below <State_Modulation>`, and described in detail under
    `ModulatorySignals <ModulatorySignal_Modulation>`.

.. _State_Creation:

Creating a State
----------------

States can be created using the constructor for one of the subclasses.  However, in general, they are created
automatically by the objects to which they belong, or by specifying the state in the constructor for the object
to which it belongs.  For example, `InputStates <InputState>` and `OutputStates <OutputStates>` can be specified,
in the **input_states** and **output_states** arguments, respectively, of the constructor for a `Mechanism`;
and a `ParameterState` can be specified in the argument of the constructor for a function of a Mechanism or Projection,
where its parameters are specified.  A state can be specified in those cases in any of the following forms:

    * an existing **State** object;
    ..
    * the name of a **state subclass** (`InputState`, `ParmeterState`, or `OutputState` - a default state of the
      corresponding type will be created, using a default value for the state that is determined by the context
      in which it is specified.
    ..
    * a **value**.  This creates a default State using the specified value as its default `value <State.value>`.
    ..
    * a **state specification dictionary**; every state specification can contain the following *KEY*:<value>
      entries, in addition to those specific to a particular state subtype (see subtype documentation):
      ..
      * *NAME*:<str> - the string is used as the name of the state;
      ..
      * *VALUE*:<value> - the value is used as the default value of the state;
      COMMENT:
          ..
          * *PROJECTIONS*:<List> - the list must contain specifications for one or more
            `projections <Projection_In_Context_Specification> to or from the state,
            depending the type of state and the context in which it is specified;
      COMMENT
      ..
      * *str*:<List> - the key is used as the name of the state, and the list must contain specifications for
        one or more `projections <Projection_In_Context_Specification>` to or from the state,
        depending on the type of state and the context in which it is specified;
        ..
    * a **2-item tuple** - the first item must be a value, used as the default value for the state,
      and the second item must be a specification for a `projection <Projection_In_Context_Specification>`
      to or from the state, depending on the type of state and the context in which it is specified;

COMMENT:
*** EXAMPLES HERE
COMMENT

.. _State_Structure:

Structure
---------

Every State is owned by either a `Mechanism <Mechanism>` or a `Projection <Projection>`. Like all PsyNeuLink
components, a State has the three following core attributes:

    * `variable <State.variable>`:  for an `InputState` and `ParameterState`,
      the value of this is determined by the  value(s) of the projection(s) that it receives (and that are listed in
      its `path_afferents <State.path_afferents>` attribute).  For an `OutputState`, it is the item of the owner
      mechanism's `value <Mechanism.value>` to which the OutputState is assigned (specified by the OutputState's
      `index <OutputState_Index>` attribute.
    ..
    * `function <State.function>`:  for an `InputState` this aggregates the values of the projections that the state
      receives (the default is `LinearCombination` that sums the values), under the potential influence of a
      `GatingSignal`;  for a `ParameterState`, it determines the value of the associated parameter, under the
      potential influence of a `ControlSignal` (for a `Mechanism`) or a `LearningSignal` (for a `MappingProjection`);
      for an OutputState, it conveys the result  of the Mechanism's function to its
      `output_values <Mechanism.output_values> attribute, under the potential influence of a `GatingSignal`.
      See  `ModulatorySignals <ModulatorySignal_Structure>` and the `AdaptiveMechanism <AdaptiveMechanism>` associated
      with each type for a description of how they can be used to modulate the `function <State.function> of a State.
    ..
    * `value <State.value>`:  for an `InputState` this is the aggregated value of the `PathWayProjections` it
      receives;  for a `ParameterState`, this determines the value of the associated parameter;
      for an `OutputState`, it is the item of the  owner Mechanism's `value <Mechanisms.value>` to which the
      OutputState is assigned, possibly modified by its `calculate <OutputState_Calculate>` attribute and/or a
      `GatingSignal`, and used as the `value <Projection.value>` of any the projections listed in its
      `efferents <OutputState.path_efferents>` attribute.

.. _State_Modulation:

Modulation
~~~~~~~~~~

Every type of State has a `mod_afferents <State.mod_afferents>` attribute, that lists the
`ModulatoryProjections <ModulatoryProjection>` it receives.  Each ModulatoryProjection comes from a `ModulatorySignal`
that specifies how it should modulate the State's `value <State.value>` when the State is
`updated <State_Execution> (see `ModulatorySignal_Modulation`).  In most cases a ModulatorySignal uses the State's
`function <State.function>` to modulate its `value <State.value>`.  The function of every State assigns one of its
parameters as its *MULTIPLICATIVE_PARAM* and another as its *MULTIPLICATIVE_PARAM*. The
`modulation <ModulatorySigal.modulation>` attribute of a ModulatorySignal determines which of these to modify
when the State uses it `fucntion <State.function>` to calculate its `value  <State.value>`.  However, the
ModulatorySignal can also be configured to override the State's `value <State.value>` (i.e., assign it directly),
or to disable modulation, using one of the values of 'ModulationParm` for its `modulation <ModulatorySignal.modulation>`
attribute (see `ModulatorySignal_Modulation` and `ModulatorySignal_Examples`).

When a State is `updated <State_Execution>`, it executes all of the ModulatoryProjections it receives.  Different
ModulatorySignals may call for different forms of modulation.  Accordingly, it seprately sums the values specified
by any ModulatorySignals for the *MULTIPLICATIVE_PARAM* of its `function <State.function>`, and similarly for the
*ADDITIVE_PARAM*.  It then applies the summed value for each to the corresponding parameter of its
`function <State.function>`.  If any of the ModulatorySignals specifies *OVERRIDE*, then the value of that
ModulatorySignal is used as the State's `value <State.value>`.

.. note::
   'OVERRIDE <ModulatorySignal_Modulation>' can be specified for **only one** ModulatoryProjection to a State;
   specifying it for more than one causes an error.

.. _State_Execution:

Execution
---------

States cannot be executed.  They are updated when the component to which they belong is executed.  InputStates and
ParameterStates belonging to a Mechanism are updated before the Mechanism's function is called.  OutputStates
are updated after the Mechanism's function is called.  When a State is updated, it executes any Projections that
project to it (listed in its `all_afferents <State.all_afferents>` attribute.  It uses the values it receives from any
`PathWayProjections` (listed in its `path_afferents` attribute) as the variable for its `function <State.function>`,
and the values it receives from any `ModulatoryProjections` (listed in its `mod_afferents` attribute) to determine
the parameters of its `function <State.function>` (as described `above <State_Modulation>`).  It then calls its
`function <State.function>` to determine its `value <State.value>`. This conforms to a "lazy evaluation" protocol
(see :ref:`Lazy Evaluation <LINK>` for a more detailed discussion).

.. _State_Class_Reference:

Class Reference
---------------

"""

import inspect
import copy
import collections
from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Functions.Function import _get_modulated_param
from PsyNeuLink.Components.Projections.Projection import projection_keywords, _is_projection_spec
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

state_keywords = component_keywords.copy()
state_keywords.update({STATE_VALUE,
                       STATE_PARAMS,
                       STATE_PROJECTIONS,
                       MODULATORY_PROJECTIONS,
                       PROJECTION_TYPE,
                       LEARNING_PROJECTION_PARAMS,
                       LEARNING_SIGNAL_SPECS,
                       CONTROL_PROJECTION_PARAMS,
                       CONTROL_SIGNAL_SPECS,
                       GATING_PROJECTION_PARAMS
                       })

def _is_state_type (spec):
    if issubclass(spec, State):
        return True
    return False

# Note:  This is created only for assignment of default projection types for each state subclass (see .__init__.py)
#        Individual stateRegistries (used for naming) are created for each mechanism
StateRegistry = {}

class StateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# State factory method:
# def state(name=NotImplemented, params=NotImplemented, context=None):
#         """Instantiates default or specified subclass of State
#
#        If called w/o arguments or 1st argument=NotImplemented, instantiates default subclass (ParameterState)
#         If called with a name string:
#             - if registered in owner mechanism's state_registry as name of a subclass, instantiates that class
#             - otherwise, uses it as the name for an instantiation of the default subclass, and instantiates that
#         If a params dictionary is included, it is passed to the subclass
#
#         :param name:
#         :param param_defaults:
#         :return:
#         """
#
#         # Call to instantiate a particular subclass, so look up in MechanismRegistry
#         if name in mechanism's _stateRegistry:
#             return _stateRegistry[name].mechanismSubclass(params)
#         # Name is not in MechanismRegistry or is not provided, so instantiate default subclass
#         else:
#             # from Components.Defaults import DefaultState
#             return DefaultState(name, params)



# DOCUMENT:  INSTANTATION CREATES AN ATTIRBUTE ON THE OWNER MECHANISM WITH THE STATE'S NAME + kwValueSuffix
#            THAT IS UPDATED BY THE STATE'S value setter METHOD (USED BY LOGGING OF MECHANISM ENTRIES)
class State_Base(State):
    """
    State_Base(        \
    owner,             \
    params=None,       \
    name=None,         \
    prefs=None)

    Abstract class for State.

    .. note::
       States should NEVER be instantiated by a direct call to the base class.
       They should be instantiated by calling the constructor for the desired subclass,
       or using other methods for specifying a state (see :ref:`State_Creation`).

    COMMENT:
        Description
        -----------
            Represents and updates the state of an input, output or parameter of a mechanism
                - receives inputs from projections (self.path_afferents, STATE_PROJECTIONS)
                - input_states and parameterStates: combines inputs from all projections (mapping, control or learning)
                    and uses this as variable of function to update the value attribute
                - outputStates: represent values of output of function
            Value attribute:
                 - is updated by the execute method (which calls state's function)
                 - can be used as sender (input) to one or more projections
                 - can be accessed by KVO
            Constraints:
                - value must be compatible with variable of function
                - value must be compatible with receiver.value for all projections it receives

            Subclasses:
                Must implement:
                    componentType
                    ParamClassDefaults with:
                        + FUNCTION (or <subclass>.function
                        + FUNCTION_PARAMS (optional)
                        + PROJECTION_TYPE - specifies type of projection to use for instantiation of default subclass
                Standard subclasses and constraints:
                    InputState - used as input state for Mechanism;  additional constraint:
                        - value must be compatible with variable of owner's function method
                    OutputState - used as output state for Mechanism;  additional constraint:
                        - value must be compatible with the output of the owner's function
                    MechanismsParameterState - used as state for Mechanism parameter;  additional constraint:
                        - output of function must be compatible with the parameter's value

        Class attributes
        ----------------
            + componentCategory = kwStateFunctionCategory
            + className = STATE
            + suffix
            + classPreference (PreferenceSet): StatePreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
            + variableClassDefault (value): [0]
            + requiredParamClassDefaultTypes = {FUNCTION_PARAMS : [dict],    # Subclass function params
                                               PROJECTION_TYPE: [str, Projection]})   # Default projection type
            + paramClassDefaults (dict): {STATE_PROJECTIONS: []}             # Projections to States
            + paramNames (dict)
            + owner (Mechansim)
            + FUNCTION (Function class or object, or method)

        Class methods
        -------------
            - set_value(value) -
                validates and assigns value, and updates observers
                returns None
            - update_state(context) -
                updates self.value by combining all projections and using them to compute new value
                return None

        StateRegistry
        -------------
            Used by .__init__.py to assign default projection types to each state subclass
            Note:
            * All states that belong to a given owner are registered in the owner's _stateRegistry,
                which maintains a dict for each state type that it uses, a count for all instances of that type,
                and a dictionary of those instances;  NONE of these are registered in the StateRegistry
                This is so that the same name can be used for instances of a state type by different owners
                    without adding index suffixes for that name across owners,
                    while still indexing multiple uses of the same base name within an owner

        Arguments
        ---------
        - value (value) - establishes type of value attribute and initializes it (default: [0])
        - owner(Mechanism) - assigns state to mechanism (default: NotImplemented)
        - params (dict):  (if absent, default state is implemented)
            + FUNCTION (method)         |  Implemented in subclasses; used in update()
            + FUNCTION_PARAMS (dict) |
            + STATE_PROJECTIONS:<projection specification or list of ones>
                if absent, no projections will be created
                projection specification can be: (see Projection for details)
                    + Projection object
                    + Projection class
                    + specification dict
                    + a list containing any or all of the above
                    if dict, must contain entries specifying a projection:
                        + PROJECTION_TYPE:<Projection class>: must be a subclass of Projection
                        + PROJECTION_PARAMS:<dict>? - must be dict of params for PROJECTION_TYPE
        - name (str): if it is not specified, a default based on the class is assigned in register_category,
                            of the form: className+n where n is the n'th instantiation of the class
        - prefs (PreferenceSet or specification dict):
             if it is omitted, a PreferenceSet will be constructed using the classPreferences for the subclass
             dict entries must have a preference keyPath as their key, and a PreferenceEntry or setting as their value
             (see Description under PreferenceSet for details)
        - context (str): must be a reference to a subclass, or an exception will be raised
    COMMENT

    Attributes
    ----------

    owner : Mechanism or Projection
        object to which the state belongs.

    base_value : number, list or np.ndarray
        value with which the state was initialized.
    
    all_afferents : Optional[List[Projection]]
        list of all Projections received by the State (i.e., for which it is a `receiver <Projection.receiver>`.

    path_afferents : Optional[List[Projection]]
        list all `PathwayProjections <PathwayProjection>` received by the State.
        (note:  only `InputStates <InputState> have path_efferents;  the list is empty for other types of States).

    mod_afferents : Optional[List[GatingProjection]]
        list of all `ModulatoryProjections <ModulatoryProjection>` received by the State.

    efferents : Optional[List[Projection]]
        list of outoging Projections from the State (i.e., for which is a `sender <Projection.sender>`
        (note:  only `OutputStates <OutputState> have efferents;  the list is empty for other types of States).

    function : TransferFunction : default determined by type
        used to determine the state's own value from the value of the projection(s) it receives;  the parameters that 
        the TrasnferFunction identifies as ADDITIVE and MULTIPLICATIVE are subject to modulation by a 
        `ModualtoryProjection <ModulatoryProjection_Structure>`. 

    value : number, list or np.ndarray
        current value of the state (updated by `update <State.update>` method).

    name : str : default <State subclass>-<index>
        the name of the state.
        Specified in the **name** argument of the constructor for the state;  if not is specified,
        a default is assigned by StateRegistry based on the states's subclass
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, states names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation).

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the state.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).

    """

    #region CLASS ATTRIBUTES

    componentCategory = kwStateComponentCategory
    className = STATE
    suffix = " " + className
    paramsType = None

    registry = StateRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY

    variableClassDefault = [0]

    requiredParamClassDefaultTypes = Component.requiredParamClassDefaultTypes.copy()
    requiredParamClassDefaultTypes.update({FUNCTION_PARAMS : [dict],
                                           PROJECTION_TYPE: [str, Projection]})   # Default projection type
    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({STATE_PROJECTIONS:[],
                               MODULATORY_PROJECTIONS:[]})
    paramNames = paramClassDefaults.keys()

    @tc.typecheck
    def __init__(self,
                 owner:tc.any(Mechanism, Projection),
                 variable=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kargs):
        """Initialize subclass that computes and represents the value of a particular state of a mechanism

        This is used by subclasses to implement the inputState(s), outputState(s), and parameterState(s) of a Mechanism.

        Arguments:
            - owner (Mechanism):
                 mechanism with which state is associated (default: NotImplemented)
                 this argument is required, as can't instantiate a State without an owning Mechanism
            - variable (value): value of the state:
                must be list or tuple of numbers, or a number (in which case it will be converted to a single-item list)
                must match input and output of state's update function, and any sending or receiving projections
            - params (dict):
                + if absent, implements default State determined by PROJECTION_TYPE param
                + if dict, can have the following entries:
                    + STATE_PROJECTIONS:<Projection object, Projection class, dict, or list of either or both>
                        if absent, no projections will be created
                        if dict, must contain entries specifying a projection:
                            + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                            + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            - name (str): string with name of state (default: name of owner + suffix + instanceIndex)
            - prefs (dict): dictionary containing system preferences (default: Prefs.DEFAULTS)
            - context (str)
            - **kargs (dict): dictionary of arguments using the following keywords for each of the above kargs:
                # + STATE_VALUE = value
                + VARIABLE = variable
                + STATE_PARAMS = params
                + kwStateName = name
                + kwStatePrefs = prefs
                + kwStateContext = context
                NOTES:
                    * these are used for dictionary specification of a State in param declarations
                    * they take precedence over arguments specified directly in the call to __init__()
        """
        if kargs:
            try:
                # # MODIFIED 5/10/17 OLD:
                # variable = kargs[STATE_VALUE]
                # MODIFIED 5/10/17 NEW:
                variable = kargs[VARIABLE]
                # MODIFIED 5/10/17 END
            except (KeyError, NameError):
                pass
            try:
                params = kargs[STATE_PARAMS]
            except (KeyError, NameError):
                pass
            try:
                name = kargs[kwStateName]
            except (KeyError, NameError):
                pass
            try:
                prefs = kargs[kwStatePrefs]
            except (KeyError, NameError):
                pass
            try:
                context = kargs[kwStateContext]
            except (KeyError, NameError):
                pass

        # Enforce that only called from subclass
        if not isinstance(context, State_Base):
            raise StateError("Direct call to abstract class State() is not allowed; "
                                      "use state() or one of the following subclasses: {0}".
                                      format(", ".join("{!s}".format(key) for (key) in StateRegistry.keys())))

        # Enforce that subclass must implement and _execute method
        if not hasattr(self, '_execute'):
            raise StateError("{}, as a subclass of {}, must implement an _execute() method".
                             format(self.__class__.__name__, STATE))

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(params=params)

        # # VALIDATE owner
        # if isinstance(owner, (Mechanism, Projection)):
        #     self.owner = owner
        # else:
        #     raise StateError("\'owner\' argument ({0}) for {1} must be a mechanism or projection".
        #                               format(owner, name))
        self.owner = owner

        # Register state with StateRegistry of owner (mechanism to which the state is being assigned)
        register_category(entry=self,
                          base_class=State_Base,
                          name=name,
                          registry=owner._stateRegistry,
                          # sub_group_attr='owner',
                          context=context)

        self.path_afferents = []
        self.mod_afferents = []
        self.efferents = []
        self._stateful = False

        self._path_proj_values = []
        # Create dict with entries for each ModualationParam and initialize - used in update() to coo
        self._mod_proj_values = {}
        for attrib, value in get_class_attributes(ModulationParam):
            self._mod_proj_values[getattr(ModulationParam,attrib)] = []

        # VALIDATE VARIABLE, PARAM_SPECS, AND INSTANTIATE self.function
        super(State_Base, self).__init__(variable_default=variable,
                                         param_defaults=params,
                                         name=name,
                                         prefs=prefs,
                                         context=context.__class__.__name__)

        # INSTANTIATE PROJECTION_SPECS SPECIFIED IN PARAM_SPECS
        try:
            projections = self.paramsCurrent[STATE_PROJECTIONS]
        except KeyError:
            # No projections specified, so none will be created here
            # IMPLEMENTATION NOTE:  This is where a default projection would be implemented
            #                       if params = NotImplemented or there is no param[STATE_PROJECTIONS]
            pass
        else:
            if projections:
                self._instantiate_projections_to_state(projections=projections, context=context)

    def _validate_variable(self, variable, context=None):
        """Validate variable and assign validated values to self.variable

        Sets self.base_value = self.value = self.variable = variable
        Insures that it is a number of list or tuple of numbers

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True
        """

        super(State,self)._validate_variable(variable, context)

        if not context:
            context = kwAssign + ' Base Value'
        else:
            context = context + kwAssign + ' Base Value'

        # # MODIFIED 6/1/17 OLD:
        # self.base_value = self.variable
        # MODIFIED 6/1/17 END

    def _validate_params(self, request_set, target_set=None, context=None):
        """validate projection specification(s)

        Call super (Component._validate_params()
        Validate following params:
            + STATE_PROJECTIONS:  <entry or list of entries>; each entry must be one of the following:
                + Projection object
                + Projection class
                + specification dict, with the following entries:
                    + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                    + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
            # IMPLEMENTATION NOTE: TBI - When learning projection is implemented
            # + FUNCTION_PARAMS:  <dict>, every entry of which must be one of the following:
            #     ParameterState, projection, 2-item tuple or value
        """

        if STATE_PROJECTIONS in target_set:
            # if projection specification is an object or class reference, needs to be wrapped in a list
            # - to be consistent with paramClassDefaults
            # - for consistency of treatment below
            projections = target_set[STATE_PROJECTIONS]
            if not isinstance(projections, list):
                projections = [projections]
        else:
            # If no projections, ignore (none will be created)
            projections = None

        super(State, self)._validate_params(request_set, target_set, context=context)

        if projections:
            # Validate projection specs in list
            from PsyNeuLink.Components.Projections import Projection
            for projection in projections:
                try:
                    issubclass(projection, Projection)
                except TypeError:
                    if (isinstance(projection, Projection) or iscompatible(projection. dict)):
                        continue
                    else:
                        if self.prefs.verbosePref:
                            print("{0} in {1} is not a projection, projection type, or specification dict; "
                                  "{2} will be used to create default {3} for {4}".
                                format(projection,
                                       self.__class__.__name__,
                                       target_set[PROJECTION_TYPE],
                                       self.owner.name))

    def _instantiate_function(self, context=None):
        """Insure that output of function (self.value) is compatible with its input (self.variable)

        This constraint reflects the role of State functions:
            they simply update the value of the State;
            accordingly, their variable and value must be compatible
        """

        var_is_matrix = False
        # If variable is a matrix (e.g., for the MATRIX parameterState of a MappingProjection),
        #     it needs to be embedded in a list so that it is properly handled by LinearCombination
        #     (i.e., solo matrix is returned intact, rather than treated as arrays to be combined);
        # Notes:
        #     * this is not a problem when LinearCombination is called in state.update(), since that puts
        #         projection values in a list before calling LinearCombination to combine them
        #     * it is removed from the list below, after calling _instantiate_function
        # FIX: UPDATE WITH MODULATION_MODS REMOVE THE FOLLOWING COMMENT:
        #     * no change is made to PARAMETER_MODULATION_FUNCTION here (matrices may be multiplied or added)
        #         (that is handled by the individual state subclasses (e.g., ADD is enforced for MATRIX parameterState)
        if ((inspect.isclass(self.function) and issubclass(self.function, LinearCombination) or
                 isinstance(self.function, LinearCombination)) and
                (isinstance(self.variable, np.matrix) or
                (isinstance(self.variable, np.ndarray) and self.variable.ndim >= 2))):
            self.variable = [self.variable]
            var_is_matrix = True

        super()._instantiate_function(context=context)

        # If it is a matrix, remove from list in which it was embedded after instantiating and evaluating function
        if var_is_matrix:
            self.variable = self.variable[0]

        # Insure that output of the function (self.value) is compatible with (same format as) its input (self.variable)
        #     (this enforces constraint that State functions should only combine values from multiple projections,
        #     but not transform them in any other way;  so the format of its value should be the same as its variable).
        if not iscompatible(self.variable, self.value):
            raise StateError("Output ({0}: {1}) of function ({2}) for {3} {4} of {5}"
                                      " must be the same format as its input ({6}: {7})".
                                      format(type(self.value).__name__,
                                             self.value,
                                             self.function.__self__.componentName,
                                             self.name,
                                             self.__class__.__name__,
                                             self.owner.name,
                                             self.variable.__class__.__name__,
                                             self.variable))

    def _instantiate_projections_to_state(self, projections, context=None):
        """Instantiate projections to a state and assign them to self.path_afferents

        For each spec in projections arg, check that it is one or a list of any of the following:
        + Projection class (or keyword string constant for one):
            implements default projection for projection class
        + Projection object:
            checks that receiver is self
            checks that projection function output is compatible with self.value
        + State object:
            check that it is compatible with (i.e., a legimate sender for) projection
            assign as sender of the projection
        + [TBI: State class: instantiate default State]
        + [TBI: Mechanism object]
        + specification dict (usually from STATE_PROJECTIONS entry of params dict):
            checks that projection function output is compatible with self.value
            implements projection
            dict must contain:
                + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
        If any of the conditions above fail:
            a default projection is instantiated using self.paramsCurrent[PROJECTION_TYPE]
        For each projection:
            if it is a MappingProjection, it is added to self.path_afferents
            if it is a LearningProjection, ControlProjection, or GatingProjection, it is added to self.mod_afferents
        If kwMStateProjections is absent or empty, no projections are created
        """

        from PsyNeuLink.Components.Projections.Projection import Projection_Base
        from PsyNeuLink.Components.Projections.PathwayProjections.PathwayProjection \
            import PathwayProjection_Base
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection \
            import ModulatoryProjection_Base
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

        # If specification is not a list, wrap it in one for consistency of treatment below
        # (since specification can be a list, so easier to treat any as a list)
        projection_list = projections
        if not isinstance(projection_list, list):
            projection_list = [projection_list]

        state_name_string = self.name
        item_prefix_string = ""
        item_suffix_string = state_name_string + " ({} for {})".format(self.__class__.__name__, self.owner.name,)
        default_string = ""
        kwDefault = "default "

        default_projection_type = self.paramClassDefaults[PROJECTION_TYPE]

        # Instantiate each projection specification in the projection_list, and
        # - insure it is in self.path_afferents
        # - insure the output of its function is compatible with self.value
        for projection_spec in projection_list:

            # If there is more than one projection specified, construct messages for use in case of failure
            if len(projection_list) > 1:
                item_prefix_string = "Item {0} of projection list for {1}: ".\
                    format(projection_list.index(projection_spec)+1, state_name_string)
                item_suffix_string = ""

# FIX: FROM HERE TO BOTTOM OF METHOD SHOULD ALL BE HANDLED IN __init__() FOR PROJECTION_SPEC
# FIX: OR A _parse_projection_spec METHOD
            projection_object = None # flags whether projection object has been instantiated; doesn't store object
            projection_type = None   # stores type of projection to instantiate
            sender = None
            projection_params = {}

            # PARSE AND INSTANTIATE PROJECTION_SPEC --------------------------------------------------------------------

            # If projection_spec is a Projection object:
            # - call _check_projection_receiver() to check that receiver is self; if not, it:
            #     returns object with receiver reassigned to self if chosen by user
            #     else, returns new (default) PROJECTION_TYPE object with self as receiver
            #     note: in that case, projection will be in self.path_afferents list
            if isinstance(projection_spec, Projection_Base):
                if projection_spec.value is DEFERRED_INITIALIZATION:
                    if isinstance(projection_spec, ModulatoryProjection_Base):
                        # Assign projection to mod_afferents
                        self.mod_afferents.append(projection_spec)
                        projection_spec.init_args[RECEIVER] = self
                        # Skip any further initialization for now
                        #   (remainder will occur as part of deferred init for
                        #    ControlProjection or GatingProjection)
                        continue

                    # Complete init for other (presumably Mapping) projections
                    else:
                        # Assume init was deferred because receiver could not be determined previously
                        #  (e.g., specified in function arg for receiver object, or as standalone projection in script)
                        # Assign receiver to init_args and call _deferred_init for projection
                        projection_spec.init_args[RECEIVER] = self
                        projection_spec.init_args['name'] = self.owner.name+' '+self.name+' '+projection_spec.className
                        # FIX: REINSTATE:
                        # projection_spec.init_args['context'] = context
                        projection_spec._deferred_init()  # XXX
                projection_object, default_class_name = self._check_projection_receiver(
                                                                                    projection_spec=projection_spec,
                                                                                    messages=[item_prefix_string,
                                                                                              item_suffix_string,
                                                                                              state_name_string],
                                                                                    context=self)
                # If projection's name has not been assigned, base it on State's name:
                if default_class_name:
                    # projection_object.name = projection_object.name.replace(default_class_name, self.name)
                    projection_object.name = self.name + '_' + projection_object.name
                    # Used for error message
                    default_string = kwDefault
# FIX:  REPLACE DEFAULT NAME (RETURNED AS DEFAULT) PROJECTION_SPEC NAME WITH State'S NAME, LEAVING INDEXED SUFFIX INTACT

            # MODIFIED 6/19/17 NEW:
            # If projection_spec is a State
            # - assume
            elif isinstance(projection_spec, State):
                # Check that State is appropriate for type of projection
                _check_projection_sender_compatiability(self, default_projection_type, type(projection_spec))
                # Assign State as projections's sender (for use below)
                sender = projection_spec
            # MODIFIED 6/19/17 END

            # FIX: IMPLEMENT FOLLOWING;
            #      ASSUME PRIMARY OutputState OF ProcessingMechanism FOR MappingProjection OR raise exception
            #      ASSUME FIRST ModulatorySignal OF AdaptiveMechanism FOR ModulatoryProjection OR raise exception
            # # MODIFIED 6/19/17 NEW:
            # elif isinstance(projection_spec, Mechanism):
            #     pass
            # # MODIFIED 6/19/17 END


            # If projection_spec is a dict:
            # - get projection_type
            # - get projection_params
            # Note: this gets projection_type but does NOT not instantiate projection; so,
            #       projection is NOT yet in self.path_afferents list
            elif isinstance(projection_spec, dict):
                # Get projection type from specification dict
                try:
                    projection_type = projection_spec[PROJECTION_TYPE]
                except KeyError:
                    projection_type = default_projection_type
                    default_string = kwDefault
                    if self.prefs.verbosePref:
                        warnings.warn("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                              format(item_prefix_string,
                                     PROJECTION_TYPE,
                                     STATE_PROJECTIONS,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))
                else:
                    # IMPLEMENTATION NOTE:  can add more informative reporting here about reason for failure
                    projection_type, error_str = self._parse_projection_ref(projection_spec=projection_type,
                                                                           context=self)
                    if error_str and self.prefs.verbosePref:
                        warnings.warn("{0}{1} {2}; default {4} will be assigned".
                              format(item_prefix_string,
                                     PROJECTION_TYPE,
                                     error_str,
                                     STATE_PROJECTIONS,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

                # Get projection params from specification dict
                try:
                    projection_params = projection_spec[PROJECTION_PARAMS]
                except KeyError:
                    if self.prefs.verbosePref:
                        warnings.warn("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                              format(item_prefix_string,
                                     PROJECTION_PARAMS,
                                     STATE_PROJECTIONS, state_name_string,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

            # Check if projection_spec is class ref or keyword string constant for one
            # Note: this gets projection_type but does NOT instantiate the projection (that happens below),
            #       so projection is NOT yet in self.path_afferents list
            else:
                projection_type, err_str = self._parse_projection_ref(projection_spec=projection_spec,context=self)
                if err_str and self.verbosePref:
                    warnings.warn("{0}{1} {2}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_TYPE,
                                 err_str,
                                 STATE_PROJECTIONS,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

            # If neither projection_object nor projection_type have been assigned, assign default type
            # Note: this gets projection_type but does NOT instantiate projection; so,
            #       projection is NOT yet in self.path_afferents list
            if not projection_object and not projection_type:
                    projection_type = default_projection_type
                    default_string = kwDefault
                    if self.prefs.verbosePref:
                        warnings.warn("{0}{1} is not a Projection object or specification for one{2}; "
                              "default {3} will be assigned".
                              format(item_prefix_string,
                                     projection_spec.name,
                                     item_suffix_string,
                                     default_projection_type.__class__.__name__))

            # If projection_object has not been assigned, instantiate projection_type
            # Note: this automatically assigns projection to self.path_afferents and to it's sender's efferents list;
            #       when a projection is instantiated, it assigns itself to:
            #           its receiver's .path_afferents attribute (in Projection._instantiate_receiver)
            #           its sender's .efferents attribute (in Projection._instantiate_sender)
            if not projection_object:
                kwargs = {RECEIVER:self,
                          NAME:self.owner.name+' '+self.name+' '+projection_type.className,
                          PARAMS:projection_params,
                          CONTEXT:context}
                # If the projection_spec was a State (see above) and assigned as the sender, assign to SENDER arg
                if sender:
                    kwargs.update({SENDER:sender})
                # If the projection was specified with a keyword or attribute value
                #     then move it to the relevant entry of the params dict for the projection
                # If projection_spec was in the form of a matrix keyword, move it to a matrix entry in the params dict
                if issubclass(projection_type, PathwayProjection_Base) and projection_spec in MATRIX_KEYWORD_SET:
                    kwargs.update({MATRIX:projection_spec})
                # If projection_spec was in the form of a ModulationParam value,
                #    move it to a MODULATION entry in the params dict
                elif (issubclass(projection_type, ModulatoryProjection_Base) and
                          isinstance(projection_spec, ModulationParam)):
                    kwargs[PARAMS].update({MODULATION:projection_spec})
                projection_spec = projection_type(**kwargs)

            # Check that output of projection's function (projection_spec.value is compatible with
            #    variable of the State to which it projects;  if it is not, raise exception:
            # The buck stops here; can't modify projection's function to accommodate the State,
            #    or there would be an unmanageable regress of reassigning projections,
            #    requiring reassignment or modification of sender outputStates, etc.

            # Initialization of projection is deferred
            if projection_spec.value is DEFERRED_INITIALIZATION:
                # Assign instantiated "stub" so it is found on deferred initialization pass (see Process)
                if isinstance(projection_spec, ModulatoryProjection_Base):
                    self.mod_afferents.append(projection_spec)
                else:
                    self.path_afferents.append(projection_spec)
                continue

            # Projection was instantiated, so:
            #    - validate value
            #    - assign to State's path_afferents or mod_afferents list
            # If it is a ModualatoryProjection:
            #    - check that projection's value is compatible with value of the function param being modulated
            #    - assign projection to mod_afferents
            if isinstance(projection_spec, ModulatoryProjection_Base):
                function_param_value = _get_modulated_param(self, projection_spec).function_param_val
                # Match the projection's value with the value of the function parameter
                mod_proj_spec_value = type_match(projection_spec.value, type(function_param_value))
                # If the match was successful (i.e., they are compatible), assign the projection to mod_afferents
                if function_param_value is None or iscompatible(function_param_value, mod_proj_spec_value):
                    # Avoid duplicates, since instantiation of projection (e.g, by Mechanism)
                    #    may have already called this method and assigned projection to self.mod_afferents
                    if not projection_spec in self.mod_afferents:
                        self.mod_afferents.append(projection_spec)
                    continue
            # Otherwise:
            #    - check that projection's value is compatible with the state's variable
            #    - assign projection to path_afferents
            else:
                if iscompatible(self.variable, projection_spec.value):
                    # This is needed to avoid duplicates, since instantiation of projection (e.g., of ControlProjection)
                    #    may have already called this method and assigned projection to self.path_afferents list
                    if not projection_spec in self.path_afferents:
                        self.path_afferents.append(projection_spec)
                    continue

            # Projection specification is not valid
            raise StateError("{}Output of function for {}{} ( ({})) is not compatible with value of {} ({})".
                             format(item_prefix_string,
                                    default_string,
                                    projection_spec.name,
                                    projection_spec.value,
                                    item_suffix_string,
                                    self.value))

    def _instantiate_projection_from_state(self, projection_spec, receiver, context=None):
        """Instantiate outgoing projection from a state and assign it to self.efferents

        Check that projection_spec is one of the following:
        + Projection class (or keyword string constant for one):
            implements default projection for projection class
        + Projection object:
            checks that sender is self
            checks that self.value is compatible with projection's function variable
        + specification dict:
            checks that self.value is compatiable with projection's function variable
            implements projection
            dict must contain:
                + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
        If any of the conditions above fail:
            a default projection is instantiated using self.paramsCurrent[PROJECTION_TYPE]
        Projection is added to self.efferents
        If kwMStateProjections is absent or empty, no projections are created
        """

        from PsyNeuLink.Components.Projections.Projection import Projection_Base

        state_name_string = self.name
        item_prefix_string = ""
        item_suffix_string = state_name_string + " ({} for {})".format(self.__class__.__name__, self.owner.name,)
        default_string = ""
        kwDefault = "default "

        # # MODIFIED 12/1/16 OLD:
        # default_projection_type = self.paramsCurrent[PROJECTION_TYPE]
        # MODIFIED 12/1/16 NEW:
        default_projection_type = self.paramClassDefaults[PROJECTION_TYPE]
        # MODIFIED 12/1/16 END

        # Instantiate projection specification and
        # - insure it is in self.efferents
        # - insure self.value is compatible with the projection's function variable

# FIX: FROM HERE TO BOTTOM OF METHOD SHOULD ALL BE HANDLED IN __init__() FOR PROJECTION_SPEC
        projection_object = None # flags whether projection object has been instantiated; doesn't store object
        projection_type = None   # stores type of projection to instantiate
        projection_params = {}

        # VALIDATE RECEIVER
        # Must be an InputState or ParameterState
        from PsyNeuLink.Components.States.InputState import InputState
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        if not isinstance(receiver, (InputState, ParameterState)):
            raise StateError("Receiver {} of {} from {} must be an inputState or parameterState".
                             format(receiver, projection_spec, self.name))

        # INSTANTIATE PROJECTION_SPEC
        # If projection_spec is a Projection object:
        # - call _check_projection_sender() to check that sender is self; if not, it:
        #     returns object with sender reassigned to self if chosen by user
        #     else, returns new (default) PROJECTION_TYPE object with self as sender
        #     note: in that case, projection will be in self.efferents list
        if isinstance(projection_spec, Projection_Base):
            projection_object, default_class_name = self._check_projection_sender(projection_spec=projection_spec,
                                                                                 receiver=receiver,
                                                                                 messages=[item_prefix_string,
                                                                                           item_suffix_string,
                                                                                           state_name_string],
                                                                                 context=self)
            # If projection's name has not been assigned, base it on State's name:
            if default_class_name:
                # projection_object.name = projection_object.name.replace(default_class_name, self.name)
                projection_object.name = self.name + '_' + projection_object.name
                # Used for error message
                default_string = kwDefault
# FIX:  REPLACE DEFAULT NAME (RETURNED AS DEFAULT) PROJECTION_SPEC NAME WITH State'S NAME, LEAVING INDEXED SUFFIX INTACT

        # If projection_spec is a dict:
        # - get projection_type
        # - get projection_params
        # Note: this gets projection_type but does NOT not instantiate projection; so,
        #       projection is NOT yet in self.efferents list
        elif isinstance(projection_spec, dict):
            # Get projection type from specification dict
            try:
                projection_type = projection_spec[PROJECTION_TYPE]
            except KeyError:
                projection_type = default_projection_type
                default_string = kwDefault
                if self.prefs.verbosePref:
                    print("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_TYPE,
                                 STATE_PROJECTIONS,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))
            else:
                # IMPLEMENTATION NOTE:  can add more informative reporting here about reason for failure
                projection_type, error_str = self._parse_projection_ref(projection_spec=projection_type,
                                                                       context=self)
                if error_str:
                    print("{0}{1} {2}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_TYPE,
                                 error_str,
                                 STATE_PROJECTIONS,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

            # Get projection params from specification dict
            try:
                projection_params = projection_spec[PROJECTION_PARAMS]
            except KeyError:
                if self.prefs.verbosePref:
                    print("{0}{1} not specified in {2} params{3}; default {4} will be assigned".
                          format(item_prefix_string,
                                 PROJECTION_PARAMS,
                                 STATE_PROJECTIONS, state_name_string,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

        # Check if projection_spec is class ref or keyword string constant for one
        # Note: this gets projection_type but does NOT instantiate the projection,
        #       so projection is NOT yet in self.efferents list
        else:
            projection_type, err_str = self._parse_projection_ref(projection_spec=projection_spec,context=self)
            if err_str:
                print("{0}{1} {2}; default {4} will be assigned".
                      format(item_prefix_string,
                             PROJECTION_TYPE,
                             err_str,
                             STATE_PROJECTIONS,
                             item_suffix_string,
                             default_projection_type.__class__.__name__))

        # If neither projection_object nor projection_type have been assigned, assign default type
        # Note: this gets projection_type but does NOT not instantiate projection; so,
        #       projection is NOT yet in self.path_afferents list
        if not projection_object and not projection_type:
                projection_type = default_projection_type
                default_string = kwDefault
                if self.prefs.verbosePref:
                    print("{0}{1} is not a Projection object or specification for one{2}; "
                          "default {3} will be assigned".
                          format(item_prefix_string,
                                 projection_spec.name,
                                 item_suffix_string,
                                 default_projection_type.__class__.__name__))

        # If projection_object has not been assigned, instantiate projection_type
        # Note: this automatically assigns projection to self.efferents and
        #       to it's receiver's afferents list:
        #           when a projection is instantiated, it assigns itself to:
        #               its receiver's .path_afferents attribute (in Projection._instantiate_receiver)
        #               its sender's .efferents list attribute (in Projection._instantiate_sender)
        if not projection_object:
            projection_spec = projection_type(sender=self,
                                              receiver=receiver,
                                              name=self.name+'_'+projection_type.className,
                                              params=projection_params,
                                              context=context)

        # Check that self.value is compatible with projection's function variable
        if not iscompatible(self.value, projection_spec.variable):
            raise StateError("{0}Output ({1}) of {2} is not compatible with variable ({3}) of function for {4}".
                  format(
                         # item_prefix_string,
                         # self.value,
                         # self.name,
                         # projection_spec.variable,
                         # projection_spec.name,
                         # item_suffix_string))
                         item_prefix_string,
                         self.value,
                         item_suffix_string,
                         projection_spec.variable,
                         projection_spec.name
                         ))

        # If projection is valid, assign to State's efferents list
        else:
            # This is needed to avoid duplicates, since instantiation of projection may have already called this method
            #    and assigned projection to self.efferents list
            if not projection_spec in self.efferents:
                self.efferents.append(projection_spec)

    def _check_projection_receiver(self, projection_spec, messages=None, context=None):
        """Check whether Projection object references State as receiver and, if not, return default Projection object

        Arguments:
        - projection_spec (Projection object)
        - message (list): list of three strings - prefix and suffix for error/warning message, and State name
        - context (object): ref to State object; used to identify PROJECTION_TYPE and name

        Returns: tuple (Projection object, str); second value is name of default projection, else None

        :param self:
        :param projection_spec: (Projection object)
        :param messages: (list)
        :param context: (State object)
        :return: (tuple) Projection object, str) - second value is false if default was returned
        """

        prefix = 0
        suffix = 1
        name = 2
        if messages is None:
            messages = ["","","",context.__class__.__name__]
        message = "{}{} is a projection of the correct type for {}, but its receiver is not assigned to {}." \
                  " \nReassign (r) or use default projection (d)?:".format(messages[prefix],
                                                                           projection_spec.name,
                                                                           projection_spec.receiver.name,
                                                                           messages[suffix])

        if projection_spec.receiver is not self:
            reassign = input(message)
            while reassign != 'r' and reassign != 'd':
                reassign = input("Reassign {0} to {1} or use default (r/d)?:".
                                 format(projection_spec.name, messages[name]))
            # User chose to reassign, so return projection object with State as its receiver
            if reassign == 'r':
                projection_spec.receiver = self
                # IMPLEMENTATION NOTE: allow the following, since it is being carried out by State itself
                self.path_afferents.append(projection_spec)
                if self.prefs.verbosePref:
                    print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
                return (projection_spec, None)
            # User chose to assign default, so return default projection object
            elif reassign == 'd':
                print("Default {0} will be used for {1}".
                      format(projection_spec.name, messages[name]))
                return (self.paramsCurrent[PROJECTION_TYPE](receiver=self),
                        self.paramsCurrent[PROJECTION_TYPE].className)
                #     print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
            else:
                raise StateError("Program error:  reassign should be r or d")

        return (projection_spec, None)

    def _check_projection_sender(self, projection_spec, receiver, messages=None, context=None):
        """Check whether Projection object references State as sender and, if not, return default Projection object

        Arguments:
        - projection_spec (Projection object)
        - message (list): list of three strings - prefix and suffix for error/warning message, and State name
        - context (object): ref to State object; used to identify PROJECTION_TYPE and name

        Returns: tuple (Projection object, str); second value is name of default projection, else None

        :param self:
        :param projection_spec: (Projection object)
        :param messages: (list)
        :param context: (State object)
        :return: (tuple) Projection object, str) - second value is false if default was returned
        """

        prefix = 0
        suffix = 1
        name = 2
        if messages is None:
            messages = ["","","",context.__class__.__name__]
        #FIX: NEED TO GET projection_spec.name VS .__name__ STRAIGHT BELOW
        message = "{}{} is a projection of the correct type for {}, but its sender is not assigned to {}." \
                  " \nReassign (r) or use default projection(d)?:".format(messages[prefix],
                                                                          projection_spec.name,
                                                                          projection_spec.sender,
                                                                          messages[suffix])

        if not projection_spec.sender is self:
            reassign = input(message)
            while reassign != 'r' and reassign != 'd':
                reassign = input("Reassign {0} to {1} or use default (r/d)?:".
                                 format(projection_spec.name, messages[name]))
            # User chose to reassign, so return projection object with State as its sender
            if reassign == 'r':
                projection_spec.sender = self
                # IMPLEMENTATION NOTE: allow the following, since it is being carried out by State itself
                self.efferents.append(projection_spec)
                if self.prefs.verbosePref:
                    print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
                return (projection_spec, None)
            # User chose to assign default, so return default projection object
            elif reassign == 'd':
                print("Default {0} will be used for {1}".
                      format(projection_spec.name, messages[name]))
                return (self.paramsCurrent[PROJECTION_TYPE](sender=self, receiver=receiver),
                        self.paramsCurrent[PROJECTION_TYPE].className)
                #     print("{0} reassigned to {1}".format(projection_spec.name, messages[name]))
            else:
                raise StateError("Program error:  reassign should be r or d")

        return (projection_spec, None)

    def _parse_projection_ref(self,
                             projection_spec,
                             # messages=NotImplemented,
                             context=None):
        """Take projection ref and return ref to corresponding type or, if invalid, to  default for context

        Arguments:
        - projection_spec (Projection subclass or str):  str must be a keyword constant for a Projection subclass
        - context (str):

        Returns tuple: (Projection subclass or None, error string)

        :param projection_spec: (Projection subclass or str)
        :param messages: (list)
        :param context: (State object)
        :return: (Projection subclass, string)
        """
        try:
            # Try projection spec as class ref
            is_projection_class = issubclass(projection_spec, Projection)
        except TypeError:
            # Try projection spec as keyword string constant
            if isinstance(projection_spec, str):
                try:
                    from PsyNeuLink.Components.Projections.Projection import ProjectionRegistry
                    projection_spec = ProjectionRegistry[projection_spec].subclass
                except KeyError:
                    # projection_spec was not a recognized key
                    return (None, "not found in ProjectionRegistry")
                # projection_spec was legitimate keyword
                else:
                    return (projection_spec, None)
            # projection_spec was neither a class reference nor a keyword
            else:
                return (None, "neither a class reference nor a keyword")
        else:
            # projection_spec was a legitimate class
            if is_projection_class:
                return (projection_spec, None)
            # projection_spec was class but not Projection
            else:
                return (None, "not a Projection subclass")#


    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):
        """Update each projection, combine them, and assign return result

        Call update for each projection in self.path_afferents (passing specified params)
        Note: only update LearningSignals if context == LEARNING; otherwise, just get their value
        Call self.function (default: LinearCombination function) to combine their values
        Returns combined values of

    Arguments:
    - context (str)

    :param context: (str)
    :return: None

    """

        # SET UP -------------------------------------------------------------------------------------------------------

        # Get state-specific param_specs
        try:
            # Get State params
            self.stateParams = params[self.paramsType]
        except (KeyError, TypeError):
            self.stateParams = {}
        except (AttributeError):
            raise StateError("PROGRAM ERROR: paramsType not specified for {}".format(self.name))
        #endregion

        # Flag format of input
        if isinstance(self.value, numbers.Number):
            # Treat as single real value
            value_is_number = True
        else:
            # Treat as vector (list or np.array)
            value_is_number = False

        # AGGREGATE INPUT FROM PROJECTIONS -----------------------------------------------------------------------------

        # Get type-specific params from PROJECTION_PARAMS
        mapping_params = merge_param_dicts(self.stateParams, MAPPING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        learning_projection_params = merge_param_dicts(self.stateParams, LEARNING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        control_projection_params = merge_param_dicts(self.stateParams, CONTROL_PROJECTION_PARAMS, PROJECTION_PARAMS)
        gating_projection_params = merge_param_dicts(self.stateParams, GATING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        #endregion

        #For each projection: get its params, pass them to it, get the projection's value, and append to relevant list
        self._path_proj_values = []
        for value in self._mod_proj_values:
            self._mod_proj_values[value] = []

        from PsyNeuLink.Components.Process import ProcessInputState
        from PsyNeuLink.Components.Projections.PathwayProjections.PathwayProjection \
            import PathwayProjection_Base
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ModulatoryProjection \
            import ModulatoryProjection_Base
        from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

        # If owner is a Mechanism, get its execution_id
        if isinstance(self.owner, (Mechanism, Process)):
            self_id = self.owner._execution_id
        # If owner is a MappingProjection, get it's sender's execution_id
        elif isinstance(self.owner, MappingProjection):
            self_id = self.owner.sender.owner._execution_id
        else:
            raise StateError("PROGRAM ERROR: Object ({}) of type {} has a {}, but this is only allowed for "
                             "Mechanisms and MappingProjections".
                             format(self.owner.name, self.owner.__class__.__name__, self.__class__.__name__,))


        modulatory_override = False

        # Get values of all Projections
        for projection in self.all_afferents:

            # Only update if sender has also executed in this round
            #     (i.e., has same execution_id as owner)
            # Get sender's execution id
            if hasattr(projection, 'sender'):
                sender = projection.sender
            else:
                if self.verbosePref:
                    warnings.warn("{} to {} {} of {} ignored [has no sender]".format(projection.__class__.__name__,
                                                                                     self.name,
                                                                                     self.__class__.__name__,
                                                                                     self.owner.name))
                continue

            sender_id = sender.owner._execution_id
            if sender_id != self_id:
                continue

            # Only accept projections from a Process to which the owner Mechanism belongs
            if isinstance(sender, ProcessInputState):
                if not sender.owner in self.owner.processes.keys():
                    continue

            # Merge with relevant projection type-specific params
            if isinstance(projection, MappingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, mapping_params, )
            elif isinstance(projection, LearningProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, learning_projection_params)
            elif isinstance(projection, ControlProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, control_projection_params)
            elif isinstance(projection, GatingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, gating_projection_params)
            if not projection_params:
                projection_params = None

            # FIX: UPDATE FOR LEARNING
            # Update LearningSignals only if context == LEARNING;  otherwise, assign zero for projection_value
            # Note: done here rather than in its own method in order to exploit parsing of params above
            if isinstance(projection, LearningProjection) and not LEARNING in context:
                # projection_value = projection.value
                projection_value = projection.value * 0.0
            else:
                projection_value = projection.execute(params=projection_params,
                                                      time_scale=time_scale,
                                                      context=context)

            # If this is initialization run and projection initialization has been deferred, pass
            if INITIALIZING in context and projection_value is DEFERRED_INITIALIZATION:
                continue

            if isinstance(projection, PathwayProjection_Base):
                # Add projection_value to list of TransmissiveProjection values (for aggregation below)
                self._path_proj_values.append(projection_value)

            # If it is a ModulatoryProjection, add its value to the list in the dict entry for the relevant mod_param
            elif isinstance(projection, ModulatoryProjection_Base):
                # Get the meta_param to be modulated from modulation attribute of the  projection's ModulatorySignal
                #    and get the function parameter to be modulated to type_match the projection value below
                mod_meta_param, mod_param_name, mod_param_value = _get_modulated_param(self, projection)
                # If meta_param is DISABLE, ignore the ModulatoryProjection
                if mod_meta_param is Modulation.DISABLE:
                    continue
                if mod_meta_param is Modulation.OVERRIDE:
                    # If paramValidationPref is set, allow all projections to be processed
                    #    to be sure there are no other conflicting OVERRIDES assigned
                    if self.owner.paramValidationPref:
                        if modulatory_override:
                            raise StateError("Illegal assignment of {} to more than one {} ({} and {})".
                                             format(MODULATION_OVERRIDE, MODULATORY_SIGNAL,
                                                    projection.name, modulatory_override[2]))
                        modulatory_override = (MODULATION_OVERRIDE, projection_value, projection)
                        continue
                    # Otherwise, for efficiency, assign OVERRIDE value to State here and return
                    else:
                        self.value = type_match(projection_value, type(self.value))
                        return
                else:
                    mod_value = type_match(projection_value, type(mod_param_value))
                self._mod_proj_values[mod_meta_param].append(mod_value)

        # Handle ModulatoryProjection OVERRIDE
        #    if there is one and it wasn't been handled above (i.e., if paramValidation is set)
        if modulatory_override:
            self.value = type_match(modulatory_override[1], type(self.value))
            return

        # AGGREGATE ModulatoryProjection VALUES  -----------------------------------------------------------------------

        # For each modulated parameter of the state's function, 
        #    combine any values received from the relevant projections into a single modulation value
        #    and assign that to the relevant entry in the params dict for the state's function.
        for mod_param, value_list in self._mod_proj_values.items():
            if value_list:
                aggregated_mod_val = mod_param.reduce(value_list)
                function_param = self.function_object.params[mod_param.attrib_name]
                if not FUNCTION_PARAMS in self.stateParams:
                    self.stateParams[FUNCTION_PARAMS] = {function_param: aggregated_mod_val}
                else:
                    self.stateParams[FUNCTION_PARAMS].update({function_param: aggregated_mod_val})

        # CALL STATE'S function TO GET ITS VALUE  ----------------------------------------------------------------------
        try:
            # pass only function params (which implement the effects of any ModulatoryProjections)
            function_params = self.stateParams[FUNCTION_PARAMS]
        except (KeyError, TypeError):
            function_params = None
        self.value = self._execute(function_params=function_params, context=context)

    def execute(self, input=None, time_scale=None, params=None, context=None):
        return self.function(variable=input, params=params, time_scale=time_scale, context=context)

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, assignment):
        self._owner = assignment

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):

        from math import isnan
        if isinstance(assignment, np.ndarray) and assignment.ndim == 2 and isnan(assignment[0][0]):
                    TEST = True

        self._value = assignment

        # Store value in log if specified
        # Get logPref
        if self.prefs:
            log_pref = self.prefs.logPref

        # Get context
        try:
            curr_frame = inspect.currentframe()
            prev_frame = inspect.getouterframes(curr_frame, 2)
            context = inspect.getargvalues(prev_frame[1][0]).locals['context']
        except KeyError:
            context = ""

        # If context is consistent with log_pref, record value to log
        if (log_pref is LogLevel.ALL_ASSIGNMENTS or
                (log_pref is LogLevel.EXECUTION and EXECUTING in context) or
                (log_pref is LogLevel.VALUE_ASSIGNMENT and (EXECUTING in context and kwAssign in context))):
            self.owner.log.entries[self.name] = LogEntry(CurrentTime(), context, assignment)
            # self.owner.log.entries[self.name] = LogEntry(CentralClock, context, assignment)

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, assignment):
        self._projections = assignment

    @property
    def all_afferents(self):
        return self.path_afferents + self.mod_afferents


def _instantiate_state_list(owner,
                           state_list,              # list of State specs, (state_spec, params) tuples, or None
                           state_type,              # StateType subclass
                           state_param_identifier,  # used to specify state_type state(s) in params[]
                           constraint_value,       # value(s) used as default for state and to check compatibility
                           constraint_value_name,  # name of constraint_value type (e.g. variable, output...)
                           context=None):
    """Instantiate and return a ContentAddressableList of states specified in state_list

    Arguments:
    - state_type (class): State class to be instantiated
    - state_list (list): List of State specifications (generally from owner.paramsCurrent[kw<State>]),
                             each itme of which must be a:
                                 string (used as name)
                                 value (used as constraint value)
                                 # ??CORRECT: (state_spec, params_dict) tuple  
                                     SHOULDN'T IT BE: (state_spec, projection) tuple?
                                 dict (key=name, value=constraint_value or param dict) 
                         if None, instantiate a single default state using constraint_value as state_spec
    - state_param_identifier (str): kw used to identify set of states in params;  must be one of:
        - INPUT_STATE
        - OUTPUT_STATE
    - constraint_value (2D np.array): set of 1D np.ndarrays used as default values and
        for compatibility testing in instantiation of state(s):
        - INPUT_STATE: self.variable
        - OUTPUT_STATE: self.value
        ?? ** Note:
        * this is ignored if param turns out to be a dict (entry value used instead)
    - constraint_value_name (str):  passed to State._instantiate_state(), used in error messages
    - context (str)

    If state_list is None:
        - instantiate a default State using constraint_value,
        - place as the single entry of the list returned.
    Otherwise, if state_list is:
        - a single value:
            instantiate it (if necessary) and place as the single entry in an OrderedDict
        - a list:
            instantiate each item (if necessary) and place in a ContentAddressableList
    In each case, generate a ContentAddressableList with one or more entries, assigning:
        # the key for each entry the name of the outputState if provided,
        #     otherwise, use MECHANISM<state_type>States-n (incrementing n for each additional entry)
        # the state value for each entry to the corresponding item of the mechanism's state_type state's value
        # the dict to both self.<state_type>States and paramsCurrent[MECHANISM<state_type>States]
        # self.<state_type>State to self.<state_type>States[0] (the first entry of the dict)
    Notes:
        * if there is only one state, but the value of the mechanism's state_type has more than one item:
            assign it to the sole state, which is assumed to have a multi-item value
        * if there is more than one state:
            the number of states must match length of mechanisms state_type value or an exception is raised
    """

    state_entries = state_list
    # FIX: INSTANTIATE DICT SPEC BELOW FOR ENTRIES IN state_list

    # If no states were passed in, instantiate a default state_type using constraint_value
    if not state_entries:
        # assign constraint_value as single item in a list, to be used as state_spec below
        state_entries = constraint_value

        # issue warning if in VERBOSE mode:
        if owner.prefs.verbosePref:
            print("No {0} specified for {1}; default will be created using {2} of function ({3})"
                  " as its value".format(state_param_identifier,
                                         owner.__class__.__name__,
                                         constraint_value_name,
                                         constraint_value))

    # States should be either in a list, or possibly an np.array (from constraint_value assignment above):
    if isinstance(state_entries, (ContentAddressableList, list, np.ndarray)):

        # VALIDATE THAT NUMBER OF STATES IS COMPATIBLE WITH NUMBER OF CONSTRAINT VALUES
        num_states = len(state_entries)

        # Check that constraint_value is an indexable object, the items of which are the constraints for each state
        # Notes
        # * generally, this will be a list or an np.ndarray (either >= 2D np.array or with a dtype=object)
        # * for OutputStates, this should correspond to its value
        try:
            # Insure that constraint_value is an indexible item (list, >=2D np.darray, or otherwise)
            num_constraint_items = len(constraint_value)
        except:
            raise StateError("PROGRAM ERROR: constraint_value ({0}) for {1} of {2}"
                                 " must be an indexable object (e.g., list or np.ndarray)".
                                 format(constraint_value, constraint_value_name, state_type.__name__))

        # IMPLEMENTATION NOTE: NO LONGER VALID SINCE outputStates CAN NOW BE ONE TO MANY OR MANY TO ONE
        #                      WITH RESPECT TO ITEMS OF constraint_value (I.E., owner.value)
        # If number of states exceeds number of items in constraint_value, raise exception
        if num_states > num_constraint_items:
            raise StateError("There are too many {0} specified ({1}) in {2} "
                                 "for the number of values ({3}) in the {4} of its function".
                                 format(state_param_identifier,
                                        num_states,
                                        owner.__class__.__name__,
                                        num_constraint_items,
                                        constraint_value_name))

        # If number of states is less than number of items in constraint_value, raise exception
        elif num_states < num_constraint_items:
            raise StateError("There are fewer {0} specified ({1}) than the number of values ({2}) "
                                 "in the {3} of the function for {4}".
                                 format(state_param_identifier,
                                        num_states,
                                        num_constraint_items,
                                        constraint_value_name,
                                        owner.name))

        # INSTANTIATE EACH STATE

        # Iterate through list or state_dict:
        # - instantiate each item or entry as state_type State
        # - get name, and use as key to assign as entry in self.<*>states
        states = ContentAddressableList(component_type=State_Base,
                                        name=owner.name+' ContentAddressableList of ' + state_param_identifier)

        # Instantiate state for entry in list or dict
        # Note: if state_entries is a list, state_spec is the item, and key is its index in the list
        for index, state_spec in state_entries if isinstance(state_entries, dict) else enumerate(state_entries):
            state_name = ""

            # State_entries is a dict, so use:
            # - entry index as state's name
            # - entry value as state_spec
            if isinstance(index, str):
                state_name = index
                state_constraint_value = constraint_value
                # Note: state_spec has already been assigned to entry value by enumeration above
                # MODIFIED 12/11/16 NEW:
                # If it is an "exposed" number, make it a 1d np.array
                if isinstance(state_spec, numbers.Number):
                    state_spec = np.atleast_1d(state_spec)
                # MODIFIED 12/11/16 END
                state_params = None

            # State_entries is a list
            else:
                if isinstance(state_spec, tuple):
                    if not len(state_spec) == 2:
                        raise StateError("List of {}s to instantiate for {} has tuple with more than 2 items:"
                                                  " {}".format(state_type.__name__, owner.name, state_spec))

                    state_spec, state_params = state_spec
                    if not (isinstance(state_params, dict) or state_params is None):
                        raise StateError("In list of {}s to instantiate for {}, second item of tuple "
                                                  "({}) must be a params dict or None:".
                                                  format(state_type.__name__, owner.name, state_params))
                else:
                    state_params = None

                # If state_spec is a string, then use:
                # - string as the name for a default state
                # - index (index in list) to get corresponding value from constraint_value as state_spec
                # - assign same item of constraint_value as the constraint
                if isinstance(state_spec, str):
                    # Use state_spec as state_name if it has not yet been used
                    if not state_name is state_spec and not state_name in states:
                        state_name = state_spec
                    # Add index suffix to name if it is already been used
                    # Note: avoid any chance of duplicate names (will cause current state to overwrite previous one)
                    else:
                        state_name = state_spec + '_' + str(index)
                    state_spec = constraint_value[index]
                    state_constraint_value = constraint_value[index]

                # FIX: 5/21/17  ADD, AND DEAL WITH state_spec AND state_constraint
                # elif isinstance(state_spec, dict):
                #     # If state_spec has NAME entry
                #     if NAME in state_spec:
                #         # If it has been used, add suffix to it
                #         if state_name is state_spec[NAME]:
                #             state_name = state_spec[NAME] + '_' + str(key)
                #         # Otherwise, use it
                #         else:
                #             state_name = state_spec[NAME]
                #     state_spec = ??
                #     state_constraint_value = ??


                # If state_spec is NOT a string, then:
                # - use default name (which is incremented for each instance in register_categories)
                # - use item as state_spec (i.e., assume it is a specification for a State)
                #   Note:  still need to get indexed element of constraint_value,
                #          since it was passed in as a 2D array (one for each state)
                else:
                    # If only one state, don't add index suffix
                    if num_states == 1:
                        state_name = 'Default_' + state_param_identifier[:-1]
                    # Add incremented index suffix for each state name
                    else:
                        state_name = 'Default_' + state_param_identifier[:-1] + "-" + str(index+1)
                    # If it is an "exposed" number, make it a 1d np.array
                    if isinstance(state_spec, numbers.Number):
                        state_spec = np.atleast_1d(state_spec)

                    state_constraint_value = constraint_value[index]

            state = _instantiate_state(owner=owner,
                                       state_type=state_type,
                                       state_name=state_name,
                                       state_spec=state_spec,
                                       state_params=state_params,
                                       constraint_value=state_constraint_value,
                                       constraint_value_name=constraint_value_name,
                                       context=context)

            # Get name of state, and use as index to assign to states ContentAddressableList
            states[state.name] = state
        return states

    else:
        # This shouldn't happen, as MECHANISM<*>States was validated to be one of the above in _validate_params
        raise StateError("PROGRAM ERROR: {} for {} is not a recognized \'{}\' specification for {}; "
                         "it should have been converted to a list in Mechanism._validate_params)".
                         format(state_entries, owner.name, state_param_identifier, owner.__class__.__name__))


def _instantiate_state(owner,                  # Object to which state will belong
                      state_type,              # State subclass
                      state_name,              # Name for state (also used to refer to subclass in prompts)
                      state_spec,              # State subclass, object, spec dict, tuple, projection, value or str
                      state_params,            # params for state
                      constraint_value,        # Value used to check compatibility
                      constraint_value_name,   # Name of constraint_value's type (e.g. variable, output...)
                      context=None):
    """Instantiate a State of specified type, with a value that is compatible with constraint_value

    Constraint value must be a number or a list or tuple of numbers
    (since it is used as the variable for instantiating the requested state)

    If state_spec is a:
    + State class:
        implement default using constraint_value
    + State object:
        check owner is owner (if not, user is given options in _check_state_ownership)
        check compatibility of value with constraint_value
    + 2-item tuple: (only allowed for ParameterState spec)
        assign first item to state_spec
            if it is a string:
                test if it is a keyword and get its value by calling keyword method of owner's execute method
                otherwise, return None (suppress assignment of parameterState)
        assign second item to STATE_PARAMS{STATE_PROJECTIONS:<projection>}
    + Projection object:
        assign constraint_value to value
        assign projection to STATE_PARAMS{STATE_PROJECTIONS:<projection>}
    + Projection class (or keyword string constant for one):
        assign constraint_value to value
        assign projection class spec to STATE_PARAMS{STATE_PROJECTIONS:<projection>}
    + specification dict for State (see XXX for context):
        check compatibility of STATE_VALUE with constraint_value
    + value:
        implement default using the value
    + str:
        test if it is a keyword and get its value by calling keyword method of owner's execute method
        # otherwise, return None (suppress assignment of parameterState)
        otherwise, implement default using the string as its name
    Check compatibility with constraint_value
    If any of the conditions above fail:
        a default State of specified type is instantiated using constraint_value as value

    If state_params is specified, include as params arg with instantiation of state
    
    Returns a state or None
    """

    # IMPLEMENTATION NOTE: CONSIDER MOVING MUCH IF NOT ALL OF THIS TO State.__init__()

    # FIX: IF VARIABLE IS IN state_params EXTRACT IT AND ASSIGN IT TO constraint_value 5/9/17

    # VALIDATE ARGS
    if not inspect.isclass(state_type) or not issubclass(state_type, State):
        raise StateError("PROGRAM ERROR: state_type arg ({}) for _instantiate_state must be a State subclass".
                         format(state_type))
    if not isinstance(state_name, str):
        raise StateError("PROGRAM ERROR: state_name arg ({}) for _instantiate_state must be a string".
                             format(state_name))
    if not isinstance(constraint_value_name, str):
        raise StateError("PROGRAM ERROR: constraint_value_name arg ({}) for _instantiate_state must be a string".
                             format(constraint_value_name))

    state_params = state_params or {}

    # PARSE constraint_value
    constraint_dict = _parse_state_spec(owner=owner,
                                        state_type=state_type,
                                        state_spec=constraint_value,
                                        value=None,
                                        params=None)
    constraint_value = constraint_dict[VARIABLE]
    # constraint_value = constraint_dict[VALUE]

    # PARSE state_spec using constraint_value as default for value
    state_spec = _parse_state_spec(owner=owner,
                                   state_type=state_type,
                                   state_spec=state_spec,
                                   # state_spec=state_spec,
                                   name=state_name,
                                   params=state_params,
                                   value=constraint_value)

    # state_spec is State object
    # - check that its value attribute matches the constraint_value
    # - check that its owner = owner
    # - if either fails, assign default
    if isinstance(state_spec, state_type):
        # Check that State's value is compatible with Mechanism's variable
        if iscompatible(state_spec.value, constraint_value):
            # Check that Mechanism is State's owner;  if it is not, user is given options
            state =  _check_state_ownership(owner, state_name, state_spec)
            if state:
                return state
            else:
                # State was rejected, and assignment of default selected
                state = constraint_value
        else:
            # State's value doesn't match constraint_value, so assign default
            if owner.verbosePref:
                warnings.warn("Value of {} for {} ({}, {}) does not match expected ({}); "
                              "default {} will be assigned)".
                              format(state_type.__name__,
                                     owner.name,
                                     state_spec.name,
                                     state_spec.value,
                                     constraint_value,
                                     state_type.__name__))
            # state = constraint_value
            state_variable = constraint_value

    # Otherwise, state_spec should now be a state specification dict
    state_variable = state_spec[VARIABLE]
    state_value = state_spec[VALUE]

    # Check that it's variable is compatible with constraint_value, and if not, assign the latter as default variable
    if constraint_value is not None and not iscompatible(state_variable, constraint_value):
        if owner.prefs.verbosePref:
            print("{} is not compatible with constraint value ({}) specified for {} of {};  latter will be used".
                  format(VARIABLE, constraint_value, state_type, owner.name))
        state_variable = constraint_value
    # else:
    #     constraint_value = state_variable

    # INSTANTIATE STATE:
    # Note: this will be either a default State instantiated using constraint_value as its value
    #       or one determined by a specification dict, depending on which of the following obtained above:
    # - state_spec was a 2-item tuple
    # - state_spec was a specification dict
    # - state_spec was a value
    # - value of specified State was incompatible with constraint_value
    # - owner of State was not owner and user chose to implement default
    # IMPLEMENTATION NOTE:
    # - setting prefs=NotImplemented causes TypeDefaultPreferences to be assigned (from ComponentPreferenceSet)
    # - alternative would be prefs=owner.prefs, causing state to inherit the prefs of its owner;

    #  Convert constraint_value to np.array to match state_variable (which, as output of function, will be an np.array)
    constraint_value = convert_to_np_array(constraint_value,1)

    # Implement default State
    state = state_type(owner=owner,
                       reference_value=constraint_value,
                       variable=state_variable,
                       name=state_spec[NAME],
                       params=state_spec[PARAMS],
                       prefs=None,
                       context=context)

# FIX LOG: ADD NAME TO LIST OF MECHANISM'S VALUE ATTRIBUTES FOR USE BY LOGGING ENTRIES
    # This is done here to register name with Mechanism's stateValues[] list
    # It must be consistent with value setter method in State
# FIX LOG: MOVE THIS TO MECHANISM STATE __init__ (WHERE IT CAN BE KEPT CONSISTENT WITH setter METHOD??
#      OR MAYBE JUST REGISTER THE NAME, WITHOUT SETTING THE
# FIX: 2/17/17:  COMMENTED THIS OUT SINCE IT CREATES AN ATTRIBUTE ON OWNER THAT IS NAMED <state.name.value>
#                NOT SURE WHAT THE PURPOSE IS
#     setattr(owner, state.name+'.value', state.value)

    return state

def _check_parameter_state_value(owner, param_name, value):
    """Check that parameter value (<ParameterState>.value) is compatible with value in paramClassDefault

    :param param_name: (str)
    :param value: (value)
    :return: (value)
    """
    default_value = owner.paramClassDefaults[param_name]
    if iscompatible(value, default_value):
        return value
    else:
        if owner.prefs.verbosePref:
            print("Format is incorrect for value ({0}) of {1} in {2};  default ({3}) will be used.".
                  format(value, param_name, owner.name, default_value))
        return default_value

def _check_state_ownership(owner, param_name, mechanism_state):
    """Check whether State's owner is owner and if not offer options how to handle it

    If State's owner is not owner, options offered to:
    - reassign it to owner
    - make a copy and assign to owner
    - return None => caller should assign default

    :param param_name: (str)
    :param mechanism_state: (State)
    :param context: (str)
    :return: (State or None)
    """

    if mechanism_state.owner != owner:
        reassign = input("\nState for \'{0}\' parameter, assigned to {1} in {2}, already belongs to {3}. "
                         "You can reassign it (r), copy it (c), or assign default (d):".
                         format(mechanism_state.name, param_name, owner.name,
                                mechanism_state.owner.name))
        while reassign != 'r' and reassign != 'c' and reassign != 'd':
            reassign = input("\nReassign (r), copy (c), or default (d):".
                             format(mechanism_state.name, param_name, owner.name,
                                    mechanism_state.owner.name))

            if reassign == 'r':
                while reassign != 'y' and reassign != 'n':
                    reassign = input("\nYou are certain you want to reassign it {0}? (y/n):".
                                     format(param_name))
                if reassign == 'y':
                    # Note: assumed that parameters have already been checked for compatibility with assignment
                    return mechanism_state

        # Make copy of state
        if reassign == 'c':
            import copy
            # # MODIFIED 10/28/16 OLD:
            # mechanism_state = copy.deepcopy(mechanism_state)
            # MODIFIED 10/28/16 NEW:
            # if owner.verbosePref:
                # warnings.warn("WARNING: at present, 'deepcopy' can be used to copy states, "
                #               "so some components of {} might be missing".format(mechanism_state.name))
            print("WARNING: at present, 'deepcopy' can be used to copy states, "
                  "so some components of {} assigned to {} might be missing".
                  format(mechanism_state.name, append_type_to_name(owner)))
            mechanism_state = copy.copy(mechanism_state)
            # MODIFIED 10/28/16 END

        # Assign owner to chosen state
        mechanism_state.owner = owner
    return mechanism_state


def _check_projection_sender_compatiability(owner, projection_type, sender_type):
    from PsyNeuLink.Components.States.OutputState import OutputState
    from PsyNeuLink.Components.States.ModulatorySignals.LearningSignal import LearningSignal
    from PsyNeuLink.Components.States.ModulatorySignals.ControlSignal import ControlSignal
    from PsyNeuLink.Components.States.ModulatorySignals.GatingSignal import GatingSignal
    from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
    from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
    from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection

    if issubclass(projection_type, MappingProjection) and issubclass(sender_type, OutputState):
        return
    if issubclass(projection_type, LearningProjection) and issubclass(sender_type, LearningSignal):
        return
    if issubclass(projection_type, ControlProjection) and issubclass(sender_type, ControlSignal):
        return
    if issubclass(projection_type, GatingProjection) and issubclass(sender_type, GatingSignal):
        return
    else:
        raise StateError("Illegal specification of a {} for {} of {}".
                         format(sender_type, owner.__class__.__name__, owner.owner.name))


# FIX 5/23/17:  UPDATE TO ACCOMODATE (param, ControlSignal) TUPLE
@tc.typecheck
def _parse_state_spec(owner,
                      state_type:_is_state_type,
                      state_spec,
                      name:tc.optional(str)=None,
                      variable=None,
                      value=None,
                      # projections:tc.any(list, bool)=[],
                      # modulatory_projections:tc.any(list,bool)=[],
                      params=None,
                      force_dict=False):

    """Return either state object or state specification dict for state_spec
    
    If state_spec is or resolves to a state object:
        if force_dict is False:  return state object 
        if force_dict is True: parse into state specification_dictionary 
            (replacing any components with their id to avoid problems with deepcopy)   
    Otherwise, return state specification dictionary using arguments provided as defaults
    Warn if variable is assigned the default value, and verbosePref is set on owner.
    **value** arg should generally be a constraint for the value of the state;  
        if state_spec is a Projection, and method is being called from:
            InputState, value should be the projection's value; 
            ParameterState, value should be the projection's value; 
            OutputState, value should be the projection's variable
    Any entries with keys other than XXX are moved to entries of the dict in the PARAMS entry
    """

    from PsyNeuLink.Components.Projections.Projection import projection_keywords

    # # IMPLEMENTATION NOTE:  ONLY CALLED IF force_dict=True;  CAN AVOID BY NEVER SETTING THAT OPTION TO True
    # #                       STILL NEEDS WORK: SEEMS TO SET PARAMS AND SO CAUSES CALL TO assign_params TO BAD EFFECT
    # # Get convert state object into state specification dictionary,
    # #     replacing any set, dict or Component with its id to avoid problems with deepcopy
    # @tc.typecheck
    # def _state_dict(state:State):
    #     @tc.typecheck
    #     # Checks if item is Component and returns its id if it is
    #     def filter_params(item):
    #         if isinstance(item, (set, dict, Component)):
    #             item = id(item)
    #         return item
    #     if hasattr(state, 'params') and state.params:
    #         for param in state.params:
    #             if isinstance(param, collections.Iterable) and not isinstance(param, str):
    #                 for index, item in param if isinstance(param, dict) else enumerate(param):
    #                     state.params[param][index] = filter_params(item)
    #             else:
    #                 state.params[param] = filter_params(state.params[param])
    #     return dict(**{NAME:state.name,
    #                   VARIABLE:state.variable,
    #                   VALUE:state.value,
    #                   # PARAMS:{STATE_PROJECTIONS:state.pathway_projections}})
    #                   PARAMS:state.params})

    # Validate that state_type is a State class
    if not inspect.isclass(state_type) or not issubclass(state_type, State):
        raise StateError("\'state_type\' arg ({}) must be a sublcass of {}".format(state_type,
                                                                                   State.__name__))
    state_type_name = state_type.__name__

    # State object:
    # - check that it is of the specified type and, if so:
    #     - if force_dict is False, return the primary state object
    #     - if force_dict is True, get state's attributes and return their values in a state specification dictionary

    if isinstance(state_spec, State):
        if isinstance(state_spec, state_type):
            # if force_dict:
            #     return _state_dict(state_spec)
            # else:
            #     return state_spec
            return state_spec
        else:
            raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
                             "class of state being instantiated ({})".format(state_spec, state_type_name))

    # Mechanism object:
    # - call owner to get primary state of specified type;
    # - if force_dict is False, return the primary state object
    # - if force_dict is True, get primary state's attributes and return their values in a state specification dict
    if isinstance(state_spec, Mechanism):
        primary_state = owner._get_primary_state(state_type)
        # if force_dict:
        #     return _state_dict(primary_state)
        # else:
        #     return primary_state
        return primary_state

    # # Avoid modifying any objects passed in via state_spec
    # state_spec = copy.deepcopy(state_spec)

    # params = params or {}

    if params:
        # If variable is specified in state_params, use that
        if VARIABLE in params and params[VARIABLE] is not None:
            variable = params[VARIABLE]

    # Create default dict for return
    state_dict = {NAME: name,
                  VARIABLE: variable,
                  VALUE: value,
                  PARAMS: params}

    # State class
    if inspect.isclass(state_spec) and issubclass(state_spec, State):
        if state_spec is state_type:
            state_dict[VARIABLE] = state_spec.variableClassDefault
        else:
            raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
                             "class of state being instantiated ({})".format(state_spec, state_type_name))

    # # State object [PARSED INTO DICT HERE]
    # elif isinstance(state_spec, State):
    #     if state_spec is state_type:
    #         name = state_spec.name
    #         # variable = state_spec.value
    #         # variable = state_spec.variableClassDefault
    #         variable = state_spec.variable
    #         value = state_spec.value
    #         modulatory_projections =  state_spec.mod_projections
    #         params = state_spec.user_params.copy()
    #     else:
    #         raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
    #                          "class of state being instantiated ({})".format(state_spec, state_type))

    # Specification dict
    # - move any entries other than for standard args into dict in params entry
    elif isinstance(state_spec, dict):
        # Dict has a single entry in which the key is not a recognized keyword,
        #    so assume it is of the form {<STATE_NAME>:<STATE SPECIFICATION DICT>}:
        #    assign STATE_NAME as name, and return parsed SPECIFICATION_DICT
        if len(state_spec) == 1 and list(state_spec.keys())[0] not in (state_keywords | STANDARD_ARGS):
            name, state_spec = list(state_spec.items())[0]
            state_dict = _parse_state_spec(owner=owner,
                                           state_type=state_type,
                                           state_spec=state_spec,
                                           name=name,
                                           variable=variable,
                                           value=value,
                                           # projections=projections,
                                           # modulatory_projections=modulatory_projections,
                                           # params=params)
                                           )
            if state_dict[PARAMS]:
                params = params or {}
                # params.update(state_dict[PARAMS])
                # Use name specified as key in state_spec (overrides one in SPEFICATION_DICT if specified):
                state_dict[PARAMS].update(params)

        else:
            # Warn if VARIABLE was not in dict
            if not VARIABLE in state_spec and owner.prefs.verbosePref:
                print("{} missing from specification dict for {} of {};  default ({}) will be used".
                      format(VARIABLE, state_type, owner.name, state_spec))
            # Move all params-relevant entries from state_spec into params
            for spec in [param_spec for param_spec in state_spec.copy()
                         if not param_spec in STANDARD_ARGS]:
                params = params or {}
                params[spec] = state_spec[spec]
                # MODIFIED 6/5/17 OLD: [REINSTATED, BUT CAUSING TROUBLE IN STROOP TEST SCRIPT]
                del state_spec[spec]
                # MODIFIED 6/5/17 END
            state_dict.update(state_spec)
            # state_dict = state_spec
            if params:
                if state_dict[PARAMS] is None:
                    state_dict[PARAMS] = {}
                state_dict[PARAMS].update(params)

    # # 2-item tuple (spec, projection)
    # 2-item tuple (spec, Component)
    elif isinstance(state_spec, tuple):
        if len(state_spec) != 2:
            raise StateError("Tuple provided as state_spec for {} of {} ({}) must have exactly two items".
                             format(state_type_name, owner.name, state_spec))
        # IMPLEMENTATION NOTE: Mechanism allowed in tuple to accomodate specification of param for ControlSignal
        if not (_is_projection_spec(state_spec[1]) or isinstance(state_spec[1], (Mechanism, State))):
            raise StateError("2nd item of tuple in state_spec for {} of {} ({}) must be a specification "
                             "for a mechanism, state, or projection".
                             format(state_type_name, owner.__class__.__name__, state_spec[1]))
        # Put projection spec from second item of tuple in params
        params = params or {}
        # FIX 5/23/17: NEED TO HANDLE NON-MODULATORY PROJECTION SPECS
        params.update({STATE_PROJECTIONS:[state_spec[1]]})

        # Parse state_spec in first item of tuple (without params)
        state_dict = _parse_state_spec(owner=owner,
                                       state_type=state_type,
                                       state_spec=state_spec[0],
                                       name=name,
                                       variable=variable,
                                       value=value,
                                       # projections=projections,
                                       params={})
        # Add params (with projection spec) to any params specified in first item of tuple
        if state_dict[PARAMS] is None:
            state_dict[PARAMS] = {}
        state_dict[PARAMS].update(params)

    # Projection class, object, or keyword:
    #     set variable to value and assign projection spec to STATE_PROJECTIONS entry in params
    # IMPLEMENTATION NOTE:  It is the caller's responsibility to assign the value arg
    #                           appropriately for the state being requested, for:
    #                               InputState, projection's value;
    #                               ParameterState, projection's (= parameter's) value;
    #                               OutputState, projection's variable .
    # Don't allow matrix keywords -- force them to be converted from a string into a value (below)
    elif _is_projection_spec(state_spec, include_matrix_keywords=False):
        # state_spec = state_variable
        state_dict[VARIABLE] =  value
        if state_dict[PARAMS] is None:
            state_dict[PARAMS] = {}
        state_dict[PARAMS].update({STATE_PROJECTIONS:[state_spec]})

    # string (keyword or name specification)
    elif isinstance(state_spec, str):
        # Check if it is a keyword
        spec = get_param_value_for_keyword(owner, state_spec)
        # A value was returned, so use value of keyword as variable
        if spec is not None:
            state_dict[VARIABLE] = spec
            if owner.prefs.verbosePref:
                print("{} not specified for {} of {};  default ({}) will be used".
                      format(VARIABLE, state_type, owner.name, value))
        # It is not a keyword, so treat string as the name for the state
        else:
            state_dict[NAME] = state_spec

    # function; try to resolve to a value, otherwise return None to suppress instantiation of state
    elif isinstance(state_spec, function_type):
        state_dict[VALUE] = get_param_value_for_function(owner, state_spec)
        if state_dict[VALUE] is None:
            # return None
            raise StateError("PROGRAM ERROR: state_spec for {} of {} is a function ({}), "
                             "but it failed to return a value".format(state_type_name, owner.name, state_spec))

    # value, so use as variable of input_state
    elif is_value_spec(state_spec):
        state_dict[VARIABLE] = state_spec
        state_dict[VALUE] = state_spec

    elif state_spec is None:
        # pass
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is None".format(state_type_name, owner.name))

    else:
        if name and hasattr(owner, name):
            owner_name = owner.name
        else:
            owner_name = owner.__class__.__name__
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is an unrecognized specification ({})".
                         format(state_type_name, owner.name, state_spec))


    # If variable is none, use value:
    if state_dict[VARIABLE] is None:
        state_dict[VARIABLE] = state_dict[VALUE]

    return state_dict
