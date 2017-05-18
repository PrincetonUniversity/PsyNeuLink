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
provide by them.  The value of a state can be modulated by a `ModulatoryProjection`. There are three types of states, 
all of which are used by `mechanisms <Mechanism>`, one of which is used by `MappingProjections <MappingProjection>`, 
and all of which are subject to modulation by particular types of ModulatoryProjections, as summarized below:

* **InputState**:
     used by a mechanism to receive input from `MappingProjections <MappingProjection>`;  its value can be modulated 
     by a `GatingProjection`.

* **ParameterState**:
    * used by a mechanism to represent the value of one of its parameters, or a parameter of its :keyword:`function`,
      possibly modulated by a `ControlProjection`;
    * used by a `MappingProjection` to represent the value of its `matrix <MappingProjection.MappingProjection.matrix>`
      parameter, possibly modulated by a `LearningProjection`.

* **OutputState**:
    * used by a mechanism to send its to any outgoing projection(s):
      * `MappingProjection` for a `ProcessingMechanism <ProcessingMechanism>`;
      * `LearningProjection` for a `MonitoringMechanism <MonitoringMechanism>`.
      * `GatingProjection` for a `GatingMechanism <GatingMechanism>`;
      * `ControlProjection` for a `ControlMechanism <ControlMechanism>`.
      Its value can be modulated by a `GatingProjection`. 

.. _State_Creation:

Creating a State
----------------

States can be created using the constructor for one of the subclasses.  However, in general, they are created
automatically by the objects to which they belong, and/or through specification in context (e.g., when
`specifying the parameters <ParameterState_Specifying_Parameters>` of a mechanism or its function to be controlled,
or of a `MappingProjections to be learned <MappingProjection_Tuple_Specification>).

Structure
---------

Every state is owned by either a `mechanism <Mechanism>` or a `projection <Projection>`. Like all PsyNeuLink
components, a state has the three following core attributes:

    * `variable <State.variable>`:  for an `inputState <InputState>` and `parameterState <ParameterState>`,
      the value of this is determined by the  value(s) of the projection(s) that it receives (and that are listed in
      its `afferents <State.afferents>` attribute).  For an `outputState <OutputState>`,
      it is the item of the owner mechanism's :keyword:`value` to which the outputState is assigned (specified by the
      outputStates `index <OutputState_Index>` attribute.
    ..
    * `function <State.function>`:  for an `inputState <InputState>` this aggregates the values of the projections 
      that the state receives (the default is `LinearCombination` that sums the values), under the potential influence
      of a `Gating` projection;  for a `parameterState <ParameterState>`, it determines the value of the associated 
      parameter, under the potential influence of a `ControlProjection` (for a `Mechanism`) or a `LearningProjection`
      (for a `MappingProjection`);  for an outputState, it conveys the result  of the mechanism's function to its
      output_values, under the potential influence of a `GatingProjection`.  
      See `ModulatoryProjections <ModulatoryProjection_Structure>` for a description of how these can be used to 
      influence the `function <State.function> of a state.
    ..
    * `value <State.value>`:  for an `inputState <InputState>` this is the aggregated value of the projections it 
      receives;  for a `parameterState <ParameterState>`, this determines the value of the associated parameter;  
      for an `outputState <OutputState>`, it is the item of the  owner mechanism's :keyword:`value` to which the 
      outputState is assigned, possibly modified by its `calculate <OutputState_Calculate>` attribute.

Execution
---------

State cannot be executed.  They are updated when the component to which they belong is executed.  InputStates and 
parameterStates belonging to a mechanism are updated before the mechanism's function is called.  OutputStates
are updated after the mechanism's function is called.  When a state is updated, it executes any projections that 
project to it (listed in its `afferents <State.afferents>` attribute), uses the values 
it receives from them as the variable for its `function <State.function>`, and calls that to determine its own 
`value <State.value>`. This conforms to a "lazy evaluation" protocol (see :ref:`Lazy Evaluation <LINK>` for a more 
detailed discussion).

.. _State_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.Projections.Projection import projection_keywords

state_keywords = component_keywords.copy().update({STATE_VALUE,
                                                   STATE_PARAMS,
                                                   STATE_PROJECTIONS,
                                                   MODULATORY_PROJECTIONS,
                                                   PROJECTION_TYPE})

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
    State_Base(  \
    owner        \
    value=None,  \
    params=None, \
    name=None,   \
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
                - receives inputs from projections (self.afferents, STATE_PROJECTIONS)
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

    baseValue : number, list or np.ndarray
        value with which the state was initialized.

    afferents : Optional[List[Projection]]
        list of projections for which the state is a :keyword:`receiver`.

    efferents : Optional[List[Projection]]
        list of projections for which the state is a :keyword:`sender`.

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

    kpState = "State"
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

    #endregion

    def __init__(self,
                 owner,
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

        if not isinstance(context, State_Base):
            raise StateError("Direct call to abstract class State() is not allowed; "
                                      "use state() or one of the following subclasses: {0}".
                                      format(", ".join("{!s}".format(key) for (key) in StateRegistry.keys())))

        # FROM MECHANISM:
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)

        # VALIDATE owner
        if isinstance(owner, (Mechanism, Projection)):
            self.owner = owner
        else:
            raise StateError("\'owner\' argument ({0}) for {1} must be a mechanism or projection".
                                      format(owner, name))

        register_category(entry=self,
                          base_class=State_Base,
                          name=name,
                          registry=owner._stateRegistry,
                          # sub_group_attr='owner',
                          context=context)

        self.afferents = []
        self.efferents = []

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

# # FIX LOG: EITHER GET RID OF THIS NOW THAT @property HAS BEEN IMPLEMENTED, OR AT LEAST INTEGRATE WITH IT
#         # add state to KVO observer dict
#         self.observers = {self.kpState: []}
#
# # FIX: WHY IS THIS COMMENTED OUT?  IS IT HANDLED BY SUBCLASSES??
    # def register_category(self):
    #     register_mechanism_state_subclass(self)

    def _validate_variable(self, variable, context=None):
        """Validate variable and assign validated values to self.variable

        Sets self.baseValue = self.value = self.variable = variable
        Insures that it is a number of list or tuple of numbers

        This overrides the class method, to perform more detailed type checking
        See explanation in class method.
        Note:  this method (or the class version) is called only if the parameter_validation attribute is True

        :param variable: (anything but a dict) - variable to be validated:
        :param context: (str)
        :return none:
        """

        super(State,self)._validate_variable(variable, context)

        if not context:
            context = kwAssign + ' Base Value'
        else:
            context = context + kwAssign + ' Base Value'

        self.baseValue = self.variable

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

        :param request_set:
        :param target_set:
        :param context:
        :return:
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

        :param context:
        :return:
        """

        var_is_matrix = False
        # If variable is a matrix (e.g., for the MATRIX parameterState of a MappingProjection),
        #     it needs to be embedded in a list so that it is properly handled in by LinearCombination
        #     (i.e., solo matrix is returned intact, rather than treated as arrays to be combined);
        # Notes:
        #     * this is not a problem when LinearCombination is called in state.update(), since that puts
        #         projection values in a list before calling LinearCombination to combine them
        #     * it is removed from the list below, after calling _instantiate_function
        #     * no change is made to PARAMETER_MODULATION_FUNCTION here (matrices may be multiplied or added)
        #         (that is handled by the indivudal state subclasses (e.g., ADD is enforced for MATRIX parameterState)
        if (isinstance(self.variable, np.matrix) or
                (isinstance(self.variable, np.ndarray) and self.variable.ndim >= 2)):
            self.variable = [self.variable]
            var_is_matrix = True

        super()._instantiate_function(context=context)

        # If it is a matrix, remove from list in which it was embedded after instantiating and evaluating function
        if var_is_matrix:
            self.variable = self.variable[0]

        # Insure that output of function (self.value) is compatible with (same format as) its input (self.variable)
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
        """Instantiate projections to a state and assign them to self.afferents

        For each projection spec in STATE_PROJECTIONS, check that it is one or a list of any of the following:
        + Projection class (or keyword string constant for one):
            implements default projection for projection class
        + Projection object:
            checks that receiver is self
            checks that projection function output is compatible with self.value
        + specification dict:
            checks that projection function output is compatible with self.value
            implements projection
            dict must contain:
                + PROJECTION_TYPE:<Projection class> - must be a subclass of Projection
                + PROJECTION_PARAMS:<dict> - must be dict of params for PROJECTION_TYPE
        If any of the conditions above fail:
            a default projection is instantiated using self.paramsCurrent[PROJECTION_TYPE]
        Each projection in the list is added to self.afferents
        If kwMStateProjections is absent or empty, no projections are created

        :param context: (str)
        :return state: (State)
        """

        from PsyNeuLink.Components.Projections.Projection import Projection_Base
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

        # MODIFIED 12/1/16 OLD:
        # default_projection_type = self.paramsCurrent[PROJECTION_TYPE]
        # MODIFIED 12/1/16 NEW:
        default_projection_type = self.paramClassDefaults[PROJECTION_TYPE]
        # MODIFIED 12/1/16 END

        # Instantiate each projection specification in the projection_list, and
        # - insure it is in self.afferents
        # - insure the output of its function is compatible with self.value
        for projection_spec in projection_list:

            # If there is more than one projection specified, construct messages for use in case of failure
            if len(projection_list) > 1:
                item_prefix_string = "Item {0} of projection list for {1}: ".\
                    format(projection_list.index(projection_spec)+1, state_name_string)
                item_suffix_string = ""

# FIX: FROM HERE TO BOTTOM OF METHOD SHOULD ALL BE HANDLED IN __init__() FOR PROJECTION_SPEC
            projection_object = None # flags whether projection object has been instantiated; doesn't store object
            projection_type = None   # stores type of projection to instantiate
            projection_params = {}

            # INSTANTIATE PROJECTION_SPEC
            # If projection_spec is a Projection object:
            # - call _check_projection_receiver() to check that receiver is self; if not, it:
            #     returns object with receiver reassigned to self if chosen by user
            #     else, returns new (default) PROJECTION_TYPE object with self as receiver
            #     note: in that case, projection will be in self.afferents list
            if isinstance(projection_spec, Projection_Base):
                if projection_spec.value is DEFERRED_INITIALIZATION:
                    from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
                    # from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection
                    from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
                    # if isinstance(projection_spec, (LearningProjection, GatingProjection, ControlProjection)):
                    if isinstance(projection_spec, (LearningProjection, ControlProjection)):
                        # Assign projection to parameterState
                        self.afferents.append(projection_spec)
                        projection_spec.init_args[kwReceiver] = self
                        # Skip any further initialization for now
                        #   (remainder will occur as part of deferred init for
                        #    LearningProjection, ControlProjection or GatingProjection)
                        continue
                    # Complete init for other projections (e.g., ControlProjection)
                    else:
                        # Assume init was deferred because receiver could not be determined previously
                        #  (e.g., specified in function arg for receiver object, or as standalone projection in script)
                        # Assign receiver to init_args and call _deferred_init for projection
                        projection_spec.init_args[kwReceiver] = self
                        projection_spec.init_args['name'] = self.owner.name+' '+self.name+' '+projection_spec.className
                        # FIX: REINSTATE:
                        # projection_spec.init_args['context'] = context
                        projection_spec._deferred_init()
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

            # If projection_spec is a dict:
            # - get projection_type
            # - get projection_params
            # Note: this gets projection_type but does NOT not instantiate projection; so,
            #       projection is NOT yet in self.afferents list
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
            # Note: this gets projection_type but does NOT instantiate the projection (that happens below),
            #       so projection is NOT yet in self.afferents list
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
            #       projection is NOT yet in self.afferents list
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
            # Note: this automatically assigns projection to self.afferents and
            #       to it's sender's efferents list:
            #           when a projection is instantiated, it assigns itself to:
            #               its receiver's .afferents attribute (in Projection._instantiate_receiver)
            #               its sender's .efferents list attribute (in Projection._instantiate_sender)
            if not projection_object:
                projection_spec = projection_type(receiver=self,
                                                  name=self.owner.name+' '+self.name+' '+projection_type.className,
                                                  # name=self.owner.name + ' '+projection_type.className,
                                                  params=projection_params,
                                                  context=context)

            # Check that output of projection's function (projection_spec.value is compatible with
            #    variable of the State to which it projects;  if it is not, raise exception:
            # The buck stops here; can't modify projection's function to accommodate the State,
            #    or there would be an unmanageable regress of reassigning projections,
            #    requiring reassignment or modification of sender outputStates, etc.

            # Initialization of projection is deferred
            if projection_spec.value is DEFERRED_INITIALIZATION:
                # Assign instantiated "stub" so it is found on deferred initialization pass (see Process)
                self.afferents.append(projection_spec)
                continue

            # Projection specification is valid, so assign projection to State's afferents list
            elif iscompatible(self.variable, projection_spec.value):
                # This is needed to avoid duplicates, since instantiation of projection (e.g., of ControlProjection)
                #    may have already called this method and assigned projection to self.afferents list
                if not projection_spec in self.afferents:
                    self.afferents.append(projection_spec)

            # Projection specification is not valid
            else:
                raise StateError("{}Output of function for {}{} ( ({})) is not compatible with value of {} ({})".
                                 format(item_prefix_string,
                                        default_string,
                                        projection_spec.name,
                                        projection_spec.value,
                                        item_suffix_string,
                                        self.value))

    def _instantiate_projection_from_state(self, projection_spec, receiver, context=None):
        """Instantiate projection from a state and assign it to self.efferents

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

        :param context: (str)
        :return state: (State)
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
        #       projection is NOT yet in self.afferents list
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
        #               its receiver's .afferents attribute (in Projection._instantiate_receiver)
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
                self.afferents.append(projection_spec)
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
                return (None, "not a Projection subclass")

# # FIX: NO LONGER USED;  SUPERCEDED BY value setter METHOD ABOVE.  INCOROPRATE VALIDATION THERE??
#     def add_observer_for_keypath(self, object, keypath):
#         self.observers[keypath].append(object)
#
# # IMPLEMENTATION NOTE:  USE DECORATOR TO MAKE SURE THIS IS CALLED WHENEVER state.value IS ASSIGNED
#     def set_value(self, new_value):
#         """Validate value, assign it, and update any observers
#
#         Uses valueClassDefault as the template for validating new_value
#         :param new_value:
#         :return:
#         """
#
#         # Validate new_value
#         if self.prefs.paramValidationPref:
#             if not isinstance(new_value, self.variableInstanceDefault):
#                 raise StateError("Value {0} of {1} must be of type {2}".
#                                      format(new_value, self.name, self.variableInstanceDefault))
#             # Check that each element is a number
#             for element in new_value:
#                 if not isinstance(element, numbers.Number):
#                     raise StateError("Item {0} ({1}) in value of {2} is not a number".
#                                          format(new_value.index(element), element, self.name))
#
#         old_value = self.value
#
#         # Set value
#         self.value = new_value
#
#         # Update observers
#         if self.observers:
#         # if len(self.observers[self.kpState]):
#             for observer in self.observers[self.kpState]:
#                 observer.observe_value_at_keypath(self.kpState, old_value, new_value)
#


    def update(self, params=None, time_scale=TimeScale.TRIAL, context=None):
        """Update each projection, combine them, and assign result to value

        Call update for each projection in self.afferents (passing specified params)
        Note: only update LearningSignals if context == LEARNING; otherwise, just get their value
        Call self.function (default: LinearCombination function) to combine their values
        Assign result to self.value

    Arguments:
    - context (str)

    :param context: (str)
    :return: None

    """

        #region GET STATE-SPECIFIC PARAM_SPECS
        try:
            # Get State params
            self.stateParams = params[self.paramsType]
        except (KeyError, TypeError):
            self.stateParams = None
        except (AttributeError):
            raise StateError("PROGRAM ERROR: paramsType not specified for {}".format(self.name))
        #endregion

        #region FLAG FORMAT OF INPUT
        if isinstance(self.value, numbers.Number):
            # Treat as single real value
            value_is_number = True
        else:
            # Treat as vector (list or np.array)
            value_is_number = False
        #endregion

        #region AGGREGATE INPUT FROM PROJECTION_SPECS

        #region Get type-specific params from PROJECTION_PARAMS
        mapping_params = merge_param_dicts(self.stateParams, MAPPING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        control_projection_params = merge_param_dicts(self.stateParams, CONTROL_PROJECTION_PARAMS, PROJECTION_PARAMS)
        learning_projection_params = merge_param_dicts(self.stateParams, LEARNING_PROJECTION_PARAMS, PROJECTION_PARAMS)
        #endregion

        #region For each projection: get its params, pass them to it, and get the projection's value
        projection_value_list = []

        from PsyNeuLink.Components.Process import ProcessInputState
        from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection

        for projection in self.afferents:

            if hasattr(projection, 'sender'):
                sender = projection.sender
            else:
                if self.verbosePref:
                    warnings.warn("{} to {} {} of {} ignored [has no sender]".format(projection.__class__.__name__,
                                                                                     self.name,
                                                                                     self.__class__.__name__,
                                                                                     self.owner.name))
                continue

            # Only update if sender has also executed in this round (i.e., has matching execution_id)
            if isinstance(self.owner, (Mechanism, Process)):
                if sender.owner._execution_id != self.owner._execution_id:
                    continue
            elif isinstance(self.owner, MappingProjection):
                if sender.owner._execution_id != self.owner.sender.owner._execution_id:
                    continue
            else:
                raise StateError("PROGRAM ERROR: Object ({}) of type {} has a {}, but this is only allowed for "
                                 "Mechanisms and MappingProjections".
                                 format(self.owner.name, self.owner.__class__.__name__, self.__class__.__name__,))

            # FIX: FOR EACH PROJECTION TO INPUT_STATE, CHECK IF SENDER IS FROM PROCESS INPUT OR TARGET INPUT
            # FIX: IF SO, ONLY INCLUDE IF THEY BELONG TO CURRENT PROCESS;
            if isinstance(sender, ProcessInputState):
                if not sender.owner in self.owner.processes.keys():
                    continue

            from PsyNeuLink.Components.Projections.TransmissiveProjections.MappingProjection import MappingProjection
            from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
            from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection

            # Merge with relevant projection type-specific params
            if isinstance(projection, MappingProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, mapping_params, )
            elif isinstance(projection, ControlProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, control_projection_params)
            elif isinstance(projection, LearningProjection):
                projection_params = merge_param_dicts(self.stateParams, projection.name, learning_projection_params)
            if not projection_params:
                projection_params = None

            # Update LearningSignals only if context == LEARNING;  otherwise, just get current value
            # Note: done here rather than in its own method in order to exploit parsing of params above
            if isinstance(projection, LearningProjection):
                if LEARNING in context:
                    projection_value = projection.execute(time_scale=time_scale,
                                                          params=projection_params,
                                                          context=context)
                else:
                    projection_value = projection.value

                TEST = True

            else:
                # Update all non-LearningProjections and get value
                projection_value = projection.execute(params=projection_params,
                                                      time_scale=time_scale,
                                                      context=context)

            # If this is initialization run and projection initialization has been deferred, pass
            if INITIALIZING in context and projection_value is DEFERRED_INITIALIZATION:
                continue
            # Add projection_value to list (for aggregation below)
            projection_value_list.append(projection_value)
        #endregion

        #region Aggregate projection values

        # If there were projections:
        if projection_value_list:

            try:
                # pass only function params
                function_params = self.stateParams[FUNCTION_PARAMS]
            except (KeyError, TypeError):
                function_params = None

            # Combine projection values
            combined_values = self.function(variable=projection_value_list,
                                            params=function_params,
                                            context=context)

            # If self.value is a number, convert combined_values back to number
            if value_is_number and combined_values:
                combined_values = combined_values[0]

        # There were no projections
        else:
            # mark combined_values as none, so that (after being assigned to self.value)
            #    it is ignored in execute method (i.e., not combined with baseValue)
            combined_values = None
        #endregion

        #region ASSIGN STATE VALUE
        context = context + kwAggregate + ' Projection Inputs'
        self.value = combined_values
        #endregion

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
    def baseValue(self):
        return self._baseValue

    @baseValue.setter
    def baseValue(self, value):
        self._baseValue = value

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, assignment):
        self._projections = assignment

    # @property
    # def afferents(self):
    #     return self._afferents
    #
    # @afferents.setter
    # def afferents(self, assignment):
    #     self._afferents = assignment

# **************************************************************************************

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
        states = ContentAddressableList(component_type=State_Base)

        # Instantiate state for entry in list or dict
        # Note: if state_entries is a list, state_spec is the item, and key is its index in the list
        for key, state_spec in state_entries if isinstance(state_entries, dict) else enumerate(state_entries):
            state_name = ""

            # State_entries is a dict, so use:
            # - entry key as state's name
            # - entry value as state_spec
            if isinstance(key, str):
                state_name = key
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
                # - key (index in list) to get corresponding value from constraint_value as state_spec
                # - assign same item of constraint_value as the constraint
                if isinstance(state_spec, str):
                    # Use state_spec as state_name if it has not yet been used
                    if not state_name is state_spec and not state_name in states:
                        state_name = state_spec
                    # Add index suffix to name if it is already been used
                    # Note: avoid any chance of duplicate names (will cause current state to overwrite previous one)
                    else:
                        state_name = state_spec + '_' + str(key)
                    state_spec = constraint_value[key]
                    state_constraint_value = constraint_value[key]

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
                        state_name = 'Default_' + state_param_identifier[:-1] + "-" + str(key+1)
                    # MODIFIED 12/11/16 NEW:
                    # If it is an "exposed" number, make it a 1d np.array
                    if isinstance(state_spec, numbers.Number):
                        state_spec = np.atleast_1d(state_spec)
                    # MODIFIED 12/11/16 END

                    state_constraint_value = constraint_value[key]

            state = _instantiate_state(owner=owner,
                                                state_type=state_type,
                                                state_name=state_name,
                                                state_spec=state_spec,
                                                state_params=state_params,
                                                constraint_value=state_constraint_value,
                                                constraint_value_name=constraint_value_name,
                                                context=context)

            # Get name of state, and use as key to assign to states OrderedDict
            states[state.name] = state
        return states

    else:
        # This shouldn't happen, as MECHANISM<*>States was validated to be one of the above in _validate_params
        raise StateError("PROGRAM ERROR: {} for {} is not a recognized \'{}\' specification for {}; "
                         "it should have been converted to a list in Mechanism._validate_params)".
                         format(state_entries, owner.name, state_param_identifier, owner.__class__.__name__))


def _instantiate_state(owner,                   # Object to which state will belong
                      state_type,              # State subclass
                      state_name,              # Name used to refer to subclass in prompts
                      state_spec,              # State subclass, object, spec dict or value
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
    + Projection object:
        assign constraint_value to value
        assign projection to STATE_PARAMS{STATE_PROJECTIONS:<projection>}
    + Projection class (or keyword string constant for one):
        assign constraint_value to value
        assign projection class spec to STATE_PARAMS{STATE_PROJECTIONS:<projection>}
    + specification dict for State (see XXX for context):
        check compatibility of STATE_VALUE with constraint_value
    + 2-item tuple: (only allowed for ParameterState spec)
        assign first item to state_spec
            if it is a string:
                test if it is a keyword and get its value by calling keyword method of owner's execute method
                otherwise, return None (suppress assignment of parameterState)
        assign second item to STATE_PARAMS{STATE_PROJECTIONS:<projection>}
    + value:
        if it is a string:
            test if it is a keyword and get its value by calling keyword method of owner's execute method
            otherwise, return None (suppress assignment of parameterState)
        check compatibility with constraint_value
    If any of the conditions above fail:
        a default State of specified type is instantiated using constraint_value as value

    If state_params is specified, include with instantiation of state

    :param context: (str)
    :return state: (State)
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

    # Used locally to report type of specification for State
    #  if value is not compatible with constraint_value
    spec_type = None

    # PARSE constraint_value
    constraint_dict = _parse_state_spec(owner=owner,
                                        state_type=state_type,
                                        state_spec=constraint_value,
                                        value=None,
                                        params=None)
    constraint_value = constraint_dict[VALUE]

    # PARSE state_spec using constraint_value as default for value
    state_spec = _parse_state_spec(owner=owner,
                                   state_type=state_type,
                                   state_spec=state_spec,
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
            state = constraint_value
            spec_type = state_name

    # Otherwise, state_spec should now be a state specification dict
    state_variable = state_spec[VARIABLE]
    state_value = state_spec[VALUE]

    if not iscompatible(state_variable, constraint_value):
        if owner.prefs.verbosePref:
            print("{} is not compatible with constraint value ({}) specified for {} of {};  latter will be used".
                  format(VARIABLE, constraint_value, state_type, owner.name))
        state_variable = constraint_value
        spec_type = VARIABLE


    # ----------------------------------------------------------------------------------------------
    # XXX MAKE SURE RELEVANT THINGS BELOW ARE GETTING DONE IN parse_state_spec

    # IMPLEMENTATION NOTE:  CONSOLIDATE ALL THE PROJECTION-RELATED STUFF BELOW:

    # FIX: MOVE THIS TO METHOD THAT CAN ALSO BE CALLED BY Function._instantiate_function()
    PARAM_SPEC = 0
    PROJECTION_SPEC = 1
    #region
    # 2-item tuple (param_value, projection_spec) [convenience notation for projection to parameterState]:
    # If state_type is ParameterState, and state_spec is a tuple with two items, the second of which is a
    #    projection specification (MAPPING_PROJECTION, CONTROL_PROJECTION, LEARNING_PROJECTION, CONTROL or LEARNING,
    #    or class ref to one of those), allow it
    # - check that first item matches constraint_value and assign to state_variable
    # - assign second item as projection to STATE_PARAMS:{STATE_PROJECTIONS:<projection>}
    # Note: validity of projection specification or compatibility of projection's variable or function output
    #       with state value is handled in State._instantiate_projections_to_state
    # IMPLEMENTATION NOTE:
    #    - need to do some checking on state_spec[PROJECTION_SPEC] to see if it is a projection
    #      since it could just be a numeric tuple used for the variable of a state;
    #      could check string against ProjectionRegistry (as done in _parse_projection_ref in State)
    if (isinstance(state_spec, tuple) and len(state_spec) is 2 and
            (state_spec[PROJECTION_SPEC] in {MAPPING_PROJECTION,
                                             CONTROL_PROJECTION,
                                             LEARNING_PROJECTION,
                                             CONTROL,
                                             LEARNING} or
                 isinstance(state_spec[PROJECTION_SPEC], Projection) or
                 (inspect.isclass(state_spec[PROJECTION_SPEC]) and issubclass(state_spec[PROJECTION_SPEC], Projection)))
        ):
        from PsyNeuLink.Components.States.ParameterState import ParameterState
        if not issubclass(state_type, ParameterState):
            raise StateError("Tuple with projection spec ({0}) not permitted as specification "
                                      "for {1} (in {2})".format(state_spec, state_type.__name__, owner.name))
        state_variable =  state_spec[PARAM_SPEC]
        projection_to_state = state_spec[PROJECTION_SPEC]
        # If it is a string, assume it is a keyword and try to resolve to value
        if isinstance(state_variable, str):
            # Evaluate keyword to get template for state_variable
            state_variable = get_param_value_for_keyword(owner, state_variable)
            if state_variable is None:
                return None
        # If it is a function, call to resolve to value
        if isinstance(state_variable, function_type):
            state_variable = get_param_value_for_function(owner, state_variable)
            if state_variable is None:
                return None

        constraint_value = state_variable
        state_params.update({STATE_PROJECTIONS:[projection_to_state]})
    #endregion

    #region Keyword String
    if isinstance(state_spec, str):
        if state_spec in projection_keywords:
            state_spec = state_variable
            state_variable = constraint_value
        else:
            state_spec = get_param_value_for_keyword(owner, state_spec)
            if state_spec is None:
                return None
    #endregion

    #region Function
    if isinstance(state_spec, function_type):
        state_spec = get_param_value_for_function(owner, state_spec)
        if state_spec is None:
            return None
    #endregion

    # Projection
    # If state_spec is a Projection object or Projection class
    # - assign constraint_value to state_variable
    # - assign tuple[1] to STATE_PARAMS:{STATE_PROJECTIONS:[<projection>]}
    # Note: validity of projection specification or compatibility of projection's variable or function output
    #       with state value is handled in State._instantiate_projections_to_state
    try:
        # Projection class
        issubclass(state_spec, Projection)
        state_variable =  constraint_value
        state_params.update({STATE_PROJECTIONS:[state_spec]})
    except TypeError:
        # Projection object
        if isinstance(state_spec, (Projection, str)):
            state_variable =  constraint_value
            state_params.update({STATE_PROJECTIONS:[state_spec]})

    # Do one last check for compatibility of value with constraint_value (in case state_spec was a value)
    if not iscompatible(state_variable, constraint_value):
        state_variable = constraint_value
        spec_type = state_name

    # WARN IF DEFAULT (constraint_value) HAS BEEN ASSIGNED
    # spec_type has been assigned, so iscompatible() failed above and constraint value was assigned
    if spec_type:
        if owner.prefs.verbosePref:
            print("Value ({0}) of {1} (type: {2}) is not compatible with {3} ({4}) of {6};"
                  " default {4} will be created using {5}".
                  format(state_variable,
                         state_name,
                         spec_type,
                         constraint_value_name,
                         constraint_value.__class__.__name__,
                         constraint_value,
                         owner.__class__.__name__))

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

    #  Convert constraint_value to np.array to match state_variable (which, as output of function, will be np.array)
    constraint_value = convert_to_np_array(constraint_value,1)

    # Implement default State
    state = state_type(owner=owner,
                       reference_value=constraint_value,
                       variable=state_variable,
                       name=state_name,
                       params=state_params,
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


def _parse_state_spec(owner,
                      state_type,
                      state_spec,
                      name=None,
                      variable=None,
                      value=None,
                      projections=None,
                      modulatory_projections=None,
                      params=None):
    """Return either state object or state specification dict for state_spec
    
    If state_spec resolves to a state object, return that;  
        otherwise, return state specification dictionary using arguments provided as defaults
    Warn if variable is assigned is assigned the default value, and verbosePref is set on owner. 
    
    """

    # State object:
    # - check that it is of the specified type and, if so, return it
    if isinstance(state_spec, State):
        if state_spec is state_type:
            return state_spec
        else:
            raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
                             "class of state being instantiated ({})".format(state_spec, state_type))

    # Mechanism object:
    # - call owner to return primary state of specified type
    if isinstance(state_spec, Mechanism):
        return owner._get_primary_state(state_type)

    # If variable is specified in state_params, use that
    if params is not None and VARIABLE in params and params[VARIABLE] is not None:
        variable = params[VARIABLE]

    # MODIFIED 5/17/17 FROM _instantiate_state: ----------------------------------------------------------------

    # # Projection class, object, or keyword: set to paramClassDefaults (of owner or owner's function)
    # from PsyNeuLink.Components.Projections.Projection import projection_keywords
    # if ((isinstance(state_spec, str) and state_spec in projection_keywords) or
    #         isinstance(state_spec, Projection) or
    #         (inspect.isclass(state_spec) and issubclass(state_spec, Projection))):
    #     from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
    #     from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
    #     from PsyNeuLink.Components.Projections.ModulatoryProjections.GatingProjection import GatingProjection
    #     # Disallow if it is not a LearningProjection, ControlProjection or GatingProjection
    #     if (state_spec in {LEARNING_PROJECTION, CONTROL_PROJECTION, GATING_PROJECTION} or
    #                 isinstance(state_spec, (LearningProjection, ControlProjection, GatingProjection)) or
    #                 (inspect.isclass(state_spec) and
    #                      issubclass(state_spec, (LearningProjection, ControlProjection, GatingProjection)))
    #             ):
    #         try:
    #             constraint_value = owner.paramClassDefaults[state_name]
    #         # If parameter is not for owner itself, try owner's function
    #         except KeyError:
    #             constraint_value = owner.user_params[FUNCTION].paramClassDefaults[state_name]

    # Create default dict for return
    state_dict = {NAME: name,
                  VARIABLE: variable,
                  VALUE: value,
                  STATE_PROJECTIONS: projections,
                  MODULATORY_PROJECTIONS: modulatory_projections,
                  PARAMS: params}

    # State class
    if inspect.isclass(state_spec) and issubclass(state_spec, State):
        if state_spec is state_type:
            state_dict[VARIABLE] = state_spec.variableClassDefault
        else:
            raise StateError("PROGRAM ERROR: state_spec specified as class ({}) that does not match "
                             "class of state being instantiated ({})".format(state_spec, state_type))

    # # State object
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
    elif isinstance(state_spec, dict):
        # Dict has a single entry of the form {STATE_NAME:SPECIFICATION_DICT},
        #     so assign STATE_NAME as name, and return parsed SPECIFICATION_DICT
        if len(state_spec) == 1 and list(state_spec.keys())[0] not in state_keywords:
            name, state_spec = list(state_spec.items())[0]
            state_dict = _parse_state_spec(owner=owner,
                                           state_type=state_type,
                                           state_spec=state_spec,
                                           name=name,
                                           variable=variable,
                                           value=value,
                                           projections=projections,
                                           modulatory_projections=modulatory_projections,
                                           params=params)
            # Use name specified as key in state_spec (rather than one in SPEFICATION_DICT if specified):
            state_dict.update({NAME:name})

        else:
            # Warn if VARIABLE was not in dict
            if not VARIABLE in state_spec and owner.prefs.verbosePref:
                print("{} missing from specification dict for {} of {};  default ({}) will be used".
                      format(VARIABLE, state_type, owner.name, state_spec))
            # FIX: DO THIS: Put entries (except NAME, VARIABLE AND VALUE) in params
            # Move all params-relevant entries from state_spec into params
            for spec in [param_spec for param_spec in state_spec.copy()
                         if not param_spec in {NAME, VARIABLE, PREFS_ARG, CONTEXT}]:
                # if spec in {NAME, VARIABLE, PREFS_ARG, CONTEXT}:
                #     continue
                if not params:
                    params = {}
                params[spec] = state_spec[spec]
                del state_spec[spec]
            state_dict.update(state_spec)


    # Tuple
    elif isinstance(state_spec, tuple):
        # Parse state_spec in first item of tuple
        state_dict = _parse_state_spec(owner=owner,
                                       state_type=state_type,
                                       state_spec=state_spec[0],
                                       name=name,
                                       variable=variable,
                                       value=value,
                                       projections=projections,
                                       params=params)
        # Add projection_spec from second item in tuple and return dict
        state_dict.update({modulatory_projections:state_spec[1]})

    # string
    elif isinstance(state_spec, str):
        # Test whether it is a keyword for the owner, in which case it should resolve to a value
        if state_spec in projection_keywords:
            # state_spec = state_variable
            # state_variable = constraint_value
            state_dict[VARIABLE]=value
        ??else:
        spec = get_param_value_for_keyword(owner, state_spec)
        # A value was returned, so use as variable
        if spec is not None:
            state_dict[VARIABLE] = spec
            if owner.prefs.verbosePref:
                print("{} not specified for {} of {};  default ({}) will be used".
                      format(VARIABLE, state_type, owner.name, constraint_value))
        # It is not a keyword, so treat string as the name for the state
        else:
            state_dict[NAME] = spec

    # function; try to resolve to a value, otherwise return None to suppress instantiation of state
    elif isinstance(state_spec, function_type):
        state_dict[VALUE] = get_param_value_for_function(owner, state_spec)
        if state_dict[VALUE] is None:
            # return None
            raise StateError("PROGRAM ERROR: state_spec for {} of {} is a function ({}), "
                             "but it failed to return a value".format(state_type,owner.name, state_spec))

    # value, so use as variable of input_state
    elif is_value_spec(state_spec):
        state_dict[VARIABLE] = state_spec

    elif state_spec is None:
        # pass
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is None".format(state_type,owner.name))

    else:
        if name and hasattr(owner, name):
            owner_name = owner.name
        else:
            owner_name = owner.__class__.__name__
        raise StateError("PROGRAM ERROR: state_spec for {} of {} is an unrecognized specification ({})".
                         format(state_type, owner.name, state_spec))

    return state_dict

