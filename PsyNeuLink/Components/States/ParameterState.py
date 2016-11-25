# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  ParameterState ******************************************************

"""
**[DOCUMENTATION STILL UNDER CONSTRUCTION]**

Overview
--------

A parameterState belongs to either a mechanism or a projection, and accepts one or more ControlProjections and/or
LearningProjections that modify the parameters of its owner's ``function``.   A list of the projections received by
a parameterState is kept in its ``receivesFromProjections`` attribute.  It's ``function`` combines the values of
these inputs, and uses the result to modify the value of the ``function`` parameter for which it is responsible.

.. _ParameterState_Creation:

Creating a ParameterState
-------------------------

A parameterState can be created by calling its constructor, but in general this is not necessary or advisable, as
parameterStates are created automatically when the object to which they belong (a mechanism or a projection) is
created.  One parameterState is created for each parameter of the object's ``function``.  Each parameterState is
created using the specification for the corresponding parameter of the ``function``, as described below.

.. _ParameterState_Specifying_Parameters:

Specifying Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

COMMENT:
  XXXX EXPLAIN THAT PARAMETER SPECIFICATION CREATES ParameterState FOR IT
  XXXX EXPLAIN baseValue (reference figure)

When a function is specified for an object, its parameters can be assigned in two ways:

* in **constructor** for the function, where that is used for the ``function`` argument of the object,
  as in the example below::

    my_mechanism = SomeMechanism(function=SomeFunction(SOME_PARAM=1, SOME_OTHER_PARAM=2)

* or in the :keyword:`FUNCTION_PARAMS` entry of a parameter dictionary used for the ``params`` argument of the object,
  as in the example below::

    my_mechanism = SomeMechanism(function=SomeFunction
                                 params={FUNCTION_PARAMS:{SOME_PARAM=1, SOME_OTHER_PARAM=2}})


When the parameterStates for the function are created, the values specified in the ``function`` argument

Why would you do the latter???:  cass in wich

The parameters of a :keyword:`function` can be specified in two ways:  in a constructor for the function, where it
use as a ``function`` argument


 (where it is specified as a ``function`` argument;  or in the
:keyword:`FUNCTION_PARAMS` entry of a parameter dictionary



used in
the ``function`` argument of the mechanism or projection;  or in the :keyword:`FUNCTION_PARAMS`
entry of a parameter dictionary used for the ``params`` argument of the mechanism or projection [LINK].  The value
must be a dictionary, the enties of which have a key
that is the name of a function parameter, and the value of which is one of the following:

    COMMENT:
       XXXX VERIFY THAT THIS IS TRUE:
    COMMENT
    * A **value**.  This must be a valid the value of the parameter.  The creates a default parameterState and
      assigns the value as its ``baseValue``. [LINK]
    ..
    * An existing **parameterState** object or the name of one.  It's name must be the name of a parameter of the
      owner's ``function``, and its value must be a valid for that parameter.  This capability is provided
      for generality and potential future use, but its use is not advised.
    ..
    COMMENT:
       XXXX VERIFY THAT THIS IS TRUE:
    COMMENT
    * A **Projection subclass**. This creates a default parameterState, assigns the parameter's default value as
      the parameterState's ``baseValue``, and creates and assigns a projection to it of the specified type.
      The ``sender`` for the projection is specified by XXXX?????
    ..
    * A **Projection object** or **projection specification dictionary** [LINK].  This creates a default
      parameterState, and assigns the ``value`` of projection as the parameterState's ``baseValue``.
      This must be valid for the parameter.
    ..
    + ParamValueProjection tuple:
        value will be used as variable to instantiate a default ParameterState
        projection will be assigned as projection to ParameterState
    ..
    * A :any:`ParamValueProjection` or 2-item (value, projection) tuple.  This creates a default parameterState using
      the ``value`` item as its ``baesValue``. If the projection item is an existing projection or a constructor for
      one, it is assigned the parameter as its ``receiver``.  If the projection item is the name of a Projection
      subclass, a default projection of the specified type is created, and assigned the parameterState as its
      ``receiver``.

    .. note::
       In all cases, the resulting value of the parameterState must be compatible (that is, have the same number and
       type of elements) as the parameter of the ``function`` with which it is associated.

  COMMENT:
    XXXX ??MOVE THIS TO ControlSignal
  COMENT
  *Assigning a ControlProjection*

  A control signal can be assigned to a parameter, wherever the parameter value is specified, by using a tuple with
  two items. The first item is the value of the parameter, and the second item is either :keyword:`CONTROL_PROJECTION`,
  the name of the ControlProjection class, or a call to its constructor.  In the following example, a mechanism is
  created with a function that has three parameters::

    my_mechanism = SomeMechanism(function=SomeFunction(param_1=1.0,
                                                       param_2=(0.5, ControlProjection))
                                                       param_3=(36, ControlProjection(function=Logistic)))

  The first parameter of the mechanism's function is assigned a value directly, the second parameter is assigned a
  ControlProjection, and the third is assigned a
  :ref:`ControlProjection with a specified function <ControlProjection_Structure>`.

The value of function parameters can also be modified using a runtime parameters dictionary where a mechanism is
specified in a process ``pathway`` (see XXX), or in the ``params`` argument  of a mechanism's ``execute`` or ``run``
methods (see :ref:`Mechanism_Runtime_Parameters`).  The figure below shows how these factors are combined by the
parameterState to determine the paramter value for a function.

    **Role of ParameterStates in Controlling the Parameter Value of a Function**

    .. figure:: _static/ParameterState_fig.*
       :alt: ParameterState
       :scale: 75 %

       ..

       +--------------+--------------------------------------------------------------------+
       | Component    | Impact on Parameter Value                                          |
       +==============+====================================================================+
       | Brown (A)    | baseValue of drift rate parameter of DDM function                  |
       +--------------+--------------------------------------------------------------------+
       | Purple (B)   | runtime specification of drift rate parameter                      |
       +--------------+--------------------------------------------------------------------+
       | Red (C)      | runtime parameter influences ControlProjection-modulated baseValue |
       +--------------+--------------------------------------------------------------------+
       | Green (D)    | combined controlSignals modulate baseValue                         |
       +--------------+--------------------------------------------------------------------+
       | Blue (E)     | parameterState function combines ControlProjection                 |
       +--------------+--------------------------------------------------------------------+




parameter_modulation_operation:  ModulationOperation - list values and their meaning

"""

from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.State import _instantiate_state
from PsyNeuLink.Components.Functions.Function import *

# class ParameterStateLog(IntEnum):
#     NONE            = 0
#     TIME_STAMP      = 1 << 0
#     ALL = TIME_STAMP
#     DEFAULTS = NONE


class ParameterStateError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


# class ParameterState_Base(State_Base):
class ParameterState(State_Base):
    """Implement subclass type of State that represents parameter value for function of a Mechanism

    Definition for ParameterState componentType in State category of Function class

    Description:
        The ParameterState class is a componentType in the State category of Function,
        Its FUNCTION executes the projections that it receives and updates the ParameterState's value

    Instantiation:
        - ParameterStates can be instantiated in one of two ways:
            - directly: requires explicit specification of its value and owner;
                - specification of value can be any of the forms allowed for specifying a State
                    (default value will be inferred from anything other than a value or ParamValueProjection tuple)
                - owner must be a reference to a Mechanism object, or DefaultProcessingMechanism_Base will be used
            - as part of the instantiation of a mechanism:
                - the mechanism for which it is being instantiated will automatically be used as the owner
                - the value of the owner's param for which the ParameterState is being instantiated
                    will be used as its variable (that must also be compatible with its self.value)
        - self.variable must be compatible with self.value (enforced in _validate_variable)
            note: although it may receive multiple projections, the output of each must conform to self.variable,
                  as they will be combined to produce a single value that must be compatible with self.variable
        - self.function (= params[FUNCTION]) must be Function.LinearCombination (enforced in _validate_params)

    Execution:
        - get ParameterStateParams
        - pass params to super, which aggregates inputs from projections
        - combine input from projections (processed in super) with baseValue using paramModulationOperation
        - combine result with value specified at runtime in PARAMETER_STATE_PARAMS
        - assign result to self.value

    StateRegistry:
        All ParameterStates are registered in StateRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        ParameterStates can be named explicitly (using the name argument). If this argument is omitted,
         it will be assigned "ParameterState" with a hyphenated, indexed suffix ('ParameterState-n')

    Parameters:
        The default for FUNCTION is LinearCombination using kwAritmentic.Operation.PRODUCT:
           self.value is multiplied by  the output of each of the  projections it receives
               (generally ControlProjections)
# IMPLEMENTATION NOTE:  *** CONFIRM THAT THIS IS TRUE:
        FUNCTION can be set to another function, so long as it has type kwLinearCombinationFunction
        The parameters of FUNCTION can be set:
            - by including them at initialization (param[FUNCTION] = <function>(sender, params)
            - calling the adjust method, which changes their default values (param[FUNCTION].adjust(params)
            - at run time, which changes their values for just for that call (self.execute(sender, params)
    Class attributes:
        + componentType (str) = kwMechanisParameterState
        + classPreferences
        + classPreferenceLevel (PreferenceLevel.Type)
        + paramClassDefaults (dict)
            + FUNCTION (LinearCombination)
            + FUNCTION_PARAMS  (Operation.PRODUCT)
            + PROJECTION_TYPE (CONTROL_PROJECTION)
            + PARAMETER_MODULATION_OPERATION   (ModulationOperation.MULTIPLY)
        + paramNames (dict)
    Class methods:
        _instantiate_function: insures that function is ARITHMETIC) (default: Operation.PRODUCT)
        update_state: updates self.value from projections, baseValue and runtime in PARAMETER_STATE_PARAMS

    Instance attributes:
        + paramInstanceDefaults (dict) - defaults for instance (created and validated in Components init)
        + params (dict) - set currently in effect
        + paramNames (list) - list of keys for the params dictionary
        + owner (Mechanism)
        + value (value)
        + params (dict)
        + baseValue (value)
        + projections (list)
        + modulationOperation (ModulationOperation)
        + name (str)
        + prefs (dict)

    Instance methods:
        none
    """

    #region CLASS ATTRIBUTES

    componentType = kwParameterState
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
                 value=None,
                 function=LinearCombination(operation=PRODUCT),
                 parameter_modulation_operation=ModulationOperation.MULTIPLY,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                 parameter_modulation_operation=parameter_modulation_operation,
                                                 params=params)

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        super(ParameterState, self).__init__(owner,
                                             value=value,
                                             params=params,
                                             name=name,
                                             prefs=prefs,
                                             context=self)

        self.modulationOperation = self.paramsCurrent[PARAMETER_MODULATION_OPERATION]

    def _validate_params(self, request_set, target_set=NotImplemented, context=None):
        """Insure that parameterState (as identified by its name) is for a valid parameter of its owner's function
        """
        if not self.name in self.owner.function_params.keys():
            raise ParameterStateError("Name of requested parameterState ({}) does not refer to a valid parameter "
                                      "of the function ({}) of its owner ({})".
                                      format(self.name,
                                             # self.owner.function_object.__class__.__name__,
                                             self.owner.function_object.componentName,
                                             self.owner.name))

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

        # If parameterState is for a matrix of a MappingProjection,
        #     its parameter_modulation_operation should be SUM (rather than PRODUCT)
        #         so that weight changes (e.g., from a learningSignals) are added rather than multiplied
        if self.name == MATRIX:
            # IMPLEMENT / TEST: ZZZ 10/20/16 THIS SHOULD BE ABLE TO REPLACE SPECIFICATION IN LEARNING PROJECTION
            self.params[PARAMETER_MODULATION_OPERATION] = ModulationOperation.ADD

        super()._instantiate_function(context=context)

        # Insure that function is LinearCombination
        if not isinstance(self.function.__self__, (LinearCombination)):
            raise StateError("Function {0} for {1} of {2} must be of LinearCombination type".
                                 format(self.function.__self__.componentName, FUNCTION, self.name))

        # # Insure that output of function (self.value) is compatible with relevant parameter value
        if not iscompatible(self.value, self.reference_value):
            raise ParameterStateError("Value ({0}) of {1} for {2} mechanism is not compatible with "
                                           "the variable ({3}) of its function".
                                           format(self.value,
                                                  self.name,
                                                  self.owner.name,
                                                  self.owner.variable))


    def update(self, params=NotImplemented, time_scale=TimeScale.TRIAL, context=None):
        """Parse params for parameterState params and XXX ***

# DOCUMENTATION:  MORE HERE:
        - get ParameterStateParams
        - pass params to super, which aggregates inputs from projections
        - combine input from projections (processed in super) with baseValue using paramModulationOperation
        - combine result with value specified at runtime in PARAMETER_STATE_PARAMS
        - assign result to self.value

        :param params:
        :param time_scale:
        :param context:
        :return:
        """

        super().update(params=params,
                       time_scale=time_scale,
                       context=context)

        #region COMBINE PROJECTIONS INPUT WITH BASE PARAM VALUE
        try:
            # Check whether modulationOperation has been specified at runtime
            self.modulationOperation = self.stateParams[PARAMETER_MODULATION_OPERATION]
        except (KeyError, TypeError):
            # If not, try to get from params (possibly passed from projection to ParameterState)
            try:
                self.modulationOperation = params[PARAMETER_MODULATION_OPERATION]
            except (KeyError, TypeError):
                pass
            # If not, ignore (leave self.modulationOperation assigned to previous value)
            pass

        # If self.value has not been set, assign to baseValue
        if self.value is None:
            if not context:
                context = kwAssign + ' Base Value'
            else:
                context = context + kwAssign + ' Base Value'
            self.value = self.baseValue

        # Otherwise, combine param's value with baseValue using modulatonOperation
        else:
            if not context:
                context = kwAssign + ' Modulated Value'
            else:
                context = context + kwAssign + ' Modulated Value'
            self.value = self.modulationOperation(self.baseValue, self.value)
        #endregion

        #region APPLY RUNTIME PARAM VALUES
        # If there are not any runtime params, or functionRuntimeParamsPref is disabled, return
        if (self.stateParams is NotImplemented or
                    self.prefs.functionRuntimeParamsPref is ModulationOperation.DISABLED):
            return

        # Assign class-level pref as default operation
        default_operation = self.prefs.functionRuntimeParamsPref

        try:
            value, operation = self.stateParams[self.name]

        except KeyError:
            # No runtime param for this param state
            return

        except TypeError:
            # If single ("exposed") value, use default_operation (class-level functionRuntimeParamsPref)
            self.value = default_operation(self.stateParams[self.name], self.value)
        else:
            # If tuple, use param-specific ModulationOperation as operation
            self.value = operation(value, self.value)

            # Assign class-level pref as default operation
            default_operation = self.prefs.functionRuntimeParamsPref
        #endregion

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):
        self._value = assignment

def _instantiate_parameter_states(owner, context=None):
    """Call _instantiate_state_list() to instantiate ParameterStates for subclass' function

    Instantiate parameter states for params specified in FUNCTION_PARAMS unless PARAMETER_STATES == False
    Use constraints (for compatibility checking) from paramsCurrent (inherited from paramClassDefaults)

    :param context:
    :return:
    """

    owner.parameterStates = {}

    try:
        function_param_specs = owner.paramsCurrent[FUNCTION_PARAMS]
    except KeyError:
        # No need to warn, as that already occurred in _validate_params (above)
        return
    else:
        try:
            no_parameter_states = not owner.params[PARAMETER_STATES]
        except KeyError:
            # PARAMETER_STATES not specified, so continue
            pass
        else:
            # PARAMETER_STATES was set to False, so do not instantiate any parameterStates
            if no_parameter_states:
                return
            # TBI / IMPLEMENT: use specs to implement paramterStates below
            # Notes:
            # * functionParams are still available in paramsCurrent;
            # # just no parameterStates instantiated for them.

        # Instantiate parameterState for each param in functionParams, using its value as the state_spec
        for param_name, param_value in function_param_specs.items():

            state = _instantiate_state(owner=owner,
                                      state_type=ParameterState,
                                      state_name=param_name,
                                      state_spec=param_value,
                                      state_params=None,
                                      constraint_value=param_value,
                                      constraint_value_name=param_name,
                                      context=context)
            if state:
                owner.parameterStates[param_name] = state
