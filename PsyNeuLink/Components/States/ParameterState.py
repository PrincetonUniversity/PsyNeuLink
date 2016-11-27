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

COMMENT:
  OLD VERSION
A parameterState belongs to either a mechanism or a MappingProjection, and is used to represent and possibly modify
the value of a parameter of it's owner's ``function``. It can receive one or more ControlProjections and/or
LearningProjections that modify that parameter.   A list of the projections received by a parameterState is kept in
its ``receivesFromProjections`` attribute.  It's ``function`` combines the values ofthese inputs, and uses the result
to modify the value of the parameter of its owner's ``funtion`` for which it is responsible.
COMMENT

A parameterState belongs to either a mechanism or a MappingProjection, and is used to represent and possibly modify
the value of a parameter of its owner or it owner's function. It can receive one or more ControlProjections and/or
LearningProjections that modify that parameter.   A list of the projections received by a parameterState is kept in
its ``receivesFromProjections`` attribute.  It's ``function`` combines the values of these inputs, and uses the result
to modify the value of the parameter for which it is responsible.


.. _ParameterState_Creation:

Creating a ParameterState
-------------------------

COMMENT:
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

OLD VERSION:
A parameterState can be created by calling its constructor, but in general this is not necessary or advisable, as
parameterStates are created automatically when the mechanism or projection to which they belong is created.  One
parameterState is created for each parameter of the object's ``function``.  Each parameterState is
created using the specification of the parameter for which it is responsible, as described below.

COMMENT

A parameterState can be created by calling its constructor, but in general this is not necessary or advisable, as
parameterStates are created automatically when the mechanism or projection to which they belong is created.  One
parameterState is created for each parameter of the object and its ``function``.  Each parameterState is
created using the specification of the parameter for which it is responsible, as described below.

.. _ParameterState_Specifying_Parameters:

Specifying Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter of a ``function`` can be specified in one of three ways:  in the constructor for its ``function``;  in
the :keyword:`FUNCTION_PARAMS` entry of a parameter dictionary for the object to which the ``function`` belongs; or,
in some cases, as an argument in the constructor for the object to which the ``function`` belongs (see
:ref:`Component_Specifying_Functions_and_Parameters` for details).  The items specified for the parameter are used
to create its ParameterState.  The value specified (either explicitly, or by default) is assigned to the
parameterState's ``baseValue`` attribute, and any projection specified is assigned to its ``receiveFromProjections``
attribute.  The ``baseValue`` is assigned as the value of the parameter for ``function`` when it executes,
after it is modified by any ControlProjections or LearningProjections received by the ParameterState (see
:ref:`Execution` for details).  The specification of a parameter can take any of the following forms:

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
    * A **Projection subclass**.  This creates a default parameterState, assigns the parameter's default value as
      the parameterState's ``baseValue``, and creates and assigns a projection to it of the specified type.
      The projection must be a ControlProjection or LearningProjection.
    ..
    * A **Projection object** or **projection specification dictionary** [LINK].  This creates a default
      parameterState, assigns the ``value`` of projection as the parameterState's ``baseValue``, and assigns the
      parameter state as the ``receiver`` for the projection.  The projection must be a ControlProjection or
      LearningProjection, and its value must be a valid one for the parameter.
    ..
    * A :any:`ParamValueProjection` or 2-item (value, projection) tuple.  This creates a default parameterState using
      the ``value`` (1st) item of the tuple as its ``baesValue``. If the ``projection`` (2nd) item of the tuple is an
      existing projection or a constructor for one, it is assigned the parameter as its ``receiver``, and the
      projection is assigned to the parameterState's ``receivesFromProjections`` attribute. If the projection item
      is the name of a Projection subclass, a default projection of the specified type is created, and assigned the
      parameterState as its ``receiver``.  In either case, the projection must be a ControlProjection or
      LearningProjection, and its value must be a valid one for the parameter.

Examples
~~~~~~~~

In the following example, a mechanism is created with a function that has four parameters,
each of which is specified using a different format::

    my_mechanism = SomeMechanism(function=SomeFunction(param_a=1.0,
                                                       param_b=(0.5, ControlProjection),
                                                       param_c=(36, ControlProjection(function=Logistic),
                                                       param_d=ControlProjection)))

The first parameter of the mechanism's function (``param_a``) is assigned a value directly; the second (``param_b``) is
assigned a value and a ControlProjection; the third (``param_c``) is assigned a value and a :ref:`ControlProjection
with a specified function  <ControlProjection_Structure>`; and the fourth (``param_d``) is assigned just a
ControlProjection (the default vaue for the parameter will be used).

In this example, a MappingProjection is created in which its ``matrix`` parameter is assigned a
a random weight matrix (using a :ref:`matrix keyword <Matrix_Keywords>`) and :doc:`LearningProjection`::

    my_mapping_projection = MappingProjection(sender=my_input_mechanism,
                                              receiver=my_output_mechanism,
                                              matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningProjection))

.. note::
   the ``matrix`` parameter belongs to the MappingProjection's ``function``;  however, since it has only one
   standard function, its arguments are available in the constructor for the projection
   (see :ref:`Component_Specifying_Functions_and_Parameters` for a more detailed explanation).


.. _ParameterState_Structure:

Structure
---------

function vs. parameter_modulation_operation
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


Every parameterState is owned by a :doc:`mechanism <Mechanism>` or :doc:`MappingProjection`. It can receive one or more
:ref:`ControlProjections <ControlProjection>` or :ref:`LearningProjections <LearningProjection>` from other mechanisms.
A list of projections received by a parameterState is maintained in its ``receivesFromProjections`` attribute.
Like all PsyNeuLink components, it has the three following fundamental attributes:

* ``variable``:  this serves as a template for the ``value`` of each projection that the parameterState receives;
  each must match both the number and type of elements of its ``variable``.

* ``function``:  this performs an elementwise (Hadamard) aggregation  of the ``values`` of the projections
   received by the parameterState.  The default function is :any:`LinearCombination` that multiplies the values.
   A custom function can be specified (e.g., to perform a Hadamard sum, or to handle non-numeric values in
   some way), so long as it generates a result that is compatible with the ``value`` expected for the parameterState
XXX IS THIS TRUE:
   It assigns the result to the parameterState's ``value`` attribute.
XXX DOES THIS COMBINE WITH BASEVALUE, OR IS THAT DONE AFTETWARDS?

* ``value``:  this is the aggregated value of the projections received by the parameterState, assigned to it by the
  parameterState's ``function``.  It must be compatible
  COMMENT:
  both with the inputState's ``variable`` (since the ``function``
  of an inputState only combines the values of its projections, but does not otherwise transform its input),
  COMMENT
  with its corresponding item of the owner mechanism's ``variable``.

In addition, a parameterState has two other attributes that are used to determine the value of the ``function``
parameter for which it is responsible:

.. ParameterState_BaseValue:

* ``baseValue``:  this is the default value of the ``function`` parameter for which the parameterState is responsible.
  It is combined with the parameterState's value (i.e., the aggregated values received from its projections) to
  determine the value of the ``function`` parameter for which the parameterState is responsible
  (see :ref:`figure <ParameterState_Figure>` below).

.. ParameterState_Parameter_Modulation_Operation:

* ``parameterModulationOperation``: determines how the parameterState's ``value`` (i.e., the aggregrated values
  received from its projections) is combined with its ``baseValue`` to generate the value assigned to the ``function``
  parameter for which it is responsible (see :ref:`figure <ParameterState_Figure>` below).  This must be a value of
  :any:`ModulationOperation`;  the default is :keyword:`PRODUCT`.

COMMENT:
   XXXX DOCUMENT THAT THIS CAN BE SPECIFIED IN A TUPLE WITH PARAM VALUE (INSTEAD OF PROJECTION) AS PER FIGURE?
COMMMENT

.. _ParameterState_Execution:

Execution
---------

States cannot be executed directly.  They are executed when the mechanism to which they belong is executed. When this
occurs, each parameterState executes any projections it receives, calls its own ``function`` to aggregate their
values, and then assigns this to the parameter of its owner's ``function`` for which it is responsible.  The
value  of the parameter is determined by the ``baseValue`` of its parameterState, modified by the value of any
projections it receives.  The way in which it is modified is determined by

.. ParameterState_Runtime_Parameters:

The value of function parameters can also be modified when the function's object is executed.  This can be done by
specifying runtime parameters for a mechanism where it is specified in the ``pathway`` of a process or in mechanism's
``execute`` or ``run`` methods (see :ref:`Mechanism_Runtime_Parameters`).
COMMENT:
   XXXXX MAKE SURE ROLE OF ParamModulationOperation FOR runtime params IS EXPLAINED THERE (OR EXPLAIN HERE)
COMMENT

COMMENT:
.. ParameterState_Parameter_Modulation_Operation:

XXXX EXPLAIN:
parameter_modulation_operation:  ModulationOperation - list values and their meaning
see ref:`Mapping_Parameter_Modulation_Operation`

        - get ParameterStateParams
        - pass params to super, which aggregates inputs from projections
        - combine input from projections (processed in super) with baseValue using paramModulationOperation
        - combine result with value specified at runtime in PARAMETER_STATE_PARAMS
        - assign result to self.value

COMMENT


COMMENT:
 XXXXX NEED TO MODIFY DESCRIPTION AND/OR FIGURE TO DEAL WITH LEARNING SIGNALS
COMMENT

.. _ParameterState_Figure:

The figure below shows how these factors are combined by the parameterState to determine the parameter value for a
function:

    **How a ParameterState Determines the Value of a Parameter of its Owner's Function**

    .. figure:: _static/ParameterState_fig.*
       :alt: ParameterState
       :scale: 75 %

       ..

       +--------------+--------------------------------------------------------------------+
       | Component    | Impact of ParameterState on Parameter Value                        |
       +==============+====================================================================+
       | A (brown)    | ``baseValue`` (default value of parameter of owner's ``function``) |
       +--------------+--------------------------------------------------------------------+
       | B (purple)   | runtime specification of parameter value                           |
       +--------------+--------------------------------------------------------------------+
       | C (red)      | runtime parameter influences projection-modulated ``baseValue``    |
       +--------------+--------------------------------------------------------------------+
       | D (green)    | combined projection values modulate ``baseValue``                  |
       +--------------+--------------------------------------------------------------------+
       | E (blue)     | parameterState's ``function`` combines ``value`` of  projections   |
       +--------------+--------------------------------------------------------------------+


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
    """
    ParameterState(                                              \
    owner,                                                       \
    reference_value=None                                         \
    value=None,                                                  \
    function=LinearCombination(operation=PRODUCT),               \
    parameter_modulation_operation=ModulationOperation.MULTIPLY, \
    params=None,                                                 \
    name=None,                                                   \
    prefs=None)

    Implements subclass of State that represents and possibly modifies the parameter value for a function

    COMMENT:

        Description
        -----------
            The ParameterState class is a componentType in the State category of Function,
            Its FUNCTION executes the projections that it receives and updates the ParameterState's value

        Class attributes
        ----------------
            + componentType (str) = kwMechanisParameterState
            + classPreferences
            + classPreferenceLevel (PreferenceLevel.Type)
            + paramClassDefaults (dict)
                + FUNCTION (LinearCombination)
                + FUNCTION_PARAMS  (Operation.PRODUCT)
                + PROJECTION_TYPE (CONTROL_PROJECTION)
                + PARAMETER_MODULATION_OPERATION   (ModulationOperation.MULTIPLY)
            + paramNames (dict)

        Class methods
        -------------
            _instantiate_function: insures that function is ARITHMETIC) (default: Operation.PRODUCT)
            update_state: updates self.value from projections, baseValue and runtime in PARAMETER_STATE_PARAMS

        StateRegistry
        -------------
            All ParameterStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism
        the mechanism to which the parameterState belongs;  it must be specified or determinable from the context in
        which the parameterState is created.

    reference_value : number, list or np.ndarray
        the default value of the parameter for which the parameterState is responsible.

    value : number, list or np.ndarray
        used as the template for ``variable``.

    function : Function or method : default LinearCombination(operation=SUM)
        function used to aggregate the values of the projections received by the parameterState.
        It must produce a result that has the same format (number and type of elements) as its input.

    parameter_modulation_operation : ModulationOperation : default ModulationOperation.MULTIPLY
        specifies the operation by which the values of the projections received by the parameterState are used
        to modify its ``baseValue`` before assigning it to the parameter for which it is responsible.

    params : Optional[Dict[param keyword, param value]]
        a dictionary that can be used to specify the parameters for the inputState, parameters for its function,
        and/or a custom function and its parameters (see :doc:`Component` for specification of a params dict).

    name : str : default InputState-<index>
        a string used for the name of the inputState.
        If not is specified, a default is assigned by StateRegistry of the mechanism to which the inputState belongs
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the PreferenceSet for the inputState.
        If it is not specified, a default is assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details) [LINK].
    COMMENT


    Attributes
    ----------
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
            # Check whether ModulationOperation for projections has been specified at runtime
            # Note: this is distinct from ModulationOperation for runtime parameter (handled below)
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
        # # MODIFIED 11/27/16 OLD:
        # default_operation = self.prefs.functionRuntimeParamsPref
        # MODIFIED 11/27/16 NEW:
        default_operation = self.prefs.functionRuntimeParamsPref[0]
        # MODIFIED 11/27/16 END

        # If there is a runtime param specified, could be a (parameter value, ModulationOperation) tuple
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

            # # Assign class-level pref as default operation
            # # # MODIFIED 11/27/16 OLD: [REDUNDANT WITH ABOVE
            # # default_operation = self.prefs.functionRuntimeParamsPref
            # # MODIFIED 11/27/16 END
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

    # FIX: MODIFY THIS TO USE user_params (STILL TREATING function_param_specs AS BELOW)

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
