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

A parameterState belongs to either a `mechanism <Mechanism>` or a `MappingProjection`, and is used to represent and
possibly modify the value of a parameter of its owner or it owner's function.  It can receive one or more
`ControlProjections <ControlProjection>` and/or `LearningProjections <LearningProjection>` that modify that
parameter.   The projections received by a parameterState are listed in its
`afferents <ParameterState.afferents>` attribute.
Its `function <ParameterState.function>` combines the values of these inputs, and uses the result to modify the value
of the parameter for which it is responsible.


.. _ParameterState_Creation:

Creating a ParameterState
-------------------------

A parameterState can be created by calling its constructor, but in general this is not necessary or advisable as
parameterStates are created automatically when the mechanism or projection to which they belong is created.  The
`owner <ParamaterState.owner>` of a parameterState must be a `mechanism <Mechanism>` or `MappingProjection`.  If the
`owner <ParamaterState.owner>` is not explicitly specified, and can't be determined by context, the parameterState
will be assigned to the :ref:`DefaultProcessingMechanism`.  One parameterState is created for each configurable
parameter of its owner, as well as for each parameter that has been specified for that component's :keyword:`function`.
Each parameterState is created using the specification of the parameter for which it is responsible, as described below.

.. _ParameterState_Specifying_Parameters:

Specifying Parameters
~~~~~~~~~~~~~~~~~~~~~

Parameters can be specified in one of several places:

    * In the **argument for the parameter** of the constructor for the `component <Component>` to which the parameter 
      belongs (see :ref:`Component_Specifying_Functions_and_Parameters` for additional details).
    ..
    * In a **parameter specification dictionary** assigned to the **params** argument in the constructor for the 
      component to which the parameter belongs. The entry for each parameter must use the name of the parameter
      (or a corresponding keyword) as its key, and the parameter's specification as its value (see 
      `examples <ParameterState_Specification_Examples>` below). Parameters for a component's :keyword:`function` 
      must be specified in an entry with the key FUNCTION_PARAMS, the value of which is a parameter dictionary 
      containing an entry for each of the function's parameters to be specified.  When a value is assigned to a 
      parameter in a parameter dictionary, it overrides any value assigned to the argument for the parameter in the 
      component's constructor.
    ..
    * By direct assignment to the corresponding attribute of the component to which the parameter belongs.  The
      attribute always has the same name as the parameter and can be referenced using standard python attribute
      notation (e.g., myComponent.paramter_name).      
    ..
    * In the `assign_params` method for the component.
    ..
    * When the component is executed, in the **runtime_params** argument of a call to component's
      `execute <Mechanism.Mechanism_Base.execute>`
      COMMENT:
          or `run <Mechanism.Mechanism_Base.run>` methods
      COMMENT
      method (only for a mechanism), or in a tuple with the mechanism where it is specified as part of the
      `pathway` for a process (see :ref:`Runtime Specification <ParameterState_Runtime_Parameters>` below).

The value specified for the parameter (either explicitly or by default) is assigned as the parameterState's
`base_value <ParameterState.base_value>`, and any projections assigned to it are added to its
`receiveFromProjections <ParameterState.afferents>` attribute. When the parameterState's owner is
executed, the parameterState's `base_value <ParameterState.base_value>` is combined with the value of the projections
it receives to determine the value of the parameter for which the parameterState is responsible
(see `ParameterState_Execution` for details).

The specification of a parameter can take any of the following forms:

    * A **value**.  This must be a valid value for the parameter.  it creates a default parameterState,
      assigns the parameter's default value as the parameterState's `base_value <ParameterState.base_value>`,
      and assigns the parameter's name as the name of the parameterState.
    ..
    * A reference to an existing **parameterState** object.
      COMMENT:
      It's name must be the name of a parameter of the
      owner's ``function``, and its
      COMMENT
      Its value must be a valid one for the parameter.

      .. note::
          This capability is provided for generality and potential
          future use, but its current use is not advised.
    ..
    * A `projection specification <Projection_In_Context_Specification>`.  This creates a default parameterState,
      assigns the parameter's default value as the parameterState's `base_value <ParameterState.base_value>`,
      and assigns the parameter's name as the name of the parameterState.  It also creates and/or assigns the
      specified projection, and assigns the parameterState as the projection's
      `receiver <Projection.Projection.receiver>`.  The projection must be a `ControlProjection` or
      `LearningProjection`, and its value must be a valid one for the parameter.
    ..
    * A 2-item (value, projection specification) **tuple**.  This creates a default
      parameterState, uses the value (1st) item of the tuple as parameterState's
      `base_value <ParameterState.base_value>`, and assigns the parameter's name as the name of the parameterState.
      The projection (2nd) item of the tuple is used to create and/or assign the specified projection, that is assigned
      the parameterState as its `receiver <Projection.Projection.receiver>`.  The projection must be a
      `ControlProjection` or `LearningProjection`, and its value must be a valid one for the parameter.

      .. note::
          Currently, the :keyword:`function` of a component, although it can be specified a parameter value,
          cannot be assigned
          COMMENT:
          a ControlProjection, LearningProjection, or a runtime specification.
          COMMENT
          a ControlProjection or a LearningProjection. This may change in the future.

The **default value** assigned to a parameterState is the default value of the argument for the parameter in the
constructor for the parameter's owner.  If the value of a parameter is specified as `None`, `NotImplemented`,
or any other non-numeric value that is not one of those listed above, then no parameter state is created and the
parameter cannot be modified by a `ControlProjection`, 'LearningProjection', or 
`runtime specification <ParameterState_Runtime_Parameters>`.

COMMENT:
    - No parameterState is created for parameters that are:
       assigned a non-numeric value (including None, NotImplemented, False or True)
          unless it is:
              a tuple (could be one specifying ControlProjection, LearningProjection or Modulation)
              a dict with an entry with the key FUNCTION_PARAMS and a value that is a dict (otherwise exclude)
       a function
           IMPLEMENTATION NOTE: FUNCTION_RUNTIME_PARAM_NOT_SUPPORTED
           (this is because paramInstanceDefaults[FUNCTION] could be a class rather than an bound method;
           i.e., not yet instantiated;  could be rectified by assignment in _instantiate_function)

    - self.variable must be compatible with self.value (enforced in _validate_variable)
        note: although it may receive multiple projections, the output of each must conform to self.variable,
              as they will be combined to produce a single value that must be compatible with self.variable
COMMENT


.. _ParameterState_Specification_Examples:

**Examples**

In the following example, a mechanism is created with a function that has four parameters,
each of which is specified using a different format::

    my_mechanism = SomeMechanism(function=SomeFunction(param_a=1.0,
                                                       param_b=(0.5, ControlProjection),
                                                       param_c=(36, ControlProjection(function=Logistic),
                                                       param_d=ControlProjection)))

The first parameter of the mechanism's function (``param_a``) is assigned a value directly; the second (``param_b``) is
assigned a value and a ControlProjection; the third (``param_c``) is assigned a value and a
`ControlProjection with a specified function  <ControlProjection_Structure>`; and the fourth (``param_d``) is
assigned just a `ControlProjection` (the default value for the parameter will be used).

In the following example, a `MappingProjection` is created, and its
`matrix <MappingProjection.MappingProjection.matrix>` parameter is assigned a random weight matrix (using a
`matrix keyword <Matrix_Keywords>`) and `LearningProjection`::

    my_mapping_projection = MappingProjection(sender=my_input_mechanism,
                                              receiver=my_output_mechanism,
                                              matrix=(RANDOM_CONNECTIVITY_MATRIX, LearningProjection))

.. note::
   the `matrix <MappingProjection.MappingProjection.matrix>` parameter belongs to the MappingProjection's
   `function <MappingProjection.MappingProjection.function>`;  however, since it has only one standard function,
   its arguments are available in the constructor for the projection (see
   `Component_Specifying_Functions_and_Parameters` for a more detailed explanation).

COMMENT:
    ADD EXAMPLE USING A PARAMS DICT, INCLUDING FUNCTION_PARAMS, AND assign_params
COMMENT

.. _ParameterState_Structure:

Structure
---------

Every parameterState is owned by a `mechanism <Mechanism>` or `MappingProjection`. It can receive one or more
`ControlProjections <ControlProjection>` or `LearningProjections <LearningProjection>`.  However, the format (the
number and type of its elements) of each must match the value of the parameter for which the parameterState is
responsible.  When the parameterState is updated (i.e., the owner is executed) the values of its projections are
combined (using the  parameterState's `function <ParameterState.function>`) and the result is used to modify the
parameter for which the parameterState is responsible (see `Execution <ParameterState_Execution>` below).  The
projections received by a parameterState are listed in its `receiveFromProjections
<ParameterState.afferents>` attribute. Like all PsyNeuLink components, it has the three following core
attributes:

* `variable <ParameterState.variable>`:  this serves as a template for the `value <Projection.Projection.value>` of
  each projection that the parameterState receives.  It must match the format (the number and type of elements) of the
  parameter for which the parameterState is responsible. Any projections the parameterState receives must, it turn,
  match the format of :keyword:`variable`.

* `function <ParameterState.function>`:  this performs an elementwise (Hadamard) aggregation  of the values of the
  projections received by the parameterState.  The default function is `LinearCombination` that multiplies the
  values. A custom function can be specified (e.g., to perform a Hadamard sum, or to handle non-numeric values in
  some way), so long as it generates a result that is compatible with the `value <ParameterState.value>` of the
  parameterState.

* `value <ParameterState.value>`:  this is the value assigned to the parameter for which the parameterState is
  responsible.  It is the `base_value <ParameterState.base_value>` of the parameterState, modified by
  aggregated value of the projections received by the parameterState returned by the
  `function <ParameterState.function>.

In addition, a parameterState has two other attributes that are used to determine the value it assigns to the
parameter for which it is responsible (as shown in the `figure <ParameterState_Figure>` below):

.. ParameterState_BaseValue:

* `base_value <ParameterState.base_value>`:  this is the default value of the parameter for which the
  parameterState is responsible.  It is combined with the result of the parameterState's
  `function <ParameterState.function>` to determine the value of the parameter for which the parameterState is
  responsible.

.. ParameterState_Parameter_Modulation_Operation:

* `parameterModulationOperation <ParameterState.parameterModulationOperation>`: this determines how the
  result of the parameterState's `function <ParameterState.function>` (the aggregated values of the projections it
  receives) is combined with its `base_value <ParameterState.base_value>` to generate the value of the parameter
  for which it is responsible.  This must be a value of `Modulation`.  It can be specified in either
  the **parameter_modulation_operation** argument of the parameterState's constructor, or in a 
  PARAMETER_MODULATION_OPERATION entry of a `parameter dictionary <ParameterState_Specifying_Parameters>` in 
  either the **params** argument of the parameterState's constructor or within a PARAMETER_STATE_PARAMS 
  dictionary in a `runtime specification <ParameterState_Runtime_Parameters>`. The default is value is
  `Modulation.PRODUCT`, which multiples the parameterState's `base_value <ParameterState.base_value>` by the 
  aggregated value of the result of the parameterState's `function <ParameterState.function>` to determine the value 
  of the parameter.

All of the user-modifiable parameters of a component are listed in its `user_params <Component.user_params>` attribute, 
which is a read-only dictionary with an entry for each parameter.  The parameters of a component can be 
modified individually by assigning a value to the corresponding attribute, or in groups using the component's 
`assign_params <Component.assign_params>` method.  The parameters for a component's `function <Component.function>` 
are listed in its `function_params <Component.function_params>` attribute, which is a read-only dictionary with an 
entry for each of its function's parameter.  The parameters of a component's function can be modified by
assigning a value to the corresponding attribute of the component's `function_object <Component.function_object>` 
attribute (e.g., myMechanism.function_object.my_parameter), or in a FUNCTION_PARAMS dict in `assign_params`.  

.. _ParameterState_Figure:

The figure below shows how the specifications for a parameter are combined by its parameterState to determine the
parameter's value.

    **How a ParameterState Determines the Value of a Parameter**

    .. figure:: _static/ParameterState_fig_without_runtime_params.pdf
       :alt: ParameterState
       :scale: 75 %

       ..

       +--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
       | Component    | Impact of ParameterState on Parameter Value                                                                                                                  |
       +==============+==============================================================================================================================================================+
       | A (brown)    | `base_value <ParameterState.base_value>` (default value of the parameter)                                                                                      |
       +--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
       | B (blue)     | parameterState's `function <ParameterState.function>` combines `value <Projection.Projection.value>` of projections                                          |
       +--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
       | C (green)    | parameterState's `parameterModulationOperation <ParameterState.parameterModulationOperation>` combines projections and `base_value <ParameterState.base_value>`|
       +--------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

       In example, the values for the parameters (shown in brown) -- ``param_x`` for the mechanism, and ``param_y``
       for its ``function`` -- specify the :py:data:`base_value <ParameterState.base_value>` of the paramterState for
       each parameter (labeled "A" in the figure).  These are the values that will be used for
       those parameters absent any other influences. However, ``param_y`` is assigned a ControlProjection, so its value
       will also be determined by the value of the ControlProjection to its parameterState;  that will be combined
       with its `base_value <ParameterState.base_value>` using the
       `parameterModulationOperation <ParameterState.parameterModulationOperation>`
       specified for the mechanism (``Modulation.SUM``, shown in green, and labeled "C" in the figure). If
       there had been more than one ControlProjection specified, their values would have been combined using the
       parameterState's `function <ParameterState.function>` (lableled "B" in the figure), before combining the result
       with the
       base_value.


.. _ParameterState_Execution:

Execution
---------

A parameterState cannot be executed directly.  It is executed when the mechanism to which it belongs is executed.
When this occurs, the parameterState executes any `ControlProjections` and/or `LearningProjections` it receives, and
calls its `function <ParameterState.function>` to aggregate their values.  It then combines the result with the
parameterState's `base_value <ParameterState.base_value>` using its
`parameterModulationOperation <ParameterState.parameterModulationOperation>` attribute, combines the result with any 
`runtime specification <ParameterState_Runtime_Parameters>` for the parameter using the `Modulation` 
specified for runtime parameters, and finally assigns the result as the `value <ParameterState.value>` of the 
parameterState.  This is used as the value of the parameter for which the parameterState is responsible.

.. _ParameterState_Runtime_Parameters:

Runtime Specification of Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   This is an advanced feature, that is generally not required for most applications.

In general, it should not be necessary to modify parameters programmatically each time a process or system is
executed or run; ordinarily, this should be done using `control projections <ControlProjection>` and/or
`learning projections <LearningProjection>`.  However, if necessary, it is possible to modify parameters
"on-the-fly" in two ways:  by specifying runtime parameters for a mechanism as part of a tuple where it is
specified in the `pathway <Process.Process_Base.pathway>` of a process, or in the
`execute <Mechanism.Mechanism_Base.execute>`
COMMENT:
    or :py:meth:`run <Mechanism.Mechanism_Base.run>` methods
COMMENT
method for a mechanism, process or system (see `Mechanism_Runtime_Parameters`).  By default, runtime assignment of
a parameter value is *one-time*:  that is, it applies only for the round of execution in which it is specified,
and the parameter's value returns to the default for the instance of the component to which it belongs after
execution.  The `runtimeParamsStickyAssignmentPref` can be used to specify persistent assignment of a runtime value,
however in general it is better to modify a parameter's value permantently by assigning the value directly its 
corresponding attribute, or using the `assign_params` method of its component.

COMMENT:
    IS THE MECHANISM TUPLE SPECIFICATION ONE TIME OR EACH TIME? <- BUG IN merge_dictionary()
    IS THE RUN AND EXECUTE SPECIFICATION ONE TRIAL OR ALL TRIALS IN THAT RUN?
COMMENT

.. note::
   At this time, runtime specification can be used only  for the parameters of a mechanism or of its ``function``.
   Since the function itself is not currently assigned a parameterState, it cannot be modified at runtime;  nor is
   there currently a method for runtime specification for the parameters of a MappingProjection.  These may be
   supported in the future.
 
.. _ParameterState_Runtime_Figure:

COMMENT:
   XXXXX MAKE SURE ROLE OF ParamModulationOperation FOR runtime params IS EXPLAINED THERE (OR EXPLAIN HERE)
   XXXX DOCUMENT THAT MOD OP CAN BE SPECIFIED IN A TUPLE WITH PARAM VALUE (INSTEAD OF PROJECTION) AS PER FIGURE?
COMMENT

The figure below shows how runtime paramter specification combines the others ways to specify a parameter's value:

    .. figure:: _static/ParameterState_fig_with_runtime_params.pdf
       :alt: ParameterState
       :scale: 75 %

       ..

       +--------------+--------------------------------------------------------------------+
       | Component    | Impact of ParameterState on Parameter Value (including at runtime) |
       +==============+====================================================================+
       | A (brown)    | ``base_value`` (default value of parameter``)                       |
       +--------------+--------------------------------------------------------------------+
       | B (blue)     | parameterState's ``function`` combines ``value`` of  projections   |
       +--------------+--------------------------------------------------------------------+
       | C (green)    | runtime parameter influences projection-modulated ``base_value``    |
       +--------------+--------------------------------------------------------------------+
       | D (violet)   | runtime specification of parameter value                           |
       +--------------+--------------------------------------------------------------------+
       | E (red)      | combined projection values modulate ``base_value``                  |
       +--------------+--------------------------------------------------------------------+
       
       * 1st example:  param_x is given a runtime value (violet) but no runtime Modulation;
         param_y is given a runtime value (violet) and also a runtime Modulation (red);
        the parameterState's parameterModulationOperation is set to MULTIPLY (green).
       ..
       * 2nd example:  param_x is given a runtime value (violet) and also a runtime Modulation (red);
         param_y is given a runtime value (violet) but no runtime Modulation;
         the parameterState's parameterModulationOperation is set to SUM (green)

       COMMENT:
           NOTES: CAPS FOR PARAM SPECIFICATION IN DICTS -> KEYWORDS
                  AUGMENT FIGURE TO SHOW PARAM SPECIFICATIONS FOR BOTH THE OBJECT AND ITS FUNCTION
       COMMENT

.. _ParameterState_Class_Reference:

Class Reference
---------------

"""

from PsyNeuLink.Components.Functions.Function import *
from PsyNeuLink.Components.States.State import *
from PsyNeuLink.Components.States.State import _instantiate_state


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
    parameter_modulation_operation=Modulation.MULTIPLY,          \
    params=None,                                                 \
    name=None,                                                   \
    prefs=None)

    Implements a subclass of `State` that represents and possibly modifies the value of a parameter for a mechanism,
    projection, or function.

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
                + PARAMETER_MODULATION_OPERATION   (Modulation.MULTIPLY)
            + paramNames (dict)

        Class methods
        -------------
            _instantiate_function: insures that function is ARITHMETIC) (default: Operation.PRODUCT)
            update_state: updates self.value from projections, base_value and runtime in PARAMETER_STATE_PARAMS

        StateRegistry
        -------------
            All ParameterStates are registered in StateRegistry, which maintains an entry for the subclass,
              a count for all instances of it, and a dictionary of those instances

    COMMENT


    Arguments
    ---------

    owner : Mechanism or Projection
        the `mechanism <Mechanism>` or `projection <Projection>` to which to which the parameterState belongs; it must
        be specified or determinable from the context in which the parameterState is created. The owner of a
        parameterState for the parameter of a :keyword:`function` should be specified as the mechanism or projection
        to which the function belongs.

    reference_value : number, list or np.ndarray
        specifies the default value of the parameter for which the parameterState is responsible.

    variable : number, list or np.ndarray
        specifies the template for the parametersState's `variable <ParameterState.variable>`.

    function : Function or method : default LinearCombination(operation=SUM)
        specifies the function used to aggregate the values of the projections received by the parameterState.
        It must produce a result that has the same format (number and type of elements) as its input.

    COMMENT:
        parameter_modulation_operation : Modulation : default Modulation.MULTIPLY
            specifies the operation by which the values of the projections received by the parameterState are used
            to modify its `base_value <ParameterState.base_value>` before assigning it as the value of the parameter for
            which the parameterState is responsible.
    COMMENT

    params : Optional[Dict[param keyword, param value]]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can be used to specify the parameters for
        the parameterState or its function, and/or a custom function and its parameters.  Values specified for
        parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default InputState-<index>
        a string used for the name of the inputState.
        If not is specified, a default is assigned by StateRegistry of the mechanism to which the inputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : State.classPreferences]
        the `PreferenceSet` for the inputState.
        If it is not specified, a default is assigned using `classPreferences` defined in __init__.py
        (see :doc:`PreferenceSet <LINK>` for details).


    Attributes
    ----------

    owner : Mechanism
        the mechanism to which the parameterState belongs.

    afferents : Optional[List[Projection]]
        a list of the projections received by the parameterState (i.e., for which it is a
        `receiver <Projection.Projection.receiver>`); generally these are `ControlProjection(s) <ControlProjection>`
        and/or `LearningProjection(s) <LearningProjection>`.

    variable : number, list or np.ndarray
        the template for the `value <Projection.Projection.value>` of each projection that the parameterState receives,
        each of which must match the format (number and types of elements) of the parameterState's :keyword:`variable`.

    function : CombinationFunction : default LinearCombination(operation=PRODUCT))
        performs an element-wise (Hadamard) aggregation  of the `value <Projecction.Projection.value>` of each
        projection received by the parameterState.

    COMMENT:
        base_value : number, list or np.ndarray
            the default value for the parameterState.  It is combined with the aggregated value of any projections it
            receives using its `parameterModulationOperation <ParameterState.parameterModulationOperation>`
            and then assigned to `value <ParameterState.value>`.
    
        parameterModulationOperation : Modulation : default Modulation.PRODUCT
            the arithmetic operation used to combine the aggregated value of any projections is receives
            (the result of the parameterState's `function <ParameterState.function>`) with its
            `base_value <ParameterState.base_value>`, the result of which is assigned to `value <ParameterState.value>`.
    COMMENT

    value : number, list or np.ndarray
        the aggregated value of the projections received by the ParameterState, combined with the
        `base_value <ParameterState.base_value>` using its
        `parameterModulationOperation <ParameterState.parameterModulationOperation>`
        COMMENT:
        as well as any runtime specification
        COMMENT
        .  This is the value assigned to the parameter for which the parameterState is responsible.

    name : str : default <State subclass>-<index>
        the name of the inputState.
        Specified in the **name** argument of the constructor for the outputState.  If not is specified, a default is
        assigned by the StateRegistry of the mechanism to which the outputState belongs
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

        .. note::
            Unlike other PsyNeuLink components, states names are "scoped" within a mechanism, meaning that states with
            the same name are permitted in different mechanisms.  However, they are *not* permitted in the same
            mechanism: states within a mechanism with the same base name are appended an index in the order of their
            creation.

    prefs : PreferenceSet or specification dict : State.classPreferences
        the `PreferenceSet` for the inputState.
        Specified in the **prefs** argument of the constructor for the projection;  if it is not specified, a default is
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
                 function=Linear(),
                 parameter_modulation_operation=Modulation.MULTIPLY,
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
                                                  parameter_modulation_operation=parameter_modulation_operation,
                                                  params=params)

        self.reference_value = reference_value

        # Validate sender (as variable) and params, and assign to variable and paramsInstanceDefaults
        # Note: pass name of mechanism (to override assignment of componentName in super.__init__)
        super(ParameterState, self).__init__(owner,
                                             variable=variable,
                                             params=params,
                                             name=name,
                                             prefs=prefs,
                                             context=self)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that parameterState (as identified by its name) is for a valid parameter of the owner

        Parameter can be either owner's, or owner's function_object
        """

        # If the parameter is not in either the owner's user_params dict or its function_params dict, throw exception
        if not self.name in self.owner.user_params.keys() and not self.name in self.owner.function_params.keys():
            raise ParameterStateError("Name of requested parameterState ({}) does not refer to a valid parameter "
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

        # # FIX: UPDATE FOR LEARNING [COMMENTED OUT]
        # # If parameterState is for a matrix of a MappingProjection,
        # #     its parameter_modulation_operation should be SUM (rather than PRODUCT)
        # #         so that weight changes (e.g., from a learningSignals) are added rather than multiplied
        # if self.name == MATRIX:
        #     # IMPLEMENT / TEST: 10/20/16 THIS SHOULD BE ABLE TO REPLACE SPECIFICATION IN LEARNING PROJECTION
        #     self.params[PARAMETER_MODULATION_OPERATION] = Modulation.ADD

        super()._instantiate_function(context=context)

        # # Insure that output of function (self.value) is compatible with relevant parameter's reference_value
        if not iscompatible(self.value, self.reference_value):
            raise ParameterStateError("Value ({0}) of the {1} parameterState for the {2} mechanism is not compatible "
                                      "the type of value expected for that parameter ({3})".
                                           format(self.value,
                                                  self.name,
                                                  self.owner.name,
                                                  self.reference_value))

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
        return value

    @property
    def trans_projections(self):
        raise ParameterStateError("PROGRAM ERROR: Attempt to access trans_projection for {};"
                                  "it is a {} which does not have {}s".
                                  format(self.name, PARAMETER_STATE, TRANSMISSIVE_PROJECTION))

    @trans_projections.setter
    def trans_projections(self, value):
        raise ParameterStateError("PROGRAM ERROR: Attempt to assign trans_projection to {};"
                                  "it is a {} which cannot accept {}s".
                                  format(self.name, PARAMETER_STATE, TRANSMISSIVE_PROJECTION))


def _instantiate_parameter_states(owner, context=None):
    """Call _instantiate_parameter_state for all params in user_params to instantiate ParameterStates for them

    If owner.params[PARAMETER_STATE] is None or False:
        - no parameterStates will be instantiated.
    Otherwise, instantiate parameterState for each allowable param in owner.user_params

    """

    # TBI / IMPLEMENT: use specs to implement paramterStates below

    owner._parameter_states = ContentAddressableList(ParameterState)

    # Check that parameterStates for owner have not been explicitly suppressed (by assigning to None)
    try:
        no_parameter_states = not owner.params[PARAMETER_STATES]
        # PARAMETER_STATES for owner was suppressed (set to False or None), so do not instantiate any parameterStates
        if no_parameter_states:
            return
    except KeyError:
        # PARAMETER_STATES not specified at all, so OK to continue and construct them
        pass

    try:
        owner.user_params
    except AttributeError:
        return

    # Instantiate parameterState for each param in user_params (including all params in function_params dict),
    #     using its value as the state_spec
    # IMPLEMENTATION NOTE:  Use user_params_for_instantiation since user_params may have been overwritten
    #                       when defaults were assigned to paramsCurrent in Component.__init__,
    #                       (since that will assign values to the properties of each param;
    #                       and that, in turn, will overwrite their current values with the defaults from paramsCurrent)
    for param_name, param_value in owner.user_params_for_instantiation.items():
        _instantiate_parameter_state(owner, param_name, param_value, context=context)


def _instantiate_parameter_state(owner, param_name, param_value, context):
    """Call _instantiate_state for allowable params, to instantiate a ParameterState for it

    Include ones in owner.user_params[FUNCTION_PARAMS] (nested iteration through that dict)
    Exclude if it is a:
        parameterState that already exists (e.g., in case of a call from Component.assign_params)
        non-numeric value (including None, NotImplemented, False or True)
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

    # # Skip if parameterState already exists (e.g., in case of call from Component.assign_params)
    # if param_name in owner.parameterStates:
    #     return

    from PsyNeuLink.Components.Projections.Projection import Projection
    # Allow numerics but omit booleans (which are treated by is_numeric as numerical)
    if is_numeric(param_value) and not isinstance(param_value, bool):
        pass
    # Only allow a FUNCTION_PARAMS dict
    elif isinstance(param_value, ReadOnlyOrderedDict) and param_name is FUNCTION_PARAMS:
        pass
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
    # Allow tuples (could be spec that includes a projection or Modulation)
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
            #    Therefore, None will be passed as the constraint_value, which will cause validation of the
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

            # if isinstance(owner, MappingProjection) and function_param_name is MATRIX:
            #     state_params = {FUNCTION: LinearCombination}
            # else:
            #     state_params = None
            # Assign parameterState for function_param to the component
            state = _instantiate_state(owner=owner,
                                      state_type=ParameterState,
                                      state_name=function_param_name,
                                      state_spec=function_param_value,
                                      state_params=None,
                                      constraint_value=function_param_value,
                                      constraint_value_name=function_param_name,
                                      context=context)
            if state:
                owner._parameter_states[function_param_name] = state

    elif _is_legal_param_value(owner, param_value):
        state = _instantiate_state(owner=owner,
                                  state_type=ParameterState,
                                  state_name=param_name,
                                  state_spec=param_value,
                                  state_params=None,
                                  constraint_value=param_value,
                                  constraint_value_name=param_name,
                                  context=context)
        if state:
            owner._parameter_states[param_name] = state


def _is_legal_param_value(owner, value):

    # LEGAL PARAMETER VALUES:

    # lists, arrays numeric values
    if is_value_spec(value) or isinstance(value, tuple):
        return True

    # keyword that resolves to one of the above
    if get_param_value_for_keyword(owner, value) is not None:
        return True

    # Assignment of ParameterState for Component objects, function or method are not currently supported
    if isinstance(value, (function_type, method_type, Component)):
        return False


def _get_parameter_state(sender_owner, sender_type, param_name, component):
    """Return parameterState for named parameter of a mechanism requested by owner
    """

    # Validate that component is a Mechanism or Projection
    if not isinstance(component, (Mechanism, Projection)):
        raise ParameterStateError("Request for {} of a component ({}) that is not a {} or {}".
                                  format(PARAMETER_STATE, component, MECHANISM, PROJECTION))

    try:
        return component._parameter_states[param_name]
    except KeyError:
        # Check that param (named by str) is an attribute of the mechanism
        if not (hasattr(component, param_name) or hasattr(component.function_object, param_name)):
            raise ParameterStateError("{} (in specification of {}  {}) is not an attribute "
                                        "of {} or its function"
                                        .format(param_name, sender_type, sender_owner.name, component))
        # Check that the mechanism has a parameterState for the param
        if not param_name in component._parameter_states.names:
            raise ParameterStateError("There is no ParameterState for the parameter ({}) of {} "
                                        "specified in {} for {}".
                                        format(param_name, component.name, sender_type, sender_owner.name))

