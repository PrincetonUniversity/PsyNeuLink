#
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# *****************************************  USER-DEFINED FUNCTION  ****************************************************

import ctypes
import numpy as np
import typecheck as tc

from inspect import signature, _empty

from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.function import FunctionError, Function_Base
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    ADDITIVE_PARAM, CONTEXT, CUSTOM_FUNCTION, EXECUTION_ID, MULTIPLICATIVE_PARAM, OWNER, PARAMS, \
    PARAMETER_PORT_PARAMS, SELF, USER_DEFINED_FUNCTION, USER_DEFINED_FUNCTION_TYPE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences import is_pref_set
from psyneulink.core.globals.utilities import iscompatible

from psyneulink.core import llvm as pnlvm

__all__ = ['UserDefinedFunction']

class UserDefinedFunction(Function_Base):
    """UserDefinedFunction(  \
    custom_function=None,    \
    default_variable=None,   \
    params=None,             \
    owner=None,              \
    name=None,               \
    prefs=None)

    .. _UDF_Description:

    A UserDefinedFunction (UDF) is used to "wrap" a Python function or method, including a lamdba function,
    as a PsyNeuLink `Function <Function>`, so that it can be used as the `function <Component.function>` of a
    `Component <Component>`.  This is done automatically if a Python function or method is assigned as the `function
    <Component.function>` attribute of a Component.  A Python function or method can also be wrapped explicitly,
    using the UserDefinedFunction constructor, and assigning the Python function or method to its **custom_function**
    argument.  A Python function or method wrapped as a UDF must obey the following conventions to be treated
    correctly:

    .. _UDF_Variable:

    * It must have **at least one argument** (that can be a positional or a keyword argument);  this will be treated
      as the `variable <UserDefinedFunction.variable>` attribute of the UDF's `function <UserDefinedFunction.function>`.
      When the UDF calls the function or method that it wraps, an initial attempt is made to do so with **variable**
      as the name of the first argument; if that fails, it is called positionally.  The argument is always passed as a
      2d np.array, that may contain one or more items (elements in axis 0), depending upon the Component to which the
      UDF is assigned.  It is the user's responsibility to insure that the number of items expected in the first
      argument of the function or method is compatible with the circumstances in which it will be called.
    ..
    .. _UDF_Additional_Arguments:

    * It may have have **any number of additional arguments** (positional and/or keyword);  these are treated as
      parameters of the UDF, and can be modulated by `ModulatorySignals <ModulatorySignal>` like the parameters of
      ordinary PsyNeuLink `Functions <Function>`.  If the UDF is assigned to (or automatically created for) a
      `Mechanism <Mechanism>` or `Projection <Projection>`, these parameters are each automatically assigned a
      `ParameterPort` so that they can be modulated by `ControlSignals <ControlSignal>` or `LearningSignals
      <LearningSignal>`, respectively.  If the UDF is assigned to (or automatically created for) an `InputPort` or
      `OutputPort`, and any of the parameters are specified as `Function_Modulatory_Params` (see `below
      <UDF_Modulatory_Params>`), then they can be modulated by `GatingSignals <GatingSignal>`. The function or method
      wrapped by the UDF is called with these parameters by their name and with their current values (i.e.,
      as determined by any `ModulatorySignals <ModulatorySignal>` assigned to them).
    ..
    .. _UDF_Params_Context:

    * It may include **self**, **owner**, **context**, and/or **params** arguments;  none of these are
      required, but can be included to gain access to the standard `Function` parameters (such as the history of its
      parameter values), those of the `Component` to which it has been assigned (i.e., its `owner <Function.owner>`,
      and/or receive information about the current conditions of execution.   When the function or method is called,
      an initial attempt is made to pass these arguments; if that fails, it is called again without them.
    ..
    .. _UDF_Modulatory_Params:

    * The parameters of a UDF can be specified as `Function_Modulatory_Params` in a `parameter specification dictionary
      <ParameterPort_Specification>` assigned to the **params** argument of the constructor for either the Python
      function or method, or of an explicitly defined UDF (see `examples below <UDF_Modulatory_Params_Examples>`).
      It can include either or both of the following two entries:
         *MULTIPLICATIVE_PARAM*: <parameter name>\n
         *ADDITIVE_PARAM*: <parameter name>
      These are used only when the UDF is assigned as the `function <Port_Base.function>` of an InputPort or
      OutputPort that receives one more more `GatingProjections <GatingProjection>`.

      COMMENT:
      # IMPLEMENT INTERFACE FOR OTHER MODULATION TYPES (i.e., for ability to add new custom ones)
      COMMENT

    .. tip::
       The format of the `variable <UserDefinedFunction.variable>` passed to the `custom_function
       <UserDefinedFunction.custom_function>` function can be verified by adding a ``print(variable)`` or
       ``print(type(variable))`` statement to the function.

    Examples
    --------

    **Assigning a custom function to a Mechanism**

    .. _UDF_Lambda_Function_Examples:

    The following example assigns a simple lambda function that returns the sum of the elements of a 1d array) to a
    `TransferMechanism`::

        >>> import psyneulink as pnl
        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0,0,0]],
        ...                                   function=lambda x:sum(x[0]))
        >>> my_mech.execute(input = [1, 2, 3])
        array([[6]])

    Note that the function treats its argument, x, as a 2d array, and accesses its first item for the calculation.
    This is because  the `variable <Mechanism_Base.variable>` of ``my_mech`` is defined in the **size** argument of
    its constructor as having a single item (a 1d array of length 3;  (see `size <Component.size>`).  In the
    following example, a function is defined for a Mechanism in which the variable has two items, that are summed by
    the function::

        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=lambda x: x[0] + x[1])
        >>> my_mech.execute(input = [[1],[2]])
        array([[3]])

    .. _UDF_Defined_Function_Examples:

    The **function** argument can also be assigned a function defined in Python::

        >>> def my_fct(variable):
        ...     return variable[0] + variable[1]
        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=my_fct)

    This will produce the same result as the last example.  This can be useful for assigning the function to more than
    one Component.

    More complicated functions, including ones with more than one parameter can also be used;  for example::

        >>> def my_sinusoidal_fct(input=[[0],[0]],
        ...                       phase=0,
        ...                       amplitude=1):
        ...    frequency = input[0]
        ...    t = input[1]
        ...    return amplitude * np.sin(2 * np.pi * frequency * t + phase)
        >>> my_wave_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                        function=my_sinusoidal_fct)

    Note that in this example, ``input`` is used as the name of the first argument, instead of ``variable``
    as in the examples above. The name of the first argument of a function to be "wrapped" as a UDF does not matter;
    in general it is good practice to use ``variable``, as the `variable <Component.variable>` of the Component
    to which the UDF is assigned is what is passed to the function as its first argument.  However, if it is helpful to
    name it something else, that is fine.

    Notice also that ``my_sinusoidal_fct`` takes two values in its ``input`` argument, that it assigns to the
    ``frequency`` and ``t`` variables of the function.  While  it could have been specified more compactly as a 1d array
    with two elements (i.e. [0,0]), it is specified in the example as a 2d array with two items to make it clear that
    it matches the format of the **default_variable** for the ProcessingMechanism to which it will be assigned,
    which requires it be formatted this way (since the `variable <Component.variable>` of all Components are converted
    to a 2d array).

    ``my_sinusoidal_fct`` also has two other arguments, ``phase`` and ``amplitude``.   When it is assigned to
    ``my_wave_mech``, those parameters are assigned to `ParameterPorts <ParameterPort>` of ``my_wave_mech``, which
    that be used to modify their values by `ControlSignals <ControlSignal>` (see `example below <_
    UDF_Control_Signal_Example>`).

    .. _UDF_Explicit_Creation_Examples:

    In all of the examples above, a UDF was automatically created for the functions assigned to the Mechanism.  A UDF
    can also be created explicitly, as follows:

        >>> my_sinusoidal_UDF = pnl.UserDefinedFunction(custom_function=my_sinusoidal_fct)
        >>> my_wave_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                        function=my_sinusoidal_UDF)

    When the UDF is created explicitly, parameters of the function can be included as arguments to its constructor,
    to assign them default values that differ from the those in the definition of the function, or for parameters
    that don't define default values.  For example::

        >>> my_sinusoidal_UDF = pnl.UserDefinedFunction(custom_function=my_sinusoidal_fct,
        ...                                  phase=10,
        ...                                  amplitude=3)
        >>> my_wave_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                        function=my_sinusoidal_UDF)

    assigns ``my_sinusoidal_fct`` as the `function <Mechanism_Base.function>` for ``my_mech``, but with the default
    values of its ``phase`` and ``amplitude`` parameters assigned new values.  This can be useful for assigning the
    same function to different Mechanisms with different default values.

    .. _UDF_Control_Signal_Example:

    Explicitly defining the UDF can also be used to `specify control <ControlSignal_Specification>` for parameters of
    the function, as in the following example::

        >>> my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=UserDefinedFunction(custom_function=my_sinusoidal_fct,
        ...                                                                amplitude=(1.0, pnl.CONTROL)))

    This specifies that the default value of the ``amplitude`` parameter of ``my_sinusoidal_fct`` be ``1.0``, but
    its value should be modulated by a `ControlSignal`.

    COMMENT:
    Note:  if a function explicitly defined in a UDF does not assign a default value to its first argument (i.e.,
    it is a positional argument), then the UDF that must define the variable, as in:

    Note:  if the function does not assign a default value to its first argument i.e., it is a positional arg),
    then if it is explicitly wrapped in a UDF that must define the variable, as in:
        xxx my_mech = pnl.ProcessingMechanism(default_variable=[[0],[0]],
        ...                                   function=UserDefinedFunction(default_variable=[[0],[0]],
        ...                                                                custom_function=my_sinusoidal_fct,
        ...                                                                amplitude=(1.0, pnl.CONTROL)))

    This is required so that the format of the variable can be checked for compatibilty with other Components
    with which it interacts.

    .. note::
       Built-in Python functions and methods (including numpy functions) cannot be assigned to a UDF

    COMMENT

    Custom functions can be as elaborate as desired, and can even include other PsyNeuLink `Functions <Function>`
    indirectly, such as::

        >>> import psyneulink as pnl
        >>> L = pnl.Logistic(gain = 2)
        >>> def my_fct(variable):
        ...     return L(variable) + 2
        >>> my_mech = pnl.ProcessingMechanism(size = 3, function = my_fct)
        >>> my_mech.execute(input = [1, 2, 3])  #doctest: +SKIP
        array([[2.88079708, 2.98201379, 2.99752738]])


    .. _UDF_Assign_to_Port_Examples:

    **Assigning of a custom function to a Port**

    A custom function can also be assigned as the `function <Port_Base.function>` of an `InputPort` or `OutputPort`.
    For example, the following assigns ``my_sinusoidal_fct`` to the `function <OutputPort.function>` of an OutputPort
    of ``my_mech``, rather the Mechanism's `function <Mechanism_Base.function>`::

        >>> my_wave_mech = pnl.ProcessingMechanism(size=1,
        ...                                        function=pnl.Linear,
        ...                                        output_ports=[{pnl.NAME: 'SINUSOIDAL OUTPUT',
        ...                                                       pnl.VARIABLE: [(pnl.OWNER_VALUE, 0),pnl.EXECUTION_COUNT],
        ...                                                       pnl.FUNCTION: my_sinusoidal_fct}])

    For details on how to specify a function of an OutputPort, see `OutputPort Customization <OutputPort_Customization>`.
    Below is an example plot of the output of the 'SINUSOIDAL OUTPUT' `OutputPort` from my_wave_mech above, as the
    execution count increments, when the input to the mechanism is 0.005 for 1000 runs::

.. figure:: _static/sinusoid_005.png
   :alt: Sinusoid function
   :scale: 50 %

.. _UDF_Modulatory_Params_Examples:

    The parameters of a custom function assigned to an InputPort or OutputPort can also be used for `gating
    <GatingMechanism_Specifying_Gating>`.  However, this requires that its `Function_Modulatory_Params` be specified
    (see `above <UDF_Modulatory_Params>`). This can be done by including a **params** argument in the definition of
    the function itself::

        >>> def my_sinusoidal_fct(input=[[0],[0]],
        ...                      phase=0,
        ...                      amplitude=1,
        ...                      params={pnl.ADDITIVE_PARAM:'phase',
        ...                              pnl.MULTIPLICATIVE_PARAM:'amplitude'}):
        ...    frequency = input[0]
        ...    t = input[1]
        ...    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    or in the explicit creation of a UDF::

        >>> my_sinusoidal_UDF = pnl.UserDefinedFunction(custom_function=my_sinusoidal_fct,
        ...                                             phase=0,
        ...                                             amplitude=1,
        ...                                             params={pnl.ADDITIVE_PARAM:'phase',
        ...                                                     pnl.MULTIPLICATIVE_PARAM:'amplitude'})


    The ``phase`` and ``amplitude`` parameters of ``my_sinusoidal_fct`` can now be used as the
    `Function_Modulatory_Params` for gating any InputPort or OutputPort to which the function is assigned (see
    `GatingMechanism_Specifying_Gating` and `GatingSignal_Examples`).

    **Class Definition:**


    Arguments
    ---------

    COMMENT:
        CW 1/26/18: Again, commented here is the old version, because I'm afraid I may have missed some functionality.
        custom_function : function
        specifies function to "wrap." It can be any function, take any arguments (including standard ones,
        such as :keyword:`params` and :keyword:`context`) and return any value(s), so long as these are consistent
        with the context in which the UserDefinedFunction will be used.
    COMMENT
    custom_function : function
        specifies the function to "wrap." It can be any function or method, including a lambda function;
        see `above <UDF_Description>` for additional details.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the function.
        This can be used to define an `additive_param <UserDefinedFunction.additive_param>` and/or
        `multiplicative_param <UserDefinedFunction.multiplicative_param>` for the UDF, by including one or both
        of the following entries:\n
          *ADDITIVE_PARAM*: <param_name>\n
          *MULTIPLICATIVE_PARAM*: <param_name>\n
        Values specified for parameters in the dictionary override any assigned to those parameters in arguments of
        the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable: value
        format and default value of the function "wrapped" by the UDF.

    custom_function : function
        the user-specified function: called by the Function's `owner <Function_Base.owner>` when it is executed.

    additive_param : str
        this contains the name of the additive_param, if one has been specified for the UDF
        (see `above <UDF_Modulatory_Params>` for details).

    multiplicative_param : str
        this contains the name of the multiplicative_param, if one has been specified for the UDF
        (see `above <UDF_Modulatory_Params>` for details).

    COMMENT:
    enable_output_type_conversion : Bool : False
        specifies whether `function output type conversion <Function_Output_Type_Conversion>` is enabled.

    output_type : FunctionOutputType : None
        used to specify the return type for the `function <Function_Base.function>`;  `functionOuputTypeConversion`
        must be enabled and implemented for the class (see `FunctionOutputType <Function_Output_Type_Conversion>`
        for details).
    COMMENT

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Function; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
    """

    componentName = USER_DEFINED_FUNCTION
    componentType = USER_DEFINED_FUNCTION_TYPE

    class Parameters(Function_Base.Parameters):
        """
            Attributes
            ----------

                custom_function
                    see `custom_function <UserDefinedFunction.custom_function>`

                    :default value: None
                    :type:
        """
        custom_function = Parameter(
            None,
            stateful=False,
            loggable=False,
        )

    @tc.typecheck
    def __init__(self,
                 custom_function=None,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs: tc.optional(is_pref_set) = None,
                 **kwargs):

        def get_cust_fct_args(custom_function):
            """Get args of custom_function
            Return:
                - value of first arg (to be used as default_variable for UDF)
                - dict with all others (to be assigned as params of UDF)
                - dict with default values (from function definition, else set to None)
            """
            try:
                arg_names = custom_function.__code__.co_varnames
            except AttributeError:
                raise FunctionError("Can't get __code__ for custom_function")
            args = {}
            defaults = {}
            for arg_name, arg in signature(custom_function).parameters.items():

                # MODIFIED 3/6/19 NEW: [JDC]
                # Custom function specified owner as arg
                if arg_name in {SELF, OWNER, CONTEXT}:
                    # Flag for inclusion in call to function
                    if arg_name == SELF:
                        self.self_arg = True
                    elif arg_name == OWNER:
                        self.owner_arg = True
                    else:
                        self.context_arg = True
                    # Skip rest, as these don't need to be params
                    continue
                # MODIFIED 3/6/19 END

                # Use definition from the function as default;
                #    this allows UDF to assign a value for this instance (including a MODULATORY spec)
                #    while assigning an actual value to current/defaults
                if arg.default is _empty:
                    defaults[arg_name] = None

                else:
                    defaults[arg_name] = arg.default

                # If arg is specified in the constructor for the UDF, assign that as its value
                if arg_name in kwargs:
                    args[arg_name] = kwargs[arg_name]
                # Otherwise, use the default value from the definition of the function
                else:
                    args[arg_name] = defaults[arg_name]

            # Assign default value of first arg as variable and remove from dict
            variable = args[arg_names[0]]
            if variable is _empty:
                variable = None
            del args[arg_names[0]]

            return variable, args, defaults

        self.self_arg = False
        self.owner_arg = False
        self.context_arg = False

        # Get variable and names of other any other args for custom_function and assign to cust_fct_params
        if params is not None and CUSTOM_FUNCTION in params:
            custom_function = params[CUSTOM_FUNCTION]
        try:
            cust_fct_variable, self.cust_fct_params, defaults = get_cust_fct_args(custom_function)
        except FunctionError:
            raise FunctionError("Assignment of a built-in function or method ({}) to a {} is not supported".
                                format(custom_function, self.__class__.__name__))

        # If params is specified as arg in custom function's definition, move it to params in UDF's constructor
        if PARAMS in self.cust_fct_params:
            if self.cust_fct_params[PARAMS]:
                if params:
                    params.update(self.cust_fct_params)
                else:
                    params = self.cust_fct_params[PARAMS]
            del self.cust_fct_params[PARAMS]

        # If context is specified as arg in custom function's definition, delete it
        if CONTEXT in self.cust_fct_params:
            if self.cust_fct_params[CONTEXT]:
                context = self.cust_fct_params[CONTEXT]
            del self.cust_fct_params[CONTEXT]

        # Assign variable to default_variable if default_variable was not specified
        if default_variable is None:
            default_variable = cust_fct_variable
        elif cust_fct_variable and not iscompatible(default_variable, cust_fct_variable):
            owner_name = ' ({})'.format(owner.name) if owner else ''
            cust_fct_name = repr(custom_function.__name__)
            raise FunctionError("Value passed as \'default_variable\' for {} {} ({}) conflicts with specification of "
                                "first argument in constructor for {} itself ({}). "
                                "Try modifying specification of \'default_variable\' "
                                "for object to which {} is being assigned{}, and/or insuring that "
                                "the first argument of {} is at least a 2d array".
                                format(self.__class__.__name__, cust_fct_name, default_variable,
                                       cust_fct_name, cust_fct_variable, cust_fct_name, owner_name, cust_fct_name))

        super().__init__(
            default_variable=default_variable,
            custom_function=custom_function,
            params=params,
            owner=owner,
            prefs=prefs,
            **self.cust_fct_params
        )

    def _handle_illegal_kwargs(self, **kwargs):
        super()._handle_illegal_kwargs(
            **{k: kwargs[k] for k in kwargs if k not in self.cust_fct_params}
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        pass

    def _instantiate_attributes_before_function(self, function=None, context=None):
        super()._instantiate_attributes_before_function(function=function, context=context)
        # create transient Parameters objects for custom function params
        # done here because they need to be present before _instantiate_value which calls self.function
        for param_name in self.cust_fct_params:
            p = Parameter(self.cust_fct_params[param_name], modulable=True)
            setattr(self.parameters, param_name, p)

            p._set(p.default_value, context, skip_history=True)

    def _function(self, variable, context=None, **kwargs):

        # Update value of parms in cust_fct_params
        for param in self.cust_fct_params:

            # First check for value passed in params as runtime param:
            if PARAMS in kwargs and kwargs[PARAMS] is not None and param in kwargs[PARAMS]:
                self.cust_fct_params[param] = kwargs[PARAMS][param]
            else:
                # Otherwise, get current value from ParameterPort (in case it is being modulated by ControlSignal(s)
                self.cust_fct_params[param] = self._get_current_function_param(param, context)

        call_params = self.cust_fct_params.copy()

        # # MODIFIED 3/6/19 NEW: [JDC]
        # Add any of these that were included in the definition of the custom function:
        if self.self_arg:
            call_params[SELF] = self
        if self.owner_arg:
            call_params[OWNER] = self.owner
        if self.context_arg:
            call_params[CONTEXT] = context
        # MODIFIED 3/6/19 END

        kwargs.update(call_params)

        try:
            # Try calling with full list of args (including context and params)
            value = self.custom_function(variable, **kwargs)
        except TypeError:
            # Try calling with just variable and cust_fct_params
            value = self.custom_function(variable, **call_params)

        return self.convert_output_type(value)

    def _gen_llvm_function_body(self, ctx, builder, params, state,
                                arg_in, arg_out, *, tags:frozenset):

        # Instantiate needed ctypes
        arg_in_ct = pnlvm._convert_llvm_ir_to_ctype(arg_in.type.pointee)
        params_ct = pnlvm._convert_llvm_ir_to_ctype(params.type.pointee)
        state_ct = pnlvm._convert_llvm_ir_to_ctype(state.type.pointee)
        arg_out_ct = pnlvm._convert_llvm_ir_to_ctype(arg_out.type.pointee)
        wrapper_ct = ctypes.CFUNCTYPE(None,
                                      ctypes.POINTER(params_ct),
                                      ctypes.POINTER(state_ct),
                                      ctypes.POINTER(arg_in_ct),
                                      ctypes.POINTER(arg_out_ct))

        # we don't support passing any stateful params
        for i, p in enumerate(self.llvm_state_ids):
            assert p not in self.cust_fct_params

        def _carr_to_list(carr):
            try:
                return [_carr_to_list(x) for x in carr]
            except TypeError:
                return carr

        def _assign_to_carr(carr, vals):
            assert len(carr) == len(vals)
            for i, x in enumerate(vals):
                try:
                    carr[i] = x
                except TypeError:
                    _assign_to_carr(carr[i], x)

        def _wrapper(params, state, arg_in, arg_out):
            variable = _carr_to_list(arg_in.contents)

            llvm_params = {}
            for i, p in enumerate(self.llvm_param_ids):
                if p in self.cust_fct_params:
                    field_name = params.contents._fields_[i][0]
                    val = getattr(params.contents, field_name)
                    llvm_params[p] = val

            if self.context_arg:
                # FIXME: We can't get the context
                #        and do not support runtime params
                llvm_params[CONTEXT] = None
                llvm_params[PARAMS] = None

            value = self.custom_function(np.asfarray(variable), **llvm_params)
            _assign_to_carr(arg_out.contents, np.atleast_2d(value))

        self.__wrapper_f = wrapper_ct(_wrapper)
        # To get the right callback pointer, we need to cast to void*
        wrapper_address = ctypes.cast(self.__wrapper_f, ctypes.c_void_p)
        # Direct pointer constants don't work
        wrapper_ptr = builder.inttoptr(pnlvm.ir.IntType(64)(wrapper_address.value), builder.function.type)
        builder.call(wrapper_ptr, [params, state, arg_in, arg_out])

        return builder
