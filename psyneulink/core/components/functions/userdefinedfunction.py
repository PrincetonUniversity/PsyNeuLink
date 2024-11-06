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

import builtins
import numpy as np
from beartype import beartype

from psyneulink._typing import Optional
from inspect import signature, _empty, getsourcelines, getsourcefile, getclosurevars
import ast

from psyneulink.core.components.functions.function import FunctionError, Function_Base
from psyneulink.core.globals.keywords import \
    CONTEXT, CUSTOM_FUNCTION, OWNER, PARAMS, \
    SELF, USER_DEFINED_FUNCTION, USER_DEFINED_FUNCTION_TYPE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences import ValidPrefSet
from psyneulink.core.globals.utilities import _is_module_class, iscompatible

from psyneulink.core import llvm as pnlvm

__all__ = ['UserDefinedFunction']


class _ExpressionVisitor(ast.NodeVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vars = set()
        self.functions = set()

    def visit_Name(self, node):
        if node.id not in dir(builtins):
            self.vars.add(node.id)

    def visit_Call(self, node):
        try:
            # gives top level module name if module function used
            func_id = node.func.value.id
        except AttributeError:
            func_id = node.func.id

        if func_id not in dir(builtins):
            self.functions.add(func_id)

        for c in ast.iter_child_nodes(node):
            self.visit(c)


class UserDefinedFunction(Function_Base):
    """UserDefinedFunction(  \
    custom_function=None,    \
    default_variable=None,   \
    params=None,             \
    owner=None,              \
    name=None,               \
    prefs=None)

    .. _UDF_Description:

    A UserDefinedFunction (UDF) is used to "wrap" a Python function or method, lamdba function,
    or an expression written in string format
    as a PsyNeuLink `Function <Function>`, so that it can be used as the `function <Component.function>` of a
    `Component <Component>`.  This is done automatically if a Python function or method is assigned as the `function
    <Component.function>` attribute of a Component.  A Python function or method can also be wrapped explicitly,
    using the UserDefinedFunction constructor, and assigning the Python function or method to its **custom_function**
    argument.  A Python function or method wrapped as a UDF must obey the following conventions to be treated
    correctly:

    .. _UDF_Variable:

    * If providing a Python function, method, or lambda function, it must have **at least one argument** (that can be a positional or a keyword argument);  this will be treated
      as the `variable <UserDefinedFunction.variable>` attribute of the UDF's `function <UserDefinedFunction.function>`.
      When the UDF calls the function or method that it wraps, an initial attempt is made to do so with **variable**
      as the name of the first argument; if that fails, it is called positionally.  The argument is always passed as a
      2d np.array, that may contain one or more items (elements in axis 0), depending upon the Component to which the
      UDF is assigned.  It is the user's responsibility to insure that the number of items expected in the first
      argument of the function or method is compatible with the circumstances in which it will be called.
      If providing a string expression, **variable** is optional. However, if **variable** is not included in
      the expression, the resulting UDF will not use **variable** at all in its calculation.
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
    This is because  the `variable <Mechanism_Base.variable>` of ``my_mech`` is defined in the **input_shapes** argument of
    its constructor as having a single item (a 1d array of length 3;  (see `input_shapes <Component.input_shapes>`).  In the
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

    .. _UDF_String_Expression_Function_Examples:

    The **function** argument may also be an expression written as a string::

        >>> my_mech = pnl.ProcessingMechanism(function='sum(variable, 2)')
        >>> my_mech.execute(input=[1])
        array([[3]])

    This option is primarily designed for compatibility with other packages that use string expressions as
    their main description of computation and may be less flexible or reliable than the previous styles.

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
        >>> my_mech = pnl.ProcessingMechanism(input_shapes = 3, function = my_fct)
        >>> my_mech.execute(input = [1, 2, 3])  #doctest: +SKIP
        array([[2.88079708, 2.98201379, 2.99752738]])


    .. _UDF_Assign_to_Port_Examples:

    **Assigning of a custom function to a Port**

    A custom function can also be assigned as the `function <Port_Base.function>` of an `InputPort` or `OutputPort`.
    For example, the following assigns ``my_sinusoidal_fct`` to the `function <OutputPort.function>` of an OutputPort
    of ``my_mech``, rather the Mechanism's `function <Mechanism_Base.function>`::

        >>> my_wave_mech = pnl.ProcessingMechanism(input_shapes=1,
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

.. _UDF_Compilation:

    **Compiling a User Defined Function**

    User defined functions may also be `automatically compiled <Composition_Compilation>`, by adding them as a mechanism or projection function.
    There are several restrictions to take into account:

.. _UDF_Compilation_Restrictions:

    * *Lambda Functions* -- User defined functions currently do not support Python Lambda functions

    * *Loops* -- User defined functions currently do not support Loops

    * *Python Data Types* -- User defined functions currently do not support *dict* and *class* types

    * *Nested Functions* -- User defined functions currently do not support recursive calls, nested functions, or closures

    * *Slicing and comprehensions* -- User defined functions currently have limited support for slicing, and do not support comprehensions

    * *Libraries* -- User defined functions currently do not support libraries, aside from **NumPy** (with limited support)

.. _UDF_Compilation_Numpy:

    **NumPy Support for Compiled User Defined Functions**

    Compiled User Defined Functions also provide access to limited compiled NumPy functionality; The supported state_features are listed as follows:

    * *Data Types* -- Numpy Arrays and Matrices are supported, as long as their dimensionality is less than 3. In addition, the elementwise multiplication and addition of NumPy arrays and matrices is fully supported

    * *Numerical functions* -- the `exp` and `tanh` methods are currently supported in compiled mode

    It is highly recommended that users who require additional functionality request it as an issue `here <https://github.com/PrincetonUniversity/PsyNeuLink/issues>`_.

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
            pnl_internal=True,
        )

    @check_user_specified
    @beartype
    def __init__(self,
                 custom_function=None,
                 default_variable=None,
                 params=None,
                 owner=None,
                 prefs:  Optional[ValidPrefSet] = None,
                 stateful_parameter=None,
                 **kwargs):

        def get_cust_fct_args(custom_function):
            """Get args of custom_function
            Return:
                - value of first arg (to be used as default_variable for UDF)
                - dict with all others (to be assigned as params of UDF)
                - dict with default values (from function definition, else set to None)
            """
            try:
                custom_function_signature = signature(custom_function)
            except ValueError:
                raise FunctionError(
                    "Assignment of a function or method ({}) without an "
                    "inspect.signature to a {} is not supported".format(
                        custom_function, self.__class__.__name__
                    )
                )
            except TypeError:
                v = _ExpressionVisitor()
                v.visit(ast.parse(custom_function))
                parameters = v.vars.union(v.functions)

                if 'variable' in parameters:
                    parameters.remove('variable')
                    variable = kwargs['variable']
                else:
                    variable = None

                args = {}
                for p in parameters:
                    if '.' not in p:  # assume . indicates external module function call
                        try:
                            args[p] = kwargs[p]
                        except KeyError:
                            args[p] = None

                return variable, args, args

            args = {}
            defaults = {}
            for arg_name, arg in custom_function_signature.parameters.items():

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
            # .keys is ordered
            first_arg_name = list(custom_function_signature.parameters.keys())[0]
            variable = args[first_arg_name]
            if variable is _empty:
                variable = None
            del args[first_arg_name]

            return variable, args, defaults

        self.self_arg = False
        self.owner_arg = False
        self.context_arg = False

        # Get variable and names of other any other args for custom_function and assign to cust_fct_params
        if params is not None and CUSTOM_FUNCTION in params:
            custom_function = params[CUSTOM_FUNCTION]

        cust_fct_variable, self.cust_fct_params, defaults = get_cust_fct_args(custom_function)

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

        if stateful_parameter is not None:
            if stateful_parameter not in self.cust_fct_params:
                raise FunctionError(
                    f'{stateful_parameter} specified as integration parameter is not a parameter of {custom_function}'
                )
        self.stateful_parameter = stateful_parameter

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

    def _get_allowed_arguments(self):
        return super()._get_allowed_arguments().union(self.cust_fct_params)

    def _validate_params(self, request_set, target_set=None, context=None):
        pass

    def _initialize_parameters(self, context=None, **param_defaults):
        # pass custom parameter values here so they can be created as
        # Parameters in Component._initialize_parameters and
        # automatically handled as if they were normal Parameters
        for param_name in self.cust_fct_params:
            param_defaults[param_name] = Parameter(self.cust_fct_params[param_name], modulable=True)

        super()._initialize_parameters(context, **param_defaults)

    def _function(self, variable, context=None, **kwargs):
        call_params = self.cust_fct_params.copy()

        # Update value of parms in cust_fct_params
        for param in call_params:

            # First check for value passed in params as runtime param:
            if PARAMS in kwargs and kwargs[PARAMS] is not None and param in kwargs[PARAMS]:
                call_params[param] = kwargs[PARAMS][param]
            elif param in kwargs:
                call_params[param] = kwargs[param]
            else:
                # Otherwise, get current value from ParameterPort (in case it is being modulated by ControlSignal(s)
                call_params[param] = self._get_current_parameter_value(param, context)

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
        except TypeError as e:
            if "'str' object is not callable" != str(e):
                # Try calling with just variable and cust_fct_params
                value = self.custom_function(variable, **call_params)
            else:
                value = eval(self.custom_function, kwargs)

        if self.stateful_parameter is not None and not self.is_initializing:
            # use external set here because we don't control custom_function
            getattr(self.parameters, self.stateful_parameter).set(value, context)

        return self.convert_output_type(value)

    def _gen_llvm_function_body(self, ctx, builder, params, state,
                                arg_in, arg_out, *, tags:frozenset):

        # Check for global and nonlocal vars. we can't compile those.
        closure_vars = getclosurevars(self.custom_function)
        assert len(closure_vars.nonlocals) == 0, "Compiling functions with non-local variables is not supported!"

        srcfile = getsourcefile(self.custom_function)
        first_line = getsourcelines(self.custom_function)[1]

        with open(srcfile) as f:
            for node in ast.walk(ast.parse(f.read(), srcfile)):
                if getattr(node, 'lineno', -1) == first_line and isinstance(node, (ast.FunctionDef, ast.Lambda)):
                    func_ast = node
                    break
                func_ast = None

        assert func_ast is not None, "UDF function source code not found"

        func_globals = closure_vars.globals
        assert len(func_globals) == 0 or (
               len(func_globals) == 1 and np in func_globals.values()), \
               "Compiling functions with global variables is not supported! ({})".format(closure_vars.globals)
        func_params = {param_id: ctx.get_param_or_state_ptr(builder, self, param_id, param_struct_ptr=params) for param_id in self.llvm_param_ids}

        pnlvm.codegen.UserDefinedFunctionVisitor(ctx, builder, func_globals, func_params, arg_in, arg_out).visit(func_ast)

        # The generic '_gen_llvm' will append another ret void to this block
        post_block = builder.append_basic_block(name="post_udf")
        builder.position_at_start(post_block)
        return builder

    def as_mdf_model(self):
        import math
        import modeci_mdf.functions.standard

        model = super().as_mdf_model()
        ext_function_str = None

        if self.custom_function in [
            func_dict['function']
            for name, func_dict
            in modeci_mdf.functions.standard.mdf_functions.items()
        ]:
            ext_function_str = self.custom_function.__name__

        if _is_module_class(self.custom_function, math):
            ext_function_str = f'{self.custom_function.__module__}.{self.custom_function.__name__}'

        if ext_function_str is not None:
            model.metadata['custom_function'] = ext_function_str
            del model.metadata['type']

        return model
