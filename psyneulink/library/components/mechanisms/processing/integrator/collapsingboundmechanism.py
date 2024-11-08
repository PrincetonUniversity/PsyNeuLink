# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  IntegratorMechanism *************************************************

"""

Contents
--------

  * `IntegratorMechanism_Overview`
  * `IntegratorMechanism_Creation`
  * `IntegratorMechanism_Structure`
  * `IntegratorMechanism_Execution`
  * `IntegratorMechanism_Class_Reference`


.. _IntegratorMechanism_Overview:

Overview
--------

An IntegratorMechanism integrates its input, possibly based on its prior values.  The input can be a single
scalar value or an array of scalars (list or 1d np.array).  If it is a list or array, then each value is
independently integrated.  The default function (`IntegratorFunction`) can be parametrized to implement either a simple
increment rate, additive accumulator, or an (exponentially weighted) time-averaging of its input.  It can also be
assigned a custom function.

.. _IntegratorMechanism_Creation:

Creating an IntegratorMechanism
-------------------------------

An IntegratorMechanism can be created directly by calling its constructor, or using the `mechanism` command and
specifying *INTEGRATOR_MECHANISM* as its **mech_spec** argument.  Its function is specified in the **function**
argument, which can be parametrized by calling its constructor with parameter values::

    >>> import psyneulink as pnl
    >>> my_time_averaging_mechanism = pnl.IntegratorMechanism(function=pnl.AdaptiveIntegrator(rate=0.5))

The **default_variable** argument specifies the format of its input (i.e., whether it is a single scalar or an
array), as well as the value to use if none is provided when Mechanism is executed.  Alternatively, the **input_shapes**
argument can be used to specify the length of the array, in which case it will be initialized with all zeros.

.. _IntegratorMechanism_Structure:

Structure
---------

An IntegratorMechanism has a single `InputPort`, the `value <InputPort.InputPort.value>` of which is
used as the  `variable <Mechanism_Base.variable>` for its `function <IntegratorMechanism.function>`.
The default for `function <IntegratorMechanism.function>` is `AdaptiveIntegrator(rate=0.5)`. However,
a custom function can also be specified,  so long as it takes a numeric value, or a list or np.ndarray of numeric
values as its input, and returns a value of the same type and format.  The Mechanism has a single `OutputPort`,
the `value <OutputPort.OutputPort.value>` of which is assigned the result of  the call to the Mechanism's
`function  <IntegratorMechanism.function>`.

.. _IntegratorMechanism_Execution:

Execution
---------

When an IntegratorMechanism is executed, it carries out the specified integration, and assigns the result to the
`value <Mechanism_Base.value>` of its `primary OutputPort <OutputPort_Primary>`.  For the default function
(`IntegratorFunction`), if the value specified for **default_variable** is a list or array, or **input_shapes** is greater
than 1, each element of the array is independently integrated.  If its `rate <IntegratorFunction.rate>` parameter is a
single value, that rate is used for integrating each element. If the `rate <IntegratorFunction.rate>` parameter is a
list or array, then each element is used as the rate for the corresponding element of the input (in this case, `rate
<IntegratorFunction.rate>` must be the same length as the value specified for **default_variable** or **input_shapes**).
Integration can be reset to the value of its `function <IntegratorMechanism.function>`\\s `initializer by setting
its `reset <IntegratorMechanism.reset>` parameter to a non-zero value, as described below.

.. _IntegratorMechanism_Reset:

*Resetting the IntegratorMechanism*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An IntegatorMechanism has a `modulable <ModulatorySignal_Modulation>` `reset <IntergatorMechanism.reset>` parameter
that can be used to reset its value to the value of its `function <IntegratorMechanism.function>`\\s `initializer
<IntegratorFunction.initializer>`. This also clears the `value <Mechanism_Base.value>` `history <Parameter.history>`,
thus effectively setting the `previous_value <IntegratorFunction.previous_value>`  of its `function
<IntegratorMechanism.function>` to None.

The `reset <IntegratorMechanism.reset>` parameter can be used to reset the IntegratorMechanism under the control of a
`ControlMechanism`.  This simplest way to do this is to specify the `reset <IntegratorMechanism.reset>` parameter of
the IntgeratorMechanism in the **control** argument of the ControlMechanism's constructor, and to specify *OVERRIDE*
in its **modulation** argument, as in the following example::

    >>> my_integrator = IntegratorMechanism()
    >>> ctl_mech = pnl.ControlMechanism(modulation=pnl.OVERRIDE, control=(pnl.RESET, my_integrator))

In this case, any non-zero value of the ControlMechanism's `ControlSignal` will reset the IntegratorMechanism.
*OVERRIDE* must be used as its `modulation <ControlMechanism.modulation>` parameter (instead of its default value
of *MULTIPLICATIVE*), so that the value of the ControlMechanism's `ControlSignal` is assigned directly to the
IntegratorMechanism's `reset <IntegratorMechanism.reset>` parameter (otherwise, since the default of the `reset
<IntegratorMechanism.reset>` parameter is 0, the ControlSignal's value has no effect). An alternative is to specify
the **reset_default** agument in the IntegratorMechanism constructor with a non-zero value, and while allowing the
ControlMechanism to use its default value for `modulation <ControlMechanism.modulation>` (i.e., *MULTIPLICATIVE*)::

    >>> my_integrator = IntegratorMechanism(reset_default=1)
    >>> ctl_mech = pnl.ControlMechanism(control=(pnl.RESET, my_integrator))

In this case, a ControlSignal with a zero value suppresses a reset by multiplying the `reset
<IntegratorMechanism.reset>` parameter by 0, whereas a ControlSignal with a non-zero value multiples the `reset
<IntegratorMechanism.reset>` parameter's non-zero default value, resulting in a non-zero value that elicits a reset.

.. _IntegratorMechanism_Class_Reference:

Class Reference
---------------

"""
from collections.abc import Iterable

from beartype import beartype

from psyneulink._typing import Optional, Union
import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.stateful.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, INTEGRATOR_MECHANISM, VARIABLE, PREFERENCE_SET_NAME, RESET
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'DEFAULT_RATE', 'IntegratorMechanism', 'IntegratorMechanismError'
]

# IntegratorMechanism parameter keywords:
DEFAULT_RATE = 0.5

class IntegratorMechanismError(MechanismError):
    pass


class IntegratorMechanism(ProcessingMechanism_Base):
    """
    IntegratorMechanism( \
        function=AdaptiveIntegrator(rate=0.5))

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that integrates its input.
    See `Mechanism <Mechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    function : IntegratorFunction : default IntegratorFunction
        specifies the function used to integrate the input.  Must take a single numeric value, or a list or np.array
        of values, and return one of the same form.

    reset_default : number, list or np.ndarray : default 0
        specifies the default value used for the `reset <IntegratorMechanism.reset>` parameter.

    Attributes
    ----------

    reset : int, float or 1d array of length 1 : default 0
        if non-zero, the IntegratorMechanism's `reset <Mechanism_Base.reset>` method is called, which resets the
        `value <IntegratorMechanism.value>` of the IntegratorMechanism to its initial value (see
        `IntegratorMechanism_Reset` for additional details).

    """

    componentType = INTEGRATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'IntegratorMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class Parameters(ProcessingMechanism_Base.Parameters):
        """
            Attributes
            ----------

                function
                    see `function <IntegratorMechanism.function>`

                    :default value: `AdaptiveIntegrator`(initializer=numpy.array([0]), rate=0.5)
                    :type: `Function`

                reset
                    see `reset <IntegratorMechanism.reset>`

                    :default value: None
                    :type: 'list or np.ndarray'
        """
        function = Parameter(AdaptiveIntegrator(rate=0.5), stateful=False, loggable=False)
        reset = Parameter([0], modulable=True, constructor_argument='reset_default')

        #
    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 input_shapes=None,
                 input_ports:Optional[Union[list, dict]]=None,
                 function=None,
                 reset_default=0,
                 output_ports:Optional[Union[str, Iterable]]=None,
                 params=None,
                 name=None,
                 prefs:   Optional[ValidPrefSet] = None,
                 **kwargs):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        super(IntegratorMechanism, self).__init__(default_variable=default_variable,
                                                  input_shapes=input_shapes,
                                                  function=function,
                                                  reset_default=reset_default,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  input_ports=input_ports,
                                                  output_ports=output_ports,
                                                  **kwargs)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED

    # def _parse_function_variable(self, variable, context=None, context=None):
    #     super()._parse_function_variable(variable, context, context)

    def _handle_default_variable(self, default_variable=None, input_shapes=None, input_ports=None, function=None, params=None):
        """If any parameters with len>1 have been specified for the Mechanism's function, and Mechanism's
        default_variable has not been specified, reshape Mechanism's variable to match function's,
        but make sure function's has the same outer dimensionality as the Mechanism's
        """

        # Get variable for Mechanism
        user_specified = False
        if default_variable is not None:
            variable = np.atleast_1d(default_variable)
            user_specified = True
        else:
            variable = self.parameters.variable.default_value
            user_specified = self.parameters.variable._user_specified

        # Only bother if an instantiated function was specified for the Mechanism
        if isinstance(function, Function):
            function_variable = function.parameters.variable.default_value
            function_variable_len = function_variable.shape[-1]
            variable_len = variable.shape[-1]

            # Raise error if:
            # - the length of both Mechanism and function variable are greater than 1 and they don't match, or
            # - the Mechanism's variable length is 1 and the function's is > 1 (in which case would like to assign
            #   shape of function's variable to that of Mechanism) but Mechanism's variable is user-specified.
            if ((variable_len>1 and function_variable_len>1 and variable_len!=function_variable_len) or
                (function_variable_len>1 and variable_len==1 and user_specified)):
                raise IntegratorMechanismError(f"Shape of {repr(VARIABLE)} for function specified for {self.name} "
                                               f"({function.name}: {function.variable.shape}) does not match "
                                               f"the shape of the {repr(DEFAULT_VARIABLE)} specified for the "
                                               f"{repr(Mechanism.__name__)}.")

            # If length of Mechanism's variable is 1 but the function's is longer,
            #     reshape Mechanism's inner dimension to match function
            elif variable_len==1 and function_variable_len>1:
                variable_shape = list(variable.shape)
                variable_shape[-1] = function_variable.shape[-1]
                # self.parameters.variable.default_value = np.zeros(tuple(variable_shape))
                variable = np.zeros(tuple(variable_shape))
            else:
                variable = default_variable
        else:
            variable = default_variable

            # IMPLEMENTATON NOTE:
            #    Don't worry about case in which length of function's variable is 1 and Mechanism's is > 1
            #    as the reshaping of the function's variable will be taken care of in _instantiate_function

        return super()._handle_default_variable(default_variable=variable,
                                                input_shapes=input_shapes,
                                                input_ports=input_ports,
                                                function=function,
                                                params=params)

    def _execute(self, variable=None, context=None, runtime_params=None, **kwargs):
        """Override to check for call to reset by ControlSignal"""
        # IMPLEMENTATION NOTE:
        #  This could be augmented to use reset parameter value as argument to reset()
        #  if it is the same shape an an initializer for the Mechanism
        value = super()._execute(variable=variable, context=context, runtime_params=runtime_params, **kwargs)
        # No need to reset during initialization (which will occur if **reset_default** != 0)
        if not self.is_initializing:
            if np.array(self._get_current_parameter_value(RESET,context)).squeeze():
                self.reset(context=context)
                value = self.parameters.value._get(context).reshape(value.shape)
        return value
