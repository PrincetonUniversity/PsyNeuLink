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
array), as well as the value to use if none is provided when Mechanism is executed.  Alternatively, the **size**
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

When an IntegratorMechanism is executed, it carries out the specified integration, and assigns the
result to the `value <Mechanism_Base.value>` of its `primary OutputPort <OutputPort_Primary>`.  For the default
function (`IntegratorFunction`), if the value specified for **default_variable** is a list or array, or **size** is greater
than 1, each element of the array is independently integrated.  If its `rate <IntegratorFunction.rate>` parameter is a
single value,  that rate will be used for integrating each element.  If the `rate <IntegratorFunction.rate>` parameter is a
list or array, then each element will be used as the rate for the corresponding element of the input (in this case,
`rate <IntegratorFunction.rate>` must be the same length as the value specified for **default_variable** or **size**).


.. _IntegratorMechanism_Class_Reference:

Class Reference
---------------

"""
from collections.abc import Iterable

import typecheck as tc
import numpy as np

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    DEFAULT_VARIABLE, INTEGRATOR_MECHANISM, RESULTS, VARIABLE, PREFERENCE_SET_NAME
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'DEFAULT_RATE', 'IntegratorMechanism', 'IntegratorMechanismError'
]

# IntegratorMechanism parameter keywords:
DEFAULT_RATE = 0.5

class IntegratorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


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
        """
        function = Parameter(AdaptiveIntegrator(rate=0.5), stateful=False, loggable=False)

        #
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.any(list, dict))=None,
                 function=None,
                 output_ports:tc.optional(tc.any(str, Iterable))=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        super(IntegratorMechanism, self).__init__(default_variable=default_variable,
                                                  size=size,
                                                  function=function,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  input_ports=input_ports,
                                                  output_ports=output_ports,
                                                  **kwargs)

        # IMPLEMENT: INITIALIZE LOG ENTRIES, NOW THAT ALL PARTS OF THE MECHANISM HAVE BEEN INSTANTIATED

    # def _parse_function_variable(self, variable, context=None, context=None):
    #     super()._parse_function_variable(variable, context, context)

    def _handle_default_variable(self, default_variable=None, size=None, input_ports=None, function=None, params=None):
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

            # IMPLEMENTATON NOTE:
            #    Don't worry about case in which length of function's variable is 1 and Mechanism's is > 1
            #    as the reshaping of the function's variable will be taken care of in _instantiate_function

        return super()._handle_default_variable(default_variable=variable,
                                                size=size,
                                                input_ports=input_ports,
                                                function=function,
                                                params=params)
