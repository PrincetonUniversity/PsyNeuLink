# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  UnrestrictedProcessingMechanism *************************************************

"""
Overview
--------

An UnrestrictedProcessingMechanism is the simplest mechanism in PsyNeuLink. It does not have any extra arguments or
specialized validation. Any valid PsyNeuLink Function, including a `UserDefinedFunction` may be the function of an
UnrestricedProcessingMechanism.

.. _UnrestrictedProcessingMechanism_Creation:

Creating an UnrestrictedProcessingMechanism
-------------------------------

An UnrestrictedProcessingMechanism can be created directly by calling its constructor. Its function is specified in the
**function** argument, which can be parametrized by calling its constructor with parameter values::

    >>> import psyneulink as pnl
    >>> my_unrestricted_mechanism = pnl.UnrestrictedProcessingMechanism(function=pnl.Linear)

COMMENT:

.. _UnrestrictedProcessingMechanism_Structure

Structure
---------

An UnrestrictedProcessingMechanism has a single `InputState`, the `value <InputState.InputState.value>` of which is
used as the  `variable <UnrestrictedProcessingMechanism.variable>` for its `function <UnrestrictedProcessingMechanism.function>`.
The default for `function <UnrestrictedProcessingMechanism.function>` is `Linear`. However,
a custom function can also be specified,  so long as it takes a numeric value, or a list or np.ndarray of numeric
values as its input, and returns a value of the same type and format.  The Mechanism has a single `OutputState`,
the `value <OutputState.OutputState.value>` of which is assigned the result of  the call to the Mechanism's
`function  <UnrestrictedProcessingMechanism.function>`.

.. _UnrestrictedProcessingMechanism_Execution

Execution
---------

When an UnrestrictedProcessingMechanism is executed, it carries out the specified integration, and assigns the
result to the `value <UnrestrictedProcessingMechanism.value>` of its `primary OutputState <OutputState_Primary>`.  For the default
function (`Integrator`), if the value specified for **default_variable** is a list or array, or **size** is greater
than 1, each element of the array is independently integrated.  If its `rate <Integrator.rate>` parameter is a
single value,  that rate will be used for integrating each element.  If the `rate <Integrator.rate>` parameter is a
list or array, then each element will be used as the rate for the corresponding element of the input (in this case,
`rate <Integrator.rate>` must be the same length as the value specified for **default_variable** or **size**).

COMMENT

.. _UnrestrictedProcessingMechanism_Class_Reference:

Class Reference
---------------

"""
import typecheck as tc

from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism_Base
from psyneulink.globals.keywords import UNRESTRICTED_PROCESSING_MECHANISM, OUTPUT_STATES, PREDICTION_MECHANISM_OUTPUT, kwPreferenceSetName
from psyneulink.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel


__all__ = [
    'DEFAULT_RATE', 'UnrestrictedProcessingMechanism', 'UnrestrictedProcessingMechanismError'
]

# UnrestrictedProcessingMechanism parameter keywords:
DEFAULT_RATE = 0.5

class UnrestrictedProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class UnrestrictedProcessingMechanism(ProcessingMechanism_Base):
    """
    UnrestrictedProcessingMechanism(                            \
    default_variable=None,                               \
    size=None,                                              \
    function=Linear, \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Subclass of `ProcessingMechanism <ProcessingMechanism>` that does not have any specialized features.

    Arguments
    ---------

    default_variable : number, list or np.ndarray
        the input to the Mechanism to use if none is provided in a call to its
        `execute <Mechanism_Base.execute>` or `run <Mechanism_Base.run>` methods;
        also serves as a template to specify the length of `variable <UnrestrictedProcessingMechanism.variable>` for
        `function <UnrestrictedProcessingMechanism.function>`, and the `primary outputState <OutputState_Primary>` of the
        Mechanism.

    size : int, list or np.ndarray of ints
        specifies default_variable as array(s) of zeros if **default_variable** is not passed as an argument;
        if **default_variable** is specified, it takes precedence over the specification of **size**.
        As an example, the following mechanisms are equivalent::
            T1 = TransferMechanism(size = [3, 2])
            T2 = TransferMechanism(default_variable = [[0, 0, 0], [0, 0]])

    function : PsyNeuLink Function : default Linear
        specifies the function used to compute the output

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, parameters for its `function <UnrestrictedProcessingMechanism.function>`, and/or a custom function and its
        parameters.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    name : str : default see `name <UnrestrictedProcessingMechanism.name>`
        specifies the name of the UnrestrictedProcessingMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the UnrestrictedProcessingMechanism; see `prefs <UnrestrictedProcessingMechanism.prefs>` for details.

    Attributes
    ----------
    variable : value: default
        the input to Mechanism's ``function``.

    name : str
        the name of the UnrestrictedProcessingMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the UnrestrictedProcessingMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentType = UNRESTRICTED_PROCESSING_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'UnrestrictedProcessingMechanismCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class ClassDefaults(ProcessingMechanism_Base.ClassDefaults):
        # Sets template for variable (input)
        variable = [[0]]

    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        OUTPUT_STATES:[PREDICTION_MECHANISM_OUTPUT]

    })

    from psyneulink.components.functions.function import Linear

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_states:tc.optional(tc.any(list, dict))=None,
                 function=Linear,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        if default_variable is None and size is None:
            default_variable = self.ClassDefaults.variable

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(function=function,
                                                  params=params)

        super(UnrestrictedProcessingMechanism, self).__init__(variable=default_variable,
                                                  size=size,
                                                  input_states=input_states,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  context=self)

    # MODIFIED 6/2/17 NEW:
    @property
    def previous_value(self):
        return self.function_object.previous_value
    # MODIFIED 6/2/17 END


