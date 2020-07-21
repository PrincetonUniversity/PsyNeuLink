# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  ProcessingMechanism ****************************************************

"""

Contents
--------

  * `ProcessingMechanism_Overview`
  * `ProcessingMechanism_Creation`
  * `ProcessingMechanism_Structure`
  * `ProcessingMechanism_Execution`
  * `ProcessingMechanism_Class_Reference`


.. _ProcessingMechanism_Overview:

Overview
--------

A ProcessingMechanism is a type of `Mechanism <>` that transforms its input in some way.  A ProcessingMechanism always
receives its input either from another Mechanism, or from the input to a `Composition` when it is
executed.  Similarly, its output is generally conveyed to another Mechanism or used as the output for a Composition.

The ProcessingMechanism is the simplest mechanism in PsyNeuLink. It does not have any extra arguments or
specialized validation. Almost any PsyNeuLink Function, including the `UserDefinedFunction`, may be the function of a
ProcessingMechanism. Currently, the only exception is `BackPropagation`. Subtypes of
ProcessingMechanism have more specialized features, and often have restrictions on which Functions are allowed.

The output of a ProcessingMechanism may also be used by a `ModulatoryMechanism <ModulatoryMechanism>` to modify the
parameters of other components (or its own parameters). ProcessingMechanisms are always executed before all
ModulatoryMechanisms in the Composition to which they belong, so that any modifications made by the ModulatoryMechanism
are available to all ProcessingMechanisms in the next `TRIAL <TimeScale.TRIAL>`.

.. _ProcessingMechanism_Creation:

Creating a ProcessingMechanism
------------------------------

A ProcessingMechanism is created by calling its constructor.

Its `function <Mechanism_Base.function>` is specified in the **function** argument, which may be the name of a
`Function <Function>` class:

    >>> import psyneulink as pnl
    >>> my_linear_processing_mechanism = pnl.ProcessingMechanism(function=pnl.Linear)

in which case all of the function's parameters will be set to their default values.

Alternatively, the **function** argument may be a call to a Function constructor, in which case values may be specified
for the Function's parameters:

    >>> my_logistic_processing_mechanism = pnl.ProcessingMechanism(function=pnl.Logistic(gain=1.0, bias=-4))


.. _ProcessingMechanism_Structure:

Structure
---------

A ProcessingMechanism has the same structure as a `Mechanism <Mechanism>`, with the addition of several
`StandardOutputPorts <OutputPort_Standard>` to its `standard_output_ports
<ProcessingMechanism.standard_output_ports>` attribute.

See documentation for individual subtypes of ProcessingMechanism for more specific information about their structure.

.. _ProcessingMechanism_Execution:

Execution
---------

The execution of a ProcessingMechanism follows the same sequence of actions as a standard `Mechanism <Mechanism>`
(see `Mechanism_Execution`).

.. _ProcessingMechanism_Class_Reference:

Class Reference
---------------

"""

from collections.abc import Iterable

import typecheck as tc
import numpy as np

from psyneulink.core.components.functions.transferfunctions import Linear, SoftMax
from psyneulink.core.components.functions.selectionfunctions import OneHot
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.globals.keywords import \
    FUNCTION, MAX_ABS_INDICATOR, MAX_ABS_ONE_HOT, MAX_ABS_VAL, MAX_INDICATOR, MAX_ONE_HOT, MAX_VAL, MEAN, MEDIAN, \
    NAME, PROB, PROCESSING_MECHANISM, PREFERENCE_SET_NAME, STANDARD_DEVIATION, VARIANCE
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'ProcessingMechanismError',
]


class ProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


# # These are defined here because STANDARD_DEVIATION AND VARIANCE
# #    are already defined in Keywords in lower case (used as arg for Functions).
# STD_DEV_OUTPUT_PORT_NAME = 'STANDARD_DEVIATION'
# VARIANCE_OUTPUT_PORT_NAME = 'VARIANCE'


class ProcessingMechanism_Base(Mechanism_Base):
    """Subclass of `Mechanism <Mechanism>`.

    This is a TYPE and subclasses are SUBTYPES.  its primary purpose is to implement TYPE level preferences for all
    processing mechanisms.

    .. note::
       ProcessingMechanism_Base is an abstract class and should *never* be instantiated by a call to its constructor.
       It should be instantiated using the constructor for `ProcessingMechanism` or one of its  `subclasses
       <ProcessingMechanism_Subtypes>`.

   """

    componentType = "ProcessingMechanism"

    is_self_learner = False  # CW 11/27/17: a flag; "True" if this mech learns on its own. See use in LeabraMechanism

    classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to TYPE automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ProcessingMechanismClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}


    standard_output_ports = Mechanism_Base.standard_output_ports.copy()
    standard_output_ports.extend([{NAME:MEAN,
                                   FUNCTION:lambda x: np.mean(x)},
                                  {NAME: MEDIAN,
                                   FUNCTION:lambda x: np.median(x)},
                                  {NAME: STANDARD_DEVIATION,
                                   FUNCTION:lambda x: np.std(x)},
                                  {NAME: VARIANCE,
                                   FUNCTION:lambda x: np.var(x)},
                                  {NAME: MAX_VAL,
                                   FUNCTION:lambda x: np.max(x)},
                                  {NAME: MAX_ABS_VAL,
                                   FUNCTION:lambda x: np.max(np.absolute(x))},
                                  {NAME: MAX_ONE_HOT,
                                   FUNCTION: OneHot(mode=MAX_VAL)},
                                  {NAME: MAX_ABS_ONE_HOT,
                                   FUNCTION: OneHot(mode=MAX_ABS_VAL)},
                                  {NAME: MAX_INDICATOR,
                                   FUNCTION: OneHot(mode=MAX_INDICATOR)},
                                  {NAME: MAX_ABS_INDICATOR,
                                   FUNCTION: OneHot(mode=MAX_ABS_INDICATOR)},
                                  {NAME: PROB,
                                   FUNCTION: SoftMax(output=PROB)}])
    standard_output_port_names = [i['name'] for i in standard_output_ports]

    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports=None,
                 function=None,
                 output_ports=None,
                 params=None,
                 name=None,
                 prefs=None,
                 context=None,
                 **kwargs
                 ):
        """Abstract class for processing mechanisms

        :param variable: (value)
        :param size: (int or list/array of ints)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        :param context: (str)
        """

        super().__init__(default_variable=default_variable,
                         size=size,
                         input_ports=input_ports,
                         function=function,
                         output_ports=output_ports,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=context,
                         **kwargs
                         )

    def _validate_inputs(self, inputs=None):
        # Let mechanism itself do validation of the input
        pass

__all__ = [
    'DEFAULT_RATE', 'ProcessingMechanism', 'ProcessingMechanismError'
]

# ProcessingMechanism parameter keywords:
DEFAULT_RATE = 0.5

class ProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ProcessingMechanism(ProcessingMechanism_Base):
    """
    Implements instance of `ProcessingMechanism_Base <ProcessingMechanism>` subclass of `Mechanism <Mechanism>`.
    See `Mechanism <Mechanism_Class_Reference>` and `subclasses <ProcessingMechanism_Subtypes>` of ProcessingMechanism
    for arguments and additional attributes.

    Attributes
    ----------

    standard_output_ports : list[dict]
      list of the dictionary specifications for `StandardOutputPorts <OutputPort_Standard>` that can be assigned as
      `OutputPorts <OutputPort>`, in addition to the `standard_output_ports <Mechanism_Base.standard_output_ports>`
      of a `Mechanism <Mechanism>`; each assigns as the `value <OutputPort.value>` of the OutputPort a quantity
      calculated over the elements of the first item in the outermost dimension (axis 0) of the Mechanism`s `value
      <Mechanism_Base.value>`. `Subclasses <ProcessingMechanism_Subtypes>` of ProcessingMechanism may extend this
      list to include additional `StandardOutputPorts <OutputPort_Standard>`.

     *MEAN* : float
       mean of the elements.

     *MEDIAN* : float
       median of the elements.

     *STANDARD_DEVIATION* : float
       standard deviation of the elements.

     *VARIANCE* : float
       variance of the elements.

     *MAX_VAL* : float
       greatest signed value of the elements.

     *MAX_ABS_VAL* : float
       greatest absolute value of the elements.

     *MAX_ONE_HOT* : float
       element with the greatest signed value is assigned that value, all others are assigned 0.

     *MAX_ABS_ONE_HOT* : float
       element with the greatest absolute value is assigned that value, all others are assigned 0.

     *MAX_INDICATOR* : 1d array
       element with the greatest signed value is assigned 1, all others are assigned 0.

     *MAX_ABS_INDICATOR* : 1d array
       element with the greatest absolute value is assigned 1, all others are assigned 0.

     *PROB* : float
       element chosen probabilistically based on softmax distribution is assigned its value, all others are assigned 0.

    """

    componentType = PROCESSING_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'ProcessingMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.any(list, dict))=None,
                 output_ports:tc.optional(tc.any(str, Iterable))=None,
                 function=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        super(ProcessingMechanism, self).__init__(default_variable=default_variable,
                                                  size=size,
                                                  input_ports=input_ports,
                                                  function=function,
                                                  output_ports=output_ports,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  **kwargs)
