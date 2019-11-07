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
are available to all ProcessingMechanisms in the next `TRIAL`.

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

A ProcessingMechanism has the same basic structure as a `Mechanism <Mechanism>`.  See the documentation for
individual subtypes of ProcessingMechanism for more specific information about their structure.

.. _ProcessingMechanism_Execution:

Execution
---------

Three main tasks are completed each time a ProcessingMechanism executes:

1. The ProcessingMechanism updates its `InputPort`(s), and their values are used to assemble `variable
<Mechanism_Base.variable>`. Each InputPort `value <InputPort.value>` (often there is only one `InputPort`) is
added to an outer array, such that each item of variable corresponds to an InputPort `value <InputPort.value>`.

2. The ProcessingMechanism's `variable <Mechanism_Base.variable>` is handed off as the input to the
ProcessingMechanism's `function <Mechanism_Base.function>`, and the function executes.

3. The result of the ProcessingMechanism's `function <Mechanism_Base.function>` is placed in the Mechanism's
`value <Mechanism_Base.value>` attribute, and OutputPorts are generated based on `value <Mechanism_Base.value>`.

A ProcessingMechanism may be executed by calling its execute method directly:

    >>> my_simple_processing_mechanism = pnl.ProcessingMechanism()      #doctest: +SKIP
    >>> my_simple_processing_mechanism.execute(1.0)                     #doctest: +SKIP

This option is intended for testing and debugging purposes.  More commonly, a mechanism is executed when the
`Composition` to which it belongs is `run <Composition_Run>`.

"""

from collections.abc import Iterable

import typecheck as tc
import numpy as np

from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.globals.keywords import PROCESSING_MECHANISM, PREFERENCE_SET_NAME
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'ProcessingMechanismError',
]

# ControlMechanismRegistry = {}


class ProcessingMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value



# # These are defined here because STANDARD_DEVIATION AND VARIANCE
# #    are already defined in Keywords in lower case (used as arg for Functions).
# STD_DEV_OUTPUT_PORT_NAME = 'STANDARD_DEVIATION'
# VARIANCE_OUTPUT_PORT_NAME = 'variance'


class ProcessingMechanism_Base(Mechanism_Base):
    # DOCUMENTATION: this is a TYPE and subclasses are SUBTYPES
    #                primary purpose is to implement TYPE level preferences for all processing mechanisms
    #                inherits all attributes and methods of Mechanism -- see Mechanism for documentation
    """Subclass of `Mechanism <Mechanism>` that implements processing in a `Pathway`.

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

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()

    standard_output_ports = Mechanism_Base.standard_output_ports.copy()
    standard_output_port_names = Mechanism_Base.standard_output_port_names.copy()

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

        self.system = None

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
    Subclass of `ProcessingMechanism <ProcessingMechanism>` that does not have any specialized features.
    See Mechanism `Mechanism_Class_Reference` for arguments of constructor and attributes.
    """

    componentType = PROCESSING_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'ProcessingMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    paramClassDefaults = ProcessingMechanism_Base.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:tc.optional(tc.any(list, dict))=None,
                 output_ports:tc.optional(tc.any(str, Iterable))=None,
                 function=Linear,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs):
        # Assign args to params and functionParams dicts
        params = self._assign_args_to_param_dicts(function=function,
                                                  input_ports=input_ports,
                                                  output_ports=output_ports,
                                                  params=params)

        super(ProcessingMechanism, self).__init__(default_variable=default_variable,
                                                  size=size,
                                                  input_ports=input_ports,
                                                  function=function,
                                                  output_ports=output_ports,
                                                  params=params,
                                                  name=name,
                                                  prefs=prefs,
                                                  **kwargs)
