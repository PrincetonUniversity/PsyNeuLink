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
    - `ProcessingMechanism_Configuration`
      - `ProcessingMechanism_Parallel_Processing`
      - `ProcessingMechanism_Divergent_Processing`
      - `ProcessingMechanism_Custom_Processing`
  * `ProcessingMechanism_Structure`
  * `ProcessingMechanism_Execution`
  * `ProcessingMechanism_Class_Reference`


.. _ProcessingMechanism_Overview:

Overview
--------

A ProcessingMechanism is a type of `Mechanism` that transforms its input in some way.  A ProcessingMechanism always
receives its input either from another Mechanism, or from the input to a `Composition` when it is
executed.  Similarly, its output is generally conveyed to another Mechanism or used as the output for a Composition.

The ProcessingMechanism is the simplest mechanism in PsyNeuLink. It does not have any extra arguments or specialized
validation. Almost any PsyNeuLink `Function`, including the `UserDefinedFunction`, may be the function of a
ProcessingMechanism. Currently, the only exception is `BackPropagation`. Subtypes of ProcessingMechanism have more
specialized features, and often have restrictions on which Functions are allowed.

The output of a ProcessingMechanism may also be used by a `ModulatoryMechanism <ModulatoryMechanism>` to modify the
parameters of other components (or its own parameters), as in the case of an `Objective Mechanism <ObjectiveMechanism>`
that is specialized for this purpose.

.. _ProcessingMechanism_Creation:

Creating a ProcessingMechanism
------------------------------

A ProcessingMechanism is created by calling its constructor. By default, a ProcessingMechanism is assigned:

- a single `InputPort`, that receives (and sums the `value <Projection_Base.value>`) of any `Projection`\\s to it;

- `Linear` as its `function <Mechanism_Base.function>`;

- a single `OutputPort`, the `value <OutputPort.value>` of which is the result of the linear transformation
  of its input (which, if no `parameters <Parameters>` are assigned to it function, is the same as its input).

However, it can be configured otherwise, for various forms of processing, as described below.

.. _ProcessingMechanism_Configuration:

*Configuring a ProcessingMechanism*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As with any `Mechanism`, the number of InputPorts can be specified using the **input_ports**, **default_variable** or
**size** arguments of the constructor (see `Mechanism_InputPorts`), and OutputPorts can be specified using the
**output_ports** argument (see `Mechanism_OutputPorts`).  These can be used to configure processing in a variety of
ways. Some common ones are described below (also see `ProcessingMechanism_Examples`).

    .. note::
       All of the configurations below assume that:

       - the ProcessingMechanism's `function <Mechanism_Base.function>` is a `TransferFunction`

       - any specified **InputPorts and/or OutputPorts** use their **default** `function <Port_Base.function>`,

       - the `variable <OutputPort.variable>` of **OutputPorts** is **not specified**.

       If any of these is not the case, then the results may differ from those described below.

.. _ProcessingMechanism_Parallel_Processing:

Parallel Processing
^^^^^^^^^^^^^^^^^^^

- *Multiple InputPorts* and either *no or an equal number of OutputPorts*:
  one `OutputPort` is created for (and given the same name as) each `InputPort`; the `value <OutputPort.value>` of
  each is assigned the result of the transformation of the `value <InputPort.value>` of its corresponding InputPort.
  This exploits the property of a TransferFunction, which preserves the shape of its input, implementing parallel
  processing streams in which the same `function <Mechanism_Base.function>` is used to process the input to each
  InputPort independently of the others (including during `learning <Composition_Learning>`)

.. _ProcessingMechanism_Divergent_Processing:

Divergent Processing
^^^^^^^^^^^^^^^^^^^^

- *One InputPort* and *multiple outputPorts*:
  all OutputPorts receive the result of the ProcessingMechanism's `function <Mechanism_Base.function>`
  applied to the `value <InputPort.value>` of the `InputPort`.

.. _ProcessingMechanism_Custom_Processing:

Custom Processing
^^^^^^^^^^^^^^^^^

- *Multiple* but an *unequal* number of *InputPorts and OutputPorts*:

  - if there are *fewer* OutputPorts than InputPorts, each OutputPort is assigned a name and `value <OutputPort.value>`
    corresponding to the name and result of processing of the input to the corresponding InputPort. In this case, the
    inputs to the additional InputPorts are ignored; however, this can be modified by explicitly specifying the
    `variable <OutputPort.variable>` for the OutputPort(s) (see `OutputPort_Custom_Variable`).

  - if there are *more* OutputPorts than Inputports, then all are assigned a `default name <OutputPort.name>`,
    and a `value <OutputPort.value>` that is the result of processing the input to the *first* (`primary
    <InputPort_Primary>`) InputPort`; this can be modified by explicitly specifying the variable for the OutputPort(s)
    (see `OutputPort_Custom_Variable`).

  .. warning::
     An unequal number of InputPorts and OutputPorts is not supported for `learning <Composition_Learning>`


.. _ProcessingMechanism_Structure:

Structure
---------

A ProcessingMechanism has the same structure as a `Mechanism <Mechanism>`, with the addition of several
`StandardOutputPorts <OutputPort_Standard>` to its `standard_output_ports <ProcessingMechanism.standard_output_ports>`
attribute.

See documentation for individual subtypes of ProcessingMechanism for more specific information about their structure.

.. _ProcessingMechanism_Execution:

Execution
---------

The execution of a ProcessingMechanism follows the same sequence of actions as a standard `Mechanism <Mechanism>`
(see `Mechanism_Execution`).

See `ProcessingMechanism_Parallel_Processing` above for a description of how processing is affect by the number of
InputPorts and OutputPorts.


.. _ProcessingMechanism_Examples:

Examples
--------

The `function <Mechanism_Base.function>` of a ProcessingMechanism is specified in the **function** argument,
which may be the name of a `Function <Function>` class:

    >>> import psyneulink as pnl
    >>> my_linear_processing_mechanism = pnl.ProcessingMechanism(function=pnl.Linear)

in which case all of the function's parameters will be set to their default values.

Alternatively, the **function** argument may be a call to a Function constructor, in which case values may be specified
for the Function's parameters:

    >>> my_logistic_processing_mechanism = pnl.ProcessingMechanism(function=pnl.Logistic(gain=1.0, bias=-4))

COMMENT:
EXMAMPLES OF PARLLEL AND NON_PARALLEL PROCESSING

If the function is *not* a `TransferFunction` then, if not specified otherwise, a single OutputPort is created,
the `value <OutputPort.value>` of which is the result of the function's transformation of the `value
<InputPort.value>` the number of InputPorts required by the function for its input.
FIX: EXAMPLE HERE OF LinearCombination Function
COMMENT



.. _ProcessingMechanism_Class_Reference:

Class Reference
---------------

"""

from collections.abc import Iterable

from beartype import beartype

from psyneulink._typing import Optional, Union
import numpy as np

from psyneulink.core.components.functions.nonstateful.transferfunctions import SoftMax
from psyneulink.core.components.functions.nonstateful.selectionfunctions import OneHot
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base, Mechanism, MechanismError
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.globals.keywords import \
    FUNCTION, MAX_ABS_INDICATOR, MAX_ABS_ONE_HOT, MAX_ABS_VAL, MAX_INDICATOR, MAX_ONE_HOT, MAX_VAL, MEAN, MEDIAN, \
    NAME, PROB, PROCESSING_MECHANISM, PREFERENCE_SET_NAME, STANDARD_DEVIATION, VARIANCE, VARIABLE, OWNER_VALUE
from psyneulink.core.globals.parameters import check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.context import ContextFlags

__all__ = [
    'ProcessingMechanismError',
]


class ProcessingMechanismError(MechanismError):
    pass


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
                                   VARIABLE: OWNER_VALUE,
                                   FUNCTION: SoftMax(output=PROB)}])
    standard_output_port_names = [i['name'] for i in standard_output_ports]

    @check_user_specified
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

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 size=None,
                 input_ports:Optional[Union[Iterable, Mechanism, OutputPort, InputPort]]=None,
                 output_ports:Optional[Union[str, Iterable]]=None,
                 function=None,
                 params=None,
                 name=None,
                 prefs:   Optional[ValidPrefSet] = None,
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

    def _instantiate_output_ports(self, context=None):
        """If no OutputPorts have been specified, len(variable) >1 and function is TransferFunction,
        create an OutputPort for each item of variable
        """
        from psyneulink.core.components.ports.outputport import _instantiate_output_ports
        from psyneulink.core.components.functions.nonstateful.transferfunctions import TransferFunction

        if (len(self.defaults.variable) > 1 and isinstance(self.function, TransferFunction)):
            # More than one InputPort, and funciton is TransferFunction, so implement corresponding OutputPorts
            output_ports = []

            if self.output_ports is None:
                # No OutputPorts have been specified, so:
                # - create one for each item of Mechanism's variable (i.e., InputPort)
                # - name it the same as corresponding InputPort
                for i in range(len(self.defaults.variable)):
                    output_ports.append({NAME: f'{self.input_ports[i].name}',
                                         VARIABLE: (OWNER_VALUE, i)})
                return _instantiate_output_ports(owner=self, output_ports=output_ports, context=context)

            elif len(self.output_ports) <= len(self.defaults.variable):
                # Some OutputPorts have been specified, but fewer than there are items of variable, so:
                for i, output_port in enumerate(self.output_ports):
                    # - for each:
                    #   - if no name is specified, name it the same as corresponding InputPort
                    #   - if no variable is specified, assign the corresponding item of Mechanism's value
                    if isinstance(output_port, str):
                        output_ports.append({NAME: output_port,
                                             VARIABLE: (OWNER_VALUE, i)})
                    elif isinstance(output_port, dict):
                        if not output_port:
                            output_ports.append({NAME: f'{self.input_ports[i].name}',
                                                 VARIABLE: (OWNER_VALUE, i)})
                        else:
                            variable = output_port[VARIABLE] if VARIABLE in output_port else (OWNER_VALUE, i)
                            output_port.update({VARIABLE: variable})
                            if NAME in output_port:
                                output_port.update({NAME: output_port[NAME]})
                            elif isinstance(variable, tuple) and variable[0] is OWNER_VALUE:
                                output_port.update({NAME: f'{self.input_ports[variable[1]].name}'})
                            output_ports.append(output_port)
                    elif isinstance(output_port, OutputPort):
                        if output_port.initialization_status & ContextFlags.DEFERRED_INIT:
                            if output_port._init_args[VARIABLE] is None:
                                # FIX: HACK;  FOR SOME REASON, ASSIGNING (OWNER_VALUE, i) TO VARIABLE ARG DOESN'T WORK
                                output_port._variable_spec = (OWNER_VALUE, i)
                            if output_port._init_args[NAME] is None:
                                if (isinstance(output_port._variable_spec, tuple)
                                        and output_port._variable_spec[0] is OWNER_VALUE):
                                    output_port._init_args[NAME] = \
                                        f'{self.input_ports[output_port._variable_spec[1]].name}'
                        output_ports.append(output_port)

                return _instantiate_output_ports(owner=self, output_ports=output_ports, context=context)

        # Multiple but unequal numbers of InputPorts and OutputPorts, so follow default protocol
        #   (i.e., assign value of all OutputPorts to the first item of Mechanism's value unless otherwise specified)
        return super()._instantiate_output_ports(context=context)
