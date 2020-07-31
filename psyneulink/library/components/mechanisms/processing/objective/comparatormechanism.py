# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************  ComparatorMechanism ***************************************************

"""

Contents
--------

  * `ComparatorMechanism_Overview`
  * `ComparatorMechanism_Creation`
  * `ComparatorMechanism_Structure`
  * `ComparatorMechanism_Execution`
  * `ComparatorMechanism_Example`
  * `ComparatorMechanism_Class_Reference`


.. _ComparatorMechanism_Overview:

Overview
--------

A ComparatorMechanism is a subclass of `ObjectiveMechanism` that receives two inputs (a sample and a target), compares
them using its `function <ComparatorMechanism.function>`, and places the calculated discrepancy between the two in its
*OUTCOME* `OutputPort <ComparatorMechanism.output_port>`.

.. _ComparatorMechanism_Creation:

Creating a ComparatorMechanism
------------------------------

ComparatorMechanisms are generally created automatically when other PsyNeuLink components are created (such as
`LearningMechanisms <LearningMechanism_Creation>`).  A ComparatorMechanism can also be created directly by calling
its constructor.  Its **sample** and **target** arguments are used to specify the OutputPorts that provide the
sample and target inputs, respectively (see `ObjectiveMechanism_Monitored_ports` for details concerning their
specification, which are special versions of an ObjectiveMechanism's **monitor** argument).  When the
ComparatorMechanism is created, two InputPorts are created, one each for its sample and target inputs (and named,
by default, *SAMPLE* and *TARGET*). Each is assigned a MappingProjection from the corresponding OutputPort specified
in the **sample** and **target** arguments.

It is important to recognize that the value of the *SAMPLE* and *TARGET* InputPorts must have the same length and type,
so that they can be compared using the ComparatorMechanism's `function <ComparatorMechanism.function>`.  By default,
they use the format of the OutputPorts specified in the **sample** and **target** arguments, respectively,
and the `MappingProjection` to each uses an `IDENTITY_MATRIX`.  Therefore, for the default configuration, the
OutputPorts specified in the **sample** and **target** arguments must have values of the same length and type.
If these differ, the **input_ports** argument can be used to explicitly specify the format of the ComparatorMechanism's
*SAMPLE* and *TARGET* InputPorts, to insure they are compatible with one another (as well as to customize their
names, if desired).  If the **input_ports** argument is used, *both* the sample and target InputPorts must be
specified.  Any of the formats for `specifying InputPorts <InputPort_Specification>` can be used in the argument.
If values are assigned for the InputPorts, they must be of equal length and type.  Their types must
also be compatible with the value of the OutputPorts specified in the **sample** and **target** arguments.  However,
the length specified for an InputPort can differ from its corresponding OutputPort;  in that case, by default, the
MappingProjection created uses a `FULL_CONNECTIVITY` matrix.  Thus, OutputPorts of differing lengths can be mapped
to the sample and target InputPorts of a ComparatorMechanism (see the `example <ComparatorMechanism_Example>` below),
so long as the latter are of the same length.  If a projection other than a `FULL_CONNECTIVITY` matrix is needed, this
can be specified using the *PROJECTION* entry of a `Port specification dictionary <Port_Specification>` for the
InputPort in the **input_ports** argument.

.. _ComparatorMechanism_Structure:

Structure
---------

A ComparatorMechanism has two `input_ports <ComparatorMechanism.input_ports>`, each of which receives a
`MappingProjection` from a corresponding OutputPort specified in the **sample** and **target** arguments of its
constructor.  The InputPorts are listed in the Mechanism's `input_ports <ComparatorMechanism.input_ports>` attribute
and named, respectively, *SAMPLE* and *TARGET*.  The OutputPorts from which they receive their projections (specified
in the **sample** and **target** arguments) are listed in the Mechanism's `sample <ComparatorMechanism.sample>` and
`target <ComparatorMechanism.target>` attributes as well as in its `monitor <ComparatorMechanism.monitor>` attribute.
The ComparatorMechanism's `function <ComparatorMechanism.function>` compares the value of the sample and target
InputPorts.  By default, it uses a `LinearCombination` function, assigning the sample InputPort a `weight
<LinearCombination.weight>` of *-1* and the target a `weight <LinearCombination.weight>` of *1*, so that the sample
is subtracted from the target.  However, the `function <ComparatorMechanism.function>` can be customized, so long as
it is replaced with one that takes two arrays with the same format as its inputs and generates a similar array as its
result. The result is assigned as the value of the Comparator Mechanism's *OUTCOME* (`primary <OutputPort_Primary>`)
OutputPort.

.. _ComparatorMechanism_Execution:

Execution
---------

When a ComparatorMechanism is executed, it updates its input_ports with the values of the OutputPorts specified
in its **sample** and **target** arguments, and then uses its `function <ComparatorMechanism.function>` to
compare these.  By default, the result is assigned to the `value <Mechanism_Base.value>` of its *OUTCOME*
`output_port <ComparatorMechanism.output_port>`, and as the first item of the Mechanism's
`output_values <ComparatorMechanism.output_values>` attribute.

.. _ComparatorMechanism_Example:

Example
-------

.. _ComparatorMechanism_Default_Input_Value_Example:

*Formatting InputPort values*

The **default_variable** argument can be used to specify a particular format for the SAMPLE and/or TARGET InputPorts
of a ComparatorMechanism.  This can be useful when one or both of these differ from the format of the
OutputPort(s) specified in the **sample** and **target** arguments. For example, for `Reinforcement Learning
<Reinforcement>`, a ComparatorMechanism is used to monitor an action selection Mechanism (the sample), and compare
this with a reinforcement signal (the target).  In the example below, the action selection Mechanism is a
`TransferMechanism` that uses the `SoftMax` function (and the `PROB <Softmax.PROB>` as its output format) to select
an action.  This generates a vector with a single non-zero value (the selected action). Because the output is a vector,
specifying it as the ComparatorMechanism's **sample** argument will generate a corresponding InputPort with a vector
as its value.  This will not match the reward signal specified in the ComparatorMechanism's **target** argument, the
value of which is a single scalar.  This can be dealt with by explicitly specifying the format for the SAMPLE and
TARGET InputPorts in the **default_variable** argument of the ComparatorMechanism's constructor, as follows::

    >>> import psyneulink as pnl
    >>> my_action_selection_mech = pnl.TransferMechanism(size=5,
    ...                                                  function=pnl.SoftMax(output=pnl.PROB))

    >>> my_reward_mech = pnl.TransferMechanism()

    >>> my_comparator_mech = pnl.ComparatorMechanism(default_variable = [[0],[0]],
    ...                                              sample=my_action_selection_mech,
    ...                                              target=my_reward_mech)

Note that ``my_action_selection_mechanism`` is specified to take an array of length 5 as its input, and therefore
generate one of the same length as its `primary output <OutputPort_Primary>`.  Since it is assigned as the **sample**
of the ComparatorMechanism, by default this will create a *SAMPLE* InputPort of length 5, that will not match the
length of the *TARGET* InputPort (the default for which is length 1).  This is taken care of, by specifying the
**default_variable** argument as an array with two single-value arrays (corresponding to the *SAMPLE* and *TARGET*
InputPorts). (In this example, the **sample** and **target** arguments are specified as Mechanisms since,
by default, each has only a single (`primary <OutputPort_Primary>`) OutputPort, that will be used;  if either had
more than one OutputPort, and one of those was desired, it would have had to be specified explicitly in the
**sample** or **target** argument).

.. _ComparatorMechanism_Class_Reference:

Class Reference
---------------

"""

from collections.abc import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.port import _parse_port_spec
from psyneulink.core.globals.context import Context
from psyneulink.core.globals.keywords import \
    COMPARATOR_MECHANISM, FUNCTION, INPUT_PORTS, NAME, OUTCOME, SAMPLE, TARGET, VARIABLE, PREFERENCE_SET_NAME, MSE, SSE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import \
    is_numeric, is_value_spec, iscompatible, kwCompatibilityLength, kwCompatibilityNumeric, recursive_update
from psyneulink.core.globals.utilities import safe_len

__all__ = [
    'ComparatorMechanism', 'ComparatorMechanismError'
]


class ComparatorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ComparatorMechanism(ObjectiveMechanism):
    """
    ComparatorMechanism(                                \
        sample,                                         \
        target,                                         \
        input_ports=[SAMPLE,TARGET]                     \
        function=LinearCombination(weights=[[-1],[1]],  \
        output_ports=OUTCOME)

    Subclass of `ObjectiveMechanism` that compares the values of two `OutputPorts <OutputPort>`.
    See `ObjectiveMechanism <ObjectiveMechanism_Class_Reference>` for additional arguments and attributes.


    Arguments
    ---------

    sample : OutputPort, Mechanism, value, or string
        specifies the value to compare with the `target` by the `function <ComparatorMechanism.function>`.

    target :  OutputPort, Mechanism, value, or string
        specifies the value with which the `sample` is compared by the `function <ComparatorMechanism.function>`.

    input_ports :  List[InputPort, value, str or dict] or Dict[] : default [SAMPLE, TARGET]
        specifies the names and/or formats to use for the values of the sample and target InputPorts;
        by default they are named *SAMPLE* and *TARGET*, and their formats are match the value of the OutputPorts
        specified in the **sample** and **target** arguments, respectively (see `ComparatorMechanism_Structure`
        for additional details).

    function :  Function, function or method : default Distance(metric=DIFFERENCE)
        specifies the `function <Comparator.function>` used to compare the `sample` with the `target`.


    Attributes
    ----------

    COMMENT:
    default_variable : Optional[List[array] or 2d np.array]
    COMMENT

    sample : OutputPort
        determines the value to compare with the `target` by the `function <ComparatorMechanism.function>`.

    target : OutputPort
        determines the value with which `sample` is compared by the `function <ComparatorMechanism.function>`.

    input_ports : ContentAddressableList[InputPort, InputPort]
        contains the two InputPorts named, by default, *SAMPLE* and *TARGET*, each of which receives a
        `MappingProjection` from the OutputPorts referenced by the `sample` and `target` attributes
        (see `ComparatorMechanism_Structure` for additional details).

    function : CombinationFunction, function or method
        used to compare the `sample` with the `target`.  It can be any PsyNeuLink `CombinationFunction`,
        or a python function that takes a 2d array with two items and returns a 1d array of the same length
        as the two input items.

    output_port : OutputPort
        contains the `primary <OutputPort_Primary>` OutputPort of the ComparatorMechanism; the default is
        its *OUTCOME* OutputPort, the value of which is equal to the `value <ComparatorMechanism.value>`
        attribute of the ComparatorMechanism.

    output_ports : ContentAddressableList[OutputPort]
        contains, by default, only the *OUTCOME* (primary) OutputPort of the ComparatorMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *OUTCOME* OutputPort.

    standard_output_ports : list[str]
        list of `Standard OutputPorts <OutputPort_Standard>` that includes the following in addition to the
        `standard_output_ports <ObjectiveMechanism.standard_output_ports>` of an `ObjectiveMechanism`:

        .. _COMPARATOR_MECHANISM_SSE

        *SSE*
            the value of the sum squared error of the Mechanism's function

        .. _COMPARATOR_MECHANISM_MSE

        *MSE*
            the value of the mean squared error of the Mechanism's function

    """
    componentType = COMPARATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'ComparatorCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class Parameters(ObjectiveMechanism.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ComparatorMechanism.variable>`

                    :default value: numpy.array([[0], [0]])
                    :type: ``numpy.ndarray``
                    :read only: True

                function
                    see `function <ComparatorMechanism.function>`

                    :default value: `LinearCombination`(weights=numpy.array([[-1], [ 1]]))
                    :type: `Function`

                output_ports
                    see `output_ports <ComparatorMechanism.output_ports>`

                    :default value: [`OUTCOME`]
                    :type: ``list``
                    :read only: True

                sample
                    see `sample <ComparatorMechanism.sample>`

                    :default value: None
                    :type:

                target
                    see `target <ComparatorMechanism.target>`

                    :default value: None
                    :type:
        """
        # By default, ComparatorMechanism compares two 1D np.array input_ports
        variable = Parameter(np.array([[0], [0]]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        function = Parameter(LinearCombination(weights=[[-1], [1]]), stateful=False, loggable=False)
        sample = None
        target = None

        output_ports = Parameter(
            [OUTCOME],
            stateful=False,
            loggable=False,
            read_only=True,
            structural=True,
        )

    # ComparatorMechanism parameter and control signal assignments):

    standard_output_ports = ObjectiveMechanism.standard_output_ports.copy()
    standard_output_ports.extend([{NAME: SSE,
                                   FUNCTION: lambda x: np.sum(x * x)},
                                  {NAME: MSE,
                                   FUNCTION: lambda x: np.sum(x * x) / safe_len(x)}])
    standard_output_port_names = ObjectiveMechanism.standard_output_port_names.copy()
    standard_output_port_names.extend([SSE, MSE])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 sample: tc.optional(tc.any(OutputPort, Mechanism_Base, dict, is_numeric, str))=None,
                 target: tc.optional(tc.any(OutputPort, Mechanism_Base, dict, is_numeric, str))=None,
                 function=None,
                 output_ports:tc.optional(tc.optional(tc.any(str, Iterable))) = None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **kwargs
                 ):

        input_ports = kwargs.pop(INPUT_PORTS, {})
        if input_ports:
            input_ports = {INPUT_PORTS: input_ports}

        input_ports = self._merge_legacy_constructor_args(sample, target, default_variable, input_ports)

        # Default output_ports is specified in constructor as a tuple rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if isinstance(output_ports, (str, tuple)):
            output_ports = list(output_ports)

        # IMPLEMENTATION NOTE: The following prevents the default from being updated by subsequent assignment
        #                     (in this case, to [OUTCOME, {NAME= MSE}]), but fails to expose default in IDE
        # output_ports = output_ports or [OUTCOME, MSE]

        super().__init__(monitor=input_ports,
                         function=function,
                         output_ports=output_ports, # prevent default from getting overwritten by later assign
                         params=params,
                         name=name,
                         prefs=prefs,
                         **kwargs
                         )

        # Require Projection to TARGET InputPort (already required for SAMPLE as primary InputPort)
        self.input_ports[1].parameters.require_projection_in_composition._set(True, Context())

    def _validate_params(self, request_set, target_set=None, context=None):
        """If sample and target values are specified, validate that they are compatible
        """

        if INPUT_PORTS in request_set and request_set[INPUT_PORTS] is not None:
            input_ports = request_set[INPUT_PORTS]

            # Validate that there are exactly two input_ports (for sample and target)
            num_input_ports = len(input_ports)
            if num_input_ports != 2:
                raise ComparatorMechanismError(f"{INPUT_PORTS} arg is specified for {self.__class__.__name__} "
                                               f"({len(input_ports)}), so it must have exactly 2 items, "
                                               f"one each for {SAMPLE} and {TARGET}.")

            # Validate that input_ports are specified as dicts
            if not all(isinstance(input_port,dict) for input_port in input_ports):
                raise ComparatorMechanismError("PROGRAM ERROR: all items in input_port args must be converted to dicts"
                                               " by calling Port._parse_port_spec() before calling super().__init__")

            # Validate length of variable for sample = target
            if VARIABLE in input_ports[0]:
                # input_ports arg specified in standard port specification dict format
                lengths = [len(input_port[VARIABLE]) if input_port[VARIABLE] is not None else 0
                           for input_port in input_ports]
            else:
                # input_ports arg specified in {<Port_Name>:<PORT SPECIFICATION DICT>} format
                lengths = [len(list(input_port_dict.values())[0][VARIABLE]) for input_port_dict in input_ports]

            if lengths[0] != lengths[1]:
                raise ComparatorMechanismError(f"Length of value specified for {SAMPLE} InputPort "
                                               f"of {self.__class__.__name__} ({lengths[0]}) must be "
                                               f"same as length of value specified for {TARGET} ({lengths[1]}).")

        elif SAMPLE in request_set and TARGET in request_set:

            sample = request_set[SAMPLE]
            if isinstance(sample, InputPort):
                sample_value = sample.value
            elif isinstance(sample, Mechanism):
                sample_value = sample.input_value[0]
            elif is_value_spec(sample):
                sample_value = sample
            else:
                sample_value = None

            target = request_set[TARGET]
            if isinstance(target, InputPort):
                target_value = target.value
            elif isinstance(target, Mechanism):
                target_value = target.input_value[0]
            elif is_value_spec(target):
                target_value = target
            else:
                target_value = None

            if sample is not None and target is not None:
                if not iscompatible(sample, target, **{kwCompatibilityLength: True,
                                                       kwCompatibilityNumeric: True}):
                    raise ComparatorMechanismError(f"The length of the sample ({len(sample)}) "
                                                   f"must be the same as for the target ({len(target)})"
                                                   f"for {self.__class__.__name__} {self.name}.")

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

    def _merge_legacy_constructor_args(self, sample, target, default_variable=None, input_ports=None):

        # USE sample and target TO CREATE AN InputPort specfication dictionary for each;
        # DO SAME FOR InputPorts argument, USE TO OVERWRITE ANY SPECIFICATIONS IN sample AND target DICTS
        # TRY tuple format AS WAY OF PROVIDED CONSOLIDATED variable AND OutputPort specifications

        sample_dict = _parse_port_spec(owner=self,
                                        port_type=InputPort,
                                        port_spec=sample,
                                        name=SAMPLE)

        target_dict = _parse_port_spec(owner=self,
                                        port_type=InputPort,
                                        port_spec=target,
                                        name=TARGET)

        # If either the default_variable arg or the input_ports arg is provided:
        #    - validate that there are exactly two items in default_variable or input_ports list
        #    - if there is an input_ports list, parse it and use it to update sample and target dicts
        if input_ports:
            input_ports = input_ports[INPUT_PORTS]
            # print("type input_ports = {}".format(type(input_ports)))
            if not isinstance(input_ports, list):
                raise ComparatorMechanismError(f"If an '{INPUT_PORTS}' argument is included in the constructor "
                                               f"for a {ComparatorMechanism.__name__} it must be a list with "
                                               f"two {InputPort.__name__} specifications.")

        input_ports = input_ports or default_variable

        if input_ports is not None:
            if len(input_ports)!=2:
                raise ComparatorMechanismError(f"If an \'input_ports\' arg is included in the constructor for a "
                                               f"{ComparatorMechanism.__name__}, it must be a list with exactly "
                                               f"two items (not {len(input_ports)}).")

            sample_input_port_dict = _parse_port_spec(owner=self,
                                                        port_type=InputPort,
                                                        port_spec=input_ports[0],
                                                        name=SAMPLE,
                                                        value=None)

            target_input_port_dict = _parse_port_spec(owner=self,
                                                        port_type=InputPort,
                                                        port_spec=input_ports[1],
                                                        name=TARGET,
                                                        value=None)

            sample_dict = recursive_update(sample_dict, sample_input_port_dict)
            target_dict = recursive_update(target_dict, target_input_port_dict)

        return [sample_dict, target_dict]
