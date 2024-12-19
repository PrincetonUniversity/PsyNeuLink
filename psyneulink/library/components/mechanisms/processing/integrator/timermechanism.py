# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  TimerMechanism *************************************************

"""

Contents
--------

  * `TimerMechanism_Overview`
  * `TimerMechanism_Creation`
  * `TimerMechanism_Structure`
  * `TimerMechanism_Execution`
  * `TimerMechanism_Examples`
  * `TimerMechanism_Class_Reference`


.. _TimerMechanism_Overview:

Overview
--------

A TimerMechanism is a type of `IntegratorMechanism` the `value <Mechanism_Base.value>` of which begins at a specified
`start <TimerMechanism.start>` value, is changed monotonically each time it is executed, until it reaches a specified
`end <TimerMechanism.end>` value.  The number of executions it takes to do so is determined by a combination of its
`duration <TimerMechanism.duration>` and `increment <TimerMechanism.increment>` parameters, and whether or not it
receives any input; and the path that its `value <Mechanism_Base.value>` takes is determined by its `trajectory
<TimerMechanism.trajectory>` Function. It can be reset to its starting value by calling its `reset
<TimerMechanism.reset>` method, or by modulating its `reset <TimerMechanism.reset>` Parameter with a `ControlSignal`.

A TimerMechanism can be used to implement a variety of time-based processes, such as the collapse of a boundary over
time, or the rise of a value to a threshold. It can also be configured to execute multiple such processes in parallel,
each with its own starting and ending values, as well as direction, increment and input (on a given execution), all of
which use the same `trajectory <TimerMechanism.trajectory>` Function.

.. _TimerMechanism_Creation:

Creating a TimerMechanism
-------------------------

A TimerMechanism can be created directly by calling its constructor, or using the `mechanism` command and specifying
*TIMER_MECHANISM* as its **mech_spec** argument. It can be created with or without a source of input. By default, a
TimerMechanisms increments linearly, starting at 0, incrementing by 0.01 each time it is executed, and stopping when it
reaches 1. However, the shape, starting, ending and rate of increment can all be configured.  The shape of the timer's
progression is specified by it **trajectory** argument, which must be a `TimerFunction` or an appropriately configured
`UserDefinedFunction` (see `below <TimerMechanism_Trajectory_Function>` for details); the starting and ending `values
<Mechanism_Base.value>` of the timer are specified by its **start** and **end** arguments, respectively, and the ammount
it progresses each time the Mechanimsm is executed (in the absence of input) is specified by its **increment** argument.

.. _TimerMechanism_Structure:

Structure
---------

A TimerMechanism may or may not have a source of input, and has a single `OutputPort`.  Its `function
<TimerMechanism.function>` is an `SimpleIntegrator` `Function` that determines the input to the TimerMechanism's
`trajectory <TimerMechanism.trajectory>` Function which, in turn, determines the `value <Mechanism_Base.value>` of the
TimerMechanism. The TimerMechanism's `function <TimerMechanism.function>` is always a `SimpleIntegrator` that always
starts at 0 and, each time it is executed, increments the value of the TimerMechanism based either on its input or, if
it receives none, then its `increment <TimerMechanism.increment>` Parameter (see `TimerMechanism_Execution` for
additional details).

    .. technical_note::
       The TimerMechanism's `function <TimerMechanism.function>` cannot be modified; its `rate
       <SimpleIntegrator.rate>` is set to the value specified for the TimerMechanism's `increment
       <TimerMechanism.increment>` Parameter, and its `initializer <SimpleIntegrator.initializer>`,
       `noise <SimpleIntegrator.noise>` and `offset <SimpleIntegrator.offset>` Parameters are fixed at 0

.. _TimerMechanism_Trajectory_Function:

Its `trajectory <TimerMechanism.trajectory>` Function must be either a `TimerFunction` or a `UserDefinedFunction` that
has **initial**, **final** and *duration** parameters, and takes a single numeric value, a list or an np.array as its
`variable <Function_Base.variable>`.

    .. technical_note::
       The `trajectory <TimerMechanism.trajectory>` Function is an auxilliary function that takes the result of the
       TimerMechanism's `function <TimerMechanism.function>` and transforms it to implement the specified trajectory
       of the timer's `value <Mechanism_Base.value>`. The value of the `start <TimerMechanism.start>` parameter is
       assigned as the `initial <TimerFunction.initial>` Parameter of its `trajectory <TimerMechanism.trajectory>`
       Function; its `end <TimerMechanism.end>` Parameter is assigned as the `final <TimerFunction.final>` Parameter of
       the `trajectory <TimerMechanism.trajectory>` Function, and its `duration <TimerMechanism.duration>` Parameter
       is assigned as the `duration <TimerFunction.duration>` Parameter of the `trajectory <TimerMechanism.trajectory>`;
       which is also used to set the TimerMechanism's `finished <TimeMechanism.finished>` attribute.

.. _TimerMechanism_Modulation:

A TimerMechanism's `start <TimerMechanism.start>`, `end <TimerMechanism.end>` and `increment <TimerMechanism.increment>`
parameters can be `modulated <ModulatorySignal_Modulation>` by a `ControlMechanism`.

.. _TimerMechanism_Execution:

Execution
---------

When a TimerMechanism is executed, it advances its value by adding either its `input <TimerMechanism.input>`
or its `increment <TimerMechanism.increment>` Parameter to the `previous_value <IntegratorFunction.previous_value>`
of its `function <TimerMechanism.function>`, that is then passed to the `Function` specified by its `trajectory
<TimerMechanism.trajectory>` Parameter, the result of which is assigned to the `value <Mechanism_Base.value>` of the
TimerMechansim's `primary OutputPort <OutputPort_Primary>`. If its `default_variable <TimerMechanism.default_variable>`
is a scalar or of length 1, or  its **input_shapes** is specified as 1, the TimerMechanism generates a scalar value
as its output; otherwise, it generates an array of values, each element of which is independently advanced. If its
`increment <TimerMechanism.increment>` Parameter is a single value, that is used for advancing all elements; if the
`increment <TimerMechanism.increment>` Parameter is a list or array, then each element is used to increment the
corresponding element of the timer array.

The TimerMechanism stops advancing after the `value <Function_Base.value>` of its `function <TimerMechanism.function>`
equals the TimerMechanism's `duration <TimerMechanism.duration>` Parameter.  If the TimerMechanism receives no input,
then it will stop advancing after the number of executions = `duration <TimerMechanism.duration>` / `increment
<TimerMechanism.increment>`.  If the TimerMechanism receives input, then the number of executions is determined by the
number of times it is executed, and the amount of input it receives on each execution.  When the TimerMechanism stops
advancing, it sets its `finished <TimerMechanism.finished>` attribute to `True`, and its `value <Mechanism_Base.value>`
remains equal to its `end <TimerMechanism.end>` on any further executions, unless it is reset.  If the TimerMechanism
is reset, its `value <Mechanism_Base.value>` is set to its `start <TimerMechanism.start>` value.

    .. hint::
       A TimerMechanism's `finished <TimerMechanism.finished>` attribute can be used together with a `Scheduler`
       `Condition` to make the processing of other `Mechanisms <Mechanism>` contingent on the TimerMechanism
       reaching its end value.

    .. note::
       The TimerMechanism's `finished <TimerMechanism.finished>` attribute is not used to determine when it stops
       advancing; rather, it is set to `True` when the TimerMechanism's `value <Mechanism_Base.value>` reaches its
       `end <TimerMechanism.end>` value.

If a TimerMechanism continues to be executed after its `finished <TimerMechanism.finished>` attribute is True, its
`value <Mechanism_Base.value>` remains equal to its `end <TimerMechanism.end>` value.  The TimerMechanism can be reset
to its `start <TimerMechanism.start>` value by setting its `reset <TimerMechanism.reset>` Parameter to a non-zero value,
as described for a standard `IntegratorMechanism <IntegratorMechanism_Reset>`.

.. _TimerMechanism_Examples:

Examples
--------

The following example creates a TimerMechanism with a linear decay from 1 to 0:

    >>> import psyneulink as pnl
    >>> my_timer_mechanism = pnl.TimerMechanism(trajectory=LINEAR, start=1, end=0))


.. _TimerMechanism_Class_Reference:

Class Reference
---------------

"""
from collections.abc import Iterable

from beartype import beartype

from psyneulink._typing import Optional, Union
import numpy as np

from psyneulink.core.components.functions.nonstateful.timerfunctions import TimerFunction, LinearTimer
from psyneulink.core.components.functions.stateful.integratorfunctions import IntegratorFunction, SimpleIntegrator
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.globals.keywords import TIMER_MECHANISM, PREFERENCE_SET_NAME, RESET, TRAJECTORY
from psyneulink.core.globals.parameters import Parameter, FunctionParameter, check_user_specified
from psyneulink.core.globals.preferences.basepreferenceset import ValidPrefSet, REPORT_OUTPUT_PREF
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel

__all__ = [
    'DEFAULT_RATE', 'TimerMechanism', 'TimerMechanismError'
]

# TimerMechanism parameter keywords:
DEFAULT_RATE = 1


class TimerMechanismError(MechanismError):
    pass


class TimerMechanism(IntegratorMechanism):
    """
    TimerMechanism(                             \
        start=0,                                \
        increment=1,                            \
        end=1,                                  \
        function=SimpleIntegrator(rate=1),      \
        trajectory=LinearTimer,                 \
        duration=1)

    Subclass of `IntegratorMechanism` that advances its input until it reaches a specified value.
    See `IntegratorMechanism <IntegratorMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    start : scalar, list or array : default 0
        specifies the starting `value <Mechanism_Base.value>` of the timer; if a list or array, the length must be
        the same as specified for **default_variable** or **input_shapes** (see `TimerMechanism_Execution` for
        additional details).

    increment : scalar, list or array : default 1
        specifies the amount by which the `previous_value <IntegratorFunction.previous_value>` of the
        TimerMechanism's `function <TimerMechanism.function>` is incremented each time the TimerMechanism is
        executed in the absence of input; if a list or array, the length must be the same as specified for
        **default_variable** or **input_shapes** (see `TimerMechanism_Execution` for additional details).

    function : IntegratorFunction : default SimpleIntegrator(rate=1)
        specifies the function used to increment the input to the TimerMechanism's `trajectory
        <TimerMechanism.trajectory>` Function; must take a single numeric value, or a list or np.array
        of values, and return one of the same shape.

    trajectory : TransferFunction or UserDefinedFunction : default LinearTimer
        specifies the shape of the timer's trajectory; must be a supported `TransferFunction`
        (see `trajectory function <TimerMechanism_Trajectory_Function>`)

    end : scalar, list or array : default 1
        specifies the value of its `trajectory <TimerMechamism.trajectory>` function at which the timer stops advancing;
        if a list or array, the length must be the same as specified for **default_variable** or **input_shapes** (see
        `TimerMechanism_Execution` for additional details).

    duration : scalar, list or array : default 1
        specifies the value of its `variable <Mechanism_Base.variable>` at which the timer stops advancing; if a list
        or array, the length must be the same as specified for **default_variable** or **input_shapes** (see
        `TimerMechanism_Execution` for additional details).

    reset_default : number, list or np.ndarray : default 0
        specifies the default value used for the `reset <TimerMechanism.reset>` parameter.

    Attributes
    ----------

    start : scalar, list or array
        determines the starting `value <Mechanism_Base.value>` of the timer; assigned as the `start` Parameter of the
        TimerMechanism's `trajectory <TimerMechanism.function>` if it has one; otherwise, it is computed if possible
        or, an error is raised (see `TimerMechanism_Execution` for additional details).

    increment : scalar, list or array
        determines the amount by which the `previous_value <IntegratorFunction.previous_value>` of the
        TimerMechanism's `function <TimerMechanism.function>` is incremented each time the TimerMechanism is
        executed in the absence of input; assigned as the `rate <IntegratorFunction.rate>` of the TimerMechanism's
        `function <TimerMechanism.function>` (see `TimerMechanism_Execution` for additional details).

    function : IntegratorFunction
        determines the function used to advance the input to the TimerMechanism's `trajectorytrajectory
        <TimerMechanism.trajectory>` Function; if the TimerMechanism receives an external input, that is
        used to advance the timer; otherwise, the `increment <TimerMechanism.increment>` Parameter is used.

    trajectory : TimerFunction or UserDefinedFunction
        determines the `Function` used to transform the ouput of the TimerMechanism's `function
        <TimerMechanism.function>` to generate its output; this determines the shape of the trajectory
        of the TimerMechanism's `value <Mechanism_Base.value>`.

    end : scalar, list or array : default 1
        determines the value of its `trajectory <TimerMechamism.trajectory>` function at which the timer stops
        advancing; if a list or array, the length must be the same as specified for **default_variable** or
        **input_shapes** (see `TimerMechanism_Execution` for additional details).

    duration : scalar, list or array
        determines the value at which the timer stops advancing, after which it sets its `finished
        <TimerMechanism.finished>` Parameter to `True` and `value <Mechanism_Base.value>` remains equal to
        its `duration <TimerMechanism.duration>` value.

    finished : bool
        indicates whether the TimerMechanism has reached its `duration <TimerMechanism.duration>` value
        (see `TimerMechanism_Execution` for additional details).

    reset : int, float or 1d array of length 1 : default 0
        if non-zero, the TimerMechanism's `reset <Mechanism_Base.reset>` method is called, which resets the
        `value <Mechanism_Base.value>` of the TimerMechanism to its initial value (see
        `TimerMechanism_Reset` for additional details).

    """
    componentType = TIMER_MECHANISM

    classPreferenceLevel = PreferenceLevel.TYPE
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    classPreferences = {
        PREFERENCE_SET_NAME: 'TimerMechanismCustomClassPreferences',
        REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class Parameters(IntegratorMechanism.Parameters):
        """
            Attributes
            ----------

                finished
                    see `finished <TimerMechanism.finished>`

                    :default value: False
                    :type: `bool`

                duration
                    see `duration <TimerMechanism.duration>`

                    :default value: 1
                    :type: `float`

                function
                    see `function <TimerMechanism.function>`

                    :default value: `SimpleIntegrator`(initializer=numpy.array([0]), rate=1)
                    :type: `Function`

                increment
                    see `increment <TimerMechanism.increment>`

                    :default value: 1
                    :type: `float`

                start
                    see `start <TimerMechanism.start>`

                    :default value: 0
                    :type: `float`

                end
                    see `end <TimerMechanism.end>`

                    :default value: 1
                    :type: `float`

                trajectory
                    see `trajectory <TimerMechanism.trajectory>`

                    :default value: `LinearTimer`
                    :type: `Function`
        """
        function = Parameter(SimpleIntegrator, stateful=False, loggable=False)
        trajectory = Parameter(LinearTimer, stateful=False, loggable=False)
        start = FunctionParameter(0, function_name='trajectory', function_parameter_name='initial', primary=True)
        increment = FunctionParameter(.01, function_name='function', function_parameter_name='rate', primary=True)
        end = FunctionParameter(1, function_name='trajectory', function_parameter_name='final', primary=True )
        duration = FunctionParameter(1, function_name='trajectory', function_parameter_name='duration', primary=True )
        finished = Parameter(False, stateful=True, loggable=True)

        def _validate_trajectory(self, trajectory):
            if not (isinstance(trajectory, TimerFunction)
                    or (isinstance(trajectory, type) and issubclass(trajectory, TimerFunction))):
                return f'must be a TimerFunction'

        def _validate_start(self, start):
            if not isinstance(start, (int, float, list, np.ndarray)):
                return f'must be an int, float or a list or array of either'

        def _validate_increment(self, increment):
            if not isinstance(increment, (int, float, list, np.ndarray)):
                return f'must be an int, float or a list or array of either'

        def _validate_end(self, end):
            if not isinstance(end, (int, float, list, np.ndarray)):
                return f'must be an int, float or a list or array of either'

        def _validate_durat(self, duration):
            if not isinstance(duration, (int, float, list, np.ndarray)):
                return f'must be an int, float or a list or array of either'

    @check_user_specified
    @beartype
    def __init__(self,
                 input_shapes=None,
                 start=None,
                 increment=None,
                 function=None,
                 trajectory=None,
                 end=None,
                 duration=None,
                 params=None,
                 name=None,
                 prefs: Optional[ValidPrefSet] = None,
                 **kwargs):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        super(TimerMechanism, self).__init__(default_variable=1,
                                             input_shapes=input_shapes,
                                             start=start,
                                             increment=increment,
                                             function=function,
                                             trajectory=trajectory,
                                             end=end,
                                             duration=duration,
                                             finished=False,
                                             params=params,
                                             name=name,
                                             prefs=prefs,
                                             **kwargs)

    def _execute(self, variable=None, context=None, runtime_params=None, **kwargs):
        """Override to check for call to reset by ControlSignal"""
        if not self.is_initializing and self.parameters.finished.get(context):
            return self.trajectory(self.function.parameters.previous_value._get(context))

        x = super()._execute(variable=variable, context=context, runtime_params=runtime_params, **kwargs)
        y = self.trajectory(x)

        # No need to reset during initialization (which will occur if **reset_default** != 0)
        if not self.is_initializing:

            if np.allclose(x,self.parameters.duration.get(context)):
                self.parameters.finished._set(True, context)

            # If reset Parameter has been set, reset() TimerMechanism to start value
            if np.array(self._get_current_parameter_value(RESET,context)).squeeze():
                self.reset(self.parameters.start._get(context), context=context)
                y = self.parameters.value._get(context).reshape(y.shape)

        return y

    def reset(self, *args, force=False, context=None, **kwargs):
        super().reset(*args, force=force, context=context, **kwargs)
        self.finished = False
