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

A TimerMechanism is a form of `IntegratorMechanism` that starts at a specified `value <Mechanism_Base.value>`, that
is changed monotonically each time it is executed, until it reaches a specified end value. The path that its change
in `value <Mechanism_Base.value>` takes is specified by its `trajectory <TimerMechanism.trajectory>` Function. If
it receives an imput, the timer is advanced by that ammount; otherwise it is advanced by the value of its `increment
<TimerMechanism.increment>` parameter. It can be reset to its starting value by calling its `reset
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
reaches 1. Howver, the shape, starting, ending and rate of increment call all be configured.  The shape of the timer's
progression is specified by it **trajectory** argument, which must be a `TimerFunction` or an appropriately configured 
`UserDefinedFunction` (see `below <TimerMechanism_Trajectory_Function>` for details); the starting and ending `values
<Mechanism_Base.value>` of the timer are specified by its **start** and **end** arguments, respectively, and the
ammount it progresses each time the Mechanimsm is executed in the absence of input) can be specified by its
**increment** argument.

COMMENT:
TBI
The direction of its progression can be specified using the **direction** argument, which can be *INCREASING* or
*DECREASING*.
COMMENT

.. _TimerMechanism_Structure:

Structure
---------

A TimerMechanism may or may not have a source of input, and has a single `OutputPort`.  Its `function
<TimerMechanism.function>` is an `SimpleIntegrator` `Function` that always starts at 0 and, each time it is executed,
increments by the value of the TimerMechanism based either on its input or, if it receives none, then its `increment
<TimerMechanism.increment>` Parameter (see `TimerMechanism_Execution` for additional details).

    .. technical_note::
       The TimerMechanism's `function <TimerMechanism.function>` cannot be modified; its `rate
       <SimpleIntegrator.rate>` is set to the value specified for the TimerMechanism's `increment
       <TimerMechanism.increment>` Parameter, and its `initializer <SimpleIntegrator.initializer>`,
       `noise <SimpleIntegrator.noise>` and `offset <SimpleIntegrator.offset>` Parameters are fixed at 0.

.. _TimerMechanism_Trajectory_Function:

Its `trajectory <TimerMechanism.trajectory>` must be either a `TimerFunction` or a `UserDefinedFunction` that
has **start** and **end** parameters, and takes a single numeric value, a list or an np.array as its `variable
<Function_Base.variable>`.

    .. technical_note::
       The `trajectory <TimerMechanism.trajectory>` Function is an auxilliary function that takes the result of the
       TimerMechanism's `function <TimerMechanism.function>` and transforms it to implement the specified trajectory
       of the timer's `value <Mechanism_Base.value>`. The value of the `start <TimerMechanism.start>` parameter is
       assigned as the `start <TimerFunction.start>` Parameter of its `trajectory <TimerMechanism.trajectory>`
       Function; its `increment <TimerMechanism.increment>` Parameter is assigned as the `rate <TimerFunction.rate>`
       Parameter of the `trajectory <TimerMechanism.trajectory>` Function; and its `end <TimerMechanism.end>` Parameter
       is assigned as the `end <TimerFunction.end>` Parameter of the `trajectory <TimerMechanism.trajectory>` Function,
       which is also used to set the TimerMechanism's `complete <TImeMechanism.complete>` attribute.

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
corresponding element of the timer array.  The TimerMechanism stop advancing when its value reaches its `end
<TimerMechanism.end>` Parameter, at which point it sets its `complete <TimerMechanism.complete>` Parameter to True,
and stops advancing its value. This can be used together with a `Scheduler` `Condition` to make the processing
of other `Mechanisms <Mechanism>` contingent on the TimerMechanism reaching its end value. The Timer can be reset to
its `start <TimerMechanism.start>` `value <Mechanism_Base.value>` by setting  its `reset <TimerMechanism.reset>`
parameter to a non-zero value, as described for a standard `IntegratorMechanism <IntegratorMechanism_Reset>`.

COMMENT:
FIX: ADD ADVICE SECTION HERE OR EXAMPLES FOR TIMERS THAT IMPLEMENT RISING TO A THRESHOLD AND COLLAPSING TO BOUND
EXPONENTIAL GROWTH: **trajectory** = EXPONENTIAL;  **direction** = *INCREASING*
:math: s-1\ +\ e^{\frac{-\ln\left(1-\frac{d}{s}\right)}{f}x}
python: start -1 np.exp(-np.ln(1-threhold/start)/end)
start = offset (y value at which growth begins)
threshold = distance above starting y at end
end = scale (x value at which end should occur

DECELLERATING GROWTH: **trajectory** = EXPONENTIAL;  **direction** = *INCREASING*
# :math: s+r\left(1-e^{\frac{\ln\left(1-\frac{d}{r}\right)}{f}x}\right)
# python: start + rate * (1 - np.exp(np.ln(1 - threshold / rate) / end) * x))
# start = offset (y value at which growth begins)
# end = scale (contingent on threshold; i.e., scale should cause y=threshold at x=end)
----------------
:math: s+r\left(1-e^{\frac{\ln\left(1-t\right)}{f}x}\right)
python: start + threshold * (1 - np.exp(np.ln(1 - threshold) / end) * x))
start = offset (y value at which growth begins)
end = scale (y = tolerance * start at x = end)
threshold = distance above starting y at end
tolerance = 0.01 (default) (i.e., y = within 1% of threshold at x = end)

EXPONENTIAL DECAY:   **trajectory** = EXPONENTIAL; **direction** = *DECREASING*
:math:   offset\ + bias e^{-\left(\frac{variable\ln\left(\frac{1}{tolerance}\right)}{scale}\right)}
[DESMOS: o\ +se^{-\left(\frac{x\ln\left(\frac{1}{t}\right)}{f}\right)}]
python: offset + bias * np.exp(-variable * np.ln(1/tolerance) / scale)

FOR TIMER:
offset = offset (y offset for entire function)
start = bias (starting y value relative to offset: s + o = y for x = 0)
end = scale (y = (tolerance * start + offset) at x = end)
tolerance = 0.01 (default) (i.e., y = (1% of start) + offset at x = end)

ACCELERATING DECAY:  **trajectory** = EXPONENTIAL; **direction** = *INCREASING*
LogarithmicDecay Function
:math: start + e^{-e}(1-e^x)\frac{n}{e^{end-e-0.4^{end}}}
python: start + np.exp(-e)*(1-np.exp(x))*n/np.exp(end-e-0.4**end)
start = offset (y value at which decay begins)
end = scale (x value at which y=0)

.. technical_note::
   This is an empirically-derived function;  the value of 0.4 is used to ensure that the function reaches 0 at the
    specified end value. If anyone has an analytic solution, please add it here.

`function <LogarithmicDecay._function>` returns exponentially decaying transform of `variable
<LogarithmicDecay.variable>`, that has a value of `start <LogarithmicDecay.start>` + `offset
<LogarithmicDecay.offset>` at `variable <LogarithmicDecay.variable>` = 0, and a value of `threshold
<LogarithmicDecay.end>` * `start <LogarithmicDecay.start>` + `offset <LogarithmicDecay.offset>` at
`variable at `variable <LogarithmicDecay.variable>` = `end <LogarithmicDecay.end>`:
COMMENT

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

from psyneulink.core.components.functions.function import Function
from psyneulink.core.components.functions.nonstateful.transferfunctions import TransferFunction, Linear, Exponential
from psyneulink.core.components.functions.nonstateful.timerfunctions import TimerFunction
from psyneulink.core.components.functions.stateful.integratorfunctions import IntegratorFunction, SimpleIntegrator
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE, TIMER_MECHANISM, VARIABLE, PREFERENCE_SET_NAME, RESET
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
        function=SimpleIntegrator(rate=1),      \
        trajectory=Linear,                      \
        end=1)

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

    COMMENT:
    TBI
    direction : INCREASING, DECREASING, or list of either/both : default INCREASING
        specifies whether the timer progresses in the direction of increasing or decreasing values; if a list or
        array, the length must be the same as specified for **default_variable** or **input_shapes**
        (see `TimerMechanism_Execution` for additional details).
    COMMENT

    end : scalar, list or array : default 1
        specifies the value at which the timer stops advancing; if a list or array, the length must be the same as
        specified for **default_variable** or **input_shapes** (see `TimerMechanism_Execution` for additional details).

    function : IntegratorFunction : default SimpleIntegrator(rate=1)
        specifies the function used to increment the input; must take a single numeric value, or a list or np.array
        of values, and return one of the same shape.

    trajectory : TransferFunction or UserDefinedFunction : default LINEAR
        specifies the shape of the timer's trajectory; must be a supported `TransferFunction` (see XXX

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

    COMMENT:
    direction : INCREASING, DECREASING, or list of either/both
        determines whether the timer progresses in the direction of increasing or decreasing values.
        (see `TimerMechanism_Execution` for additional details).
    COMMENT

    function : IntegratorFunction
        determines the function used to advance the input to the TimerMechanism's `trajectorytrajectory
        <TimerMechanism.trajectory>` Function; if the TimerMechanism receives an external input, that is
        used to advance the timer; otherwise, the `increment <TimerMechanism.increment>` Parameter is used.

    trajectory : TransferFunction or UserDefinedFunction
        determines the `Function` used to transform the ouput of the TimerMechanism's `function
        <TimerMechanism.function>` to generate its output; this determines the shape of the trajectory
        of the TimerMechanism's `value <Mechanism_Base.value>`.

    end : scalar, list or array
        determines the value at which the timer stops advancing, after which it sets its `complete
        <TimerMechanism.complete>` Parameter to `True` and `value <Mechanism_Base.value>` remains equal to
        its `end <TimerMechanism.end>` value.

    complete : bool
        indicates whether the TimerMechanism has reached its `end <TimerMechanism.end>` value
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

                complete
                    see `complete <TimerMechanism.complete>`

                    :default value: `False
                    :type: `bool`

                end
                    see `end <TimerMechanism.end>`

                    :default value: `AdaptiveIntegrator`(initializer=numpy.array([0]), rate=0.5)
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

                    :default value: `AdaptiveIntegrator`(initializer=numpy.array([0]), rate=0.5)
                    :type: `float`

                trajectory
                    see `trajectory <TimerMechanism.trajectory>`

                    :default value: `Linear`
                    :type: `Function`
        """
        function = Parameter(SimpleIntegrator, stateful=False, loggable=False)
        trajectory = Parameter(Linear, stateful=False, loggable=False)
        start = FunctionParameter(0, function_name='trajectory', function_parameter_name='start', primary=True)
        increment = FunctionParameter(1, function_name='function', function_parameter_name='rate', primary=True)
        end = FunctionParameter(1, function_name='trajectory', function_parameter_name='end', primary=True )
        complete = Parameter(False, stateful=True, loggable=True)

        # FIX: WRITE VALIDATION METHODS AND REMOVE TYPE HINTS

    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 input_shapes=None,
                 start:Optional[Union[int, float, list, np.ndarray]]=None,
                 increment:Optional[Union[int, float, list, np.ndarray]]=None,
                 function:Optional[IntegratorFunction]=None,
                 trajectory:Optional[TimerFunction]=None,
                 end:Optional[Union[int, float, list, np.ndarray]]=None,
                 params=None,
                 name=None,
                 prefs:   Optional[ValidPrefSet] = None,
                 **kwargs):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        # FIX: ASSIGN start -> initializer, increment -> rate, end -> is_finished

        super(TimerMechanism, self).__init__(default_variable=default_variable,
                                             input_shapes=input_shapes,
                                             start=start,
                                             increment=increment,
                                             function=function,
                                             trajectory=trajectory,
                                             end=end,
                                             complete=False,
                                             params=params,
                                             name=name,
                                             prefs=prefs,
                                             **kwargs)
        assert True

    def _execute(self, variable=None, context=None, runtime_params=None, **kwargs):
        """Override to check for call to reset by ControlSignal"""
        if not self.is_initializing and self.parameters.complete.get(context):
            return self.parameters.previous_value._get(context)

        value = super()._execute(variable=variable, context=context, runtime_params=runtime_params, **kwargs)
        value = self.trajectory(value)

        # FIX: WHY IS THIS AFTER RATHER THAN BEFORE EXECUTION?
        # No need to reset during initialization (which will occur if **reset_default** != 0)
        if not self.is_initializing:

            if value == self.parameters.end.get(context):
                self.parameters.complete._set(True, context)

            if np.array(self._get_current_parameter_value(RESET,context)).squeeze():
                self.reset(context=context)
                value = self.parameters.value._get(context).reshape(value.shape)

        return value
