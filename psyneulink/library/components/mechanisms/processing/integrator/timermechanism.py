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
    - `TimerMechanism_Processing`
    - `TimerMechanism_Reset`
  * `TimerMechanism_Examples`
  * `TimerMechanism_Class_Reference`


.. _TimerMechanism_Overview:

Overview
--------

A TimerMechanism is a form of `IntegratorMechanism` that advances its output until it reaches a specified value. It's
starting and ending values, as well as the
COMMENT:
direction,
COMMENT
rate and trajectory of its progression can be specified.
If an input is provided, the timer is advanced by that ammount; otherwise it is advanced by the value of its `increment
<TimerMechanism.increment>` parameter. A TimerMechanism can be reset to its starting value by calling its `reset
<TimerMechanism.reset>` method, or by modulating its `reset <TimerMechanism.reset>` Parameter with a `ControlSignal`.

A TimerMechanism can be used to implement a variety of time-based processes, such as the collapse of a boundary over
time, or the rise of a value to a threshold. It can also be configured to execute multiple such processes in parallel,
each with its own starting and ending values, as well as direction, increment and input (on a given execution), all of
which use the trajectory.

.. _TimerMechanism_Creation:

Creating a TimerMechanism
-------------------------

A TimerMechanism can be created directly by calling its constructor, or using the `mechanism` command and specifying
*TIMER_MECHANISM* as its **mech_spec** argument. It can be created with or without a source of input. The shape of the
timer's trajectory is specified by it **trajecotry** argument, which must be a `TransferFunction`
COMMENT:
TBI
as well as its **direction** argument, which can be *INCREASING* or *DECREASING*
COMMENT
(see `TimerMechanism_Processing` below for more details).The starting and ending values of the timer are specified by
its **start** and **end** arguments, respectively, and the ammount it progresses (in the absence of input) each time
the Mechanimsm is executed can be specified by its **increment** argument.

.. _TimerMechanism_Structure:

Structure
---------

A TimerMechanism may or may not have a source of input, and has a single `OutputPort`. Its `start
<TimerMechanism.start>`, `end <TimerMechanism.end>` and `increment <TimerMechanism.rate>` parameters can be `modulated
<ModulatorySignal_Modulation>` by a `ControlMechanism` to change its rate of

.. technical_note::
   A TimerMechanism is an `IntegratorMechanism` that, in addition to its `function <TimerMechanism.function>`, has an
   auxilliary `TransferFunction` -- its `trajectory <TimerMechanism.trajectory>` Parameter -- that takes the result of
   its `function <TimerMechanism.function>` and transforms it to implement the specified trajectory of the timer's
   `value <Mechanism_Base.value>`. The value of the `start <TimerMechanism.start>` parameter is assigned as the
   `initializer <IntegratorFunction.initializer>` of the TimerMechanism's `function <TimerMechanism.function>`; its
   `increment <TimerMechanism.increment>` parameter is assigned as the `rate <IntegratorFunction.rate>` of the
   TimerMechanism's `function <TimerMechanism.function>`; and its `end <TimerMechanism.end>` parameter is used to set
   the `is_finished <TimerMechanism.is_finished>` attribute of the TimerMechanism.

.. _TimerMechanism_Execution:

Execution
---------

.. _TimerMechanism_Processing:

*Processing*
~~~~~~~~~~~~

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
<TimerMechanism.end>` Parameter, at which point it sets its `is_finished <TimerMechanism.is_finished>` attribute to
`True`, and stops advancing its value. This can be used together with the  `WhenFinished` `Condition` to make the
processing of other `Mechanisms <Mechanism>` contingent on the TimerMechanism reaching its end value. The Timer can be
reset to its `start <TimerMechanism.start>` value by setting  its `reset <TimerMechanism.reset>` parameter to a
non-zero value, as described below.

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
AcceleratingDecay Function
:math: start + e^{-e}(1-e^x)\frac{n}{e^{end-e-0.4^{end}}}
python: start + np.exp(-e)*(1-np.exp(x))*n/np.exp(end-e-0.4**end)
start = offset (y value at which decay begins)
end = scale (x value at which y=0)

.. technical_note::
   This is an empirically-derived function;  the value of 0.4 is used to ensure that the function reaches 0 at the
    specified end value. If anyone has an analytic solution, please add it here.

`function <AcceleratingDecay._function>` returns exponentially decaying transform of `variable
<AcceleratingDecay.variable>`, that has a value of `start <AcceleratingDecay.start>` + `offset
<AcceleratingDecay.offset>` at `variable <AcceleratingDecay.variable>` = 0, and a value of `threshold
<AcceleratingDecay.end>` * `start <AcceleratingDecay.start>` + `offset <AcceleratingDecay.offset>` at
`variable at `variable <AcceleratingDecay.variable>` = `end <AcceleratingDecay.end>`:
COMMENT

.. _TimerMechanism_Reset:

*Reset*
~~~~~~~

A TimeMechanism has a `modulable <ModulatorySignal_Modulation>` `reset <IntergatorMechanism.reset>` Parameter
that can be used to reset its value to the value of its `function <TimerMechanism.function>`\\s `initializer
<IntegratorFunction.initializer>`. This also clears the `value <Mechanism_Base.value>` `history <Parameter.history>`,
thus effectively setting the `previous_value <IntegratorFunction.previous_value>`  of its `function
<TimerMechanism.function>` to None.

The `reset <TimerMechanism.reset>` parameter can be used to reset the TimerMechanism under the control of a
`ControlMechanism`.  This simplest way to do this is to specify the `reset <TimerMechanism.reset>` parameter of
the IntgeratorMechanism in the **control** argument of the ControlMechanism's constructor, and to specify *OVERRIDE*
in its **modulation** argument, as in the following example::

    >>> my_integrator = TimerMechanism()
    >>> ctl_mech = pnl.ControlMechanism(modulation=pnl.OVERRIDE, control=(pnl.RESET, my_integrator))

In this case, any non-zero value of the ControlMechanism's `ControlSignal` will reset the TimerMechanism.
*OVERRIDE* must be used as its `modulation <ControlMechanism.modulation>` parameter (instead of its default value
of *MULTIPLICATIVE*), so that the value of the ControlMechanism's `ControlSignal` is assigned directly to the
TimerMechanism's `reset <TimerMechanism.reset>` parameter (otherwise, since the default of the `reset
<TimerMechanism.reset>` parameter is 0, the ControlSignal's value has no effect). An alternative is to specify
the **reset_default** agument in the TimerMechanism constructor with a non-zero value, and while allowing the
ControlMechanism to use its default value for `modulation <ControlMechanism.modulation>` (i.e., *MULTIPLICATIVE*)::

    >>> my_integrator = TimerMechanism(reset_default=1)
    >>> ctl_mech = pnl.ControlMechanism(control=(pnl.RESET, my_integrator))

In this case, a ControlSignal with a zero value suppresses a reset by multiplying the `reset
<TimerMechanism.reset>` parameter by 0, whereas a ControlSignal with a non-zero value multiples the `reset
<TimerMechanism.reset>` parameter's non-zero default value, resulting in a non-zero value that elicits a reset.

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
from psyneulink.core.components.functions.nonstateful.transferfunctions import TransferFunction, Linear
from psyneulink.core.components.functions.stateful.integratorfunctions import IntegratorFunction, SimpleIntegrator
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.mechanism import Mechanism, MechanismError
from psyneulink.core.globals.keywords import DEFAULT_VARIABLE, TIMER_MECHANISM, VARIABLE, PREFERENCE_SET_NAME, RESET
from psyneulink.core.globals.parameters import Parameter, check_user_specified
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
    TimerMechanism( \
        function=AdaptiveIntegrator(rate=0.5))

    Subclass of `IntegratorMechanism` that advances its input until it reaches a specified value.
    See `IntegratorMechanism <IntegratorMechanism_Class_Reference>` for additional arguments and attributes.

    Arguments
    ---------

    start : scalar, list or array : default 0
        specifies the starting value of the timer; if a list or array, the length must be the same as specified
        for **default_variable** or **input_shapes** (see `TimerMechanism_Processing` for additional details).

    increment : scalar, list or array : default 1
        specifies the amount by which the `previous_value <IntegratorFunction.previous_value>` of the
        TimerMechanism's `function <TimerMechanism.function>` is incremented each time the TimerMechanism is
        executed in the absence of input; if a list or array, the length must be the same as specified for
        **default_variable** or **input_shapes** (see `TimerMechanism_Processing` for additional details).

    COMMENT:
    TBI
    direction : INCREASING, DECREASING, or list of either/both : default INCREASING
        specifies whether the timer progresses in the direction of increasing or decreasing values; if a list or
        array, the length must be the same as specified for **default_variable** or **input_shapes**
        (see `TimerMechanism_Processing` for additional details).
    COMMENT

    end : scalar, list or array : default 1
        specifies the value at which the timer stops advancing; if a list or array, the length must be the same as
        specified for **default_variable** or **input_shapes** (see `TimerMechanism_Processing` for additional details).

    function : IntegratorFunction : default SimpleIntegrator(rate=1)
        specifies the function used to increment the input; must take a single numeric value, or a list or np.array
        of values, and return one of the same shape.

    trajectory : TransferFunction : default LINEAR
        specifies the shape of the timer's trajectory; must be a TransferFunction that takes a single numeric value,
        or a list or np.array of values, and returns one of the same shape.

    reset_default : number, list or np.ndarray : default 0
        specifies the default value used for the `reset <TimerMechanism.reset>` parameter.

    Attributes
    ----------

    start : scalar, list or array
        determines the starting value of the timer; assigned as the `initializer <IntegratorFunction.initializer>`
        Parameter of the TimerMechanism's `function <TimerMechanism.function>` (see `TimerMechanism_Processing` for
        additional details).

    increment : scalar, list or array
        determines the amount by which the `previous_value <IntegratorFunction.previous_value>` of the
        TimerMechanism's `function <TimerMechanism.function>` is incremented each time the TimerMechanism is
        executed in the absence of input; assigned as the `rate <IntegratorFunction.rate>` of the TimerMechanism's
        `function <TimerMechanism.function>` (see `TimerMechanism_Processing` for additional details).

    COMMENT:
    direction : INCREASING, DECREASING, or list of either/both
        determines whether the timer progresses in the direction of increasing or decreasing values.
        (see `TimerMechanism_Processing` for additional details).
    COMMENT

    end : scalar, list or array
        determines the value at which the timer stops advancing, after which it maintains that value;
        (see `TimerMechanism_Processing` for additional details).

    function : IntegratorFunction
        determines the function used to advance the input to the TimerMechanism's `trajectorytrajectory
        <TimerMechanism.trajectory>` Function; if the TimerMechanism receives an external input, that is
        used to advance the timer; otherwise, the `increment <TimerMechanism.increment>` Parameter is used.

    trajectory : TransferFunction
        determines the `Function` used to transform the ouput of the TimerMechanism's `function
        <TimerMechanism.function>` to generate its output; this determines the shape of the trajectory
        of the TimerMechanism's `value <TimerMechanism.value>`.

    reset : int, float or 1d array of length 1 : default 0
        if non-zero, the TimerMechanism's `reset <Mechanism_Base.reset>` method is called, which resets the
        `value <TimerMechanism.value>` of the TimerMechanism to its initial value (see
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
        start = Parameter(0, stateful=True, loggable=False)
        increment = Parameter(1, stateful=True, loggable=False)
        end = Parameter(1, stateful=True, loggable=False)
        function = Parameter(SimpleIntegrator(rate=1), stateful=False, loggable=False)
        trajectory = Parameter(Linear(), stateful=False, loggable=False)

        #
    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 input_shapes=None,
                 # FIX: TYPE HINTS NEEDED HERE OR HANDLED BY Parameters?
                 start:Optional[Union[int, float, list, np.ndarray]]=None,
                 increment:Optional[Union[int, float, list, np.ndarray]]=1,
                 end:Optional[Union[int, float, list, np.ndarray]]=1,
                 input_ports:Optional[Union[list, dict]]=None,
                 function:Optional[IntegratorFunction]=SimpleIntegrator(rate=1),
                 trajectory:Optional[TransferFunction]=Linear(),
                 reset_default=0,
                 output_ports:Optional[Union[str, Iterable]]=None,
                 params=None,
                 name=None,
                 prefs:   Optional[ValidPrefSet] = None,
                 **kwargs):
        """Assign type-level preferences, default input value (SigmoidLayer_DEFAULT_BIAS) and call super.__init__
        """

        super(TimerMechanism, self).__init__(default_variable=default_variable,
                                             input_shapes=input_shapes,
                                             start=start,
                                             increment=increment,
                                             end=end,
                                             function=function,
                                             trajectory=trajectory,
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
                raise TimerMechanismError(f"Shape of {repr(VARIABLE)} for function specified for {self.name} "
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
