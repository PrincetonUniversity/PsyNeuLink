
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***********************************************  RUN MODULE **********************************************************

"""

Overview
========

.. _Run_Overview:

The :keyword:`run` function is used for executing a Mechanism, Process or System.  It can be called directly, however
it is typically invoked by calling the :keyword:`run` method of the Component to be run.  It  executes a Component by
calling the Component's :keyword:`execute` method.  While a Component's :keyword:`execute` method can be called
directly, using its :keyword:`run` method is easier because it:

    * allows multiple rounds of execution to be run in sequence, whereas the :keyword:`execute` method of a Component
      runs only a single execution of the object;
    ..
    * uses simpler formats for specifying `inputs <Composition_Run_Inputs>` and `targets <Run_Targets>`;
    ..
    * automatically aggregates results across executions and stores them in the results attribute of the object.

Understanding a few basic concepts about how the :keyword:`run` function operates will make it easier to use the
:keyword:`execute` and :keyword:`run` methods of PsyNeuLink Components.  These are discussed below.


.. _Run_Scope_of_Execution:

*Execution Contexts*
====================

An *execution context* is a scope of execution which has its own set of values for Components and their `parameters <Parameters>`.
This is designed to prevent computations from interfering with each other, when Components are reused, which often occurs
when using multiple or nested Compositions, or running `simulations <OptimizationControlMechanism_Execution>`. Each execution context is
or is associated with an *execution_id*, which is often a user-readable string. An *execution_id* can be specified in a call to `Composition.run`,
or left unspecified, in which case the Composition's `default execution_id <Composition.default_execution_id>` would be used. When
looking for values after a run, it's important to know the execution context you are interested in, as shown below

::

        >>> import psyneulink as pnl
        >>> c = pnl.Composition()
        >>> d = pnl.Composition()
        >>> t = pnl.TransferMechanism()
        >>> c.add_node(t)
        >>> d.add_node(t)

        >>> t.execute(1)
        array([[1.]])
        >>> c.run({t: 5})
        [[array([5.])]]
        >>> d.run({t: 10})
        [[array([10.])]]
        >>> c.run({t: 20}, context='custom execution id')
        [[array([20.])]]

        # context None
        >>> print(t.parameters.value.get())
        [[1.]]
        >>> print(t.parameters.value.get(c))
        [[5.]]
        >>> print(t.parameters.value.get(d))
        [[10.]]
        >>> print(t.parameters.value.get('custom execution id'))
        [[20.]]

In general, anything that happens outside of a Composition run and without an explicit setting of execution context
occurs in the `None` execution context.


For Developers
--------------

.. _Run_Execution_Contexts_Init:

Initialization of Execution Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The parameter values for any execution context can be copied into another execution context by using \
Component._initialize_from_context, which when called on a Component copies the values for all its parameters \
and recursively for all of the Component's `_dependent_components <Component._dependent_components>`

- `_dependent_components <Component._dependent_components>` should be added to for any new Component that requires \
other Components to function properly (beyond "standard" things like Component.function, \
or Mechanism.input_ports, as these are added in the proper classes' _dependent_components)
    - the intent is that with ``_dependent_components`` set properly, calling \
    ``obj._initialize_from_context(new_context, base_context)`` should be sufficient to run obj \
    under **new_context**
    - a good example of a "nonstandard" override is `OptimizationControlMechanism._dependent_components`

.. _Run_Timing:

*Timing*
========

When :keyword:`run` is called by a Component, it calls that Component's :keyword:`execute` method once for each
`input <Composition_Run_Inputs>`  (or set of inputs) specified in the call to :keyword:`run`, which constitutes a `TRIAL` of
execution.  For each `TRIAL`, the Component makes repeated `calls to its Scheduler <Scheduler_Execution>`,
executing the Components it specifies in each `TIME_STEP`, until every Component has been executed at least once or
another `termination condition <Scheduler_Termination_Conditions>` is met.  The `Scheduler` can be used in combination
with `Condition` specifications for individual Components to execute different Components at different time scales.

.. _Composition_Run_Inputs:

*Inputs*
========

The :keyword:`run` function presents the inputs for each `TRIAL` to the input_ports of the relevant Mechanisms in
the `scope of execution <Run_Scope_of_Execution>`. These are specified in the **inputs** argument of a Component's
:keyword:`execute` or :keyword:`run` method.

Inputs are specified in a Python dictionary where the keys are `ORIGIN` Mechanisms, and the values are lists in which
the i-th element represents the input value to the Mechanism on trial i. Each input value must be compatible with the
shape of the mechanism's `external_input_values <MechanismBase.external_input_values>`. This means that the inputs to
an origin mechanism are usually specified by a list of 2d lists/arrays, though `some shorthand notations are allowed
<Input_Specification_Examples>`. Any InputPorts that are not represented in `external_input_values
<MechanismBase.external_input_values>` will not receive a user-specified input value.

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
        ...                          default_variable=[[0.0, 0.0]])
        >>> b = pnl.TransferMechanism(name='b',
        ...                          default_variable=[[0.0], [0.0]])
        >>> c = pnl.TransferMechanism(name='c')

        >>> p1 = pnl.Process(pathway=[a, c],
        ...                 name='p1')
        >>> p2 = pnl.Process(pathway=[b, c],
        ...                 name='p2')

        >>> s = pnl.System(processes=[p1, p2])

        >>> input_dictionary = {a: [[[1.0, 1.0]], [[1.0, 1.0]]],
        ...                    b: [[[2.0], [3.0]], [[2.0], [3.0]]]}

        >>> s.run(inputs=input_dictionary)

COMMENT:
    .. _Composition_Run_Inputs_Fig:

    .. figure:: _static/input_spec_variables.svg
       :alt: Example input specifications with variable
COMMENT

.. _Composition_Run_Inputs_Fig_States:

.. figure:: _static/input_spec_states.svg
   :alt: Example input specifications with input ports

.. note::
    Keep in mind that a mechanism's `external_input_values <MechanismBase.external_input_values>` attribute contains
    the concatenation of the values of its external InputPorts. Any InputPorts marked as "internal", such as
    InputPorts that receive recurrent Projections, are excluded from this value. A mechanism's `external_input_values
    <MechanismBase.external_input_values>` attribute is always a 2d list in which the index i element is the value of
    the Mechanism's index i InputPort. In many cases, `external_input_values <MechanismBase.external_input_values>` is
    the same as `variable <MechanismBase.variable>`

The number of inputs specified **must** be the same for all origin mechanisms in the system. In other words, all of the
values in the input dictionary must have the same length.

If num_trials is not in use, the number of inputs provided determines the number of trials in the run. For example, if
five inputs are provided for each origin mechanism, and num_trials is not specified, the system will execute five times.

+----------------------+-------+------+------+------+------+
| Trial #              |0      |1     |2     |3     |4     |
+----------------------+-------+------+------+------+------+
| Input to Mechanism a |1.0    |2.0   |3.0   |4.0   |5.0   |
+----------------------+-------+------+------+------+------+

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a')
        >>> b = pnl.TransferMechanism(name='b')

        >>> p1 = pnl.Process(pathway=[a, b])

        >>> s = pnl.System(processes=[p1])

        >>> input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        >>> s.run(inputs=input_dictionary)

If num_trials is in use, `run` will iterate over the inputs until num_trials is reached. For example, if five inputs
are provided for each `ORIGIN` mechanism, and num_trials = 7, the system will execute seven times. The first two
items in the list of inputs will be used on trial 5 and trial 6, respectively.

+----------------------+-------+------+------+------+------+------+------+
| Trial #              |0      |1     |2     |3     |4     |5     |6     |
+----------------------+-------+------+------+------+------+------+------+
| Input to Mechanism a |1.0    |2.0   |3.0   |4.0   |5.0   |1.0   |2.0   |
+----------------------+-------+------+------+------+------+------+------+

::

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        s.run(inputs=input_dictionary,
              num_trials=7)

.. _Input_Specification_Examples:

For convenience, condensed versions of the input specification described above are also accepted in the following
situations:

* **Case 1: Origin mechanism has only one InputPort**
+--------------------------+-------+------+------+------+------+
| Trial #                  |0      |1     |2     |3     |4     |
+--------------------------+-------+------+------+------+------+
| Input to **Mechanism a** |1.0    |2.0   |3.0   |4.0   |5.0   |
+--------------------------+-------+------+------+------+------+

Complete input specification:

::

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        s.run(inputs=input_dictionary)
..

Shorthand - drop the outer list on each input because **Mechanism a** only has one InputPort:

::

        input_dictionary = {a: [[1.0], [2.0], [3.0], [4.0], [5.0]]}

        s.run(inputs=input_dictionary)
..

Shorthand - drop the remaining list on each input because **Mechanism a**'s one InputPort's value is length 1:

::

        input_dictionary = {a: [1.0, 2.0, 3.0, 4.0, 5.0]}

        s.run(inputs=input_dictionary)
..

* **Case 2: Only one input is provided for the mechanism**

+--------------------------+------------------+
| Trial #                  |0                 |
+--------------------------+------------------+
| Input to **Mechanism a** |[[1.0], [2.0]]    |
+--------------------------+------------------+

Complete input specification:

::

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0], [0.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0], [2.0]]]}

        s.run(inputs=input_dictionary)
..

Shorthand - drop the outer list on **Mechanism a**'s input specification because there is only one trial:

::

        input_dictionary = {a: [[1.0], [2.0]]}

        s.run(inputs=input_dictionary)
..

* **Case 3: The same input is used on all trials**

+--------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Trial #                  |0                  |1                  |2                  |3                  |4                  |
+--------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Input to **Mechanism a** | [[1.0], [2.0]]    | [[1.0], [2.0]]    | [[1.0], [2.0]]    | [[1.0], [2.0]]    | [[1.0], [2.0]]    |
+--------------------------+-------------------+-------------------+-------------------+-------------------+-------------------+

Complete input specification:

::

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0], [0.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]]]}

        s.run(inputs=input_dictionary)
..

Shorthand - drop the outer list on **Mechanism a**'s input specification and use `num_trials` to repeat the input value

::

        input_dictionary = {a: [[1.0], [2.0]]}

        s.run(inputs=input_dictionary,
              num_trials=5)
..

* **Case 4: There is only one origin mechanism**

+--------------------------+-------------------+-------------------+
| Trial #                  |0                  |1                  |
+--------------------------+-------------------+-------------------+
| Input to **Mechanism a** | [1.0, 2.0, 3.0]   |  [1.0, 2.0, 3.0]  |
+--------------------------+-------------------+-------------------+

Complete input specification:

::

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[1.0, 2.0, 3.0]])
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = input_dictionary = {a: [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}

        s.run(inputs=input_dictionary)
..

Shorthand - specify **Mechanism a**'s inputs in a list because it is the only origin mechanism

::

        input_list = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        s.run(inputs=input_list)
..

.. _Run_Runtime_Parameters:

*Runtime Parameters*
====================

Runtime parameters are alternate parameter values that a Mechanism only uses under certain conditions. They are
specified in a nested dictionary containing (value, condition) tuples that correspond to parameters and Function
parameters of Mechanisms, which is passed into the `runtime_params <Run.runtime_params>` argument of `Run`.

Outer dictionary:
    - *key* - Mechanism
    - *value* - Runtime Parameter Specification Dictionary

Runtime Parameter Specification Dictionary:
    - *key* - keyword corresponding to a parameter of the Mechanism or its Function
    - *value* - tuple in which the index 0 item is the runtime parameter value, and the index 1 item is a `Condition`

If a runtime parameter is meant to be used throughout the `Run`, then the `Condition` may be omitted and the `Always`
`Condition` will be assigned by default:

>>> import psyneulink as pnl

>>> T = pnl.TransferMechanism()
>>> P = pnl.Process(pathway=[T])
>>> S = pnl.System(processes=[P])
>>> T.function.slope  # slope starts out at 1.0
1.0

>>> # During the following run, 10.0 will be used as the slope
>>> S.run(inputs={T: 2.0},
...       runtime_params={T: {"slope": 10.0}})
[ 20.]

>>> T.function.slope  # After the run, T.slope resets to 1.0

Otherwise, the runtime parameter value will be used on all executions of the
`Run` during which the `Condition` is True:

>>> T = pnl.TransferMechanism()
>>> P = pnl.Process(pathway=[T])
>>> S = pnl.System(processes=[P])

>>> T.function.intercept     # intercept starts out at 0.0
>>> T.function.slope         # slope starts out at 1.0

>>> S.run(inputs={T: 2.0},
...       runtime_params={T: {"intercept": (5.0, pnl.AfterTrial(1)),
...                           "slope": (2.0, pnl.AtTrial(3))}},
...       num_trials=5)
[[np.array([2.])], [np.array([2.])], [np.array([7.])], [np.array([9.])], [np.array([7.])]]

The table below shows how runtime parameters were applied to the intercept and slope parameters of Mechanism T in the
example above.

+-------------+--------+--------+--------+--------+--------+
|             |Trial 0 |Trial 1 |Trial 2 |Trial 3 |Trial 4 |
+=============+========+========+========+========+========+
| Intercept   |0.0     |0.0     |5.0     |5.0     |5.0     |
+-------------+--------+--------+--------+--------+--------+
| Slope       |1.0     |1.0     |1.0     |2.0     |0.0     |
+-------------+--------+--------+--------+--------+--------+
| Value       |2.0     |2.0     |7.0     |9.0     |7.0     |
+-------------+--------+--------+--------+--------+--------+

as indicated by the results of S.run(), the original parameter values were used on trials 0 and 1,
the runtime intercept was used on trials 2, 3, and 4, and the runtime slope was used on trial 3.

.. note::
    Runtime parameter values are subject to the same type, value, and shape requirements as the original parameter
    value.

COMMENT:
.. _Run_Initial_Values:

*Initial Values*
================

Any Mechanism that is the `sender <Projection_Base.sender>` of a Projection that closes a loop in a Process
or System, and that is not an `ORIGIN` Mechanism, is designated as `INITIALIZE_CYCLE`. An initial value can be assigned
to such Mechanisms, that will be used to initialize them when the Process or System is first run.  These values are
specified in the **initial_values** argument of :keyword:`run`, as a dictionary. The key for each entry must
be a Mechanism designated as `INITIALIZE_CYCLE`, and its value an input for the Mechanism to be used as its initial
value.  The size of the input (length of the outermost level if it is a list, or axis 0 if it is an np.ndarray),
must equal the number of InputPorts of the Mechanism, and the size of each value must match (in number and type of
elements) that of the `variable <InputPort.InputPort.variable>` for the corresponding InputPort.
COMMENT

.. _Run_Targets:

*Targets*
=========

If learning is specified for a `Process <Process_Learning_Sequence>` or `System <System_Execution_Learning>`, then
target values for each `TRIAL` must be provided for each `TARGET` Mechanism in the Process or System being run.  These
are specified in the **targets** argument of the :keyword:`execute` or :keyword:`run` method.

Recall that the `TARGET`, or `ComparatorMechanism`, of a learning sequence receives a TARGET, which is provided by the
user at run time, and a SAMPLE, which is received from a projection sent by the last mechanism of the learning sequence.
The TARGET and SAMPLE values for a particular `TARGET` Mechanism must have the same shape. See `learning sequence
<Process_Learning_Sequence>` for more details on how these components relate to each other.

The standard format for specifying targets is a Python dictionary where the keys are the last mechanism of each learning
sequence, and the values are lists in which the i-th element represents the target value for that learning sequence on
trial i. There must be the same number of keys in the target specification dictionary as there are `TARGET` Mechanisms
in the system. Each target value must be compatible with the shape of the `TARGET` mechanism's TARGET `InputPort
<ComparatorMechanism.input_ports>`. This means that for a given key (which is always the last mechanism of the
learning sequence) in the target specification dictionary, the value is usually a list of 1d lists/arrays.

The number of targets specified for each Mechanism must equal the number specified for the **inputs** argument;  as
with **inputs**, if the number of `TRIAL` \\s specified is greater than the number of inputs (and targets), then the
list will be cycled until the number of `TRIAL` \\s specified is completed.

+------------------------------------------+--------------+--------------+
| Trial #                                  |0             |   1          |
+------------------------------------------+--------------+--------------+
| Target value for the learning sequence   | [1.0, 1.0]   |   [2.0, 2.0] |
| containing **Mechanism b**               |              |              |
+------------------------------------------+--------------+--------------+
| Target value for the learning sequence   |  [1.0]       |   [2.0]      |
| containing **Mechanism c**               |              |              |
+------------------------------------------+--------------+--------------+

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name="a")
        >>> b = pnl.TransferMechanism(name="b",
        ...                           default_variable=np.array([[0.0, 0.0]]))
        >>> c = pnl.TransferMechanism(name="c")

        >>> learning_sequence_1 = pnl.Process(name="learning-sequence-1",
        ...                                   pathway=[a, b],
        ...                                   learning=pnl.ENABLED)
        >>> learning_sequence_2 = pnl.Process(name="learning-sequence-2",
        ...                                   pathway=[a, c],
        ...                                   learning=pnl.ENABLED)


        >>> s = pnl.System(name="learning-system",
        ...                processes=[learning_sequence_1, learning_sequence_2])

        >>> input_dictionary = {a: [[[0.1]], [[0.2]]]}

        >>> target_dictionary = {b: [[1.0, 1.0], [2.0, 2.0]],
        ...                      c: [[1.0], [2.0]]}

        >>> s.run(inputs=input_dictionary,
        ...       targets=target_dictionary)

.. _Run_Targets_Fig:

.. figure:: _static/target_spec_dictionary.svg
   :alt: Example of dictionary format of target specification

Alternatively, the value for a given key (last mechanism in the learning sequence) in the target specification
dictionary may be a function. The output of that function must be compatible with the shape of the `TARGET` mechanism's
TARGET `InputPort <ComparatorMechanism.input_ports>`. The function will be executed at the start of the learning
portion of each trial. This format allows targets to be constructed programmatically, in response
to computations made during the run.

::

        >>> a = TransferMechanism(name="a")
        >>> b = TransferMechanism(name="b",
        ...                       default_variable=np.array([[0.0, 0.0]]))

        >>> learning_sequence = Process(name="learning-sequence",
        ...                             pathway=[A, B],
        ...                             learning=ENABLED)

        >>> s = System(name="learning-system",
        ...            processes=[LP])

        >>> def target_function():
        ...     val_1 = NormalDist(mean=3.0)()
        ...     val_2 = NormalDist(mean=3.0)()
        ...     target_value = np.array([val_1, val_2])
        ...     return target_value

        >>> s.run(inputs={A: [[[1.0]], [[2.0]], [[3.0]]]},
        ...       targets={B: target_function})

.. note::

    Target specification dictionaries that provide values for multiple learning sequences may contain functions for some
    learning sequences and lists of values for others.

Finally, for convenience, if there is only one learning sequence in a system, the targets may be specified in a list,
rather than a dictionary.

+------------------------------------------+-------+------+------+------+------+
| Trial #                                  |0      |1     |2     |3     |4     |
+------------------------------------------+-------+------+------+------+------+
| Target corresponding to  **Mechanism b** |1.0    |2.0   |3.0   |4.0   |5.0   |
+------------------------------------------+-------+------+------+------+------+

Complete input specification:

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a')
        >>> b = pnl.TransferMechanism(name='b')

        >>> p1 = pnl.Process(pathway=[a, b])

        >>> s = pnl.System(processes=[p1])

        >>> input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}
        >>> target_dictionary = {b: [[1.0], [2.0], [3.0], [4.0], [5.0]]}

        >>> s.run(inputs=input_dictionary,
        ...       targets=target_dictionary)

Shorthand - specify the targets in a list because there is only one learning sequence:

::

        >>> target_list = [[1.0], [2.0], [3.0], [4.0], [5.0]]

        >>> s.run(inputs=input_dictionary,
        ...       targets=target_list)


.. _Run_Class_Reference:

Class Reference
===============

"""

import datetime
import types
import warnings

from collections.abc import Iterable
from numbers import Number

import numpy as np
import typecheck as tc

from psyneulink.core.components.shellclasses import Mechanism, Process_Base, System_Base
from psyneulink.core.globals.context import ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import INPUT_LABELS_DICT, MECHANISM, OUTPUT_LABELS_DICT, PROCESS, RUN, SAMPLE, SYSTEM, TARGET
from psyneulink.core.globals.log import LogCondition
from psyneulink.core.globals.utilities import call_with_pruned_args
from psyneulink.core.scheduling.time import TimeScale

__all__ = [
    'RunError', 'run'
]

class RunError(Exception):
     def __init__(obj, error_value):
         obj.error_value = error_value

     def __str__(obj):
         return repr(obj.error_value)

@tc.typecheck
@handle_external_context()
def run(obj,
        inputs=None,
        num_trials:tc.optional(int)=None,
        initialize:bool=False,
        initial_values:tc.optional(tc.any(list, dict, np.ndarray))=None,
        targets=None,
        learning:tc.optional(bool)=None,
        call_before_trial:tc.optional(callable)=None,
        call_after_trial:tc.optional(callable)=None,
        call_before_time_step:tc.optional(callable)=None,
        call_after_time_step:tc.optional(callable)=None,
        termination_processing=None,
        termination_learning=None,
        runtime_params=None,
        context=None,
        ):
    """run(                      \
    inputs,                      \
    num_trials=None,             \
    initialize=False,            \
    intial_values=None,          \
    targets=None,                \
    learning=None,               \
    call_before_trial=None,      \
    call_after_trial=None,       \
    call_before_time_step=None,  \
    call_after_time_step=None,   \
    termination_processing=None, \
    termination_learning=None,   \
    runtime_params=None,         \
    )

    Run a sequence of executions for a `Process` or `System`.

    COMMENT:
        First, validate inputs (and targets, if learning is enabled).  Then, for each `TRIAL`:
            * call call_before_trial if specified;
            * for each time_step in the trial:
                * call call_before_time_step if specified;
                * call ``obj.execute`` with inputs, and append result to ``obj.results``;
                * call call_after_time_step if specified;
            * call call_after_trial if specified.
        Return ``obj.results``.

        The inputs argument must be a list or an np.ndarray array of the appropriate dimensionality:
            * the inner-most dimension must equal the length of obj.defaults.variable (i.e., the input to the obj);
            * for Mechanism format, the length of the value of all entries must be equal (== number of executions);
            * the outer-most dimension is the number of input sets (num_input_sets) specified (one per execution)
                Note: num_input_sets need not equal num_trials (the number of executions to actually run)
                      if num_trials > num_input_sets:
                          executions will cycle through input_sets, with the final one being only a partial cycle
                      if num_trials < num_input_sets:
                          the executions will only partially sample the input sets
    COMMENT

   Arguments
   ---------

    inputs : List[input] or ndarray(input) : default default_variable for a single `TRIAL`
        the input for each `TRIAL` in a sequence (see `Composition_Run_Inputs` for detailed description of formatting
        requirements and options).

    num_trials : int : default None
        the number of `TRIAL` \\s to run.  If it is `None` (the default), then a number of `TRIAL` \\s run will be equal
        equal to the number of items specified in the **inputs** argument.  If **num_trials** exceeds the number of
        inputs, then the inputs will be cycled until the number of `TRIAL` \\s specified have been run.

    initialize : bool default False
        calls the `initialize <System.initialize>` method of the System prior to the first `TRIAL`.

    initial_values : Dict[Mechanism:List[input]], List[input] or np.ndarray(input) : default None
        the initial values assigned to Mechanisms designated as `INITIALIZE_CYCLE`.

    targets : dict : default None
        the target values assigned to the `ComparatorMechanism` of each learning sequence on each `TRIAL`.

    learning : bool :  default None
        enables or disables learning during execution for a `Process <Process_Execution_Learning>` or
        `System <System_Execution_Learning>`.  If it is not specified, the current state of learning is left intact.
        If it is `True`, learning is forced on; if it is `False`, learning is forced off.

    call_before_trial : Function : default `None`
        called before each `TRIAL` in the sequence is run.

    call_after_trial : Function : default `None`
        called after each `TRIAL` in the sequence is run.

    call_before_time_step : Function : default ``None`
        called before each `TIME_STEP` is executed.

    call_after_time_step : Function : default `None`
        called after each `TIME_STEP` is executed.

    termination_processing : Dict[TimeScale: Condition]
        a dictionary containing `Condition`\\ s that signal the end of the associated `TimeScale` within the :ref:`processing
        phase of execution <System_Execution_Processing>`

    termination_learning : Dict[TimeScale: Condition]
        a dictionary containing `Condition`\\ s that signal the end of the associated `TimeScale` within the :ref:`learning
        phase of execution <System_Execution_Learning>`

    runtime_params : Dict[Mechanism: Dict[Parameter: Tuple(Value, Condition)]]
        nested dictionary of (value, `Condition`) tuples for parameters of Mechanisms of the Composition; specifies
        alternate parameter values to be used only during this `Run` when the specified `Condition` is met.

        Outer dictionary:
            - *key* - Mechanism
            - *value* - Runtime Parameter Specification Dictionary

        Runtime Parameter Specification Dictionary:
            - *key* - keyword corresponding to a parameter of the Mechanism
            - *value* - tuple in which the index 0 item is the runtime parameter value, and the index 1 item is a
              `Condition`

        See `Run_Runtime_Parameters` for more details and examples of valid dictionaries.

   Returns
   -------

    <obj>.results : List[OutputPort.value]
        list of the values, for each `TRIAL`, of the OutputPorts for a Mechanism run directly,
        or of the OutputPorts of the `TERMINAL` Mechanisms for the Process or System run.
    """
    from psyneulink.core.globals.context import ContextFlags

    if inputs is None:
        inputs = {}

    # small version of 'sequence' format in the once case where it was still working (single origin mechanism)
    if isinstance(inputs, (list, np.ndarray)):
        if len(obj.origin_mechanisms) == 1:
            inputs = {obj.origin_mechanisms[0]: inputs}
        else:
            raise RunError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                           "mechanisms.".format(obj.name, len(obj.origin_mechanisms)))
    elif not isinstance(inputs, dict) and not isinstance(inputs, str):
        if len(obj.origin_mechanisms) == 1:
            raise RunError("Inputs to {} must be specified in a list or in a dictionary with the origin mechanism({}) "
                           "as its only key".format(obj.name, obj.origin_mechanisms[0].name))
        else:
            raise RunError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                           "mechanisms.".format(obj.name, len(obj.origin_mechanisms)))

    inputs, num_inputs_sets = _adjust_stimulus_dict(obj, inputs)

    if num_trials is not None:
        num_trials = num_trials
    else:
        num_trials = num_inputs_sets

    # num_trials = num_trials or num_inputs_sets  # num_trials may be provided by user, otherwise = # of input sets

    if targets is not None:

        if isinstance(targets, dict):
            targets, num_targets = _adjust_target_dict(obj, targets)

        elif isinstance(targets, (list, np.ndarray)):
            # small version of former 'sequence' format -- only allowed if there is a single Target mechanism
            if len(obj.target_mechanisms) == 1:
                targets = {obj.target_mechanisms[0].input_ports[SAMPLE].path_afferents[0].sender.owner: targets}
                targets, num_targets = _adjust_target_dict(obj, targets)
            else:
                raise RunError("Target values for {} must be specified in a dictionary.".format(obj.name))

        elif isinstance(targets, types.FunctionType):
            if len(obj.target_mechanisms) == 1:
                targets = {obj.target_mechanisms[0].input_ports[SAMPLE].path_afferents[0].sender.owner: targets}
                targets, num_targets = _adjust_target_dict(obj, targets)
            else:
                raise RunError("Target values for {} must be specified in a dictionary.".format(obj.name))
        else:
            raise RunError("Target values for {} must be specified in a dictionary.".format(obj.name))

        # if num_targets = -1, all targets were specified as functions
        if num_targets != num_inputs_sets and num_targets != -1:
            raise RunError("Number of target values specified ({}) for each learning sequence in {} must equal the "
                           "number of input values specified ({}) for each origin mechanism in {}."
                           .format(num_targets, obj.name, num_inputs_sets, obj.name))

    object_type = _get_object_type(obj)

    obj.targets = targets

    # SET LEARNING (if relevant)
    # FIX: THIS NEEDS TO BE DONE FOR EACH PROCESS IF THIS CALL TO run() IS FOR SYSTEM
    #      IMPLEMENT learning_enabled FOR SYSTEM, WHICH FORCES LEARNING OF PROCESSES WHEN SYSTEM EXECUTES?
    #      OR MAKE LEARNING A PARAM THAT IS PASSED IN execute
    # If learning is specified, buffer current state and set to specified state
    if learning is not None:
        try:
            learning_state_buffer = obj._learning_enabled
        except AttributeError:
            if obj.verbosePref:
                warnings.warn("WARNING: learning not enabled for {}".format(obj.name))
        else:
            if learning is True:
                obj._learning_enabled = True

            elif learning is False:
                obj._learning_enabled = False

    # SET LEARNING_RATE, if specified, for all learningProjections in process or system
    if obj.learning_rate is not None:
        from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
        for learning_mech in obj.learning_mechanisms.mechanisms:
            for projection in learning_mech.output_port.efferents:
                if isinstance(projection, LearningProjection):
                    projection.function_obj.learning_rate = obj.learning_rate

    # INITIALIZATION
    if initialize:
        obj.initialize(context=context)

    # SET UP TIMING
    if object_type == MECHANISM:
        time_steps = 1
    else:
        time_steps = obj.numPhases

    # EXECUTE
    execution_inputs = {}
    execution_targets = {}
    for execution in range(num_trials):

        if call_before_trial:
            call_with_pruned_args(call_before_trial, context=context)

        for time_step in range(time_steps):

            result = None

            if call_before_time_step:
                call_with_pruned_args(call_before_time_step, context=context)

            # Reinitialize any mechanisms that has a 'reinitialize_when' condition and it is satisfied
            for mechanism in obj.mechanisms:
                if hasattr(mechanism, "reinitialize_when") and mechanism.parameters.has_initializers._get(context):
                    if mechanism.reinitialize_when.is_satisfied(scheduler=obj.scheduler, context=context):
                        mechanism.reinitialize(None, context=context)

            input_num = execution % num_inputs_sets

            for mech in inputs:
                execution_inputs[mech] = inputs[mech][input_num]
            if object_type == SYSTEM:
                obj.inputs = execution_inputs

            # Assign targets:
            if targets is not None:

                if isinstance(targets, types.FunctionType):
                    obj.target = targets
                else:
                    for mech in targets:
                        if callable(targets[mech]):
                            execution_targets[mech] = targets[mech]
                        else:
                            execution_targets[mech] = targets[mech][input_num]
                    if object_type is SYSTEM:
                        obj.target = execution_targets
                        obj.current_targets = execution_targets

            if context.source == ContextFlags.COMMAND_LINE or ContextFlags.SIMULATION not in context.execution_phase:
                context.execution_phase = ContextFlags.PROCESSING
                context.composition = obj

            result = obj.execute(
                input=execution_inputs,
                target=execution_targets,
                context=context,
                termination_processing=termination_processing,
                termination_learning=termination_learning,
                runtime_params=runtime_params,

            )

            if call_after_time_step:
                call_with_pruned_args(call_after_time_step, context=context)

        if ContextFlags.SIMULATION not in context.execution_phase:
            if isinstance(result, Iterable):
                result_copy = result.copy()
            else:
                result_copy = result
            obj.results.append(result_copy)

        if call_after_trial:
            call_with_pruned_args(call_after_trial, context=context)

        from psyneulink.core.globals.log import _log_trials_and_runs, ContextFlags
        _log_trials_and_runs(
            composition=obj,
            curr_condition=LogCondition.TRIAL,
            context=context,
        )

    try:
        obj.scheduler.date_last_run_end = datetime.datetime.now()
        obj.scheduler_learning.date_last_run_end = datetime.datetime.now()

        for sched in [obj.scheduler, obj.scheduler_learning]:
            try:
                sched.get_clock(context)._increment_time(TimeScale.RUN)
            except KeyError:
                # learning scheduler may not have been execute, so may not have
                # created a Clock for context
                pass

    except AttributeError:
        # this will fail on processes, which do not have schedulers
        pass

    # Restore learning state
    try:
        learning_state_buffer
    except UnboundLocalError:
        pass
    else:
        obj._learning_enabled = learning_state_buffer

    from psyneulink.core.globals.log import _log_trials_and_runs
    _log_trials_and_runs(
        composition=obj,
        curr_condition=LogCondition.RUN,
        context=context
    )

    return obj.results

@tc.typecheck
def _input_matches_external_input_port_values(input, value_to_compare):
    # input ports are uniform
    if np.shape(np.atleast_2d(input)) == np.shape(value_to_compare):
        return "homogeneous"
    # input ports have different lengths
    elif len(np.shape(value_to_compare)) == 1 and isinstance(value_to_compare[0], (list, np.ndarray)):
        for i in range(len(input)):
            if len(input[i]) != len(value_to_compare[i]):
                return False
        return "heterogeneous"
    return False

def _target_matches_input_port_variable(target, input_port_variable):
    if np.shape(np.atleast_1d(target)) == np.shape(input_port_variable):
        return True
    return False

def _adjust_stimulus_dict(obj, stimuli):

    #  STEP 0:  parse any labels into array entries
    need_parse_input_labels = []
    for mech in obj.origin_mechanisms:
        if hasattr(mech, "input_labels_dict"):
            if mech.input_labels_dict is not None and mech.input_labels_dict != {}:
                need_parse_input_labels.append(mech)
    if len(need_parse_input_labels) > 0:
        stimuli = _parse_input_labels(obj, stimuli, need_parse_input_labels)

    # STEP 1: validate that there is a one-to-one mapping of input entries to origin mechanisms

    # Check that all of the mechanisms listed in the inputs dict are ORIGIN mechanisms in the object
    for mech in stimuli.keys():
        if not mech in obj.origin_mechanisms.mechanisms:
            raise RunError("{} in inputs dict for {} is not one of its ORIGIN mechanisms".
                           format(mech.name, obj.name))

    # Check that all of the ORIGIN mechanisms in the obj are represented by entries in the inputs dict
    # If not, assign their default variable to the dict
    for mech in obj.origin_mechanisms:
        if not mech in stimuli:
            stimuli[mech] = mech.defaults.variable.copy()

    # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

    # (1) Replace any user provided convenience notations with values that match the following specs:
    # a - all dictionary values are lists containing and input value on each trial (even if only one trial)
    # b - each input value is a 2d array that matches external_input_values
    # example: { Mech1: [Fully_specified_input_for_mech1_on_trial_1, Fully_specified_input_for_mech1_on_trial_2 … ],
    #            Mech2: [Fully_specified_input_for_mech2_on_trial_1, Fully_specified_input_for_mech2_on_trial_2 … ]}
    # (2) Verify that all mechanism values provide the same number of inputs (check length of each dictionary value)

    adjusted_stimuli = {}
    num_input_sets = -1

    for mech, stim_list in stimuli.items():

        check_spec_type = _input_matches_external_input_port_values(stim_list, mech.external_input_values
                                                                     )
        # If a mechanism provided a single input, wrap it in one more list in order to represent trials
        if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
            if check_spec_type == "homogeneous":
                # np.atleast_2d will catch any single-input ports specified without an outer list
                # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                adjusted_stimuli[mech] = [np.atleast_2d(stim_list)]
            else:
                adjusted_stimuli[mech] = [stim_list]

            # verify that all mechanisms have provided the same number of inputs
            if num_input_sets == -1:
                num_input_sets = 1
            elif num_input_sets != 1:
                raise RunError("Input specification for {} is not valid. The number of inputs (1) provided for {} "
                               "conflicts with at least one other mechanism's input specification.".format(obj.name,
                                                                                                           mech.name))
        else:
            adjusted_stimuli[mech] = []
            for stim in stimuli[mech]:
                check_spec_type = _input_matches_external_input_port_values(stim, mech.external_input_values)

                # loop over each input to verify that it matches external_input_values
                if check_spec_type == False:
                    err_msg = "Input stimulus ({}) for {} is incompatible with its external_input_values ({}).".\
                        format(stim, mech.name, mech.external_input_values
)
                    # 8/3/17 CW: The error message implementation here is very hacky; but it's at least not a hack
                    # for "functionality" but rather a hack for user clarity
                    if "KWTA" in str(type(mech)):
                        err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other" \
                                            " values) to represent the outside stimulus for the inhibition InputPort"
                    raise RunError(err_msg)
                elif check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input ports specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[mech].append(np.atleast_2d(stim))
                else:
                    adjusted_stimuli[mech].append(stim)

            # verify that all mechanisms have provided the same number of inputs
            if num_input_sets == -1:
                num_input_sets = len(stimuli[mech])
            elif num_input_sets != len(stimuli[mech]):
                raise RunError("Input specification for {} is not valid. The number of inputs ({}) provided for {} "
                               "conflicts with at least one other mechanism's input specification."
                               .format(obj.name, (stimuli[mech]), mech.name))

    return adjusted_stimuli, num_input_sets

def _adjust_target_dict(component, target_dict):

    #  STEP 0:  parse any labels into array entries
    need_parse_target_labels = []
    for mech in target_dict:
        if hasattr(mech, "output_labels_dict"):
            if mech.output_labels_dict is not None and mech.output_labels_dict != {}:
                need_parse_target_labels.append(mech)
    if len(need_parse_target_labels) > 0:
        target_dict = _parse_target_labels(component, target_dict, need_parse_target_labels)

    # STEP 1: validate that there is a one-to-one mapping of target entries and target mechanisms
    for target_mechanism in component.target_mechanisms:
        # If any projection to a target does not have a sender in the stimulus dict, raise an exception
        if not any(mech is projection.sender.owner for
                   projection in target_mechanism.input_ports[SAMPLE].path_afferents
                   for mech in target_dict.keys()):
                raise RunError("Entry for {} is missing from specification of targets for run of {}".
                               format(target_mechanism.input_ports[SAMPLE].path_afferents[0].sender.owner.name,
                                      component.name))

    for mech in target_dict:
        # If any mechanism in the target dict does not have a projection to a target, raise an error
        if not any(target is projection.receiver.owner for
                   projection in mech.output_port.efferents
                   for target in component.target_mechanisms):
            raise RunError("{} does not project to a target Mechanism in {}".format(mech.name, component.name))

        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

        # (1) Replace any user provided convenience notations with values that match the following specs:
        # a - all dictionary values are lists containing a target value on each trial (even if only one trial)
        # b - each input value is at least a 1d array that matches the variable of the TARGET InputPort

        # (2) Verify that all mechanism values provide the same number of inputs (check length of each dictionary value)

    adjusted_targets = {}
    num_targets = -1
    for mech, target_list in target_dict.items():
        if isinstance(target_list, (float, list, np.ndarray)):
            for efferent_projection in mech.output_port.efferents:
                for input_port in efferent_projection.receiver.owner.input_ports:
                    if input_port.name == TARGET:
                        input_port_variable = input_port.socket_template
                        break
            num_targets = -1

            # first check if only one target was provided:
            if np.shape(np.atleast_1d(target_list)) == np.shape(input_port_variable):
                adjusted_targets[mech] = [np.atleast_1d(target_list)]
                if num_targets == -1:
                    num_targets = 1
                elif num_targets != 1:
                    raise RunError("Target specification for {} is not valid. The number of targets (1) provided for {}"
                                   "conflicts with at least one other mechanism's target specification."
                                   .format(component.name, mech.name))

            # iterate over list and check that each candidate target is compatible with corresponding TARGET InputPort
            elif isinstance(target_list, (list, np.ndarray)):
                adjusted_targets[mech] = []
                for target_value in target_list:
                    if np.shape(np.atleast_1d(target_value)) == np.shape(input_port_variable):
                        adjusted_targets[mech].append(np.atleast_1d(target_value))
                    else:
                        raise RunError("Target specification ({}) for {} is not valid. The shape of {} is not compatible "
                                       "with the TARGET InputPort of the corresponding ComparatorMechanism ({})"
                                       .format(target_list, mech.name, target_value,
                                               mech.output_port.efferents[0].receiver.owner.name))
                current_num_targets = len(adjusted_targets[mech])
                # verify that all mechanisms have provided the same number of inputs
                if num_targets == -1:
                    num_targets = current_num_targets
                elif num_targets != current_num_targets:
                    raise RunError("Target specification for {} is not valid. The number of targets ({}) provided for {}"
                                   "conflicts with at least one other mechanism's target specification."
                                   .format(component.name, current_num_targets, mech.name))

        elif callable(target_list):
            _validate_target_function(target_list, mech.output_port.efferents[0].receiver.owner, mech)
            adjusted_targets[mech] = target_list
    return adjusted_targets, num_targets

@tc.typecheck
def _parse_input_labels(obj, stimuli, mechanisms_to_parse):

    def get_input_for_label(mech, key):
        """check mech.input_labels_dict for key"""
        try:
            return mech.input_labels_dict[key]
        except KeyError:
            raise RunError("No entry \'{}\' found for input to {} in {} for mech.name".
                           format(key, obj.name, INPUT_LABELS_DICT, mech.name))

    if len(mechanisms_to_parse) == 1:
        if isinstance(stimuli, float):
            return stimuli
        elif isinstance(stimuli, str):
            stimuli = {mechanisms_to_parse[0]: [stimuli]}

    for mech in mechanisms_to_parse:
        inputs = stimuli[mech]

        # Check for subdicts
        subdicts = False
        for k in mech.input_labels_dict:
            value = mech.input_labels_dict[k]
            if isinstance(value, dict):
                subdicts = True
                break

        if subdicts:    # If there are subdicts, validate
            # if len(mech.input_labels_dict) != len(mech.input_ports):
            #     raise RunError("If input labels are specified at the level of input ports, then one InputPort label "
            #                    "sub-dictionary must be provided for each InputPort. {} has {} InputPort label "
            #                    "sub-dictionaries, but {} input ports.".format(mech.name,
            #                                                                    len(mech.input_labels_dict),
            #                                                                    len(mech.input_ports)))
            for k in mech.input_labels_dict:
                value = mech.input_labels_dict[k]
                if not isinstance(value, dict):
                    raise RunError("A sub-dictionary  of label:value pairs was not specified for the InputPort {} of "
                                   "{}. If input labels are specified at the level of InputPorts, then a sub-dictionary"
                                   " must be provided for each InputPort in the input labels dictionary"
                                   .format(k, mech.name))

            # If there is only one subdict, then we already know that we are in the correct InputPort
            num_input_labels = len(mech.input_labels_dict)
            if num_input_labels == 1:
                # there is only one key, but we don't know what it is
                for k in mech.input_labels_dict:
                    for i in range(len(inputs)):
                        # if the whole input spec is a string, look up its value
                        if isinstance(inputs[i], str):
                            inputs[i] = mech.input_labels_dict[k][inputs[i]]
                        # otherwise, index into [0] because we know that this label is for the primary InputPort
                        elif isinstance(inputs[i][0], str):
                            inputs[i][0] = mech.input_labels_dict[k][inputs[i][0]]

            else:
                for trial_stimulus in inputs:
                    for input_port_index in range(len(trial_stimulus)):
                        if isinstance(trial_stimulus[input_port_index], str):
                            label_to_parse = trial_stimulus[input_port_index]
                            input_port_name = mech.input_ports[input_port_index].name
                            if input_port_index in mech.input_labels_dict:
                                trial_stimulus[input_port_index] = \
                                    mech.input_labels_dict[input_port_index][label_to_parse]
                            elif input_port_name in mech.input_labels_dict:
                                trial_stimulus[input_port_index] = \
                                    mech.input_labels_dict[input_port_name][label_to_parse]

        else:
            for i, stim in enumerate(inputs):
                # "Burrow" down to determine whether there's a number at the "bottom";
                #     if so, leave as is; otherwise, check if its a string and, if so, get value for label
                if isinstance(stim, (list, np.ndarray)): # format of stimuli dict is at least: [[???]...?]
                    for j, item in enumerate(stim):
                        if isinstance(item, (Number, list, np.ndarray)): # format of stimuli dict is [[int or []...?]]
                            continue # leave input item as is
                        elif isinstance(item, str): # format of stimuli dict is [[label]...]
                            # inputs[i][j] = get_input_for_label(mech, item, stim)
                            inputs[i][j] = get_input_for_label(mech, item)
                elif isinstance(stim, str):
                    inputs[i] = get_input_for_label(mech, stim)
        return stimuli

def _parse_target_labels(obj, target_dict, mechanisms_to_parse):
    if len(mechanisms_to_parse) == 1:
        if isinstance(target_dict, float):
            return target_dict
        elif isinstance(target_dict, str):
            target_dict= {mechanisms_to_parse[0]: [target_dict]}
        elif isinstance(target_dict, (list, np.ndarray)):
            target_dict = {mechanisms_to_parse[0]: target_dict}
    def get_target_for_label(mech, key):
        """check mech.input_labels_dict for key"""

        try:
            return mech.output_labels_dict[key]
        except KeyError:
            raise RunError("No entry \'{}\' found for input to {} in {} for mech.name".
                           format(key, obj.name, OUTPUT_LABELS_DICT, mech.name))

    for mech in mechanisms_to_parse:
        targets = target_dict[mech]
        # Check for subdicts
        subdicts = False
        for k in mech.output_labels_dict:
            value = mech.output_labels_dict[k]
            if isinstance(value, dict):
                subdicts = True
                break

        if subdicts:    # If there are subdicts, validate
            for key in mech.output_labels_dict:
                output_port = mech.output_ports[key]
                for proj in output_port.efferents:
                    if proj.receiver.name == SAMPLE:
                        output_port_index = mech.output_ports.index(output_port)
                        output_port_name = output_port.name

            for i in range(len(targets)):
                trial_target = targets[i]
                if isinstance(trial_target, str):
                    if output_port_index in mech.output_labels_dict:
                        targets[i] = mech.output_labels_dict[output_port_index][trial_target]
                    elif output_port_name in mech.output_labels_dict:
                        targets[i] = mech.output_labels_dict[output_port_name][trial_target]

        else:
            for i, stim in enumerate(targets):
                # "Burrow" down to determine whether there's a number at the "bottom";
                #     if so, leave as is; otherwise, check if its a string and, if so, get value for label
                if isinstance(stim, (list, np.ndarray)): # format of stimuli dict is at least: [[???]...?]
                    for j, item in enumerate(stim):
                        if isinstance(item, (Number, list, np.ndarray)): # format of stimuli dict is [[int or []...?]]
                            continue # leave input item as is
                        elif isinstance(item, str): # format of stimuli dict is [[label]...]
                            # targets[i][j] = get_input_for_label(mech, item, stim)
                            targets[i][j] = get_target_for_label(mech, item)
                        else:
                            raise RunError("Unrecognized specification ({}) in stimulus {} of entry "
                                           "for {} in targets dictionary specified for {}".
                                           format(item, i, mech.name, obj.name))
                elif isinstance(stim, str):
                    targets[i] = get_target_for_label(mech, stim)
                else:
                    raise RunError("Unrecognized specification ({}) for stimulus {} in entry "
                                   "for {} of targets dictionary specified for {}".
                                   format(stim, i, mech.name, obj.name))
    return target_dict
def _validate_target_function(target_function, target_mechanism, sample_mechanism):

    generated_targets = np.atleast_1d(target_function())
    expected_shape = target_mechanism.input_ports[TARGET].socket_template
    if np.shape(generated_targets) != np.shape(expected_shape):
            raise RunError("Target values generated by target function ({}) are not compatible with TARGET InputPort "
                           "of {} ({}). See {} entry in target specification dictionary. "
                           .format(generated_targets, target_mechanism.name, expected_shape, sample_mechanism.name))

def _get_object_type(obj):
    if isinstance(obj, Mechanism):
        return MECHANISM
    elif isinstance(obj, Process_Base):
        return PROCESS
    elif isinstance(obj, System_Base):
        return SYSTEM
    else:
        raise RunError("{} type not supported by Run module".format(obj.__class__.__name__))
