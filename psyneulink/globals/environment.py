
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***********************************************  RUN MODULE **********************************************************

"""

Overview
--------

.. _Run_Overview:

The :keyword:`run` function is used for executing a Mechanism, Process or System.  It can be called directly, however
it is typically invoked by calling the :keyword:`run` method of the Component to be run.  It  executes a Component by
calling the Component's :keyword:`execute` method.  While a Component's :keyword:`execute` method can be called
directly, using its :keyword:`run` method is easier because it:

    * allows multiple rounds of execution to be run in sequence, whereas the :keyword:`execute` method of a Component
      runs only a single execution of the object;
    ..
    * uses simpler formats for specifying `inputs <Run_Inputs>` and `targets <Run_Targets>`;
    ..
    * automatically aggregates results across executions and stores them in the results attribute of the object.

Understanding a few basic concepts about how the :keyword:`run` function operates will make it easier to use the
:keyword:`execute` and :keyword:`run` methods of PsyNeuLink Components.  These are discussed below.


.. _Run_Scope_of_Execution:

Scope of Execution
~~~~~~~~~~~~~~~~~~

When the :keyword:`run` method of a Component is called, it executes that Component and all others within its scope of
execution.  For a `Mechanism <Mechanism>`, the scope of execution is simply the Mechanism itself.  For a `Process`,
the scope of
execution is all of the Mechanisms specified in its `pathway` attribute.  For a `System`, the scope of execution is
all of the Mechanisms in the Processes specified in the System's `processes <System.processes>` attribute.

.. _Run_Timing:

Timing
~~~~~~

When :keyword:`run` is called by a Component, it calls that Component's :keyword:`execute` method once for each
`input <Run_Inputs>`  (or set of inputs) specified in the call to :keyword:`run`, which constitutes a `TRIAL` of
execution.  For each `TRIAL`, the Component makes repeated `calls to its Scheduler <Scheduler_Execution>`,
executing the Components it specifies in each `TIME_STEP`, until every Component has been executed at least once or
another `termination condition <Scheduler_Termination_Conditions>` is met.  The `Scheduler` can be used in combination
with `Condition` specifications for individual Components to execute different Components at different time scales.

.. _Run_Inputs:

Inputs
~~~~~~

The :keyword:`run` function presents the inputs for each `TRIAL` to the input_states of the relevant Mechanisms in
the `scope of execution <Run_Scope_of_Execution>`. These are specified in the **inputs** argument of a Component's
:keyword:`execute` or :keyword:`run` method.

Inputs are specified in a Python dictionary where the keys are `ORIGIN` Mechanisms, and the values are lists in which
the i-th element represents the input value to the mechanism on trial i. Each input value must be compatible with the
shape of the mechanism's variable. This means that the inputs to an origin mechanism are usually specified by a
list of 2d lists/arrays, though `some shorthand notations are allowed <Input_Specification_Examples>`.

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

.. _Run_Inputs_Fig:

.. figure:: _static/input_spec_variables.svg
   :alt: Example input specifications with variable


.. note::
    Keep in mind that a mechanism's variable is the concatenation of its input states. In other words, a fully specified
    mechanism variable is a 2d list/array in which the i-th element is the variable of the mechanism's i-th input state.
    Because of this `relationship between a mechanism's variable and its input states <Mechanism_Figure>`, it is also
    valid to think about the input specification for a given origin mechanism as a nested list of values for each input
    state on each trial.

    .. _Run_Inputs_Fig_States:

    .. figure:: _static/input_spec_states.svg
       :alt: Example input specifications with input states

The number of inputs specified **must** be the same for all origin mechanisms in the system. In other words, all of the
values in the input dictionary must have the same length.

If num_trials is not in use, the number of inputs provided determines the number of trials in the run. For example, if
five inputs are provided for each origin mechanism, and num_trials is not specified, the system will execute five times.

+----------------------+-------+------+------+------+------+
| Trial #              |1      |2     |3     |4     |5     |
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
items in the list of inputs will be used on the 6th and 7th trials, respectively.

+----------------------+-------+------+------+------+------+------+------+
| Trial #              |1      |2     |3     |4     |5     |6     |7     |
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

* **Case 1: Origin mechanism has only one input state**
+--------------------------+-------+------+------+------+------+
| Trial #                  |1      |2     |3     |4     |5     |
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

Shorthand - drop the outer list on each input because **Mechanism a** only has one input state:

::

        input_dictionary = {a: [[1.0], [2.0], [3.0], [4.0], [5.0]]}

        s.run(inputs=input_dictionary)
..

Shorthand - drop the remaining list on each input because **Mechanism a**'s variable is length 1:

::

        input_dictionary = {a: [1.0, 2.0, 3.0, 4.0, 5.0]}

        s.run(inputs=input_dictionary)
..

* **Case 2: Only one input is provided for the mechanism**

+--------------------------+------------------+
| Trial #                  |1                 |
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
| Trial #                  |1                  |2                  |3                  |4                  |5                  |
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
| Trial #                  |1                  |2                  |
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

COMMENT:
.. _Run_Initial_Values:

Initial Values
~~~~~~~~~~~~~~

Any Mechanism that is the `sender <Projection.Projection.sender>` of a Projection that closes a loop in a Process or
System, and that is not an `ORIGIN` Mechanism, is designated as `INITIALIZE_CYCLE`. An initial value can be assigned
to such Mechanisms, that will be used to initialize them when the Process or System is first run.  These values are
specified in the **initial_values** argument of :keyword:`run`, as a dictionary. The key for each entry must
be a Mechanism designated as `INITIALIZE_CYCLE`, and its value an input for the Mechanism to be used as its initial
value.  The size of the input (length of the outermost level if it is a list, or axis 0 if it is an np.ndarray),
must equal the number of InputStates of the Mechanism, and the size of each value must match (in number and type of
elements) that of the `variable <InputState.InputState.variable>` for the corresponding InputState.
COMMENT

.. _Run_Targets:

Targets
~~~~~~~

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
in the system. Each target value must be compatible with the shape of the `TARGET` mechanism's TARGET `input state
<ComparatorMechanism.input_states>`. This means that for a given key (which is always the last mechanism of the
learning sequence) in the target specification dictionary, the value is usually a list of 1d lists/arrays.

The number of targets specified for each Mechanism must equal the number specified for the **inputs** argument;  as
with **inputs**, if the number of `TRIAL` \\s specified is greater than the number of inputs (and targets), then the
list will be cycled until the number of `TRIAL` \\s specified is completed.

+------------------------------------------+--------------+--------------+
| Trial #                                  |1             |   2          |
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
TARGET `input state <ComparatorMechanism.input_states>`. The function will be executed at the start of the learning
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
        ...     val_1 = NormalDist(mean=3.0).function()
        ...     val_2 = NormalDist(mean=3.0).function()
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
| Trial #                                  |1      |2     |3     |4     |5     |
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
---------------

"""

import datetime
import warnings
from collections import Iterable
from numbers import Number

import numpy as np
import typecheck as tc

from psyneulink.components.component import function_type
from psyneulink.components.shellclasses import Mechanism, Process_Base, System_Base
from psyneulink.globals.context import ContextFlags
from psyneulink.globals.keywords import INPUT_LABELS_DICT, MECHANISM, \
    PROCESS, RUN, SAMPLE, SYSTEM, TARGET
from psyneulink.globals.log import LogCondition
from psyneulink.scheduling.time import TimeScale

__all__ = [
    'EXECUTION_SET_DIM', 'MECHANISM_DIM', 'RunError', 'STATE_DIM', 'run'
]

EXECUTION_SET_DIM = 0
MECHANISM_DIM = 2
STATE_DIM = 3  # Note: only meaningful if mechanisms are homnogenous (i.e., all have the same number of states -- see chart below):

class RunError(Exception):
     def __init__(object, error_value):
         object.error_value = error_value

     def __str__(object):
         return repr(object.error_value)

@tc.typecheck
def run(object,
        inputs,
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
        context=ContextFlags.COMMAND_LINE):
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
    call_after_time_step=None,   \)

    Run a sequence of executions for a `Process` or `System`.

    COMMENT:
        First, validate inputs (and targets, if learning is enabled).  Then, for each `TRIAL`:
            * call call_before_trial if specified;
            * for each time_step in the trial:
                * call call_before_time_step if specified;
                * call ``object.execute`` with inputs, and append result to ``object.results``;
                * call call_after_time_step if specified;
            * call call_after_trial if specified.
        Return ``object.results``.

        The inputs argument must be a list or an np.ndarray array of the appropriate dimensionality:
            * the inner-most dimension must equal the length of object.instance_defaults.variable (i.e., the input to the object);
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
        the input for each `TRIAL` in a sequence (see `Run_Inputs` for detailed description of formatting
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

   Returns
   -------

    <object>.results : List[OutputState.value]
        list of the values, for each `TRIAL`, of the OutputStates for a Mechanism run directly,
        or of the OutputStates of the `TERMINAL` Mechanisms for the Process or System run.
    """
    from psyneulink.globals.context import ContextFlags

    # small version of 'sequence' format in the once case where it was still working (single origin mechanism)
    if isinstance(inputs, (list, np.ndarray)):
        if len(object.origin_mechanisms) == 1:
            inputs = {object.origin_mechanisms[0]: inputs}
        else:
            raise RunError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                           "mechanisms.".format(object.name, len(object.origin_mechanisms)))
    elif not isinstance(inputs, dict):
        if len(object.origin_mechanisms) == 1:
            raise RunError("Inputs to {} must be specified in a list or in a dictionary with the origin mechanism({}) "
                           "as its only key".format(object.name, object.origin_mechanisms[0].name))
        else:
            raise RunError("Inputs to {} must be specified in a dictionary with a key for each of its {} origin "
                           "mechanisms.".format(object.name, len(object.origin_mechanisms)))

    inputs, num_inputs_sets = _adjust_stimulus_dict(object, inputs)

    if num_trials is not None:
        num_trials = num_trials
    else:
        num_trials = num_inputs_sets

    # num_trials = num_trials or num_inputs_sets  # num_trials may be provided by user, otherwise = # of input sets

    if targets is not None:

        if isinstance(targets, dict):
            targets, num_targets = _adjust_target_dict(object, targets)

        elif isinstance(targets, (list, np.ndarray)):
            # small version of former 'sequence' format -- only allowed if there is a single Target mechanism
            if len(object.target_mechanisms) == 1:
                targets = {object.target_mechanisms[0].input_states[SAMPLE].path_afferents[0].sender.owner: targets}
                targets, num_targets = _adjust_target_dict(object, targets)
            else:
                raise RunError("Target values for {} must be specified in a dictionary.".format(object.name))

        elif isinstance(targets, function_type):
            if len(object.target_mechanisms) == 1:
                targets = {object.target_mechanisms[0].input_states[SAMPLE].path_afferents[0].sender.owner: targets}
                targets, num_targets = _adjust_target_dict(object, targets)
            else:
                raise RunError("Target values for {} must be specified in a dictionary.".format(object.name))
        else:
            raise RunError("Target values for {} must be specified in a dictionary.".format(object.name))

        # if num_targets = -1, all targets were specified as functions
        if num_targets != num_inputs_sets and num_targets != -1:
            raise RunError("Number of target values specified ({}) for each learning sequence in {} must equal the "
                           "number of input values specified ({}) for each origin mechanism in {}."
                           .format(num_targets, object.name, num_inputs_sets, object.name))

    object_type = _get_object_type(object)

    object.targets = targets

    # SET LEARNING (if relevant)
    # FIX: THIS NEEDS TO BE DONE FOR EACH PROCESS IF THIS CALL TO run() IS FOR SYSTEM
    #      IMPLEMENT learning_enabled FOR SYSTEM, WHICH FORCES LEARNING OF PROCESSES WHEN SYSTEM EXECUTES?
    #      OR MAKE LEARNING A PARAM THAT IS PASSED IN execute
    # If learning is specified, buffer current state and set to specified state
    if learning is not None:
        try:
            learning_state_buffer = object._learning_enabled
        except AttributeError:
            if object.verbosePref:
                warnings.warn("WARNING: learning not enabled for {}".format(object.name))
        else:
            if learning is True:
                object._learning_enabled = True

            elif learning is False:
                object._learning_enabled = False

    # SET LEARNING_RATE, if specified, for all learningProjections in process or system
    if object.learning_rate is not None:
        from psyneulink.components.projections.modulatory.learningprojection import LearningProjection
        for learning_mech in object.learning_mechanisms.mechanisms:
            for projection in learning_mech.output_state.efferents:
                if isinstance(projection, LearningProjection):
                    projection.function_object.learning_rate = object.learning_rate

    # Class-specific validation:
    if not object.context.flags:
        object.context.initialization_status = ContextFlags.VALIDATING
        object.context.string = RUN + "validating " + object.name

    # INITIALIZATION
    if initialize:
        object.initialize()

    # SET UP TIMING
    if object_type == MECHANISM:
        time_steps = 1
    else:
        time_steps = object.numPhases

    # EXECUTE
    execution_inputs = {}
    execution_targets = {}
    for execution in range(num_trials):

        execution_id = _get_unique_id()

        if call_before_trial:
            call_before_trial()

        for time_step in range(time_steps):

            if call_before_time_step:
                call_before_time_step()

            input_num = execution%num_inputs_sets

            for mech in inputs:
                execution_inputs[mech] = inputs[mech][input_num]
            if object_type == SYSTEM:
                object.inputs = execution_inputs

            # Assign targets:
            if targets is not None:

                if isinstance(targets, function_type):
                    object.target = targets
                else:
                    for mech in targets:
                        if callable(targets[mech]):
                            execution_targets[mech] = targets[mech]
                        else:
                            execution_targets[mech] = targets[mech][input_num]
                    if object_type is SYSTEM:
                        object.target = execution_targets
                        object.current_targets = execution_targets

            if context == ContextFlags.COMMAND_LINE and not object.context.execution_phase == ContextFlags.SIMULATION:
                object.context.execution_phase = ContextFlags.PROCESSING
                object.context.string = RUN + ": EXECUTING " + object_type.upper() + " " + object.name

            result = object.execute(
                input=execution_inputs,
                execution_id=execution_id,
                termination_processing=termination_processing,
                termination_learning=termination_learning,
                context=context
            )

            if call_after_time_step:
                call_after_time_step()

        # object.results.append(result)
        if isinstance(result, Iterable):
            result_copy = result.copy()
        else:
            result_copy = result
        object.results.append(result_copy)

        if call_after_trial:
            call_after_trial()

        from psyneulink.globals.log import _log_trials_and_runs, ContextFlags
        _log_trials_and_runs(composition=object,
                             curr_condition=LogCondition.TRIAL,
                             context=context)

    try:
        object.scheduler_processing.date_last_run_end = datetime.datetime.now()
        object.scheduler_learning.date_last_run_end = datetime.datetime.now()

        for sched in [object.scheduler_processing, object.scheduler_learning]:
            sched.clock._increment_time(TimeScale.RUN)
    except AttributeError:
        # this will fail on processes, which do not have schedulers
        pass

    # Restore learning state
    try:
        learning_state_buffer
    except UnboundLocalError:
        pass
    else:
        object._learning_enabled = learning_state_buffer

    from psyneulink.globals.log import _log_trials_and_runs
    _log_trials_and_runs(composition=object,
                         curr_condition=LogCondition.RUN,
                         context=context)

    return object.results

@tc.typecheck

def _input_matches_variable(input, var):
    # input states are uniform
    if np.shape(np.atleast_2d(input)) == np.shape(var):
        return "homogeneous"
    # input states have different lengths
    elif len(np.shape(var)) == 1 and isinstance(var[0], (list, np.ndarray)):
        for i in range(len(input)):
            if len(input[i]) != len(var[i]):
                return False
        return "heterogeneous"
    return False

def _target_matches_input_state_variable(target, input_state_variable):
    if np.shape(np.atleast_1d(target)) == np.shape(input_state_variable):
        return True
    return False

def _adjust_stimulus_dict(obj, stimuli):

    #  STEP 0:  parse any labels into array entries
    if any(mech.input_labels_dict for mech in obj.origin_mechanisms):
        _parse_input_labels(obj, stimuli)

    # STEP 1: validate that there is a one-to-one mapping of input entries to origin mechanisms

    # Check that all of the mechanisms listed in the inputs dict are ORIGIN mechanisms in the object
    for mech in stimuli.keys():
        if not mech in obj.origin_mechanisms.mechanisms:
            raise RunError("{} in inputs dict for {} is not one of its ORIGIN mechanisms".
                           format(mech.name, obj.name))
    # Check that all of the ORIGIN mechanisms in the obj are represented by entries in the inputs dict
    for mech in obj.origin_mechanisms:
        if not mech in stimuli:
            raise RunError("Entry for ORIGIN Mechanism {} is missing from the inputs dict for {}".
                           format(mech.name, obj.name))

    # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

    # (1) Replace any user provided convenience notations with values that match the following specs:
    # a - all dictionary values are lists containing and input value on each trial (even if only one trial)
    # b - each input value is a 2d array that matches variable
    # example: { Mech1: [Fully_specified_input_for_mech1_on_trial_1, Fully_specified_input_for_mech1_on_trial_2 … ],
    #            Mech2: [Fully_specified_input_for_mech2_on_trial_1, Fully_specified_input_for_mech2_on_trial_2 … ]}
    # (2) Verify that all mechanism values provide the same number of inputs (check length of each dictionary value)

    adjusted_stimuli = {}
    num_input_sets = -1

    for mech, stim_list in stimuli.items():

        check_spec_type = _input_matches_variable(stim_list, mech.instance_defaults.variable)
        # If a mechanism provided a single input, wrap it in one more list in order to represent trials
        if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
            if check_spec_type == "homogeneous":
                # np.atleast_2d will catch any single-input states specified without an outer list
                # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                adjusted_stimuli[mech] = [np.atleast_2d(stim_list)]
            else:
                adjusted_stimuli[mech] = [stim_list]

            # verify that all mechanisms have provided the same number of inputs
            if num_input_sets == -1:
                num_input_sets = 1
            elif num_input_sets != 1:
                raise RunError("Input specification for {} is not valid. The number of inputs (1) provided for {}"
                               "conflicts with at least one other mechanism's input specification.".format(obj.name,
                                                                                                           mech.name))
        else:
            adjusted_stimuli[mech] = []
            for stim in stimuli[mech]:
                check_spec_type = _input_matches_variable(stim, mech.instance_defaults.variable)
                # loop over each input to verify that it matches variable
                if check_spec_type == False:
                    err_msg = "Input stimulus ({}) for {} is incompatible with its variable ({}).".\
                        format(stim, mech.name, mech.instance_defaults.variable)
                    # 8/3/17 CW: I admit the error message implementation here is very hacky; but it's at least not a hack
                    # for "functionality" but rather a hack for user clarity
                    if "KWTA" in str(type(mech)):
                        err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other values)" \
                                            " to represent the outside stimulus for the inhibition input state, and " \
                                            "for systems, put your inputs"
                    raise RunError(err_msg)
                elif check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input states specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[mech].append(np.atleast_2d(stim))
                else:
                    adjusted_stimuli[mech].append(stim)

            # verify that all mechanisms have provided the same number of inputs
            if num_input_sets == -1:
                num_input_sets = len(stimuli[mech])
            elif num_input_sets != len(stimuli[mech]):
                raise RunError("Input specification for {} is not valid. The number of inputs ({}) provided for {}"
                               "conflicts with at least one other mechanism's input specification."
                               .format(obj.name, (stimuli[mech]), mech.name))

    return adjusted_stimuli, num_input_sets

def _adjust_target_dict(component, target_dict):

    #  STEP 0:  parse any labels into array entries
    if any(mech.input_labels_dict for mech in component.target_mechanisms):
        _parse_input_labels(component, target_dict)

    # STEP 1: validate that there is a one-to-one mapping of target entries and target mechanisms
    for target_mechanism in component.target_mechanisms:
        # If any projection to a target does not have a sender in the stimulus dict, raise an exception
        if not any(mech is projection.sender.owner for
                   projection in target_mechanism.input_states[SAMPLE].path_afferents
                   for mech in target_dict.keys()):
                raise RunError("Entry for {} is missing from specification of targets for run of {}".
                               format(target_mechanism.input_states[SAMPLE].path_afferents[0].sender.owner.name,
                                      component.name))

    for mech in target_dict:
        # If any mechanism in the target dict does not have a projection to a target, raise an error
        if not any(target is projection.receiver.owner for
                   projection in mech.output_state.efferents
                   for target in component.target_mechanisms):
            raise RunError("{} does not project to a target Mechanism in {}".format(mech.name, component.name))

        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

        # (1) Replace any user provided convenience notations with values that match the following specs:
        # a - all dictionary values are lists containing a target value on each trial (even if only one trial)
        # b - each input value is at least a 1d array that matches the variable of the TARGET input state

        # (2) Verify that all mechanism values provide the same number of inputs (check length of each dictionary value)

    adjusted_targets = {}
    num_targets = -1
    for mech, target_list in target_dict.items():
        if isinstance(target_list, (float, list, np.ndarray)):
            input_state_variable = mech.output_state.efferents[0].receiver.owner.input_states[TARGET].instance_defaults.variable
            num_targets = -1

            # first check if only one target was provided:
            if np.shape(np.atleast_1d(target_list)) == np.shape(input_state_variable):
                adjusted_targets[mech] = [np.atleast_1d(target_list)]
                if num_targets == -1:
                    num_targets = 1
                elif num_targets != 1:
                    raise RunError("Target specification for {} is not valid. The number of targets (1) provided for {}"
                                   "conflicts with at least one other mechanism's target specification."
                                   .format(component.name, mech.name))

            # iterate over list and check that each candidate target is compatible with corresponding TARGET input state
            elif isinstance(target_list, (list, np.ndarray)):
                adjusted_targets[mech] = []
                for target_value in target_list:
                    if np.shape(np.atleast_1d(target_value)) == np.shape(input_state_variable):
                        adjusted_targets[mech].append(np.atleast_1d(target_value))
                    else:
                        raise RunError("Target specification ({}) for {} is not valid. The shape of {} is not compatible "
                                       "with the TARGET input state of the corresponding ComparatorMechanism ({})"
                                       .format(target_list, mech.name, target_value,
                                               mech.output_state.efferents[0].receiver.owner.name))
                current_num_targets = len(adjusted_targets[mech])
                # verify that all mechanisms have provided the same number of inputs
                if num_targets == -1:
                    num_targets = current_num_targets
                elif num_targets != current_num_targets:
                    raise RunError("Target specification for {} is not valid. The number of targets ({}) provided for {}"
                                   "conflicts with at least one other mechanism's target specification."
                                   .format(component.name, current_num_targets, mech.name))

        elif callable(target_list):
            _validate_target_function(target_list, mech.output_state.efferents[0].receiver.owner, mech)
            adjusted_targets[mech] = target_list
    return adjusted_targets, num_targets


@tc.typecheck
def _parse_input_labels(obj, stimuli:dict):
    from psyneulink.components.states.inputstate import InputState

    # def get_input_for_label(mech, key, input_array=None):
    def get_input_for_label(mech, key, subdicts, input_array=None):
        """check mech.input_labels_dict for key
        If input_array is passed, need to check for subdicts (should be one for each InputState of mech)"""

        # FIX: FOR SOME REASON dict IN TEST BELOW IS TREATED AS AN UNBOUND LOCAL VARIABLE
        # subdicts = isinstance(list(mech.input_labels_dict.keys())[0], dict)

        if input_array is None:
            if subdicts:
                raise RunError("Attempt to reference a label for a stimulus at top level of {} for {},"
                               "which contains subdictionaries for each of its {}s".
                               format(INPUT_LABELS_DICT, mech.name, InputState))
            try:
                return mech.input_labels_dict[key]
            except KeyError:
                raise RunError("No entry \'{}\' found for input to {} in {} for mech.name".
                               format(key, obj.name, INPUT_LABELS_DICT, mech.name))
        else:
            if not subdicts:
                try:
                    return mech.input_labels_dict[key]
                except KeyError:
                    raise RunError("No entry \'{}\' found for input to {} in {} for mech.name".
                                   format(key, obj.name, INPUT_LABELS_DICT, mech.name))
            else:
                # if subdicts, look exhaustively for any instances of the label in keys of all subdicts
                name_value_pairs = []
                for name, dict in mech.input_labels.items():
                    if key in dict:
                        name_value_pairs.append((name,dict[key]))
                if len(name_value_pairs)==1:
                    # if only one found, use its value
                    return name_value_pairs[0][1]
                else:
                    # if more than one is found, now know that "convenience notation" has not been used
                    #     check that number of items in input_array == number of states
                    if len(input_array) != len(mech.input_states):
                        raise RunError("Number of items in input for {} of {} ({}) "
                                       "does not match the number of its {}s ({})".
                                       format(mech.name, obj.name, len(input_array),
                                              InputState, len(mech.input_states)))
                    # use index of item in outer array and key (int or name of state) to determine which subdict to use
                    input_index = input_array.index(key)

                    # try to match input_index against index in name_value_pairs[0];
                    value = [item[1] for item in name_value_pairs if item[0]==input_index]
                    if value:
                        return value[0]
                    else:
                        # otherwise, match against index associated with name of state in name_value_pairs
                        value = [item[1] for item in name_value_pairs if mech.input_states.index(item[0])==input_index]
                        if value:
                            return value[0]
                        else:
                            raise RunError("Unable to find value for label ({}) in {} for {} of {}".
                                           format(key, INPUT_LABELS_DICT, mech.name, obj.name))

    for mech, inputs in stimuli.items():

        subdicts = isinstance(list(mech.input_labels_dict.keys())[0], dict)

        if any(isinstance(input, str) for input in inputs) and not mech.input_labels_dict:
            raise RunError("Labels can not be used to specify the inputs to {} since it does not have an {}".
                           format(mech.name, INPUT_LABELS_DICT))
        for i, stim in enumerate(inputs):
            # "Burrow" down to determine whether there's a number at the "bottom";
            #     if so, leave as is; otherwise, check if its a string and, if so, get value for label
            if isinstance(stim, (list, np.ndarray)): # format of stimuli dict is at least: [[???]...?]
                for j, item in enumerate(stim):
                    if isinstance(item, (Number, list, np.ndarray)): # format of stimuli dict is [[int or []...?]]
                        continue # leave input item as is
                    elif isinstance(item, str): # format of stimuli dict is [[label]...]
                        # inputs[i][j] = get_input_for_label(mech, item, stim)
                        inputs[i][j] = get_input_for_label(mech, item, subdicts, stim)
                    else:
                        raise RunError("Unrecognized specification ({}) in stimulus {} of entry "
                                       "for {} in inputs dictionary specified for {}".
                                       format(item, i, mech.name, obj.name))
            elif isinstance(stim, str):
                # Don't pass input_array as no need to check for subdicts
                # inputs[i] = get_input_for_label(mech, stim)
                inputs[i] = get_input_for_label(mech, stim, subdicts)
            else:
                raise RunError("Unrecognized specification ({}) for stimulus {} in entry "
                               "for {} of inputs dictionary specified for {}".
                               format(stim, i, mech.name, obj.name))

def _validate_target_function(target_function, target_mechanism, sample_mechanism):

    generated_targets = np.atleast_1d(target_function())
    expected_shape = target_mechanism.input_states[TARGET].instance_defaults.variable
    if np.shape(generated_targets) != np.shape(expected_shape):
            raise RunError("Target values generated by target function ({}) are not compatible with TARGET input state "
                           "of {} ({}). See {} entry in target specification dictionary. "
                           .format(generated_targets, target_mechanism.name, expected_shape, sample_mechanism.name))

def _get_object_type(object):
    if isinstance(object, Mechanism):
        return MECHANISM
    elif isinstance(object, Process_Base):
        return PROCESS
    elif isinstance(object, System_Base):
        return SYSTEM
    else:
        raise RunError("{} type not supported by Run module".format(object.__class__.__name__))


import uuid
def _get_unique_id():
    return uuid.uuid4()
