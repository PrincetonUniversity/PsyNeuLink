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

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a',
                                  default_variable=[[0.0, 0.0]])
        b = pnl.TransferMechanism(name='b',
                                  default_variable=[[0.0], [0.0]])
        c = pnl.TransferMechanism(name='c')

        p1 = pnl.Process(pathway=[a, c],
                         name='p1')
        p2 = pnl.Process(pathway=[b, c],
                         name='p2')

        s = pnl.System(processes=[p1, p2])

        input_dictionary = {a: [[[1.0, 1.0]], [[1.0, 1.0]]],
                            b: [[[2.0], [3.0]], [[2.0], [3.0]]]}

        s.run(inputs=input_dictionary)

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

        import psyneulink as pnl

        a = pnl.TransferMechanism(name='a')
        b = pnl.TransferMechanism(name='b')

        p1 = pnl.Process(pathway=[a, b])

        s = pnl.System(processes=[p1])

        input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        s.run(inputs=input_dictionary)

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
are specified in the **targets** argument of the :keyword:`execute` or :keyword:`run` method, which can be in
any of three formats.  The two formats used for **inputs** (`Sequence <Run_Inputs_Sequence_Format>` and
`Mechanism <Run_Inputs_Mechanism_Format>` format) can also be used for targets.  However, the format of the lists or
ndarrays is simpler, since each `TARGET` Mechanism is assigned only a single target value, so there is never the need
for the extra level of nesting (or dimension of ndarray) used for InputStates in the specification of **inputs**.
Details concerning the use of the `Sequence <Run_Targets_Sequence_Format>`  and
`Mechanism <Run_Targets_Mechanism_Format>` formats for targets is described below. Targets can also be specified
as a `function <Run_Targets_Function_Format>` (for example, to allow the target to depend on the outcome of processing).

If either the Sequence or Mechanism format is used, then the number of targets specified for each Mechanism must equal
the number specified for the **inputs** argument;  as with **inputs**, if the number of `TRIAL` \\s specified is greater
than the number of inputs (and targets), then the list will be cycled until the number of `TRIAL` \\s specified is
completed.  If a function is used for the **targets**, then it will be used to generate a target for each `TRIAL`.

The number of targets specified in the Sequence or Mechanism formats for each `TRIAL`, or generated using
the function format, must equal the number of `TARGET` Mechanisms for the Process or System being run (see Process
`target_mechanisms <Process.Process.target_mechanisms>` or
System `targetMechanism <System.target_mechanisms>` respectively), and the value of each target must
match (in number and type of elements) that  of the `target <ComparatorMechanism.ComparatorMechanism.target>`
attribute of the `TARGET` Mechanism for which it is intended.  Furthermore, if a range is specified for the output of
the `TERMINAL` Mechanism with which the target is compared (that is, the Mechanism that provides the
`ComparatorMechanism's <ComparatorMechanism>` `sample <ComparatorMechanism.ComparatorMechanism.sample>`
value, then the target must be within that range (for example, if the `TERMINAL` Mechanism is a
`TransferMechanism` that uses a `Logistic` function, its `range <TransferMechanism.TransferMechanism.range>` is
[0,1], so the target must be within that range).

.. _Run_Targets_Sequence_Format:

Sequence Format
^^^^^^^^^^^^^^^

*(List[values] or ndarray):* -- there are at most three levels of nesting (or dimensions) required for
targets:  one for `TRIAL` \\s, one for Mechanisms, and one for the elements of each input.  For a System
with more than one `TARGET` Mechanism, the targets must be specified in the same order as they appear in the System's
`target_mechanisms <System.target_mechanisms>` attribute.  This should be the same order in which
they are declared, and can be displayed using the System's `show <System.show>` method). All
other requirements are the same as the `Sequence format <Run_Inputs_Sequence_Format>` for **inputs**.

.. _Run_Targets_Mechanism_Format:

Mechanism Format
^^^^^^^^^^^^^^^^
*(Dict[Mechanism, List[values] or ndarray]):* -- there must be one entry in the dictionary for each of the `TARGET`
Mechanisms in the Process or System being run, though the entries can be specified in any order (making this format
easier to use. The value of each entry is a list or ndarray of the target values for that Mechanism, one for each
`TRIAL`.  There are at most two levels of nesting (or dimensions) required for each entry: one for the `TRIAL`,
and the other for the elements of each input.  In all other respects, the format is the same as the
`Mechanism format <Run_Inputs_Mechanism_Format>` for **inputs**.

.. _Run_Targets_Function_Format:

Function Format
^^^^^^^^^^^^^^^

*[Function]:* -- the function must return an array with a number of items equal to the number of `TARGET` Mechanisms
for the Process or System being run, each of which must match (in number and type of elements) the
`target <ComparatorMechanism.ComparatorMechanism.target>` attribute of the `TARGET` Mechanism for which it is intended.
This format allows targets to be constructed programmatically, in response to computations made during the run.

COMMENT:
    ADD EXAMPLE HERE
COMMENT

.. _Run_Class_Reference:

Class Reference
---------------

"""

import warnings

from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.components.component import ExecutionStatus, function_type
from psyneulink.components.process import ProcessInputState
from psyneulink.components.shellclasses import Mechanism, Process_Base, System_Base
from psyneulink.globals.keywords import EVC_SIMULATION, MECHANISM, PROCESS, PROCESSES_DIM, RUN, SAMPLE, SYSTEM, TARGET
from psyneulink.globals.utilities import append_type_to_name, iscompatible
from psyneulink.scheduling.timescale import CentralClock, TimeScale

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
        reset_clock:bool=True,
        initialize:bool=False,
        initial_values:tc.optional(tc.any(list, dict, np.ndarray))=None,
        targets:tc.optional(tc.any(list, dict, np.ndarray, function_type))=None,
        learning:tc.optional(bool)=None,
        call_before_trial:tc.optional(callable)=None,
        call_after_trial:tc.optional(callable)=None,
        call_before_time_step:tc.optional(callable)=None,
        call_after_time_step:tc.optional(callable)=None,
        clock=CentralClock,
        time_scale:tc.optional(tc.enum(TimeScale.TRIAL, TimeScale.TIME_STEP))=None,
        termination_processing=None,
        termination_learning=None,
        context=None):
    """run(                      \
    inputs,                      \
    num_trials=None,             \
    reset_clock=True,            \
    initialize=False,            \
    intial_values=None,          \
    targets=None,                \
    learning=None,               \
    call_before_trial=None,      \
    call_after_trial=None,       \
    call_before_time_step=None,  \
    call_after_time_step=None,   \
    clock=CentralClock,          \
    time_scale=None)

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

    reset_clock : bool : default True
        if `True`, resets `CentralClock` to 0 before a sequence of `TRIAL` \\s.

    initialize : bool default False
        calls the `initialize <System.initialize>` method of the System prior to the first `TRIAL`.

    initial_values : Dict[Mechanism:List[input]], List[input] or np.ndarray(input) : default None
        the initial values assigned to Mechanisms designated as `INITIALIZE_CYCLE`.

    targets : List[input] or np.ndarray(input) : default None
        the target values assigned to the `ComparatorMechanism` for each `TRIAL` (used for learning).
        The length must be equal to **inputs**.

    learning : bool :  default None
        enables or disables learning during execution for a `Process <Process_Execution_Learning>` or
        `System <System_Execution_Learning>`.  If it is not specified, the current state of learning is left intact.
        If it is `True`, learning is forced on; if it is `False`, learning is forced off.

    call_before_trial : Function : default= `None`
        called before each `TRIAL` in the sequence is run.

    call_after_trial : Function : default= `None`
        called after each `TRIAL` in the sequence is run.

    call_before_time_step : Function : default= ``None`
        called before each `TIME_STEP` is executed.

    call_after_time_step : Function : default= `None`
        called after each `TIME_STEP` is executed.

   Returns
   -------

    <object>.results : List[OutputState.value]
        list of the values, for each `TRIAL`, of the OutputStates for a Mechanism run directly,
        or of the OutputStates of the `TERMINAL` Mechanisms for the Process or System run.
    """

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

    num_trials = num_trials or num_inputs_sets  # num_trials may be provided by user, otherwise = # of input sets

    if targets:
        if isinstance(targets, dict):
            targets = _adjust_target_dict(object, targets)
        elif not isinstance(targets, function_type):
            raise RunError("Targets for {} must be a dictionary or function.".format(object.name))
        _validate_targets(object, targets, num_inputs_sets, context=context)

    object_type = _get_object_type(object)

    object.targets = targets

    time_scale = time_scale or TimeScale.TRIAL

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
    context = context or RUN + "validating " + object.name


    # INITIALIZATION
    if reset_clock:
        clock.trial = 0
        clock.time_step = 0
    if initialize:
        object.initialize()

    # SET UP TIMING
    if object_type == MECHANISM:
        time_steps = 1
    else:
        time_steps = object.numPhases

    # EXECUTE
    execution_inputs = {}
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

                # IMPLEMENTATION NOTE:  USE input_num since # of inputs must equal # targets,
                #                       whereas targets can be assigned a function (so can't be used to generated #)
                elif object_type == PROCESS:
                    # object.target = targets[input_num][time_step]
                    object.target = targets[input_num][time_step]

                elif object_type == SYSTEM:
                    object.current_targets = targets[input_num]
            # MODIFIED 3/16/17 END
            if RUN in context and not EVC_SIMULATION in context:
                context = RUN + ": EXECUTING " + object_type.upper() + " " + object.name
                object.execution_status = ExecutionStatus.EXECUTING
            result = object.execute(input=execution_inputs,
                                    execution_id=execution_id,
                                    clock=clock,
                                    time_scale=time_scale,
                                    termination_processing=termination_processing,
                                    termination_learning=termination_learning,
                                    context=context)

            if call_after_time_step:
                call_after_time_step()

            clock.time_step += 1

        # object.results.append(result)
        if isinstance(result, Iterable):
            result_copy = result.copy()
        else:
            result_copy = result
        object.results.append(result_copy)

        if call_after_trial:
            call_after_trial()

        clock.trial += 1

    # Restore learning state
    try:
        learning_state_buffer
    except UnboundLocalError:
        pass
    else:
        object._learning_enabled = learning_state_buffer

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

def _adjust_stimulus_dict(obj, stimuli):

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

def _adjust_target_dict(object, stimuli):
    object_type = _get_object_type(object)

    # FIX: RE-WRITE USING NEXT AND StopIteration EXCEPTION ON FAIL TO FIND (THIS GIVES SPECIFICS)
    # FIX: TRY USING compare METHOD OF DICT OR LIST?
    # Check that every target in the process or system receives a projection from a mechanism named in the dict
    for target in object.target_mechanisms:
        # If any projection to a target does not have a sender in the stimulus dict, raise an exception
        if not any(mech is projection.sender.owner for
                   projection in target.input_states[SAMPLE].path_afferents
                   for mech in stimuli.keys()):
                raise RunError("Entry for {} is missing from specification of targets for run of {}".
                               format(target.input_states[SAMPLE].
                                      afferents[0].sender.owner.name,
                                      object.name))

    # FIX: COULD JUST IGNORE THOSE, OR WARN ABOUT THEM IF VERBOSE?

    # Check that each target referenced in the dict (key)
    #     is the name of a mechanism that projects to a target (comparator) in the system
    terminal_to_target_mapping = {}
    for mech in stimuli.keys():
        # If any mechanism in the stimulus dict does not have a projection to the target, raise an exception
        if not any(target is projection.receiver.owner for
                   projection in mech.output_state.efferents
                   for target in object.target_mechanisms):
            raise RunError("{} is not a target Mechanism in {}".format(mech.name, object.name))
        # Get target mech (comparator) for each entry in stimuli dict:
        terminal_to_target_mapping[mech] = mech.output_state.efferents[0]

    # Insure that target lists in dict are accessed in the same order as the
    #   targets in the system's target_mechanisms list, by reassigning targets to an OrderedDict:
    from collections import OrderedDict
    ordered_targets = OrderedDict()
    for target in object.target_mechanisms:
        # Get the process to which the TARGET mechanism belongs:
        try:
            process = next(projection.sender.owner for
                           projection in target.input_states[TARGET].path_afferents if
                           isinstance(projection.sender, ProcessInputState))
        except StopIteration:
            raise RunError("PROGRAM ERROR: No process found for TARGET Mechanism ({}) "
                           "supposed to be in target_mechanisms for {}".
                           format(target.name, object.name))
        # Get stimuli specified for TERMINAL mechanism of process associated with TARGET mechanism
        terminal_mech = process.terminal_mechanisms[0]
        try:
            ordered_targets[terminal_mech] = stimuli[terminal_mech]
        except KeyError:
            raise RunError("{} (of {} process) not found target specification for run of {}".
                           format(terminal_mech, object.name))
    stimuli = ordered_targets

    # Convert all items to 2D arrays:
    # - to match standard format of mech.instance_defaults.variable
    # - to deal with case in which the lists have only one stimulus, one more more has length > 1,
    #     and those are specified as lists or 1D arrays (which would be misinterpreted as > 1 stimulus)

    stim_lists = list(stimuli.values())
    num_input_sets = len(stim_lists[EXECUTION_SET_DIM])

    # Check that all lists have the same number of stimuli
    if not all(len(np.array(stim_list)) == num_input_sets for stim_list in stim_lists):
        raise RunError("The length of all the stimulus lists must be the same")

    stim_list = []

    for i in range(num_input_sets):
        stims_in_execution = []
        for mech in stimuli:
            stims_in_execution.append(stimuli[mech][i])
        stim_list.append(stims_in_execution)

    try:
        stim_list = np.array(stim_list)
    except ValueError:
        for exec in range(len(stim_list)):
            for phase in range(len(stim_list[exec])):
                for mech in range(len(stim_list[exec][phase])):
                    stim_list[exec][phase][mech] = stim_list[exec][phase][mech].tolist()
        stim_list = np.array(stim_list)

    return np.array(stim_list)



def _validate_targets(object, targets, num_input_sets, context=None):
    """
    num_targets = number of target stimuli per execution
    num_targets_sets = number sets of targets (one for each execution) in targets;  must match num_input_sets
    """

    object_type = _get_object_type(object)
    num_target_sets = None

    if isinstance(targets, function_type):
        # Check that function returns a number of items equal to the number of target mechanisms
        generated_targets = targets()
        num_targets = len(generated_targets)
        num_target_mechs = len(object.target_mechanisms)
        if num_targets != num_target_mechs:
            raise RunError("function for target argument of run returns {} items "
                           "but {} has {} targets".
                           format(num_targets, object.name, num_target_mechs))

        # Check that each target generated is compatible with the targetMechanism for which it is intended
        for target, targetMechanism in zip(generated_targets, object.target_mechanisms):
            target_len = np.size(target)
            if target_len != np.size(targetMechanism.input_states[TARGET].instance_defaults.variable):
                if num_target_sets > 1:
                    plural = 's'
                else:
                    plural = ''
                raise RunError("Length ({}) of target{} specified for run of {}"
                                   " does not match expected target length of {}".
                                   format(target_len, plural, append_type_to_name(object),
                                          np.size(object.target_mechanism.target)))
        return

    if object_type is PROCESS:

        # If learning is enabled, validate target
        if object._learning_enabled:
            target_array = np.atleast_2d(targets)
            target_len = np.size(target_array[0])
            num_target_sets = np.size(target_array, 0)

            if target_len != np.size(object.target_mechanism.input_states[TARGET].instance_defaults.variable):
                if num_target_sets > 1:
                    plural = 's'
                else:
                    plural = ''
                raise RunError("Length ({}) of target{} specified for run of {}"
                                   " does not match expected target length of {}".
                                   format(target_len, plural, append_type_to_name(object),
                                          np.size(object.target_mechanism.target)))

            if any(np.size(target) != target_len for target in target_array):
                raise RunError("Not all of the targets specified for {} are of the same length".
                                   format(append_type_to_name(object)))

            if num_target_sets != num_input_sets:
                raise RunError("Number of targets ({}) does not match number of inputs ({}) specified in run of {}".
                                   format(num_target_sets, num_input_sets, append_type_to_name(object)))

    elif object_type is SYSTEM:

        # FIX: VALIDATE THE LEARNING IS ENABLED
        # FIX: CONSOLIDATE WITH TESTS FOR PROCESS ABOVE?

        # If the system has any process with learning enabled
        if any(process._learning_enabled for process in object.processes):

            HOMOGENOUS_TARGETS = 1
            HETEROGENOUS_TARGETS = 0

            if targets.dtype in {np.dtype('int'), np.dtype('float')}:
                process_structure = HOMOGENOUS_TARGETS
            elif targets.dtype is np.dtype('O'):
                process_structure = HETEROGENOUS_TARGETS
            else:
                raise RunError("Unknown data type for inputs in {}".format(object.name))

            # Processed targets for a system should be 1 dim less than inputs (since don't include phase)
            # If inputs to processes of system are heterogenous, inputs.ndim should be 2:
            # If inputs to processes of system are homogeneous, inputs.ndim should be 3:
            expected_dim = 2 + process_structure
            if targets.ndim != expected_dim:
                raise RunError("targets arg in call to {}.run() must be a {}D np.array or comparable list".
                                  format(object.name, expected_dim))

            # FIX: PROCESS_DIM IS NOT THE RIGHT VALUE HERE, AGAIN BECAUSE IT IS A 3D NOT A 4D ARRAY (NO PHASES)
            # # MODIFIED 2/16/17 OLD:
            # num_target_sets = np.size(targets,PROCESSES_DIM-1)
            # MODIFIED 2/16/17 NEW:
            num_target_sets = targets.shape[0]
            num_targets_per_set = np.size(targets,PROCESSES_DIM-1)
            # MODIFIED 2/16/17 END
            # Check that number of target values in each execution equals the number of target mechanisms in the system
            if num_targets_per_set != len(object.target_mechanisms):
                raise RunError("The number of target values for each execution ({}) in the call to {}.run() "
                                  "does not match the number of Processes in the System ({})".
                                  format(
                                         # np.size(targets,PROCESSES_DIM),
                                         num_targets_per_set,
                                         object.name,
                                         len(object.origin_mechanisms)))

            # MODIFIED 12/23/16 NEW:
            # Validate that each target is compatible with its corresponding targetMechanism
            # FIX: CONSOLIDATE WITH TESTS FOR PROCESS AND FOR function_type ABOVE
            # FIX: MAKE SURE THAT ITEMS IN targets ARE ALIGNED WITH CORRESPONDING object.target_mechanisms
            target_array = np.atleast_2d(targets)

            for target, targetMechanism in zip(targets, object.target_mechanisms):
                target_len = np.size(target)
                if target_len != np.size(targetMechanism.input_states[TARGET].instance_defaults.variable):
                    if num_targets_per_set > 1:
                        plural = 's'
                    else:
                        plural = ''
                    raise RunError("Length ({}) of target{} specified for run of {}"
                                       " does not match expected target length of {}".
                                       format(target_len, plural, append_type_to_name(object),
                                              np.size(targetMechanism.input_states[TARGET].instance_defaults.variable)))

                if any(np.size(target) != target_len for target in target_array):
                    raise RunError("Not all of the targets specified for {} are of the same length".
                                       format(append_type_to_name(object)))

                if num_target_sets != num_input_sets:
                    raise RunError("Number of targets ({}) does not match number of inputs ({}) specified in run of {}".
                                       format(num_target_sets, num_input_sets, append_type_to_name(object)))
            # MODIFIED 12/23/16 END

    else:
        raise RunError("PROGRAM ERRROR: {} type not currently supported by _validate_targets in Run module for ".
                       format(object.__class__.__name__))

    return num_target_sets

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
