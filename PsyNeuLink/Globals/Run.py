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
all of the Mechanisms in the Processes specified in the System's `processes <System_Base.processes>` attribute.

.. _Run_Timing:

Timing
~~~~~~

When :keyword:`run` is called by a Component, it calls that Component's :keyword:`execute` method once for each
`input <Run_Inputs>`  (or set of inputs) specified in the call to :keyword:`run`, which constitutes a `TRIAL` of
execution.  For each `TRIAL`, the Component makes repeated `calls to its Scheduler <Scheduler_Execution>`,
executing the Components it specifies in each `TIME_STEP`, until every Component has been executed at least once or
another `termination condition <Scheduler_Termination_Conditions>` is met.  The `Scheduler` can be used in combination
with `Condition` specifications for individual Components to execute different Components at different time scales.

.. note::
   The **time_scale** argument of :keyword:`run`, described below, is currently not fully implemented,
   but will be in a subsequent version.

.. _Run_Time_Scale::

The :keyword:`run` function also has a **time_scale** argument, that can be used to globally specify the time_scale
parameter for those Components that make use of it.  Any value of `TimeScale` can be specified; how it is interpreted
is determined by the Component. For example, some Mechanisms that perform integration (such as the `DDM`) offer the
option of using an analytic solution (that computes the integral in a single `TIME_STEP`), or a numerical method
(that carries out one step of integration per `TIME_STEP`).  If `TimeScale.TIME_STEP` is assigned as the value of the
**time_scale** argument in a call to :keyword:`run`, those Mechanisms will use their numerical integration method,
whereas if `TimeScale.TRIAL` is assigned they will use their analytic solution.  Similarly, for `TimeScale.TIME_STEP`,
`TransferMechanisms <TransferMechanism>` integrate their input prior to applying the transfer function (sometimes
referred to as "time averaging" or "cascade mode"), whereas for `TimeScale.TRIAL` they apply it to their current input
(i.e., execute their transfer an "instantaneously").

.. _Run_Inputs:

Inputs
~~~~~~

The :keyword:`run` function presents the inputs for each `TRIAL` to the input_states of the relevant Mechanisms in
the `scope of execution <Run_Scope_of_Execution>`. These are specified in the **inputs** argument of a Component's
:keyword:`execute` or :keyword:`run` method. For a Mechanism, they comprise the `variable <InputState.variable>` for
each of the Mechanism's `InputStates <InputState>`.  For a Process or System, they comprise the
`variable <InputState.variable>` for the InputState(s) of the `ORIGIN` Mechanism(s).  Inputs can be specified in one
of two ways: `Sequence format <Run_Inputs_Sequence_Format>` and `Mechanism format <Run_Dict_format>`.
Sequence format is more complex, but does not require the specification of Mechanisms by name, and thus may better
suited for automated means of generating inputs.  Mechanism format requires that inputs be assigned to Mechanisms by
name, but is easier to use (as the order in which the inputs are specified does not matter, so long as they are paired
with their Mechanisms).  Both formats require that inputs be specified as nested lists or ndarrays, that define the
number of trials, mechanisms, InputStates and elements for each input.  These factors determine the levels of nesting
required for a list, or the dimensionality (number of axes) for an ndarray.  They are described below, followed by a
description of the two formats.

.. note::
   The descriptions below are for completeness, and are intended as a technical reference;  or most uses of
   of :keyword:`run` methods, it is only necessary to understand the relatively simple Mechanism formate
   described `below <Run_Inputs_Mechanism_Format>`.

.. _Run_Nesting_Factors:

* **Number of TRIALS**.  If the **inputs** argument contains the input for more than one `TRIAL`, then the outermost
  level of the list, or axis 0 of the ndarray, is used for the `TRIAL` \\s, each item of which contains the
  set inputs for a given `TRIAL`.  Otherwise, it is used for the next relevant factor in the list below.  If the
  number of inputs specified is less than the number of `TRIAL` \\s, then the input list is cycled until the full
  number of `TRIAL` \\s is completed.
..
* **Number of Mechanisms.** If :keyword:`run` is used for a System, and it has more than one `ORIGIN` Mechanism, then
  the next level of nesting of a list, or next higher axis of an ndarray, is used for the `ORIGIN` Mechanisms, with
  each item containing the inputs for a given `ORIGIN` Mechanism within a `TRIAL`.  This factor is not relevant
  when :keyword:`run` is used for a single Mechanism, a Process (which only ever has one `ORIGIN` Mechanism),
  or a System that has only one `ORIGIN` Mechanism.  It is also not relevant for the
  `Mechanism format <Run_Inputs_Mechanism_Format>`, since that separates the inputs for each Mechanism into individual
  entries of a dictionary.
..
* **Number of InputStates.** In general, Mechanisms have only a single (`primary <Mechanism_InputStates>`) InputState;
  however, some types of Mechanisms can have more than one.  If any `ORIGIN` Mechanism in a Pocess or System has more
  than one InputState, then the next level of nesting of a list, or next higher axis of an ndarray, is used for the
  set of InputStates for each Mechanism.
..
* **Number of elements for the value of an InputState.** The input for an InputState can be a single element (e.g.,
  a scalar) or have multiple elements (e.g., a vector).  By convention, even if the input to an InputState is a single
  element, it should nevertheless always be specified as a list or a 1d np.array (it is internally converted to the
  latter).  PsyNeuLink can usually parse single-element inputs specified as a stand-alone value (e.g., as a number
  not in a list or ndarray).  Nevertheless, it is best to embed such inputs in a single-element list or a 1d array,
  both for clarity and to insure consistent treatment of nested lists and ndarrays.  If this convention is followed,
  then the number of elements for a given input should not affect nesting of lists or dimensionality (number of axes)
  of ndarrays of an **inputs** argument.

With these factors in mind, the **inputs** argument can be specified in the simplest form possible (least
number of nestings for a list, or lowest dimension of an ndarray).  It can be specified using one of two formats:

.. _Run_Inputs_Sequence_Format:

Sequence Format
^^^^^^^^^^^^^^^

.. note::
   This format is included for backward compatability, but may not be supported in the future.  It is **strongly**
   recommended that the `Mechanism format <Run_Inputs_Mechanism_Format>` be used instead.  That said, please feel
   free to convey any strong preference for this format to the development team, so that an informed decision can
   be made about future inclusion of this format.

*(List[values] or ndarray)* -- this uses a nested list or ndarray to fully specify the input for
each `TRIAL` in a sequence.  It is more complex than the `Mechanism format <Run_Inputs_Mechanism_Format>`,
and for Systems requires that the inputs for each Mechanism be specified in the same order in which those Mechanisms
appear in the System's `origin_mechanisms <System_Base.origin_mechanisms>` attribute.  This is generally the
same order in which they are declared, and can be displayed using the System's `show <System_Base.show>`
method). Although this format is more complex, it may be better suited to automated input generation, since it does
not require that Mechanisms be referenced explicitly (though it is allowed). The following provides a description of
the Sequence format for all of the combinations of factors describe `above <Run_Inputs_Mechanism_Format>`.
The `figure <Run_Sequence_Format_Fig>` below shows examples.

    *Lists:* if there is more than one `TRIAL`, then the outermost level of the list is used for the sequence of
    `TRIALS`.  If there is only one `ORIGIN` Mechanism and it has only one InputState (the most common
    case), then a single sublist is used for the input of each `TRIAL`.  If the `ORIGIN` Mechanism has more
    than one InputState, then the entry for each `TRIAL` is a sublist of the InputStates, each entry of which is a
    sublist containing the input for that InputState.  If there is more than one Mechanism, but none have more than
    one InputState, then a sublist is used for each Mechanism in each `TRIAL`, within which a sublist is used for the
    input for that Mechanism.  If there is more than one Mechanism, and any have more than one InputState,
    then a sublist is used for each Mechanism for each `TRIAL`, within which a sublist is used for each
    InputState of the corresponding Mechanism, and inside that a sublist is used for the input for each InputState.

    *ndarray:*  axis 0 is used for the first factor (`TRIAL`, Mechanism, InputState or input) for which there is only
    one item, axis 1 is used for the next factor for which there is only one item, and so forth.  For example, if there
    is more than one `TRIAL`, only one `ORIGIN` Mechanism, and that has only one InputState (the most common case),
    then axis 0 is used for `TRIAL`, and axis 1 for inputs per `TRIAL`.  At the other extreme, if there are multiple
    `TRIALS`, more than one `ORIGIN` Mechanism, and more than one InputState for one or more of the `ORIGIN` Mechanisms,
    then axis 0 is used for `TRIAL` `s, axis 1 for Mechanisms within `TRIAL`, axis 2 for InputStates of each Mechanism,
    and axis 3 for the input to each InputState of a Mechanism.  Note that if *any* Mechanism being run (directly, or as
    one of the `ORIGIN` Mechanisms of a Process or System) has more than one InputState, then an axis must be
    committed to InputStates, and the input to every InputState of every Mechanism must be specified in that axis
    (i.e., even for those Mechanisms that have a single InputState).

    .. _Run_Sequence_Format_Fig:

    .. figure:: _static/Sequence_format_input_specs_fig.*
       :alt: Example input specifications in Sequence format
       :scale: 75 %
       :align: center

       Example input specifications in Sequence format

.. _Run_Inputs_Mechanism_Format:

Mechanism Format
^^^^^^^^^^^^^^^^

*(Dict[Mechanism, List[values] or ndarray])* -- this provides a simpler format for specifying :keyword:`inputs` than
the Sequence format, and does not require that the inputs for each Mechanism be specified in a particular order.
However, it requires that each Mechanism that receives inputs be referenced explicitly (instead of by order),
which may be less suitable for automated forms of input generation.  It uses a dictionary, each entry of which is the
sequence of inputs for an `ORIGIN` Mechanism;  there must be one such entry for each of the `ORIGIN` Mechanisms of the
Process or System being run.  The key for each entry is the `ORIGIN` Mechanism, and the value contains either a list
or ndarray specifying the sequence of inputs for that Mechanism, one for each `TRIAL` to be run.  If a list is used,
and the Mechanism has more than one InputState, then a sublist is used in each item of the list to specify the inputs
for each of the Mechanism's InputStates for that `TRIAL`.  If an ndarray is used, axis 0 is used for the sequence of
`TRIAL` \\s. If the Mechanism has a single InputState, then axis 1 is used for the input for each `TRIAL.  If the
Mechanism has multiple InputStates, then axis 1 is used for the InputStates, and axis 2 is used for the input to each
InputState for each `TRIAL`.

    .. figure:: _static/Mechanism_format_input_specs_fig.*
       :alt: Mechanism format input specification
       :align: center

       Mechanism format input specification

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
`target_mechanisms <Process.Process_Base.target_mechanisms>` or
System `targetMechanism <System_Base.target_mechanisms>` respectively), and the value of each target must
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
`target_mechanisms <System_Base.target_mechanisms>` attribute.  This should be the same order in which
they are declared, and can be displayed using the System's `show <System_Base.show>` method). All
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

from PsyNeuLink.Components.Component import ExecutionStatus, function_type
from PsyNeuLink.Components.Process import ProcessInputState
from PsyNeuLink.Components.ShellClasses import Mechanism, Process, System
from PsyNeuLink.Globals.Keywords import EVC_SIMULATION, MECHANISM, PROCESS, PROCESSES_DIM, RUN, SAMPLE, SYSTEM, TARGET
from PsyNeuLink.Globals.Utilities import append_type_to_name, iscompatible
from PsyNeuLink.Scheduling.TimeScale import CentralClock, TimeScale

HOMOGENOUS = 1
HETEROGENOUS = 0

EXECUTION_SET_DIM = 0
PHASE_DIM = 1
MECHANISM_DIM = 2
STATE_DIM = 3  # Note: only meaningful if mechanisms are homnogenous (i.e., all have the same number of states -- see chart below):

# Axis 0         |---------------------------------------------------exec set----------------------------------------------------|
# Axis 1           |-----------------------phase-----------------------|   |-----------------------phase-----------------------|
# Axis 2             |--------mech---------|   |--------mech---------|       |--------mech---------|   |--------mech---------|
# Axis 3               |-state--|-state--|       |-state--|-state--|           |-state--|-state--|       |-state--|-state--|
#
# HOMOGENOUS: (ndim = 5)
# a = np.array([ [ [ [ [ 0, 0 ] , [ 1, 1 ] ] , [ [ 2, 2 ] , [ 3, 3 ] ] ] , [ [ [ 4, 4 ] , [ 5, 5 ] ] , [ [ 6, 6 ] , [ 7, 7 ] ] ] ] ,
#                [ [ [ [ 8, 8 ] , [ 9, 9 ] ] , [ [10, 10] , [11, 11] ] ] , [ [ [12, 12] , [13, 13] ] , [ [14, 14] , [15, 15] ] ] ] ])
#
# HETEROGENOUS:
# State sizes (ndim = 4)
# b = np.array([ [ [ [ [    0 ] , [ 1, 1 ] ] , [ [    2 ] , [ 3, 3 ] ] ] , [ [ [    4 ] , [ 5, 5 ] ] , [ [    6 ] , [ 7, 7 ] ] ] ] ,
#                [ [ [ [    8 ] , [ 9, 9 ] ] , [ [    10] , [    11] ] ] , [ [ [    12] , [    13] ] , [ [    14] , [    15] ] ] ] ])
#
# States per mechanism  (ndim = 3)
# c = np.array([ [ [ [ [ 0, 0 ]            ] , [ [ 2, 2 ] , [ 3, 3 ] ] ] , [ [            [ 5, 5 ] ] , [ [ 6, 6 ] , [ 7, 7 ] ] ] ] ,
#                [ [ [ [ 8, 8 ]            ] , [ [10, 10] , [11, 11] ] ] , [ [            [13, 13] ] , [ [14, 14] , [15, 15] ] ] ] ])
#
# Both (ndim = 3)
# d = np.array([ [ [ [ [    0 ]            ] , [ [ 2, 2 ] , [    3 ] ] ] , [ [ [    4 ]            ] , [ [ 6, 6 ] , [    7 ] ] ] ] ,
#                [ [ [ [    8 ]            ] , [ [10, 10] , [    11] ] ] , [ [ [    12]            ] , [ [14, 14] , [    15] ] ] ] ])

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
        calls the `initialize <System_Base.initialize>` method of the System prior to the first `TRIAL`.

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

    time_scale : TimeScale :  default TimeScale.TRIAL
        specifies time scale for Components that implement different forms of execution for different values of
        `TimeScale`.

   Returns
   -------

    <object>.results : List[OutputState.value]
        list of the values, for each `TRIAL`, of the OutputStates for a Mechanism run directly,
        or of the OutputStates of the `TERMINAL` Mechanisms for the Process or System run.
    """

    inputs = _construct_stimulus_sets(object, inputs)

    if targets:
        targets = _construct_stimulus_sets(object, targets, is_target=True)

    object_type = _get_object_type(object)

    if object_type in {MECHANISM, PROCESS}:
        # Insure inputs is 3D to accommodate TIME_STEP dimension assumed by Function.run()
        inputs = np.array(inputs)
        if object_type is MECHANISM:
            mech_len = np.size(object.instance_defaults.variable)
        else:
            mech_len = np.size(object.first_mechanism.instance_defaults.variable)
        # If input dimension is 1 and size is same as input for first mechanism,
        # there is only one input for one execution, so promote dimensionality to 3
        if inputs.ndim == 1 and np.size(inputs) == mech_len:
            while inputs.ndim < 3:
                inputs = np.array([inputs])
        if inputs.ndim == 2 and all(np.size(input) == mech_len for input in inputs):
            inputs = np.expand_dims(inputs, axis=1)
        # FIX:
        # Otherwise, assume multiple executions...
        # MORE HERE

    object.targets = targets

    time_scale = time_scale or TimeScale.TRIAL

    # num_trials = num_trials or len(inputs)
    # num_trials = num_trials or np.size(inputs,(inputs.ndim-1))
    # num_trials = num_trials or np.size(inputs, 0)
    # num_trials = num_trials or np.size(inputs, inputs.ndim-3)
    num_trials = num_trials or np.size(inputs, EXECUTION_SET_DIM)

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
        from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
        for learning_mech in object.learning_mechanisms.mechanisms:
            for projection in learning_mech.output_state.efferents:
                if isinstance(projection, LearningProjection):
                    projection.function_object.learning_rate = object.learning_rate

    # VALIDATE INPUTS: COMMON TO PROCESS AND SYSTEM
    # Input is empty
    if inputs is None or isinstance(inputs, np.ndarray) and not np.size(inputs):
        raise RunError("No inputs arg for \'{}\'.run(): must be a list or np.array of stimuli)".format(object.name))

    # Input must be a list or np.array
    if not isinstance(inputs, (list, np.ndarray)):
        raise RunError("The input must be a list or np.array")

    inputs = np.array(inputs)
    inputs = np.atleast_2d(inputs)

    # Insure that all input sets have the same length
    if any(len(input_set) != len(inputs[0]) for input_set in inputs):
        raise RunError("The length of at least one input in the series is not the same as the rest")

    # Class-specific validation:
    context = context or RUN + "validating " + object.name
    num_inputs_sets = _validate_inputs(object=object, inputs=inputs, context=context)
    if targets is not None:
        _validate_targets(object, targets, num_inputs_sets, context=context)

    if object.verbosePref:
        shape = inputs.shape
        print('Inputs for run of {}: \n'
              '- executions: {}\n'
              '- phases per execution: {}\n'
              '- mechanisms per execution: {}\n'.
              format(object.name, shape[EXECUTION_SET_DIM], shape[PHASE_DIM], shape[MECHANISM_DIM]))

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
    for execution in range(num_trials):

        execution_id = _get_unique_id()

        if call_before_trial:
            call_before_trial()

        for time_step in range(time_steps):

            if call_before_time_step:
                call_before_time_step()

            input_num = execution%len(inputs)
            input = inputs[input_num][time_step]
            if object_type == SYSTEM:
                object.inputs = input

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
            result = object.execute(input=input,
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
def _construct_stimulus_sets(object, stimuli, is_target=False):
    """Return an nparray of stimuli suitable for use as inputs arg for System.run()

    If inputs is a list:
        - the first item in the list can be a header:
            it must contain the names of the origin mechanisms of the System
            in the order in which the inputs are specified in each subsequent item
        - the length of each item must equal the number of origin mechanisms in the System
        - each item should contain a sub-list of inputs for each `ORIGIN` Mechanism in the System

    If inputs is a dict, for each entry:
        - the number of entries must equal the number of `ORIGIN` Mechanisms in the System
        - key must be the name of an origin Mechanism in the System
        - value must be a list of input values for the Mechanism, one for each exeuction
        - the length of all value lists must be the same

    Automatically assign input values to proper phases for Mechanism, and assigns zero to other phases

    For each trial,
       for each time_step
           for each `ORIGIN` Mechanism:
               if phase (from mech tuple) is modulus of time step:
                   draw from each list; else pad with zero
    DIMENSIONS:
       axis 0: num_input_sets
       axis 1: object._phaseSpecMax
       axis 2: len(object.origin_mechanisms)
       axis 3: len(mech.input_states)
       axis 4: items of input_states

    Notes:
    * Construct as lists and then convert to np.array, since size of inputs can be different for different mechs
        so can't initialize a simple (regular) np.array;  this means that stim_list dtype may also be 'O'
    * Code below is not pretty, but needs to test for cases in which inputs have different sizes

    """

    object_type = _get_object_type(object)

    # Stimuli in Sequence format
    if isinstance(stimuli, (list, np.ndarray)):
        stim_list = _construct_from_stimulus_list(object, stimuli, is_target=is_target)

    # Stimuli in Mechanism format
    elif isinstance(stimuli, dict):
        stim_list = _construct_from_stimulus_dict(object, stimuli, is_target=is_target)

    elif is_target and isinstance(stimuli, function_type):
        return stimuli

    else:
        if is_target:
            stim_type = 'targets'
        else:
            stim_type = 'inputs'
        raise RunError("{} arg for {}._construct_stimulus_sets() must be a dict or list".
                          format(stim_type, object.name))

    stim_list_array = np.array(stim_list)
    return stim_list_array

def _construct_from_stimulus_list(object, stimuli, is_target, context=None):
    object_type = _get_object_type(object)

    # Check for header
    headers = None
    if isinstance(stimuli[0],Iterable) and any(isinstance(header, Mechanism) for header in stimuli[0]):
        headers = stimuli[0]
        del stimuli[0]
        for mech in object.origin_mechanisms:
            if not mech in headers:
                raise RunError("Header is missing for ORIGIN Mechanism {} in stimulus list".
                                  format(mech.name, object.name))
        for mech in headers:
            if not mech in object.origin_mechanisms.mechanisms:
                raise RunError("{} in header for stimulus list is not an ORIGIN Mechanism in {}".
                                  format(mech.name, object.name))

    inputs_array = np.array(stimuli)
    if inputs_array.dtype in {np.dtype('int64'),np.dtype('float64')}:
        max_dim = 2
    elif inputs_array.dtype is np.dtype('O'):
        max_dim = 1
    else:
        raise RunError("Unknown data type for inputs in {}".format(object.name))
    while inputs_array.ndim > max_dim:
        # inputs_array = np.hstack(inputs_array)
        inputs_array = np.concatenate(inputs_array)
    inputs = inputs_array.tolist()

    context = context or RUN + ' constructing stimuli for ' + object.name
    num_input_sets = _validate_inputs(object=object,
                                      inputs=inputs,
                                      num_phases=1,
                                      is_target=is_target,
                                      context=context)

    # If inputs are for a mechanism or process, no need to deal with phase so just return
    if object_type in {MECHANISM, PROCESS} or is_target:
        return inputs

    mechs = list(object.origin_mechanisms)
    num_mechs = len(object.origin_mechanisms)
    inputs_flattened = np.hstack(inputs)
    # inputs_flattened = np.concatenate(inputs)
    input_elem = 0    # Used for indexing w/o headers
    execution_offset = 0  # Used for indexing w/ headers
    stim_list = []

    # NOTE: never happens as guessed
    # if object.numPhases > 1:
    #     print('NUM PHASES: {0}'.format(object.numPhases))
    #     import code
    #     code.interact(local=locals())

    for execution in range(num_input_sets):
        execution_len = 0  # Used for indexing w/ headers
        stimuli_in_execution = []
        for phase in range(object.numPhases):
            stimuli_in_phase = []
            for mech_num in range(num_mechs):
                mech = list(object.origin_mechanisms.mechs)[mech_num]
                mech_len = np.size(mechs[mech_num].instance_defaults.variable)
                # Assign stimulus of appropriate size for mech and fill with 0's
                stimulus = np.zeros(mech_len)
                # Assign input elements to stimulus if phase is correct one for mech
                for stim_elem in range(mech_len):
                    if headers:
                        input_index = headers.index(mech) + execution_offset
                    else:
                        input_index = input_elem
                    stimulus[stim_elem] = inputs_flattened[input_index]
                    input_elem += 1
                    execution_len += 1
                # Otherwise, assign vector of 0's with proper length
                stimuli_in_phase.append(stimulus)
            stimuli_in_execution.append(stimuli_in_phase)
        stim_list.append(stimuli_in_execution)
        execution_offset += execution_len
    return stim_list

def _construct_from_stimulus_dict(object, stimuli, is_target):

    object_type = _get_object_type(object)

    # Stimuli are inputs:
    #    validate that there is a one-to-one mapping of input entries to origin mechanisms in the process or system.
    if not is_target:
        # Check that all of the mechanisms listed in the inputs dict are ORIGIN mechanisms in the object
        for mech in stimuli.keys():
            if not mech in object.origin_mechanisms.mechanisms:
                raise RunError("{} in inputs dict for {} is not one of its ORIGIN mechanisms".
                               format(mech.name, object.name))
        # Check that all of the ORIGIN mechanisms in the object are represented by entries in the inputs dict
        for mech in object.origin_mechanisms:
            if not mech in stimuli:
                raise RunError("Entry for ORIGIN Mechanism {} is missing from the inputs dict for {}".
                               format(mech.name, object.name))

    # Note: no need to order entries for inputs (as with targets, below) as that only matters for systems,
    #       and is handled where stimuli for a system are assigned to phases below

    # Stimuli are targets:
    #    - validate that there is a one-to-one mapping of target entries to target mechanisms in the process or system;
    #    - insure that order of target stimuli in dict parallels order of target mechanisms in target_mechanisms list
    else:
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

    # Check that all of the stimuli in each list are compatible with the corresponding mechanism's variable
    for mech, stim_list in stimuli.items():

        # First entry in stimulus list is a single item (possibly an item in a simple list or 1D array)
        if not isinstance(stim_list[0], Iterable):
            # If mech.instance_defaults.variable is also of length 1
            if np.size(mech.instance_defaults.variable) == 1:
                # Wrap each entry in a list
                for i in range(len(stim_list)):
                    stimuli[mech][i] = [stim_list[i]]
            # Length of mech.instance_defaults.variable is > 1, so check if length of list matches it
            elif len(stim_list) == np.size(mech.instance_defaults.variable):
                # Assume that the list consists of a single stimulus, so wrap it in list
                stimuli[mech] = [stim_list]
            else:
                raise RunError("Stimuli for {} of {} are not properly formatted ({})".
                                  # format(append_type_to_name(mech),object.name, stimuli[mech]))
                                  format(mech.name, object.name, stimuli[mech]))

        for stim in stimuli[mech]:
            if not iscompatible(np.atleast_2d(stim), mech.instance_defaults.variable):
                err_msg = "Input stimulus ({}) for {} is incompatible with its variable ({}).".\
                    format(stim, mech.name, mech.instance_defaults.variable)
                # 8/3/17 CW: I admit the error message implementation here is very hacky; but it's at least not a hack
                # for "functionality" but rather a hack for user clarity
                if "KWTA" in str(type(mech)):
                    err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other values)" \
                                        " to represent the outside stimulus for the inhibition input state, and " \
                                        "for systems, put your inputs"
                raise RunError(err_msg)

    stim_lists = list(stimuli.values())
    num_input_sets = len(stim_lists[EXECUTION_SET_DIM])

    # Check that all lists have the same number of stimuli
    if not all(len(np.array(stim_list)) == num_input_sets for stim_list in stim_lists):
        raise RunError("The length of all the stimulus lists must be the same")

    stim_list = []

    # If stimuli are for a process or are targets, construct stimulus list from dict without worrying about phases
    if object_type in {MECHANISM, PROCESS} or is_target:
        for i in range(num_input_sets):
            stims_in_execution = []
            for mech in stimuli:
                stims_in_execution.append(stimuli[mech][i])
            stim_list.append(stims_in_execution)

    # NOTE: never happens as guessed
    # if object.numPhases > 1:
    #     print('NUM PHASES: {0}'.format(object.numPhases))
    #     import code
    #     code.interact(local=locals())

    # Otherwise, for inputs to a system, construct stimulus from dict with phases
    elif object_type is SYSTEM:
        for execution in range(num_input_sets):
            stimuli_in_execution = []
            for phase in range(object.numPhases):
                stimuli_in_phase = []
                # Only assign inputs to origin_mechanisms
                #    and assign them in the order they appear in origin_mechanisms and fill out each phase
                for mech in object.origin_mechanisms.mechs:
                    # Assign input elements to stimulus if phase is correct one for mech

                    # Get stimulus for mech for current execution, and enforce 2d to accomodate input_states per mech
                    stimulus = np.atleast_2d(stimuli[mech][execution])
                    if not isinstance(stimulus, Iterable):
                        stimulus = np.atleast_2d([stimulus])

                    stimuli_in_phase.append(stimulus)

                stimuli_in_execution.append(stimuli_in_phase)
            stim_list.append(stimuli_in_execution)

    else:
        raise RunError("PROGRAM ERROR: illegal type for run ({}); should have been caught by _get_object_type ".
                       format(object_type))

    try:
        stim_list = np.array(stim_list)
    except ValueError:
        # for i in range(len(stim_list[0][0])):
        #     stim_list[0][0][i][0]
        #     stim_list[0][0][i] = np.array(stim_list[0][0][i])
        # for exec in range(len(stim_list)):
        #     for phase in range(len(stim_list[exec])):
        #         for mech in range(len(stim_list[exec][phase])):
        #             stim_list[exec][phase][mech] = [stim_list[exec][phase][mech].tolist()]
        for exec in range(len(stim_list)):
            for phase in range(len(stim_list[exec])):
                for mech in range(len(stim_list[exec][phase])):
                    stim_list[exec][phase][mech] = stim_list[exec][phase][mech].tolist()
        stim_list = np.array(stim_list)

    return np.array(stim_list)

def _validate_inputs(object, inputs=None, is_target=False, num_phases=None, context=None):
    """Validate inputs for _construct_inputs() and object.run()

    If inputs is an np.ndarray:
        inputs must be 3D (if inputs to each process are different lengths) or 4D (if they are homogenous):
            axis 0 (outer-most): inputs for each execution of the run (len == number of executions to be run)
                (note: this is validated in super().run()
            axis 1: inputs for each time step of a trial (len == _phaseSpecMax of System (no. of time_steps per trial)
            axis 2: inputs to the System, one for each Process (len == number of Processes in System)

    returns number of input_sets (one per execution)
    """

    object_type = _get_object_type(object)

    if object_type is PROCESS:

        if isinstance(inputs, list):
            inputs = np.array(inputs)

        # If inputs to process are heterogenous, inputs.ndim should be 2:
        if inputs.dtype is np.dtype('O') and inputs.ndim != 2:
            raise RunError("inputs arg in call to {}.run() must be a 2D np.array or comparable list".
                              format(object.name))

        # If inputs to process are homogeneous, inputs.ndim should be 2 if length of input == 1, else 3:
        if inputs.dtype in {np.dtype('int64'),np.dtype('float64')}:
            # Get a sample length (use first, since it is convenient and all are the same)
            mech_len = len(object.first_mechanism.instance_defaults.variable)
            if not ((mech_len == 1 and inputs.ndim == 2) or inputs.ndim == 3):
                raise RunError("inputs arg in call to {}.run() must be a 3d np.array or comparable list".
                                  format(object.name))

        num_input_sets = np.size(inputs, inputs.ndim-3)

        return num_input_sets

    elif object_type is SYSTEM:

        if is_target:
            num_phases = 1
        else:
            num_phases = num_phases or object.numPhases

        if not isinstance(inputs, np.ndarray):
            # raise RunError("PROGRAM ERROR: inputs must an ndarray")
            inputs = np.array(inputs)

        states_per_mech_heterog = False
        size_of_states_heterog = False
        if inputs.dtype in {np.dtype('int64'),np.dtype('float64')}:
            input_homogenity = HOMOGENOUS
        elif inputs.dtype is np.dtype('O'):
            input_homogenity = HETEROGENOUS
            # Determine whether the number of states/mech is homogenous
            num_states_in_first_mech = len(object.origin_mechanisms[0].input_states)
            if any(len(mech.input_states) != num_states_in_first_mech for mech in object.origin_mechanisms):
                states_per_mech_heterog = True
            # Determine whether the size of all states is homogenous
            size_of_first_state = len(object.origin_mechanisms[0].input_states[0].value)
            for origin_mech in object.origin_mechanisms:
                if any(len(state.value) != size_of_first_state for state in origin_mech.input_states):
                    size_of_states_heterog = True
        else:
            raise RunError("Unknown data type for inputs in {}".format(object.name))

        if is_target:    # No phase dimension, so one less than for stimulus inputs
            # If targets are homogeneous, inputs.ndim should be 4:
            # If targets are heterogenous:
            #   if states/mech are homogenous, inputs.ndim should be 3
            #   if states/mech are heterogenous, inputs.ndim should be 2
            expected_dim = 2 + input_homogenity + states_per_mech_heterog
        else:            # Stimuli have phases, so one extra dimension
            # If inputs are homogeneous, inputs.ndim should be 5;
            # If inputs are heterogenous:
            #   if states sizes are heterogenous, inputs.ndim should be 4
            #   if states/mech are heterogenous, inputs.ndim should be 3
            #   if both are heterogenous, inputs.ndim should be 3
            if input_homogenity:
                expected_dim = 5
            elif states_per_mech_heterog:
                expected_dim = 3
            elif size_of_states_heterog:
                expected_dim = 4
            else:
                raise RunError("PROGRAM ERROR: Unexpected shape of inputs: {}".format(inputs.shape))

        if inputs.ndim != expected_dim:
            raise RunError("inputs arg in call to {}.run() must be a {}d np.array or comparable list".
                              format(object.name, expected_dim))

        if np.size(inputs,PROCESSES_DIM) != len(object.origin_mechanisms):
            raise RunError("The number of inputs for each execution ({}) in the call to {}.run() "
                              "does not match the number of Processes in the System ({})".
                              format(np.size(inputs,PROCESSES_DIM),
                                     object.name,
                                     len(object.origin_mechanisms)))

        # Check that length of each input matches length of corresponding origin mechanism over all executions and phases
        if is_target:
            mechs = list(object.target_mechanisms)
        else:
            mechs = list(object.origin_mechanisms)
        num_mechs = len(mechs)
        inputs_array = np.array(inputs)
        num_execution_sets = inputs_array.shape[EXECUTION_SET_DIM]
        for execution_set_num in range(num_execution_sets):
            execution_set = inputs_array[execution_set_num]
            for phase_num in range(num_phases):
                inputs_for_phase = execution_set[phase_num]
                if len(inputs_for_phase) != num_mechs:
                    raise RunError("Number of mechanisms ({}) in input for phase {} should be {}".
                                   format(len(inputs_for_phase), phase_num, num_mechs))
                for mech_num in range(num_mechs):
                    input_for_mech = inputs_for_phase[mech_num]
                    if len(input_for_mech) != len(mechs[mech_num].input_values):
                        raise RunError("Number of states ({}) in input for {} should be {}".
                                       format(len(input_for_mech),
                                              mechs[mech_num].name,
                                              len(mechs[mech_num].input_values)))
                    for state_num in range(len(input_for_mech)):
                        input_for_state = mechs[mech_num].input_values[state_num]
                        if len(input_for_state) != len(mechs[mech_num].input_values[state_num]):
                            raise RunError("Length of state {} ({}) in input for {} should be {}".
                                           format(list(mechs[mech_num].input_states)[state_num],
                                                  len(input_for_state),
                                                  mechs[mech_num].name,
                                                  len(mechs[mech_num].input_values[state_num])))
        return num_execution_sets

    else:
        raise RunError("PROGRAM ERRROR: {} type not currently supported by _validate_inputs in Run module for ".
                       format(object.__class__.__name__))

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

            if targets.dtype in {np.dtype('int64'),np.dtype('float64')}:
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
    elif isinstance(object, Process):
        return PROCESS
    elif isinstance(object, System):
        return SYSTEM
    else:
        raise RunError("{} type not supported by Run module".format(object.__class__.__name__))


import uuid
def _get_unique_id():
    return uuid.uuid4()
