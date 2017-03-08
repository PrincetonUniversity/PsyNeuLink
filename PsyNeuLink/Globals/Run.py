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

The :keyword:`run` function is used for executing a mechanism, process or system.  It can be called directly, however
it is typically invoked by calling the :keyword:`run` method of the object to be run.  It  executes an object by
calling the object's :keyword:`execute` method.  While an object's :keyword:`execute` method can be called directly,
using its :keyword:`run` method is much easier because it:

    * allows multiple rounds of execution to be run in sequence, whereas the :keyword:`execute` method of an object
      runs only a single execution of the object;
    ..
    * uses simpler formats for specifying `inputs <Run_Inputs>` and `targets <Run_Targets>`;
    ..
COMMENT:
    *** THIS WILL NEED TO BE UPDATED ONCE SCHEDULER IS IMPLEMENTED
COMMENT
    * manages timing factors (such as updating the `CentralClock <TimeScale.CentralClock>` and presenting
    inputs in the correct `phase of execution <System_Execution_Phase>` of a system.
    ..
    * automatically aggregates results across executions and stores them in the results attribute of the object.

COMMENT:
Note:: The ``run`` function uses the ``construct_input`` function to convert the input into the format required by
``execute`` methods.
COMMENT

Understanding a few basic concepts about how the :keyword:`run` function operates will make it easier to use the
:keyword:`execute` and :keyword:`run` methods of PsyNeuLink objects.  These are discussed below.


Scope of Execution
~~~~~~~~~~~~~~~~~~

When the :keyword:`run` method of an object is called, it executes that object and all others within its scope of
execution.  For a `mechanism <Mechanism>`, the scope of execution is simply that mechanism.  For a `process <Process>`,
the scope of execution is all of the mechanisms specified in its `pathway` attribute.  For a `system <System>`,
the scope of execution is all of the mechanisms in the processes specified in the system's
`processes <System.System_Base.processes>` attribute.

.. _Run_Timing:

Timing
~~~~~~

COMMENT:
    *** THIS WILL NEED TO BE UPDATED ONCE SCHEDULER IS IMPLEMENTED
COMMENT

PsyNeuLink supports two time scales for executing objects: `TIME_STEP <TimeScale.TimeScale.TIME_STEP>` and
`TRIAL <TimeScale.TimeScale.TRIAL>`.  Every mechanism defines how it is executed at one or both of these time
scales, and its current mode of execution is determined by its `timeScale <Mechanism.Mechanism_Base.timeScale>`
attribute.

.. _Run_TIME_STEP:

* `TIME_STEP <TimeScale.TimeScale.TIME_STEP>`:  this mode of execution is a mechanism's closest approximation
  to continuous, or "real time" processing.  Execution of a `time_step` is defined as a single execution of all objects
  in the scope of execution at their `time_step` time scale.  Mechanisms called upon to execute a `time_step` that do
  not support that time scale of execution have the option of generating an exception, being ignored, or providing
  their trial mode response, either on the first `time_step`, every `time_step`, or the last `time_step` in the sequence
  being run.

.. _Run_TRIAL:

* `TRIAL <TimeScale.TimeScale.TIME_TRIAL>`: this mode of execution is the "ballistic" execution of a
  mechanism to a state that would have been achieved with `time_step` execution to a specified criterion.  The
  criterion can be specified in terms of the number of `time_steps`, or a condition to be met by the mechanism's
  output.  It is up to the mechanism how it implements its `trial` mode of execution (e.g., whether this is done by
  internal numerical iteration or an analytic calculation). Execution of a `trial` is defined as the execution of a
  `trial` of all of the objects in the scope of execution.

The :keyword:`time_scale` argument of an :keyword:`execute` or :keyword:`run` method determines the time scale for each
round of execution: a `time_step` or a `trial`.  When a `process <Process>` is run, each mechanism is executed in the
order that it appears in the process' `pathway`, once per round of execution.  When a `system <System>` is run,
the order of execution is determined by the system's `executionList` attribute, which is based on the system's
`graph` (a list of the dependencies among all of the mechanisms in the system).  Execution of the mechanisms in a
system also depends on the `phaseSpec` of each mechanism: this determines *when* in an execution sequence it should
be executed. The `CentralClock <TimeScale.CentralClock>` is used to control timing, so executing a system
requires that it be appropriately updated.

The :keyword:`run` function handles all of the above factors automatically.


.. _Run_Inputs:

Inputs
~~~~~~

COMMENT:
    OUT-TAKES
    The inputs for a single execution must contain a value for each :doc:`inputState <InputState> of each
    :py:data:`ORIGIN` <Keywords.Keywords.ORIGIN>` mechanism in the process or system, using the same format
    used for the format of the input for the execute method of a process or system.  This can be specified as a
    nested set of lists, or an ndarray.  The exact structure is determined by a number of factors, as described below.
    the number of `ORIGIN` mechanisms involved (a process has only one, but a system can have several), the
    number of inputStates for each `ORIGIN` mechanism, and whether the input to those inputStates is
    single-element (such as scalars), multi-element (such as vectors) or a mix.  For the run method, the structure is
    further determined by whether only a single execution or multiple executions is specified.  Rather than specifying a
    single format structure that must be used for all purposes (which would necessarily be the most complex one),
    PsyNeuLink is designed to be flexible, allowing use of the simplest structure necessary to describe the input for a
    particular process or input, which can vary according to circumstance.  Examples are provided below.  In all cases,
    either nested lists or ndarrays can be used, in which the innermost level (highest axis of an ndarray) is used to
    specify the input values for a given inputState (if any are multi-element), the next nested level (second highest
    axis) is used to specify the different inputStates of a given mechanism (if any have more than one), the level
    (axis) after that is used to specify the different `ORIGIN` mechanisms (if there is more than one), and
    finally the outermost level (lowest axis) is used to specify different trials (if there is more than one to be run).

    PsyNeuLink affords flexibility of input format that PsyNeuLink allows, the structure of the input can vary
    (i.e., the levels of nesting of the list, or dimensionality and shape of the ndarray used to specify it).
    The run function handles all of these formats seamlessly, so that whathever notation is simplest and easiest
    for a given purpose can be used (though, as noted above, it is best to consistently specify the input value of
    an inputstae as a list or array (axis of an ndarray).
COMMENT

The :keyword:`run` function presents the inputs for each round of execution to the inputStates of the relevant
mechanisms. These are specified in the :keyword:`inputs` argument of the :keyword:`execute` or :keyword:`run` method.
For a mechanism, they comprise the input value for each of the mechanism's `inputStates <InputState>`.  For a process
or system, they comprise the input values for the inputState(s) of the `ORIGIN` mechanism(s).  Input values can be
specified in one of two ways: `sequence format <Run_Inputs_Sequence_Format>` and `mechanism format <Run_Dict_format>`.
Sequence format is more complex, but does not require the specification of mechanisms by name, and thus may better
suited for automated means of generating inputs.  Mechanism format requires that inputs be assigned to mechanisms by
name, but is easier to use (as the order in which the mechanisms are specified does not matter).  Both formats require
that inputs be specified as nested lists or ndarrays, that define the number of executions, mechanisms, inputStates
and elements of each input value.  These factors determine the levels of nesting required for a list, or
the dimensionality (number of axes) for an ndarray.  They are described below, followed by a description of the two
formats.

* **Number of rounds of execution**.  If the :keyword:`inputs` argument contains the input for more than one round of
  execution (i.e., multiple time_steps and/or trials), then the outermost level of the list, or axis 0 of the ndarray,
  is used for the rounds of execution, each item of which contains the set inputs for a given round.  Otherwise, it is
  used for the next relevant factor in the list below.  If the number of inputs specified is less than the number of
  executions, then the input list is cycled until the full number of executions is completed.
..
* **Number of mechanisms.** If :keyword:`run` is used for a system, and it has more than one `ORIGIN` mechanism, then
  the next level of nesting of a list, or next higher axis of an ndarray, is used for the `ORIGIN` mechanisms, with
  each item containing the inputs for a given `ORIGIN` mechanism within a round.  This factor is not relevant
  when run is used for a single mechanism, a process (which only ever has one `ORIGIN` mechanism), or a system that
  has only one `ORIGIN` mechanism.  It is also not relevant for the `mechanism format <Run_Inputs_Mechanism_Format>`,
  since that separates the inputs for each mechanism into separate entries of a dictionary.
..
* **Number of inputStates.** In general, mechanisms have a single ("primary") inputState; however, some types of
  mechanisms can have more than one (see `Mechanism_InputStates`).  If any `ORIGIN` mechanism in a process or
  system has more than one inputState, then the next level of nesting of a list, or next higher axis of an ndarray,
  is used for the set of inputStates for each mechanism.
..
* **Number of elements for the value of an inputState.** The input for an inputState can be a single element (e.g.,
  a scalar) or have multiple elements (e.g., a vector).  By convention, even if the input to an inputState is only a
  single element, it should nevertheless always be specified as a list or a 1d np.array (it is internally converted to
  the latter).  PsyNeuLink can usually parse single-element inputs specified as a stand-alone value (e.g., as a number
  not in a list or ndarray).  Nevertheless, it is best to embed such inputs in a single-element list or a 1d array,
  both for clarity and to insure consistent treatment of nested lists and ndarrays.  If this convention is followed,
  then the number of elements for a given input should not affect nesting of lists or dimensionality (number of axes)
  of ndarrays of an :keyword:`inputs` argument.

With these factors in mind, the :keyword:`inputs` argument can be specified in the simplest form possible (least
number of nestings for a list, or lowest dimension of an ndarray).  It can be specified using one of two formats:

.. _Run_Inputs_Sequence_Format:

Sequence Format
^^^^^^^^^^^^^^^

*(List[values] or ndarray)* -- this uses a nested list or ndarray to fully specify the input for
each round of execution in a sequence.  It is more complex than the `mechanism format <Run_Inputs_Mechanism_Format>`,
and for systems requires that the inputs for each mechanism be specified in the same order in which those mechanisms
appear in the system's `originMechanisms <System.System_Base.originMechanisms>` attribute.  This is
generally the same order in which they are declared, and can be displayed using the system's
`show <System.System_Base.show>` method). Although this is format is more demanding, it may be better suited to
automated input generation, since it does not require that mechanisms be referenced explicitly (though it is
allowed). The following provides a description of the sequence format for all of the combinations of factors listed
above.  The `figure <Run_Sequence_Format_Fig>` below shows examples.

    *Lists:* if there is more than one round, then the outermost level of the list is used for the sequence of
    executions.  If there is only one `ORIGIN` mechanism and it has only one inputState (the most common
    case), then a single sublist is used for the input of each round.  If the `ORIGIN` mechanism has more
    than one inputState, then the entry for each round is a sublist of the inputStates, each entry of which is a 
    sublist containing the input for that inputState.  If there is more than one mechanism, but none have more than 
    one inputState, then a sublist is used for each mechanism in each round, within which a sublist is used for the
    input for that mechanism.  If there is more than one mechanism, and any have more than one inputState,
    then a sublist is used for each mechanism for each round, within which a sublist is used for each
    inputState of the corresponding mechanism, and inside that a sublist is used for the input for each inputState.

    *ndarray:*  axis 0 is used for the first factor (round, mechanism, inputState or input) for which there is only one
    item, axis 1 is used for the next factor for which there is only one item, and so forth.  For example, if there is
    more than one round, only one `ORIGIN` mechanism, and that has only one inputState (the most common case),
    then axis 0 is used for round, and axis 1 for inputs per round.  In the extreme, if there are multiple rounds,
    more than one `ORIGIN` mechanism, and more than one inputState for one or more of the `ORIGIN` mechanisms,
    then axis 0 is used for rounds, axis 1 for mechanisms within round, axis 2 for inputStates of each mechanism, and
    axis 3 for the input to each inputState of a mechanism.  Note that if *any* mechanism being run (directly, or as
    one of the `ORIGIN` mechanisms of a process or system) has more than one inputState, then an axis must be
    committed to inputStates, and the input to every inputState of every mechanism must be specified in that axis
    (i.e., even for those mechanisms that have a single inputState).

    .. _Run_Sequence_Format_Fig:

    .. figure:: _static/Sequence_format_input_specs_fig.*
       :alt: Example input specifications in sequence format
       :scale: 75 %
       :align: center

       Example input specifications in sequence format

.. _Run_Inputs_Mechanism_Format:

Mechanism Format
^^^^^^^^^^^^^^^^

*(Dict[mechanism, List[values] or ndarray])* -- this provides a simpler format for specifying :keyword:`inputs` than
the sequence format, and does not require that the inputs for each mechanism be specified in a particular order.
However, it requires that each mechanism that receives inputs be referenced explicitly (instead of by order),
which may be less suitable for automated forms of input generation.  It uses a dictionary, each entry of which is the
sequence of inputs for an `ORIGIN` mechanism;  there must be one such entry for each of the `ORIGIN` mechanisms of the
process or system being run.  The key for each entry is the `ORIGIN` mechanism, and the value contains either a list
or ndarray specifying the sequence of inputs for that mechanism, one for each round of execution.  If a list is used,
and the mechanism has more than one inputState, then a sublist is used in each item of the list to specify the inputs
for each of the mechanism's inputStates for that round.  If an ndarray is used, axis 0 is used for the sequence of
rounds. If the mechanism has a single inputState, then axis 1 is used for the input for each round.  If the mechanism
has multiple inputStates, then axis 1 is used for the inputStates, and axis 2 is used for the input to each
inputState for each round.

    .. figure:: _static/Mechanism_format_input_specs_fig.*
       :alt: Mechanism format input specification
       :align: center

       Mechanism format input specification

.. _Run_Initial_Values:

Initial Values
~~~~~~~~~~~~~~

Any mechanism that is the `sender <Projection.Projection.sender>` of a projection that closes a loop in a process or
system, and that is not an `ORIGIN` mechanism, is designated as `INITIALIZE_CYCLE`. An initial value can be assigned
to such mechanisms, that will be used to initialize the process or system when it is first run.  These values are
specified in the :keyword:`initial_values` argument of :keyword:`run`, as a dictionary. The key for each entry must
be a mechanism designated as `INITIALIZE_CYCLE`, and its value an input for the mechanism to be used as its initial
value.  The size of the input (length of the outermost level if it is a list, or axis 0 if it is an np.ndarray),
must equal the number of inputStates of the mechanism, and the size of each value must match (in number and type of
elements) that of the `variable <InputState.InputState.variable>` for the corresponding inputState.

.. _Run_Targets:

Targets
~~~~~~~

If learning is specified for a `process <Process_Learning>` or `system <System_Execution_Learning>`, then target values
for each round of execution must be provided for each `TARGET` mechanism in the process or system being run.  These
are specified in the :keyword:`targets` argument of the :keyword:`execute` or :keyword:`run` method, which can be in
any of three formats.  The two formats used for :keyword:`inputs` (`sequence <Run_Inputs_Sequence_Format>` and
`mechanism <Run_Inputs_Mechanism_Format>` format) can also be used for targets.  However, the format of the lists or
ndarrays is simpler, since each `TARGET` mechanism is assigned only a single target value, so there is never the need
for the extra level of nesting (or dimension of ndarray) used for inputStates in the specification of :keyword:`inputs`.
Details concerning the use of the `sequence <Run_Targets_Sequence_Format>`  and
`mechanism <Run_Targets_Mechanism_Format>` formats for targets is described below. Targets can also be specified
as a `function <Run_Targets_Function_Format>` (for example, to allow the target to depend on the outcome of processing).

If either the sequence or mechanism format is used, then the number of targets specified for each mechanism must
equal the number specified for the :keyword:`inputs` argument;  as with :keyword:`inputs`, if the number of executions
specified is greater than the number of inputs (and targets), then the list will be cycled until the number of
executions specified is completed.  If a function is used for the :keyword:`targets`, then it will be used to generate
a target for each round of execution.

The number of targets specified in the sequence or mechanism formats for each round of execution, or generated using
the function format, must equal the number of `TARGET` mechanisms for the process or system being run (see process
`targetMechanism <Process.Process_Base.targetMechanisms>` or
system `targetMechanism <System.System_Base.targetMechanisms>` respectively), and the value of each target must
match (in number and type of elements) that  of the `target <ComparatorMechanism.ComparatorMechanism.target>`
attribute of the `TARGET` mechanism for which it is intended.  Furthermore, if a range is specified for the output of
the `TERMINAL` mechanism with which the target is compared (that is, the mechanism that provides the
`ComparatorMechanism's <ComparatorMechanism>` `sample <ComparatorMechanism.ComparatorMechanism.sample>`
value, then the target must be within that range (for example, if the `TERMINAL` mechanism is a
`TransferMechanism` that uses a `Logistic` function, it's `range <TransferMechanism.TransferMechanism.range>` is
[0,1], so the target must be within that range).

.. _Run_Targets_Sequence_Format:

Sequence Format
^^^^^^^^^^^^^^^

*(List[values] or ndarray):* -- there are at most three levels of nesting (or dimensions) required for
:keyword:`targets`:  one for executions, one for mechanisms, and one for the elements of each input.  For a system
with more than one `TARGET` mechanism, the targets must be specified in the same order as they appear in the system's
`targetMechanisms <System.System_Base.targetMechanisms>` attribute.  This should be the same order in which
they are declared, and can be displayed using the system's `show <System.System_Base.show>` method). All
other requirements are the same as the `sequence format <Run_Inputs_Sequence_Format>` for :keyword:`inputs`.

.. _Run_Targets_Mechanism_Format:

Mechanism Format
^^^^^^^^^^^^^^^^
*(Dict[mechanism, List[values] or ndarray]):* -- there must be one entry in the dictionary for each of the `TARGET`
mechanisms in the process or system being run, though the entries can be specified in any order.  For this reason,
this format may be easier (and safer) to use. The value of each entry is a list or ndarray of the target values for
that mechanism, one for each round of execution. There are at most two levels of nesting (or dimensions)
required for each entry: one for the execution, and the other for the elements of each input.  In all other respects,
the format is the same as the `mechanism format <Run_Inputs_Mechanism_Format>` for :keyword:`inputs`.

.. _Run_Targets_Function_Format:

Function Format
^^^^^^^^^^^^^^^

*[Function]:* -- the function must return an array with a number of items equal to the number of `TARGET` mechanisms
for the process  or system being run, each of which must match (in number and type of elements) the
`target <ComparatorMechanism.ComparatorMechanism.target>` attribute of the `TARGET` mechanism for which it is
intended. This format allows targets to be constructed programmatically, in response to computations made during the
run.

COMMENT:
    ADD EXAMPLE HERE
COMMENT

COMMENT:
   Module Contents
       system() factory method:  instantiate system
       System_Base: class definition
COMMENT


.. _Run_Class_Reference:

Class Reference
---------------

"""


import numpy as np
from collections import Iterable
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Components.Component import function_type
from PsyNeuLink.Components.System import System
from PsyNeuLink.Components.Process import Process, ProcessInputState
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import COMPARATOR_SAMPLE, \
                                                                                      COMPARATOR_TARGET

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
        num_executions:tc.optional(int)=None,
        reset_clock:bool=True,
        initialize:bool=False,
        intial_values:tc.optional(tc.any(list, np.ndarray))=None,
        targets:tc.optional(tc.any(list, dict, np.ndarray, function_type))=None,
        learning:tc.optional(bool)=None,
        call_before_trial:tc.optional(function_type)=None,
        call_after_trial:tc.optional(function_type)=None,
        call_before_time_step:tc.optional(function_type)=None,
        call_after_time_step:tc.optional(function_type)=None,
        clock=CentralClock,
        time_scale:tc.optional(tc.enum(TimeScale.TRIAL, TimeScale.TIME_STEP))=None,
        context=None):
    """run(                         \
    inputs,                      \
    num_executions=None,         \
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

    Run a sequence of executions for a `process <Process>` or `system <System>`.

    COMMENT:
        First, validate inputs (and targets, if learning is enabled).  Then, for each round of execution:
            * call call_before_trial if specified;
            * for each time_step in the trial:
                * call call_before_time_step if specified;
                * call ``object.execute`` with inputs, and append result to ``object.results``;
                * call call_after_time_step if specified;
            * call call_after_trial if specified.
        Return ``object.results``.

        The inputs argument must be a list or an np.ndarray array of the appropriate dimensionality:
            * the inner-most dimension must equal the length of object.variable (i.e., the input to the object);
            * for mechanism format, the length of the value of all entries must be equal (== number of executions);
            * the outer-most dimension is the number of input sets (num_input_sets) specified (one per execution)
                Note: num_input_sets need not equal num_executions (the number of executions to actually run)
                      if num_executions > num_input_sets:
                          executions will cycle through input_sets, with the final one being only a partial cycle
                      if num_executions < num_input_sets:
                          the executions will only partially sample the input sets
    COMMENT

   Arguments
   ---------

    inputs : List[input] or ndarray(input) : default default_input_value for a single execution
        the input for each execution in a sequence (see `Run_Inputs` for detailed description of formatting
        requirements and options).

    num_executions : int : default None
        the number of executions to carry out.  If it is `None` (the default), then a number of executions will be
        carried out equal to the number of :keyword:`inputs`.  If :keyword:`num_executions` exceeds the number of
        :keyword:`inputs`, then the :keyword:`inputs` will be cycled until the number of executions specified is
        completed.

    reset_clock : bool : default True
        if :keyword:`True`, resets `CentralClock` to 0 before a sequence of executions.

    initialize : bool default False
        calls the `initialize <System.System_Base.initialize>` method of the system prior to a sequence of executions.

    initial_values : Dict[Mechanism, List[input] or np.ndarray(input)] : default None
        the initial values assigned to mechanisms designated as `INITIALIZE_CYCLE`.

    targets : List[input] or np.ndarray(input) : default None
        the target values assigned to the `MonitoringMechanism` for each execution (used for learning).
        The length must be equal to :keyword:`inputs`.

    learning : bool :  default None
        enables or disables learning during execution for a `process <Process_Learning>` or
        `system <System_Execution_Learning>`.  If it is not specified, the current state of learning is left intact.
        If it is :keyword:`True`, learning is forced on; if it is :keyword:`False`, learning is forced off.

    call_before_trial : Function : default= `None`
        called before each `trial` in the sequence is executed.

    call_after_trial : Function : default= `None`
        called after each `trial` in the sequence is executed.

    call_before_time_step : Function : default= ``None`
        called before each `time_step` is executed.

    call_after_time_step : Function : default= `None`
        called after each `time_step` is executed.

    time_scale : TimeScale :  default TimeScale.TRIAL
        specifies whether mechanisms are executed for a single time_step or a trial

    Returns
    -------

    <object>.results : List[outputState.value]
        list of the values, for each execution, of the outputStates for a mechanism run directly,
        or of the outputStates of the `TERMINAL` mechanisms for the process or system run
    """

    inputs = _construct_stimulus_sets(object, inputs)
    if targets:
        targets = _construct_stimulus_sets(object, targets, is_target=True)

    object_type = _get_obect_type(object)

    if object_type in {MECHANISM, PROCESS}:
        # Insure inputs is 3D to accommodate TIME_STEP dimension assumed by Function.run()
        inputs = np.array(inputs)
        if object_type is MECHANISM:
            mech_len = np.size(object.variable)
        else:
            mech_len = np.size(object.firstMechanism.variable)
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

    # num_executions = num_executions or len(inputs)
    # num_executions = num_executions or np.size(inputs,(inputs.ndim-1))
    # num_executions = num_executions or np.size(inputs, 0)
    # num_executions = num_executions or np.size(inputs, inputs.ndim-3)
    num_executions = num_executions or np.size(inputs, EXECUTION_SET_DIM)

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
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
        from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
        for learning_mech in object.monitoringMechanisms.mechanisms:
            for projection in learning_mech.outputState.sendsToProjections:
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
    for execution in range(num_executions):

        execution_id = _get_get_execution_id()

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

                # IMPLEMENTATION NOTE:  USE input_num since it # of inputs must equal # targets,
                #                       where as targets can be assigned a function (so can't be used to generated #)
                elif object_type == PROCESS:
                    # object.target = targets[input_num][time_step]
                    object.target = targets[input_num][time_step]

                elif object_type == SYSTEM:
                    object.current_targets = targets[input_num]

            if RUN in context and not EVC_SIMULATION in context:
                context = RUN + ": EXECUTING " + object_type.upper() + " " + object.name
            result = object.execute(input=input,
                                    execution_id=execution_id,
                                    clock=clock,
                                    time_scale=time_scale,
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
    """Return an nparray of stimuli suitable for use as inputs arg for system.run()

    If inputs is a list:
        - the first item in the list can be a header:
            it must contain the names of the origin mechanisms of the system
            in the order in which the inputs are specified in each subsequent item
        - the length of each item must equal the number of origin mechanisms in the system
        - each item should contain a sub-list of inputs for each origin mechanism in the system

    If inputs is a dict, for each entry:
        - the number of entries must equal the number of origin mechanisms in the system
        - key must be the name of an origin mechanism in the system
        - value must be a list of input values for the mechanism, one for each exeuction
        - the length of all value lists must be the same

    Automatically assign input values to proper phases for mechanism, and assigns zero to other phases

    For each trial,
       for each time_step
           for each origin mechanism:
               if phase (from mech tuple) is modulus of time step:
                   draw from each list; else pad with zero
    DIMENSIONS:
       axis 0: num_input_sets
       axis 1: object._phaseSpecMax
       axis 2: len(object.originMechanisms)
       axis 3: len(mech.inputStates)
       axis 4: items of inputStates

    Notes:
    * Construct as lists and then convert to np.array, since size of inputs can be different for different mechs
        so can't initialize a simple (regular) np.array;  this means that stim_list dtype may also be 'O'
    * Code below is not pretty, but needs to test for cases in which inputs have different sizes

    """

    object_type = _get_obect_type(object)

    # Stimuli in sequence format
    if isinstance(stimuli, (list, np.ndarray)):
        stim_list = _construct_from_stimulus_list(object, stimuli, is_target=is_target)

    # Stimuli in mechanism format
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

    object_type = _get_obect_type(object)

    # Check for header
    headers = None
    if isinstance(stimuli[0],Iterable) and any(isinstance(header, Mechanism) for header in stimuli[0]):
        headers = stimuli[0]
        del stimuli[0]
        for mech in object.originMechanisms:
            if not mech in headers:
                raise RunError("Header is missing for origin mechanism {} in stimulus list".
                                  format(mech.name, object.name))
        for mech in headers:
            if not mech in object.originMechanisms.mechanisms:
                raise RunError("{} in header for stimulus list is not an origin mechanism in {}".
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

    mechs = list(object.originMechanisms)
    num_mechs = len(object.originMechanisms)
    inputs_flattened = np.hstack(inputs)
    # inputs_flattened = np.concatenate(inputs)
    input_elem = 0    # Used for indexing w/o headers
    execution_offset = 0  # Used for indexing w/ headers
    stim_list = []

    for execution in range(num_input_sets):
        execution_len = 0  # Used for indexing w/ headers
        stimuli_in_execution = []
        for phase in range(object.numPhases):
            stimuli_in_phase = []
            for mech_num in range(num_mechs):
                mech, runtime_params, phase_spec = list(object.originMechanisms.mech_tuples)[mech_num]
                mech_len = np.size(mechs[mech_num].variable)
                # Assign stimulus of appropriate size for mech and fill with 0's
                stimulus = np.zeros(mech_len)
                # Assign input elements to stimulus if phase is correct one for mech
                if phase == phase_spec:
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

    object_type = _get_obect_type(object)

    # Stimuli are inputs:
    #    validate that there is a one-to-one mapping of input entries to origin mechanisms in the process or system.
    if not is_target:
        for mech in object.originMechanisms:
            if not mech in stimuli:
                raise RunError("Stimulus list is missing for origin mechanism {}".format(mech.name, object.name))
        for mech in stimuli.keys():
            if not mech in object.originMechanisms.mechanisms:
                raise RunError("{} is not an origin mechanism in {}".format(mech.name, object.name))

    # Note: no need to order entries for inputs (as with targets, below) as that only matters for systems,
    #       and is handled where stimuli for a system are assigned to phases below

    # Stimuli are targets:
    #    - validate that there is a one-to-one mapping of target entries to target mechanisms in the process or system;
    #    - insure that order of target stimuli in dict parallels order of target mechanisms in targetMechanisms list
    else:
        # FIX: RE-WRITE USING NEXT AND StopIteration EXCEPTION ON FAIL TO FIND (THIS GIVES SPECIFICS)
        # FIX: TRY USING compare METHOD OF DICT OR LIST?
        # Check that every target in the process or system receives a projection from a mechanism named in the dict
        # from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import SAMPLE
        for target in object.targetMechanisms:
            # If any projection to a target does not have a sender in the stimulus dict, raise an exception
            if not any(mech is projection.sender.owner for
                       projection in target.inputStates[SAMPLE].receivesFromProjections
                       for mech in stimuli.keys()):
                    raise RunError("Entry for {} is missing from specification of targets for run of {}".
                                   format(target.inputStates[COMPARATOR_SAMPLE].
                                          receivesFromProjections[0].sender.owner.name,
                                          object.name))

        # FIX: COULD JUST IGNORE THOSE, OR WARN ABOUT THEM IF VERBOSE?

        # Check that each target referenced in the dict (key)
        #     is the name of a mechanism that projects to a target (comparator) in the system
        terminal_to_target_mapping = {}
        for mech in stimuli.keys():
            # If any mechanism in the stimulus dict does not have a projection to the target, raise an exception
            if not any(target is projection.receiver.owner for
                       projection in mech.outputState.sendsToProjections
                       for target in object.targetMechanisms):
                raise RunError("{} is not a target mechanism in {}".format(mech.name, object.name))
            # Get target mech (comparator) for each entry in stimuli dict:
            terminal_to_target_mapping[mech] = mech.outputState.sendsToProjections[0]

        # Insure that target lists in dict are accessed in the same order as the
        #   targets in the system's targetMechanisms list, by reassigning targets to an OrderedDict:
        from collections import OrderedDict
        ordered_targets = OrderedDict()
        for target in object.targetMechanisms:
            # Get the process to which the TARGET mechanism belongs:
            try:
                process = next(projection.sender.owner for
                               projection in target.inputStates[TARGET].receivesFromProjections if
                               isinstance(projection.sender, ProcessInputState))
            except StopIteration:
                raise RunError("PROGRAM ERROR: No process found for target mechanism ({}) "
                               "supposed to be in targetMechanisms for {}".
                               format(target.name, object.name))
            # Get stimuli specified for TERMINAL mechanism of process associated with TARGET mechanism
            terminal_mech = process.terminalMechanisms[0]
            try:
                ordered_targets[terminal_mech] = stimuli[terminal_mech]
            except KeyError:
                raise RunError("{} (of {} process) not found target specification for run of {}".
                               format(terminal_mech, object.name))
        stimuli = ordered_targets

    # Convert all items to 2D arrays:
    # - to match standard format of mech.variable
    # - to deal with case in which the lists have only one stimulus, one more more has length > 1,
    #     and those are specified as lists or 1D arrays (which would be misinterpreted as > 1 stimulus)

    # Check that all of the stimuli in each list are compatible with the corresponding mechanism's variable
    for mech, stim_list in stimuli.items():

        # First entry in stimulus list is a single item (possibly an item in a simple list or 1D array)
        if not isinstance(stim_list[0], Iterable):
            # If mech.variable is also of length 1
            if np.size(mech.variable) == 1:
                # Wrap each entry in a list
                for i in range(len(stim_list)):
                    stimuli[mech][i] = [stim_list[i]]
            # Length of mech.variable is > 1, so check if length of list matches it
            elif len(stim_list) == np.size(mech.variable):
                # Assume that the list consists of a single stimulus, so wrap it in list
                stimuli[mech] = [stim_list]
            else:
                raise RunError("Stimuli for {} of {} are not properly formatted ({})".
                                  format(append_type_to_name(mech),object.name))

        for stim in stimuli[mech]:
            if not iscompatible(np.atleast_2d(stim), mech.variable):
                raise RunError("Incompatible stimuli ({}) for {} ({})".
                                  format(stim, append_type_to_name(mech), mech.variable))

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

    # Otherwise, for inputs to a system, construct stimulus from dict with phases
    elif object_type is SYSTEM:
        for execution in range(num_input_sets):
            stimuli_in_execution = []
            for phase in range(object.numPhases):
                stimuli_in_phase = []
                # Only assign inputs to originMechanisms
                #    and assign them in the order they appear in originMechanisms and fill out each phase
                for mech, runtime_params, phase_spec in object.originMechanisms.mech_tuples:
                    # Assign input elements to stimulus if phase is correct one for mech
                    if phase == phase_spec:
                        # Get stimulus for mech for current execution, and enforce 2d to accomodate inputStates per mech
                        stimulus = np.atleast_2d(stimuli[mech][execution])
                        if not isinstance(stimulus, Iterable):
                            stimulus = np.atleast_2d([stimulus])
                    # Otherwise, pad stimulus for this phase with zeros
                    else:
                        if not isinstance(stimuli[mech][execution], Iterable):
                            stimulus = np.atleast_2d(np.zeros(1))
                        else:
                            stimulus = np.atleast_2d(np.zeros(len(stimuli[mech][execution])))
                    stimuli_in_phase.append(stimulus)

                stimuli_in_execution.append(stimuli_in_phase)
            stim_list.append(stimuli_in_execution)

    else:
        raise RunError("PROGRAM ERROR: illegal type for run ({}); should have been caught by _get_obect_type ".
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
            axis 1: inputs for each time step of a trial (len == _phaseSpecMax of system (no. of time_steps per trial)
            axis 2: inputs to the system, one for each process (len == number of processes in system)

    returns number of input_sets (one per execution)
    """

    object_type = _get_obect_type(object)

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
            mech_len = len(object.firstMechanism.variable)
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
            num_states_in_first_mech = len(object.originMechanisms[0].inputStates)
            if any(len(mech.inputStates) != num_states_in_first_mech for mech in object.originMechanisms):
                states_per_mech_heterog = True
            # Determine whether the size of all states is homogenous
            size_of_first_state = len(list(object.originMechanisms[0].inputStates.values())[0].value)
            for origin_mech in object.originMechanisms:
                if any(len(state.value) != size_of_first_state for state in origin_mech.inputStates.values()):
                    size_of_states_heterog = True
        else:
            raise RunError("Unknown data type for inputs in {}".format(object.name))

        if is_target:   # No phase dimension, so one less than for stimulus inputs
            # If targets are homogeneous, inputs.ndim should be 4:
            # If targets are heterogenous:
            #   if states/mech are homogenous, inputs.ndim should be 3
            #   if states/mech are heterogenous, inputs.ndim should be 2
            expected_dim = 2 + input_homogenity + states_per_mech_heterog
        else: # Stimuli, which have phases, so one extra dimension
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
                raise RunError("PROGRAM ERROR: Unexepcted shape of intputs: {}".format(inputs.shape))

        if inputs.ndim != expected_dim:
            raise RunError("inputs arg in call to {}.run() must be a {}d np.array or comparable list".
                              format(object.name, expected_dim))

        if np.size(inputs,PROCESSES_DIM) != len(object.originMechanisms):
            raise RunError("The number of inputs for each execution ({}) in the call to {}.run() "
                              "does not match the number of processes in the system ({})".
                              format(np.size(inputs,PROCESSES_DIM),
                                     object.name,
                                     len(object.originMechanisms)))

        # Check that length of each input matches length of corresponding origin mechanism over all executions and phases
        if is_target:
            mechs = list(object.targetMechanisms)
        else:
            mechs = list(object.originMechanisms)
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
                    if len(input_for_mech) != len(mechs[mech_num].inputValue):
                        raise RunError("Number of states ({}) in input for {} should be {}".
                                       format(len(input_for_mech),
                                              mechs[mech_num].name,
                                              len(mechs[mech_num].inputValue)))
                    for state_num in range(len(input_for_mech)):
                        input_for_state = mechs[mech_num].inputValue[state_num]
                        if len(input_for_state) != len(mechs[mech_num].inputValue[state_num]):
                            raise RunError("Length of state {} ({}) in input for {} should be {}".
                                           format(list(mechs[mech_num].inputStates)[state_num],
                                                  len(input_for_state),
                                                  mechs[mech_num].name,
                                                  len(mechs[mech_num].inputValue[state_num])))
        return num_execution_sets

    else:
        raise RunError("PROGRAM ERRROR: {} type not currently supported by _validate_inputs in Run module for ".
                       format(object.__class__.__name__))

def _validate_targets(object, targets, num_input_sets, context=None):
    """
    num_targets = number of target stimuli per execution
    num_targets_sets = number sets of targets (one for each execution) in targets;  must match num_input_sets
    """

    object_type = _get_obect_type(object)
    num_target_sets = None

    if isinstance(targets, function_type):
        # Check that function returns a number of items equal to the number of target mechanisms
        generated_targets = targets()
        num_targets = len(generated_targets)
        num_target_mechs = len(object.targetMechanisms)
        if num_targets != num_target_mechs:
            raise RunError("function for target argument of run returns {} items "
                           "but {} has {} targets".
                           format(num_targets, object.name, num_target_mechs))

        # Check that each target generated is compatible with the targetMechanism for which it is intended
        for target, targetMechanism in zip(generated_targets, object.targetMechanisms):
            target_len = np.size(target)
            if target_len != np.size(targetMechanism.inputStates[TARGET].variable):
                if num_target_sets > 1:
                    plural = 's'
                else:
                    plural = ''
                raise RunError("Length ({}) of target{} specified for run of {}"
                                   " does not match expected target length of {}".
                                   format(target_len, plural, append_type_to_name(object),
                                          np.size(object.targetMechanism.target)))
        return

    if object_type is PROCESS:

        # If learning is enabled, validate target
        if object._learning_enabled:
            target_array = np.atleast_2d(targets)
            target_len = np.size(target_array[0])
            num_target_sets = np.size(target_array, 0)

            if target_len != np.size(object.targetMechanism.inputStates[TARGET].variable):
                if num_target_sets > 1:
                    plural = 's'
                else:
                    plural = ''
                raise RunError("Length ({}) of target{} specified for run of {}"
                                   " does not match expected target length of {}".
                                   format(target_len, plural, append_type_to_name(object),
                                          np.size(object.targetMechanism.target)))

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
            if num_targets_per_set != len(object.targetMechanisms):
                raise RunError("The number of target values for each execution ({}) in the call to {}.run() "
                                  "does not match the number of processes in the system ({})".
                                  format(
                                         # np.size(targets,PROCESSES_DIM),
                                         num_targets_per_set,
                                         object.name,
                                         len(object.originMechanisms)))

            # MODIFIED 12/23/16 NEW:
            # Validate that each target is compatible with its corresponding targetMechanism
            # FIX: CONSOLIDATE WITH TESTS FOR PROCESS AND FOR function_type ABOVE
            # FIX: MAKE SURE THAT ITEMS IN targets ARE ALIGNED WITH CORRESPONDING object.targetMechanisms
            target_array = np.atleast_2d(targets)

            for target, targetMechanism in zip(targets, object.targetMechanisms):
                target_len = np.size(target)
                if target_len != np.size(targetMechanism.inputStates[TARGET].variable):
                    if num_targets_per_set > 1:
                        plural = 's'
                    else:
                        plural = ''
                    raise RunError("Length ({}) of target{} specified for run of {}"
                                       " does not match expected target length of {}".
                                       format(target_len, plural, append_type_to_name(object),
                                              np.size(targetMechanism.inputStates[TARGET].variable)))

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

def _get_obect_type(object):
    if isinstance(object, Mechanism):
        return MECHANISM
    elif isinstance(object, Process):
        return PROCESS
    elif isinstance(object, System):
        return SYSTEM
    else:
        raise RunError("{} type not supported by Run module".format(object.__class__.__name__))
    

import uuid
def _get_get_execution_id():
    return uuid.uuid4()
