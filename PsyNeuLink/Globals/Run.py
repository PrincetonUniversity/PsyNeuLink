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

The ``run`` function is used for executing a mechanism, process or system.  It can be called directly, however,
it is typically invoked by calling the ``run`` method of the object to be run.  It  executes an object by calling the
object's ``execute`` method.  While an object's execute method can be called directy, using ``run`` is much easier
because it:

    * allows multiple rounds of execution to be run in sequence; the ``execute`` method of an object
      runs only a single execution of the object);

    * uses simpler formats for specifying inputs (see below);

    * manages timing factors (such as updating the ``CentralClock`` and presenting inputs at the correct time
      for execution (see below).

    * automatically aggregates results across executions and stores these in the results attribute of the object.

COMMENT:
Note:: The ``run`` function uses the ``construct_input`` function to convert the input into the format required by
``execute`` methods.
COMMENT

There are a few concepts to understand that will help in using the ``run`` function.  These are discussed below.

.. _Run_Timing:

Timing
~~~~~~

When ``run`` is called, it executes objects within the scope of execution.  For a mechanism [LINK], the scope of 
execution is simply that mechanism.  For a process [LINK], the scope of execution is all of the mechanisms specified 
in its ``pathway`` [LINK] parameter.  For a system [LINK], the scope of execution is all of the mechanisms in the 
processes specified in the system's ``processes`` parameter.

PsyNeuLink supports two time scales for executing objects:``time_step`` and ``trial``.  Every mechanism in
PsyNeuLink defines its execution at one or both of these time scales, and its current mode of execution can be set
using its ``TimeScale`` [LINK] parameter.

* **time_step**:  this mode of execution is a mechanism's closest approximation to continuous, or "real time"
  processing.  Execution of a time_step is defined as a single execution of all objects in the scope of
  execution for at their time_step time scale.  Mechanisms called upon to execute a time_step that do not support
  that time scale of execution have the option of generating an exception, being ignored, or providing their trial
  mode response, either on the first time_step, every time_step, or the last time_step in the sequence being run
  (see Timing [LINK] for further details).

* **trial**: this mode of execution is defined as the "ballistic" execution of a mechanism to a state that would
  have been achieved with time_step execution to a specified criterion.  The criterion can be specified in terms of
  the number of time_steps, or a condition to be met by the mechanism's output.  It is up to the mechanism how it
  implements its trial mode of execution (i.e., whether this is done by internal numerical iteration or an analytic
  calculation). A trial of execution is defined as the execution of a trial of all of the objects in the scope of
  execution.

The TimeScale argument to ``run`` determines the time scale for each round of execution: a time_step or a trial.
When a process is run, each  mechanism is executed in the order that it appears in the ``pathway`` parameter of the
process, once per round of execution.  When a system is run, the order of execution is determined by the system's `
`executionList``, which is based on a graph analysis of the system's processes that determines the dependencies
among its mechanisms (both within and  between processes). Execution of the mechanisms in a system also depends on
the ``phaseSpec`` of each mechanism: this determines *when* in an execution sequence it should be run.  The
``CentralClock``[LINK] is used to control timing, so executing a system requires that the ``CentralClock``
be appropriately updated.

The ``run`` function handles all of the above factors automatically.


.. _Run_Inputs:

Inputs
~~~~~~

COMMENT:
    OUT-TAKES
    The inputs for a single execution must contain a value for each inputState [LINK] of each :keyword:`ORIGIN`
    mechanism [LINK] in the process or system, using the same format used for the format of the input for the execute
    method of a process or system.  This can be specified as a nested set of lists, or an ndarray.  The exact structure
    is determined by a number of factors, as described below.
    the number of :keyword:`ORIGIN` mechanisms involved (a process has only one, but a system can have several), the
    number of inputStates for each :keyword:`ORIGIN` mechanism, and whether the input to those inputStates is
    single-element (such as scalars), multi-element (such as vectors) or a mix.  For the run method, the structure is
    further determined by whether only a single execution or multiple executions is specified.  Rather than specifying a
    single format structure that must be used for all purposes (which would necessarily be the most complex one),
    PsyNeuLink is designed to be flexible, allowing use of the simplest structure necessary to describe the input for a
    particular process or input, which can vary according to circumstance.  Examples are provided below.  In all cases,
    either nested lists or ndarrays can be used, in which the innermost level (highest axis of an ndarray) is used to
    specify the input values for a given inputState (if any are multi-element), the next nested level (second highest
    axis) is used to specify the different inputStates of a given mechanism (if any have more than one), the level
    (axis) after that is used to specify the different :keyword:`ORIGIN` mechanisms (if there is more than one), and
    finally the outermost level (lowest axis) is used to specify different trials (if there is more than one to be run).

    PsyNeuLink affords flexibility of input format that PsyNeuLink allows, the structure of the input can vary
    (i.e., the levels of nesting of the list, or dimensionality and shape of the ndarray used to specify it).
    The run function handles all of these formats seamlessly, so that whathever notation is simplest and easiest
    for a given purpose can be used (though, as noted above, it is best to consistently specify the input value of
    an inputstae as a list or array (axis of an ndarray).
COMMENT

The primary purpose of the ``run`` function is to present the inputs for each round of execution to the inputStates of
the relevant mechanisms.  The ``inputs`` argument is used to specify those input values.  For a mechanism, this is the
input values for each of the mechanism's inputStates.  For a process or system, it assigns the inputs to the
inputState(s) of the :keyword:`ORIGIN` mechanism(s) [LINK].  Input values can be specified either as a nested list or
an ndarray. There are four factors that determine the levels of nesting for a list, or the dimensionality
(number of axes) for an ndarray:

* **Number of rounds of execution**.  If the ``inputs`` argument contains the input for more than one round of
  execution (i.e., multiple time_steps and/or trials), then the outermost level of the list, or axis 0 of the ndarray,
  is used for the rounds of execution, each item of which contains the set inputs for a given round.  Otherwise, it is
  used for the next relevant factor in the list below.

* **Number of mechanisms.** If run is used for a system, and it  has more than one :keyword:`ORIGIN` mechanism, then
  the next level of nesting of a list, or next higher axis of an ndarray, is used for the :keyword:`ORIGIN` mechanisms,
  with each item containing the inputs for a given :keyword:`ORIGIN` mechanism within a round.  This factor is not
  relevant when run is used for a single mechanism, a process (which only ever has one :keyword:`ORIGIN` mechanism),
  or a system that has only one :keyword:`ORIGIN` mechanism.

* **Number of inputStates.** In general, mechanisms have a single ("primary") inputState [LINK];  however, some types
  of mechanisms can have more than one (e.g., ComparatorMechanisms [LINK] have two: one for their ``sample_input`` and
  the other for their ``target_input``).  If any :keyword:`ORIGIN` mechanism in a process or system has more than one
  inputState, then the next level of nesting of a list, or next higher axis of an ndarray, is used for the set of
  inputStates for each mechanism.

* **Number of elements for the value of an inputState.** The input to an inputState can be a single element (e.g.,
  a scalar) or have multiple elements (e.g., a vector).  By convention, even if the input to an inputState is only a
  single element, it should nevertheless always be specified as a list or a 1d np.array (it is internally converted to
  the latter by PsyNeuLink).  PsyNeuLink can usually parse single-element inputs specified as a stand-alone value
  (e.g., as a number not in a list or ndarray).  Nevertheless, it is best to embed such inputs in a single-element
  list or a 1d array, both for clarity and to insure consistent treatment of nested lists and ndarrays.  If this
  convention is followed, then the number of elements for a given input should not affect nesting of lists or
  dimensionality (number of axes) of ndarrays of an inputs argument.

With these factors in mind, inputs can be specified in the simplest form possible (least number of nestings for a list,
or lowest dimension of an ndarray).  Inputs can be specified using one of two formats:  *execution* format or
*mechanism* format.

**Sequence format** *(List[values] or ndarray):*

    This uses a nested list or ndarray to fully specify the input for each round of execution in a sequence.
    The following provides a description of the sequeunce format for all of the combinations of the factors listed 
    above. The figure shows examples.

    *Lists:* if there is more than one round, then the outermost level of the list is used for the sequence of 
    executions.  If there is only one :keyword:`ORIGIN` mechanism and it has only one inputState (the most common 
    case), then is a single sublist is used for the input of each round.  If the :keyword:`ORIGIN` mechanism has more 
    than one inputState, then the entry for each round is a sublist of the inputStates, each entry of which is a 
    sublist containing the input for that inputState.  If there is more than one mechanism, but none have more than 
    one inputState, then a sublist is used for each mechanism in each round, within which a sublist is used for the
    input for that mechanism.  If there is more than one mechanism, and any have more than one inputState,
    then a sublist is used for each mechanism for each round, within which a sublist is used for each
    inputState of the corresponding mechanism, and inside that a sublist is used for the input for each inputState.

    *ndarray:*  axis 0 is used for the first factor (round, mechanism, inputState or input) for which there is only one
    item, axis 1 is used for the next factor for which there is only one item, and so forth.  For example, if there is
    more than one round, only one :keyword:`ORIGIN` mechanism, and that has only one inputState (the most common case),
    then axis 0 is used for round, and axis 1 for inputs per round.  In the extreme, if there are multiple rounds,
    more than one :keyword:`ORIGIN` mechanism, and more than on inputState for any of the :keyword:`ORIGIN` mechanisms,
    then axis 0 is used for rounds, axis 1 for mechanisms within rouund, axis 2 for inputStates of each mechanim, and
    axis 3 for the input to each inputState of a mechanism.  Note that if *any* mechanism being run (directly, or as
    one of the :keyword:`ORIGIN` mechanisms of a process or system) has more than one inputState, then an axis must be
    committed to inputStates, and the input to every inputState of every mechanism must be specified in that axis
    (i.e., even for those mechanisms that have a single inputState).

    .. figure:: _static/Sequence_format_input_specs_fig.*
       :alt: Example input specifications in sequence format
       :scale: 75 %
       :align: center

       Example input specifications in sequence format

**Mechanism format** *(Dict[mechanism, List[values] or ndarray]):*
    This provides a simpler format for specifying inputs.  It uses a dictionary, each entry of which is the sequence of
    inputs for a given :keyword:`ORIGIN` mechanism.  The key for each entry is the :keyword:`ORIGIN` mechanism, and the
    value contains either a list or ndarray specifying the sequence of inputs for that mechanism, one for each round
    of execution.  If a list is used, and the mechanism has more than one inputState, then a sublist is used in each 
    item of the list to specify the inputs for each of the mechanism's inputStates for that round.  If an ndarray is 
    used, axis 0 is used for the sequence of rounds. If the mechanism has a single inputState, then axis 1 is used for 
    the input for each round.  If the mechanism has multiple inputStates, then axis 1 is used for the inputStates, 
    and axis 2 is used for the input to each inputState for each round.

    .. figure:: _static/Mechanism_format_input_specs_fig.*
       :alt: Mechanism format input specification
       :align: center

       Mechanism format input specification

.. _Run_Initial_Values:

Initial Values
~~~~~~~~~~~~~~

Any mechanism that is the source (sender) of a projection that closes a processing loop in a process or system, and
that is not an :keyword:`ORIGIN` mechanism, is designated as :keyword:`INITIALIZE_CYCLE`. [LINK]  An initial value
can be assigned to such mechanisms, that will be used to initialize the process or system when it is first run.  These
values are specified in the ``initial_values`` argument of ``run``, as a dict.  The key for each entry must be a
mechanism designated as :keyword:`INITIALIZE_CYCLE`, and its value an input for the mechanism to be used as its
initial value.  The size of the input (length of the outermost level if it is a list, or axis 0 if it is an np.ndarray),
must equal the number of inputStates of the mechanism, and the size of each value must match that of the variable
for the corresponding inputState.

.. _Run_Targets:

Targets
~~~~~~~

If a process or system uses learning, then target values must be provided (in the ``targets`` argument of ``run``).
Like inputs, targets can be specified as a list or ndarray.  The size of the targets argument (length of the outermost
level if a nested list, or axis 0 if an ndarray) must equal that of the inputs argument, and the size of each target
must match that of the corresponding item of the target inputState for the monitoringMechanism of the process or system.


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
from PsyNeuLink.Components.Process import Process
from PsyNeuLink.Components.Mechanisms.Mechanism import Mechanism

class RunError(Exception):
     def __init__(object, error_value):
         object.error_value = error_value

     def __str__(object):
         return repr(object.error_value)

MECHANISM = "mechanism"
PROCESS = "process"
SYSTEM = 'system'

@tc.typecheck
def run(object,
        # inputs:tc.any(list, dict, np.ndarray),
        inputs,
        num_executions:tc.optional(int)=None,
        reset_clock:bool=True,
        initialize:bool=False,
        intial_values:tc.optional(tc.any(list, np.ndarray))=None,
        targets:tc.optional(tc.any(list, np.ndarray))=None,
        learning:tc.optional(bool)=None,
        call_before_trial:tc.optional(function_type)=None,
        call_after_trial:tc.optional(function_type)=None,
        call_before_time_step:tc.optional(function_type)=None,
        call_after_time_step:tc.optional(function_type)=None,
        time_scale:tc.optional(tc.enum)=None):
    """Run a sequence of executions for a process or system

    First, validate inputs.  Then, for each round of execution:

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

    The targets argument must be the same length as the inputs argument.

    .. note::
        * if num_executions is :keyword:`None`, a number of executions is run equal to the length of the input
          (i.e., size of axis 0)

    Arguments
    ---------

    inputs : List[input] or ndarray(input) : default default_input_value for a single execution
        the input for each execution in a sequence (see :doc:`Run` for detailed description of formatting
        requirements and options).

    reset_clock : bool : default :keyword:`True`
        if :keyword:`True`, resets ``CentralClock`` to 0 before a sequence of executions.

    initialize : bool default :keyword:`False`
        calls the ``initialize`` method of the system prior to a sequence of executions.

    initial_values : Dict[Mechanism, List[input] or np.ndarray(input)] : default :keyword:`None`
        the initial values for mechanisms designated as :keyword:`INITIALIZE_CYCLE` [LINK].

    targets : List[input] or np.ndarray(input) : default :keyword:`None`
        the target values for monitoring mechanisms for each execution (used for learning).
        The length must be equal to inputs.

    learning : bool :  default :keyword:`None`
        enables or disables learning during execution.
        If it is not specified, current state is left intact.
        If :keyword:`True`, learning is forced on; if :keyword:`False`, learning is forced off.

    call_before_trial : Function : default= :keyword:`None`
        called before each trial in the sequence is executed.

    call_after_trial : Function : default= :keyword:`None`
        called after each trial in the sequence is executed.

    call_before_time_step : Function : default= `:keyword:`None`
        called before each time_step is executed.

    call_after_time_step : Function : default= :keyword:`None`
        called after each time_step is executed.

    time_scale : TimeScale :  default TimeScale.TRIAL
        specifies whether mechanisms are executed for a single time step or a trial

    Returns
    -------

    <object>.results : List[outputState.value]
        list of the values, for each execution, of the outputStates for a mechanism run directly,
        or of the outputStates of the :keyword:`TERMINAL` mechanisms for the process or system run
    """

    inputs = _construct_inputs(object, inputs, targets)

    object_type = get_object_type(object)

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

        object.target = targets

    time_scale = time_scale or TimeScale.TRIAL

    # num_executions = num_executions or len(inputs)
    # num_executions = num_executions or np.size(inputs,(inputs.ndim-1))
    # num_executions = num_executions or np.size(inputs, 0)
    num_executions = num_executions or np.size(inputs, inputs.ndim-3)

    # SET LEARNING (if relevant)
    # FIX: THIS NEEDS TO BE DONE FOR EACH PROCESS IF THIS CALL TO run() IS FOR SYSTEM
    #      IMPLEMENT learning_enabled FOR SYSTEM, WHICH FORCES LEARNING OF PROCESSES WHEN SYSTEM EXECUTES?
    #      OR MAKE LEARNING A PARAM THAT IS PASSED IN execute
    # If learning is specified, buffer current state and set to specified state
    if not learning is None:
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

    # VALIDATE INPUTS: COMMON TO PROCESS AND SYSTEM
    # Input is empty
    if inputs is None or isinstance(inputs, np.ndarray) and not np.size(inputs):
        raise SystemError("No inputs arg for \'{}\'.run(): must be a list or np.array of stimuli)".format(object.name))

    # Input must be a list or np.array
    if not isinstance(inputs, (list, np.ndarray)):
        raise RunError("The input must be a list or np.array")

    inputs = np.array(inputs)
    inputs = np.atleast_2d(inputs)

    # Insure that all input sets have the same length
    if any(len(input_set) != len(inputs[0]) for input_set in inputs):
        raise RunError("The length of at least one input in the series is not the same as the rest")

    # Class-specific validation:
    num_inputs_sets = _validate_inputs(object=object, inputs=inputs, context="Run " + object.name)
    _validate_targets(object, targets, num_inputs_sets)

    if reset_clock:
        CentralClock.trial = 0
        CentralClock.time_step = 0

    if initialize:
        object.initialize()

    if object_type == MECHANISM:
        time_steps = 1
    else:
        time_steps = object.numPhases

    for execution in range(num_executions):

        if call_before_trial:
            call_before_trial()

        for time_step in range(time_steps):

            if call_before_time_step:
                call_before_time_step()

            input_num = execution%len(inputs)

            if object_type == PROCESS and targets:
                object.target = targets[input_num]

            result = object.execute(inputs[input_num][time_step],time_scale=time_scale)

            if call_after_time_step:
                call_after_time_step()

            CentralClock.time_step += 1

        # object.results.append(result)
        if isinstance(result, Iterable):
            result_copy = result.copy()
        else:
            result_copy = result
        object.results.append(result_copy)

        if call_after_trial:
            call_after_trial()

        CentralClock.trial += 1

    # Restore learning state
    try:
        learning_state_buffer
    except UnboundLocalError:
        pass
    else:
        object._learning_enabled = learning_state_buffer

    return object.results

@tc.typecheck
def _construct_inputs(object, inputs, targets=None):
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
       axis 0: num_executions
       axis 1: object._phaseSpecMax
       axis 2: len(object.originMechanisms)
       axis 3: len(mech.inputStates)

    Notes:
    * Construct as lists and then convert to np.array, since size of inputs can be different for different mechs
        so can't initialize a simple (regular) np.array;  this means that stim_list dtype may also be 'O'
    * Code below is not pretty, but needs to test for cases in which inputs have different sizes

    """

    object_type = get_object_type(object)

    # Stimuli in list format
    if isinstance(inputs, (list, np.ndarray)):
        stim_list = _construct_from_stimulus_list(object, inputs, targets)

    # Stimuli in dict format
    elif isinstance(inputs, dict):
        stim_list = _construct_from_stimulus_dict(object, inputs, targets)

    else:
        raise SystemError("inputs arg for {}._construct_inputs() must be a dict or list".format(object.name))

    stim_list_array = np.array(stim_list)
    return stim_list_array

def _construct_from_stimulus_list(object, stimuli, targets):

    object_type = get_object_type(object)

    # Check for header
    headers = None
    if isinstance(stimuli[0],Iterable) and any(isinstance(header, Mechanism) for header in stimuli[0]):
        headers = stimuli[0]
        del stimuli[0]
        for mech in object.originMechanisms:
            if not mech in headers:
                raise SystemError("Header is missing for origin mechanism {} in stimulus list".
                                  format(mech.name, object.name))
        for mech in headers:
            if not mech in object.originMechanisms.mechanisms:
                raise SystemError("{} in header for stimulus list is not an origin mechanism in {}".
                                  format(mech.name, object.name))

    inputs_array = np.array(stimuli)
    if inputs_array.dtype in {np.dtype('int64'),np.dtype('float64')}:
        max_dim = 2
    elif inputs_array.dtype is np.dtype('O'):
        max_dim = 1
    else:
        raise SystemError("Unknown data type for inputs in {}".format(object.name))
    while inputs_array.ndim > max_dim:
        # inputs_array = np.hstack(inputs_array)
        inputs_array = np.concatenate(inputs_array)
    inputs = inputs_array.tolist()

    num_executions = _validate_inputs_and_targets(object=object,
                                 inputs=inputs,
                                 targets=targets,
                                 num_phases=1,
                                 context='contruct_inputs for ' + object.name)

    # If inputs are for a process, no need to deal with phase so just return
    if object_type in {MECHANISM, PROCESS}:
        return inputs

    mechs = list(object.originMechanisms)
    num_mechs = len(object.originMechanisms)
    inputs_flattened = np.hstack(inputs)
    # inputs_flattened = np.concatenate(inputs)
    input_elem = 0    # Used for indexing w/o headers
    execution_offset = 0  # Used for indexing w/ headers
    stim_list = []

    for execution in range(num_executions):
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
                        # stimulus[stim_elem] = inputs_flattened[input_elem]
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

def _construct_from_stimulus_dict(object, stimuli, targets):

    object_type = get_object_type(object)

    # Validate that there is a one-to-one mapping of entries to origin mechanisms in the system
    for mech in object.originMechanisms:
        if not mech in stimuli:
            raise SystemError("Stimulus list is missing for origin mechanism {}".format(mech.name, object.name))
    for mech in stimuli.keys():
        if not mech in object.originMechanisms.mechanisms:
            raise SystemError("{} is not an origin mechanism in {}".format(mech.name, object.name))

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
                raise SystemError("Inputs for {} of {} are not properly formatted ({})".
                                  format(append_type_to_name(mech),object.name))

        for stim in stimuli[mech]:
            # # MODIFIED 10/28/16 OLD:
            # if not iscompatible(stim, mech.variable):
            # MODIFIED 10/28/16 NEW:
            if not iscompatible(np.atleast_2d(stim), mech.variable):
            # MODIFIED 10/28/16 END
                raise SystemError("Incompatible input ({}) for {} ({})".
                                  format(stim, append_type_to_name(mech), mech.variable))

    stim_lists = list(stimuli.values())
    num_executions = len(stim_lists[0])

    # Check that all lists have the same number of stimuli
    if not all(len(np.array(stim_list)) == num_executions for stim_list in stim_lists):
        raise SystemError("The length of all the stimulus lists must be the same")

    stim_list = []

    # If stimuli are for a process, construct stimulus list from dict without worrying about phases
    if object_type in {MECHANISM, PROCESS}:
        for i in range(num_executions):
            stims_in_execution = []
            for mech in stimuli:
                stims_in_execution.append(stimuli[mech][i])
            stim_list.append(stims_in_execution)

    # Otherwise, for system, construct stimulus from dict with phases
    else:
        for execution in range(num_executions):
            stimuli_in_execution = []
            for phase in range(object.numPhases):
                stimuli_in_phase = []
                for mech, runtime_params, phase_spec in object.originMechanisms.mech_tuples:
                    for process, status in mech.processes.items():
                        if process._isControllerProcess:
                            continue
                        if mech.systems[object] in {ORIGIN, SINGLETON}:
                            if phase == phase_spec:
                                stimulus = np.array(stimuli[mech][execution])
                                if not isinstance(stimulus, Iterable):
                                    stimulus = np.array([stimulus])
                            else:
                                if not isinstance(stimuli[mech][execution], Iterable):
                                    stimulus = np.zeros(1)
                                else:
                                    stimulus = np.zeros(len(stimuli[mech][execution]))
                        stimuli_in_phase.append(stimulus)
                stimuli_in_execution.append(stimuli_in_phase)
            stim_list.append(stimuli_in_execution)
    return stim_list

def _validate_inputs(object, inputs=None, num_phases=None, context=None):
    """Validate inputs for _construct_inputs() and object.run()

    If inputs is an np.ndarray:
        inputs must be 3D (if inputs to each process are different lengths) or 4D (if they are homogenous):
            axis 0 (outer-most): inputs for each execution of the run (len == number of executions to be run)
                (note: this is validated in super().run()
            axis 1: inputs for each time step of a trial (len == _phaseSpecMax of system (no. of time_steps per trial)
            axis 2: inputs to the system, one for each process (len == number of processes in system)

    returns number of executions implicit in inputs
    """

    object_type = get_object_type(object)

    if object_type is PROCESS:

        if isinstance(inputs, list):
            inputs = np.array(inputs)

        # If inputs to process are heterogeneous, inputs.ndim should be 2:
        if inputs.dtype is np.dtype('O') and inputs.ndim != 2:
            raise SystemError("inputs arg in call to {}.run() must be a 2D np.array or comparable list".
                              format(object.name))

        # If inputs to process are homogeneous, inputs.ndim should be 2 if length of input == 1, else 3:
        if inputs.dtype in {np.dtype('int64'),np.dtype('float64')}:
            mech_len = len(object.firstMechanism.variable)
            if not ((mech_len == 1 and inputs.ndim == 2) or inputs.ndim == 3):
                raise SystemError("inputs arg in call to {}.run() must be a 3d np.array or comparable list".
                                  format(object.name))

        return np.size(inputs, inputs.ndim-3)

    elif object_type is SYSTEM:

        num_phases = num_phases or object.numPhases

        if isinstance(inputs, np.ndarray):

            HOMOGENOUS_INPUTS = 1
            HETEROGENOUS_INPUTS = 0

            if inputs.dtype in {np.dtype('int64'),np.dtype('float64')}:
                process_structure = HOMOGENOUS_INPUTS
            elif inputs.dtype is np.dtype('O'):
                process_structure = HETEROGENOUS_INPUTS
            else:
                raise SystemError("Unknown data type for inputs in {}".format(object.name))

            # If inputs to processes of system are heterogeneous, inputs.ndim should be 3:
            # If inputs to processes of system are homogeneous, inputs.ndim should be 4:
            expected_dim = 3 + process_structure

            if inputs.ndim != expected_dim:
                raise SystemError("inputs arg in call to {}.run() must be a {}D np.array or comparable list".
                                  format(object.name, expected_dim))

            if np.size(inputs,PROCESSES_DIM) != len(object.originMechanisms):
                raise SystemError("The number of inputs for each execution ({}) in the call to {}.run() "
                                  "does not match the number of processes in the system ({})".
                                  format(np.size(inputs,PROCESSES_DIM),
                                         object.name,
                                         len(object.originMechanisms)))

        # FIX: STANDARDIZE DIMENSIONALITY SO THAT np.take CAN BE USED

        # Check that length of each input matches length of corresponding origin mechanism over all executions and phases
        # Calcluate total number of executions
        num_mechs = len(object.originMechanisms)
        mechs = list(object.originMechanisms)
        num_executions = 0
        executions_remain = True
        input_num = 0
        inputs_array = np.array(inputs)
        while executions_remain:
            try:
                for mech_num in range(num_mechs):
                    # input = inputs[input_num]
                    mech_len = np.size(mechs[mech_num].variable)
                    # FIX: WORRIED ABOUT THIS AND THE MAGIC NUMBER -2 BELOW:
                    # If inputs_array is just a list of numbers and its length equals the input to the mechanism
                    #    then there is just one input and one execution
                    if inputs_array.ndim == 1 and len(inputs) == mech_len:
                        input_num += 1
                        executions_remain = False
                        continue
                    input = np.take(inputs_array,input_num,inputs_array.ndim-2)
                    if np.size(input) != mech_len * num_phases:
                       # If size of input didn't match length of mech variable,
                       #  may be that inputs for each mech are embedded within list/array
                        if isinstance(input, Iterable):
                            inner_input_num = 0
                            for inner_input in input:
                                mech_len = np.size(mechs[inner_input_num].variable)
                                # Handles assymetric input lengths:
                                if (isinstance(inner_input, Iterable) and
                                            np.size(np.concatenate(inner_input)) != mech_len * num_phases):
                                    for item in inner_input:
                                        if np.size(item) != mech_len * num_phases:
                                            raise SystemError("Length ({}) of stimulus ({}) does not match length ({}) "
                                                              "of input for {} in execution {}".
                                                              format(len(inputs[inner_input_num]),
                                                                     inputs[inner_input_num],
                                                                     mech_len,
                                                                     append_type_to_name(mechs[inner_input_num],'mechanism'),
                                                                     num_executions))
                                        inner_input_num += 1
                                        mech_len = np.size(mechs[inner_input_num].variable)
                                elif np.size(inner_input) != mech_len * num_phases:
                                    raise SystemError("Length ({}) of stimulus ({}) does not match length ({}) "
                                                      "of input for {} in execution {}".
                                                      format(len(inputs[inner_input_num]), inputs[inner_input_num], mech_len,
                                                      append_type_to_name(mechs[inner_input_num],'mechanism'), num_executions))
                                else:
                                    inner_input_num += 1
                            input_num += 1
                            break
                    input_num += 1
                num_executions += 1
            except IndexError:
                executions_remain = False

        return num_executions

def _validate_targets(object, targets, num_executions):
    """
    num_targets = number of target stimuli per execution
    num_target_executions = number sets of targets (one for each execution) in targets;  must match num_executions
    """

    object_type = get_object_type(object)

    if object_type is PROCESS:

        # If learning is enabled, validate target
        if targets and object._learning_enabled:
            target_array = np.atleast_2d(targets)
            target_len = np.size(target_array[0])
            num_targets = np.size(target_array, 0)

            if target_len != np.size(object.comparatorMechanism.target):
                if num_targets > 1:
                    plural = 's'
                else:
                    plural = ''
                raise RunError("Length ({}) of target{} specified for run of {}"
                                   " does not match expected target length of {}".
                                   format(target_len, plural, append_type_to_name(object),
                                          np.size(object.comparatorMechanism.target)))

            if any(np.size(target) != target_len for target in target_array):
                raise RunError("Not all of the targets specified for {} are of the same length".
                                   format(append_type_to_name(object)))

            if num_targets != num_executions:
                raise RunError("Number of targets ({}) does not match number of inputs ({}) specified in run of {}".
                                   format(num_targets, num_executions, append_type_to_name(object)))

    elif object_type is SYSTEM:
        # Check that number of target values in each execution equals the number of target mechanisms in the system
        if targets and any(process._learning_enabled for process in object.processes):
            if np.size(targets,0) != len(object.targetMechanisms):
                raise SystemError("The number of target values ({}) for each execution in the call to {}.run() "
                                  "does not match the number of target mechanisms ({}) in the system".
                                  format(np.size(targets,0),
                                         object.name,
                                         len(object.targetMechanisms)))
    else:
        pass

def get_object_type(object):
    if isinstance(object, Mechanism):
        return MECHANISM
    elif isinstance(object, Process):
        return PROCESS
    elif isinstance(object, System):
        return SYSTEM
    else:
        raise RunError("{} type not supported by Run module".format(object.__class__.__name__))

