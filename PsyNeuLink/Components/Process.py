# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *********************************************  Process ***************************************************************


# *****************************************    PROCESS CLASS    ********************************************************

"""
..
    Sections:
      * :ref:`Process_Overview`
      * :ref:`Process_Creation`
      * :ref:`Process_Structure`
         * :ref:`Process_Pathway`
         * :ref:`Process_Mechanisms`
         * :ref:`Process_Projections`
         * :ref:`Process_Input_And_Output`
         * :ref:`Process_Learning`
      * :ref:`Process_Execution`
      * :ref:`Process_Class_Reference`


.. _Process_Overview:

Overview
--------

A process is a sequence of mechanisms linked by projections. Executing a process executes all of its mechanisms
in the order in which they are listed in its `pathway` attribute:  a list of `mechanisms <Mechanism> and (optional)
`projection <Projection>` specifications.  Projections can be specified among any mechanisms in a process, including
to themselves.  However, a process cannot involve any "branching" (that is, one-to-many or many-to-one projections);
that must be done using a `System`. Mechanisms in a process can also project to mechanisms in other processes,
but these will only have an effect if all of the processes involved are members of the same `system <System>`.
Projections between mechanisms can be trained, by assigning `LearningProjections <LearningProjection>` to them.
Learning can also be specified for the entire process, in which case all of the projections among mechanisms in the
process will be trained. Processes can be constructed and executed on their own.  More commonly, however, they are used
to construct a `System`.

.. _Process_Creation:

Creating a Process
------------------

A process is created by calling the :py:func:`process` function. The mechanisms to be included are specified in a list
in the `pathway` argument, in the order in which they should be executed by the process.  The mechanism entries can be
separated by `projections <Projection>` used to connect them.  If no arguments are provided to the `pathway` argument,
a process with a single :ref:`default mechanism <LINK>` is created.

.. _Process_Structure:

Structure
---------

.. _Process_Pathway:

Pathway
~~~~~~~

A process is defined primarily by its `pathway` attribute, which is a list of `mechanisms <Mechanism>` and
`projections <Projection>`.  The list defines an array of mechanisms that will executed in the order specified. Each
mechanism in the pathway must project at least to the next one in the pathway, though it can project to others and
also receive recurrent (feedback) projections from them.  However, pathways cannot be used to construct branching
patterns;  that requires the use of a  `System`.  The mechanisms specified in the `pathway` for a process are generally
`ProcessingMechanisms <ProcessingMechanism>`.  The projections between mechanisms in a process must be
`MappingProjections <MappingProjection>`.  These transmit the output of a mechanism (the projection's
`sender <MappingProjection.MappingProjection.sender>`) to the input of another mechanism (the projection's
`receiver <MappingProjection.MappingProjection.receiver>`). Specification of a `pathway` requires, at the least, a list
of mechanisms.  These can be specified directly, or in a **tuple** that also contains a set of
`runtime parameters <Mechanism_Runtime_Parameters>` and/or a `phase <System_Execution_Phase>` specification (see
`below <Process_Mechanism_Specification>`). Projections between a pair of mechanisms can be specified by
interposing
them in the list between the pair.  When no projection appears between two adjacent mechanisms in the pathway, and
there is no otherwise specified projection between them, PsyNeuLink assigns a default projection. Specifying the
components of a pathway is described in detail below.

.. _Process_Mechanisms:

Mechanisms
~~~~~~~~~~

The mechanisms of a process must be listed in the `pathway` argument of the :py:func:`process` function explicitly,
in the order to be executed.  The first mechanism in the process is designated as the `ORIGIN`, and receives as its
input any input provided to the process' `execute <Process_Base.execute>` or `run <Process_Base.run>` methods. The last
mechanism listed in the `pathway` is designated as the `TERMINAL` mechanism, and its output is assigned as the output
of the process when it is executed.

.. note::
   The `ORIGIN` and `TERMINAL` mechanisms of a process are not necessarily the `ORIGIN` and/or `TERMINAL` mechanisms
   of the `system <System_Mechanisms>` to which it belongs.  The designations of a mechanism's status in the process(es)
   to which it belongs are listed in its `processes <Mechanism.Mechanism_Base.processes>` attribute.

.. _Process_Mechanism_Specification:

Mechanisms can be specified in the `pathway` argument of :py:func:`process` in one of two ways:  directly or
in a *MechanismTuple*.  Direct  specification can use any supported format for `specifying a mechanism
<Mechanism_Creation>`. Alternatively, mechanisms can be specified using a MechanismTuple, the first item of which
is the mechanism, and the second and third (optional) items are a set of
`runtime parameters <Mechanism_Runtime_Parameters>` and a `phase <System_Execution_Phase>` specification. Runtime
parameters are used for that mechanism when the process (or a system to which it belongs) is executed; otherwise
they do not remain associated with the mechanism.  The phase is used when the mechanism is executed as part of a
system, to specify when within the trial the mechanism should be executed.  Either the runtime parameters or the phase
can be omitted from a MechanismTuple (if the phase is omitted, the default value of 0 will be assigned).

.. note::
   Irrespective of the format in which a mechanism is specified in a `pathway`, it's entry is
   converted internally to a MechanismTuple, and listed in the process' `mechanisms` attribute.

The same mechanism can appear more than once in a `pathway`, as one means of generating a recurrent processing loop
(another is to specify this in the projections -- see below).


.. _Process_Projections:

Projections
~~~~~~~~~~~

Projections between mechanisms in the `pathway` of a process are specified in one of three ways:

* Inline specification
    Projection specifications can be interposed between any two mechanisms in the `pathway` list.  This creates a
    projection from the preceding mechanism in the list to the one that follows it.  The projection specification can
    be an instance of a `MappingProjection`, the class name :keyword:`MappingProjection`, a
    `matrix keyword <Matrix_Keywords>` for a type of MappingProjection (`IDENTITY_MATRIX`, `FULL_CONNECTIVITY_MATRIX`,
    or `RANDOM_CONNECTIVITY_MATRIX`), or a dictionary with `specifications for the projection <Projection_Creation>`.

* Stand-alone projection
    When a projection is created on its own, it can be assigned :ref:`sender <MappingProjection_Sender>`
    and :ref:`receiver <MappingProjection_Receiver>` mechanisms. If both are in the process, then that
    projection will be used when creating the process.  Stand-alone specification of a projection between two
    mechanisms in a process takes precedence over default or inline specification; that is, the stand-alone
    projection will be used in place of any that is specified in the pathway. Stand-alone specification is required
    to implement projections between mechanisms that are not adjacent to one another in the `pathway` list.

* Default assignment
    For any mechanism that does not receive a projection from another mechanism in the process (specified using one of
    the methods above), a `MappingProjection` is automatically created from the mechanism that precedes it in the
    `pathway`.  If the format of the preceding mechanism's output matches that of the next mechanism, then
    `IDENTITY_MATRIX` is used for the projection;  if the formats do not match, or
    `learning has been specified <Process_Learning>` either for the projection or the process,
    then a `FULL_CONNECTIVITY_MATRIX` is used. If the mechanism is the `ORIGIN` mechanism (i.e., first in the
    `pathway`), a `ProcessInputState <Process_Input_And_Output>` will be used as the sender,
    and an `IDENTITY_MATRIX` is used for the projection.


.. _Process_Input_And_Ouput:

Process input and output
~~~~~~~~~~~~~~~~~~~~~~~~

The input to a process is a list or 2d np.array provided as an argument in its `execute <Process_Base.execute>`
or `run <Process_Base.run>` methods, and assigned to its :py:data:`input <Process_Base.input>` attribute.
When a process is created, a set of `ProcessInputStates <processInputStates>` and `MappingProjections
<MappingProjection>` are automatically generated to transmit the process' input to its `ORIGIN` mechanism, as follows:

* if the number of items in the `input` is the same as the number of `ORIGIN` input_states:
    a MappingProjection is created for each value of the input to an inputState of the `ORIGIN` mechanism;

* if the `input` has only one item but the `ORIGIN` mechanism has more than one inputState:
    a single ProcessInputState is created with projections to each of the `ORIGIN` mechanism input_states;

* if the `input` has more than one item but the `ORIGIN` mechanism has only one inputState:
    a ProcessInputState is created for each input item, and all project to the `ORIGIN` mechanism's inputState;

* otherwise, if both the `input` and `ORIGIN` mechanism have more than one inputState, but the numbers are not equal:
    an error message is generated indicating that the there is an ambiguous mapping from the Process'
    input value to `ORIGIN` mechanism's input_states.

The output of a process is a 2d np.array containing the values of its `TERMINAL` mechanism's outputStates.

.. _Process_Learning:

Learning
~~~~~~~~

Learning modifies projections between mechanisms in a process's `pathway`, so that the input to each projection`s
`sender <MappingProjection_Sender>` produces the desired ("target") output from its
`receiver <MappingProjection_Receiver>`.  Learning occurs when a projection or process for which learning has been
specified is executed.  Learning can be specified for a particular projection in a process, or for the entire
process. It is specified for a particular projection by including a `LearningProjection specification
<LearningProjection_Creation>` in the specification for the projection.  It is specified for the entire process by
assigning either a `LearningProjection` specification, or the keyword `LEARNING` to the `learning` argument of the
process` constructor.  Specifying learning for a process will implement it for all eligible projections in the
process (i.e., all `MappingProjections <MappingProjection>`, excluding projections from
the process' inputState to its `ORIGIN` mechanism, and projections from the `TERMINAL` mechanism to
the process' outputState). When learning is specified for the process, all projections in the process will be trained
so that input to the process (i.e., its `ORIGIN` mechanism) will generate the specified target value as its
output (i.e., the output of the `TERMINAL` mechanism). In either case, all mechanisms that receive projections for
which learning has been specified must be `compatible with learning <LearningProjection>`).

When learning is specified , the following objects are automatically created for each projection involved (see figure
below):
    * a `MonitoringMechanism` used to evaluate the output of the projection's `receiver <MappingProjection_Receiver>`
      against a target value;
    ..
    * a `MappingProjection` from the projection's `receiver <MappingProjection_Receiver>` to the MonitoringMechanism;
    ..
    * a `LearningProjection` from the MonitoringMechanism to the projection being learned.

Different learning algorithms can be specified (e.g., `Reinforcement` or `BackPropagation`), that implement the
MonitoringMechanisms and LearningSignals required for the specified type of learning. However,  as noted above,
all mechanisms that receive projections being learned must be compatible with learning.

When a process or any of its projections is specified for learning, a set of `target values <Run_Targets>`
must be provided (along with the `inputs <input>`) as an argument to the process' `execute <Process_Base.execute>` or
`run <Process_Base.run>` method.

.. _Process_Learning_Figure:

**Figure: Learning in PsyNeuLink**

.. figure:: _static/PNL_learning_fig.png
   :alt: Schematic of learning mechanisms and LearningProjections in a process

   Learning using the `BackPropagation` learning algorithm in a three-layered network, using a `TransferMechanism` for
   each layer.

.. _Process_Execution:

Execution
---------

A process can be executed as part of a `system <System>` or on its own.  On its own, it can be executed by calling
either its `execute <Process_Base.execute>` or `run <Process_Base.run>` methods.  When a process is
executed, its `input` is conveyed to the `ORIGIN` mechanism (first mechanism in the pathway).  By default,
the the input value is presented only once.  If the `ORIGIN` mechanism is executed again in the same round of execution
(e.g., if it appears again in the pathway, or receives recurrent projections), the input is not presented again.
However, the input can be "clamped" on using the `clamp_input` argument of `execute <Process_Base.execute>` or
`run <Process_Base.run>`.  After the `ORIGIN` mechanism is executed, each subsequent mechanism in the `pathway` is
executed in sequence (irrespective of any `phase` specification).  If a mechanism is specified in the pathway in a
`MechanismTuple <Process_Mechanism_Specification>`, then the runtime parameters are applied and the mechanism is
executed using them (see `Mechanism` for parameter specification).  Finally the output of the `TERMINAL` mechanism
(last one in the pathway) is assigned as the output of the process.  If `learning <Process_Learning>` has been
specified for the process or any of the projections in its `pathway`, then the relevant learning mechanisms are
executed. These calculate changes that will be made to the corresponding projections.

.. note::
   The changes to a projection induced by learning are not applied until the mechanisms that receive those
   projections are next executed; see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

Examples
--------

*Specification of mechanisms in a pathway:*  The first mechanism is specified as a reference to an instance,
the second as a default instance of a mechanism type, and the third in MechanismTuple format (specifying a reference
to a mechanism that should receive some_params at runtime; note: the phase is omitted and so will be assigned the
default value of 0)::

    mechanism_1 = TransferMechanism()
    mechanism_2 = DDM()
    some_params = {PARAMETER_STATE_PARAMS:{THRESHOLD:2,NOISE:0.1}}
    my_process = process(pathway=[mechanism_1, TransferMechanism, (mechanism_2, some_params)])

*Default projection specification:*  The `pathway` for this process uses default projection specifications; as a
result, a `MappingProjection` is automatically instantiated between each of the mechanisms listed::

    my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])


*Inline projection specification using an existing projection:*  In this `pathway`, ``projection_A`` is specified as
the projection between the first and second mechanisms; a default projection will be created between ``mechanism_2``
and ``mechanism_3``::

    projection_A = MappingProjection()
    my_process = process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])

*Inline projection specification using a keyword:*  In this `pathway`, a `RANDOM_CONNECTIVITY_MATRIX <Matrix_Keywords>`
is assigned as the projection between the first and second mechanisms::

    my_process = process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])

*Stand-alone projection specification:*  In this `pathway`, ``projection_A`` is explicilty specified as a projection
between ``mechanism_1`` and ``mechanism_2``, and so will be used as the projection between them in ``my_process``;
a default projection will be created between ``mechanism_2`` and ``mechanism_3``::

    projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
    my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])

*Process that implements learning:*  This `pathway` implements a series of mechanisms with projections between them,
all of which will be learned using `BackPropagation` (the default learning algorithm).  Note that it uses the `Logistic`
function, which is compatible with BackPropagation::

    mechanism_1 = TransferMechanism(function=Logistic)
    mechanism_2 = TransferMechanism(function=Logistic)
    mechanism_3 = TransferMechanism(function=Logistic)

.. XXX USE EXAMPLE BELOW THAT CORRESPONDS TO CURRENT FUNCTIONALITY (WHETHER TARGET MUST BE SPECIFIED)
    # my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3],
    #                      learning=LEARNING)
    my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3],
                         learning=LEARNING,
                         target=[0])

.. ADD EXAMPLE HERE WHEN FUNCTIONALITY IS AVAILABLE
   *Process with individual projections that implement learning:*

    mechanism_1 = TransferMechanism(function=Logistic)
    mechanism_2 = TransferMechanism(function=Logistic)
    mechanism_3 = TransferMechanism(function=Logistic)
    # my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3],
    #                      learning=LEARNING)



COMMENT:
    Module Contents
        process() factory method:  instantiate process
        Process_Base: class definition
        ProcessInputState: class definition
        ProcessList: class definition
COMMENT

.. _Process_Class_Reference:

Class Reference
---------------

"""

import math
import re

import PsyNeuLink.Components
import PsyNeuLink.Components
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import LearningMechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import LearningMechanism
from PsyNeuLink.Components.Mechanisms.Mechanism import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import \
    ObjectiveMechanism
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection, \
    _is_learning_spec
from PsyNeuLink.Components.Projections.Projection import _is_projection_spec, _is_projection_subclass, \
    _add_projection_to
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.States.ParameterState import ParameterState
from PsyNeuLink.Components.States.State import _instantiate_state_list, _instantiate_state
from PsyNeuLink.Globals.Registry import register_category

# *****************************************    PROCESS CLASS    ********************************************************

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

DEFAULT_PHASE_SPEC = 0

# FIX: NOT WORKING WHEN ACCESSED AS DEFAULT:
DEFAULT_PROJECTION_MATRIX = AUTO_ASSIGN_MATRIX
# DEFAULT_PROJECTION_MATRIX = IDENTITY_MATRIX

ProcessRegistry = {}


class ProcessError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


# Process factory method:
@tc.typecheck
def process(process_spec=None,
            default_input_value=None,
            pathway=None,
            initial_values:dict={},
            clamp_input:tc.optional(tc.enum(SOFT_CLAMP, HARD_CLAMP))=None,
            default_projection_matrix=DEFAULT_PROJECTION_MATRIX,
            learning:tc.optional(_is_learning_spec)=None,
            learning_rate:tc.optional(parameter_spec)=None,
            target=None,
            params=None,
            name=None,
            prefs:is_pref_set=None,
            context=None):

    """
    process(                                                \
    process_spec=None,                                      \
    default_input_value=None,                               \
    pathway=None,                                           \
    initial_values=None,                                    \
    clamp_input=None,                                       \
    default_projection_matrix=None,                         \
    learning=None,                                          \
    learning_rate=None                                      \
    target=None,                                            \
    params=None,                                            \
    name=None,                                              \
    prefs=None)

    Factory method for Process: returns an instance of Process.  If called with no arguments, returns an instance of
    Process with a single ref:`DefaultMechanism <LINK>`.  See `Process` for class description.

    COMMENT:
       REPLACE DefaultMechanism BELOW USING Inline markup
    COMMENT

    Arguments
    ---------

    process_spec : Optional[str or Dict[param keyword, param value]]
        specification for the process to create.
        If it is `None`, returns an instance of Process with a single :ref:`DefaultMechanism <LINK>`;
        if it is a string, uses it as the name for the process;
        if it is a dict, the key for each entry must be a parameter name, and its value the value to assign to that
        parameter (these will be used to instantiate the process, and will override any values assigned
        to the arguments in the call to :py:func:`process`). If a name is not specified, the nth instance created
        will be named by using the process' `componentType <Process_Base.componentType>` attribute as the base and
        adding an indexed suffix: componentType-n.

    default_input_value : List[values] or ndarray :  default default input value of ORIGIN mechanism
        the input to the process used if none is provided in a call to the `execute <Process_Base.execute>` or `run
        <Process_Base.run>` method. This must be the same length as the `ORIGIN` mechanism's input.

    pathway : List[mechanism spec[, projection spec], mechanism spec...] : default List[DefaultMechanism]
        the set of mechanisms and projections between them to execute when the process is executed.  Each mechanism
        must a `ProcessingMechanism <ProcessingMechanism>`.  The specification for each can be an instance,
        a class name (creates a default instance), or a `specification dictionary <Mechanism_Creation>`.  Each
        projection must be a `MappingProjection`. The specification for each can be the class name (creates a
        default instance), an instance, or a `specification dictionary <Projection_Creation>`.

    initial_values : Optional[Dict[mechanism, param value]] : default None
        a dictionary of values used to initialize the specified mechanisms. The key for each entry must be a mechanism
        object, and the value must be a number, list or np.array that must be compatible with the format of
        the mechanism's `value <Mechanism.Mechanism_Base.value>` attribute. Mechanisms not specified will be initialized
        with their `default_input_value <Mechanism.Mechanism_Base.default_input_value>`.

    clamp_input : Optional[keyword] : default None
        specifies whether the input to the process continues to be applied to the `ORIGIN` mechanism after
        its initial execution.  The following keywords can be used:
        ..
            * `None`: Process input is used only for the first execution of the `ORIGIN` mechanism
              in a round of executions.

            * SOFT_CLAMP: combines the process' input with input from any other projections to the
              `ORIGIN` mechanism every time it is executed in a round of executions.

            * HARD_CLAMP: applies the process' input in place of any other sources of input to the
              `ORIGIN` mechanism every time it is executed in a round of executions.

    default_projection_matrix : keyword, list or ndarray : default DEFAULT_PROJECTION_MATRIX,
        the type of matrix used for default projections (see `matrix` parameter for `MappingProjection`).

    learning : Optional[LearningProjection spec]
        implements `learning <LearningProjection_CreationLearningSignal>` for all eligible projections in the process.

    learning_rate : float : None
        set the learning rate for all mechanisms in the process (see `learning_rate` attribute for additional
        information).

    target : List or ndarray : default ndarray of zeroes
        the value assigned as the TARGET input for the `ObjectiveMechanism` to which the `TERMINAL` mechanism of the
        process projects (and assigned as its `TARGET` mechanism; used for `learning <Process_Learning>`).
        It must be the same length as the `TERMINAL` mechanism's output.

    params : Optional[Dict[param keyword, param value]
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can include any of the parameters above;
        the parameter's name is used as the keyword for its entry. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default Process-<index>
        a string used for the name of the process
        (see Registry module for conventions used in naming, including for default and duplicate names)

    prefs : PreferenceSet or specification dict : Process.classPreferences
        the `PreferenceSet` for process (see ComponentPreferenceSet module for specification of PreferenceSet)

    COMMENT:
    context : str : default ''None''
           string used for contextualization of instantiation, hierarchical calls, executions, etc.
    COMMENT

    Returns
    -------
    instance of Process : Process

    """

    # MODIFIED 9/20/16 NEW:  REPLACED IN ARG ABOVE WITH None
    pathway = pathway or [Mechanism_Base.defaultMechanism]
    # MODIFIED 9/20/16 END

    # # Called with a keyword
    # if process_spec in ProcessRegistry:
    #     return ProcessRegistry[process_spec].processSubclass(params=params, context=context)
    #
    # Called with a string that is not in the Registry, so return default type with the name specified by the string
    if isinstance(process_spec, str):
        return Process_Base(name=process_spec, params=params, context=context)

    # Called with Process specification dict (with type and params as entries within it), so:
    #    - return a Process instantiated using args passed in process_spec
    elif isinstance(process_spec, dict):
        return Process_Base(context=context, **process_spec)

    # Called without a specification, so return Process with default mechanism
    elif process_spec is None:
        return Process_Base(default_input_value=default_input_value,
                            pathway=pathway,
                            initial_values=initial_values,
                            clamp_input=clamp_input,
                            default_projection_matrix=default_projection_matrix,
                            learning=learning,
                            learning_rate=learning_rate,
                            target=target,
                            params=params,
                            name=name,
                            prefs=prefs,
                            context=context)

    # Can't be anything else, so return empty
    else:
        return None


kwProcessInputState = 'ProcessInputState'
kwTarget = 'target'
from PsyNeuLink.Components.States.OutputState import OutputState

# DOCUMENT:  HOW DO MULTIPLE PROCESS INPUTS RELATE TO # OF INPUTSTATES IN FIRST MECHANISM
#            WHAT HAPPENS IF LENGTH OF INPUT TO PROCESS DOESN'T MATCH LENGTH OF VARIABLE FOR FIRST MECHANISM??


class Process_Base(Process):
    """
    Process_Base(process_spec=None,                         \
    default_input_value=None,                               \
    pathway=None,                                           \
    initial_values={},                                      \
    clamp_input:=None,                                      \
    default_projection_matrix=DEFAULT_PROJECTION_MATRIX,    \
    learning=None,                                          \
    learning_rate=None                                      \
    target=None,                                            \
    params=None,                                            \
    name=None,                                              \
    prefs=None,                                             \
    context=None)

    Abstract class for Process.

    .. note::
       Processes should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the :class:`process` factory method (see it for description of parameters).

    COMMENT:
        Description
        -----------
            Process is a Category of the Component class.
            It implements a Process that is used to execute a sequence of mechanisms connected by projections.
            NOTES:
                * if no pathway or time_scale is provided:
                    a single mechanism of Mechanism class default mechanism and TRIAL are used
                * process.input is set to the inputState.value of the first mechanism in the pathway
                * process.output is set to the outputState.value of the last mechanism in the pathway

        Class attributes
        ----------------
        componentCategory : str : default kwProcessFunctionCategory
        className : str : default kwProcessFunctionCategory
        suffix : str : default "<kwMechanismFunctionCategory>"
        registry : dict : default ProcessRegistry
        classPreference : PreferenceSet : default ProcessPreferenceSet instantiated in __init__()
        classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + variableClassDefault = inputValueSystemDefault                     # Used as default input value to Process)
        + paramClassDefaults = {PATHWAY: [Mechanism_Base.defaultMechanism],
                                TIME_SCALE: TimeScale.TRIAL}

        Class methods
        -------------
            - execute(input, control_signal_allocations, time_scale):
                executes the process by calling execute_functions of the mechanisms (in order) in the pathway list
                assigns input to sender.output (and passed through mapping) of first mechanism in the pathway list
                assigns output of last mechanism in the pathway list to self.output
                returns output after either one time_step or the full trial (determined by time_scale)
            - register_process(): registers process with ProcessRegistry
            [TBI: - adjust(control_signal_allocations=NotImplemented):
                modifies the control_signal_allocations while the process is executing;
                calling it without control_signal_allocations functions like interrogate
                returns (responseState, accuracy)
            [TBI: - interrogate(): returns (responseState, accuracy)
            [TBI: - terminate(): terminates the process and returns output
            [TBI: - accuracy(target):
                a function that uses target together with the pathway's output.value(s)
                and its accuracyFunction to return an accuracy measure;
                the target must be in a pathway-appropriate format (checked with call)

        ProcessRegistry
        ---------------
            All Processes are registered in ProcessRegistry, which maintains a dict for the subclass,
              a count for all instances of it, and a dictionary of those instances
    COMMENT

    Attributes
    ----------

    componentType : "Process"

    pathway : List[(Mechanism, dict, int), (projection, LearningProjection spec, None), (Mechanism, dict, int)...]
        specifies the list of mechanisms that are executed (in the order specified) when the process executes.
        Entries are alternating tuples specifying mechanisms and projections.  For mechanism tuples, the dict specifies
        a set of `runtime parameters <Mechanism_Runtime_Parameters>` to use for execution of the mechanism,
        and the int specifies the `phase <System_Execution_Phase>` in which the mechanism should be executed when the
        process to which it belongs is executed by a system. For projection tuples, the LearningProjection
        specification can be a `LearningProjection` object, the class or the `LEARNING_PROJECTION` keyword (which
        specifies a default instance) or the constructor for a LearningProjection (including parameters).  The second
        and third items of mechanism tuples, and the second item of projection tuples are optional and therefore may
        be `None`. The third item of projection tuples is currently not used and is always `None`.

        .. note::
             The value of this attribute is constructed from the `pathway` argument of the :py:func:`process`
             function, the entries of which do not necessarily have to have all items in a tuple, or even be in tuple
             form.  All entries of the `pathway` argument are converted to tuples when assigned to the `pathway`
             attribute.  Entries that are not tuples must be a mechanism or projection.  For tuple entries,
             the first item must be a mechanism or projection;  the second is optional, and `None` is entered for
             missing values; the third item is optional for mechanism tuples (0 is the default) and is currently
             ignored for projection tuples (and assigned `None`, as it is not currently in use).

    processInputStates : Optional[List[ProcessInputState]]
        used to represent the input to the process, and transmit this to the inputState(s) of its `ORIGIN`
        mechanism.  Each processInputState sends a MappingProjection to one or more input_states of the
        `ORIGIN` mechanism.

    input :  Optional[List[value] or ndarray]
        input to the process on each round of execution;  it is assigned the value of the :keyword:`input` argument
        in a call to the process` `execute <Process_Base.execute>`  or `run <Process_Base.run>` method. Its items are
        assigned as the value of the corresponding ProcessInputStates in `processInputStates`, and must match the
        format of the `variable <Mechanism.Mechanism_Base.variable>` for the process' `ORIGIN` mechanism.

        .. note:: The :keyword:`input` attribute of a process preserves its value throughout the execution of the
                  process. It's value is assigned to the `variable <Mechanism.Mechanism_Base.variable>` attribute of
                  the `ORIGIN` mechanism at the start of execution.  After that, by default, that mechanism's
                  :keyword:`variable` attribute is zeroed. This is so that if the `ORIGIN` mechanism is executed
                  again in the same round of execution (e.g., if it is part of a recurrent loop) it does not continue
                  to receive the initial input to the Process.  However, this behavior can be modified with the process'
                  `clamp_input` attribute.

    input_value :  2d np.array : default ``variableInstanceDefault``
        same as the :keyword:`variable` attribute of the process; contains the values of the ProcessInputStates in its
        `processInputStates` attribute.

    clamp_input : Optional[keyword]
        determines whether the process' input continues to be applied to the `ORIGIN` mechanism if it is executed again
        within the same round of execution.  It can tae the following values:

        * `None`: applies the process' `input` to the `ORIGIN` mechanism only once (the first time it is executed)
          in a given round of the process' execution.

        * `SOFT_CLAMP`: combines the process' `input` with input from any other projections to the
          `ORIGIN` mechanism every time the latter is executed within a round of the process' execution.

        * `HARD_CLAMP`: applies the process' `input` to the `ORIGIN` mechanism in place of any other sources of input
          every time it is executed.

    initial_values : Optional[Dict[mechanism, param value]]
        a dictionary of values used to initialize the specified mechanisms. The key for each entry is a mechanism
        object, and the value is a number, list or np.array that must be compatible with the format of
        the mechanism's `value <Mechanism.Mechanism_Base.value>` attribute. Mechanisms not specified will be
        initialized with their `default_input_value <Mechanism.Mechanism_Base.default_input_value>`.

    value: 2d. np.array
        the value of the `primary outputState <OutputState_Primary>` of the `TERMINAL` mechanism of the process.

    outputState : State
        the `primary outputState <OutputState_Primary>` of the `TERMINAL` mechanism in the process.

      .. _mechs : List[MechanismTuple]
             :class:`MechanismTuple` for all mechanisms in the process, listed in the order specified in pathway.
             MechanismTuples are of the form: (mechanism, runtime_params, phase) where runtime_params is dictionary
             of {argument keyword: argument values} entries and phase is an int.
             Note:  the list includes monitoring mechanisms (used for learning).

      .. _allMechanisms : MechanismList
             Contains all mechanisms in the system (based on _mechs).

      .. _origin_mechs : List[MechanismTuple]
             Contains a tuple for the `ORIGIN` mechanism of the process.
             (Note:  the use of a list is for compatibility with the MechanismList object)

      .. _terminal_mechs : List[MechanismTuple]
             Contains a tuple for the `TERMINAL` mechanism of the process.
             (Note:  the use of a list is for compatibility with the MechanismList object)

      .. _target_mechs : List[MechanismTuple]
             Contains a tuple for the `TARGET` mechanism of the process.
             (Note:  the use of a list is for compatibility with the MechanismList object)

      .. _monitoring_mechs : List[MechanismTuple]
             `MechanismTuples <Mechanism.MechanismTuples>` for all `MonitoringMechanism <MonitoringMechanisms>` in the
             process (used for learning).

      .. mechanisms : List[Mechanism]
             List of all mechanisms in the process.
             property that points to _allMechanisms.mechanisms (see below).

    mechanismNames : List[str]
        the names of the mechanisms in the process.

        .. property that points to _allMechanisms.names (see below).

    mechanisms : List[Mechanism]
        a list of the mechanisms in the process.

    originMechanisms : MechanismList
        a list with the `ORIGIN` mechanism of the process.

        .. note:: A process can have only one `ORIGIN` mechanism; the use of a list is for compatibility with
                  methods that are also used for systems.

        COMMENT:
            based on _origin_mechs;  process.input contains the input to `ORIGIN` mechanism.
        COMMENT

    terminalMechanisms : MechanismList
        a list with the `TERMINAL` mechanism of the process.

        .. note:: A process can have only one `TERMINAL` mechanism; the use of a list is for compatibility with
                  methods that are also used for systems.

        COMMENT:
            based on _terminal_mechs; process.output contains the output of the `TERMINAL` mechanism.
        COMMENT

    monitoringMechanisms : MechanismList
        a list of all of the monitoring mechanisms in the process.

        .. based on _monitoring_mechs

    targetMechanisms : MechanismList
        a list with the `TARGET` mechanism of the process.

        .. note:: A process can have only one `TARGET` mechanism; the use of a list is for compatibility with
                  methods that are also used for systems.

        COMMENT:
            based on _target_mechs
        COMMENT

    systems : List[System]
        a list of the systems to which the process belongs.

      .. _phaseSpecMax : int : default 0
             phase of last (set of) ProcessingMechanism(s) to be executed in the process.
             It is assigned to the ``phaseSpec`` for the mechanism in the pathway with the largest ``phaseSpec`` value.

    numPhases : int : default 1
        the number of ref:`phases <System_Execution_Phase>` for the process.

        COMMENT:
            It is assigned as ``_phaseSpecMax + 1``.
        COMMENT

      .. _isControllerProcess : bool : :keyword:`False`
             identifies whether the process is an internal one created by a ControlMechanism.

    learning : Optional[LearningProjection]
        determines whether the process is configured for learning.  The value can be a `LearningProjection`,
        the keyword `LEARNING_PROJECTION`, the name of the class, or a call to its constructor.

        .. note::  If an existing `LearningProjection` or a call to the constructor is used for the specification,
                   the object itself will **not** be used as the LearningProjection for the process. Rather it
                   will be used as a template (including any parameters that are specified) for creating
                   LearningProjections for all of the `MappingProjections <MappingProjection>` in the process.

                   .. _learning_enabled : bool
                      indicates whether or not learning is enabled.  This only has effect if the ``learning`` parameter
                      has been specified (see above).

    learning_rate : float : default None
        determines the learning rate for all mechanisms in the process.  This overrides any values set for the
        function of individual LearningProjections, and persists for all subsequent runs of the system.  If it is
        set to None, then the learning_rate is determined by the last value assigned to each LearningProjection
        (either directly, or following a run of any process or system to which the LearningProjection belongs and
        for which a learning_rate was set).

    results : List[outputState.value]
        a list of return values from a sequence of executions of the process.

    timeScale : TimeScale : default TimeScale.TRIAL
        determines the default `TimeScale` value used by mechanisms in the pathway.

    name : str : default Process-<index>
        the name of the process.
        Specified in the **name** argument of the constructor for the process;
        if not is specified, a default is assigned by ProcessRegistry
        (see :ref:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Process.classPreferences
        the `PreferenceSet` for the process.
        Specified in the **prefs** argument of the constructor for the process;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py (see :ref:`PreferenceSet <LINK>` for details).


    """

    componentCategory = kwProcessComponentCategory
    className = componentCategory
    suffix = " " + className
    componentType = "Process"

    registry = ProcessRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # These will override those specified in TypeDefaultPreferences
    # classPreferences = {
    #     kwPreferenceSetName: 'ProcessCustomClassPreferences',
    #     kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}
    # Use inputValueSystemDefault as default input to process

    variableClassDefault = None

    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({TIME_SCALE: TimeScale.TRIAL,
                               '_execution_id': None,
                               PATHWAY: None,
                               'input':[],
                               'processInputStates': [],
                               'targets': None,
                               'targetInputStates': [],
                               'systems': [],
                               '_phaseSpecMax': 0,
                               '_isControllerProcess': False
                               })

    default_pathway = [Mechanism_Base.defaultMechanism]

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 pathway=default_pathway,
                 initial_values=None,
                 clamp_input=None,
                 default_projection_matrix=DEFAULT_PROJECTION_MATRIX,
                 learning=None,
                 learning_rate=None,
                 target=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 context=None):

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(pathway=pathway,
                                                  initial_values=initial_values,
                                                  clamp_input=clamp_input,
                                                  default_projection_matrix=default_projection_matrix,
                                                  learning=learning,
                                                  learning_rate=learning_rate,
                                                  target=target,
                                                  params=params)
        self.function = self.execute

        register_category(entry=self,
                          base_class=Process_Base,
                          name=name,
                          registry=ProcessRegistry,
                          context=context)

        if not context:
            # context = self.__class__.__name__
            context = INITIALIZING + self.name + kwSeparator + PROCESS_INIT

        super(Process_Base, self).__init__(variable_default=default_input_value,
                                           param_defaults=params,
                                           name=self.name,
                                           prefs=prefs,
                                           context=context)


    def _validate_variable(self, variable, context=None):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state

        :param variable:
        :param context:
        :return:
        """

        super(Process_Base, self)._validate_variable(variable, context)

        # Force Process variable specification to be a 2D array (to accommodate multiple input states of 1st mech):
        if self.variableClassDefault:
            self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        if variable is not None:
            self.variable = convert_to_np_array(self.variable, 2)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate initial_values args
           Note: validation of target (for learning) is deferred until _instantiate_target since,
                 if it doesn't have a TARGET mechanism (see _check_for_target_mechanism),
                 it will not need a target.
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Note: don't confuse target_set (argument of validate_params) with self.target (process attribute for learning)
        if INITIAL_VALUES in target_set and target_set[INITIAL_VALUES]:
            for mech, value in target_set[INITIAL_VALUES].items():
                if not isinstance(mech, Mechanism):
                    raise SystemError("{} (key for entry in initial_values arg for \'{}\') "
                                      "is not a Mechanism object".format(mech, self.name))

    def _instantiate_attributes_before_function(self, context=None):
        """Call methods that must be run before function method is instantiated

        Need to do this before _instantiate_function as mechanisms in pathway must be instantiated
            in order to assign input projection and self.outputState to first and last mechanisms, respectively

        :param context:
        :return:
        """
        self._instantiate_pathway(context=context)
        # super(Process_Base, self)._instantiate_function(context=context)

    def _instantiate_function(self, context=None):
        """Override Function._instantiate_function:

        This is necessary to:
        - insure there is no FUNCTION specified (not allowed for a Process object)
        - suppress validation (and attendant execution) of Process execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in the pathway have already been validated;
            Note: this means learning is not validated either
        """

        if self.paramsCurrent[FUNCTION] != self.execute:
            print("Process object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.paramsCurrent[FUNCTION], FUNCTION)
            self.paramsCurrent[FUNCTION] = self.execute
        # If validation pref is set, instantiate and execute the Process
        if self.prefs.paramValidationPref:
            super(Process_Base, self)._instantiate_function(context=context)
        # Otherwise, just set Process output info to the corresponding info for the last mechanism in the pathway
        else:
            self.value = self.pathway[-1].output_state.value

# DOCUMENTATION:
#         Uses paramClassDefaults[PATHWAY] == [Mechanism_Base.defaultMechanism] as default
#         1) ITERATE THROUGH CONFIG LIST TO PARSE AND INSTANTIATE EACH MECHANISM ITEM
#             - RAISE EXCEPTION IF TWO PROJECTIONS IN A ROW
#         2) ITERATE THROUGH CONFIG LIST AND ASSIGN PROJECTIONS (NOW THAT ALL MECHANISMS ARE INSTANTIATED)
#
#

    def _instantiate_pathway(self, context):
        # DOCUMENT:  Projections SPECIFIED IN A PATHWAY MUST BE A MappingProjection
        # DOCUMENT:
        # Each item in Pathway can be a Mechanism or Projection object, class ref, or specification dict,
        #     str as name for a default Mechanism,
        #     keyword (IDENTITY_MATRIX or FULL_CONNECTIVITY_MATRIX) as specification for a default Projection,
        #     or a tuple with any of the above as the first item and a param dict as the second
        """Construct pathway list of Mechanisms and Projections used to execute process

        Iterate through Pathway, parsing and instantiating each Mechanism item;
            - raise exception if two Projections are found in a row;
            - for last Mechanism in Pathway, assign ouputState to Process.outputState
        Iterate through Pathway, assigning Projections to Mechanisms:
            - first Mechanism in Pathway:
                if it does NOT already have any projections:
                    assign projection(s) from ProcessInputState(s) to corresponding Mechanism.input_state(s):
                if it DOES already has a projection, and it is from:
                    (A) the current Process input, leave intact
                    (B) another Process input, if verbose warn
                    (C) another mechanism in the current process, if verbose warn about recurrence
                    (D) a mechanism not in the current Process or System, if verbose warn
                    (E) another mechanism in the current System, OK so ignore
                    (F) from something other than a mechanism in the system, so warn (irrespective of verbose)
                    (G) a Process in something other than a System, so warn (irrespective of verbose)
            - subsequent Mechanisms:
                assign projections from each Mechanism to the next one in the list:
                - if Projection is explicitly specified as item between them in the list, use that;
                - if Projection is NOT explicitly specified,
                    but the next Mechanism already has a projection from the previous one, use that;
                - otherwise, instantiate a default MappingProjection from previous mechanism to next:
                    use kwIdentity (identity matrix) if len(sender.value == len(receiver.variable)
                    use FULL_CONNECTIVITY_MATRIX (full connectivity matrix with unit weights) if the lengths are not equal
                    use FULL_CONNECTIVITY_MATRIX (full connectivity matrix with unit weights) if LEARNING has been set

        :param context:
        :return:
        """
        pathway = self.paramsCurrent[PATHWAY]
        self._mechs = []
        self._monitoring_mechs = []
        self._target_mechs = []

        self._standardize_config_entries(pathway=pathway, context=context)

        # VALIDATE PATHWAY THEN PARSE AND INSTANTIATE MECHANISM ENTRIES  ------------------------------------
        self._parse_and_instantiate_mechanism_entries(pathway=pathway, context=context)

        # Identify origin and terminal mechanisms in the process and
        #    and assign the mechanism's status in the process to its entry in the mechanism's processes dict
        self.firstMechanism = pathway[0]
        self.firstMechanism.processes[self] = ORIGIN
        self._origin_mechs = [pathway[0]]
        self.originMechanisms = MechanismList(self, self._origin_mechs)

        # Assign last mechanism in pathwway to lastMechanism attribute
        i = -1
        while not isinstance(pathway[i],Mechanism_Base):
            i -=1
        self.lastMechanism = pathway[i]

        if self.lastMechanism is self.firstMechanism:
            self.lastMechanism.processes[self] = SINGLETON
        else:
            self.lastMechanism.processes[self] = TERMINAL
        self._terminal_mechs = [pathway[-1]]
        self.terminalMechanisms = MechanismList(self, self._terminal_mechs)

        # # Assign process outputState to last mechanisms in pathway
        # self.outputState = self.lastMechanism.outputState

        # PARSE AND INSTANTIATE PROJECTION ENTRIES  ------------------------------------

        self._parse_and_instantiate_projection_entries(pathway=pathway, context=context)

        self.pathway = pathway

        self._instantiate__deferred_inits(context=context)

        if self.learning:
            self._check_for_target_mechanism()
            if self.targetMechanism:
                self._instantiate_target_input()
            self._learning_enabled = True
        else:
            self._learning_enabled = False

        self._allMechanisms = MechanismList(self, self._mechs)
        self.monitoringMechanisms = MechanismList(self, self._monitoring_mechs)
        self.targetMechanisms = MechanismList(self, self._target_mechs)

    def _standardize_config_entries(self, pathway, context=None):

        from PsyNeuLink.Components.Mechanisms.Mechanism import _is_mechanism_spec
# FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process._validate_params
        self.runtime_params_dict = {}

        # Kristen modified 5/24
        # in  ALL mechanism tuples, the middle entry is set to zero (formerly used for specifying runtime params)
        # rightmost entry is set to zero (formerly phase spec)
        # if _is_mechanism_spec, runtime_params_dict[mechanism] is set to actual runtime params

        for i in range(len(pathway)):
            config_item = pathway[i]
            # if this element of the pathway is a tuple
            if isinstance(config_item, tuple):
                # and the tuple has 1 item
                if len(config_item) is 1:
                    # if the tuple contains either a mechanism or a projection
                    if _is_mechanism_spec(config_item[0]) or _is_projection_spec(config_item[0]):

                        # Replace it with just the mech or proj
                        pathway[i] = config_item
                        # if it's a mechanism, set the runtime params to None
                        if _is_mechanism_spec(config_item[0]):
                            self.runtime_params_dict[config_item[0]] = None
                    # otherwise the tuple is not valid
                    else:
                        raise ProcessError("First item of tuple ({}) in entry {} of pathway for {}"
                                           " is neither a mechanism nor a projection specification".
                                           format(config_item[0], i, self.name))
                # If the tuple has two items
                if len(config_item) is 2:

                    # Replace it with just the mech or proj
                    pathway[i] = config_item[0]

                    # If it's a mechanism
                    if _is_mechanism_spec(config_item[0]):
                        # and its second element is a dict
                        if isinstance(config_item[1], dict):
                            # set the mechanism's runtime params to be the second element
                            self.runtime_params_dict[config_item[0]] = config_item[1]
                        # if the second element is not a dict, then it's not valid
                        else:
                            raise ProcessError("Second item of tuple ({}) in item {} of pathway for {}"
                                               " is not a params dict.".
                                               format(config_item[1], i, self.name))
                    # if the first element is not a mechanism, then it's not valid
                    else:
                        raise ProcessError("Projection cannot have a runtime params dict".format(config_item[0],
                                                                                                 i, self.name))
                # config_item should not have more than 2 elements
                if len(config_item) > 2:
                    raise ProcessError("The tuple for item {} of pathway for {} has more than two items {}".
                                       format(i, self.name, config_item))
            else:
                # If the item is a mechanism or a projection
                if _is_mechanism_spec(pathway[i]) or _is_projection_spec(pathway[i]):
                    # if it's a mechanism, set runtime params to None
                    if _is_mechanism_spec(pathway[i]):
                        self.runtime_params_dict[pathway[i]] = None

                else:
                    raise ProcessError("Item of {} of pathway for {}"
                                       " is neither a mechanism nor a projection specification".
                                       format(i, self.name))

    def _parse_and_instantiate_mechanism_entries(self, pathway, context=None):

# FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process._validate_params
        # - make sure first entry is not a Projection
        # - make sure Projection entries do NOT occur back-to-back (i.e., no two in a row)
        # - instantiate Mechanism entries

        previous_item_was_projection = False

        for i in range(len(pathway)):
            item = pathway[i]

            # Get max phaseSpec for Mechanisms in pathway
            # if not phase_spec:
            #     phase_spec = 0
            # self._phaseSpecMax = int(max(math.floor(float(phase_spec)), self._phaseSpecMax))

            # VALIDATE PLACEMENT OF PROJECTION ENTRIES  ----------------------------------------------------------

            # Can't be first entry, and can never have two in a row

            # Config entry is a Projection
            if _is_projection_spec(item):
                # Projection not allowed as first entry
                if i==0:
                    raise ProcessError("Projection cannot be first entry in pathway ({0})".format(self.name))
                # Projections not allowed back-to-back
                if previous_item_was_projection:
                    raise ProcessError("Illegal sequence of two adjacent projections ({0}:{1} and {1}:{2})"
                                       " in pathway for {3}".
                                       format(i-1, pathway[i-1], i, pathway[i], self.name))
                previous_item_was_projection = True
                continue

            previous_item_was_projection = False
            mech = item

            # INSTANTIATE MECHANISM  -----------------------------------------------------------------------------

            # Must do this before assigning projections (below)
            # Mechanism entry must be a Mechanism object, class, specification dict, str, or (Mechanism, params) tuple
            # Don't use params item of tuple (if present) to instantiate Mechanism, as they are runtime only params

            # Entry is NOT already a Mechanism object
            if not isinstance(mech, Mechanism):
                # Note: need full pathname for mechanism factory method, as "mechanism" is used as local variable below
                mech = PsyNeuLink.Components.Mechanisms.Mechanism.mechanism(mech, context=context)
                if not mech:
                    raise ProcessError("Entry {0} ({1}) is not a recognized form of Mechanism specification".
                                       format(i, mech))
                # Params in mech tuple must be a dict or None
                # if params and not isinstance(params, dict):
                #     raise ProcessError("Params entry ({0}) of tuple in item {1} of pathway for {2} is not a dict".
                #                           format(params, i, self.name))
                # Replace Pathway entry with new tuple containing instantiated Mechanism object and params
                pathway[i] = mech


            # Entry IS already a Mechanism object
            # Add entry to _mechs and name to mechanismNames list
            # mech.phaseSpec = phase_spec
            # Add Process to the mechanism's list of processes to which it belongs
            if not self in mech.processes:
                mech.processes[self] = INTERNAL
                self._mechs.append(pathway[i])
            # self.mechanismNames.append(mech.name)

            # FIX: ADD RECURRENT PROJECTION AND MECHANISM
            # IMPLEMENTATION NOTE:  THIS IS A TOTAL HACK TO ALLOW RECURRENT MECHANISMS IN THE CURRENT SYSTEM
            #                       SHOULD BE HANDLED MORE APPROPRIATELY IN COMPOSITION
            # If this is the last mechanism in the pathway, and it has a recurrent projection,
            #    add that to the pathway so that it can be identified and assigned for learning if so specified
            if i+1 == len(pathway):
                if any(any(proj.receiver.owner is mech
                           for proj in state.efferents)
                       for state in mech.output_states):
                    for state in mech.output_states:
                        for proj in state.efferents:
                            if proj.receiver.owner is mech:
                                pathway.append(proj)
                                pathway.append(pathway[i-2])


        # Validate initial values
        # FIX: CHECK WHETHER ALL MECHANISMS DESIGNATED AS INITALIZE HAVE AN INITIAL_VALUES ENTRY
        if self.initial_values:
            for mech, value in self.initial_values.items():
                if not mech in self.mechanisms:
                    raise SystemError("{} (entry in initial_values arg) is not a Mechanism in pathway for \'{}\'".
                                      format(mech.name, self.name))
                if not iscompatible(value, mech.variable):
                    raise SystemError("{} (in initial_values arg for {}) is not a valid value for {}".
                                      format(value,
                                             append_type_to_name(self),
                                             append_type_to_name(mech)))

    def _parse_and_instantiate_projection_entries(self, pathway, context=None):

        # ASSIGN DEFAULT PROJECTION PARAMS

        # If learning is specified for the Process, add to default projection params
        if self.learning:

            # if spec is LEARNING (convenience spec), change to projection version of keyword for consistency below
            if self.learning is LEARNING:
                self.learning = LEARNING_PROJECTION

            # FIX: IF self.learning IS AN ACTUAL LearningProjection OBJECT, NEED TO RESPECIFY AS CLASS + PARAMS
            # FIX:     OR CAN THE SAME LearningProjection OBJECT BE SHARED BY MULTIPLE PROJECTIONS?
            # FIX:     DOES IT HAVE ANY INTERNAL STATE VARIABLES OR PARAMS THAT NEED TO BE PROJECTIONS-SPECIFIC?
            # FIX:     MAKE IT A COPY?

            matrix_spec = (self.default_projection_matrix, self.learning)
        else:
            matrix_spec = self.default_projection_matrix

        projection_params = {FUNCTION_PARAMS:
                                 {MATRIX: matrix_spec}}

        for i in range(len(pathway)):
                item = pathway[i]

                #region FIRST ENTRY

                # Must be a Mechanism (enforced above)
                # Assign input(s) from Process to it if it doesn't already have any
                # Note: does not include learning (even if specified for the process)
                if i == 0:
                    # Relabel for clarity
                    mech = item

                    # Check if first Mechanism already has any projections and, if so, issue appropriate warning
                    if mech.input_state.path_afferents:
                        self._issue_warning_about_existing_projections(mech, context)

                    # Assign input projection from Process
                    self._assign_process_input_projections(mech, context=context)
                    continue
                #endregion

                #region SUBSEQUENT ENTRIES

                # Item is a Mechanism
                item = item
                if isinstance(item, Mechanism):

                    preceding_item = pathway[i-1]


                    # PRECEDING ITEM IS A PROJECTION
                    if isinstance(preceding_item, Projection):
                        if self.learning:

                            # Check if preceding_item has a matrix parameterState and, if so, it has any learningSignals
                            # If it does, assign them to learning_projections
                            try:
                                learning_projections = list(projection for
                                                        projection in
                                                        preceding_item._parameter_states[MATRIX].mod_afferents
                                                        if isinstance(projection, LearningProjection))

                            # preceding_item doesn't have a parameterStates attrib, so assign one with self.learning
                            except AttributeError:
                                # Instantiate parameterStates Ordered dict with ParameterState and self.learning
                                preceding_item._parameter_states = _instantiate_state_list(
                                                                                owner=preceding_item,
                                                                                state_list=[(MATRIX,
                                                                                             self.learning)],
                                                                                state_type=ParameterState,
                                                                                state_param_identifier=PARAMETER_STATE,
                                                                                constraint_value=self.learning,
                                                                                constraint_value_name=LEARNING_PROJECTION,
                                                                                context=context)

                            # preceding_item has parameterStates but not (yet!) one for MATRIX, so instantiate it
                            except KeyError:
                                # Instantiate ParameterState for MATRIX
                                preceding_item._parameter_states[MATRIX] = _instantiate_state(
                                                                                owner=preceding_item,
                                                                                state_type=ParameterState,
                                                                                state_name=MATRIX,
                                                                                state_spec=PARAMETER_STATE,
                                                                                state_params=self.learning,
                                                                                constraint_value=self.learning,
                                                                                constraint_value_name=LEARNING_PROJECTION,
                                                                                context=context)
                            # preceding_item has parameterState for MATRIX,
                            else:
                                if not learning_projections:
                                    # Add learningProjection to projection if it doesn't have one
                                    _add_projection_to(preceding_item,
                                                      preceding_item._parameter_states[MATRIX],
                                                      projection_spec=self.learning)
                        continue

                    # Preceding item was a Mechanism, so check if a Projection needs to be instantiated between them
                    # Check if Mechanism already has a projection from the preceding Mechanism, by testing whether the
                    #    preceding mechanism is the sender of any projections received by the current one's inputState
    # FIX: THIS SHOULD BE DONE FOR ALL INPUTSTATES
    # FIX: POTENTIAL PROBLEM - EVC *CAN* HAVE MULTIPLE PROJECTIONS FROM (DIFFERENT outputStates OF) THE SAME MECHANISM

                    # PRECEDING ITEM IS A MECHANISM
                    projection_list = item.input_state.path_afferents
                    projection_found = False
                    for projection in projection_list:
                        # Current mechanism DOES receive a projection from the preceding item
                        if preceding_item == projection.sender.owner:
                            projection_found = True
                            if self.learning:
                                # Make sure projection includes a learningSignal and add one if it doesn't
                                try:
                                    matrix_param_state = projection._parameter_states[MATRIX]

                                # projection doesn't have a parameterStates attrib, so assign one with self.learning
                                except AttributeError:
                                    # Instantiate parameterStates Ordered dict with ParameterState for self.learning
                                    projection._parameter_states = _instantiate_state_list(
                                                                                owner=preceding_item,
                                                                                state_list=[(MATRIX,
                                                                                             self.learning)],
                                                                                state_type=ParameterState,
                                                                                state_param_identifier=PARAMETER_STATE,
                                                                                constraint_value=self.learning,
                                                                                constraint_value_name=LEARNING_PROJECTION,
                                                                                context=context)

                                # projection has parameterStates but not (yet!) one for MATRIX,
                                #    so instantiate it with self.learning
                                except KeyError:
                                    # Instantiate ParameterState for MATRIX
                                    projection._parameter_states[MATRIX] = _instantiate_state(
                                                                                owner=preceding_item,
                                                                                state_type=ParameterState,
                                                                                state_name=MATRIX,
                                                                                state_spec=PARAMETER_STATE,
                                                                                state_params=self.learning,
                                                                                constraint_value=self.learning,
                                                                                constraint_value_name=LEARNING_PROJECTION,
                                                                                context=context)

                                # Check if projection's matrix param has a learningSignal
                                else:
                                    if not (any(isinstance(projection, LearningProjection) for
                                                projection in matrix_param_state.mod_afferents)):
                                        _add_projection_to(projection,
                                                          matrix_param_state,
                                                          projection_spec=self.learning)

                                if self.prefs.verbosePref:
                                    print("LearningProjection added to projection from mechanism {0} to mechanism {1} "
                                          "in pathway of {2}".format(preceding_item.name, item.name, self.name))
                            break

                    if not projection_found:
                        # No projection found, so instantiate MappingProjection from preceding mech to current one;
                        # Note:  If self.learning arg is specified, it has already been added to projection_params above
                        MappingProjection(sender=preceding_item,
                                          receiver=item,
                                          params=projection_params,
                                          name='{} from {} to {}'.
                                          format(MAPPING_PROJECTION, preceding_item.name, item.name)
                                          )
                        if self.prefs.verbosePref:
                            print("MappingProjection added from mechanism {0} to mechanism {1}"
                                  " in pathway of {2}".format(preceding_item.name, item.name, self.name))

                # Item is a Projection or specification for one
                else:
                    # Instantiate Projection, assigning mechanism in previous entry as sender and next one as receiver
                    # IMPLEMENTATION NOTE:  FOR NOW:
                    #    - ASSUME THAT PROJECTION SPECIFICATION (IN item) IS ONE OF THE FOLLOWING:
                    #        + Projection object
                    #        + Matrix object
                    # #        +  Matrix keyword (IDENTITY_MATRIX or FULL_CONNECTIVITY_MATRIX)
                    #        +  Matrix keyword (use "is_projection" to validate)
                    #    - params IS IGNORED
    # 9/5/16:
    # FIX: IMPLEMENT _validate_params TO VALIDATE PROJECTION SPEC USING Projection.is_projection
    # FIX: ADD SPECIFICATION OF PROJECTION BY KEYWORD:
    # FIX: ADD learningSignal spec if specified at Process level (overrided individual projection spec?)

                    # FIX: PARSE/VALIDATE ALL FORMS OF PROJECTION SPEC (ITEM PART OF TUPLE) HERE:
                    # FIX:                                                          CLASS, OBJECT, DICT, STR, TUPLE??
                    # IMPLEMENT: MOVE State._instantiate_projections_to_state(), _check_projection_receiver()
                    #            and _parse_projection_ref() all to Projection_Base.__init__() and call that
                    #           VALIDATION OF PROJECTION OBJECT:
                    #                MAKE SURE IT IS A MappingProjection
                    #                CHECK THAT SENDER IS pathway[i-1][OBJECT_ITEM]
                    #                CHECK THAT RECEVIER IS pathway[i+1][OBJECT_ITEM]


                    # Get sender for projection
                    sender_mech=pathway[i-1]

                    # Get receiver for projection
                    try:
                        receiver_mech=pathway[i+1]
                    except IndexError:
                       # There are no more entries in the pathway
                       #    so the projection had better project to a mechanism already in the pathway;
                       #    otherwise, raise and exception
                       try:
                           receiver_mech = item.receiver.owner
                           if not receiver_mech in [object_item for object_item in pathway]:
                               raise AttributeError
                       except AttributeError:
                           raise ProcessError("The last entry in the pathway for {} is a project specification {}, "
                                              "so its receiver must be a mechanism in the pathway".
                                              format(self.name, item))

                    # projection spec is an instance of a MappingProjection
                    if isinstance(item, MappingProjection):
                        # Check that Projection's sender and receiver are to the mech before and after it in the list
                        # IMPLEMENT: CONSIDER ADDING LEARNING TO ITS SPECIFICATION?
    # FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process._validate_params

                        # If initialization of MappingProjection has been deferred,
                        #    check sender and receiver, assign them if they have not been assigned, and initialize it
                        if item.value is DEFERRED_INITIALIZATION:
                            # Check sender arg
                            try:
                                sender_arg = item.init_args[SENDER]
                            except AttributeError:
                                raise ProcessError("PROGRAM ERROR: Value of {} is {} but it does not have init_args".
                                                   format(item, DEFERRED_INITIALIZATION))
                            except KeyError:
                                raise ProcessError("PROGRAM ERROR: Value of {} is {} "
                                                   "but init_args does not have entry for {}".
                                                   format(item.init_args[NAME], DEFERRED_INITIALIZATION, SENDER))
                            else:
                                # If sender is not specified for the projection,
                                #    assign mechanism that precedes in pathway
                                if sender_arg is None:
                                    item.init_args[SENDER] = sender_mech
                                elif sender_arg is not sender_mech:
                                    raise ProcessError("Sender of projection ({}) specified in item {} of"
                                                       " pathway for {} is not the mechanism ({}) "
                                                       "that precedes it in the pathway".
                                                       format(item.init_args[NAME],
                                                              i, self.name, sender_mech.name))
                            # Check receiver arg
                            try:
                                receiver_arg = item.init_args[kwReceiverArg]
                            except AttributeError:
                                raise ProcessError("PROGRAM ERROR: Value of {} is {} but it does not have init_args".
                                                   format(item, DEFERRED_INITIALIZATION))
                            except KeyError:
                                raise ProcessError("PROGRAM ERROR: Value of {} is {} "
                                                   "but init_args does not have entry for {}".
                                                   format(item.init_args[NAME], DEFERRED_INITIALIZATION, kwReceiverArg))
                            else:
                                # If receiver is not specified for the projection,
                                #    assign mechanism that follows it in the pathway
                                if receiver_arg is None:
                                    item.init_args[kwReceiverArg] = receiver_mech
                                elif receiver_arg is not receiver_mech:
                                    raise ProcessError("Receiver of projection ({}) specified in item {} of"
                                                       " pathway for {} is not the mechanism ({}) "
                                                       "that follows it in the pathway".
                                                       format(item.init_args[NAME],
                                                              i, self.name, receiver_mech.name))

                            # Complete initialization of projection
                            item._deferred_init()

                        if not item.sender.owner is sender_mech:
                            raise ProcessError("Sender of projection ({}) specified in item {} of pathway for {} "
                                               "is not the mechanism ({}) that precedes it in the pathway".
                                               format(item.name, i, self.name, sender_mech.name))
                        if not item.receiver.owner is receiver_mech:
                            raise ProcessError("Receiver of projection ({}) specified in item {} of pathway for "
                                               "{} is not the mechanism ({}) that follows it in the pathway".
                                               format(item.name, i, self.name, sender_mech.name))
                        projection = item

                        # TEST
                        # if params:
                        #     projection.matrix = params

                    # projection spec is a MappingProjection class reference
                    elif inspect.isclass(item) and issubclass(item, MappingProjection):
                        # if params:
                        #     # Note:  If self.learning is specified, it has already been added to projection_params above
                        #     projection_params = params
                        projection = MappingProjection(sender=sender_mech,
                                             receiver=receiver_mech,
                                             # params=projection_params
                                                       )

                    # projection spec is a matrix spec, a keyword for one, or a (matrix, LearningProjection) tuple
                    # Note: this is tested above by call to _is_projection_spec()
                    elif (isinstance(item, (np.matrix, str, tuple)) or
                              (isinstance(item, np.ndarray) and item.ndim == 2)):
                        # If a LearningProjection is explicitly specified for this projection, use it
                        # if params:
                        #     matrix_spec = (item, params)
                        # If a LearningProjection is not specified for this projection but self.learning is, use that
                        if self.learning:
                            matrix_spec = (item, self.learning)
                        # Otherwise, do not include any LearningProjection
                        else:
                            matrix_spec = item
                        projection = MappingProjection(sender=sender_mech,
                                             receiver=receiver_mech,
                                             matrix=matrix_spec)
                    else:
                        raise ProcessError("Item {0} ({1}) of pathway for {2} is not "
                                           "a valid mechanism or projection specification".format(i, item, self.name))
                    # Reassign Pathway entry
                    #    with Projection as OBJECT item and original params as PARAMS item of the tuple
                    # IMPLEMENTATION NOTE:  params is currently ignored
                    pathway[i] = projection


    def _issue_warning_about_existing_projections(self, mechanism, context=None):

        # Check where the projection(s) is/are from and, if verbose pref is set, issue appropriate warnings
        for projection in mechanism.input_state.all_afferents:

            # Projection to first Mechanism in Pathway comes from a Process input
            if isinstance(projection.sender, ProcessInputState):
                # If it is:
                # (A) from self, ignore
                # (B) from another Process, warn if verbose pref is set
                if not projection.sender.owner is self:
                    if self.prefs.verbosePref:
                        print("WARNING: {0} in pathway for {1} already has an input from {2} "
                              "that will be used".
                              format(mechanism.name, self.name, projection.sender.owner.name))
                    return

            # (C) Projection to first Mechanism in Pathway comes from one in the Process' _mechs;
            #     so warn if verbose pref is set
            if projection.sender.owner in self._mechs:
                if self.prefs.verbosePref:
                    print("WARNING: first mechanism ({0}) in pathway for {1} receives "
                          "a (recurrent) projection from another mechanism {2} in {1}".
                          format(mechanism.name, self.name, projection.sender.owner.name))

            # Projection to first Mechanism in Pathway comes from a Mechanism not in the Process;
            #    check if Process is in a System, and projection is from another Mechanism in the System
            else:
                try:
                    if (inspect.isclass(context) and issubclass(context, System)):
                        # Relabel for clarity
                        system = context
                    else:
                        system = None
                except:
                    # Process is NOT being implemented as part of a System, so projection is from elsewhere;
                    #  (D)  Issue warning if verbose
                    if self.prefs.verbosePref:
                        print("WARNING: first mechanism ({0}) in pathway for {1} receives a "
                              "projection ({2}) that is not part of {1} or the System it is in".
                              format(mechanism.name, self.name, projection.sender.owner.name))
                else:
                    # Process IS being implemented as part of a System,
                    if system:
                        # (E) Projection is from another Mechanism in the System
                        #    (most likely the last in a previous Process)
                        if mechanism in system.mechanisms:
                            pass
                        # (F) Projection is from something other than a mechanism,
                        #     so warn irrespective of verbose (since can't be a Process input
                        #     which was checked above)
                        else:
                            print("First mechanism ({0}) in pathway for {1}"
                                  " receives a projection {2} that is not in {1} "
                                  "or its System ({3}); it will be ignored and "
                                  "a projection assigned to it by {3}".
                                  format(mechanism.name,
                                         self.name,
                                         projection.sender.owner.name,
                                         context.name))
                    # Process is being implemented in something other than a System
                    #    so warn (irrespecive of verbose)
                    else:
                        print("WARNING:  Process ({0}) being instantiated in context "
                                           "({1}) other than a System ".format(self.name, context))

    def _assign_process_input_projections(self, mechanism, context=None):
        """Create projection(s) for each item in Process input to inputState(s) of the specified Mechanism

        For each item in Process input:
        - create process_input_state, as sender for MappingProjection to the mechanism.input_state
        - create the MappingProjection (with process_input_state as sender, and mechanism as receiver)

        If len(Process.input) == len(mechanism.variable):
            - create one projection for each of the mechanism.input_state(s)
        If len(Process.input) == 1 but len(mechanism.variable) > 1:
            - create a projection for each of the mechanism.input_states, and provide Process.input[value] to each
        If len(Process.input) > 1 but len(mechanism.variable) == 1:
            - create one projection for each Process.input[value] and assign all to mechanism.input_state
        Otherwise,  if len(Process.input) != len(mechanism.variable) and both > 1:
            - raise exception:  ambiguous mapping from Process input values to mechanism's input_states

        :param mechanism:
        :return:
        """

        # FIX: LENGTH OF EACH PROCESS INPUT STATE SHOUD BE MATCHED TO LENGTH OF INPUT STATE FOR CORRESPONDING ORIGIN MECHANISM

        # If input was not provided, generate defaults to match format of ORIGIN mechanisms for process
        if self.variable is None:
            self.variable = []
            seen = set()
            # mech_list = list(object_item for object_item in self._mechs)
            for mech in self._mechs:
                # Skip repeat mechansims (don't add another element to self.variable)
                if mech in seen:
                    continue
                else:
                    seen.add(mech)
                if mech.processes[self] in {ORIGIN, SINGLETON}:
                    self.variable.extend(mech.variable)
        process_input = convert_to_np_array(self.variable,2)

        # Get number of Process inputs
        num_process_inputs = len(process_input)

        # Get number of mechanism.input_states
        #    - assume mechanism.variable is a 2D np.array, and that
        #    - there is one inputState for each item (1D array) in mechanism.variable
        num_mechanism_input_states = len(mechanism.variable)

        # There is a mismatch between number of Process inputs and number of mechanism.input_states:
        if num_process_inputs > 1 and num_mechanism_input_states > 1 and num_process_inputs != num_mechanism_input_states:
            raise ProcessError("Mismatch between number of input values ({0}) for {1} and "
                               "number of input_states ({2}) for {3}".format(num_process_inputs,
                                                                            self.name,
                                                                            num_mechanism_input_states,
                                                                            mechanism.name))

        # Create input state for each item of Process input, and assign to list
        for i in range(num_process_inputs):
            process_input_state = ProcessInputState(owner=self,
                                                    variable=process_input[i],
                                                    prefs=self.prefs)
            self.processInputStates.append(process_input_state)

        from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

        # If there is the same number of Process input values and mechanism.input_states, assign one to each
        if num_process_inputs == num_mechanism_input_states:
            for i in range(num_mechanism_input_states):
                # Insure that each Process input value is compatible with corresponding variable of mechanism.input_state
                # MODIFIED 4/3/17 NEW:
                input_state_variable = mechanism.input_states[i].variable
                # MODIFIED 4/3/17 END
                if not iscompatible(process_input[i], input_state_variable):
                    raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                       "variable for corresponding inputState of {3}".
                                       format(i, process_input[i], self.name, mechanism.name))
                # Create MappingProjection from Process input state to corresponding mechanism.input_state
                MappingProjection(sender=self.processInputStates[i],
                                  receiver=mechanism.input_states[i],
                                  name=self.name+'_Input Projection',
                                  context=context)
                if self.prefs.verbosePref:
                    print("Assigned input value {0} ({1}) of {2} to corresponding inputState of {3}".
                          format(i, process_input[i], self.name, mechanism.name))

        # If the number of Process inputs and mechanism.input_states is unequal, but only a single of one or the other:
        # - if there is a single Process input value and multiple mechanism.input_states,
        #     instantiate a single Process input state with projections to each of the mechanism.input_states
        # - if there are multiple Process input values and a single mechanism.input_state,
        #     instantiate multiple Process input states each with a projection to the single mechanism.input_state
        else:
            for i in range(num_mechanism_input_states):
                for j in range(num_process_inputs):
                    if not iscompatible(process_input[j], mechanism.variable[i]):
                        raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                           "variable ({3}) for inputState {4} of {5}".
                                           format(j, process_input[j], self.name,
                                                  mechanism.variable[i], i, mechanism.name))
                    # Create MappingProjection from Process buffer_intput_state to corresponding mechanism.input_state
                    MappingProjection(sender=self.processInputStates[j],
                            receiver=mechanism.input_states[i],
                            name=self.name+'_Input Projection')
                    if self.prefs.verbosePref:
                        print("Assigned input value {0} ({1}) of {2} to inputState {3} of {4}".
                              format(j, process_input[j], self.name, i, mechanism.name))

        mechanism._receivesProcessInput = True

    def _assign_input_values(self, input, context=None):
        """Validate input, assign each item (1D array) in input to corresponding process_input_state

        Returns converted version of input

        Args:
            input:

        Returns:

        """
        # Validate input
        if input is None:
            input = self.firstMechanism.variableInstanceDefault
            if (self.prefs.verbosePref and
                    not (not context or COMPONENT_INIT in context)):
                print("- No input provided;  default will be used: {0}")

        else:
            # Insure that input is a list of 1D array items, one for each processInputState
            # If input is a single number, wrap in a list
            from numpy import ndarray
            if isinstance(input, numbers.Number) or (isinstance(input, ndarray) and input.ndim == 0):
                input = [input]
            # If input is a list of numbers, wrap in an outer list (for processing below)
            if all(isinstance(i, numbers.Number) for i in input):
                input = [input]

        if len(self.processInputStates) != len(input):
            raise ProcessError("Length ({}) of input to {} does not match the number "
                               "required for the inputs of its origin mechanisms ({}) ".
                               format(len(input), self.name, len(self.processInputStates)))

        # Assign items in input to value of each process_input_state
        for i in range (len(self.processInputStates)):
            self.processInputStates[i].value = input[i]

        return input

    def _update_input(self):
        for s, i in zip(self.processInputStates, range(len(self.processInputStates))):
            self.input = s.value

    def _instantiate__deferred_inits(self, context=None):
        """Instantiate any objects in the Process that have deferred their initialization

        Description:
            For learning:
                go through _mechs in reverse order of pathway since
                    LearningProjections are processed from the output (where the training signal is provided) backwards
                exhaustively check all of components of each mechanism,
                    including all projections to its input_states and parameterStates
                initialize all items that specified deferred initialization
                construct a _monitoring_mechs of mechanism tuples (mech, params, phase_spec):
                    assign phase_spec for each MonitoringMechanism = self._phaseSpecMax + 1 (i.e., execute them last)
                add _monitoring_mechs to the Process' _mechs
                assign input projection from Process to first mechanism in _monitoring_mechs

        IMPLEMENTATION NOTE: assume that the only projection to a projection is a LearningProjection
                             this is implemented to be fully general, but at present may be overkill
                             since the only objects that currently use deferred initialization are LearningProjections
        """

        # For each mechanism in the Process, in backwards order through its _mechs
        for item in reversed(self._mechs):
            mech = item
            mech._deferred_init()

            # For each inputState of the mechanism
            for input_state in mech.input_states:
                input_state._deferred_init()
                # Restrict projections to those from mechanisms in the current process
                projections = []
                for projection in input_state.all_afferents:
                    try:
                        if self in projection.sender.owner.processes:
                            projections.append(projection)
                    except AttributeError:
                        pass
                self._instantiate__deferred_init_projections(projections, context=context)

            # For each parameterState of the mechanism
            for parameter_state in mech._parameter_states:
                parameter_state._deferred_init() # XXX
                # MODIFIED 5/2/17 OLD:
                # self._instantiate__deferred_init_projections(parameter_state.path_afferents)
                # MODIFIED 5/2/17 NEW:
                # Defer instantiation of ControlProjections to System
                #   and there should not be any other type of projection to the ParameterState of a Mechanism
                from PsyNeuLink.Components.Projections.ModulatoryProjections.ControlProjection import ControlProjection
                if not all(isinstance(proj, ControlProjection) for proj in parameter_state.mod_afferents):
                    raise ProcessError("PROGRAM ERROR:  non-ControlProjection found to ParameterState for a Mechanism")
                # MODIFIED 5/2/17 END

        # Label monitoring mechanisms and add _monitoring_mechs to _mechs for execution
        if self._monitoring_mechs:

            # Add designations to newly created MonitoringMechanisms:
            for object_item in self._monitoring_mechs:
                mech = object_item
                # If
                # - mech is a TARGET ObjectiveMechanism, and
                # - the mech that projects to mech is a TERMINAL for the current process, and
                # - current process has learning specified
                # then designate mech as a TARGET
                if (isinstance(mech, ObjectiveMechanism) and
                        # any(projection.sender.owner.processes[self] == TERMINAL
                        #     for projection in mech.input_states[SAMPLE].path_afferents) and
                        mech._learning_role is TARGET and
                        self.learning
                            ):
                    object_item.processes[self] = TARGET
                else:
                    # mech must be a LearningMechanism;
                    # If a learning_rate has been specified for the process, assign that to all LearningMechanisms
                    #    for which a mechanism-specific learning_rate has NOT been assigned
                    if (self.learning_rate is not None and
                                mech.function_object.learning_rate is None):
                        mech.function_object.learning_rate = self.learning_rate

                    # Assign its label
                    object_item.processes[self] = MONITORING

            # Add _monitoring_mechs to _mechs
            self._mechs.extend(self._monitoring_mechs)

            # IMPLEMENTATION NOTE:
            #   MonitoringMechanisms for learning are assigned _phaseSpecMax;
            #   this is so that they will run after the last ProcessingMechansisms have run

    def _instantiate__deferred_init_projections(self, projection_list, context=None):

        # For each projection in the list
        for projection in projection_list:
            projection._deferred_init() # XXX

            # FIX:  WHY DOESN'T THE PROJECTION HANDLE THIS? (I.E., IN ITS deferred_init() METHOD?)
            # For each parameter_state of the projection
            try:
                for parameter_state in projection._parameter_states:
                    # Initialize each projection to the parameterState (learning or control)
                    # IMPLEMENTATION NOTE:  SHOULD ControlProjections BE IGNORED HERE?
                    for param_projection in parameter_state.mod_afferents:
                        param_projection._deferred_init(context=context)
                        if isinstance(param_projection, LearningProjection):
                            # Get ObjectiveMechanism if there is one, and add to _monitoring_mechs
                            try:
                                objective_mechanism = projection.objective_mechanism
                            except AttributeError:
                                pass
                            else:
                                # If objective_mechanism is not already in _monitoring_mechs,
                                #     pack in tuple and add it
                                if objective_mechanism and not objective_mechanism in self._monitoring_mechs:
                                    # objective_object_item = objective_mechanism
                                    self._monitoring_mechs.append(objective_mechanism)
                            # Get LearningMechanism and add to _monitoring_mechs; raise exception if not found
                            try:
                                learning_mechanism = projection.learning_mechanism
                            except AttributeError:
                                raise ProcessError("{} is missing a LearningMechanism".format(param_projection.name))
                            else:
                                # If learning_mechanism is not already in _monitoring_mechs,
                                #     pack in tuple and add it
                                if learning_mechanism and not any(learning_mechanism is object_item for
                                                                    object_item in self._monitoring_mechs):
                                    # learning_object_item = learning_mechanism
                                    self._monitoring_mechs.append(learning_mechanism)

            # Not all Projection subclasses instantiate parameterStates
            except AttributeError as e:
                if 'parameterStates' in e.args[0]:
                    pass
                else:
                    error_msg = 'Error in attempt to initialize LearningProjection ({}) for {}: \"{}\"'.\
                        format(param_projection.name, projection.name, e.args[0])
                    raise ProcessError(error_msg)

    def _check_for_target_mechanism(self):
        """Check for and assign TARGET ObjectiveMechanism to use for reporting error during learning.

         This should only be called if self.learning is specified
         Check that there is one and only one TARGET ObjectiveMechanism for the process
         Assign targetMechanism to self.targetMechanism,
             assign self to targetMechanism.processes,
             and report assignment if verbose
        """

        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ObjectiveMechanism import ObjectiveMechanism
        def trace_learning_objective_mechanism_projections(mech):
            """Recursively trace projections to Objective mechanisms;
                   return TARGET ObjectiveMechanism if one is found upstream;
                   return None if no TARGET ObjectiveMechanism is found.
            """
            for input_state in mech.input_states.values():
                for projection in input_state.path_afferents:
                    sender = projection.sender.owner
                    # If projection is not from another ObjectiveMechanism, ignore
                    if not isinstance(sender, (ObjectiveMechanism)):
                        continue
                    if isinstance(sender, ObjectiveMechanism) and sender._learning_role is TARGET:
                        return sender
                    if sender.input_states:
                        target_mech = trace_learning_objective_mechanism_projections(sender)
                        if target_mech:
                            return target_mech
                        else:
                            continue
                    else:
                        continue

        if not self.learning:
            raise ProcessError("PROGRAM ERROR: _check_for_target_mechanism should only be called"
                               " for a process if it has a learning specification")

        target_mechs = list(object_item
                           for object_item in self._mechs
                            if (isinstance(object_item, ObjectiveMechanism) and
                                object_item._learning_role is TARGET))

        if not target_mechs:

            # Trace projections to first learning ObjectiveMechanism (which is for the last mechanism in the process)
            #   (in case terminal mechanism of process is part of another process that has learning implemented)
            #    in which case, shouldn't assign target ObjectiveMechanism, but rather just a LearningMechanism)
            try:
                target_mech = trace_learning_objective_mechanism_projections(self._monitoring_mechs[0][0])
            except IndexError:
                raise ProcessError("Learning specified for {} but no ObjectiveMechanisms or LearningMechanisms found"
                                   .format(self.name))

            if target_mech:
                if self.prefs.verbosePref:
                    warnings.warn("{} itself has no Target Mechanism, but its TERMINAL_MECHANISM ({}) "
                                  "appears to be in one or more pathways ({}) that has one".
                                                      format(self.name,
                                                             # list(self.terminalMechanisms)[0].name,
                                                             self.lastMechanism.name,
                                                             list(process.name for process in target_mech.processes)))
                self.targetMechanism = None
            else:

                raise ProcessError("PROGRAM ERROR: {} has a learning specification ({}) "
                                   "but no TARGET ObjectiveMechanism".format(self.name, self.learning))

        elif len(target_mechs) > 1:
            target_mech_names = list(targetMechanism.name for targetMechanism in target_mechs)
            raise ProcessError("PROGRAM ERROR: {} has more than one targetMechanism mechanism: {}".
                               format(self.name, target_mech_names))

        else:
            self.targetMechanism = target_mechs[0]
            self._target_mechs.append(target_mechs[0])
            if self.prefs.verbosePref:
                print("\'{}\' assigned as TARGET ObjectiveMechanism for output of \'{}\'".
                      format(self.targetMechanism.name, self.name))

    def _instantiate_target_input(self):

        if self.target is None:
            raise ProcessError("Learning has been specified for {} and it has a TARGET ObjectiveMechanism, "
                               "so it must also have a target.".format(self.name))

        target = np.atleast_1d(self.target)

        # Create ProcessInputState for target and assign to targetMechanism's target inputState
        target_mech_target = self.targetMechanism.input_states[TARGET]

        # Check that length of process' target input matches length of targetMechanism's target input
        if len(target) != len(target_mech_target.variable):
            raise ProcessError("Length of target ({}) does not match length of input for targetMechanism in {}".
                               format(len(target), len(target_mech_target.variable)))

        target_input_state = ProcessInputState(owner=self,
                                                variable=target,
                                                prefs=self.prefs,
                                                name=TARGET)
        self.targetInputStates.append(target_input_state)

        # Add MappingProjection from target_input_state to MonitoringMechanism's target inputState
        from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
        MappingProjection(sender=target_input_state,
                receiver=target_mech_target,
                name=self.name+'_Input Projection to '+target_mech_target.name)

    def initialize(self):
        """Assign the values specified for each mechanism in the process' `initial_values` attribute.
        """
        # FIX:  INITIALIZE PROCESS INPUTS??
        for mech, value in self.initial_values.items():
            mech.initialize(value)

    def execute(self,
                input=None,
                # params=None,
                target=None,
                execution_id=None,
                clock=CentralClock,
                time_scale=None,
                # time_scale=TimeScale.TRIAL,
                runtime_params=None,
                termination_processing=None,
                termination_learning=None,
                context=None
                ):
        """Execute the mechanisms specified in the process` `pathway` attribute.

        COMMENT:
            First check that input is provided (required) and appropriate.
            Then execute each mechanism in the order they appear in the `pathway` list.
        COMMENT

        Arguments
        ---------

        input : List[value] or ndarray: default input to process
            input used to execute the process.
            This must be compatible with the input of the `ORIGIN` mechanism (the first in its `pathway`).

        time_scale : TimeScale :  default TimeScale.TRIAL
            specifies whether mechanisms are executed for a single time step or a trial.

        params : Dict[param keyword, param value] :  default None
            a `parameter dictionary <ParameterState_Specifying_Parameters>` that can include any of the parameters used
            as arguments to instantiate the object. Use parameter's name as the keyword for its entry.  Values specified
            for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

        COMMENT:
            context : str : default EXECUTING + self.name
                a string used for contextualization of instantiation, hierarchical calls, executions, etc.
        COMMENT

        Returns
        -------

        output of process : ndarray
            output of process` `TERMINAL` mechanism (the last in its `pathway`).

        COMMENT:
           IMPLEMENTATION NOTE:
           Still need to:
           * coordinate execution of multiple processes (in particular, mechanisms that appear in more than one process)
           * deal with different time scales
        COMMENT

        """

        if not context:
            context = EXECUTING + " " + PROCESS + " " + self.name

        from PsyNeuLink.Globals.Run import _get_unique_id
        self._execution_id = execution_id or _get_unique_id()
        for mech in self.mechanisms:
            mech._execution_id = self._execution_id

        # Report output if reporting preference is on and this is not an initialization run
        report_output = self.prefs.reportOutputPref and context and EXECUTING in context


        # FIX: CONSOLIDATE/REARRANGE _assign_input_values, _check_args, AND ASIGNMENT OF input TO self.variable
        # FIX: (SO THAT assign_input_value DOESN'T HAVE TO RETURN input

        self.input = self._assign_input_values(input=input, context=context)

        self._check_args(self.input,runtime_params)

        self.timeScale = time_scale or TimeScale.TRIAL

        # Use Process self.input as input to first Mechanism in Pathway
        self.variable = self.input

        # Generate header and report input
        if report_output:
            self._report_process_initiation(separator=True)

        # Execute each Mechanism in the pathway, in the order listed, except those used for learning
        for mechanism in self._mechs:
            if (isinstance(mechanism, LearningMechanism) or
                    (isinstance(mechanism, ObjectiveMechanism) and mechanism._role is LEARNING)):
                continue

            # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
            mechanism.execute(clock=clock,
                              time_scale=self.timeScale,
                              # time_scale=time_scale,
                              # runtime_params=params,
                              context=context)
            if report_output:
                # FIX: USE clamp_input OPTION HERE, AND ADD HARD_CLAMP AND SOFT_CLAMP
                self._report_mechanism_execution(mechanism)

            if mechanism is self.firstMechanism and not self.clamp_input:
                # Zero self.input to first mechanism after first run
                #     in case it is repeated in the pathway or receives a recurrent projection
                self.variable = self.variable * 0

        # Execute LearningMechanisms
        if self._learning_enabled:

            self._execute_learning(target=target, clock=clock, context=context)
            # self._execute_learning(clock=clock, time_scale=time_scale, context=context)

        if report_output:
            self._report_process_completion(separator=True)

        # FIX:  SHOULD THIS BE JUST THE VALUE OF THE PRIMARY OUTPUTSTATE, OR OF ALL OF THEM?
        return self.output_state.value

    def _execute_learning(self, target=None, clock=CentralClock, context=None):
    # def _execute_learning(self, clock=CentralClock, time_scale=TimeScale.TRIAL, context=None):
        """ Update each LearningProjection for mechanisms in _mechs of process

        # Begin with projection(s) to last Mechanism in _mechs, and work backwards

        """

        # FIRST, assign targets

        # If target was provided to execute, use that;  otherwise, will use value provided on instantiation
        #
        if target is not None:
            self.target = target

        # If targets were specified as a function in call to Run() or in System,
        #  call the function now and assign value to targetInputStates
        #    (i.e., after execution of the pathways, but before learning)
        # Note:  this accommodates functions that predicate the target on the outcome of processing
        #        (e.g., for rewards in reinforcement learning)
        elif isinstance(self.targets, function_type):
            self.target = self.targets()
            # FIX: DOES THIS NEED TO BE A LOOP?  ISN'T THERE ONLY EVER ONE targetInputState FOR A PROCESS?

        # Assign target to targetInputState (ProcessInputState that projects to targetMechanism for the process)
        for i, target_input_state in zip(range(len(self.targetInputStates)), self.targetInputStates):
            target_input_state.value = self.target[i]

        # # Zero any input from projections to target from any other processes
        # # Note: there is only one targetMechanism in a Process, so can assume it is first item and no need to iterate
        for process in list(self.targetMechanisms)[0].processes:
            process.targetInputStates[0].value *= 0
        if callable(self.target):
            self.targetInputStates[0].variable = self.target()
        else:
            self.targetInputStates[0].value = np.array(self.target)

        # # MODIFIED 3/22/17 NEW:
        # # NEXT, implement process learning_rate param if specified:
        # #    embed it in a param specification dict for inclusion with runtime_params
        # process_learning_rate_spec_dict = None
        # if self.learning_rate is not None:
        #     process_learning_rate_spec_dict = {LEARNING_RATE: self.learning_rate}
        # # MODIFIED 3/22/17 END

        # THEN, execute Objective and LearningMechanisms
        for mechanism in self._monitoring_mechs:
            # # MODIFIED 3/22/17 NEW:
            # # If learning_rate was specified for process and this is a LearningMechanism
            # if process_learning_rate_spec_dict is not None and isinstance(mechanism, LearningMechanism):
            #     # Add to any existing params
            #     if params is not None:
            #         params.update(process_learning_rate_spec_dict)
            #     # Or just assign if none
            #     else:
            #         params = process_learning_rate_spec_dict
            # # MODIFIED 3/22/17 END

            mechanism.execute(clock=clock,
                              time_scale=self.timeScale,
                              # runtime_params=params,
                              context=context)


        # FINALLY, execute LearningProjections to MappingProjections in the process' pathway
        for item in self._mechs:
            mech = item
            # params = item.params

            # IMPLEMENTATION NOTE:
            #    This implementation restricts learning to parameterStates of projections to input_states
            #    That means that other parameters (e.g. object or function parameters) are not currenlty learnable

            # For each inputState of the mechanism
            for input_state in mech.input_states:
                # For each projection in the list
                for projection in input_state.path_afferents:

                    # Skip learning if projection is an input from the Process or a system
                    # or comes from a mechanism that belongs to another process
                    #    (this is to prevent "double-training" of projections from mechanisms belonging
                    #     to different processes when call to _execute_learning() comes from a system)
                    sender = projection.sender.owner
                    if isinstance(sender, Process_Base) or not self in (sender.processes):
                        continue

                    # For each parameter_state of the projection
                    try:
                        for parameter_state in projection._parameter_states:
                            # Call parameter_state.update with LEARNING in context to update LearningSignals
                            # Note: do this rather just calling LearningSignals directly
                            #       since parameter_state.update() handles parsing of LearningProjection-specific params
                            context = context + SEPARATOR_BAR + LEARNING

                            # NOTE: This will need to be updated when runtime params are reenabled
                            # parameter_state.update(params=params, time_scale=TimeScale.TRIAL, context=context)
                            parameter_state.update(time_scale=TimeScale.TRIAL, context=context)

                    # Not all Projection subclasses instantiate parameterStates
                    except AttributeError as e:
                        pass

    def run(self,
            inputs,
            num_executions=None,
            reset_clock=True,
            initialize=False,
            targets=None,
            learning=None,
            call_before_trial=None,
            call_after_trial=None,
            call_before_time_step=None,
            call_after_time_step=None,
            time_scale=None):
        """Run a sequence of executions

        COMMENT:
            Call execute method for each execution in a sequence specified by the `inputs` argument (required).
            See `Run` for details of formatting input specifications.
        COMMENT

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_input_value for a single execution
            input for each in a sequence of executions (see :doc:`Run` for a detailed description of formatting
            requirements and options).

        reset_clock : bool : default True
            reset `CentralClock <TimeScale.CentralClock>` to 0 before a sequence of executions.

        initialize : bool default False
            call the process' `initialize` method before a sequence of executions.

        targets : List[input] or np.ndarray(input) : default None
            target value(s) assigned to the process` `target <Process_Base.targetMechanisms>` mechanism for each
            execution (during learning).  The length (of the outermost level if a nested list, or lowest axis if an
            ndarray) must be equal to that of the `inputs` argument (see above).

        learning : bool :  default None
            enables or disables learning during execution.
            If it is not specified, current state is left intact.
            If :keyword:`True`, learning is forced on; if :keyword:`False`, learning is forced off.

        call_before_trial : Function : default None
            called before each trial in the sequence is executed.

        call_after_trial : Function : default None
            called after each trial in the sequence is executed.

        call_before_time_step : Function : default None
            called before each time_step of each trial is executed.

        call_after_time_step : Function : default None
            called after each time_step of each trial is executed.

        time_scale : TimeScale :  default TimeScale.TRIAL
            specifies whether mechanisms are executed for a single `time_step or a trial <Run_Timing>`.

        Returns
        -------

        <process>.results : List[outputState.value]
            list of the value of the outputState for each `TERMINAL` mechanism of the system returned for
            each execution.

        """
        from PsyNeuLink.Globals.Run import run
        return run(self,
                   inputs=inputs,
                   num_executions=num_executions,
                   reset_clock=reset_clock,
                   initialize=initialize,
                   targets=targets,
                   learning=learning,
                   call_before_trial=call_before_trial,
                   call_after_trial=call_after_trial,
                   call_before_time_step=call_before_time_step,
                   call_after_time_step=call_after_time_step,
                   time_scale=time_scale)
    def _report_process_initiation(self, input=None, separator=False):
        """
        Parameters
        ----------
        input : ndarray
            input to ORIGIN mechanism for current execution.  By default, it is the value specified by the
            ProcessInputState that projects to the ORIGIN mechanism.  Used by system to specify the input
            from the SystemInputState when the ORIGIN mechanism is executed as part of that sysetm.

        separator : boolean
            determines whether separator is printed above output

        Returns
        -------

        """
        if separator:
            print("\n\n****************************************\n")

        print("\n\'{}\' executing with:\n- pathway: [{}]".
              format(append_type_to_name(self),
              # format(self.name,
                     re.sub(r'[\[,\],\n]','',str(self.mechanismNames))))
        # # MODIFIED 2/17/17 OLD:
        # variable = [list(i) for i in self.variable]
        # print("- variable: {1}".format(self, re.sub('[\[,\],\n]','',
        #                                          str([[float("{:0.3}".format(float(i)))
        #                                                for i in value] for value in variable]))))
        # MODIFIED 2/17/17 NEW:
        if input is None:
            input = self.input
        print("- input: {}".format(input))
        # MODIFIED 2/17/17 END


    def _report_mechanism_execution(self, mechanism):
        # DEPRECATED: Reporting of mechanism execution relegated to individual mechanism prefs
        pass
        # print("\n{0} executed {1}:\n- output: {2}\n\n--------------------------------------".
        #       format(self.name,
        #              mechanism.name,
        #              re.sub('[\[,\],\n]','',
        #                     str(mechanism.outputState.value))))

    def _report_process_completion(self, separator=False):

        print("\n\'{}' completed:\n- output: {}".
              format(append_type_to_name(self),
                     re.sub(r'[\[,\],\n]','',str([float("{:0.3}".format(float(i))) for i in self.output_state.value]))))

        if self.learning:
            from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms.ComparatorMechanism \
                import MSE
            for mech in self.targetMechanisms:
                if not MSE in mech.output_states:
                    continue
                print("\n- MSE: {:0.3}".
                      format(float(mech.output_states[MSE].value)))

        elif separator:
            print("\n\n****************************************\n")

    def show(self, options=None):
        """Print list of all mechanisms in the process, followed by its `ORIGIN` and `TERMINAL` mechanisms.

        Arguments
        ---------

        options : InspectionOptions
            [TBI]
        """

        # # IMPLEMENTATION NOTE:  Stub for implementing options:
        # if options and self.InspectOptions.ALL_OUTPUT_LABELS in options:
        #     pass

        print ("\n---------------------------------------------------------")
        print ("\n{}\n".format(self.name))

        print ("\tLearning enabled: {}".format(self._learning_enabled))

        # print ("\n\tMechanisms:")
        # for mech_name in self.mechanismNames:
        #     print ("\t\t{}".format(mech_name))

        print ("\n\tMechanisms:")
        for object_item in self._mechs:
            print ("\t\t{} (phase: {})".format(object_item.name, object_item.phase))


        print ("\n\tOrigin mechanism: ".format(self.name))
        for object_item in self.originMechanisms.mechs_sorted:
            print("\t\t{} (phase: {})".format(object_item.name, object_item.phase))

        print ("\n\tTerminal mechanism: ".format(self.name))
        for object_item in self.terminalMechanisms.mechs_sorted:
            print("\t\t{} (phase: {})".format(object_item.name, object_item.phase))
            for output_state_name in object_item.output_states:
                print("\t\t\t{0}".format(output_state_name))

        print ("\n---------------------------------------------------------")



    @property
    def mechanisms(self):
        return self._allMechanisms.mechanisms

    @property
    def mechanismNames(self):
        return self._allMechanisms.names

    @property
    def variableInstanceDefault(self):
        return self._variableInstanceDefault

    @variableInstanceDefault.setter
    def variableInstanceDefault(self, value):
        assigned = -1
        try:
            value
        except ValueError as e:
            pass
        self._variableInstanceDefault = value

    # @property
    # def input_value(self):
    #     return self.variable

    @property
    def output_state(self):
        return self.lastMechanism.output_state

    @property
    def output(self):
        # FIX: THESE NEED TO BE PROPERLY MAPPED
        return np.array(list(item.value for item in self.lastMechanism.output_states.values()))

    @property
    def numPhases(self):
        return self._phaseSpecMax + 1

class ProcessInputState(OutputState):
    """Encodes either an input to or target for the process and transmits it to the corresponding mechanism

    Each instance encodes one of the following:
    - an item of the `input <Process.input>` to the process (a 1d array in the 2d input array) and provides it to a
        `MappingProjection` that projects to one or more `input_states <Mechanism.Mechanism_Base.input_states>` of the
        `ORIGIN` mechanism in the process.
    - a `target <Process.target>` to the process (also a 1d array) and provides it to a `MappingProjection` that
         projects to the `TARGET` mechanism of the process.

    (See :ref:`Process_Input_And_OuputProcess` for an explanation of the mapping from processInputStates to
    `ORIGIN` mechanism input_states when there is more than one process input value and/or mechanism inputState)

    .. Declared as a sublcass of OutputState so that it is recognized as a legitimate sender to a Projection
       in Projection._instantiate_sender()

       self.value is used to represent the corresponding item of the input arg to process.execute or process.run

    """
    def __init__(self, owner=None, variable=None, name=None, prefs=None):
        """Pass variable to MappingProjection from Process to first Mechanism in Pathway

        :param variable:
        """
        if not name:
            self.name = owner.name + "_" + kwProcessInputState
        else:
            self.name = owner.name + "_" + name
        self.prefs = prefs
        self.efferents = []
        self.owner = owner
        self.value = variable
        # MODIFIED 2/17/17 NEW:
        # self.owner.input = self.value
        # MODIFIED 2/17/17 END
        # self.path_afferents = []
        # from PsyNeuLink.Components.States.OutputState import PRIMARY_OUTPUT_STATE
        # from PsyNeuLink.Components.Functions.Function import Linear
        # self.index = PRIMARY_OUTPUT_STATE
        # self.calculate = Linear

    # MODIFIED 2/1717 NEW:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, assignment):
        self._value = assignment
        self.owner._update_input()
    # MODIFIED 2/1717 END


ProcessTuple = namedtuple('ProcessTuple', 'process, input')


class ProcessList(UserList):
    """Provides access to items in (process, process_input) tuples in a list of ProcessTuples
    """

    def __init__(self, owner, tuples_list:list):
        super().__init__()
        self.owner = owner
        for item in tuples_list:
            if not isinstance(item, ProcessTuple):
                raise ProcessError("{} in the tuples_list arg of ProcessList() is not a ProcessTuple".format(item))
        self.process_tuples = tuples_list

    def __getitem__(self, item):
        # return list(self.process_tuples[item])[PROCESS]
        return self.process_tuples[item].process


    def __setitem__(self, key, value):
        raise ("MyList is read only ")

    def __len__(self):
        return (len(self.process_tuples))

    def _get_tuple_for_process(self, process):
        """Return first process tuple containing specified process from list of process_tuples
        """
        # FIX:
        # if list(item[MECHANISM] for item in self.mechs).count(mech):
        #     if self.owner.verbosePref:
        #         print("PROGRAM ERROR:  {} found in more than one object_item in {} in {}".
        #               format(append_type_to_name(mech), self.__class__.__name__, self.owner.name))
        return next((ProcessTuple for ProcessTuple in self.process_tuples if ProcessTuple.process is process), None)

    @property
    def process_tuples_sorted(self):
        """Return list of mechs sorted by mechanism name"""
        return sorted(self.process_tuples, key=lambda process_tuple: process_tuple[0].name)

    @property
    def processes(self):
        """Return list of all processes in ProcessList
        """
        # MODIFIED 11/1/16 OLD:
        return list(item.process for item in self.process_tuples)
        # # MODIFIED 11/1/16 NEW:
        # return sorted(list(item.process for item in self.process_tuples), key=lambda process: process.name)
        # MODIFIED 11/1/16 END

    @property
    def processNames(self):
        """Return names of all processes in ProcessList
        """
        # MODIFIED 11/1/16 OLD:
        return list(item.process.name for item in self.process_tuples)
        # # MODIFIED 11/1/16 NEW:
        # return sorted(list(item.process.name for item in self.process_tuples))
        # MODIFIED 11/1/16 END

    @property
    def _mechs(self):
        return self.__mechs__

    @_mechs.setter
    def _mechs(self, value):
        self.__mechs__ = value
