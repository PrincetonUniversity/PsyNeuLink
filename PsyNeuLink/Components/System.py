
# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *****************************************    SYSTEM MODULE    ********************************************************

"""
..
    Sections:
      * `System_Overview`
      * `System_Creation`
      * `System_Structure`
         * `System_Graph`
         * `System_Mechanisms`
      * `System_Execution`
         * `System_Execution_Order`
         * `System_Execution_Phase`
         * `System_Execution_Input_And_Initialization`
         * `System_Execution_Learning`
         * `System_Execution_Control`
      * `System_Class_Reference`


.. _System_Overview:

Overview
--------

A system is a collection of `processes <Process>` that are executed together.  Executing a system executes all of the
`mechanisms <Mechanism>` in its processes in a structured order.  `Projections <Projection>` between mechanisms in
different processes within the system are permitted, as are recurrent projections, but projections from mechanisms
in other systems are ignored (PsyNeuLink does not support ESP).  A system can include three types of mechanisms:

* `ProcessingMechanism`
    These receive input from one or more projections, transform the input in some way,
    and assign the result as their output.

* `ControlMechanism`
    These monitor the output of other mechanisms for use in controlling the parameters of other mechanisms or their
    functions.

* `MonitoringMechanism`
    These monitor the output of other mechanisms for use in modifying the parameters of projections (learning)

.. _System_Creation:

Creating a System
-----------------

Systems are created by calling the :py:func:`system` function.  If no arguments are provided, a system with a
single process containing a single :ref:`default mechanism <LINK>` will be returned. Whenever a system is created,
a `ControlMechanism <ControlMechanism>` is created for it and assigned as its `controller`.  The controller can be
specified by assigning an existing ControlMechanism to the  :keyword:`controller`  argument of the system's constructor,
or specifying a class of ControlMechanism;  if none is specified, a `DefaultControlMechanism` is created.

.. _System_Structure:

Structure
---------

.. _System_Graph:

Graph
~~~~~

When an instance of a system is created, a graph is constructed that describes the connections (edges) among its
mechanisms (nodes).  The graph is assigned to the system's `graph` attribute.  This is a dictionary of dependencies,
that can be passed to graph theoretical tools for analysis.  A system can have recurrent processing pathways, such as
feedback loops, in which case the system will have a cyclic graph.  PsyNeuLink also uses the graph of a
system to determine the order in which its mechanisms are executed.  In order to do so in an orderly manner, however,
the graph must be acyclic.  So, for execution, PsyNeuLink constructs an `executionGraph` from the system's `graph`.
If the  system is acyclic, these are the same.  If the system is cyclic, then the `executionGraph` is a subset of the
`graph` in which the dependencies (edges) associated with projections that close a loop have been removed. Note that
this only impacts the order of execution;  the projections themselves remain in effect, and will be fully functional
during the execution of the affected mechanisms (see :ref:`System_Execution` below for a more detailed description).

.. _System_Mechanisms:

Mechanisms
~~~~~~~~~~

The mechanisms in a system are assigned designations based on the position they occupy in the `graph`
and/or the role they play in a system:

    `ORIGIN`: receives input to the system, and does not receive projections from any other ProcessingMechanisms;

    `TERMINAL`: provides output from the system, and does not send projections to any other ProcessingMechanisms;

    `SINGLETON`: both an `ORIGIN` and a `TERMINAL` mechanism;

    `INITIALIZE_CYCLE`: sends a projection that closes a recurrent loop;
    can be assigned an initial value;

    `CYCLE`: receives a projection that closes a recurrent loop;

    `CONTROL`: monitors the value of another mechanism for use in controlling parameter values;

    `MONITORING`: monitors the value of another mechanism for use in learning;

    `TARGET`: ObjectiveMechanism that monitors a `TERMINAL` mechanism of a process

    `INTERNAL`: ProcessingMechanism that does not fall into any of the categories above.

    .. note:: Any `ORIGIN` and `TERMINAL` mechanisms of a system must be, respectively,
       the `ORIGIN` or `TERMINAL` of any process(es) to which they belong.  However, it is not
       necessarily the case that the `ORIGIN` and/or `TERMINAL` mechanism of a process is also the
       `ORIGIN` and/or `TERMINAL` of a system to which the process belongs (see the Chain example below).

    .. note: designations are stored in the mechanism.systems attribute (see _instantiate_graph below, and Mechanism)


COMMENT:
    .. _System_Control:
    Control
    ~~~~~~~

    .. _System_Learning:
    Learning
    ~~~~~~~~

    Based on process

COMMENT

.. _System_Full_Fig:

**Components of a System**

.. figure:: _static/System_full_fig.pdf
   :alt: Overview of major PsyNeuLink components
   :scale: 75 %

   Two :doc:`processes <Process>` are shown, both belonging to the same :doc:`system <System>`.  Each process has a
   series of :doc:`ProcessingMechanisms <ProcessingMechanism>` linked by :doc:`MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism.  Each ProcessingMechanism is labeled with its designation in
   the system.  The `TERMINAL` mechanism for both processes projects to an `ObjectiveMechanism` that is used to
   drive `learning <LearningProjection>` in Process B. It also projects to a separate ObjectiveMechanism that is used
   for control of ProcessingMechanisms in both Processes A and B.  Note that the mechanisms and
   projections responsible for learning and control belong to the system and can monitor and/or control mechanisms
   belonging to more than one process (as shown for control in this figure).

.. _System_Execution:

Execution
---------

A system can be executed by calling either its `execute <System_Base.execute>` or `run <System_Base.execute>` methods.
`execute <System_Base.execute>` executes each mechanism in the system once, whereas `run <System_Base.execute>`
allows a series of executions to be carried out.

.. _System_Execution_Order:

Order
~~~~~
Mechanisms are executed in a topologically sorted order, based on the order in which they are listed in their
processes. When a mechanism is executed, it receives input from any other mechanisms that project to it within the
system,  but not from mechanisms outside the system (PsyNeuLink does not support ESP).  The order of execution is
determined by the system's `executionGraph` attribute, which is a subset of the system's `graph` that has been
"pruned" to be acyclic (i.e., devoid of recurrent loops).  While the `executionGraph` is acyclic, all recurrent
projections in the system remain intact during execution and can be
`initialized <System_Execution_Input_And_Initialization>` at the start of execution.

.. _System_Execution_Phase:

Phase
~~~~~
Execution occurs in passes through a system called *phases*.  Each phase corresponds to a single `time_step <LINK>`.
When executing a system in `trial <LINK>` mode, a trial is defined as the number of phases (time_steps) required to
execute a trial of every mechanism in the system.  During each phase of execution, only the mechanisms assigned to
that phase are executed.   Mechanisms are assigned a phase where they are listed in the `pathway` of a
`process <Process>`. When a mechanism is executed, it receives input from any other mechanisms that project to it
within the system.

.. _System_Execution_Input_And_Initialization:

Input and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~
The input to a system is specified in the :keyword:`input` argument of either its `execute <System_Base.execute>` or
`run <System_Base.run>` method. In both cases, the input for a single trial must be a list or ndarray of values,
each of which is an appropriate input for the corresponding `ORIGIN` mechanism (listed in
`originMechanisms <System_Base.originMechanisms>`). If the `execute <System_Base.execute>` method is used,
input for only a single trial is provided, and only a single trial is executed.  The `run <System_Base.run>` method
can be used for a sequence of executions (time_steps or trials), by providing it with a list or ndarray of inputs,
one for each round of execution.  In both cases, two other types of input can be provided:  a list or ndarray of
initialization values, and a list or ndarray of target values. Initialization values are assigned, at the start
of execution, as input to mechanisms that close recurrent loops (designated as `INITIALIZE_CYCLE`, and listed in
`recurrentInitMechanisms`), and target values are assigned as the TARGET input of the system's `TARGET` mechanisms
(see learning below;  also, see `Run` for additional details of formatting input specifications).

.. _System_Execution_Learning:

Learning
~~~~~~~~
The system will execute learning if it is specified for any process in the system.  The system's `learning` attribute
indicates whether learning is enabled for the system. Learning is executed for any
components (individual projections or processes) for which it is specified after all processing mechanisms in the
system have been executed, but before the controller is executed (see below). The stimuli (both inputs and targets for
learning) can be specified in either of two formats, sequence or mechanism, that are described in the :doc:`Run` module;
see `Run_Inputs` and `Run_Targets`).  Both formats require that an input be provided for each `ORIGIN` mechanism of
the system (listed in its `originMechanisms <System_Base.originMechanisms>` attribute).  If the targets are specified
in sequence or mechanism format, one target must be provided for each `TARGET` mechanism (listed in its
`targetMechanisms <System_Base.targetMechanisms>` attribute).  Targets can also be specified in a
`function format <Run_Targets_Function_Format>`, which generates a target for each execution of the mechanism.

.. note::
   A :py:data:`targetMechanism <Process.Process_Base.targetMechanisms>` of a process is not necessarily a
   :py:data:`targetMechanism <System_Base.targetMechanisms>` of the system to which it belongs
   (see :ref:`LearningProjection_Targets`).

.. _System_Execution_Control:

Control
~~~~~~~
Every system is associated with a single `controller`.  The controller monitors the outputState(s) of one or more
mechanisms in the system (listed in its `monitored_output_states` attribute), and uses that information to set the
value of parameters for those or other mechanisms in the system, or their functions
(see :ref:`ControlMechanism_Monitored_OutputStates` for a description of how to specify which outputStates are
monitored, and :ref:`ControlProjection_Creation` for specifying parameters to be controlled). The controller is
executed after all other mechanisms in the system are executed, and sets the values of any parameters that it
controls, which then take effect in the next round of execution.

COMMENT:
   Examples
   --------
   XXX ADD EXAMPLES HERE FROM 'System Graph and Input Test Script'
   .. note::  All of the example systems below use the following set of mechanisms.  However, in practice, they must be
      created separately for each system;  using the same mechanisms and processes in multiple systems can produce
      confusing results.

   Module Contents
   system() factory method:  instantiate system
   System_Base: class definition
COMMENT

.. _System_Class_Reference:

Class Reference
---------------

"""

import math
import re
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import LearningMechanism
from collections import OrderedDict
from collections import UserList, Iterable
from toposort import *

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismList, MechanismTuple,\
                                                       OBJECT_ITEM, PARAMS_ITEM, PHASE_ITEM
from PsyNeuLink.Components.Mechanisms.Mechanism import MonitoredOutputStatesOption
from PsyNeuLink.Components.Process import ProcessInputState, ProcessList, ProcessTuple
from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection, _is_learning_spec
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Globals.TimeScale import TimeScale
from PsyNeuLink.scheduling.Scheduler import Scheduler

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# inspect() keywords
SCHEDULER = "scheduler"
PROCESSES = 'processes'
MECHANISMS = 'mechanisms'
ORIGIN_MECHANISMS = 'origin_mechanisms'
INPUT_ARRAY = 'input_array'
RECURRENT_MECHANISMS = 'recurrent_mechanisms'
RECURRENT_INIT_ARRAY = 'recurrent_init_array'
TERMINAL_MECHANISMS = 'terminal_mechanisms'
OUTPUT_STATE_NAMES = 'output_state_names'
OUTPUT_VALUE_ARRAY = 'output_value_array'
NUM_PHASES_PER_TRIAL = 'num_phases'
MONITORING_MECHANISMS = 'monitoring_mechanisms'
TARGET_MECHANISMS = 'target_mechanisms'
LEARNING_PROJECTION_RECEIVERS = 'learning_projection_receivers'
CONTROL_MECHANISMS = 'control_mechanisms'
CONTROL_PROJECTION_RECEIVERS = 'control_projection_receivers'

SystemRegistry = {}

kwSystemInputState = 'SystemInputState'


class SystemWarning(Warning):
     def __init__(self, error_value):
         self.error_value = error_value

class SystemError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


# FIX:  IMPLEMENT DEFAULT PROCESS
# FIX:  NEED TO CREATE THE PROJECTIONS FROM THE PROCESS TO THE FIRST MECHANISM IN PROCESS FIRST SINCE,
# FIX:  ONCE IT IS IN THE GRAPH, IT IS NOT LONGER EASY TO DETERMINE WHICH IS WHICH IS WHICH (SINCE SETS ARE NOT ORDERED)

from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components import SystemDefaultControlMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism


# System factory method:
@tc.typecheck
def system(default_input_value=None,
           processes:list=[],
           scheduler = None,
           initial_values:dict={},
           controller=SystemDefaultControlMechanism,
           enable_controller:bool=False,
           monitor_for_control:list=[MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES],
           # learning:tc.optional(_is_learning_spec)=None,
           learning_rate:tc.optional(parameter_spec)=None,
           targets:tc.optional(tc.any(list, np.ndarray))=None,
           params:tc.optional(dict)=None,
           name:tc.optional(str)=None,
           prefs:is_pref_set=None,
           context=None):
    """
    system(                                   \
    default_input_value=None,                 \
    processes=None,                           \
    initial_values=None,                      \
    controller=SystemDefaultControlMechanism, \
    enable_controller=:keyword:`False`,       \
    monitor_for_control=`None`,               \
    learning_rate=None,                       \
    targets=None,                             \
    params=None,                              \
    name=None,                                \
    prefs=None)

    COMMENT:
       VERSION WITH learning
        system(                                   \
        default_input_value=None,                 \
        processes=None,                           \
        initial_values=None,                      \
        controller=SystemDefaultControlMechanism, \
        enable_controller=:keyword:`False`,       \
        monitor_for_control=`None`,               \
        learning=None,                            \
        targets=None                              \
        params=None,                              \
        name=None,                                \
        prefs=None)
    COMMENT

    Factory method for System: returns instance of System.

    If called with no arguments, returns an instance of System with a single default process and mechanism;
    if called with a name string, that is used as the name of the instance of System returned;
    if a params dictionary is included, it is passed to the instantiated system.

    See :class:`System_Base` for class description

    Arguments
    ---------

    default_input_value : list or ndarray of values : default default input for `ORIGIN` mechanism of each Process
        the input to the system if none is provided in a call to the `execute <System_Base.exeucte> or
        `run <System_Base.run> methods. Should contain one item corresponding to the input of each `ORIGIN` mechanism
        in the system.
        COMMENT:
            REPLACE DefaultProcess BELOW USING Inline markup
        COMMENT

    processes : list of process specifications : default list('DefaultProcess')
        a list of the processes to include in the system.
        Each process specification can be an instance, the class name (creates a default Process), or a specification
        dictionary (see `Process` for details).

    initial_values : dict of mechanism:value entries
        a dictionary of values used to initialize mechanisms that close recurrent loops (designated as
        `INITIALIZE_CYCLE`). The key for each entry is a mechanism object, and the value is a number,
        list or 1d np.array that must be compatible with the format of the first item of the mechanism's value
        (i.e., mechanism.value[0]).

    controller : ControlMechanism : default DefaultController
        specifies the `ControlMechanism` used to monitor the value of the outputState(s) for mechanisms specified in
        `monitor_for_control`, and that specify the value of `ControlProjections` in the system.

    enable_controller :  bool : default :keyword:`False`
        specifies whether the `controller` is executed during system execution.

    monitor_for_control : list of OutputState objects or specifications : default None
        specifies the outputStates of the `TERMINAL` mechanisms in the system to be monitored by its `controller`
        (see `ControlMechanism_Monitored_OutputStates` for specifying the `monitor_for_control` argument).

    COMMENT:
        learning : Optional[LearningProjection spec]
            implements `learning <LearningProjection_CreationLearningSignal>` for all processes in the system.
    COMMENT

    learning_rate : float : None
        set the learning rate for all mechanisms in the system (see `learning_rate` attribute for additional
        information).

    targets : Optional[List[List]], 2d np.ndarray] : default ndarrays of zeroes
        the values assigned to the TARGET input of each `TARGET` mechanism in the system (listed in its
        `targetMechanisms` attribute).  There must be the same number of items as there are `targetMechanisms`,
        and each item must have the same format (length and number of elements) as the TARGET input
        for each of the corresponding `TARGET` mechanism.

    params : dict : default None
        a `parameter dictionary <ParameterState_Specifying_Parameters>` that can include any of the parameters above;
        the parameter's name should be used as the key for its entry. Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default System-<index>
        a string used for the name of the system
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names)

    prefs : PreferenceSet or specification dict : System.classPreferences
        the `PreferenceSet` for system (see :doc:`ComponentPreferenceSet <LINK>` for specification of PreferenceSet)

    COMMENT:
    context : str : default None
        string used for contextualization of instantiation, hierarchical calls, executions, etc.
    COMMENT

    Returns
    -------
    instance of System : System

    """


    # Called with descriptor keyword
    if not processes:
        processes = [process()]

    return System_Base(default_input_value=default_input_value,
                       processes=processes,
                       controller=controller,
                       scheduler= scheduler,
                       initial_values=initial_values,
                       enable_controller=enable_controller,
                       monitor_for_control=monitor_for_control,
                       # learning=learning,
                       learning_rate=learning_rate,
                       targets=targets,
                       params=params,
                       name=name,
                       prefs=prefs,
                       context=context)


class System_Base(System):
    """
    System_Base(                              \
    default_input_value=None,                 \
    processes=None,                           \
    initial_values=None,                      \
    controller=SystemDefaultControlMechanism, \
    enable_controller=:keyword:`False`,       \
    monitor_for_control=`None`,               \
    learning_rate=None,                       \
    targets=None,                             \
    params=None,                              \
    name=None,                                \
    prefs=None)

    COMMENT:
        VERSION WITH learning
        System_Base(                              \
        default_input_value=None,                 \
        processes=None,                           \
        initial_values=None,                      \
        controller=SystemDefaultControlMechanism, \
        enable_controller=:keyword:`False`,       \
        monitor_for_control=`None`,               \
        learning=None,                            \
        targets=None,                             \
        params=None,                              \
        name=None,                                \
        prefs=None)
    COMMENT

    Abstract class for System.

    .. note::
       Systems should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the :func:`system` factory method (see it for description of parameters).

    COMMENT:
        Description
        -----------
            System is a Category of the Component class.
            It implements a System that is used to execute a collection of processes.

       Class attributes
       ----------------
        + componentCategory (str): kwProcessFunctionCategory
        + className (str): kwProcessFunctionCategory
        + suffix (str): " <kwMechanismFunctionCategory>"
        + registry (dict): ProcessRegistry
        + classPreference (PreferenceSet): ProcessPreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + variableClassDefault = inputValueSystemDefault                     # Used as default input value to Process)
        + paramClassDefaults = {kwProcesses: [Mechanism_Base.defaultMechanism],
                                CONTROLLER: DefaultController,
                                TIME_SCALE: TimeScale.TRIAL}
       Class methods
       -------------
        - _validate_variable(variable, context):  insures that variable is 3D np.array (one 2D for each Process)
        - _instantiate_attributes_before_function(context):  calls self._instantiate_graph
        - _instantiate_function(context): validates only if self.prefs.paramValidationPref is set
        - _instantiate_graph(input, context):  instantiates Processes in self.process and constructs executionList
        - identify_origin_and_terminal_mechanisms():  assign self.originMechanisms and self.terminalMechanisms
        - _assign_output_states():  assign outputStates of System (currently = terminalMechanisms)
        - execute(input, time_scale, context):  executes Mechanisms in order specified by executionList
        - variableInstanceDefaults(value):  setter for variableInstanceDefaults;  does some kind of error checking??

       SystemRegistry
       --------------
        Register in SystemRegistry, which maintains a dict for the subclass, a count for all instances of it,
         and a dictionary of those instances

        TBI: MAKE THESE convenience lists, akin to self.terminalMechanisms
        + input (list): contains Process.input for each process in self.processes
        + output (list): containts Process.ouput for each process in self.processes
        [TBI: + input (list): each item is the Process.input object for the corresponding Process in self.processes]
        [TBI: + outputs (list): each item is the Process.output object for the corresponding Process in self.processes]
    COMMENT

    Attributes
    ----------

    componentType : SYSTEM

    processes : list of Process objects
        list of processes in the system specified by the `process` parameter.

        .. can be appended with prediction processes by EVCMechanism
           used with self.input to constsruct self.process_tuples

        .. _processList : ProcessList
            Provides access to (process, input) tuples.
            Derived from self.input and self.processes.
            Used to construct :py:data:`executionGraph <System_Base.executionGraph>` and execute the System

    controller : ControlMechanism : default DefaultController
        the ControlMechanism used to monitor the value of the outputState(s) for mechanisms specified in
        ``monitor_for_control`` argument, and specify the value of ControlProjections in the system.

    enable_controller :  bool : default :keyword:`False`
        determines whether the `controller` is executed during system execution.

    learning : bool : default False
        indicates whether learning is being used;  is set to True if learning is specified for any processes
        in the system or for the system itself.

    learning_rate : float : default None
        determines the learning rate for all mechanisms in the system.  This overrides any values set for the
        function of individual LearningProjections, and persists for all subsequent runs of the system.  If it is
        set to None, then the learning_rate is determined by the last value assigned to each LearningProjection
        (either directly, or following a run of any process or system to which the LearningProjection belongs and
        for which a learning_rate was set).

    targets : 2d nparray : default zeroes
        used as template for the values of the system's `targetInputStates`, and to represent the targets specified in
        the :keyword:`targets` argument of system's `execute <System.execute>` and `run <System.run>` methods.

    graph : OrderedDict
        contains a graph of all of the mechanisms in the system.
        Each entry specifies a set of <Receiver>: {sender, sender...} dependencies.
        The key of each entry is a receiver mech_tuple, and
        the value is a set of mech_tuples that send projections to that receiver.
        If a key (receiver) has no dependents, its value is an empty set.

    executionGraph : OrderedDict
        contains an acyclic subset of the system's `graph`, hierarchically organized by a toposort.
        Used to specify the order in which mechanisms are executed.

    execution_sets : list of sets
        contains a list of mechanism sets.
        Each set contains mechanism to be executed at the same time.
        The sets are ordered in the sequence with which they should be executed.

    executionList : list of Mechanism objects
        contains a list of mechanisms in the order in which they are executed.
        The list is a random sample of the permissible orders constrained by the `executionGraph`.

    mechanisms : list of Mechanism objects
        contains a list of all mechanisms in the system.

        .. property that points to _allMechanisms.mechanisms (see below)

    mechanismsDict : Dict[Mechanism:Process]
        contains a dictionary of all mechanisms in the system, listing the processes to which they belong.
        The key of each entry is a `Mechanism` object, and the value of each entry is a list of `processes <Process>`.

        .. Note: the following attributes use lists of tuples (mechanism, runtime_param, phaseSpec) and MechanismList
              xxx_mech_tuples are lists of tuples defined in the Process pathways;
                  tuples are used because runtime_params and phaseSpec are attributes that need
                  to be able to be specified differently for the same mechanism in different contexts
                  and thus are not easily managed as mechanism attributes
              xxxMechanismLists point to MechanismList objects that provide access to information
                  about the mechanism <type> listed in mech_tuples (i.e., the mechanisms, names, etc.)

        .. _all_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all mechanisms in the system (serve as keys in self.graph).

        .. _allMechanisms : MechanismList
            Contains all mechanisms in the system (based on _all_mech_tuples).

        .. _origin_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all ORIGIN mechanisms in the system.

        .. _terminal_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all TERMINAL mechanisms in the system.

        .. _monitoring_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all MonitoringMechanisms in the system (used for learning).

        .. _target_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all TARGET `ObjectiveMechanisms <ObjectiveMechanism>`  in the system that are a `TERMINAL`
            for at least on process to which it belongs and that process has learning enabled --  the criteria for
            being a target used in learning.

        .. _learning_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all LearningMechanisms in the system (used for learning).

        .. _control_mech_tuple : list of a single (mechanism, runtime_param, phaseSpec) tuple
            Tuple for the controller in the system.

    originMechanisms : MechanismList
        contains all `ORIGIN` mechanisms in the system (i.e., that don't receive projections from any other
        mechanisms.

        .. based on _origin_mech_tuples
           system.input contains the input to each `ORIGIN` mechanism

    terminalMechanisms : MechanismList
        contains all `TERMINAL` mechanisms in the system (i.e., that don't project to any other ProcessingMechanisms).

        .. based on _terminal_mech_tuples
           system.ouput contains the output of each TERMINAL mechanism

    recurrentInitMechanisms : MechanismList
        contains mechanisms with recurrent projections that are candidates for
        `initialization <System_Execution_Input_And_Initialization>`.

    monitoringMechanisms : MechanismList)
        contains all `MONITORING` mechanisms in the system (used for learning).
        COMMENET:
            based on _monitoring_mech_tuples)
        COMMENT

    targetMechanisms : MechanismList)
        contains all `TARGET` mechanisms in the system (used for learning.
        COMMENT:
            based on _target_mech_tuples)
        COMMENT

    targetInputStates : List[SystemInputState]
        one item for each `TARGET` mechanism in the system (listed in `targetMechanisms`).  Used to represent the
        :keyword:`targets` specified in the system's `execute <System.execute>` and `run <System.run>` methods, and
        provide their values to the the TARGET inputState of each `TARGET` mechanism during execution.

    COMMENT:
       IS THIS CORRECT:
    COMMENT

    controlMechanisms : MechanismList
        contains `controller` of the system
        COMMENT:
            ??and any other `ControlMechanisms <ControlMechanism>` in the system
            (based on _control_mech_tuples).
        COMMENT

    value : 3D ndarray
        contains an array of 2D arrays, each of which is the `outputValue `of a `TERMINAL` mechanism in the system.

        .. _phaseSpecMax : int
            Maximum phase specified for any mechanism in system.  Determines the phase of the last (set of)
            ProcessingMechanism(s) to be executed in the system.

    numPhases : int
        number of phases for system (read-only).

        .. implemented as an @property attribute; = _phaseSpecMax + 1

    initial_values : list or ndarray of values :  default array of zero arrays
        values used to initialize mechanisms that close recurrent loops (designated as `INITIALIZE_CYCLE`).
        Must be the same length as the list of `INITIALIZE_CYCLE` mechanisms in the system contained in
        `recurrentInitMechanisms`.

    timeScale : TimeScale  : default TimeScale.TRIAL
        determines the default `TimeScale` value used by mechanisms in the system.

    results : List[outputState.value]
        list of return values (outputState.value) from the sequence of executions.

    name : str : default System-<index>
        the name of the system;
        Specified in the `name` argument of the constructor for the system;
        if not is specified, a default is assigned by SystemRegistry
        (see :doc:`Registry <LINK>` for conventions used in naming, including for default and duplicate names).


    prefs : PreferenceSet or specification dict : System.classPreferences
        the `PreferenceSet` for system.
        Specified in the `prefs` argument of the constructor for the system;  if it is not specified, a default is
        assigned using `classPreferences` defined in __init__.py
        (see :ref:`PreferenceSet <LINK>` for details).

    """

    componentCategory = kwSystemComponentCategory
    className = componentCategory
    suffix = " " + className
    componentType = "System"

    registry = SystemRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # These will override those specified in CategoryDefaultPreferences
    # classPreferences = {
    #     kwPreferenceSetName: 'SystemCustomClassPreferences',
    #     kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # Use inputValueSystemDefault as default input to process
    # variableClassDefault = inputValueSystemDefault
    variableClassDefault = None

    paramClassDefaults = Component.paramClassDefaults.copy()
    paramClassDefaults.update({TIME_SCALE: TimeScale.TRIAL})

    @tc.typecheck
    def __init__(self,
                 default_input_value=None,
                 processes=None,
                 initial_values=None,
                 controller=SystemDefaultControlMechanism,
                 enable_controller=False,
                 monitor_for_control=None,
                 # learning=None,
                 learning_rate=None,
                 targets=None,
                 params=None,
                 name=None,
                 scheduler = None,
                 prefs:is_pref_set=None,
                 context=None):

        processes = processes or []
        monitor_for_control = monitor_for_control or [MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES]

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self._assign_args_to_param_dicts(processes=processes,
                                                  initial_values=initial_values,
                                                  controller=controller,
                                                  enable_controller=enable_controller,
                                                  monitor_for_control=monitor_for_control,
                                                  learning_rate=learning_rate,
                                                  targets=targets,
                                                  params=params)

        self.function = self.execute
        self.outputStates = {}
        self._phaseSpecMax = 0
        self.stimulusInputStates = []
        self.inputs = []
        self.current_input = None
        self.targetInputStates = []
        self.targets = None
        self.current_targets = None
        self.learning = False
        self.scheduler = scheduler

        register_category(entry=self,
                          base_class=System_Base,
                          name=name,
                          registry=SystemRegistry,
                          context=context)

        if not context:
            # context = INITIALIZING + self.name
            context = INITIALIZING + self.name + kwSeparator + SYSTEM_INIT

        super().__init__(variable_default=default_input_value,
                         param_defaults=params,
                         name=self.name,
                         prefs=prefs,
                         context=context)

        self._execution_id = None

        # Get/assign controller

        # Controller is DefaultControlMechanism
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.DefaultControlMechanism import DefaultControlMechanism
        if self.paramsCurrent[CONTROLLER] is DefaultControlMechanism:
            # Get DefaultController from MechanismRegistry
            from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismRegistry
            self.controller = list(MechanismRegistry[DEFAULT_CONTROL_MECHANISM].instanceDict.values())[0]
        # Controller is not DefaultControlMechanism
        else:
            # Instantiate specified controller
            # MODIFIED 11/6/16 OLD:
            self.controller = self.paramsCurrent[CONTROLLER](params={SYSTEM: self})
            # # MODIFIED 11/6/16 NEW:
            # self.controller = self.paramsCurrent[CONTROLLER](system=self)
            # MODIFIED 11/6/16 END

        # Check whether controller has input, and if not then disable
        try:
            has_input_states = bool(self.controller.inputStates)
        except:
            has_input_states = False
        if not has_input_states:
            # If controller was enabled (and verbose is set), warn that it has been disabled
            if self.enable_controller and self.prefs.verbosePref:
                print("{} for {} has no inputStates, so controller will be disabled".
                      format(self.controller.name, self.name))
            self.enable_controller = False


        # Compare _phaseSpecMax with controller's phaseSpec, and assign default if it is not specified
        try:
            # Get phaseSpec from controller
            self._phaseSpecMax = max(self._phaseSpecMax, self.controller.phaseSpec)
        except (AttributeError, TypeError):
            # Controller phaseSpec not specified
            try:
                # Assign System specification of Controller phaseSpec if provided
                self.controller.phaseSpec = self.paramsCurrent[CONROLLER_PHASE_SPEC]
                self._phaseSpecMax = max(self._phaseSpecMax, self.controller.phaseSpec)
            except:
                # No System specification, so use System max as default
                self.controller.phaseSpec = self._phaseSpecMax

        # IMPLEMENT CORRECT REPORTING HERE
        # if self.prefs.reportOutputPref:
        #     print("\n{0} initialized with:\n- pathway: [{1}]".
        #           # format(self.name, self.pathwayMechanismNames.__str__().strip("[]")))
        #           format(self.name, self.names.__str__().strip("[]")))

    def _validate_variable(self, variable, context=None):
        """Convert variableClassDefault and self.variable to 2D np.array: one 1D value for each input state
        """
        super(System_Base, self)._validate_variable(variable, context)

        # # MODIFIED 6/26/16 OLD:
        # # Force System variable specification to be a 2D array (to accommodate multiple input states of 1st mech(s)):
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        # self.variable = convert_to_np_array(self.variable, 2)
        # FIX:  THIS CURRENTLY FAILS:
        # # MODIFIED 6/26/16 NEW:
        # # Force System variable specification to be a 3D array (to accommodate input states for each Process):
        # self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 3)
        # self.variable = convert_to_np_array(self.variable, 3)
        # MODIFIED 10/2/16 NEWER:
        # Force System variable specification to be a 2D array (to accommodate multiple input states of 1st mech(s)):
        if variable is None:
            return
        self.variableClassDefault = convert_to_np_array(self.variableClassDefault, 2)
        self.variable = convert_to_np_array(self.variable, 2)

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate controller, processes and initial_values
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        controller = target_set[CONTROLLER]
        if (not isinstance(controller, ControlMechanism_Base) and
                not (inspect.isclass(controller) and issubclass(controller, ControlMechanism_Base))):
            raise SystemError("{} (controller arg for \'{}\') is not a ControllerMechanism or subclass of one".
                              format(controller, self.name))

        for process in target_set[kwProcesses]:
            if not isinstance(process, Process):
                raise SystemError("{} (in processes arg for \'{}\') is not a Process object".format(process, self.name))

        for mech, value in target_set[kwInitialValues].items():
            if not isinstance(mech, Mechanism):
                raise SystemError("{} (key for entry in initial_values arg for \'{}\') "
                                  "is not a Mechanism object".format(mech, self.name))

    def _instantiate_attributes_before_function(self, context=None):
        """Instantiate processes and graph

        These calls must be made before _instantiate_function as the latter may be called during init for validation
        """
        self._instantiate_processes(input=self.variable, context=context)
        self._instantiate_graph(context=context)
        self._instantiate_learning_graph(context=context)

    def _instantiate_function(self, context=None):
        """Suppress validation of function

        This is necessary to:
        - insure there is no FUNCTION specified (not allowed for a System object)
        - suppress validation (and attendant execution) of System execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in kwProcesses have already been validated
        """

        if self.paramsCurrent[FUNCTION] != self.execute:
            print("System object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.paramsCurrent[FUNCTION], FUNCTION)
            self.paramsCurrent[FUNCTION] = self.execute

        # If validation pref is set, instantiate and execute the System
        if self.prefs.paramValidationPref:
            super(System_Base, self)._instantiate_function(context=context)
        # Otherwise, just set System output info to the corresponding info for the last mechanism(s) in self.processes
        else:
            self.value = self.processes[-1].outputState.value

    def _instantiate_processes(self, input=None, context=None):
# FIX: ALLOW Projections (??ProjectionTiming TUPLES) TO BE INTERPOSED BETWEEN MECHANISMS IN PATHWAY
# FIX: AUGMENT LinearMatrix TO USE FULL_CONNECTIVITY_MATRIX IF len(sender) != len(receiver)
        """Instantiate processes of system

        Use self.processes (populated by self.paramsCurrent[kwProcesses] in Function._assign_args_to_param_dicts
        If self.processes is empty, instantiate default process by calling process()
        Iterate through self.processes, instantiating each (including the input to each input projection)
        If input is specified, check that it's length equals the number of processes
        If input is not specified, compose from the input for each Process (value specified or, if None, default)
        Note: specification of input for system takes precedence over specification for processes

        # ??STILL THE CASE, OR MOVED TO _instantiate_graph:
        Iterate through Process._mech_tuples for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.executionGraph
        """

        # # MODIFIED 2/8/17 OLD:  [SEE BELOW]
        # self.variable = []
        # MODIFIED 2/8/17 END
        self.mechanismsDict = {}
        self._all_mech_tuples = []
        self._allMechanisms = MechanismList(self, self._all_mech_tuples)

        # Get list of processes specified in arg to init, possibly appended by EVCMechanism (with prediction processes)
        processes_spec = self.processes

        # Assign default Process if PROCESS is empty, or invalid
        if not processes_spec:
            from PsyNeuLink.Components.Process import Process_Base
            processes_spec.append(ProcessTuple(Process_Base(), None))

        # If input to system is specified, number of items must equal number of processes with origin mechanisms
        if input is not None and len(input) != len(self.originMechanisms):
            raise SystemError("Number of items in input ({}) must equal number of processes ({}) in {} ".
                              format(len(input), len(self.originMechanisms),self.name))

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS

        # Convert all entries to (process, input) tuples, with None as filler for absent input
        input_index = input_index_curr = 0
        for i in range(len(processes_spec)):

            # MODIFIED 2/8/17 NEW:
            # Get list of origin mechanisms for processes that have already been converted
            #   (for use below in assigning input)
            orig_mechs_already_processed = list(p[0].originMechanisms[0] for
                                                p in processes_spec if isinstance(p,ProcessTuple))
            # MODIFIED 2/8/17 END

            # Entry is not a tuple
            #    presumably it is a process spec, so enter it as first item of ProcessTuple
            if not isinstance(processes_spec[i], tuple):
                processes_spec[i] = ProcessTuple(processes_spec[i], None)

            # Entry is a tuple but not a ProcessTuple, so convert it
            if isinstance(processes_spec[i], tuple) and not isinstance(processes_spec[i], ProcessTuple):
                processes_spec[i] = ProcessTuple(processes_spec[i][0], processes_spec[i][1])

            # Input was NOT provided on command line, so get it from the process
            if input is None:
                process = processes_spec[i].process
                process_input = []
                for process_input_state in process.processInputStates:
                    process_input.extend(process_input_state.value)
                processes_spec[i] = ProcessTuple(process, process_input)
            # Input was provided on command line, so assign that to input item of tuple
            else:
                # Assign None as input to processes implemented by controller (controller provides their input)
                #    (e.g., prediction processes implemented by EVCMechanism)
                if processes_spec[i].process._isControllerProcess:
                    processes_spec[i] = ProcessTuple(processes_spec[i].process, None)
                else:
                    # MODIFIED 2/8/17 NEW:
                    # Replace input item in tuple with one from command line
                    # Note:  check if origin mechanism for current process is same as any previous one;
                    #        if it is, use that one (and don't increment index for input
                    #        otherwise, assign input and increment input_index
                    try:
                        input_index_curr = orig_mechs_already_processed.index(processes_spec[i][0].originMechanisms[0])
                    except ValueError:
                        input_index += 1
                    processes_spec[i] = ProcessTuple(processes_spec[i].process, input[input_index_curr])
                    input_index_curr = input_index
                    # MODIFIED 2/8/17 END

            # Validate input
            if (processes_spec[i].input is not None and
                    not isinstance(processes_spec[i].input,(numbers.Number, list, np.ndarray))):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, processes_spec[i].input))

            process = processes_spec[i].process
            process_input = processes_spec[i].input

            # # MODIFIED 2/8/17 OLD: [MOVED ASSIGNMENT OF self.variable TO _instantiate_graph()
            # #                       SINCE THAT IS WHERE SYSTEM'S ORIGIN MECHANISMS ARE IDENTIFIED]
            # self.variable.append(process_input)
            # # MODIFIED 2/8/17 END

            # IMPLEMENT: THIS IS WHERE LEARNING SPECIFIED FOR A SYSTEM SHOULD BE IMPLEMENTED FOR EACH PROCESS IN THE
            #            SYSTEM;  NOTE:  IF THE PROCESS IS ALREADY INSTANTIATED WITHOUT LEARNING
            #            (FIRST CONDITIONAL BELOW), MAY NEED TO BE RE-INSTANTIATED WITH LEARNING
            #            (QUESTION:  WHERE TO GET SPECS FOR PROCESS FOR RE-INSTANTIATION??)

            # If process item is a Process object, assign process_input as default
            if isinstance(process, Process):
                if process_input is not None:
                    process._assign_defaults(variable=process_input, context=context)
                # If learning_rate is specified for system but not for process, then apply to process
                # # MODIFIED 3/21/17 OLD:
                # if self.learning_rate and not process.learning_rate:
                    # # FIX:  assign_params WANTS TO CREATE A ParamaterState ON process FOR learning_rate
                    # process.assign_params(request_set={LEARNING_RATE:self.learning_rate})
                # # MODIFIED 3/21/17 NEW:[learning_rate SHOULD BE NOT BE RE-ASSIGNED FOR PROCESS, BUT RATHER ON EXECUTE]
                # if self.learning_rate is not None and process.learning_rate is None:
                #     process.learning_rate = self.learning_rate
                # # MODIFIED 3/21/17 END

            # Otherwise, instantiate Process
            else:
                if inspect.isclass(process) and issubclass(process, Process):
                    # FIX: MAKE SURE THIS IS CORRECT
                    # Provide self as context, so that Process knows it is part of a System (and which one)
                    # Note: this is used by Process._instantiate_pathway() when instantiating first Mechanism
                    #           in Pathway, to override instantiation of projections from Process.input_state
                    process = Process(default_input_value=process_input,
                                      learning_rate=self.learning_rate,
                                      context=self)
                elif isinstance(process, dict):
                    # IMPLEMENT:  HANDLE Process specification dict here;
                    #             include process_input as ??param, and context=self
                    raise SystemError("Attempt to instantiate process {0} in kwProcesses of {1} "
                                      "using a Process specification dict: not currently supported".
                                      format(process.name, self.name))
                else:
                    raise SystemError("Entry {0} of kwProcesses ({1}) must be a Process object, class, or a "
                                      "specification dict for a Process".format(i, process))

            # # process should now be a Process object;  assign to processList
            # self.processList.append(process)

            # Assign the Process a reference to this System
            process.systems.append(self)
            if process.learning:
                self.learning = True

            # Get max of Process phaseSpecs
            self._phaseSpecMax = int(max(math.floor(process._phaseSpecMax), self._phaseSpecMax))

            # Iterate through mechanism tuples in Process' mech_tuples
            #     to construct self._all_mech_tuples and mechanismsDict
            # FIX: ??REPLACE WITH:  for sender_mech_tuple in Process._mech_tuples
            for sender_mech_tuple in process._mech_tuples:

                sender_mech = sender_mech_tuple.mechanism

                # THIS IS NOW DONE IN _instantiate_graph
                # # Add system to the Mechanism's list of systems of which it is member
                # if not self in sender_mech_tuple[MECHANISM].systems:
                #     sender_mech.systems[self] = INTERNAL

                # Assign sender mechanism entry in self.mechanismsDict, with mech_tuple as key and its Process as value
                #     (this is used by Process._instantiate_pathway() to determine if Process is part of System)
                # If the sender is already in the System's mechanisms dict
                if sender_mech_tuple.mechanism in self.mechanismsDict:
                    existing_mech_tuple = self._allMechanisms._get_tuple_for_mech(sender_mech)
                    if not sender_mech_tuple is existing_mech_tuple:
                        # Contents of tuple are the same, so use the tuple in _allMechanisms
                        if (sender_mech_tuple.phase == existing_mech_tuple.phase and
                                    sender_mech_tuple.params == existing_mech_tuple.params):
                            pass
                        # Contents of tuple are different, so raise exception
                        else:
                            if sender_mech_tuple.phase != existing_mech_tuple.phase:
                                offending_tuple_field = 'phase'
                                offending_value = PHASE_ITEM
                            else:
                                offending_tuple_field = 'process_input'
                                offending_value = PARAMS_ITEM
                            raise SystemError("The same mechanism in different processes must have the same parameters:"
                                              "the {} ({}) for {} in {} does not match the value({}) in {}".
                                              format(offending_tuple_field,
                                                     sender_mech_tuple.mechanism,
                                                     sender_mech_tuple[offending_value],
                                                     process,
                                                     existing_mech_tuple[offending_value],
                                                     self.mechanismsDict[sender_mech_tuple.mechanism]
                                                     ))
                    # Add to entry's list
                    self.mechanismsDict[sender_mech].append(process)
                else:
                    # Add new entry
                    self.mechanismsDict[sender_mech] = [process]
                if not sender_mech_tuple in self._all_mech_tuples:
                    self._all_mech_tuples.append(sender_mech_tuple)

            process._allMechanisms = MechanismList(process, tuples_list=process._mech_tuples)

        # # MODIFIED 2/8/17 OLD: [SEE ABOVE]
        # self.variable = convert_to_np_array(self.variable, 2)
        # # MODIFIED 2/8/17 END
        #
        # # Instantiate processList using process_tuples, and point self.processes to it
        # # Note: this also points self.params[kwProcesses] to self.processes
        self.process_tuples = processes_spec
        self._processList = ProcessList(self, self.process_tuples)
        self.processes = self._processList.processes

    def _instantiate_graph(self, context=None):
        """Construct graph (full) and executionGraph (acyclic) of system

        Instantate a graph of all of the mechanisms in the system and their dependencies,
            designate a type for each mechanism in the graph,
            instantiate the executionGraph, a subset of the graph with any cycles removed,
                and topologically sorted into a sequentially ordered list of sets
                containing mechanisms to be executed at the same time

        graph contains a dictionary of dependency sets for all mechanisms in the system:
            reciever_mech_tuple : {sender_mech_tuple, sender_mech_tuple...}
        executionGraph contains an acyclic subset of graph used to determine sequence of mechanism execution;

        They are constructed as follows:
            sequence through self.processes;  for each process:
                begin with process.firstMechanism (assign as ORIGIN if it doesn't receive any projections)
                traverse all projections
                for each mechanism encountered (receiver), assign to its dependency set the previous (sender) mechanism
                for each assignment, use toposort to test whether the dependency introduced a cycle; if so:
                    eliminate the dependent from the executionGraph, and designate it as CYCLE (unless it is an ORIGIN)
                    designate the sender as INITIALIZE_CYCLE (it can receive and initial_value specification)
                if a mechanism doe not project to any other ProcessingMechanisms (ignore monitoring and control mechs):
                    assign as TERMINAL unless it is already an ORIGIN, in which case assign as SINGLETON

        Construct execution_sets and exeuction_list

        Assign MechanismLists:
            allMechanisms
            originMechanisms
            terminalMechanisms
            recurrentInitMechanisms (INITIALIZE_CYCLE)
            monitoringMechansims
            controlMechanisms

        Validate initial_values

        """

        # Use to recursively traverse processes
        def build_dependency_sets_by_traversing_projections(sender_mech):

            # If sender is an ObjectiveMechanism being used for learning or control, or a LearningMechanism,
            # Assign as MONITORING and move on
            if ((isinstance(sender_mech, ObjectiveMechanism) and sender_mech.role) or
                    isinstance(sender_mech, LearningMechanism)):
                sender_mech.systems[self] = MONITORING
                return

            # Delete any projections to mechanism from processes or mechanisms in processes not in current system
            for input_state in sender_mech.inputStates.values():
                for projection in input_state.receivesFromProjections:
                    sender = projection.sender.owner
                    system_processes = self.processes
                    if isinstance(sender, Process):
                        if not sender in system_processes:
                            del projection
                    elif not all(sender_process in system_processes for sender_process in sender.processes):
                        del projection

            # If sender_mech has no projections left, raise exception
            if not any(any(projection for projection in input_state.receivesFromProjections)
                       for input_state in sender_mech.inputStates.values()):
                raise SystemError("{} only receives projections from other processes or mechanisms not"
                                  " in the current system ({})".format(sender_mech.name, self.name))

            # Assign as TERMINAL (or SINGLETON) if it:
            #    - is not an Objective Mechanism used for Learning or Control and
            #    - has no outgoing projections or
            #    -     only ones to ObjectiveMechanism(s) used for Learning or Control and
            # Note:  SINGLETON is assigned if mechanism is already a TERMINAL;  indicates that it is both
            #        an ORIGIN AND A TERMINAL and thus must be the only mechanism in its process
            # It is not a ControlMechanism
            if (

                not (isinstance(sender_mech, ControlMechanism_Base) or
                    # It is not an ObjectiveMechanism used for Learning or Control
                    (isinstance(sender_mech, ObjectiveMechanism) and sender_mech.role in (LEARNING,CONTROL))) and
                        # All of its projections
                        all(
                            all(
                                # are to ControlMechanism(s)...
                                isinstance(projection.receiver.owner, ControlMechanism_Base) or
                                 # or ObjectiveMechanism(s) used for Learning or Control
                                 (isinstance(projection.receiver.owner, ObjectiveMechanism) and
                                             projection.receiver.owner.role in (LEARNING, CONTROL))
                            for projection in output_state.sendsToProjections)
                        for output_state in sender_mech.outputStates.values())):
                try:
                    if sender_mech.systems[self] is ORIGIN:
                        sender_mech.systems[self] = SINGLETON
                    else:
                        sender_mech.systems[self] = TERMINAL
                except KeyError:
                    sender_mech.systems[self] = TERMINAL
                return

            for outputState in sender_mech.outputStates.values():

                for projection in outputState.sendsToProjections:
                    receiver = projection.receiver.owner
                    receiver_tuple = self._allMechanisms._get_tuple_for_mech(receiver)

                    # MODIFIED 2/8/17 NEW:
                    # If receiver is not in system's list of mechanisms, must belong to a process that has
                    #    not been included in the system, so ignore it
                    if not receiver_tuple:
                        continue
                    # MODIFIED 2/8/17 END

                    try:
                        self.graph[receiver_tuple].add(self._allMechanisms._get_tuple_for_mech(sender_mech))
                    except KeyError:
                        self.graph[receiver_tuple] = {self._allMechanisms._get_tuple_for_mech(sender_mech)}

                    # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                    # Do not include dependency (or receiver on sender) in executionGraph for this projection
                    #  and end this branch of the traversal if the receiver has already been encountered,
                    #  but do mark for initialization
                    # Notes:
                    # * This is because it is a feedback connection, which introduces a cycle into the graph
                    #     that precludes use of toposort to determine order of execution;
                    #     however, the feedback projection will still be used during execution
                    #     so the sending mechanism should be designated as INITIALIZE_CYCLE
                    # * Check for receiver mechanism and not its tuple,
                    #     since the same mechanism can appear in more than one tuple (e.g., with different phases)
                    #     and would introduce a cycle irrespective of the tuple in which it appears in the graph
                    # FIX: MODIFY THIS TO (GO BACK TO) USING if receiver_tuple in self.executionGraph
                    # FIX  BUT CHECK THAT THEY ARE IN DIFFERENT PHASES
                    if receiver in self.execution_graph_mechs:
                        # Try assigning receiver as dependent of current mechanism and test toposort
                        try:
                            # If receiver_tuple already has dependencies in its set, add sender_mech to set
                            if self.executionGraph[receiver_tuple]:
                                self.executionGraph[receiver_tuple].\
                                    add(self._allMechanisms._get_tuple_for_mech(sender_mech))
                            # If receiver_tuple set is empty, assign sender_mech to set
                            else:
                                self.executionGraph[receiver_tuple] = \
                                    {self._allMechanisms._get_tuple_for_mech(sender_mech)}
                            # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                            list(toposort(self.executionGraph))
                        # If making receiver dependent on sender produced a cycle (feedback loop), remove from graph
                        except ValueError:
                            self.executionGraph[receiver_tuple].\
                                remove(self._allMechanisms._get_tuple_for_mech(sender_mech))
                            # Assign sender_mech INITIALIZE_CYCLE as system status if not ORIGIN or not yet assigned
                            if not sender_mech.systems or not (sender_mech.systems[self] in {ORIGIN, SINGLETON}):
                                sender_mech.systems[self] = INITIALIZE_CYCLE
                            if not (receiver.systems[self] in {ORIGIN, SINGLETON}):
                                receiver.systems[self] = CYCLE
                            continue

                    else:
                        # Assign receiver as dependent on sender mechanism
                        try:
                            # FIX: THIS WILL ADD SENDER_MECH IF RECEIVER IS IN GRAPH BUT = set()
                            # FIX: DOES THAT SCREW UP ORIGINS?
                            self.executionGraph[receiver_tuple].\
                                add(self._allMechanisms._get_tuple_for_mech(sender_mech))
                        except KeyError:
                            self.executionGraph[receiver_tuple] = \
                                {self._allMechanisms._get_tuple_for_mech(sender_mech)}

                    if not sender_mech.systems:
                        sender_mech.systems[self] = INTERNAL

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver)

        self.graph = OrderedDict()
        self.executionGraph = OrderedDict()


        # Sort for consistency of output
        sorted_processes = sorted(self.processes, key=lambda process : process.name)

        for process in sorted_processes:
            first_mech = process.firstMechanism

            # Treat as ORIGIN if ALL projections to the first mechanism in the process are from:
            #    - the process itself (ProcessInputState)
            #    - another mechanism in the in process (i.e., feedback projections from *within* the process)
            #    - mechanisms from other process for which it is an origin
            # Notes:
            # * This precludes a mechanism that is an ORIGIN of a process from being an ORIGIN for the system
            #       if it receives any projections from any other mechanisms in the system (including other processes)
            #       other than ones in processes for which it is also their ORIGIN
            # * This does allow a mechanism to be the ORIGIN (but *only* the ORIGIN) for > 1 process in the system
            if all(
                    all(
                            # All projections must be from a process (i.e., ProcessInputState) to which it belongs
                            # # MODIFIED 2/8/17 OLD:
                            # #          [THIS CHECKED FOR PROCESS IN SYSTEM'S LIST OF PROCESSES
                            # #           IT CRASHED IF first_mech WAS ASSIGNED TO ANY PROCESS THAT WAS NOT ALSO
                            # #           ASSIGNED TO THE SYSTEM TO WHICH THE first_mech BELONGS
                            #  projection.sender.owner in sorted_processes or
                            # MODIFIED 2/8/17 NEW:
                            #          [THIS CHECKS THAT PROJECTION IS FROM A PROCESS IN first_mech's LIST OF PROCESSES]
                            #           PROBABLY ISN"T NECESSARY, AS IT SHOULD BE COVERED BY INITIAL ASSIGNMENT OF PROJ]
                            projection.sender.owner in first_mech.processes or
                            # MODIFIED 2/8/17 END
                            # or from mechanisms within its own process (e.g., [a, b, a])
                            projection.sender.owner in list(process.mechanisms) or
                            # or from mechanisms in other processes for which it is also an ORIGIN ([a,b,a], [a,c,a])
                            all(ORIGIN in first_mech.processes[proc]
                                for proc in projection.sender.owner.processes
                                if isinstance(projection.sender.owner,Mechanism))
                        # For all the projections to each inputState
                        for projection in input_state.receivesFromProjections)
                    # For all inputStates for the first_mech
                    for input_state in first_mech.inputStates.values()):
                # Assign its set value as empty, marking it as a "leaf" in the graph
                mech_tuple = self._allMechanisms._get_tuple_for_mech(first_mech)
                self.graph[mech_tuple] = set()
                self.executionGraph[mech_tuple] = set()
                first_mech.systems[self] = ORIGIN

            build_dependency_sets_by_traversing_projections(first_mech)

        # MODIFIED 4/1/17 NEW:
        # HACK TO LABEL TERMINAL MECHS -- SHOULD HAVE BEEN HANDLED ABOVE
        # LABELS ANY MECH AS A TARGET THAT PROJECTION TO AN ObjectiveMechanism WITH LEARNING AS ITS role
        for mech in self.mechanisms:
            for output_state in mech.outputStates.values():
                for projection in output_state.sendsToProjections:
                    receiver = projection.receiver.owner
                    if isinstance(receiver, ObjectiveMechanism) and receiver.role == LEARNING:
                        mech.systems[self] = TERMINAL
                        break
                if mech.systems[self] == TERMINAL:
                    break
        # MODIFIED 4/1/17 END

        # Print graph
        if self.verbosePref:
            warnings.warn("In the system graph for \'{}\':".format(self.name))
            for receiver_mech_tuple, dep_set in self.executionGraph.items():
                mech = receiver_mech_tuple.mechanism
                if not dep_set:
                    print("\t\'{}\' is an {} mechanism".
                          format(mech.name, mech.systems[self]))
                else:
                    status = mech.systems[self]
                    if status is TERMINAL:
                        status = 'a ' + status
                    elif status in {INTERNAL, INITIALIZE_CYCLE}:
                        status = 'an ' + status
                    print("\t\'{}\' is {} mechanism that receives projections from:".format(mech.name, status))
                    for sender_mech_tuple in dep_set:
                        print("\t\t\'{}\'".format(sender_mech_tuple.mechanism.name))

        # For each mechanism (represented by its tuple) in the graph, add entry to relevant list(s)
        # Note: ignore mechanisms belonging to controllerProcesses (e.g., instantiated by EVCMechanism)
        #       as they are for internal use only;
        #       this also ignored learning-related mechanisms (they are handled below)
        self._origin_mech_tuples = []
        self._terminal_mech_tuples = []
        self.recurrent_init_mech_tuples = []
        self._control_mech_tuple = []

        for mech_tuple in self.executionGraph:

            mech = mech_tuple.mechanism

            if mech.systems[self] in {ORIGIN, SINGLETON}:
                for process, status in mech.processes.items():
                    if process._isControllerProcess:
                        continue
                    self._origin_mech_tuples.append(mech_tuple)
                    break

            if mech_tuple.mechanism.systems[self] in {TERMINAL, SINGLETON}:
                for process, status in mech.processes.items():
                    if process._isControllerProcess:
                        continue
                    self._terminal_mech_tuples.append(mech_tuple)
                    break

            if mech_tuple.mechanism.systems[self] in {INITIALIZE_CYCLE}:
                for process, status in mech.processes.items():
                    if process._isControllerProcess:
                        continue
                    self.recurrent_init_mech_tuples.append(mech_tuple)
                    break

            if isinstance(mech_tuple.mechanism, ControlMechanism_Base):
                if not mech_tuple.mechanism in self._control_mech_tuple:
                    self._control_mech_tuple.append(mech_tuple)

        self.originMechanisms = MechanismList(self, self._origin_mech_tuples)
        self.terminalMechanisms = MechanismList(self, self._terminal_mech_tuples)
        self.recurrentInitMechanisms = MechanismList(self, self.recurrent_init_mech_tuples)
        self.controlMechanism = MechanismList(self, self._control_mech_tuple)

        try:
            self.execution_sets = list(toposort(self.executionGraph))
        except ValueError as e:
            if 'Cyclic dependencies exist' in e.args[0]:
                # if self.verbosePref:
                # print('{} has feedback connections; be sure that the following items are properly initialized:'.
                #       format(self.name))
                raise SystemError("PROGRAM ERROR: cycle (feedback loop) in {} not detected by _instantiate_graph ".
                                  format(self.name))

        # Create instance of sequential (execution) list:
        # MODIFIED 10/31/16 OLD:
        # self.executionList = toposort_flatten(self.executionGraph, sort=False)
        # MODIFIED 10/31/16 NEW:
        temp = toposort_flatten(self.executionGraph, sort=False)
        self.executionList = self._toposort_with_ordered_mech_tuples(self.executionGraph)
        # MODIFIED 10/31/16 END

        # MODIFIED 2/8/17 NEW:
        # Construct self.variable from inputs to ORIGIN mechanisms
        self.variable = []
        for mech in self.originMechanisms:
            orig_mech_input = []
            for input_state in mech.inputStates.values():
                orig_mech_input.extend(input_state.value)
            self.variable.append(orig_mech_input)
        self.variable = convert_to_np_array(self.variable, 2)
        # MODIFIED 2/8/17 END

        # Instantiate StimulusInputStates
        self._instantiate_stimulus_inputs()

        # Validate initial values
        # FIX: CHECK WHETHER ALL MECHANISMS DESIGNATED AS INITIALIZE HAVE AN INITIAL_VALUES ENTRY
        # FIX: ONLY CHECKS FIRST ITEM OF self._value_template (ASSUMES THAT IS ALL THAT WILL GET ASSIGNED)
        # FIX: ONLY CHECK ONES THAT RECEIVE PROJECTIONS
        for mech, value in self.initial_values.items():
            if not mech in self.execution_graph_mechs:
                raise SystemError("{} (entry in initial_values arg) is not a Mechanism in \'{}\'".
                                  format(mech.name, self.name))
            mech._update_value
            if not iscompatible(value, mech._value_template[0]):
                raise SystemError("{} (in initial_values arg for \'{}\') is not a valid value for {}".
                                  format(value, self.name, append_type_to_name(self)))

    def _instantiate_stimulus_inputs(self, context=None):

# FIX: ZERO VALUE OF ALL ProcessInputStates BEFORE EXECUTING
# FIX: RENAME SystemInputState -> SystemInputState

        # Create SystemInputState for each ORIGIN mechanism in originMechanisms and
        #    assign MappingProjection from the SystemInputState to the ORIGIN mechanism
        for i, origin_mech in zip(range(len(self.originMechanisms)), self.originMechanisms):

            # Skip if ORIGIN mechanism already has a projection from a SystemInputState in current system
            # (this avoids duplication from multiple passes through _instantiate_graph)
            if any(self is projection.sender.owner for projection in origin_mech.inputState.receivesFromProjections):
                continue

            # Check, for each ORIGIN mechanism, that the length of the corresponding item of self.variable matches the
            # length of the ORIGIN inputState's variable attribute
            if len(self.variable[i]) != len(origin_mech.inputState.variable):
                raise SystemError("Length of input {} ({}) does not match the length of the input ({}) for the "
                                  "corresponding ORIGIN mechanism ()".
                                   format(i,
                                          len(self.variable[i]),
                                          len(origin_mech.inputState.variable),
                                          origin_mech.name))

            stimulus_input_state = SystemInputState(owner=self,
                                                        variable=origin_mech.inputState.variable,
                                                        prefs=self.prefs,
                                                        name="System Input {}".format(i))
            self.stimulusInputStates.append(stimulus_input_state)
            self.inputs.append(stimulus_input_state.value)

            # Add MappingProjection from stimulus_input_state to ORIGIN mechainsm's inputState
            from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
            MappingProjection(sender=stimulus_input_state,
                    receiver=origin_mech,
                    name=self.name+' Input Projection to '+origin_mech.name)


    def _instantiate_learning_graph(self, context=None):
        """Build graph of monitoringMechanisms and learningProjections for use in learning
        """

        self.learningGraph = OrderedDict()
        self.learningExecutionGraph = OrderedDict()

        def build_dependency_sets_by_traversing_projections(sender_mech, process):

            # MappingProjections are legal recipients of learning projections (hence the call)
            #  but do not send any projections, so no need to consider further
            from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
            if isinstance(sender_mech, MappingProjection):
                return

            # All other sender_mechs must be either a MonitoringMechanism or an ObjectiveMechanism with role=LEARNING
            elif not (isinstance(sender_mech, LearningMechanism) or
                          (isinstance(sender_mech, ObjectiveMechanism) and sender_mech.role is LEARNING)):
                raise SystemError("PROGRAM ERROR: {} is not a legal object for learning graph;"
                                  "must be a LearningMechanism or an ObjectiveMechanism".
                                  format(sender_mech))


            # MODIFIED 3/12/17 NEW:

            # MANAGE TARGET ObjectiveMechanism FOR INTERNAL or TERMINAL CONVERGENCE of PATHWAYS

            # If sender_mech is an ObjectiveMechanism, and:
            #    - none of the mechanisms that project to it are are a TERMINAL mechanism for the current process, or
            #    - all of the mechanisms that project to it already have an ObjectiveMechanism, then:
            #        - do not include the ObjectiveMechanism in the graph;
            #        - be sure that its outputState projects to the ERROR_SIGNAL inputState of a LearningMechanism
            #            (labelled "learning_mech" here -- raise an exception if it does not;
            #        - determine whether learning_mech's ERROR_SIGNAL inputState receives any other projections
            #            from another ObjectiveMechanism or LearningMechanism (labelled "error_signal_projection" here)
            #            -- if it does, be sure that it is from the same system and if so return;
            #               (note:  this shouldn't be true, but the test is here for completeness and sanity-checking)
            #        - if learning_mech's ERROR_SIGNAL inputState does not receive any projections from
            #            another objectiveMechanism and/or LearningMechanism in the system, then:
            #            - find the sender to the ObjectiveMechanism (labelled "error_source" here)
            #            - find the 1st projection from error_source that projects to the ACTIVATION_INPUT inputState of
            #                a LearningMechanism (labelled "error_signal" here)
            #            - instantiate a MappingProjection from error_signal to learning_mech
            #                projected
            # IMPLEMENTATION NOTE: Composition should allow 1st condition if user indicates internal TARGET is desired;
            #                      for now, however, assuming this is not desired (i.e., only TERMINAL mechanisms
            #                      should project to ObjectiveMechanisms) and always replace internal
            #                      ObjectiveMechanism with projection from a LearningMechanism (if it is available)

            # FIX: RELABEL "sender_mech" as "obj_mech" here

            if isinstance(sender_mech, ObjectiveMechanism) and len(self.learningExecutionGraph):

                # TERMINAL CONVERGENCE
                # All of the mechanisms that project to sender_mech
                #    project to another ObjectiveMechanism already in the learning_graph
                if all(
                        any(
                                (isinstance(receiver_mech, ObjectiveMechanism) and
                                 # its already in a dependency set in the learningExecutionGraph
                                         receiver_mech in set.union(*list(self.learningExecutionGraph.values())) and
                                     not receiver_mech is sender_mech)
                                # receivers of senders to sender_mech
                                for receiver_mech in [proj.receiver.owner for proj in
                                                      mech.outputState.sendsToProjections])
                        # senders to sender_mech
                        for mech in [proj.sender.owner
                                     for proj in sender_mech.inputStates[SAMPLE].receivesFromProjections]):

                    # Get the ProcessingMechanism that projected to sender_mech
                    error_source_mech = sender_mech.inputStates[SAMPLE].receivesFromProjections[0].sender.owner

                    # Get the other ObjectiveMechanism to which the error_source projects (in addition to sender_mech)
                    other_obj_mech = next((projection.receiver.owner for projection in
                                           error_source_mech.outputState.sendsToProjections if
                                           isinstance(projection.receiver.owner, ObjectiveMechanism)), None)
                    sender_mech = other_obj_mech

                # INTERNAL CONVERGENCE
                # None of the mechanisms that project to it are a TERMINAL mechanism
                elif not all(all(projection.sender.owner.processes[proc] is TERMINAL
                                 for proc in projection.sender.owner.processes)
                             for projection in sender_mech.inputStates[SAMPLE].receivesFromProjections):

                    # Get the LearningMechanism to which the sender_mech projected
                    try:
                        learning_mech = sender_mech.outputState.sendsToProjections[0].receiver.owner
                        if not isinstance(learning_mech, LearningMechanism):
                            raise AttributeError
                    except AttributeError:
                        raise SystemError("{} does not project to a LearningMechanism in the same process {}".
                                          format(sender_mech.name, process.name))

                    from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningAuxilliary \
                        import ACTIVATION_INPUT, ERROR_SIGNAL

                    # Get the ProcessingMechanism that projected to sender_mech
                    error_source_mech = sender_mech.inputStates[SAMPLE].receivesFromProjections[0].sender.owner

                    # Get the other LearningMechanism to which the error_source projects (in addition to sender_mech)
                    error_signal_mech = next((projection.receiver.owner for projection in
                                              error_source_mech.outputState.sendsToProjections if
                                              projection.receiver.name is ACTIVATION_INPUT), None)


                    # Check if learning_mech receives an error_signal_projection
                    #    from any other ObjectiveMechanism or LearningMechanism in the system;
                    # If it does, get the first one found
                    error_signal_projection = next ((projection for projection
                                                     in learning_mech.inputStates[ERROR_SIGNAL].receivesFromProjections
                                                     if (isinstance(projection.sender.owner,(ObjectiveMechanism,
                                                                                            LearningMechanism)) and
                                                     not projection.sender.owner is sender_mech and
                                                     self in projection.sender.owner.systems.values())), None)
                    # If learning_mech receives another error_signal projection,
                    #    reassign sender_mech to the sender of that projection
                    if error_signal_projection:
                        if self.verbosePref:
                            warnings.warn("Although {} a TERMINAL mechanism for the {} process, it is an "
                                          "internal mechanism for other proesses in the {} system; therefore "
                                          "its ObjectiveMechanism ({}) will be replaced with the {} LearningMechanism".
                                          format(error_source_mech.name,
                                                 process.name,
                                                 self.name,
                                                 sender_mech.name,
                                                 error_signal_mech))
                        sender_mech = error_signal_projection.sender.owner

                    # FIX:  FINISH DOCUMENTATION HERE ABOUT HOW THIS IS DIFFERENT THAN ABOVE
                    if error_signal_mech is None:
                        raise SystemError("Could not find projection to an {} inputState of a LearningMechanism for "
                                          "the ProcessingMechanism ({}) that projects to {} in the {} process"
                                          "".format(ACTIVATION_INPUT,
                                                    error_source_mech.name,
                                                    sender_mech.name,
                                                    process.name))
                    else:
                        mp = MappingProjection(sender=error_signal_mech.outputStates[ERROR_SIGNAL],
                                               receiver=learning_mech.inputStates[ERROR_SIGNAL],
                                               matrix=IDENTITY_MATRIX)
                        if mp is None:
                            raise SystemError("Could not instantiate a MappingProjection "
                                              "from {} to {} for the {} process".
                                              format(error_signal_mech.name, learning_mech.name))

                        sender_mech = error_signal_mech
            # MODIFIED 3/12/17 END


            # Delete any projections to mechanism from processes or mechanisms in processes not in current system
            for input_state in sender_mech.inputStates.values():
                for projection in input_state.receivesFromProjections:
                    sender = projection.sender.owner
                    system_processes = self.processes
                    if isinstance(sender, Process):
                        if not sender in system_processes:
                            del projection
                    elif not all(sender_process in system_processes for sender_process in sender.processes):
                        del projection

            # If sender_mech has no projections left, raise exception
            if not any(any(projection for projection in input_state.receivesFromProjections)
                       for input_state in sender_mech.inputStates.values()):
                raise SystemError("{} only receives projections from other processes or mechanisms not"
                                  " in the current system ({})".format(sender_mech.name, self.name))

            for outputState in sender_mech.outputStates.values():

                for projection in outputState.sendsToProjections:
                    receiver = projection.receiver.owner
                    try:
                        self.learningGraph[receiver].add(sender_mech)
                    except KeyError:
                        self.learningGraph[receiver] = {sender_mech}

                    # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                    # Do not include dependency (or receiver on sender) in learningExecutionGraph for this projection
                    #  and end this branch of the traversal if the receiver has already been encountered,
                    #  but do mark for initialization
                    # Notes:
                    # * This is because it is a feedback connection, which introduces a cycle into the learningGraph
                    #     that precludes use of toposort to determine order of execution;
                    #     however, the feedback projection will still be used during execution
                    #     so the sending mechanism should be designated as INITIALIZE_CYCLE
                    # * Check for receiver mechanism and not its tuple,
                    #     since the same mechanism can appear in more than one tuple (e.g., with different phases)
                    #     and would introduce a cycle irrespective of the tuple in which it appears in the learningGraph

                    if receiver in self.learningExecutionGraph:
                    # if receiver in self.learning_execution_graph_mechs:
                        # Try assigning receiver as dependent of current mechanism and test toposort
                        try:
                            # If receiver already has dependencies in its set, add sender_mech to set
                            if self.learningExecutionGraph[receiver]:
                                self.learningExecutionGraph[receiver].add(sender_mech)
                            # If receiver set is empty, assign sender_mech to set
                            else:
                                self.learningExecutionGraph[receiver] = {sender_mech}
                            # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                            list(toposort(self.learningExecutionGraph))
                        # If making receiver dependent on sender produced a cycle, remove from learningGraph
                        except ValueError:
                            self.learningExecutionGraph[receiver].remove(sender_mech)
                            receiver.systems[self] = CYCLE
                            continue

                    else:
                        # Assign receiver as dependent on sender mechanism
                        try:
                            # FIX: THIS WILL ADD SENDER_MECH IF RECEIVER IS IN GRAPH BUT = set()
                            # FIX: DOES THAT SCREW UP ORIGINS?
                            self.learningExecutionGraph[receiver].add(sender_mech)
                        except KeyError:
                            self.learningExecutionGraph[receiver] = {sender_mech}

                    if not sender_mech.systems:
                        sender_mech.systems[self] = MONITORING

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver, process)

        # Sort for consistency of output
        sorted_processes = sorted(self.processes, key=lambda process : process.name)

        # This assumes that the first mechanism in process.monitoringMechanisms is the last in the learning sequence
        # (i.e., that the list is being traversed "backwards")
        for process in sorted_processes:
            if process.learning and process._learning_enabled:
                build_dependency_sets_by_traversing_projections(process.monitoringMechanisms[0], process)

        # FIX: USE TOPOSORT TO FIND, OR AT LEAST CONFIRM, TARGET MECHANISMS, WHICH SHOULD EQUAL COMPARATOR MECHANISMS
        self.learningExecutionList = toposort_flatten(self.learningExecutionGraph, sort=False)
        # self.learningExecutionList = self._toposort_with_ordered_mech_tuples(self.learningExecutionGraph)

        # Construct monitoringMechanisms and targetMechanisms MechanismLists

        # MODIFIED 3/12/17 NEW: [MOVED FROM _instantiate_graph]
        self._monitoring_mech_tuples = []
        self._target_mech_tuples = []

        from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
        for item in self.learningExecutionList:
            if isinstance(item, MappingProjection):
                continue

            # If a learning_rate has been specified for the system, assign that to all LearningMechanisms
            #    for which a mechanism-specific learning_rate has NOT been assigned
            if (isinstance(item, LearningMechanism) and
                        self.learning_rate is not None and
                        item.function_object.learning_rate is None):
                item.function_object.learning_rate = self.learning_rate

            mech_tuple = self._allMechanisms._get_tuple_for_mech(item)
            if not mech_tuple in self._monitoring_mech_tuples:
                self._monitoring_mech_tuples.append(mech_tuple)
            if isinstance(item, ObjectiveMechanism) and not mech_tuple in self._target_mech_tuples:
                self._target_mech_tuples.append(mech_tuple)
        self.monitoringMechanisms = MechanismList(self, self._monitoring_mech_tuples)
        self.targetMechanisms = MechanismList(self, self._target_mech_tuples)
        # MODIFIED 3/12/17 END

        # Instantiate TargetInputStates
        self._instantiate_target_inputs()

    def _instantiate_target_inputs(self, context=None):

        if self.learning and self.targets is None:
            if not self.targetMechanisms:
                raise SystemError("PROGRAM ERROR: Learning has been specified for {} but it has no targetMechanisms".
                                  format(self.name))
            elif len(self.targetMechanisms)==1:
                error_msg = "Learning has been specified for {} so a target must also be specified"
            else:
                error_msg = "Learning has been specified for {} but no targets have been specified."
            raise SystemError(error_msg.format(self.name))

        self.targets = np.atleast_2d(self.targets)

        # Create SystemInputState for each TARGET mechanism in targetMechanisms and
        #    assign MappingProjection from the SystemInputState
        #    to the TARGET mechanism's TARGET inputSate
        #    (i.e., from the SystemInputState to the ComparatorMechanism)
        for i, target_mech in zip(range(len(self.targetMechanisms)), self.targetMechanisms):

            # Create ProcessInputState for each target and assign to targetMechanism's target inputState
            target_mech_TARGET_input_state = target_mech.inputStates[TARGET]

            # Check, for each TARGET mechanism, that the length of the corresponding item of targets matches the length
            #    of the TARGET (ComparatorMechanism) target inputState's variable attribute
            if len(self.targets[i]) != len(target_mech_TARGET_input_state.variable):
                raise SystemError("Length of target ({}: {}) does not match the length ({}) of the target "
                                  "expected for its TARGET mechanism {}".
                                   format(len(self.targets[i]),
                                          self.targets[i],
                                          len(target_mech_TARGET_input_state.variable),
                                          target_mech.name))

            system_target_input_state = SystemInputState(owner=self,
                                                        variable=target_mech_TARGET_input_state.variable,
                                                        prefs=self.prefs,
                                                        name="System Target {}".format(i))
            self.targetInputStates.append(system_target_input_state)

            # Add MappingProjection from system_target_input_state to TARGET mechainsm's target inputState
            from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
            MappingProjection(sender=system_target_input_state,
                    receiver=target_mech_TARGET_input_state,
                    name=self.name+' Input Projection to '+target_mech_TARGET_input_state.name)

    def _assign_output_states(self):
        """Assign outputStates for System (the values of which will comprise System.value)

        Assign the outputs of terminal Mechanisms in the graph to the system's outputValue

        Note:
        * Current implementation simply assigns terminal mechanisms as outputStates
        * This method is included so that sublcasses and/or future versions can override it to make custom assignments

        """
        for mech in self.terminalMechanisms.mechanisms:
            self.outputStates[mech.name] = mech.outputStates

    def initialize(self):
        """Assign :py:data:`initial_values <System_Base.initialize>` to mechanisms designated as \
        `INITIALIZE_CYCLE` and contained in recurrentInitMechanisms.
        """
        # FIX:  INITIALIZE PROCESS INPUT??
        # FIX: CHECK THAT ALL MECHANISMS ARE INITIALIZED FOR WHICH mech.system[SELF]==INITIALIZE
        # FIX: ADD OPTION THAT IMPLEMENTS/ENFORCES INITIALIZATION
        # FIX: ADD SOFT_CLAMP AND HARD_CLAMP OPTIONS
        # FIX: ONLY ASSIGN ONES THAT RECEIVE PROJECTIONS
        for mech, value in self.initial_values.items():
            mech.initialize(value)

    def execute(self,
                input=None,
                target=None,
                execution_id=None,
                clock=CentralClock,
                time_scale=None,
                termination_conditions={ts: None for ts in TimeScale},
                # time_scale=TimeScale.TRIAL
                context=None):
        """Execute mechanisms in system at specified :ref:`phases <System_Execution_Phase>` in order \
        specified by the :py:data:`executionGraph <System_Base.executionGraph>` attribute.

        Assign items of input to `ORIGIN` mechanisms

        Execute mechanisms in the order specified in executionList and with phases equal to
        ``CentralClock.time_step % numPhases``.

        Execute any learning components specified at the appropriate phase.

        Execute controller after all mechanisms have been executed (after each numPhases)

        .. Execution:
            - the input arg in system.execute() or run() is provided as input to ORIGIN mechanisms (and system.input);
                As with a process, ORIGIN mechanisms will receive their input only once (first execution)
                    unless clamp_input (or SOFT_CLAMP or HARD_CLAMP) are specified, in which case they will continue to
            - execute() calls mechanism.execute() for each mechanism in its execute_graph in sequence
            - outputs of TERMINAL mechanisms are assigned as system.ouputValue
            - system.controller is executed after execution of all mechanisms in the system
            - notes:
                * the same mechanism can be listed more than once in a system, inducing recurrent processing

        Arguments
        ---------
        input : list or ndarray
            a list or array of input value arrays, one for each `ORIGIN` mechanism in the system.

            .. [TBI: time_scale : TimeScale : default TimeScale.TRIAL
               specifies a default TimeScale for the system]

            .. context : str

        Returns
        -------
        output values of system : 3d ndarray
            Each item is a 2d array that contains arrays for each outputState.value of each TERMINAL mechanism

        """

        if not context:
            context = EXECUTING + " " + SYSTEM + " " + self.name

        # Update execution_id for self and all mechanisms in graph (including learning) and controller
        from PsyNeuLink.Globals.Run import _get_unique_id
        self._execution_id = execution_id or _get_unique_id()
        # FIX: GO THROUGH LEARNING GRAPH HERE AND ASSIGN EXECUTION TOKENS FOR ALL MECHANISMS IN IT
        # self.learningExecutionList
        for mech in self.execution_graph_mechs:
            mech._execution_id = self._execution_id
        for learning_mech in self.learningExecutionList:
            learning_mech._execution_id = self._execution_id
        self.controller._execution_id = self._execution_id
        if self.controller.inputStates:
            for state in self.controller.inputStates.values():
                for projection in state.receivesFromProjections:
                    projection.sender.owner._execution_id = self._execution_id

        self._report_system_output = self.prefs.reportOutputPref and context and EXECUTING in context
        if self._report_system_output:
            self._report_process_output = any(process.reportOutputPref for process in self.processes)

        self.timeScale = time_scale or TimeScale.TRIAL

        # FIX: MOVE TO RUN??
        #region ASSIGN INPUTS TO SystemInputStates
        #    that will be used as the input to the MappingProjection to each ORIGIN mechanism
        num_origin_mechs = len(list(self.originMechanisms))

        if input is None:
            if (self.prefs.verbosePref and
                    not (not context or COMPONENT_INIT in context)):
                print("- No input provided;  default will be used: {0}")
            input = np.zeros_like(self.variable)
            for i in range(num_origin_mechs):
                input[i] = self.originMechanisms[i].variableInstanceDefault

        else:
            num_inputs = np.size(input,0)

            # Check if input items are of different lengths (indicated by dtype == np.dtype('O'))
            if num_inputs != num_origin_mechs:
                num_inputs = np.size(input)
               # Check that number of inputs matches number of ORIGIN mechanisms
                if isinstance(input, np.ndarray) and input.dtype is np.dtype('O') and num_inputs == num_origin_mechs:
                    pass
                else:
                    raise SystemError("Number of items in input ({0}) to {1} does not match "
                                      "its number of origin Mechanisms ({2})".
                                      format(num_inputs, self.name,  num_origin_mechs ))

            # Get SystemInputState that projects to each ORIGIN mechanism and assign input to it
            for i, origin_mech in zip(range(num_origin_mechs), self.originMechanisms):
                # For each inputState of the ORIGIN mechansim
                input_states = list(origin_mech.inputStates.values())
                for j, input_state in zip(range(len(origin_mech.inputStates)), input_states):
                   # Get the input from each projection to that inputState (from the corresponding SystemInputState)
                    system_input_state = next(projection.sender for projection in input_state.receivesFromProjections
                                              if isinstance(projection.sender, SystemInputState))
                    if system_input_state:
                        print("SYSTEM_INPUT_STATE.VALUE", system_input_state.value)
                        system_input_state.value = input[i][j]
                    else:
                        raise SystemError("Failed to find expected SystemInputState for {}".format(origin_mech.name))

        self.input = input
        #endregion

        if self._report_system_output:
            self._report_system_initiation(clock=clock)


        #region EXECUTE MECHANISMS

        # TEST PRINT:
        # for i in range(len(self.executionList)):
        #     print(self.executionList[i][0].name)
        # sorted_list = list(mech_tuple[0].name for mech_tuple in self.executionList)

        # Execute system without learning on projections (that will be taken care of in _execute_learning()
        self._execute_processing(clock=clock, termination_conditions=termination_conditions, context=context)
        #endregion

        # region EXECUTE LEARNING FOR EACH PROCESS

        # Don't execute learning for simulation runs
        if not EVC_SIMULATION in context:
            self._execute_learning(clock=clock, context=context + LEARNING)
        # endregion


        #region EXECUTE CONTROLLER
# FIX: 1) RETRY APPENDING TO EXECUTE LIST AND COMPARING TO THIS VERSION
# FIX: 2) REASSIGN INPUT TO SYSTEM FROM ONE DESIGNATED FOR EVC SIMULUS (E.G., StimulusPrediction)

        # Only call controller if this is not a controller simulation run (to avoid infinite recursion)
        if not EVC_SIMULATION in context and self.enable_controller:
            try:
                if self.controller.phaseSpec == (clock.time_step % self.numPhases):
                    self.controller.execute(clock=clock,
                                            time_scale=TimeScale.TRIAL,
                                            runtime_params=None,
                                            context=context)
                    if self._report_system_output:
                        print("{0}: {1} executed".format(self.name, self.controller.name))

            except AttributeError as error_msg:
                if not 'INIT' in context:
                    raise SystemError("Problem executing controller for {}: {}".format(self.name, error_msg))
        #endregion

        # Report completion of system execution and value of designated outputs
        if self._report_system_output:
            self._report_system_completion(clock=clock)

        return self.terminalMechanisms.outputStateValues

    def _execute_processing(self, clock=CentralClock, termination_conditions={ts: None for ts in TimeScale}, context=None):
    # def _execute_processing(self, clock=CentralClock, time_scale=TimeScale.Trial, context=None):
        # Execute each Mechanism in self.executionList, in the order listed during its phase
        if self.scheduler is None:
            raise SystemError('System.py:_execute_processing - {0}\'s scheduler is None, must be initialized before execution'.format(self.name))
        for next_execution_set in self.scheduler.run(termination_conds=termination_conditions):
            for mechanism in next_execution_set:
                # mechanism, params, phase_spec = self.executionList[i]
                params = self._allMechanisms._get_tuple_for_mech(mechanism).params
                mechanism.execute(
                                    # clock=clock,
                                    # time_scale=self.timeScale,
                                    # time_scale=time_scale,
                                    runtime_params=params,
                                    context = context
                                    )

    def _execute_learning(self, clock=CentralClock, context=None):
        # Execute each monitoringMechanism as well as learning projections in self.learningExecutionList

        # FIRST, if targets were specified as a function, call the function now
        #    (i.e., after execution of the pathways, but before learning)
        # Note:  this accomodates functions that predicate the target on the outcome of processing
        #        (e.g., for rewards in reinforcement learning)
        if isinstance(self.targets, function_type):
            self.current_targets = self.targets()

        for i, target_mech in zip(range(len(self.targetMechanisms)), self.targetMechanisms):
        # Assign each item of targets to the value of the targetInputState for the TARGET mechanism
        #    and zero the value of all ProcessInputStates that project to the TARGET mechanism
            self.targetInputStates[i].value = self.current_targets[i]

        # NEXT, execute all components involved in learning
        for component in self.learningExecutionList:

            from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
            if isinstance(component, MappingProjection):
                continue

            params = None

            component_type = component.componentType

            processes = list(component.processes.keys())

            # Sort for consistency of reporting:
            process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
            process_names = list(p.name for p in process_keys_sorted)

            context_str = str("{} | {}: {} [in processes: {}]".
                              format(context,
                                     component_type,
                                     component.name,
                                     re.sub('[\[,\],\n]','',str(process_names))))

            # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
            component.execute(clock=clock,
                              time_scale=self.timeScale,
                              runtime_params=params,
                              # time_scale=time_scale,
                              context=context_str)
            # # TEST PRINT:
            # print ("EXECUTING MONITORING UPDATES: ", component.name)

        # THEN update all MappingProjections
        for component in self.learningExecutionList:

            if isinstance(component, (LearningMechanism, ObjectiveMechanism)):
                continue
            if not isinstance(component, MappingProjection):
                raise SystemError("PROGRAM ERROR:  Attempted learning on non-MappingProjection")

            component_type = "mappingProjection"
            processes = list(component.sender.owner.processes.keys())


            # Sort for consistency of reporting:
            process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
            process_names = list(p.name for p in process_keys_sorted)

            context_str = str("{} | {}: {} [in processes: {}]".
                              format(context,
                                     component_type,
                                     component.name,
                                     re.sub('[\[,\],\n]','',str(process_names))))

            component.parameterStates[MATRIX].update(time_scale=TimeScale.TRIAL, context=context_str)

            # TEST PRINT:
            # print ("EXECUTING WEIGHT UPDATES: ", component.name)

        # FINALLY report outputs
        if self._report_system_output and self._report_process_output:
            # Report learning for targetMechanisms (and the processes to which they belong)
            # Sort for consistency of reporting:
            print("\n\'{}' learning completed:".format(self.name))

            for target_mech in self.targetMechanisms:
                processes = list(target_mech.processes.keys())
                process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                process_names = list(p.name for p in process_keys_sorted)
                # print("\n\'- Target: {}' error: {} (processes: {})".
                print("- error for target {}': {}".
                      format(append_type_to_name(target_mech),
                             re.sub('[\[,\],\n]','',str([float("{:0.3}".format(float(i)))
                                                         for i in target_mech.outputState.value])),
                             ))
                             # process_names))

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
            clock=CentralClock,
            time_scale=None,
            termination_conditions=None,
            context=None):
        """Run a sequence of executions

        Call execute method for each execution in a sequence specified by inputs.  See :doc:`Run` for details of
        formatting input specifications.

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_input_value for a single execution
            the input for each in a sequence of executions (see :doc:`Run` for detailed description of formatting
            requirements and options).

        reset_clock : bool : default :keyword:`True`
            if True, resets the :py:class:`CentralClock <TimeScale.CentralClock>` to 0 before a sequence of executions.

        initialize : bool default :keyword:`False`
            if :keyword:`True`, calls the :py:meth:`initialize <System_Base.initialize>` method of the system before a
            sequence of executions.

        targets : List[input] or np.ndarray(input) : default `None`
            the target values for the MonitoringMechanisms of the system for each execution (used for learning).
            The length (of the outermost level if a nested list, or lowest axis if an ndarray) must be equal to that
            of ``inputs``.

        learning : bool :  default `None`
            enables or disables learning during execution.
            If it is not specified, the current state is left intact.
            If it is :keyword:`True`, learning is forced on; if it is :keyword:`False`, learning is forced off.

        call_before_trial : Function : default= `None`
            called before each trial in the sequence is executed.

        call_after_trial : Function : default= `None`
            called after each trial in the sequence is executed.

        call_before_time_step : Function : default= `None`
            called before each time_step of each trial is executed.

        call_after_time_step : Function : default= `None`
            called after each time_step of each trial is executed.

        time_scale : TimeScale :  default TimeScale.TRIAL
            specifies whether mechanisms are executed for a single time step or a trial.

        Returns
        -------

        COMMMENT:
            OLD;  CORRECT?
            <system>.results : List[outputState.value]
                list of the value of the outputState for each `TERMINAL` mechanism of the system returned for
                each execution.
        COMMMENT
        <system>.results : List[Mechanism.OutputValue]
            list of the OutputValue for each `TERMINAL` mechanism of the system returned for each execution.

        """
        if self.scheduler is None:
            self.scheduler = Scheduler(system=self)

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
                   time_scale=time_scale,
                   termination_conditions=termination_conditions,
                   clock=clock,
                   context=context)

    def _report_system_initiation(self, clock=CentralClock):
        """Prints iniiation message, time_step, and list of processes in system being executed
        """

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        if clock.time_step == 0:
            print("\n\'{}\'{} executing with: **** (time_step {}) ".
                  format(self.name, system_string, clock.time_step))
            processes = list(process.name for process in self.processes)
            print("- processes: {}".format(processes))
            if np.size(self.input, 0) == 1:
                input_string = ''
            else:
                input_string = 's'
            print("- input{}: {}".format(input_string, self.input.tolist()))

        else:
            print("\n\'{}\'{} executing ********** (time_step {}) ".
                  format(self.name, system_string, clock.time_step))

    def _report_system_completion(self, clock=CentralClock):
        """Prints completion message and outputValue of system
        """

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        # Print output value of primary (first) outputState of each terminal Mechanism in System
        # IMPLEMENTATION NOTE:  add options for what to print (primary, all or monitored outputStates)
        print("\n\'{}\'{} completed ***********(time_step {})".format(self.name, system_string, clock.time_step))
        for mech_tuple in self._terminal_mech_tuples:
            if mech_tuple.mechanism.phaseSpec == (clock.time_step % self.numPhases):
                print("- output for {0}: {1}".
                      format(mech_tuple.mechanism.name,
                             re.sub('[\[,\],\n]','',str(["{:0.3}".
                                                format(float(i)) for i in mech_tuple.mechanism.outputState.value]))))
        if self.learning:
            from PsyNeuLink.Components.Projections.LearningProjection import TARGET_MSE
            for mech in self.targetMechanisms:
                print("\n- MSE: {:0.3}".
                      format(float(mech.outputStates[TARGET_MSE].value)))


    # TBI:
    # class InspectOptions(AutoNumber):
    #     """Option value keywords for `inspect` and `show` methods.
    #     """
    #     ALL = ()
    #     """Show all values.
    #     """
    #     EXECUTION_SETS = ()
    #     """Show `execution_sets` attribute."""
    #     ExecutionList = ()
    #     """Show `executionList` attribute."""
    #     ATTRIBUTES = ()
    #     """Show system's attributes."""
    #     ALL_OUTPUTS = ()
    #     """"""
    #     ALL_OUTPUT_LABELS = ()
    #     """"""
    #     PRIMARY_OUTPUTS = ()
    #     """"""
    #     PRIMARY_OUTPUT_LABELS = ()
    #     """"""
    #     MONITORED_OUTPUTS = ()
    #     """"""
    #     MONITORED_OUTPUT_LABELS = ()
    #     """"""
    #     FLAT_OUTPUT = ()
    #     """"""
    #     DICT_OUTPUT = ()
    #     """"""

    def show(self, options=None):
        """Print ``execution_sets``, ``executionList``, `ORIGIN`, `TERMINAL` mechanisms,
        `TARGET` mechahinsms, ``outputs`` and their labels for the system.

        Arguments
        ---------

        options : InspectionOptions
            [TBI]
        """

        # # IMPLEMENTATION NOTE:  Stub for implementing options:
        # if options and self.InspectOptions.ALL_OUTPUT_LABELS in options:
        #     pass

        print ("\n---------------------------------------------------------")
        print ("\n{0}".format(self.name))


        print ("\n\tControl enabled: {0}".format(self.enable_controller))
        print ("\n\tProcesses:")

        for process in self.processes:
            print ("\t\t{} [learning enabled: {}]".format(process.name, process._learning_enabled))


        # Print execution_sets (output of toposort)
        print ("\n\tExecution sets: ".format(self.name))
        # Sort for consistency of output
        execution_sets_sorted = sorted(self.execution_sets)
        for i in range(len(execution_sets_sorted)):
        # for i in range(len(self.execution_sets)):
            print ("\t\tSet {0}:\n\t\t\t".format(i),end='')
            print("{ ",end='')
            sorted_mechs_names_in_set = sorted(list(mech_tuple.mechanism.name
                                                    for mech_tuple in self.execution_sets[i]))
            for name in sorted_mechs_names_in_set:
                print("{0} ".format(name), end='')
            print("}")

        # Print executionList sorted by phase and including EVC mechanism

        # Sort executionList by phase
        sorted_execution_list = self.executionList.copy()


        # Sort by phaseSpec and, within each phase, by mechanism name
        sorted_execution_list.sort(key=lambda mech_tuple: mech_tuple.phase)


        # Add controller to execution list for printing if enabled
        if self.enable_controller:
            sorted_execution_list.append(MechanismTuple(self.controller, None, self.controller.phaseSpec))


        mech_names_from_exec_list = list(mech_tuple.mechanism.name for mech_tuple in self.executionList)
        mech_names_from_sorted_exec_list = list(mech_tuple.mechanism.name for mech_tuple in sorted_execution_list)

        print ("\n\tExecution list: ".format(self.name))
        phase = 0
        print("\t\tPhase {}:".format(phase))
        for mech_tuple in sorted_execution_list:
            if mech_tuple.phase != phase:
                phase = mech_tuple.phase
                print("\t\tPhase {}:".format(phase))
            print ("\t\t\t{}".format(mech_tuple.mechanism.name))

        print ("\n\tOrigin mechanisms: ".format(self.name))
        for mech_tuple in self.originMechanisms.mech_tuples_sorted:
            print("\t\t{0} (phase: {1})".format(mech_tuple.mechanism.name, mech_tuple.phase))

        print ("\n\tTerminal mechanisms: ".format(self.name))
        for mech_tuple in self.terminalMechanisms.mech_tuples_sorted:
            print("\t\t{0} (phase: {1})".format(mech_tuple.mechanism.name, mech_tuple.phase))
            for output_state_name in mech_tuple.mechanism.outputStates:
                print("\t\t\t{0}".format(output_state_name))

        # if any(process.learning for process in self.processes):
        if self.learning:
            print ("\n\tTarget mechanisms: ".format(self.name))
            for mech_tuple in self.targetMechanisms.mech_tuples:
                print("\t\t{0} (phase: {1})".format(mech_tuple.mechanism.name, mech_tuple.phase))

        print ("\n---------------------------------------------------------")


    def inspect(self):
        """Return dictionary with system attributes and values

        Diciontary contains entries for the following attributes and values:

            :keyword:`PROCESSES`: list of processes in system

            :keyword:`MECHANISMS`: list of all mechanisms in the system

            :keyword:`ORIGIN_MECHANISMS`: list of ORIGIN mechanisms

            :keyword:`INPUT_ARRAY`: ndarray of the inputs to the ORIGIN mechanisms

            :keyword:`RECURRENT_MECHANISMS`:  list of INITALIZE_CYCLE mechanisms

            :keyword:`RECURRENT_INIT_ARRAY`: ndarray of initial_values

            :keyword:`TERMINAL_MECHANISMS`:list of TERMINAL mechanisms

            :keyword:`OUTPUT_STATE_NAMES`: list of outputState names corrresponding to 1D arrays in output_value_array

            :keyword:`OUTPUT_VALUE_ARRAY`:3D ndarray of 2D arrays of output.value arrays of outputStates for all
            `TERMINAL` mechs

            :keyword:`NUM_PHASES_PER_TRIAL`:number of phases required to execute all mechanisms in the system

            :keyword:`MONITORING_MECHANISMS`:list of MONITORING mechanisms

            `TARGET`:list of TARGET mechanisms

            :keyword:`LEARNING_PROJECTION_RECEIVERS`:list of MappingProjections that receive learning projections

            :keyword:`CONTROL_MECHANISMS`:list of CONTROL mechanisms

            :keyword:`CONTROL_PROJECTION_RECEIVERS`:list of parameterStates that receive learning projections

        Returns
        -------
        Dictionary of system attributes and values : dict

        """

        input_array = []
        for mech in list(self.originMechanisms.mechanisms):
            input_array.append(mech.value)
        input_array = np.array(input_array)

        recurrent_init_array = []
        for mech in list(self.recurrentInitMechanisms.mechanisms):
            recurrent_init_array.append(mech.value)
        recurrent_init_array = np.array(recurrent_init_array)

        output_state_names = []
        output_value_array = []
        for mech in list(self.terminalMechanisms.mechanisms):
            output_value_array.append(mech.outputValue)
            for name in mech.outputStates:
                output_state_names.append(name)
        output_value_array = np.array(output_value_array)

        from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
        from PsyNeuLink.Components.Projections.LearningProjection import LearningProjection
        learning_projections = []
        controlled_parameters = []
        for mech in list(self.mechanisms):
            for parameter_state in mech.parameterStates:
                try:
                    for projection in parameter_state.receivesFromProjections:
                        if isinstance(projection, ControlProjection):
                            controlled_parameters.append(parameter_state)
                except AttributeError:
                    pass
            for output_state in mech.outputStates:
                try:
                    for projection in output_state.sendsToProjections:
                        for parameter_state in projection.paramaterStates:
                            for sender in parameter_state.receivesFromProjections:
                                if isinstance(sender, LearningProjection):
                                    learning_projections.append(projection)
                except AttributeError:
                    pass

        inspect_dict = {
            PROCESSES: self.processes,
            MECHANISMS: self.mechanisms,
            ORIGIN_MECHANISMS: self.originMechanisms.mechanisms,
            INPUT_ARRAY: input_array,
            RECURRENT_MECHANISMS: self.recurrentInitMechanisms,
            RECURRENT_INIT_ARRAY: recurrent_init_array,
            TERMINAL_MECHANISMS: self.terminalMechanisms.mechanisms,
            OUTPUT_STATE_NAMES: output_state_names,
            OUTPUT_VALUE_ARRAY: output_value_array,
            NUM_PHASES_PER_TRIAL: self.numPhases,
            MONITORING_MECHANISMS: self.monitoringMechanisms,
            TARGET_MECHANISMS: self.targetMechanisms,
            LEARNING_PROJECTION_RECEIVERS: learning_projections,
            CONTROL_MECHANISMS: self.controlMechanism,
            CONTROL_PROJECTION_RECEIVERS: controlled_parameters,
        }

        return inspect_dict

    def _toposort_with_ordered_mech_tuples(self, data):
        """Returns a single list of dependencies, sorted by mech_tuple[MECHANISM].name"""
        result = []
        for dependency_set in toposort(data):
            d_iter = iter(dependency_set)
            result.extend(sorted(dependency_set, key=lambda item : next(d_iter).mechanism.name))
        return result

    def _cache_state(self):

        # http://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
        # import pickle
        #
        # a = {'hello': 'world'}
        #
        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open('filename.pickle', 'rb') as handle:
        #     b = pickle.load(handle)
        #
        # print a == b

        # >>> import dill
        # >>> pik = dill.dumps(d)

        # import pickle
        # with open('cached_PNL_sys.pickle', 'wb') as handle:
        #     pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # import dill
        # self.cached_system = dill.dumps(self, recurse=True)

        # def mechanisms_cache:
        #     self.input_value = []
        #     self.value= []
        #     self.output_value = []
        #
        # for mech in self.mechanisms:
        #     for
        pass

    def _restore_state(self):
        pass

    @property
    def mechanisms(self):
        """List of all mechanisms in the system

        Returns
        -------
        all mechanisms in the system : List[mechanism]

        """
        return self._allMechanisms.mechanisms

    # # MODIFIED 11/1/16 NEW:
    # @property
    # def processes(self):
    #     return sorted(self._processList.processes)

    @property
    def inputValue(self):
        """Value of input to system

        Returns
        -------
        value of input to system : ndarray
        """
        return self.variable

    @property
    def numPhases(self):
        """Number of phases required to execute all ProcessingMechanisms in the system

        Equals maximum phase value of ProcessingMechanisms in the system + 1

        Returns
        -------
        number of phases in system : int

        """
        return self._phaseSpecMax + 1

    @property
    def execution_graph_mechs(self):
        """Mechanisms whose mech_tuples appear as keys in self.executionGraph

        :rtype: list of Mechanism objects
        """
        return list(mech_tuple[0] for mech_tuple in self.executionGraph)

    def show_graph(self, output_fmt='pdf', direction = 'BT'):
        """Generate simple visualization of execution graph, showing dimensions

        Arguments
        ---------

        output_fmt : 'jupyter' or 'pdf'
            'pdf' will generate and open a pdf with the visualization,

            'jupyter' will simply return graphviz graph the object (ideal for working in jupyter/ipython notebooks)

        direction : 'BT', 'TB', 'LR', or 'RL' correspond to bottom to top, top to bottom, left to right, and right to left
            rank direction of graph


        """
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import LearningMechanism

        import graphviz as gv

        system_graph = self.graph
        learning_graph=self.learningGraph

        # build graph and configure visualisation settings
        G = gv.Digraph(engine = "dot",
                       node_attr  = {'fontsize':'12', 'fontname': 'arial', 'shape':'oval'},
                       edge_attr  = {'arrowhead':'halfopen', 'fontsize': '10', 'fontname': 'arial'},
                       graph_attr = {"rankdir" : direction} )

        # work with system graph
        rcvrs = list(system_graph.keys())
        # loop through receivers
        for rcvr in rcvrs:
            if isinstance(rcvr[0], ObjectiveMechanism) or isinstance(rcvr[0], LearningMechanism):
                continue
            rcvr_name = rcvr[0].name
            rcvr_shape = rcvr[0].variable.shape[1]
            rcvr_label = " {} ({}) ".format(rcvr_name, rcvr_shape)

            # loop through senders
            sndrs = system_graph[rcvr]
            for sndr in sndrs:
                sndr_name = sndr[0].name
                sndr_shape = sndr[0].variable.shape[1]
                sndr_label = " {} ({}) ".format(sndr_name, sndr_shape)

                # find edge name
                projs = sndr[0].outputState.sendsToProjections
                for proj in projs:
                    if proj.receiver.owner == rcvr[0]:
                        edge_name = proj.name
                        edge_shape = proj.matrix.shape
                edge_label = " {} {} ".format(edge_name, edge_shape)
                G.edge(sndr_label, rcvr_label, label = edge_label)

        if   output_fmt == 'pdf':
            G.view(self.name.replace(" ", "-"), cleanup=True)
        elif output_fmt == 'jupyter':
            return G

    def show_graph_with_learning(self, output_fmt='pdf', direction = 'BT', learning_color='green'):
        """Generate visualization of interconnections between all mechanisms and projections, including all learning machinery

        Arguments
        ---------

        output_fmt : 'jupyter' or 'pdf'
            pdf to generate and open a pdf with the visualization,
            jupyter to simply return the object (ideal for working in jupyter/ipython notebooks)

        direction : 'BT', 'TB', 'LR', or 'RL' correspond to bottom to top, top to bottom, left to right, and right to left
            rank direction of graph

        learning_color : determines with what color to draw all the learning machinery

        Returns
        -------

        Graphviz graph object if output_fmt is 'jupyter'

        """
        from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.ObjectiveMechanism import ObjectiveMechanism
        from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.LearningMechanisms.LearningMechanism import LearningMechanism
        from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection

        import graphviz as gv

        system_graph = self.graph
        learning_graph=self.learningGraph

        # build graph and configure visualisation settings
        G = gv.Digraph(engine = "dot",
                       node_attr  = {'fontsize':'12', 'fontname': 'arial', 'shape':'oval'},
                       edge_attr  = {'arrowhead':'halfopen', 'fontsize': '10', 'fontname': 'arial'},
                       graph_attr = {"rankdir" : direction} )

        # work with system graph
        rcvrs = list(system_graph.keys())
        # loop through receivers
        for rcvr in rcvrs:
            rcvr_name = rcvr[0].name
            rcvr_label = rcvr_name

            # loop through senders
            sndrs = system_graph[rcvr]
            for sndr in sndrs:
                sndr_name = sndr[0].name
                sndr_label = sndr_name

                # find edge name
                projs = sndr[0].outputState.sendsToProjections
                for proj in projs:
                    if proj.receiver.owner == rcvr[0]:
                        edge_name = proj.name
                        draw_node = proj.has_learning_projection
                edge_label = edge_name
                #### CHANGE MADE HERE ###
                # if rcvr is learning mechanism, draw arrow with learning color
                if isinstance(rcvr[0], LearningMechanism) or isinstance(rcvr[0], ObjectiveMechanism):
                    arrow_color=learning_color
                else:
                    arrow_color="black"
                if not draw_node:
                    G.edge(sndr_label, rcvr_label, label = edge_label, color=arrow_color)
                else:

                    G.node(sndr_label, shape="oval")
                    G.node(edge_label, shape="diamond")
                    G.node(rcvr_label, shape="oval")
                    G.edge(sndr_label, edge_label, arrowhead='none')
                    G.edge(edge_label, rcvr_label)
                #### CHANGE MADE HERE ###

        rcvrs = list(learning_graph.keys())

        for rcvr in rcvrs:
                # if rcvr is projection
                if isinstance(rcvr, MappingProjection):
                    # for each sndr of rcvr
                    sndrs = learning_graph[rcvr]
                    for sndr in sndrs:
                        edge_label = rcvr.parameterStates['matrix'].receivesFromProjections[0].name
                        G.edge(sndr.name, rcvr.name, color=learning_color, label = edge_label)
                else:
                    sndrs = learning_graph[rcvr]
                    for sndr in sndrs:
                        projs = sndr.outputState.sendsToProjections
                        for proj in projs:
                            if proj.receiver.owner == rcvr:
                                edge_name = proj.name
                        G.node(rcvr.name, color=learning_color)
                        G.node(sndr.name, color=learning_color)
                        G.edge(sndr.name, rcvr.name, color=learning_color, label=edge_name)

        if   output_fmt == 'pdf':
            G.view(self.name.replace(" ", "-"), cleanup=True)
        elif output_fmt == 'jupyter':
            return G


SYSTEM_TARGET_INPUT_STATE = 'SystemInputState'

from PsyNeuLink.Components.States.OutputState import OutputState
class SystemInputState(OutputState):
    """Encodes target for the system and transmits it to a `TARGET` mechanism in the system

    Each instance encodes a `target <System.target>` to the system (also a 1d array in 2d array of
    `targets <System.targets>`) and provides it to a `MappingProjection` that projects to a `TARGET`
     mechanism of the system.

    .. Declared as a sublcass of OutputState so that it is recognized as a legitimate sender to a Projection
       in Projection._instantiate_sender()

       self.value is used to represent the item of the targets arg to system.execute or system.run

    """
    def __init__(self, owner=None, variable=None, name=None, prefs=None):
        """Pass variable to MappingProjection from Process to first Mechanism in Pathway

        :param variable:
        """
        if not name:
            self.name = owner.name + "_" + SYSTEM_TARGET_INPUT_STATE
        else:
            self.name = owner.name + "_" + name
        self.prefs = prefs
        self.sendsToProjections = []
        self.owner = owner
        self.value = variable
