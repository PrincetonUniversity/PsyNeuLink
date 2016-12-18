
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
      * :ref:`System_Overview`
      * :ref:`System_Creation`
      * :ref:`System_Structure`
         * :ref:`System_Graph`
         * :ref:`System_Mechanisms`
      * :ref:`System_Execution`
         * :ref:`System_Execution_Order`
         * :ref:`System_Execution_Phase`
         * :ref:`System_Execution_Input_And_Initialization`
         * :ref:`System_Execution_Learning`
         * :ref:`System_Execution_Control`
      * :ref:`System_Class_Reference`


.. _System_Overview:

Overview
--------

A system is a collection of processes that are executed together.  Executing a system executes all of the mechanisms
in its processes in a structured order.  Projections between mechanisms in different processes within the system
are permitted, as are recurrent projections, but projections from mechanisms in other systems are ignored
(PsyNeuLink does not support ESP).  A system can include three types of mechanisms:

* :doc:`ProcessingMechanism`
    These receive input from one or more projections, transform the input in some way, and assign the result
    as their output.

* :doc:`MonitoringMechanism`
    These monitor the output of other mechanisms for use in modifying the parameteres of projections (learning)

* :doc:`ControlMechanism`
    These monitor the output of other mechanisms for use on controlling the parameters of other mechanisms

.. _System_Creation:

Creating a System
-----------------

Systems are created by calling the :py:func:`system` function.  If no arguments are provided, a system with a
single process containing a single default mechanism will be returned (see [LINK for default] for default mechanism).
Whenever a system is created, a :doc:`ControlMechanism` is created for it and assigned as its controller.  The
controller can be specified using the :py:data:`controller <System_Base.controller>` parameter, by referencing an
existing ControlMechanism, or simply specifying its class;   if one is none specified,
a :doc:`DefaultControlMechanism` is created.

.. _System_Structure:

Structure
---------

.. _System_Graph:

Graph
~~~~~

When an instance of a system is created, a graph is constructed that describes the connections (edges) among its
mechanisms (nodes).  The graph is stored in the system's :py:data:`graph <System_Base.graph>` attribute,
as a dict of dependencies,  that can be passed to graph theoretical tools for analysis.  A system can contain
recurrent paths, such as feedback  loops, in which case the system will have a cyclic graph.  PsyNeuLink also uses
the graph of a system to determine the order in which its mechanisms are executed.  In order to execute such systems
in an orderly manner, the graph must be acyclic.  So, for execution, PsyNeuLink constructs an
:py:data:`executionGraph <System_Base.executionGraph>` from the system's :py:data:`graph <System_Base.graph>`.
If the  system is acyclic, these are the same.  However, if the graph is cyclic, then the
:py:data:`executionGraph <System_Base.executionGraph>` is a subset of the :py:data:`graph <System_Base.graph>`
in which the dependencies (edges) associated with projections that close a loop have been removed. Note that
this only impacts the order of execution;  the projections themselves remain in effect, and will be fully functional
during the execution of the affected mechanisms (see :ref:`System_Execution` below for a more detailed discussion).


.. _System_Mechanisms:

Mechanisms
~~~~~~~~~~

Mechanisms are assigned the following designations based on the position they occupy in the graph structure and/or
the role they play in a system:

    COMMENT:
        DOCUMENTATION: MAKE EACH :keyword:`<KEYWORD>` AN ATTRIBUTE OF A CLASS OF KEYWORDS WHERE THEY ARE DESCRIBED.
        keywords
    COMMENT

    :py:data:`ORIGIN <Keywords.Keywords.ORIGIN>`: receives input to the system, and begins execution;

    :py:data:`TERMINAL <Keywords.Keywords.TERMINAL>`: final point of execution, and provides an output of the system;

    :py:data:`SINGLETON <Keywords.Keywords.SINGLETON>`: both an :keyword:`ORIGIN` and a :keyword:`TERMINAL` mechanism;

    :py:data:`CYCLE <Keywords.Keywords.CYCLE>`: receives a projection that closes a recurrent loop;

    :py:data:`INITIALIZE_CYCLE <Keywords.Keywords.INITIALIZE_CYCLE>`: sends a projection that closes a recurrent loop;
    can be assigned an initial value;

    :py:data:`MONITORING <Keywords.Keywords.MONITORING>`: monitors value of another mechanism for use in learning;

    :py:data:`TARGET <Keywords.Keywords.TARGET>`: ComparatorMechanism that monitors a
    :keyword:`TERMINAL` mechanism of a process

    :py:data:`CONTROL <Keywords.Keywords.CONTROL>`: monitors the value of another mechanism used to control
    parameters values;

    :py:data:`INTERNAL <Keywords.Keywords.INTERNAL>`: processing mechanism that does not fall into any of the
    categories above.

    .. note:: Any :keyword:`ORIGIN` and :keyword:`TERMINAL` mechanisms of a system must be, respectively,
       the :keyword:`ORIGIN` or :keyword:`TERMINAL` of any process(es) to which they belong.  However, it is not
       necessarily the case that the :keyword:`ORIGIN` and/or :keyword:`TERMINAL` mechanism of a process is also the
       :keyword:`ORIGIN` and/or :keyword:`TERMINAL` of a system to which the process belongs
       (see the Chain example below).

    .. note: designations are stored in the mechanism.systems attribute (see _instantiate_graph below, and Mechanism)


COMMENT:
    Control
    ~~~~~~~

    Learning
    ~~~~~~~~

    Based on process

COMMENT

.. _System_Full_Fig:

**Components of a System**

.. figure:: _static/System_full_fig.png
   :alt: Overview of major PsyNeuLink components
   :scale: 75 %

   Two :doc:`processes <Process>` are shown, both belonging to the same :doc:`system <System>`.  Each process has a
   series of :doc:`ProcessingMechanisms <ProcessingMechanism>` linked by :doc:`MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism.  The latter projects to a
   :doc:`MonitoringMechanism <MonitoringMechanism>` that is used to drive :doc:`learning <LearningProjection>` in
   Process B. It also projects to a :doc:`ControlMechanism <ControlMechanism>` that :doc:`controls <ControlProjection>`
   ProcessingMechanisms in both Processes A and B.  Note that while the mechanisms and projections responsible for
   learning belong to, and are restricted to a single process, those responsible for control belong to the system and
   can monitor and/or control mechanisms belonging to more than one process.


.. _System_Execution:

Execution
---------

A system can be executed by calling either its :py:meth:`execute <System_Base.execute>` or
:py:meth:`run <System_Base.run>` methods. :py:meth:`execute <System_Base.execute>` executes each mechanism in the
system once, whereas :py:meth:`run <System_Base.run>` allows a series of executions to be carried out.

.. _System_Execution_Order:

Order
~~~~~
Mechanisms are executed in a topologically sorted order, based on the order in which they are listed in their
processes. When a mechanism is executed, it receives input from any other mechanisms that project to it within the
system,  but not from mechanisms outside the system (PsyNeuLink does not support ESP).  The order of execution is
represented by the :py:data:`executionGraph <System_Base.executionGraph>`, which is a subset of the system's graph that
has been "pruned" to be acyclic (i.e., devoid of recurrent loops).  While the
:py:data:`executionGraph <System_Base.executionGraph>` is acyclic, all recurrent projections in the system remain
intact during execution and can be initialized at the start of execution (see below).

.. _System_Execution_Phase:

Phase
~~~~~
Execution occurs in passes through a system called phases.  Each phase corresponds to a single time_step.
When executing a system in trial mode, a trial is defined as the number of phases (time_steps) required to execute
a trial of every mechanism in the system.  During each phase of execution, only the mechanisms assigned to that phase
are executed.   Mechanisms are assigned a phase where they are listed in the pathway of a process (see Process).
When a mechanism is executed, it receives input from any other mechanisms that project to it within the system.

.. _System_Execution_Input_And_Initialization:

Input and Initialization
~~~~~~~~~~~~~~~~~~~~~~~~
The input to a system is specified in the ``input`` argument of either its :py:meth:`execute <System_Base.execute>` or
:py:meth:`run <System_Base.run>` method. In both cases, the input for a single trial must be a list or ndarray of
values, each of which is an appropriate input for the corresponding :keyword:`ORIGIN` mechanism (listed in
:py:data:`originMechanisms <System_Base.originMechanisms>`). If the :py:meth:`execute <System_Base.execute>` method
is used, input for only a single trial is provided, and only a single trial is executed.  The
:py:meth:`run <System_Base.run>` method can be used for a sequence of executions (time_steps or trials),
by providing it with a list or ndarray of inputs, one for each round of execution.  In both cases, two other types of
input can be provided:  a list or ndarray of initialization values, and a list or ndarray of target values.
Initialization values are assigned, at the start execution, as input to mechanisms that close recurrent loops
(designated as :keyword:`INITIALIZE_CYCLE`, and listed in
:py:data:`recurrentInitMechanisms <System_Base.recurrentInitMechanisms>`), and target values are assigned to the
:py:data:`target <ComparatorMechanism.ComparatorMechanism.target>` attribute of :doc:`ComparatorMechanisms`
(see learning below;  also, see :doc:`Run` for additional details of formatting input specifications).

.. _System_Execution_Learning:

Learning
~~~~~~~~
The system will execute learning for any process that specifies it.  Learning is executed for each process
after all processing mechanisms in the system have been executed, but before the controller is executed (see below).
The stimuli (both inputs and targets for learning) can be specified in one of two formats (list or dict),
that are described in the :doc:`Run` module.  Both formats require that an input be provided for each
:py:data:`ORIGIN <Keywords.Keywords.ORIGIN>` mechanism of the system (listed in its
:py:data:`originMechanisms <System_Base.originMechanisms>` attribute), and that a target be provided for each
:py:data:`TARGET` <Keywords.Keywords.TARGET>` mechanism (listed in its
:py:data:`targetMechanisms <System_Base.targetMechanisms>` attribute).

.. note::
   A :py:data:`targetMechanism <Process.Process_Base.targetMechanisms>` of a process is not necessarily a
   :py:data:`targetMechanism <System_Base.targetMechanisms>` of the system to which it belongs
   (see :ref:`LearningProjection_Targets`).

.. _System_Execution_Control:

Control
~~~~~~~
Every system is associated with a single :py:data:`controller <System_Base.controller>`.  The controller monitors
the outputState(s) of one or more mechanisms in the system
COMMENT:
    (listed in :py:data:`monitoredOutputStates <System_Base.monitoredOutputStates>`),
COMMENT
and uses that information to set the value of parameters for those or other mechanisms in the system, or their functions
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
from collections import UserList, Iterable

from toposort import *
# from PsyNeuLink.Globals.toposort.toposort import toposort

from PsyNeuLink.Globals.Registry import register_category
from PsyNeuLink.Components.ShellClasses import *
from PsyNeuLink.Components.Process import ProcessInputState, ProcessList, ProcessTuple
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismList, MechanismTuple
from PsyNeuLink.Components.Mechanisms.Mechanism import MonitoredOutputStatesOption
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.ComparatorMechanism import ComparatorMechanism
from PsyNeuLink.Components.Mechanisms.MonitoringMechanisms.MonitoringMechanism import MonitoringMechanism_Base
from PsyNeuLink.Components.Mechanisms.ControlMechanisms.ControlMechanism import ControlMechanism_Base

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# inspect() keywords
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

from PsyNeuLink.Components import SystemDefaultControlMechanism
from PsyNeuLink.Components.Process import process


# System factory method:
@tc.typecheck
def system(default_input_value=None,
           processes:list=[],
           initial_values:dict={},
           controller=SystemDefaultControlMechanism,
           enable_controller:bool=False,
           monitor_for_control:list=[MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES],
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
    monitor_for_control=:keyword:`None`,  \
    params=None,                              \
    name=None,                                \
    prefs=None)

    Factory method for System: returns instance of System.

    If called with no arguments, returns an instance of System with a single default process and mechanism;
    if called with a name string, that is used as the name of the instance of System returned;
    if a params dictionary is included, it is passed to the instantiated system.

    See :class:`System_Base` for class description

    Arguments
    ---------

    default_input_value : list or ndarray of values : default default inputs for :keyword:`ORIGIN` mechanism of each Process
        the input to the system if none is provided in a call to the execute() method or run() function.
        Should contain one item corresponding to the input of each :keyword:`ORIGIN` mechanism in the system.

        COMMENT:
            REPLACE DefaultProcess BELOW USING Inline markup
        COMMENT

    processes : list of process specifications : default list(''DefaultProcess'')
        a list of the processes to include in the system.
        Process specifications can be an instance, the class name (creates a default Process, or a specification
        dictionary (see Processes for details).

    initial_values : dict of mechanism:value entries
        a dictionary of values used to initialize mechanisms that close recurrent loops (designated as
        :keyword:`INITIALIZE_CYCLE`). The key for each entry is a mechanism object, and the value is a number,
        list or 1d np.array that must be compatible with the format of the first item of the mechanism's value
        (i.e., mechanism.value[0]).

    controller : ControlMechanism : default DefaultController
        specifies the ControlMechanism used to monitor the value of the outputState(s) for mechanisms specified in
        monitor_for_control, and specify the value of ControlProjections in the system.

    enable_controller :  bool : default :keyword:`False`
        specifies whether the :py:data:`controller <System_Base.controller>` is executed during system execution.

    monitor_for_control : list of OutputState objects or specifications : default None
        specifies the outputStates of the :keyword:`TERMINAL` mechanisms in the system to be monitored by its controller
        (see :ref:`ControlMechanism_Monitored_OutputStates` for specifying the ``monitor_for_control`` argument).

    params : dict : default None
        a dictionary that can include any of the parameters above; use the parameter's name as the keyword for its entry
        values in the dictionary will override argument values

    name : str : default System-<index>
        a string used for the name of the system
        (see Registry module for conventions used in naming, including for default and duplicate names)

    prefs : PreferenceSet or specification dict : System.classPreferences
        the PreferenceSet for system (see ComponentPreferenceSet module for specification of PreferenceSet)

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
                       initial_values=initial_values,
                       enable_controller=enable_controller,
                       monitor_for_control=monitor_for_control,
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
    monitor_for_control=:keyword:`None`,      \
    params=None,                              \
    name=None,                                \
    prefs=None)

    Abstract class for System.

    .. note::
       Systems should NEVER be instantiated by a direct call to the base class.
       They should be instantiated using the :class:`system` factory method (see it for description of parameters).

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
                                kwController: DefaultController,
                                TIME_SCALE: TimeScale.TRIAL}
       Class methods
       -------------
        - _validate_variable(variable, context):  insures that variable is 3D np.array (one 2D for each Process)
        - _instantiate_attributes_before_function(context):  calls self._instantiate_graph
        - _instantiate_function(context): validates only if self.prefs.paramValidationPref is set
        - _instantiate_graph(inputs, context):  instantiates Processes in self.process and constructs executionList
        - identify_origin_and_terminal_mechanisms():  assign self.originMechanisms and self.terminalMechanisms
        - _assign_output_states():  assign outputStates of System (currently = terminalMechanisms)
        - execute(inputs, time_scale, context):  executes Mechanisms in order specified by executionList
        - variableInstanceDefaults(value):  setter for variableInstanceDefaults;  does some kind of error checking??

       SystemRegistry
       --------------
        Register in SystemRegistry, which maintains a dict for the subclass, a count for all instances of it,
         and a dictionary of those instances

        TBI: MAKE THESE convenience lists, akin to self.terminalMechanisms
        + input (list): contains Process.input for each process in self.processes
        + output (list): containts Process.ouput for each process in self.processes
        [TBI: + inputs (list): each item is the Process.input object for the corresponding Process in self.processes]
        [TBI: + outputs (list): each item is the Process.output object for the corresponding Process in self.processes]
    COMMENT

    Attributes
    ----------

    processes : list of Process objects
        list of processes in the system specified by the process parameter.

        .. can be appended with prediction processes by EVCMechanism
           used with self.inputs to constsruct self.process_tuples

        .. _processList : ProcessList
            Provides access to (process, input) tuples.
            Derived from self.inputs and self.processes.
            Used to construct :py:data:`executionGraph <System_Base.executionGraph>` and execute the System

    controller : ControlMechanism : default DefaultController
        the ControlMechanism used to monitor the value of the outputState(s) for mechanisms specified in
        ``monitor_for_control`` argument, and specify the value of ControlProjections in the system.

    enable_controller :  bool : default :keyword:`False`
        determines whether the :py:data:`controller <System_Base.controller>` is executed during system execution.

    graph : OrderedDict
        contains a graph of all of the mechanisms in the system.
        Each entry specifies a set of <Receiver>: {sender, sender...} dependencies.
        The key of each entry is a receiver mech_tuple, and
        the value is a set of mech_tuples that send projections to that receiver.
        If a key (receiver) has no dependents, its value is an empty set.

    executionGraph : OrderedDict
        contains an acyclic subset of the system's graph, hierarchically organized by a toposort.
        Used to specify the order in which mechanisms are executed.

    execution_sets : list of sets
        contains a list of mechanism sets.
        Each set contains mechanism to be executed at the same time.
        The sets are ordered in the sequence with which they should be executed.

    executionList : list of Mechanism objects
        contains a list of mechanisms in the order in which they are executed.
        The list is a random sample of the permissible orders constrained by the
        :py:data:`executionGraph <System_Base.executionGraph>`

    mechanisms : list of Mechanism objects
        contains a list of all mechanisms in the system.

        .. property that points to _allMechanisms.mechanisms (see below)

    mechanismsDict : Dict[Mechanism:Process]
        contains a dictionary of all mechanisms in the system, listing the processes to which they belong.
        The key of each entry is a Mechanism object, and the value of each entry is a list of processes.

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
            Tuples for all ComparatorMechanisms  in the system that are a :keyword:`TERMINAL` for at least on process
            to which it belongs and that process has learning enabled --  the criteria for being a target used in
            learning.

        .. _learning_mech_tuples : list of (mechanism, runtime_param, phaseSpec) tuples
            Tuples for all LearningMechanisms in the system (used for learning).

        .. _control_mech_tuple : list of a single (mechanism, runtime_param, phaseSpec) tuple
            Tuple for the controller in the system.

    originMechanisms : MechanismList
        contains all :keyword:`ORIGIN` mechanisms in the system (i.e., that don't receive projections from any other
        mechanisms.

        .. based on _origin_mech_tuples
           system.input contains the input to each :keyword:`ORIGIN` mechanism

    terminalMechanisms : MechanismList
        contains all :keyword:`TERMINAL` mechanisms in the system (i.e., that don't project to any other mechanisms).

        .. based on _terminal_mech_tuples
           system.ouput contains the output of each TERMINAL mechanism

    recurrentInitMechanisms : MechanismList
        contains mechanisms with recurrent projections that are candidates for \
        :ref:`initialization <System_Execution_Input_And_Initialization>`.

    monitoringMechanisms : MechanismList)
        contains all MONITORING mechanisms in the system (used for learning; based on _monitoring_mech_tuples).

    targetMechanisms : MechanismList)
        contains all TARGET mechanisms in the system (used for learning; based on _target_mech_tuples).

    COMMENT:
       IS THIS CORRECT:
    COMMENT

    controlMechanisms : MechanismList
        contains controller (CONTROL mechanism) of the system (based on _control_mech_tuples).

    value : 3D ndarray
        contains an array of 2D arrays, each of which is the outputValue of a TERMINAL mechanism in the system.

        .. _phaseSpecMax : int
            Maximum phase specified for any mechanism in system.  Determines the phase of the last (set of)
            ProcessingMechanism(s) to be executed in the system.

    numPhases : int
        number of phases for system (read-only).

        .. implemented as an @property attribute; = _phaseSpecMax + 1

    initial_values : list or ndarray of values :  default array of zero arrays
        values used to initialize mechanisms that close recurrent loops (designated as :keyword:`INITIALIZE_CYCLE`).
        Must be the same length as the list of :keyword:`INITIAL_CYCLE` mechanisms in the system contained in
        :py:data:`recurrentInitMechanisms <System_Base.recurrentInitMechanisms>.

    timeScale : TimeScale  : default TimeScale.TRIAL
        determines the default TimeScale value used by mechanisms in the system.

    results : List[outputState.value]
        list of return values (outputState.value) from the sequence of executions.

    name : str : default System-<index>
        the name of the system;
        Specified in the name argument of the call to create the system;
        if not is specified, a default is assigned by SystemRegistry
        (see :doc:`Registry` for conventions used in naming, including for default and duplicate names).[LINK]


    prefs : PreferenceSet or specification dict : System.classPreferences
        the PreferenceSet for system.
        Specified in the prefs argument of the call to create the system;  if it is not specified, a default is
        assigned using ``classPreferences`` defined in __init__.py
        (see Description under PreferenceSet for details).[LINK]

    """

    componentCategory = kwProcessFunctionCategory
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
                 params=None,
                 name=None,
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
                                                 params=params)

        # MODIFIED 12/15/16 OLD:
        # self.pathway = None
        # MODIFIED 12/15/16 END
        self.outputStates = {}
        self._phaseSpecMax = 0
        self.function = self.execute

        register_category(entry=self,
                          base_class=System_Base,
                          name=name,
                          registry=SystemRegistry,
                          context=context)

        if not context:
            # context = INITIALIZING + self.name
            context = INITIALIZING + self.name + kwSeparator + kwSystemInit

        super().__init__(variable_default=default_input_value,
                         param_defaults=params,
                         name=self.name,
                         prefs=prefs,
                         context=context)

        # Get/assign controller

        # Controller is DefaultControlMechanism
        from PsyNeuLink.Components.Mechanisms.ControlMechanisms.DefaultControlMechanism import DefaultControlMechanism
        if self.paramsCurrent[kwController] is DefaultControlMechanism:
            # Get DefaultController from MechanismRegistry
            from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismRegistry
            self.controller = list(MechanismRegistry[kwDefaultControlMechanism].instanceDict.values())[0]
        # Controller is not DefaultControlMechanism
        else:
            # Instantiate specified controller
            # MODIFIED 11/6/16 OLD:
            self.controller = self.paramsCurrent[kwController](params={SYSTEM: self})
            # # MODIFIED 11/6/16 NEW:
            # self.controller = self.paramsCurrent[kwController](system=self)
            # MODIFIED 11/6/16 END

        # Check whether controller has inputs, and if not then disable
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
                self.controller.phaseSpec = self.paramsCurrent[kwControllerPhaseSpec]
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

        controller = target_set[kwController]
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
        self._instantiate_processes(inputs=self.variable, context=context)
        self._instantiate_graph(context=context)

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

# FIX: ALLOW Projections (??ProjectionTiming TUPLES) TO BE INTERPOSED BETWEEN MECHANISMS IN PATHWAY
# FIX: AUGMENT LinearMatrix TO USE FULL_CONNECTIVITY_MATRIX IF len(sender) != len(receiver)
    def _instantiate_processes(self, inputs=None, context=None):
        """Instantiate processes of system

        Use self.processes (populated by self.paramsCurrent[kwProcesses] in Function._assign_args_to_param_dicts
        If self.processes is empty, instantiate default process by calling process()
        Iterate through self.processes, instantiating each (including the input to each input projection)
        If inputs is specified, check that it's length equals the number of processes
        If inputs is not specified, compose from the input for each Process (value specified or, if None, default)
        Note: specification of input for system takes precedence over specification for processes

        # ??STILL THE CASE, OR MOVED TO _instantiate_graph:
        Iterate through Process._mech_tuples for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.executionGraph
        """

        self.variable = []
        self.mechanismsDict = {}
        self._all_mech_tuples = []
        self._allMechanisms = MechanismList(self, self._all_mech_tuples)

        # Get list of processes specified in arg to init, possiblly appended by EVCMechanism (with prediction processes)
        processes_spec = self.processes

        # Assign default Process if kwProcess is empty, or invalid
        if not processes_spec:
            from PsyNeuLink.Components.Process import Process_Base
            processes_spec.append(ProcessTuple(Process_Base(), None))

        # If inputs to system are specified, number must equal number of processes with origin mechanisms
        if not inputs is None and len(inputs) != len(self.originMechanisms):
            raise SystemError("Number of inputs ({0}) must equal number of processes ({1}) in {} ".
                              format(len(inputs), len(self.originMechanisms)),
                              self.name)

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS

        # Convert all entries to (process, input) tuples, with None as filler for absent inputs
        for i in range(len(processes_spec)):

            # Entry is not a tuple
            #    presumably it is a process spec, so enter it as first item of ProcessTuple
            if not isinstance(processes_spec[i], tuple):
                processes_spec[i] = ProcessTuple(processes_spec[i], None)

            # Entry is a tuple but not a ProcessTuple, so convert it
            if isinstance(processes_spec[i], tuple) and not isinstance(processes_spec[i], ProcessTuple):
                processes_spec[i] = ProcessTuple(processes_spec[i][0], processes_spec[i][1])

            if inputs is None:
                # FIX: ASSIGN PROCESS INPUTS TO SYSTEM INPUTS
                process = processes_spec[i].process
                process_input = []
                for process_input_state in process.processInputStates:
                    process_input.extend(process_input_state.value)
                processes_spec[i] = ProcessTuple(process, process_input)
            # If input was provided on command line, assign that to input item of tuple
            else:
                # Assign None as input to processes implemented by controller (controller provides their input)
                #    (e.g., prediction processes implemented by EVCMechanism)
                if processes_spec[i].process._isControllerProcess:
                    processes_spec[i] = ProcessTuple(processes_spec[i].process, None)
                else:
                    # Replace input item in tuple with one from variable
                    processes_spec[i] = ProcessTuple(processes_spec[i].process, inputs[i])

            # Validate input
            if (not processes_spec[i].input is None and
                    not isinstance(processes_spec[i].input,(numbers.Number, list, np.ndarray))):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, processes_spec[i].input))

            process = processes_spec[i].process
            input = processes_spec[i].input
            self.variable.append(input)

            # If process item is a Process object, assign input as default
            if isinstance(process, Process):
                if not input is None:
                    process._assign_defaults(variable=input, context=context)

            # Otherwise, instantiate Process
            if not isinstance(process, Process):
                if inspect.isclass(process) and issubclass(process, Process):
                    # FIX: MAKE SURE THIS IS CORRECT
                    # Provide self as context, so that Process knows it is part of a Sysetm (and which one)
                    # Note: this is used by Process._instantiate_pathway() when instantiating first Mechanism
                    #           in Pathway, to override instantiation of projections from Process.input_state
                    process = Process(default_input_value=input, context=self)
                elif isinstance(process, dict):
                    # IMPLEMENT:  HANDLE Process specification dict here; include input as ??param, and context=self
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

            # Get max of Process phaseSpecs
            self._phaseSpecMax = int(max(math.floor(process._phaseSpecMax), self._phaseSpecMax))

            # FIX: SHOULD BE ABLE TO PASS INPUTS HERE, NO?  PASSED IN VIA VARIABLE, ONE FOR EACH PROCESS
            # FIX: MODIFY _instantiate_pathway TO ACCEPT input AS ARG
            # NEEDED?? WASN"T IT INSTANTIATED ABOVE WHEN PROCESS WAS INSTANTIATED??
            # process._instantiate_pathway(self.variable[i], context=context)

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
                                offending_value = PHASE_SPEC
                            else:
                                offending_tuple_field = 'input'
                                offending_value = PARAMS
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

            # MODIFIED 10/16/16 OLD:
            # process.mechanisms = MechanismList(process, tuples_list=process._mech_tuples)
            # MODIFIED 10/16/16 NEW:
            process._allMechanisms = MechanismList(process, tuples_list=process._mech_tuples)
            # MODIFIED 10/16/16 END

        self.variable = convert_to_np_array(self.variable, 2)

        # Instantiate processList using process_tuples, and point self.processes to it
        # Note: this also points self.params[kwProcesses] to self.processes
        self.process_tuples = processes_spec
        self._processList = ProcessList(self, self.process_tuples)
        # MODIFIED 11/1/16 OLD:
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

            # Assign as TERMINAL (or SINGLETON) if it has no outgoing projections and is not a ComparatorMechanism or
            #     it projects only to ComparatorMechanism(s)
            # Note:  SINGLETON is assigned if mechanism is already a TERMINAL;  indicates that it is both
            #        an ORIGIN AND A TERMINAL and thus must be the only mechanism in its process
            if (not isinstance(sender_mech, (MonitoringMechanism_Base, ControlMechanism_Base)) and
                    all(all(isinstance(projection.receiver.owner, (MonitoringMechanism_Base, ControlMechanism_Base))
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
                                self.executionGraph[receiver_tuple].add(self._allMechanisms._get_tuple_for_mech(sender_mech))
                            # If receiver_tuple set is empty, assign sender_mech to set
                            else:
                                self.executionGraph[receiver_tuple] = {self._allMechanisms._get_tuple_for_mech(sender_mech)}
                            # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                            list(toposort(self.executionGraph))
                        # If making receiver dependent on sender produced a cycle (feedback loop), remove from graph
                        except ValueError:
                            self.executionGraph[receiver_tuple].remove(self._allMechanisms._get_tuple_for_mech(sender_mech))
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
                            self.executionGraph[receiver_tuple].add(self._allMechanisms._get_tuple_for_mech(sender_mech))
                        except KeyError:
                            self.executionGraph[receiver_tuple] = {self._allMechanisms._get_tuple_for_mech(sender_mech)}

                    if not sender_mech.systems:
                        sender_mech.systems[self] = INTERNAL

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver)

        from collections import OrderedDict
        self.graph = OrderedDict()
        self.executionGraph = OrderedDict()


        # Sort for consistency of output
        sorted_processes = sorted(self.processes, key=lambda process : process.name)

        for process in sorted_processes:
            first_mech = process.firstMechanism
            # Treat as ORIGIN if ALL projections to the first mechanism in the process are from:
            #    - the process itself (ProcessInputState
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
                            projection.sender.owner in sorted_processes or
                            # or from mechanisms within its own process (e.g., [a, b, a])
                            projection.sender.owner in list(process.mechanisms) or
                            # or from mechanisms in oher processes for which it is also an ORIGIN ([a, b, a], [a, c, a])
                            all(ORIGIN in first_mech.processes[proc] for proc in projection.sender.owner.processes)
                        for projection in input_state.receivesFromProjections)
                    for input_state in first_mech.inputStates.values()):
                # Assign its set value as empty, marking it as a "leaf" in the graph
                mech_tuple = self._allMechanisms._get_tuple_for_mech(first_mech)
                self.graph[mech_tuple] = set()
                self.executionGraph[mech_tuple] = set()
                first_mech.systems[self] = ORIGIN

            build_dependency_sets_by_traversing_projections(first_mech)

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

        self._monitoring_mech_tuples = []
        self._target_mech_tuples = []

        for mech_tuple in self._all_mech_tuples:

            if isinstance(mech_tuple.mechanism, MonitoringMechanism_Base):
                if not mech_tuple.mechanism in self._monitoring_mech_tuples:
                    self._monitoring_mech_tuples.append(mech_tuple)

            if isinstance(mech_tuple.mechanism, ComparatorMechanism):
                if not mech_tuple.mechanism in self._target_mech_tuples:
                    self._target_mech_tuples.append(mech_tuple)


        self.originMechanisms = MechanismList(self, self._origin_mech_tuples)
        self.terminalMechanisms = MechanismList(self, self._terminal_mech_tuples)
        self.recurrentInitMechanisms = MechanismList(self, self.recurrent_init_mech_tuples)
        self.controlMechanism = MechanismList(self, self._control_mech_tuple)
        self.monitoringMechanisms = MechanismList(self, self._monitoring_mech_tuples)
        self.targetMechanisms = MechanismList(self, self._target_mech_tuples)

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

        # Validate initial values
        # FIX: CHECK WHETHER ALL MECHANISMS DESIGNATED AS INITALIZE HAVE AN INITIAL_VALUES ENTRY
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
        :keyword:`INITIALIZE_CYCLE` and contained in recurrentInitMechanisms.
        """
        # FIX:  INITIALIZE PROCESS INPUTS??
        # FIX: CHECK THAT ALL MECHANISMS ARE INITIALIZED FOR WHICH mech.system[SELF]==INITIALIZE
        # FIX: ADD OPTION THAT IMPLEMENTS/ENFORCES INITIALIZATION
        # FIX: ADD SOFT_CLAMP AND HARD_CLAMP OPTIONS
        # FIX: ONLY ASSIGN ONES THAT RECEIVE PROJECTIONS
        for mech, value in self.initial_values.items():
            mech.initialize(value)

    def execute(self,
                inputs=None,
                time_scale=None,
                context=None):
        """Execute mechanisms in system at specified :ref:`phases <System_Execution_Phase>` in order \
        specified by the :py:data:`executionGraph <System_Base.executionGraph>` attribute.

        Assign inputs to :keyword:`ORIGIN` mechanisms

        Execute mechanisms in the order specified in executionList and with phases equal to
        ``CentralClock.time_step % numPhases``.

        Execute learning for processes (for those that specify it) at the appropriate phase.

        Execute controller after all mechanisms have been executed (after each numPhases)

        .. Execution:
            - the inputs arg in system.execute() or run() is provided as input to ORIGIN mechanisms (and system.input);
                As with a process, ORIGIN mechanisms will receive their input only once (first execution)
                    unless clamp_input (or SOFT_CLAMP or HARD_CLAMP) are specified, in which case they will continue to
            - execute() calls mechanism.execute() for each mechanism in its execute_graph in sequence
            - outputs of TERMINAL mechanisms are assigned as system.ouputValue
            - system.controller is executed after execution of all mechanisms in the system
            - notes:
                * the same mechanism can be listed more than once in a system, inducing recurrent processing

        Arguments
        ---------
        inputs : list or ndarray
            a list or array of input value arrays, one for each :keyword:`ORIGIN` mechanism in the system.

            .. [TBI: time_scale : TimeScale : default TimeScale.TRIAL
               specifies a default TimeScale for the system]

            .. context : str

        Returns
        -------
        output values of system : 3d ndarray
            Each item is a 2d array that contains arrays for each outputState.value of each TERMINAL mechanism

        """

        if not context:
            context = EXECUTING + self.name
        report_system_output = self.prefs.reportOutputPref and context and EXECUTING in context
        if report_system_output:
            report_process_output = any(process.reportOutputPref for process in self.processes)

        self.timeScale = time_scale or TimeScale.TRIAL

        #region ASSIGN INPUTS TO PROCESSES
        # Assign each item of input to the value of a Process.input_state which, in turn, will be used as
        #    the input to the MappingProjection to the first (origin) Mechanism in that Process' pathway
        num_origin_mechs = len(list(self.originMechanisms))
        if inputs is None:
            # # MODIFIED 12/4/16 OLD:
            # pass
            # MODIFIED 12/4/16 NEW:
            if (self.prefs.verbosePref and
                    not (not context or COMPONENT_INIT in context)):
                print("- No input provided;  default will be used: {0}")
            inputs = np.zeros_like(self.variable)
            for i in range(num_origin_mechs):
                inputs[i] = self.originMechanisms[i].variableInstanceDefault
            # MODIFIED 12/4/16 END

        else:
            num_inputs = np.size(inputs,0)
            if num_inputs != num_origin_mechs:
                # Check if inputs are of different lengths (indicated by dtype == np.dtype('O'))
                num_inputs = np.size(inputs)
                if isinstance(inputs, np.ndarray) and inputs.dtype is np.dtype('O') and num_inputs == num_origin_mechs:
                    pass
                else:
                    raise SystemError("Number of inputs ({0}) to {1} does not match "
                                      "its number of origin Mechanisms ({2})".
                                      format(num_inputs, self.name,  num_origin_mechs ))
            for i in range(num_inputs):
                input = inputs[i]
                process = self.processes[i]

                # Make sure there is an input, and if so convert it to 2D np.ndarray (required by Process
                if input is None:
                    continue
                else:
                    # Assign input as value of corresponding Process inputState
                    process._assign_input_values(input=input, context=context)
        self.inputs = inputs
        #endregion

        if report_system_output:
            self._report_system_initiation()


        #region EXECUTE MECHANISMS

        # TEST PRINT:
        # for i in range(len(self.executionList)):
        #     print(self.executionList[i][0].name)
        sorted_list = list(mech_tuple[0].name for mech_tuple in self.executionList)

        # Execute each Mechanism in self.executionList, in the order listed during its phase
        for i in range(len(self.executionList)):

            mechanism, params, phase_spec = self.executionList[i]

            if report_system_output and report_process_output:
                # Report initiation of process(es) for which mechanism is an ORIGIN
                # Sort for consistency of reporting:
                processes = list(mechanism.processes.keys())
                process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                for process in process_keys_sorted:
                    if mechanism.processes[process] in {ORIGIN, SINGLETON} and process.reportOutputPref:
                        process._report_process_initiation()

                # for process, status in mechanism.processes.items():
                #     if status in {ORIGIN, SINGLETON} and process.reportOutputPref:
                #         process._report_process_initiation()

            # Only update Mechanism on time_step(s) determined by its phaseSpec (specified in Mechanism's Process entry)
# FIX: NEED TO IMPLEMENT FRACTIONAL UPDATES (IN Mechanism.update()) FOR phaseSpec VALUES THAT HAVE A DECIMAL COMPONENT
            if phase_spec == (CentralClock.time_step % self.numPhases):
                # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
                mechanism.execute(time_scale=self.timeScale,
                                 runtime_params=params,
                                 context=context)

                # IMPLEMENTATION NOTE:  ONLY DO THE FOLLOWING IF THERE IS NOT A SIMILAR STATEMENT FOR MECHANISM ITSELF
                # Report completion of process(es) for which mechanism is a TERMINAL
                if report_system_output:
                    if report_process_output:
                        # Sort for consistency of reporting:
                        processes = list(mechanism.processes.keys())
                        process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                        for process in process_keys_sorted:
                            # MODIFIED 12/4/16 NEW:
                            if process.learning and process._learning_enabled:
                                continue
                            # MODIFIED 12/4/16 END
                            if mechanism.processes[process] == TERMINAL and process.reportOutputPref:
                                process._report_process_completion()

            if not i:
                # Zero input to first mechanism after first run (in case it is repeated in the pathway)
                # IMPLEMENTATION NOTE:  in future version, add option to allow Process to continue to provide inputs
                # FIX: USE clamp_input OPTION HERE, AND ADD HARD_CLAMP AND SOFT_CLAMP
                # # MODIFIED 10/2/16 OLD:
                # self.variable = self.variable * 0
                # # MODIFIED 10/2/16 NEW:
                # self.variable = self.inputs * 0
                # MODIFIED 10/2/16 NEWER:
                self.variable = convert_to_np_array(self.inputs, 2) * 0
                # MODIFIED 10/2/16 END
            i += 1
        #endregion


        # region EXECUTE LEARNING FOR EACH PROCESS

        # FIX: NEED TO CHECK PHASE HERE
        # Don't execute learning for simulation runs
        if not EVC_SIMULATION in context:
            for process in self.processes:
                if process.learning and process._learning_enabled:
                    # for mech in process.monitoring_mechanisms:
                    for mech_tuple in process.monitoringMechanisms.mech_tuples:
                        mech_tuple.mechanism.execute(time_scale=self.timeScale,
                                                     runtime_params=mech_tuple.params,
                                                     context=context)

                    process._execute_learning(context=context)

                    # Report output of the process if reporting is enabled for system and process
                    #   and the process has a targetMechanism
                    #   (note: it may not, as it may terminate on an internal mechanism of another process
                    #          and therefore use that process' targetMechanism as the source of its error signal)
                    if report_system_output and report_process_output and process.targetMechanisms:
                            process._report_process_completion()
        # endregion


        #region EXECUTE CONTROLLER

# FIX: 1) RETRY APPENDING TO EXECUTE LIST AND COMPARING TO THIS VERSION
# FIX: 2) REASSIGN INPUT TO SYSTEM FROM ONE DESIGNATED FOR EVC SIMULUS (E.G., StimulusPrediction)

        # Only call controller if this is not a controller simulation run (to avoid infinite recursion)
        if not EVC_SIMULATION in context and self.enable_controller:
            try:
                if self.controller.phaseSpec == (CentralClock.time_step % self.numPhases):
                    self.controller.execute(time_scale=TimeScale.TRIAL,
                                            runtime_params=None,
                                            context=context)
                    if report_system_output:
                        print("{0}: {1} executed".format(self.name, self.controller.name))

            except AttributeError:
                if not 'INIT' in context:
                    raise SystemError("PROGRAM ERROR: no controller instantiated for {0}".format(self.name))
        #endregion

        # Report completion of system execution and value of designated outputs
        if report_system_output:
            self._report_system_completion()

        return self.terminalMechanisms.outputStateValues

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

        targets : List[input] or np.ndarray(input) : default :keyword:`None`
            the target values for the MonitoringMechanisms of the system for each execution (used for learning).
            The length (of the outermost level if a nested list, or lowest axis if an ndarray) must be equal to that
            of ``inputs``.

        learning : bool :  default :keyword:`None`
            enables or disables learning during execution.
            If it is not specified, the current state is left intact.
            If it is :keyword:`True`, learning is forced on; if it is :keyword:`False`, learning is forced off.

        call_before_trial : Function : default= :keyword:`None`
            called before each trial in the sequence is executed.

        call_after_trial : Function : default= :keyword:`None`
            called after each trial in the sequence is executed.

        call_before_time_step : Function : default= :keyword:`None`
            called before each time_step of each trial is executed.

        call_after_time_step : Function : default= :keyword:`None`
            called after each time_step of each trial is executed.

        time_scale : TimeScale :  default TimeScale.TRIAL
            specifies whether mechanisms are executed for a single time step or a trial.

        Returns
        -------

        <system>.results : List[outputState.value]
            list of the value of the outputState for each :keyword:`TERMINAL` mechanism of the system returned for
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

    def _report_system_initiation(self):
        """Prints iniiation message, time_step, and list of processes in system being executed
        """

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        if CentralClock.time_step == 0:
            print("\n\'{}\'{} executing with: **** (time_step {}) ".
                  format(self.name, system_string, CentralClock.time_step))
            processes = list(process.name for process in self.processes)
            print("- processes: {}".format(processes))


        else:
            print("\n\'{}\'{} executing ********** (time_step {}) ".
                  format(self.name, system_string, CentralClock.time_step))

    def _report_system_completion(self):
        """Prints completion message and outputValue of system
        """

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        # Print output value of primary (first) outputState of each terminal Mechanism in System
        # IMPLEMENTATION NOTE:  add options for what to print (primary, all or monitored outputStates)
        print("\n\'{}\'{} completed ***********(time_step {})".format(self.name, system_string, CentralClock.time_step))
        for mech_tuple in self._terminal_mech_tuples:
            if mech_tuple.mechanism.phaseSpec == (CentralClock.time_step % self.numPhases):
                print("- output for {0}: {1}".format(mech_tuple.mechanism.name,
                                                 re.sub('[\[,\],\n]','',str(["{:0.3}".format(float(i)) for i in mech_tuple.mechanism.outputState.value]))))


    class InspectOptions(AutoNumber):
        """Option value keywords for inspect() and show() methods

        Values:

            :keyword:`ALL`

            :keyword:`EXECUTION_SETS`

            :keyword:`ExecutionList`

            :keyword:`ATTRIBUTES`

            :keyword:`ALL_OUTPUTS`

            :keyword:`ALL_OUTPUT_LABELS`

            :keyword:`PRIMARY_OUTPUTS`

            :keyword:`PRIMARY_OUTPUT_LABELS`

            :keyword:`MONITORED_OUTPUTS`

            :keyword:`MONITORED_OUTPUT_LABELS`

            :keyword:`FLAT_OUTPUT`

            :keyword:`DICT_OUTPUT`

        """
        ALL = ()
        EXECUTION_SETS = ()
        ExecutionList = ()
        ATTRIBUTES = ()
        ALL_OUTPUTS = ()
        ALL_OUTPUT_LABELS = ()
        PRIMARY_OUTPUTS = ()
        PRIMARY_OUTPUT_LABELS = ()
        MONITORED_OUTPUTS = ()
        MONITORED_OUTPUT_LABELS = ()
        FLAT_OUTPUT = ()
        DICT_OUTPUT = ()

    def show(self, options=None):
        """Print ``execution_sets``, ``executionList``, :keyword:`ORIGIN` and :keyword:`TERMINAL` mechanisms,
        ``outputs`` and their labels for the system.

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
            :keyword:`TERMINAL` mechs

            :keyword:`NUM_PHASES_PER_TRIAL`:number of phases required to execute all mechanisms in the system

            :keyword:`MONITORING_MECHANISMS`:list of MONITORING mechanisms

            :keyword:`TARGET_MECHANISMS`:list of TARGET mechanisms

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
