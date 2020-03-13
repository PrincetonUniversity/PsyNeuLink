
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

A System is a `Composition <Composition>` that is a collection of `Processes <Process>` all of which are executed
together. Executing a System executes all of the `Mechanisms <Mechanism>` in its Processes in a structured order.
`Projections <Projection>` between Mechanisms in different Processes within the System are permitted, as are recurrent
Projections, but Projections from Mechanisms in other Systems are ignored (PsyNeuLink does not support ESP).  A System
can also be assigned a `ControlMechanism <ControlMechanism>` as its `controller <System.controller>`, that can be
used to control parameters of other `Mechanisms <Mechanism>` (or their `functions <Mechanism_Base.function>` in the
System.

.. _System_Creation:

Creating a System
-----------------

Systems are created by instantiating the `System` class.  If no arguments are provided, a System with a single `Process`
containing a single `default_mechanism <Mechanism_Base.default_mechanism>` is created.  More commonly, a System is
created from one or more `Processes <Process>` that are specified in the **processes**  argument of the `System`
class, and listed in its `processes <System.processes>` attribute.

.. note::
   At present, only `Processes <Process>` can be assigned to a System; `Mechanisms <Mechanism>` cannot be assigned
   directly to a System.  They must be assigned to the `pathway <Process_Pathway>` of a Process, and then that Process
   must be included in the **processes** argument of the `System` class.

.. _System_Control_Specification:

*Specifying Control*
~~~~~~~~~~~~~~~~~~~~

A controller can also be specified for the System, in the **controller** argument of the `System`.  This can be an
existing `ControlMechanism`, a constructor for one, or a class of ControlMechanism in which case a default
instance of that class will be created.  If an existing ControlMechanism or the constructor for one is used, then
the `OutputPorts it monitors <ControlMechanism_ObjectiveMechanism>` and the `parameters it controls
<ControlMechanism_ControlSignals>` can be specified using its `objective_mechanism
<ControlMechanism.objective_mechanism>` and `control_signals <ControlMechanism.control_signals>`
attributes, respectively.  In addition, these can be specified in the **monitor_for_control** and **control_signal**
arguments of the `System`, as described below.

* **monitor_for_control** argument -- used to specify OutputPorts of Mechanisms in the System that should be
  monitored by the `ObjectiveMechanism` associated with the System's `controller <System.controller>` (see
  `ControlMechanism_ObjectiveMechanism`);  these are used in addition to any specified for the ControlMechanism or
  its ObjectiveMechanism.  These can be specified in the **monitor_for_control** argument of the `System` using
  any of the ways used to specify the *monitored_output_ports* for an ObjectiveMechanism (see
  `ObjectiveMechanism_Monitor`).  In addition, the **monitor_for_control** argument supports two
  other forms of specification:

  * **string** -- must be the `name <OutputPort.name>` of an `OutputPort` of a `Mechanism <Mechanism>` in the System
    (see third example under `System_Control_Examples`).  This can be used anywhere a reference to an OutputPort can
    ordinarily be used (e.g., in an `InputPort tuple specification <InputPort_Tuple_Specification>`). Any OutputPort
    with a name matching the string will be monitored, including ones with the same name that belong to different
    Mechanisms within the System. If an OutputPort of a particular Mechanism is desired, and it shares its name with
    other Mechanisms in the System, then it must be referenced explicitly (see `InputPort specification
    <InputPort_Specification>`, and examples under `System_Control_Examples`).

  * **MonitoredOutputPortsOption** -- must be a value of `MonitoredOutputPortsOption`, and must appear alone or as a
    single item in the list specifying the **monitor_for_control** argument;  any other specification(s) included in
    the list will take precedence.  The MonitoredOutputPortsOption applies to all of the Mechanisms in the System
    except its `controller <System.controller>` and `LearningMechanisms <LearningMechanism>`. The
    *PRIMARY_OUTPUT_PORTS* value specifies that the `primary OutputPort <OutputPort_Primary>` of every Mechanism be
    monitored, whereas *ALL_OUTPUT_PORTS* specifies that *every* OutputPort of every Mechanism be monitored.

  The default for the **monitor_for_control** argument is *MonitoredOutputPortsOption.PRIMARY_OUTPUT_PORTS*.
  The OutputPorts specified in the **monitor_for_control** argument are added to any already specified for the
  ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>`, and the full set is listed in
  the ControlMechanism's `monitored_output_ports <EVCControlMechanism.monitored_output_ports>` attribute, and its
  ObjectiveMechanism's `monitored_output_ports <ObjectiveMechanism.monitored_output_ports>` attribute).
..
* **control_signals** argument -- used to specify the parameters of Components in the System to be controlled. These
  can be specified in any of the ways used to `specify ControlSignals <ControlMechanism_ControlSignals>` in the
  *control_signals* argument of a ControlMechanism. These are added to any `ControlSignals <ControlSignal>` that have
  already been specified for the `controller <System.controller>` (listed in its `control_signals
  <ControlMechanism.control_signals>` attribute), and any parameters that have directly been `specified for
  control <ParameterPort_Specification>` within the System (see `System_Control` below for additional details).

.. _System_Structure:

Structure
---------

The Components of a System are shown in the figure below and summarized in the sections that follow.

.. _System_Full_Fig:

.. figure:: _static/System_full_fig.svg
   :alt: Overview of major PsyNeuLink components
   :scale: 75 %

   Two `Processes <Process>` are shown, both belonging to the same System.  Each Process has a
   series of :doc:`ProcessingMechanisms <ProcessingMechanism>` linked by :doc:`MappingProjections <MappingProjection>`,
   that converge on a common final ProcessingMechanism.  Each ProcessingMechanism is labeled with its designation in
   the System.  The `TERMINAL` Mechanism for both Processes projects to an `ObjectiveMechanism` that is used to
   drive `learning <LearningProjection>` in Process B. It also projects to a separate ObjectiveMechanism that is used
   for control of ProcessingMechanisms in both Processes A and B.  Note that the Mechanisms and
   Projections responsible for learning and control belong to the System and can monitor and/or control Mechanisms
   belonging to more than one Process (as shown for control in this figure).


.. _System_Mechanisms:

*Mechanisms*
~~~~~~~~~~~~

The `Mechanisms <Mechanism>` in a System are assigned designations based on the position they occupy in the `graph
<System.graph>` and/or the role they play in a System:

    `ORIGIN`: receives input to the System (provided in the `execute <System.execute>` or `run
    <System.run> method), and does not receive a `Projection <Projection>` from any other `ProcessingMechanisms
    <ProcessingMechanism>`.

    `TERMINAL`: provides output from the System, and does not send Projections to any other ProcessingMechanisms.

    `SINGLETON`: both an `ORIGIN` and a `TERMINAL` Mechanism.

    `INITIALIZE_CYCLE`: sends a Projection that closes a recurrent loop; can be assigned an initial value.

    `CYCLE`: receives a Projection that closes a recurrent loop.

    `CONTROL`: monitors the value of another Mechanism for use in controlling parameter values.

    `LEARNING`: monitors the value of another Mechanism for use in learning.

    `TARGET`: ComparatorMechanism that monitors a `TERMINAL` Mechanism of a Process and compares it to a corresponding
    value provided in the `execute <System.execute>` or `run <System.run>` method.

    `INTERNAL`: ProcessingMechanism that does not fall into any of the categories above.

    .. note::
       Any `ORIGIN` and `TERMINAL` Mechanisms of a System must be, respectively, the `ORIGIN` or `TERMINAL` of any
       Process(es) to which they belong.  However, it is not necessarily the case that the `ORIGIN` and/or `TERMINAL`
       Mechanism of a Process is also the `ORIGIN` and/or `TERMINAL` of a System to which the Process belongs (see
       `example <LearningProjection_Output_vs_Terminal_Figure>`).

    .. note:: designations are stored in the `systems <Mechanism.systems>` attribute of a `Mechanism <Mechanism>`.
    COMMENT:
    (see _instantiate_graph below)
    COMMENT

.. _System_Graph:

*Graph*
~~~~~~~

When a System is created, a graph is constructed that describes the `Projections <Projection>` (edges) among its
`Mechanisms <Mechanism>` (nodes). The graph is assigned to the System's `graph <System.graph>` attribute.  A
System's `graph <System.graph>` can be displayed using its `System.show_graph` method.  The `graph
<System.graph>` is stored as a dictionary of dependencies that can be passed to graph theoretical tools for
analysis.  A System can have recurrent Processing pathways, such as feedback loops;  that is, the System's `graph
<System.graph> can be *cyclic*.  PsyNeuLink also uses the `graph <System.graph>` to determine the order in
which its Mechanisms are executed.  To do so in an orderly manner, however, the graph must be *acyclic*.  To address
this, PsyNeuLink constructs an `execution_graph <System.execution_graph>` from the System's `graph
<System.graph>`. If the  System is acyclic, these are the same. If the System is cyclic, then the `execution_graph
<System.execution_graph>` is a subset of the `graph <System.graph>` in which the dependencies (edges)
associated with Projections that close a loop have been removed. Note that this only impacts the order of execution;
the Projections themselves remain in effect, and will be fully functional during the execution of the Mechanisms
to and from which they project (see `System_Execution` below for a more detailed description).

COMMENT:
    ADD FIGURE OF GRAPH FOR SYSTEM SHOWN IN FIGURE ABOVE
COMMENT

.. _System_Scheduler:

*Scheduler*
~~~~~~~~~~~

Every System has two `Schedulers <Scheduler>`, one that handles the ordering of execution of its Components for
`processing <System_Execution_Processing>` (assigned to its `scheduler` attribute), and one that
does the same for `learning <System_Execution_Learning>` (assigned to its `scheduler_learning` attribute).
The `scheduler` can be assigned in the **scheduler** argument of the System's constructor;  if it is not
specified, a default `Scheduler` is created automatically.   The `scheduler_learning` is always assigned automatically.
The System's Schedulers base the ordering of execution of its Components based on the order in which they are listed
in the `pathway <Process.pathway>`\\s of the `Processes <Process>` used to construct the System, constrained by any
`Conditions <Condition>` that have been created for individual Components and assigned to the System's Schedulers (see
`Scheduler`, `Condition <Condition_Creation>`, `System_Execution_Processing`, and `System_Execution_Learning` for
additional details).  Both schedulers maintain a `Clock` that can be used to access their current `time
<Time_Overview>`.

.. _System_Control:

*Control*
~~~~~~~~~

A System can be assigned a `ControlMechanism` as its `controller <System.controller>`, that can be  used to
control parameters of other `Mechanisms <Mechanism>` in the System. Although any number of ControlMechanism can be
assigned to and executed within a System, a System can have only one `controller <System.controller>`, that is
executed after all of the other Components in the System have been executed, including any other ControlMechanisms (see
`System Execution <System_Execution>`). When a ControlMechanism is assigned to or created by a System, it inherits
specifications made for the System as follows:

  * the OutputPorts specified to be monitored in the System's **monitor_for_control** argument are added to those
    that may have already been specified for the ControlMechanism's `objective_mechanism
    <ControlMechanism.objective_mechanism>` (the full set is listed in the ControlMechanism's `monitored_output_ports
    <EVCControlMechanism.monitored_output_ports>` attribute, and its ObjectiveMechanism's `monitored_output_ports
    <ObjectiveMechanism.monitored_output_ports>` attribute); see `System_Control_Specification` for additional details of how
    to specify OutputPorts to be monitored.

  * a `ControlSignal` and `ControlProjection` is assigned to the ControlMechanism for every parameter that has been
    `specified for control <ParameterPort_Specification>` in the System;  these are added to any that the
    ControlMechanism may already have (listed in its `control_signals <ControlMechanism.control_signals>` attribute).

See `System_Control_Specification` above, `ControlMechanism <ControlMechanism>` and `ModulatorySignal_Modulation`
for details of how control operates, and `System_Execution_Control` below for a description of how it is engaged
when a System is executed. The control Components of a System can be displayed using the System's `show_graph
<System.show_graph>` method with its **show_control** argument assigned as `True`.

.. _System_Learning:

*Learning*
~~~~~~~~~~

A System cannot itself be specified for learning.  However, if learning has been specified for any of its `processes
<System.processes>`, then it will be `implemented <LearningMechanism_Learning_Configurations>` and `executed
<System_Execution_Learning>` as part of the System.  Note, however, that for the learning Components of a Process to
be implemented by a System, learning must be `specified for the entire Process <Process_Learning_Specification>`. The
learning Components of a System can be displayed using the System's `System.show_graph` method with its
**show_learning** argument assigned as `True` or *ALL*.


.. _System_Execution:

Execution
---------

A System can be executed by calling either its `execute <System.execute>` or `run <System.execute>` methods.
`execute <System.execute>` executes the System once; that is, it executes a single `TRIAL`.
`run <System.run>` allows a series of `TRIAL`\\s to be executed, one for each input in the **inputs** argument
of the call to `run <System.run>`.  For each `TRIAL`, it makes a series of calls to the `run <Scheduler.run>`
method of the relevant `Scheduler` (see `System_Execution_Processing` and `System_Execution_Learning` below), and
executes the Components returned by that Scheduler (constituting a `TIME_STEP` of execution), until every Component in
the System has been executed at least once, or another `termination condition <Scheduler_Termination_Conditions>` is
met.  The execution of each `TRIAL` occurs in four phases: `initialization <System_Execution_Input_And_Initialization>`,
`processing <System_Execution_Processing>`, `learning <System_Execution_Learning>`, and
`control <System_Execution_Control>`, each of which is described below.


.. _System_Execution_Input_And_Initialization:

*Input and Initialization*
~~~~~~~~~~~~~~~~~~~~~~~~~~

The input to a System is specified in the **input** argument of either its `execute <System.execute>` or
`run <System.run>` method. In both cases, the input for a single `TRIAL` must be a list or ndarray of values,
each of which is an appropriate input for the corresponding `ORIGIN` Mechanism (listed in the System's
`origin_mechanisms <System.origin_mechanisms>` attribute). If the `execute <System.execute>` method is used,
input for only a single `TRIAL` is provided, and only a single `TRIAL` is executed.  The `run <System.run>` method
can be used for a sequence of `TRIAL`\\s, by providing it with a list or ndarray of inputs, one for each `TRIAL`.  In
both cases, two other types of input can be provided in corresponding arguments of the `run <System.run>` method:
a list or ndarray of **initial_values**, and a list or ndarray of **target** values. The **initial_values** are
assigned at the start of a `TRIAL` as input to Mechanisms that close recurrent loops (designated as `INITIALIZE_CYCLE`,
and listed in the System's `recurrent_init_mechanisms <System.recurrent_init_mechanisms>` attribute), and
**target** values are assigned as the *TARGET* input of the System's `TARGET` Mechanisms (see
`System_Execution_Learning` below;  also, see `Run` for additional details of formatting input specifications).


.. _System_Execution_Processing:

*Processing*
~~~~~~~~~~~~

Once the relevant inputs have been assigned, the `ProcessingMechanisms <ProcessingMechanism>` of the System are executed
in the order they are listed in the `Processes <Process>` used to construct the System.  When a Mechanism is executed,
it receives input from any other Mechanisms that project to it within the System,  but not from any Mechanisms outside
the System (PsyNeuLink does not support ESP).  The order of execution is determined by the System's `execution_graph`
attribute, which is a subset of the System's `graph <System.graph>` that has been "pruned" to be acyclic (i.e.,
devoid of recurrent loops (see `System_Graph` above).  While the `execution_graph` is acyclic, all recurrent Projections
in the System remain intact during execution and can be `initialized <System_Execution_Input_And_Initialization>` at
the start of execution. The order in which Components are executed can also be customized, using the System's
`System_Scheduler` in combination with `Condition` specifications for individual Components, to execute different
Components at different time scales, or to introduce dependencies among them (e.g., require that a recurrent Mechanism
settle before another one execute -- see `example <Condition_Recurrent_Example>`).


.. _System_Execution_Learning:

*Learning*
~~~~~~~~~~

A System executes learning if it is specified for one or more `Processes <Process_Learning_Sequence>` in the System.
The System's `learning <System.learning>` attribute indicates whether learning is enabled for the System. Learning
is executed for any Components (individual Projections or Processes) for which it is `specified
<Process_Learning_Sequence>` after the  `processing <System_Execution_Processing>` of each `TRIAL` has completed, but
before the `controller <System.controller> is executed <System_Execution_Control>`.

The learning Components of a System can be displayed using the System's `show_graph <System.show_graph>` method with its
**show_learning** argument assigned `True` or *ALL*. The target values used for learning can be specified in either of
two formats: dictionary or function, which are described in the `Run` module (see `Run_Targets`). Both formats require
that a target value be provided for each `TARGET` Mechanism of the System (listed in its `target_nodes
<System.target_nodes>` attribute).

.. note::
   A `TARGET` Mechanism of a Process is not necessarily one of the `TARGET` Mechanisms of the System to which it belongs
   (see `TARGET Mechanisms <LearningMechanism_Targets>`).  Also, the changes to a System induced by learning are not
   applied until the Mechanisms that receive the Projections being learned are next executed; see :ref:`Lazy Evaluation
   <LINK>` for an explanation of "lazy" updating).


.. _System_Execution_Control:

*Control*
~~~~~~~~~

The System's `controller <System.controller>` is executed in the last phase of execution in a `TRIAL`, after all
other Mechanisms in the System have executed.  Although a System may have more than one `ControlMechanism`, only one
can be assigned as its `controller <System.controller>`;  all other ControlMechanisms are executed during the
`processing `System_Execution_Processing` phase of the `TRIAL` like any other Mechanism.  The `controller
<System.controller>` uses its `objective_mechanism <ControlMechanism.objective_mechanism>` to monitor and evaluate
the `OutputPort(s) <OutputPort>` of Mechanisms in the System; based on the information it receives from that
`ObjectiveMechanism`, it modulates the value of the parameters of Components in the System that have been `specified
for control <ControlMechanism_ControlSignals>`, which then take effect in the next `TRIAL` (see `System_Control` for
additional information about control). The control Components of a System can be displayed using the System's
`show_graph`method with its **show_control** argument assigned `True`.


.. _System_Examples:

Examples
--------
COMMENT
   XXX ADD EXAMPLES HERE FROM 'System Graph and Input Test Script'
   .. note::  All of the example Systems below use the following set of Mechanisms.  However, in practice, they must be
      created separately for each System;  using the same Mechanisms and Processes in multiple Systems can produce
      confusing results.

   Module Contents
   System: class definition
COMMENT

.. _System_Control_Examples:

*Specifying Control for a System*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example specifies an `EVCControlMechanism` as the controller for a System with two `Processes <Process>`
that include two `Mechanisms <Mechanism>` (not shown)::

    my_system = System(processes=[TaskExecutionProcess, RewardProcess],
                       controller=EVCControlMechanism(objective_mechanism=
                                                   ObjectiveMechanism(
                                                       monitored_output_ports=[
                                                           Reward,
                                                           Decision.output_ports[PROBABILITY_UPPER_THRESHOLD],
                                                           (Decision.output_ports[RESPONSE_TIME], -1, 1)]))
                                                       function=LinearCombination(operation=PRODUCT))

A constructor is used to specify the EVCControlMechanism that includes a constructor specifying its `objective_mechanism
<ControlMechanism.objective_mechanism>`;  the **monitored_output_ports** argument of the ObjectiveMechanism's constructor
is used to specify that it should monitor the `primary OutputPort <OutputPort_Primary>` of the Reward Mechanism
and the *PROBABILITY_UPPER_THRESHOLD* and *RESPONSE_TIME* and, specifying how it should combine them (see the `example
<ControlMechanism_Examples>` under ControlMechanism for an explanation). Note that the **function** argument for the
ObjectiveMechanism's constructor is also specified;  this is because an ObjectiveMechanism uses *SUM* as the default
for the `operation <LinearCombination.operation>` of its `LinearCombination` function, whereas as the EVCControlMechanism
requires *PRODUCT* -- in this case, to properly use the weight and exponents specified for the RESPONSE_TIME
OutputPort of Decision (see `note <EVCControlMechanism_Objective_Mechanism_Function_Note>` in EVCControlMechanism for
a more complete explanation).  Note that both the EVCControlMechanism and/or the ObjectiveMechanism could have been
constructed separately, and then referenced in the **controller** argument of ``my_system`` and **objective_mechanism**
argument of the EVCControlMechanism, respectively.

The same configuration can be specified in a more concise, though less "transparent" form, as follows::

    my_system = System(processes=[TaskExecutionProcess, RewardProcess],
                       controller=EVCControlMechanism(objective_mechanism=[
                                                             Reward,
                                                             Decision.output_ports[PROBABILITY_UPPER_THRESHOLD],
                                                             (Decision.output_ports[RESPONSE_TIME], -1, 1)])))

Here, the constructor for the ObjectiveMechanism is elided, and the **objective_mechanism** argument for the
EVCControlMechanism is specified as a list of OutputPorts (see `ControlMechanism_ObjectiveMechanism`).

The specification can be made even simpler, but with some additional considerations that must be kept in mind,
as follows::

    my_system = System(processes=[TaskExecutionProcess, RewardProcess],
                       controller=EVCControlMechanism,
                       monitor_for_control=[Reward,
                                            PROBABILITY_UPPER_THRESHOLD,
                                            RESPONSE_TIME, 1, -1)],

Here, the *controller** for ``my_system`` is specified as the EVCControlMechanism, which will created a default
EVCControlMechanism. The OutputPorts to be monitored are specified in the **monitor_for_control** argument for
``my_system``.  Note that here they can be referenced simply by name; when ``my_system`` is created, it will search
all of its Mechanisms for OutputPorts with those names, and assign them to the `monitored_output_ports
<ObjectiveMechanism>` attribute of the EVCControlMechanism's `objective_mechanism
<EVCControlMechanism.objective_mechanism>` (see `System_Control_Specification` for a more detailed explanation of how
OutputPorts are assigned to be monitored by a System's `controller <System.controller>`).  While this form of the
specification is much simpler, it less flexible (i.e., it can't be used to customize the ObjectiveMechanism used by
the EVCControlMechanism or its `function <ObjectiveMechanism.function>`.

.. _System_Class_Reference:

Class Reference
---------------

"""

import inspect
import itertools
import logging
import math
import numbers
import re
import warnings

from PIL import Image
from collections.abc import Iterable
from collections import OrderedDict
from os import path, remove

import numpy as np
import typecheck as tc

from toposort import toposort, toposort_flatten

from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism, MonitoredOutputPortTuple
from psyneulink.core.components.mechanisms.modulatory.learning.learningauxiliary import _assign_error_signal_projections, _get_learning_mechanisms
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism, LearningTiming
from psyneulink.core.components.mechanisms.mechanism import MechanismList
from psyneulink.core.components.mechanisms.processing.objectivemechanism import DEFAULT_MONITORED_PORT_EXPONENT, DEFAULT_MONITORED_PORT_MATRIX, DEFAULT_MONITORED_PORT_WEIGHT, ObjectiveMechanism
from psyneulink.core.components.process import Process, ProcessInputPort, ProcessList, ProcessTuple
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import Projection
from psyneulink.core.components.shellclasses import Mechanism, Process_Base, System_Base
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    ALL, BOLD, COMPONENT, CONDITION, CONTROL, CONTROLLER, CYCLE, FUNCTION, FUNCTIONS, \
    INITIALIZE_CYCLE, INITIAL_VALUES, INTERNAL, LABELS, LEARNING, MATRIX, MONITOR_FOR_CONTROL, \
    OBJECTIVE_MECHANISM, ORIGIN, OUTCOME, PROJECTIONS, ROLES, SAMPLE, SINGLETON, TARGET, TERMINAL, VALUES, \
    SYSTEM_COMPONENT_CATEGORY
from psyneulink.core.globals.log import Log
from psyneulink.core.globals.parameters import Defaults, Parameter
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.preferences.systempreferenceset import SystemPreferenceSet, is_sys_pref_set
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import AutoNumber, ContentAddressableList, append_type_to_name, call_with_pruned_args, convert_to_np_array, iscompatible
from psyneulink.core.scheduling.condition import AtPass, AtTimeStep, Never
from psyneulink.core.scheduling.scheduler import Always, Condition, Scheduler
from psyneulink.library.components.mechanisms.modulatory.learning.autoassociativelearningmechanism import AutoAssociativeLearningMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

__all__ = [
    'CONTROL_MECHANISM', 'CONTROL_PROJECTION_RECEIVERS', 'defaultInstanceCount', 'DURATION',
    'EXECUTION_SET', 'INITIAL_FRAME', 'INPUT_ARRAY',
    'kwSystemInputPort',
    'LEARNING_MECHANISMS', 'LEARNING_PROJECTION_RECEIVERS', 'MECHANISMS', 'MonitoredOutputPortTuple', 'MOVIE_NAME',
    'NUM_PHASES_PER_TRIAL', 'NUM_TRIALS', 'ORIGIN_MECHANISMS', 'OUTPUT_PORT_NAMES', 'OUTPUT_VALUE_ARRAY',
    'PROCESSES', 'RECURRENT_INIT_ARRAY', 'RECURRENT_MECHANISMS',
    'SAVE_IMAGES', 'SCHEDULER',
    'System', 'sys', 'SYSTEM_TARGET_INPUT_PORT', 'SystemError', 'SystemInputPort', 'SystemRegistry',
    'SystemWarning', 'TARGET_MECHANISMS', 'TERMINAL_MECHANISMS', 'UNIT', 'osf'
]

logger = logging.getLogger(__name__)

# ProcessRegistry ------------------------------------------------------------------------------------------------------

defaultInstanceCount = 0 # Number of default instances (used to index name)

# inspect() keywords
SCHEDULER = 'scheduler'
PROCESSES = 'processes'
MECHANISMS = 'mechanisms'
INPUT_ARRAY = 'input_array'
RECURRENT_MECHANISMS = 'recurrent_mechanisms'
RECURRENT_INIT_ARRAY = 'recurrent_init_array'
TERMINAL_MECHANISMS = 'terminal_mechanisms'
OUTPUT_PORT_NAMES = 'output_port_names'
OUTPUT_VALUE_ARRAY = 'output_value_array'
NUM_PHASES_PER_TRIAL = 'num_phases'
TARGET_MECHANISMS = 'target_nodes'
LEARNING_PROJECTION_RECEIVERS = 'learning_projection_receivers'
LEARNING_MECHANISMS = 'learning_mechanisms'
CONTROL_MECHANISM = 'control_mechanism'
CONTROL_PROJECTION_RECEIVERS = 'control_projection_receivers'
ORIGIN_MECHANISMS = 'ORIGIN_MECHANISMS'

SystemRegistry = {}

kwSystemInputPort = 'SystemInputPort'

class MonitoredOutputPortsOption(AutoNumber):
    """Specifies OutputPorts to be monitored by a `ControlMechanism <ControlMechanism>`
    (see `ObjectiveMechanism_Monitor` for a more complete description of their meanings.
    """
    ONLY_SPECIFIED_OUTPUT_PORTS = ()
    """Only monitor explicitly specified OutputPorts."""
    PRIMARY_OUTPUT_PORTS = ()
    """Monitor only the `primary OutputPort <OutputPort_Primary>` of a Mechanism."""
    ALL_OUTPUT_PORTS = ()
    """Monitor all OutputPorts <Mechanism_Base.output_ports>` of a Mechanism."""
    NUM_MONITOR_STATES_OPTIONS = ()

# Indices for items in tuple format used for specifying monitored_output_ports using weights and exponents
OUTPUT_PORT_INDEX = 0
WEIGHT_INDEX = 1
EXPONENT_INDEX = 2
MATRIX_INDEX = 3

# show_graph options
SHOW_CONTROL = 'show_control'
SHOW_LEARNING = 'show_learning'

# Animation Keywords
NUM_TRIALS = 'num_trials'
UNIT = 'unit'
DURATION = 'duration'
MOVIE_NAME = 'movie_name'
SAVE_IMAGES = 'save_images'
SHOW = 'show'
INITIAL_FRAME = 'INITIAL_FRAME'

EXECUTION_SET = 'EXECUTION_SET'

class SystemWarning(Warning):
     def __init__(self, error_value):
         self.error_value = error_value


class SystemError(Exception):
     def __init__(self, error_value):
         self.error_value = error_value

     def __str__(self):
         return repr(self.error_value)


def sys(*args, **kwargs):
    """Factory method

    **args** can be `Mechanisms <Mechanism>`, `Projections <Projection>` and/or lists containing either, but must
    conform to the format for the specification of the `pathway <Process.pathway>` argument of a `Process`.  If none
    of the args is a list, then all are treated as a single Process (i.e., pathway specification). If any args are
    lists, each is treated as a pathway specification for a Process; any other args not in a list **must be Mechanisms**
    (i.e., none can be Projections), and each is used to create a singleton Process.

    **kwargs** can be any arguments of the `System` constructor.
    """

    processes = []
    if not any(isinstance(arg, (list, Process)) for arg in args):
        processes = Process(pathway=list(args))
    else:
        for arg in args:
            if isinstance(arg, Process):
                proc = arg
            else:
                if not isinstance(arg, list):
                    arg = [arg]
                proc = Process(pathway=arg)
            processes.append(proc)

    return System(processes=processes, **kwargs)

def osf(f):
    print(4)

# FIX:  IMPLEMENT DEFAULT PROCESS
# FIX:  NEED TO CREATE THE PROJECTIONS FROM THE PROCESS TO THE FIRST MECHANISM IN PROCESS FIRST SINCE,
# FIX:  ONCE IT IS IN THE GRAPH, IT IS NOT LONGER EASY TO DETERMINE WHICH IS WHICH IS WHICH (SINCE SETS ARE NOT ORDERED)

class System(System_Base):
    """

    System(                                         \
        default_variable=None,                      \
        size=None,                                  \
        processes=None,                             \
        initial_values=None,                        \
        controller=None,                            \
        enable_controller=:keyword:`False`,         \
        monitor_for_control=None,                   \
        control_signals=None,                       \
        learning_rate=None,                         \
        targets=None,                               \
        reinitialize_mechanisms_when=AtTimeStep(0), \
        scheduler=None,                             \
        params=None,                                \
        name=None,                                  \
        prefs=None)

    Base class for System.

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
        + class_defaults.variable = inputValueSystemDefault                     # Used as default input value to Process)

       Class methods
       -------------
        - _validate_variable(variable, context):  insures that variable is 3D np.array (one 2D for each Process)
        - _instantiate_attributes_before_function(context):  calls self._instantiate_graph
        - _instantiate_function(context): validates only if self.prefs.paramValidationPref is set
        - _instantiate_graph(input, context):  instantiates Processes in self.process and constructs execution_list
        - _instantiate_controller(): instantiates ControlMechanism in **controller** argument or assigned to attribute
        - identify_origin_and_terminal_mechanisms():  assign self.origin_mechanisms and self.terminalMechanisms
        - _assign_output_ports():  assign OutputPorts of System (currently = terminalMechanisms)
        - execute(input, context):  executes Mechanisms in order specified by execution_list
        - defaults.variable(value):  setter for defaults.variable;  does some kind of error checking??

       SystemRegistry
       --------------
        Register in SystemRegistry, which maintains a dict for the subclass, a count for all instances of it,
         and a dictionary of those instances

        TBI: MAKE THESE convenience lists, akin to self.terminalMechanisms
        + input (list): contains Process.input for each Process in self.processes
        + output (list): containts Process.ouput for each Process in self.processes
        [TBI: + input (list): each item is the Process.input object for the corresponding Process in self.processes]
        [TBI: + outputs (list): each item is the Process.output object for the corresponding Process in self.processes]
    COMMENT

    Attributes
    ----------

    componentType : SYSTEM

    processes : list of Process objects
        list of `Processes <Process>` in the System specified by the **processes** argument of the constructor.

        .. can be appended with prediction Processes by EVCControlMechanism
           used with self.input to constsruct self.process_tuples

        .. _processList : ProcessList
            Provides access to (process, input) tuples.
            Derived from self.input and self.processes.
            Used to construct :py:data:`execution_graph <System.execution_graph>` and execute the System

    controller : ControlMechanism : default SystemDefaultControlMechanism
        the `ControlMechanism <ControlMechanism>` used to monitor the `value <OutputPort.value>` of the `OutputPort(s)
        <OutputPort>` and/or `Mechanisms <Mechanism>` specified in the **monitor_for_control** argument,
        and that controls the parameters specified in the **control_signals** argument of the System's constructor.

    enable_controller :  bool : default :keyword:`False`
        determines whether the `controller <System.controller>` is executed during System execution.

    learning : bool : default False
        indicates whether learning is enabled for the System;  is set to `True` if learning is specified for any
        `Processes <Process>` in the System.

    learning_rate : float : default None
        determines the learning_rate for all `LearningMechanisms <LearningMechanism>` in the System.  This overrides any
        values set for the function of individual LearningMechanisms or `LearningSignals <LearningSignal>`, and persists
        for all subsequent executions of the System.  If it is set to `None`, then the `learning_rate
        <System.learning_rate>` is determined by last value assigned to each LearningMechanism (either directly,
        or following the execution of any `Process` or System to which the LearningMechanism belongs and for which a
        `learning_rate <LearningMechanism.learning_rate>` was set).

    targets : 2d nparray
        used as template for the values of the System's `target_input_ports`, and to represent the targets specified in
        the **targets** argument of System's `execute <System.execute>` and `run <System.run>` methods.

    graph : OrderedDict
        contains a graph of all of the Components in the System. Each entry specifies a set of <Receiver>: {sender,
        sender...} dependencies.  The key of each entry is a receiver Component, and the value is a set of Mechanisms
        that send Projections to that receiver. If a key (receiver) has no dependents, its value is an empty set.

    execution_graph : OrderedDict
        contains an acyclic subset of the System's `graph <System.graph>`, hierarchically organized by a
        `toposort <https://en.wikipedia.org/wiki/Topological_sorting>`_. Used to specify the order in which
        Components are `executed <System_Execution>`.

    execution_sets : list of sets
        contains a list of Component sets. Each set contains Components to be executed at the same time.
        The sets are ordered in the sequence with which they should be executed.

    execution_list : list of Mechanisms and/or Projections
        contains a list of Components in the order in which they are `executed <System_Execution>`.
        The list is a random sample of the permissible orders constrained by the `execution_graph` and produced by the
        `toposort <https://en.wikipedia.org/wiki/Topological_sorting>`_.

    mechanisms : list of Mechanism objects
        contains a list of all `Mechanisms <Mechanism>` in the System.

        .. property that points to _all_mechanisms.mechanisms (see below)

    mechanismsDict : Dict[Mechanism: Process]
        contains a dictionary of all Mechanisms in the System, listing the Processes to which they belong. The key of
        each entry is a `Mechanism <Mechanism>` object, and the value of each entry is a list of `Processes <Process>`.

        .. Note: the following attributes use lists of tuples (Mechanism, runtime_param, phaseSpec) and MechanismList
              xxx_mechs are lists of tuples defined in the Process pathways;
                  tuples are used because runtime_params and phaseSpec are attributes that need
                  to be able to be specified differently for the same Mechanism in different contexts
                  and thus are not easily managed as Mechanism attributes
              xxxMechanismLists point to MechanismList objects that provide access to information
                  about the Mechanism <type> listed in mechs (i.e., the Mechanisms, names, etc.)

        .. _all_mechs : list of (Mechanism, runtime_param, phaseSpec) tuples
            Tuples for all Mechanisms in the System (serve as keys in self.graph).

        .. _all_mechanisms : MechanismList
            Contains all Mechanisms in the System (based on _all_mechs).

        .. _origin_mechs : list of (Mechanism, runtime_param, phaseSpec) tuples
            Tuples for all ORIGIN Mechanisms in the System.

        .. _terminal_mechs : list of (Mechanism, runtime_param, phaseSpec) tuples
            Tuples for all TERMINAL Mechanisms in the System.

        .. _learning_mechs : list of (Mechanism, runtime_param, phaseSpec) tuples
            Tuples for all LearningMechanisms in the System.

        .. _target_mechs : list of (Mechanism, runtime_param, phaseSpec) tuples
            Tuples for all TARGET `ObjectiveMechanisms <ObjectiveMechanism>`  in the System that are a `TERMINAL`
            for at least one Process to which it belongs and that Process has learning enabled --  the criteria for
            being a target used in learning.

        .. _learning_mechs : list of (Mechanism, runtime_param, phaseSpec) tuples
            Tuples for all LearningMechanisms in the System (used for learning).

        .. _control_mechs : list of a single (Mechanism, runtime_param, phaseSpec) tuple
            Tuple for the controller in the System.

    origin_mechanisms : MechanismList
        all `ORIGIN` Mechanisms in the System (i.e., that don't receive `Projections <Projection>` from any other
        `Mechanisms <Mechanism>`, listed in ``origin_mechanisms.data``.

        .. based on _origin_mechs
           System.input contains the input to each `ORIGIN` Mechanism

    terminalMechanisms : MechanismList
        all `TERMINAL` Mechanisms in the System (i.e., that don't project to any other `ProcessingMechanisms
        <ProcessingMechanism>`), listed in ``terminalMechanisms.data``.

        .. based on _terminal_mechs
           System.ouput contains the output of each TERMINAL Mechanism

    recurrent_init_mechanisms : MechanismList
        `Mechanisms <Mechanism>` with recurrent `Projections <Projection>` that are candidates for `initialization
        <System_Execution_Input_And_Initialization>`, listed in ``recurrent_init_mechanisms.data``.

    learning_mechanisms : MechanismList
        all `LearningMechanisms <LearningMechanism>` in the System, listed in ``learning_mechanisms.data``.

    target_mechanisms : MechanismList
        all `TARGET` Mechanisms in the System (used for `learning <System_Execution_Learning>`), listed in
        ``target_nodes.data``.
        COMMENT:
            based on _target_mechs)
        COMMENT

    target_input_ports : List[SystemInputPort]
        one item for each `TARGET` Mechanism in the System (listed in its `target_nodes
        <System.target_mechansims>` attribute).  Used to represent the values specified in the **targets**
        argument of the System's `execute <System.execute>` and `run <System.run>` methods, and to provide
        thoese values to the TARGET `InputPort` of each `TARGET` Mechanism during `execution
        <System_Execution_Learning>`.


        .. control_mechanism : MechanismList
            contains the `ControlMechanism <ControlMechanism>` that is the `controller <System.controller>` of the
            System.
            COMMENT:
                ??and any other `ControlMechanisms <ControlMechanism>` in the System
                (based on _control_mechs).
            COMMENT

    value : 3D ndarray
        contains an array of 2D arrays, each of which is the `output_values <Mechanism_Base.output_values>` of a
        `TERMINAL` Mechanism in the System.

        .. _phaseSpecMax : int
            Maximum phase specified for any Mechanism in System.  Determines the phase of the last (set of)
            ProcessingMechanism(s) to be executed in the System.

        .. numPhases : int
            number of phases for System (read-only).

            .. implemented as an @property attribute; = _phaseSpecMax + 1

    initial_values : list or ndarray of values
        values used to initialize Mechanisms that close recurrent loops (designated as `INITIALIZE_CYCLE`).
        Length must equal the number of `INITIALIZE_CYCLE` Mechanisms listed in the System's
        `recurrent_init_mechanisms <System.recurrent_init_mechanisms>` attribute.

    results : List[OutputPort.value]
        list of return values from the sequence of executions.  Each item is a 2d array containing the `output_values
        <Mechanism.output_values>` of each `TERMINAL` Mechanism of the System for a given execution.
        Excludes simulated runs.

    simulation_results : List[OutputPort.value]
        list of return values from the sequence of executions in simulation run(s) of the System; requires
        recordSimulationPref to be `True`.  Each item is a 2d array containing the `output_values
        <Mechanism.output_values>` of each `TERMINAL` Mechanism in the System for a given execution in the simulation.
        Excludes values from non-simulation runs.

    name : str
        the name of the System; if it is not specified in the **name** argument of the constructor, a default is
        assigned by SystemRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the System; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """

    componentCategory = SYSTEM_COMPONENT_CATEGORY
    className = componentCategory
    suffix = " " + className
    componentType = "System"

    registry = SystemRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # These will override those specified in CATEGORY_DEFAULT_PREFERENCES
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'SystemCustomClassPreferences',
    #     REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}
    # classPreferences = {
    #     kwReportSimulationPref: 'SystemCustomClassPreferences',
    #     REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # Use inputValueSystemDefault as default input to process
    class Parameters(System_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <System.variable>`

                    :default value: None
                    :type:

                enable_controller
                    see `enable_controller <System.enable_controller>`

                    :default value: None
                    :type:

                initial_values
                    see `initial_values <System.initial_values>`

                    :default value: None
                    :type:

                learning_rate
                    see `learning_rate <System.learning_rate>`

                    :default value: None
                    :type:

                monitor_for_control
                    see `monitor_for_control <System.monitor_for_control>`

                    :default value: None
                    :type:

                processes
                    see `processes <System.processes>`

                    :default value: None
                    :type:

                targets
                    see `targets <System.targets>`

                    :default value: None
                    :type:
        """
        variable = Parameter(None, pnl_internal=True, constructor_argument='default_variable')

        processes = None
        initial_values = None
        enable_controller = None
        monitor_for_control = None
        learning_rate = None
        targets = None


    # FIX 5/23/17: ADD control_signals ARGUMENT HERE (AND DOCUMENT IT ABOVE)
    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 processes=None,
                 initial_values=None,
                 controller=None,
                 enable_controller=False,
                 monitor_for_control=None,
                 control_signals=None,
                 # learning=None,
                 learning_rate=None,
                 targets=None,
                 reinitialize_mechanisms_when=AtTimeStep(0),
                 scheduler=None,
                 params=None,
                 name=None,
                 prefs:is_sys_pref_set=None,
                 context=None):

        # Required to defer assignment of self.controller by setter
        #     until the rest of the System has been instantiated
        self.initialization_status = ContextFlags.INITIALIZING

        processes = processes or []
        if not isinstance(processes, list):
            processes = [processes]
        monitor_for_control = monitor_for_control or [MonitoredOutputPortsOption.PRIMARY_OUTPUT_PORTS]
        self.control_signals_arg = control_signals or []
        if not isinstance(self.control_signals_arg, list):
            self.control_signals_arg = [self.control_signals_arg]
        if not isinstance(monitor_for_control, list):
            monitor_for_control = [monitor_for_control]

        # If controller has already been instantiated, flag its ObjectiveMechanism as belonging to a controller
        #    so that it is recognized as such the System in _instantiate_system_graph()
        #    (can't actually assign ControlMechanism as controller here, as _instantiate_controller needs parsed graph)
        if isinstance(controller, ControlMechanism) and controller.objective_mechanism is not None:
            controller.objective_mechanism.for_controller = True
            for proj in controller.objective_mechanism.path_afferents:
                proj._activate_for_compositions(self)

        self.projections = []

        self.inputs = []
        self.stimulusInputPorts = []
        self.target_input_ports = []

        self.scheduler = scheduler
        self.scheduler_learning = None
        self.termination_learning = None

        self._phaseSpecMax = 0
        self.learning = False

        register_category(entry=self,
                          base_class=System,
                          name=name,
                          registry=SystemRegistry,
                          context=context)

        prefs = SystemPreferenceSet(owner=self, prefs=prefs, context=context)

        if not context:
            context = Context(source=ContextFlags.COMPOSITION)
            self.initialization_status = ContextFlags.INITIALIZING

        self.default_execution_id = self.name

        super().__init__(
            default_variable=default_variable,
            size=size,
            processes=processes,
            initial_values=initial_values,
            enable_controller=enable_controller,
            monitor_for_control=monitor_for_control,
            learning_rate=learning_rate,
            targets=targets,
            param_defaults=params,
            name=self.name,
            prefs=prefs,
        )

        self.reinitialize_mechanisms_when = reinitialize_mechanisms_when

        # Assign controller
        self._instantiate_controller(control_mech_spec=controller, context=context)

        # Assign processing scheduler (learning_scheduler is assigned in _instantiate_learning_graph)
        if self.scheduler is None:
            self.scheduler = Scheduler(system=self, default_execution_id=self.default_execution_id)

        # IMPLEMENT CORRECT REPORTING HERE
        # if self.prefs.reportOutputPref:
        #     print("\n{0} initialized with:\n- pathway: [{1}]".
        #           # format(self.name, self.pathwayMechanismNames.__str__().strip("[]")))
        #           format(self.name, self.names.__str__().strip("[]")))

    def _assign_reinitialize_condition_to_mechanisms(self, reinitialize_mechanisms_when):
        """
        Assign the Condition specified in the reinitialize_mechanisms_when argument to the reinitialize_when attribute
        of each Mechanism in the System.
        """
        if not isinstance(reinitialize_mechanisms_when, Condition):
            raise SystemError("{} is not a valid specification for reinitialize_mechanisms_when of {}. "
                              "reinitialize_mechanisms_when must be a Condition.".format(reinitialize_mechanisms_when,
                                                                                         self.name))
        for mechanism in self.mechanisms:
            if hasattr(mechanism, "reinitialize_when"):
                if isinstance(mechanism.reinitialize_when, Never):
                    mechanism.reinitialize_when = reinitialize_mechanisms_when

    def _validate_variable(self, variable, context=None):
        """Convert variable to 2D np.array: \
        one 1D value for each InputPort
        """
        super(System, self)._validate_variable(variable, context)

        # Force System variable specification to be a 2D array (to accommodate multiple input ports of 1st mech(s)):
        if variable is None:
            return

        return variable

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate controller, processes and initial_values
        """
        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        if CONTROLLER in target_set and target_set[CONTROLLER] is not None:
            controller = target_set[CONTROLLER]
            if (not isinstance(controller, ControlMechanism) and
                    not (inspect.isclass(controller) and issubclass(controller, ControlMechanism))):
                raise SystemError("{} (controller arg for \'{}\') is not a ControllerMechanism or subclass of one".
                                  format(controller, self.name))

        for process in target_set[PROCESSES]:
            if not isinstance(process, Process_Base):
                raise SystemError("{} (in processes arg for \'{}\') is not a Process object".format(process, self.name))

        if INITIAL_VALUES in target_set and target_set[INITIAL_VALUES] is not None:
            for mech, value in target_set[INITIAL_VALUES].items():
                if not isinstance(mech, Mechanism):
                    raise SystemError("{} (key for entry in initial_values arg for \'{}\') "
                                      "is not a Mechanism object".format(mech, self.name))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Instantiate processes and graph

        These calls must be made before _instantiate_function as the latter may be called during init for validation
        :param function:
        """
        self._instantiate_processes(input=self.defaults.variable, context=context)
        self._instantiate_graph(context=context)
        self._instantiate_learning_graph(context=context)

    def _instantiate_function(self, function, function_params=None, context=None):
        """Suppress validation of function

        This is necessary to:
        - insure there is no FUNCTION specified (not allowed for a System object)
        - suppress validation (and attendant execution) of System execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in PROCESSES have already been validated
        """

        if self.function != self.execute:
            print("System object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.function, FUNCTION)
            self.function = self.execute

    def _instantiate_value(self, context=None):
        # If validation pref is set, execute the System
        if self.prefs.paramValidationPref:
            super()._instantiate_value(context=context)
        # Otherwise, just set System output info to the corresponding info for the last mechanism(s) in self.processes
        else:
            value = self.processes[-1].output_port.value
            try:
                # Could be mutable, so assign copy
                self.defaults.value = value.copy()
            except AttributeError:
                # Immutable, so just assign value
                self.defaults.value = value

    def _instantiate_processes(self, input=None, context=None):
# FIX: ALLOW Projections (??ProjectionTiming TUPLES) TO BE INTERPOSED BETWEEN MECHANISMS IN PATHWAY
# FIX: AUGMENT LinearMatrix TO USE FULL_CONNECTIVITY_MATRIX IF len(sender) != len(receiver)
        """Instantiate processes of System

        Use self.processes
        If self.processes is empty, instantiate default process by calling process()
        Iterate through self.processes, instantiating each (including the input to each input projection)
        If input is specified, check that it's length equals the number of processes
        If input is not specified, compose from the input for each Process (value specified or, if None, default)
        Note: specification of input for System takes precedence over specification for Processes

        # ??STILL THE CASE, OR MOVED TO _instantiate_graph:
        Iterate through Process._mechs for each Process;  for each sequential pair:
            - create set entry:  <receiving Mechanism>: {<sending Mechanism>}
            - add each pair as an entry in self.execution_graph
        """

        self.mechanismsDict = {}
        self._all_mechs = []
        self._all_mechanisms = MechanismList(self, self._all_mechs)

        # Get list of processes specified in arg to init,
        #    possibly appended by EVCControlMechanism (with prediction processes)
        processes_spec = self.processes

        # Assign default Process if PROCESS is empty, or invalid
        if not processes_spec:
            from psyneulink.core.components.process import Process
            processes_spec.append(ProcessTuple(Process(), None))

        # If input to system is specified, number of items must equal number of processes with origin mechanisms
        if input is not None and len(input) != len(self.origin_mechanisms):
            raise SystemError("Number of items in input ({}) must equal number of processes ({}) in {} ".
                              format(len(input), len(self.origin_mechanisms),self.name))

        #region VALIDATE EACH ENTRY, STANDARDIZE FORMAT AND INSTANTIATE PROCESS

        # Convert all entries to (process, input) tuples, with None as filler for absent input
        input_index = input_index_curr = 0
        for i in range(len(processes_spec)):

            # Get list of origin mechanisms for processes that have already been converted
            #   (for use below in assigning input)
            orig_mechs_already_processed = list(p[0].origin_mechanisms[0] for
                                                p in processes_spec if isinstance(p,ProcessTuple))

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
                for process_input_port in process.process_input_ports:
                    process_input.extend(process_input_port.value)
                processes_spec[i] = ProcessTuple(process, process_input)
            # Input was provided on command line, so assign that to input item of tuple
            else:
                # Assign None as input to processes implemented by controller (controller provides their input)
                #    (e.g., prediction processes implemented by EVCControlMechanism)
                if processes_spec[i].process._isControllerProcess:
                    processes_spec[i] = ProcessTuple(processes_spec[i].process, None)
                else:
                    # Replace input item in tuple with one from command line
                    # Note:  check if origin mechanism for current process is same as any previous one;
                    #        if it is, use that one (and don't increment index for input
                    #        otherwise, assign input and increment input_index
                    try:
                        input_index_curr = orig_mechs_already_processed.index(processes_spec[i][0].origin_mechanisms[0])
                    except ValueError:
                        input_index += 1
                    processes_spec[i] = ProcessTuple(processes_spec[i].process, input[input_index_curr])
                    input_index_curr = input_index

            # Validate input
            if (processes_spec[i].input is not None and
                    not isinstance(processes_spec[i].input,(numbers.Number, list, np.ndarray))):
                raise SystemError("Second item of entry {0} ({1}) must be an input value".
                                  format(i, processes_spec[i].input))

            process = processes_spec[i].process
            process_input = processes_spec[i].input

            # IMPLEMENT: THIS IS WHERE LEARNING SPECIFIED FOR A SYSTEM SHOULD BE IMPLEMENTED FOR EACH PROCESS IN THE
            #            SYSTEM;  NOTE:  IF THE PROCESS IS ALREADY INSTANTIATED WITHOUT LEARNING
            #            (FIRST CONDITIONAL BELOW), MAY NEED TO BE RE-INSTANTIATED WITH LEARNING
            #            (QUESTION:  WHERE TO GET SPECS FOR PROCESS FOR RE-INSTANTIATION??)

            # If process item is a Process object, assign process_input as default
            if isinstance(process, Process_Base):
                if process_input is not None:
                    process._instantiate_defaults(variable=process_input, context=context)
            else:
                raise SystemError("Entry {0} of PROCESSES ({1}) for {} must be a Process object".
                                  format(i, process, self.name))

            # # process should now be a Process object;  assign to processList
            # self.processList.append(process)

            # Assign the Process a reference to this System
            process.systems.append(self)
            if process._learning_enabled:
                self.learning = True

            # Get max of Process phaseSpecs
            self._phaseSpecMax = int(max(math.floor(process._phaseSpecMax), self._phaseSpecMax))

            # Iterate through mechanism tuples in Process' mechs
            #     to construct self._all_mechs and mechanismsDict
            # FIX: ??REPLACE WITH:  for sender_object_item in Process._mechs
            for sender_object_item in process._mechs:

                sender_mech = sender_object_item

                # THIS IS NOW DONE IN _instantiate_graph
                # # Add system to the Mechanism's list of systems of which it is member
                # if not self in sender_object_item[MECHANISM].systems:
                #     sender_mech._add_system(self, INTERNAL)

                # Assign sender mechanism entry in self.mechanismsDict, with object_item as key and its Process as value
                #     (this is used by Process._instantiate_pathway() to determine if Process is part of System)
                # If the sender is already in the System's mechanisms dict
                if sender_object_item in self.mechanismsDict:
                    # existing_object_item = self._all_mechanisms._get_tuple_for_mech(sender_mech)
                    # Add to entry's list
                    self.mechanismsDict[sender_mech].append(process)
                else:
                    # Add new entry
                    self.mechanismsDict[sender_mech] = [process]
                if not sender_object_item in self._all_mechs:
                    self._all_mechs.append(sender_object_item)

                # Add ObjectiveMechanism for ControlMechanism if the latter is not the System's controller
                if (isinstance(sender_object_item, ControlMechanism)
                    and (self.controller is None or not sender_object_item is self.controller)
                    and isinstance(sender_object_item.objective_mechanism, ObjectiveMechanism)
                    and not sender_object_item.objective_mechanism in self._all_mechs):
                    self._all_mechs.append(sender_object_item.objective_mechanism)

            process._all_mechanisms = MechanismList(process, components_list=process._mechs)
            for proj in process.projections:
                if not isinstance(proj.sender, ProcessInputPort):
                    proj._activate_for_compositions(self)

        # Call all ControlMechanisms to allow them to implement specification of ALL
        #    in monitor_for_control and/or control_signals arguments of their constructors
        for mech in self.mechanisms:
            pass

        # # Instantiate processList using process_tuples, and point self.processes to it
        # # Note: this also points self.processes to self.processes
        self.process_tuples = processes_spec
        self._processList = ProcessList(self, self.process_tuples)
        self.processes = self._processList.processes

    def _instantiate_graph(self, context=None):
        """Construct graph (full) and execution_graph (acyclic) of System

        Instantate a graph of all of the Mechanisms in the System and their dependencies,
            designate a type for each Mechanism in the graph,
            instantiate the execution_graph, a subset of the graph with any cycles removed,
                and topologically sorted into a sequentially ordered list of sets
                containing mechanisms to be executed at the same time

        graph contains a dictionary of dependency sets for all Mechanisms in the System:
            reciever_object_item : {sender_object_item, sender_object_item...}
        execution_graph contains an acyclic subset of graph used to determine sequence of Mechanism execution;

        They are constructed as follows:
            sequence through self.processes;  for each Process:
                begin with process.first_mechanism (assign as `ORIGIN` if it doesn't receive any Projections)
                traverse all Projections
                for each Mechanism encountered (receiver), assign to its dependency set the previous (sender) Mechanism
                for each assignment, use toposort to test whether the dependency introduced a cycle; if so:
                    eliminate the dependent from execution_graph, and designate it as `CYCLE` (unless it is an `ORIGIN`)
                    designate the sender as `INITIALIZE_CYCLE` (it can receive and initial_value specification)
                if a Mechanism doe not project to any other ProcessingMechanisms (ignore learning and control mechs):
                    assign as `TERMINAL` unless it is already an `ORIGIN`, in which case assign as `SINGLETON`

        Construct execution_sets and exeuction_list

        Assign MechanismLists:
            allMechanisms
            origin_mechanisms
            terminalMechanisms
            recurrent_init_mechanisms (INITIALIZE_CYCLE)
            learning_mechanisms
            control_mechanism

        Validate initial_values

        """
        from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism

        def is_monitoring_mech(mech):
            if ((isinstance(mech, ObjectiveMechanism) and mech._role) or
                    isinstance(mech, (LearningMechanism, ControlMechanism))):
                return True
            else:
                return False

        def is_in_system(mech):
            if set(self.processes).intersection(set(mech.processes)):
                return True
            return False

        # Use to recursively traverse processes
        def build_dependency_sets_by_traversing_projections(sender_mech):

            # DEAL WITH LEARNING AND CONTROL MECHANISMS -----------------------------------------------------------
            #  (and their ObjectiveMechanisms)
            if is_monitoring_mech(sender_mech):

                # LearningMechanisms or ObjectiveMechanism used for learning:  label as LEARNING and return
                if (isinstance(sender_mech, LearningMechanism) or
                        (isinstance(sender_mech, ObjectiveMechanism) and sender_mech._role is LEARNING)):
                    sender_mech._add_system(self, LEARNING)
                    return
                # System's controller or ObjectiveMechanism that projects it: label as CONTROL and return
                # IMPLEMENTATION NOTE:  This allows ObjectiveMechanisms to be included in the System's execution_graph
                #                           that project to Mechanisms other than the System's controller.
                #                       If the ObjectiveMechanism projects to the controller and other Mechanisms
                #                           a warning is issued and those other projections are ignored.
                elif (sender_mech is self.controller or
                          (isinstance(sender_mech, ObjectiveMechanism) and sender_mech.for_controller)):
                    sender_mech._add_system(self, CONTROL)
                    obj_mech_rcvrs = [[projection.receiver.owner for projection in output_port.efferents]
                             for output_port in sender_mech.output_ports]
                    if len(obj_mech_rcvrs) > 1:
                        warnings.warning("{0} projects to multiple {1}s {2}. If an {3} projects to the controller"
                                         "of a {4}, its projection to any other {1}s is not currently supported; "
                                         "these have been ignored in the System graph".
                                         format(sender_mech.name,
                                                Mechanism.__name__ ,
                                                obj_mech_rcvrs,
                                                ObjectiveMechanism.__name__,
                                                System.__name__))
                    return
                # If sender is a ControlMechanism that is not the controller for the System,
                #    assign its dependency to its ObjectiveMechanism and label as INTERNAL
                elif (isinstance(sender_mech, ControlMechanism)
                      and is_in_system(sender_mech)):
                    sender_mech._add_system(self, INTERNAL)

            # PRUNE ANY NON-SYSTEM COMPONENTS ---------------------------------------------------------------------

            # Delete any projections to mechanism from processes or mechanisms in processes not in current system
            for input_port in sender_mech.input_ports:
                for projection in input_port.all_afferents:
                    sender = projection.sender.owner
                    system_processes = self.processes
                    if isinstance(sender, Process_Base):
                        if not sender in system_processes:
                            del projection
                    elif not all(sender_process in system_processes for sender_process in sender.processes):
                        del projection

            # If sender_mech has no projections left, raise exception
            if not any(any(projection for projection in input_port.all_afferents)
                       for input_port in sender_mech.input_ports):
                raise SystemError("{} only receives Projections from other Processes or Mechanisms not"
                                  " in the current System ({})".format(sender_mech.name, self.name))

            # ASSIGN TERMINAL MECHANISM(S) -----------------------------------------------------------------------

            # Assign as TERMINAL (or SINGLETON) if it:
            #    - it is not a ControlMechanism and
            #    - it is not an Objective Mechanism used for Learning or Control and
            #    - it has no outgoing projections or
            #          only ones to ObjectiveMechanism(s) used for Learning or Control
            # Note:  SINGLETON is assigned if mechanism is already a TERMINAL;  indicates that it is both
            #        an ORIGIN AND A TERMINAL and thus must be the only mechanism in its process
            if (
                not (isinstance(sender_mech, ControlMechanism) or
                # FIX: ALLOW IT TO BE TERMINAL IF IT PROJECTS ONLY TO A ControlMechanism or ObjectiveMechanism for one
                    # It is not an ObjectiveMechanism used for Learning or for the controller of the System
                    (isinstance(sender_mech, ObjectiveMechanism) and sender_mech._role in (LEARNING,CONTROL)))
                    and
                        # All of its projections
                        all(
                            all(
                                # are to ControlMechanism(s)...
                                isinstance(projection.receiver.owner, (ControlMechanism, LearningMechanism))
                                    # or to ObjectiveMechanism(s) used for Learning or Control...
                                    or (isinstance(projection.receiver.owner, ObjectiveMechanism)
                                        and projection.receiver.owner._role in (LEARNING, CONTROL))
                                # or are to itself!
                                or projection.receiver.owner is sender_mech
                            for projection in output_port.efferents)
                        for output_port in sender_mech.output_ports)):
                try:
                    if sender_mech.systems[self] is ORIGIN:
                        sender_mech._add_system(self, SINGLETON)
                    else:
                        sender_mech._add_system(self, TERMINAL)
                except KeyError:
                    sender_mech._add_system(self, TERMINAL)
                # If sender_mech has projections to ControlMechanism and/or Objective Mechanisms used for control
                #    that are NOT the System's controller, then continue to track those projections
                #    for dependents to add to the execution_graph;
                if any(
                        any(
                            # Projection to a ControlMechanism that is not the System's controller
                                    (isinstance(projection.receiver.owner, ControlMechanism)
                                     and not projection.receiver.owner is self.controller)
                            # or Projection to an ObjectiveMechanism that is not for the System's controller
                            or (isinstance(projection.receiver.owner, ObjectiveMechanism)
                                and projection.receiver.owner._role is CONTROL
                                and (self.controller is None or (self.controller is not None
                                and not projection.receiver.owner is self.controller.objective_mechanism)))
                                    for projection in output_port.efferents)
                        for output_port in sender_mech.output_ports):
                    pass

                # If sender_mech projects to an LearningMechanism that executes in the execution phase,
                #    let it pass, as that is a legitimate dependent that should be included in the execution_list
                elif any(
                        (projection.receiver.owner.learning_timing is LearningTiming.EXECUTION_PHASE
                         for projection in output_port.efferents)
                        for output_port in sender_mech.output_ports):
                    pass
                # Otherwise, don't track any of the TERMINAL Mechanism's projections
                else:
                    return

            # FIND DEPENDENTS AND ADD TO GRAPH ---------------------------------------------------------------------

            if not sender_mech.output_ports:
                return

            for output_port in sender_mech.output_ports:

                for projection in output_port.efferents:
                    receiver = projection.receiver.owner
                    # receiver_tuple = self._all_mechanisms._get_tuple_for_mech(receiver)

                    # If receiver is not in system's list of mechanisms,
                    #    must belong to a process that has NOT been included in the system,
                    #    ignore it unless it is an AutoAssociativeLearningMechanism for the sender
                    if (not receiver or
                            # MODIFIED 7/28/17 CW: added a check for auto-recurrent projections
                            #                      (i.e. receiver is sender_mech)
                            # FIX: JDC: NOT SURE WE WANT THIS CHECK, AS IT PRECLUDES IDENTIFYING MECHANISMS
                            # FIX:      THAT SHOULD BE IDENTIFIED AS CYCLES AND ASSIGNED INITIALIZATION ROLE
                            receiver is sender_mech
                            # MODIFIED 7/8/17 END
                            # Exclude any Mechanisms not in any processes belonging to the current System
                            or not is_in_system(receiver)
                    ):
                        continue
                    if is_monitoring_mech(receiver):
                        # Check if receiver is:
                        #    the controller for the System or the ObjectiveMechanism for one, or
                        #    a LearningMechanism or the ObjectiveMechanism for one
                        if (receiver is self.controller
                                or isinstance(receiver, LearningMechanism)
                                or (self.controller is not None and
                                    isinstance(receiver, self.controller.objective_mechanism))
                                or (isinstance(receiver, ObjectiveMechanism) and receiver._role is LEARNING)):
                            # If it is a LearningMechanism for the sender_mech that executes in the execution phase,
                            #    include it
                            if (isinstance(receiver, LearningMechanism) and
                                    receiver.learning_timing is LearningTiming.EXECUTION_PHASE):
                                # If it is an AutoAssociativeLearningMechanism, check that it projects to itself
                                if isinstance(receiver, AutoAssociativeLearningMechanism):
                                    if not receiver == sender_mech.learning_mechanism:
                                        raise SystemError("PROGRAM ERROR: {} is an {} that receives a projection "
                                                          "from {} but does not project to its {}".
                                                          format(receiver.name,
                                                                 AutoAssociativeLearningMechanism.__name__,
                                                                 sender_mech.name,
                                                                 AutoAssociativeProjection.__name__))
                            # Otherwise, exclude from execute_graph
                            else:
                                continue
                    try:
                        self.graph[receiver].add(sender_mech)
                    except KeyError:
                        self.graph[receiver] = {sender_mech}

                    # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                    # Do not include dependency (or receiver on sender) in execution_graph for this projection
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
                    # FIX: MODIFY THIS TO (GO BACK TO) USING if receiver_tuple in self.execution_graph
                    # FIX  BUT CHECK THAT THEY ARE IN DIFFERENT PHASES
                    if receiver in self.execution_graph:
                        # Try assigning receiver as dependent of current mechanism and test toposort
                        try:
                            # If receiver_tuple already has dependencies in its set, add sender_mech to set
                            if self.execution_graph[receiver]:
                                self.execution_graph[receiver].add(sender_mech)
                            # If receiver set is empty, assign sender_mech to set
                            else:
                                self.execution_graph[receiver] = {sender_mech}
                            # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                            list(toposort(self.execution_graph))
                        # If making receiver dependent on sender produced a cycle (feedback loop), remove from graph
                        except ValueError:
                            self.execution_graph[receiver].remove(sender_mech)
                            # Assign sender_mech INITIALIZE_CYCLE as system status if not ORIGIN or not yet assigned
                            if not sender_mech.systems or not (sender_mech.systems[self] in
                                                               {ORIGIN, SINGLETON,TERMINAL}):
                                sender_mech._add_system(self, INITIALIZE_CYCLE)
                            if not (receiver.systems[self] in {ORIGIN, SINGLETON, TERMINAL}):
                                receiver._add_system(self, CYCLE)
                            continue

                    else:
                        # Assign receiver as dependent on sender mechanism
                        try:
                            # FIX: THIS WILL ADD SENDER_MECH IF RECEIVER IS IN GRAPH BUT = set()
                            # FIX: DOES THAT SCREW UP ORIGINS?
                            self.execution_graph[receiver].\
                                add(sender_mech)
                        except KeyError:
                            self.execution_graph[receiver] = \
                                {sender_mech}

                    if not sender_mech.systems:
                        sender_mech._add_system(self, INTERNAL)

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver)

        self.graph = OrderedDict()
        self.execution_graph = OrderedDict()

        # Sort for consistency of output
        sorted_processes = sorted(self.processes, key=lambda process : process.name)

        for process in sorted_processes:
            first_mech = process.first_mechanism

            # Treat as ORIGIN if ALL projections to the first mechanism in the process are from:
            #    - the process itself (ProcessInputPort)
            #    - another mechanism in the in process (i.e., feedback projections from *within* the process)
            #    - mechanisms from other process for which it is an origin
            # Notes:
            # * This precludes a mechanism that is an ORIGIN of a process from being an ORIGIN for the system
            #       if it receives any projections from any other mechanisms in the system (including other processes)
            #       other than ones in processes for which it is also their ORIGIN
            # * This does allow a mechanism to be the ORIGIN (but *only* the ORIGIN) for > 1 process in the system
            try:
                if all(
                        all(
                                # All projections must be from a process (i.e., ProcessInputPort) to which it belongs
                                projection.sender.owner in first_mech.processes or
                                # or from mechanisms within its own process (e.g., [a, b, a])
                                projection.sender.owner in list(process.mechanisms) or
                                # or from Mechanisms in other processes for which it is also an ORIGIN ([a,b,a],[a,c,a])
                                not isinstance(projection.sender.owner, Mechanism)
                                or all(
                                    ORIGIN in first_mech.processes[proc]
                                    for proc in projection.sender.owner.processes
                                )
                            # For all the projections to each InputPort
                            for projection in input_port.path_afferents)
                        # For all input_ports for the first_mech
                        for input_port in first_mech.input_ports):
                    # Assign its set value as empty, marking it as a "leaf" in the graph
                    object_item = first_mech
                    self.graph[object_item] = set()
                    self.execution_graph[object_item] = set()
                    first_mech._add_system(self, ORIGIN)
            except KeyError as e:
                # IMPLEMENTATION NOTE:
                # This occurs if a Mechanism belongs to one (or more) Process(es) in the System but not ALL of them;
                #    it is because each Mechanism in the test above ("ORIGIN in first_mech.processes[proc]")
                #     is examined for all Processes in the System);
                # FIX: 10/3/17 - this should be factored into the tests above so that the exception does not occur
                if isinstance(e.args[0], Process_Base):
                    pass
                else:
                    raise SystemError(e)

            build_dependency_sets_by_traversing_projections(first_mech)

        # Print graph
        if self.verbosePref:
            warnings.warn("In the System graph for \'{}\':".format(self.name))
            for receiver_object_item, dep_set in self.execution_graph.items():
                mech = receiver_object_item
                if not dep_set:
                    print("\t'{}' is an {} Mechanism".
                          format(mech.name, mech.systems[self]))
                else:
                    status = mech.systems[self]
                    if status is TERMINAL:
                        status = 'a ' + status
                    elif status in {INTERNAL, INITIALIZE_CYCLE}:
                        status = 'an ' + status
                    print("\t'{}' is {} Mechanism that receives Projections from:".format(mech.name, status))
                    for sender_object_item in dep_set:
                        print("\t\t\'{}\'".format(sender_object_item.name))

        # For each mechanism (represented by its tuple) in the graph, add entry to relevant list(s)
        # Note: ignore mechanisms belonging to controllerProcesses (e.g., instantiated by EVCControlMechanism)
        #       as they are for internal use only;
        #       this also ignored learning-related mechanisms (they are handled below)
        self._origin_mechs = []
        self._terminal_mechs = []
        self._recurrent_init_mechs = []
        self._control_mechs = []

        for object_item in self.execution_graph:

            mech = object_item

            if mech.systems[self] in {ORIGIN, SINGLETON}:
                for process, status in mech.processes.items():
                    if process._isControllerProcess:
                        continue
                    self._origin_mechs.append(object_item)
                    break

            if object_item.systems[self] in {TERMINAL, SINGLETON}:
                for process, status in mech.processes.items():
                    if process._isControllerProcess:
                        continue
                    self._terminal_mechs.append(object_item)
                    break

            if object_item.systems[self] in {INITIALIZE_CYCLE}:
                for process, status in mech.processes.items():
                    if process._isControllerProcess:
                        continue
                    self._recurrent_init_mechs.append(object_item)
                    break

            if isinstance(object_item, ControlMechanism):
                if not object_item in self._control_mechs:
                    self._control_mechs.append(object_item)

        self.origin_mechanisms = MechanismList(self, self._origin_mechs)
        self.terminal_mechanisms = MechanismList(self, self._terminal_mechs)
        self.recurrent_init_mechanisms = MechanismList(self, self._recurrent_init_mechs)
        self.control_mechanisms = MechanismList(self, self._control_mechs) # Used for inspection and in case there
                                                                              # are multiple controllers in the future

        try:
            self.execution_sets = list(toposort(self.execution_graph))
        except ValueError as e:
            if 'Cyclic dependencies exist' in e.args[0]:
                # if self.verbosePref:
                # print('{} has feedback connections; be sure that the following items are properly initialized:'.
                #       format(self.name))
                raise SystemError("PROGRAM ERROR: cycle (feedback loop) in {} not detected by _instantiate_graph ".
                                  format(self.name))

        # Create instance of sequential (execution) list:
        self.execution_list = self._toposort_with_ordered_mechs(self.execution_graph)

        # Construct self.defaults.variable from inputs to ORIGIN mechanisms
        new_variable = []
        for mech in self.origin_mechanisms:
            orig_mech_input = []
            for input_port in mech.input_ports:
                orig_mech_input.append(input_port.value)
            new_variable.append(orig_mech_input)
        self.defaults.variable = convert_to_np_array(new_variable, 2)
        # should add Utility to allow conversion to 3D array
        # An example: when InputPort values are vectors, then self.defaults.variable is a 3D array because
        # an origin mechanism could have multiple input ports if there is a recurrent InputPort. However,
        # if InputPort values are all non-vector objects, such as strings, then self.defaults.variable
        # would be a 2D array. so we should convert that to a 3D array

        # Instantiate StimulusInputPorts
        self._instantiate_stimulus_inputs(context=context)

        # Validate initial values
        # FIX: CHECK WHETHER ALL MECHANISMS DESIGNATED AS INITIALIZE HAVE AN INITIAL_VALUES ENTRY
        # FIX: ONLY CHECKS FIRST ITEM OF self.defaults.value (ASSUMES THAT IS ALL THAT WILL GET ASSIGNED)
        # FIX: ONLY CHECK ONES THAT RECEIVE PROJECTIONS
        if self.initial_values is not None:
            for mech, value in self.initial_values.items():
                if not mech in self.execution_graph:
                    raise SystemError("{} (entry in initial_values arg) is not a Mechanism in \'{}\'".
                                      format(mech.name, self.name))
                if not iscompatible(value, mech.defaults.value[0]):
                    raise SystemError("{} (in initial_values arg for \'{}\') is not a valid value for {}".
                                      format(value, self.name, append_type_to_name(self)))

    def _instantiate_stimulus_inputs(self, context=None):

# FIX: ZERO VALUE OF ALL ProcessInputPorts BEFORE EXECUTING
# FIX: RENAME SystemInputPort -> SystemInputPort

        # Create SystemInputPort for each ORIGIN mechanism in origin_mechanisms and
        #    assign MappingProjection from the SystemInputPort to the ORIGIN mechanism
        for i, origin_mech in zip(range(len(self.origin_mechanisms)), self.origin_mechanisms):

            # Skip if ORIGIN mechanism already has a projection from a SystemInputPort in current system
            # (this avoids duplication from multiple passes through _instantiate_graph)
            if any(self is projection.sender.owner for projection in origin_mech.input_port.path_afferents):
                continue
            # added a for loop to iterate over origin_mech.input_ports to allow for multiple input ports in an
            # origin mechanism (useful only if origin mechanism is a KWTAMechanism) Check, for each ORIGIN mechanism,
            # that the length of the corresponding item of self.defaults.variable matches the length of the
            #  ORIGIN inputPort's defaults.variable attribute
            for j in range(len(origin_mech.input_ports)):
                if origin_mech.input_ports[j].internal_only:
                    continue
                if len(self.defaults.variable[i][j]) != origin_mech.input_ports[j].socket_width:
                    raise SystemError("Length of input {} ({}) does not match the length of the input ({}) for the "
                                      "corresponding ORIGIN Mechanism ()".
                                      format(i,
                                             len(self.defaults.variable[i][j]),
                                             origin_mech.input_ports[j].socket_width,
                                             origin_mech.name))
                stimulus_input_port = SystemInputPort(owner=self,
                                                        variable=origin_mech.input_ports[j].socket_template,
                                                        prefs=self.prefs,
                                                        name="System Input Port to Mechansism {}, Input Port {}".
                                                        format(origin_mech.name,j),
                                                        context=context)
                self.stimulusInputPorts.append(stimulus_input_port)
                self.inputs.append(stimulus_input_port.value)

                # Add MappingProjection from stimulus_input_port to ORIGIN mechainsm's inputPort
                from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
                proj = MappingProjection(sender=stimulus_input_port,
                                  receiver=origin_mech.input_ports[j],
                                  name=self.name + ' Input Projection to ' + origin_mech.name + ' Input Port ' + str(j))
                proj._activate_for_compositions(self)


    def _instantiate_learning_graph(self, context=None):
        """Build graph of LearningMechanism and LearningProjections
        """
        from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
            LearningMechanism

        self.learningGraph = OrderedDict()
        self.learning_execution_graph = OrderedDict()

        def build_dependency_sets_by_traversing_projections(sender_mech, process):

            # MappingProjections are legal recipients of learning projections (hence the call)
            #  but do not send any projections, so no need to consider further
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            if isinstance(sender_mech, MappingProjection):
                return

            # Exclude LearningMechanisms that execute in the execution phase
            #    as they are included in (and executed as part of) System.graph
            elif (isinstance(sender_mech, LearningMechanism) and
                  sender_mech.learning_timing is LearningTiming.EXECUTION_PHASE):
                return

            # All other sender_mechs must be either a LearningMechanism or a ComparatorMechanism with role=LEARNING
            elif not (isinstance(sender_mech, LearningMechanism) or
                          (isinstance(sender_mech, ObjectiveMechanism) and sender_mech._role is LEARNING)):
                raise SystemError("PROGRAM ERROR: {} is not a legal object for learning graph;"
                                  "must be a LearningMechanism or an ObjectiveMechanism".
                                  format(sender_mech))

            # MANAGE TARGET ObjectiveMechanism FOR INTERNAL or TERMINAL CONVERGENCE of PATHWAYS

            # If sender_mech is an ObjectiveMechanism, and:
            #    - none of the Mechanisms that project to it are are a TERMINAL Mechanism for the current Process, or
            #    - all of the Mechanisms that project to it already have an ObjectiveMechanism,
            # Then:
            #    - do not include the ObjectiveMechanism in the graph;
            #    - be sure that its outputPort projects to the ERROR_SIGNAL inputPort of a LearningMechanism
            #        (labelled "learning_mech" here -- raise an exception if it does not;
            #    - determine whether learning_mech's ERROR_SIGNAL inputPort receives any other projections
            #        from another ObjectiveMechanism or LearningMechanism (labelled "error_signal_projection" here)
            #        -- if it does, be sure that it is from the same system and if so return;
            #           (note:  this shouldn't be true, but the test is here for completeness and sanity-checking)
            #    - if learning_mech's ERROR_SIGNAL inputPort does not receive any projections from
            #        another objectiveMechanism and/or LearningMechanism in the system, then:
            #        - find the sender to the ObjectiveMechanism (labelled "error_source" here)
            #        - find the 1st projection from error_source that projects to the ACTIVATION_INPUT inputPort of
            #            a LearningMechanism (labelled "error_signal" here)
            #        - instantiate a MappingProjection from error_signal to learning_mech
            #            projected
            # IMPLEMENTATION NOTE: Composition should allow 1st condition if user indicates internal TARGET is desired;
            #                  for now, however, assume this is not desired (i.e., only TERMINAL mechanisms
            #                  should project to ObjectiveMechanisms) and always replace internal
            #                  ObjectiveMechanism with projection from a LearningMechanism (if it is available)
            # Otherwise:
            #     - include it in the graph

            obj_mech_replaced = False

            if isinstance(sender_mech, ObjectiveMechanism):

                # For clarity, rename as obj_mech
                obj_mech = sender_mech

                # Get the LearningMechanism to which the obj_mech projects
                try:
                    learning_mech = obj_mech.output_port.efferents[0].receiver.owner
                    if not isinstance(learning_mech, LearningMechanism):
                        raise AttributeError
                except AttributeError:
                    raise SystemError("{} in {} does not project to a LearningMechanism".
                                      format(obj_mech.name, process.name))

                # Make sure sample_mech is referenced by learning_mech as is output_source
                sample_mech = obj_mech.input_ports[SAMPLE].path_afferents[0].sender.owner
                if sample_mech != learning_mech.output_source:
                    raise SystemError("PROGRAM ERROR: learning_mecch ({}) does not properly reference sample_mech ({})"
                                      "in {} of {}".format(learning_mech.name,sample_mech.name,process.name,self.name))

                # ObjectiveMechanism is the 1st item in the learning_execution_graph, so could be for:
                #    - the last Mechanism in a learning sequence, or
                #    - a TERMINAL Mechanism of the System
                if len(self.learning_execution_graph) == 0:
                    # If is the last item in a learning sequence,
                    #    doesn't matter if it is a TERMINAL Mechanism;  needs to remain as a Target for the System
                    if not any(proj.has_learning_projection and self in proj.receiver.owner.systems
                               for proj in sample_mech.output_port.efferents):
                        pass
                    # If sample_mech is:
                    #    - NOT for a TERMINAL Mechanism of the current System
                    # Then:
                    #    - obj_mech should NOT be included in the learning_execution_graph and
                    #    - should be replaced with appropriate projections to sample_mechs's afferent LearningMechanisms
                    elif not sample_mech.systems[self] is TERMINAL:
                        _assign_error_signal_projections(sample_mech, system=self, scope=self, objective_mech=obj_mech)
                        # Don't process ObjectiveMechanism any further (since its been replaced)
                        return

                # NOT 1st item in the learning_execution_graph, so it must be for the TERMINAL Mechanism of a Process
                else:

                    # TERMINAL CONVERGENCE
                    # All of the mechanisms that project to obj_mech
                    #    project to another ObjectiveMechanism already in the learning_graph
                    if all(
                            any((isinstance(receiver_mech, ObjectiveMechanism) and
                                 # its already in a dependency set in the learning_execution_graph
                                 receiver_mech in set.union(*list(self.learning_execution_graph.values())) and
                                 not receiver_mech is obj_mech)
                                # receivers of senders to obj_mech
                                for receiver_mech in [proj.receiver.owner for proj in
                                                      mech.output_port.efferents])
                            # senders to obj_mech
                            for mech in [proj.sender.owner
                                         for proj in obj_mech.input_ports[SAMPLE].path_afferents]):

                        # Get the other ObjectiveMechanism to which the error_source projects (in addition to obj_mech)
                        other_obj_mech = next((projection.receiver.owner for projection in
                                               sample_mech.output_port.efferents if
                                               isinstance(projection.receiver.owner, ObjectiveMechanism)), None)
                        if not other_obj_mech is obj_mech:
                            sender_mech = other_obj_mech
                            sender_mech._add_process(process, TARGET)
                            obj_mech_replaced = TERMINAL
                            # Move error_signal Projections from old obj_mech to new one (now sender_mech)
                            for error_signal_proj in obj_mech.output_ports[OUTCOME].efferents:
                                # IMPLEMENTATION NOTE:  MOVE TO COMPOSITION WHEN THAT HAS BEEN IMPLEMENTED
                                proj = MappingProjection(sender=sender_mech, receiver=error_signal_proj.receiver)
                                proj._activate_for_compositions(self)
                                _assign_error_signal_projections(sample_mech, self, scope=process, objective_mech=obj_mech)
                                # sender_mech.output_ports[OUTCOME].efferents.append(error_signal_proj)

                    # INTERNAL CONVERGENCE
                    # None of the mechanisms that project to it are a TERMINAL mechanism
                    elif (not all(all(projection.sender.owner.processes[proc] is TERMINAL
                                     for proc in projection.sender.owner.processes)
                                 for projection in obj_mech.input_ports[SAMPLE].path_afferents)
                          # and it is not for the last Mechanism in a learning sequence
                          and any(proj.has_learning_projection and self in proj.receiver.owner.systems
                                  for proj in sample_mech.output_port.efferents)
                    ):
                        _assign_error_signal_projections(processing_mech=sample_mech,
                                                         system=self,
                                                         objective_mech=obj_mech)
                        obj_mech_replaced = INTERNAL

                self.learningGraph[sender_mech]=None

            # FIX: TEST FOR CROSSING:
            # FIX:  (LEARNINGMECHANISM FOR INTERNAL MECHANISM THAT HAS >1 PROJECTION TO MECHANISMS IN THE SAME SYSTEM
            #
            # - IDENTIFY ALL OF THE OUTGOING PROJECTIONS FROM THE MECHANISMS ABOVE THAT ARE:
            #     - BEING LEARNED
            #     - PROJECT TO A MECHANISM IN THE CURRENT SYSTEM
            # - ASSIGN MAPPING PROJECTION TO NEW ERROR_SIGNAL INPUT_PORT FOR sender_mech

            # sender_mech is a LearningMechanism:
            else:
                # For each of the ProcessingMechanisms that receive Projections being trained by sender_mech
                for processing_mech in [proj.receiver.owner for proj in sender_mech.learned_projections]:
                    # If it is an INTERNAL Mechanism for the System,
                    #    make sure that the LearningMechanisms for all of its afferent Projections being learned
                    #    receive error_signals from the LearningMechanisms of all it afferent Projections being learned.
                    if processing_mech.systems[self] == INTERNAL:
                        _assign_error_signal_projections(processing_mech, self)

            # If sender_mech has no Projections left, raise exception
            if not any(any(projection for projection in input_port.path_afferents)
                       for input_port in sender_mech.input_ports):
                raise SystemError("{} only receives Projections from other Processes or Mechanisms not"
                                  " in the current System ({})".format(sender_mech.name, self.name))

            # For all of the sender_mech's ERROR_SIGNALs and LEARNING_SIGNALs
            for output_port in sender_mech.output_ports:

                # Add them to the learning_graph
                for projection in output_port.efferents:
                    receiver = projection.receiver.owner

                    if obj_mech_replaced == INTERNAL:
                        ignore, senders = _get_learning_mechanisms(sample_mech, process)
                    else:
                        senders = [sender_mech]

                    for sender_mech in senders:
                        try:
                            # FIX: 2/10/18 IF sender_mech IS A REPLACED OBJ_MECH,
                            # FIX:         THEN SHOULD ADD THE LM THAT PROJECTS TO RECEIVER AS THE SENDER, NOT THE OBJ_MECH
                            self.learningGraph[receiver].add(sender_mech)
                        except KeyError:
                            self.learningGraph[receiver] = {sender_mech}

                        # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                        # Do not include dependency (or receiver on sender) in learning_execution_graph for this Projection
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

                        if receiver in self.learning_execution_graph:
                        # if receiver in self.learning_execution_graph_mechs:
                            # Try assigning receiver as dependent of current mechanism and test toposort
                            try:
                                # If receiver already has dependencies in its set, add sender_mech to set
                                if self.learning_execution_graph[receiver]:
                                    self.learning_execution_graph[receiver].add(sender_mech)
                                # If receiver set is empty, assign sender_mech to set
                                else:
                                    self.learning_execution_graph[receiver] = {sender_mech}
                                # Use toposort to test whether the added dependency produced a cycle (feedback loop)
                                list(toposort(self.learning_execution_graph))
                            # If making receiver dependent on sender produced a cycle, remove from learningGraph
                            except ValueError:
                                self.learning_execution_graph[receiver].remove(sender_mech)
                                receiver._add_system(self, CYCLE)
                                continue

                        else:
                            # Assign receiver as dependent on sender mechanism
                            try:
                                # FIX: THIS WILL ADD SENDER_MECH IF RECEIVER IS IN GRAPH BUT = set()
                                # FIX: DOES THAT SCREW UP ORIGINS?
                                self.learning_execution_graph[receiver].add(sender_mech)
                            except KeyError:
                                self.learning_execution_graph[receiver] = {sender_mech}

                        if not sender_mech.systems:
                            sender_mech._add_system(self, LEARNING)

                    # Traverse list of mechanisms in process recursively
                    build_dependency_sets_by_traversing_projections(receiver, process)

        # Sort for consistency of output
        sorted_processes = sorted(self.processes, key=lambda process : process.name)

        # This assumes that the first Mechanism in process.learning_mechanisms is the last in the learning sequence
        # (i.e., that the list is being traversed "backwards")
        # However, it does not assume any meaningful order for the Processes (other than alphabetical).
        for process in sorted_processes:
            if process.learning and process._learning_enabled:
                build_dependency_sets_by_traversing_projections(process.learning_mechanisms[0], process)

        # FIX: USE TOPOSORT TO FIND, OR AT LEAST CONFIRM, TARGET MECHANISMS, WHICH SHOULD EQUAL COMPARATOR MECHANISMS
        self.learning_execution_list = toposort_flatten(self.learning_execution_graph, sort=False)
        # self.learning_execution_list = self._toposort_with_ordered_mechs(self.learning_execution_graph)

        # Construct learning_mechanisms and target_nodes MechanismLists

        self._learning_mechs = []
        self._target_mechs = []

        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        for item in self.learning_execution_list:
            if isinstance(item, MappingProjection):
                continue

            # If a learning_rate has been specified for the system, assign that to all LearningMechanism
            #    for which a mechanism-specific learning_rate has NOT been assigned
            if (isinstance(item, LearningMechanism) and
                        self.learning_rate is not None and
                        item.function.learning_rate is None):
                item.function.learning_rate = self.learning_rate

            if not item in self._learning_mechs:
                self._learning_mechs.append(item)
            if isinstance(item, ObjectiveMechanism) and not item in self._target_mechs:
                self._target_mechs.append(item)
        self.learning_mechanisms = MechanismList(self, self._learning_mechs)
        self.target_mechanisms = MechanismList(self, self._target_mechs)

        # Instantiate TargetInputPorts
        self._instantiate_target_inputs(context=context)

        # Assign scheduler for _execute_learning:
        if self.scheduler_learning is None:
            self.scheduler_learning = Scheduler(graph=self.learning_execution_graph, default_execution_id=self.default_execution_id)

        # Assign conditions to scheduler_learning:
        #   insure that MappingProjections execute after all LearningMechanisms have executed in _execute_learning
        condition_set = {}
        for item in self.learning_execution_list:
            if isinstance(item, MappingProjection):
                condition_set[item] = AtPass(1)
            else:
                condition_set[item] = AtPass(0)
        self.scheduler_learning.add_condition_set(condition_set)


    def _instantiate_target_inputs(self, context=None):

        if self.learning and self.targets is None:
            # MODIFIED CW and KM 1/29/18: changed below from error to warning
            if not self.target_mechanisms:
                if self.verbosePref:
                    warnings.warn("WARNING: Learning has been specified for {} but it has no target_nodes. This "
                                  "is okay if the learning (e.g. Hebbian learning) does not need a target.".
                                  format(self.name))
                return
            # target arg was not specified in System's constructor,
            #    so use the value of the TARGET InputPort for the TARGET Mechanism(s) as the default
            self.targets = [target.input_ports[TARGET].value for target in self.target_mechanisms]
            if self.verbosePref:
                warnings.warn("Learning has been specified for {} but its \'targets\' argument was not specified;"
                              "default will be used ({})".format(self.name, self.targets))

        # Create SystemInputPort for each TARGET mechanism in target_mechanisms and
        #    assign MappingProjection from the SystemInputPort to the ORIGIN mechanism

        if isinstance(self.targets, dict):
            for target_mech in self.target_mechanisms:

                # Skip if TARGET InputPort already has a projection from a SystemInputPort in current system
                if any(self is projection.sender.owner for projection in target_mech.input_ports[TARGET].path_afferents):
                    continue

                sample_mechanism = target_mech.input_ports[SAMPLE].path_afferents[0].sender.owner
                TARGET_input_port = target_mech.input_ports[TARGET]

                if len(self.targets[sample_mechanism]) != len(TARGET_input_port.value):
                            raise SystemError("Length {} of target ({}, {}) does not match the length ({}) of the target "
                                              "expected for its TARGET Mechanism {}".
                                               format(len(self.targets[sample_mechanism]),
                                                      sample_mechanism.name,
                                                      self.targets[sample_mechanism],
                                                      len(TARGET_input_port.value),
                                                      target_mech.name))

                system_target_input_port = SystemInputPort(owner=self,
                                                        variable=TARGET_input_port.defaults.variable,
                                                        prefs=self.prefs,
                                                        name="System Target for {}".format(target_mech.name),
                                                        context=context)
                self.target_input_ports.append(system_target_input_port)

                # Add MappingProjection from system_target_input_port to TARGET mechanism's target inputPort
                from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
                proj = MappingProjection(sender=system_target_input_port,
                                         receiver=TARGET_input_port,
                                         name=self.name + ' Input Projection to ' + TARGET_input_port.name)
                proj._activate_for_compositions(self)

        elif isinstance(self.targets, list):

            # more than one target
            if len(self.target_mechanisms) > 1:
                if len(self.targets) != len(self.target_mechanisms):
                    raise SystemError("Number of target specifications provided ({}) does not match number of target "
                                      "mechanisms ({}) in {}".format(len(self.targets),
                                                                     len(self.target_mechanisms),
                                                                     self.name))

            # only one target, verify that it is wrapped in an outer list
            elif len(self.target_mechanisms) == 1:
                if len(np.shape(self.targets)) < 2:
                    self.targets = [self.targets]




            # Create SystemInputPort for each TARGET mechanism in target_nodes and
            #    assign MappingProjection from the SystemInputPort
            #    to the TARGET mechanism's TARGET inputSate
            #    (i.e., from the SystemInputPort to the ComparatorMechanism)
            for i, target_mech in zip(range(len(self.target_mechanisms)), self.target_mechanisms):

                # Create ProcessInputPort for each target and assign to targetMechanism's target inputPort
                target_mech_TARGET_input_port = target_mech.input_ports[TARGET]

                # Check, for each TARGET mechanism, that the length of the corresponding item of targets matches the length
                #    of the TARGET (ComparatorMechanism) target inputPort's defaults.variable attribute
                if len(self.targets[i]) != len(target_mech_TARGET_input_port.value):
                    raise SystemError("Length of target ({}: {}) does not match the length ({}) of the target "
                                      "expected for its TARGET Mechanism {}".
                                      format(len(self.targets[i]),
                                             self.targets[i],
                                             len(target_mech_TARGET_input_port.value),
                                             target_mech.name))

                system_target_input_port = SystemInputPort(
                    owner=self,
                    variable=target_mech_TARGET_input_port.value,
                    prefs=self.prefs,
                    name="System Target {}".format(i),
                    context=context)
                self.target_input_ports.append(system_target_input_port)

                # Add MappingProjection from system_target_input_port to TARGET mechanism's target inputPort
                from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
                proj = MappingProjection(sender=system_target_input_port,
                                  receiver=target_mech_TARGET_input_port,
                                  name=self.name + ' Input Projection to ' + target_mech_TARGET_input_port.name)
                proj._activate_for_compositions(self)

    def _assign_output_ports(self):
        """Assign OutputPorts for System (the values of which will comprise System.value)

        Assign the outputs of terminal Mechanisms in the graph to the System's output_values

        Note:
        * Current implementation simply assigns TERMINAL Mechanisms as OutputPorts
        * This method is included so that sublcasses and/or future versions can override it to make custom assignments

        """
        for mech in self.terminal_mechanisms.mechanisms:
            self.output_ports[mech.name] = mech.output_ports

    def _instantiate_controller(self, control_mech_spec, context=None):

        if control_mech_spec is None:
            return

        # Warn for request to assign the ControlMechanism already assigned
        if control_mech_spec is self.controller and self.prefs.verbosePref:
            warnings.warn("{} has already been assigned as the {} for {}; assignment ignored".
                          format(control_mech_spec, CONTROLLER, self.name))
            return

        # An existing ControlMechanism is being assigned, possibly one declared in the System's constructor
        if isinstance(control_mech_spec, ControlMechanism):
            control_mech_spec.assign_as_controller(self, context=context)
            controller = control_mech_spec

        # A ControlMechanism class or subclass is being used to specify the controller
        elif inspect.isclass(control_mech_spec) and issubclass(control_mech_spec, ControlMechanism):
            # Instantiate controller from class specification using:
            #   monitored_output_ports for System to specify its objective_mechanism (as list of OutputPorts to be monitored)
            #   ControlSignals for System returned by _get_system_control_signals()
            controller = control_mech_spec(
                    system=self,
                    objective_mechanism=self._get_monitored_output_ports_for_system(context=context),
                    control_signals=self._get_control_signals_for_system(self.control_signals_arg, context=context))
            controller._activate_projections_for_compositions(self)

        else:
            raise SystemError("Specification for {} of {} ({}) is not ControlMechanism".
                              format(CONTROLLER, self.name, control_mech_spec))

        # Warn if current one is being replaced
        if self.controller and self.prefs.verbosePref:
            warnings.warn("The existing {} for {} ({}) is being replaced by {}".
                          format(CONTROLLER, self.name, self.controller.name, controller.name))

        # Make assignment
        self._controller = controller

        # Add controller's ObjectiveMechanism to the System's execution_list and execution_graph
        self.execution_list.append(self.controller.objective_mechanism)
        self.execution_graph[self.controller.objective_mechanism] = set(self.execution_list[:-1])

        # Check whether controller has input, and if not then disable
        has_input_ports = isinstance(self.controller.input_ports, ContentAddressableList)

        if not has_input_ports:
            # If controller was enabled (and verbose is set), warn that it has been disabled
            if self.enable_controller and self.prefs.verbosePref:
                print("{} for {} has no input_ports, so controller will be disabled".
                      format(self.controller.name, self.name))
            self.enable_controller = False

        self.simulation_results = []

    def _get_monitored_output_ports_for_system(self, controller=None, context=None):
        """
        Parse a list of OutputPort specifications for System, controller, Mechanisms and/or their OutputPorts:
            - if specification in output_port is None:
                 do NOT monitor this port (this overrides any other specifications)
            - if an OutputPort is specified in *any* MONITOR_FOR_CONTROL, monitor it (this overrides any other specs)
            - if a Mechanism is terminal and/or specified in the System or `controller <Systsem.controller>`:
                if MonitoredOutputPortsOptions is PRIMARY_OUTPUT_PORTS:  monitor only its primary (first) OutputPort
                if MonitoredOutputPortsOptions is ALL_OUTPUT_PORTS:  monitor all of its OutputPorts
            Note: precedence is given to MonitoredOutputPortsOptions specification in Mechanism > controller > System

        Notes:
        * MonitoredOutputPortsOption is an AutoNumbered Enum declared in ControlMechanism
            - it specifies options for assigning OutputPorts of TERMINAL Mechanisms in the System
                to controller.monitored_output_ports;  the options are:
                + PRIMARY_OUTPUT_PORTS: assign only the `primary OutputPort <OutputPort_Primary>` for each
                  TERMINAL Mechanism
                + ALL_OUTPUT_PORTS: assign all of the outputPorts of each terminal Mechanism
            - precedence is given to MonitoredOutputPortsOptions specification in Mechanism > controller > System
        * controller.monitored_output_ports is a list, each item of which is an OutputPort from which a Projection
            will be instantiated to a corresponding InputPort of the ControlMechanism
        * controller.input_ports is the usual ordered dict of ports,
            each of which receives a Projection from a corresponding OutputPort in controller.monitored_output_ports

        Returns list of MonitoredOutputPortTuples: (OutputPort, weight, exponent, matrix)

        """
        # PARSE SPECS

        # Get OutputPorts already being -- or specified to be -- monitored by controller
        if controller is not None and not inspect.isclass(controller):
            try:
                # Get from monitored_output_ports attribute if controller is already implemented
                monitored_output_ports = controller.monitored_output_ports.copy() or []
                # Convert them to MonitoredOutputPortTuple specifications (for treatment below)
                monitored_output_port_specs = []
                for monitored_output_port, input_port in zip(monitored_output_ports,
                                                               controller.objective_mechanism.input_ports):
                    projection = input_port.path_afferents[0]
                    if not projection.sender is monitored_output_port:
                        raise SystemError("PROGRAM ERROR: Problem identifying projection ({}) for "
                                          "monitored_output_port ({}) specified for {} ({}) assigned to {}".
                                          format(projection.name,
                                                 monitored_output_port.name,
                                                 ControlMechanism.__name__,
                                                 controller.name,
                                                 self.name))
                    monitored_output_port_specs.append(MonitoredOutputPortTuple(monitored_output_port,
                                                                                  projection.weight,
                                                                                  projection.exponent,
                                                                                  projection.matrix))

                controller_specs = monitored_output_port_specs
            except AttributeError:
                # If controller has no monitored_output_ports attribute, it has not yet been fully instantiated
                #    (i.e., the call to this method is part of its instantiation by a System)
                #    so, get specification from the **object_mechanism** argument
                if isinstance(controller.objective_mechanism, list):
                    # **objective_mechanism** argument was specified as a list
                    controller_specs = controller.objective_mechanism.copy() or []
                elif isinstance(controller.objective_mechanism, ObjectiveMechanism):
                    # **objective_mechanism** argument was specified as an ObjectiveMechanism, which has presumably
                    # already been instantiated, so use its monitored_output_ports attribute
                    controller_specs = controller.objective_mechanism.monitored_output_ports
        else:
            controller_specs = []

        # Get system's MONITOR_FOR_CONTROL specifications
        system_specs = self.monitor_for_control.copy()

        # If controller_specs has a MonitoredOutputPortsOption specification, remove any such spec from system specs
        if controller_specs:
            if (any(isinstance(item, MonitoredOutputPortsOption) for item in controller_specs)):
                option_item = next((item for item in system_specs if isinstance(item,MonitoredOutputPortsOption)),None)
                if option_item is not None:
                    del system_specs[option_item]
            for item in controller_specs:
                if item in system_specs:
                    del system_specs[system_specs.index(item)]

        # Combine controller and system specs
        # If there are none, assign PRIMARY_OUTPUT_PORTS as default
        all_specs = controller_specs + system_specs or [MonitoredOutputPortsOption.PRIMARY_OUTPUT_PORTS]

        # Convert references to Mechanisms and/or OutputPorts in all_specs to MonitoredOutputPortTuples;
        # Each spec to be converted should be one of the following:
        #    - a MonitoredOutputPortsOption (parsed below);
        #    - a MonitoredOutputPortsTuple (returned by _get_monitored_ports_for_system when
        #          specs were initially processed by the System to parse its *monitor_for_control* argument;
        #    - a specification for an existing Mechanism or OutputPorts from the *monitor_for_control* arg of System.
        all_specs_extracted_from_tuples=[]
        all_specs_parsed=[]
        for i, spec in enumerate(all_specs):

            # Leave MonitoredOutputPortsOption and MonitoredOutputPortsTuple spec in place;
            #    these are parsed later on
            if isinstance(spec, MonitoredOutputPortsOption):
                all_specs_extracted_from_tuples.append(spec)
                all_specs_parsed.append(spec)
                continue
            if isinstance(spec, MonitoredOutputPortTuple):
                all_specs_extracted_from_tuples.append(spec.output_port)
                all_specs_parsed.append(spec)
                continue

            # spec is from *monitor_for_control* arg, so convert/parse into MonitoredOutputPortTuple(s)
            # Note:  assign parsed spec(s) to a list, as there may be more than one (that will be added to all_specs)
            monitored_output_port_tuples = []

            weight=DEFAULT_MONITORED_PORT_WEIGHT
            exponent=DEFAULT_MONITORED_PORT_EXPONENT
            matrix=DEFAULT_MONITORED_PORT_MATRIX

            # spec is a tuple
            # - put OutputPort(s) in spec
            # - assign any weight, exponent, and/or matrix specified
            if isinstance(spec, tuple):
                # 2-item tuple (<OutputPort(s) name(s)>, <Mechanism>)
                if len(spec) == 2:
                    # FIX: DO ERROR CHECK ON THE FOLLOWING / ALLOW LIST OF PORTS
                    spec = spec[1].output_ports[spec[0]]
                # 3-item tuple (<OutputPort(s) spec>, weight, exponent)
                elif len(spec) == 3:
                    spec, weight, exponent = spec
                # 4-item tuple (<OutputPort(s) spec>, weight, exponent, matrix)
                elif len(spec) == 4:
                    spec, weight, exponent, matrix = spec

            if not isinstance(spec, list):
                spec_list = [spec]

            for spec in spec_list:
                # spec is an OutputPort or Mechanism
                if isinstance(spec, (OutputPort, Mechanism)):
                    # spec is an OutputPort, so use it
                    if isinstance(spec, OutputPort):
                        output_ports = [spec]
                    # spec is Mechanism, so use the Port's owner, and get the relevant OutputPort(s)
                    elif isinstance(spec, Mechanism):
                        if (
                            hasattr(spec, MONITOR_FOR_CONTROL)
                            and spec.monitor_for_control is MonitoredOutputPortsOption.ALL_OUTPUT_PORTS
                        ):
                            output_ports = spec.output_ports
                        else:
                            output_ports = [spec.output_port]
                    for output_port in output_ports:
                        monitored_output_port_tuples.extend(
                                [MonitoredOutputPortTuple(output_port=output_port,
                                                           weight=weight,
                                                           exponent=exponent,
                                                           matrix=matrix)])
                # spec is a string
                elif isinstance(spec, str):
                    # Search System for Mechanisms with OutputPorts with the string as their name
                    for mech in self.mechanisms:
                        for output_port in mech.output_ports:
                            if output_port.name == spec:
                                monitored_output_port_tuples.extend(
                                        [MonitoredOutputPortTuple(output_port=output_port,
                                                                   weight=weight,
                                                                   exponent=exponent,
                                                                   matrix=matrix)])

                else:
                    raise SystemError("Specification of item in \'{}\' arg in constructor for {} ({}) "
                                      "is not a recognized specification for an {}".
                                      format(MONITOR_FOR_CONTROL, self.name, spec, OutputPort.__name__))

                all_specs_parsed.extend(monitored_output_port_tuples)
                all_specs_extracted_from_tuples.extend([item.output_port for item in monitored_output_port_tuples])

        all_specs = all_specs_parsed

        try:
            all (isinstance(item, (OutputPort, MonitoredOutputPortsOption))
                 for item in all_specs_extracted_from_tuples)
        except:
            raise SystemError("PROGRAM ERROR: Fail to parse items of \'{}\' arg ({}) in constructor for {}".
                              format(MONITOR_FOR_CONTROL, self.name, spec, OutputPort.__name__))

        # Get MonitoredOutputPortsOptions if specified for controller or System, and make sure there is only one:
        option_specs = [item for item in all_specs_extracted_from_tuples
                        if isinstance(item, MonitoredOutputPortsOption)]
        if not option_specs:
            ctlr_or_sys_option_spec = None
        elif len(option_specs) == 1:
            ctlr_or_sys_option_spec = option_specs[0]
        else:
            raise SystemError("PROGRAM ERROR: More than one MonitoredOutputPortsOption specified "
                              "for OutputPorts to be monitored in {}: {}".
                           format(self.name, option_specs))

        # Get MONITOR_FOR_CONTROL specifications for each Mechanism and OutputPort in the System
        # Assign OutputPorts to monitored_output_ports
        monitored_output_ports = []

        # Notes:
        # * Use all_specs to accumulate specs from all mechanisms and their outputPorts
        #     (for use in generating exponents and weights below)
        # * Use local_specs to combine *only current* Mechanism's specs with those from controller and system specs;
        #     this allows the specs for each Mechanism and its OutputPorts to be evaluated independently of any others
        controller_and_system_specs = all_specs_extracted_from_tuples.copy()

        for mech in self.mechanisms:

            # For each Mechanism:
            # - add its specifications to all_specs (for use below in generating exponents and weights)
            # - extract references to Mechanisms and outputPorts from any tuples, and add specs to local_specs
            # - assign MonitoredOutputPortsOptions (if any) to option_spec, (overrides one from controller or system)
            # - use local_specs (which now has this Mechanism's specs with those from controller and system specs)
            #     to assign outputPorts to monitored_output_ports

            local_specs = controller_and_system_specs.copy()
            option_spec = ctlr_or_sys_option_spec

            # PARSE MECHANISM'S SPECS

            # Get MONITOR_FOR_CONTROL specification from Mechanism
            try:
                mech_specs = mech.monitor_for_control

                if mech_specs is NotImplemented:
                    raise AttributeError

                # Setting MONITOR_FOR_CONTROL to None specifies Mechanism's OutputPort(s) should NOT be monitored
                if mech_specs is None:
                    raise ValueError

            # Mechanism's MONITOR_FOR_CONTROL is absent or NotImplemented, so proceed to parse OutputPort(s) specs
            except (KeyError, AttributeError):
                pass

            # Mechanism's MONITOR_FOR_CONTROL is set to None, so do NOT monitor any of its outputPorts
            except ValueError:
                continue

            # Parse specs in Mechanism's MONITOR_FOR_CONTROL
            else:

                # Add mech_specs to all_specs
                all_specs.extend(mech_specs)

                # Extract refs from tuples and add to local_specs
                for item in mech_specs:
                    if isinstance(item, tuple):
                        local_specs.append(item[OUTPUT_PORT_INDEX])
                        continue
                    local_specs.append(item)

                # Get MonitoredOutputPortsOptions if specified for Mechanism, and make sure there is only one:
                #    if there is one, use it in place of any specified for controller or system
                option_specs = [item for item in mech_specs if isinstance(item, MonitoredOutputPortsOption)]
                if not option_specs:
                    option_spec = ctlr_or_sys_option_spec
                elif option_specs and len(option_specs) == 1:
                    option_spec = option_specs[0]
                else:
                    raise SystemError("PROGRAM ERROR: More than one MonitoredOutputPortsOption specified in {}: {}".
                                   format(mech.name, option_specs))

            # PARSE OutputPort'S SPECS

            for output_port in mech.output_ports:

                # Get MONITOR_FOR_CONTROL specification from OutputPort
                try:
                    output_port_specs = output_port.monitor_for_control
                    if output_port_specs is NotImplemented:
                        raise AttributeError

                    # Setting MONITOR_FOR_CONTROL to None specifies OutputPort should NOT be monitored
                    if output_port_specs is None:
                        raise ValueError

                # OutputPort's MONITOR_FOR_CONTROL is absent or NotImplemented, so ignore
                except (KeyError, AttributeError):
                    pass

                # OutputPort's MONITOR_FOR_CONTROL is set to None, so do NOT monitor it
                except ValueError:
                    continue

                # Parse specs in OutputPort's MONITOR_FOR_CONTROL
                else:

                    # Note: no need to look for MonitoredOutputPortsOption as it has no meaning
                    #       as a specification for an OutputPort

                    # Add OutputPort specs to all_specs and local_specs
                    all_specs.extend(output_port_specs)

                    # Extract refs from tuples and add to local_specs
                    for item in output_port_specs:
                        if isinstance(item, tuple):
                            local_specs.append(item[OUTPUT_PORT_INDEX])
                            continue
                        local_specs.append(item)

            # Ignore MonitoredOutputPortsOption if any outputPorts are explicitly specified for the Mechanism
            for output_port in mech.output_ports:
                if (output_port in local_specs or output_port.name in local_specs):
                    option_spec = None


            # ASSIGN SPECIFIED OUTPUTPORTS FOR MECHANISM TO monitored_output_ports

            for output_port in mech.output_ports:

                # If OutputPort is named or referenced anywhere, include it
                if (output_port in local_specs or output_port.name in local_specs):
                    monitored_output_ports.append(output_port)
                    continue

    # FIX: NEED TO DEAL WITH SITUATION IN WHICH MonitoredOutputPortsOptions IS SPECIFIED, BUT MECHANISM IS NEITHER IN
    # THE LIST NOR IS IT A TERMINAL MECHANISM

                # If:
                #   Mechanism is named or referenced in any specification
                #   or a MonitoredOutputPortsOptions value is in local_specs (i.e., was specified for a Mechanism)
                #   or it is a terminal Mechanism
                elif (mech.name in local_specs or mech in local_specs or
                              any(isinstance(spec, MonitoredOutputPortsOption) for spec in local_specs) or
                              mech in self.terminal_mechanisms.mechanisms):
                    #
                    if (not (mech.name in local_specs or mech in local_specs) and
                            not mech in self.terminal_mechanisms.mechanisms):
                        continue

                    # If MonitoredOutputPortsOption is PRIMARY_OUTPUT_PORTS and OutputPort is primary, include it
                    if option_spec is MonitoredOutputPortsOption.PRIMARY_OUTPUT_PORTS:
                        if output_port is mech.output_port:
                            monitored_output_ports.append(output_port)
                            continue
                    # If MonitoredOutputPortsOption is ALL_OUTPUT_PORTS, include it
                    elif option_spec is MonitoredOutputPortsOption.ALL_OUTPUT_PORTS:
                        monitored_output_ports.append(output_port)
                    elif mech.name in local_specs or mech in local_specs:
                        if output_port is mech.output_port:
                            monitored_output_ports.append(output_port)
                            continue
                    elif option_spec is None:
                        continue
                    else:
                        raise SystemError("PROGRAM ERROR: unrecognized specification of MONITOR_FOR_CONTROL for "
                                       "{0} of {1}".
                                       format(output_port.name, mech.name))


        # ASSIGN EXPONENTS, WEIGHTS and MATRICES

        # Get and assign specification of weights, exponents and matrices
        #    for Mechanisms or OutputPorts specified in tuples
        output_port_tuples = [MonitoredOutputPortTuple(output_port=item, weight=None, exponent=None, matrix=None)
                               for item in monitored_output_ports]
        for spec in all_specs:
            if isinstance(spec, MonitoredOutputPortTuple):
                object_spec = spec.output_port
                # For each OutputPort in monitored_output_ports
                for i, output_port_tuple in enumerate(output_port_tuples):
                    output_port = output_port_tuple.output_port
                    # If either that OutputPort or its owner is the object specified in the tuple
                    if (output_port is object_spec
                        or output_port.name is object_spec
                        or output_port.owner is object_spec):
                        # Assign the weight, exponent and matrix specified in the spec to the output_port_tuple
                        # (can't just assign spec, as its output_port entry may be an unparsed string rather than
                        #  an actual OutputPort)
                        output_port_tuples[i] = MonitoredOutputPortTuple(output_port=output_port,
                                                                           weight=spec.weight,
                                                                           exponent=spec.exponent,
                                                                           matrix=spec.matrix)
        return output_port_tuples

    def _validate_monitored_ports_in_system(self, monitored_ports, context=None):
        for spec in monitored_ports:
            # if not any((spec is mech.name or spec in mech.output_ports.names)
            if not any((spec in {mech, mech.name} or spec in mech.output_ports or spec in mech.output_ports.names)
                       for mech in self.mechanisms):
                if isinstance(spec, OutputPort):
                    spec_str = "{} {} of {}".format(spec.name, OutputPort.__name__, spec.owner.name)
                else:
                    spec_str = spec
                raise SystemError("Specification of {} arg for {} appears to be a list of "
                                            "Mechanisms and/or OutputPorts to be monitored, but one "
                                            "of them ({}) is in a different System".
                                            format(OBJECTIVE_MECHANISM, self.name, spec_str))

    def _get_control_signals_for_system(self, control_signals=None, context=None):
        """Generate and return a list of control_signal_specs for System

        Generate list from:
           ControlSignal specifications passed in from the **control_signals** argument.
           ParameterPorts of the System's Mechanisms that have been assigned ControlProjections with deferred_init();
               Note: this includes any for which a ControlSignal rather than a ControlProjection
                     was used to specify control for a parameter (e.g., in a 2-item tuple specification for the
                     parameter); the initialization of the ControlProjection and, if specified, the ControlSignal
                     are completed in the call to _instantiate_control_signal() by the ControlMechanism.
        """
        control_signal_specs = control_signals or []
        for mech in self.mechanisms:
            for parameter_port in mech._parameter_ports:
                for projection in parameter_port.mod_afferents:
                    # If Projection was deferred for init, instantiate its ControlSignal and then initialize it
                    if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                        try:
                            proj_control_signal_specs = projection._init_args['control_signal_params'] or {}
                        except (KeyError, TypeError):
                            proj_control_signal_specs = {}
                        proj_control_signal_specs.update({PROJECTIONS: [projection]})
                        control_signal_specs.append(proj_control_signal_specs)
        return control_signal_specs

    def _validate_control_signals(self, control_signals, context=None):
        if control_signals:
            for control_signal in control_signals:
                for control_projection in control_signal.efferents:
                    if not any(control_projection.receiver in mech.parameter_ports for mech in self.mechanisms):
                        raise SystemError("A parameter controlled by a ControlSignal of a controller "
                                          "being assigned to {} is not in that System".format(self.name))

    def _add_mechanism_conditions(self, context=None):

        condition_set = {}
        for item in self.execution_list:
            if hasattr(item, CONDITION) and item.condition and not item in self.scheduler.conditions:
                condition_set[item] = item.condition
        self.scheduler.add_condition_set(condition_set)

        # FIX: DEAL WITH LEARNING PROJECTIONS HERE (ADD CONDITIONS ATTRIBUTE?)
        condition_set = {}
        for item in self.learning_execution_list:
            if hasattr(item, CONDITION) and item.condition and not item in self.scheduler_learning.conditions:
                condition_set[item] = item.condition
        self.scheduler_learning.add_condition_set(condition_set)

    def _parse_runtime_params(self, runtime_params):
        if runtime_params is None:
            return {}
        for mechanism in runtime_params:
            for param in runtime_params[mechanism]:
                if isinstance(runtime_params[mechanism][param], tuple):
                    if len(runtime_params[mechanism][param]) == 1:
                        runtime_params[mechanism][param] = (runtime_params[mechanism][param], Always())
                    elif len(runtime_params[mechanism][param]) != 2:
                        raise SystemError("Invalid runtime parameter specification ({}) for {}'s {} parameter in {}. "
                                          "Must be a tuple of the form (parameter value, condition), or simply the "
                                          "parameter value. ".format(runtime_params[mechanism][param],
                                                                     mechanism.name,
                                                                     param,
                                                                     self.name))
                else:
                    runtime_params[mechanism][param] = (runtime_params[mechanism][param], Always())
        return runtime_params

    def initialize(self, context=None):
        """Assign `initial_values <System.initialize>` to mechanisms designated as `INITIALIZE_CYCLE` \and
        contained in recurrent_init_mechanisms.
        """
        # FIX:  INITIALIZE PROCESS INPUT??
        # FIX: CHECK THAT ALL MECHANISMS ARE INITIALIZED FOR WHICH mech.system[SELF]==INITIALIZE
        # FIX: ADD OPTION THAT IMPLEMENTS/ENFORCES INITIALIZATION
        # FIX: ADD SOFT_CLAMP AND HARD_CLAMP OPTIONS
        # FIX: ONLY ASSIGN ONES THAT RECEIVE PROJECTIONS
        for mech, value in self.initial_values.items():
            mech.initialize(value, context=context)

    # execute previously always set source to ContextFlags.COMPOSITION
    # @handle_external_context(source=ContextFlags.COMPOSITION, execution_phase=ContextFlags.PROCESSING)
    def execute(self,
                input=None,
                target=None,
                context=None,
                termination_processing=None,
                termination_learning=None,
                runtime_params=None,
                ):
        """Execute mechanisms in System in order specified by the `execution_graph <System.execution_graph>` attribute.

        Assign items of input to `ORIGIN` mechanisms

        Execute any learning components specified at the appropriate phase.

        Execute controller after all other Mechanisms have been executed.

        .. Execution:
            - the input arg in System.execute() or run() is provided as input to ORIGIN mechanisms (and
              System.input);
                As with a process, `ORIGIN` Mechanisms will receive their input only once (first execution)
                    unless clamp_input (or SOFT_CLAMP or HARD_CLAMP) are specified, in which case they will continue to
            - execute() calls Mechanism.execute() for each Mechanism in its execute_graph in sequence
            - outputs of `TERMINAL` Mechanisms are assigned as System.ouputValue
            - System.controller is executed after execution of all Mechanisms in the System
            - notes:
                * the same Mechanism can be listed more than once in a System, inducing recurrent processing

        Arguments
        ---------
        input : list or ndarray
            a list or array of input value arrays, one for each `ORIGIN` Mechanism in the System.

        termination_processing : Dict[TimeScale: Condition]
            a dictionary containing `Condition`\\ s that signal the end of the associated `TimeScale` within the :ref:`processing
            phase of execution <System_Execution_Processing>`

        termination_learning : Dict[TimeScale: Condition]
            a dictionary containing `Condition`\\ s that signal the end of the associated `TimeScale` within the :ref:`learning
            phase of execution <System_Execution_Learning>`

            .. context : str

        Returns
        -------
        output values of System : 3d ndarray
            Each item is a 2d array that contains arrays for each OutputPort.value of each `TERMINAL` Mechanism

        """

        if self.scheduler is None:
            self.scheduler = Scheduler(system=self, default_execution_id=self.default_execution_id)

        if self.scheduler_learning is None:
            self.scheduler_learning = Scheduler(graph=self.learning_execution_graph, default_execution_id=self.default_execution_id)

        self._add_mechanism_conditions(context=context)

        runtime_params = self._parse_runtime_params(runtime_params)

        if not hasattr(self, '_animate'):
            # These are meant to be assigned in run method;  needed here for direct call to execute method
            self._animate = False
            self._component_execution_count = 0

        if not context:
            context = Context(
                source=ContextFlags.COMPOSITION,
                execution_phase=ContextFlags.PROCESSING,
                composition=self
            )

        context.composition = self

        self._report_system_output = (
            self.prefs.reportOutputPref
            and context.execution_phase & (ContextFlags.PROCESSING | ContextFlags.LEARNING)
        )

        if self._report_system_output:
            self._report_process_output = any(process.reportOutputPref for process in self.processes)

        # FIX: MOVE TO RUN??
        # ASSIGN INPUTS TO SystemInputPorts
        #    that will be used as the input to the MappingProjection to each ORIGIN mechanism
        num_origin_mechs = len(list(self.origin_mechanisms))

        if input is None:
            if (
                self.prefs.verbosePref
                and not (
                    context.source == ContextFlags.COMMAND_LINE
                    or self.initialization_status == ContextFlags.INITIALIZING
                )
            ):
                print("- No input provided;  default will be used: {0}")
            input = np.zeros_like(self.defaults.variable)
            for i in range(num_origin_mechs):
                input[i] = self.origin_mechanisms[i].defaults.variable

        else:
            num_inputs = len(input)
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

            # Get SystemInputPort that projects to each ORIGIN Mechanism and assign input to it
            for origin_mech in self.origin_mechanisms:
                # For each inputPort of the ORIGIN mechanism

                for j in range(len(origin_mech.external_input_ports)):
                   # Get the input from each Projection to that InputPort (from the corresponding SystemInputPort)
                    system_input_port = next((projection.sender
                                               for projection in origin_mech.input_ports[j].path_afferents
                                               if isinstance(projection.sender, SystemInputPort)), None)

                    if system_input_port:
                        if isinstance(input, dict):
                            system_input_port.parameters.value._set(input[origin_mech][j], context)

                        else:
                            system_input_port.parameters.value._set(input[j], context)
                    else:
                        logger.warning("Failed to find expected SystemInputPort "
                                       "for {} at InputPort number ({}), ({})".
                              format(origin_mech.name, j + 1, origin_mech.input_ports[j]))
                        # raise SystemError("Failed to find expected SystemInputPort for {}".format(origin_mech.name))

        self.input = input

        # termination_processing should be treated like a runtime param -- if nothing is passed in, then use the attr
        if termination_processing is None:

            termination_processing = self.termination_processing

        self.termination_learning = termination_learning

        if self._report_system_output:
            self._report_system_initiation(context)


        # Generate first frame of animation without any active_items
        if self._animate is not False:
            # if this fails, the scheduler has no data for context.execution_id yet. It also may be the first, so fall back to default execution_id
            try:
                self.show_graph(active_items=INITIAL_FRAME, **self._animate, output_fmt='gif', context=context)
            except KeyError:
                old_eid = context.execution_id
                context.execution_id = self.default_execution_id
                self.show_graph(active_items=INITIAL_FRAME, **self._animate, output_fmt='gif', context=context)
                context.execution_id = old_eid

        # EXECUTE MECHANISMS

        # TEST PRINT:
        # for i in range(len(self.execution_list)):
        #     print(self.execution_list[i][0].name)
        # sorted_list = list(object_item[0].name for object_item in self.execution_list)

        # Execute system without learning on projections (that will be taken care of in _execute_learning()
        context.add_flag(ContextFlags.PROCESSING)
        self._execute_processing(
            context=context,
            runtime_params=runtime_params,
            termination_processing=termination_processing,

        )
        context.remove_flag(ContextFlags.PROCESSING)

        outcome = self.terminal_mechanisms.get_output_port_values(context)

        if self.recordSimulationPref and ContextFlags.SIMULATION in context.execution_phase:
            self.simulation_results.append(outcome)

        # EXECUTE LEARNING FOR EACH PROCESS

        # Execute learning except for simulation runs
        if ContextFlags.SIMULATION not in context.execution_phase and self.learning:
            context.add_flag(ContextFlags.LEARNING)
            context.add_flag(ContextFlags.LEARNING_MODE)
            self._execute_learning(target=target, context=context)
            context.remove_flag(ContextFlags.LEARNING)
            context.add_flag(ContextFlags.LEARNING_MODE)

        # EXECUTE CONTROLLER
        # FIX: 1) RETRY APPENDING TO EXECUTE LIST AND COMPARING TO THIS VERSION
        # FIX: 2) REASSIGN INPUT TO SYSTEM FROM ONE DESIGNATED FOR EVC SIMULUS (E.G., StimulusPrediction)

        # Only call controller if this is not a controller simulation run (to avoid infinite recursion)
        if ContextFlags.SIMULATION not in context.execution_phase and self.enable_controller:
            context.add_flag(ContextFlags.CONTROL)
            try:
                self.controller.execute(
                    context=context,
                    runtime_params=None,

                )

                if self._animate != False and SHOW_CONTROL in self._animate and self._animate[SHOW_CONTROL]:
                    self.show_graph(active_items=self.controller, **self._animate, output_fmt='gif', context=context)
                self._component_execution_count += 1

                if self._report_system_output:
                    print("{0}: {1} executed".format(self.name, self.controller.name))

            except AttributeError as error_msg:
                if self.initialization_status != ContextFlags.INITIALIZING:
                    error_msg.args += ("PROGRAM ERROR: Problem executing controller ({}) for {}".format(self.controller.name, self.name),)
                    raise

            context.remove_flag(ContextFlags.CONTROL)

        # Report completion of system execution and value of designated outputs
        if self._report_system_output:
            self._report_system_completion(context)

        return outcome

    def _execute_processing(self, runtime_params, termination_processing, context=None):
        # Execute each Mechanism in self.execution_list, in the order listed during its phase
        # Only update Mechanism on time_step(s) determined by its phaseSpec (specified in Mechanism's Process entry)
        # FIX: NEED TO IMPLEMENT FRACTIONAL UPDATES (IN Mechanism.update())
        # FIX:    FOR phaseSpec VALUES THAT HAVE A DECIMAL COMPONENT
        if self.scheduler is None:
            raise SystemError('System.py:_execute_processing - {0}\'s scheduler is None, '
                              'must be initialized before execution'.format(self.name))
        logger.debug('{0}.scheduler processing termination conditions: {1}'.format(self, termination_processing))

        for next_execution_set in self.scheduler.run(context=context, termination_conds=termination_processing):
            logger.debug('Running next_execution_set {0}'.format(next_execution_set))
            i = 0

            if not self._animate is False and self._animate_unit is EXECUTION_SET:
                self.show_graph(active_items=next_execution_set, **self._animate, output_fmt='gif', context=context)

            for mechanism in next_execution_set:
                logger.debug('\tRunning Mechanism {0}'.format(mechanism))

                processes = list(mechanism.processes.keys())
                process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                process_names = list(p.name for p in process_keys_sorted)

                # Set up runtime params and context
                execution_runtime_params = {}
                if mechanism in runtime_params:
                    for param in runtime_params[mechanism]:
                        if runtime_params[mechanism][param][1].is_satisfied(scheduler=self.scheduler, context=context):
                            execution_runtime_params[param] = runtime_params[mechanism][param][0]

                # FIX: DO THIS LOCALLY IN LearningMechanism?? IF SO, NEEDS TO BE ABLE TO GET EXECUTION_ID
                if (isinstance(mechanism, LearningMechanism) and
                        mechanism.learning_timing is LearningTiming.EXECUTION_PHASE):
                    context.execution_phase = ContextFlags.LEARNING
                # Execute
                # # TEST PRINT:
                # print("\nEXECUTING System._execute_processing\n")
                mechanism.execute(context=context, runtime_params=execution_runtime_params)

                if not self._animate is False and self._animate_unit is COMPONENT:
                    self.show_graph(active_items=mechanism, **self._animate, output_fmt='gif', context=context)
                self._component_execution_count += 1

                # Reset runtime params and context
                if context.execution_id in mechanism._runtime_params_reset:
                    for key in mechanism._runtime_params_reset[context.execution_id]:
                        mechanism._set_parameter_value(key, mechanism._runtime_params_reset[context.execution_id][key], context)
                mechanism._runtime_params_reset[context.execution_id] = {}

                if context.execution_id in mechanism.function._runtime_params_reset:
                    for key in mechanism.function._runtime_params_reset[context.execution_id]:
                        mechanism.function._set_parameter_value(key, mechanism.function._runtime_params_reset[context.execution_id][key], context)
                mechanism.function._runtime_params_reset[context.execution_id] = {}

                if self._report_system_output and  self._report_process_output:

                    # REPORT COMPLETION OF PROCESS IF ORIGIN:
                    # Report initiation of process(es) for which mechanism is an ORIGIN
                    # Sort for consistency of reporting:
                    processes = list(mechanism.processes.keys())
                    process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                    for process in process_keys_sorted:
                        if mechanism.processes[process] in {ORIGIN, SINGLETON} and process.reportOutputPref:
                            process._report_process_initiation(input=mechanism.get_input_values(context)[0])

                    # REPORT COMPLETION OF PROCESS IF TERMINAL:
                    # Report completion of process(es) for which mechanism is a TERMINAL
                    # Sort for consistency of reporting:
                    processes = list(mechanism.processes.keys())
                    process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                    for process in process_keys_sorted:
                        if process.learning and process._learning_enabled:
                            continue
                        if mechanism.processes[process] == TERMINAL and process.reportOutputPref:
                            process._report_process_completion()

                # # TEST PRINT 7/22/19
                # print(f'Executed {mechanism.name}: \n\tvariable: {mechanism.variable}\n\tvalue: {mechanism.value}')

            if i == 0:
                # Zero input to first mechanism after first run (in case it is repeated in the pathway)
                # IMPLEMENTATION NOTE:  in future version, add option to allow Process to continue to provide input
                # FIX: USE clamp_input OPTION HERE, AND ADD HARD_CLAMP AND SOFT_CLAMP
                pass
            i += 1

    def _execute_learning(self, target=None, context=None):
        # Execute each LearningMechanism as well as LearningProjections in self.learning_execution_list

        # FIRST, if targets were specified as a function, call the function now
        #    (i.e., after execution of the pathways, but before learning)
        # Note:  this accomodates functions that predicate the target on the outcome of processing
        #        (e.g., for rewards in reinforcement learning)
        if not hasattr(self, "target"):
            self.target = self.targets
        if target is None or len(target) == 0:
            target = self.target
        if isinstance(target, dict):
            for i in range(len(self.target_mechanisms)):

                terminal_mechanism = self.target_mechanisms[i].input_ports[SAMPLE].path_afferents[0].sender.owner
                target_value = target[terminal_mechanism]
                if callable(target_value):
                    val = call_with_pruned_args(target_value, context=context, composition=self)
                    self.target_input_ports[i].parameters.value._set(val, context)
                else:
                    self.target_input_ports[i].parameters.value._set(target_value, context)

        elif isinstance(target, (list, np.ndarray)):
            for i in range(len(self.target_mechanisms)):
                self.target_input_ports[i].parameters.value._set(target[i], context)

        # THEN, execute all components involved in learning
        if self.scheduler_learning is None:
            raise SystemError('System.py:_execute_learning - {0}\'s scheduler is None, '
                              'must be initialized before execution'.format(self.name))
        logger.debug('{0}.scheduler learning termination conditions: {1}'.format(self, self.termination_learning))

        for next_execution_set in self.scheduler_learning.run(context=context, termination_conds=self.termination_learning):
            logger.debug('Running next_execution_set {0}'.format(next_execution_set))

            if (not self._animate is False and
                    self._animate_unit is EXECUTION_SET and
                    SHOW_LEARNING in self._animate and self._animate[SHOW_LEARNING]):
                # mech = [mech for mech in next_execution_set if isinstance(mech, Mechanism)]
                self.show_graph(active_items=list(next_execution_set), **self._animate, output_fmt='gif', context=context)

            for component in next_execution_set:
                logger.debug('\tRunning component {0}'.format(component))
                context.add_flag(ContextFlags.LEARNING_MODE)
                context.execution_phase = ContextFlags.LEARNING
                
                if isinstance(component, Mechanism):
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
                                             re.sub(r'[\[,\],\n]','',str(process_names))))
                    # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
                    component.execute(context=context, runtime_params=params)

                    # # TEST PRINT 7/22/19
                    # print(f'Executed {component.name}: \n\tvariable: {component.variable}\n\tvalue: {component.value}')

                elif isinstance(component, MappingProjection):
                    processes = list(component.sender.owner.processes.keys())
                    component._parameter_ports[MATRIX]._update(context=context)

                if not self._animate is False:
                    if (self._animate_unit is COMPONENT and
                            SHOW_LEARNING in self._animate and self._animate[SHOW_LEARNING]):
                            self.show_graph(active_items=component, **self._animate, output_fmt='gif', context=context)

                self._component_execution_count += 1

                # # TEST PRINT LEARNING:
                # print ("EXECUTING LEARNING UPDATES: ", component.name)

                # # TEST PRINT LEARNING:
                # print(component._parameter_ports[MATRIX].value)
                context.remove_flag(ContextFlags.LEARNING_MODE)
        
        # FINALLY report outputs
        if self._report_system_output and self._report_process_output:
            # Report learning for target_nodes (and the processes to which they belong)
            # Sort for consistency of reporting:
            print("\n\'{}' learning completed:".format(self.name))

            for target_mech in self.target_mechanisms:
                processes = list(target_mech.processes.keys())
                process_keys_sorted = sorted(processes, key=lambda i : processes[processes.index(i)].name)
                process_names = list(p.name for p in process_keys_sorted)
                # print("\n\'- Target: {}' error: {} (processes: {})".
                print("- error for target ({}): {}".
                      # format(append_type_to_name(target_mech),
                      format(target_mech.name,
                             re.sub(r'[\[,\],\n]','',str([float("{:0.3}".format(float(i)))
                                                         for i in target_mech.output_port.parameters.value._get(context)])),
                             ))
                             # process_names))

    # correct here? happens in code but maybe should be COMMAND_LINE
    @handle_external_context(source=ContextFlags.COMPOSITION)
    def run(self,
            inputs=None,
            num_trials=None,
            initialize=False,
            initial_values=None,
            targets=None,
            learning=None,
            call_before_trial=None,
            call_after_trial=None,
            call_before_time_step=None,
            call_after_time_step=None,
            termination_processing=None,
            termination_learning=None,
            runtime_params=None,
            reinitialize_values=None,
            context=None,
            base_context=Context(execution_id=None),
            animate=False,
            ):
        """Run a sequence of executions

        Call execute method for each execution in a sequence specified by inputs.  See :doc:`Run` for details of
        formatting input specifications. The **animate** argument can be used to generate a movie of the execution of
        the System.

        .. note::
           Use of the animation argument relies on `imageio <http://imageio.github.io>`_, which must be
           installed and imported (standard with PsyNeuLink pip install)

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_variable for a single execution
            the input for each in a sequence of executions (see :doc:`Run` for detailed description of formatting
            requirements and options).

        initialize : bool default :keyword:`False`
            if `True`, calls the :py:meth:`initialize <System.initialize>` method of the System before a
            sequence of executions.

        initial_values : Dict[Mechanism: List[input] or np.ndarray(input)] : default None
            the initial values assigned to Mechanisms designated as `INITIALIZE_CYCLE`.

        targets : List[input] or np.ndarray(input) : default `None`
            the target values for the LearningMechanisms of the System for each execution.
            The length (of the outermost level if a nested list, or lowest axis if an ndarray) must be equal to that
            of ``inputs``.

        learning : bool :  default `None`
            enables or disables learning during execution.
            If it is not specified, the current state is left intact.
            If it is `True`, learning is forced on; if it is :keyword:`False`, learning is forced off.

        call_before_trial : Function : default `None`
            called before each trial in the sequence is executed.

        call_after_trial : Function : default `None`
            called after each trial in the sequence is executed.

        call_before_time_step : Function : default `None`
            called before each time_step of each trial is executed.

        call_after_time_step : Function : default `None`
            called after each time_step of each trial is executed.

        termination_processing : Dict[TimeScale: Condition]
            a dictionary containing `Condition`\\ s that signal the end of the associated `TimeScale` within the :ref:`processing
            phase of execution <System_Execution_Processing>`

        termination_learning : Dict[TimeScale: Condition]
            a dictionary containing `Condition`\\ s that signal the end of the associated `TimeScale` within the :ref:`learning
            phase of execution <System_Execution_Learning>`

        reinitialize_values : Dict[Mechanism: List[reinitialization values] or np.ndarray(reinitialization values)
            a dictionary containing Mechanism: value pairs. Each Mechanism in the dictionary calls its `reinitialize
            <Mechanism_Base.reinitialize>` method at the start of the Run. The Mechanism's value in the
            reinitialize_values dictionary is passed into its `reinitialize <Mechanism_Base.reinitialize>` method. See
            the `reinitialize method <IntegratorFunction.reinitialize>` of the `function <Mechanism_Base.function>`
            or `integrator_function <TransferMechanism.integrator_function>` of the Mechanism for details on which
            values must be passed in as arguments. Keep in mind that only stateful Mechanisms may be reinitialized, and
            that Mechanisms in reinitialize_values will reinitialize regardless of whether their `reinitialize_when
            <Component.reinitialize_when>` Condition is satisfied.

        animate : dict or bool : False
            specifies use of the `show_graph <System.show_graph>` method to generate a gif movie showing the sequence
            of Components executed in a trial.  A dict can be specified containing options to pass to the `show_graph
            <System.show_graph>` method;  each key must be an argument of the `show_graph <System.show_graph>`
            method, and its value a specification for that argument.  The entries listed below can also be included
            in the dict to specify parameters of the animation.  If the **animate** argument is specified simply as
            `True`, defaults are used for all arguments of `show_graph <System.show_graph>` and the options below:

            * *UNIT*: *EXECUTION_SET* or *COMPONENT* (default=\\ *EXECUTION_SET*\\ ) -- specifies which Components to
              treat as active in each call to `show_graph <System.show_graph>`. *COMPONENT* generates an image for the
              execution of each Component.  *EXECUTION_SET* generates an image for each `execution_set
              <System.execution_sets>`, showing all of the Components in that set as active.

            * *DURATION*: float (default=0.75) -- specifies the duration (in seconds) of each image in the movie.

            * *NUM_TRIALS*: int (default=1) -- specifies the number of trials to animate;  by default, this is 1.
              If the number specified is less than the total number of trials being run, only the number specified are
              animated; if it is greater than the number of trials being run, only the number being run are animated.

            * *MOVIE_NAME*: str (default=\\ `name <System.name>` + 'movie') -- specifies the name to be used for the
              movie file; it is automatically appended with '.gif'.

            * *SAVE_IMAGES*: bool (default=\\ `False`\\ ) -- specifies whether to save each of the images used to
              construct the animation in separate gif files, in addition to the file containing the animation.

            * *SHOW*: bool (default=\\ `False`\\ ) -- specifies whether to show the animation after it is constructed,
              using the OS's default viewer.

        Examples
        --------

        This figure shows an animation of the System in the Multilayer-Learning example script, with
        the show_graph **show_learning** argument specified as *ALL*:

        .. _System_Multilayer_Learning_movie:

        .. figure:: _static/Multilayer_Learning_movie.gif
           :alt: Animation of System in Multilayer-Learning example script
           :scale: 50 %

        This figure shows an animation of the System in the EVC Gratton example script, with
        the show_graph **show_control** argument specified as *ALL* and *UNIT* specified as *EXECUTION_SET*:

        .. _System_EVC_Gratton_movie:

        .. figure:: _static/EVC_Gratton_movie.gif
           :alt: Animation of System in EVC Gratton example script
           :scale: 150 %

        Returns
        -------

        <System>.results : List[Mechanism.OutputValue]
            list of the OutputValue for each `TERMINAL` Mechanism of the System returned for each execution.

        """
        if runtime_params is None:
            runtime_params = {}

        if reinitialize_values is None:
            reinitialize_values = {}

        if context.execution_id is None:
            context.execution_id = self.default_execution_id

        context.composition = self

        # initialize from base context but don't overwrite any values already set for this context
        if ContextFlags.SIMULATION not in context.execution_phase:
            self._initialize_from_context(context, base_context, override=False)

        for mechanism in reinitialize_values:
            mechanism.reinitialize(*reinitialize_values[mechanism], context=context)

        self.initial_values = initial_values

        self._component_execution_count=0

        # Set animation attributes
        if animate is True:
            animate = {}
        self._animate = animate
        if isinstance(self._animate, dict):
            # Assign directory for animation files
            here = path.abspath(path.dirname(__file__))
            self._animate_directory = path.join(here, '../../show_graph output/' + self.name + " gifs")
            # try:
            #     rmtree(self._animate_directory)
            # except:
            #     pass
            self._animate_unit = self._animate.pop(UNIT, EXECUTION_SET)
            self._image_duration = self._animate.pop(DURATION, 0.75)
            self._animate_num_trials = self._animate.pop(NUM_TRIALS, 1)
            self._movie_filename = self._animate.pop(MOVIE_NAME, self.name + ' movie') + '.gif'
            self._save_images = self._animate.pop(SAVE_IMAGES, False)
            self._show_animation = self._animate.pop(SHOW, False)
            if not self._animate_unit in {COMPONENT, EXECUTION_SET}:
                raise SystemError("{} entry of {} argument for {} method of {} ({}) must be {} or {}".
                                  format(repr(UNIT), repr('animate'), repr('run'),
                                         self.name, self._animate_unit, repr(COMPONENT), repr(EXECUTION_SET)))
            if not isinstance(self._image_duration, (int, float)):
                raise SystemError("{} entry of {} argument for {} method of {} ({}) must be an int or a float".
                                  format(repr(DURATION), repr('animate'), repr('run'),
                                         self.name, self._image_duration))
            if not isinstance(self._animate_num_trials, int):
                raise SystemError("{} entry of {} argument for {} method of {} ({}) must an integer".
                                  format(repr(NUM_TRIALS), repr('animate'), repr('run'),
                                         self.name, self._animate_num_trials, repr('show_graph')))
            if not isinstance(self._movie_filename, str):
                raise SystemError("{} entry of {} argument for {} method of {} ({}) must be a string".
                                  format(repr(MOVIE_NAME), repr('animate'), repr('run'),
                                         self.name, self._movie_filename))
            if not isinstance(self._save_images, bool):
                raise SystemError("{} entry of {} argument for {} method of {} ({}) must be {} or {}".
                                  format(repr(MOVIE_NAME), repr('animate'), repr('run'),
                                         self.name, self._save_images, repr(True), repr(False)))
            if not isinstance(self._save_images, bool):
                raise SystemError("{} entry of {} argument for {} method of {} ({}) must be {} or {}".
                                  format(repr(SHOW), repr('animate'), repr('run'),
                                         self.name, self._show_animation, repr(True), repr(False)))

        elif self._animate:
            # self._animate should now be False or a dict
            raise SystemError("{} argument for {} method of {} ({}) must boolean or "
                              "a dictionary of argument specifications for its {} method".
                              format(repr('animate'), repr('run'), self.name, self._animate, repr('show_graph')))

        logger.debug(inputs)

        context.source = ContextFlags.COMPOSITION
        from psyneulink.core.globals.environment import run
        result = run(self,
                   inputs=inputs,
                   num_trials=num_trials,
                   initialize=initialize,
                   initial_values=initial_values,
                   targets=targets,
                   learning=learning,
                   call_before_trial=call_before_trial,
                   call_after_trial=call_after_trial,
                   call_before_time_step=call_before_time_step,
                   call_after_time_step=call_after_time_step,
                   termination_processing=termination_processing,
                   termination_learning=termination_learning,
                   runtime_params=runtime_params,
                   context=context)

        if self._animate is not False:
            # Save list of gifs in self._animation as movie file
            movie_path = self._animate_directory + '/' + self._movie_filename
            self._animation[0].save(fp=movie_path,
                                    format='GIF',
                                    save_all=True,
                                    append_images=self._animation[1:],
                                    duration=self._image_duration * 1000,
                                    loop=0)
            print('\nSaved movie for {}: {}'.format(self.name, self._movie_filename))
            if self._show_animation:
                movie = Image.open(movie_path)
                movie.show()

        return result

    def _report_system_initiation(self, inputs=None, context=None):
        """Prints iniiation message, time_step, and list of Processes in System being executed
        """

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        # replace this with updated Clock
        if False:
            print("\n\'{}\'{} executing with: **** (Time: {}) ".
                  format(self.name, system_string, self.scheduler.clock.simple_time))
            processes = list(process.name for process in self.processes)
            print("- processes: {}".format(processes))
            print("self.input = ", self.input)
            if np.size(self.input) == 1:
                input_string = ''
            else:
                input_string = 's'
            print("- input{}: {}".format(input_string, self.input))

        else:
            try:
                time = self.scheduler.get_clock(context).simple_time
            except KeyError:
                time = self.scheduler.clock.simple_time

            print("\n\'{}\'{} executing ********** (Time: {}) ".
                  format(self.name, system_string, time))

    def _report_system_completion(self, context=None):
        """Prints completion message and output_values of system
        """

        if 'system' in self.name or 'System' in self.name:
            system_string = ''
        else:
            system_string = ' system'

        # Print output value of primary (first) outputPort of each terminal Mechanism in System
        # IMPLEMENTATION NOTE:  add options for what to print (primary, all or monitored outputPorts)
        print("\n\'{}\'{} completed ***********(Time: {})".format(self.name, system_string, self.scheduler.get_clock(context).simple_time))
        if self.learning:
            from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import MSE
            for mech in self.target_mechanisms:
                if not MSE in mech.output_ports:
                    continue
                print("\n- MSE: {:0.3}".
                      format(float(mech.output_ports[MSE].parameters.value._get(context))))


    # TBI:
    # class InspectOptions(AutoNumber):
    #     """Option value keywords for `inspect` and `show` methods.
    #     """
    #     ALL = ()
    #     """Show all values.
    #     """
    #     EXECUTION_SETS = ()
    #     """Show `execution_sets` attribute."""
    #     execution_list = ()
    #     """Show `execution_list` attribute."""
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
        """Print ``execution_sets``, ``execution_list``, `ORIGIN`, `TERMINAL` Mechanisms,
        `TARGET` Mechanisms, ``outputs`` and their labels for the System.

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
            sorted_mechs_names_in_set = sorted(list(object_item.name
                                                    for object_item in self.execution_sets[i]))
            for name in sorted_mechs_names_in_set:
                print("{0} ".format(name), end='')
            print("}")

        # Print execution_list sorted by phase and including EVC mechanism

        # Sort execution_list by phase
        sorted_execution_list = self.execution_list.copy()


        # Sort by phaseSpec and, within each phase, by mechanism name
        # sorted_execution_list.sort(key=lambda object_item: object_item.phase)


        # Add controller to execution list for printing if enabled
        if self.enable_controller:
            sorted_execution_list.append(self.controller)

    def inspect(self):
        """Return dictionary with system attributes and values

        Diciontary contains entries for the following attributes and values:

            PROCESSES: list of `Processes <Process>` in system;

            MECHANISMS: list of all `Mechanisms <Mechanism>` in the system;

            ORIGIN_MECHANISMS: list of `ORIGIN` Mechanisms;

            INPUT_ARRAY: ndarray of the inputs to the `ORIGIN` Mechanisms;

            RECURRENT_MECHANISMS:  list of `INITALIZE_CYCLE` Mechanisms;

            RECURRENT_INIT_ARRAY: ndarray of initial_values;

            TERMINAL_MECHANISMS: list of `TERMINAL` Mechanisms;

            OUTPUT_PORT_NAMES: list of `OutputPort` names corresponding to 1D arrays in output_value_array;

            OUTPUT_VALUE_ARRAY: 3D ndarray of 2D arrays of output.value arrays of OutputPorts for all `TERMINAL`
            Mechanisms;

            NUM_PHASES_PER_TRIAL: number of phases required to execute all Mechanisms in the system;

            LEARNING_MECHANISMS: list of `LearningMechanisms <LearningMechanism>`;

            TARGET: list of `TARGET` Mechanisms;

            LEARNING_PROJECTION_RECEIVERS: list of `MappingProjections <MappingProjection>` that receive learning
            projections;

            CONTROL_MECHANISM: `ControlMechanism <ControlMechanism>` of the System;

            CONTROL_PROJECTION_RECEIVERS: list of `ParameterPorts <ParameterPort>` that receive learning projections.

        Returns
        -------
        Dictionary of System attributes and values : dict

        """

        input_array = []
        for mech in list(self.origin_mechanisms.mechanisms):
            input_array.append(mech.value)
        input_array = np.array(input_array)

        recurrent_init_array = []
        for mech in list(self.recurrent_init_mechanisms.mechanisms):
            recurrent_init_array.append(mech.value)
        recurrent_init_array = np.array(recurrent_init_array)

        output_port_names = []
        output_value_array = []
        for mech in list(self.terminal_mechanisms.mechanisms):
            output_value_array.append(mech.output_values)
            for name in mech.output_ports:
                output_port_names.append(name)
        output_value_array = np.array(output_value_array)

        from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
        from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
        learning_projections = []
        controlled_parameters = []
        for mech in list(self.mechanisms):
            for parameter_port in mech._parameter_ports:
                try:
                    for projection in parameter_port.mod_afferents:
                        if isinstance(projection, ControlProjection):
                            controlled_parameters.append(parameter_port)
                except AttributeError:
                    pass
            for output_port in mech.output_ports:
                try:
                    for projection in output_port.efferents:
                        for parameter_port in projection.paramaterStates:
                            for sender in parameter_port.mod_afferents:
                                if isinstance(sender, LearningProjection):
                                    learning_projections.append(projection)
                except AttributeError:
                    pass

        inspect_dict = {
            PROCESSES: self.processes,
            MECHANISMS: self.mechanisms,
            ORIGIN_MECHANISMS: self.origin_mechanisms.mechanisms,
            INPUT_ARRAY: input_array,
            RECURRENT_MECHANISMS: self.recurrent_init_mechanisms,
            RECURRENT_INIT_ARRAY: recurrent_init_array,
            TERMINAL_MECHANISMS: self.terminal_mechanisms.mechanisms,
            OUTPUT_PORT_NAMES: output_port_names,
            OUTPUT_VALUE_ARRAY: output_value_array,
            NUM_PHASES_PER_TRIAL: self.numPhases,
            LEARNING_MECHANISMS: self.learning_mechanisms,
            TARGET_MECHANISMS: self.target_mechanisms,
            LEARNING_PROJECTION_RECEIVERS: learning_projections,
            CONTROL_MECHANISM: self.control_mechanisms,
            CONTROL_PROJECTION_RECEIVERS: controlled_parameters,
        }

        return inspect_dict

    def _toposort_with_ordered_mechs(self, data):
        """Returns a single list of dependencies, sorted by object_item[MECHANISM].name"""
        result = []
        for dependency_set in toposort(data):
            d_iter = iter(dependency_set)
            result.extend(sorted(dependency_set, key=lambda item : next(d_iter).name))
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

    def _add_projection(self, projection):
        self.projections.append(projection)

    @property
    def function(self):
        return self.execute

    @property
    def termination_processing(self):
        return self.scheduler.termination_conds

    @termination_processing.setter
    def termination_processing(self, termination_conds):
        self.scheduler.termination_conds = termination_conds


    @property
    def reinitialize_mechanisms_when(self):
        return self._reinitialize_mechanisms_when

    @reinitialize_mechanisms_when.setter
    def reinitialize_mechanisms_when(self, new_condition):

        # Validate
        if not isinstance(new_condition, Condition):
            raise SystemError("{} is not a valid specification for reinitialize_mechanisms_when of {}. "
                              "reinitialize_mechanisms_when must be a Condition.".format(new_condition, self.name))

        # assign to backing field
        self._reinitialize_mechanisms_when = new_condition

        for mechanism in self.mechanisms:
            if hasattr(mechanism, "reinitialize_when"):
                # assign to all mechanisms that do not already have a user-specified condition
                if isinstance(mechanism.reinitialize_when, Never):
                    mechanism.reinitialize_when = new_condition

    @property
    def mechanisms(self):
        """List of all mechanisms in the system

        Returns
        -------
        all mechanisms in the system : List[Mechanism]

        """
        return self._all_mechanisms.mechanisms

    @property
    def stateful_mechanisms(self):
        """
        List of all mechanisms in the system that are currently marked as stateful (mechanism.has_initializers = True)

        Returns
        -------
        all stateful mechanisms in the system : List[Mechanism]

        """

        stateful_mechanisms = []
        for mechanism in self.mechanisms:
            if mechanism.has_initializers:
                stateful_mechanisms.append(mechanism)

        return stateful_mechanisms

    @property
    def mechanism_conditions(self):
        # return [mech.condition for mech in self.mechanisms if hasattr(mech, CONDITION)]
        return dict({mech:mech.condition for mech in self.mechanisms if hasattr(mech, CONDITION)})

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
    def controller(self):
        try:
            return self._controller
        except AttributeError:
            self._controller = None
            return self._controller

    @controller.setter
    def controller(self, control_mech_spec):
        self._instantiate_controller(control_mech_spec, context=Context(source=ContextFlags.PROPERTY))

    @property
    def control_signals(self):
        if self.controller is None:
            return None
        else:
            return self.controller.control_signals

    @property
    def recordSimulationPref(self):
        return self.prefs.recordSimulationPref

    @recordSimulationPref.setter
    def recordSimulationPref(self, setting):
        self.prefs.recordSimulationPref = setting

    def _get_label(self, item, show_dimensions=None, show_roles=None):

        # For Mechanisms, show length of each InputPort and OutputPort
        if isinstance(item, Mechanism):
            if show_roles:
                try:
                    role = item.systems[self]
                    role = role or ""
                except KeyError:
                    if isinstance(item, ControlMechanism) and hasattr(item, 'system'):
                        role = 'CONTROLLER'
                    else:
                        role = ""
                name = "{}\n[{}]".format(item.name, role)
            else:
                name = item.name

            if show_dimensions in {ALL, MECHANISMS}:
                input_str = "in ({})".format(",".join(str(input_port.socket_width)
                                                      for input_port in item.input_ports))
                output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_port.value)))
                                                        for output_port in item.output_ports))
                return "{}\n{}\n{}".format(output_str, name, input_str)
            else:
                return name

        # For Projection, show dimensions of matrix
        elif isinstance(item, Projection):
            if show_dimensions in {ALL, PROJECTIONS}:
                # MappingProjections use matrix
                if isinstance(item, MappingProjection):
                    value = np.array(item.matrix)
                    dim_string = "({})".format("x".join([str(i) for i in value.shape]))
                    return "{}\n{}".format(item.name, dim_string)
                # ModulatoryProjections use value
                else:
                    value = np.array(item.value)
                    dim_string = "({})".format(len(value))
                    return "{}\n{}".format(item.name, dim_string)
            else:
                return item.name

        elif isinstance(item, (System, SystemInputPort)):
            if "SYSTEM" in item.name.upper():
                return item.name + ' Target Input'
            else:
                return "{}\nSystem".format(item.name)

        else:
            raise SystemError("Unrecognized node type ({}) in graph for {}".format(item, self.name))

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.execution_list,
            self.learning_execution_list,
            self.projections,
            [self.controller] if self.controller is not None else [],
        ))

    @handle_external_context()
    def show_graph(self,
                   show_processes = False,
                   show_learning = False,
                   show_control = False,
                   show_prediction_mechanisms = False,
                   show_roles = False,
                   show_dimensions = False,
                   show_mechanism_structure=False,
                   show_headers=True,
                   show_projection_labels=False,
                   direction = 'BT',
                   active_items = None,
                   active_color = BOLD,
                   origin_color = 'green',
                   terminal_color = 'red',
                   origin_and_terminal_color = 'brown',
                   learning_color = 'orange',
                   control_color='blue',
                   prediction_mechanism_color='pink',
                   system_color = 'purple',
                   output_fmt='pdf',
                   context=NotImplemented,
                   ):
        """Generate a display of the graph structure of Mechanisms and Projections in the System.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        Displays a graph showing the structure of the System (based on the `System's graph <System.graph>`).
        By default, only the primary processing Components are shown, and Mechanisms are displayed as simple nodes.
        However, the **show_mechanism_structure** argument can be used to display more detailed information about
        each Mechanism, including its Ports and, optionally, the `function <Component.function>` and `value
        <Component.value>` of the Mechanism and each of its Ports (using the **show_functions** and **show_values**
        arguments, respectively).  The **show_dimension** argument can be used to display the dimensions of each
        Mechanism and Projection.  The **show_processes** argument arranges Mechanisms and Projections into the
        Processes to which they belong. The **show_learning** and **show_control** arguments can be used to
        show the Components associated with `learning <LearningMechanism>` and those associated with the
        System's `controller <System_Control>`.

        `Mechanisms <Mechanism>` are always displayed as nodes.  If **show_mechanism_structure** is `True`,
        Mechanism nodes are subdivided into sections for its Ports with information about each determined by the
        **show_values** and **show_functions** specifications.  Otherwise, Mechanism nodes are simple ovals.
        `ORIGIN` and  `TERMINAL` Mechanisms of the System are displayed with thicker borders in a colors specified
        for each. `Projections <Projection>` are displayed as labelled arrows, unless **show_learning** is specified,
        in which case `MappingProjections <MappingProjection> are displayed as diamond-shaped nodes, and any
        `LearningProjections <LearningProjecction>` as labelled arrows that point to them.

        COMMENT:
        node shapes: https://graphviz.gitlab.io/_pages/doc/info/shapes.html
        arrow shapes: https://graphviz.gitlab.io/_pages/doc/info/arrows.html
        colors: https://graphviz.gitlab.io/_pages/doc/info/colors.html
        COMMENT

        .. _System_Projection_Arrow_Corruption:

        .. note::
           There are two unresolved anomalies associated with show_graph (it is uncertain whether they are bugs in
           PsyNeuLink, Graphviz, or an interaction between the two):

           1) When both **show_mechanism_structure** and **show_processes** are specified together with
              **show_learning** and/or **show_control**, under some arcane conditions Projection arrows can be
              distorted and/or orphaned.  We have confirmed that this does not reflect a corruption of the underlying
              graph structure, and the System should execute normally.

           2) Specifying **show_processes** but not setting **show_headers** to `False` raises a GraphViz exception;
              to deal with this, if **show_processes** is specified, **show_headers** is automatically set to `False`.

           COMMENT:
               See IMPLEMENTATION NOTE under _assign_control_components() for description of the problem
           COMMENT

        Examples
        --------

        The figure below shows different renderings of the following System that can be generated using its
        show_graph method::

            import psyneulink as pnl
            mech_1 = pnl.TransferMechanism(name='Mech 1', size=3, output_ports=[pnl.RESULT, pnl.MEAN])
            mech_2 = pnl.TransferMechanism(name='Mech 2', size=5)
            mech_3 = pnl.TransferMechanism(name='Mech 3', size=2, function=pnl.Logistic(gain=pnl.CONTROL))
            my_process_A = pnl.Process(pathway=[mech_1, mech_3], learning=pnl.ENABLED)
            my_process_B = pnl.Process(pathway=[mech_2, mech_3])
            my_system = pnl.System(processes=[my_process_A, my_process_B],
                                   controller=pnl.ControlMechanism(name='my_system Controller'),
                                   monitor_for_control=[(pnl.MEAN, mech_1)],
                                   enable_controller=True)

        .. _System_show_graph_figure:

        **Output of show_graph using different options**

        .. figure:: _static/show_graph_figure.svg
           :alt: System graph examples
           :scale: 150 %

           Examples of renderings generated by the show_graph method with different options specified, and the call
           to the show_graph method used to generate each rendering shown below each example. **Panel A** shows the
           simplest rendering, with just Processing Components displayed; `ORIGIN` Mechanisms are shown in red,
           and the `TERMINAL` Mechanism in green.  **Panel B** shows the same graph with `MappingProjection` names
           and Component dimensions displayed.  **Panel C** shows the learning Components of the System displayed (in
           orange).  **Panel D** shows the control Components of the System displayed (in blue).  **Panel E** shows
           both learning and control Components;  the learning components are shown with all `LearningProjections
           <LearningProjection>` shown (by specifying show_learning=pnl.ALL).  **Panel F** shows a detailed view of
           the Processing Components, using the show_mechanism_structure option, that includes Component labels and
           values.  **Panel G** show a simpler rendering using the show_mechanism_structure, that only shows
           Component names, but includes the control Components (using the show_control option).


        Arguments
        ---------

        show_processes : bool : False
            specifies whether to organize the `ProcessingMechanisms <ProcessMechanism>` into the `Processes <Process>`
            to which they belong, with each Process shown in its own box.  If a Component belongs to more than one
            Process, it is shown in a separate box along with any others that belong to the same combination of
            Processes;  these represent intersections of Processes within the System.

        show_mechanism_structure : bool, VALUES, FUNCTIONS or ALL : default False
            specifies whether or not to show a detailed representation of each `Mechanism <Mechanism>` in the graph,
            including its `Ports`;  can have the following settings:

            * `True` -- shows Ports of Mechanism, but not information about the `value
              <Component.value>` or `function <Component.function>` of the Mechanism or its Ports.

            * *VALUES* -- shows the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <Port_Base.value>` of each of its Ports.

            * *LABELS* -- shows the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <Port_Base.value>` of each of its Ports, using any labels for the values of InputPorts and
              OutputPorts specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
              `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

            * *FUNCTIONS* -- shows the `function <Mechanism_Base.function>` of the Mechanism and the `function
              <Port_Base.function>` of its InputPorts and OutputPorts.

            * *ROLES* -- shows the `role <System_Mechanisms>` of the Mechanism in the System in square brackets
              (but not any of the other information;  use *ALL* to show ROLES with other information).

            * *ALL* -- shows both `value <Component.value>` and `function <Component.function>` of the Mechanism and
              its Ports (using labels for the values, if specified;  see above).

            Any combination of the settings above can also be specified in a list that is assigned to
            show_mechanism_structure

        COMMENT:
            and, optionally, the `function <Component.function>` and `value <Component.value>` of each (these
            can be specified using the **show_functions** and **show_values** arguments.  If this option is
            specified, Projections are connected to and from the Port that is the `sender <Projection_Base.sender>`
            or `receiver <Projection_Base.receiver>` of each.
        COMMENT

        show_headers : bool : default False
            specifies whether or not to show headers in the subfields of a Mechanism's node;  only takes effect if
            **show_mechanism_structure** is specified (see above).

        COMMENT:
        show_functions : bool : default False
            specifies whether or not to show `function <Component.function>` of Mechanisms and their Ports in the
            graph (enclosed by parentheses);  this requires **show_mechanism_structure** to be specified as `True`
            to take effect.

        show_values : bool : default False
            specifies whether or not to show `value <Component.value>` of Mechanisms and their Ports in the graph
            (prefixed by "=");  this requires **show_mechanism_structure** to be specified as `True` to take effect.
        COMMENT

        show_projection_labels : bool : default False
            specifies whether or not to show names of projections.

        show_learning : bool or ALL : default False
            specifies whether or not to show the learning components of the system;
            they will all be displayed in the color specified for **learning_color**.
            Projections that receive a `LearningProjection` will be shown as a diamond-shaped node.
            if set to *ALL*, all Projections associated with learning will be shown:  the LearningProjections
            as well as from `ProcessingMechanisms <ProcessingMechanism>` to `LearningMechanisms <LearningMechanism>`
            that convey error and activation information;  if set to `True`, only the LearningPojections are shown.

        show_control :  bool : default False
            specifies whether or not to show the control components of the system;
            they will all be displayed in the color specified for **control_color**.

        show_prediction_mechanisms :  bool : default False
            specifies whether or not to show the `PredictionMechanisms <Prediction_Mechanism>` for the
            `controller <System.controller>` of the System; they will be displayed in the color specified for
            **prediction_mechanism_color**.  Only applicable if **show_control** is `True`.

        show_roles : bool : default False
            specifies whether or not to include the `role <System_Mechanisms>` that each Mechanism plays in the System
            (enclosed by square brackets); 'ORIGIN' and 'TERMINAL' Mechanisms are also displayed in a color specified
            by the **origin_color**, **terminal_color** and **origin_and_terminal_color** arguments (see below).

        show_dimensions : bool, MECHANISMS, PROJECTIONS or ALL : default False
            specifies whether or not to show dimensions of Mechanisms (and/or MappingProjections when show_learning
            is `True`);  can have the following settings:

            * *MECHANISMS* -- shows `Mechanism <Mechanism>` input and output dimensions.  Input dimensions are shown
              in parentheses below the name of the Mechanism; each number represents the dimension of the `variable
              <InputPort.variable>` for each `InputPort` of the Mechanism; Output dimensions are shown above
              the name of the Mechanism; each number represents the dimension for `value <OutputPort.value>` of each
              of `OutputPort` of the Mechanism.

            * *PROJECTIONS* -- shows `MappingProjection` `matrix <MappingProjection.matrix>` dimensions.  Each is
              shown in (<dim>x<dim>...) format;  for standard 2x2 "weight" matrix, the first entry is the number of
              rows (input dimension) and the second the number of columns (output dimension).

            * *ALL* -- eqivalent to `True`; shows dimensions for both Mechanisms and Projections (see above for
              formats).

        direction : keyword : default 'BT'
            'BT': bottom to top; 'TB': top to bottom; 'LR': left to right; and 'RL`: right to left.

        active_items : List[Component] : default None
            specifies one or more items in the graph to display in the color specified by *active_color**.

        active_color : keyword : default 'yellow'
            specifies how to highlight the item(s) specified in *active_items**:  either a color recognized
            by GraphViz, or the keyword *BOLD*.

        origin_color : keyword : default 'green',
            specifies the color in which the `ORIGIN` Mechanisms of the System are displayed.

        terminal_color : keyword : default 'red',
            specifies the color in which the `TERMINAL` Mechanisms of the System are displayed.

        origin_and_terminal_color : keyword : default 'brown'
            specifies the color in which Mechanisms that are both
            an `ORIGIN` and a `TERMINAL` of the System are displayed.

        learning_color : keyword : default `green`
            specifies the color in which the learning components are displayed.

        control_color : keyword : default `blue`
            specifies the color in which the learning components are displayed (note: if the System's
            `controller <System.controller>` is an `EVCControlMechanism`, then a link is shown in pink from the
            `prediction Mechanisms <EVCControlMechanism_Prediction_Mechanisms>` it creates to the corresponding
            `ORIGIN` Mechanisms of the System, to indicate that although no projection are created for these,
            the prediction Mechanisms determine the input to the `ORIGIN` Mechanisms when the EVCControlMechanism
            `simulates execution <EVCControlMechanism_Execution>` of the System).

        prediction_mechanism_color : keyword : default `pink`
            specifies the color in which the `prediction_mechanisms
            <EVCControlMechanism.prediction_mechanisms>` are displayed for a System using an `EVCControlMechanism`

        system_color : keyword : default `purple`
            specifies the color in which the node representing input from the System is displayed.

        output_fmt : keyword : default 'pdf'
            'pdf': generate and open a pdf with the visualization;
            'jupyter': return the object (ideal for working in jupyter/ipython notebooks).

        Returns
        -------

        display of system : `pdf` or Graphviz graph object
            'pdf' (placed in current directory) if :keyword:`output_fmt` arg is 'pdf';
            Graphviz graph object if :keyword:`output_fmt` arg is 'jupyter'.

        """

        # if active_item and self.scheduler.clock.time.trial >= self._animate_num_trials:
        #     return

        # IMPLEMENTATION NOTE:
        #    The helper methods below (_assign_XXX__components) all take the main graph *and* subgraph as arguments:
        #        - the main graph (G) is used to assign edges
        #        - the subgraph (sg) is used to assign nodes to Processes if **show_processes** is specified
        #          (otherwise, it should simply be passed G)

        # HELPER METHODS

        if context is NotImplemented:
            context = self.default_execution_id

        tc.typecheck
        def _assign_processing_components(G, sg, rcvr,
                                          processes:tc.optional(list)=None,
                                          subgraphs:tc.optional(dict)=None):
            """Assign nodes to graph, or subgraph for rcvr in any of the specified **processes**."""

            from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism

            rcvr_rank = 'same'
            # Set rcvr color and penwidth info
            if ORIGIN in rcvr.systems[self] and TERMINAL in rcvr.systems[self]:
                if rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = origin_and_terminal_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    rcvr_color = origin_and_terminal_color
                    rcvr_penwidth = str(bold_width)
            elif ORIGIN in rcvr.systems[self]:
                if rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = origin_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    rcvr_color = origin_color
                    rcvr_penwidth = str(bold_width)
                rcvr_rank = origin_rank
            elif TERMINAL in rcvr.systems[self]:
                if rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = terminal_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(bold_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    rcvr_color = terminal_color
                    rcvr_penwidth = str(bold_width)
                rcvr_rank = terminal_rank
            elif rcvr in active_items:
                if active_color is BOLD:
                    if LEARNING in rcvr.systems[self]:
                        rcvr_color = learning_color
                    else:
                        rcvr_color = default_node_color
                else:
                    rcvr_color = active_color
                rcvr_penwidth = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            elif LEARNING in rcvr.systems[self]:
                rcvr_color = learning_color
                rcvr_penwidth = str(default_width)
            else:
                rcvr_color = default_node_color
                rcvr_penwidth = str(default_width)

            if isinstance(rcvr, LearningMechanism):
                if rcvr.learning_timing is LearningTiming.EXECUTION_PHASE and not show_learning:
                    return

            # Implement rcvr node
            rcvr_label=self._get_label(rcvr, show_dimensions, show_roles)
            if show_mechanism_structure:
                sg.node(rcvr_label,
                        rcvr._show_structure(**mech_struct_args),
                        color=rcvr_color,
                        rank=rcvr_rank,
                        penwidth=rcvr_penwidth)
            else:
                sg.node(rcvr_label,
                        shape=mechanism_shape,
                        color=rcvr_color,
                        rank=rcvr_rank,
                        penwidth=rcvr_penwidth)

            # handle auto-recurrent projections
            for input_port in rcvr.input_ports:
                for proj in input_port.path_afferents:
                    if proj.sender.owner is not rcvr:
                        continue
                    if show_mechanism_structure:
                        sndr_proj_label = '{}:{}-{}'.format(rcvr_label, OutputPort.__name__, proj.sender.name)
                        proc_mech_rcvr_label = '{}:{}-{}'.format(rcvr_label, InputPort.__name__, proj.receiver.name)
                    else:
                        sndr_proj_label = proc_mech_rcvr_label = rcvr_label
                    if show_projection_labels:
                        edge_label = self._get_label(proj, show_dimensions, show_roles)
                    else:
                        edge_label = ''
                    try:
                        has_learning = proj.has_learning_projection is not None
                    except AttributeError:
                        has_learning = None

                    # Handle learning components for AutoassociativeProjection
                    #  calls _assign_learning_components,
                    #  but need to manage it from here since MappingProjection needs be shown as node rather than edge
                    if show_learning and has_learning:
                        # Render projection as node
                        if proj in active_items:
                            if active_color is BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        proj_label = self._get_label(proj, show_dimensions, show_roles)
                        render_projection_as_node(G=G, sg=sg, processes=processes,
                                                  rcvr=rcvr, proj=proj,
                                                  label=proj_label,
                                                  rcvr_label=proc_mech_rcvr_label,
                                                  sndr_label=sndr_proj_label,
                                                  proj_color=proj_color,
                                                  proj_width=proj_width)
                    else:
                        # show projection as edge
                        if proj.sender in active_items:
                            if active_color is BOLD:
                                proj_color = default_node_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = default_node_color
                            proj_width = str(default_width)
                        G.edge(sndr_proj_label, proc_mech_rcvr_label, label=edge_label,
                               color=proj_color, penwidth=proj_width)

            # if rcvr is a LearningMechanism, break, as those are handled below
            if isinstance(rcvr, LearningMechanism):
                return
            # if recvr is ObjectiveMechanism for System's controller, break, as those handled below
            if isinstance(rcvr, ObjectiveMechanism) and rcvr.for_controller is True:
                return

            # loop through senders to implement edges
            sndrs = system_graph[rcvr]

            for sndr in sndrs:
                if not processes or any(p in processes for p in sndr.processes.keys()):

                    # Set sndr info

                    sndr_label = self._get_label(sndr, show_dimensions, show_roles)

                    # find edge name
                    for output_port in sndr.output_ports:
                        projs = output_port.efferents
                        for proj in projs:
                            if proj.receiver.owner == rcvr:
                                if show_mechanism_structure:
                                    sndr_proj_label = '{}:{}-{}'.\
                                        format(sndr_label, OutputPort.__name__, proj.sender.name)
                                    proc_mech_rcvr_label = '{}:{}-{}'.\
                                        format(rcvr_label, proj.receiver.__class__.__name__, proj.receiver.name)
                                        # format(rcvr_label, InputPort.__name__, proj.receiver.name)
                                else:
                                    sndr_proj_label = sndr_label
                                    proc_mech_rcvr_label = rcvr_label
                                edge_name = self._get_label(proj, show_dimensions, show_roles)
                                # edge_shape = proj.matrix.shape
                                try:
                                    has_learning = proj.has_learning_projection is not None
                                except AttributeError:
                                    has_learning = None
                                selected_proj = proj
                    edge_label = edge_name

                    # Render projections
                    if any(item in active_items for item in {selected_proj, selected_proj.receiver.owner}):
                        if active_color is BOLD:
                            if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                                proj_color = learning_color
                            else:
                                proj_color = default_node_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True

                    # Projection to or from a LearningMechanism
                    elif (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                        proj_color = learning_color
                        proj_width = str(default_width)

                    else:
                        proj_color = default_node_color
                        proj_width = str(default_width)
                    proc_mech_label = edge_label

                    if show_learning and has_learning:
                        # Render Projection as node
                        #    (do it here rather than in _assign_learning_components,
                        #     as it needs afferent and efferent edges to other nodes)
                        # IMPLEMENTATION NOTE: Projections can't yet use structured nodes:
                        deferred = not render_projection_as_node(G=G, sg=sg, processes=processes,
                                                                 rcvr=rcvr, proj=selected_proj,
                                                                 label=proc_mech_label,
                                                                 rcvr_label=proc_mech_rcvr_label,
                                                                 sndr_label=sndr_proj_label,
                                                                 proj_color=proj_color,
                                                                 proj_width=proj_width)
                        # Deferred if it is the last Mechanism in a learning sequence
                        # (see _render_projection_as_node)
                        if deferred:
                            continue
                    else:
                        # Render Projection normally (as edge)
                        if show_projection_labels:
                            label = proc_mech_label
                        else:
                            label = ''
                        G.edge(sndr_proj_label, proc_mech_rcvr_label, label=label,
                               color=proj_color, penwidth=proj_width)

            # Add node for Projection to the last Mechanism in a learning sequence in its originating Process
            # (i.e., the Process to which its sender belongs)
            if show_learning and processes:
                # If the subgraph is for more than one Process (i.e., it is an intersection of Processes)
                #    then skip, as the point is to assign the Projection node to the single Process to which it belongs
                if len(processes)>1:
                    return
                # Get current Process of ones to which rcvr belongs
                proc = set(rcvr.processes.keys()).intersection(processes).pop()
                # Check whether the rcvr projects to any Mechanism that is the last in a learning sequence
                for proj in rcvr.efferents:
                    try:
                        # If recvr projects to a ComparatorMecchanism used for Learning in the same Process as the recvr
                        if (any(isinstance(p.receiver.owner, ComparatorMechanism)
                               and p.receiver.owner._role == LEARNING
                               and proc in p.receiver.owner.processes
                               for p in proj.receiver.owner.efferents)):
                            # FIX: HANDLE show_mechanism_structure label assignment here
                            proc_mech_label = self._get_label(proj, show_dimensions, show_roles)
                            sndr_label = self._get_label(proj.sender.owner, show_dimensions, show_roles)
                            rcvr_label = self._get_label(proj.receiver.owner, show_dimensions, show_roles)
                            if show_mechanism_structure:
                                sg.node(proc_mech_label, shape=projection_shape)
                                # G.edge(sndr_label, proc_mech_label, arrowhead='none')
                                # G.edge(proc_mech_label, rcvr_label)
                                sndr_proj_label = '{}:{}-{}'.format(sndr_label, OutputPort.__name__, proj.sender.name)
                                proc_mech_rcvr_label = '{}:{}-{}'.format(rcvr_label, InputPort.__name__, proj.receiver.name)
                                G.edge(sndr_proj_label, proc_mech_label, arrowhead='none')
                                G.edge(proc_mech_label, proc_mech_rcvr_label)
                            else:
                                sg.node(proc_mech_label, shape=projection_shape)
                                G.edge(sndr_label, proc_mech_label, arrowhead='none')
                                G.edge(proc_mech_label, rcvr_label)
                    except KeyError:
                        pass

        tc.typecheck
        def _assign_learning_components(G, sg, lg, rcvr, processes:tc.optional(list)=None):
            """Assign learning nodes and edges to graph, or subgraph for rcvr in any of the specified **processes**"""

            # if rcvr is Projection, skip (handled in _assign_processing_components)
            if isinstance(rcvr, MappingProjection):
                return

            # Get rcvr info
            rcvr_label = self._get_label(rcvr, show_dimensions, show_roles)
            if rcvr in active_items:
                if active_color is BOLD:
                    rcvr_color = learning_color
                else:
                    rcvr_color = active_color
                rcvr_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                rcvr_color = learning_color
                rcvr_width = str(default_width)

            # rcvr is a LearningMechanism or ObjectiveMechanism (ComparatorMechanism)
            if self not in rcvr.systems and processes and not any(p in rcvr.processes for p in processes):
                return

            # Implement node for Mechanism
            if show_mechanism_structure:
                sg.node(rcvr_label,
                        rcvr._show_structure(**mech_struct_args),
                        rank=obj_mech_rank, color=rcvr_color, penwidth=rcvr_width)
            else:
                sg.node(rcvr_label,
                        color=rcvr_color, penwidth=rcvr_width,
                        rank=obj_mech_rank, shape=mechanism_shape)

            # Projections to ObjectiveMechanism
            if isinstance(rcvr, ObjectiveMechanism):
                if (self in rcvr.systems
                        and rcvr._role is LEARNING
                        and show_learning is ALL):
                    # Projections to ObjectiveMechanism
                    for input_port in rcvr.input_ports:
                        for proj in input_port.path_afferents:

                            smpl_or_trgt_src = proj.sender.owner

                            # Skip any Projections from ProcesInputPorts
                            if isinstance(smpl_or_trgt_src, Process):
                                continue

                            # Projection is from System
                            # Create node for System "TARGET" input
                            # Note: Mechanism.show_structure is not called for SystemInterfaceMechanism
                            elif isinstance(smpl_or_trgt_src, System):

                                if smpl_or_trgt_src in active_items:
                                    if active_color is BOLD:
                                        smpl_or_trgt_src_color = system_color
                                    else:
                                        smpl_or_trgt_src_color = active_color
                                    smpl_or_trgt_src_width = str(bold_width + active_thicker_by)
                                    self.active_item_rendered = True
                                else:
                                    smpl_or_trgt_src_color = system_color
                                    smpl_or_trgt_src_width = str(bold_width)

                                sg.node(self._get_label(smpl_or_trgt_src, show_dimensions, show_roles),
                                        color=smpl_or_trgt_src_color,
                                        rank='min',
                                        penwidth=smpl_or_trgt_src_width)

                            if proj.receiver.owner in active_items:
                                if active_color is BOLD:
                                    learning_proj_color = learning_color
                                else:
                                    learning_proj_color = active_color
                                learning_proj_width = str(default_width + active_thicker_by)
                                self.active_item_rendered = True
                            else:
                                learning_proj_color = learning_color
                                learning_proj_width = str(default_width)
                            if show_projection_labels:
                                edge_label = proj.name
                            else:
                                edge_label = ''

                            sndr_label = self._get_label(smpl_or_trgt_src, show_dimensions, show_roles)
                            rcvr_label = self._get_label(proj.receiver.owner, show_dimensions, show_roles)
                            if show_mechanism_structure:
                                proc_mech_rcvr_label = rcvr_label +':' + InputPort.__name__ + '-' + proj.receiver.name
                            else:
                                proc_mech_rcvr_label = rcvr_label

                            G.edge(sndr_label, proc_mech_rcvr_label, label=edge_label,
                                   color=learning_proj_color, penwidth=learning_proj_width)
                return

            # Implement edges for Projections to LearningMechanism
            #    from other LearningMechanisms or ObjectiveMechanism, and from ProcessingMechanisms if 'ALL' is set
            for input_port in rcvr.input_ports:
                for proj in input_port.path_afferents:

                    if proj.receiver.owner in active_items:
                        if active_color is BOLD:
                            learning_proj_color = learning_color
                        else:
                            learning_proj_color = active_color
                        learning_proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        learning_proj_color = learning_color
                        learning_proj_width = str(default_width)

                    # Get sndr info
                    sndr = proj.sender.owner
                    sndr_label = self._get_label(sndr, show_dimensions, show_roles)

                    # Create an edge for the Projection to the LearningMechanism if:
                    #    - it is from another LearningMechanism in the same System
                    #    - it is from an ObjectiveMechanism used for learning in the same System
                    #    - **show_learning** argument was specifid as ALL
                    if (((isinstance(sndr, LearningMechanism)
                          or (isinstance(sndr, ObjectiveMechanism) and sndr._role is LEARNING)))
                            or show_learning is ALL):
                        if not self in sndr.systems:
                            continue
                        if show_projection_labels:
                            edge_label = proj.name
                        else:
                            edge_label = ''
                        if show_mechanism_structure:
                            G.edge(sndr_label + ':' + OutputPort.__name__ + '-' + proj.sender.name,
                                   rcvr_label + ':' + InputPort.__name__ + '-' + proj.receiver.name,
                                   label=edge_label, color=learning_proj_color, penwidth=learning_proj_width)
                        else:
                            G.edge(sndr_label, rcvr_label, label=edge_label,
                                   color=learning_proj_color, penwidth=learning_proj_width)

        def render_projection_as_node(G, sg, processes,
                                      rcvr, proj, label,
                                      proj_color, proj_width,
                                      sndr_label=None,
                                      rcvr_label=None):
            from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism

            proj_receiver = proj.receiver.owner
            proj_learning_in_execution_phase = (proj.has_learning_projection and
                                                proj.has_learning_projection.sender.owner.learning_timing is
                                                LearningTiming.EXECUTION_PHASE)

            # If Projection has a LearningProjection from a LearningMechanism
            #    that executes in the execution_phase, include it here
            if proj_learning_in_execution_phase:
                learning_mech = proj.parameter_ports[MATRIX].mod_afferents[0].sender.owner
                learning_rcvrs = [learning_mech, proj]
                learning_graph={proj:{learning_mech}}
                for lr in learning_rcvrs:
                    _assign_learning_components(G, sg, learning_graph, lr, processes)

            # If the recvr is the last Mechanism in a learning sequence in any of the processes passed in,
            #     assignment of nodes for Projections to it will be taken care of below
            #     to insure that they are assigned to the Process(es) from which they originate
            if processes:
                # Get any processes to which recvr belongs
                procs = list(set(proj.sender.owner.processes.keys()).intersection(processes))
                # If recvr projects to any ComparatorMechanism used for learning in the same Process as rcvr
                if (any(isinstance(proj_receiver, ComparatorMechanism)
                       and proj_receiver._role == LEARNING
                       and set(procs).intersection(proj_receiver.processes)
                       for proj in rcvr.efferents)):
                    return False

            # Node for Projection
            sg.node(label, shape=projection_shape, color=proj_color, penwidth=proj_width)

            # FIX: ??
            if proj_receiver in active_items:
                # edge_color = proj_color
                # edge_width = str(proj_width)
                if active_color is BOLD:
                    edge_color = proj_color
                else:
                    edge_color = active_color
                edge_width = str(default_width + active_thicker_by)
            else:
                edge_color = default_node_color
                edge_width = str(default_width)

            # Edges to and from Projection node
            if sndr_label:
                G.edge(sndr_label, label, arrowhead='none',
                       color=edge_color, penwidth=edge_width)
            if rcvr_label:
                G.edge(label, rcvr_label,
                       color=edge_color, penwidth=edge_width)

            # LearningProjection(s) to node
            if proj in active_items or (proj_learning_in_execution_phase and proj_receiver in active_items):
                if active_color is BOLD:
                    learning_proj_color = learning_color
                else:
                    learning_proj_color = active_color
                learning_proj_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                learning_proj_color = learning_color
                learning_proj_width = str(default_width)
            sndrs = proj._parameter_ports['matrix'].mod_afferents # GET ALL LearningProjections to proj
            for sndr in sndrs:
                sndr_label = self._get_label(sndr.sender.owner, show_dimensions, show_roles)
                rcvr_label = self._get_label(proj, show_dimensions, show_roles)
                if show_projection_labels:
                    edge_label = proj._parameter_ports['matrix'].mod_afferents[0].name
                else:
                    edge_label = ''
                if show_mechanism_structure:
                    G.edge(sndr_label + ':' + OutputPort.__name__ + '-' + 'LearningSignal',
                           rcvr_label,
                           label=edge_label,
                           color=learning_proj_color, penwidth=learning_proj_width)
                else:
                    G.edge(sndr_label, rcvr_label, label = edge_label,
                           color=learning_proj_color, penwidth=learning_proj_width)
            return True

        def _assign_control_components(G, sg, show_prediction_mechanisms):
            """Assign control nodes and edges to graph, or subgraph for rcvr in any of the specified **processes**."""

            controller = self.controller
            if controller in active_items:
                if active_color is BOLD:
                    ctlr_color = control_color
                else:
                    ctlr_color = active_color
                ctlr_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                ctlr_color = control_color
                ctlr_width = str(default_width)

            if controller is None:
                print ("\nWARNING: {} has not been assigned a \'controller\', so \'show_control\' option "
                       "can't be used in its show_graph() method\n".format(self.name))
                return

            # get projection from ObjectiveMechanism to ControlMechanism
            objmech_ctlr_proj = controller.input_port.path_afferents[0]
            if controller in active_items:
                if active_color is BOLD:
                    objmech_ctlr_proj_color = control_color
                else:
                    objmech_ctlr_proj_color = active_color
                objmech_ctlr_proj_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                objmech_ctlr_proj_color = control_color
                objmech_ctlr_proj_width = str(default_width)

            # get ObjectiveMechanism
            objmech = objmech_ctlr_proj.sender.owner
            if objmech in active_items:
                if active_color is BOLD:
                    objmech_color = control_color
                else:
                    objmech_color = active_color
                objmech_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                objmech_color = control_color
                objmech_width = str(default_width)

            ctlr_label = self._get_label(controller, show_dimensions, show_roles)
            objmech_label = self._get_label(objmech, show_dimensions, show_roles)
            if show_mechanism_structure:
                sg.node(ctlr_label,
                        controller._show_structure(**mech_struct_args),
                        color=ctlr_color,
                        penwidth=ctlr_width,
                        rank = control_rank
                       )
                sg.node(objmech_label,
                        objmech._show_structure(**mech_struct_args),
                        color=objmech_color,
                        penwidth=ctlr_width,
                        rank = control_rank
                        )
            else:
                sg.node(ctlr_label,
                        color=ctlr_color, penwidth=ctlr_width, shape=mechanism_shape,
                        rank=control_rank)
                sg.node(objmech_label,
                        color=objmech_color, penwidth=objmech_width, shape=mechanism_shape,
                        rank=control_rank)

            # objmech to controller edge
            if show_projection_labels:
                edge_label = objmech_ctlr_proj.name
            else:
                edge_label = ''
            if show_mechanism_structure:
                obj_to_ctrl_label = objmech_label + ':' + OutputPort.__name__ + '-' + objmech_ctlr_proj.sender.name
                ctlr_from_obj_label = ctlr_label + ':' + InputPort.__name__ + '-' + objmech_ctlr_proj.receiver.name
            else:
                obj_to_ctrl_label = objmech_label
                ctlr_from_obj_label = ctlr_label
            G.edge(obj_to_ctrl_label, ctlr_from_obj_label, label=edge_label,
                   color=objmech_ctlr_proj_color, penwidth=objmech_ctlr_proj_width)

            # IMPLEMENTATION NOTE:
            #   When two (or more?) Processes (e.g., A and B) have homologous constructions, and a ControlProjection is
            #   assigned to a ProcessingMechanism in one Process (e.g., the 1st one in Process A) and a
            #   ProcessingMechanism in the other Process corresponding to the next in the sequence (e.g., the 2nd one
            #   in Process B) the Projection arrow for the first one get corrupted and sometimes one or more of the
            #   following warning/error messages appear in the console:
            # Warning: Arrow type "arial" unknown - ignoring
            # Warning: Unable to reclaim box space in spline routing for edge "ProcessingMechanism4 ComparatorMechanism
            # [LEARNING]" -> "LearningMechanism for MappingProjection from ProcessingMechanism3 to ProcessingMechanism4
            # [LEARNING]". Something is probably seriously wrong.
            # These do not appear to reflect corruptions of the graph structure and/or execution.

            # outgoing edges (from controller to ProcessingMechanisms)
            for control_signal in controller.control_signals:
                for ctl_proj in control_signal.efferents:
                    proc_mech_label = self._get_label(ctl_proj.receiver.owner, show_dimensions, show_roles)
                    if controller in active_items:
                        if active_color is BOLD:
                            ctl_proj_color = control_color
                        else:
                            ctl_proj_color = active_color
                        ctl_proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        ctl_proj_color = control_color
                        ctl_proj_width = str(default_width)
                    if show_projection_labels:
                        edge_label = ctl_proj.name
                    else:
                        edge_label = ''
                    if show_mechanism_structure:
                        ctl_sndr_label = ctlr_label + ':' + OutputPort.__name__ + '-' + control_signal.name
                        proc_mech_rcvr_label = \
                            proc_mech_label + ':' + ParameterPort.__name__ + '-' + ctl_proj.receiver.name
                    else:
                        ctl_sndr_label = ctlr_label
                        proc_mech_rcvr_label = proc_mech_label
                    G.edge(ctl_sndr_label,
                           proc_mech_rcvr_label,
                           label=edge_label,
                           color=ctl_proj_color,
                           penwidth=ctl_proj_width
                           )

            # incoming edges (from monitored mechs to objective mechanism)
            for input_port in objmech.input_ports:
                for projection in input_port.path_afferents:
                    if objmech in active_items:
                        if active_color is BOLD:
                            proj_color = control_color
                        else:
                            proj_color = active_color
                        proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        proj_color = control_color
                        proj_width = str(default_width)
                    if show_mechanism_structure:
                        sndr_proj_label = self._get_label(projection.sender.owner, show_dimensions, show_roles) +\
                                          ':' + OutputPort.__name__ + '-' + projection.sender.name
                        objmech_proj_label = objmech_label + ':' + InputPort.__name__ + '-' + input_port.name
                    else:
                        sndr_proj_label = self._get_label(projection.sender.owner, show_dimensions, show_roles)
                        objmech_proj_label = self._get_label(objmech, show_dimensions, show_roles)
                    if show_projection_labels:
                        edge_label = projection.name
                    else:
                        edge_label = ''
                    G.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                           color=proj_color, penwidth=proj_width)

            # prediction mechanisms
            if show_prediction_mechanisms:
                for mech in self.execution_list:
                    if mech in active_items:
                        if active_color is BOLD:
                            pred_mech_color = prediction_mechanism_color
                        else:
                            pred_mech_color = active_color
                        pred_mech_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        pred_mech_color = prediction_mechanism_color
                        pred_mech_width = str(default_width)
                    if mech._role is CONTROL and hasattr(mech, 'origin_mech'):
                        recvr = mech.origin_mech
                        recvr_label = self._get_label(recvr, show_dimensions, show_roles)
                        # IMPLEMENTATION NOTE:
                        #     THIS IS HERE FOR FUTURE COMPATIBILITY WITH FULL IMPLEMENTATION OF PredictionMechanisms
                        if show_mechanism_structure and False:
                            proj = mech.output_port.efferents[0]
                            if proj in active_items:
                                if active_color is BOLD:
                                    pred_proj_color = prediction_mechanism_color
                                else:
                                    pred_proj_color = active_color
                                pred_proj_width = str(default_width + active_thicker_by)
                                self.active_item_rendered = True
                            else:
                                pred_proj_color = prediction_mechanism_color
                                pred_proj_width = str(default_width)
                            sg.node(mech.name,
                                    shape=mech._show_structure(**mech_struct_args),
                                    color=pred_mech_color,
                                    penwidth=pred_mech_width)

                            G.edge(mech.name + ':' + OutputPort.__name__ + '-' + mech.output_port.name,
                                   recvr_label + ':' + InputPort.__name__ + '-' + proj.receiver.name,
                                   label=' prediction assignment',
                                   color=pred_proj_color, penwidth=pred_proj_width)
                        else:
                            sg.node(self._get_label(mech, show_dimensions, show_roles),
                                    color=pred_mech_color, shape=mechanism_shape, penwidth=pred_mech_width)
                            G.edge(self._get_label(mech, show_dimensions, show_roles),
                                   recvr_label,
                                   label=' prediction assignment',
                                   color=prediction_mechanism_color)

        # MAIN BODY OF METHOD:

        import graphviz as gv

        system_graph = self.graph
        learning_graph=self.learningGraph

        if show_dimensions == True:
            show_dimensions = ALL
        if show_processes:
            show_headers = False

        if not active_items:
            active_items = []
        elif active_items is INITIAL_FRAME:
            active_items = [INITIAL_FRAME]
        elif not isinstance(active_items, Iterable):
            active_items = [active_items]
        elif not isinstance(active_items, list):
            active_items = list(active_items)
        for item in active_items:
            if not isinstance(item, Component) and item is not INITIAL_FRAME:
                raise SystemError("PROGRAM ERROR: Item ({}) specified in {} argument for {} method of {} is not a {}".
                                  format(item, repr('active_items'), repr('show_graph'), self.name, Component.__name__))

        self.active_item_rendered = False

        # Argument values used to call Mechanism.show_structure()
        if isinstance(show_mechanism_structure, (list, tuple, set)):
            mech_struct_args = {'composition':self,
                                'show_roles':any(key in show_mechanism_structure for key in {ROLES, ALL}),
                                'show_functions':any(key in show_mechanism_structure for key in {FUNCTIONS, ALL}),
                                'show_values':any(key in show_mechanism_structure for key in {VALUES, ALL}),
                                'use_labels':any(key in show_mechanism_structure for key in {LABELS, ALL}),
                                'show_headers':show_headers,
                                'output_fmt':'struct',
                                'context':context}
        else:
            mech_struct_args = {'composition':self,
                                'show_roles':show_mechanism_structure in {ROLES, ALL},
                                'show_functions':show_mechanism_structure in {FUNCTIONS, ALL},
                                'show_values':show_mechanism_structure in {VALUES, LABELS, ALL},
                                'use_labels':show_mechanism_structure in {LABELS, ALL},
                                'show_headers':show_headers,
                                'output_fmt':'struct',
                                'context':context}

        default_node_color = 'black'
        mechanism_shape = 'oval'
        projection_shape = 'diamond'
        # projection_shape = 'point'
        # projection_shape = 'Mdiamond'
        # projection_shape = 'hexagon'

        bold_width = 3
        default_width = 1
        active_thicker_by = 2

        pos = None

        origin_rank = 'source'
        control_rank = 'min'
        obj_mech_rank = 'sink'
        terminal_rank = 'max'
        learning_rank = 'sink'

        # build graph and configure visualisation settings
        G = gv.Digraph(
                name = self.name,
                engine = "dot",
                # engine = "fdp",
                # engine = "neato",
                # engine = "circo",
                node_attr  = {
                    'fontsize':'12',
                    'fontname':'arial',
                    # 'shape':mechanism_shape,
                    'shape':'record',
                    'color':default_node_color,
                    'penwidth':str(default_width)
                },
                edge_attr  = {
                    # 'arrowhead':'halfopen',
                    'fontsize': '10',
                    'fontname': 'arial'
                },
                graph_attr = {
                    "rankdir" : direction,
                    'overlap' : "False"
                },
        )
        # G.attr(compound = 'True')

        # get System's ProcessingMechanisms
        rcvrs = list(system_graph.keys())
        # learning_rcvrs = list(learning_graph.keys())
        learning_rcvrs = self.learning_execution_list

        # MANAGE ProcessingMechanisms and LearningMechanisms

        # if show_processes is specified, create subgraphs for each Process
        if show_processes:

            # Manage Processes
            process_intersections = {}
            subgraphs = {}  # Entries: Process:sg
            for process in self.processes:
                subgraph_name = 'cluster_' + process.name
                subgraph_label = process.name
                with G.subgraph(name=subgraph_name) as sg:
                    subgraphs[process.name]=sg
                    sg.attr(label=subgraph_label)
                    sg.attr(rank = 'same')
                    # sg.attr(style='filled')
                    # sg.attr(color='lightgrey')

                    # loop through receivers and assign to the subgraph any that belong to the current Process
                    for r in rcvrs:
                        intersection = [p for p in self.processes if p in r.processes]
                        # If the rcvr is in only one Process, add it to the subgraph for that Process
                        if len(intersection)==1:
                            # If the rcvr is in the current Process, assign it to the subgraph
                            if process in intersection:
                                _assign_processing_components(G, sg, r, [process])
                        # Otherwise, assign rcvr to entry in dict for process intersection (subgraph is created below)
                        else:
                            intersection_name = ' and '.join([p.name for p in intersection])
                            if not intersection_name in process_intersections:
                                process_intersections[intersection_name] = [r]
                            else:
                                if r not in process_intersections[intersection_name]:
                                    process_intersections[intersection_name].append(r)

                    # loop through learning Components and assign to the subgraph any that belong to the current Process
                    if show_learning:
                        for l in learning_rcvrs:
                            if isinstance(l, Projection):
                                processes = l.sender.owner.processes
                            else:
                                processes = l.processes
                            intersection = [p for p in self.processes if p in processes]
                            # If the Component is in only one Process, add it to the subgraph for that Process
                            if len(intersection)==1:
                                if process in processes:
                                    _assign_learning_components(G, sg, learning_graph, l, [process])
                            # Otherwise, assign Component to entry in dict for process intersection (subgraph is
                            # created below)
                            else:
                                intersection_name = ' and '.join([p.name for p in intersection])
                                if not intersection_name in process_intersections:
                                    process_intersections[intersection_name] = [l]
                                else:
                                    if l not in process_intersections[intersection_name]:
                                        process_intersections[intersection_name].append(l)

            # Create a process for each unique intersection and assign rcvrs to that
            for intersection_name, mech_list in process_intersections.items():
                with G.subgraph(name='cluster_' + intersection_name) as sg:
                    sg.attr(label=intersection_name)
                    # get list of processes in the intersection (to pass to _assign_processing_components)
                    processes = [p for p in self.processes if p.name in intersection_name]
                    # loop through receivers and assign to the subgraph any that belong to the current Process
                    for r in mech_list:
                        if r in self.graph:
                            _assign_processing_components(G, sg, r, processes, subgraphs)
                        elif r in self.learningGraph:
                            _assign_learning_components(G, sg, learning_graph, r, processes)
                        else:
                            raise SystemError("PROGRAM ERROR: Component in interaction process ({}) is not in "
                                              "{}'s graph or learningGraph".format(r.name, self.name))

        else:
            for r in rcvrs:
                _assign_processing_components(G, G, r)
            # Add learning-related Components to graph if show_learning
            if show_learning:
                for rcvr in learning_rcvrs:
                    # if 'Auto' in rcvr.name:
                    #     break
                    _assign_learning_components(G, G, learning_graph, rcvr)

        # MANAGE CONTROL Components

        # Add control-related Components to graph if show_control
        if show_control:
            if show_processes:
                with G.subgraph(name='cluster_CONTROLLER') as sg:
                    sg.attr(label='CONTROLLER')
                    sg.attr(rank='top')
                    # sg.attr(style='filled')
                    # sg.attr(color='lightgrey')
                    _assign_control_components(G, sg, show_prediction_mechanisms)
            else:
                _assign_control_components(G, G, show_prediction_mechanisms)

        # GENERATE OUTPUT

        # Show as pdf
        if output_fmt == 'pdf':
            # G.format = 'svg'
            G.view(self.name.replace(" ", "-"), cleanup=True, directory='show_graph OUTPUT/PDFS')

        # Generate images for animation
        elif output_fmt == 'gif':
            if self.active_item_rendered or INITIAL_FRAME in active_items:
                G.format = 'gif'
                execution_phase = context.execution_phase
                if INITIAL_FRAME in active_items:
                    time_string = ''
                    phase_string = ''
                elif execution_phase & (ContextFlags.PROCESSING | ContextFlags.IDLE):
                    # time_string = repr(self.scheduler.clock.simple_time)
                    time = self.scheduler.get_clock(context).time
                    time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}".\
                        format(time.run, time.trial, time.pass_, time.time_step)
                    phase_string = 'Processing Phase - '
                elif execution_phase == ContextFlags.LEARNING:
                    time = self.scheduler_learning.get_clock(context).time
                    time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}".\
                        format(time.run, time.trial, time.pass_, time.time_step)
                    phase_string = 'Learning Phase - '
                elif execution_phase == ContextFlags.CONTROL:
                    time_string = ''
                    phase_string = 'Control phase'
                else:
                    raise SystemError("PROGRAM ERROR:  Unrecognized phase during execution of {}".format(self.name))
                label = '\n{}\n{}{}\n'.format(self.name, phase_string, time_string)
                G.attr(label=label)
                G.attr(labelloc='b')
                G.attr(fontname='Helvetica')
                G.attr(fontsize='14')
                if INITIAL_FRAME in active_items:
                    index = '-'
                else:
                    index = repr(self._component_execution_count)
                image_filename = repr(self.scheduler.get_clock(context).simple_time.trial) + '-' + index + '-'
                image_file = self._animate_directory + '/' + image_filename + '.gif'
                G.render(filename = image_filename,
                         directory=self._animate_directory,
                         cleanup=True,
                         # view=True
                         )
                # Append gif to self._animation
                image = Image.open(image_file)
                if not self._save_images:
                    remove(image_file)
                if not hasattr(self, '_animation'):
                    self._animation = [image]
                else:
                    self._animation.append(image)

        # Return graph to show in jupyter
        elif output_fmt == 'jupyter':
            return G


SYSTEM_TARGET_INPUT_PORT = 'SystemInputPort'

from psyneulink.core.components.ports.outputport import OutputPort
class SystemInputPort(OutputPort):
    """Represents inputs and targets specified in a call to the System's `execute <Process.execute>` and `run
    <Process.run>` methods.

    COMMENT:
        Each instance encodes a `target <System.target>` to the system (also a 1d array in 2d array of
        `targets <System.targets>`) and provides it to a `MappingProjection` that projects to a `TARGET`
        Mechanism of the System.

        .. Declared as a subclass of OutputPort so that it is recognized as a legitimate sender to a Projection
           in Projection_Base._instantiate_sender()

           value is used to represent the item of the targets arg to system.execute or system.run
    COMMENT

    A SystemInputPort is created for each `InputPort` of each `ORIGIN` Mechanism in `origin_mechanisms`, and for the
    *TARGET* `InputPort <ComparatorMechanism_Structure>` of each `ComparatorMechanism <ComparatorMechanism>` listed
    in `target_nodes <System.target_nodes>`.  A `MappingProjection` is created that projects to each
    of these InputPorts from the corresponding SystemInputPort.  When the System's `execute <System.execute>` or
    `run <System.run>` method is called, each item of its **inputs** and **targets** arguments is assigned as
    the `value <SystemInputPort.value>` of a SystemInputPort, which is then conveyed to the
    corresponding InputPort of the `origin_mechanisms <System.origin_mechanisms>` and `terminal_mechanisms
    <System.terminal_mechanisms>`.  See `System_Mechanisms` and `System_Execution` for additional details.

    """
    class Parameters(OutputPort.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <SystemInputPort.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                value
                    see `value <SystemInputPort.value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True
        """
        # just grabs input from the process
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        value = Parameter(np.array([0]), read_only=True, pnl_internal=True)

    def __init__(self, owner=None, variable=None, name=None, prefs=None, context=None):
        """Pass variable to MappingProjection from Process to first Mechanism in Pathway

        :param variable:
        """
        if not name:
            self.name = owner.name + "_" + SYSTEM_TARGET_INPUT_PORT
        else:
            self.name = owner.name + "_" + name
        self.initialization_status = ContextFlags.INITIALIZING
        self.prefs = prefs
        self.log = Log(owner=self)
        self.path_afferents = []
        self.owner = owner

        self.parameters = self.Parameters(owner=self, parent=self.class_parameters)
        self.defaults = Defaults(owner=self, variable=variable, value=variable)

        self.parameters.value._set(variable, context)
