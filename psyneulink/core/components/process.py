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
         * :ref:`Process_Learning_Sequence`
      * :ref:`Process_Execution`
         * :ref:`Process_Execution_Learning`
      * :ref:`Process_Class_Reference`


.. _Process_Overview:

Overview
--------

A Process is the simplest form of `Composition <Composition>`, made up of a `lineal <Process_Footnotes>` sequence of
`Mechanisms <Mechanism>` linked by `Projections <Projection>`.  Processes can be executed on their own, but most
commonly they are used to compose a `System`, which is the most powerful form of Composition in PsyNeuLink.  Processes
are nevertheless useful, as they define a simpler unit of processing than a System (e.g., for debugging, or for use in
multiple Systems), and are used as the unit of `learning <System_Learning>` within a System.  The general features of
Processes are summarized below, followed by a more detailed description in the sections that follow.

Mechanisms and Projections are composed into a Process by assigning them to the Process' `pathway
<Process.pathway>` attribute. Executing a Process executes all of its Mechanisms in the order in which they are
listed in its `pathway  <Process.pathway>`.  Projections can be specified among any Mechanisms in a Process,
including to themselves, however they must compose a `lineal <Process_Footnotes>` sequence.  A Process cannot involve
any "branching" (beyond what may be produced by recurrent loops within the Process); that must be done by using a
Process  to compose each branch, and then composing the Processes into a  `System`. Mechanisms in a Process can project
to and  receive Projections from Mechanisms in other Processes, however these will not have any effect when the Process
is  executed; these will only have an effect if all of the Processes involved are members of the same System and the
`System is executed <System_Execution_Processing>`.

Projections between Mechanisms can be trained by `specifying them for learning
<MappingProjection_Learning_Specification>`.  Learning can also be `specified for the entire Process
<Process_Learning_Specification>`, in which case all of the Projections among Mechanisms in the Process are trained
(see `Process_Learning_Sequence` below).

.. _Process_Creation:

Creating a Process
------------------

A Process is created by instantiating the `Process` class. The Mechanisms to be included are specified in a list in its
**pathway** argument, in the order in which they should be executed by the Process.  The Mechanism entries can be
separated by `Projections <Projection>` used to connect them.  If no arguments are provided to the **pathway** argument,
a Process with an empty pathway is created.

.. _Process_Structure:

Structure
---------

.. _Process_Pathway:

*Pathway*
~~~~~~~~~

A Process is defined by its `pathway <Process.pathway>` attribute, which is a list of `Mechanisms <Mechanism>` and
`Projections <Projection>`, that are executed in the order in which they are specified in the list. Each Mechanism in
the `pathway <Process.pathway>` must project at least to the next one in the `pathway <Process.pathway>`,
though it can project to others, and receive recurrent (feedback) Projections from them.  However, a `pathway
<Process.pathway>` cannot include branching patterns beyond any produced by recurrent loops (see `Examples
<Process_Examples>` below);  that is, a Mechanism cannot project to another Mechanism that falls outside the `lineal
<Process_Footnotes>` sequence of the `pathway <Process.pathway>` To compose more complex, branched, structures,
a Process should be created for each "branch", and these used to compose a `System <System_Creation>`.

The Mechanisms specified in the `pathway` for a Process must be `ProcessingMechanisms <ProcessingMechanism>`, and
the projections between Mechanisms in a Process must be `MappingProjections <MappingProjection>`.  These transmit the
output of a Mechanism (the Projection's `sender <MappingProjection.MappingProjection.sender>`) to the input of
another Mechanism (the Projection's `receiver <MappingProjection.MappingProjection.receiver>`). Specification of a
`pathway` requires, at the least, a list of Mechanisms.  Each of these can be specified directly, or using a **tuple**
that also contains a set of `runtime parameters <Mechanism_Runtime_Parameters>` (see `below
<Process_Mechanism_Specification>`). A Projection between a pair of Mechanisms can be specified by interposing it in
the list between the pair.  If no Projection is specified between two adjacent Mechanisms in the `pathway
<Process.pathway>`, and there is no otherwise specified Projection between them, a default MappingProjection is
automatically created when the Process is created that projects from the first to the second member of the pair.
Specifying the Components of a pathway is described in detail below.

.. _Process_Mechanisms:

*Mechanisms*
~~~~~~~~~~~~

The `Mechanisms <Mechanism>` of a Process must be listed explicitly in the **pathway** argument of the `Process`
class, in the order they are to be executed when the Process (or any System to which it belongs) is `executed
<Process_Execution>`.  The first Mechanism in a Process is designated as its `ORIGIN` Mechanism, and is assigned to its
`origin_mechanism <Process.origin_mechanism>` attribute; it receives as its input any `input
<Process_Input_And_Output>` provided to the Process' `execute <Process.execute>` or `run <Process.run>`
methods. The last Mechanism listed in the `pathway <Process.pathway>` is designated as the `TERMINAL` Mechanism,
and is assigned to its `terminal_mechanism <Process.terminal_mechanism>` attribute; its `output_values
<Mechanism_Base.output_values>` is assigned as the `output <Process_Output>` of the Process.

.. _Process_Mechanism_Initialize_Cycle:

Any Mechanism that sends a Projection that closes a recurrent loop within the `pathway <Process.pathway>` is
designated as `INITIALIZE_CYCLE`; whenever that Mechanism is `initialized <Process_Execution_Initialization>`,
it is assigned the value specified for it in the **initial_values** argument of the Process' `execute
<Process.execute>` or `run <Process.run>` methods. Mechanisms that receive a Projection from one designated
`INITIALIZE_CYCLE` are themselves designated as `CYCLE`.  All other Mechanisms in the `pathway <Process.pathway>`
are designated as `INTERNAL`.

.. note::
   The `origin_mechanism <Process.origin_mechanism>` and `terminal_mechanism <Process.terminal_mechanism>`
   of a Process are not necessarily the `ORIGIN` and/or `TERMINAL` Mechanisms of the System(s)to which the Process
   belongs (see `example <LearningProjection_Output_vs_Terminal_Figure>`).  The designations of a Mechanism's status
   in the Process(es) to which it belongs are listed in its `processes <Mechanism_Base.processes>` attribute.

.. _Process_Mechanism_Specification:

Mechanisms can be specified in the **pathway** argument of the `Process` class in one of two ways:

    * **Directly** -- using any of the ways used to `specify a Mechanism <Mechanism_Creation>`.
    ..
    * **MechanismTuple** -- the first item must be a specification for the Mechanism using any of the ways used to
      `specify a Mechanism <Mechanism_Creation>`;  the second must be a set of `runtime parameters
      <Mechanism_Runtime_Parameters>`. Runtime parameters are used for that Mechanism when the Process (or a System
      to which it belongs) is executed; otherwise they do not remain associated with the Mechanism.

The same Mechanism can appear more than once in a `pathway <Process.pathway>`, as one means of generating a
recurrent processing loop (another is to specify this in the Projections -- see below).

.. _Process_Projections:

*Projections*
~~~~~~~~~~~~~

`MappingProjections <MappingProjection>` between Mechanisms in the `pathway <Process.pathway>` of a Process can be
specified in any of the following ways:

  * **Inline specification** -- a MappingProjection specification can be interposed between any two Mechanisms in the
    `pathway <Process.pathway>` list. This creates a Projection from the preceding Mechanism in the list to the
    one that follows it.  It can be specified using any of the ways used to `specify a Projection
    <Projection_Specification>` or the `matrix parameter <MappingProjection_Matrix_Specification>` of one.
  ..

  .. _Process_Tuple_Specification:

  * **Tuple learning specification** -- this can be used in the same way as an inline specification;  the first item
    must a MappingProjection specification that takes the same form as an inline specification, and the second must be
    a `learning specification <MappingProjection_Learning_Tuple_Specification>`.
  ..
  * **Stand-alone MappingProjection** -- when a Projection is `created <Projection_Creation>` on its own,
    it can be assigned a `sender <Projection_Sender>` and/or a `receiver <Projection_Receiver>`
    Mechanism. If both are in the Process, then that Projection will be used when creating the Process.  Stand-alone
    specification of a MappingProjection between two Mechanisms in a Process takes precedence over any other
    form of specification; that is, the stand-alone Projection will be used in place of any that is specified between
    the Mechanisms in a `pathway <Process.pathway>`. Stand-alone specification is required to implement
    MappingProjections between Mechanisms that are not adjacent to one another in the `pathway <Process.pathway>`.
  ..
  * **Default assignment** -- for any Mechanism that does not receive a MappingProjection from another Mechanism in the
    Process (specified using one of the methods above), a `MappingProjection` is automatically created from the
    Mechanism that precedes it in the `pathway <Process.pathway>`. If the format of the `value <OutputPort.value>`
    of the preceding Mechanism's `primary OutputPort <OutputPort_Primary>` matches that of the next Mechanism, then an
    `IDENTITY_MATRIX` is used for the Projection's `matrix <MappingProjection.matrix>` parameter;  if the formats do not
    match, or `learning has been specified <Process_Learning_Sequence>` either for the Projection or the Process, then a
    `FULL_CONNECTIVITY_MATRIX` is used.  If the Mechanism is the `origin_mechanism <Process.origin_mechanism>`
    (i.e., first in the `pathway <Process.pathway>`), a `ProcessInputPort <Process_Input_And_Output>` is used
    as the `sender <MappingProjection.sender>`, and an `IDENTITY_MATRIX` is used for the MappingProjection.

.. _Process_Input_And_Output:

*Process input and output*
~~~~~~~~~~~~~~~~~~~~~~~~~~

The `input <Process.input>` of a Process is a list or 2d np.array provided as the **input** argument in its
`execute <Process.execute>` method, or the **inputs** argument of its `run <Process.run>` method. When a
Process is created, a set of `ProcessInputPorts <ProcessInputPort>` (listed in its `process_input_ports` attribute)
and `MappingProjections <MappingProjection>` are automatically created to transmit the Process' `input
<Process.input>` to its `origin_mechanism <Process.origin_mechanism>`, as follows:

    * if the number of items in the **input** is the same as the number of `InputPorts <InputPort>` for the
      `origin_mechanism <Process.origin_mechanism>`, a MappingProjection is created for each item of the input to a
      distinct InputPort of the `origin_mechanism <Process.origin_mechanism>`;
    ..
    * if the **input** has only one item but the `origin_mechanism <Process.origin_mechanism>` has more than one
      InputPort, a single `ProcessInputPort <ProcessInputPort>` is created with Projections to each of the
      `origin_mechanism <Process.origin_mechanism>`'s InputPorts;
    ..
    * if the **input** has more than one item but the `origin_mechanism <Process.origin_mechanism>` has only one
      InputPort, a `ProcessInputPort <ProcessInputPort>` is created for each item of the input, and all project to
      the `origin_mechanism <Process.origin_mechanism>`'s InputPort;
    ..
    * otherwise, if the **input** has more than one item and the `origin_mechanism <Process.origin_mechanism>` has
      more than one InputPort, but the numbers are not equal, an error message is generated indicating that there is an
      ambiguous mapping from the Process' **input** value to `origin_mechanism <Process.origin_mechanism>`'s
      InputPorts.

.. _Process_Output:

The output of a Process is assigned as the `output_values <Mechanism_Base.output_values>` attribute of its `TERMINAL`
Mechanism.

.. _Process_Learning_Sequence:

*Learning*
~~~~~~~~~~

Learning operates over a *learning sequence*: a contiguous sequence of `ProcessingMechanisms <ProcessingMechanism>` in
a Process `pathway <Process.pathway>`, and the `MappingProjections <MappingProjection>` between them, that have
been specified for learning. Learning modifies the `matrix <MappingProjection.matrix>` parameter of the
MappingProjections in the sequence, so that the input to the first ProcessingMechanism in the sequence generates an
output from the last ProcessingMechanism that matches as closely as possible the target specified for the sequence
(see `Process_Execution_Learning` below for a more detailed description).

.. _Process_Learning_Specification:

Learning can be `specified for individual (or subsets of) MappingProjections
<MappingProjection_Learning_Specification>`, or for the entire Process.  It is specified for the entire process by
assigning a specification for a `LearningProjection <LearningProjection_Creation>` or `LearningSignal
<LearningSignal_Specification>` specification, or the keyword *ENABLED*, to the **learning** argument of the
Process' constructor.  Specifying learning for a Process implements it for all MappingProjections in the Process (except
those that project from the `process_input_ports` to the `origin_mechanism <Process.origin_mechanism>`), which
are treated as a single learning sequence.  Mechanisms that receive MappingProjections for which learning has been
specified must be compatible with learning (that is, their `function <Mechanism_Base.function>` must be compatible with
the `function <LearningMechanism.function>` of the `LearningMechanism` for the MappingProjections they receive (see
`LearningMechanism_Function`).

.. _Process_Learning_Components:

The following Components are created for each learning sequence specified for a Process (see figure below):

    * a `TARGET` `ComparatorMechanism` (assigned to the Process' `target_nodes <Process.target_nodes>`
      attribute), that is used to `calculate an error signal <ComparatorMechanism_Execution>` for the sequence, by
      comparing `a specified output <LearningMechanism_Activation_Output>` of the last Mechanism in the learning
      sequence (received in the ComparatorMechanism's *SAMPLE* `InputPort <ComparatorMechanism_Structure>`) with the
      item of the **target** argument in Process' `execute <Process.execute>` or `run <Process.run>` method
      corresponding to the learning sequence (received in the ComparatorMechanism's *TARGET* `InputPort
      <ComparatorMechanism_Structure>`).
    ..
    * a MappingProjection that projects from the last ProcessingMechanism in the sequence to the *SAMPLE* `InputPort
      <ComparatorMechanism_Structure>` of the `TARGET` Mechanism;
    ..
    * a ProcessingInputPort to represent the corresponding item of the **target** argument of the Process' `execute
      <Process.execute>` and `run <Process.run>` methods;
    ..
    * a MappingProjection that projects from the `ProcessInputPort <ProcessInputPort>` for the **target** item to the
      *TARGET* `InputPort <ComparatorMechanism_Structure>` of the `TARGET` Mechanism;
    ..
    * a `LearningMechanism` for each MappingProjection in the sequence that calculates the `learning_signal
      <LearningMechanism.learning_signal>` used to modify the `matrix <MappingProjection.matrix>` parameter for that
      MappingProjection, along with a `LearningSignal` and `LearningProjection` that convey the `learning_signal
      <LearningMechanism.learning_signal>` to the MappingProjection's *MATRIX* `ParameterPort
      <Mapping_Matrix_ParameterPort>` (additional MappingProjections are created for the LearningMechanism -- see
      `LearningMechanism_Learning_Configurations` for details).

    .. note::
       The Components created when learning is specified for individual MappingProjections of a Process (or subsets of
       them) take effect only if the Process is executed on its own (i.e., using its `execute <Process.execute>`
       or `run <Process.run>` methods.  For learning to in a Process when it is `executed as part of a System
       <System_Execution_Learning>`, learning must be specified for the *entire Process*, as described above.

COMMENT:
    XXX ?HOW:
    Different learning algorithms can be specified (e.g., `Reinforcement` or `BackPropagation`), that implement the
    Mechanisms and LearningSignals required for the specified type of learning. However,  as noted above,
    all Mechanisms that receive Projections being learned must be compatible with learning.
COMMENT

.. _Process_Learning_Figure:

**Figure: Learning Components in a Process**

.. figure:: _static/Process_Learning_fig.svg
   :alt: Schematic of LearningMechanism and LearningProjections in a Process

   Learning using the `BackPropagation` learning algorithm in a three-layered network, using a `TransferMechanism` for
   each layer (capitalized labels in Mechanism components are their `designated roles
   <Mechanism_Role_In_Processes_And_Systems>` in the Process -- see also `Process_Mechanisms` and `Keywords`).

.. _Process_Execution:

Execution
---------

A Process can be executed as part of a `System <System_Execution>` or on its own.  On its own, it is executed by calling
either its `execute <Process.execute>` or `run <Process.run>` method.  `execute <Process.execute>` executes
the Process once (that is, it executes a single `TRIAL`);  `run <Process.run>` allows a series of `TRIAL`\\s to be
executed.

. _Process_Processing

*Processing*
~~~~~~~~~~~~

When a Process is executed, its `input` is conveyed to the `origin_mechanism <Process.origin_mechanism>`
(the first Mechanism in the `pathway <Process.pathway>`).  By default, the input is presented only once.  If
the `origin_mechanism <Process.origin_mechanism>` is executed again in the same `PASS` of execution (e.g., if it
appears again in the pathway, or receives recurrent projections), the input is not presented again. However, the input
can be "clamped" on using the **clamp_input** argument of `execute <Process.execute>` or `run <Process.run>`.
After the `origin_mechanism <Process.origin_mechanism>` is executed, each subsequent Mechanism in the `pathway` is
executed in sequence.  If a Mechanism is specified in the pathway using a `MechanismTuple
<Process_Mechanism_Specification>`, then the `runtime parameters <Mechanism_Runtime_Parameters>` are applied and the
Mechanism is executed using them (see `Mechanism <Mechanism_ParameterPorts>` for parameter specification).  Finally the
output of the `terminal_mechanism <Process.terminal_mechanism>` (the last one in the pathway) is assigned as the
`output <Process_Output>` of the Process.

.. note::
   Processes do not use a `Scheduler`; each Mechanism is executed once, in the order listed in its `pathway` attribute.
   To more precisely control the order of, and/or any dependencies in, the sequence of executions, the Process
   should be used to construct a `System`, together with `Conditions <Condition>` to implement a custom schedule.


.. _Process_Execution_Initialization

The `input <Process_Input_And_Output>` to a Process is specified in the **input** argument of either its `execute
<Process.execute>` or `run <Process.run>` method. In both cases, the input for a single `TRIAL` must be a
number, list or ndarray of values that is compatible with the `variable <Mechanism_Base.variable>` of the
`origin_mechanism <Process.origin_mechanism>`. If the `execute <Process.execute>` method is used, input for
only a single `TRIAL` is provided, and only a single `TRIAL` is executed.  The `run <System.run>` method can be
used for a sequence of `TRIAL`\\s, by providing it with a list or ndarray of inputs, one for each `TRIAL`.  In both
cases, two other types of input can be provided in corresponding arguments of the `execute <Process.execute>`
and `run <Process.run>` methods: a  list or ndarray of **initial_values**, and a list or ndarray of **target**
values. The **initial_values** are assigned as input to Mechanisms that close recurrent loops (designated as
`INITIALIZE_CYCLE`) at the start of a `TRIAL` (if **initialize** is set to `True`), and/or whenever the Process`
`initialize <Process.initialize>` method is called; **target** values are assigned as the *TARGET* input of the
`target_nodes <Process.target_nodes>` in each `TRIAL` of execution, if `learning
<Process_Learning_Sequence>` has been specified (see the next setion for how Learning is executed; also,
see `Run` documentation for additional details of formatting `Run_Input` and `Run_Target` specifications of the
`run <Process.run>` method).

.. _Process_Execution_Learning:

*Learning*
~~~~~~~~~~

If `learning <Process_Learning_Sequence>` has been specified for the Process or any of the projections in its `pathway
<Process.pathway>`, then the learning Components described `above <Process_Learning_Components>` are executed after
all of the ProcessingMechanisms in the `pathway <Process.pathway>` have executed.  The learning Components
calculate changes that will be  made to `matrix <MappingProjection.matrix>` of the MappingProjections involved.  This
requires that a set of `target values <Run_Targets>` be provided (along with the **inputs**) in the **targets**
argument of the Process' `execute <Process.execute>` or `run <Process.run>` method, one for each `learning
sequence <Process_Learning_Sequence>`. These are used to calculate a `learning_signal
<LearningMechanism.learning_signal>` for each MappingProjection in a learning sequence. This is conveyed by a
`LearningProjection` as a `weight_change_matrix <LearningProjection.weight_change_matrix>` to the MappingProjection's
*MATRIX* `ParameterPort <Mapping_Matrix_ParameterPort>`, that  is used to modify the MappingProjection's `matrix
<MappingProjection.matrix>` parameter when it executes.

.. note::
   The changes to a Projection induced by learning are not applied until the Mechanisms that receive those
   projections are next executed (see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

The `learning_signal <LearningMechanism>`\\s for a learning sequence are calculated, for each sequence, so as to reduce
the difference between the value received by the *TARGET* Mechanism in its *SAMPLE* `InputPort
<ComparatorMechanism_Structure>` (see `above <Process_Learning_Sequence>`) and the target value for the sequence
specified in the corresponding item of the **target** argument of the Process' `execute <Process.execute>` or
`run <Process.run>` method.


.. _Process_Examples:

Examples
--------

*Specification of Mechanisms in a pathway:*  The first Mechanism in the example below is specified as a reference to an
instance, the second as a default instance of a Mechanism type, and the third in `MechanismTuple format
<Process_Mechanism_Specification>`, specifying a reference to a Mechanism that should receive my_params at runtime::

    mechanism_1 = TransferMechanism()
    mechanism_2 = DDM()
    some_params = {PARAMETER_PORT_PARAMS:{THRESHOLD:2,NOISE:0.1}}
    my_process = Process(pathway=[mechanism_1, TransferMechanism, (mechanism_2, my_params)])

*Default Projection specification:*  The `pathway` for this Process uses default Projection specifications; as a
result, a `MappingProjection` is automatically instantiated between each of the Mechanisms listed::

    my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3])


*Inline Projection specification using an existing Projection:*  In this `pathway <Process.pathway>`,
``projection_A`` is specified as the Projection between the first and second Mechanisms; a default Projection is
created between ``mechanism_2`` and ``mechanism_3``::

    projection_A = MappingProjection()
    my_process = Process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])

*Inline Projection specification using a keyword:*  In this `pathway <Process.pathway>`, a
`RANDOM_CONNECTIVITY_MATRIX` is used to specify the Projection between the first and second Mechanisms::

    my_process = Process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])

*Stand-alone Projection specification:*  In this `pathway <Process.pathway>`, ``projection_A`` is explicitly
specified as a Projection between ``mechanism_1`` and ``mechanism_2``, and so is used as the Projection between them
in ``my_process``; a default Projection is created between ``mechanism_2`` and ``mechanism_3``::

    projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
    my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3])

*Process that implements learning:*  This `pathway <Process.pathway>` implements a series of Mechanisms with
Projections between them, all of which will be learned using `BackPropagation` (the default learning algorithm).
Note that it uses the `Logistic` function, which is compatible with BackPropagation::

    mechanism_1 = TransferMechanism(function=Logistic)
    mechanism_2 = TransferMechanism(function=Logistic)
    mechanism_3 = TransferMechanism(function=Logistic)
    my_process = Process(pathway=[mechanism_1, mechanism_2, mechanism_3],
                         learning=ENABLED,
                         target=[0])

*Process with individual Projections that implement learning:* This `pathway <Process.pathway>` implements learning
for two MappingProjections (between ``mechanism_1`` and ``mechanism_2``, and ``mechanism_3`` and ``mechanism_4``).
Since they are not contiguous, two `learning sequences <Process_Learning_Sequence>` are created, with `TARGET`
Mechanisms assigned to ``mechanism_2`` and ``mechanism_4`` (that will be listed in ``my_process.target_nodes``)::

    mechanism_1 = TransferMechanism(function=Logistic)
    mechanism_2 = TransferMechanism(function=Logistic)
    mechanism_3 = TransferMechanism(function=Logistic)
    mechanism_4 = TransferMechanism(function=Logistic)
    my_process = Process(pathway=[mechanism_1,
                                  MappingProjection(matrix=(RANDOM_CONNECTIVITY_MATRIX, LEARNING),
                                  mechanism_2,
                                  mechanism_3,
                                  MappingProjection(matrix=(RANDOM_CONNECTIVITY_MATRIX, LEARNING)),
                                  mechanism_4])


.. _Process_Footnotes:

Footnotes
---------

*lineal*:  this term is used rather than "linear" to refer to the flow of processing -- i.e., the graph structure
of the Process -- rather than the (potentially non-linear) processing characteristics of its individual Components.


.. _Process_Class_Reference:

Class Reference
---------------

"""

import copy
import inspect
import itertools
import numbers
import re
import types
import warnings

from collections import UserList, namedtuple

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import Component
from psyneulink.core.components.mechanisms.mechanism import MechanismList, Mechanism_Base
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import _add_projection_to, _is_projection_spec
from psyneulink.core.components.shellclasses import Mechanism, Process_Base, Projection, System_Base
from psyneulink.core.components.ports.modulatorysignals.learningsignal import LearningSignal
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.port import _instantiate_port, _instantiate_port_list
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    AUTO_ASSIGN_MATRIX, ENABLED, FUNCTION, FUNCTION_PARAMS, INITIAL_VALUES, INTERNAL, LEARNING, LEARNING_PROJECTION, \
    MAPPING_PROJECTION, MATRIX, NAME, OBJECTIVE_MECHANISM, ORIGIN, PARAMETER_PORT, PATHWAY, SENDER, SINGLETON, \
    TARGET, TERMINAL, PROCESS_COMPONENT_CATEGORY, RECEIVER_ARG
from psyneulink.core.globals.parameters import Defaults, Parameter
from psyneulink.core.globals.preferences.basepreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import append_type_to_name, convert_to_np_array, iscompatible

__all__ = [
    'DEFAULT_PHASE_SPEC', 'DEFAULT_PROJECTION_MATRIX', 'defaultInstanceCount', 'kwProcessInputPort', 'kwTarget',
    'Process', 'proc', 'ProcessError', 'ProcessInputPort', 'ProcessList', 'ProcessRegistry', 'ProcessTuple',
]

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

kwProcessInputPort = 'ProcessInputPort'
kwTarget = 'target'
from psyneulink.core.components.ports.outputport import OutputPort

# DOCUMENT:  HOW DO MULTIPLE PROCESS INPUTS RELATE TO # OF InputPortS IN FIRST MECHANISM
#            WHAT HAPPENS IF LENGTH OF INPUT TO PROCESS DOESN'T MATCH LENGTH OF VARIABLE FOR FIRST MECHANISM??


def proc(*args, **kwargs):
    """Factory method

    **args** can be `Mechanisms <Mechanism>` with our without `Projections <Projection>`, or a list of them,
    that conform to the format for the `pathway <Process.pathway>` argument of a `Process`.

    **kwargs** can be any arguments of the `Process` constructor.
    """
    return Process(pathway=list(args), **kwargs)


class Process(Process_Base):
    """
    Process(process_spec=None,                           \
    default_variable=None,                               \
    pathway=None,                                        \
    initial_values={},                                   \
    clamp_input:=None,                                   \
    default_projection_matrix=DEFAULT_PROJECTION_MATRIX, \
    learning=None,                                       \
    learning_rate=None                                   \
    target=None,                                         \
    params=None,                                         \
    name=None,                                           \
    prefs=None

    Base class for Process.

    COMMENT:
        Description
        -----------
            Process is a Category of the Component class.
            It implements a Process that is used to execute a sequence of Mechanisms connected by projections.
            NOTES:
                * if no pathway is provided:
                    no mechanism is used
                * the input to the Process is assigned as the input to its ORIGIN Mechanism
                * the output of the Process is taken as the value of the primary OutputPort of its TERMINAL Mechanism

        Class attributes
        ----------------
        componentCategory : str : default kwProcessFunctionCategory
        className : str : default kwProcessFunctionCategory
        suffix : str : default "<kwMechanismFunctionCategory>"
        registry : dict : default ProcessRegistry
        classPreference : PreferenceSet : default ProcessPreferenceSet instantiated in __init__()
        classPreferenceLevel (PreferenceLevel): PreferenceLevel.CATEGORY
        + class_defaults.variable = inputValueSystemDefault                     # Used as default input value to Process)

        Class methods
        -------------
            - execute(input, control_signal_allocations):
                executes the Process by calling execute_functions of the Mechanisms (in order) in the pathway list
                assigns input to sender.output (and passed through mapping) of first Mechanism in the pathway list
                assigns output of last Mechanism in the pathway list to self.output
            - register_process(): registers Process with ProcessRegistry
            [TBI: - adjust(control_signal_allocations=NotImplemented):
                modifies the control_signal_allocations while the Process is executing;
                calling it without control_signal_allocations functions like interrogate
                returns (responseState, accuracy)
            [TBI: - interrogate(): returns (responseState, accuracy)
            [TBI: - terminate(): terminates the Process and returns output
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

    pathway : List[ProcessingMechanism, MappingProjection, ProcessingMechanism...]
        the `ProcessingMechanisms <ProcessingMechanism>` and `MappingProjections <MappingProjection>` between them that
        are executed in the order listed when the Process `executes <Process_Execution>`.

    process_input_ports : List[ProcessInputPort]
        represent the input to the Process when it is executed.  Each `ProcessInputPort <ProcessInputPort>` represents
        an item of the `input <Process.base>` to a corresponding `InputPort` of the Process' `origin_mechanism
        <Process.origin_mechanism>` (see `Process_Input_And_Output` for details).

    input :  List[value] or ndarray
        input to the Process for each `TRIAL` of execution;  it is assigned the value of the **input** argument
        in a call to the Process' `execute <Process.execute>`  or `run <Process.run>` method. Each of its
        items is assigned as the `value <InputPort.value>` of the corresponding `ProcessInputPort <ProcessInputPort>`
        in `process_input_ports`, and each must match the format of the corresponding item of the `variable
        <Mechanism_Base.variable>` for the Process' `origin_mechanism <Process.origin_mechanism>`
        (see `Process_Input_And_Output` for details).

        .. note::
            The `input <Process.input>` attribute of a Process preserves its value throughout the execution of the
            Process. It's value is assigned to the `variable <Mechanism_Base.variable>` attribute of the
            `origin_mechanism <Process.origin_mechanism>` at the start of execution.  After that, by default, that
            Mechanism's `variable <Mechanism_Base.variable>` attribute is zeroed. This is so that if the
            `origin_mechanism <Process.origin_mechanism>` is executed again in the same `TRIAL` of execution
            (e.g., if it is part of a recurrent loop) it does not continue to receive the initial input to the
            Process.  However, this behavior can be modified with the Process' `clamp_input <Process.clamp_input>`
            attribute.

    COMMENT
        input_value :  2d np.array : default ``defaults.variable``
            same as the `variable <Process.variable>` attribute of the Process; contains the `value
            <InputPort.value>` of each ProcessInputPort in its `process_input_ports` attribute.
    COMMENT

    clamp_input : Optional[keyword]
        determines whether the Process' `input <Process.input>` continues to be applied to the `origin_mechanism
        <Process.origin_mechanism>` if it is executed again within the same `TRIAL`.  It can take the following
        values:

        * `None`: applies the Process' `input <Process.input>` to the `origin_mechanism
          <Process.origin_mechanism>` only once (the first time it is executed) in a given `TRIAL` of execution.

        * `SOFT_CLAMP`: combines the Process' `input <Process.input>` with input from any other Projections to the
          `origin_mechanism <Process.origin_mechanism>` every time the latter is executed within a `TRIAL` of
          execution.

        * `HARD_CLAMP`: applies the Process' `input <Process.input>` to the `origin_mechanism
          <Process.origin_mechanism>` to the exclusion of any other source(s) of input every time the Process is
          executed.

    initial_values : Dict[ProcessingMechanism, param value]
        values used to initialize ProcessingMechanisms designated as `INITIALIZE_CYCLE` whenever the Process'
        `initialize <Process.initialize>` method is called. The key for each entry is a ProcessingMechanism, and
        the value is a number, list or np.array that is assigned to that Mechanism's `value <Mechanism_Base.value>`
        attribute whenever it is initialized. `ProcessingMechanisms <ProcessingMechanism>` that are designated as
        `INITIALIZE_CYCLE` but not included in the **initial_values** specification are initialized with the value of
        their `variable <Mechanism_Base.variable>` attribute (i.e., the default input for that Mechanism).

    value: 2d np.array
        same as the `value <OutputPort.value>` of the `primary OutputPort <OutputPort_Primary>` of
        `terminal_mechanism <Process.terminal_mechanism>`.

    output_port : Port
        the `primary OutputPort <OutputPort_Primary>` of `terminal_mechanism <Process.terminal_mechanism>`.

    output : list
        same as the `output_values <Mechanism_Base.output_values>` attribute of `terminal_mechanism
        <Process.terminal_mechanism>`.

    COMMENT
    .. _mechs : List[MechanismTuple]
         :class:`MechanismTuple` for all Mechanisms in the Process, listed in the order specified in pathway.
         MechanismTuples are of the form: (Mechanism, runtime_params, phase) where runtime_params is dictionary
         of {argument keyword: argument values} entries and phase is an int.
         Note:  the list includes ComparatorMechanisms and LearningMechanism.

    .. _all_mechanisms : MechanismList
         Contains all Mechanisms in the System (based on _mechs).

    .. _origin_mechs : List[MechanismTuple]
         Contains a tuple for the `ORIGIN` Mechanism of the Process.
         (Note:  the use of a list is for compatibility with the MechanismList object)

    .. _terminal_mechs : List[MechanismTuple]
         Contains a tuple for the `TERMINAL` Mechanism of the Process.
         (Note:  the use of a list is for compatibility with the MechanismList object)

    .. _target_mechs : List[MechanismTuple]
         Contains a tuple for the `TARGET` Mechanism of the Process.
         (Note:  the use of a list is for compatibility with the MechanismList object)

    .. _learning_mechs : List[MechanismTuple]
         `MechanismTuple
         Process (used for learning).

    .. mechanisms : List[Mechanism]
         List of all Mechanisms in the Process.
         property that points to _all_mechanisms.mechanisms (see below).
    COMMENT

    mechanism_names : List[str]
        the names of the Mechanisms listed in the `Mechanisms <Process.mechanisms>` attribute.

        .. property that points to _all_mechanisms.names (see below).

    mechanisms : List[Mechanism]
        *all* of the Mechanisms in the Process, including those in the `pathway <Process.pathway>`
        and those created for `learning <Process_Learning_Sequence>`.

    origin_mechanism : Mechanism
        the `ORIGIN` Mechanism of the Process (see `Process Mechanisms <Process_Mechanisms>` for a description).

    COMMENT
    ..  origin_mechanisms : MechanismList
            a list with the `ORIGIN` Mechanism of the Process.

            .. note:: A Process can have only one `ORIGIN` Mechanism; the use of a list is for compatibility with
                      methods that are also used for Systems.
    COMMENT

    terminal_mechanism : Mechanism
        the `TERMINAL` Mechanism of the Process (see `Process Mechanisms <Process_Mechanisms>` for a description).

    COMMENT
    ..  terminalMechanisms : MechanismList
            a list with the `TERMINAL` Mechanism of the Process.

            .. note:: A Process can have only one `TERMINAL` Mechanism; the use of a list is for compatibility with
                      methods that are also used for Systems.
    COMMENT

    learning_mechanisms : MechanismList
        all of the `LearningMechanism in the Process <Process_Learning_Sequence>`, listed in
        ``learning_mechanisms.data``.

        .. based on _learning_mechs

    target_mechanisms : MechanismList
        the `TARGET` Mechanisms for the Process, listed in ``target_nodes.data``;  each is a `ComparatorMechanism`
        associated with the last ProcessingMechanism of a `learning sequence <Process_Learning_Sequence>` in the
        Process;

        COMMENT:
        .. note:: A Process can have only one `TARGET` Mechanism; the use of a list is for compatibility with
                  methods that are also used for Systems.
        COMMENT

        COMMENT:
            based on _target_mechs
        COMMENT

    systems : List[System]
        the `Systems <System>` to which the Process belongs.

      .. _phaseSpecMax : int : default 0
             phase of last (set of) ProcessingMechanism(s) to be executed in the Process.
             It is assigned to the ``phaseSpec`` for the Mechanism in the pathway with the largest ``phaseSpec`` value.

      .. numPhases : int : default 1
            the number of :ref:`phases <System_Execution_Phase>` for the Process.

        COMMENT:
            It is assigned as ``_phaseSpecMax + 1``.
        COMMENT

      .. _isControllerProcess : bool : :keyword:`False`
             identifies whether the Process is an internal one created by a ControlMechanism.

    learning : Optional[LearningProjection]
        indicates whether the Process is configured for learning.  If it has a value other than `None`, then `learning
        has been configured <Process_Learning_Specification>` for one or more `MappingProjections <MappingProjection>`
        in the Process;  if it is `None`, none of MappingProjections in the Process has been configured for learning.

        .. note::
           The `learning <Process.learning>` attribute of a Process may have a value other than `None` even
           if no assignment is made to the **learning** argument of the `process` command;  this occurs if one or more
           MappingProjections in the Process are `specified individually for learning
           <Process_Learning_Specification>`.

        COMMENT:
        .. note::  If an existing `LearningProjection` or a call to the constructor is used for the specification,
                   the object itself will **not** be used as the LearningProjection for the Process. Rather it
                   will be used as a template (including any parameters that are specified) for creating
                   LearningProjections for all of the `MappingProjections <MappingProjection>` in the Process.

                   .. _learning_enabled : bool
                      indicates whether or not learning is enabled.  This only has effect if the ``learning`` parameter
                      has been specified (see above).
        COMMENT

    learning_rate : float : default None
        determines the `learning_rate <LearningMechanism.learning_rate>` used for `MappingProjections
        <MappingProjection>` `specified for learning <Process_Learning_Sequence>` in the Process that do not have their
        `learning_rate <LearningProjection.learning_rate>` otherwise specified.   If is `None`, and the Process is
        executed as part of a `System`, and the System has a `learning_rate <System.learning_rate>` specified,
        then that is the value used.  Otherwise, the default value of the :keyword:`learning_rate` parameter for the
        `function <LearningMechanism.function>` of the `LearningMechanism associated with each MappingProjection
        <Process_Learning_Sequence>` is used.  If a :keyword:`learning_rate` is specified for the `LearningSignal
        <LearningSignal_Learning_Rate>` or `LearningProjection <LearningProjection_Function_and_Learning_Rate>`
        associated with a MappingProjection, that is applied in addition to any specified for the Process or the
        relevant LearningMechanism.

    results : List[OutputPort.value]
        the return values from a sequence of executions of the Process;  its value is `None` if the Process has not
        been executed.

    name : str
        the name of the Process; if it is not specified in the **name** argument of the constructor, a
        default is assigned by ProcessRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Process; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """

    componentCategory = PROCESS_COMPONENT_CATEGORY
    className = componentCategory
    suffix = " " + className
    componentType = "Process"

    registry = ProcessRegistry

    classPreferenceLevel = PreferenceLevel.CATEGORY
    # These will override those specified in TYPE_DEFAULT_PREFERENCES
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'ProcessCustomClassPreferences',
    #     REPORT_OUTPUT_PREF: PreferenceEntry(False, PreferenceLevel.INSTANCE)}
    # Use inputValueSystemDefault as default input to process

    class Parameters(Process_Base.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Process.variable>`

                    :default value: None
                    :type:

                clamp_input
                    see `clamp_input <Process.clamp_input>`

                    :default value: None
                    :type:

                default_projection_matrix
                    see `default_projection_matrix <Process.default_projection_matrix>`

                    :default value: `AUTO_ASSIGN_MATRIX`
                    :type: ``str``

                initial_values
                    see `initial_values <Process.initial_values>`

                    :default value: None
                    :type:

                input
                    see `input <Process.input>`

                    :default value: []
                    :type: ``list``

                learning
                    see `learning <Process.learning>`

                    :default value: None
                    :type:

                learning_rate
                    see `learning_rate <Process.learning_rate>`

                    :default value: None
                    :type:

                pathway
                    see `pathway <Process_Pathway>`

                    :default value: None
                    :type:

                process_input_ports
                    see `process_input_ports <Process.process_input_ports>`

                    :default value: []
                    :type: ``list``

                systems
                    see `systems <Process.systems>`

                    :default value: []
                    :type: ``list``

                target
                    see `target <Process.target>`

                    :default value: None
                    :type:

                target_input_ports
                    see `target_input_ports <Process.target_input_ports>`

                    :default value: []
                    :type: ``list``

                targets
                    see `targets <Process.targets>`

                    :default value: None
                    :type:
        """
        variable = None
        input = []
        pathway = None

        process_input_ports = []
        targets = None
        target_input_ports = []
        systems = []

        initial_values = None
        clamp_input = None
        default_projection_matrix = DEFAULT_PROJECTION_MATRIX
        learning = None

        learning_rate = None
        target = None

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 size=None,
                 pathway=None,
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

        pathway = pathway or []
        self.projections = []

        register_category(entry=self,
                          base_class=Process,
                          name=name,
                          registry=ProcessRegistry,
                          context=context)

        if not context:
            self.initialization_status = ContextFlags.INITIALIZING
        # If input was not provided, generate defaults to match format of ORIGIN mechanisms for process
        if default_variable is None and len(pathway) > 0:
            default_variable = pathway[0].defaults.variable

        self.default_execution_id = self.name
        self._phaseSpecMax = 0
        self._isControllerProcess = False

        super(Process, self).__init__(
            default_variable=default_variable,
            size=size,
            param_defaults=params,
            name=self.name,
            pathway=pathway,
            initial_values=initial_values,
            clamp_input=clamp_input,
            default_projection_matrix=default_projection_matrix,
            learning=learning,
            learning_rate=learning_rate,
            target=target,
            prefs=prefs
        )

    def _parse_arg_variable(self, variable):
        if variable is None:
            return None

        return super()._parse_arg_variable(convert_to_np_array(variable, dimension=2))

    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate initial_values args
           Note: validation of target (for learning) is deferred until _instantiate_target since,
                 if it doesn't have a TARGET Mechanism (see _check_for_target_mechanisms),
                 it will not need a target.
        """

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        # Note: target_set (argument of validate_params) should not be confused with
        #       self.target (process attribute for learning)
        if INITIAL_VALUES in target_set and target_set[INITIAL_VALUES]:
            for mech, value in target_set[INITIAL_VALUES].items():
                if not isinstance(mech, Mechanism):
                    raise SystemError("{} (key for entry in initial_values arg for \'{}\') "
                                      "is not a Mechanism object".format(mech, self.name))

    def _instantiate_attributes_before_function(self, function=None, context=None):
        """Call methods that must be run before function method is instantiated

        Need to do this before _instantiate_function as mechanisms in pathway must be instantiated
            in order to assign input Projection and self.outputPort to first and last mechanisms, respectively

        :param function:
        :param context:
        :return:
        """
        self._instantiate_pathway(context=context)

    def _instantiate_function(self, function, function_params=None, context=None):
        """Override Function._instantiate_function:

        This is necessary to:
        - insure there is no FUNCTION specified (not allowed for a Process object)
        - suppress validation (and attendant execution) of Process execute method (unless VALIDATE_PROCESS is set)
            since generally there is no need, as all of the mechanisms in the pathway have already been validated;
            Note: this means learning is not validated either
        """

        if self.function != self.execute:
            print("Process object ({0}) should not have a specification ({1}) for a {2} param;  it will be ignored").\
                format(self.name, self.function, FUNCTION)
            self.function = self.execute

# DOCUMENTATION:

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
            - for last Mechanism in Pathway, assign ouputPort to Process.outputPort
        Iterate through Pathway, assigning Projections to Mechanisms:
            - first Mechanism in Pathway:
                if it does NOT already have any projections:
                    assign Projection(s) from ProcessInputPort(s) to corresponding Mechanism.input_port(s):
                if it DOES already has a Projection, and it is from:
                    (A) the current Process input, leave intact
                    (B) another Process input, if verbose warn
                    (C) another Mechanism in the current process, if verbose warn about recurrence
                    (D) a Mechanism not in the current Process or System, if verbose warn
                    (E) another Mechanism in the current System, OK so ignore
                    (F) from something other than a Mechanism in the System, so warn (irrespective of verbose)
                    (G) a Process in something other than a System, so warn (irrespective of verbose)
            - subsequent Mechanisms:
                assign projections from each Mechanism to the next one in the list:
                - if Projection is explicitly specified as item between them in the list, use that;
                - if Projection is NOT explicitly specified,
                    but the next Mechanism already has a Projection from the previous one, use that;
                - otherwise, instantiate a default MappingProjection from previous Mechanism to next:
                    use kwIdentity (identity matrix) if len(sender.value) == len(receiver.defaults.variable)
                    use FULL_CONNECTIVITY_MATRIX (full connectivity matrix with unit weights) if the lengths are not equal
                    use FULL_CONNECTIVITY_MATRIX (full connectivity matrix with unit weights) if LEARNING has been set

        :param context:
        :return:
        """
        pathway = self.pathway
        self._mechs = []
        self._learning_mechs = []
        self._target_mechs = []

        # VALIDATE PATHWAY THEN PARSE AND INSTANTIATE MECHANISM ENTRIES  ------------------------------------
        self._parse_and_instantiate_mechanism_entries(pathway=pathway, context=context)

        # Identify ORIGIN and TERMINAL Mechanisms in the Process and
        #    and assign the Mechanism's status in the Process to its entry in the Mechanism's processes dict

        # Move any ControlMechanisms in the pathway to the end
        from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
        for i, item in enumerate(pathway):
            if len(pathway)>1 and isinstance(item, ControlMechanism):
                pathway += [pathway.pop(i)]

        # Identify and assign first Mechanism as first_mechanism and ORIGIN
        self.first_mechanism = pathway[0]
        self.first_mechanism._add_process(self, ORIGIN)
        self._origin_mechs = [pathway[0]]
        self.origin_mechanisms = MechanismList(self, self._origin_mechs)

        # Identify and assign last Mechanism as last_mechanism and ORIGIN
        i = -1
        while (not isinstance(pathway[i],Mechanism_Base) or
               (isinstance(pathway[i], ControlMechanism) and len(pathway)>1)):
            i -=1
        self.last_mechanism = pathway[i]

        if self.last_mechanism is self.first_mechanism:
            self.last_mechanism._add_process(self, SINGLETON)
        else:
            self.last_mechanism._add_process(self, TERMINAL)
        self._terminal_mechs = [pathway[-1]]
        self.terminal_mechanisms = MechanismList(self, self._terminal_mechs)

        # # Assign process OutputPort to last mechanisms in pathway
        # self.outputPort = self.last_mechanism.outputPort

        # PARSE AND INSTANTIATE PROJECTION ENTRIES  ------------------------------------

        self._parse_and_instantiate_projection_entries(pathway=pathway, context=context)

        self.pathway = pathway

        self._instantiate__deferred_inits(context=context)

        if self.learning:
            if self._check_for_target_mechanisms():
                if self._target_mechs:
                    self._instantiate_target_input(context=context)
                self._learning_enabled = True
            else:
                self._learning_enabled = False
        else:
            self._learning_enabled = False

        self._all_mechanisms = MechanismList(self, self._mechs)
        self.learning_mechanisms = MechanismList(self, self._learning_mechs)
        self.target_mechanisms = MechanismList(self, self._target_mechs)

    def _instantiate_value(self, context=None):
        # If validation pref is set, execute the Process
        if self.prefs.paramValidationPref:
            super()._instantiate_value(context=context)
        # Otherwise, just set Process output info to the corresponding info for the last mechanism in the pathway
        else:
            # MODIFIED 6/24/18 OLD:
            # value = self.pathway[-1].output_port.value
            # MODIFIED 6/24/18 NEW:
            value = self.last_mechanism.output_port.value
            # MODIFIED 6/24/18 END
            try:
                # Could be mutable, so assign copy
                self.defaults.value = value.copy()
            except AttributeError:
                # Immutable, so just assign value
                self.defaults.value = value

    def _parse_and_instantiate_mechanism_entries(self, pathway, context=None):

# FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process._validate_params
        # - make sure first entry is not a Projection
        # - make sure Projection entries do NOT occur back-to-back (i.e., no two in a row)
        # - instantiate Mechanism entries

        previous_item_was_projection = False

        for i in range(len(pathway)):
            item = pathway[i]

            # VALIDATE PLACEMENT OF PROJECTION ENTRIES  ----------------------------------------------------------

            # Can't be first entry, and can never have two in a row

            # Config entry is a Projection
            if _is_projection_spec(item, proj_type=Projection):
                # Projection not allowed as first entry
                if i==0:
                    raise ProcessError("Projection cannot be first entry in pathway ({0})".format(self.name))
                # Projections not allowed back-to-back
                if previous_item_was_projection:
                    raise ProcessError("Illegal sequence of two adjacent projections ({0}:{1} and {1}:{2})"
                                       " in pathway for {3}".
                                       format(i - 1, pathway[i - 1], i, pathway[i], self.name))
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
                raise ProcessError("Entry {0} ({1}) is not a recognized form of Mechanism specification".format(i, mech))
                # Params in mech tuple must be a dict or None
                # if params and not isinstance(params, dict):
                #     raise ProcessError("Params entry ({0}) of tuple in item {1} of pathway for {2} is not a dict".
                #                           format(params, i, self.name))
                # Replace Pathway entry with new tuple containing instantiated Mechanism object and params
                pathway[i] = mech


            # Entry IS already a Mechanism object
            # Add entry to _mechs and name to mechanism_names list
            # Add Process to the mechanism's list of processes to which it belongs
            if not self in mech.processes:
                mech._add_process(self, INTERNAL)
                self._mechs.append(pathway[i])
            # self.mechanism_names.append(mech.name)

            try:
                # previously this was only getting called for ControlMechanisms, but GatingMechanisms
                # need to activate their projections too! This is not being tested for anywhere
                mech._activate_projections_for_compositions(self)
            except AttributeError:
                pass

            # FIX: ADD RECURRENT PROJECTION AND MECHANISM
            # IMPLEMENTATION NOTE:  THIS IS A TOTAL HACK TO ALLOW SELF-RECURRENT MECHANISMS IN THE CURRENT SYSTEM
            #                       SHOULD BE HANDLED MORE APPROPRIATELY IN COMPOSITION
            # If this is the last mechanism in the pathway, and it has a self-recurrent Projection,
            #    add that to the pathway so that it can be identified and assigned for learning if so specified
            if i + 1 == len(pathway):
                if mech.output_ports and any(any(proj.receiver.owner is mech
                           for proj in port.efferents)
                       for port in mech.output_ports):
                    for port in mech.output_ports:
                        for proj in port.efferents:
                            if proj.receiver.owner is mech:
                                pathway.append(proj)
                                pathway.append(pathway[i])


        # Validate initial values
        # FIX: CHECK WHETHER ALL MECHANISMS DESIGNATED AS INITALIZE HAVE AN INITIAL_VALUES ENTRY
        if self.initial_values:
            for mech, value in self.initial_values.items():
                if not mech in self.mechanisms:
                    raise SystemError("{} (entry in initial_values arg) is not a Mechanism in pathway for \'{}\'".
                                      format(mech.name, self.name))
                if not iscompatible(value, mech.defaults.variable):
                    raise SystemError("{} (in initial_values arg for {}) is not a valid value for {}".
                                      format(value,
                                             append_type_to_name(self),
                                             append_type_to_name(mech)))

    def _parse_and_instantiate_projection_entries(self, pathway, context=None):
        from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism

        # ASSIGN DEFAULT PROJECTION PARAMS

        # If learning is specified for the Process, add learning specification to default Projection params
        #    and store any learning-related specifications
        if self.learning:

            # if spec is LEARNING or ENABLED (convenience spec),
            #    change to Projection version of keyword for consistency below
            if self.learning in {LEARNING, LEARNING_PROJECTION, ENABLED}:
                self.learning = LEARNING

            # FIX: IF self.learning IS AN ACTUAL LearningProjection OBJECT, NEED TO RESPECIFY AS CLASS + PARAMS
            # FIX:     OR CAN THE SAME LearningProjection OBJECT BE SHARED BY MULTIPLE PROJECTIONS?
            # FIX:     DOES IT HAVE ANY INTERNAL PORT VARIABLES OR PARAMS THAT NEED TO BE PROJECTIONS-SPECIFIC?
            # FIX:     MAKE IT A COPY?

            matrix_spec = (self.default_projection_matrix, self.learning)
        else:
            matrix_spec = self.default_projection_matrix

        projection_params = {FUNCTION_PARAMS: {MATRIX: matrix_spec}, MATRIX: matrix_spec}

        for i in range(len(pathway)):
            item = pathway[i]
            learning_projection_specified = False
            # FIRST ENTRY

            # Must be a Mechanism (enforced above)
            # Assign input(s) from Process to it if it doesn't already have any
            # Note: does not include learning (even if specified for the process)
            if i == 0:
                # Relabel for clarity
                mech = item

                # Check if first Mechanism already has any projections and, if so, issue appropriate warning
                if mech.input_port.path_afferents:
                    self._warn_about_existing_projections_to_first_mechanism(mech, context)

                # Assign input Projection from Process
                self._assign_process_input_projections(mech, context=context)
                continue


            # SUBSEQUENT ENTRIES

            # Item is a Mechanism
            item = item
            if isinstance(item, Mechanism):

                preceding_item = pathway[i - 1]

                # PRECEDING ITEM IS A PROJECTION
                if isinstance(preceding_item, Projection):
                    if self.learning:

                        # Check if preceding_item has a matrix ParameterPort and, if so, it has any learningSignals
                        # If it does, assign them to learning_projections
                        try:
                            learning_projections = list(
                                projection for projection in preceding_item._parameter_ports[MATRIX].mod_afferents
                                if isinstance(projection, LearningProjection)
                            )

                        # FIX: 10/3/17: USE OF TUPLE AS ITEM IN port_list ARGS BELOW IS NO LONGER SUPPORTED
                        #               NEED TO REFORMAT SPECS FOR port_list BELOW
                        #               (NOTE: THESE EXCEPTIONS ARE NOT BEING CALLED IN CURRENT TEST SUITES)
                        # preceding_item doesn't have a _parameter_ports attrib, so assign one with self.learning
                        except AttributeError:
                            # Instantiate _parameter_ports Ordered dict with ParameterPort and self.learning
                            preceding_item._parameter_ports = _instantiate_port_list(
                                    owner=preceding_item,
                                    port_list=[(MATRIX, self.learning)],
                                    port_types=ParameterPort,
                                    port_Param_identifier=PARAMETER_PORT,
                                    reference_value=self.learning,
                                    reference_value_name=LEARNING_PROJECTION,
                                    context=context
                            )

                        # preceding_item has _parameter_ports but not (yet!) one for MATRIX, so instantiate it
                        except KeyError:
                            # Instantiate ParameterPort for MATRIX
                            preceding_item._parameter_ports[MATRIX] = _instantiate_port(
                                owner=preceding_item,
                                port_type=ParameterPort,
                                name=MATRIX,
                                # # FIX: NOT SURE IF THIS IS CORRECT:
                                # port_spec=PARAMETER_PORT,
                                reference_value=self.learning,
                                reference_value_name=LEARNING_PROJECTION,
                                params=self.learning,
                                context=context
                            )
                        # preceding_item has ParameterPort for MATRIX,
                        else:
                            if not learning_projections:
                                # Add learningProjection to Projection if it doesn't have one
                                projs = _add_projection_to(
                                    preceding_item,
                                    preceding_item._parameter_ports[MATRIX],
                                    projection_spec=self.learning
                                )
                                for proj in projs:
                                    proj._activate_for_compositions(self)
                                    self._add_projection(proj)
                    continue

                # Preceding item was a Mechanism, so check if a Projection needs to be instantiated between them
                # Check if Mechanism already has a Projection from the preceding Mechanism, by testing whether the
                #    preceding mechanism is the sender of any projections received by the current one's inputPort
# FIX: THIS SHOULD BE DONE FOR ALL InputPortS
# FIX: POTENTIAL PROBLEM - EVC *CAN* HAVE MULTIPLE PROJECTIONS FROM (DIFFERENT OutputPorts OF) THE SAME MECHANISM

                # PRECEDING ITEM IS A MECHANISM
                projection_list = item.input_port.path_afferents
                projection_found = False
                for projection in projection_list:
                    # Current mechanism DOES receive a Projection from the preceding item
                    # DEPRECATED: this allows any projection existing between A->B to automatically be added
                    # to this process
                    if preceding_item == projection.sender.owner:
                        projection_found = True
                        if self.learning:
                            # Make sure Projection includes a learningSignal and add one if it doesn't
                            try:
                                matrix_param_port = projection._parameter_ports[MATRIX]

                            # Projection doesn't have a _parameter_ports attrib, so assign one with self.learning
                            except AttributeError:
                                # Instantiate _parameter_ports Ordered dict with ParameterPort for self.learning
                                projection._parameter_ports = _instantiate_port_list(
                                    owner=preceding_item,
                                    port_list=[(MATRIX, self.learning)],
                                    port_types=ParameterPort,
                                    port_Param_identifier=PARAMETER_PORT,
                                    reference_value=self.learning,
                                    reference_value_name=LEARNING_PROJECTION,
                                    context=context
                                )

                            # Projection has _parameter_ports but not (yet!) one for MATRIX,
                            #    so instantiate it with self.learning
                            except KeyError:
                                # Instantiate ParameterPort for MATRIX
                                projection._parameter_ports[MATRIX] = _instantiate_port(
                                    owner=preceding_item,
                                    port_type=ParameterPort,
                                    name=MATRIX,
                                    # port_spec=PARAMETER_PORT,
                                    reference_value=self.learning,
                                    reference_value_name=LEARNING_PROJECTION,
                                    params=self.learning,
                                    context=context
                                )

                            # Check if Projection's matrix param has a learningSignal
                            else:
                                if not (
                                    any(
                                        isinstance(projection, LearningProjection)
                                        for projection in matrix_param_port.mod_afferents
                                    )
                                ):
                                    projs = _add_projection_to(
                                        projection,
                                        matrix_param_port,
                                        projection_spec=self.learning
                                    )
                                    for p in projs:
                                        p._activate_for_compositions(self)

                            if self.prefs.verbosePref:
                                print("LearningProjection added to Projection from Mechanism {0} to Mechanism {1} "
                                      "in pathway of {2}".format(preceding_item.name, item.name, self.name))

                        # remove this to enforce that projections need to be explicitly added to Compositions
                        # left in for backwards compatibility
                        # DEPRECATED
                        projection._activate_for_compositions(self)
                        # warnings.warn(
                        #     'The ability for Process to associate with itself all projections between '
                        #     'subsequent mechanisms implicitly is deprecated. In the future, you must '
                        #     'explicitly state the projections you want included in any Composition.',
                        #     FutureWarning
                        # )

                        break

                if not projection_found:
                    # No Projection found, so instantiate MappingProjection from preceding mech to current one;
                    # Note: if self.learning arg is specified, it has already been added to projection_params above

                    # MODIFIED 9/19/17 NEW:
                    #     [ALLOWS ControlMechanism AND ASSOCIATED ObjectiveMechanism TO BE ADDED TO PATHWAY)
                    # If it is a ControlMechanism with an associated ObjectiveMechanism, try projecting to that
                    if isinstance(item, ControlMechanism) and item.objective_mechanism is not None:
                        # If it already has an associated ObjectiveMechanism, make sure it has been implemented
                        if not isinstance(item.objective_mechanism, Mechanism):
                            raise ProcessError(
                                "{} included in {} for {} ({}) has an {} arugment, but it is not an {}".format(
                                    ControlMechanism.__name__,
                                    PATHWAY,
                                    self.name,
                                    item.objective_mechanism,
                                    OBJECTIVE_MECHANISM,
                                    ObjectiveMechanism.name
                                )
                            )
                        # Check whether ObjectiveMechanism already receives a projection
                        #     from the preceding Mechanism in the pathway
                        # if not any(projection.sender.owner is preceding_item
                        #            for projection in item.objective_mechanism.input_port.path_afferents):
                        item._objective_projection._activate_for_compositions(self)
                        if (
                            not any(
                                any(
                                    projection.sender.owner is preceding_item
                                    for projection in input_port.path_afferents
                                )
                                for input_port in item.objective_mechanism.input_ports
                            )
                        ):
                            # Assign projection from preceding Mechanism in pathway to ObjectiveMechanism
                            receiver = item.objective_mechanism

                        else:
                            # Ignore (ObjectiveMechanism already as a projection from the Mechanism)
                            for input_port in item.objective_mechanism.input_ports:
                                for projection in input_port.path_afferents:
                                    if projection.sender.owner is preceding_item:
                                        projection._activate_for_compositions(self)
                            continue
                    else:
                        receiver = item
                    # MODIFIED 9/19/17 END

                    projection = MappingProjection(
                        sender=preceding_item,
                        receiver=receiver,
                        params=copy.copy(projection_params),
                        name='{} from {} to {}'.format(MAPPING_PROJECTION, preceding_item.name, item.name)
                    )

                    projection._activate_for_compositions(self)
                    for mod_proj in itertools.chain.from_iterable([p.mod_afferents for p in projection.parameter_ports]):
                        mod_proj._activate_for_compositions(self)

                    if self.prefs.verbosePref:
                        print("MappingProjection added from Mechanism {0} to Mechanism {1}"
                              " in pathway of {2}".format(preceding_item.name, item.name, self.name))

            # Item is a Projection or specification for one
            else:
                # Instantiate Projection, assigning Mechanism in previous entry as sender and next one as receiver
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
                # IMPLEMENT: MOVE Port._instantiate_projections_to_port(), _check_projection_receiver()
                #            and _parse_projection_keyword() all to Projection_Base.__init__() and call that
                #           VALIDATION OF PROJECTION OBJECT:
                #                MAKE SURE IT IS A MappingProjection
                #                CHECK THAT SENDER IS pathway[i-1][OBJECT_ITEM]
                #                CHECK THAT RECEVIER IS pathway[i+1][OBJECT_ITEM]

                # Get sender for Projection
                sender_mech = pathway[i - 1]

                # Get receiver for Projection
                try:
                    receiver_mech = pathway[i + 1]
                except IndexError:
                    # There are no more entries in the pathway
                    #    so the Projection had better project to a mechanism already in the pathway;
                    #    otherwise, raise and exception
                    try:
                        receiver_mech = item.receiver.owner
                        if receiver_mech not in [object_item for object_item in pathway]:
                            raise AttributeError
                    except AttributeError:
                        raise ProcessError(
                            "The last entry in the pathway for {} is a project specification {}, "
                            "so its receiver must be a Mechanism in the pathway".format(self.name, item)
                        )

                # # Check if there is already a projection between the sender and receiver
                # if self._check_for_duplicate_projection(sender_mech, receiver_mech, item, i):
                #     continue

                # Projection spec is an instance of a MappingProjection
                if isinstance(item, MappingProjection):
                    # Check that Projection's sender and receiver are to the mech before and after it in the list
                    # IMPLEMENT: CONSIDER ADDING LEARNING TO ITS SPECIFICATION?
                    # FIX: SHOULD MOVE VALIDATION COMPONENTS BELOW TO Process._validate_params

                    # If initialization of MappingProjection has been deferred,
                    #    check sender and receiver, assign them if they have not been assigned, and initialize it
                    if item.initialization_status == ContextFlags.DEFERRED_INIT:
                        # Check sender arg
                        try:
                            sender_arg = item._init_args[SENDER]
                        except AttributeError:
                            raise ProcessError(
                                "PROGRAM ERROR: initialization_status of {} is {} but "
                                "it does not have _init_args".format(
                                    item, ContextFlags.DEFERRED_INIT.name
                                )
                            )
                        except KeyError:
                            raise ProcessError(
                                "PROGRAM ERROR: Value of {} is {} but "
                                "_init_args does not have entry for {}".format(
                                    item._init_args[NAME], ContextFlags.DEFERRED_INIT.name, SENDER
                                )
                            )
                        else:
                            # If sender is not specified for the Projection,
                            #    assign mechanism that precedes in pathway
                            if sender_arg is None:
                                item._init_args[SENDER] = sender_mech
                            elif sender_arg is not sender_mech:
                                raise ProcessError(
                                    "Sender of Projection ({}) specified in item {} of"
                                    " pathway for {} is not the Mechanism ({}) "
                                    "that precedes it in the pathway".format(
                                        item._init_args[NAME], i, self.name, sender_mech.name
                                    )
                                )
                        # Check receiver arg
                        try:
                            receiver_arg = item._init_args[RECEIVER_ARG]
                        except AttributeError:
                            raise ProcessError(
                                "PROGRAM ERROR: initialization_status of {} is {} "
                                "but it does not have _init_args".format(
                                    item, ContextFlags.DEFERRED_INIT
                                )
                            )
                        except KeyError:
                            raise ProcessError(
                                "PROGRAM ERROR: initialization_status of {} is {} "
                                "but _init_args does not have entry for {}".format(
                                    item._init_args[NAME], ContextFlags.DEFERRED_INIT, RECEIVER_ARG
                                )
                            )
                        else:
                            # If receiver is not specified for the Projection,
                            #    assign mechanism that follows it in the pathway
                            if receiver_arg is None:
                                item._init_args[RECEIVER_ARG] = receiver_mech
                            elif receiver_arg is not receiver_mech:
                                raise ProcessError(
                                    "Receiver of Projection ({}) specified in item {} of"
                                    " pathway for {} is not the Mechanism ({}) "
                                    "that follows it in the pathway". format(
                                        item._init_args[NAME], i, self.name, receiver_mech.name
                                    )
                                )

                        # Check if it is specified for learning
                        matrix_spec = item._init_args[MATRIX]
                        if (
                            isinstance(matrix_spec, tuple)
                            and (
                                matrix_spec[1] in {LEARNING, LEARNING_PROJECTION}
                                or isinstance(matrix_spec[1], (LearningProjection, LearningSignal))
                            )
                        ):
                            self.learning = True

                        # Complete initialization of Projection
                        item._deferred_init(context=context)

                    if item.sender.owner is not sender_mech:
                        raise ProcessError("Sender of Projection ({}) specified in item {} of pathway for {} "
                                           "is not the Mechanism ({}) that precedes it in the pathway".
                                           format(item.name, i, self.name, sender_mech.name))
                    if item.receiver.owner is not receiver_mech:
                        raise ProcessError("Receiver of Projection ({}) specified in item {} of pathway for "
                                           "{} is not the Mechanism ({}) that follows it in the pathway".
                                           format(item.name, i, self.name, sender_mech.name))
                    projection = item

                    if projection.has_learning_projection:
                        self.learning = True
                    # TEST
                    # if params:
                    #     projection.matrix = params

                # Projection spec is a MappingProjection class reference
                elif inspect.isclass(item) and issubclass(item, MappingProjection):
                    # if params:
                    #     # Note:  If self.learning is specified, it has already been added to projection_params above
                    #     projection_params = params
                    projection = MappingProjection(
                        sender=sender_mech,
                        receiver=receiver_mech,
                        # params=projection_params
                    )

                # Projection spec is a matrix spec, a keyword for one, or a (matrix, LearningProjection) tuple
                # Note: this is tested above by call to _is_projection_spec()
                elif (
                    isinstance(item, (np.matrix, str, tuple))
                    or (isinstance(item, np.ndarray) and item.ndim == 2)
                ):
                    # If a LearningProjection is explicitly specified for this Projection, use it
                    if isinstance(item, tuple):
                        matrix_spec = item
                        learning_projection_specified = True
                    # If a LearningProjection is not specified for this Projection but self.learning is, use that
                    elif self.learning:
                        matrix_spec = (item, self.learning)
                    # Otherwise, do not include any LearningProjection
                    else:
                        matrix_spec = item

                    projection = MappingProjection(
                        sender=sender_mech,
                        receiver=receiver_mech,
                        matrix=matrix_spec
                    )
                else:
                    raise ProcessError(
                        "Item {0} ({1}) of pathway for {2} is not "
                        "a valid Mechanism or Projection specification".format(i, item, self.name)
                    )
                # Reassign Pathway entry
                #    with Projection as OBJECT item and original params as PARAMS item of the tuple
                # IMPLEMENTATION NOTE:  params is currently ignored
                pathway[i] = projection

                projection._activate_for_compositions(self)

        if learning_projection_specified:
            self.learning = LEARNING

    def _check_for_duplicate_projection(self, sndr_mech, rcvr_mech, proj_spec, pathway_index):
        """Check if there is already a projection between sndr_mech and rcvr_mech
        If so:
            - if it has just found the same project (e.g., as in case of AutoAssociativeProjection), let pass
            - otherwise:
                - if verbosePref, warn
                - replace proj_spec with existing projection
        """

        for input_port in rcvr_mech.input_ports:
            for proj in input_port.path_afferents:
                if proj.sender.owner is sndr_mech:
                    # Skip recurrent projections
                    try:
                        if self.pathway[pathway_index] == proj:
                            continue
                    except:
                        pass
                    if self.prefs.verbosePref:
                        print("WARNING: Duplicate {} specified between {} and {} ({}) in {}; it will be ignored".
                              format(Projection.__name__, sndr_mech.name, rcvr_mech.name, proj_spec, self.name))
                    self.pathway[pathway_index] = proj
                    return True
        return False

    def _warn_about_existing_projections_to_first_mechanism(self, mechanism, context=None):

        # Check where the Projection(s) is/are from and, if verbose pref is set, issue appropriate warnings
        for projection in mechanism.input_port.all_afferents:

            # Projection to first Mechanism in Pathway comes from a Process input
            if isinstance(projection.sender, ProcessInputPort):
                # If it is:
                # (A) from self, ignore
                # (B) from another Process, warn if verbose pref is set
                if not projection.sender.owner is self:
                    if self.prefs.verbosePref:
                        print("WARNING: {0} in pathway for {1} already has an input from {2} that will be used".
                              format(mechanism.name, self.name, projection.sender.owner.name))
                    return

            # (C) Projection to first Mechanism in Pathway comes from one in the Process' _mechs;
            #     so warn if verbose pref is set
            if projection.sender.owner in self._mechs:
                if self.prefs.verbosePref:
                    print("WARNING: first Mechanism ({0}) in pathway for {1} receives "
                          "a (recurrent) Projection from another Mechanism {2} in {1}".
                          format(mechanism.name, self.name, projection.sender.owner.name))

            # Projection to first Mechanism in Pathway comes from a Mechanism not in the Process;
            #    check if Process is in a System, and Projection is from another Mechanism in the System
            else:
                try:
                    if (inspect.isclass(context) and issubclass(context, System_Base)):
                        # Relabel for clarity
                        system = context
                    else:
                        system = None
                except:
                    # Process is NOT being implemented as part of a System, so Projection is from elsewhere;
                    #  (D)  Issue warning if verbose
                    if self.prefs.verbosePref:
                        print("WARNING: first Mechanism ({0}) in pathway for {1} receives a "
                              "Projection ({2}) that is not part of {1} or the System it is in".
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
                            print("First Mechanism ({0}) in pathway for {1}"
                                  " receives a Projection {2} that is not in {1} "
                                  "or its System ({3}); it will be ignored and "
                                  "a Projection assigned to it by {3}".
                                  format(mechanism.name,
                                         self.name,
                                         projection.sender.owner.name,
                                         context.name))
                    # Process is being implemented in something other than a System
                    #    so warn (irrespecive of verbose)
                    elif self.verbosePref:
                        print("WARNING:  Process ({}) is being instantiated outside of a System".
                              format(self.name))

    def _assign_process_input_projections(self, mechanism, context=None):
        """Create Projection(s) for each item in Process input to InputPort(s) of the specified Mechanism

        For each item in Process input:
        - create process_input_port, as sender for MappingProjection to the ORIGIN Mechanism.input_port
        - create the MappingProjection (with process_input_port as sender, and ORIGIN Mechanism as receiver)

        If number of Process inputs == len(ORIGIN Mechanism.defaults.variable):
            - create one Projection for each of the ORIGIN Mechanism.input_port(s)
        If number of Process inputs == 1 but len(ORIGIN Mechanism.defaults.variable) > 1:
            - create a Projection for each of the ORIGIN Mechanism.input_ports, and provide Process' input to each
        If number of Process inputs > 1 but len(ORIGIN Mechanism.defaults.variable) == 1:
            - create one Projection for each Process input and assign all to ORIGIN Mechanism.input_port
        Otherwise,  if number of Process inputs != len(ORIGIN Mechanism.defaults.) and both > 1:
            - raise exception:  ambiguous mapping from Process input values to ORIGIN Mechanism's input_ports

        :param Mechanism:
        :return:
        """

        # FIX: LENGTH OF EACH PROCESS INPUTPORT SHOUD BE MATCHED TO LENGTH OF INPUTPORT FOR CORRESPONDING ORIGIN MECHANISM

        process_input = self.defaults.variable

        # Get number of Process inputs
        num_process_inputs = len(process_input)

        # Get number of mechanism.input_ports
        #    - assume mechanism.defaults.variable is a 2D np.array, and that
        #    - there is one inputPort for each item (1D array) in mechanism.defaults.variable
        num_mechanism_input_ports = len(mechanism.defaults.variable)

        # There is a mismatch between number of Process inputs and number of mechanism.input_ports:
        if num_process_inputs > 1 and num_mechanism_input_ports > 1 and num_process_inputs != num_mechanism_input_ports:
            raise ProcessError("Mismatch between number of input values ({0}) for {1} and "
                               "number of input_ports ({2}) for {3}".format(num_process_inputs,
                                                                            self.name,
                                                                            num_mechanism_input_ports,
                                                                            mechanism.name))

        # Create InputPort for each item of Process input, and assign to list
        for i in range(num_process_inputs):
            process_input_port = ProcessInputPort(owner=self,
                                                    variable=process_input[i],
                                                    prefs=self.prefs)
            self.process_input_ports.append(process_input_port)

        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection

        # If there is the same number of Process input values and mechanism.input_ports, assign one to each
        if num_process_inputs == num_mechanism_input_ports:
            for i in range(num_mechanism_input_ports):
                if mechanism.input_ports[i].internal_only:
                    continue
                # Insure that each Process input value is compatible with corresponding variable of mechanism.input_port
                input_port_variable = mechanism.input_ports[i].socket_template
                if not iscompatible(process_input[i], input_port_variable):
                    raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                       "variable for corresponding inputPort of {3} (format: {4})".
                                       format(i, process_input[i], self.name, mechanism.name, input_port_variable))
                # Create MappingProjection from Process InputPort to corresponding mechanism.input_port
                proj = MappingProjection(sender=self.process_input_ports[i],
                                         receiver=mechanism.input_ports[i],
                                         name=self.name + '_Input Projection')
                proj._activate_for_compositions(self)
                if self.prefs.verbosePref:
                    print("Assigned input value {0} ({1}) of {2} to corresponding inputPort of {3}".
                          format(i, process_input[i], self.name, mechanism.name))

        # If the number of Process inputs and mechanism.input_ports is unequal, but only a single of one or the other:
        # - if there is a single Process input value and multiple mechanism.input_ports,
        #     instantiate a single Process InputPort with projections to each of the mechanism.input_ports
        # - if there are multiple Process input values and a single mechanism.input_port,
        #     instantiate multiple Process input ports each with a Projection to the single mechanism.input_port
        else:
            for i in range(num_mechanism_input_ports):
                if mechanism.input_ports[i].internal_only:
                    continue
                for j in range(num_process_inputs):
                    if not iscompatible(process_input[j], mechanism.defaults.variable[i]):
                        raise ProcessError("Input value {0} ({1}) for {2} is not compatible with "
                                           "variable ({3}) for inputPort {4} of {5}".
                                           format(j, process_input[j], self.name,
                                                  mechanism.defaults.variable[i], i, mechanism.name))
                    # Create MappingProjection from Process buffer_intput_port to corresponding mechanism.input_port
                    proj = MappingProjection(sender=self.process_input_ports[j],
                            receiver=mechanism.input_ports[i],
                            name=self.name + '_Input Projection')
                    proj._activate_for_compositions(self)
                    if self.prefs.verbosePref:
                        print("Assigned input value {0} ({1}) of {2} to inputPort {3} of {4}".
                              format(j, process_input[j], self.name, i, mechanism.name))

        mechanism._receivesProcessInput = True

    def _assign_input_values(self, input, context=None):
        """Validate input, assign each item (1D array) in input to corresponding process_input_port

        Returns converted version of input

        Args:
            input:

        Returns:

        """

        if isinstance(input, dict):
            input = list(input.values())[0]
        # Validate input
        if input is None:
            input = self.first_mechanism.defaults.variable
            if (
                self.prefs.verbosePref
                and not (
                    context.source == ContextFlags.COMMAND_LINE
                    or self.initializaton_status == ContextFlags.INITIALIZING
                )
            ):
                print("- No input provided;  default will be used: {0}")

        else:
            # Insure that input is a list of 1D array items, one for each processInputPort
            # If input is a single number, wrap in a list
            from numpy import ndarray
            if isinstance(input, numbers.Number) or (isinstance(input, ndarray) and input.ndim == 0):
                input = [input]
            # If input is a list of numbers, wrap in an outer list (for processing below)
            if all(isinstance(i, numbers.Number) for i in input):
                input = [input]

        if len(self.process_input_ports) != len(input):
            raise ProcessError("Length ({}) of input to {} does not match the number "
                               "required for the inputs of its origin Mechanisms ({}) ".
                               format(len(input), self.name, len(self.process_input_ports)))

        # Assign items in input to value of each process_input_port
        for i in range(len(self.process_input_ports)):
            self.process_input_ports[i].parameters.value._set(input[i], context)

        return input

    def _update_input(self):
        for s, i in zip(self.process_input_ports, range(len(self.process_input_ports))):
            self.input = s.value

    def _instantiate__deferred_inits(self, context=None):
        """Instantiate any objects in the Process that have deferred their initialization

        Description:
            For learning:
                go through _mechs in reverse order of pathway since
                    LearningProjections are processed from the output (where the training signal is provided) backwards
                exhaustively check all of Components of each Mechanism,
                    including all projections to its input_ports and _parameter_ports
                initialize all items that specified deferred initialization
                construct a _learning_mechs of Mechanism tuples (mech, params):
                add _learning_mechs to the Process' _mechs
                assign input Projection from Process to first Mechanism in _learning_mechs

        IMPLEMENTATION NOTE: assume that the only Projection to a Projection is a LearningProjection
                             this is implemented to be fully general, but at present may be overkill
                             since the only objects that currently use deferred initialization are LearningProjections
        """

        # For each mechanism in the Process, in backwards order through its _mechs
        for item in reversed(self._mechs):
            mech = item
            mech._deferred_init(context=context)

            # For each inputPort of the mechanism
            for input_port in mech.input_ports:
                input_port._deferred_init(context=context)
                # Restrict projections to those from mechanisms in the current process
                projections = []
                for projection in input_port.all_afferents:
                    try:
                        if self in projection.sender.owner.processes:
                            projections.append(projection)
                            self._add_projection(projection)
                    except AttributeError:
                        pass
                self._instantiate__deferred_init_projections(projections, context=context)

            # For each ParameterPort of the mechanism
            for parameter_port in mech._parameter_ports:
                parameter_port._deferred_init(context=context)
                # MODIFIED 5/2/17 OLD:
                # self._instantiate__deferred_init_projections(parameter_port.path_afferents)
                # MODIFIED 5/2/17 NEW:
                # Defer instantiation of ControlProjections to System
                #   and there should not be any other type of Projection to the ParameterPort of a Mechanism
                from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
                if not all(isinstance(proj, ControlProjection) for proj in parameter_port.mod_afferents):
                    raise ProcessError("PROGRAM ERROR:  non-ControlProjection found to ParameterPort for a Mechanism")
                # MODIFIED 5/2/17 END

        # CHANGYAN NOTE: check this spot
        # Label monitoring mechanisms and add _learning_mechs to _mechs for execution
        if self._learning_mechs:

            # Add designations to newly created LearningMechanism:
            for object_item in self._learning_mechs:
                mech = object_item
                # If
                # - mech is a TARGET ObjectiveMechanism, and
                # - the mech that projects to mech is a TERMINAL for the current process, and
                # - current process has learning specified
                # then designate mech as a TARGET
                if (isinstance(mech, ObjectiveMechanism) and
                        # any(projection.sender.owner.processes[self] == TERMINAL
                        #     for projection in mech.input_ports[SAMPLE].path_afferents) and
                        mech._learning_role is TARGET and
                        self.learning
                            ):
                    object_item._add_process(self, TARGET)
                else:
                    # mech must be a LearningMechanism;
                    # If a learning_rate has been specified for the process, assign that to all LearningMechanism
                    #    for which a mechanism-specific learning_rate has NOT been assigned
                    if (
                        self.learning_rate is not None
                        and not mech.function.parameters.learning_rate._user_specified
                    ):
                        mech.function.learning_rate = self.learning_rate

                    # Assign its label
                    object_item._add_process(self, LEARNING)

            # Add _learning_mechs to _mechs
            self._mechs.extend(self._learning_mechs)

            # IMPLEMENTATION NOTE:
            #   LearningMechanism are assigned _phaseSpecMax;
            #   this is so that they will run after the last ProcessingMechansisms have run

    def _instantiate__deferred_init_projections(self, projection_list, context=None):

        # For each Projection in the list
        for projection in projection_list:
            projection._deferred_init(context=context)

            # FIX:  WHY DOESN'T THE PROJECTION HANDLE THIS? (I.E., IN ITS deferred_init() METHOD?)
            # For each parameter_port of the Projection
            try:
                for parameter_port in projection._parameter_ports:
                    # Initialize each Projection to the ParameterPort (learning or control)
                    # IMPLEMENTATION NOTE:  SHOULD ControlProjections BE IGNORED HERE?
                    for param_projection in parameter_port.mod_afferents:
                        param_projection._deferred_init(context=context)
                        if isinstance(param_projection, LearningProjection):
                            # Get ObjectiveMechanism if there is one, and add to _learning_mechs
                            try:
                                objective_mechanism = projection.objective_mechanism
                            except AttributeError:
                                pass
                            else:
                                # If objective_mechanism is not already in _learning_mechs,
                                #     pack in tuple and add it
                                if objective_mechanism and not objective_mechanism in self._learning_mechs:
                                    # objective_object_item = objective_mechanism
                                    self._learning_mechs.append(objective_mechanism)
                            # Get LearningMechanism and add to _learning_mechs; raise exception if not found
                            try:
                                learning_mechanism = projection.learning_mechanism
                            except AttributeError:
                                raise ProcessError("{} is missing a LearningMechanism".format(param_projection.name))
                            else:
                                # If learning_mechanism is not already in _learning_mechs,
                                #     pack in tuple and add it
                                if (learning_mechanism and not any(learning_mechanism is object_item for
                                                                   object_item in self._learning_mechs)) :
                                    self._learning_mechs.append(learning_mechanism)
                        try:
                            lc = param_projection._learning_components
                            for proj in [
                                param_projection,
                                lc.error_projection,
                                lc._activation_mech_input_projection,
                                lc._activation_mech_output_projection,
                            ]:
                                proj._activate_for_compositions(self)
                                self._add_projection(proj)

                            for proj in lc.learning_mechanism.projections:
                                proj._activate_for_compositions(self)
                                self._add_projection(proj)
                        except AttributeError:
                            pass

            # Not all Projection subclasses instantiate ParameterPorts
            except TypeError as e:
                if 'parameterPorts' in e.args[0]:
                    pass
                else:
                    error_msg = 'Error in attempt to initialize LearningProjection ({}) for {}: \"{}\"'.\
                        format(param_projection.name, projection.name, e.args[0])
                    raise

    def _check_for_target_mechanisms(self):
        """Check for and assign TARGET ObjectiveMechanism to use for reporting error during learning.

         This should only be called if self.learning is specified
         Identify TARGET Mechanisms and assign to self.target_nodes,
             assign self to each TARGET Mechanism
             and report assignment if verbose

         Returns True of TARGET Mechanisms are found and/or assigned, else False
        """

        from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
        from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism \
            import ACTIVATION_INPUT
        def trace_learning_objective_mechanism_projections(mech):
            """Recursively trace projections to Objective mechanisms;
                   return TARGET ObjectiveMechanism if one is found upstream;
                   return None if no TARGET ObjectiveMechanism is found.
            """
            for input_port in mech.input_ports:
                for projection in input_port.path_afferents:
                    sender = projection.sender.owner
                    # If Projection is not from another ObjectiveMechanism, ignore
                    if not isinstance(sender, (ObjectiveMechanism)):
                        continue
                    if isinstance(sender, ObjectiveMechanism) and sender._learning_role is TARGET:
                        return sender
                    if sender.input_ports:
                        target_mech = trace_learning_objective_mechanism_projections(sender)
                        if target_mech:
                            return target_mech
                        else:
                            continue
                    else:
                        continue

        if not self.learning:
            raise ProcessError("PROGRAM ERROR: _check_for_target_mechanisms should only be called"
                               " for a process if it has a learning specification")

        target_mechs = list(object_item
                           for object_item in self._mechs
                            if (isinstance(object_item, ObjectiveMechanism) and
                                object_item._learning_role is TARGET))

        if target_mechs:

            # self.target_nodes = target_mechs
            self._target_mechs = target_mechs
            if self.prefs.verbosePref:
                print("\'{}\' assigned as TARGET Mechanism(s) for \'{}\'".
                      format([mech.name for mech in self._target_mechs], self.name))
            return True


        # No target_mechs already specified, so get from learning_mechanism
        elif self._learning_mechs:
            last_learning_mech  = self._learning_mechs[0]

            # Trace projections to first learning ObjectiveMechanism, which is for the last mechanism in the process,
            #   unless TERMINAL mechanism of process is part of another process that has learning implemented
            #    in which case, shouldn't assign target ObjectiveMechanism, but rather just a LearningMechanism)
            # NOTE: ignores need for ObjectiveMechanism for AutoAssociativeLearning
            try:
                target_mech = trace_learning_objective_mechanism_projections(last_learning_mech)
            except IndexError:
                raise ProcessError("Learning specified for {} but no ObjectiveMechanisms or LearningMechanism found"
                                   .format(self.name))

            if target_mech:
                if self.prefs.verbosePref:
                    warnings.warn("{} itself has no Target Mechanism, but its TERMINAL_MECHANISM ({}) "
                                  "appears to be in one or more pathways ({}) that has one".
                                                      format(self.name,
                                                             # list(self.terminalMechanisms)[0].name,
                                                             self.last_mechanism.name,
                                                             list(process.name for process in target_mech.processes)))
            # Check for AutoAssociativeLearningMechanism:
            #    its *ACTIVATION_INPUT* InputPort should receive a projection from the same Mechanism
            #    that receives a MappingProjection to which its *LEARNING_SIGNAL* projects
            elif any(projection.sender.owner in [projection.sender.owner
                                                 for projection in
                                                 last_learning_mech.input_ports[ACTIVATION_INPUT].path_afferents]
                     for projection in
                     last_learning_mech.input_ports[ACTIVATION_INPUT].path_afferents):
                pass
            else:

                raise ProcessError("PROGRAM ERROR: {} has a learning specification ({}) "
                                   "but no TARGET ObjectiveMechanism".format(self.name, self.learning))
            return True

        else:
            return False

    def _instantiate_target_input(self, context=None):

        if self.target is None:
            # target arg was not specified in Process' constructor,
            #    so use the value of the TARGET InputPort for each TARGET Mechanism as the default
            self.target = [mech.input_ports[TARGET].value for mech in self._target_mechs]
            if self.verbosePref:
                warnings.warn("Learning has been specified for {} and it has TARGET Mechanism(s), but its "
                              "\'target\' argument was not specified; default value(s) will be used ({})".
                              format(self.name, self.target))
        else:
            self.target = np.atleast_2d(self.target)

        # Create ProcessInputPort for each item of target and
        #   assign to TARGET inputPort of each item of _target_mechs
        for target_mech, target in zip(self._target_mechs, self.target):
            target_mech_target = target_mech.input_ports[TARGET]

            target = np.atleast_1d(target)

            # Check that length of process' target input matches length of TARGET Mechanism's target input
            if len(target) != len(target_mech_target.value):
                raise ProcessError("Length of target ({}) does not match length of input for TARGET Mechanism {} ({})".
                                   format(len(target),
                                          target_mech.name,
                                          len(target_mech_target.value)))

            target_input_port = ProcessInputPort(owner=self,
                                                    variable=target,
                                                    prefs=self.prefs,
                                                    name=TARGET)
            self.target_input_ports.append(target_input_port)

            # Add MappingProjection from target_input_port to ComparatorMechanism's TARGET InputPort
            from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
            proj = MappingProjection(sender=target_input_port,
                    receiver=target_mech_target,
                    name=self.name + '_Input Projection to ' + target_mech_target.name)
            proj._activate_for_compositions(self)
    # MODIFIED 8/14/17 END



    def initialize(self, context=None):
        """Assign the values specified for each Mechanism in the process' `initial_values` attribute.  This usually
        occurs at the beginning of one or a series of executions invoked by a call to the Process` `execute
        <Process.execute>` or `run <Process.run>` methods.
        """
        # FIX:  INITIALIZE PROCESS INPUTS??
        for mech, value in self.initial_values.items():
            mech.initialize(value, context)

    # correct here? happens in code but maybe should be COMMAND_LINE
    @handle_external_context(source=ContextFlags.COMPOSITION)
    def execute(
        self,
        input=None,
        target=None,
        context=None,
        base_context=Context(execution_id=None),
        runtime_params=None,
        termination_processing=None,
        termination_learning=None,

    ):
        """Execute the Mechanisms specified in the process` `pathway` attribute.

        COMMENT:
            First check that input is provided (required) and appropriate.
            Then execute each Mechanism in the order they appear in the `pathway` list.
        COMMENT

        Arguments
        ---------

        input : List[value] or ndarray: default zeroes
            specifies the value(s) of the Process' `input <Process.input>` for the `execution <Process_Execution>`;
            it is provided as the input to the `origin_mechanism <Process.origin_mechanism>` and must be compatible
            (in number and type of items) with its `variable <Mechanism_Base.variable>` attribute (see
            `Process_Input_And_Output` for details).

        target : List[value] or ndarray: default None
            specifies the target value assigned to each of the `target_nodes <Process.target_nodes>` for
            the `execution <Process_Execution>`.  Each item is assigned to the *TARGET* `InputPort
            <ComparatorMechanism_Structure>` of the corresponding `ComparatorMechanism` in `target_nodes
            <Process.target_nodes>`; the number of items must equal the number of items in
            `target_nodes <Process.target_nodes>`, and each item of **target** be compatible with the
            `variable <InputPort.variable>` attribute of the *TARGET* `InputPort <ComparatorMechanism_Structure>`
            for the corresponding `ComparatorMechanism` in `target_nodes <Process.target_nodes>`.

        params : Dict[param keyword: param value] :  default None
            a `parameter dictionary <ParameterPort_Specification>` that can include any of the parameters used
            as arguments to instantiate the object. Use parameter's name as the keyword for its entry.  Values specified
            for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

        COMMENT:
            context : str : default EXECUTING + self.name
                a string used for contextualization of instantiation, hierarchical calls, executions, etc.
        COMMENT

        Returns
        -------

        output of Process : ndarray
            the `value <OutputPort.value>` of the `primary OutputPort <OutputPort_Primary>` of the
            `terminal_mechanism <Process.terminal_mechanism>` of the Process.

        COMMENT:
        output of Process : list
            value of the Process' `output <Process.output>` attribute (same as the `output_values
            <Mechanism_Base.output_values>` attribute of the `terminal_mechanism <Process.terminal_mechanism>`.
        COMMENT

        COMMENT:
           IMPLEMENTATION NOTE:
           Still need to:
           * coordinate execution of multiple processes (in particular, Mechanisms that appear in more than one process)
           * deal with different time scales
        COMMENT

        """
        from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import LearningMechanism

        if context.execution_id is None:
            context.execution_id = self.default_execution_id

        context.composition = self

        # initialize from base context but don't overwrite any values already set for this context
        self._initialize_from_context(context, base_context, override=False)

        # Report output if reporting preference is on and this is not an initialization run
        report_output = self.prefs.reportOutputPref and self.initialization_status == ContextFlags.INITIALIZED

        # FIX: CONSOLIDATE/REARRANGE _assign_input_values, _check_args, AND ASSIGNMENT OF input TO variable
        # FIX: (SO THAT assign_input_value DOESN'T HAVE TO RETURN input

        variable = self._assign_input_values(input=input, context=context)

        self._check_args(variable, runtime_params, context=context)

        # Use Process self.input as input to first Mechanism in Pathway
        self.parameters.input._set(variable, context)

        # Generate header and report input
        if report_output:
            self._report_process_initiation(input=variable, separator=True)

        # Execute each Mechanism in the pathway, in the order listed, except those used for learning
        for mechanism in self._mechs:
            if (isinstance(mechanism, LearningMechanism) or
                    (isinstance(mechanism, ObjectiveMechanism) and mechanism._role is LEARNING)):
                continue

            # Execute Mechanism
            # Note:  DON'T include input arg, as that will be resolved by mechanism from its sender projections
            context.source = ContextFlags.PROCESS
            context.execution_phase = ContextFlags.PROCESSING
            mechanism.execute(context=context)
            context.execution_phase = ContextFlags.IDLE

            if report_output:
                # FIX: USE clamp_input OPTION HERE, AND ADD HARD_CLAMP AND SOFT_CLAMP
                self._report_mechanism_execution(mechanism)

            if mechanism is self.first_mechanism and not self.clamp_input:
                # Zero self.input to first mechanism after first run
                #     in case it is repeated in the pathway or receives a recurrent Projection
                variable = variable * 0

        # Execute LearningMechanisms
        if self._learning_enabled:
            self._execute_learning(context=context, target=target)

        if report_output:
            self._report_process_completion(separator=True, context=context)

        # FIX:  WHICH SHOULD THIS BE?
        return self.output_port.parameters.value._get(context)
        # return self.output

    def _execute_learning(self, context=None, target=None):
        """Update each LearningProjection for mechanisms in _mechs of process

        # Begin with Projection(s) to last Mechanism in _mechs, and work backwards
        """

        # FIRST, assign targets

        # If target was provided to execute, use that;  otherwise, will use value provided on instantiation
        #
        if target is not None:
            target = np.atleast_2d(target)
        else:
            target = self.target

        # If targets were specified as a function in call to Run() or in System (and assigned to self.targets),
        #  call the function now (i.e., after execution of the pathways, but before learning)
        #  and assign value to self.target (that will be used below to assign values to target_input_ports)
        # Note:  this accommodates functions that predicate the target on the outcome of processing
        #        (e.g., for rewards in reinforcement learning)
        if isinstance(target, types.FunctionType):
            target = target()

        # If target itself is callable, call that now
        if callable(target):
            target = target()

        # Assign items of self.target to target_input_ports
        #   (ProcessInputPorts that project to corresponding target_nodes for the Process)
        for i, target_input_port in zip(range(len(self.target_input_ports)), self.target_input_ports):
            target_input_port.parameters.value._set(target[i], context)

        # # Zero any input from projections to target(s) from any other processes
        for target_mech in self.target_mechanisms:
            for process in target_mech.processes:
                if process is self:
                    continue
                for target_input_port in  process.target_input_ports:
                    target_input_port.value *= 0

        # THEN, execute ComparatorMechanism and LearningMechanism
        context.add_flag(ContextFlags.LEARNING_MODE)
        context.add_flag(ContextFlags.LEARNING)
        for mechanism in self._learning_mechs:
            mechanism.execute(context=context)

        # FINALLY, execute LearningProjections to MappingProjections in the process' pathway
        for mech in self._mechs:
            # IMPLEMENTATION NOTE:
            #    This implementation restricts learning to ParameterPorts of projections to input_ports
            #    That means that other parameters (e.g. object or function parameters) are not currenlty learnable

            # For each inputPort of the mechanism
            for input_port in mech.input_ports:
                # For each Projection in the list
                for projection in input_port.path_afferents:

                    # Skip learning if Projection is an input from the Process or a system
                    # or comes from a mechanism that belongs to another process
                    #    (this is to prevent "double-training" of projections from mechanisms belonging
                    #     to different processes when call to _execute_learning() comes from a System)
                    sender = projection.sender.owner
                    if isinstance(sender, Process) or not self in (sender.processes):
                        continue

                    # Call parameter_port._update with LEARNING in context to update LearningSignals
                    # Note: context is set on the projection,
                    #    as the ParameterPorts are assigned their owner's context in their update methods
                    # Note: do this rather just calling LearningSignals directly
                    #       since parameter_port._update() handles parsing of LearningProjection-specific params

                    # For each parameter_port of the Projection
                    try:
                        for parameter_port in projection._parameter_ports:

                            # Skip learning if the LearningMechanism to which the LearningProjection belongs is disabled
                            if all(projection.sender.owner.learning_enabled is False
                                   for projection in parameter_port.mod_afferents):
                                continue

                            # NOTE: This will need to be updated when runtime params are re-enabled
                            # parameter_port._update(params=params, context=context)
                            parameter_port._update(context=context)

                    # Not all Projection subclasses instantiate ParameterPorts
                    except AttributeError as e:
                        if e.args[0] == '_parameter_ports':
                            pass
                        else:
                            raise ProcessError("PROGRAM ERROR: unrecognized attribute (\'{}\') encountered "
                                               "while attempting to update {} {} of {}".
                                               format(e.args[0], parameter_port.name, ParameterPort.__name__,
                                                      projection.name))

        context.remove_flag(ContextFlags.LEARNING)
        context.remove_flag(ContextFlags.LEARNING_MODE)

    def run(self,
            inputs,
            num_trials=None,
            initialize=False,
            initial_values=None,
            targets=None,
            learning=None,
            call_before_trial=None,
            call_after_trial=None,
            call_before_time_step=None,
            call_after_time_step=None
    ):
        """Run a sequence of executions

        COMMENT:
            Call execute method for each execution in a sequence specified by the `inputs` argument (required).
            See `Run` for details of formatting input specifications.
        COMMENT

        Arguments
        ---------

        inputs : List[input] or ndarray(input) : default default_variable for a single execution
            specifies the input used to `execute <Process_Execution>` the Process for each `TRIAL` in a sequence of
            executions  (see `Run` for formatting requirements and options). Each item of the outermost level (if a
            nested list) or axis 0 (if an ndarray) is assigned as the `input <Process.input>` to the Process for the
            corresponding `TRIAL`, and therefore must be compatible (in number and type of items) with the `variable
            <Mechanism_Base.variable>` of the `origin_mechanism <Process.origin_mechanism>` for the Process. If the
            number of items is less than **num_trials**, the **inputs** are cycled until the number of `TRIALS`\\s
            specified in **num_trials** has been executed.

        num_trials : int : default None
            number of `TRIAL`\\s to execute.  If the number exceeds the number of **inputs** specified, they are cycled
            until the number of `TRIALS`\\s specified in **num_trials** has been executed.

        initialize : bool default False
            specifies whether to call the Process` `initialize <Process.initialize>` method before executing
            each `TRIAL`;  if it is `False`, then `initialize <Process.initialize>` is called only *once*,
            before the first `TRIAL` executed.

        initial_values : ProcessingMechanism, List[input] or np.ndarray(input)] : default None
            specifies the values used to initialize `ProcessingMechanisms <ProcessingMechanism>` designated as
            `INITIALIZE_CYCLE` whenever the Process' `initialize <Process.initialize>` method is called. The key
            for each entry must be a ProcessingMechanism `designated <Process_Mechanism_Initialize_Cycle>`
            `INITIALIZE_CYCLE`, and the value must be a number, list or np.array that is compatible with the format
            of the ProcessingMechanism's `value <Mechanism_Base.value>` attribute. ProcessingMechanisms designated as
            `INITIALIZE_CYCLE` but not specified in **initial_values** are initialized with the value of their
            `variable <Mechanism_Base.variable>` attribute (the default input for that Mechanism).

        targets : List[input] or np.ndarray(input) : default None
            specifies the target value assigned to each of the `target_nodes <Process.target_nodes>` in
            each `TRIAL` of execution.  Each item of the outermost level (if a nested list) or axis 0 (if an ndarray)
            corresponds to a single `TRIAL`;  the number of items must equal the number of items in the **inputs**
            argument.  Each item is assigned to the *TARGET* `InputPort <ComparatorMechanism_Structure>` of the
            corresponding `ComparatorMechanism` in `target_nodes <Process.target_nodes>`; the number of
            items must equal the number of items in `target_nodes <Process.target_nodes>`, and each item
            of **target** be compatible with the `variable <InputPort.variable>` attribute of the *TARGET* `InputPort
            <ComparatorMechanism_Structure>` for the corresponding `ComparatorMechanism` in `target_nodes
            <Process.target_nodes>`.

        learning : bool :  default None
            enables or disables `learning <Process_Execution_Learning>` during execution.
            If it is not specified, its current value (from possible prior assignment) is left intact.
            If `True`, learning is forced on; if `False`, learning is forced off.

        call_before_trial : Function : default None
            called before each `TRIAL` in the sequence is executed.

        call_after_trial : Function : default None
            called after each `TRIAL` in the sequence is executed.

        call_before_time_step : Function : default None
            called before each `TIME_STEP` of each trial is executed.

        call_after_time_step : Function : default None
            called after each `TIME_STEP` of each trial is executed.

        Returns
        -------

        <Process>.results : List[OutputPort.value]
            list of the `value <OutputPort.value>`\\s of the `primary OutputPort <OutputPort_Primary>` for the
            `terminal_mechanism <Process.terminal_mechanism>` of the Process returned for each execution.

        """

        if initial_values is None and self.initial_values:
            initial_values = self.initial_values

        from psyneulink.core.globals.environment import run
        return run(self,
                   inputs=inputs,
                   num_trials=num_trials,
                   initialize=initialize,
                   initial_values=initial_values,
                   targets=targets,
                   learning=learning,
                   call_before_trial=call_before_trial,
                   call_after_trial=call_after_trial,
                   call_before_time_step=call_before_time_step,
                   call_after_time_step=call_after_time_step)

    def _report_process_initiation(self, input=None, separator=False):
        """
        Parameters
        ----------
        input : ndarray
            input to ORIGIN Mechanism for current execution.  By default, it is the value specified by the
            `ProcessInputPort <ProcessInputPort>` that projects to the ORIGIN Mechanism.  Used by system to specify
            the input from the `SystemInputPort <SystemInputPort>` when the ORIGIN Mechanism is executed as part of
            that System.

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
                     re.sub(r'[\[,\],\n]','',str(self.mechanism_names))))
        if input is None:
            input = self.input
        print("- input: {}".format(input))


    def _report_mechanism_execution(self, mechanism):
        # DEPRECATED: Reporting of mechanism execution relegated to individual mechanism prefs
        pass
        # print("\n{0} executed {1}:\n- output: {2}\n\n--------------------------------------".
        #       format(self.name,
        #              mechanism.name,
        #              re.sub('[\[,\],\n]','',
        #                     str(mechanism.outputPort.value))))

    def _report_process_completion(self, context=None, separator=False):

        print("\n\'{}' completed:\n- output: {}".
              format(append_type_to_name(self),
                     re.sub(r'[\[,\],\n]','',str([float("{:0.3}".format(float(i))) for i in self.output_port.parameters.value.get(context)]))))

        if self.learning:
            from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import MSE
            for mech in self.target_mechanisms:
                if not MSE in mech.output_ports:
                    continue
                print("\n- MSE: {:0.3}".
                      format(float(mech.output_ports[MSE].parameters.value.get(context))))

        elif separator:
            print("\n\n****************************************\n")

    def show(self, options=None):
        """Print list of all Mechanisms in the process, followed by its `ORIGIN` and `TERMINAL` Mechanisms.

        Arguments
        ---------

        COMMENT:
        options : InspectionOptions
            [TBI]
        COMMENT
        """

        # # IMPLEMENTATION NOTE:  Stub for implementing options:
        # if options and self.InspectOptions.ALL_OUTPUT_LABELS in options:
        #     pass

        print ("\n---------------------------------------------------------")
        print ("\n{}\n".format(self.name))

        print ("\tLearning enabled: {}".format(self._learning_enabled))

        # print ("\n\tMechanisms:")
        # for mech_name in self.mechanism_names:
        #     print ("\t\t{}".format(mech_name))

        print ("\n\tMechanisms:")
        for object_item in self._mechs:
            print ("\t\t{}".format(object_item.name))


        print ("\n\tOrigin Mechanism: ".format(self.name))
        print("\t\t{}".format(self.origin_mechanism.name))

        print ("\n\tTerminal Mechanism: ".format(self.name))
        print("\t\t{}".format(self.terminal_mechanism.name))
        for output_port in self.terminal_mechanism.output_ports:
            print("\t\t\t{0}".format(output_port.name))

        print ("\n---------------------------------------------------------")

    def _add_projection(self, projection):
        self.projections.append(projection)

    @property
    def function(self):
        return self.execute

    @property
    def mechanisms(self):
        return self._all_mechanisms.mechanisms

    @property
    def mechanism_names(self):
        return self._all_mechanisms.names

    @property
    def output_port(self):
        return self.last_mechanism.output_port

    @property
    def output(self):
        # FIX: THESE NEED TO BE PROPERLY MAPPED
        # return np.array(list(item.value for item in self.last_mechanism.output_ports.values()))
        return self.last_mechanism.output_values

    @property
    def origin_mechanism(self):
        return self.first_mechanism

    @property
    def terminal_mechanism(self):
        return self.last_mechanism

    @property
    def numPhases(self):
        return self._phaseSpecMax + 1

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self._mechs,
            self._learning_mechs,
            self.projections,
        ))


class ProcessInputPort(OutputPort):
    """Represents inputs and targets specified in a call to the Process' `execute <Process.execute>` and `run
    <Process.run>` methods.

    COMMENT:
        Each instance encodes one of the following:
        - an input to the Process and provides it to a `MappingProjection` that projects to one or more
            `input_ports <Mechanism_Base.input_ports>` of the `ORIGIN` Mechanism in the process.
        - a target to the Process (also a 1d array) and provides it to a `MappingProjection` that
             projects to the `TARGET` Mechanism of the process.
    COMMENT

    .. _ProcessInputPort:

    A ProcessInputPort is created for each `InputPort` of the `origin_mechanism`, and for the *TARGET* `InputPort
    <ComparatorMechanism_Structure>` of each `ComparatorMechanism <ComparatorMechanism>` listed in `target_nodes
    <Process.target_nodes>`.  A `MappingProjection` is created that projects to each of these InputPorts
    from the corresponding ProcessingInputPort.  When the Process' `execute <Process.execute>` or
    `run <Process.run>` method is called, each item of its **inputs** and **targets** arguments is assigned as
    the `value <ProcessInputPort.value>` of a ProcessInputPort, which is then conveyed to the
    corresponding InputPort of the `origin_mechanism <Process.origin_mechanism>` and `terminal_mechanisms
    <Process.terminal_mechanisms>`.  See `Process_Input_And_Output` for additional details.

    COMMENT:
    .. Declared as a sublcass of OutputPort so that it is recognized as a legitimate sender to a Projection
       in Projection_Base._instantiate_sender()

       value is used to represent the corresponding item of the input arg to process.execute or process.run
    COMMENT

    """
    class Parameters(OutputPort.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ProcessInputPort.variable>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True

                value
                    see `value <ProcessInputPort.value>`

                    :default value: numpy.array([0])
                    :type: ``numpy.ndarray``
                    :read only: True
        """
        # just grabs input from the process
        variable = Parameter(np.array([0]), read_only=True, pnl_internal=True, constructor_argument='default_variable')
        value = Parameter(np.array([0]), read_only=True, pnl_internal=True)

    def __init__(self, owner=None, variable=None, name=None, prefs=None):
        """Pass variable to MappingProjection from Process to first Mechanism in Pathway

        :param variable:
        """
        if not name:
            self.name = owner.name + "_" + kwProcessInputPort
        else:
            self.name = owner.name + "_" + name
        self.prefs = prefs
        self.path_afferents = []
        self.owner = owner

        self.parameters = self.Parameters(owner=self, parent=self.class_parameters)
        self.defaults = Defaults(owner=self, variable=variable, value=variable)

        self.parameters.value._set(variable, Context())

        # self.index = PRIMARY
        # self.assign = None


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
        """Return list of mechs sorted by Mechanism name"""
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
