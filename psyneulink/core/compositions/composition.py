# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Composition ************************************************************

"""
..
    Sections:
      * `Composition_Overview`

.. _Composition_Overview:

Overview
--------

Composition is the base class for objects that combine PsyNeuLink `Components <Component>` into an executable model.
It defines a common set of attributes possessed, and methods used by all Composition objects.

Composition "Nodes" are `Mechanisms <Mechanism>` and/or nested `Compositions <Composition>`. `Projections
<Projection>` connect two Nodes. The Composition's `graph <Composition.graph>` stores the structural relationships
among the Nodes of a Composition and the Projections that connect them.

The Composition's `scheduler <Scheduler>` generates an execution queue based on these structural dependencies,
allowing for other user-specified scheduling and termination conditions to be mixed in.

.. _Composition_Creation:

Creating a Composition
----------------------

A generic Composition can be created by calling the constructor, and then adding `Components <Component>` using the
following Composition methods:

    - `add_node <Composition.add_node>`
        Adds a node to the Composition
    - `add_nodes <Composition.add_nodes>`
        Adds mutiple nodes to the Composition
    - `add_projection <Composition.add_projection>`
        Adds a connection between a pair of nodes in the Composition
    - `add_projections <Composition.add_projections>`
        Adds connection between multiple pairs of nodes in the Composition
    - `add_linear_processing_pathway <Composition.add_linear_processing_pathway>`
        Adds and connects a list of nodes and/or Projections to the Composition;
        Inserts a default Projection between any adjacent Nodes

.. note::
  Only Nodes and Projections added to a Composition via the methods above constitute a Composition, even if
  other Nodes and/or Projections are constructed in the same script.

In the following script comp_0, comp_1 and comp_2 are identical, but constructed using different methods.

    *Create Mechanisms:*

    >>> import psyneulink as pnl
    >>> A = pnl.ProcessingMechanism(name='A')
    >>> B = pnl.ProcessingMechanism(name='B')
    >>> C = pnl.ProcessingMechanism(name='C')

    *Create Projections:*

    >>> A_to_B = pnl.MappingProjection(name="A-to-B")
    >>> B_to_C = pnl.MappingProjection(name="B-to-C")

    *Create Composition; Add Nodes (Mechanisms) and Projections via the add_linear_processing_pathway method:*

    >>> comp_0 = pnl.Composition(name='comp-0')
    >>> comp_0.add_linear_processing_pathway(pathway=[A, A_to_B, B, B_to_C, C])

    *Create Composition; Add Nodes (Mechanisms) and Projections via the add_nodes and add_projection methods:*

    >>> comp_1 = pnl.Composition(name='comp-1')
    >>> comp_1.add_nodes(nodes=[A, B, C])
    >>> comp_1.add_projection(projection=A_to_B)
    >>> comp_1.add_projection(projection=B_to_C)

    *Create Composition; Add Nodes (Mechanisms) and Projections via the add_node and add_projection methods:*

    >>> comp_2 = pnl.Composition(name='comp-2')
    >>> comp_2.add_node(node=A)
    >>> comp_2.add_node(node=B)
    >>> comp_2.add_node(node=C)
    >>> comp_2.add_projection(projection=A_to_B)
    >>> comp_2.add_projection(projection=B_to_C)

    *Run each Composition:*

    >>> input_dict = {A: [[[1.0]]]}
    >>> comp_0_output = comp_0.run(inputs=input_dict)
    >>> comp_1_output = comp_1.run(inputs=input_dict)
    >>> comp_2_output = comp_2.run(inputs=input_dict)

.. _Running_a_Composition:

Running a Composition
---------------------

.. _Run_Inputs:

*Inputs*
========

The :keyword:`run` method presents the inputs for each `TRIAL` to the input_states of the INPUT Nodes in
the `scope of execution <Run_Scope_of_Execution>`. These input values are specified in the **inputs** argument of a
Composition's :keyword:`execute` or :keyword:`run` method.
COMMENT:
    From KAM 2/7/19 - not sure "scope of execution" is the right phrase. To me, it implies that only a subset of the
    nodes in the Composition belong to the "scope of execution". What we want to convey (I think) is that ALL of the
    Nodes execute, but they do so in a "state" (history, parameter vals) corresponding to a particular execution id.
COMMENT

The standard way to specificy inputs is a Python dictionary in which each key is an `INPUT <NodeRole.INPUT>` Node and
each value is a list. The lists represent the inputs to the key `INPUT <NodeRole.INPUT>` Nodes, such that the i-th
element of the list represents the input value to the key Node on trial i.

.. _Run_Inputs_Fig_States:

.. figure:: _static/input_spec_states.svg
   :alt: Example input specifications with input states


Each input value must be compatible with the shape of the key `INPUT <NodeRole.INPUT>` Node's `external_input_values
<MechanismBase.external_input_values>`. As a result, each item in the list of inputs is typically a 2d list/array,
though `some shorthand notations are allowed <Input_Specification_Examples>`.

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
        ...                           default_variable=[[0.0, 0.0]])
        >>> b = pnl.TransferMechanism(name='b',
        ...                           default_variable=[[0.0], [0.0]])
        >>> c = pnl.TransferMechanism(name='c')

        >>> pathway1 = [a, c]
        >>> pathway2 = [b, c]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)
        >>> comp.add_linear_processing_pathway(pathway2)

        >>> input_dictionary = {a: [[[1.0, 1.0]], [[1.0, 1.0]]],
        ...                     b: [[[2.0], [3.0]], [[2.0], [3.0]]]}

        >>> comp.run(inputs=input_dictionary)

.. note::
    A Node's `external_input_values <MechanismBase.external_input_values>` attribute is always a 2d list in which the
    index i element is the value of the Node's index i `external_input_state <MechanismBase.external_input_states>`. In
    many cases, `external_input_values <MechanismBase.external_input_values>` is the same as `variable
    <MechanismBase.variable>`. Keep in mind that any InputStates marked as "internal" are excluded from
    `external_input_values <MechanismBase.external_input_values>`, and do not receive user-specified input values.

If num_trials is not in use, the number of inputs provided determines the number of trials in the run. For example, if
five inputs are provided for each INPUT Node, and num_trials is not specified, the Composition executes five times.

+----------------------+-------+------+------+------+------+
| Trial #              |0      |1     |2     |3     |4     |
+----------------------+-------+------+------+------+------+
| Input to Mechanism a |1.0    |2.0   |3.0   |4.0   |5.0   |
+----------------------+-------+------+------+------+------+

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a')
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        >>> comp.run(inputs=input_dictionary)

The number of inputs specified **must** be the same for all Nodes in the input dictionary (except for any Nodes for
which only one input is specified). In other words, all of the values in the input dictionary must have the same length
as each other (or length 1).

If num_trials is in use, `run` iterates over the inputs until num_trials is reached. For example, if five inputs
are provided for each `INPUT <NodeRole.INPUT>` Node, and num_trials = 7, the system executes seven times. The input
values from trials 0 and 1 are used again on trials 5 and 6, respectively.

+----------------------+-------+------+------+------+------+------+------+
| Trial #              |0      |1     |2     |3     |4     |5     |6     |
+----------------------+-------+------+------+------+------+------+------+
| Input to Mechanism a |1.0    |2.0   |3.0   |4.0   |5.0   |1.0   |2.0   |
+----------------------+-------+------+------+------+------+------+------+

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a')
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        >>> comp.run(inputs=input_dictionary,
        ...          num_trials=7)

.. _Input_Specification_Examples:

For convenience, condensed versions of the input specification described above are also accepted in the following
situations:

* **Case 1: INPUT Node has only one input state**
+--------------------------+-------+------+------+------+------+
| Trial #                  |0      |1     |2     |3     |4     |
+--------------------------+-------+------+------+------+------+
| Input to **Mechanism a** |1.0    |2.0   |3.0   |4.0   |5.0   |
+--------------------------+-------+------+------+------+------+

Complete input specification:

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a')
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> input_dictionary = {a: [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]}

        >>> comp.run(inputs=input_dictionary)

Shorthand - drop the outer list on each input because **Mechanism a** only has one input state:

        >>> input_dictionary = {a: [[1.0], [2.0], [3.0], [4.0], [5.0]]}

        >>> comp.run(inputs=input_dictionary)

Shorthand - drop the remaining list on each input because **Mechanism a**'s one input state's value is length 1:

        >>> input_dictionary = {a: [1.0, 2.0, 3.0, 4.0, 5.0]}

        >>> comp.run(inputs=input_dictionary)

* **Case 2: Only one input is provided for the INPUT Node**

+--------------------------+------------------+
| Trial #                  |0                 |
+--------------------------+------------------+
| Input to **Mechanism a** |[[1.0], [2.0]]    |
+--------------------------+------------------+

Complete input specification:

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
                                      default_variable=[[0.0], [0.0]])
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> input_dictionary = {a: [[[1.0], [2.0]]]}

        >>> comp.run(inputs=input_dictionary)

Shorthand - drop the outer list on **Mechanism a**'s input specification because there is only one trial:

        >>> input_dictionary = {a: [[1.0], [2.0]]}

        >>> comp.run(inputs=input_dictionary)

* **Case 3: The same input is used on all trials**

+--------------------------+----------------+-----------------+----------------+----------------+----------------+
| Trial #                  |0               |1                |2               |3               |4               |
+--------------------------+----------------+-----------------+----------------+----------------+----------------+
| Input to **Mechanism a** | [[1.0], [2.0]] | [[1.0], [2.0]]  | [[1.0], [2.0]] | [[1.0], [2.0]] | [[1.0], [2.0]] |
+--------------------------+----------------+-----------------+----------------+----------------+----------------+

Complete input specification:

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
        ...                           default_variable=[[0.0], [0.0]])
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> input_dictionary = {a: [[[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]]]}

        >>> comp.run(inputs=input_dictionary)
..

Shorthand - drop the outer list on **Mechanism a**'s input specification and use `num_trials` to repeat the input value

::

        >>> input_dictionary = {a: [[1.0], [2.0]]}

        >>> comp.run(inputs=input_dictionary,
        ...          num_trials=5)
..

* **Case 4: There is only one INPUT Node**

+--------------------------+-------------------+-------------------+
| Trial #                  |0                  |1                  |
+--------------------------+-------------------+-------------------+
| Input to **Mechanism a** | [1.0, 2.0, 3.0]   |  [1.0, 2.0, 3.0]  |
+--------------------------+-------------------+-------------------+

Complete input specification:

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
        ...                           default_variable=[[1.0, 2.0, 3.0]])
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> input_dictionary = input_dictionary = {a: [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}

        >>> comp.run(inputs=input_dictionary)
..

Shorthand - specify **Mechanism a**'s inputs in a list because it is the only INPUT Node

::

        >>> input_list = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        >>> comp.run(inputs=input_list)
..

.. _Run_Inputs_Interactive:

*Interactive Inputs*
====================

An alternative way to specify inputs is with a function. The function must return a dictionary that satisfies
the rules above for standard input specification. The only difference is that on each execution, the function returns
the input values for each INPUT Node for a single trial.

COMMENT:
The script below, for example, uses a function to specify inputs in order to interact with the Gym Forarger
Environment.

..
    import psyneulink as pnl

    a = pnl.TransferMechanism(name='a')
    b = pnl.TransferMechanism(name='b')

    pathway1 = [a, b]

    comp = Composition(name='comp')

    comp.add_linear_processing_pathway(pathway1)

    def input_function(env, result):
        action = np.where(result[0] == 0, 0, result[0] / np.abs(result[0]))
        env_step = env.step(action)
        observation = env_step[0]
        done = env_step[2]
        if not done:
            # NEW: This function MUST return a dictionary of input values for a single trial for each INPUT node
            return {player: [observation[player_coord_idx]],
                    prey: [observation[prey_coord_idx]]}
        return done
        return {a: [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]}

    comp.run(inputs=input_dictionary)

COMMENT


.. _Run_Scope_of_Execution:

*Execution Contexts*
====================

An *execution context* is a scope of execution which has its own set of values for Components and their `parameters
<Parameters>`. This is designed to prevent computations from interfering with each other, when Components are reused,
which often occurs when using multiple or nested Compositions, or running `simulations
<OptimizationControlMechanism_Execution>`. Each execution context is or is associated with an *execution_id*,
which is often a user-readable string. An *execution_id* can be specified in a call to `Composition.run`, or left
unspecified, in which case the Composition's `default execution_id <Composition.default_execution_id>` would be used.
When looking for values after a run, it's important to know the execution context you are interested in, as shown below.

::

        >>> import psyneulink as pnl
        >>> c = pnl.Composition()
        >>> d = pnl.Composition()
        >>> t = pnl.TransferMechanism()
        >>> c.add_node(t)
        >>> d.add_node(t)

        >>> t.execute(1)
        array([[1.]])
        >>> c.run({t: 5})
        [[array([5.])]]
        >>> d.run({t: 10})
        [[array([10.])]]
        >>> c.run({t: 20}, execution_id='custom execution id')
        [[array([20.])]]

        # context None
        >>> print(t.parameters.value.get())
        [[1.]]
        >>> print(t.parameters.value.get(c))
        [[5.]]
        >>> print(t.parameters.value.get(d))
        [[10.]]
        >>> print(t.parameters.value.get('custom execution id'))
        [[20.]]

In general, anything that happens outside of a Composition run and without an explicit setting of execution context
occurs in the `None` execution context.

.. _Composition_Controller:

Controller
----------

A Composition can be assigned a `controller <Composition.controller>`.  This is a `ModulatoryMechanism`, or a subclass
of one, that modulates the parameters of Components within the Composition.  It typically does this based on the output
of an `ObjectiveMechanism` that evaluates the value of other Mechanisms in the Composition, and provides the result to
the `controller <Composition.controller>`.

.. _Composition_Controller_Assignment:

Assigning a Controller
======================

A `controller <Composition.controller>` can be assigned either by specifying it in the **controller** argument of the
Composition's constructor, or using its `add_controller <Composition.add_controller>` method.

COMMENT:
TBI [PARALLELING SYSTEM]:
Specyfing Parameters to Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A controller can also be specified for the System, in the **controller** argument of the `System`.  This can be an
existing `ControlMechanism`, a constructor for one, or a class of ControlMechanism in which case a default
instance of that class will be created.  If an existing ControlMechanism or the constructor for one is used, then
the `OutputStates it monitors <ControlMechanism_ObjectiveMechanism>` and the `parameters it controls
<ControlMechanism_Control_Signals>` can be specified using its `objective_mechanism
<ControlMechanism.objective_mechanism>` and `control_signals <ControlMechanism.control_signals>`
attributes, respectively.  In addition, these can be specified in the **monitor_for_control** and **control_signal**
arguments of the `System`, as described below.

* **monitor_for_control** argument -- used to specify OutputStates of Mechanisms in the System that should be
  monitored by the `ObjectiveMechanism` associated with the System's `controller <System.controller>` (see
  `ControlMechanism_ObjectiveMechanism`);  these are used in addition to any specified for the ControlMechanism or
  its ObjectiveMechanism.  These can be specified in the **monitor_for_control** argument of the `System` using
  any of the ways used to specify the *monitored_output_states* for an ObjectiveMechanism (see
  `ObjectiveMechanism_Monitor`).  In addition, the **monitor_for_control** argument supports two
  other forms of specification:

  * **string** -- must be the `name <OutputState.name>` of an `OutputState` of a `Mechanism <Mechanism>` in the System
    (see third example under `System_Control_Examples`).  This can be used anywhere a reference to an OutputState can
    ordinarily be used (e.g., in an `InputState tuple specification <InputState_Tuple_Specification>`). Any OutputState
    with a name matching the string will be monitored, including ones with the same name that belong to different
    Mechanisms within the System. If an OutputState of a particular Mechanism is desired, and it shares its name with
    other Mechanisms in the System, then it must be referenced explicitly (see `InputState specification
    <InputState_Specification>`, and examples under `System_Control_Examples`).
  |
  * **MonitoredOutputStatesOption** -- must be a value of `MonitoredOutputStatesOption`, and must appear alone or as a
    single item in the list specifying the **monitor_for_control** argument;  any other specification(s) included in
    the list will take precedence.  The MonitoredOutputStatesOption applies to all of the Mechanisms in the System
    except its `controller <System.controller>` and `LearningMechanisms <LearningMechanism>`. The
    *PRIMARY_OUTPUT_STATES* value specifies that the `primary OutputState <OutputState_Primary>` of every Mechanism be
    monitored, whereas *ALL_OUTPUT_STATES* specifies that *every* OutputState of every Mechanism be monitored.
  |
  The default for the **monitor_for_control** argument is *MonitoredOutputStatesOption.PRIMARY_OUTPUT_STATES*.
  The OutputStates specified in the **monitor_for_control** argument are added to any already specified for the
  ControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>`, and the full set is listed in
  the ControlMechanism's `monitored_output_states <EVCControlMechanism.monitored_output_states>` attribute, and its
  ObjectiveMechanism's `monitored_output_states <ObjectiveMechanism.monitored_output_states>` attribute).
..
* **control_signals** argument -- used to specify the parameters of Components in the System to be controlled. These
  can be specified in any of the ways used to `specify ControlSignals <ControlMechanism_Control_Signals>` in the
  *control_signals* argument of a ControlMechanism. These are added to any `ControlSignals <ControlSignal>` that have
  already been specified for the `controller <System.controller>` (listed in its `control_signals
  <ControlMechanism.control_signals>` attribute), and any parameters that have directly been `specified for
  control <ParameterState_Specification>` within the System (see `System_Control` below for additional details).
COMMENT

.. _Composition_Controller_Execution:

Controller Execution
====================

The `controller <Composition.controller>` is executed only if the Composition's `enable_controller
<Composition.enable_controller>` attribute is True.  This generally done automatically when the `controller
<Composition.controller>` is `assigned <Composition_Controller_Assignment>`.  If enabled, the `controller
<Composition.controller>` is generally executed either before or after all of the other Components in the Composition
have been executed, as determined by the Composition's `controller_mode <Composition.controller_mode>` attribute.
However, the Composition's `controller_condition <Composition.controller_condition>` attribute can be used to
customize when it is executed.  All three of these attributes can be specified in corresponding arguments of the
Composition's constructor, or programmatically after it is constructed by assigning the desired value to the
attribute.

For Developers
--------------

.. _Run_Execution_Contexts_Init:

Initialization of Execution Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The parameter values for any execution context can be copied into another execution context by using \
Component._initialize_from_context, which when called on a Component copies the values for all its parameters \
and recursively for all of the Component's `_dependent_components <Component._dependent_components>`

- `_dependent_components <Component._dependent_components>` should be added to for any new Component that requires \
other Components to function properly (beyond "standard" things like Component.function, \
or Mechanism.input_states, as these are added in the proper classes' _dependent_components)
    - the intent is that with ``_dependent_components`` set properly, calling \
    ``obj._initialize_from_context(new_execution_id, base_execution_id)`` should be sufficient to run obj \
    under **new_execution_id**
    - a good example of a "nonstandard" override is `OptimizationControlMechanism._dependent_components`

Debugging Tips
^^^^^^^^^^^^^^
If you receive an error like below, while checking for a context value for example,

::

    self.parameters.context._get(execution_id).execution_phase == ContextStatus.PROCESSING
    AttributeError: 'NoneType' object has no attribute 'execution_phase'

this means that there was no context value found for execution_id, and can be indicative that execution_id
was not initialized to the values of another execution context, which normally happens during execution.
See `Execution Contexts initialization <Run_Execution_Contexts_Init>`.

.. _Run_Timing:

*Timing*
========

When `run <Composition.run>` is called by a Composition, it calls that Composition's `execute <Composition.execute>`
method once for each `input <Run_Inputs>`  (or set of inputs) specified in the call to `run <Composition.run>`,
which constitutes a `TRIAL` of execution.  For each `TRIAL`, the Component makes repeated `calls to its Scheduler
<Scheduler_Execution>`, executing the Components it specifies in each `TIME_STEP`, until every Component has been
executed at least once or
another `termination condition <Scheduler_Termination_Conditions>` is met.  The `Scheduler` can be used in combination
with `Condition` specifications for individual Components to execute different Components at different time scales.

Runtime Params


.. _Visualizing_a_Composition:

Visualizing a Composition
-------------------------

The `show_graph <Composition.show_graph>` method generates a display of the graph structure of Nodes (Mechanisms and
Nested Compositions) and Projections in the Composition (based on the Composition's `processing graph
<Composition.processing_graph>`).

By default, Nodes are shown as ovals labeled by their `names <Mechanism.name>`, with the Composition's `INPUT
<NodeRole.INPUT>` Mechanisms shown in green, its `OUTPUT <NodeRole.OUTPUT>` Mechanisms shown in red, and Projections
shown as unlabeled arrows, as illustrated for the Composition in the example below:

.. _System_show_graph_basic_figure:

+-----------------------------------------------------------+-------------------------------------------+
| >>> from psyneulink import *                              | .. figure:: _static/show_graph_basic.svg  |
| >>> a = ProcessingMechanism(                              |                                           |
|               name='A',                                   |                                           |
| ...           size=3,                                     |                                           |
| ...           output_states=[RESULTS, OUTPUT_MEAN]        |                                           |
| ...           )                                           |                                           |
| >>> b = ProcessingMechanism(                              |                                           |
| ...           name='B',                                   |                                           |
| ...           size=5                                      |                                           |
| ...           )                                           |                                           |
| >>> c = ProcessingMechanism(                              |                                           |
| ...           name='C',                                   |                                           |
| ...           size=2,                                     |                                           |
| ...           function=Logistic(gain=pnl.CONTROL)         |                                           |
| ...           )                                           |                                           |
| >>> comp = Composition(                                   |                                           |
| ...           name='Comp',                                |                                           |
| ...           enable_controller=True           |                                           |
| ...           )                                           |                                           |
| >>> comp.add_linear_processing_pathway([a,c])             |                                           |
| >>> comp.add_linear_processing_pathway([b,c])             |                                           |
| >>> ctlr = OptimizationControlMechanism(                  |                                           |
| ...            name='Controller',                         |                                           |
| ...            monitor_for_control=[(pnl.OUTPUT_MEAN, a)],|                                           |
| ...            function=GridSearch,                       |                                           |
| ...            control_signals=(GAIN, c),                 |                                           |
| ...            agent_rep=comp                             |                                           |
| ...            )                                          |                                           |
| >>> comp.add_controller(ctlr)                  |                                           |
+-----------------------------------------------------------+-------------------------------------------+

Note that the Composition's `controller <Composition.controller>` is not shown by default.  However this
can be shown, along with other information, using options in the Composition's `show_graph <Composition.show_graph>`
method.  The figure below shows several examples.

.. _System_show_graph_figure:

**Output of show_graph using different options**

.. figure:: _static/show_graph_figure.svg
   :alt: System graph examples
   :scale: 150 %

   Displays of the Composition used in the `example above <System_show_graph_basic_figure>`, generated using various
   options of its `show_graph <Composition.show_graph>` method. **Panel A** shows the graph with its Projections labeled
   and Component dimensions displayed.  **Panel B** shows the `controller <Composition.controller>` for the
   Composition and its associated `ObjectiveMechanism` using the **show_controller** option (controller-related
   Components are displayed in blue by default).  **Panel C** adds the Composition's `CompositionInterfaceMechanisms
   <CompositionInterfaceMechanism>` using the **show_cim** option. **Panel D** shows a detailed view of the Mechanisms
   using the **show_node_structure** option, that includes their `States <State>` and their `roles <NodeRole>` in the
   Composition. **Panel E** show an even more detailed view using **show_node_structure** as well as **show_cim**.

If a Composition has one ore more Compositions nested as Nodes within it, then these can be shown using the
**show_nested** option. For example, if two Compositions identical to **comp** in the `example above
<System_show_graph_basic_figure>` are added as the nodes of the linear processing pathway of a third* **comp** *,
these can be shown as follows:

        +-------------------------------------------+-------------------------------------------+
        |    >>> comp.show_graph()                  | .. figure:: _static/nested.svg            |
        +-------------------------------------------+-------------------------------------------+
        |    >>> comp.show_graph(show_nested=True)  | .. figure:: _static/show_nested.svg       |
        |                                           |                                           |
        +-------------------------------------------+-------------------------------------------+



.. _Composition_Class_Reference:

Class Reference
---------------

"""

import collections
import inspect
import itertools
import logging
import warnings

import numpy as np
import typecheck as tc
import uuid

from PIL import Image
from os import path, remove

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component, ComponentsMeta, function_type
from psyneulink.core.components.functions.interfacefunctions import InterfaceStateMap
from psyneulink.core.components.functions.learningfunctions import Reinforcement, BackPropagation, TDLearning
from psyneulink.core.components.functions.combinationfunctions import LinearCombination, PredictionErrorDeltaFunction
from psyneulink.core.components.mechanisms.mechanism import MechanismRegistry, Mechanism_Base
from psyneulink.core.components.mechanisms.adaptive.modulatorymechanism import ModulatoryMechanism
from psyneulink.core.components.mechanisms.adaptive.control.optimizationcontrolmechanism import OptimizationControlMechanism
from psyneulink.core.components.mechanisms.adaptive.learning.learningmechanism import \
    LearningMechanism, ACTIVATION_OUTPUT, ERROR_SIGNAL
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.projections.projection import DuplicateProjectionError
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.shellclasses import Composition_Base
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.parameterstate import ParameterState
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.globals.registry import remove_instance_from_registry
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import \
    AFTER, ALL, BEFORE, BOLD, COMPARATOR_MECHANISM, COMPONENT, CONTROLLER, CONDITIONS, FUNCTIONS, HARD_CLAMP, \
    IDENTITY_MATRIX, INPUT, LABELS, LEARNED_PROJECTION, LEARNING_MECHANISM, \
    MATRIX, MATRIX_KEYWORD_VALUES, MECHANISM, MECHANISMS, NAME, NO_CLAMP, \
    ONLINE, OUTCOME, OUTPUT, OWNER_VALUE, PROJECTIONS, PULSE_CLAMP, ROLES, \
    SAMPLE, SIMULATIONS, SOFT_CLAMP, TARGET, TARGET_MECHANISM, VALUES, VARIABLE, WEIGHT
from psyneulink.core.globals.log import CompositionLog, LogCondition
from psyneulink.core.globals.parameters import Defaults, Parameter, ParametersBase
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import ContentAddressableList, NodeRole, call_with_pruned_args, convert_to_list
from psyneulink.core.scheduling.condition import All, Always, Condition, EveryNCalls
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism, MSE
from psyneulink.library.components.mechanisms.processing.objective.predictionerrormechanism import PredictionErrorMechanism

__all__ = [

    'Composition', 'CompositionError', 'CompositionRegistry', 'MECH_FUNCTION_PARAMS', 'STATE_FUNCTION_PARAMS'
]

# show_graph animation options
NUM_TRIALS = 'num_trials'
NUM_RUNS = 'num_Runs'
UNIT = 'unit'
DURATION = 'duration'
MOVIE_DIR = 'movie_dir'
MOVIE_NAME = 'movie_name'
SAVE_IMAGES = 'save_images'
SHOW = 'show'
INITIAL_FRAME = 'INITIAL_FRAME'
EXECUTION_SET = 'EXECUTION_SET'
SHOW_CIM = 'show_cim'
SHOW_CONTROLLER = 'show_controller'
SHOW_LEARNING = 'show_learning'


logger = logging.getLogger(__name__)

CompositionRegistry = {}


class CompositionError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class RunError(Exception):

    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)

class Vertex(object):
    """
        Stores a Component for use with a `Graph`

        Arguments
        ---------

        component : Component
            the `Component <Component>` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`

        Attributes
        ----------

        component : Component
            the `Component <Component>` represented by this Vertex

        parents : list[Vertex]
            the `Vertices <Vertex>` corresponding to the incoming edges of this `Vertex`

        children : list[Vertex]
            the `Vertices <Vertex>` corresponding to the outgoing edges of this `Vertex`
    """

    def __init__(self, component, parents=None, children=None, feedback=None):
        self.component = component
        if parents is not None:
            self.parents = parents
        else:
            self.parents = []
        if children is not None:
            self.children = children
        else:
            self.children = []

        self.feedback = feedback
        self.backward_sources = set()

    def __repr__(self):
        return '(Vertex {0} {1})'.format(id(self), self.component)


class Graph(object):
    """
        A Graph of vertices and edges/

        Attributes
        ----------

        comp_to_vertex : Dict[`Component <Component>` : `Vertex`]
            maps `Component` in the graph to the `Vertices <Vertex>` that represent them.

        vertices : List[Vertex]
            the `Vertices <Vertex>` contained in this Graph.

    """

    def __init__(self):
        self.comp_to_vertex = collections.OrderedDict()  # Translate from PNL Mech, Comp or Proj to corresponding vertex
        self.vertices = []  # List of vertices within graph

    def copy(self):
        """
            Returns
            -------

            A copy of the Graph. `Vertices <Vertex>` are distinct from their originals, and point to the same
            `Component <Component>` object : `Graph`
        """
        g = Graph()

        for vertex in self.vertices:
            g.add_vertex(Vertex(vertex.component, feedback=vertex.feedback))

        for i in range(len(self.vertices)):
            g.vertices[i].parents = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in
                                     self.vertices[i].parents]
            g.vertices[i].children = [g.comp_to_vertex[parent_vertex.component] for parent_vertex in
                                      self.vertices[i].children]

        return g

    def add_component(self, component, feedback=False):
        if component in [vertex.component for vertex in self.vertices]:
            logger.info('Component {1} is already in graph {0}'.format(component, self))
        else:
            vertex = Vertex(component, feedback=feedback)
            self.comp_to_vertex[component] = vertex
            self.add_vertex(vertex)

    def add_vertex(self, vertex):
        if vertex in self.vertices:
            logger.info('Vertex {1} is already in graph {0}'.format(vertex, self))
        else:
            self.vertices.append(vertex)
            self.comp_to_vertex[vertex.component] = vertex

    def remove_component(self, component):
        try:
            self.remove_vertex(self.comp_to_vertex[component])
        except KeyError as e:
            raise CompositionError('Component {1} not found in graph {2}: {0}'.format(e, component, self))

    def remove_vertex(self, vertex):
        try:
            for parent in vertex.parents:
                parent.children.remove(vertex)
            for child in vertex.children:
                child.parents.remove(vertex)

            self.vertices.remove(vertex)
            del self.comp_to_vertex[vertex.component]
            # TODO:
            #   check if this removal puts the graph in an inconsistent state
        except ValueError as e:
            raise CompositionError('Vertex {1} not found in graph {2}: {0}'.format(e, vertex, self))

    def connect_components(self, parent, child):
        try:
            self.connect_vertices(self.comp_to_vertex[parent], self.comp_to_vertex[child])
        except KeyError as e:
            if parent not in self.comp_to_vertex:
                raise CompositionError("Sender ({}) of {} ({}) not (yet) assigned".
                                       format(repr(parent.name), Projection.__name__, repr(child.name)))
            elif child not in self.comp_to_vertex:
                raise CompositionError("{} ({}) to {} not (yet) assigned".
                                       format(Projection.__name__, repr(parent.name), repr(child.name)))
            else:
                raise KeyError(e)

    def connect_vertices(self, parent, child):
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def get_parents_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose parents will be returned

            Returns
            -------

            A list[Vertex] of the parent `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        """
        return self.comp_to_vertex[component].parents

    def get_children_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        """
        return self.comp_to_vertex[component].children

    def get_forward_children_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose parents will be returned

            Returns
            -------

            A list[Vertex] of the parent `Vertices <Vertex>` of the Vertex associated with **component**: list[`Vertex`]
        """
        forward_children = []
        for child in self.comp_to_vertex[component].children:
            if component not in self.comp_to_vertex[child.component].backward_sources:
                forward_children.append(child)
        return forward_children

    def get_forward_parents_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose parents will be returned

            Returns
            -------

            A list[Vertex] of the parent `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        """
        forward_parents = []
        for parent in self.comp_to_vertex[component].parents:
            if parent.component not in self.comp_to_vertex[component].backward_sources:
                forward_parents.append(parent)
        return forward_parents

    def get_backward_children_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        """
        backward_children = []
        for child in self.comp_to_vertex[component].children:
            if component in self.comp_to_vertex[child.component].backward_sources:
                backward_children.append(child)
        return backward_children

    def get_backward_parents_from_component(self, component):
        """
            Arguments
            ---------

            component : Component
                the Component whose children will be returned

            Returns
            -------

            A list[Vertex] of the child `Vertices <Vertex>` of the Vertex associated with **component** : list[`Vertex`]
        """

        return list(self.comp_to_vertex[component].backward_sources)


# Options for show_node_structure argument of show_graph()
MECH_FUNCTION_PARAMS = "MECHANISM_FUNCTION_PARAMS"
STATE_FUNCTION_PARAMS = "STATE_FUNCTION_PARAMS"


class Composition(Composition_Base, metaclass=ComponentsMeta):
    """
        Composition

        Arguments
        ---------

        name: str

        controller:   `OptimizationControlmechanism` : default None
            specifies the `OptimizationControlMechanism` to use as the Composition's `controller
            <Composition.controller>` (see `Composition_Controller` for details).

        enable_controller: bool : default None
            specifies whether the Composition's `controller <Composition.controller>` is executed when the
            Composition is executed.  Set to True by default if **controller** specified;  if set to False,
            the `controller <Composition.controller>` is ignored when the Composition is executed.

        controller_mode: Enum[BEOFRE|AFTER] : default AFTER
            specifies whether the controller is executed before or after the rest of the Composition
            in each trial.  Must be either the keyword *BEFORE* or *AFTER*.

        controller_condition: Condition : default Always
            specifies when the Composition's `controller <Composition.controller>` is executed in a trial.

        enable_learning: bool : default True
            specifies whether `LearningMechanisms <LearningMechanism>` in the Composition are executed when it is
            executed.


        Attributes
        ----------

        graph : `Graph`
            the full `Graph` associated with this Composition. Contains both Nodes (`Mechanisms <Mechanism>` or
            `Compositions <Composition>`) and `Projections <Projection>`

        nodes : `list[Mechanisms and Compositions]`
            a list of all Nodes (`Mechanisms <Mechanism>` and/or `Compositions <Composition>`) contained in
            this Composition

        input_CIM : `CompositionInterfaceMechanism`
            aggregates input values for the INPUT nodes of the Composition. If the Composition is nested, then the
            input_CIM and its InputStates serve as proxies for the Composition itself in terms of afferent projections.

        input_CIM_states : dict
            a dictionary in which keys are InputStates of INPUT Nodes in a composition, and values are lists
            containing two items: the corresponding InputState and OutputState on the input_CIM.

        afferents : ContentAddressableList
            a list of all of the `Projections <Projection>` to the Composition's `input_CIM`.

        output_CIM : `CompositionInterfaceMechanism`
            aggregates output values from the OUTPUT nodes of the Composition. If the Composition is nested, then the
            output_CIM and its OutputStates serve as proxies for Composition itself in terms of efferent projections.

        output_CIM_states : dict
            a dictionary in which keys are OutputStates of OUTPUT Nodes in a composition, and values are lists
            containing two items: the corresponding InputState and OutputState on the input_CIM.

        efferents : ContentAddressableList
            a list of all of the `Projections <Projection>` from the Composition's `output_CIM`.

        env : Gym Forager Environment : default: None
            stores a Gym Forager Environment so that the Composition may interact with this environment within a
            single call to `run <Composition.run>`.

        shadows : dict
            a dictionary in which the keys are all in the Composition and the values are lists of any Nodes that
            `shadow <InputState_Shadow_Inputs>` the original Node's input.

        controller : OptimizationControlMechanism
            identifies the `OptimizationControlMechanism` used as the Composition's controller
            (see `Composition_Controller` for details).

        enable_controller : bool
            determines whether the Composition's `controller <Composition.controller>` is executed in each trial
            (see controller_mode <Composition.controller_mode>` for timing of execution).  Set to True by default
            if `controller <Composition.controller>` is specified.  Setting it to False suppresses exectuion of the
            `controller <Composition.controller>`.

        controller_mode :  BEFORE or AFTER
            determines whether the controller is executed before or after the rest of the `Composition`
            is executed on each trial.

        controller_condition : Condition
            specifies whether the controller is executed in a given trial.  The default is `Always`, which
            executes the controller on every trial.

        default_execution_id
            if no *execution_id* is specified in a call to run, this *execution_id* is used;  by default,
            it is the Composition's `name <Composition.name>`.

        execution_ids : set
            stores all execution_ids used by this Composition.

        results : 3d array
            stores the `output_values <Mechanism_Base.output_values>` of the `OUTPUT` Mechanisms in the Composition for
            every `TRIAL <TimeScale.TRIAL>` executed in a call to `run <Composition.run>`.  Each item in the outermost
            dimension (axis 0) of the array corresponds to a trial; each item within a trial corresponds to the
            `output_values <Mechanism_Base.output_values>` of an `OUTPUT` Mechanism.

        simulation_results : 3d array
            stores the `results <Composition.results>` for executions of the Composition when it is executed using
            its `evaluate <Composition.evaluate>` method.

        retain_old_simulation_data : bool
            if True, all Parameter values generated during simulations will be saved for later inspection;
            if False, simulation values will be deleted unless otherwise specified by individual Parameters

        COMMENT:
        name : str
            see `name <Composition_Name>`

        prefs : PreferenceSet
            see `prefs <Composition_Prefs>`
        COMMENT

    """
    # Composition now inherits from Component, so registry inherits name None
    componentType = 'Composition'

    class Parameters(ParametersBase):
        """
            Attributes
            ----------

                results
                    see `results <Composition.results>`

                    :default value: []
                    :type: list

                simulation_results
                    see `simulation_results <Composition.simulation_results>`

                    :default value: []
                    :type: list

        """
        results = Parameter([], loggable=False)
        simulation_results = Parameter([], loggable=False)
        retain_old_simulation_data = Parameter(False, stateful=False, loggable=False)

    class _CompilationData(ParametersBase):
        ptx_execution = None
        parameter_struct = None
        context_struct = None
        data_struct = None
        scheduler_conditions = None

    def __init__(
            self,
            name=None,
            controller:ModulatoryMechanism=None,
            enable_controller=None,
            controller_mode:tc.enum(BEFORE,AFTER)=AFTER,
            controller_condition:Condition=Always(),
            enable_learning=False,
            retain_old_simulation_data=None,
            **param_defaults
    ):
        # also sets name
        register_category(
            entry=self,
            base_class=Composition,
            registry=CompositionRegistry,
            name=name,
        )

        # core attribute
        self.graph = Graph()  # Graph of the Composition
        self._graph_processing = None
        self.nodes = ContentAddressableList(component_type=Component)
        self.required_node_roles = []

        # 'env' attr required for dynamic inputs generated by gym forager env
        self.env = None

        # Interface Mechanisms
        self.input_CIM = CompositionInterfaceMechanism(name=self.name + " Input_CIM",
                                                       composition=self)
        self.output_CIM = CompositionInterfaceMechanism(name=self.name + " Output_CIM",
                                                        composition=self)
        self.input_CIM_states = {}
        self.output_CIM_states = {}

        self.shadows = {}

        self.enable_controller = enable_controller
        self.controller = controller
        self.controller_mode = controller_mode
        self.controller_condition = controller_condition
        self.controller_condition.owner = self.controller

        self.default_execution_id = self.name
        self.execution_ids = {self.default_execution_id}

        self.projections = ContentAddressableList(component_type=Component)

        self._scheduler_processing = None

        self.enable_learning = False

        # status attributes
        self.graph_consistent = True  # Tracks if Composition is in runnable state (no dangling projections (what else?)
        self.needs_update_graph = True  # Tracks if Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True  # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler_processing = True  # Tracks if the processing scheduler needs to be regenerated

        self.nodes_to_roles = collections.OrderedDict()

        self.feedback_senders = set()
        self.feedback_receivers = set()

        self.parameters = self.Parameters(owner=self, parent=self.class_parameters)
        self.defaults = Defaults(
            owner=self,
            retain_old_simulation_data=retain_old_simulation_data,
            **{k: v for (k, v) in param_defaults.items() if hasattr(self.parameters, k)}
        )
        self._initialize_parameters()

        # Compiled resources
        self.__generated_node_wrappers = {}
        self.__generated_run = None
        self.__generated_simulation = None
        self.__generated_sim_run = None

        self._compilation_data = self._CompilationData(owner=self)

        self.log = CompositionLog(owner=self)
        self._terminal_backprop_sequences = {}

    def __repr__(self):
        return '({0} {1})'.format(type(self).__name__, self.name)

    @property
    def graph_processing(self):
        """
            The Composition's processing graph (contains only `Mechanisms <Mechanism>`.

            :getter: Returns the processing graph, and builds the graph if it needs updating since the last access.
        """
        if self.needs_update_graph_processing or self._graph_processing is None:
            self._update_processing_graph()

        return self._graph_processing

    @property
    def scheduler_processing(self):
        """
            A default `Scheduler` automatically generated by the Composition, used for the
            (`processing <System_Execution_Processing>` phase of execution.

            :getter: Returns the default processing scheduler, and builds it if it needs updating since the last access.
        """
        if self.needs_update_scheduler_processing or not isinstance(self._scheduler_processing, Scheduler):
            old_scheduler = self._scheduler_processing
            self._scheduler_processing = Scheduler(graph=self.graph_processing, execution_id=self.default_execution_id)

            if old_scheduler is not None:
                self._scheduler_processing.add_condition_set(old_scheduler.conditions)

            self.needs_update_scheduler_processing = False

        return self._scheduler_processing

    @scheduler_processing.setter
    def scheduler_processing(self, value: Scheduler):
        warnings.warn(
            f'If {self} is changed (nodes or projections are added or removed), scheduler_processing '
            ' will be rebuilt, and will be different than the Scheduler you are now setting it to.',
            stacklevel=2
        )

        self._scheduler_processing = value

    @property
    def termination_processing(self):
        return self.scheduler_processing.termination_conds

    @termination_processing.setter
    def termination_processing(self, termination_conds):
        self.scheduler_processing.termination_conds = termination_conds

    def _get_unique_id(self):
        return uuid.uuid4()

    def _update_shadows_dict(self, node):
        # Create an empty entry for this node in the Composition's "shadows" dict
        # If any other nodes shadow this node, they will be added to the list
        if node not in self.shadows:
            self.shadows[node] = []

        # If this node is shadowing another node, then add it to that node's entry in the Composition's "shadows" dict
        for input_state in node.input_states:
            if hasattr(input_state, "shadow_inputs") and input_state.shadow_inputs is not None:
                if node not in self.shadows[input_state.shadow_inputs.owner]:
                    self.shadows[input_state.shadow_inputs.owner].append(node)

    def add_node(self, node, required_roles=None):
        """
            Add a Composition Node (`Mechanism` or `Composition`) to the Composition, if it is not already added

            Arguments
            ---------

            node : `Mechanism` or `Composition`
                the node to be added to the Composition

            required_roles : `NodeRole` or list of NodeRoles
                any NodeRoles roles that this node should have in addition to those determined by analyze graph.
        """

        self._update_shadows_dict(node)

        if node not in [vertex.component for vertex in
                        self.graph.vertices]:  # Only add if it doesn't already exist in graph
            node.is_processing = True
            self.graph.add_component(node)  # Set incoming edge list of node to empty
            self.nodes.append(node)
            self.nodes_to_roles[node] = set()

            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler_processing = True

            try:
                # activate any projections the node requires
                node._activate_projections_for_compositions(self)
            except AttributeError:
                pass

        if hasattr(node, "aux_components"):

            projections = []
            # Add all "nodes" to the composition first (in case projections reference them)
            for component in node.aux_components:
                if isinstance(component, (Mechanism, Composition)):
                    if isinstance(component, Composition):
                        component._analyze_graph()
                    self.add_node(component)
                elif isinstance(component, Projection):
                    projections.append((component, False))
                elif isinstance(component, tuple):
                    if isinstance(component[0], Projection):
                        if isinstance(component[1], bool):
                            projections.append(component)
                        else:
                            raise CompositionError("Invalid component specification ({}) in {}'s aux_components. If a "
                                                   "tuple is used to specify a Projection, then the index 0 item must "
                                                   "be the Projection, and the index 1 item must be the feedback "
                                                   "specification (True or False).".format(component, node.name))
                    elif isinstance(component[0], (Mechanism, Composition)):
                        if isinstance(component[1], NodeRole):
                            self.add_node(node=component[0], required_roles=component[1])
                        elif isinstance(component[1], list):
                            if isinstance(component[1][0], NodeRole):
                                self.add_node(node=component[0], required_roles=component[1])
                            else:
                                raise CompositionError("Invalid component specification ({}) in {}'s aux_components. "
                                                       "If a tuple is used to specify a Mechanism or Composition, then "
                                                       "the index 0 item must be the node, and the index 1 item must "
                                                       "be the required_roles".format(component, node.name))

                        else:
                            raise CompositionError("Invalid component specification ({}) in {}'s aux_components. If a "
                                                   "tuple is used to specify a Mechanism or Composition, then the "
                                                   "index 0 item must be the node, and the index 1 item must be the "
                                                   "required_roles".format(component, node.name))
                    else:
                        raise CompositionError("Invalid component specification ({}) in {}'s aux_components. If a tuple"
                                               " is specified, then the index 0 item must be a Projection, Mechanism, "
                                               "or Composition.".format(component, node.name))
                else:
                    raise CompositionError("Invalid component ({}) in {}'s aux_components. Must be a Mechanism, "
                                           "Composition, Projection, or tuple."
                                           .format(component.name, node.name))

            # Add all projections to the composition
            for proj_spec in projections:
                self.add_projection(projection=proj_spec[0], feedback=proj_spec[1])
        if required_roles:
            if not isinstance(required_roles, list):
                required_roles = [required_roles]
            for required_role in required_roles:
                self.add_required_node_role(node, required_role)

        for input_state in node.input_states:
            if hasattr(input_state, "shadow_inputs") and input_state.shadow_inputs is not None:
                for proj in input_state.shadow_inputs.path_afferents:
                    sender = proj.sender
                    if sender.owner != self.input_CIM:
                        self.add_projection(projection=MappingProjection(sender=proj.sender, receiver=input_state),
                                            sender=proj.sender.owner,
                                            receiver=node)

    def add_nodes(self, nodes, required_roles=None):
        """
            Add a list of Composition Nodes (`Mechanism` or `Composition`) to the Composition,

            Arguments
            ---------

            nodes : list
                the nodes to be added to the Composition.  Each item of the list must be a `Mechanism`,
                a `Composition` or a role-specification tuple with a Mechanism or Composition as the first item,
                and a `NodeRole` or list of those as the second item;  any NodeRoles in a role-specification tuple
                are applied in addition to those specified in the **required_roles** argument.

            required_roles : `NodeRole` or list of NodeRoles
                NodeRoles to assign to the nodes in addition to those determined by analyze graph;
                these apply to any items in the list of nodes that are not in a tuple;  these apply to any specified
                in any role-specification tuples in the **nodes** argument.
        """
        if not isinstance(nodes, list):
            raise CompositionError(f"Arg for 'add_nodes' method of '{self.name}' {Composition.__name__} "
                                   f"must be a list of nodes or (node, required_roles) tuples")
        for node in nodes:
            if isinstance(node, (Mechanism, Composition)):
                self.add_node(node, required_roles)
            elif isinstance(node, tuple):
                node_specific_roles = convert_to_list(node[1])
                if required_roles:
                    node_specific_roles.append(required_roles)
                self.add_node(node=node[0], required_roles=node_specific_roles)
            else:
                raise CompositionError(f"Node specified in 'add_nodes' method of '{self.name}' {Composition.__name__} "
                                       f"({node}) must be a {Mechanism.__name__}, {Composition.__name__}, "
                                       f"or a tuple containing one of those and a {NodeRole.__name__} or list of them")

    def remove_nodes(self, nodes):
        if not isinstance(nodes, (list, Mechanism, Composition)):
            assert False, 'Argument of remove_nodes must be a Mechanism, Composition or list containing either or both'
        nodes = convert_to_list(nodes)
        for node in nodes:
            for proj in node.afferents + node.efferents:
                try:
                    del self.projections[proj]
                except ValueError:
                    # why are these not present?
                    pass

                try:
                    self.graph.remove_component(proj)
                except CompositionError:
                    # why are these not present?
                    pass

            self.graph.remove_component(node)
            del self.nodes_to_roles[node]
            node_role_pairs = [item for item in self.required_node_roles if item[0] is node]
            for item in node_role_pairs:
                self.required_node_roles.remove(item)
            del self.nodes[node]

    def add_controller(self, controller:ModulatoryMechanism):
        """
        Add an `OptimizationControlMechanism` as the `controller
        <Composition.controller>` of the Composition, which gives the OCM access to the
        `Composition`'s `evaluate <Composition.evaluate>` method. This allows the OCM to use simulations to determine
        an optimal Control policy.
        """

        if not isinstance(controller, ModulatoryMechanism):
            raise CompositionError(f"Specification of {repr(CONTROLLER)} arg for {self.name} "
                                   f"must be a {repr(ModulatoryMechanism.__name__)} ")

        self.controller = controller
        self.controller.composition = self

        if self.controller.objective_mechanism:
            self.add_node(self.controller.objective_mechanism)

        self.enable_controller = True

        controller._activate_projections_for_compositions(self)
        self._analyze_graph()
        self._update_shadows_dict(controller)

        # Skip first (OUTCOME) input_state
        for input_state in controller.input_states[1:]:
            if hasattr(input_state, "shadow_inputs") and input_state.shadow_inputs is not None:
                for proj in input_state.shadow_inputs.path_afferents:
                    sender = proj.sender
                    if sender.owner != self.input_CIM:
                        self.add_projection(projection=MappingProjection(sender=sender, receiver=input_state),
                                            sender=sender.owner,
                                            receiver=controller)
                        shadow_proj._activate_for_compositions(self)
                    else:
                        shadow_proj = MappingProjection(sender=proj.sender, receiver=input_state)
                        shadow_proj._activate_for_compositions(self)
            # MODIFIED 6/11/19 NEW: [JDC]
            for proj in input_state.path_afferents:
                proj._activate_for_compositions(self)
            # MODIFIED 6/11/19 END

    # # MODIFIED 7/22/19 OLD:
    # def _parse_projection_spec(self, projection, name):
    #     if isinstance(projection, (np.ndarray, np.matrix, list)):
    #         return MappingProjection(matrix=projection, name=name)
    #     elif isinstance(projection, str):
    #         if projection in MATRIX_KEYWORD_VALUES:
    #             return MappingProjection(matrix=projection, name=name)
    #         else:
    #             raise CompositionError("Invalid projection ({}) specified for {}.".format(projection, self.name))
    #     elif isinstance(projection, ModulatoryProjection_Base):
    #         return projection
    #     elif projection is None:
    #         return MappingProjection(name=name)
    #     elif not isinstance(projection, Projection):
    #         raise CompositionError("Invalid projection ({}) specified for {}. Must be a Projection."
    #                                .format(projection, self.name))
    #     return projection
    # MODIFIED 7/22/19 NEW:
    def _parse_projection_spec(self, projection, sender=None, receiver=None, name=None):
        if isinstance(projection, (np.ndarray, np.matrix, list)):
            return MappingProjection(matrix=projection, sender=sender, receiver=receiver, name=name)
        elif isinstance(projection, str):
            if projection in MATRIX_KEYWORD_VALUES:
                return MappingProjection(matrix=projection, sender=sender, receiver=receiver, name=name)
            else:
                raise CompositionError("Invalid projection ({}) specified for {}.".format(projection, self.name))
        elif isinstance(projection, ModulatoryProjection_Base):
            return projection
        elif projection is None:
            return MappingProjection(sender=sender, receiver=receiver, name=name)
        elif not isinstance(projection, Projection):
            raise CompositionError("Invalid projection ({}) specified for {}. Must be a Projection."
                                   .format(projection, self.name))
        return projection
    # MODIFIED 7/22/19 END

    def _parse_sender_spec(self, projection, sender):

        # if a sender was not passed, check for a sender OutputState stored on the Projection object
        if sender is None:
            if hasattr(projection, "sender"):
                sender = projection.sender.owner
            else:
                raise CompositionError(f"{projection.name} is missing a sender specification. "
                                       f"For a Projection to be added to a Composition a sender must be specified, "
                                       "either on the Projection or in the call to Composition.add_projection(). ")

        # initialize all receiver-related variables
        graph_sender = sender_mechanism = sender_output_state = sender

        nested_compositions = []
        if isinstance(sender, Mechanism):
            # Mechanism spec -- update sender_output_state to reference primary OutputState
            sender_output_state = sender.output_state

        elif isinstance(sender, OutputState):
            # InputState spec -- update sender_mechanism and graph_sender to reference owner Mechanism
            sender_mechanism = graph_sender = sender.owner

        elif isinstance(sender, Composition):
            # Nested Composition Spec -- update sender_mechanism to CIM; sender_output_state to CIM's primary O.S.
            sender_mechanism = sender.output_CIM
            sender_output_state = sender_mechanism.output_state
            nested_compositions.append(sender)

        else:
            raise CompositionError("sender arg ({}) of call to add_projection method of {} is not a {}, {} or {}".
                                   format(sender, self.name,
                                          Mechanism.__name__, OutputState.__name__, Composition.__name__))

        if (not isinstance(sender_mechanism, CompositionInterfaceMechanism)
                and not isinstance(sender, Composition)
                and sender_mechanism not in self.nodes):
            sender_name = sender.name

            # if the sender is IN a nested Composition AND sender is an OUTPUT Node
            # then use the corresponding CIM on the nested comp as the sender going forward
            sender, sender_output_state, graph_sender, sender_mechanism = \
                self._get_nested_node_CIM_state(sender_mechanism,
                                                sender_output_state,
                                                NodeRole.OUTPUT)
            nested_compositions.append(graph_sender)
            if sender is None:
                receiver_name = 'node'
                if hasattr(projection, 'receiver'):
                    receiver_name = f'{repr(projection.receiver.owner.name)}'
                raise CompositionError(f"A {Projection.__name__} specified to {receiver_name} in {self.name} "
                                       f"has a sender ({repr(sender_name)}) that is not (yet) in it "
                                       f"or any of its nested {Composition.__name__}s.")

        if hasattr(projection, "sender"):
            if projection.sender.owner != sender and \
                    projection.sender.owner != graph_sender and \
                    projection.sender.owner != sender_mechanism:
                raise CompositionError("The position of {} in {} conflicts with its sender attribute."
                                       .format(projection.name, self.name))

        return sender, sender_mechanism, graph_sender, nested_compositions

    def _parse_receiver_spec(self, projection, receiver, sender, learning_projection):

        # if a receiver was not passed, check for a receiver InputState stored on the Projection object
        if receiver is None:
            if hasattr(projection, "receiver"):
                receiver = projection.receiver.owner
            else:
                raise CompositionError("For a Projection to be added to a Composition, a receiver must be specified, "
                                       "either on the Projection or in the call to Composition.add_projection(). {}"
                                       " is missing a receiver specification. ".format(projection.name))

        # initialize all receiver-related variables
        graph_receiver = receiver_mechanism = receiver_input_state = receiver

        nested_compositions = []
        if isinstance(receiver, Mechanism):
            # Mechanism spec -- update receiver_input_state to reference primary InputState
            receiver_input_state = receiver.input_state

        elif isinstance(receiver, InputState):
            # InputState spec -- update receiver_mechanism and graph_receiver to reference owner Mechanism
            receiver_mechanism = graph_receiver = receiver.owner

        elif isinstance(receiver, Composition):
            # Nested Composition Spec -- update receiver_mechanism to CIM; receiver_input_state to CIM's primary I.S.
            receiver_mechanism = receiver.input_CIM
            receiver_input_state = receiver_mechanism.input_state
            nested_compositions.append(receiver)

        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        elif isinstance(receiver, AutoAssociativeProjection):
            receiver_mechanism = receiver.owner_mech
            receiver_input_state = receiver_mechanism.input_state
            learning_projection = True

        elif isinstance(sender, LearningMechanism):
            receiver_mechanism = receiver.receiver.owner
            receiver_input_state = receiver_mechanism.input_state
            learning_projection = True

        else:
            raise CompositionError("receiver arg ({}) of call to add_projection method of {} is not a {}, {} or {}".
                                   format(receiver, self.name,
                                          Mechanism.__name__, InputState.__name__, Composition.__name__))

        if not isinstance(receiver_mechanism, CompositionInterfaceMechanism) \
                and not isinstance(receiver, Composition)\
                and receiver_mechanism not in self.nodes\
                and not learning_projection:

            # if the receiver is IN a nested Composition AND receiver is an INPUT Node
            # then use the corresponding CIM on the nested comp as the receiver going forward
            receiver, receiver_input_state, graph_receiver, receiver_mechanism = \
                self._get_nested_node_CIM_state(receiver_mechanism, receiver_input_state, NodeRole.INPUT)

            nested_compositions.append(graph_receiver)
            # Otherwise, there was a mistake in the spec
            if receiver is None:
                raise CompositionError("receiver arg ({}) in call to add_projection method of {} "
                                       "is not in it or any of its nested {}s ".
                                       format(repr(receiver), self.name, Composition.__name__, ))

        return receiver, receiver_mechanism, graph_receiver, receiver_input_state, \
               nested_compositions, learning_projection

    def add_projection(self,
                       projection=None,
                       sender=None,
                       receiver=None,
                       feedback=False,
                       learning_projection=False,
                       name=None,
                       allow_duplicates=False
                       ):
        """Add **projection** to the Composition, if one with the same sender and receiver doesn't already exist.

        If **projection** is not specified, create a default `MappingProjection` using **sender** and **receiver**.

        If **projection** is specified:

        • if **projection** has already been instantiated, and **sender** and **receiver** are also specified,
          they must match the `sender <MappingProjection.sender>` and `receiver <MappingProjection.receiver>`
          of **projection**.

        • if the status of **projection** is `deferred_init`:

          - if its `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` attributes are not
            specified, then **sender** and/or **receiver** are used.

          - if `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` attributes are specified,
            they must match **sender** and/or **receiver** if those have also been specified.

          - if a Projection between the specified sender and receiver does *not* already exist, it is initialized; if
            it *does* already exist, the request to add it is ignored, however requests to shadow it and/or mark it as
            a`feedback` Projection are implemented (in case it has not already been done for the existing Projection).

        .. note::
           If **projection** is an instantiated projection (i.e., not in `deferred_init`) and one already exists between
           its `sender <Projection_Base.sender>` and `receiver <Projection_Base.receiver>` a warning is generated.

        COMMENT:
        IMPLEMENTATION NOTE:
            Duplicates are determined by the **States** to which they project, not the Mechanisms (to allow
            multiple Projections to exist between the same pair of Mechanisms using different States).
            -
            If an already instantiated Projection is passed to add_projection and is a duplicate of an existing one,
            it is detected and suppresed, with a warning, in State._instantiate_projections_to_state.
            -
            If a Projection with deferred_init status is a duplicate, it is fully suppressed here,
            as these are generated by add_linear_processing_pathway if the pathway overlaps with an existing one,
            and so warnings are unnecessary and would be confusing to users.
        COMMENT

        Arguments
        ---------

        sender : Mechanism, Composition, or OutputState
            the sender of **projection**

        projection : Projection, matrix
            the projection to add

        receiver : Mechanism, Composition, or InputState
            the receiver of **projection**

        feedback : bool
            When False (default) all Nodes within a cycle containing this Projection execute in parallel. This
            means that each Projections within the cycle actually passes to its `receiver <Projection_Base.receiver>`
            the `value <Projection.value>` of its `sender <Projection_Base.sender>` from the previous execution.
            When True, this Projection "breaks" the cycle, such that all Nodes execute in sequence, and only the
            Projection marked as 'feedback' passes to its `receiver <Projection_Base.receiver>` the
            `value <Projection.value>` of its `sender <Projection_Base.sender>` from the previous execution.

        Returns
        -------

        projection if added, else None

    """

        # FIX: 7/22/19 [JDC] - THIS COULD BE CLEANED UP MORE

        # # MODIFIED 7/22/19 OLD: [JDC] MOVED TO BELOW
        # projection = self._parse_projection_spec(projection, name)
        # duplicate = False
        # MODIFIED 7/22/19 NEWER:
        try:
            # projection = self._parse_projection_spec(projection, sender, receiver, name)
            projection = self._parse_projection_spec(projection, name)
        except DuplicateProjectionError:
            return projection
        duplicate = False
        # MODIFIED 7/22/19 END

        # Parse sender and receiver specs
        sender, sender_mechanism, graph_sender, nested_compositions = self._parse_sender_spec(projection, sender)
        receiver, receiver_mechanism, graph_receiver, receiver_input_state, nested_compositions, learning_projection = \
            self._parse_receiver_spec(projection, receiver, sender, learning_projection)

        # # MODIFIED 7/22/19 NEW: [JDC] MOVED FROM ABOVE
        # try:
        #     projection = self._parse_projection_spec(projection, sender, receiver, name)
        # except DuplicateProjectionError:
        #     return None
        # duplicate = False
        # MODIFIED 7/22/19 END

        # If Deferred init
        if projection.context.initialization_status == ContextFlags.DEFERRED_INIT:
            # If sender or receiver are State specs, use those;  otherwise, use graph node (Mechanism or Composition)
            if not isinstance(sender, OutputState):
                # sender = graph_sender
                sender = sender_mechanism
            if not isinstance(receiver, InputState):
                # receiver = graph_receiver
                receiver = receiver_mechanism
            # Check if Projection to be initialized already exists;  if so, mark as duplicate and skip
            duplicate = self._check_for_existing_projection(sender=sender, receiver=receiver)
            if not duplicate:
                # Initialize Projection
                projection.init_args['sender'] = sender
                projection.init_args['receiver'] = receiver
                try:
                    projection._deferred_init(context=" INITIALIZING ")
                except DuplicateProjectionError:
                    return projection

        # MODIFIED 7/22/19 NEW: [JDC]
        elif self._check_for_existing_projection(projection, sender=sender, receiver=receiver):
            return projection
        # MODIFIED 7/22/19 END

        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        if sender_mechanism != self.input_CIM and receiver_mechanism != self.output_CIM \
                and projection not in [vertex.component for vertex in self.graph.vertices] and not learning_projection:

            projection.is_processing = False
            # KDM 5/24/19: removing below rename because it results in several duplicates
            # projection.name = f'{sender} to {receiver}'
            self.graph.add_component(projection, feedback=feedback)

            try:
                self.graph.connect_components(graph_sender, projection)
                self.graph.connect_components(projection, graph_receiver)
            except CompositionError as c:
                raise CompositionError(f"{c.args[0]} to {self.name}.")

        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        if not duplicate:
            self._validate_projection(projection,
                                      sender, receiver,
                                      sender_mechanism, receiver_mechanism,
                                      learning_projection)
            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler_processing = True

            projection._activate_for_compositions(self)
            for comp in nested_compositions:
                projection._activate_for_compositions(comp)

        # Note: do all of the following even if Projection is a duplicate,
        #   as these conditions shoud apply to the exisiting one (and it won't hurt to try again if they do)

        # Create "shadow" projections to any input states that are meant to shadow this projection's receiver
        # (note: do this even if there is a duplciate and they are not allowed, as still want to shadow that projection)
        if receiver_mechanism in self.shadows and len(self.shadows[receiver_mechanism]) > 0:
            for shadow in self.shadows[receiver_mechanism]:
                for input_state in shadow.input_states:
                    if input_state.shadow_inputs is not None:
                        if input_state.shadow_inputs.owner == receiver:
                            # TBI: Copy the projection type/matrix value of the projection that is being shadowed
                            self.add_projection(MappingProjection(sender=sender, receiver=input_state),
                                                sender_mechanism, shadow)
        if feedback:
            self.feedback_senders.add(sender_mechanism)
            self.feedback_receivers.add(receiver_mechanism)

        return projection

    def _add_projection(self, projection):
        self.projections.append(projection)

    def add_projections(self, projections=None):
        """
            Calls `add_projection <Composition.add_projection>` for each Projection in the *projections* list. Each
            Projection must have its `sender <Projection_Base.sender>` and `receiver <Projection_Base.receiver>`
            already specified.  If an item in the list is a list of projections, called recursively on that list.

            Arguments
            ---------

            projections : list of Projections
                list of Projections to be added to the Composition
        """

        if isinstance(projections, list):
            for projection in projections:
                if isinstance(projection, list):
                    self.add_projections(projection)
                elif isinstance(projection, Projection) and \
                        hasattr(projection, "sender") and \
                        hasattr(projection, "receiver"):
                    self.add_projection(projection)
                else:
                    raise CompositionError("Invalid projections specification for {}. The add_projections method of "
                                           "Composition requires a list of Projections, each of which must have a "
                                           "sender and a receiver.".format(self.name))
        else:
            raise CompositionError("Invalid projections specification for {}. The add_projections method of "
                                   "Composition requires a list of Projections, each of which must have a "
                                   "sender and a receiver.".format(self.name))

    def remove_projection(self, projection):
        # step 1 - remove Vertex from Graph
        if projection in [vertex.component for vertex in self.graph.vertices]:
            vert = self.graph.comp_to_vertex[projection]
            self.graph.remove_vertex(vert)
        # step 2 - remove Projection from Composition's list
        if projection in self.projections:
            self.projections.remove(projection)

        # step 3 - TBI? remove Projection from afferents & efferents lists of any node

    def add_pathway(self, path):
        """
            Adds an existing Pathway to the current Composition

            Arguments
            ---------

            path: the Pathway (Composition) to be added
        """

        # identify nodes and projections
        nodes, projections = [], []
        for c in path.graph.vertices:
            if isinstance(c.component, Mechanism):
                nodes.append(c.component)
            elif isinstance(c.component, Composition):
                nodes.append(c.component)
            elif isinstance(c.component, Projection):
                projections.append(c.component)

        # add all nodes first
        for node in nodes:
            self.add_node(node)

        # then projections
        for p in projections:
            self.add_projection(p, p.sender.owner, p.receiver.owner)

        self._analyze_graph()

    def add_linear_processing_pathway(self, pathway, feedback=False, *args):
        """Add sequence of Mechanisms or Compositions possibly with intercolated Projections
        Tuples (Mechanism, NodeRole(s)) can be used to assign required_roles to Mechanisms.
        """
        nodes = []
        # First, verify that the pathway begins with a node
        if not isinstance(pathway, (list, tuple)):
            raise CompositionError(f"First arg for add_linear_processing_pathway method of '{self.name}' "
                                   f"{Composition.__name__} must be a list of nodes")

        if isinstance(pathway[0], (Mechanism, Composition, tuple)):
            self.add_nodes([pathway[0]]) # Use add_nodes so that node spec can also be a tuple with required_roles
            nodes.append(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg
            raise CompositionError("The first item in a linear processing pathway must be a Node (Mechanism or "
                                   "Composition).")

        # Then, add all of the remaining nodes in the pathway
        for c in range(1, len(pathway)):
            # if the current item is a Mechanism, Composition or (Mechanism, NodeRole(s)) tuple, add it
            if isinstance(pathway[c], (Mechanism, Composition, tuple)):
                self.add_nodes([pathway[c]])
                nodes.append(pathway[c])
        projections = []

        # Then, loop through and validate that the Mechanism-Projection relationships make sense
        # and add MappingProjection(s) where needed
        for c in range(1, len(pathway)):
            # if the current item is a Node
            if isinstance(pathway[c], (Mechanism, Composition, tuple)):
                if isinstance(pathway[c - 1], (Mechanism, Composition, tuple)):
                    # if the previous item was also a Composition Node, add a mapping projection between them
                    proj = self.add_projection(sender=pathway[c - 1],
                                               receiver=pathway[c],
                                               feedback=feedback)
                    # # MODIFIED 7/22/19 OLD:
                    # projections.append(proj)
                    # MODIFIED 7/22/19 NEW: [JDC]
                    if proj:
                        projections.append(proj)
                    # MODIFIED 7/22/19 END
            # if the current item is a Projection specification
            elif isinstance(pathway[c], (Projection, np.ndarray, np.matrix, str, list)):
                if c == len(pathway) - 1:
                    raise CompositionError("{} is the last item in the pathway. A projection cannot be the last item in"
                                           " a linear processing pathway.".format(pathway[c]))
                # confirm that it is between two nodes, then add the projection
                if isinstance(pathway[c - 1], (Mechanism, Composition)) \
                        and isinstance(pathway[c + 1], (Mechanism, Composition)):
                    proj = pathway[c]
                    # # MODIFIED 7/22/19 OLD:
                    # if isinstance(pathway[c], (np.ndarray, np.matrix, list)):
                    #     proj = MappingProjection(sender=pathway[c - 1],
                    #                              matrix=pathway[c],
                    #                              receiver=pathway[c + 1])
                    # MODIFIED 7/22/19 NEW:
                    try:
                        if isinstance(pathway[c], (np.ndarray, np.matrix, list)):
                            proj = MappingProjection(sender=pathway[c - 1],
                                                     matrix=pathway[c],
                                                     receiver=pathway[c + 1])
                    except DuplicateProjectionError:
                        # FIX: 7/22/19 ADD WARNING HERE??
                        pass
                    # MODIFIED 7/22/19 END

                    # # MODIFIED 7/22/19 OLD:
                    # self.add_projection(projection=proj,
                    #                     sender=pathway[c - 1],
                    #                     receiver=pathway[c + 1],
                    #                     feedback=feedback,
                    #                     allow_duplicates=False)
                    # projections.append(proj)
                    # MODIFIED 7/22/19 NEW: [JDC]
                    proj = self.add_projection(projection=proj,
                                               sender=pathway[c - 1],
                                               receiver=pathway[c + 1],
                                               feedback=feedback,
                                               allow_duplicates=False)
                    if proj:
                        projections.append(proj)
                    # MODIFIED 7/22/19 END

                else:
                    raise CompositionError(
                        "{} is not between two Composition Nodes. A Projection in a linear processing pathway must be "
                        "preceded by a Composition Node (Mechanism or Composition) and followed by a Composition Node"
                            .format(pathway[c]))
            else:
                raise CompositionError("{} is not a Projection or a Composition node (Mechanism or Composition). A "
                                       "linear processing pathway must be made up of Projections and Composition Nodes."
                                       .format(pathway[c]))
        # interleave nodes and projections
        explicit_pathway = [nodes[0]]
        for i in range(len(projections)):
            explicit_pathway.append(projections[i])
            explicit_pathway.append(nodes[i+1])

        return explicit_pathway

    def _create_rl_learning_related_mechanisms(self,
                                               input_source,
                                               output_source,
                                               error_function,
                                               learned_projection,
                                               learning_rate,
                                               learning_update):
        # Create learning components
        target_mechanism = ProcessingMechanism(name='Target')

        comparator_mechanism = ComparatorMechanism(name='Comparator',
                                                   target={NAME: TARGET,
                                                           VARIABLE: [0.]},
                                                   sample={NAME: SAMPLE,
                                                           VARIABLE: [0.], WEIGHT: -1},
                                                   function=error_function,
                                                   output_states=[OUTCOME, MSE])

        learning_mechanism = \
            LearningMechanism(function=Reinforcement(default_variable=[input_source.output_states[0].value,
                                                                       output_source.output_states[0].value,
                                                                       comparator_mechanism.output_states[0].value],
                                                     learning_rate=learning_rate),
                              default_variable=[input_source.output_states[0].value,
                                                output_source.output_states[0].value,
                                                comparator_mechanism.output_states[0].value],
                              error_sources=comparator_mechanism,
                              learning_enabled=learning_update,
                              in_composition=True,
                              name="Learning Mechanism for " + learned_projection.name)

        # FIX 5/29/19 [JDC]:  MIGHT WANT TO TEST HERE WHETHER IT IS IN A BP CHAIN AND, IF SO, AND NOT LAST, THEN
        #  REQUIRE IT
        learning_mechanism.output_states[ERROR_SIGNAL].parameters.require_projection_in_composition._set(False,
                                                                                                        override=True)
        self.enable_learning = True
        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_td_related_mechanisms(self,
                                      input_source,
                                      output_source,
                                      error_function,
                                      learned_projection,
                                      learning_rate,
                                      learning_update):
        # Create learning components
        target_mechanism = ProcessingMechanism(name='Target',
                                               default_variable=output_source.defaults.value)

        comparator_mechanism = PredictionErrorMechanism(
            sample={NAME: SAMPLE,
                    VARIABLE: output_source.defaults.value},
            target={NAME: TARGET,
                    VARIABLE: output_source.defaults.value},
            function=PredictionErrorDeltaFunction(gamma=1.0),
            # name="{} {}".format(output_source.name
            #                     # PREDICTION_ERROR_MECHANISM
            #                     )
        )

        learning_mechanism = LearningMechanism(function=TDLearning(learning_rate=learning_rate),
                                               default_variable=[input_source.output_states[0].defaults.value,
                                                                 output_source.output_states[0].defaults.value,
                                                                 comparator_mechanism.output_states[0].defaults.value],
                                               error_sources=comparator_mechanism,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name)
        self.enable_learning = True
        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_terminal_backprop_sequence_components(self,
                                                    input_source,
                                                    output_source,
                                                    error_function,
                                                    learned_projection,
                                                    learning_rate,
                                                    learning_update):
        """Create ComparatorMechanism, LearningMechanism and LearningProjection for Component in learning sequence"""

        target_mechanism = ProcessingMechanism(name='Target',
                                               default_variable=output_source.output_states[0].value)

        comparator_mechanism = ComparatorMechanism(name='Comparator',
                                                   target={NAME: TARGET,
                                                           VARIABLE: target_mechanism.output_states[0].value},
                                                   sample={NAME: SAMPLE,
                                                           VARIABLE: output_source.output_states[0].value,
                                                           WEIGHT: -1},
                                                   function=error_function,
                                                   output_states=[OUTCOME, MSE])

        learning_function = BackPropagation(default_variable=[input_source.output_states[0].value,
                                                              output_source.output_states[0].value,
                                                              comparator_mechanism.output_states[0].value],
                                            activation_derivative_fct=output_source.function.derivative,
                                            learning_rate=learning_rate)

        learning_mechanism = LearningMechanism(function=learning_function,
                                               default_variable=[input_source.output_states[0].value,
                                                                 output_source.output_states[0].value,
                                                                 comparator_mechanism.output_states[0].value],
                                               error_sources=comparator_mechanism,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name)

        self.add_nodes(nodes=[(target_mechanism, NodeRole.TARGET),
                              comparator_mechanism,
                              learning_mechanism],
                       required_roles=NodeRole.LEARNING)

        learning_related_projections = self._create_learning_related_projections(input_source,
                                                                                 output_source,
                                                                                 target_mechanism,
                                                                                 comparator_mechanism,
                                                                                 learning_mechanism)
        self.add_projections(learning_related_projections)

        learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
        self.add_projection(learning_projection, feedback=True)

        self.enable_learning = True

        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_multilayer_backprop_components(self,
                                               input_source,
                                               output_source,
                                               learned_projection,
                                               learning_rate,
                                               learning_update,
                                               previous_learning_mechanism):

        learning_function = BackPropagation(default_variable=[input_source.output_states[0].value,
                                                              output_source.output_states[0].value,
                                                              previous_learning_mechanism.output_states[0].value],
                                            activation_derivative_fct=output_source.function.derivative,
                                            learning_rate=learning_rate)

        learning_mechanism = LearningMechanism(function=learning_function,
                                               default_variable=[input_source.output_states[0].value,
                                                                 output_source.output_states[0].value,
                                                                 previous_learning_mechanism.output_states[0].value],
                                               error_sources=previous_learning_mechanism,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name)

        if not learning_mechanism.output_states[ERROR_SIGNAL].efferents:
            learning_mechanism.output_states[ERROR_SIGNAL].parameters.require_projection_in_composition.set(False,
                                                                                                            override=True)

        self.add_node(learning_mechanism, required_roles=NodeRole.LEARNING)

        act_in_projection = MappingProjection(sender=input_source.output_states[0],
                                              receiver=learning_mechanism.input_states[0])
        act_out_projection = MappingProjection(sender=output_source.output_states[0],
                                               receiver=learning_mechanism.input_states[1])
        error_projection = MappingProjection(sender=previous_learning_mechanism.output_states[0],
                                             receiver=learning_mechanism.input_states[2])

        self.add_projections([act_in_projection, act_out_projection, error_projection])
        learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
        self.add_projection(learning_projection, feedback=True)

        return learning_mechanism

    def _create_learning_related_projections(self, input_source, output_source, target, comparator, learning_mechanism):
        # construct learning related mapping projections
        # FIX 5/29/19 [JDC]:  REPLACE INDICES BELOW WITH RELEVANT KEYWORDS
        sample_projection = MappingProjection(sender=output_source,
                                              receiver=comparator.input_states[0])
        target_projection = MappingProjection(sender=target,
                                              receiver=comparator.input_states[1])
        act_in_projection = MappingProjection(sender=input_source.output_states[0],
                                              receiver=learning_mechanism.input_states[0])
        act_out_projection = MappingProjection(sender=output_source.output_states[0],
                                               receiver=learning_mechanism.input_states[1])
        # MODIFIED 7/22/19 OLD:
        error_signal_projection = MappingProjection(sender=comparator.output_states[OUTCOME],
                                                    receiver=learning_mechanism.input_states[2])
        return [target_projection, sample_projection, error_signal_projection, act_out_projection, act_in_projection]
        # # MODIFIED 7/22/19 XXX NEW:
        # error_signal_projections = []
        # for learning_mech in learning_mechanism.dependent_learning_mechanisms:
        #     error_signal_projections.append(MappingProjection(sender=comparator.output_states[OUTCOME],
        #                                                       receiver=learning_mechanism.input_states[2]))
        return [target_projection, sample_projection, error_signal_projections, act_out_projection, act_in_projection]
        # MODIFIED 7/22/19 END

    def _create_learning_projection(self, learning_mechanism, learned_projection):

        learning_projection = LearningProjection(name="Learning Projection",
                                                 sender=learning_mechanism.learning_signals[0],
                                                 receiver=learned_projection.parameter_states["matrix"])

        learned_projection.has_learning_projection = True

        return learning_projection

    def _unpack_processing_components_of_learning_pathway(self, processing_pathway):
        # unpack processing components and add to composition
        if len(processing_pathway) == 3:
            input_source, learned_projection, output_source = processing_pathway
        elif len(processing_pathway) == 2:
            input_source, output_source = processing_pathway
            learned_projection = MappingProjection(sender=input_source, receiver=output_source)
        else:
            raise CompositionError(f"Too many components in learning pathway: {pathway}. Only single-layer learning "
                                   f"is supported by this method. See AutodiffComposition for other learning models.")
        return input_source, output_source, learned_projection

    def add_reinforcement_learning_pathway(self, pathway, learning_rate=0.05, error_function=None,
                                           learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE):
        """
        Arguments
        ---------

        pathway: List
            list containing either [Node1, Node2] or [Node1, MappingProjection, Node2]. If a projection is
            specified, that projection is the learned projection. Otherwise, a default MappingProjection is
            automatically generated for the learned projection.

        learning_rate : float : default 0.05
            specifies the `learning_rate <ReinforcementLearning.learning_rate>` used for the `ReinforcementLearning`
            function of the `LearningMechanism` in the **pathway**.

        error_function : function : default LinearCombination
            specifies the function assigned to `ComparatorMechanism` used to compute the error from the target and
            the output (`value <Mechanism_Base.value>`) of the `TARGET` Mechanism in the **pathway**).

        learning_update : Optional[bool|ONLINE|AFTER] : default AFTER
            specifies when the `matrix <MappingProjection.matrix>` parameter of the `learned_projection` is updated
            in each `TRIAL` when the Composition executes;  it is assigned as the default value for the
            `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanism
            <LearningMechanism>` in the pathway, and its `LearningProjection` (see `learning_enabled
            <LearningMechanism.learning_enabled>` for meaning of values).

        Returns
        --------

        A dictionary of components that were automatically generated and added to the Composition in order to
        implement ReinforcementLearning in the pathway.

        {LEARNING_MECHANISM: learning_mechanism,
         COMPARATOR_MECHANISM: comparator,
         TARGET_MECHANISM: target,
         LEARNED_PROJECTION: learned_projection}
        """

        if not error_function:
            error_function = LinearCombination()

        # Processing Components
        input_source, output_source, learned_projection = \
            self._unpack_processing_components_of_learning_pathway(pathway)
        self.add_linear_processing_pathway([input_source, learned_projection, output_source])
        self.add_required_node_role(output_source, NodeRole.OUTPUT)

        # Learning Components
        target, comparator, learning_mechanism = self._create_rl_learning_related_mechanisms(input_source,
                                                                                             output_source,
                                                                                             error_function,
                                                                                             learned_projection,
                                                                                             learning_rate,
                                                                                             learning_update)
        self.add_nodes([(target, NodeRole.TARGET), comparator, learning_mechanism],
                       required_roles=NodeRole.LEARNING)

        learning_related_projections = self._create_learning_related_projections(input_source,
                                                                                 output_source,
                                                                                 target,
                                                                                 comparator,
                                                                                 learning_mechanism)
        self.add_projections(learning_related_projections)

        learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
        self.add_projection(learning_projection, learning_projection=True)

        learning_related_components = {LEARNING_MECHANISM: learning_mechanism,
                                       COMPARATOR_MECHANISM: comparator,
                                       TARGET_MECHANISM: target,
                                       LEARNED_PROJECTION: learned_projection}

        return learning_related_components

    def add_td_learning_pathway(self, pathway, learning_rate=0.05, error_function=None,
                                learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE):
        """
        Arguments
        ---------

        pathway: List
            list containing either [Node1, Node2] or [Node1, MappingProjection, Node2]. If a projection is
            specified, that projection is the learned projection. Otherwise, a default MappingProjection is
            automatically generated for the learned projection.

        learning_rate : float : default 0.05
            specifies the `learning_rate <TDLearning.learning_rate>` used for the `TDLearning` function of the
            `LearningMechanism` in the **pathway**.

        error_function : function : default LinearCombination
            specifies the function assigned to `ComparatorMechanism` used to compute the error from the target and
            the output (`value <Mechanism_Base.value>`) of the `TARGET` Mechanism in the **pathway**).

        learning_update : Optional[bool|ONLINE|AFTER] : default AFTER
            specifies when the `matrix <MappingProjection.matrix>` parameter of the `learned_projection` is updated
            in each `TRIAL` when the Composition executes;  it is assigned as the default value for the
            `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanism
            <LearningMechanism>` in the pathway, and its `LearningProjection` (see `learning_enabled
            <LearningMechanism.learning_enabled>` for meaning of values).

        Returns
        --------

        A dictionary of components that were automatically generated and added to the Composition in order to
        implement TDLearning in the pathway.

        {LEARNING_MECHANISM: learning_mechanism,
         COMPARATOR_MECHANISM: comparator,
         TARGET_MECHANISM: target,
         LEARNED_PROJECTION: learned_projection}
        """

        if not error_function:
            error_function = LinearCombination()

        # Processing Components
        input_source, output_source, learned_projection = \
            self._unpack_processing_components_of_learning_pathway(pathway)
        self.add_linear_processing_pathway([input_source, learned_projection, output_source])

        # Learning Components
        target, comparator, learning_mechanism = self._create_td_related_mechanisms(input_source,
                                                                                    output_source,
                                                                                    error_function,
                                                                                    learned_projection,
                                                                                    learning_rate,
                                                                                    learning_update)
        self.add_nodes([(target, NodeRole.TARGET), comparator, learning_mechanism], required_roles=NodeRole.LEARNING)

        learning_related_projections = self._create_learning_related_projections(input_source,
                                                                                 output_source,
                                                                                 target,
                                                                                 comparator,
                                                                                 learning_mechanism)
        self.add_projections(learning_related_projections)

        learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
        self.add_projection(learning_projection, learning_projection=True)

        learning_related_components = {LEARNING_MECHANISM: learning_mechanism,
                                       COMPARATOR_MECHANISM: comparator,
                                       TARGET_MECHANISM: target,
                                       LEARNED_PROJECTION: learned_projection}

        return learning_related_components

    def add_backpropagation_pathway(self, pathway, learning_rate=0.05, error_function=None,
                                     learning_update:tc.optional(tc.any(bool, tc.enum(ONLINE, AFTER)))=AFTER):
        """Add linear processing pathway with backpropogation learning

        Arguments
        ---------
        pathway : list
            specifies list of nodes for the pathway (see `add_linear_processing_pathway` for details of specification).

        pathway: List
            specifies nodes of the pathway for the learning sequence  (see `add_linear_processing_pathway` for
            details of specification).  Any `MappingProjections <MappingProjection>` specified or constructed for the
            pathway are assigned as `learned_projections`.

        learning_rate : float : default 0.05
            specifies the `learning_rate <Backpropagation.learning_rate>` used for the `Backpropagation` function of
            the `LearningMechanisms <LearningMechanism>` in the **pathway**.

        error_function : function : default LinearCombination
            specifies the function assigned to `ComparatorMechanism` used to compute the error from the target and the
            output (`value <Mechanism_Base.value>`) of the `TARGET` (last) Mechanism in the **pathway**).

        learning_update : Optional[bool|ONLINE|AFTER] : default AFTER
            specifies when the `matrix <MappingProjection.matrix>` parameters of the `learned_projections` are updated
            in each `TRIAL` when the Composition executes;  it is assigned as the default value for the
            `learning_enabled <LearningMechanism.learning_enabled>` attribute of the `LearningMechanisms
            <LearningMechanism>` in the pathway, and their `LearningProjections <LearningProjection>`
            (see `learning_enabled <LearningMechanism.learning_enabled>` for meaning of values).

        Returns
        --------

        A dictionary of components that were automatically generated and added to the Composition in order to
        implement Backpropagation along the pathway.

        {LEARNING_MECHANISM: learning_mechanism,
         COMPARATOR_MECHANISM: comparator,
         TARGET_MECHANISM: target,
         LEARNED_PROJECTION: learned_projection}
        """


        if not error_function:
            error_function = LinearCombination()

        # Add pathway to graph and get its full specification
        processing_pathway = self.add_linear_processing_pathway(pathway)

        path_length = len(processing_pathway)

        # Pathway length must be >=3 (Mechanism, Projection, Mechanism
        if path_length >= 3:
            # get the "terminal_sequence" --
            # the last 2 nodes in the back prop pathway and the projection between them
            # these components are are processed separately because
            # they inform the construction of the Target and Comparator mechs
            terminal_sequence = processing_pathway[path_length - 3: path_length]
        else:
            raise CompositionError(f"Backpropagation pathway specification ({pathway}) must not contain "
                                   f"at least three components "
                                   f"([{Mechanism.__name__}, {Projection.__name__}, {Mechanism.__name__}]).")

        # Unpack and process terminal_sequence:
        input_source, learned_projection, output_source = terminal_sequence

        # If pathway includes existing terminal_sequence for the output_source, use that
        if output_source in self._terminal_backprop_sequences:
            target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
            comparator = self._terminal_backprop_sequences[output_source][COMPARATOR_MECHANISM]
            learning_mechanism = self._terminal_backprop_sequences[output_source][LEARNING_MECHANISM]
            sequence_end = path_length-3

        # # MODIFIED 7/22/19 NEW: [JDC]
        # # FIX: ALTERNATIVE IS TO TEST WHETHER IT PROJECTIONS TO ANY MECHANISMS WITH LEARNING ROLE
        # Otherwise, if output_source already projects to a learning Mechanism, integrate with existing sequence
        elif any(isinstance(p.receiver.owner, LearningMechanism) for p in output_source.efferents):
            # FIX: ASSIGN target, comparator and learning_mechanism AS ABOVE?  OR CREATE NEW METHOD TO CREATE THEM?
            # Set learning_mechanism to the one to which output_source projects
            learning_mechanism = next((p.receiver.owner for p in output_source.efferents
                                       if isinstance(p.receiver.owner, LearningMechanism)))
            # # Use existing target and comparator to learning_mechanism for Mechanism to which output_source project
            # target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
            # comparator = self._terminal_backprop_sequences[output_source][COMPARATOR_MECHANISM]
            target = None
            comparator = None
            sequence_end = path_length-1
        # MODIFIED 7/22/19 END
        # Otherwise create terminal_sequence for the sequence,
        #    and eliminate existing terminal_sequences previously created for Mechanisms now in the pathway
        else:

            # MODIFIED 7/22/19 NEW: [JDC]
            # Eliminate existing comparators and targets for Mechanisms now in the pathway that were output_sources
            #   (i.e., ones that belong to previously-created sequences that overlap with the current one)
            old_learning_mechanisms = []
            for old_output_source in [m for m in pathway[:-1:] if isinstance(m, Mechanism)]:

                old_comparator = next((p.receiver.owner for p in old_output_source.efferents
                                       if (isinstance(p.receiver.owner, ComparatorMechanism)
                                           and p.receiver.owner in self.get_nodes_by_role(NodeRole.LEARNING))),
                                      None)
                if old_comparator:
                    old_target = next((p.sender.owner for p in old_comparator.input_states[TARGET].path_afferents
                                       if p.sender.owner in self.get_nodes_by_role(NodeRole.TARGET)),
                                      None)
                    self.remove_nodes([old_comparator, old_target])
                    Mechanism_Base._delete_mechanism(old_comparator)
                    Mechanism_Base._delete_mechanism(old_target)
                    old_learning_mechanisms.append(next((p.receiver.owner for p in old_output_source.efferents
                                                         if (isinstance(p.receiver.owner, LearningMechanism)
                                                             and p.receiver.owner in self.nodes
                                                             and ACTIVATION_OUTPUT in p.receiver.name )),
                                                        None))
            # MODIFIED 7/22/19 END

            # Create teminal_sequence
            target, comparator, learning_mechanism = \
                self._create_terminal_backprop_sequence_components(input_source,
                                                                   output_source,
                                                                   error_function,
                                                                   learned_projection,
                                                                   learning_rate,
                                                                   learning_update)
            self._terminal_backprop_sequences[output_source] = {LEARNING_MECHANISM: learning_mechanism,
                                                                TARGET_MECHANISM: target,
                                                                COMPARATOR_MECHANISM: comparator}

            # # MODIFIED 7/22/19 NEW:
            if old_learning_mechanisms:
                self.add_projection(sender=learning_mechanism.output_states[ERROR_SIGNAL],
                                    receiver=old_learning_mechanisms[0].input_states[ERROR_SIGNAL])
                old_learning_mechanisms[0].error_matrices[0] = \
                    learning_mechanism.learned_projections[0].parameter_states[MATRIX]
            # # MODIFIED 7/22/19 NEWER:
            # # Update old_learning_mechanism
            # for old_learning_mech in old_learning_mechanisms:
            #     # Get LearningMechanism(s) providing new source(s) of error_signal(s) (replaces old comparator)
            #     # FIX 7/22/19: STILL NEED TO ADD CHECK IF LEARNING PROJECTION
            #     #              p.parameter_states[MATRIX].mod_afferents[0]IS IN COMPOSITION
            #     for error_sources in [p.parameter_states[MATRIX].mod_afferents[0].sender.owner
            #                           for p in old_learning_mech.output_source.efferents
            #                           if (p.has_learning_projection and p in self.projections]:
            #         new_error_source = old_learning_mech.input_states[ACTIVATION_OUTPUT].path_afferents[0].sender.owner
            #         # Get index of error_signal associated with new error_source
            #         # Add projection from that
            #         self.add_projection(sender=new_error_source.output_states[ERROR_SIGNAL],
            #                             receiver=old_learning_mech.input_states[ERROR_SIGNAL])
            #         # Update old_learning_mech's error_matrices with new projection
            #         error_matrix = \
            #             old_learning_mech.input_states[ERROR_SIGNAL].path_afferents[0].sender.owner.learning_signal.receiver.matrix
            # MODIFIED 7/22/19 OLD


            sequence_end = path_length-3

        # loop backwards through the rest of the pathway to create and connect
        # the remaining learning mechanisms
        learning_mechanisms = [learning_mechanism]
        learned_projections = [learned_projection]
        for i in range(sequence_end, 1, -2):
            # set variables for this iteration
            input_source = processing_pathway[i-2]
            learned_projection = processing_pathway[i-1]
            output_source = processing_pathway[i]
            previous_learning_mechanism = learning_mechanisms[-1]

            new_learning_mechanism = self._create_multilayer_backprop_components(input_source,
                                                                                 output_source,
                                                                                 learned_projection,
                                                                                 learning_rate,
                                                                                 learning_update,
                                                                                 previous_learning_mechanism)

            learning_mechanisms.append(new_learning_mechanism)
            learned_projections.append(learned_projection)

        learning_related_components = {LEARNING_MECHANISM: learning_mechanisms,
                                       COMPARATOR_MECHANISM: comparator,
                                       TARGET_MECHANISM: target,
                                       LEARNED_PROJECTION: learned_projections}

        # MODIFIED 7/22/19 NEW: [JDC]
        self._analyze_graph()
        # MODIFIED 7/22/19 END

        return learning_related_components

    def _validate_projection(self,
                             projection,
                             sender, receiver,
                             graph_sender,
                             graph_receiver,
                             learning_projection,
                             ):

        # FIX: [JDC 6/8/19] SHOULDN'T THERE BE A CHECK FOR THEM LearningProjections? OR ARE THOSE DONE ELSEWHERE?
        # Skip this validation on learning projections because they have non-standard senders and receivers
        if not learning_projection:
            if projection.sender.owner != graph_sender:
                raise CompositionError("{}'s sender assignment [{}] is incompatible with the positions of these "
                                       "Components in the Composition.".format(projection, sender))
            if projection.receiver.owner != graph_receiver:
                raise CompositionError("{}'s receiver assignment [{}] is incompatible with the positions of these "
                                       "Components in the Composition.".format(projection, receiver))

    def _get_original_senders(self, input_state, projections):
        original_senders = set()
        for original_projection in projections:
            if original_projection in self.projections:
                original_senders.add(original_projection.sender)
                correct_sender = original_projection.sender
                shadow_found = False
                for shadow_projection in input_state.path_afferents:
                    if shadow_projection.sender == correct_sender:
                        shadow_found = True
                        break
                if not shadow_found:
                    # TBI - Shadow projection type? Matrix value?
                    new_projection = MappingProjection(sender=correct_sender,
                                                       receiver=input_state)
                    self.add_projection(new_projection, sender=correct_sender, receiver=input_state)
        return original_senders

    def _update_shadow_projections(self):
        for node in self.nodes:
            for input_state in node.input_states:
                if input_state.shadow_inputs:
                    original_senders = self._get_original_senders(input_state, input_state.shadow_inputs.path_afferents)
                    for shadow_projection in input_state.path_afferents:
                        if shadow_projection.sender not in original_senders:
                            self.remove_projection(shadow_projection)

            # If the node does not have any roles, it is internal
            if len(self.get_roles_by_node(node)) == 0:
                self._add_node_role(node, NodeRole.INTERNAL)

    def _check_for_projection_assignments(self):
        """Check that all Projections and States with require_projection_in_composition attribute are configured.

        Validate that all InputStates with require_projection_in_composition == True have an afferent Projection.
        Validate that all OuputStates with require_projection_in_composition == True have an efferent Projection.
        Validate that all Projections have senders and receivers.
        """
        projections = self.projections.copy()

        for node in self.nodes:
            if isinstance(node, Projection):
                projections.append(node)
                continue
            for input_state in node.input_states:
                if input_state.require_projection_in_composition and not input_state.path_afferents:
                    warnings.warn(f'{InputState.__name__} ({input_state.name}) of {node.name} '
                                  f'doesn\'t have any afferent {Projection.__name__}s')
            for output_state in node.output_states:
                if output_state.require_projection_in_composition and not output_state.efferents:
                    warnings.warn(f'{OutputState.__name__} ({output_state.name}) of {node.name} '
                                  f'doesn\'t have any efferent {Projection.__name__}s')

        for projection in projections:
            if not projection.sender:
                warnings.warn(f'{Projection.__name__} {projection.name} is missing a sender')
            if not projection.receiver:
                warnings.warn(f'{Projection.__name__} {projection.name} is missing a receiver')

    def _check_for_existing_projection(self,
                                         projection=None,
                                         sender=None,
                                         receiver=None):
        assert projection or (sender and receiver), \
            f'_check_for_existing_projection must be passed a projection or a sender and receiver'

        if projection:
            sender = projection.sender
            receiver = projection.receiver
        else:
            if isinstance(sender, Mechanism):
                sender = sender.output_state
            elif isinstance(sender, Composition):
                sender = sender.output_CIM.output_state
            if isinstance(receiver, Mechanism):
                receiver = receiver.input_state
            elif isinstance(receiver, Composition):
                receiver = receiver.input_CIM.input_state

        if [proj for proj in sender.efferents if proj.receiver is receiver and proj in self.projections]:
            return True
        return False

    def _analyze_consideration_queue(self, q, objective_mechanism):
        """Assigns NodeRole.ORIGIN to all nodes in the first entry of the consideration queue and NodeRole.TERMINAL to
            all nodes in the last entry of the consideration queue. The ObjectiveMechanism of a controller
            may not be NodeRole.TERMINAL, so if the ObjectiveMechanism is the only node in the last entry of the
            consideration queue, then the second-to-last entry is NodeRole.TERMINAL instead. """
        for node in q[0]:
            self._add_node_role(node, NodeRole.ORIGIN)

        for node in list(q)[-1]:
            if node != objective_mechanism:
                self._add_node_role(node, NodeRole.TERMINAL)
            elif len(q[-1]) < 2:
                for previous_node in q[-2]:
                    self._add_node_role(previous_node, NodeRole.TERMINAL)

    def _determine_node_roles(self):
        # Clear old roles
        self.nodes_to_roles.update({k: set() for k in self.nodes_to_roles})

        # Required Roles
        for node_role_pair in self.required_node_roles:
            self._add_node_role(node_role_pair[0], node_role_pair[1])

        objective_mechanism = None
        if self.controller and self.enable_controller and self.controller.objective_mechanism:
            objective_mechanism = self.controller.objective_mechanism
            self._add_node_role(objective_mechanism, NodeRole.OBJECTIVE)

        # Use Scheduler.consideration_queue to check for ORIGIN and TERMINAL Nodes:
        if self.scheduler_processing.consideration_queue:
            self._analyze_consideration_queue(self.scheduler_processing.consideration_queue, objective_mechanism)

        # Cycles
        for node in self.scheduler_processing.cycle_nodes:
            self._add_node_role(node, NodeRole.CYCLE)

        # "Feedback" projections
        for node in self.feedback_senders:
            self._add_node_role(node, NodeRole.FEEDBACK_SENDER)

        for node in self.feedback_receivers:
            self._add_node_role(node, NodeRole.FEEDBACK_RECEIVER)

        # Required Roles
        for node_role_pair in self.required_node_roles:
            self._add_node_role(node_role_pair[0], node_role_pair[1])

        # If INPUT nodes were not specified by user, ORIGIN nodes become INPUT nodes
        if not self.get_nodes_by_role(NodeRole.INPUT):
            origin_nodes = self.get_nodes_by_role(NodeRole.ORIGIN)
            for node in origin_nodes:
                self._add_node_role(node, NodeRole.INPUT)

        # If OUTPUT nodes were not specified by user, TERMINAL nodes become OUTPUT nodes.
        # If there are LearningMechanisms, OUTPUT node is the last non-learning-related node.
        # If there are no TERMINAL nodes either, then the last node added to the Composition becomes the OUTPUT node.
        if not self.get_nodes_by_role(NodeRole.OUTPUT):
            if self.get_nodes_by_role(NodeRole.LEARNING):
                # FIX: ADD COMMENT HERE
                # terminal_nodes = [[n for n in self.nodes if not NodeRole.LEARNING in self.nodes_to_roles[n]][-1]]
                terminal_nodes = list([items for items in self.scheduler_processing.consideration_queue
                                       if any([item for item in items
                                               if not NodeRole.LEARNING in self.nodes_to_roles[item]])])[-1]
            else:
                terminal_nodes = self.get_nodes_by_role(NodeRole.TERMINAL)
            if not terminal_nodes:
                try:
                    # FIX: ADD COMMENT HERE
                    terminal_nodes = list([items for items in self.scheduler_processing.consideration_queue
                                           if any([item for item in items
                                                   if not NodeRole.LEARNING in self.nodes_to_roles[item]])])[-1]
                except IndexError:
                    terminal_nodes = []
            for node in terminal_nodes:
                self._add_node_role(node, NodeRole.OUTPUT)

    def _analyze_graph(self):
        """
        Assigns `NodeRoles <NodeRoles>` to nodes based on the structure of the `Graph`.

        By default, if _analyze_graph determines that a node is `ORIGIN <NodeRole.ORIGIN>`, it is also given the role
        `INPUT <NodeRole.INPUT>`. Similarly, if _analyze_graph determines that a node is `TERMINAL
        <NodeRole.TERMINAL>`, it is also given the role `OUTPUT <NodeRole.OUTPUT>`.

        However, if the required_roles argument of `add_node <Composition.add_node>` is used to set any node in the
        Composition to `INPUT <NodeRole.INPUT>`, then the `ORIGIN <NodeRole.ORIGIN>` nodes are not set to `INPUT
        <NodeRole.INPUT>` by default. If the required_roles argument of `add_node <Composition.add_node>` is used
        to set any node in the Composition to `OUTPUT <NodeRole.OUTPUT>`, then the `TERMINAL <NodeRole.TERMINAL>`
        nodes are not set to `OUTPUT <NodeRole.OUTPUT>` by default.

        :param graph:
        :param context:
        :return:
        """

        self._determine_node_roles()
        self._create_CIM_states()
        self._update_shadow_projections()
        self._check_for_projection_assignments()
        self.needs_update_graph = False

    def _update_processing_graph(self):
        """
        Constructs the processing graph (the graph that contains only Nodes as vertices)
        from the composition's full graph
        """
        logger.debug('Updating processing graph')

        self._graph_processing = self.graph.copy()

        def remove_vertex(vertex):
            logger.debug('Removing', vertex)
            for parent in vertex.parents:
                for child in vertex.children:
                    if vertex.feedback:
                        child.backward_sources.add(parent.component)
                    self._graph_processing.connect_vertices(parent, child)
            # ensure that children get handled
            if len(vertex.parents) == 0:
                for child in vertex.children:
                    if vertex.feedback:
                        child.backward_sources.add(parent.component)

            for node in cur_vertex.parents + cur_vertex.children:
                logger.debug(
                    'New parents for vertex {0}: \n\t{1}\nchildren: \n\t{2}'.format(
                        node, node.parents, node.children
                    )
                )

            logger.debug('Removing vertex {0}'.format(cur_vertex))

            self._graph_processing.remove_vertex(vertex)

        # copy to avoid iteration problems when deleting
        vert_list = self._graph_processing.vertices.copy()
        for cur_vertex in vert_list:
            logger.debug('Examining', cur_vertex)
            if not cur_vertex.component.is_processing:
                remove_vertex(cur_vertex)

        self.needs_update_graph_processing = False

    def get_nodes_by_role(self, role):
        """
            Returns a List of Composition Nodes in this Composition that have the *role* specified

            Arguments
            _________

            role : NodeRole
                the List of nodes having this role to return

            Returns
            -------

            List of Composition Nodes with `NodeRole` *role* : List(`Mechanisms <Mechanism>` and
            `Compositions <Composition>`)
        """
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        try:
            return [node for node in self.nodes if role in self.nodes_to_roles[node]]

        except KeyError as e:
            raise CompositionError('Node missing from {0}.nodes_to_roles: {1}'.format(self, e))

    def get_roles_by_node(self, node):
        try:
            return self.nodes_to_roles[node]
        except KeyError:
            raise CompositionError('Node {0} not found in {1}.nodes_to_roles'.format(node, self))

    def _set_node_roles(self, node, roles):
        self._clear_node_roles(node)
        for role in roles:
            self._add_node_role(role)

    def _clear_node_roles(self, node):
        if node in self.nodes_to_roles:
            self.nodes_to_roles[node] = set()

    def _add_node_role(self, node, role):
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        self.nodes_to_roles[node].add(role)

    def _remove_node_role(self, node, role):
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        self.nodes_to_roles[node].remove(role)

    tc.typecheck

    def _get_nested_node_CIM_state(self,
                                   node: Mechanism,
                                   node_state: tc.any(InputState, OutputState),
                                   role: tc.enum(NodeRole.INPUT, NodeRole.OUTPUT)
                                   ):
        """Check for node in nested Composition
        Return relevant state of relevant CIM if found and nested Composition in which it was found, else (None, None)
        """

        nested_comp = CIM_state_for_nested_node = CIM = None

        nested_comps = [c for c in self.nodes if isinstance(c, Composition)]
        for nc in nested_comps:
            if node in nc.nodes:
                # Must be assigned Node.Role of INPUT or OUTPUT (depending on receiver vs sender)
                if role not in nc.nodes_to_roles[node]:
                    raise CompositionError("{} found in nested {} of {} ({}) but without required {} ({})".
                                           format(node.name, Composition.__name__, self.name, nc.name,
                                                  NodeRole.__name__, repr(role)))
                # With the current implementation, there should never be multiple nested compositions that contain the
                # same mechanism -- because all nested compositions are passed the same execution ID
                # if CIM_state_for_nested_node:
                #     warnings.warn("{} found with {} of {} in more than one nested {} of {}; "
                #                   "only first one found (in {}) will be used".
                #                   format(node.name, NodeRole.__name__, repr(role),
                #                          Composition.__name__, self.name, nested_comp.name))
                #     continue

                if isinstance(node_state, InputState):
                    CIM_state_for_nested_node = nc.input_CIM_states[node_state][0]
                    CIM = nc.input_CIM
                elif isinstance(node_state, OutputState):
                    CIM_state_for_nested_node = nc.output_CIM_states[node_state][1]
                    CIM = nc.output_CIM
                else:
                    # IMPLEMENTATION NOTE:  Place marker for future implementation of ParameterState handling
                    #                       However, typecheck above should have caught this
                    assert False

                nested_comp = nc
                break

        return CIM_state_for_nested_node, CIM_state_for_nested_node, nested_comp, CIM

    def add_required_node_role(self, node, role):
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        node_role_pair = (node, role)
        if node_role_pair not in self.required_node_roles:
            self.required_node_roles.append(node_role_pair)

    def remove_required_node_role(self, node, role):
        if role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        node_role_pair = (node, role)
        if node_role_pair in self.required_node_roles:
            self.required_node_roles.remove(node_role_pair)

    def _create_CIM_states(self, context=None):

        """
            - remove the default InputState and OutputState from the CIMs if this is the first time that real
              InputStates and OutputStates are being added to the CIMs

            - create a corresponding InputState and OutputState on the `input_CIM <Composition.input_CIM>` for each
              InputState of each INPUT node. Connect the OutputState on the input_CIM to the INPUT node's corresponding
              InputState via a standard MappingProjection.

            - create a corresponding InputState and OutputState on the `output_CIM <Composition.output_CIM>` for each
              OutputState of each OUTPUT node. Connect the OUTPUT node's OutputState to the output_CIM's corresponding
              InputState via a standard MappingProjection.

            - build two dictionaries:

                (1) input_CIM_states = { INPUT Node InputState: (InputCIM InputState, InputCIM OutputState) }

                (2) output_CIM_states = { OUTPUT Node OutputState: (OutputCIM InputState, OutputCIM OutputState) }

            - if the Node has any shadows, create the appropriate projections as needed.

            - delete all of the above for any node States which were previously, but are no longer, classified as
              INPUT/OUTPUT

        """

        if not self.input_CIM.connected_to_composition:
            self.input_CIM.input_states.remove(self.input_CIM.input_state)
            self.input_CIM.output_states.remove(self.input_CIM.output_state)
            self.input_CIM.connected_to_composition = True

        if not self.output_CIM.connected_to_composition:
            self.output_CIM.input_states.remove(self.output_CIM.input_state)
            self.output_CIM.output_states.remove(self.output_CIM.output_state)
            self.output_CIM.connected_to_composition = True

        current_input_node_input_states = set()

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        for node in input_nodes:

            for input_state in node.external_input_states:
                # add it to our set of current input states
                current_input_node_input_states.add(input_state)

                # if there is not a corresponding CIM output state, add one
                if input_state not in set(self.input_CIM_states.keys()):
                    interface_input_state = InputState(owner=self.input_CIM,
                                                       variable=input_state.defaults.value,
                                                       reference_value=input_state.defaults.value,
                                                       name="INPUT_CIM_" + node.name + "_" + input_state.name)

                    interface_output_state = OutputState(owner=self.input_CIM,
                                                         variable=OWNER_VALUE,
                                                         default_variable=self.input_CIM.defaults.variable,
                                                         function=InterfaceStateMap(
                                                             corresponding_input_state=interface_input_state),
                                                         name="INPUT_CIM_" + node.name + "_" + input_state.name)

                    self.input_CIM_states[input_state] = [interface_input_state, interface_output_state]

                    projection = MappingProjection(sender=interface_output_state,
                                                   receiver=input_state,
                                                   matrix=IDENTITY_MATRIX,
                                                   name="(" + interface_output_state.name + ") to ("
                                                        + input_state.owner.name + "-" + input_state.name + ")")
                    projection._activate_for_compositions(self)

                    if isinstance(node, Composition):
                        projection._activate_for_compositions(node)

        new_shadow_projections = {}

        # for any entirely new shadow_projections, create a MappingProjection object and add to projections
        for output_state, input_state in new_shadow_projections:
            if new_shadow_projections[(output_state, input_state)] is None:
                shadow_projection = MappingProjection(sender=output_state,
                                                      receiver=input_state,
                                                      name="(" + output_state.name + ") to ("
                                                           + input_state.owner.name + "-" + input_state.name + ")")
                shadow_projection._activate_for_compositions(self)

        sends_to_input_states = set(self.input_CIM_states.keys())

        # For any states still registered on the CIM that does not map to a corresponding INPUT node I.S.:
        for input_state in sends_to_input_states.difference(current_input_node_input_states):
            for projection in input_state.path_afferents:
                if projection.sender == self.input_CIM_states[input_state][1]:
                    # remove the corresponding projection from the INPUT node's path afferents
                    input_state.path_afferents.remove(projection)

                    # projection.receiver.efferents.remove(projection)
                    # Bug? ^^ projection is not in receiver.efferents??
                    if projection.receiver.owner in self.shadows and len(self.shadows[projection.receiver.owner]) > 0:
                        for shadow in self.shadows[projection.receiver.owner]:
                            for shadow_input_state in shadow.input_states:
                                for shadow_projection in shadow_input_state.path_afferents:
                                    if shadow_projection.sender == self.input_CIM_states[input_state][1]:
                                        shadow_input_state.path_afferents.remove(shadow_projection)

            # remove the CIM input and output states associated with this INPUT node input state
            self.input_CIM.input_states.remove(self.input_CIM_states[input_state][0])
            self.input_CIM.output_states.remove(self.input_CIM_states[input_state][1])

            # and from the dictionary of CIM output state/input state pairs
            del self.input_CIM_states[input_state]

        # OUTPUT CIMS
        # loop over all OUTPUT nodes
        current_output_node_output_states = set()
        for node in self.get_nodes_by_role(NodeRole.OUTPUT):
            for output_state in node.output_states:
                current_output_node_output_states.add(output_state)
                # if there is not a corresponding CIM output state, add one
                if output_state not in set(self.output_CIM_states.keys()):

                    interface_input_state = InputState(owner=self.output_CIM,
                                                       variable=output_state.defaults.value,
                                                       reference_value=output_state.defaults.value,
                                                       name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    interface_output_state = OutputState(
                        owner=self.output_CIM,
                        variable=OWNER_VALUE,
                        function=InterfaceStateMap(corresponding_input_state=interface_input_state),
                        reference_value=output_state.defaults.value,
                        name="OUTPUT_CIM_" + node.name + "_" + output_state.name)

                    self.output_CIM_states[output_state] = [interface_input_state, interface_output_state]

                    proj_name = "(" + output_state.name + ") to (" + interface_input_state.name + ")"

                    proj = MappingProjection(
                        sender=output_state,
                        receiver=interface_input_state,
                        matrix=IDENTITY_MATRIX,
                        name=proj_name
                    )
                    proj._activate_for_compositions(self)
                    if isinstance(node, Composition):
                        proj._activate_for_compositions(node)

        previous_output_node_output_states = set(self.output_CIM_states.keys())
        for output_state in previous_output_node_output_states.difference(current_output_node_output_states):
            # remove the CIM input and output states associated with this Terminal Node output state
            self.output_CIM.remove_states(self.output_CIM_states[output_state][0])
            self.output_CIM.remove_states(self.output_CIM_states[output_state][1])
            del self.output_CIM_states[output_state]

    def _assign_values_to_input_CIM(self, inputs, execution_id=None):
        """
            Assign values from input dictionary to the InputStates of the Input CIM, then execute the Input CIM

        """

        build_CIM_input = []

        for input_state in self.input_CIM.input_states:
            # "input_state" is an InputState on the input CIM

            for key in self.input_CIM_states:
                # "key" is an InputState on an origin Node of the Composition
                if self.input_CIM_states[key][0] == input_state:
                    origin_input_state = key
                    origin_node = key.owner
                    index = origin_node.input_states.index(origin_input_state)

                    if isinstance(origin_node, CompositionInterfaceMechanism):
                        index = origin_node.input_states.index(origin_input_state)
                        origin_node = origin_node.composition

                    if origin_node in inputs:
                        value = inputs[origin_node][index]

                    else:
                        value = origin_node.defaults.variable[index]

            build_CIM_input.append(value)

        self.input_CIM.execute(build_CIM_input, execution_id=execution_id)

    def _assign_execution_ids(self, execution_id=None):
        """
            assigns the same execution id to each Node in the composition's processing graph as well as the CIMs.
            The execution id is either specified in the user's call to run(), or from the Composition's
            **default_execution_id**
        """

        # Traverse processing graph and assign one uuid to all of its nodes
        if execution_id is None:
            execution_id = self.default_execution_id

        if execution_id not in self.execution_ids:
            self.execution_ids.add(execution_id)

        return execution_id

    def _identify_clamp_inputs(self, list_type, input_type, origins):
        # clamp type of this list is same as the one the user set for the whole composition; return all nodes
        if list_type == input_type:
            return origins
        # the user specified different types of clamps for each origin node; generate a list accordingly
        elif isinstance(input_type, dict):
            return [k for k, v in input_type.items() if list_type == v]
        # clamp type of this list is NOT same as the one the user set for the whole composition; return empty list
        else:
            return []

    def _parse_runtime_params(self, runtime_params):
        if runtime_params is None:
            return {}
        for node in runtime_params:
            for param in runtime_params[node]:
                if isinstance(runtime_params[node][param], tuple):
                    if len(runtime_params[node][param]) == 1:
                        runtime_params[node][param] = (runtime_params[node][param], Always())
                    elif len(runtime_params[node][param]) != 2:
                        raise CompositionError(
                            "Invalid runtime parameter specification ({}) for {}'s {} parameter in {}. "
                            "Must be a tuple of the form (parameter value, condition), or simply the "
                            "parameter value. ".format(runtime_params[node][param],
                                                       node.name,
                                                       param,
                                                       self.name))
                else:
                    runtime_params[node][param] = (runtime_params[node][param], Always())
        return runtime_params

    def _get_graph_node_label(self, item, show_dimensions=None):
        if not isinstance(item, (Mechanism, Composition, Projection)):
            raise CompositionError("Unrecognized node type ({}) in graph for {}".format(item, self.name))
        # TBI Show Dimensions
        name = item.name

        if show_dimensions in {ALL, MECHANISMS} and isinstance(item, Mechanism):
            input_str = "in ({})".format(",".join(str(input_state.socket_width)
                                                  for input_state in item.input_states))
            output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_state.value)))
                                                    for output_state in item.output_states))
            return "{}\n{}\n{}".format(output_str, name, input_str)
        if show_dimensions in {ALL, PROJECTIONS} and isinstance(item, Projection):
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

        if isinstance(item, CompositionInterfaceMechanism):
            name = name.replace('Input_CIM','INPUT')
            name = name.replace('Output_CIM', 'OUTPUT')

        return name

    def _set_up_animation(self, execution_id):

        self._component_animation_execution_count = None

        if isinstance(self._animate, dict):
            # Assign directory for animation files
            from psyneulink._version import root_dir
            default_dir = root_dir + '/../show_graph output/GIFs/' + self.name # + " gifs"
            # try:
            #     rmtree(self._animate_directory)
            # except:
            #     pass
            self._animate_unit = self._animate.pop(UNIT, EXECUTION_SET)
            self._image_duration = self._animate.pop(DURATION, 0.75)
            self._animate_num_runs = self._animate.pop(NUM_RUNS, 1)
            self._animate_num_trials = self._animate.pop(NUM_TRIALS, 1)
            self._animate_simulations = self._animate.pop(SIMULATIONS, False)
            self._movie_filename = self._animate.pop(MOVIE_NAME, self.name + ' movie') + '.gif'
            self._animation_directory = self._animate.pop(MOVIE_DIR, default_dir)
            self._save_images = self._animate.pop(SAVE_IMAGES, False)
            self._show_animation = self._animate.pop(SHOW, False)
            if not self._animate_unit in {COMPONENT, EXECUTION_SET}:
                raise SystemError(f"{repr(UNIT)} entry of {repr('animate')} argument for {self.name} method "
                                  f"of {repr('run')} ({self._animate_unit}) "
                                  f"must be {repr(COMPONENT)} or {repr(EXECUTION_SET)}.")
            if not isinstance(self._image_duration, (int, float)):
                raise SystemError(f"{repr(DURATION)} entry of {repr('animate')} argument for {repr('run')} method of "
                                  f"{self.name} ({self._image_duration}) must be an int or a float.")
            if not isinstance(self._animate_num_runs, int):
                raise SystemError(f"{repr(NUM_RUNS)} entry of {repr('animate')} argument for {repr('show_graph')} "
                                  f"method of {self.name} ({self._animate_num_runs}) must an integer.")
            if not isinstance(self._animate_num_trials, int):
                raise SystemError(f"{repr(NUM_TRIALS)} entry of {repr('animate')} argument for {repr('show_graph')} "
                                  f"method of {self.name} ({self._animate_num_trials}) must an integer.")
            if not isinstance(self._animate_simulations, bool):
                raise SystemError(f"{repr(SIMULATIONS)} entry of {repr('animate')} argument for {repr('show_graph')} "
                                  f"method of {self.name} ({self._animate_num_trials}) must a boolean.")
            if not isinstance(self._animation_directory, str):
                raise SystemError(f"{repr(MOVIE_DIR)} entry of {repr('animate')} argument for {repr('run')} "
                                  f"method of {self.name} ({self._animation_directory}) must be a string.")
            if not isinstance(self._movie_filename, str):
                raise SystemError(f"{repr(MOVIE_NAME)} entry of {repr('animate')} argument for {repr('run')} "
                                  f"method of {self.name} ({self._movie_filename}) must be a string.")
            if not isinstance(self._save_images, bool):
                raise SystemError(f"{repr(SAVE_IMAGES)} entry of {repr('animate')} argument for {repr('run')} method "
                                  f"of {self.name} ({self._save_images}) must be a boolean")
            if not isinstance(self._show_animation, bool):
                raise SystemError(f"{repr(SHOW)} entry of {repr('animate')} argument for {repr('run')} "
                                  f"method of {self.name} ({self._show_animation}) must be a boolean.")
        elif self._animate:
            # self._animate should now be False or a dict
            raise SystemError("{} argument for {} method of {} ({}) must be a boolean or "
                              "a dictionary of argument specifications for its {} method".
                              format(repr('animate'), repr('run'), self.name, self._animate, repr('show_graph')))

    def _animate_execution(self, active_items, execution_id):
        if self._component_animation_execution_count is None:
            self._component_animation_execution_count = 0
        else:
            self._component_animation_execution_count += 1
        self.show_graph(active_items=active_items,
                        **self._animate,
                        output_fmt='gif',
                        execution_id=execution_id
                        )

    @tc.typecheck
    def show_structure(self,
                       # direction = 'BT',
                       show_functions:bool=False,
                       show_values:bool=False,
                       use_labels:bool=False,
                       show_headers:bool=False,
                       show_roles:bool=False,
                       show_conditions:bool=False,
                       system=None,
                       composition=None,
                       condition:tc.optional(Condition)=None,
                       compact_cim:tc.optional(tc.enum(INPUT, OUTPUT))=None,
                       output_fmt:tc.enum('pdf','struct')='pdf'
                       ):
        """Generate a detailed display of a the structure of a Mechanism.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        Displays the structure of a Mechanism using the GraphViz `record
        <http://graphviz.readthedocs.io/en/stable/examples.html#structs-revisited-py>`_ shape.  This method is called
        by `System.show_graph` if its **show_mechanism_structure** argument is specified as `True` when it is called.

        Arguments
        ---------

        show_functions : bool : default False
            show the `function <Component.function>` of the Mechanism and each of its States.

        show_mech_function_params : bool : default False
            show the parameters of the Mechanism's `function <Component.function>` if **show_functions** is True.

        show_state_function_params : bool : default False
            show parameters for the `function <Component.function>` of the Mechanism's States if **show_functions** is
            True).

        show_values : bool : default False
            show the `value <Component.value>` of the Mechanism and each of its States (prefixed by "=").

        use_labels : bool : default False
            use labels for values if **show_values** is `True`; labels must be specified in the `input_labels_dict
            <Mechanism.input_labels_dict>` (for InputState values) and `output_labels_dict
            <Mechanism.output_labels_dict>` (for OutputState values); otherwise it is ignored.

        show_headers : bool : default False
            show the Mechanism, InputState, ParameterState and OutputState headers.

        show_roles : bool : default False
            show the `roles <Composition.NodeRoles>` of each Mechanism in the `Composition`.

        show_conditions : bool : default False
            show the `conditions <Condition>` used by `Composition` to determine whether/when to execute each Mechanism.

        system : System : default None
            specifies the `System` (to which the Mechanism must belong) for which to show its role (see **roles**);
            if this is not specified, the **show_roles** argument is ignored.

        composition : Composition : default None
            specifies the `Composition` (to which the Mechanism must belong) for which to show its role (see **roles**);
            if this is not specified, the **show_roles** argument is ignored.

        compact_cim : *INPUT* or *OUTUPT* : default None
            specifies whether to suppress InputState fields for input_CIM and OutputState fields for output_CIM.

        output_fmt : keyword : default 'pdf'
            'pdf': generate and open a pdf with the visualization;\n
            'jupyter': return the object (ideal for working in jupyter/ipython notebooks)\n
            'struct': return a string that specifies the structure of a mechanism,
            for use in a GraphViz node specification.

        """
        if composition:
            system = composition
        open_bracket = r'{'
        pipe = r' | '
        close_bracket = r'}'
        mechanism_header = r'COMPOSITION:\n'
        input_states_header = r'______CIMINPUTSTATES______\n' \
                              r'/\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ' \
                              r'\ \ \ \ \ \ \ \ \ \ \\'
        output_states_header = r'\\______\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ______/' \
                               r'\nCIMOUTPUTSTATES'

        def mech_string(mech):
            """Return string with name of mechanism possibly with function and/or value
            Inclusion of role, function and/or value is determined by arguments of call to show_structure """
            if show_headers:
                mech_header = mechanism_header
            else:
                mech_header = ''
            mech_name = r' <{0}> {1}{0}'.format(mech.name, mech_header)
            mech_role = ''
            if system and show_roles:
                try:
                    mech_role = r'\n[{}]'.format(self.systems[system])
                except KeyError:
                    # # mech_role = r'\n[{}]'.format(self.system)
                    # mech_role = r'\n[CONTROLLER]'
                    from psyneulink.core.components.mechanisms.adaptive.control.controlmechanism import \
                        ControlMechanism
                    from psyneulink.core.components.mechanisms.processing.objectivemechanism import \
                        ObjectiveMechanism
                    if isinstance(mech, ControlMechanism) and hasattr(mech, 'system'):
                        mech_role = r'\n[CONTROLLER]'
                    elif isinstance(mech, ObjectiveMechanism) and hasattr(mech, '_role'):
                        mech_role = r'\n[{}]'.format(mech._role)
                    else:
                        mech_role = ""

            mech_function = ''
            if show_functions:
                mech_function = r'\n({})'.format(mech.function.__class__.__name__)
            mech_value = ''
            if show_values:
                mech_value = r'\n={}'.format(mech.value)
            return mech_name + mech_role + mech_function + mech_value

        from psyneulink.core.globals.utilities import ContentAddressableList
        def states_string(state_list: ContentAddressableList,
                          state_type,
                          include_function: bool = False,
                          include_value: bool = False,
                          use_label: bool = False):
            """Return string with name of states in ContentAddressableList with functions and/or values as specified"""
            states = open_bracket
            for i, state in enumerate(state_list):
                if i:
                    states += pipe
                function = ''
                if include_function:
                    function = r'\n({})'.format(state.function.__class__.__name__)
                value = ''
                if include_value:
                    if use_label:
                        value = r'\n={}'.format(state.label)
                    else:
                        value = r'\n={}'.format(state.value)
                states += r'<{0}-{1}> {1}{2}{3}'.format(state_type.__name__,
                                                        state.name,
                                                        function,
                                                        value)
            states += close_bracket
            return states

        # Construct Mechanism specification
        mech = mech_string(self)

        # Construct InputStates specification
        if len(self.input_states) and compact_cim is not INPUT:
            if show_headers:
                input_states = input_states_header + pipe + states_string(self.input_states,
                                                                          InputState,
                                                                          include_function=show_functions,
                                                                          include_value=show_values,
                                                                          use_label=use_labels)
            else:
                input_states = states_string(self.input_states,
                                             InputState,
                                             include_function=show_functions,
                                             include_value=show_values,
                                             use_label=use_labels)
            input_states = pipe + input_states
        else:
            input_states = ''

        # Construct OutputStates specification
        if len(self.output_states) and compact_cim is not OUTPUT:
            if show_headers:
                output_states = states_string(self.output_states,
                                              OutputState,
                                              include_function=show_functions,
                                              include_value=show_values,
                                              use_label=use_labels) + pipe + output_states_header
            else:
                output_states = states_string(self.output_states,
                                              OutputState,
                                              include_function=show_functions,
                                              include_value=show_values,
                                              use_label=use_labels)

            output_states = output_states + pipe
        else:
            output_states = ''

        m_node_struct = open_bracket + \
                        output_states + \
                        open_bracket + mech + close_bracket + \
                        input_states + \
                        close_bracket

        if output_fmt == 'struct':
            # return m.node
            return m_node_struct

        # Make node
        import graphviz as gv
        m = gv.Digraph(  # 'mechanisms',
            # filename='mechanisms_revisited.gv',
            node_attr={'shape': 'record'},
        )
        m.node(self.name, m_node_struct, shape='record')

        if output_fmt == 'pdf':
            m.view(self.name.replace(" ", "-"), cleanup=True)

        elif output_fmt == 'jupyter':
            return m

    def _generate_gifs(self, G, active_items, execution_id):

        def create_phase_string(phase):
            return f'%16s' % phase + ' - '

        def create_time_string(time, spec):
            if spec == 'TIME':
                r = time.run
                t = time.trial
                p = time.pass_
                ts = time.time_step
            else:
                r = t = p = ts = '__'
            return f"Time(run: %2s, " % r + f"trial: %2s, " % t + f"pass: %2s, " % p + f"time_step: %2s)" % ts

        G.format = 'gif'
        execution_phase = self.parameters.context.get(execution_id).execution_phase
        time = self.scheduler_processing.get_clock(execution_id).time
        run_num = time.run
        trial_num = time.trial

        if INITIAL_FRAME in active_items:
            phase_string = create_phase_string('Initializing')
            time_string = create_time_string(time, 'BLANKS')

        elif execution_phase == ContextFlags.PROCESSING:
            phase_string = create_phase_string('Processing Phase')
            time_string = create_time_string(time, 'TIME')
        # elif execution_phase == ContextFlags.LEARNING:
        #     time = self.scheduler_learning.get_clock(execution_id).time
        #     time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}". \
        #         format(run_num, time.trial, time.pass_, time.time_step)
        #     phase_string = 'Learning Phase - '

        elif execution_phase == ContextFlags.CONTROL:
            phase_string = create_phase_string('Control Phase')
            time_string = create_time_string(time, 'TIME')

        else:
            raise CompositionError(
                f"PROGRAM ERROR:  Unrecognized phase during execution of {self.name}: {execution_phase.name}")

        label = f'\n{self.name}\n{phase_string}{time_string}\n'
        G.attr(label=label)
        G.attr(labelloc='b')
        G.attr(fontname='Monaco')
        G.attr(fontsize='14')
        index = repr(self._component_animation_execution_count)
        image_filename = '-'.join([repr(run_num), repr(trial_num), index])
        image_file = self._animation_directory + '/' + image_filename + '.gif'
        G.render(filename=image_filename,
                 directory=self._animation_directory,
                 cleanup=True,
                 # view=True
                 )
        # Append gif to self._animation
        image = Image.open(image_file)
        # TBI?
        # if not self._save_images:
        #     remove(image_file)
        if not hasattr(self, '_animation'):
            self._animation = [image]
        else:
            self._animation.append(image)
        assert True

    @tc.typecheck
    def show_graph(self,
                   show_node_structure:tc.any(bool, tc.enum(VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS,
                                                            STATE_FUNCTION_PARAMS, ROLES, ALL))=False,
                   show_nested:tc.optional(tc.any(bool,dict,tc.enum(ALL)))=ALL,
                   show_controller:bool=False,
                   show_cim:bool=False,
                   show_learning:bool=False,
                   show_headers:bool=True,
                   show_dimensions:bool=False,
                   show_projection_labels:bool=False,
                   direction:tc.enum('BT', 'TB', 'LR', 'RL')='BT',
                   # active_items:tc.optional(list)=None,
                   active_items=None,
                   active_color=BOLD,
                   input_color='green',
                   output_color='red',
                   input_and_output_color='brown',
                   controller_color='blue',
                   learning_color='orange',
                   composition_color='pink',
                   output_fmt:tc.enum('pdf','gv','jupyter','gif')='pdf',
                   execution_id=NotImplemented,
                   **kwargs):
        """
        show_graph(                           \
           show_node_structure=False,         \
           show_nested=True,                  \
           show_controller=False,             \
           show_cim=False,                    \
           show_learning=False,               \
           show_headers=True,                 \
           show_dimensions=False,             \
           show_projection_labels=False,      \
           direction='BT',                    \
           active_items=None,                 \
           active_color=BOLD,                 \
           input_color='green',               \
           output_color='red',                \
           input_and_output_color='brown',    \
           controller_color='blue',           \
           composition_color='pink',          \
           output_fmt='pdf',                  \
           execution_id=None)

        Show graphical display of Components in a Composition's graph.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        See `Visualizing a Composition <Visualizing_a_Composition>` for details and examples.

        Arguments
        ---------

        show_node_structure : bool, VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS, STATE_FUNCTION_PARAMS, ROLES, \
        or ALL : default False
            show a detailed representation of each `Mechanism` in the graph, including its `States <State>`;  can
            have any of the following settings alone or in a list:

            * `True` -- show States of Mechanism, but not information about the `value
              <Component.value>` or `function <Component.function>` of the Mechanism or its States.

            * *VALUES* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <State_Base.value>` of each of its States.

            * *LABELS* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <State_Base.value>` of each of its States, using any labels for the values of InputStates and
              OutputStates specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
              `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

            * *FUNCTIONS* -- show the `function <Mechanism_Base.function>` of the Mechanism and the `function
              <State_Base.function>` of its InputStates and OutputStates.

            * *MECH_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
              Mechanism in the Composition (only applies if *FUNCTIONS* is True).

            * *STATE_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
              State of each Mechanism in the Composition (only applies if *FUNCTIONS* is True).

            * *ROLES* -- show the `role <Composition.NodeRoles>` of the Mechanism in the Composition
              (but not any of the other information;  use *ALL* to show ROLES with other information).

            * *ALL* -- shows the role, `function <Component.function>`, and `value <Component.value>` of the
              Mechanisms in the `Composition` and their `States <State>` (using labels for
              the values, if specified -- see above), including parameters for all functions.

        show_nested : bool | dict : default ALL
            specifies whether any nested Composition(s) are shown in details as inset graphs.  A dict can be used to
            specify any of the arguments allowed for show_graph to be used for the nested Composition(s);  *ALL*
            passes all arguments specified for the main Composition to the nested one(s);  True uses the default
            values of show_graph args for the nested Composition(s).

        show_controller :  bool : default False
            specifies whether or not to show the Composition's controller and associated ObjectiveMechanism;
            these are displayed in the color specified for **controller_color**.

        show_cim : bool : default False
            specifies whether or not to show the Composition's input and out CompositionInterfaceMechanisms (CIMs)

        show_learning : bool or ALL : default False
            specifies whether or not to show the learning components of the Compositoin;
            they will all be displayed in the color specified for **learning_color**.
            Projections that receive a `LearningProjection` will be shown as a diamond-shaped node.
            If set to *ALL*, all Projections associated with learning will be shown:  the LearningProjections
            as well as from `ProcessingMechanisms <ProcessingMechanism>` to `LearningMechanisms <LearningMechanism>`
            that convey error and activation information;  if set to `True`, only the LearningPojections are shown.

        show_projection_labels : bool : default False
            specifies whether or not to show names of projections.

        show_headers : bool : default True
            specifies whether or not to show headers in the subfields of a Mechanism's node;  only takes effect if
            **show_node_structure** is specified (see above).

        show_dimensions : bool : default False
            specifies whether or not to show dimensions for the `variable <Component.variable>` and `value
            <Component.value>` of each Component in the graph (and/or MappingProjections when show_learning
            is `True`);  can have the following settings:

            * *MECHANISMS* -- shows `Mechanism` input and output dimensions.  Input dimensions are shown in parentheses
              below the name of the Mechanism; each number represents the dimension of the `variable
              <InputState.variable>` for each `InputState` of the Mechanism; Output dimensions are shown above
              the name of the Mechanism; each number represents the dimension for `value <OutputState.value>` of each
              of `OutputState` of the Mechanism.

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

        input_color : keyword : default 'green',
            specifies the display color for `INPUT <NodeRole.INPUT>` Nodes in the Composition

        output_color : keyword : default 'red',
            specifies the display color for `OUTPUT` Nodes in the Composition

        input_and_output_color : keyword : default 'brown'
            specifies the display color of nodes that are both an `INPUT <NodeRole.INPUT>` and an `OUTPUT
            <NodeRole.OUTPUT>` Node in the Composition

        input_and_output_color : keyword : default 'brown'
            specifies the display color of nodes that represented nested Compositions.

        cim_shape : default 'square'
            specifies the display color input_CIM and output_CIM nodes

        controller_color : keyword : default `blue`
            specifies the color in which the controller components are displayed

        output_fmt : keyword : default 'pdf'
            'pdf': generate and open a pdf with the visualization;
            'jupyter': return the object (for working in jupyter/ipython notebooks);
            'gv': return graphviz object
            'gif': return gif used for animation

        Returns
        -------

        display of Composition : `pdf` or Graphviz graph object
            PDF: (placed in current directory) if :keyword:`output_fmt` arg is 'pdf';
            Graphviz graph object if :keyword:`output_fmt` arg is 'gv' or 'jupyter';
            gif if :keyword:`output_fmt` arg is 'gif'.

        """

        # HELPER METHODS ----------------------------------------------------------------------

        tc.typecheck
        _locals = locals().copy()

        def _assign_processing_components(g, rcvr, show_nested):
            """Assign nodes to graph"""
            if isinstance(rcvr, Composition) and show_nested:
                # User passed args for nested Composition
                output_fmt_arg = {'output_fmt':'gv'}
                if isinstance(show_nested, dict):
                    args = show_nested
                    args.update(output_fmt_arg)
                elif show_nested is ALL:
                    # Pass args from main call to show_graph to call for nested Composition
                    args = dict({k:_locals[k] for k in list(inspect.signature(self.show_graph).parameters)})
                    args.update(output_fmt_arg)
                    if kwargs:
                        args['kwargs'] = kwargs
                    else:
                        del  args['kwargs']
                else:
                    # Use default args for nested Composition
                    args = output_fmt_arg
                nested_comp_graph = rcvr.show_graph(**args)
                nested_comp_graph.name = "cluster_"+rcvr.name
                rcvr_label = rcvr.name
                if rcvr in self.get_nodes_by_role(NodeRole.INPUT) and \
                        rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
                    nested_comp_graph.attr(color=input_and_output_color)
                elif rcvr in self.get_nodes_by_role(NodeRole.INPUT):
                    nested_comp_graph.attr(color=input_color)
                elif rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
                    nested_comp_graph.attr(color=output_color)
                nested_comp_graph.attr(label=rcvr_label)
                g.subgraph(nested_comp_graph)

            # If rcvr is a learning component and not an INPUT node,
            #    break and handle in _assign_learning_components()
            #    (node: this allows TARGET node for learning to remain marked as an INPUT node)
            if NodeRole.LEARNING in self.nodes_to_roles[rcvr] and not NodeRole.INPUT in self.nodes_to_roles[rcvr]:
                return

            # If rcvr is ObjectiveMechanism for Composition's controller,
            #    break and handle in _assign_control_components()
            if (isinstance(rcvr, ObjectiveMechanism)
                    and self.controller
                    and rcvr is self.controller.objective_mechanism):
                return

            # Implement rcvr node
            else:

                # Set rcvr color and penwidth based on node type
                rcvr_rank = 'same'
                node_shape = mechanism_shape

                # Get condition if any associated with rcvr
                if rcvr in self.scheduler_processing.conditions:
                    condition = self.scheduler_processing.conditions[rcvr]
                else:
                    condition = None

                # Input and Output Node
                if rcvr in self.get_nodes_by_role(NodeRole.INPUT) and \
                        rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
                    if rcvr in active_items:
                        if active_color is BOLD:
                            rcvr_color = input_and_output_color
                        else:
                            rcvr_color = active_color
                        rcvr_penwidth = str(bold_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        rcvr_color = input_and_output_color
                        rcvr_penwidth = str(bold_width)

                # Input Node
                elif rcvr in self.get_nodes_by_role(NodeRole.INPUT):
                    if rcvr in active_items:
                        if active_color is BOLD:
                            rcvr_color = input_color
                        else:
                            rcvr_color = active_color
                        rcvr_penwidth = str(bold_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        rcvr_color = input_color
                        rcvr_penwidth = str(bold_width)
                    rcvr_rank = input_rank

                # Output Node
                elif rcvr in self.get_nodes_by_role(NodeRole.OUTPUT):
                    if rcvr in active_items:
                        if active_color is BOLD:
                            rcvr_color = output_color
                        else:
                            rcvr_color = active_color
                        rcvr_penwidth = str(bold_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        rcvr_color = output_color
                        rcvr_penwidth = str(bold_width)
                    rcvr_rank = output_rank

                # Composition
                elif isinstance(rcvr, Composition):
                    node_shape = composition_shape
                    if rcvr in active_items:
                        if active_color is BOLD:
                            rcvr_color = composition_color
                        else:
                            rcvr_color = active_color
                        rcvr_penwidth = str(bold_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        rcvr_color = composition_color
                        rcvr_penwidth = str(bold_width)

                elif rcvr in active_items:
                    if active_color is BOLD:
                        rcvr_color = default_node_color
                    else:
                        rcvr_color = active_color
                    rcvr_penwidth = str(default_width + active_thicker_by)
                    self.active_item_rendered = True

                else:
                    rcvr_color = default_node_color
                    rcvr_penwidth = str(default_width)

                # Implement rcvr node
                rcvr_label = self._get_graph_node_label(rcvr, show_dimensions)

                if show_node_structure and isinstance(rcvr, Mechanism):
                    g.node(rcvr_label,
                           rcvr.show_structure(**node_struct_args, node_border=rcvr_penwidth, condition=condition),
                           shape=struct_shape,
                           color=rcvr_color,
                           rank=rcvr_rank,
                           penwidth=rcvr_penwidth)
                else:
                    g.node(rcvr_label,
                           shape=node_shape,
                           color=rcvr_color,
                           rank=rcvr_rank,
                           penwidth=rcvr_penwidth)

                # MODIFIED 5/29/19 OLD: [JDC] SEEMS TO BE HANDLED BY ADDITION OF CALL TO _assign_incoming_edges BELOW
                # # handle auto-recurrent projections
                # for input_state in rcvr.input_states:
                #     for proj in input_state.path_afferents:
                #         if proj.sender.owner is not rcvr:
                #             continue
                #         if show_node_structure:
                #             sndr_proj_label = '{}:{}'.format(rcvr_label, rcvr._get_port_name(proj.sender))
                #             proc_mech_rcvr_label = '{}:{}'.format(rcvr_label, rcvr._get_port_name(proj.receiver))
                #         else:
                #             sndr_proj_label = proc_mech_rcvr_label = rcvr_label
                #         if show_projection_labels:
                #             edge_label = self._get_graph_node_label(proj, show_dimensions)
                #         else:
                #             edge_label = ''
                #
                #         # show projection as edge
                #         if proj.sender in active_items:
                #             if active_color is BOLD:
                #                 proj_color = default_node_color
                #             else:
                #                 proj_color = active_color
                #             proj_width = str(default_width + active_thicker_by)
                #             self.active_item_rendered = True
                #         else:
                #             proj_color = default_node_color
                #             proj_width = str(default_width)
                #         g.edge(sndr_proj_label, proc_mech_rcvr_label, label=edge_label,
                #                color=proj_color, penwidth=proj_width)
                # # MODIFIED 5/29/19 END

            # Implement sender edges
            sndrs = processing_graph[rcvr]
            _assign_incoming_edges(g, rcvr, rcvr_label, sndrs)

        def _assign_cim_components(g, cims):

            cim_rank = 'same'

            for cim in cims:

                cim_penwidth = str(default_width)

                # ASSIGN CIM NODE ****************************************************************

                # Assign color
                # Also take opportunity to verify that cim is either input_CIM or output_CIM
                if cim is self.input_CIM:
                    if cim in active_items:
                        if active_color is BOLD:
                            cim_color = input_color
                        else:
                            cim_color = active_color
                        cim_penwidth = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        cim_color = input_color

                elif cim is self.output_CIM:
                    if cim in active_items:
                        if active_color is BOLD:
                            cim_color = output_color
                        else:
                            cim_color = active_color
                        cim_penwidth = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        cim_color = output_color

                else:
                    assert False, '_assignm_cim_components called with node that is not input_CIM or output_CIM'

                # Assign lablel
                cim_label = self._get_graph_node_label(cim, show_dimensions)

                if show_node_structure:
                    g.node(cim_label,
                           cim.show_structure(**node_struct_args, node_border=cim_penwidth, compact_cim=True),
                           shape=struct_shape,
                           color=cim_color,
                           rank=cim_rank,
                           penwidth=cim_penwidth)

                else:
                    g.node(cim_label,
                           shape=cim_shape,
                           color=cim_color,
                           rank=cim_rank,
                           penwidth=cim_penwidth)

                # ASSIGN CIM PROJECTIONS ****************************************************************

                # Projections from input_CIM to INPUT nodes
                if cim is self.input_CIM:

                    for output_state in self.input_CIM.output_states:
                        projs = output_state.efferents
                        for proj in projs:
                            input_mech = proj.receiver.owner
                            if input_mech is self.controller:
                                # Projections to contoller are handled under _assign_controller_components
                                continue
                            # Validate the Projection is to an INPUT node or a node that is shadowing one
                            if ((input_mech in self.nodes_to_roles and
                                 not NodeRole.INPUT in self.nodes_to_roles[input_mech])
                                    and (proj.receiver.shadow_inputs in self.nodes_to_roles and
                                         not NodeRole.INPUT in self.nodes_to_roles[proj.receiver.shadow_inputs])):
                                raise CompositionError("Projection from input_CIM of {} to node {} "
                                                       "that is not an {} node or shadowing its {}".
                                                       format(self.name, input_mech,
                                                              NodeRole.INPUT.name, NodeRole.INPUT.name.lower()))
                            # Construct edge name
                            input_mech_label = self._get_graph_node_label(input_mech, show_dimensions)
                            if show_node_structure:
                                cim_proj_label = '{}:{}-{}'. \
                                    format(cim_label, OutputState.__name__, proj.sender.name)
                                proc_mech_rcvr_label = '{}:{}-{}'. \
                                    format(input_mech_label, InputState.__name__, proj.receiver.name)
                            else:
                                cim_proj_label = cim_label
                                proc_mech_rcvr_label = input_mech_label

                            # Render Projection
                            if any(item in active_items for item in {proj, proj.receiver.owner}):
                                if active_color is BOLD:
                                    proj_color = default_node_color
                                else:
                                    proj_color = active_color
                                proj_width = str(default_width + active_thicker_by)
                                self.active_item_rendered = True
                            else:
                                proj_color = default_node_color
                                proj_width = str(default_width)
                            if show_projection_labels:
                                label = self._get_graph_node_label(proj, show_dimensions)
                            else:
                                label = ''
                            g.edge(cim_proj_label, proc_mech_rcvr_label, label=label,
                                   color=proj_color, penwidth=proj_width)

                # Projections from OUTPUT nodes to output_CIM
                if cim is self.output_CIM:
                    # Construct edge name
                    for input_state in self.output_CIM.input_states:
                        projs = input_state.path_afferents
                        for proj in projs:
                            # Validate the Projection is from an OUTPUT node
                            output_mech = proj.sender.owner
                            if not NodeRole.OUTPUT in self.nodes_to_roles[output_mech]:
                                raise CompositionError("Projection to output_CIM of {} from node {} "
                                                       "that is not an {} node".
                                                       format(self.name, output_mech,
                                                              NodeRole.OUTPUT.name, NodeRole.OUTPUT.name.lower()))
                            # Construct edge name
                            output_mech_label = self._get_graph_node_label(output_mech, show_dimensions)
                            if show_node_structure:
                                cim_proj_label = '{}:{}'. \
                                    format(cim_label, cim._get_port_name(proj.receiver))
                                proc_mech_sndr_label = '{}:{}'.\
                                    format(output_mech_label, output_mech._get_port_name(proj.sender))
                                    # format(output_mech_label, OutputState.__name__, proj.sender.name)
                            else:
                                cim_proj_label = cim_label
                                proc_mech_sndr_label = output_mech_label

                            # Render Projection
                            if any(item in active_items for item in {proj, proj.receiver.owner}):
                                if active_color is BOLD:
                                    proj_color = default_node_color
                                else:
                                    proj_color = active_color
                                proj_width = str(default_width + active_thicker_by)
                                self.active_item_rendered = True
                            else:
                                proj_color = default_node_color
                                proj_width = str(default_width)
                            if show_projection_labels:
                                label = self._get_graph_node_label(proj, show_dimensions)
                            else:
                                label = ''
                            g.edge(proc_mech_sndr_label, cim_proj_label, label=label,
                                   color=proj_color, penwidth=proj_width)

        def _assign_controller_components(g):
            """Assign control nodes and edges to graph """

            controller = self.controller
            if controller is None:
                warnings.warn(f"{self.name} has not been assigned a \'controller\', "
                              f"so \'show_controller\' option in call to its show_graph() method will be ignored.")
                return

            if controller in active_items:
                if active_color is BOLD:
                    ctlr_color = controller_color
                else:
                    ctlr_color = active_color
                ctlr_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                ctlr_color = controller_color
                ctlr_width = str(default_width)

            # Assign controller node
            node_shape = mechanism_shape
            ctlr_label = self._get_graph_node_label(controller, show_dimensions)
            if show_node_structure:
                g.node(ctlr_label,
                       controller.show_structure(**node_struct_args, node_border=ctlr_width,
                                                 condition=self.controller_condition),
                       shape=struct_shape,
                       color=ctlr_color,
                       penwidth=ctlr_width,
                       rank=control_rank
                       )
            else:
                g.node(ctlr_label,
                        color=ctlr_color, penwidth=ctlr_width, shape=node_shape,
                        rank=control_rank)

            # outgoing edges (from controller to ProcessingMechanisms)
            for control_signal in controller.control_signals:
                for ctl_proj in control_signal.efferents:
                    proc_mech_label = self._get_graph_node_label(ctl_proj.receiver.owner, show_dimensions)
                    if controller in active_items:
                        if active_color is BOLD:
                            ctl_proj_color = controller_color
                        else:
                            ctl_proj_color = active_color
                        ctl_proj_width = str(default_width + active_thicker_by)
                        self.active_item_rendered = True
                    else:
                        ctl_proj_color = controller_color
                        ctl_proj_width = str(default_width)
                    if show_projection_labels:
                        edge_label = ctl_proj.name
                    else:
                        edge_label = ''
                    if show_node_structure:
                        ctl_sndr_label = ctlr_label + ':' + controller._get_port_name(control_signal)
                        proc_mech_rcvr_label = \
                            proc_mech_label + ':' + controller._get_port_name(ctl_proj.receiver)
                    else:
                        ctl_sndr_label = ctlr_label
                        proc_mech_rcvr_label = proc_mech_label
                    g.edge(ctl_sndr_label,
                           proc_mech_rcvr_label,
                           label=edge_label,
                           color=ctl_proj_color,
                           penwidth=ctl_proj_width
                           )

            # If controller has objective_mechanism, assign its node and Projections
            if controller.objective_mechanism:
                # get projection from ObjectiveMechanism to ControlMechanism
                objmech_ctlr_proj = controller.input_state.path_afferents[0]
                if controller in active_items:
                    if active_color is BOLD:
                        objmech_ctlr_proj_color = controller_color
                    else:
                        objmech_ctlr_proj_color = active_color
                    objmech_ctlr_proj_width = str(default_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    objmech_ctlr_proj_color = controller_color
                    objmech_ctlr_proj_width = str(default_width)

                # get ObjectiveMechanism
                objmech = objmech_ctlr_proj.sender.owner
                if objmech in active_items:
                    if active_color is BOLD:
                        objmech_color = controller_color
                    else:
                        objmech_color = active_color
                    objmech_width = str(default_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    objmech_color = controller_color
                    objmech_width = str(default_width)

                objmech_label = self._get_graph_node_label(objmech, show_dimensions)
                if show_node_structure:
                    if objmech in self.scheduler_processing.conditions:
                        condition = self.scheduler_processing.conditions[obj_mech]
                    else:
                        condition = None
                    g.node(objmech_label,
                           objmech.show_structure(**node_struct_args, node_border=ctlr_width, condition=condition),
                           shape=struct_shape,
                           color=objmech_color,
                           penwidth=ctlr_width,
                           rank=control_rank
                           )
                else:
                    g.node(objmech_label,
                            color=objmech_color, penwidth=objmech_width, shape=node_shape,
                            rank=control_rank)

                # objmech to controller edge
                if show_projection_labels:
                    edge_label = objmech_ctlr_proj.name
                else:
                    edge_label = ''
                if show_node_structure:
                    obj_to_ctrl_label = objmech_label + ':' + objmech._get_port_name(objmech_ctlr_proj.sender)
                    ctlr_from_obj_label = ctlr_label + ':' + objmech._get_port_name(objmech_ctlr_proj.receiver)
                else:
                    obj_to_ctrl_label = objmech_label
                    ctlr_from_obj_label = ctlr_label
                g.edge(obj_to_ctrl_label, ctlr_from_obj_label, label=edge_label,
                       color=objmech_ctlr_proj_color, penwidth=objmech_ctlr_proj_width)

                # incoming edges (from monitored mechs to objective mechanism)
                for input_state in objmech.input_states:
                    for projection in input_state.path_afferents:
                        if objmech in active_items:
                            if active_color is BOLD:
                                proj_color = controller_color
                            else:
                                proj_color = active_color
                            proj_width = str(default_width + active_thicker_by)
                            self.active_item_rendered = True
                        else:
                            proj_color = controller_color
                            proj_width = str(default_width)
                        if show_node_structure:
                            sndr_proj_label = self._get_graph_node_label(projection.sender.owner, show_dimensions) + \
                                              ':' + objmech._get_port_name(projection.sender)
                            objmech_proj_label = objmech_label + ':' + objmech._get_port_name(input_state)
                        else:
                            sndr_proj_label = self._get_graph_node_label(projection.sender.owner, show_dimensions)
                            objmech_proj_label = self._get_graph_node_label(objmech, show_dimensions)
                        if show_projection_labels:
                            edge_label = projection.name
                        else:
                            edge_label = ''
                        g.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                               color=proj_color, penwidth=proj_width)

            # If controller has an agent_rep, assign its node and edges (not Projections per se)
            if hasattr(controller, 'agent_rep') and controller.agent_rep:
                # get agent_rep
                agent_rep = controller.agent_rep
                # controller is active, treat
                if controller in active_items:
                    if active_color is BOLD:
                        agent_rep_color = controller_color
                    else:
                        agent_rep_color = active_color
                    agent_rep_width = str(default_width + active_thicker_by)
                    self.active_item_rendered = True
                else:
                    agent_rep_color = controller_color
                    agent_rep_width = str(default_width)

                # agent_rep node
                agent_rep_label = self._get_graph_node_label(agent_rep, show_dimensions)
                g.node(agent_rep_label,
                        color=agent_rep_color, penwidth=agent_rep_width, shape=agent_rep_shape,
                        rank=control_rank)

                # agent_rep <-> controller edges
                g.edge(agent_rep_label, ctlr_label, color=agent_rep_color, penwidth=agent_rep_width)
                g.edge(ctlr_label, agent_rep_label, color=agent_rep_color, penwidth=agent_rep_width)

            # get any other incoming edges to controller (i.e., other than from ObjectiveMechanism)
            senders = set()
            for i in controller.input_states[1:]:
                for p in i.path_afferents:
                    senders.add(p.sender.owner)
            _assign_incoming_edges(g, controller, ctlr_label, senders, proj_color=ctl_proj_color)

        def _assign_learning_components(g):
            """Assign learning nodes and edges to graph"""

            # Get learning_components, with exception of INPUT (i.e. TARGET) nodes
            #    (i.e., allow TARGET node to continue to be marked as an INPUT node)
            learning_components = [node for node in self.learning_components
                                   if not NodeRole.INPUT in self.nodes_to_roles[node]]

            for rcvr in learning_components:
                # if rcvr is Projection, skip (handled in _assign_processing_components)
                if isinstance(rcvr, MappingProjection):
                    return

                # Get rcvr info
                rcvr_label = self._get_graph_node_label(rcvr, show_dimensions)
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
                # Implement node for Mechanism
                if show_node_structure:
                    g.node(rcvr_label,
                            rcvr.show_structure(**node_struct_args),
                            rank=learning_rank, color=rcvr_color, penwidth=rcvr_width)
                else:
                    g.node(rcvr_label,
                            color=rcvr_color, penwidth=rcvr_width,
                            rank=learning_rank, shape=mechanism_shape)

                # Implement sender edges
                sndrs = processing_graph[rcvr]
                _assign_incoming_edges(g, rcvr, rcvr_label, sndrs)

        def render_projection_as_node(g, proj, label,
                                      proj_color, proj_width,
                                      sndr_label=None,
                                      rcvr_label=None):
            from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism

            proj_receiver = proj.receiver.owner

            # Node for Projection
            g.node(label, shape=learning_projection_shape, color=proj_color, penwidth=proj_width)

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
            # if proj in active_items or (proj_learning_in_execution_phase and proj_receiver in active_items):
            if proj in active_items:
                if active_color is BOLD:
                    learning_proj_color = learning_color
                else:
                    learning_proj_color = active_color
                learning_proj_width = str(default_width + active_thicker_by)
                self.active_item_rendered = True
            else:
                learning_proj_color = learning_color
                learning_proj_width = str(default_width)
            sndrs = proj._parameter_states['matrix'].mod_afferents # GET ALL LearningProjections to proj
            for sndr in sndrs:
                sndr_label = self._get_graph_node_label(sndr.sender.owner, show_dimensions)
                rcvr_label = self._get_graph_node_label(proj, show_dimensions)
                if show_projection_labels:
                    edge_label = proj._parameter_states['matrix'].mod_afferents[0].name
                else:
                    edge_label = ''
                if show_node_structure:
                    G.edge(sndr_label + ':' + OutputState.__name__ + '-' + 'LearningSignal',
                           rcvr_label,
                           label=edge_label,
                           color=learning_proj_color, penwidth=learning_proj_width)
                else:
                    G.edge(sndr_label, rcvr_label, label = edge_label,
                           color=learning_proj_color, penwidth=learning_proj_width)
            return True

        @tc.typecheck
        def _assign_incoming_edges(g, rcvr, rcvr_label, senders, proj_color=None):
            proj_color = proj_color or default_node_color
            for sndr in senders:

                # Set sndr info

                sndr_label = self._get_graph_node_label(sndr, show_dimensions)

                # Iterate through all Projections from all OutputStates of sndr
                for output_state in sndr.output_states:
                    for proj in output_state.efferents:

                        # Skip any projections to ObjectiveMechanism for controller
                        #   (those are handled in _assign_control_components)
                        if (self.controller
                                and proj.receiver.owner is self.controller.objective_mechanism):
                            continue

                        # Only consider Projections to the rcvr
                        if ((isinstance(rcvr, (Mechanism, Projection)) and proj.receiver.owner == rcvr)
                                or (isinstance(rcvr, Composition) and proj.receiver.owner is rcvr.input_CIM)):

                            if show_node_structure and isinstance(sndr, Mechanism) and isinstance(rcvr, Mechanism):
                                sndr_proj_label = f'{sndr_label}:{sndr._get_port_name(proj.sender)}'
                                proc_mech_rcvr_label = f'{rcvr_label}:{rcvr._get_port_name(proj.receiver)}'
                            else:
                                sndr_proj_label = sndr_label
                                proc_mech_rcvr_label = rcvr_label
                            try:
                                has_learning = proj.has_learning_projection is not None
                            except AttributeError:
                                has_learning = None

                            edge_label = self._get_graph_node_label(proj, show_dimensions)

                            # Check if Projection or its receiver is active
                            if any(item in active_items for item in {proj, proj.receiver.owner}):
                                if active_color is BOLD:
                                    if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                                        proj_color = learning_color
                                    else:
                                        pass
                                else:
                                    proj_color = active_color
                                proj_width = str(default_width + active_thicker_by)
                                self.active_item_rendered = True

                            # Projection to or from a LearningMechanism
                            elif (NodeRole.LEARNING in self.nodes_to_roles[rcvr]):
                                proj_color = learning_color
                                proj_width = str(default_width)

                            else:
                                proj_width = str(default_width)
                            proc_mech_label = edge_label

                            # Render Projection as edge

                            if show_learning and has_learning:
                                # Render Projection as node
                                #    (do it here rather than in _assign_learning_components,
                                #     as it needs afferent and efferent edges to other nodes)
                                # IMPLEMENTATION NOTE: Projections can't yet use structured nodes:
                                deferred = not render_projection_as_node(g=g, proj=proj,
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
                                if show_projection_labels:
                                    label = proc_mech_label
                                else:
                                    label = ''
                                g.edge(sndr_proj_label, proc_mech_rcvr_label, label=label,
                                       color=proj_color, penwidth=proj_width)

        # SETUP AND CONSTANTS -----------------------------------------------------------------

        INITIAL_FRAME = "INITIAL_FRAME"

        if execution_id is NotImplemented:
            execution_id = self.default_execution_id

        # For backward compatibility
        if 'show_model_based_optimizer' in kwargs:
            show_controller = kwargs['show_model_based_optimizer']
            del kwargs['show_model_based_optimizer']
        if kwargs:
            raise CompositionError(f'Unrecognized argument(s) in call to show_graph method '
                                   f'of {Composition.__name__} {repr(self.name)}: {", ".join(kwargs.keys())}')

        if show_dimensions == True:
            show_dimensions = ALL

        active_items = active_items or []
        if active_items:
            active_items = convert_to_list(active_items)
            if (self.scheduler_processing.get_clock(execution_id).time.run >= self._animate_num_runs or
                    self.scheduler_processing.get_clock(execution_id).time.trial >= self._animate_num_trials):
                return

            for item in active_items:
                if not isinstance(item, Component) and item is not INITIAL_FRAME:
                    raise CompositionError(
                        "PROGRAM ERROR: Item ({}) specified in {} argument for {} method of {} is not a {}".
                        format(item, repr('active_items'), repr('show_graph'), self.name, Component.__name__))

        self.active_item_rendered = False

        # Argument values used to call Mechanism.show_structure()
        if isinstance(show_node_structure, (list, tuple, set)):
            node_struct_args = {'composition': self,
                                'show_roles': any(key in show_node_structure for key in {ROLES, ALL}),
                                'show_conditions': any(key in show_node_structure for key in {CONDITIONS, ALL}),
                                'show_functions': any(key in show_node_structure for key in {FUNCTIONS, ALL}),
                                'show_mech_function_params': any(key in show_node_structure
                                                                 for key in {MECH_FUNCTION_PARAMS, ALL}),
                                'show_state_function_params': any(key in show_node_structure
                                                                  for key in {STATE_FUNCTION_PARAMS, ALL}),
                                'show_values': any(key in show_node_structure for key in {VALUES, ALL}),
                                'use_labels': any(key in show_node_structure for key in {LABELS, ALL}),
                                'show_headers': show_headers,
                                'output_fmt': 'struct'}
        else:
            node_struct_args = {'composition': self,
                                'show_roles': show_node_structure in {ROLES, ALL},
                                'show_conditions': show_node_structure in {CONDITIONS, ALL},
                                'show_functions': show_node_structure in {FUNCTIONS, ALL},
                                'show_mech_function_params': show_node_structure in {MECH_FUNCTION_PARAMS, ALL},
                                'show_state_function_params': show_node_structure in {STATE_FUNCTION_PARAMS, ALL},
                                'show_values': show_node_structure in {VALUES, LABELS, ALL},
                                'use_labels': show_node_structure in {LABELS, ALL},
                                'show_headers': show_headers,
                                'output_fmt': 'struct'}

        default_node_color = 'black'
        mechanism_shape = 'oval'
        learning_projection_shape = 'diamond'
        struct_shape = 'plaintext' # assumes use of html
        cim_shape = 'rectangle'
        composition_shape = 'rectangle'
        agent_rep_shape = 'egg'

        bold_width = 3
        default_width = 1
        active_thicker_by = 2

        input_rank = 'source'
        control_rank = 'min'
        learning_rank = 'min'
        output_rank = 'max'

        # BUILD GRAPH ------------------------------------------------------------------------

        import graphviz as gv

        G = gv.Digraph(
            name=self.name,
            engine="dot",
            node_attr={
                'fontsize': '12',
                'fontname': 'arial',
                'shape': 'record',
                'color': default_node_color,
                'penwidth': str(default_width)
            },
            edge_attr={
                'fontsize': '10',
                'fontname': 'arial'
            },
            graph_attr={
                "rankdir": direction,
                'overlap': "False"
            },
        )

        # get all Nodes
        # FIX: call to _analyze_graph in nested calls to show_graph cause trouble
        if output_fmt != 'gv':
            self._analyze_graph()
        processing_graph = self.scheduler_processing.visual_graph
        rcvrs = list(processing_graph.keys())

        for r in rcvrs:
            _assign_processing_components(G, r, show_nested)

        # Add cim Components to graph if show_cim
        if show_cim:
            _assign_cim_components(G, [self.input_CIM, self.output_CIM])

        # Add controller-related Components to graph if show_controller
        if show_controller:
            _assign_controller_components(G)

        # Add learning-related Components to graph if show_learning
        if show_learning:
            _assign_learning_components(G)

        # Sort to put ORIGIN nodes first and controller and its objective_mechanism last
        def get_index_of_node_in_G_body(node):
            """Get index of node in G.body"""
            for i, item in enumerate(G.body):
                # Skip projections
                if node.name in item and not '->' in item:
                    return i
        for node in self.nodes:
            roles = self.get_roles_by_node(node)
            if NodeRole.INPUT in roles:
                i = get_index_of_node_in_G_body(node)
                if i is not None:
                    G.body.insert(0,G.body.pop(i))
        if self.controller and show_controller:
            i = get_index_of_node_in_G_body(self.controller)
            G.body.insert(len(G.body),G.body.pop(i))

        # GENERATE OUTPUT ---------------------------------------------------------------------

        # Show as pdf
        if output_fmt == 'pdf':
            # G.format = 'svg'
            G.view(self.name.replace(" ", "-"), cleanup=True, directory='show_graph OUTPUT/PDFS')

        # Generate images for animation
        elif output_fmt == 'gif':
            if self.active_item_rendered or INITIAL_FRAME in active_items:
                self._generate_gifs(G, active_items, execution_id)

        # Return graph to show in jupyter
        elif output_fmt == 'jupyter':
            return G

        elif output_fmt == 'gv':
            return G

    def execute(
            self,
            inputs=None,
            autodiff_stimuli=None,
            scheduler_processing=None,
            termination_processing=None,
            call_before_time_step=None,
            call_before_pass=None,
            call_after_time_step=None,
            call_after_pass=None,
            execution_id=None,
            base_execution_id=None,
            clamp_input=SOFT_CLAMP,
            runtime_params=None,
            skip_initialization=False,
            bin_execute=False,
            context=None):
        """
            Passes inputs to any Nodes receiving inputs directly from the user (via the "inputs" argument) then
            coordinates with the Scheduler to execute sets of nodes that are eligible to execute until
            termination conditions are met.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` or `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each node in the composition that receives inputs from
                the user. For each pair, the key is the node (Mechanism or Composition) and the value is an input,
                the shape of which must match the node's default variable.

            scheduler_processing : Scheduler
                the scheduler object that owns the conditions that will instruct the execution of this Composition
                If not specified, the Composition will use its automatically generated scheduler.

            execution_id
                execution_id will be set to self.default_execution_id if unspecified

            base_execution_id
                the execution_id corresponding to the execution context from which this execution will be initialized,
                if values currently do not exist for **execution_id**

            call_before_time_step : callable
                called before each `TIME_STEP` is executed
                passed the current *execution_id* (but it is not necessary for your callable to take)

            call_after_time_step : callable
                called after each `TIME_STEP` is executed
                passed the current *execution_id* (but it is not necessary for your callable to take)

            call_before_pass : callable
                called before each `PASS` is executed
                passed the current *execution_id* (but it is not necessary for your callable to take)

            call_after_pass : callable
                called after each `PASS` is executed
                passed the current *execution_id* (but it is not necessary for your callable to take)

            Returns
            ---------

            output value of the final Mechanism executed in the Composition : various
        """

        # ASSIGNMENTS **************************************************************************************************

        if bin_execute == 'Python':
            bin_execute = False

        if not hasattr(self, '_animate'):
            # These are meant to be assigned in run method;  needed here for direct call to execute method
            self._animate = False

        # KAM Note 4/29/19
        # The nested var is set to True if this Composition is nested in another Composition, otherwise False
        # Later on, this is used to determine:
        #   (1) whether to initialize from context
        #   (2) whether to assign values to CIM from input dict (if not nested) or simply execute CIM (if nested)
        nested = False
        if len(self.input_CIM.path_afferents) > 0:
            nested = True

        runtime_params = self._parse_runtime_params(runtime_params)

        # Assign the same execution_ids to all nodes in the Composition and get it (if it was None)
        execution_id = self._assign_execution_ids(execution_id)

        if not skip_initialization:
            self._assign_context_values(execution_id=execution_id, composition=self, propagate=True)

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        execution_scheduler = scheduler_processing or self.scheduler_processing

        execution_context = self.parameters.context._get(execution_id)

        if termination_processing is None:
            termination_processing = self.termination_processing

        # Skip initialization if possible (for efficiency):
        # - and(execution_id and context have not changed
        # -     structure of the graph has not changed
        # -     not a nested composition
        # -     its not a simulation)
        # - or(gym forage env is being used)
        # (e.g., when run is called externally repeated for the same environment)
        # KAM added HACK below "or self.env is None" in order to merge in interactive inputs fix for speed improvement
        # TBI: Clean way to call _initialize_from_context if execution_id has not changed, BUT composition has changed
        # for example:
        # comp.run()
        # comp.add_node(new_node)
        # comp.run().
        # execution_id has not changed on the comp, BUT new_node's execution id needs to be set from None --> ID
        if self.most_recent_execution_id != execution_id or self.env is None:
            # initialize from base context but don't overwrite any values already set for this execution_id
            if (
                not skip_initialization
                and not nested
                or execution_context is None
                and execution_context.execution_phase is not ContextFlags.SIMULATION
            ):
                self._initialize_from_context(execution_id, base_execution_id, override=False)
                self._assign_context_values(execution_id, composition=self)

        # Generate first frame of animation without any active_items
        if self._animate is not False:
            # If execution_id fails, the scheduler has no data for it yet.
            # It also may be the first, so fall back to default execution_id
            try:
                self._animate_execution(INITIAL_FRAME, execution_id)
            except KeyError:
                self._animate_execution(INITIAL_FRAME, self.default_execution_id)

        # EXECUTE INPUT CIM ********************************************************************************************

        # FIX: 6/12/19 MOVE TO EXECUTE BELOW? (i.e., with bin_execute / _comp_ex.execute_node(self.input_CIM, inputs))
        # Execute input_CIMs
        if nested:
            self.input_CIM.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING
            self.input_CIM.execute(execution_id=execution_id, context=ContextFlags.PROCESSING)
        else:
            inputs = self._adjust_execution_stimuli(inputs)
            self._assign_values_to_input_CIM(inputs, execution_id=execution_id)

        # FIX: 6/12/19 Deprecate?
        # Manage input clamping
        next_pass_before = 1
        next_pass_after = 1
        if clamp_input:
            soft_clamp_inputs = self._identify_clamp_inputs(SOFT_CLAMP, clamp_input, input_nodes)
            hard_clamp_inputs = self._identify_clamp_inputs(HARD_CLAMP, clamp_input, input_nodes)
            pulse_clamp_inputs = self._identify_clamp_inputs(PULSE_CLAMP, clamp_input, input_nodes)
            no_clamp_inputs = self._identify_clamp_inputs(NO_CLAMP, clamp_input, input_nodes)

        # Animate input_CIM
        # FIX: COORDINATE WITH REFACTORING OF PROCESSING/CONTROL CONTEXT
        #      (NOT SURE WHETHER IT CAN BE LEFT IN PROCESSING AFTER THAT)
        execution_phase_buffer = self.parameters.context.get(execution_id).execution_phase
        self.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING
        if self._animate is not False and SHOW_CIM in self._animate and self._animate[SHOW_CIM]:
            self._animate_execution(self.input_CIM, execution_id)
        self.parameters.context.get(execution_id).execution_phase = execution_phase_buffer
        # FIX: END

        # EXECUTE CONTROLLER (if specified for BEFORE) *****************************************************************

        # Compile controller execution (if compilation is specified) --------------------------------

        if bin_execute:
            is_simulation = (execution_context is not None and
                             execution_context.execution_phase == ContextFlags.SIMULATION)
            # Try running in Exec mode first
            if (bin_execute is True or str(bin_execute).endswith('Exec')):
                # There's no mode to execute simulations.
                # Simulations are run as part of the controller node wrapper.
                assert not is_simulation
                try:
                    if bin_execute is True or bin_execute.startswith('LLVM'):
                        _comp_ex = pnlvm.CompExecution(self, [execution_id])
                        _comp_ex.execute(inputs)
                        return _comp_ex.extract_node_output(self.output_CIM)
                    elif bin_execute.startswith('PTX'):
                        self.__ptx_initialize(execution_id)
                        __execution = self._compilation_data.ptx_execution._get(execution_id)
                        __execution.cuda_execute(inputs)
                        return __execution.extract_node_output(self.output_CIM)
                except Exception as e:
                    if bin_execute is not True:
                        raise e

                    string = "Failed to execute `{}': {}".format(self.name, str(e))
                    print("WARNING: {}".format(string))

            # Exec failed for some reason, we can still try node level bin_execute
            try:
                # Filter out mechanisms. Nested compositions are not executed in this mode
                # Filter out controller. Compilation of controllers is not supported yet
                mechanisms = [n for n in self._all_nodes
                              if isinstance(n, Mechanism) and (n is not self.controller or not is_simulation)]
                # Generate all mechanism wrappers
                for m in mechanisms:
                    self._get_node_wrapper(m)

                _comp_ex = pnlvm.CompExecution(self, [execution_id])
                # Compile all mechanism wrappers
                for m in mechanisms:
                    _comp_ex._set_bin_node(m)

                bin_execute = True
            except Exception as e:
                if bin_execute is not True:
                    raise e

                string = "Failed to compile wrapper for `{}' in `{}': {}".format(m.name, self.name, str(e))
                print("WARNING: {}".format(string))
                bin_execute = False

        # Execute controller --------------------------------------------------------

        if (self.enable_controller and
            self.controller_mode is BEFORE and
            self.controller_condition.is_satisfied(scheduler=execution_scheduler,
                                                   execution_context=execution_id)):

            # control phase
            # FIX: SHOULD SET CONTEXT AS CONTROL HERE AND RESET AT END (AS DONE FOR animation BELOW)
            execution_phase = self.parameters.context._get(execution_id).execution_phase
            if (
                    execution_phase != ContextFlags.INITIALIZING
                    and execution_phase != ContextFlags.SIMULATION
            ):
                if self.controller and not bin_execute:
                    # FIX: REMOVE ONCE context IS SET TO CONTROL ABOVE
                    self.controller.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING
                    # FIX: END REMOVE
                    self.controller.execute(execution_id=execution_id, context=context)

                if bin_execute:
                    _comp_ex.execute_node(self.controller)

                # MODIFIED 6/13/19 NEW: [JDC]
                # FIX: REMOVE ONCE context IS SET TO CONTROL ABOVE
                if execution_context:
                    entry_execution_phase = execution_context.execution_phase
                    execution_context.execution_phase = ContextFlags.CONTROL
                # MODIFIED 6/13/19 END

                # Animate controller (before execution)
                if self._animate != False and SHOW_CONTROLLER in self._animate and self._animate[SHOW_CONTROLLER]:
                    self._animate_execution(self.controller, execution_id)

                # MODIFIED 6/13/19 NEW: [JDC]
                # FIX: REMOVE ONCE context IS SET TO CONTROL ABOVE
                if execution_context:
                    execution_context.execution_phase = entry_execution_phase
                # MODIFIED 6/13/19 END

        # EXECUTE (each execution_set) *********************************************************************************

        # PREPROCESS (get inputs, call_before_pass, animate first frame) ----------------------------------

        if execution_context:
            # FIX: REPLACE WITH STACK ON CONTEXT ALONG WITH INSTANCES OF execution_phase_buffer
            entry_execution_phase = execution_context.execution_phase
            self.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING

        if bin_execute:
            _comp_ex.execute_node(self.input_CIM, inputs)
        #              WHY DO BOTH?  WHY NOT if-else?

        if call_before_pass:
            call_with_pruned_args(call_before_pass, execution_context=execution_id)


        # GET execution_set -------------------------------------------------------------------------
        # run scheduler to receive sets of nodes that may be executed at this time step in any order
        for next_execution_set in execution_scheduler.run(termination_conds=termination_processing,
                                                          execution_id=execution_id,
                                                          skip_trial_time_increment=True,
                                                          ):

            # SETUP EXECUTION ----------------------------------------------------------------------------

            # FIX: 6/12/19 WHY IS call_*after*_pass BEING CALLED BEFORE THE PASS?
            if call_after_pass:
                if next_pass_after == \
                        execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS,
                                                                                          TimeScale.TRIAL):
                    logger.debug('next_pass_after {0}\tscheduler pass {1}'.
                                 format(next_pass_after,
                                        execution_scheduler.clocks[
                                            execution_id].get_total_times_relative(
                                                TimeScale.PASS, TimeScale.TRIAL)))
                    call_with_pruned_args(call_after_pass, execution_context=execution_id)
                    next_pass_after += 1

            if call_before_pass:
                if next_pass_before == \
                        execution_scheduler.clocks[execution_id].get_total_times_relative(TimeScale.PASS,
                                                                                          TimeScale.TRIAL):
                    call_with_pruned_args(call_before_pass, execution_context=execution_id)
                    logger.debug('next_pass_before {0}\tscheduler pass {1}'.
                                 format(next_pass_before,
                                        execution_scheduler.clocks[
                                            execution_id].get_total_times_relative(
                                                TimeScale.PASS,
                                                TimeScale.TRIAL)))
                    next_pass_before += 1

            if call_before_time_step:
                call_with_pruned_args(call_before_time_step, execution_context=execution_id)

            # MANAGE EXECUTION OF FEEDBACK / CYCLIC GRAPHS ------------------------------------------------
            # Set up storage of all node values *before* the start of each timestep
            # If nodes within a timestep are connected by projections, those projections must pass their senders'
            # values from the beginning of the timestep (i.e. their "frozen values")
            # This ensures that the order in which nodes execute does not affect the results of this timestep
            frozen_values = {}
            new_values = {}
            if bin_execute:
                _comp_ex.freeze_values()

            # PURGE LEARNING IF NOT ENABLED ----------------------------------------------------------------

            # If learning is turned off, check for any learning related nodes and remove them from the execution set
            if not self.enable_learning:
                next_execution_set = next_execution_set - set(self.get_nodes_by_role(NodeRole.LEARNING))

            # ANIMATE execution_set ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self._animate is not False and self._animate_unit is EXECUTION_SET:
                self._animate_execution(next_execution_set, execution_id)

            # EXECUTE (each node) --------------------------------------------------------------------------

            # execute each node with EXECUTING in context
            for node in next_execution_set:

                # Store values of all nodes in this execution_set for use by other nodes in the execution set
                #    throughout this timestep (e.g., for recurrent Projections)
                frozen_values[node] = node.get_output_values(execution_id)

                # FIX: 6/12/19 Deprecate?
                # Handle input clamping
                if node in input_nodes:
                    if clamp_input:
                        if node in hard_clamp_inputs:
                            # clamp = HARD_CLAMP --> "turn off" recurrent projection
                            if hasattr(node, "recurrent_projection"):
                                node.recurrent_projection.sender.parameters.value._set([0.0], execution_id,
                                                                                      override=True)
                        elif node in no_clamp_inputs:
                            for input_state in node.input_states:
                                self.input_CIM_states[input_state][1].parameters.value._set(0.0, execution_id,
                                                                                           override=True)

                # EXECUTE A MECHANISM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                if isinstance(node, Mechanism):

                    execution_runtime_params = {}

                    if node in runtime_params:
                        for param in runtime_params[node]:
                            if runtime_params[node][param][1].is_satisfied(scheduler=execution_scheduler,
                                               # KAM 5/15/18 - not sure if this will always be the correct execution id:
                                                                           execution_context=execution_id):
                                execution_runtime_params[param] = runtime_params[node][param][0]

                    # Set context.execution_phase

                    # Set to PROCESSING by default
                    node.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING

                    # Set to LEARNING if Mechanism receives any PathwayProjections that are being learned
                    #    for which learning_enabled == True or ONLINE (i.e., not False or AFTER)
                    if self.enable_learning:
                        projections = set(self.projections).intersection(set(node.path_afferents))
                        if any([p for p in projections if
                                any([a for a in p.parameter_states[MATRIX].mod_afferents
                                     if (hasattr(a, 'learning_enabled') and a.learning_enabled in {True, ONLINE})])]):
                            node.parameters.context._get(execution_id).execution_phase = ContextFlags.LEARNING

                    # Execute node
                    if bin_execute:
                        _comp_ex.execute_node(node)
                    else:
                        if node is not self.controller:
                            node.execute(execution_id=execution_id,
                                         runtime_params=execution_runtime_params,
                                         context=ContextFlags.COMPOSITION)

                    # Reset runtime_params for node and its function if specified
                        if execution_id in node._runtime_params_reset:
                            for key in node._runtime_params_reset[execution_id]:
                                node._set_parameter_value(key, node._runtime_params_reset[execution_id][key],
                                                          execution_id)
                        node._runtime_params_reset[execution_id] = {}

                        if execution_id in node.function._runtime_params_reset:
                            for key in node.function._runtime_params_reset[execution_id]:
                                node.function._set_parameter_value(
                                        key,
                                        node.function._runtime_params_reset[execution_id][key],
                                        execution_id)

                        node.function._runtime_params_reset[execution_id] = {}

                    # TEST PRINT 7/22/19
                    print(f'Executed {node.name}: \n\tvariable: {node.parameters.variable.get(execution_id)}'
                          f'\n\tvalue: {node.parameters.value.get(execution_id)}')

                    # Set execution_phase for node's context back to IDLE
                    node.parameters.context._get(execution_id).execution_phase = ContextFlags.IDLE

                # EXECUTE A NESTED COMPOSITION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                elif isinstance(node, Composition):

                    # Set up compilation
                    if bin_execute:
                        # Values of node with compiled wrappers are in binary data structure
                        srcs = (proj.sender.owner for proj in node.input_CIM.afferents if
                                proj.sender.owner in self.__generated_node_wrappers)
                        for srnode in srcs:
                            assert srnode in self.nodes or srnode is self.input_CIM
                            data = _comp_ex.extract_frozen_node_output(srnode)
                            for i, v in enumerate(data):
                                # This sets frozen values
                                srnode.output_states[i].parameters.value._set(v, execution_id, skip_history=True,
                                                                             skip_log=True, override=True)

                    # Pass outer execution_id to nested Composition
                    node._assign_context_values(execution_id, composition=node, propagate=True)

                    # Execute Composition
                    # FIX: 6/12/19 WHERE IS COMPILED EXECUTION OF NESTED NODE?
                    # autodiff compositions must be passed extra inputs
                    pytorch_enabled = False
                    if hasattr(node, "pytorch_representation"):
                        if node.learning_enabled:
                            pytorch_enabled = True
                    # Autodiff execution
                    if pytorch_enabled:
                        ret = node.execute(inputs=autodiff_stimuli[node],
                                           execution_id=execution_id,
                                           context=ContextFlags.COMPOSITION)
                    # Standard execution
                    else:
                        ret = node.execute(execution_id=execution_id,
                                           context=ContextFlags.COMPOSITION)

                    # Get output info from compiled execution
                    if bin_execute:
                        # Update result in binary data structure
                        _comp_ex.insert_node_output(node, ret)
                        for i, v in enumerate(ret):
                            # Set current output. This will be stored to "new_values" below
                            node.output_CIM.output_states[i].parameters.value._set(v, execution_id, skip_history=True,
                                                                                  skip_log=True, override=True)

                # ANIMATE node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if self._animate is not False and self._animate_unit is COMPONENT:
                    self._animate_execution(node, execution_id)


                # MANAGE INPUTS (for next execution_set)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # FIX: 6/12/19 Deprecate?
                # Handle input clamping
                if node in input_nodes:
                    if clamp_input:
                        if node in pulse_clamp_inputs:
                            for input_state in node.input_states:
                                # clamp = None --> "turn off" input node
                                self.input_CIM_states[input_state][1].parameters.value._set(0, execution_id,
                                                                                           override=True)

                # Store new value generated by node,
                #    then set back to frozen value for use by other nodes in execution_set
                new_values[node] = node.get_output_values(execution_id)
                for i in range(len(node.output_states)):
                    node.output_states[i].parameters.value._set(frozen_values[node][i], execution_id, override=True,
                                                               skip_history=True, skip_log=True)


            # Set all nodes to new values
            for node in next_execution_set:
                for i in range(len(node.output_states)):
                    node.output_states[i].parameters.value._set(new_values[node][i], execution_id, override=True,
                                                               skip_history=True, skip_log=True)

            if call_after_time_step:
                call_with_pruned_args(call_after_time_step, execution_context=execution_id)

        # Update matrix parameter of PathwayProjections being learned with learning_enabled==AFTER
        if self.enable_learning:
            for projection in [p for p in self.projections if
                               hasattr(p, 'has_learning_projection') and p.has_learning_projection]:
                matrix_parameter_state = projection.parameter_states[MATRIX]
                if any([lp for lp in matrix_parameter_state.mod_afferents if lp.learning_enabled == AFTER]):
                    execution_phase_buffer = projection.parameters.context._get(execution_id).execution_phase
                    projection.parameters.context._get(execution_id).execution_phase = ContextFlags.LEARNING
                    projection.parameters.context._get(execution_id).string = \
                        f"Updating {ParameterState.__name__} for {projection.name} in {self.name}"
                    matrix_parameter_state.update(execution_id=execution_id,
                                                                context=ContextFlags.COMPOSITION)
                    projection.parameters.context._get(execution_id).execution_phase = execution_phase_buffer

        if call_after_pass:
            call_with_pruned_args(call_after_pass, execution_context=execution_id)


        # Animate output_CIM
        # FIX: NOT SURE WHETHER IT CAN BE LEFT IN PROCESSING AFTER THIS -
        #      COORDINATE WITH REFACTORING OF PROCESSING/CONTROL CONTEXT
        # execution_phase_buffer = self.parameters.context.get(execution_id).execution_phase
        # self.parameters.context.get(execution_id).execution_phase = ContextFlags.PROCESSING
        if self._animate is not False and SHOW_CIM in self._animate and self._animate[SHOW_CIM]:
            self._animate_execution(self.output_CIM, execution_id)
        # self.parameters.context.get(execution_id).execution_phase = execution_phase_buffer
        # FIX: END

        # Restore execution_context to state entry of execute method
        if execution_context:
            execution_context.execution_phase = entry_execution_phase

        # EXECUTE CONTROLLER (if controller_mode == AFTER) ************************************************************

        if (self.enable_controller and
                self.controller_mode == AFTER and
                self.controller_condition.is_satisfied(scheduler=execution_scheduler,
                                                       execution_context=execution_id)):
            # control phase
            # FIX: SHOULD SET CONTEXT AS CONTROL HERE AND RESET AT END (AS DONE FOR animation BELOW)
            execution_phase = execution_context.execution_phase
            if (
                    execution_phase != ContextFlags.INITIALIZING
                    and execution_phase != ContextFlags.SIMULATION
            ):

                if self.controller and not bin_execute:
                    # FIX: REMOVE ONCE context IS SET TO CONTROL ABOVE
                    self.controller.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING
                    # FIX: END REMOVE
                    self.controller.execute(execution_id=execution_id, context=context)

                if bin_execute:
                    _comp_ex.freeze_values()
                    _comp_ex.execute_node(self.controller)

                # MODIFIED 6/13/19 NEW: [JDC]
                # FIX: NEEDED TO ANIMATE CONTROL; REMOVE ONCE context IS SET TO CONTROL ABOVE
                if execution_context:
                    entry_execution_phase = execution_context.execution_phase
                    execution_context.execution_phase = ContextFlags.CONTROL
                # MODIFIED 6/13/19 END

                # Animate controller (after execution)
                if self._animate is not False and SHOW_CONTROLLER in self._animate and self._animate[SHOW_CONTROLLER]:
                    self._animate_execution(self.controller, execution_id)

                # MODIFIED 6/13/19 NEW: [JDC]
                # FIX: REMOVE ONCE context IS SET TO CONTROL ABOVE
                if execution_context:
                    execution_context.execution_phase = entry_execution_phase
                # MODIFIED 6/13/19 END

        execution_scheduler.clocks[execution_id]._increment_time(TimeScale.TRIAL)

        # REPORT RESULTS ***********************************************************************************************

        # Extract result here
        if bin_execute:
            _comp_ex.freeze_values()
            _comp_ex.execute_node(self.output_CIM)
            return _comp_ex.extract_node_output(self.output_CIM)

        self.output_CIM.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING
        self.output_CIM.execute(execution_id=execution_id, context=ContextFlags.PROCESSING)

        output_values = []
        for state in self.output_CIM.output_states:
            output_values.append(state.parameters.value._get(execution_id))

        return output_values

    def reinitialize(self, values, execution_context=NotImplemented):
        if execution_context is NotImplemented:
            execution_context = self.most_recent_execution_id

        for i in range(self.stateful_nodes):
            self.stateful_nodes[i].reinitialize(values[i], execution_context=execution_context)

    def run(
            self,
            inputs=None,
            scheduler_processing=None,
            termination_processing=None,
            num_trials=None,
            call_before_time_step=None,
            call_after_time_step=None,
            call_before_pass=None,
            call_after_pass=None,
            call_before_trial=None,
            call_after_trial=None,
            clamp_input=SOFT_CLAMP,
            bin_execute=False,
            log=False,
            initial_values=None,
            reinitialize_values=None,
            runtime_params=None,
            skip_initialization=False,
            animate=False,
            execution_id=None,
            base_execution_id=None,
            context=None):

        """Pass inputs to Composition, then execute sets of nodes that are eligible to run until termination
        conditions are met.  See `Run` for details of formatting input specifications. See `Run` for details of
        formatting input specifications. Use **animate** to generate a gif of the execution sequence.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` : list } or { `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each Node in the composition that receives inputs from
                the user. For each pair, the key is the Node and the value is a list of inputs. Each input in the
                list corresponds to a certain `TRIAL`.

            scheduler_processing : Scheduler
                the scheduler object that owns the conditions that will instruct the execution of the Composition.
                If not specified, the Composition will use its automatically generated scheduler.

            execution_id
                execution_id will be set to self.default_execution_id if unspecified

            base_execution_id
                the execution_id corresponding to the execution context from which this execution will be initialized,
                if values currently do not exist for **execution_id**

            num_trials : int
                typically, the composition will infer the number of trials from the length of its input specification.
                To reuse the same inputs across many trials, you may specify an input dictionary with lists of length 1,
                or use default inputs, and select a number of trials with num_trials.

            call_before_time_step : callable
                will be called before each `TIME_STEP` is executed.

            call_after_time_step : callable
                will be called after each `TIME_STEP` is executed.

            call_before_pass : callable
                will be called before each `PASS` is executed.

            call_after_pass : callable
                will be called after each `PASS` is executed.

            call_before_trial : callable
                will be called before each `TRIAL` is executed.

            call_after_trial : callable
                will be called after each `TRIAL` is executed.

            initial_values : Dict[Node: Node Value]
                sets the values of nodes before the start of the run. This is useful in cases where a node's value is
                used before that node executes for the first time (usually due to recurrence or control).

            runtime_params : Dict[Node: Dict[Parameter: Tuple(Value, Condition)]]
                nested dictionary of (value, `Condition`) tuples for parameters of Nodes (`Mechanisms <Mechanism>` or
                `Compositions <Composition>` of the Composition; specifies alternate parameter values to be used only
                during this `Run` when the specified `Condition` is met.

                Outer dictionary:
                    - *key* - Node
                    - *value* - Runtime Parameter Specification Dictionary

                Runtime Parameter Specification Dictionary:
                    - *key* - keyword corresponding to a parameter of the Node
                    - *value* - tuple in which the index 0 item is the runtime parameter value, and the index 1 item is
                      a `Condition`

                See `Run_Runtime_Parameters` for more details and examples of valid dictionaries.

            animate : dict or bool : False
                specifies use of the `show_graph <Composition.show_graph>` method to generate a gif movie showing the
                sequence of Components executed in a run.  A dict can be specified containing options to pass to
                the `show_graph <Composition.show_graph>` method;  each key must be a legal argument for the `show_graph
                <Composition.show_graph>` method, and its value a specification for that argument.  The entries listed
                below can also be included in the dict to specify parameters of the animation.  If the **animate**
                argument is specified simply as `True`, defaults are used for all arguments of `show_graph
                <Composition.show_graph>` and the options below:

                * *UNIT*: *EXECUTION_SET* or *COMPONENT* (default=\\ *EXECUTION_SET*\\ ) -- specifies which Components
                  to treat as active in each call to `show_graph <Composition.show_graph>`. *COMPONENT* generates an
                  image for the execution of each Component.  *EXECUTION_SET* generates an image for each `execution_set
                  <Component.execution_sets>`, showing all of the Components in that set as active.

                * *DURATION*: float (default=0.75) -- specifies the duration (in seconds) of each image in the movie.

                * *NUM_RUNS*: int (default=1) -- specifies the number of runs to animate;  by default, this is 1.
                  If the number specified is less than the total number of runs executed, only the number specified
                  are animated; if it is greater than the number of runs being executed, only the number being run are
                  animated.

                * *NUM_TRIALS*: int (default=1) -- specifies the number of trials to animate;  by default, this is 1.
                  If the number specified is less than the total number of trials being run, only the number specified
                  are animated; if it is greater than the number of trials being run, only the number being run are
                  animated.

                * *MOVIE_DIR*: str (default=project root dir) -- specifies the directdory to be used for the movie file;
                  by default a subdirectory of <root_dir>/show_graph_OUTPUT/GIFS is created using the `name
                  <Composition.name>` of the  `Composition`, and the gif files are stored there.

                * *MOVIE_NAME*: str (default=\\ `name <System.name>` + 'movie') -- specifies the name to be used for
                  the movie file; it is automatically appended with '.gif'.

                * *SAVE_IMAGES*: bool (default=\\ `False`\\ ) -- specifies whether to save each of the images used to
                  construct the animation in separate gif files, in addition to the file containing the animation.

                * *SHOW*: bool (default=\\ `False`\\ ) -- specifies whether to show the animation after it is
                  constructed, using the OS's default viewer.

            log : bool, LogCondition
                Sets the `log_condition <Parameter.log_condition>` for every primary `node <Composition.nodes>` and
                `projection <Composition.projections>` in this Composition, if it is not already set.

                .. note::
                   as when setting the `log_condition <Parameter.log_condition>` directly, a value of `True` will
                   correspond to the `EXECUTION LogCondition <LogCondition.EXECUTION>`.

        COMMENT:
        REPLACE WITH EVC/OCM EXAMPLE
        Examples
        --------

        This figure shows an animation of the Composition in the XXX example script, with
        the show_graph **show_learning** argument specified as *ALL*:

        .. _Composition_XXX_movie:

        .. figure:: _static/XXX_movie.gif
           :alt: Animation of Composition in XXX example script
           :scale: 50 %

        This figure shows an animation of the Composition in the XXX example script, with
        the show_graph **show_control** argument specified as *ALL* and *UNIT* specified as *EXECUTION_SET*:

        .. _Composition_XXX_movie:

        .. figure:: _static/XXX_movie.gif
           :alt: Animation of Composition in XXX example script
           :scale: 150 %
        COMMENT

        Returns
        ---------

        output value of the final Node executed in the composition : various
        """

        if scheduler_processing is None:
            scheduler_processing = self.scheduler_processing

        if termination_processing is None:
            termination_processing = self.termination_processing
        else:
            new_conds = self.termination_processing.copy()
            new_conds.update(termination_processing)
            termination_processing = new_conds

        if initial_values is not None:
            for node in initial_values:
                if node not in self.nodes:
                    raise CompositionError("{} (entry in initial_values arg) is not a node in \'{}\'".
                                           format(node.name, self.name))

        if reinitialize_values is None:
            reinitialize_values = {}

        for node in reinitialize_values:
            node.reinitialize(*reinitialize_values[node], execution_context=execution_id)

        try:
            if self.parameters.context._get(execution_id).execution_phase != ContextFlags.SIMULATION:
                self._analyze_graph()
        except AttributeError:
            # if context is None, it has not been created for this execution_id yet, so it is not
            # in a simulation
            self._analyze_graph()

        # set auto logging if it's not already set, and if log argument is True
        if log:
            for item in self.nodes + self.projections:
                if not isinstance(item, CompositionInterfaceMechanism):
                    for param in item.parameters:
                        if param.loggable and param.log_condition is LogCondition.OFF:
                            param.log_condition = LogCondition.EXECUTION

        # Set animation attributes
        if animate is True:
            animate = {}
        self._animate = animate
        if self._animate is not False:
            self._set_up_animation(execution_id)

        # SET UP EXECUTION -----------------------------------------------

        results = []

        execution_id = self._assign_execution_ids(execution_id)

        scheduler_processing._init_counts(execution_id=execution_id)

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        # if there is only one INPUT Node, allow inputs to be specified in a list
        if isinstance(inputs, (list, np.ndarray)):
            if len(input_nodes) == 1:
                inputs = {next(iter(input_nodes)): inputs}
            else:
                raise CompositionError(
                    f"Inputs to {self.name} must be specified in a dictionary with a key for each of its "
                    f"{len(input_nodes)} INPUT nodes ({[n.name for n in input_nodes]}).")
        elif callable(inputs):
            num_inputs_sets = 1
            autodiff_stimuli = {}
        elif not isinstance(inputs, dict):
            if len(input_nodes) == 1:
                raise CompositionError(
                    "Inputs to {} must be specified in a list or in a dictionary "
                    "with the INPUT node ({}) as its only key".
                        format(self.name, next(iter(input_nodes)).name))
            else:
                input_node_names = ", ".join([i.name for i in input_nodes])
                raise CompositionError(
                    "Inputs to {} must be specified in a dictionary "
                    "with its {} INPUT nodes ({}) as the keys and their inputs as the values".
                    format(self.name, len(input_nodes), input_node_names))
        if not callable(inputs):
            # Currently, no validation if 'inputs' arg is a function
            inputs, num_inputs_sets, autodiff_stimuli = self._adjust_stimulus_dict(inputs)

        if num_trials is not None:
            num_trials = num_trials
        else:
            num_trials = num_inputs_sets

        scheduler_processing._reset_counts_total(TimeScale.RUN, execution_id)

        execution_context = self.parameters.context._get(execution_id)

        # KDM 3/29/19: run the following not only during LLVM Run compilation, due to bug where TimeScale.RUN
        # termination condition is checked and no data yet exists. Adds slight overhead as long as run is not
        # called repeatedly (this init is repeated in Composition.execute)
        # initialize from base context but don't overwrite any values already set for this execution_id
        if (
            not skip_initialization
            and (execution_context is None or execution_context.execution_phase != ContextFlags.SIMULATION)
        ):
            self._initialize_from_context(execution_id, base_execution_id, override=False)
            self._assign_context_values(execution_id=execution_id, composition=self, propagate=True)

        is_simulation = (execution_context is not None and
                         execution_context.execution_phase == ContextFlags.SIMULATION)
        if (bin_execute is True or str(bin_execute).endswith('Run')):
            # There's no mode to run simulations.
            # Simulations are run as part of the controller node wrapper.
            assert not is_simulation
            try:
                if bin_execute is True or bin_execute.startswith('LLVM'):
                    _comp_ex = pnlvm.CompExecution(self, [execution_id])
                    results += _comp_ex.run(inputs, num_trials, num_inputs_sets)
                elif bin_execute.startswith('PTX'):
                    self.__ptx_initialize(execution_id)
                    EX = self._compilation_data.ptx_execution._get(execution_id)
                    results += EX.cuda_run(inputs, num_trials, num_inputs_sets)

                full_results = self.parameters.results._get(execution_id)
                if full_results is None:
                    full_results = results
                else:
                    full_results.extend(results)

                self.parameters.results._set(full_results, execution_id)
                # KAM added the [-1] index after changing Composition run()
                # behavior to return only last trial of run (11/7/18)
                self.most_recent_execution_id = execution_id
                return full_results[-1]

            except Exception as e:
                if bin_execute is not True:
                    raise e

                print("WARNING: Failed to Run execution `{}': {}".format(
                      self.name, str(e)))

        # Reset gym forager environment for the current trial
        if self.env:
            trial_output = np.atleast_2d(self.env.reset())

        # Loop over the length of the list of inputs - each input represents a TRIAL
        for trial_num in range(num_trials):

            # Execute call before trial "hook" (user defined function)
            if call_before_trial:
                call_with_pruned_args(call_before_trial, execution_context=execution_id)

            if termination_processing[TimeScale.RUN].is_satisfied(
                scheduler=scheduler_processing,
                execution_context=execution_id
            ):
                break

            # PROCESSING ------------------------------------------------------------------------
            # Prepare stimuli from the outside world  -- collect the inputs for this TRIAL and store them in a dict
            if callable(inputs):
                # If 'inputs' argument is a function, call the function here with results from last trial
                execution_stimuli = inputs(self.env, trial_output)
                if not isinstance(execution_stimuli, dict):
                    return trial_output
            else:
                execution_stimuli = {}
                stimulus_index = trial_num % num_inputs_sets
                for node in inputs:
                    if len(inputs[node]) == 1:
                        execution_stimuli[node] = inputs[node][0]
                        continue
                    execution_stimuli[node] = inputs[node][stimulus_index]

            execution_autodiff_stimuli = {}
            for node in autodiff_stimuli:
                if isinstance(autodiff_stimuli[node], list):
                    execution_autodiff_stimuli[node] = autodiff_stimuli[node][stimulus_index]
                else:
                    execution_autodiff_stimuli[node] = autodiff_stimuli[node]

            for node in self.nodes:
                if hasattr(node, "reinitialize_when") and node.parameters.has_initializers._get(execution_id):
                    if node.reinitialize_when.is_satisfied(scheduler=self.scheduler_processing,
                                                           execution_context=execution_id):
                        node.reinitialize(None, execution_context=execution_id)

            # execute processing
            # pass along the stimuli for this trial
            trial_output = self.execute(inputs=execution_stimuli,
                                        autodiff_stimuli=execution_autodiff_stimuli,
                                        scheduler_processing=scheduler_processing,
                                        termination_processing=termination_processing,
                                        call_before_time_step=call_before_time_step,
                                        call_before_pass=call_before_pass,
                                        call_after_time_step=call_after_time_step,
                                        call_after_pass=call_after_pass,
                                        execution_id=execution_id,
                                        base_execution_id=base_execution_id,
                                        clamp_input=clamp_input,
                                        runtime_params=runtime_params,
                                        skip_initialization=True,
                                        bin_execute=bin_execute)

            # ---------------------------------------------------------------------------------
            # store the result of this execute in case it will be the final result

            # object.results.append(result)
            if isinstance(trial_output, collections.abc.Iterable):
                result_copy = trial_output.copy()
            else:
                result_copy = trial_output

            if self.parameters.context._get(execution_id).execution_phase != ContextFlags.SIMULATION:
                results.append(result_copy)

                if not self.parameters.retain_old_simulation_data._get():
                    if self.controller is not None:
                        # if any other special parameters store simulation info that needs to be cleaned up
                        # consider dedicating a function to it here
                        # this will not be caught above because it resides in the base context (execution_id)
                        if not self.parameters.simulation_results.retain_old_simulation_data:
                            self.parameters.simulation_results._get(execution_id).clear()

                        if not self.controller.parameters.simulation_ids.retain_old_simulation_data:
                            self.controller.parameters.simulation_ids._get(execution_id).clear()

            if call_after_trial:
                call_with_pruned_args(call_after_trial, execution_context=execution_id)

        scheduler_processing.clocks[execution_id]._increment_time(TimeScale.RUN)

        full_results = self.parameters.results._get(execution_id)
        if full_results is None:
            full_results = results
        else:
            full_results.extend(results)

        self.parameters.results._set(full_results, execution_id)

        self.most_recent_execution_id = execution_id

        if self._animate is not False:
            # Save list of gifs in self._animation as movie file
            movie_path = self._animation_directory + '/' + self._movie_filename
            self._animation[0].save(fp=movie_path,
                                    format='GIF',
                                    save_all=True,
                                    append_images=self._animation[1:],
                                    duration=self._image_duration*1000,
                                    loop=0)
            print('\nSaved movie for {}: {}'.format(self.name, self._movie_filename))
            if self._show_animation:
                movie = Image.open(movie_path)
                movie.show()

        return trial_output

    def _after_agent_rep_execution(self, context=None):
        pass

    @property
    def _all_nodes(self):
        for n in self.nodes:
            yield n
        yield self.input_CIM
        yield self.output_CIM
        if self.controller:
            yield self.controller

    def _get_param_struct_type(self, ctx):
        mech_param_type_list = (ctx.get_param_struct_type(m) for m in self._all_nodes)
        proj_param_type_list = (ctx.get_param_struct_type(p) for p in self.projections)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(mech_param_type_list),
            pnlvm.ir.LiteralStructType(proj_param_type_list)))

    def _get_context_struct_type(self, ctx):
        mech_ctx_type_list = (ctx.get_context_struct_type(m) for m in self._all_nodes)
        proj_ctx_type_list = (ctx.get_context_struct_type(p) for p in self.projections)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(mech_ctx_type_list),
            pnlvm.ir.LiteralStructType(proj_ctx_type_list)))

    def _get_input_struct_type(self, ctx):
        return ctx.get_input_struct_type(self.input_CIM)

    def _get_output_struct_type(self, ctx):
        return ctx.get_output_struct_type(self.output_CIM)

    def _get_data_struct_type(self, ctx):
        output_type_list = (ctx.get_output_struct_type(m) for m in self._all_nodes)

        data = [pnlvm.ir.LiteralStructType(output_type_list)]
        for node in self.nodes:
            nested_data = ctx.get_data_struct_type(node)
            data.append(nested_data)
        return pnlvm.ir.LiteralStructType(data)

    def _get_context_initializer(self, execution_id=None, simulation=False):
        mech_contexts = (tuple(m._get_context_initializer(execution_id=execution_id))
                         for m in self._all_nodes if m is not self.controller or not simulation)
        proj_contexts = (tuple(p._get_context_initializer(execution_id=execution_id)) for p in self.projections)
        return (tuple(mech_contexts), tuple(proj_contexts))

    def _get_param_initializer(self, execution_id, simulation=False):
        mech_params = (tuple(m._get_param_initializer(execution_id))
                       for m in self._all_nodes if m is not self.controller or not simulation)
        proj_params = (tuple(p._get_param_initializer(execution_id)) for p in self.projections)
        return (tuple(mech_params), tuple(proj_params))

    def _get_flattened_controller_output(self, execution_id):
        controller_data = [os.parameters.value._get(execution_id) for os in self.controller.output_states]
        # This is an ugly hack to remove 2d arrays
        try:
            controller_data = [[c[0][0]] for c in controller_data]
        except:
            pass
        return controller_data

    def _get_data_initializer(self, execution_id=None):
        output = [(os.parameters.value._get(execution_id) for os in m.output_states) for m in self._all_nodes]
        data = [output]
        for node in self.nodes:
            nested_data = node._get_data_initializer(execution_id=execution_id) \
                if hasattr(node,'_get_data_initializer') else []
            data.append(nested_data)
        return pnlvm._tupleize(data)

    def _get_node_index(self, node):
        node_list = list(self._all_nodes)
        return node_list.index(node)

    def _get_node_wrapper(self, node, simulation=False):
        if node not in self.__generated_node_wrappers:
            class node_wrapper():
                def __init__(self, node, gen_f):
                    self._node = node
                    self._gen_f = gen_f
                def _gen_llvm_function(self):
                    return self._gen_f(self._node)
            wrapper = node_wrapper(node, self.__gen_node_wrapper)
            self.__generated_node_wrappers[node] = wrapper
            return wrapper

        return self.__generated_node_wrappers[node]

    def _gen_llvm_function(self):
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
                return ctx.gen_composition_exec(self)

    @property
    def _llvm_run(self):
        if self.__generated_run is None:
            with pnlvm.LLVMBuilderContext.get_global() as ctx:
                self.__generated_run = ctx.gen_composition_run(self)

        return self.__generated_run

    @property
    def _llvm_simulation(self):
        if self.__generated_simulation is None:
            with pnlvm.LLVMBuilderContext.get_global() as ctx:
                self.__generated_simulation = ctx.gen_composition_exec(self, True)

        return self.__generated_simulation

    @property
    def _llvm_sim_run(self):
        if self.__generated_sim_run is None:
            with pnlvm.LLVMBuilderContext.get_global() as ctx:
                self.__generated_sim_run = ctx.gen_composition_run(self, True)

        return self.__generated_sim_run

    def reinitialize(self, execution_context=NotImplemented):
        if execution_context is NotImplemented:
            execution_context = self.most_recent_execution_id

        self._compilation_data.ptx_execution.set(None, execution_context)
        self._compilation_data.parameter_struct.set(None, execution_context)
        self._compilation_data.context_struct.set(None, execution_context)
        self._compilation_data.data_struct.set(None, execution_context)
        self._compilation_data.scheduler_conditions.set(None, execution_context)

    def __ptx_initialize(self, execution_id=None):
        if self._compilation_data.ptx_execution._get(execution_id) is None:
            self._compilation_data.ptx_execution._set(pnlvm.CompExecution(self, [execution_id]), execution_id)

    def __gen_node_wrapper(self, node, simulation=False):
        assert node is not self.controller or simulation is False
        name = 'comp_sim_wrap_' if simulation else 'comp_wrap_'
        is_mech = isinstance(node, Mechanism)

        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            data_struct_ptr = ctx.get_data_struct_type(self).as_pointer()
            args = [
                ctx.get_context_struct_type(self).as_pointer(),
                ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                data_struct_ptr, data_struct_ptr]

            if not is_mech:
                # Add condition struct
                cond_gen = pnlvm.helpers.ConditionGenerator(ctx, self)
                cond_ty = cond_gen.get_condition_struct_type().as_pointer()
                args.append(cond_ty)

            builder = ctx.create_llvm_function(args, node, name + node.name)
            llvm_func = builder.function
            llvm_func.attributes.add('alwaysinline')
            for a in llvm_func.args:
                a.attributes.add('nonnull')

            context, params, comp_in, data_in, data_out = llvm_func.args[:5]
            cond_ptr = llvm_func.args[-1]

            m_function = ctx.get_llvm_function(node)

            if node is self.input_CIM:
                m_in = comp_in
                incoming_projections = []
            elif not is_mech:
                m_in = builder.alloca(m_function.args[2].type.pointee)
                incoming_projections = node.input_CIM.afferents
            else:
                m_in = builder.alloca(m_function.args[2].type.pointee)
                incoming_projections = node.afferents

            # Run all incoming projections
            # TODO: This should filter out projections with different execution ID

            for par_proj in incoming_projections:
                # Skip autoassociative projections
                if par_proj.sender.owner is par_proj.receiver.owner:
                    continue

                proj_idx = self.projections.index(par_proj)

                # Get parent mechanism
                par_mech = par_proj.sender.owner

                proj_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(proj_idx)])
                proj_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(proj_idx)])
                proj_function = ctx.get_llvm_function(par_proj)

                output_s = par_proj.sender
                assert output_s in par_mech.output_states
                if par_mech in self._all_nodes:
                    par_idx = self._get_node_index(par_mech)
                else:
                    comp = par_mech.composition
                    assert par_mech is comp.output_CIM
                    par_idx = self.nodes.index(comp)

                output_state_idx = par_mech.output_states.index(output_s)
                proj_in = builder.gep(data_in, [ctx.int32_ty(0),
                                                ctx.int32_ty(0),
                                                ctx.int32_ty(par_idx),
                                                ctx.int32_ty(output_state_idx)])

                state = par_proj.receiver
                assert state.owner is node or state.owner is node.input_CIM
                if state in state.owner.input_states:
                    state_idx = state.owner.input_states.index(state)

                    assert par_proj in state.pathway_projections
                    projection_idx = state.pathway_projections.index(par_proj)

                    # Adjust for AutoAssociative projections
                    for i in range(projection_idx):
                        if isinstance(state.pathway_projections[i], AutoAssociativeProjection):
                            projection_idx -= 1
                elif state in state.owner.parameter_states:
                    state_idx = state.owner.parameter_states.index(state) + len(state.owner.input_states)

                    assert par_proj in state.mod_afferents
                    projection_idx = state.mod_afferents.index(par_proj)
                else:
                    assert False, "State neither an input state nor a parameter state"

                assert state_idx < len(m_in.type.pointee)
                assert projection_idx < len(m_in.type.pointee.elements[state_idx])
                proj_out = builder.gep(m_in, [ctx.int32_ty(0),
                                              ctx.int32_ty(state_idx),
                                              ctx.int32_ty(projection_idx)])

                if proj_in.type != proj_function.args[2].type:
                    assert node is self.output_CIM
                    proj_in = builder.bitcast(proj_in, proj_function.args[2].type)
                builder.call(proj_function, [proj_params, proj_context, proj_in, proj_out])

            idx = ctx.int32_ty(self._get_node_index(node))
            zero = ctx.int32_ty(0)
            m_params = builder.gep(params, [zero, zero, idx])
            m_context = builder.gep(context, [zero, zero, idx])
            m_out = builder.gep(data_out, [zero, zero, idx])
            if is_mech:
                call_args = [m_params, m_context, m_in, m_out]
                if node is self.controller:
                    call_args += [params, context, data_in]
                builder.call(m_function, call_args)
            else:
                # Condition and data structures includes parent first
                nested_idx = ctx.int32_ty(self._get_node_index(node) + 1)
                m_data = builder.gep(data_in, [zero, nested_idx])
                m_cond = builder.gep(cond_ptr, [zero, nested_idx])
                builder.call(m_function, [m_context, m_params, m_in, m_data, m_cond])
                # Copy output of the nested composition to its output place
                output_idx = node._get_node_index(node.output_CIM)
                result = builder.gep(m_data, [zero, zero, ctx.int32_ty(output_idx)])
                builder.store(builder.load(result), m_out)

            builder.ret_void()

        return llvm_func

    def _get_processing_condition_set(self, node):
        dep_group = []
        for group in self.scheduler_processing.consideration_queue:
            if node in group:
                break
            dep_group = group

        # NOTE: This is not ideal we don't need to depend on
        # the entire previous group. Only our dependencies
        cond = [EveryNCalls(dep, 1) for dep in dep_group]
        if node not in self.scheduler_processing.conditions:
            cond.append(Always())
        else:
            node_conds = self.scheduler_processing.conditions[node]
            cond.append(node_conds)

        return All(*cond)

    def _input_matches_variable(self, input_value, var):
        # input_value states are uniform
        if np.shape(np.atleast_2d(input_value)) == np.shape(var):
            return "homogeneous"
        # input_value states have different lengths
        elif len(np.shape(var)) == 1 and isinstance(var[0], (list, np.ndarray)):
            for i in range(len(input_value)):
                if len(input_value[i]) != len(var[i]):
                    return False
            return "heterogeneous"
        return False

    def _adjust_stimulus_dict(self, stimuli):

        autodiff_stimuli = {}
        all_stimuli_keys = list(stimuli.keys())
        for node in all_stimuli_keys:
            if hasattr(node, "pytorch_representation"):
                if node.learning_enabled:
                    autodiff_stimuli[node] = stimuli[node]
                    del stimuli[node]

        # STEP 1A: Check that all of the nodes listed in the inputs dict are INPUT nodes in the composition
        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        for node in stimuli.keys():
            if not node in input_nodes:
                raise CompositionError("{} in inputs dict for {} is not one of its INPUT nodes".
                                       format(node.name, self.name))

        # STEP 1B: Check that all of the INPUT nodes are represented - if not, use default_external_input_values
        for node in input_nodes:
            if not node in stimuli:
                stimuli[node] = node.default_external_input_values

        # STEP 2: Loop over all dictionary entries to validate their content and adjust any convenience notations:

        # (1) Replace any user provided convenience notations with values that match the following specs:
        # a - all dictionary values are lists containing an input value for each trial (even if only one trial)
        # b - each input value is a 2d array that matches variable
        # example: { Mech1: [Fully_specified_input_for_mech1_on_trial_1, Fully_specified_input_for_mech1_on_trial_2 … ],
        #            Mech2: [Fully_specified_input_for_mech2_on_trial_1, Fully_specified_input_for_mech2_on_trial_2 … ]}
        # (2) Verify that all nodes provide the same number of inputs (check length of each dictionary value)

        adjusted_stimuli = {}
        nums_input_sets = set()
        for node, stim_list in stimuli.items():
            if isinstance(node, Composition):
                if isinstance(stim_list, dict):

                    adjusted_stimulus_dict, num_trials, autodiff_stimuli = node._adjust_stimulus_dict(stim_list)
                    translated_stimulus_dict = {}

                    # first time through the stimulus dictionary, assemble a dictionary in which the keys are input CIM
                    # InputStates and the values are lists containing the first input value
                    for nested_input_node, values in adjusted_stimulus_dict.items():
                        first_value = values[0]
                        for i in range(len(first_value)):
                            input_state = nested_input_node.external_input_states[i]
                            input_cim_input_state = node.input_CIM_states[input_state][0]
                            translated_stimulus_dict[input_cim_input_state] = [first_value[i]]
                            # then loop through the stimulus dictionary again for each remaining trial
                            for trial in range(1, num_trials):
                                translated_stimulus_dict[input_cim_input_state].append(values[trial][i])

                    adjusted_stimulus_list = []
                    for trial in range(num_trials):
                        trial_adjusted_stimulus_list = []
                        for state in node.external_input_states:
                            trial_adjusted_stimulus_list.append(translated_stimulus_dict[state][trial])
                        adjusted_stimulus_list.append(trial_adjusted_stimulus_list)
                    stimuli[node] = adjusted_stimulus_list
                    stim_list = adjusted_stimulus_list  # ADDED CW 12/21/18: This line fixed a bug, but it might be a hack

            # excludes any input states marked "internal_only" (usually recurrent)
            # KDM 3/29/19: changed to use defaults equivalent of node.external_input_values
            input_must_match = [input_state.defaults.value for input_state in node.input_states if not input_state.internal_only]

            if input_must_match == []:
                # all input states are internal_only
                continue

            check_spec_type = self._input_matches_variable(stim_list, input_must_match)
            # If a node provided a single input, wrap it in one more list in order to represent trials
            if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
                if check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input states specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[node] = [np.atleast_2d(stim_list)]
                else:
                    adjusted_stimuli[node] = [stim_list]
                nums_input_sets.add(1)

            else:
                adjusted_stimuli[node] = []
                for stim in stimuli[node]:
                    check_spec_type = self._input_matches_variable(stim, input_must_match)
                    # loop over each input to verify that it matches variable
                    if check_spec_type == False:
                        err_msg = "Input stimulus ({}) for {} is incompatible with its external_input_values ({}).". \
                            format(stim, node.name, input_must_match)
                        # 8/3/17 CW: I admit the error message implementation here is very hacky; but it's at least not a hack
                        # for "functionality" but rather a hack for user clarity
                        if "KWTA" in str(type(node)):
                            err_msg = err_msg + " For KWTA mechanisms, remember to append an array of zeros (or other values)" \
                                                " to represent the outside stimulus for the inhibition input state, and " \
                                                "for systems, put your inputs"
                        raise RunError(err_msg)
                    elif check_spec_type == "homogeneous":
                        # np.atleast_2d will catch any single-input states specified without an outer list
                        # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                        adjusted_stimuli[node].append(np.atleast_2d(stim))
                    else:
                        adjusted_stimuli[node].append(stim)
                nums_input_sets.add(len(stimuli[node]))
        if len(nums_input_sets) > 1:
            if 1 in nums_input_sets:
                nums_input_sets.remove(1)
                if len(nums_input_sets) > 1:
                    raise CompositionError("The input dictionary for {} contains input specifications of different "
                                           "lengths ({}). The same number of inputs must be provided for each node "
                                           "in a Composition.".format(self.name, nums_input_sets))
            else:
                raise CompositionError("The input dictionary for {} contains input specifications of different "
                                       "lengths ({}). The same number of inputs must be provided for each node "
                                       "in a Composition.".format(self.name, nums_input_sets))
        num_input_sets = nums_input_sets.pop()
        return adjusted_stimuli, num_input_sets, autodiff_stimuli

    def _adjust_execution_stimuli(self, stimuli):
        adjusted_stimuli = {}
        for node, stimulus in stimuli.items():
            if isinstance(node, Composition):
                input_must_match = node.external_input_values
                if isinstance(stimulus, dict):
                    adjusted_stimulus_dict = node._adjust_stimulus_dict(stimulus)
                    adjusted_stimuli[node] = adjusted_stimulus_dict
                    continue
            else:
                input_must_match = node.default_external_input_values

            check_spec_type = self._input_matches_variable(stimulus, input_must_match)
            # If a node provided a single input, wrap it in one more list in order to represent trials
            if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
                if check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input states specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[node] = np.atleast_2d(stimulus)
                else:
                    adjusted_stimuli[node] = stimulus

            else:
                raise CompositionError("Input stimulus ({}) for {} is incompatible with its variable ({})."
                                       .format(stimulus, node.name, input_must_match))
        return adjusted_stimuli

    def reshape_control_signal(self,
                                         arr):

        current_shape = np.shape(arr)
        if len(current_shape) > 2:
            newshape = (current_shape[0], current_shape[1])
            newarr = np.reshape(arr, newshape)
            arr = tuple(newarr[i].item() for i in range(len(newarr)))

        return np.array(arr)

    def _get_total_cost_of_control_allocation(self, control_allocation, execution_id, runtime_params, context):
        total_cost = 0.
        if control_allocation is not None:  # using "is not None" in case the control allocation is 0.

            base_control_allocation = self.reshape_control_signal(self.controller.parameters.value._get(execution_id))

            candidate_control_allocation = self.reshape_control_signal(control_allocation)

            # Get reconfiguration cost for candidate control signal
            reconfiguration_cost = 0.
            if callable(self.controller.compute_reconfiguration_cost):
                reconfiguration_cost = self.controller.compute_reconfiguration_cost([candidate_control_allocation,
                                                                                     base_control_allocation])
                self.controller.reconfiguration_cost.set(reconfiguration_cost, execution_id)

            # Apply candidate control signal
            self.controller._apply_control_allocation(candidate_control_allocation,
                                                                execution_id=execution_id,
                                                                runtime_params=runtime_params,
                                                                context=context)

            # Get control signal costs
            all_costs = self.controller.parameters.costs._get(execution_id) + [reconfiguration_cost]
            # Compute a total for the candidate control signal(s)
            total_cost = self.controller.combine_costs(all_costs)
        return total_cost

    def _build_predicted_inputs_dict(self, predicted_input):
        inputs = {}
        # ASSUMPTION: input_states[0] is NOT a feature and input_states[1:] are features
        # If this is not a good assumption, we need another way to look up the feature InputStates
        # of the OCM and know which InputState maps to which predicted_input value
        for j in range(len(self.controller.input_states) - 1):
            input_state = self.controller.input_states[j + 1]
            if hasattr(input_state, "shadow_inputs") and input_state.shadow_inputs is not None:
                inputs[input_state.shadow_inputs.owner] = predicted_input[j]

        return inputs

    def evaluate(
            self,
            predicted_input=None,
            control_allocation=None,
            num_simulation_trials=None,
            runtime_params=None,
            base_execution_id=None,
            execution_id=None,
            context=None,
            execution_mode=False,
    ):
        """Runs a simulation of the `Composition`, with the specified control_allocation, excluding its
           `controller <Composition.controller>` in order to return the
           `net_outcome <ControlMechanism.net_outcome>` of the Composition, according to its
           `controller <Composition.controller>` under that control_allocation. All values are
           reset to pre-simulation values at the end of the simulation. """
        # Apply candidate control to signal(s) for the upcoming simulation and determine its cost
        total_cost = self._get_total_cost_of_control_allocation(control_allocation, execution_id, runtime_params, context)

        # Build input dictionary for simulation
        inputs = self._build_predicted_inputs_dict(predicted_input)

        # Run Composition in "SIMULATION" context
        # MODIFIED 6/12/19 NEW: [JDC]
        if self._animate is not False and self._animate_simulations is not False:
            animate = self._animate
            buffer_animate_state = None
        else:
            animate = False
            buffer_animate_state = self._animate
        entry_execution_phase = self.parameters.context._get(execution_id).execution_phase
        # MODIFIED 6/12/19 END
        self.parameters.context._get(execution_id).execution_phase = ContextFlags.SIMULATION
        self.run(inputs=inputs,
                 execution_id=execution_id,
                 runtime_params=runtime_params,
                 num_trials=num_simulation_trials,
                 animate=animate,
                 context=context,
                 bin_execute=execution_mode,
                 skip_initialization=True,
                 )
        # # MODIFIED 6/12/19 OLD:
        # self.parameters.context._get(execution_id).execution_phase = ContextFlags.PROCESSING
        # MODIFIED 6/12/19 NEW: [JDC]
        self.parameters.context._get(execution_id).execution_phase = entry_execution_phase
        if buffer_animate_state:
            self._animate = buffer_animate_state
        # MODIFIED 6/12/19 END

        # Store simulation results on "base" composition
        if context.initialization_status != ContextFlags.INITIALIZING:
            try:
                self.parameters.simulation_results._get(base_execution_id).append(
                    self.get_output_values(execution_id))
            except AttributeError:
                self.parameters.simulation_results._set([self.get_output_values(execution_id)], base_execution_id)

        # Update input states in order to get correct value for "outcome" (from objective mech)
        self.controller._update_input_states(execution_id, runtime_params, context.flags_string)
        outcome = self.controller.input_state.parameters.value._get(execution_id)

        # Compute net outcome based on the cost of the simulated control allocation (usually, net = outcome - cost)
        net_outcome = self.controller.compute_net_outcome(outcome, total_cost)

        return net_outcome

    def disable_all_history(self):
        """
            When run, disables history tracking for all Parameters of all Components used in this Composition
        """
        self._set_all_parameter_properties_recursively(history_max_length=0)

    def _dict_summary(self):
        scheduler_dict = {
            'schedulers': {
                str(ContextFlags.PROCESSING): self.scheduler_processing._dict_summary()
            }
        }

        nodes_dict = {
            'nodes': {n.name: n._dict_summary() for n in self.nodes}
        }

        projections_dict = {
            'projections': {f'{p.sender.owner.name} to {p.receiver.owner.name}': p._dict_summary() for p in self.projections}
        }

        return {
            **super()._dict_summary(),
            **scheduler_dict,
            **nodes_dict,
            **projections_dict,
        }

    @property
    def input_states(self):
        """Returns all InputStates that belong to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_states

    @property
    def output_states(self):
        """Returns all OutputStates that belong to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_states

    @property
    def output_values(self):
        """Returns values of all OutputStates that belong to the Output CompositionInterfaceMechanism"""
        return self.get_output_values()

    def get_output_values(self, execution_context=None):
        return [output_state.parameters.value.get(execution_context) for output_state in self.output_CIM.output_states]

    @property
    def input_state(self):
        """Returns the index 0 InputState that belongs to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_states[0]

    @property
    def input_values(self):
        """Returns values of all InputStates that belong to the Input CompositionInterfaceMechanism"""
        return self.get_input_values()

    def get_input_values(self, execution_context=None):
        return [input_state.parameters.value.get(execution_context) for input_state in self.input_CIM.input_states]

    @property
    def runs_simulations(self):
        return True

    @property
    def simulation_results(self):
        return self.parameters.simulation_results._get(self.default_execution_id)

    #  For now, external_input_states == input_states and external_input_values == input_values
    #  They could be different in the future depending on new features (ex. if we introduce recurrent compositions)
    #  Useful to have this property for treating Compositions the same as Mechanisms in run & execute
    @property
    def external_input_states(self):
        """Returns all external InputStates that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_state for input_state in self.input_CIM.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_values(self):
        """Returns values of all external InputStates that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_state.value for input_state in self.input_CIM.input_states if not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_values(self):
        """Returns the default values of all external InputStates that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_state.defaults.value for input_state in self.input_CIM.input_states if
                    not input_state.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def stateful_nodes(self):
        """
        List of all nodes in the system that are currently marked as stateful. For Mechanisms, statefulness is
        determined by checking whether node.has_initializers is True. For Compositions, statefulness is determined
        by checking whether any of its nodes are stateful.

        Returns
        -------
        all stateful nodes in the system : List[Nodes]

        """

        stateful_nodes = []
        for node in self.nodes:
            if isinstance(node, Composition):
                if len(node.stateful_nodes) > 0:
                    stateful_nodes.append(node)
            elif node.has_initializers:
                stateful_nodes.append(node)

        return stateful_nodes

    @property
    def output_state(self):
        """Returns the index 0 OutputState that belongs to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_states[0]

    @property
    def class_parameters(self):
        return self.__class__.parameters

    @property
    def stateful_parameters(self):
        return [param for param in self.parameters if param.stateful]

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            super()._dependent_components,
            self.nodes,
            self.projections,
            [self.input_CIM, self.output_CIM],
            self.input_CIM.efferents,
            self.output_CIM.afferents,
            [self.controller] if self.controller is not None else []
        ))

    @property
    def learning_components(self):
        return [node for node in self.nodes if NodeRole.LEARNING in self.nodes_to_roles[node]]

    @property
    def afferents(self):
        return ContentAddressableList(component_type=Projection,
                                      list=[proj for proj in self.input_CIM.afferents])

    @property
    def efferents(self):
        return ContentAddressableList(component_type=Projection,
                                      list=[proj for proj in self.output_CIM.efferents])
