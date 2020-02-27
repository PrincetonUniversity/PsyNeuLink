# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Composition ************************************************************

"""

Contents
--------

  * `Composition_Overview`
  * `Composition_Creation`
      - `Composition_Nested`
  * `Composition_Run`
      - `Composition_Run_Static_Inputs`
      - `Composition_Run_Dynamic_Inputs`
      - `Composition_Scope_of_Execution`
  * `Composition_Controller`
      - `Composition_Controller_Assignment`
      - `Composition_Controller_Execution`
  * `Composition_Learning`
      - `Composition_Learning_Standard`
          • `Composition_Learning_Unsupervised`
          • `Composition_Learning_Unsupervised`
              - `Composition_Learning_Methods`
              - `Composition_Learning_Components`
              - `Composition_Learning_Execution`
      - `Composition_Learning_AutodiffComposition`
      - `Composition_Learning_UDF`
  * `Composition_Visualization`
  * `Composition_Class_Reference`


.. _Composition_Overview:

Overview
--------

.. warning:: As of PsyNeuLink 0.7.5, the API for using Compositions for Learning has been slightly changed! Please see `this link <RefactoredLearningGuide>` for more details!


Composition is the base class for objects that combine PsyNeuLink `Components <Component>` into an executable model.
It defines a common set of attributes possessed, and methods used by all Composition objects.

Composition "Nodes" are `Mechanisms <Mechanism>` and/or nested `Compositions <Composition>`. `Projections
<Projection>` connect two Nodes. The Composition's `graph <Composition.graph>` stores the structural relationships
among the Nodes of a Composition and the Projections that connect them.  The Composition's `scheduler
<Composition.scheduler>` generates an execution queue based on these structural dependencies, allowing
for other user-specified scheduling and termination conditions to be specified.

.. _Composition_Creation:

Creating a Composition
----------------------

A generic Composition can be created by calling the constructor, and then adding `Components <Component>` using the
following Composition methods:

    - `add_node <Composition.add_node>`

        adds a node to the Composition

    - `add_nodes <Composition.add_nodes>`

        adds mutiple nodes to the Composition

    - `add_projection <Composition.add_projection>`

        adds a connection between a pair of nodes in the Composition

    - `add_projections <Composition.add_projections>`

        adds connection between multiple pairs of nodes in the Composition

    - `add_linear_processing_pathway <Composition.add_linear_processing_pathway>`

        adds and connects a list of nodes and/or Projections to the Composition;
        Inserts a default Projection between any adjacent Nodes.

In addition, a Composition has the following set of `learning methods <Composition_Learning_Methods>` that can also
be used to create a Composition from (or add) pathways that implement `learning <Composition_Learning>`:

    - `add_linear_learning_pathway` <Composition.add_linear_learning_pathway>`

        adds and connects a list of nodes, including `learning components <Composition_Learning_Components>`
        needed to implement the algorithm specified in its **learning_function** argument in the specified pathway.

    - `add_reinforcement_learning_pathway <Composition.add_reinforcement_learning_pathway>`

        adds and connects a list of nodes, including `learning components <Composition_Learning_Components>`
        needed to implement `reinforcement learning` in the specified pathway;

    - `add_td_learning_pathway <Composition.add_td_learning_pathway>`

        adds and connects a list of nodes, including `learning components <Composition_Learning_Components>`
        needed to implement the `temporal differences` method of reinforcement learning` in the specified pathway;

    - `add_backpopagation_learning_pathway <Composition.add_backpopagation_learning_pathway>`

        adds and connects a list of nodes, including `learning components <Composition_Learning_Components>`
        needed to implement the `backpropagation learning algorithm` in the specified pathway.

.. note::
  Only Mechanisms and Projections added to a Composition via the methods above constitute a Composition, even if
  other Mechanism and/or Projections are constructed in the same script.

COMMENT:
• MOVE THE EXAPLES BELOW TO AN "Examples" SECTION
COMMENT
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

.. _Composition_Nested:

*Nested Compositions*
=====================

A Composition can be used as a node of another Composition, by calling `add_node <Composition.add_node>`
from the parent composition using the child Composition as an argument. Projections can then be specifed to and from
the nested composition just as for any other node.

    *Create outer Composition:*

    >>> outer_A = pnl.ProcessingMechanism(name='outer_A')
    >>> outer_B = pnl.ProcessingMechanism(name='outer_B')
    >>> outer_comp = pnl.Composition(name='outer_comp')
    >>> outer_comp.add_nodes([outer_A, outer_B])

    *Create and configure inner Composition:*

    >>> inner_A = pnl.ProcessingMechanism(name='inner_A')
    >>> inner_B = pnl.ProcessingMechanism(name='inner_B')
    >>> inner_comp = pnl.Composition(name='inner_comp')
    >>> inner_comp.add_linear_processing_pathway([inner_A, inner_B])

    *Nest inner Composition within outer Composition using `add_node <Composition.add_node>`:*

    >>> outer_comp.add_node(inner_comp)

    *Create Projections:*

    >>> outer_comp.add_projection(pnl.MappingProjection(), sender=outer_A, receiver=inner_comp)
    >>> outer_comp.add_projection(pnl.MappingProjection(), sender=inner_comp, receiver=outer_B)
    >>> input_dict = {outer_A: [[[1.0]]]}

    *Run Composition:*

    >>> outer_comp.run(inputs=input_dict)

    *Using `add_linear_processing_pathway <Composition.add_linear_processing_pathway>` with nested compositions for brevity:*

    >>> outer_A = pnl.ProcessingMechanism(name='outer_A')
    >>> outer_B = pnl.ProcessingMechanism(name='outer_B')
    >>> outer_comp = pnl.Composition(name='outer_comp')
    >>> inner_A = pnl.ProcessingMechanism(name='inner_A')
    >>> inner_B = pnl.ProcessingMechanism(name='inner_B')
    >>> inner_comp = pnl.Composition(name='inner_comp')
    >>> inner_comp.add_linear_processing_pathway([inner_A, inner_B])
    >>> outer_comp.add_linear_processing_pathway([outer_A, inner_comp, outer_B])
    >>> input_dict = {outer_A: [[[1.0]]]}
    >>> outer_comp.run(inputs=input_dict)

.. _Composition_Run:

Running a Composition
---------------------

.. _Composition_Run_Static_Inputs:

*Run with Input Dictionary*
============================

The `run <Composition.run>` method presents the inputs for each `TRIAL` to the input_ports of the INPUT Nodes in the
`scope of execution <Composition_Scope_of_Execution>`. These input values are specified in the **inputs** argument of
a Composition's `execute <Composition.execute>` or `run <Composition.run>` methods.

COMMENT:
    From KAM 2/7/19 - not sure "scope of execution" is the right phrase. To me, it implies that only a subset of the
    nodes in the Composition belong to the "scope of execution". What we want to convey (I think) is that ALL of the
    Nodes execute, but they do so in a "state" (history, parameter vals) corresponding to a particular execution id.
COMMENT

The standard way to specificy inputs is a Python dictionary in which each key is an `INPUT <NodeRole.INPUT>` Node and
each value is a list. The lists represent the inputs to the key `INPUT <NodeRole.INPUT>` Nodes, in which the i-th
element of the list represents the input value to the key Node on trial i.

.. _Composition_Run_Inputs_Fig_States:

.. figure:: _static/input_spec_states.svg
   :alt: Example input specifications with input ports


Each input value must be compatible with the shape of the key `INPUT <NodeRole.INPUT>` Node's `external_input_values
<MechanismBase.external_input_values>`. As a result, each item in the list of inputs is typically a 2d list/array,
though `some shorthand notations are allowed <Composition_Input_Specification_Examples>`.

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
    index i element is the value of the Node's index i `external_input_port <MechanismBase.external_input_ports>`. In
    many cases, `external_input_values <MechanismBase.external_input_values>` is the same as `variable
    <MechanismBase.variable>`. Keep in mind that any InputPorts marked as "internal" are excluded from
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

.. _Composition_Input_Specification_Examples:

For convenience, condensed versions of the input specification described above are also accepted in the following
situations:

* **Case 1: INPUT Node has only one InputPort**
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

Shorthand - drop the outer list on each input because **Mechanism a** only has one InputPort:

        >>> input_dictionary = {a: [[1.0], [2.0], [3.0], [4.0], [5.0]]}

        >>> comp.run(inputs=input_dictionary)

Shorthand - drop the remaining list on each input because **Mechanism a**'s one InputPort's value is length 1:

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

.. _Composition_Run_Dynamic_Inputs:

*Run with Function, Generator, or Generator Function*
===========================================================

Inputs can also be specified with a function, generator, or generator function.

A function used as input must take as its sole argument the current trial number and return a value that satisfies
all rules above for standard input specification. The only difference is that on each execution, the function must
return the input values for each INPUT Node for a single trial.

.. note::
    Default behavior when passing a function as input to a Composition is to execute for only one trial. Remember to
    set the num_trials argument of Composition.run if you intend to cycle through multiple trials.

Complete input specification:

::

        >>> import psyneulink as pnl

        >>> a = pnl.TransferMechanism(name='a',
        ...                           default_variable=[[1.0, 2.0, 3.0]])
        >>> b = pnl.TransferMechanism(name='b')

        >>> pathway1 = [a, b]

        >>> comp = pnl.Composition(name='comp')

        >>> comp.add_linear_processing_pathway(pathway1)

        >>> def function_as_input(trial_num):
        ...     a_inputs = [[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]
        ...     this_trials_inputs = {
        ...         a: a_inputs[trial_num]
        ...     }
        ...     return this_trials_inputs

        >>> comp.run(inputs=function_as_input,
        ...          num_trials=2)

..

A generator can also be used as input. On each yield, it should return a value that satisfies all rules above for
standard input specification. The only difference is that on each execution, the generator must yield the input values
for each INPUT Node for a single trial.

.. note::
    Default behavior when passing a generator is to execute until the generator is exhausted. If the num_trials
    argument of Composition.run is set, the Composition will execute EITHER until exhaustion, or until num_trials has
    been reached - whichever comes first.

Complete input specification:

::

    >>> import psyneulink as pnl

    >>> a = pnl.TransferMechanism(name='a',default_variable = [[1.0, 2.0, 3.0]])
    >>> b = pnl.TransferMechanism(name='b')

    >>> pathway1 = [a, b]

    >>> comp = pnl.Composition(name='comp')

    >>> comp.add_linear_processing_pathway(pathway1)

    >>> def generator_as_input():
    ...    a_inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    ...    for i in range(len(a_inputs)):
    ...        this_trials_inputs = {a: a_inputs[i]}
    ...        yield this_trials_inputs

    >>> generator_instance = generator_as_input()

    >>> # Because the num_trials argument is set to 2, the below call to run will result in only 2 executions of
    ... # comp, even though it would take three executions to exhaust the generator.
    >>> comp.run(inputs=generator_instance,
    ...          num_trials=2)

..

If a generator function is used, the Composition will instantiate the generator and use that as its input. Thus,
the returned generator instance of a generator function must follow the same rules as a generator instance passed
directly to the Composition.

Complete input specification:

::

    >>> import psyneulink as pnl

    >>> a = pnl.TransferMechanism(name='a',default_variable = [[1.0, 2.0, 3.0]])
    >>> b = pnl.TransferMechanism(name='b')

    >>> pathway1 = [a, b]

    >>> comp = pnl.Composition(name='comp')

    >>> comp.add_linear_processing_pathway(pathway1)

    >>> def generator_function_as_input():
    ...    a_inputs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ...    for i in range(len(a_inputs)):
    ...        this_trials_inputs = {a: a_inputs[i]}
    ...        yield this_trials_inputs

    >>> comp.run(inputs=generator_function_as_input)

..

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

COMMENT:
.. _Composition_Initial_Values_and_Feedback
FIX:  ADD SECTION ON CYCLES, FEEDBACK, INITIAL VALUES, RELEVANCE TO MODULATORY MECHANISMS REINITIALIZATION
MODIFIED FROM SYSTEM (_System_Execution_Input_And_Initialization):
..[another type] of input can be provided in corresponding arguments of the `run <System.run>` method:
a list or ndarray of **initial_values**[...] The **initial_values** are
assigned at the start of a `TRIAL` as input to Nodes that close recurrent loops (designated as `FEEDBACK_SENDER`,
and listed in the Composition's ?? attribute),
COMMENT

.. _Composition_Scope_of_Execution:

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
        >>> c.run({t: 20}, context='custom execution id')
        [[array([20.])]]

        # context None
        >>> print(t.parameters.value.get())
        [[1.]]
        >>> print(t.parameters.value.get(c))
        [[5.]]
        >>> print(t.parameters.value.get(d))
        [[10.]]
        >>> print(t.parameters.value.get('custom execution id'))
        [[20.]]Composition_Controller

In general, anything that happens outside of a Composition run and without an explicit setting of execution context
occurs in the `None` execution context.

.. _Composition_Controller:

Controlling a Composition
-------------------------

A Composition can be assigned a `controller <Composition.controller>`.  This is a `ControlMechanism`, or a subclass
of one, that modulates the parameters of Components within the Composition (including Components of nested Compositions).
It typically does this based on the output of an `ObjectiveMechanism` that evaluates the value of other Mechanisms in
the Composition, and provides the result to the `controller <Composition.controller>`.

.. _Composition_Controller_Assignment:

Assigning a Controller
======================

A `controller <Composition.controller>` can be assigned either by specifying it in the **controller** argument of the
Composition's constructor, or using its `add_controller <Composition.add_controller>` method.

COMMENT:
TBI FOR COMPOSITION
Specyfing Parameters to Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  |
  * **MonitoredOutputPortsOption** -- must be a value of `MonitoredOutputPortsOption`, and must appear alone or as a
    single item in the list specifying the **monitor_for_control** argument;  any other specification(s) included in
    the list will take precedence.  The MonitoredOutputPortsOption applies to all of the Mechanisms in the System
    except its `controller <System.controller>` and `LearningMechanisms <LearningMechanism>`. The
    *PRIMARY_OUTPUT_PORTS* value specifies that the `primary OutputPort <OutputPort_Primary>` of every Mechanism be
    monitored, whereas *ALL_OUTPUT_PORTS* specifies that *every* OutputPort of every Mechanism be monitored.
  |
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


COMMENT:
For Developers
--------------

.. _Composition_Execution_Contexts_Init:

Initialization of Execution Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- The parameter values for any execution context can be copied into another execution context by using \
Component._initialize_from_context, which when called on a Component copies the values for all its parameters \
and recursively for all of the Component's `_dependent_components <Component._dependent_components>`

- `_dependent_components <Component._dependent_components>` should be added to for any new Component that requires \
other Components to function properly (beyond "standard" things like Component.function, \
or Mechanism.input_ports, as these are added in the proper classes' _dependent_components)
    - the intent is that with ``_dependent_components`` set properly, calling \
    ``obj._initialize_from_context(new_context, base_context)`` should be sufficient to run obj \
    under **new_context**
    - a good example of a "nonstandard" override is `OptimizationControlMechanism._dependent_components`

.. _Composition_TIming:

*Timing*
========

When `run <Composition.run>` is called by a Composition, it calls that Composition's `execute <Composition.execute>`
method once for each `input <Composition_Run_Inputs>`  (or set of inputs) specified in the call to `run
<Composition.run>`, which constitutes a `TRIAL` of execution.  For each `TRIAL`, the Component makes repeated calls
to its `scheduler <Composition.scheduler>`, executing the Components it specifies in each
`TIME_STEP`, until every Component has been executed at least once or another `termination condition
<Scheduler_Termination_Conditions>` is met.  The `scheduler <Composition.scheduler>` can be
used in combination with `Condition` specifications for individual Components to execute different Components at
different time scales.

Runtime Params
COMMENT

.. _Composition_Learning:

Learning in a Composition
-------------------------
* `Composition_Learning_Standard`
* `Composition_Learning_AutodiffComposition`
* `Composition_Learning_UDF`

Learning is used to modify the `Projections <Projection>` between Mechanisms in a Composition.  More specifically,
it modifies the `matrix <MappingProjection.matrix>` parameter of those `MappingProjections <MappingProjection>`,
which implements the strengths ("weights") of the associations between representations in the Mechanisms they connect.

.. _Composition_Learning_Mode:

*Running a Composition in Learning Mode*
======================================
A Composition only learns when ran in learning mode, and when its `disable_learning` parameter is False. To run the Composition in learning mode, use the `learn <Composition.learn>` method.
See `learn <Composition.learn>` for more details.

*Implementing Learning in a Composition*
======================================
There are three ways of implementing learning in a Composition:

i) using `standard PsyNeuLink Components <Composition_Learning_Standard>`

ii) using the `AutodiffComposition <Composition_Learning_AutodiffComposition>` -- a specialized subclass of Composition that executes learning using `PyTorch <https://pytorch.org>`_

iii) using `UserDefinedFunctions <UserDefinedFunction>`.

The advantage of using standard PsyNeuLink compoments is that it
assigns each operation involved in learning to a dedicated Component. This helps make clear exactly what those
operations are, the sequence in which they are carried out, and how they interact with one another.  However,
this can also make execution inefficient, due to the "overhead" incurred by distributing the calculations over
different Components.  If more efficient computation is critical, then the `AutodiffComposition` can be used to
execute a compatible PsyNeuLink Composition in PyTorch, or one or more `UserDefinedFunctions <UserDefinedFunction>`
can be assigned to either PyTorch functions or those in any other Python environment that implements learning and
accepts and returns tensors. Each of these approaches is described in more detail below.

.. _Composition_Learning_Standard:

*Learning Using PsyNeuLink Components*
======================================

* `Composition_Learning_Unsupervised`
* `Composition_Learning_Supervised`

When learning is `implemented using standard PsyNeuLink Components <Composition_Learning_Standard>`, each calculation
and/or operation involved in learning -- including those responsible for computing errors, and for using those to
modify the Projections between Mechanisms, is assigned to a different PsyNeuLink `learning-related Component
<Composition_Learning_Components>`.  These can be used to implement any form of learning.  Learning is generally
considered to fall into two broad classes:  *unsupervised*, in which associative strenghts are modified
by mere exposure to the inputs, in order to capture structure and/or relationships among them;  and *supervised*,
which in which the associative strengths are modified so that each input generates a desired output (see
`<https://www.geeksforgeeks.org/supervised-unsupervised-learning/>`_ for a useful summary).  Both forms of
learning can be implemented in a Composition, using `LearningMechanisms <LearningMechanism>` that compute the
changes to make to the `matrix <MappingProjection.matrix>` parameter of `MappingProjections <MappingProjection>`
being learned, and `LearningProjections <LearningProjection>` that apply those changes to the MappingProjections).
In addition, supervised learning uses a `ComparatorMechanism` to compute the error between the response generated by
the Composition to the input stimulus, and the target stimulus used to designate the desired response.  In most
cases, the LearningMechanisms, LearningProjections and, where needed, ComparatorMechanism are generated automatically,
as described for each form of learning below.  However, these can also be configured manually using their constructors,
or modified by assigning values to their attributes.

.. _Composition_Learning_Unsupervised:

Unsupervised Learning
~~~~~~~~~~~~~~~~~~~~~

Undersupervised learning is implemented using a `RecurrentTransferMechanism`, setting its **enable_learning** argument
to True, and specifying the desired `LearningFunction <LearningFunctions>` in its **learning_function** argument.  The
default is `Hebbian`, however others can be specified (such as `ContrastiveHebbian` or `Kohonen`). When a
RecurrentTransferMechanism with learning enabled is added to a Composition, an `AutoAssociativeLearningMechanism` that
that is appropriate for the specified learning_function is automatically constructured and added to the Composition,
as is a `LearningProjection` from the AutoAssociativeLearningMechanism to the RecurrentTransferMechanism's
`recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.  When the Composition is run and the
RecurrentTransferMechanism is executed, its AutoAssociativeLearningMechanism is also executed, which updates the `matrix
<AutoAssociativeProjection.matrix>` of its `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
in response to its input.

COMMENT:
• DISCUSS LEARNING COMPONENTS RETURNED ONCE add_node AND add_linear_processing_pathway RETURN THEM
• ADD EXAMPLE HERE
COMMENT

.. _Composition_Learning_Supervised:

Supervised Learning
~~~~~~~~~~~~~~~~~~~

* `Composition_Learning_Methods`
* `Composition_Learning_Components`
* `Compositon_Learning_Execution`

COMMENT:
TBI:  Supervised learning is implemented using a Composition's `add_learning_pathway` method, and specifying an
appropriate `LearningFunction <LearningFunctions>` in its **learning_function** argument.
XXXMORE HERE ABOUT TYPES OF FUNCTIONS
• MODIFY REFERENCE TO LEARNING COMPONENT NAMES WHEN THEY ARE IMPLEMENTED AS AN ENUM CLASS
• ADD EXAMPLES - POINT TO ONES IN BasicsAndPrimer
COMMENT

.. _Composition_Learning_Methods:

*Learning Methods*
^^^^^^^^^^^^^^^^^^

Supervised learning is implemented using a Composition's method for the desired type of learning.  There are currently
three such methods:

    • `add_linear_learning_pathway`
    • `add_reinforcement_learning_pathway`
    • `add_td_learning_pathway`
    • `add_backpropagation_learning_pathway`.

Each uses the Composition's `add_linear_processing_pathway` method to create a  *learning sequence* specified in their
**pathway** argument: a contiguous sequence of `ProcessingMechanisms <ProcessingMechanism>` and the `MappingProjections
<MappingProjection>` between them, in which learning modifies the `matrix <MappingProjection.matrix>` parameter of the
MappingProjections in the sequence, so that the input to the first ProcessingMechanism in the sequence generates an
output from the last ProcessingMechanism that matches as closely as possible the value specified for the `target
mechanism <Process_Learning_Components>` in the **inputs** argument of the Composition's `run <Composition.run>` method.
The Mechanisms in the pathway must be compatible with learning (that is, their `function <Mechanism_Base.function>` must
be compatible with the `function <LearningMechanism.function>` of the `LearningMechanism` for the MappingProjections
they receive (see `LearningMechanism_Function`).  The Composition's `learning methods <Composition_Learning_Methods>`
return the set of learning components generates for the pathway, as described below.

.. _Composition_Learning_Components:

*Learning Components*
^^^^^^^^^^^^^^^^^^^^^

For each learning sequence specified in a `learning method <Composition_Learning_Methods>`, it creates the
following Components, and assigns to them the `NodeRoles <NodeRole>` indicated:

    .. _COMPARATOR_MECHANISM:
    * *COMPARATOR_MECHANISM* `ComparatorMechanism` -- used to `calculate an error signal
      <ComparatorMechanism_Execution>` for the sequence by comparing the value received by the ComparatorMechanism's
      *SAMPLE* `InputPort <ComparatorMechanism_Structure>` (from the `output <LearningMechanism_Activation_Output>` of
      the last Processing Mechanism in the learning sequence) with the value received in the *COMPARATOR_MECHANISM*'s
      *TARGET* `InputPort <ComparatorMechanism_Structure>` (from the *TARGET_MECHANISM* generated by the method --
      see below); this is assigned the `NodeRole` `LEARNING` in the Composition.
    ..
    .. _TARGET_MECHANISM:
    * *TARGET_MECHANISM* -- receives the value to be used by the *COMPARATOR_MECHANISM* as the target in
      computing the error signal (see above);  that value must be specified in the **inputs** argument of the
      Composition's `run <Composition.run>` method (as the input to the *TARGET_MECHANISM*; this is assigned the
      `NodeRoles <NodeRole>` `TARGET` and `LEARNING` in the Composition;
    ..
    * a MappingProjection that projects from the last ProcessingMechanism in the learning sequence to the *SAMPLE*
      `InputPort  <ComparatorMechanism_Structure>` of the *COMPARATOR_MECHANISM*;
    ..
    * a MappingProjection that projects from the *TARGET_MECHANISM* to the *TARGET* `InputPort
      <ComparatorMechanism_Structure>` of the *COMPARATOR_MECHANISM*;
    ..
    .. _LEARNING_MECHANISM:
    * a *LEARNING_MECHANISM* for each MappingProjection in the sequence, each of which calculates the `learning_signal
      <LearningMechanism.learning_signal>` used to modify the `matrix <MappingProjection.matrix>` parameter for the
      coresponding MappingProjection, along with a `LearningSignal` and `LearningProjection` that convey the
      `learning_signal <LearningMechanism.learning_signal>` to the MappingProjection's *MATRIX* `ParameterPort
      <Mapping_Matrix_ParameterPort>`;  depending on learning method, additional MappingProjections may be created to
      and/or from the LearningMechanism -- see `LearningMechanism_Learning_Configurations` for details);
      these are assigned the `NodeRole` `LEARNING` in the Composition.

The items with names in the list above are returned by the learning method in a dictionary, in which each name is the
key of an entry, and the object(s) created of that type are its value.  See `LearningMechanism_Single_Layer_Learning`
for a more detailed description and figure showing these Components.

If the learning sequence involves more than two ProcessingMechanisms (e.g. using `add_backpropagation_learning_pathway`
for a multilayered neural network), then additional LearningMechanisms are created, along with MappingProjections
that provides them with the `error_signal <LearningMechanism.error_signal>` from the preceding LearningMechanism,
and `LearningProjections <LearningProjection>` that modify the additional MappingProjections (*LEARNED_PROJECTION*\\s)
in the sequence, as shown for an example in the figure below.  These additional learning components are listed in the
*LEARNING_MECHANISM* and *LEARNED_PROJECTION* entries of the dictionary returned by the learning method.

.. _Composition_MultilayerLearning_Figure:

**Learning Components**

.. figure:: _static/Composition_Multilayer_Learning_fig.svg
   :alt: Schematic of LearningMechanism and LearningProjections in a Process
   :scale: 50 %

   Components for sequence of three Mechanisms generated by a call to a learning method (e.g.,
   ``add_backpropagation_learning_pathway(pathway=[A,B,C])``), with `NodeRole` assigned to each node in the
   Composition's `graph <Composition.graph>` (in italics below Mechanism type) and the names of the learning
   components (capitalized in italics) returned by the learning method.

.. _Composition_XOR_Example:

The following example implements a simple three-layered network that learns the XOR function
(see `figure <Composition_Learning_Output_vs_Terminal_Figure>` below)::

    # Construct Composition:
    >>> input = TransferMechanism(name='Input', default_variable=np.zeros(2))
    >>> hidden = TransferMechanism(name='Hidden', default_variable=np.zeros(10), function=Logistic())
    >>> output = TransferMechanism(name='Output', default_variable=np.zeros(1), function=Logistic())
    >>> input_weights = MappingProjection(name='Input Weights', matrix=np.random.rand(2,10))
    >>> output_weights = MappingProjection(name='Output Weights', matrix=np.random.rand(10,1))
    >>> xor_comp = Composition('XOR Composition')
    >>> learning_components = xor_comp.add_backpropagation_learning_pathway(
    >>>                       pathway=[input, input_weights, hidden, output_weights, output])
    >>> target = learning_components[TARGET_MECHANISM]

    # Create inputs:            Trial 1  Trial 2  Trial 3  Trial 4
    >>> xor_inputs = {'stimuli':[[0, 0],  [0, 1],  [1, 0],  [1, 1]],
    >>>               'targets':[  [0],     [1],     [1],     [0] ]}
    >>> xor_comp.learn(inputs={input:xor_inputs['stimuli'],
    >>>                      target:xor_inputs['targets']},
    >>>              num_trials=1,
    >>>              animate={'show_learning':True})

The description and example above pertain to simple linear sequences.  However, more complex configurations,
with convergent, divergent and/or intersecting sequences can be built using multiple calls to the learning method
(see `example <BasicsAndPrimer_Rumelhart_Model>` in `BasicsAndPrimer`).  In each call, the learning method determines
how the sequence to be added relates to any existing ones with which it abuts or intersects, and automatically creates
andconfigures the relevant learning components so that the error terms are properly computed and propagated by each
LearningMechanism to the next in the configuration. It is important to note that, in doing so, the status of a
Mechanism in the final configuration takes precedence over its status in any of the individual sequences specified
in the `learning methods <Composition_Learning_Methods>` when building the Composition.  In particular,
whereas ordinarily the last ProcessingMechanism of a sequence specified in a learning method projects to a
*COMPARATOR_MECHANISM*, this may be superceded if multiple sequences are created. This is the case if: i) the
Mechanism is in a seqence that is contiguous (i.e., abuts or intersects) with others already in the Composition,
ii) the Mechanism appears in any of those other sequences and, iii) it is not the last Mechanism in *all* of them;
in that in that case, it will not project to a *COMPARATOR_MECHANISM* (see `figure below
<Composition_Learning_Output_vs_Terminal_Figure>` for an example).  Furthermore, if it *is* the last Mechanism in all of
them (that is, all of the specified pathways converge on that Mechanism), only one *COMPARATOR_MECHANISM* is created
for that Mechanism (i.e., not one for each sequence).  Finally, it should be noted that, by default, learning components
are *not* assigned the `NodeRole` of `OUTPUT` even though they may be the `TERMINAL` Mechanism of a Composition;
conversely, even though the last Mechanism of a learning sequence projects to a *COMPARATOR_MECHANISM*, and thus is not
the `TERMINAL` node of a Composition, if it does not project to any other Mechanisms in the Composition it is
nevertheless assigned as an `OUTPUT` of the Composition.  That is, Mechanisms that would otherwise have been the
`TERMINAL` Mechanism of a Composition preserve their role as an `OUTPUT` of the Composition if they are part of a
learning sequence even though they project to another Mechanism (the *COMPARATOR_MECHANISM*) in the Composition.

.. _Composition_Learning_Output_vs_Terminal_Figure:

    **OUTPUT** vs. **TERMINAL** Roles in Learning Configuration

    .. figure:: _static/Composition_Learning_OUTPUT_vs_TERMINAL_fig.svg
       :alt: Schematic of Mechanisms and Projections involved in learning
       :scale: 50 %

       Configuration of Components generated by the creation of two intersecting learning sequences
       (e.g., ``add_backpropagation_learning_pathway(pathway=[A,B])`` and
       ``add_backpropagation_learning_pathway(pathway=[D,B,C])``).  Mechanism B is the last Mechanism of the
       sequence specified for the first pathway, and so would project to a `ComparatorMechanism`, and would be
       assigned as an `OUTPUT` node of the Composition, if that pathway was created on its own.  However,
       since Mechanims B is also in the middle of the sequence specified for the second pathway, it does not
       project to a ComparatorMechanism, and is relegated to being an `INTERNAL` node of the Composition
       Mechanism C is now the one that projects to the ComparatorMechanism and assigned as the `OUTPUT` node.

.. _Composition_Learning_Execution:

*Execution of Learning*
^^^^^^^^^^^^^^^^^^^^^^^
When a Composition is run that contains one or more learning sequences, all of the ProcessingMechanisms for a
sequence are executed first, and then its LearningComponents. This is shown in an animation of the XOR network
from the `example above <Composition_XOR_Example>`:

.. _Composition_Learning_Animation_Figure:

    **Composition with Learning**

    .. figure:: _static/Composition_XOR_animation.gif
       :alt: Animation of Composition with learning
       :scale: 50 %

       Animation of XOR Composition in example above when it is executed by calling its `learn <Composition.learn>`
       method with the argument ``animate={'show_learning':True}``.

Note that, since the `learning components <Composition_Learning_Components>` are not executed until after the
processing components, the change to the weights of the MappingProjections in the processing pathway are not
made until after it has executed.  Thus, as with `execution of a Projection <Projection_Execution>`, those
changes will not be observed in the values of their `matrix <MappingProjection.matrix>` parameters until after
they are next executed (see :ref:`Lazy Evaluation <LINK>` for an explanation of "lazy" updating).

.. _Composition_Learning_AutodiffComposition:

*Learning Using AutodiffCompositon*
===================================

COMMENT:
Change reference to example below to point to Rumelhart Semantic Network Model Script once implemented
COMMENT

`AutodiffCompositions <AutodiffComposition>` provide the ability to execute a composition using `PyTorch
<https://pytorch.org>`_ (see `example <BasicsAndPrimer_Rumelhart_Model>` in `BasicsAndPrimer`).  The
AutodiffComposition constructor provides arguments for configuring the PyTorch implementation in various ways; the
Composition is then built using the same methods (e.g., `add_node`, `add_projection`, `add_linear_processing_pathway`,
etc.) as any other Composition. Note that there is no need to use any `learning methods <Composition_Learning_Methods>`
— AutodiffCompositions automatically creates backpropagation learning pathways between all input - output node paths.
It can be run just as a standard Composition would - using `learn <AutodiffComposition.learn>` for learning mode, and
`run <AutodiffComposition.run>` for test mode.

The advantage of this approach is that it allows the Composition to be implemented in PsyNeuLink, while exploiting
the efficiency of execution in PyTorch (which can yield as much as three orders of magnitude improvement).  However,
a disadvantage is that there are restrictions on the kinds of Compositions that be implemented in this way.
First, because it relies on PyTorch, it is best suited for use with `supervised
learning <Composition_Learning_Supervised>`, although it can be used for some forms of `unsupervised learning
<Composition_Learning_Unsupervised>` that are supported in PyTorch (e.g., `self-organized maps
<https://github.com/giannisnik/som>`_).  Second, all of the Components in the Composition are be subject to and must
be with compatible with learning.   This means that it cannot be used with a Composition that contains any
`modulatory components <ModulatorySignal_Anatomy_Figure>` or that are subject to modulation, whether by
ControlMechanisms within or outside the Composition;  this includes a `controller <Composition_Controller>`
or any LearningMechanisms.  An AutodiffComposition can be `nested in a Composition <Composition_Nested>`
that has such other Components.  During learning, none of the internal Components of the AutodiffComposition (e.g.,
intermediate layers of a neural network model) are accessible to the other Components of the outer Composition,
(e.g., as sources of information, or for modulation).  However, when learning turned off, then the  AutodiffComposition
functions like any other, and all of its internal  Components accessible to other Components of the outer Composition.
Thus, as long as access to its internal Components is not needed during learning, an `AutodiffComposition` can be
trained, and then used to execute the trained Composition like any other.

.. _Composition_Learning_UDF:

*Learning Using UserDefinedFunctions*
=====================================

If execution efficiency is critical and the `AutodiffComposition` is too restrictive, a function from any Python
environment that supports learning can be assigned as the `function <Mechanism_Base.function>` of a `Mechanism
<Mechanism>`, in which case it is automatically  wrapped as `UserDefinedFunction`.  For example, the `forward and
backward methods <https://pytorch.org/docs/master/notes/extending.html>`_ of a PyTorch object can be assigned in this
way.  The advanatage of this approach is that it can be applied to any Python function that adheres to the requirements
of a `UserDefinedFunction`.  The disadvantage is that it can't be `compiled`, so efficiency may be compromised.  It must
also be carefully coordinated with the execution of other learning-related Components in the Composition, to insure
that each function is called at the appropriate times during execution.  Furthermore, as with an `AutodiffComposition`,
the internal constituents of the object (e.g., intermediates layers of a neural network model) are not accessible to
other Components in the Composition (e.g., as a source of information or for modulation).

.. _Composition_Visualization:

Visualizing a Composition
-------------------------

COMMENT:
XXX - ADD EXAMPLE OF NESTED COMPOSITION
XXX - ADD DISCUSSION OF show_controller AND show_learning
COMMENT

The `show_graph <Composition.show_graph>` method generates a display of the graph structure of Nodes (Mechanisms and
Nested Compositions) and Projections in the Composition (based on the Composition's `processing graph
<Composition.processing_graph>`).

By default, Nodes are shown as ovals labeled by their `names <Mechanism.name>`, with the Composition's `INPUT
<NodeRole.INPUT>` Mechanisms shown in green, its `OUTPUT <NodeRole.OUTPUT>` Mechanisms shown in red, and Projections
shown as unlabeled arrows, as illustrated for the Composition in the example below:

.. _Composition_show_graph_basic_figure:

+-----------------------------------------------------------+----------------------------------------------------------+
| >>> from psyneulink import *                              | .. figure:: _static/Composition_show_graph_basic_fig.svg |
| >>> a = ProcessingMechanism(                              |                                                          |
|               name='A',                                   |                                                          |
| ...           size=3,                                     |                                                          |
| ...           output_ports=[RESULT, MEAN]                 |                                                          |
| ...           )                                           |                                                          |
| >>> b = ProcessingMechanism(                              |                                                          |
| ...           name='B',                                   |                                                          |
| ...           size=5                                      |                                                          |
| ...           )                                           |                                                          |
| >>> c = ProcessingMechanism(                              |                                                          |
| ...           name='C',                                   |                                                          |
| ...           size=2,                                     |                                                          |
| ...           function=Logistic(gain=pnl.CONTROL)         |                                                          |
| ...           )                                           |                                                          |
| >>> comp = Composition(                                   |                                                          |
| ...           name='Comp',                                |                                                          |
| ...           enable_controller=True                      |                                                          |
| ...           )                                           |                                                          |
| >>> comp.add_linear_processing_pathway([a,c])             |                                                          |
| >>> comp.add_linear_processing_pathway([b,c])             |                                                          |
| >>> ctlr = OptimizationControlMechanism(                  |                                                          |
| ...            name='Controller',                         |                                                          |
| ...            monitor_for_control=[(pnl.MEAN, a)],       |                                                          |
| ...            control_signals=(GAIN, c),                 |                                                          |
| ...            agent_rep=comp                             |                                                          |
| ...            )                                          |                                                          |
| >>> comp.add_controller(ctlr)                             |                                                          |
+-----------------------------------------------------------+----------------------------------------------------------+

Note that the Composition's `controller <Composition.controller>` is not shown by default.  However this
can be shown, along with other information, using options in the Composition's `show_graph <Composition.show_graph>`
method.  The figure below shows several examples.

.. _Composition_show_graph_options_figure:

**Output of show_graph using different options**

.. figure:: _static/Composition_show_graph_options_fig.svg
   :alt: Composition graph examples
   :scale: 150 %

   Displays of the Composition in the `example above <Composition_show_graph_basic_figure>`, generated using various
   options of its `show_graph <Composition.show_graph>` method. **Panel A** shows the graph with its Projections labeled
   and Component dimensions displayed.  **Panel B** shows the `controller <Composition.controller>` for the
   Composition and its associated `ObjectiveMechanism` using the **show_controller** option (controller-related
   Components are displayed in blue by default).  **Panel C** adds the Composition's `CompositionInterfaceMechanisms
   <CompositionInterfaceMechanism>` using the **show_cim** option. **Panel D** shows a detailed view of the Mechanisms
   using the **show_node_structure** option, that includes their `Ports <Port>` and their `roles <NodeRole>` in the
   Composition. **Panel E** shows an even more detailed view using **show_node_structure** as well as **show_cim**.

If a Composition has one ore more Compositions nested as Nodes within it, these can be shown using the
**show_nested** option. For example, the pathway in the script below contains a sequence of Mechanisms
and nested Compositions in an outer Composition, ``comp``:

.. _Composition_show_graph_show_nested_figure:

+------------------------------------------------------+---------------------------------------------------------------+
| >>> mech_stim = ProcessingMechanism(name='STIMULUS') |.. figure:: _static/Composition_show_graph_show_nested_fig.svg |
| >>> mech_A1 = ProcessingMechanism(name='A1')         |                                                               |
| >>> mech_B1 = ProcessingMechanism(name='B1')         |                                                               |
| >>> comp1 = Composition(name='comp1')                |                                                               |
| >>> comp1.add_linear_processing_pathway([mech_A1,    |                                                               |
| ...                                      mech_B1])   |                                                               |
| >>> mech_A2 = ProcessingMechanism(name='A2')         |                                                               |
| >>> mech_B2 = ProcessingMechanism(name='B2')         |                                                               |
| >>> comp2 = Composition(name='comp2')                |                                                               |
| >>> comp2.add_linear_processing_pathway([mech_A2,    |                                                               |
| ...                                      mech_B2])   |                                                               |
| >>> mech_resp = ProcessingMechanism(name='RESPONSE') |                                                               |
| >>> comp = Composition()                             |                                                               |
| >>> comp.add_linear_processing_pathway([mech_stim,   |                                                               |
| ...                                     comp1, comp2,|                                                               |
| ...                                     mech_resp])  |                                                               |
| >>> comp.show_graph(show_nested=True)                |                                                               |
+------------------------------------------------------+---------------------------------------------------------------+

.. _Composition_Class_Reference:

Class Reference
---------------

"""

import collections
import inspect
import itertools
import logging
import warnings
import sys

import numpy as np
import typecheck as tc

from PIL import Image
from copy import deepcopy, copy
from inspect import isgenerator, isgeneratorfunction

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import Component, ComponentsMeta
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.interfacefunctions import InterfacePortMap
from psyneulink.core.components.functions.learningfunctions import \
    LearningFunction, Reinforcement, BackPropagation, TDLearning
from psyneulink.core.components.functions.combinationfunctions import LinearCombination, PredictionErrorDeltaFunction
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import \
    OptimizationControlMechanism
from psyneulink.core.components.mechanisms.modulatory.learning.learningmechanism import \
    LearningMechanism, ACTIVATION_INPUT_INDEX, ACTIVATION_OUTPUT_INDEX, ERROR_SIGNAL, ERROR_SIGNAL_INDEX
from psyneulink.core.components.mechanisms.processing.compositioninterfacemechanism import CompositionInterfaceMechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.components.mechanisms.modulatory.control.optimizationcontrolmechanism import AGENT_REP
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import RecurrentTransferMechanism
from psyneulink.core.components.projections.projection import DuplicateProjectionError
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.modulatory.modulatoryprojection import ModulatoryProjection_Base
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.core.components.projections.modulatory.learningprojection import LearningProjection
from psyneulink.core.components.shellclasses import Composition_Base
from psyneulink.core.components.shellclasses import Mechanism, Projection
from psyneulink.core.components.ports.port import Port
from psyneulink.core.components.ports.inputport import InputPort, SHADOW_INPUTS
from psyneulink.core.components.ports.parameterport import ParameterPort
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.globals.context import Context, ContextFlags, handle_external_context
from psyneulink.core.globals.keywords import \
    AFTER, ALL, BEFORE, BOLD, BOTH, COMPARATOR_MECHANISM, COMPONENT, COMPOSITION, CONDITIONS, \
    CONTROL, CONTROLLER, CONTROL_SIGNAL, FUNCTIONS, HARD_CLAMP, IDENTITY_MATRIX, INPUT, \
    LABELS, LEARNED_PROJECTION, LEARNING_MECHANISM, MATRIX, MATRIX_KEYWORD_VALUES, MAYBE, MECHANISM, MECHANISMS, \
    MODEL_SPEC_ID_COMPOSITION, MODEL_SPEC_ID_NODES, MODEL_SPEC_ID_PROJECTIONS, MODEL_SPEC_ID_PSYNEULINK, \
    MODEL_SPEC_ID_RECEIVER_MECH, MODEL_SPEC_ID_SENDER_MECH, MONITOR, MONITOR_FOR_CONTROL, MSE, NAME, NO_CLAMP, \
    ONLINE, OUTCOME, OUTPUT, OWNER_VALUE, PATHWAY, PROJECTION, PROJECTIONS, PULSE_CLAMP, ROLES, \
    SAMPLE, SIMULATIONS, SOFT_CLAMP, SSE, TARGET, TARGET_MECHANISM, VALUES, VARIABLE, WEIGHT
from psyneulink.core.globals.log import CompositionLog, LogCondition
from psyneulink.core.globals.parameters import Parameter, ParametersBase
from psyneulink.core.globals.registry import register_category
from psyneulink.core.globals.utilities import ContentAddressableList, NodeRole, call_with_pruned_args, convert_to_list
from psyneulink.core.scheduling.condition import All, Always, Condition, EveryNCalls, Never
from psyneulink.core.scheduling.scheduler import Scheduler
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel, PreferenceSet, _assign_prefs
from psyneulink.core.globals.preferences.basepreferenceset import BasePreferenceSet
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection
from psyneulink.library.components.mechanisms.processing.objective.comparatormechanism import ComparatorMechanism, MSE
from psyneulink.library.components.mechanisms.processing.objective.predictionerrormechanism import \
    PredictionErrorMechanism

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
        A Graph of vertices and edges.

        Attributes
        ----------

        comp_to_vertex : Dict[`Component <Component>` : `Vertex`]
            maps `Component` in the graph to the `Vertices <Vertex>` that represent them.

        vertices : List[Vertex]
            the `Vertices <Vertex>` contained in this Graph.

        dependency_dict : Dict[`Component` : Set(`Compnent`)]
            maps each Component to those from which it receives Projections

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

            # FIX 8/12/19:  MODIFIED FEEDBACK -
            #  IS THIS A CORRECT DESCRIPTION? (SAME AS get_forward_parents_from_component)
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
            # FIX 8/12/19:  MODIFIED FEEDBACK -
            #  IS THIS A CORRECT DESCRIPTION? (SAME AS get_forward_children_from_component)
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

    @property
    def dependency_dict(self):
        return dict((v.component,set(d.component for d in v.parents)) for v in self.vertices)

# Options for show_node_structure argument of show_graph()
MECH_FUNCTION_PARAMS = "MECHANISM_FUNCTION_PARAMS"
STATE_FUNCTION_PARAMS = "STATE_FUNCTION_PARAMS"


class Composition(Composition_Base, metaclass=ComponentsMeta):
    """
    Composition(
        controller=None,
        enable_controller=None,
        controller_mode=AFTER,
        controller_condition=Always,
        disable_learning=False,
        name=None,
        prefs=Composition.classPreferences
        context=None)

    Base class for Composition.

    Arguments
    ---------

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

    disable_learning: bool : default False
        specifies whether `LearningMechanisms <LearningMechanism>` in the Composition are executed when ran in `learning mode <Composition.learn>`.

    name : str : default see `name <Composition.name>`
        specifies the name of the Composition.

    prefs : PreferenceSet or specification dict : default Composition.classPreferences
        specifies the `PreferenceSet` for the Composition; see `prefs <Composition.prefs>` for details.

    Attributes
    ----------

    graph : `Graph`
        the full `Graph` associated with this Composition. Contains both Nodes (`Mechanisms <Mechanism>` or
        `Compositions <Composition>`) and `Projections <Projection>`

    nodes : `list[Mechanisms and Compositions]`
        a list of all Nodes (`Mechanisms <Mechanism>` and/or `Compositions <Composition>`) contained in
        this Composition

    input_CIM : `CompositionInterfaceMechanism`
        mediates input values for the INPUT nodes of the Composition. If the Composition is nested, then the
        input_CIM and its InputPorts serve as proxies for the Composition itself in terms of afferent projections.

    input_CIM_ports : dict
        a dictionary in which keys are InputPorts of INPUT Nodes in a composition, and values are lists
        containing two items: the corresponding InputPort and OutputPort on the input_CIM.

    afferents : ContentAddressableList
        a list of all of the `Projections <Projection>` to the Composition's `input_CIM`.

    output_CIM : `CompositionInterfaceMechanism`
        aggregates output values from the OUTPUT nodes of the Composition. If the Composition is nested, then the
        output_CIM and its OutputPorts serve as proxies for Composition itself in terms of efferent projections.

    output_CIM_ports : dict
        a dictionary in which keys are OutputPorts of OUTPUT Nodes in a composition, and values are lists
        containing two items: the corresponding InputPort and OutputPort on the input_CIM.

    efferents : ContentAddressableList
        a list of all of the `Projections <Projection>` from the Composition's `output_CIM`.

    env : Gym Forager Environment : default: None
        stores a Gym Forager Environment so that the Composition may interact with this environment within a
        single call to `run <Composition.run>`.

    shadows : dict
        a dictionary in which the keys are all in the Composition and the values are lists of any Nodes that
        `shadow <InputPort_Shadow_Inputs>` the original Node's input.

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
        if no *context* is specified in a call to run, this *context* is used;  by default,
        it is the Composition's `name <Composition.name>`.

    execution_ids : set
        stores all execution_ids used by this Composition.

    disable_learning: bool : default False
        specifies whether `LearningMechanisms <LearningMechanism>` in the Composition are executed when ran in `learning mode <Composition.learn>`.

    learning_components : list
        contains the learning-related components in the Composition, all or many of which may have been
        created automatically in a call to one of its `add_<*learning_type*>_pathway' methods (see
        `Composition_Learning` for details).  This does *not* contain the `ProcessingMechanisms
        <ProcessingMechanism>` or `MappingProjections <MappingProjection>` in the pathway(s) being learned;
        those are contained in `learning_pathways <Composition.learning_pathways>` attribute.

    learned_components : list[list]
        contains a list of the components subject to learning in the Composition (`ProcessingMechanisms
        <ProcessingMechanism>` and `MappingProjections <MappingProjection>`);  this does *not* contain the
        components used for learning; those are contained in `learning_components
        <Composition.learning_components>` attribute.

    COMMENT:
    learning_pathways : list[list]
        contains a list of the learning pathways specified for the Composition; each item contains a list of the
        `ProcessingMechanisms <ProcessingMechanism>` and `MappingProjection(s) <MappingProjection>` specified a
        a call to one of the Composition's `add_<*learning_type*>_pathway' methods (see `Composition_Learning`
        for details).  This does *not* contain the components used for learning; those are contained in
        `learning_components <Composition.learning_components>` attribute.
    COMMENT

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

    input_specification : None or dict or list or generator or function
        stores the `inputs` for executions of the Composition when it is executed using its `run <Composition.run>`
        method.

    name : str
        the name of the Composition; if it is not specified in the **name** argument of the constructor, a default
        is assigned by CompositionRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the Composition; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).

    """
    # Composition now inherits from Component, so registry inherits name None
    componentType = 'Composition'
    classPreferenceLevel = PreferenceLevel.CATEGORY

    _model_spec_generic_type_name = 'graph'

    class Parameters(ParametersBase):
        """
            Attributes
            ----------

                input_specification
                    see `input_specification <Composition.input_specification>`

                    :default value: None
                    :type:

                results
                    see `results <Composition.results>`

                    :default value: []
                    :type: ``list``

                retain_old_simulation_data
                    see `retain_old_simulation_data <Composition.retain_old_simulation_data>`

                    :default value: False
                    :type: ``bool``

                simulation_results
                    see `simulation_results <Composition.simulation_results>`

                    :default value: []
                    :type: ``list``
        """
        results = Parameter([], loggable=False, pnl_internal=True)
        simulation_results = Parameter([], loggable=False, pnl_internal=True)
        retain_old_simulation_data = Parameter(False, stateful=False, loggable=False)
        input_specification = Parameter(None, stateful=False, loggable=False, pnl_internal=True)

    class _CompilationData(ParametersBase):
        ptx_execution = None
        parameter_struct = None
        state_struct = None
        data_struct = None
        scheduler_conditions = None

    def __init__(
            self,
            name=None,
            controller:ControlMechanism=None,
            enable_controller=None,
            controller_mode:tc.enum(BEFORE,AFTER)=AFTER,
            controller_condition:Condition=Always(),
            retain_old_simulation_data=None,
            prefs=None,
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
        self.node_ordering = []

        # 'env' attr required for dynamic inputs generated by gym forager env
        self.env = None

        # Interface Mechanisms
        self.input_CIM = CompositionInterfaceMechanism(name=self.name + " Input_CIM",
                                                       composition=self)
        self.output_CIM = CompositionInterfaceMechanism(name=self.name + " Output_CIM",
                                                        composition=self)
        self.parameter_CIM = CompositionInterfaceMechanism(name=self.name + " Parameter_CIM",
                                                        composition=self)
        self.input_CIM_ports = {}
        self.output_CIM_ports = {}
        self.parameter_CIM_ports = {}

        self.shadows = {}

        self.default_execution_id = self.name
        self.execution_ids = {self.default_execution_id}

        self.projections = ContentAddressableList(component_type=Component)

        self._scheduler = None

        self.disable_learning = False

        # status attributes
        self.graph_consistent = True  # Tracks if Composition is in runnable state (no dangling projections (what else?)
        self.needs_update_graph = True  # Tracks if Composition graph has been analyzed to assign roles to components
        self.needs_update_graph_processing = True  # Tracks if the processing graph is current with the full graph
        self.needs_update_scheduler = True  # Tracks if the scheduler needs to be regenerated

        self.nodes_to_roles = collections.OrderedDict()

        self.feedback_senders = set()
        self.feedback_receivers = set()

        self._initialize_parameters(
            **param_defaults,
            retain_old_simulation_data=retain_old_simulation_data,
            context=Context(source=ContextFlags.COMPOSITION)
        )

        # Compiled resources
        self.__generated_node_wrappers = {}

        self._compilation_data = self._CompilationData(owner=self)

        # If a PreferenceSet was provided, assign to instance
        _assign_prefs(self, prefs, BasePreferenceSet)

        self.log = CompositionLog(owner=self)
        self._terminal_backprop_sequences = {}

        self.controller = None
        if controller:
            self.add_controller(controller)
        else:
            self.enable_controller = enable_controller
        self.controller_mode = controller_mode
        self.controller_condition = controller_condition
        self.controller_condition.owner = self.controller

        self._update_parameter_components()

        self.initialization_status = ContextFlags.INITIALIZED
        #FIXME: This removes `composition.parameters.values`, as it was not being
        # populated correctly in the first place. `composition.parameters.results`
        # should be used instead - in the long run, we should look into possibly
        # populating both values and results, as it would be more consistent with
        # the behavior of components
        del self.parameters.value
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
    def scheduler(self):
        """
            A default `Scheduler` automatically generated by the Composition, and used for its execution
            when it is `run <Composition_Run>`.

            :getter: Returns the default scheduler, and builds it if it needs updating since the last access.
        """
        if self.needs_update_scheduler or not isinstance(self._scheduler, Scheduler):
            old_scheduler = self._scheduler
            self._scheduler = Scheduler(graph=self.graph_processing, default_execution_id=self.default_execution_id)

            if old_scheduler is not None:
                self._scheduler.add_condition_set(old_scheduler.conditions)

            self.needs_update_scheduler = False

        return self._scheduler

    @scheduler.setter
    def scheduler(self, value: Scheduler):
        warnings.warn(
            f'If {self} is changed (nodes or projections are added or removed), scheduler '
            ' will be rebuilt, and will be different than the Scheduler you are now setting it to.',
            stacklevel=2
        )

        self._scheduler = value

    @property
    def termination_processing(self):
        return self.scheduler.termination_conds

    @termination_processing.setter
    def termination_processing(self, termination_conds):
        self.scheduler.termination_conds = termination_conds

    # ******************************************************************************************************************
    #                                              GRAPH
    # ******************************************************************************************************************

    def _analyze_graph(self, scheduler=None, context=None):
        """
        Assigns `NodeRoles <NodeRoles>` to nodes based on the structure of the `Graph`.

        By default, if _analyze_graph determines that a node is `ORIGIN <NodeRole.ORIGIN>`, it is also given the role
        `INPUT <NodeRole.INPUT>`. Similarly, if _analyze_graph determines that a node is `TERMINAL
        <NodeRole.TERMINAL>`, it is also given the role `OUTPUT <NodeRole.OUTPUT>`.

        However, if the **required_roles** argument of `add_node <Composition.add_node>` is used to set any node in the
        Composition to `INPUT <NodeRole.INPUT>`, then the `ORIGIN <NodeRole.ORIGIN>` nodes are not set to `INPUT
        <NodeRole.INPUT>` by default. If the **required_roles** argument of `add_node <Composition.add_node>` is used
        to set any node in the Composition to `OUTPUT <NodeRole.OUTPUT>`, then the `TERMINAL <NodeRole.TERMINAL>`
        nodes are not set to `OUTPUT <NodeRole.OUTPUT>` by default.
        """
        for n in self.nodes:
            try:
                n._analyze_graph(context=context)
            except AttributeError:
                pass

        self._check_feedback(scheduler=scheduler, context=context)
        self._determine_node_roles(context=context)
        self._create_CIM_ports(context=context)
        self._update_shadow_projections(context=context)
        self._check_for_projection_assignments(context=context)
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

    def _analyze_consideration_queue(self, q, objective_mechanism):
        """Assigns NodeRole.ORIGIN to all nodes in the first entry of the consideration queue and NodeRole.TERMINAL to
            all nodes in the last entry of the consideration queue. The ObjectiveMechanism of a controller
            may not be NodeRole.TERMINAL, so if the ObjectiveMechanism is the only node in the last entry of the
            consideration queue, then the second-to-last entry is NodeRole.TERMINAL instead.
        """
        for node in q[0]:
            self._add_node_role(node, NodeRole.ORIGIN)

        for node in list(q)[-1]:
            if node != objective_mechanism:
                self._add_node_role(node, NodeRole.TERMINAL)
            elif len(q[-1]) < 2:
                for previous_node in q[-2]:
                    self._add_node_role(previous_node, NodeRole.TERMINAL)


    # ******************************************************************************************************************
    #                                               NODES
    # ******************************************************************************************************************

    def add_node(self, node, required_roles=None, context=None):
        """
            Add a Composition Node (`Mechanism <Mechanism>` or `Composition`) to Composition, if it is not already added

            Arguments
            ---------

            node : `Mechanism <Mechanism>` or `Composition`
                the node to be added to the Composition

            required_roles : `NodeRole` or list of NodeRoles
                any NodeRoles roles that this node should have in addition to those determined by analyze graph.
        """

        self._update_shadows_dict(node)

        try:
            node._analyze_graph()
        except AttributeError:
            pass

        node._check_for_composition(context=context)

        # Add node to Composition's graph
        if node not in [vertex.component for vertex in
                        self.graph.vertices]:  # Only add if it doesn't already exist in graph
            node.is_processing = True
            self.graph.add_component(node)  # Set incoming edge list of node to empty
            self.nodes.append(node)
            self.node_ordering.append(node)
            self.nodes_to_roles[node] = set()

            self.needs_update_graph = True
            self.needs_update_graph_processing = True
            self.needs_update_scheduler = True

            try:
                # activate any projections the node requires
                node._activate_projections_for_compositions(self)
            except AttributeError:
                pass

        # Implement any components specified in node's aux_components attribute
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
                        if isinstance(component[1], bool) or component[1]==MAYBE:
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

            # Add all Projections to the Composition
            for proj_spec in projections:
                # The proj_spec assumes a direct connection between sender and receiver, and is therefore invalid if
                # either are nested (i.e. projections between them need to be routed through a CIM). In these cases,
                # we instantiate a new projection between sender and receiver instead of using the original spec.
                # If the sender or receiver is an AutoAssociativeProjection, then the owner will be another projection
                # instead of a mechanism, so we need to use owner_mech instead.
                sender_node = proj_spec[0].sender.owner
                receiver_node = proj_spec[0].receiver.owner
                if isinstance(sender_node, AutoAssociativeProjection):
                    sender_node = proj_spec[0].sender.owner.owner_mech
                if isinstance(receiver_node, AutoAssociativeProjection):
                    receiver_node = proj_spec[0].receiver.owner.owner_mech
                if sender_node in self.nodes and \
                        receiver_node in self.nodes:
                    self.add_projection(projection=proj_spec[0],
                                        feedback=proj_spec[1])
                else:
                    self.add_projection(sender=proj_spec[0].sender,
                                        receiver=proj_spec[0].receiver,
                                        feedback=proj_spec[1])

        # Implement required_roles
        if required_roles:
            if not isinstance(required_roles, list):
                required_roles = [required_roles]
            for required_role in required_roles:
                self.add_required_node_role(node, required_role)

        # Add projections to node from sender of any shadowed InputPorts
        for input_port in node.input_ports:
            if hasattr(input_port, SHADOW_INPUTS) and input_port.shadow_inputs is not None:
                for proj in input_port.shadow_inputs.path_afferents:
                    sender = proj.sender
                    if sender.owner != self.input_CIM:
                        self.add_projection(projection=MappingProjection(sender=proj.sender, receiver=input_port),
                                            sender=proj.sender.owner,
                                            receiver=node)

        # Add ControlSignals to controller and ControlProjections
        #     to any parameter_ports specified for control in node's constructor
        if self.controller:
            deferred_init_control_specs = node._get_parameter_port_deferred_init_control_specs()
            if deferred_init_control_specs:
                self.controller._remove_default_control_signal(type=CONTROL_SIGNAL)
                for ctl_sig_spec in deferred_init_control_specs:
                    # FIX: 9/14/19 - IS THE CONTEXT CORRECT (TRY TRACKING IN SYSTEM TO SEE WHAT CONTEXT IS):
                    control_signal = self.controller._instantiate_control_signal(control_signal=ctl_sig_spec,
                                                           context=Context(source=ContextFlags.COMPOSITION))
                    self.controller.control.append(control_signal)
                    self.controller._activate_projections_for_compositions(self)

    def add_nodes(self, nodes, required_roles=None):
        """
            Add a list of Composition Nodes (`Mechanism <Mechanism>` or `Composition`) to the Composition,

            Arguments
            ---------

            nodes : list
                the nodes to be added to the Composition.  Each item of the list must be a `Mechanism <Mechanism>`,
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
            self.node_ordering.remove(node)

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

    def get_roles_by_node(self, node):
        try:
            return self.nodes_to_roles[node]
        except KeyError:
            raise CompositionError('Node {0} not found in {1}.nodes_to_roles'.format(node, self))

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
        if role is None or role not in NodeRole:
            raise CompositionError('Invalid NodeRole: {0}'.format(role))

        try:
            return [node for node in self.nodes if role in self.nodes_to_roles[node]]

        except KeyError as e:
            raise CompositionError('Node missing from {0}.nodes_to_roles: {1}'.format(self, e))

    def _get_nested_nodes(self,
                          nested_nodes=NotImplemented,
                          root_composition=NotImplemented,
                          visited_compositions=NotImplemented):
        """Recursive search that returns all nodes of all nested compositions in a tuple with the composition they are
            embedded in.

        :return

        A list of tuples in format (node, composition) containing all nodes of all nested compositions.
        """
        if nested_nodes is NotImplemented:
            nested_nodes=[]
        if root_composition is NotImplemented:
            root_composition=self
        if visited_compositions is NotImplemented:
            visited_compositions = [self]
        for node in self.nodes:
            if node.componentType == 'Composition' and \
                    node not in visited_compositions:
                visited_compositions.append(node)
                node._get_nested_nodes(nested_nodes,
                                       root_composition,
                                       visited_compositions)
            elif root_composition is not self:
                nested_nodes.append((node,self))
        return nested_nodes

    def _get_nested_compositions(self,
                                 nested_compositions=NotImplemented,
                                 visited_compositions=NotImplemented):
        """Recursive search that returns all nested compositions.

        :return

        A list of nested compositions.
        """
        if nested_compositions is NotImplemented:
            nested_compositions=[]
        if visited_compositions is NotImplemented:
            visited_compositions = [self]
        for node in self.nodes:
            if node.componentType == 'Composition' and \
                    node not in visited_compositions:
                nested_compositions.append(node)
                visited_compositions.append(node)
                node._get_nested_compositions(nested_compositions,
                                              visited_compositions)
        return nested_compositions

    def _determine_node_roles(self, context=None):

        # Clear old roles
        self.nodes_to_roles.update({k: set() for k in self.nodes_to_roles})

        # Required Roles
        for node_role_pair in self.required_node_roles:
            self._add_node_role(node_role_pair[0], node_role_pair[1])

        objective_mechanism = None
        # # MODIFIED 10/24/19 OLD:
        # if self.controller and self.enable_controller and self.controller.objective_mechanism:
        # MODIFIED 10/24/19 NEW:
        if self.controller and self.controller.objective_mechanism:
        # MODIFIED 10/24/19 END
            objective_mechanism = self.controller.objective_mechanism
            self._add_node_role(objective_mechanism, NodeRole.CONTROLLER_OBJECTIVE)

        # Use Scheduler.consideration_queue to check for ORIGIN and TERMINAL Nodes:
        if self.scheduler.consideration_queue:
            self._analyze_consideration_queue(self.scheduler.consideration_queue, objective_mechanism)

        # A ControlMechanism should not be the TERMINAL node of a Composition
        #    (unless it is specifed as a required_role, in which case it is reassigned below)
        for node in self.nodes:
            if isinstance(node, ControlMechanism):
                if NodeRole.TERMINAL in self.nodes_to_roles[node]:
                    self.nodes_to_roles[node].remove(NodeRole.TERMINAL)
                if NodeRole.OUTPUT in self.nodes_to_roles[node]:
                    self.nodes_to_roles[node].remove(NodeRole.OUTPUT)

        # Cycles
        for node in self.scheduler.cycle_nodes:
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

        # If OUTPUT nodes were not specified by user, assign them:
        # - if there are LearningMechanisms, OUTPUT node is the last non-learning-related node.
        # - if there are no TERMINAL nodes either, then the last node added to the Composition becomes the OUTPUT node.
        if not self.get_nodes_by_role(NodeRole.OUTPUT):

            # FIX: 10/24/19: NOW MISSES controller.objective_mechanism in test_controller_objective_mech_not_terminal
            #                 if controller_enabled = False
            def remove_learning_and_control_nodes(nodes):
                output_nodes_copy = nodes.copy()
                for node in output_nodes_copy:
                    if (NodeRole.LEARNING in self.nodes_to_roles[node]
                            or NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[node]
                            or isinstance(node, ControlMechanism)
                            or (isinstance(node, ObjectiveMechanism) and node._role == CONTROL)):
                        nodes.remove(node)

            if self.get_nodes_by_role(NodeRole.LEARNING) or self.get_nodes_by_role(NodeRole.AUTOASSOCIATIVE_LEARNING):
                # FIX: ADD COMMENT HERE
                # terminal_nodes = [[n for n in self.nodes if not NodeRole.LEARNING in self.nodes_to_roles[n]][-1]]
                output_nodes = list([items for items in self.scheduler.consideration_queue
                                     if any([item for item in items
                                               if (not NodeRole.LEARNING in self.nodes_to_roles[item] and
                                                   not NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[item])
                                               ])])[-1].copy()
            else:
                output_nodes = self.get_nodes_by_role(NodeRole.TERMINAL)

            if output_nodes:
                remove_learning_and_control_nodes(output_nodes)
            else:
                try:
                    # Assign TERMINAL role to nodes that are last in the scheduler's consideration queue that are:
                    #    - not used for Learning;
                    #    - not ControlMechanisms or ObjectiveMechanisms that project to them;
                    #    - do not project to any other nodes.

                    # First, find last consideration_set in scheduler that does not contain only
                    #    learning-related nodes, ControlMechanism(s) or control-related ObjectiveMechanism(s);
                    #    note: get copy of the consideration_set, as don't want to modify one actually used by scheduler
                    output_nodes = list([items for items in self.scheduler.consideration_queue
                                           if any([item for item in items if
                                                   (not NodeRole.LEARNING in self.nodes_to_roles[item] and
                                                    not NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[item]
                                                    and not isinstance(item, ControlMechanism)
                                                    and not (isinstance(item, ObjectiveMechanism)
                                                             and item._role == CONTROL))
                                                   ])]
                                          )[-1].copy()

                    # Next, remove any learning-related nodes, ControlMechanism(s) or control-related
                    #    ObjectiveMechanism(s) that may have "snuck in" (i.e., happen to be in the set)
                    remove_learning_and_control_nodes(output_nodes)

                    # Then, add any nodes that are not learning-related or a ControlMechanism,
                    #    and that have *no* efferent Projections
                    # IMPLEMENTATION NOTE:
                    #  Do this here, as the list considers entire sets in the consideration queue,
                    #    and a node with no efferents may be in the same set as one with efferents
                    #    if they have the same dependencies.
                    for node in self.nodes:
                        if (not node.efferents
                                and not NodeRole.LEARNING in self.nodes_to_roles[node]
                                and not NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[node]
                                and not isinstance(node, ControlMechanism)
                                and not (isinstance(node, ObjectiveMechanism) and node._role == CONTROL)
                        ):
                            output_nodes.add(node)
                except IndexError:
                    output_nodes = []
            for node in output_nodes:
                self._add_node_role(node, NodeRole.OUTPUT)

            # Finally, assign TERMINAL nodes
            for node in self.nodes:
                if not node.efferents or NodeRole.FEEDBACK_SENDER in self.nodes_to_roles[node]:
                    self._add_node_role(node, NodeRole.TERMINAL)

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
    def _create_CIM_ports(self, context=None):
        """
            - remove the default InputPort and OutputPort from the CIMs if this is the first time that real
              InputPorts and OutputPorts are being added to the CIMs

            - create a corresponding InputPort and OutputPort on the `input_CIM <Composition.input_CIM>` for each
              InputPort of each INPUT node. Connect the OutputPort on the input_CIM to the INPUT node's corresponding
              InputPort via a standard MappingProjection.

            - create a corresponding InputPort and OutputPort on the `output_CIM <Composition.output_CIM>` for each
              OutputPort of each OUTPUT node. Connect the OUTPUT node's OutputPort to the output_CIM's corresponding
              InputPort via a standard MappingProjection.

            - build two dictionaries:

                (1) input_CIM_ports = { INPUT Node InputPort: (InputCIM InputPort, InputCIM OutputPort) }

                (2) output_CIM_ports = { OUTPUT Node OutputPort: (OutputCIM InputPort, OutputCIM OutputPort) }

            - if the Node has any shadows, create the appropriate projections as needed.

            - delete all of the above for any node Ports which were previously, but are no longer, classified as
              INPUT/OUTPUT

            - if composition has a controller, remove default InputPort and OutputPort of all nested compositions'
              `parameter CIMs <Composition.parameter_CIM>` which contain nodes that will be modulated and whose default
              ports have not already been removed

            - delete afferents of compositions' parameter CIMs if their sender is no longer the controller of any of
              the composition's parent compositions

            - create a corresponding InputPort and ControlSignal on the `parameter_CIM <Composition.parameter_CIM>` for
              each parameter modulated by the controller

            - instantiate and activate projections from ControlSignals of controller to corresponding InputPorts
              of nested compositions' `parameter_CIMs <Composition.parameter_CIM>`
        """

        if not self.input_CIM.connected_to_composition:
            self.input_CIM.input_ports.remove(self.input_CIM.input_port)
            self.input_CIM.output_ports.remove(self.input_CIM.output_port)
            self.input_CIM.connected_to_composition = True

        if not self.output_CIM.connected_to_composition:
            self.output_CIM.input_ports.remove(self.output_CIM.input_port)
            self.output_CIM.output_ports.remove(self.output_CIM.output_port)
            self.output_CIM.connected_to_composition = True

        current_input_node_input_ports = set()

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        for node in input_nodes:

            for input_port in node.external_input_ports:
                # add it to our set of current input ports
                current_input_node_input_ports.add(input_port)

                # if there is not a corresponding CIM OutputPort, add one
                if input_port not in set(self.input_CIM_ports.keys()):
                    interface_input_port = InputPort(owner=self.input_CIM,
                                                      variable=input_port.defaults.value,
                                                      reference_value=input_port.defaults.value,
                                                      name="INPUT_CIM_" + node.name + "_" + input_port.name)

                    interface_output_port = OutputPort(owner=self.input_CIM,
                                                        variable=OWNER_VALUE,
                                                        function=InterfacePortMap(
                                                             corresponding_input_port=interface_input_port),
                                                        name="INPUT_CIM_" + node.name + "_" + input_port.name)

                    self.input_CIM_ports[input_port] = [interface_input_port, interface_output_port]

                    projection = MappingProjection(sender=interface_output_port,
                                                   receiver=input_port,
                                                   matrix=IDENTITY_MATRIX,
                                                   name="(" + interface_output_port.name + ") to ("
                                                        + input_port.owner.name + "-" + input_port.name + ")")
                    projection._activate_for_compositions(self)

                    if isinstance(node, Composition):
                        projection._activate_for_compositions(node)

        new_shadow_projections = {}

        # for any entirely new shadow_projections, create a MappingProjection object and add to projections
        for output_port, input_port in new_shadow_projections:
            if new_shadow_projections[(output_port, input_port)] is None:
                shadow_projection = MappingProjection(sender=output_port,
                                                      receiver=input_port,
                                                      name="(" + output_port.name + ") to ("
                                                           + input_port.owner.name + "-" + input_port.name + ")")
                shadow_projection._activate_for_compositions(self)

        sends_to_input_ports = set(self.input_CIM_ports.keys())

        # For any ports still registered on the CIM that does not map to a corresponding INPUT node I.S.:
        for input_port in sends_to_input_ports.difference(current_input_node_input_ports):
            for projection in input_port.path_afferents:
                if projection.sender == self.input_CIM_ports[input_port][1]:
                    # remove the corresponding projection from the INPUT node's path afferents
                    input_port.path_afferents.remove(projection)

                    # projection.receiver.efferents.remove(projection)
                    # Bug? ^^ projection is not in receiver.efferents??
                    if projection.receiver.owner in self.shadows and len(self.shadows[projection.receiver.owner]) > 0:
                        for shadow in self.shadows[projection.receiver.owner]:
                            for shadow_input_port in shadow.input_ports:
                                for shadow_projection in shadow_input_port.path_afferents:
                                    if shadow_projection.sender == self.input_CIM_ports[input_port][1]:
                                        shadow_input_port.path_afferents.remove(shadow_projection)

            # remove the CIM input and output ports associated with this INPUT node InputPort
            self.input_CIM.input_ports.remove(self.input_CIM_ports[input_port][0])
            self.input_CIM.output_ports.remove(self.input_CIM_ports[input_port][1])

            # and from the dictionary of CIM OutputPort/InputPort pairs
            del self.input_CIM_ports[input_port]

        # OUTPUT CIMS
        # loop over all OUTPUT nodes
        current_output_node_output_ports = set()
        for node in self.get_nodes_by_role(NodeRole.OUTPUT):
            for output_port in node.output_ports:
                current_output_node_output_ports.add(output_port)
                # if there is not a corresponding CIM OutputPort, add one
                if output_port not in set(self.output_CIM_ports.keys()):

                    interface_input_port = InputPort(owner=self.output_CIM,
                                                     variable=output_port.defaults.value,
                                                     reference_value=output_port.defaults.value,
                                                     name="OUTPUT_CIM_" + node.name + "_" + output_port.name)

                    interface_output_port = OutputPort(
                            owner=self.output_CIM,
                            variable=OWNER_VALUE,
                            function=InterfacePortMap(corresponding_input_port=interface_input_port),
                            reference_value=output_port.defaults.value,
                            name="OUTPUT_CIM_" + node.name + "_" + output_port.name)

                    self.output_CIM_ports[output_port] = [interface_input_port, interface_output_port]

                    proj_name = "(" + output_port.name + ") to (" + interface_input_port.name + ")"

                    proj = MappingProjection(
                        sender=output_port,
                        receiver=interface_input_port,
                        # FIX:  This fails if OutputPorts don't all have the same dimensionality (number of axes);
                        #       see example in test_output_ports/TestOutputPorts
                        matrix=IDENTITY_MATRIX,
                        name=proj_name
                    )
                    proj._activate_for_compositions(self)
                    if isinstance(node, Composition):
                        proj._activate_for_compositions(node)

        previous_output_node_output_ports = set(self.output_CIM_ports.keys())
        for output_port in previous_output_node_output_ports.difference(current_output_node_output_ports):
            # remove the CIM input and output ports associated with this Terminal Node OutputPort
            self.output_CIM.remove_ports(self.output_CIM_ports[output_port][0])
            self.output_CIM.remove_ports(self.output_CIM_ports[output_port][1])
            del self.output_CIM_ports[output_port]

        # PARAMETER CIMS
        if self.controller:
            controller = self.controller
            nested_nodes = dict(self._get_nested_nodes())
            nested_comps = self._get_nested_compositions()
            for comp in nested_comps:
                for port in comp.parameter_CIM.input_ports:
                    for afferent in port.all_afferents:
                        if not comp in afferent.sender.owner.composition._get_nested_compositions():
                            del port._afferents_info[afferent]
                            if afferent in port.path_afferents:
                                port.path_afferents.remove(afferent)
                            if afferent in port.mod_afferents:
                                port.mod_afferents.remove(afferent)

            for modulatory_signal in controller.control_signals:
                for projection in modulatory_signal.projections:
                    receiver = projection.receiver
                    mech = receiver.owner
                    if mech in nested_nodes:
                        comp = nested_nodes[mech]
                        pcim = comp.parameter_CIM
                        pcIM_ports = comp.parameter_CIM_ports
                        if receiver not in pcIM_ports:
                            if not pcim.connected_to_composition:
                                pcim.input_ports.remove(pcim.input_port)
                                pcim.output_ports.remove(pcim.output_port)
                                pcim.connected_to_composition = True
                            modulation = modulatory_signal.owner.modulation
                            input_port = InputPort(
                                owner = pcim,
                            )
                            control_signal = ControlSignal(
                                owner = pcim,
                                modulation = modulation,
                                variable = OWNER_VALUE,
                                transfer_function=InterfacePortMap(
                                    corresponding_input_port = input_port
                                ),
                                modulates = receiver,
                                name = 'PARAMETER_CIM_' + mech.name + "_" + receiver.name
                            )
                            for projection in control_signal.projections:
                                projection._activate_for_compositions(self)
                                projection._activate_for_compositions(comp)
                            for projection in receiver.mod_afferents:
                                if projection.sender.owner == controller:
                                    receiver.mod_afferents.remove(projection)
                            pcIM_ports[receiver] = (modulatory_signal, input_port)

            for comp in nested_comps:
                pcim = comp.parameter_CIM
                connected_to_controller = False
                for afferent in pcim.afferents:
                    if afferent.sender.owner is controller:
                        connected_to_controller = True
                if not connected_to_controller:
                    for efferent in controller.efferents:
                        if efferent.receiver in pcIM_ports:
                            input_projection = MappingProjection(
                                sender = efferent.sender,
                                receiver = pcIM_ports[efferent.receiver][1]
                            )
                            input_projection._activate_for_compositions(self)
                            input_projection._activate_for_compositions(comp)

    def _get_nested_node_CIM_port(self,
                                   node: Mechanism,
                                   node_state: tc.any(InputPort, OutputPort),
                                   role: tc.enum(NodeRole.INPUT, NodeRole.OUTPUT)
                                   ):
        """Check for node in nested Composition
        Return relevant port of relevant CIM if found and nested Composition in which it was found, else (None, None)
        """

        nested_comp = CIM_port_for_nested_node = CIM = None

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
                # if CIM_port_for_nested_node:
                #     warnings.warn("{} found with {} of {} in more than one nested {} of {}; "
                #                   "only first one found (in {}) will be used".
                #                   format(node.name, NodeRole.__name__, repr(role),
                #                          Composition.__name__, self.name, nested_comp.name))
                #     continue

                if isinstance(node_state, InputPort):
                    CIM_port_for_nested_node = nc.input_CIM_ports[node_state][0]
                    CIM = nc.input_CIM
                elif isinstance(node_state, OutputPort):
                    CIM_port_for_nested_node = nc.output_CIM_ports[node_state][1]
                    CIM = nc.output_CIM
                else:
                    # IMPLEMENTATION NOTE:  Place marker for future implementation of ParameterPort handling
                    #                       However, typecheck above should have caught this
                    assert False

                nested_comp = nc
                break

        return CIM_port_for_nested_node, CIM_port_for_nested_node, nested_comp, CIM

    def _update_shadows_dict(self, node):
        # Create an empty entry for this node in the Composition's "shadows" dict
        # If any other nodes shadow this node, they will be added to the list
        if node not in self.shadows:
            self.shadows[node] = []

        nested_nodes = dict(self._get_nested_nodes())
        # If this node is shadowing another node, then add it to that node's entry in the Composition's "shadows" dict
        # If the node it's shadowing is a nested node, add it to the entry for the composition it's nested in.
        for input_port in node.input_ports:
            if hasattr(input_port, SHADOW_INPUTS) and input_port.shadow_inputs is not None:
                owner = input_port.shadow_inputs.owner
                if owner in nested_nodes:
                    owner = nested_nodes[owner]
                if node not in self.shadows[owner]:
                    self.shadows[owner].append(node)


    # ******************************************************************************************************************
    #                                            PROJECTIONS
    # ******************************************************************************************************************

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

        • if **sender** and **receiver** are specified and one or more Projections already exists between them:
          - if it is in the Composition:
            - if there is only one, the request is ignored and the existing Projection is returned
            - if there is more than one, an exception is raised as this should never be the case
          - it is NOT in the Composition:
            - if there is only one, that Projection is used;
            - if there is more than one, the last in the list (presumably the most recent) is used;
            in either case, processing continues, to activate it for the Compostion,
            construct any "shadow" projections that may be specified, and assign feedback if specified,

        • if the status of **projection** is `deferred_init`:

          - if its `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` attributes are not
            specified, then **sender** and/or **receiver** are used.

          - if `sender <Projection_Base.sender>` and/or `receiver <Projection_Base.receiver>` attributes are specified,
            they must match **sender** and/or **receiver** if those have also been specified.

          - if a Projection between the specified sender and receiver does *not* already exist, it is initialized; if
            it *does* already exist, the request to add it is ignored, however requests to shadow it and/or mark it as
            a`feedback` Projection are implemented (in case it has not already been done for the existing Projection).

        .. note::
           If **projection** is an instantiated Projection (i.e., not in `deferred_init`) and one already exists between
           its `sender <Projection_Base.sender>` and `receiver <Projection_Base.receiver>` a warning is generated.

        COMMENT:
        IMPLEMENTATION NOTE:
            Duplicates are determined by the **Ports** to which they project, not the Mechanisms (to allow
            multiple Projections to exist between the same pair of Mechanisms using different Ports).
            -
            If an already instantiated Projection is passed to add_projection and is a duplicate of an existing one,
            it is detected and suppresed, with a warning, in Port._instantiate_projections_to_port.
            -
            If a Projection with deferred_init status is a duplicate, it is fully suppressed here,
            as these are generated by add_linear_processing_pathway if the pathway overlaps with an existing one,
            and so warnings are unnecessary and would be confusing to users.
        COMMENT

        Arguments
        ---------

        sender : Mechanism, Composition, or OutputPort
            the sender of **projection**

        projection : Projection, matrix
            the projection to add

        receiver : Mechanism, Composition, or InputPort
            the receiver of **projection**

        feedback : bool
            When False (default) all Nodes within a cycle containing this Projection execute in parallel. This
            means that each Projections within the cycle actually passes to its `receiver <Projection_Base.receiver>`
            the `value <Projection_Base.value>` of its `sender <Projection_Base.sender>` from the previous execution.
            When True, this Projection "breaks" the cycle, such that all Nodes execute in sequence, and only the
            Projection marked as 'feedback' passes to its `receiver <Projection_Base.receiver>` the
            `value <Projection_Base.value>` of its `sender <Projection_Base.sender>` from the previous execution.

        Returns
        -------

        projection if added, else None

    """

        existing_projections = False

        # If a sender and receiver have been specified but not a projection,
        #    check whether there is *any* projection like that
        #    (i.e., whether it/they are already in the current Composition or not);  if so:
        #    - if there is only one, use that;
        #    - if there are several, use the last in the list (on the assumption in that it is the most recent).
        # Note:  Skip this if **projection** was specified, as it might include parameters that are different
        #        than the existing ones, in which case should use that rather than any existing ones;
        #        will handle any existing Projections that are in the current Composition below.
        if sender and receiver and projection is None:
            existing_projections = self._check_for_existing_projections(sender=sender,
                                                               receiver=receiver,
                                                               in_composition=False)
            if existing_projections:
                if isinstance(sender, Port):
                    sender_check = sender.owner
                else:
                    sender_check = sender
                if isinstance(receiver, Port):
                    receiver_check = receiver.owner
                else:
                    receiver_check = receiver
                if ((not isinstance(sender_check, CompositionInterfaceMechanism) and sender_check not in self.nodes)
                        or (not isinstance(receiver_check, CompositionInterfaceMechanism)
                            and receiver_check not in self.nodes)):
                    for proj in existing_projections:
                        self.remove_projection(proj)
                        for port in receiver_check.input_ports + sender_check.output_ports:
                            if proj in port.afferents_info:
                                del port.afferents_info[proj]
                            if proj in port.projections:
                                port.projections.remove(proj)
                            if proj in port.path_afferents:
                                port.path_afferents.remove(proj)
                            if proj in port.mod_afferents:
                                port.mod_afferents.remove(proj)
                            if proj in port.efferents:
                                port.efferents.remove(proj)
                else:
                #  Need to do stuff at end, so can't just return
                    if self.prefs.verbosePref:
                        warnings.warn(f"Several existing projections were identified between "
                                      f"{sender.name} and {receiver.name}: {[p.name for p in existing_projections]}; "
                                      f"the last of these will be used in {self.name}.")
                    projection = existing_projections[-1]

        # FIX: 9/30/19 - Why is this not an else?
        #                Because above is only for existing Projections outside of Composition, which should be
        #                used
        #                But existing one could be within, in which case want to use that one
        #                existing Projection might be deferred_init, and want t
        try:
            # Note: this does NOT initialize the Projection if it is in deferred_init
            projection = self._parse_projection_spec(projection, name)
        except DuplicateProjectionError:
            # return projection
            return

        # Parse sender and receiver specs
        sender, sender_mechanism, graph_sender, nested_compositions = self._parse_sender_spec(projection, sender)
        receiver, receiver_mechanism, graph_receiver, receiver_input_port, nested_compositions, learning_projection = \
            self._parse_receiver_spec(projection, receiver, sender, learning_projection)

        # If Deferred init
        if projection.initialization_status == ContextFlags.DEFERRED_INIT:
            # If sender or receiver are Port specs, use those;  otherwise, use graph node (Mechanism or Composition)
            if not isinstance(sender, OutputPort):
                sender = sender_mechanism
            if not isinstance(receiver, InputPort):
                receiver = receiver_mechanism
            # Check if Projection to be initialized already exists in the current Composition;
            #    if so, mark as existing_projections and skip
            existing_projections = self._check_for_existing_projections(sender=sender, receiver=receiver)
            if existing_projections:
                return
            else:
                # Initialize Projection
                projection._init_args['sender'] = sender
                projection._init_args['receiver'] = receiver
                try:
                    projection._deferred_init()
                except DuplicateProjectionError:
                    # return projection
                    return

        else:
            existing_projections = self._check_for_existing_projections(projection, sender=sender, receiver=receiver)

        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        if (sender_mechanism != self.input_CIM
                and receiver_mechanism != self.output_CIM
                and projection not in [vertex.component for vertex in self.graph.vertices]
                and not learning_projection):

            projection.is_processing = False
            # KDM 5/24/19: removing below rename because it results in several existing_projections
            # projection.name = f'{sender} to {receiver}'
            self.graph.add_component(projection, feedback=feedback)

            try:
                self.graph.connect_components(graph_sender, projection)
                self.graph.connect_components(projection, graph_receiver)
            except CompositionError as c:
                raise CompositionError(f"{c.args[0]} to {self.name}.")

        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        if not existing_projections:
            self._validate_projection(projection,
                                      sender, receiver,
                                      sender_mechanism, receiver_mechanism,
                                      learning_projection)
        self.needs_update_graph = True
        self.needs_update_graph_processing = True
        self.needs_update_scheduler = True

        projection._activate_for_compositions(self)
        for comp in nested_compositions:
            projection._activate_for_compositions(comp)

        # Note: do all of the following even if Projection is a existing_projections,
        #   as these conditions shoud apply to the exisiting one (and it won't hurt to try again if they do)

        # Create "shadow" projections to any input ports that are meant to shadow this projection's receiver
        # (note: do this even if there is a duplciate and they are not allowed, as still want to shadow that projection)
        if receiver_mechanism in self.shadows and len(self.shadows[receiver_mechanism]) > 0:
            for shadow in self.shadows[receiver_mechanism]:
                for input_port in shadow.input_ports:
                    if input_port.shadow_inputs is not None:
                        if input_port.shadow_inputs.owner == receiver:
                            # TBI: Copy the projection type/matrix value of the projection that is being shadowed
                            self.add_projection(MappingProjection(sender=sender, receiver=input_port),
                                                sender_mechanism, shadow)
        if feedback:
            self.feedback_senders.add(sender_mechanism)
            self.feedback_receivers.add(receiver_mechanism)

        return projection

    def remove_projection(self, projection):
        # step 1 - remove Vertex from Graph
        if projection in [vertex.component for vertex in self.graph.vertices]:
            vert = self.graph.comp_to_vertex[projection]
            self.graph.remove_vertex(vert)
        # step 2 - remove Projection from Composition's list
        if projection in self.projections:
            self.projections.remove(projection)

        # step 3 - TBI? remove Projection from afferents & efferents lists of any node

    def _add_projection(self, projection):
        self.projections.append(projection)

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

    def _parse_sender_spec(self, projection, sender):

        # if a sender was not passed, check for a sender OutputPort stored on the Projection object
        if sender is None:
            if hasattr(projection, "sender"):
                sender = projection.sender.owner
            else:
                raise CompositionError(f"{projection.name} is missing a sender specification. "
                                       f"For a Projection to be added to a Composition a sender must be specified, "
                                       "either on the Projection or in the call to Composition.add_projection(). ")

        # initialize all receiver-related variables
        graph_sender = sender_mechanism = sender_output_port = sender

        nested_compositions = []
        if isinstance(sender, Mechanism):
            # Mechanism spec -- update sender_output_port to reference primary OutputPort
            sender_output_port = sender.output_port

        elif isinstance(sender, OutputPort):
            # InputPort spec -- update sender_mechanism and graph_sender to reference owner Mechanism
            sender_mechanism = graph_sender = sender.owner

        elif isinstance(sender, Composition):
            # Nested Composition Spec -- update sender_mechanism to CIM; sender_output_port to CIM's primary O.S.
            sender_mechanism = sender.output_CIM
            sender_output_port = sender_mechanism.output_port
            nested_compositions.append(sender)

        else:
            raise CompositionError("sender arg ({}) of call to add_projection method of {} is not a {}, {} or {}".
                                   format(sender, self.name,
                                          Mechanism.__name__, OutputPort.__name__, Composition.__name__))

        if (not isinstance(sender_mechanism, CompositionInterfaceMechanism)
                and not isinstance(sender, Composition)
                and sender_mechanism not in self.nodes):
            if isinstance(sender, Port):
                sender_name = sender.full_name
            else:
                sender_name = sender.name

            # if the sender is IN a nested Composition AND sender is an OUTPUT Node
            # then use the corresponding CIM on the nested comp as the sender going forward
            sender, sender_output_port, graph_sender, sender_mechanism = \
                self._get_nested_node_CIM_port(sender_mechanism,
                                                sender_output_port,
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

        receiver_arg = receiver

        # if a receiver was not passed, check for a receiver InputPort stored on the Projection object
        if receiver is None:
            if hasattr(projection, "receiver"):
                receiver = projection.receiver.owner
            else:
                raise CompositionError("For a Projection to be added to a Composition, a receiver must be specified, "
                                       "either on the Projection or in the call to Composition.add_projection(). {}"
                                       " is missing a receiver specification. ".format(projection.name))

        # initialize all receiver-related variables
        graph_receiver = receiver_mechanism = receiver_input_port = receiver

        nested_compositions = []
        if isinstance(receiver, Mechanism):
            # Mechanism spec -- update receiver_input_port to reference primary InputPort
            receiver_input_port = receiver.input_port

        elif isinstance(receiver, (InputPort, ParameterPort)):
            # InputPort spec -- update receiver_mechanism and graph_receiver to reference owner Mechanism
            receiver_mechanism = graph_receiver = receiver.owner

        elif isinstance(sender, (ControlSignal, ControlMechanism)) and isinstance(receiver, ParameterPort):
            # ParameterPort spec -- update receiver_mechanism and graph_receiver to reference owner Mechanism
            receiver_mechanism = graph_receiver = receiver.owner

        elif isinstance(receiver, Composition):
            # Nested Composition Spec -- update receiver_mechanism to CIM; receiver_input_port to CIM's primary I.S.
            receiver_mechanism = receiver.input_CIM
            receiver_input_port = receiver_mechanism.input_port
            nested_compositions.append(receiver)

        # KAM HACK 2/13/19 to get hebbian learning working for PSY/NEU 330
        # Add autoassociative learning mechanism + related projections to composition as processing components
        elif isinstance(receiver, AutoAssociativeProjection):
            receiver_mechanism = receiver.owner_mech
            receiver_input_port = receiver_mechanism.input_port
            learning_projection = True

        elif isinstance(sender, LearningMechanism):
            receiver_mechanism = receiver.receiver.owner
            receiver_input_port = receiver_mechanism.input_port
            learning_projection = True

        else:
            raise CompositionError(f"receiver arg ({receiver_arg}) of call to add_projection method of {self.name} "
                                   f"is not a {Mechanism.__name__}, {InputPort.__name__} or {Composition.__name__}.")

        if (not isinstance(receiver_mechanism, CompositionInterfaceMechanism)
                and not isinstance(receiver, Composition)
                and receiver_mechanism not in self.nodes
                and not learning_projection):

            # if the receiver is IN a nested Composition AND receiver is an INPUT Node
            # then use the corresponding CIM on the nested comp as the receiver going forward
            receiver, receiver_input_port, graph_receiver, receiver_mechanism = \
                self._get_nested_node_CIM_port(receiver_mechanism, receiver_input_port, NodeRole.INPUT)

            nested_compositions.append(graph_receiver)
            # Otherwise, there was a mistake in the spec
            if receiver is None:
                # raise CompositionError(f"receiver arg ({repr(receiver_arg)}) in call to add_projection method of "
                #                        f"{self.name} is not in it or any of its nested {Composition.__name__}s.")
                if isinstance(receiver_arg, Port):
                    receiver_str = f"{receiver_arg} of {receiver_arg.owner}"
                else:
                    receiver_str = f"{receiver_arg}"
                raise CompositionError(f"{receiver_str}, specified as receiver of {Projection.__name__} from "
                                       f"{sender.name}, is not in {self.name} or any {Composition.__name__}s nested "
                                       f"within it.")

        return receiver, receiver_mechanism, graph_receiver, receiver_input_port, \
               nested_compositions, learning_projection

    def _get_original_senders(self, input_port, projections):
        original_senders = set()
        for original_projection in projections:
            if original_projection in self.projections:
                original_senders.add(original_projection.sender)
                correct_sender = original_projection.sender
                shadow_found = False
                for shadow_projection in input_port.path_afferents:
                    if shadow_projection.sender == correct_sender:
                        shadow_found = True
                        break
                if not shadow_found:
                    # TBI - Shadow projection type? Matrix value?
                    new_projection = MappingProjection(sender=correct_sender,
                                                       receiver=input_port)
                    self.add_projection(new_projection, sender=correct_sender, receiver=input_port)
        return original_senders

    def _update_shadow_projections(self, context=None):
        for node in self.nodes:
            for input_port in node.input_ports:
                if input_port.shadow_inputs:
                    original_senders = self._get_original_senders(input_port, input_port.shadow_inputs.path_afferents)
                    for shadow_projection in input_port.path_afferents:
                        if shadow_projection.sender not in original_senders:
                            self.remove_projection(shadow_projection)

            # If the node does not have any roles, it is internal
            if len(self.get_roles_by_node(node)) == 0:
                self._add_node_role(node, NodeRole.INTERNAL)

    def _check_for_projection_assignments(self, context=None):
        """Check that all Projections and Ports with require_projection_in_composition attribute are configured.

        Validate that all InputPorts with require_projection_in_composition == True have an afferent Projection.
        Validate that all OuputStates with require_projection_in_composition == True have an efferent Projection.
        Validate that all Projections have senders and receivers.
        """
        projections = self.projections.copy()

        for node in self.nodes:
            if isinstance(node, Projection):
                projections.append(node)
                continue
            for input_port in node.input_ports:
                if input_port.require_projection_in_composition and not input_port.path_afferents:
                    warnings.warn(f'{InputPort.__name__} ({input_port.name}) of {node.name} '
                                  f'doesn\'t have any afferent {Projection.__name__}s')
            for output_port in node.output_ports:
                if output_port.require_projection_in_composition and not output_port.efferents:
                    warnings.warn(f'{OutputPort.__name__} ({output_port.name}) of {node.name} '
                                  f'doesn\'t have any efferent {Projection.__name__}s in {self.name}')

        for projection in projections:
            if not projection.sender:
                warnings.warn(f'{Projection.__name__} {projection.name} is missing a sender')
            if not projection.receiver:
                warnings.warn(f'{Projection.__name__} {projection.name} is missing a receiver')

    def _check_for_existing_projections(self,
                                       projection=None,
                                       sender=None,
                                       receiver=None,
                                       in_composition:bool=True):
        """Check for Projection with same sender and receiver
        If **in_composition** is True, return only Projections found in the current Composition
        If **in_composition** is False, return only Projections that are found outside the current Composition

        Return Projection or list of Projections that satisfies the conditions, else False
        """
        assert projection or (sender and receiver), \
            f'_check_for_existing_projection must be passed a projection or a sender and receiver'

        if projection:
            sender = projection.sender
            receiver = projection.receiver
        else:
            if isinstance(sender, Mechanism):
                sender = sender.output_port
            elif isinstance(sender, Composition):
                sender = sender.output_CIM.output_port
            if isinstance(receiver, Mechanism):
                receiver = receiver.input_port
            elif isinstance(receiver, Composition):
                receiver = receiver.input_CIM.input_port
        existing_projections = [proj for proj in sender.efferents if proj.receiver is receiver]
        existing_projections_in_composition = [proj for proj in existing_projections if proj in self.projections]
        assert len(existing_projections_in_composition) <= 1, \
            f"PROGRAM ERROR: More than one identical projection found " \
            f"in {self.name}: {existing_projections_in_composition}."
        if in_composition:
            if existing_projections_in_composition:
                return existing_projections_in_composition[0]
        else:
            if existing_projections and not existing_projections_in_composition:
                return existing_projections
        return False

    def _check_feedback(self, scheduler, context=None):
        # FIX: 10/2/19 - SHOULD REALLY HANDLE THIS BY DETECTING LOOPS DIRECTLY
        """Check that feedback specification is required for projections to which it has been assigned
        Rationale:
            if, after removing the feedback designation of a Projection, structural and functional dependencies
            are the same, then the designation is not needed so remove it.
        Note:
        - graph_processing.dependency_dict is used as indication of structural dependencies
        - scheduler.dependency_dict is used as indication of functional (execution) dependencies
        """

        if scheduler:
            # If an external scheduler is provided, update it with current processing graph
            try:
                scheduler._init_consideration_queue_from_graph(self.graph_processing)
            # Ignore any cycles at this point
            except ValueError:
                pass
        else:
            scheduler = self.scheduler

        already_tested = []
        for vertex in [v for v in self.graph.vertices if v.feedback==MAYBE]:
            # projection = vertex.component
            # assert isinstance(projection, Projection), \
            #     f'PROGRAM ERROR: vertex identified with feedback=True that is not a Projection'
            if vertex in already_tested:
                continue

            v_set = [v for v in self.graph.vertices
                     if (v.feedback==MAYBE
                         and v.component.sender.owner is vertex.component.sender.owner)]

            for v in v_set:
                v.feedback = False

            # Update Composition's graph_processing
            self._update_processing_graph()

            # Update scheduler's consideration_queue based on update of graph_processing to detect any new cycles
            try:
                scheduler._init_consideration_queue_from_graph(self.graph_processing)
            except ValueError:
                # If a cycle is detected, leave feedback alone
                feedback = 'leave'

            # If, when feedback is False, the dependency_dicts for the structural and execution are the same,
            #    then no need for feedback specification, so remove it
            #       and remove assignments of sender and receiver to corresponding feedback entries of Composition
            if self.graph_processing.dependency_dict == scheduler.dependency_dict:
                feedback = 'remove'
            else:
                feedback = 'leave'

            # Remove nodes that send and receive feedback Projection from feedback_senders and feedback_receivers lists
            if feedback == 'remove':
                self.feedback_senders.remove(v.component.sender.owner)
                self.feedback_receivers.remove(v.component.receiver.owner)
            # Otherwise, restore feedback assignment and scheduler's consideration_queue
            else:
                for v in v_set:
                    v.feedback = True
                self._update_processing_graph()
                scheduler._init_consideration_queue_from_graph(self.graph_processing)
            already_tested.extend(v_set)


    # ******************************************************************************************************************
    #                                            PATHWAYS
    # ******************************************************************************************************************


    # -----------------------------------------  PROCESSING  -----------------------------------------------------------

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

    def add_linear_processing_pathway(self, pathway, *args):
        """Add sequence of Mechanisms or Compositions possibly with intercolated Projections

        A `MappingProjection` is created for each contiguous pair of `Mechanisms <Mechanism>` and/or Compositions
        in the **pathway** argument, from the `primary OutputPort <OutputPort_Primary>` of the first one to the
        `primary InputPort <InputPort_Primary>` of the second.

        Tuples (Mechanism, `NodeRoles <NodeRole>`) can be used to assign `required_roles
        <Composition.add_node.required_roles>` to Mechanisms.

        Note that any specifications of a ControlMechanism's **monitor_for_control** `argument
        <ControlMechanism_Monitor_for_Control_Argument>` or the **monitor** argument specified in the constructor
        for an ObjectiveMechanism in the **objective_mechanism** `argument <ControlMechanism_ObjectiveMechanism>`
        supercede any MappingProjections that would otherwise be created for them when specified in the **pathway**
        argument.
        """
        nodes = []

        from psyneulink.core.globals.keywords import PROJECTION, NODE
        def is_spec(entry, desired_type:tc.enum(NODE, PROJECTION)):
            """Test whether pathway entry is specified type (NODE or PROJECTION)"""
            node_specs = (Mechanism, Composition)
            proj_specs = (Projection, np.ndarray, np.matrix, str, list)
            if desired_type == NODE:
                if (isinstance(entry, node_specs)
                        or (isinstance(entry, tuple)
                            and isinstance(entry[0], node_specs)
                            and isinstance(entry[1], NodeRole))):
                    return True
            elif desired_type == PROJECTION:
                if (isinstance(entry, proj_specs)
                        or (isinstance(entry, tuple)
                            and isinstance(entry[0], proj_specs)
                            and entry[1] in {True, False, MAYBE})):
                    return True
            else:
                return False

        # First, verify that the pathway begins with a node
        if not isinstance(pathway, (list, tuple)):
            raise CompositionError(f"First argument in add_linear_processing_pathway method of '{self.name}' "
                                   f"{Composition.__name__} must be a list of nodes")

        # Then make sure the first item is a node and not a Projection
        if is_spec(pathway[0], NODE):
            self.add_nodes([pathway[0]]) # Use add_nodes so that node spec can also be a tuple with required_roles
            nodes.append(pathway[0])
        else:
            # 'MappingProjection has no attribute _name' error is thrown when pathway[0] is passed to the error msg
            raise CompositionError("The first item in a linear processing pathway must be a Node (Mechanism or "
                                   "Composition).")

        # Then, add all of the remaining nodes in the pathway
        for c in range(1, len(pathway)):
            # if the current item is a Mechanism, Composition or (Mechanism, NodeRole(s)) tuple, add it
            if is_spec(pathway[c], NODE):
                self.add_nodes([pathway[c]])
                nodes.append(pathway[c])

        # FIX 8/27/19 [JDC]:  GENERALIZE TO ControlMechanism
        # MODIFIED 8/12/19 NEW: [JDC] - AVOID DUPLCIATE CONTROL_RELATED PROJECTIONS
        # Then, delete any ControlMechanism that has its monitor_for_control attribute assigned
        #    and any ObjectiveMechanism that projects to a ControlMechanism,
        #    as well as any projections to them specified in the pathway;
        #    this is to avoid instantiating projections to them that might conflict with those
        #    instantiated by their constructors or, for a controller, _add_controller()
        items_to_delete = []
        for i, item in enumerate(pathway):
            if ((isinstance(item, ControlMechanism) and item.monitor_for_control)
                    or (isinstance(item, ObjectiveMechanism) and item._role == CONTROL)):
                items_to_delete.append(item)
                # Delete any projections to the ControlMechanism or ObjectiveMechanism specified in pathway
                if i>0 and is_spec(pathway[i - 1],PROJECTION):
                    items_to_delete.append(pathway[i - 1])
        for item in items_to_delete:
            if isinstance(item, ControlMechanism):
                arg_name = f'in the {repr(MONITOR_FOR_CONTROL)} of its constructor'
            else:
                arg_name = f'either in the {repr(MONITOR)} arg of its constructor, ' \
                           f'or in the {repr(MONITOR_FOR_CONTROL)} arg of its associated {ControlMechanism.__name__}'
            warnings.warn(f'No new {Projection.__name__}s were added to {item.name} that was included in '
                          f'the {repr(PATHWAY)} arg of add_linear_processing_pathway for {self.name}, '
                          f'since there were ones already specified {arg_name}.')
            del pathway[pathway.index(item)]
        # MODIFIED 8/12/19 END

        # Then, loop through pathway and validate that the Mechanism-Projection relationships make sense
        # and add MappingProjection(s) where needed
        projections = []
        for c in range(1, len(pathway)):

            # if the current item is a Node
            if is_spec(pathway[c], NODE):
                if is_spec(pathway[c - 1], NODE):
                    # if the previous item was also a node, add a MappingProjection between them
                    if isinstance(pathway[c - 1], tuple):
                        sender = pathway[c - 1][0]
                    else:
                        sender = pathway[c - 1]
                    if isinstance(pathway[c], tuple):
                        receiver = pathway[c][0]
                    else:
                        receiver = pathway[c]
                    proj = self.add_projection(sender=sender,
                                               receiver=receiver)
                    if proj:
                        projections.append(proj)

            # if the current item is a Projection specification
            elif is_spec(pathway[c], PROJECTION):
                if c == len(pathway) - 1:
                    raise CompositionError("{} is the last item in the pathway. A projection cannot be the last item in"
                                           " a linear processing pathway.".format(pathway[c]))
                # confirm that it is between two nodes, then add the projection
                if isinstance(pathway[c], tuple):
                    proj = pathway[c][0]
                    feedback = pathway[c][1]
                else:
                    proj = pathway[c]
                    feedback = False
                sender = pathway[c - 1]
                receiver = pathway[c + 1]
                if isinstance(sender, (Mechanism, Composition)) \
                        and isinstance(receiver, (Mechanism, Composition)):
                    try:
                        if isinstance(proj, (np.ndarray, np.matrix, list)):
                            proj = MappingProjection(sender=sender,
                                                     matrix=proj,
                                                     receiver=receiver)
                    except DuplicateProjectionError:
                        # FIX: 7/22/19 ADD WARNING HERE??
                        # FIX: 7/22/19 MAKE THIS A METHOD ON Projection??
                        duplicate = [p for p in receiver.afferents if p in sender.efferents]
                        assert len(duplicate)==1, \
                            f"PROGRAM ERROR: Could not identify duplicate on DuplicateProjectionError " \
                                f"for {Projection.__name__} between {sender.name} and {receiver.name} " \
                                f"in call to {repr('add_linear_processing_pathway')} for {self.name}."
                        duplicate = duplicate[0]
                        warning_msg = f"Projection specified between {sender.name} and {receiver.name} " \
                                      f"in call to 'add_linear_projection' for {self.name} is a duplicate of one"
                        # IMPLEMENTATION NOTE: Version that allows different Projections between same
                        #                      sender and receiver in different Compositions
                        # if duplicate in self.projections:
                        #     warnings.warn(f"{warning_msg} already in the Composition ({duplicate.name}) "
                        #                   f"and so will be ignored.")
                        #     proj=duplicate
                        # else:
                        #     if self.prefs.verbosePref:
                        #         warnings.warn(f" that already exists between those nodes ({duplicate.name}). The "
                        #                       f"new one will be used; delete it if you want to use the existing one")
                        # Version that forbids *any* duplicate Projections between same sender and receiver
                        warnings.warn(f"{warning_msg} that already exists between those nodes ({duplicate.name}) "
                                      f"and so will be ignored.")
                        proj=duplicate

                    proj = self.add_projection(projection=proj,
                                               sender=sender,
                                               receiver=receiver,
                                               feedback=feedback,
                                               allow_duplicates=False)
                    if proj:
                        projections.append(proj)

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
            explicit_pathway.append(nodes[i + 1])

        return explicit_pathway


    # ------------------------------------------  LEARNING  ------------------------------------------------------------

    def add_linear_learning_pathway(self,
                                    pathway,
                                    learning_function,
                                    loss_function=None,
                                    learning_rate=0.05,
                                    error_function=LinearCombination(),
                                    learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE):
        """Implement learning pathway (including necessary `learning components <Composition_Learning_Components>`.

        Generic method for implementing a learning pathway.  Calls `add_linear_processing_pathway` to implement
        the processing components including, if necessary, the MappingProjections between Mechanisms.  All of the
        MappingProjections (whether specified or created) are subject to learning (and are assigned as the
        `learned_projection <LearningMechanism.learned_projection>` attribute of the `LearningMechanisms
        <LeaningMechanisms>` created for the pathway.

        If **learning_function** is a sublcass of `LearningFunction <LearningFunctions>`, a class-specific
        `learning method <Composition_Learning_Methods>` is called.  Some may allow the error_function
        to be specified, in which case it must be compatible with the class of LearningFunction specified.

        If **learning_function** an instantiated function, it is assigned to all of the `LearningMechanisms
        <LearningMechanism>` created for the MappingProjections in the pathway.  A `ComparatorMechanism` is
        created to compute the error for the pathway, and assigned the function specified in **error_function**,
        which must be compatible with **learning_function**.

        See `Composition_Learning` for for a more detailed description of how learning is implemented in a
        Composition, including the `learning components` <Composition_Learning_Components>` that are created,
        as well as other `learning methods <Composition_Learning_Methods>` that can be used to implement specific
        algorithms.

        Arguments
        ---------

        pathway: List
            list containing either [Node1, Node2] or [Node1, MappingProjection, Node2]. If a projection is
            specified, that projection is the learned projection. Otherwise, a default MappingProjection is
            automatically generated for the learned projection.

        learning_rate : float : default 0.05
            specifies the `learning_rate <LearningMechanism.learning_rate>` used for the **learning_function**
            of the `LearningMechanism` in the **pathway**.

        error_function : function : default LinearCombination
            specifies the function assigned to Mechanism used to compute the error from the target and the output
            (`value <Mechanism_Base.value>`) of the `TARGET` Mechanism in the **pathway**.

            .. note::
               For most learning algorithms (and by default), a `ComparatorMechanism` is used to compute the error.
               However, some learning algorithms may use a different Mechanism (e.g., for `TDlearning` a
               `PredictionErrorMechanism` is used, which uses as its fuction `PredictionErrorDeltaFunction`.

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

        if isinstance(learning_function, type) and issubclass(learning_function, BackPropagation):
            return self._create_backpropagation_learning_pathway(pathway,
                                                                 loss_function,
                                                                 learning_rate,
                                                                 error_function,
                                                                 learning_update)

        # Processing Components
        input_source, output_source, learned_projection = \
            self._unpack_processing_components_of_learning_pathway(pathway)
        self.add_linear_processing_pathway([input_source, learned_projection, output_source])

        # FIX: CONSOLIDATE LEARNING - WAS SPECIFIC TO RL AND NOT IN TD
        self.add_required_node_role(output_source, NodeRole.OUTPUT)

        # Learning Components
        target, comparator, learning_mechanism = self._create_learning_related_mechanisms(input_source,
                                                                                          output_source,
                                                                                          error_function,
                                                                                          learning_function,
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

        # Update graph in case method is called again
        self._analyze_graph()

        return learning_related_components

    def add_reinforcement_learning_pathway(self, pathway, learning_rate=0.05, error_function=None,
                                           learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE):
        """Convenience method that calls `add_linear_learning_pathway` with **learning_function**=`Reinforcement`

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

        return self.add_linear_learning_pathway(pathway,
                                                learning_rate=learning_rate,
                                                learning_function=Reinforcement,
                                                error_function=error_function,
                                                learning_update=learning_update)

    def add_td_learning_pathway(self, pathway, learning_rate=0.05, error_function=None,
                                learning_update:tc.any(bool, tc.enum(ONLINE, AFTER))=ONLINE):
        """Convenience method that calls `add_linear_learning_pathway` with **learning_function**=`TDLearning`

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

        return self.add_linear_learning_pathway(pathway,
                                                learning_rate=learning_rate,
                                                learning_function=TDLearning,
                                                learning_update=learning_update)

    def add_backpropagation_learning_pathway(self,
                                             pathway,
                                             learning_rate=0.05,
                                             error_function=None,
                                             loss_function:tc.enum(MSE,SSE)=MSE,
                                             learning_update:tc.optional(tc.any(bool, tc.enum(ONLINE, AFTER)))=AFTER):
        """Convenience method that calls `add_linear_learning_pathway` with **learning_function**=`Backpropagation`

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

        return self.add_linear_learning_pathway(pathway,
                                                learning_rate=learning_rate,
                                                learning_function=BackPropagation,
                                                loss_function=loss_function,
                                                error_function=error_function,
                                                learning_update=learning_update)

    # NOTES:
    # Learning-type-specific creation methods should:
    # - create ComparatorMechanism and pass in as error_source (for 1st LearningMechanism in sequence in bp)
    # - Determine and pass error_sources (aka previous_learning_mechanism) (for bp)
    # - construct and pass in the learning_function
    # - do the following for last LearningMechanism in sequence:
    #   learning_mechanism.output_ports[ERROR_SIGNAL].parameters.require_projection_in_composition._set(False,
    #                                                                                                    override=True)
    #
    # Create_backprop... should pass error_function (handled by kwargs below)
    # Check for existence of Learning mechanism (or do this in creation method?);  if one exists, compare its
    #    ERROR_SIGNAL input_ports with error_sources and update/add any needed, as well as corresponding
    #    error_matrices (from their learned_projections) -- do so using LearningMechanism's add_ports method);
    #    create projections from each
    # Move creation of LearningProjections and learning-related projections (MappingProjections) here
    # ?Do add_nodes and add_projections here or in Learning-type-specific creation methods

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


    # FIX: NOT CURRENTLY USED; IMPLEMENTED FOR FUTURE USE IN GENERALIZATION OF LEARNING METHODS
    def _create_learning_components(self,
                                    sender_activity_source,   # aka input_source
                                    receiver_activity_source, # aka output_source
                                    error_sources,            # aka comparator/previous_learning_mechanism
                                    learning_function,
                                    learned_projection,
                                    learning_rate,
                                    learning_update,
                                    target_mech=None,
                                    **kwargs                  # Use of type-specific learning arguments
                                    ):

        # ONLY DO THIS IF ONE DOESN'T ALREADY EXIST (?pass in argument determing this?)
        learning_mechanism = LearningMechanism(function=learning_function,
                                               default_variable=[sender_activity_source.output_ports[0].value,
                                                                 receiver_activity_source.output_ports[0].value,
                                                                 error_sources.output_ports[0].value],
                                               error_sources=error_sources,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name,
                                               **kwargs)


        return learning_mechanism

    def _create_learning_related_mechanisms(self,
                                            input_source,
                                            output_source,
                                            error_function,
                                            learning_function,
                                            learned_projection,
                                            learning_rate,
                                            learning_update):
        """Creates *TARGET_MECHANISM*, *COMPARATOR_MECHANISM* and *LEARNING_MECHANISM* for RL and TD learning"""

        if isinstance(learning_function, type):
            if issubclass(learning_function, TDLearning):
                creation_method = self._create_td_related_mechanisms
            elif issubclass(learning_function, Reinforcement):
                creation_method = self._create_rl_related_mechanisms
            else:
                raise CompositionError(f"'learning_function' argument for add_linear_learning_pathway "
                                       f"({learning_function}) must be a class of {LearningFunction.__name__}")

            target_mechanism, comparator_mechanism, learning_mechanism  = creation_method(input_source,
                                                                                          output_source,
                                                                                          error_function,
                                                                                          learned_projection,
                                                                                          learning_rate,
                                                                                          learning_update)

        elif is_function_type(learning_function):
            target_mechanism = ProcessingMechanism(name='Target')
            comparator_mechanism = ComparatorMechanism(name='Comparator',
                                                       sample={NAME: SAMPLE,
                                                               VARIABLE: [0.], WEIGHT: -1},
                                                       target={NAME: TARGET,
                                                               VARIABLE: [0.]},
                                                       function=error_function,
                                                       output_ports=[OUTCOME, MSE])
            learning_mechanism = LearningMechanism(
                                    function=learning_function(
                                                         default_variable=[input_source.output_ports[0].value,
                                                                           output_source.output_ports[0].value,
                                                                           comparator_mechanism.output_ports[0].value],
                                                         learning_rate=learning_rate),
                                    default_variable=[input_source.output_ports[0].value,
                                                      output_source.output_ports[0].value,
                                                      comparator_mechanism.output_ports[0].value],
                                    error_sources=comparator_mechanism,
                                    learning_enabled=learning_update,
                                    in_composition=True,
                                    name="Learning Mechanism for " + learned_projection.name)
        else:
            raise CompositionError(f"'learning_function' argument of add_linear_learning_pathway "
                                   f"({learning_function}) must be a class of {LearningFunction.__name__} or a "
                                   f"learning-compatible function")

        learning_mechanism.output_ports[ERROR_SIGNAL].parameters.require_projection_in_composition._set(False,
                                                                                                         override=True)
        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_learning_related_projections(self, input_source, output_source, target, comparator, learning_mechanism):
        """Construct MappingProjections among `learning components <Composition_Learning_Components>` for pathway"""

        # FIX 5/29/19 [JDC]:  INTEGRATE WITH _get_back_prop_error_sources (RIGHT NOW, ONLY CALLED FOR TERMINAL SEQUENCE)
        try:
            sample_projection = MappingProjection(sender=output_source, receiver=comparator.input_ports[SAMPLE])
        except DuplicateProjectionError:
            sample_projection = [p for p in output_source.efferents
                                 if p in comparator.input_ports[SAMPLE].path_afferents]
        try:
            target_projection = MappingProjection(sender=target, receiver=comparator.input_ports[TARGET])
        except DuplicateProjectionError:
            target_projection = [p for p in target.efferents
                                 if p in comparator.input_ports[TARGET].path_afferents]
        act_in_projection = MappingProjection(sender=input_source.output_ports[0],
                                              receiver=learning_mechanism.input_ports[ACTIVATION_INPUT_INDEX])
        act_out_projection = MappingProjection(sender=output_source.output_ports[0],
                                               receiver=learning_mechanism.input_ports[ACTIVATION_OUTPUT_INDEX])
        # FIX CROSS_PATHWAYS 7/28/19 [JDC]: THIS MAY NEED TO USE add_ports (SINCE ONE MAY EXIST; CONSTRUCT TEST FOR IT)
        error_signal_projection = MappingProjection(sender=comparator.output_ports[OUTCOME],
                                                    receiver=learning_mechanism.input_ports[ERROR_SIGNAL_INDEX])
        return [target_projection, sample_projection, error_signal_projection, act_out_projection, act_in_projection]

    def _create_learning_projection(self, learning_mechanism, learned_projection):
        """Construct LearningProjections from LearningMechanisms to learned_projections in processing pathway"""

        learning_projection = LearningProjection(name="Learning Projection",
                                                 sender=learning_mechanism.learning_signals[0],
                                                 receiver=learned_projection.parameter_ports["matrix"])

        learned_projection.has_learning_projection = True

        return learning_projection

    def _create_rl_related_mechanisms(self,
                                      input_source,
                                      output_source,
                                      error_function,
                                      learned_projection,
                                      learning_rate,
                                      learning_update):

        target_mechanism = ProcessingMechanism(name='Target')

        comparator_mechanism = ComparatorMechanism(name='Comparator',
                                                   sample={NAME: SAMPLE,
                                                           VARIABLE: [0.], WEIGHT: -1},
                                                   target={NAME: TARGET,
                                                           VARIABLE: [0.]},
                                                   function=error_function,
                                                   output_ports=[OUTCOME, MSE])

        learning_mechanism = \
            LearningMechanism(function=Reinforcement(default_variable=[input_source.output_ports[0].value,
                                                                       output_source.output_ports[0].value,
                                                                       comparator_mechanism.output_ports[0].value],
                                                     learning_rate=learning_rate),
                              default_variable=[input_source.output_ports[0].value,
                                                output_source.output_ports[0].value,
                                                comparator_mechanism.output_ports[0].value],
                              error_sources=comparator_mechanism,
                              learning_enabled=learning_update,
                              in_composition=True,
                              name="Learning Mechanism for " + learned_projection.name)

        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_td_related_mechanisms(self,
                                      input_source,
                                      output_source,
                                      error_function,
                                      learned_projection,
                                      learning_rate,
                                      learning_update):

        target_mechanism = ProcessingMechanism(name='Target',
                                               default_variable=output_source.defaults.value)

        comparator_mechanism = PredictionErrorMechanism(name='PredictionError',
                                                        sample={NAME: SAMPLE,
                                                                VARIABLE: output_source.defaults.value},
                                                        target={NAME: TARGET,
                                                                VARIABLE: output_source.defaults.value},
                                                        function=PredictionErrorDeltaFunction(gamma=1.0))

        learning_mechanism = LearningMechanism(function=TDLearning(learning_rate=learning_rate),
                                               default_variable=[input_source.output_ports[0].defaults.value,
                                                                 output_source.output_ports[0].defaults.value,
                                                                 comparator_mechanism.output_ports[0].defaults.value],
                                               error_sources=comparator_mechanism,
                                               learning_enabled=learning_update,
                                               in_composition=True,
                                               name="Learning Mechanism for " + learned_projection.name)

        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_backpropagation_learning_pathway(self, pathway, loss_function, learning_rate=0.05, error_function=None,
                                             learning_update:tc.optional(tc.any(bool, tc.enum(ONLINE, AFTER)))=AFTER):

        # FIX: LEARNING CONSOLIDATION - Can get rid of this:
        if not error_function:
            error_function = LinearCombination()

        # Add pathway to graph and get its full specification (includes all ProcessingMechanisms and MappingProjections)
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

            # FIX CROSSED_PATHWAYS 7/28/19 [JDC]:
            #  THIS SHOULD BE INTEGRATED WITH CALL TO _create_terminal_backprop_learning_components
            #  ** NEED TO CHECK WHETHER LAST NODE IN THE SEQUENCE IS TERMINAL AND IF SO:
            #     ASSIGN USING: self.add_required_node_role(output_source, NodeRole.OUTPUT)
            # If learned_projection already has a LearningProjection (due to pathway overlap),
            #    use those terminal sequence components
            if (learned_projection.has_learning_projection
                    and any([lp for lp in learned_projection.parameter_ports[MATRIX].mod_afferents
                             if lp in self.projections])):
                target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
                comparator = self._terminal_backprop_sequences[output_source][COMPARATOR_MECHANISM]
                learning_mechanism = self._terminal_backprop_sequences[output_source][LEARNING_MECHANISM]

            # Otherwise, create new ones
            else:
                target, comparator, learning_mechanism = \
                    self._create_terminal_backprop_learning_components(input_source,
                                                                       output_source,
                                                                       error_function,
                                                                       loss_function,
                                                                       learned_projection,
                                                                       learning_rate,
                                                                       learning_update)
            sequence_end = path_length - 3

        # # FIX: ALTERNATIVE IS TO TEST WHETHER IT PROJECTIONS TO ANY MECHANISMS WITH LEARNING ROLE
        # Otherwise, if output_source already projects to a LearningMechanism, integrate with existing sequence
        elif any(isinstance(p.receiver.owner, LearningMechanism) for p in output_source.efferents):
            # Set learning_mechanism to the one to which output_source projects
            learning_mechanism = next((p.receiver.owner for p in output_source.efferents
                                       if isinstance(p.receiver.owner, LearningMechanism)))
            # # Use existing target and comparator to learning_mechanism for Mechanism to which output_source project
            # target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
            # comparator = self._terminal_backprop_sequences[output_source][COMPARATOR_MECHANISM]
            target = None
            comparator = None
            sequence_end = path_length - 1

        # Otherwise create terminal_sequence for the sequence,
        #    and eliminate existing terminal_sequences previously created for Mechanisms now in the pathway
        else:
            # Eliminate existing comparators and targets for Mechanisms now in the pathway that were output_sources
            #   (i.e., ones that belong to previously-created sequences that overlap with the current one)
            for pathway_mech in [m for m in pathway if isinstance(m, Mechanism)]:

                old_comparator = next((p.receiver.owner for p in pathway_mech.efferents
                                       if (isinstance(p.receiver.owner, ComparatorMechanism)
                                           and p.receiver.owner in self.get_nodes_by_role(NodeRole.LEARNING))),
                                      None)
                if old_comparator:
                    old_target = next((p.sender.owner for p in old_comparator.input_ports[TARGET].path_afferents
                                       if p.sender.owner in self.get_nodes_by_role(NodeRole.TARGET)),
                                      None)
                    self.remove_nodes([old_comparator, old_target])
                    # FIX CROSSING_PATHWAYS [JDC]: MAKE THE FOLLOWING A METHOD?
                    # Collect InputPorts that received error_signal projections from the old_comparator
                    #    and delete after old_comparator has been deleted
                    #    (i.e., after those InputPorts have been vacated)
                    old_error_signal_input_ports = []
                    for error_projection in old_comparator.output_port.efferents:
                        old_error_signal_input_ports.append(error_projection.receiver)
                    Mechanism_Base._delete_mechanism(old_comparator)
                    Mechanism_Base._delete_mechanism(old_target)
                    for input_port in old_error_signal_input_ports:
                        input_port.owner.remove_ports(input_port)
                    del self._terminal_backprop_sequences[pathway_mech]
                    del self.required_node_roles[self.required_node_roles.index((pathway_mech, NodeRole.OUTPUT))]

            # Create terminal_sequence
            target, comparator, learning_mechanism = \
                self._create_terminal_backprop_learning_components(input_source,
                                                                   output_source,
                                                                   error_function,
                                                                   loss_function,
                                                                   learned_projection,
                                                                   learning_rate,
                                                                   learning_update)
            self._terminal_backprop_sequences[output_source] = {LEARNING_MECHANISM: learning_mechanism,
                                                                TARGET_MECHANISM: target,
                                                                COMPARATOR_MECHANISM: comparator}
            self.add_required_node_role(pathway[-1], NodeRole.OUTPUT)

            sequence_end = path_length - 3

        # loop backwards through the rest of the pathway to create and connect
        # the remaining learning mechanisms
        learning_mechanisms = [learning_mechanism]
        learned_projections = [learned_projection]
        for i in range(sequence_end, 1, -2):
            # set variables for this iteration
            input_source = processing_pathway[i - 2]
            learned_projection = processing_pathway[i - 1]
            output_source = processing_pathway[i]

            learning_mechanism = self._create_non_terminal_backprop_learning_components(input_source,
                                                                                        output_source,
                                                                                        learned_projection,
                                                                                        learning_rate,
                                                                                        learning_update)
            learning_mechanisms.append(learning_mechanism)
            learned_projections.append(learned_projection)

        # Add error_signal projections to any learning_mechanisms that are now dependent on the new one
        for lm in learning_mechanisms:
            if lm.dependent_learning_mechanisms:
                projections = self._add_error_projection_to_dependent_learning_mechs(lm)
                self.add_projections(projections)

        # Suppress "no efferent connections" warning for:
        #    - error_signal OutputPort of last LearningMechanism in sequence
        #    - comparator
        learning_mechanisms[-1].output_ports[ERROR_SIGNAL].parameters.require_projection_in_composition.set(False,
                                                                                                             override=True)
        if comparator:
            for s in comparator.output_ports:
                s.parameters.require_projection_in_composition.set(False,
                                                                   override=True)

        learning_related_components = {LEARNING_MECHANISM: learning_mechanisms,
                                       COMPARATOR_MECHANISM: comparator,
                                       TARGET_MECHANISM: target,
                                       LEARNED_PROJECTION: learned_projections}

        # Update graph in case method is called again
        self._analyze_graph()

        return learning_related_components

    def infer_backpropagation_learning_pathways(self):
        """Convenience method that automatically creates backpropapagation learning pathways for every Input Node --> Output Node pathway"""
        self._analyze_graph()
        # returns a list of all pathways from start -> output node
        def bfs(start):
            pathways = []
            prev = {}
            queue = collections.deque([start])
            while len(queue) > 0:
                curr_node = queue.popleft()
                if NodeRole.OUTPUT in self.get_roles_by_node(curr_node):
                    p = []
                    while curr_node in prev:
                        p.insert(0, curr_node)
                        curr_node = prev[curr_node]
                    p.insert(0, curr_node)
                    pathways.append(p)
                    continue
                for projection, efferent_node in [(p, p.receiver.owner) for p in curr_node.efferents]:
                    if (not hasattr(projection,'learnable')) or (projection.learnable is False):
                        continue
                    prev[efferent_node] = projection
                    prev[projection] = curr_node
                    queue.append(efferent_node)
            return pathways

        pathways = [p for n in self.get_nodes_by_role(NodeRole.INPUT) if NodeRole.TARGET not in self.get_roles_by_node(n) for p in bfs(n)]
        for pathway in pathways:
            self.add_backpropagation_learning_pathway(pathway=pathway)

    def _create_terminal_backprop_learning_components(self,
                                                      input_source,
                                                      output_source,
                                                      error_function,
                                                      loss_function,
                                                      learned_projection,
                                                      learning_rate,
                                                      learning_update):
        """Create ComparatorMechanism, LearningMechanism and LearningProjection for Component in learning sequence"""

        # target = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
        # comparator = self._terminal_backprop_sequences[output_source][COMPARATOR_MECHANISM]
        # learning_mechanism = self._terminal_backprop_sequences[output_source][LEARNING_MECHANISM]

        # If target and comparator already exist (due to overlapping pathway), use those
        try:
            target_mechanism = self._terminal_backprop_sequences[output_source][TARGET_MECHANISM]
            comparator_mechanism = self._terminal_backprop_sequences[output_source][COMPARATOR_MECHANISM]

        # Otherwise, create new ones
        except KeyError:
            target_mechanism = ProcessingMechanism(name='Target',
                                                   default_variable=output_source.output_ports[0].value)
            comparator_mechanism = ComparatorMechanism(name='Comparator',
                                                       target={NAME: TARGET,
                                                               VARIABLE: target_mechanism.output_ports[0].value},
                                                       sample={NAME: SAMPLE,
                                                               VARIABLE: output_source.output_ports[0].value,
                                                               WEIGHT: -1},
                                                       function=error_function,
                                                       output_ports=[OUTCOME, MSE])

        learning_function = BackPropagation(default_variable=[input_source.output_ports[0].value,
                                                              output_source.output_ports[0].value,
                                                              comparator_mechanism.output_ports[0].value],
                                            activation_derivative_fct=output_source.function.derivative,
                                            learning_rate=learning_rate,
                                            loss_function=loss_function)

        learning_mechanism = LearningMechanism(function=learning_function,
                                               default_variable=[input_source.output_ports[0].value,
                                                                 output_source.output_ports[0].value,
                                                                 comparator_mechanism.output_ports[0].value],
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


        return target_mechanism, comparator_mechanism, learning_mechanism

    def _create_non_terminal_backprop_learning_components(self,
                                                          input_source,
                                                          output_source,
                                                          learned_projection,
                                                          learning_rate,
                                                          learning_update):

        # Get existing LearningMechanism if one exists (i.e., if this is a crossing point with another pathway)
        learning_mechanism = \
            next((lp.sender.owner for lp in learned_projection.parameter_ports[MATRIX].mod_afferents
                  if isinstance(lp, LearningProjection)),
                 None)

        # If learning_mechanism exists:
        #    error_sources will be empty (as they have been dealt with in self._get_back_prop_error_sources
        #    error_projections will contain list of any created to be added to the Composition below
        if learning_mechanism:
            error_sources, error_projections = self._get_back_prop_error_sources(output_source, learning_mechanism)
        # If learning_mechanism does not yet exist:
        #    error_sources will contain ones needed to create learning_mechanism
        #    error_projections will be empty since they can't be created until the learning_mechanism is created below;
        #    they will be created (using error_sources) when, and determined after learning_mechanism is created below
        else:
            error_sources, error_projections = self._get_back_prop_error_sources(output_source)
            error_signal_template = [error_source.output_ports[ERROR_SIGNAL].value for error_source in error_sources]
            default_variable = [input_source.output_ports[0].value,
                                output_source.output_ports[0].value] + error_signal_template

            learning_function = BackPropagation(default_variable=[input_source.output_ports[0].value,
                                                                  output_source.output_ports[0].value,
                                                                  error_signal_template[0]],
                                                activation_derivative_fct=output_source.function.derivative,
                                                learning_rate=learning_rate)

            learning_mechanism = LearningMechanism(function=learning_function,
                                                   # default_variable=[input_source.output_ports[0].value,
                                                   #                   output_source.output_ports[0].value,
                                                   #                   error_signal_template],
                                                   default_variable=default_variable,
                                                   error_sources=error_sources,
                                                   learning_enabled=learning_update,
                                                   in_composition=True,
                                                   name="Learning Mechanism for " + learned_projection.name)

            # Create MappingProjections from ERROR_SIGNAL OutputPort of each error_source
            #    to corresponding error_input_ports
            for i, error_source in enumerate(error_sources):
                error_projection = MappingProjection(sender=error_source,
                                                     receiver=learning_mechanism.error_signal_input_ports[i])
                error_projections.append(error_projection)

        self.add_node(learning_mechanism, required_roles=NodeRole.LEARNING)
        try:
            act_in_projection = MappingProjection(sender=input_source.output_ports[0],
                                                receiver=learning_mechanism.input_ports[0])
            act_out_projection = MappingProjection(sender=output_source.output_ports[0],
                                                receiver=learning_mechanism.input_ports[1])
            self.add_projections([act_in_projection, act_out_projection] + error_projections)

            learning_projection = self._create_learning_projection(learning_mechanism, learned_projection)
            self.add_projection(learning_projection, feedback=True)
        except DuplicateProjectionError as e:
            # we don't care if there is a duplicate
            return learning_mechanism
        except Exception as e:
            raise e

        return learning_mechanism

    def _get_back_prop_error_sources(self, receiver_activity_mech, learning_mech=None):
        # FIX CROSSED_PATHWAYS [JDC]:  GENERALIZE THIS TO HANDLE COMPARATOR/TARGET ASSIGNMENTS IN BACKPROP
        #                              AND THEN TO HANDLE ALL FORMS OF LEARNING (AS BELOW)
        #  REFACTOR TO DEAL WITH CROSSING PATHWAYS (?CREATE METHOD ON LearningMechanism TO DO THIS?):
        #  1) Determine whether this is a terminal sequence:
        #     - use arg passed in or determine from context (see current implementation in add_backpropagation_learning_pathway)
        #     - for terminal sequence, handle target and sample projections as below
        #  2) For non-terminal sequences, determine # of error_signals coming from LearningMechanisms associated with
        #     all efferentprojections of ProcessingMechanism that projects to ACTIVATION_OUTPUT of LearningMechanism
        #     - check validity of existing error_signal projections with respect to those and, if possible,
        #       their correspondence with error_matrices
        #     - check if any ERROR_SIGNAL input_ports are empty (vacated by terminal sequence elements deleted in
        #       add_projection)
        #     - call add_ports method on LearningMechanism to add new ERROR_SIGNAL input_port to its input_ports
        #       and error_matrix to its self.error_matrices attribute
        #     - add new error_signal projection
        """Add any LearningMechanisms associated with efferent projection from receiver_activity_mech"""
        error_sources = []
        error_projections = []

        # First get all efferents of receiver_activity_mech with a LearningProjection that are in current Composition
        for efferent in [p for p in receiver_activity_mech.efferents
                         if (hasattr(p, 'has_learning_projection')
                             and p.has_learning_projection
                             and p in self.projections)]:
            # Then get any LearningProjections to that efferent that are in current Composition
            for learning_projection in [mod_aff for mod_aff in efferent.parameter_ports[MATRIX].mod_afferents
                                        if (isinstance(mod_aff, LearningProjection) and mod_aff in self.projections)]:
                error_source = learning_projection.sender.owner
                if (error_source not in self.nodes  # error_source is not in the Composition
                        or (learning_mech  # learning_mech passed in
                            # the error_source is already associated with learning_mech
                            and (error_source in learning_mech.error_sources)
                            # and the error_source already sends a Projection to learning_mech
                            and (learning_mech in [p.receiver.owner for p in error_source.efferents]))):
                    continue  # ignore the error_source
                error_sources.append(error_source)

                # If learning_mech was passed in, add error_source to its list of error_signal_input_ports
                if learning_mech:
                    # FIX: REPLACE WITH learning_mech._add_error_signal_input_port ONCE IMPLEMENTED
                    error_signal_input_port = next((e for e in learning_mech.error_signal_input_ports
                                                     if not e.path_afferents), None)
                    if error_signal_input_port is None:
                        error_signal_input_port = learning_mech.add_ports(
                                                            InputPort(projections=error_source.output_ports[ERROR_SIGNAL],
                                                                      name=ERROR_SIGNAL,
                                                                      context=Context(source=ContextFlags.METHOD)),
                                                            context=Context(source=ContextFlags.METHOD))[0]
                    # Create Projection here so that don't have to worry about determining correct
                    #    error_signal_input_port of learning_mech in _create_non_terminal_backprop_learning_components
                    try:
                        error_projections.append(MappingProjection(sender=error_source.output_ports[ERROR_SIGNAL],
                                                               receiver=error_signal_input_port))
                    except DuplicateProjectionError:
                        pass
                    except Exception as e:
                        raise e

        # Return error_sources so they can be used to create a new LearningMechanism if needed
        # Return error_projections created to existing learning_mech
        #    so they can be added to the Composition by _create_non_terminal_backprop_learning_components
        return error_sources, error_projections

    def _get_backprop_error_projections(self, learning_mech, receiver_activity_mech):
        error_sources = []
        error_projections = []
        # for error_source in learning_mech.error_sources:
        #     if error_source in self.nodes:
        #         error_sources.append(error_source)
        # Add any LearningMechanisms associated with efferent projection from receiver_activity_mech
        # First get all efferents of receiver_activity_mech with a LearningProjection that are in current Composition
        for efferent in [p for p in receiver_activity_mech.efferents
                         if (hasattr(p, 'has_learning_projection')
                             and p.has_learning_projection
                             and p in self.projections)]:
            # Then any LearningProjections to that efferent that are in current Composition
            for learning_projection in [mod_aff for mod_aff in efferent.parameter_ports[MATRIX].mod_afferents
                                        if (isinstance(mod_aff, LearningProjection) and mod_aff in self.projections)]:
                error_source = learning_projection.sender.owner
                if (error_source in learning_mech.error_sources
                        and error_source in self.nodes
                        and learning_mech in [p.receiver.owner for p in error_source.efferents]):
                    continue
                error_sources.append(error_source)
                # FIX: REPLACE WITH learning_mech._add_error_signal_input_port ONCE IMPLEMENTED
                error_signal_input_port = next((e for e in learning_mech.error_signal_input_ports
                                                 if not e.path_afferents), None)
                if error_signal_input_port is None:
                    error_signal_input_port = learning_mech.add_ports(
                                                        InputPort(projections=error_source.output_ports[ERROR_SIGNAL],
                                                                  name=ERROR_SIGNAL,
                                                                  context=Context(source=ContextFlags.METHOD)),
                                                        context=Context(source=ContextFlags.METHOD))
                # DOES THE ABOVE GENERATE A PROJECTION?  IF SO, JUST GET AND RETURN THAT;  ELSE DO THE FOLLOWING:
                error_projections.append(MappingProjection(sender=error_source.output_ports[ERROR_SIGNAL],
                                                           receiver=error_signal_input_port))
        return error_projections
        #  2) For non-terminal sequences, determine # of error_signals coming from LearningMechanisms associated with
        #     all efferentprojections of ProcessingMechanism that projects to ACTIVATION_OUTPUT of LearningMechanism
        #     - check validity of existing error_signal projections with respect to those and, if possible,
        #       their correspondence with error_matrices
        #     - check if any ERROR_SIGNAL input_ports are empty (vacated by terminal sequence elements deleted in
        #       add_projection)
        #     - call add_ports method on LearningMechanism to add new ERROR_SIGNAL input_port to its input_ports
        #       and error_matrix to its self.error_matrices attribute
        #     - add new error_signal projection

    def _add_error_projection_to_dependent_learning_mechs(self, error_source):
        projections = []
        # Get all afferents to receiver_activity_mech in Composition that have LearningProjections
        for afferent in [p for p in error_source.input_source.path_afferents
                         if (p in self.projections
                             and hasattr(p, 'has_learning_projection')
                             and p.has_learning_projection)]:
            # For each LearningProjection to that afferent, if its LearningMechanism doesn't already receiver
            for learning_projection in [lp for lp in afferent.parameter_ports[MATRIX].mod_afferents
                                        if (isinstance(lp, LearningProjection)
                                            and error_source not in lp.sender.owner.error_sources)]:
                dependent_learning_mech = learning_projection.sender.owner
                error_signal_input_port = dependent_learning_mech.add_ports(
                                                    InputPort(projections=error_source.output_ports[ERROR_SIGNAL],
                                                              name=ERROR_SIGNAL,
                                                              context=Context(source=ContextFlags.METHOD)),
                                                    context=Context(source=ContextFlags.METHOD))
                projections.append(error_signal_input_port[0].path_afferents[0])
                # projections.append(MappingProjection(sender=error_source.output_ports[ERROR_SIGNAL],
                #                                      receiver=error_signal_input_port[0]))
        return projections


    # ******************************************************************************************************************
    #                                              CONTROL
    # ******************************************************************************************************************

    def add_controller(self, controller:ControlMechanism):
        """
        Add an `OptimizationControlMechanism` as the `controller
        <Composition.controller>` of the Composition, which gives the OCM access to the
        `Composition`'s `evaluate <Composition.evaluate>` method. This allows the OCM to use simulations to determine
        an optimal Control policy.
        """

        if not isinstance(controller, ControlMechanism):
            raise CompositionError(f"Specification of {repr(CONTROLLER)} arg for {self.name} "
                                   f"must be a {repr(ControlMechanism.__name__)} ")

        # VALIDATE AND ADD CONTROLLER

        if not controller.initialization_status == ContextFlags.DEFERRED_INIT:
            # Warn for request to assign the ControlMechanism already assigned and ignore
            if controller is self.controller:
                warnings.warn(f"{controller.name} has already been assigned as the {CONTROLLER} "
                              f"for {self.name}; assignment ignored.")
                return

            # Warn for request to assign ControlMechanism that is already the controller of another Composition
            if hasattr(controller, COMPOSITION) and controller.composition is not self:
                warnings.warn(f"{controller} has already been assigned as the {CONTROLLER} "
                              f"for another {COMPOSITION} ({controller.composition.name}); assignment ignored.")
                return

            # Warn if current one is being replaced
            if self.controller and self.prefs.verbosePref:
                warnings.warn(f"The existing {CONTROLLER} for {self.name} ({self.controller.name}) "
                              f"is being replaced by {controller.name}.")

        controller.composition = self
        self.controller = controller

        invalid_aux_components = self._get_invalid_aux_components(controller)

        if not invalid_aux_components:
            if self.controller.objective_mechanism:
                self.add_node(self.controller.objective_mechanism)

            self.node_ordering.append(controller)

            self.enable_controller = True

            controller._activate_projections_for_compositions(self)
            self._analyze_graph()
            self._update_shadows_dict(controller)

            # INSTANTIATE SHADOW_INPUT PROJECTIONS
            # Skip controller's first (OUTCOME) input_port (that receives the Projection from its objective_mechanism
            nested_cims = [comp.input_CIM for comp in self._get_nested_compositions()]
            input_cims= [self.input_CIM] + nested_cims
            # For the rest of the controller's input_ports if they are marked as receiving SHADOW_INPUTS,
            #    instantiate the shadowing Projection to them from the sender to the shadowed InputPort
            for input_port in controller.input_ports[1:]:
                if hasattr(input_port, SHADOW_INPUTS) and input_port.shadow_inputs is not None:
                    for proj in input_port.shadow_inputs.path_afferents:
                        sender = proj.sender
                        if sender.owner not in input_cims:
                            self.add_projection(projection=MappingProjection(sender=sender, receiver=input_port),
                                                sender=sender.owner,
                                                receiver=controller)
                            shadow_proj._activate_for_compositions(self)
                        else:
                            if not sender.owner.composition == self:
                                sender_input_cim = sender.owner
                                proj_index = sender_input_cim.output_ports.index(sender)
                                sender_corresponding_input_projection = sender_input_cim.input_ports[
                                    proj_index].path_afferents[0]
                                input_projection_sender = sender_corresponding_input_projection.sender
                                if input_projection_sender.owner == self.input_CIM:
                                    shadow_proj = MappingProjection(sender = input_projection_sender,receiver = input_port)
                                    shadow_proj._activate_for_compositions(self)
                            else:
                                try:
                                    shadow_proj = MappingProjection(sender=proj.sender, receiver=input_port)
                                    shadow_proj._activate_for_compositions(self)
                                except DuplicateProjectionError:
                                    pass
                for proj in input_port.path_afferents:
                    if not proj.sender.owner in nested_cims:
                        proj._activate_for_compositions(self)

            # Check whether controller has input, and if not then disable
            if not (isinstance(self.controller.input_ports, ContentAddressableList)
                    and self.controller.input_ports):
                # If controller was enabled, warn that it has been disabled
                if self.enable_controller:
                    warnings.warn(f"{self.controller.name} for {self.name} has no input_ports, "
                                  f"so controller will be disabled.")
                self.enable_controller = False
                return

            # ADD ANY ControlSignals SPECIFIED BY NODES IN COMPOSITION

            # Get rid of default ControlSignal if it has no ControlProjections
            controller._remove_default_control_signal(type=CONTROL_SIGNAL)

            # Add any ControlSignals specified for ParameterPorts of nodes already in the Composition
            control_signal_specs = self._get_control_signals_for_composition()
            for ctl_sig_spec in control_signal_specs:
                # FIX: 9/14/19: THIS SHOULD BE HANDLED IN _instantiate_projection_to_port
                #               CALLED FROM _instantiate_control_signal
                #               SHOULD TRAP THAT ERROR AND GENERATE CONTEXT-APPROPRIATE ERROR MESSAGE
                # Don't add any that are already on the ControlMechanism

                # FIX: 9/14/19 - IS THE CONTEXT CORRECT (TRY TRACKING IN SYSTEM TO SEE WHAT CONTEXT IS):
                new_signal = controller._instantiate_control_signal(control_signal=ctl_sig_spec,
                                                       context=Context(source=ContextFlags.COMPOSITION))
                controller.control.append(new_signal)
                # FIX: 9/15/19 - WHAT IF NODE THAT RECEIVES ControlProjection IS NOT YET IN COMPOSITON:
                #                ?DON'T ASSIGN ControlProjection?
                #                ?JUST DON'T ACTIVATE IT FOR COMPOSITON?
                #                ?PUT IT IN aux_components FOR NODE?
                #                ! TRACE THROUGH _activate_projections_for_compositions TO SEE WHAT IT CURRENTLY DOES
                controller._activate_projections_for_compositions(self)
            controller.initialization_status = ContextFlags.INITIALIZED
        else:
            controller.initialization_status = ContextFlags.DEFERRED_INIT

    def _get_control_signals_for_composition(self):
        """Return list of ControlSignals specified by nodes in the Composition

        Generate list of control signal specifications
            from ParameterPorts of Mechanisms that have been specified for control.
        The specifications can be:
            ControlProjections (with deferred_init())
            # FIX: 9/14/19 - THIS SHOULD ALREADY HAVE BEEN PARSED INTO ControlProjection WITH DEFFERRED_INIT:
            #                OTHERWISE, NEED TO ADD HANDLING OF IT BELOW
            ControlSignals (e.g., in a 2-item tuple specification for the parameter);
            Note:
                The initialization of the ControlProjection and, if specified, the ControlSignal
                are completed in the call to controller_instantiate_control_signal() in add_controller.
        Mechanism can be in the Compositon itself, or in a nested Composition that does not have its own controller.
        """

        control_signal_specs = []
        for node in self.nodes:
            if isinstance(node, Composition):
                # Get control signal specifications for nested composition if it does not have its own controller
                if node.controller:
                    node_control_signals = node._get_control_signals_for_composition()
                    if node_control_signals:
                        control_signal_specs.append(node._get_control_signals_for_composition())
            elif isinstance(node, Mechanism):
                control_signal_specs.extend(node._get_parameter_port_deferred_init_control_specs())
        return control_signal_specs

    def _build_predicted_inputs_dict(self, predicted_input):
        inputs = {}
        # ASSUMPTION: input_ports[0] is NOT a feature and input_ports[1:] are features
        # If this is not a good assumption, we need another way to look up the feature InputPorts
        # of the OCM and know which InputPort maps to which predicted_input value

        nested_nodes = dict(self._get_nested_nodes())
        for j in range(len(self.controller.input_ports) - 1):
            input_port = self.controller.input_ports[j + 1]
            if hasattr(input_port, SHADOW_INPUTS) and input_port.shadow_inputs is not None:
                owner = input_port.shadow_inputs.owner
                if not owner in nested_nodes:
                    inputs[input_port.shadow_inputs.owner] = predicted_input[j]
                else:
                    comp = nested_nodes[owner]
                    if not comp in inputs:
                        inputs[comp]=[[predicted_input[j]]]
                    else:
                        inputs[comp]=np.concatenate([[predicted_input[j]],inputs[comp][0]])
        return inputs

    def _get_invalid_aux_components(self, controller):
        valid_nodes = \
            [node for node in self.nodes.data] + \
            [node for node, composition in self._get_nested_nodes()] + \
            [self.controller, self.controller.objective_mechanism, self]
        invalid_components = []
        for aux in controller.aux_components:
            component = None
            if isinstance(aux, Projection):
                component = aux
            elif hasattr(aux, '__iter__'):
                for i in aux:
                    if isinstance(i, Projection):
                        component = i
            if not component:
                continue
            if isinstance(component, Projection):
                if hasattr(component.sender,'owner_mech'):
                    sender_node = component.sender.owner_mech
                else:
                    if isinstance(component.sender.owner, CompositionInterfaceMechanism):
                        sender_node = component.sender.owner.composition
                    else:
                        sender_node = component.sender.owner
                if hasattr(component.receiver, 'owner_mech'):
                    receiver_node = component.receiver.owner_mech
                else:
                    if isinstance(component.receiver.owner, CompositionInterfaceMechanism):
                        receiver_node = component.receiver.owner.composition
                    else:
                        receiver_node = component.receiver.owner
                if not all([sender_node in valid_nodes, receiver_node in valid_nodes]):
                    invalid_components.append(component)
        if invalid_components:
            return invalid_components
        else:
            return []

    def reshape_control_signal(self, arr):

        current_shape = np.shape(arr)
        if len(current_shape) > 2:
            newshape = (current_shape[0], current_shape[1])
            newarr = np.reshape(arr, newshape)
            arr = tuple(newarr[i].item() for i in range(len(newarr)))

        return np.array(arr)

    def _get_total_cost_of_control_allocation(self, control_allocation, context, runtime_params):
        total_cost = 0.
        if control_allocation is not None:  # using "is not None" in case the control allocation is 0.

            base_control_allocation = self.reshape_control_signal(self.controller.parameters.value._get(context))

            candidate_control_allocation = self.reshape_control_signal(control_allocation)

            # Get reconfiguration cost for candidate control signal
            reconfiguration_cost = 0.
            if callable(self.controller.compute_reconfiguration_cost):
                reconfiguration_cost = self.controller.compute_reconfiguration_cost([candidate_control_allocation,
                                                                                     base_control_allocation])
                self.controller.reconfiguration_cost.set(reconfiguration_cost, context)

            # Apply candidate control signal
            self.controller._apply_control_allocation(candidate_control_allocation,
                                                                context=context,
                                                                runtime_params=runtime_params,
                                                                )

            # Get control signal costs
            all_costs = self.controller.parameters.costs._get(context) + [reconfiguration_cost]
            # Compute a total for the candidate control signal(s)
            total_cost = self.controller.combine_costs(all_costs)
        return total_cost

    def _check_initialization_status(self):
        """Checks initialization status of controller (if applicable) and any projections or ports

        """
        # Check if controller is in deferred init
        if self.controller and self.controller.initialization_status == ContextFlags.DEFERRED_INIT:
            self.add_controller(self.controller)
            if self.controller.initialization_status == ContextFlags.DEFERRED_INIT:
                invalid_aux_components = self._get_invalid_aux_components(self.controller)
                for component in invalid_aux_components:
                    if isinstance(component, Projection):
                        if hasattr(component.receiver, 'owner_mech'):
                            owner = component.receiver.owner_mech
                        else:
                            owner = component.receiver.owner
                        raise CompositionError(
                                f"The controller of {self.name} has been specified to project to {owner.name}, "
                                f"but {owner.name} is not in {self.name} or any of its nested Compositions."
                        )

        for node in self.nodes:
            # Check for deferred init projections
            for projection in node.projections:
                if projection.initialization_status == ContextFlags.DEFERRED_INIT:
                    # NOTE:
                    #   May want to add other conditions and warnings here. Currently
                    #   just checking for unresolved control projections.
                    if isinstance(projection, ControlProjection):
                        warnings.warn(f"The {projection.receiver.name} parameter of {projection.receiver.owner.name} \n"
                                      f"is specified for control, but {self.name} does not have a controller. Please \n"
                                      f"add a controller to {self.name} or the control specification will be \n"
                                      f"ignored.")

    def evaluate(
            self,
            predicted_input=None,
            control_allocation=None,
            num_simulation_trials=None,
            runtime_params=None,
            base_context=Context(execution_id=None),
            context=None,
            execution_mode=False,
            return_results=False,
            block_simulate=False
    ):
        """Runs a simulation of the `Composition`, with the specified control_allocation, excluding its
           `controller <Composition.controller>` in order to return the
           `net_outcome <ControlMechanism.net_outcome>` of the Composition, according to its
           `controller <Composition.controller>` under that control_allocation. All values are
           reset to pre-simulation values at the end of the simulation.

           If `block_simulate` is set to True, the `controller <Composition.controller>` will attempt to use the
           entire input set provided to the `run <Composition.run>` method of the `Composition` as input for the
           simulated call to `run <Composition.run>`. If it is not, the `controller <Composition.controller>` will
           use the inputs slated for its next or previous execution, depending on whether the `controller_mode` of the
           `Composition` is set to `before` or `after`, respectively.

           .. note::
                Block simulation can not be used if the Composition's stimuli were specified as a generator. If
                `block_simulate` is set to True and the input type for the Composition was a generator,
                block simulation will be disabled for the current execution of `evaluate <Composition.evaluate>`.
        """
        # Apply candidate control to signal(s) for the upcoming simulation and determine its cost
        total_cost = self._get_total_cost_of_control_allocation(control_allocation, context, runtime_params)

        # Build input dictionary for simulation
        input_spec = self.parameters.input_specification.get(context)
        if input_spec and block_simulate and not isgenerator(input_spec):
            if isgeneratorfunction(input_spec):
                inputs = input_spec()
            elif isinstance(input_spec, dict):
                inputs = input_spec
        else:
            inputs = self._build_predicted_inputs_dict(predicted_input)

        if hasattr(self, '_input_spec') and block_simulate and isgenerator(input_spec):
            warnings.warn(f"The evaluate method of {self.name} is attempting to use block simulation, but the "
                          f"supplied input spec is a generator. Generators can not be used as inputs for block "
                          f"simulation. This evaluation will not use block simulation.")

        # HACK: _animate attribute is set in execute method, but Evaluate can be called on a Composition that has not
        # yet called the execute method, so we need to do a check here too.
        # -DTS
        if not hasattr(self, '_animate'):
            # These are meant to be assigned in run method;  needed here for direct call to execute method
            self._animate = False

        # Run Composition in "SIMULATION" context
        if self._animate is not False and self._animate_simulations is not False:
            animate = self._animate
            buffer_animate_state = None
        else:
            animate = False
            buffer_animate_state = self._animate

        context.add_flag(ContextFlags.SIMULATION)
        context.remove_flag(ContextFlags.CONTROL)
        results = self.run(inputs=inputs,
                 context=context,
                 runtime_params=runtime_params,
                 num_trials=num_simulation_trials,
                 animate=animate,
                 bin_execute=execution_mode,
                 skip_initialization=True,
                 )
        context.remove_flag(ContextFlags.SIMULATION)
        context.add_flag(ContextFlags.CONTROL)
        if buffer_animate_state:
            self._animate = buffer_animate_state

        # Store simulation results on "base" composition
        if self.initialization_status != ContextFlags.INITIALIZING:
            try:
                self.parameters.simulation_results._get(base_context).append(
                    self.get_output_values(context))
            except AttributeError:
                self.parameters.simulation_results._set([self.get_output_values(context)], base_context)

        # Update input ports in order to get correct value for "outcome" (from objective mech)
        self.controller._update_input_ports(context, runtime_params)
        outcome = self.controller.input_port.parameters.value._get(context)

        if outcome is None:
            net_outcome = 0.0
        else:
            # Compute net outcome based on the cost of the simulated control allocation (usually, net = outcome - cost)
            net_outcome = self.controller.compute_net_outcome(outcome, total_cost)

        if return_results:
            return net_outcome, results
        else:
            return net_outcome

    # ******************************************************************************************************************
    #                                                SHOW_GRAPH
    # ******************************************************************************************************************

    @tc.typecheck
    @handle_external_context(execution_id=NotImplemented)
    def show_graph(self,
                   show_node_structure:tc.any(bool, tc.enum(VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS,
                                                            STATE_FUNCTION_PARAMS, ROLES, ALL))=False,
                   show_nested:tc.optional(tc.any(bool,dict,tc.enum(ALL)))=ALL,
                   show_controller:tc.any(bool, tc.enum(AGENT_REP))=False,
                   show_cim:bool=False,
                   show_learning:bool=False,
                   show_headers:bool=True,
                   show_types:bool=False,
                   show_dimensions:bool=False,
                   show_projection_labels:bool=False,
                   direction:tc.enum('BT', 'TB', 'LR', 'RL')='BT',
                   # active_items:tc.optional(list)=None,
                   active_items=None,
                   active_color=BOLD,
                   input_color='green',
                   output_color='red',
                   input_and_output_color='brown',
                   # feedback_color='yellow',
                   controller_color='blue',
                   learning_color='orange',
                   composition_color='pink',
                   control_projection_arrow='box',
                   feedback_shape = 'septagon',
                   cim_shape='square',
                   output_fmt:tc.enum('pdf','gv','jupyter','gif')='pdf',
                   context=None,
                   **kwargs):
        """
        show_graph(                           \
           show_node_structure=False,         \
           show_nested=True,                  \
           show_controller=False,             \
           show_cim=False,                    \
           show_learning=False,               \
           show_headers=True,                 \
           show_types=False,                  \
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
           feedback_shape = 'septagon',       \
           cim_shape='square',                \
           output_fmt='pdf',                  \
           context=None)

        Show graphical display of Components in a Composition's graph.

        .. note::
           This method relies on `graphviz <http://www.graphviz.org>`_, which must be installed and imported
           (standard with PsyNeuLink pip install)

        See `Visualizing a Composition <Composition_Visualization>` for details and examples.

        Arguments
        ---------

        show_node_structure : bool, VALUES, LABELS, FUNCTIONS, MECH_FUNCTION_PARAMS, STATE_FUNCTION_PARAMS, ROLES, \
        or ALL : default False
            show a detailed representation of each `Mechanism <Mechanism>` in the graph, including its `Ports <Port>`;
            can have any of the following settings alone or in a list:

            * `True` -- show Ports of Mechanism, but not information about the `value
              <Component.value>` or `function <Component.function>` of the Mechanism or its Ports.

            * *VALUES* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <Port_Base.value>` of each of its Ports.

            * *LABELS* -- show the `value <Mechanism_Base.value>` of the Mechanism and the `value
              <Port_Base.value>` of each of its Ports, using any labels for the values of InputPorts and
              OutputPorts specified in the Mechanism's `input_labels_dict <Mechanism.input_labels_dict>` and
              `output_labels_dict <Mechanism.output_labels_dict>`, respectively.

            * *FUNCTIONS* -- show the `function <Mechanism_Base.function>` of the Mechanism and the `function
              <Port_Base.function>` of its InputPorts and OutputPorts.

            * *MECH_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
              Mechanism in the Composition (only applies if *FUNCTIONS* is True).

            * *STATE_FUNCTION_PARAMS_* -- show the parameters of the `function <Mechanism_Base.function>` for each
              Port of each Mechanism in the Composition (only applies if *FUNCTIONS* is True).

            * *ROLES* -- show the `role <Composition.NodeRoles>` of the Mechanism in the Composition
              (but not any of the other information;  use *ALL* to show ROLES with other information).

            * *ALL* -- shows the role, `function <Component.function>`, and `value <Component.value>` of the
              Mechanisms in the `Composition` and their `Ports <Port>` (using labels for
              the values, if specified -- see above), including parameters for all functions.

        show_nested : bool | dict : default ALL
            specifies whether any nested Composition(s) are shown in details as inset graphs.  A dict can be used to
            specify any of the arguments allowed for show_graph to be used for the nested Composition(s);  *ALL*
            passes all arguments specified for the main Composition to the nested one(s);  True uses the default
            values of show_graph args for the nested Composition(s).

        show_controller :  bool or AGENT_REP : default False
            specifies whether or not to show the Composition's `controller <Composition.controller>` and associated
            `objective_mechanism <ControlMechanism.objective_mechanism>` if it has one.  If the controller is an
            OptimizationControlMechanism and it has an `agent_rep <OptimizationControlMechanism>`, then specifying
            *AGENT_REP* will also show that.  All of these items are displayed in the color specified for
            **controller_color**.

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

        show_types : bool : default False
            specifies whether or not to show type (class) of `Mechanism <Mechanism>` in each node label.

        show_dimensions : bool : default False
            specifies whether or not to show dimensions for the `variable <Component.variable>` and `value
            <Component.value>` of each Component in the graph (and/or MappingProjections when show_learning
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

        input_color : keyword : default 'green',
            specifies the display color for `INPUT <NodeRole.INPUT>` Nodes in the Composition

        output_color : keyword : default 'red',
            specifies the display color for `OUTPUT` Nodes in the Composition

        input_and_output_color : keyword : default 'brown'
            specifies the display color of nodes that are both an `INPUT <NodeRole.INPUT>` and an `OUTPUT
            <NodeRole.OUTPUT>` Node in the Composition

        COMMENT:
        feedback_color : keyword : default 'yellow'
            specifies the display color of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.
        COMMENT

        controller_color : keyword : default 'blue'
            specifies the color in which the controller components are displayed

        learning_color : keyword : default 'orange'
            specifies the color in which the learning components are displayed

        composition_color : keyword : default 'brown'
            specifies the display color of nodes that represent nested Compositions.

        feedback_shape : keyword : default 'septagon'
            specifies the display shape of nodes that are assigned the `NodeRole` `FEEDBACK_SENDER`.

        cim_shape : default 'square'
            specifies the display color input_CIM and output_CIM nodes

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
                nested_comp_graph.name = "cluster_" + rcvr.name
                rcvr_label = rcvr.name
                # if rcvr in self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
                #     nested_comp_graph.attr(color=feedback_color)
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
            if ((NodeRole.LEARNING in self.nodes_to_roles[rcvr]
                 or NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[rcvr])
                    and not NodeRole.INPUT in self.nodes_to_roles[rcvr]):
                return

            # If rcvr is ObjectiveMechanism for Composition's controller,
            #    break and handle in _assign_control_components()
            if (isinstance(rcvr, ObjectiveMechanism)
                    and self.controller
                    and rcvr is self.controller.objective_mechanism):
                return

            # Implement rcvr node
            else:

                # Set rcvr shape, color, and penwidth based on node type
                rcvr_rank = 'same'

                # Feedback Node
                if rcvr in self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
                    node_shape = feedback_shape
                else:
                    node_shape = mechanism_shape

                # Get condition if any associated with rcvr
                if rcvr in self.scheduler.conditions:
                    condition = self.scheduler.conditions[rcvr]
                else:
                    condition = None

                # # Feedback Node
                # if rcvr in self.get_nodes_by_role(NodeRole.FEEDBACK_SENDER):
                #     if rcvr in active_items:
                #         if active_color is BOLD:
                #             rcvr_color = feedback_color
                #         else:
                #             rcvr_color = active_color
                #         rcvr_penwidth = str(bold_width + active_thicker_by)
                #         self.active_item_rendered = True
                #     else:
                #         rcvr_color = feedback_color
                #         rcvr_penwidth = str(bold_width)

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
                rcvr_label = self._get_graph_node_label(rcvr,
                                                        show_types,
                                                        show_dimensions)

                if show_node_structure and isinstance(rcvr, Mechanism):
                    g.node(rcvr_label,
                           rcvr._show_structure(**node_struct_args, node_border=rcvr_penwidth, condition=condition),
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
                cim_label = self._get_graph_node_label(cim, show_types, show_dimensions)

                if show_node_structure:
                    g.node(cim_label,
                           cim._show_structure(**node_struct_args, node_border=cim_penwidth, compact_cim=True),
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

                    for output_port in self.input_CIM.output_ports:
                        projs = output_port.efferents
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
                            input_mech_label = self._get_graph_node_label(input_mech,
                                                                          show_types,
                                                                          show_dimensions)
                            if show_node_structure:
                                cim_proj_label = '{}:{}-{}'. \
                                    format(cim_label, OutputPort.__name__, proj.sender.name)
                                proc_mech_rcvr_label = '{}:{}-{}'. \
                                    format(input_mech_label, InputPort.__name__, proj.receiver.name)
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
                                label = self._get_graph_node_label(proj, show_types, show_dimensions)
                            else:
                                label = ''
                            g.edge(cim_proj_label, proc_mech_rcvr_label, label=label,
                                   color=proj_color, penwidth=proj_width)

                # Projections from OUTPUT nodes to output_CIM
                if cim is self.output_CIM:
                    # Construct edge name
                    for input_port in self.output_CIM.input_ports:
                        projs = input_port.path_afferents
                        for proj in projs:
                            # Validate the Projection is from an OUTPUT node
                            output_mech = proj.sender.owner
                            if not NodeRole.OUTPUT in self.nodes_to_roles[output_mech]:
                                raise CompositionError("Projection to output_CIM of {} from node {} "
                                                       "that is not an {} node".
                                                       format(self.name, output_mech,
                                                              NodeRole.OUTPUT.name, NodeRole.OUTPUT.name.lower()))
                            # Construct edge name
                            output_mech_label = self._get_graph_node_label(output_mech,
                                                                           show_types,
                                                                           show_dimensions)
                            if show_node_structure:
                                cim_proj_label = '{}:{}'. \
                                    format(cim_label, cim._get_port_name(proj.receiver))
                                proc_mech_sndr_label = '{}:{}'.\
                                    format(output_mech_label, output_mech._get_port_name(proj.sender))
                                    # format(output_mech_label, OutputPort.__name__, proj.sender.name)
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
                                label = self._get_graph_node_label(proj, show_types, show_dimensions)
                            else:
                                label = ''
                            g.edge(proc_mech_sndr_label, cim_proj_label, label=label,
                                   color=proj_color, penwidth=proj_width)

        def _assign_controller_components(g):
            """Assign control nodes and edges to graph"""

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
            ctlr_label = self._get_graph_node_label(controller, show_types, show_dimensions)
            if show_node_structure:
                g.node(ctlr_label,
                       controller._show_structure(**node_struct_args, node_border=ctlr_width,
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
                    proc_mech_label = self._get_graph_node_label(ctl_proj.receiver.owner, show_types, show_dimensions)
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
                objmech_ctlr_proj = controller.input_port.path_afferents[0]
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

                objmech_label = self._get_graph_node_label(objmech, show_types, show_dimensions)
                if show_node_structure:
                    if objmech in self.scheduler.conditions:
                        condition = self.scheduler.conditions[objmech]
                    else:
                        condition = None
                    g.node(objmech_label,
                           objmech._show_structure(**node_struct_args, node_border=ctlr_width, condition=condition),
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
                for input_port in objmech.input_ports:
                    for projection in input_port.path_afferents:
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
                            sndr_proj_label = self._get_graph_node_label(projection.sender.owner,
                                                                         show_types,
                                                                         show_dimensions) + \
                                              ':' + objmech._get_port_name(projection.sender)
                            objmech_proj_label = objmech_label + ':' + objmech._get_port_name(input_port)
                        else:
                            sndr_proj_label = self._get_graph_node_label(projection.sender.owner,
                                                                         show_types,
                                                                         show_dimensions)
                            objmech_proj_label = self._get_graph_node_label(objmech,
                                                                            show_types,
                                                                            show_dimensions)
                        if show_projection_labels:
                            edge_label = projection.name
                        else:
                            edge_label = ''
                        g.edge(sndr_proj_label, objmech_proj_label, label=edge_label,
                               color=proj_color, penwidth=proj_width)

            # If controller has an agent_rep, assign its node and edges (not Projections per se)
            if hasattr(controller, 'agent_rep') and controller.agent_rep and show_controller==AGENT_REP :
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
                agent_rep_label = self._get_graph_node_label(agent_rep, show_types, show_dimensions)
                g.node(agent_rep_label,
                        color=agent_rep_color, penwidth=agent_rep_width, shape=agent_rep_shape,
                        rank=control_rank)

                # agent_rep <-> controller edges
                g.edge(agent_rep_label, ctlr_label, color=agent_rep_color, penwidth=agent_rep_width)
                g.edge(ctlr_label, agent_rep_label, color=agent_rep_color, penwidth=agent_rep_width)

            # get any other incoming edges to controller (i.e., other than from ObjectiveMechanism)
            senders = set()
            for i in controller.input_ports[1:]:
                for p in i.path_afferents:
                    senders.add(p.sender.owner)
            _assign_incoming_edges(g, controller, ctlr_label, senders, proj_color=ctl_proj_color)

        def _assign_learning_components(g):
            """Assign learning nodes and edges to graph"""

            # Get learning_components, with exception of INPUT (i.e. TARGET) nodes
            #    (i.e., allow TARGET node to continue to be marked as an INPUT node)
            learning_components = [node for node in self.learning_components
                                   if not NodeRole.INPUT in self.nodes_to_roles[node]]
            # learning_components.extend([node for node in self.nodes if
            #                             NodeRole.AUTOASSOCIATIVE_LEARNING in
            #                             self.nodes_to_roles[node]])

            for rcvr in learning_components:
                # if rcvr is Projection, skip (handled in _assign_processing_components)
                if isinstance(rcvr, MappingProjection):
                    return

                # Get rcvr info
                rcvr_label = self._get_graph_node_label(rcvr, show_types, show_dimensions)
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
                            rcvr._show_structure(**node_struct_args),
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
            sndrs = proj._parameter_ports['matrix'].mod_afferents # GET ALL LearningProjections to proj
            for sndr in sndrs:
                sndr_label = self._get_graph_node_label(sndr.sender.owner, show_types, show_dimensions)
                rcvr_label = self._get_graph_node_label(proj, show_types, show_dimensions)
                if show_projection_labels:
                    edge_label = proj._parameter_ports['matrix'].mod_afferents[0].name
                else:
                    edge_label = ''
                if show_node_structure:
                    G.edge(sndr_label + ':' + OutputPort.__name__ + '-' + 'LearningSignal',
                           rcvr_label,
                           label=edge_label,
                           color=learning_proj_color, penwidth=learning_proj_width)
                else:
                    G.edge(sndr_label, rcvr_label, label = edge_label,
                           color=learning_proj_color, penwidth=learning_proj_width)
            return True

        @tc.typecheck
        def _assign_incoming_edges(g, rcvr, rcvr_label, senders, proj_color=None, proj_arrow=None):

            proj_color = proj_color or default_node_color
            proj_arrow = default_projection_arrow

            for sndr in senders:

                # Set sndr info
                sndr_label = self._get_graph_node_label(sndr, show_types, show_dimensions)

                # Iterate through all Projections from all OutputPorts of sndr
                for output_port in sndr.output_ports:
                    for proj in output_port.efferents:

                        # Skip any projections to ObjectiveMechanism for controller
                        #   (those are handled in _assign_control_components)
                        if (self.controller and
                                proj.receiver.owner in {self.controller, self.controller.objective_mechanism}):
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

                            edge_label = self._get_graph_node_label(proj, show_types, show_dimensions)
                            is_learning_component = rcvr in self.learning_components or sndr in self.learning_components

                            # Check if Projection or its receiver is active
                            if any(item in active_items for item in {proj, proj.receiver.owner}):
                                if active_color is BOLD:
                                    # if (isinstance(rcvr, LearningMechanism) or isinstance(sndr, LearningMechanism)):
                                    if is_learning_component:
                                        proj_color = learning_color
                                    else:
                                        pass
                                else:
                                    proj_color = active_color
                                proj_width = str(default_width + active_thicker_by)
                                self.active_item_rendered = True

                            # Projection to or from a LearningMechanism
                            elif (NodeRole.LEARNING in self.nodes_to_roles[rcvr]
                                  or NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[rcvr]):
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
                                from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
                                if isinstance(proj, ControlProjection):
                                    arrowhead=control_projection_arrow
                                else:
                                    arrowhead=proj_arrow
                                if show_projection_labels:
                                    label = proc_mech_label
                                else:
                                    label = ''
                                g.edge(sndr_proj_label, proc_mech_rcvr_label,
                                       label=label,
                                       color=proj_color,
                                       penwidth=proj_width,
                                       arrowhead=arrowhead)

        # SETUP AND CONSTANTS -----------------------------------------------------------------

        INITIAL_FRAME = "INITIAL_FRAME"

        if context.execution_id is NotImplemented:
            context.execution_id = self.default_execution_id

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
            if (self.scheduler.get_clock(context).time.run >= self._animate_num_runs or
                    self.scheduler.get_clock(context).time.trial >= self._animate_num_trials):
                return

            for item in active_items:
                if not isinstance(item, Component) and item is not INITIAL_FRAME:
                    raise CompositionError(
                        "PROGRAM ERROR: Item ({}) specified in {} argument for {} method of {} is not a {}".
                        format(item, repr('active_items'), repr('show_graph'), self.name, Component.__name__))

        self.active_item_rendered = False

        # Argument values used to call Mechanism._show_structure()
        if isinstance(show_node_structure, (list, tuple, set)):
            node_struct_args = {'composition': self,
                                'show_roles': any(key in show_node_structure for key in {ROLES, ALL}),
                                'show_conditions': any(key in show_node_structure for key in {CONDITIONS, ALL}),
                                'show_functions': any(key in show_node_structure for key in {FUNCTIONS, ALL}),
                                'show_mech_function_params': any(key in show_node_structure
                                                                 for key in {MECH_FUNCTION_PARAMS, ALL}),
                                'show_port_function_params': any(key in show_node_structure
                                                                  for key in {STATE_FUNCTION_PARAMS, ALL}),
                                'show_values': any(key in show_node_structure for key in {VALUES, ALL}),
                                'use_labels': any(key in show_node_structure for key in {LABELS, ALL}),
                                'show_headers': show_headers,
                                'output_fmt': 'struct',
                                'context':context}
        else:
            node_struct_args = {'composition': self,
                                'show_roles': show_node_structure in {ROLES, ALL},
                                'show_conditions': show_node_structure in {CONDITIONS, ALL},
                                'show_functions': show_node_structure in {FUNCTIONS, ALL},
                                'show_mech_function_params': show_node_structure in {MECH_FUNCTION_PARAMS, ALL},
                                'show_port_function_params': show_node_structure in {STATE_FUNCTION_PARAMS, ALL},
                                'show_values': show_node_structure in {VALUES, LABELS, ALL},
                                'use_labels': show_node_structure in {LABELS, ALL},
                                'show_headers': show_headers,
                                'output_fmt': 'struct',
                                'context': context}

        # DEFAULT ATTRIBUTES ----------------------------------------------------------------

        default_node_color = 'black'
        mechanism_shape = 'oval'
        learning_projection_shape = 'diamond'
        struct_shape = 'plaintext' # assumes use of html
        cim_shape = 'rectangle'
        composition_shape = 'rectangle'
        agent_rep_shape = 'egg'
        default_projection_arrow = 'normal'

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
                'penwidth': str(default_width),
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
            self._analyze_graph(context=context)
        processing_graph = self.graph_processing.dependency_dict
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

        # Sort nodes for display
        def get_index_of_node_in_G_body(node, node_type:tc.enum(MECHANISM, PROJECTION, BOTH)):
            """Get index of node in G.body"""
            for i, item in enumerate(G.body):
                if node.name in item:
                    if node_type in {MECHANISM, BOTH}:
                        if not '->' in item:
                            return i
                    elif node_type in {PROJECTION, BOTH}:
                        if '->' in item:
                            return i
                    else:
                        assert False, f'PROGRAM ERROR: node_type not specified or illegal ({node_type})'

        for node in self.nodes:
            roles = self.get_roles_by_node(node)
            # Put INPUT node(s) first
            if NodeRole.INPUT in roles:
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(0,G.body.pop(i))
            # Put OUTPUT node(s) last (except for ControlMechanisms)
            if NodeRole.OUTPUT in roles:
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))
            # Put ControlMechanism(s) last
            if isinstance(node, ControlMechanism):
                i = get_index_of_node_in_G_body(node, MECHANISM)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))

        for proj in self.projections:
            # Put ControlProjection(s) last (along with ControlMechanis(s))
            if isinstance(proj, ControlProjection):
                i = get_index_of_node_in_G_body(node, PROJECTION)
                if i is not None:
                    G.body.insert(len(G.body),G.body.pop(i))

        if self.controller and show_controller:
            i = get_index_of_node_in_G_body(self.controller, MECHANISM)
            G.body.insert(len(G.body),G.body.pop(i))

        # GENERATE OUTPUT ---------------------------------------------------------------------

        # Show as pdf
        try:
            if output_fmt == 'pdf':
                # G.format = 'svg'
                G.view(self.name.replace(" ", "-"), cleanup=True, directory='show_graph OUTPUT/PDFS')

            # Generate images for animation
            elif output_fmt == 'gif':
                if self.active_item_rendered or INITIAL_FRAME in active_items:
                    self._generate_gifs(G, active_items, context)

            # Return graph to show in jupyter
            elif output_fmt == 'jupyter':
                return G

            elif output_fmt == 'gv':
                return G
        except:
            raise CompositionError(f"Problem displaying graph for {self.name}")

    @tc.typecheck
    def _show_structure(self,
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
                        output_fmt:tc.enum('pdf','struct')='pdf',
                        context=None
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
            show the `function <Component.function>` of the Mechanism and each of its Ports.

        show_mech_function_params : bool : default False
            show the parameters of the Mechanism's `function <Component.function>` if **show_functions** is True.

        show_port_function_params : bool : default False
            show parameters for the `function <Component.function>` of the Mechanism's Ports if **show_functions** is
            True).

        show_values : bool : default False
            show the `value <Component.value>` of the Mechanism and each of its Ports (prefixed by "=").

        use_labels : bool : default False
            use labels for values if **show_values** is `True`; labels must be specified in the `input_labels_dict
            <Mechanism.input_labels_dict>` (for InputPort values) and `output_labels_dict
            <Mechanism.output_labels_dict>` (for OutputPort values); otherwise it is ignored.

        show_headers : bool : default False
            show the Mechanism, InputPort, ParameterPort and OutputPort headers.

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
            specifies whether to suppress InputPort fields for input_CIM and OutputPort fields for output_CIM.

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
        input_ports_header = r'______CIMInputPortS______\n' \
                              r'/\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ' \
                              r'\ \ \ \ \ \ \ \ \ \ \\'
        output_ports_header = r'\\______\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ ______/' \
                               r'\nCIMOutputPortS'

        def mech_string(mech):
            """Return string with name of mechanism possibly with function and/or value
            Inclusion of role, function and/or value is determined by arguments of call to _show_structure
            """
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
                    from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import \
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
        def states_string(port_list: ContentAddressableList,
                          port_type,
                          include_function: bool = False,
                          include_value: bool = False,
                          use_label: bool = False):
            """Return string with name of ports in ContentAddressableList with functions and/or values as specified"""
            states = open_bracket
            for i, port in enumerate(port_list):
                if i:
                    states += pipe
                function = ''
                if include_function:
                    function = r'\n({})'.format(port.function.__class__.__name__)
                value = ''
                if include_value:
                    if use_label:
                        value = r'\n={}'.format(port.label)
                    else:
                        value = r'\n={}'.format(port.value)
                states += r'<{0}-{1}> {1}{2}{3}'.format(port_type.__name__,
                                                        port.name,
                                                        function,
                                                        value)
            states += close_bracket
            return states

        # Construct Mechanism specification
        mech = mech_string(self)

        # Construct InputPorts specification
        if len(self.input_ports) and compact_cim is not INPUT:
            if show_headers:
                input_ports = input_ports_header + pipe + states_string(self.input_ports,
                                                                          InputPort,
                                                                          include_function=show_functions,
                                                                          include_value=show_values,
                                                                          use_label=use_labels)
            else:
                input_ports = states_string(self.input_ports,
                                             InputPort,
                                             include_function=show_functions,
                                             include_value=show_values,
                                             use_label=use_labels)
            input_ports = pipe + input_ports
        else:
            input_ports = ''

        # Construct OutputPorts specification
        if len(self.output_ports) and compact_cim is not OUTPUT:
            if show_headers:
                output_ports = states_string(self.output_ports,
                                              OutputPort,
                                              include_function=show_functions,
                                              include_value=show_values,
                                              use_label=use_labels) + pipe + output_ports_header
            else:
                output_ports = states_string(self.output_ports,
                                              OutputPort,
                                              include_function=show_functions,
                                              include_value=show_values,
                                              use_label=use_labels)

            output_ports = output_ports + pipe
        else:
            output_ports = ''

        m_node_struct = open_bracket + \
                        output_ports + \
                        open_bracket + mech + close_bracket + \
                        input_ports + \
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

    def _get_graph_node_label(self, item, show_types=None, show_dimensions=None):
        if not isinstance(item, (Mechanism, Composition, Projection)):
            raise CompositionError("Unrecognized node type ({}) in graph for {}".format(item, self.name))
        # TBI Show Dimensions
        name = item.name

        if show_types:
            name = item.name + '\n(' + item.__class__.__name__ + ')'

        if show_dimensions in {ALL, MECHANISMS} and isinstance(item, Mechanism):
            input_str = "in ({})".format(",".join(str(input_port.socket_width)
                                                  for input_port in item.input_ports))
            output_str = "out ({})".format(",".join(str(len(np.atleast_1d(output_port.value)))
                                                    for output_port in item.output_ports))
            return f"{output_str}\n{name}\n{input_str}"
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

    def _set_up_animation(self, context):

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

    def _animate_execution(self, active_items, context):
        if self._component_animation_execution_count is None:
            self._component_animation_execution_count = 0
        else:
            self._component_animation_execution_count += 1
        self.show_graph(active_items=active_items,
                        **self._animate,
                        output_fmt='gif',
                        context=context,
                        )

    def _generate_gifs(self, G, active_items, context):

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
        execution_phase = context.execution_phase
        time = self.scheduler.get_clock(context).time
        run_num = time.run
        trial_num = time.trial

        if INITIAL_FRAME in active_items:
            phase_string = create_phase_string('Initializing')
            time_string = create_time_string(time, 'BLANKS')

        elif ContextFlags.PROCESSING in execution_phase:
            phase_string = create_phase_string('Processing Phase')
            time_string = create_time_string(time, 'TIME')
        # elif ContextFlags.LEARNING in execution_phase:
        #     time = self.scheduler_learning.get_clock(context).time
        #     time_string = "Time(run: {}, trial: {}, pass: {}, time_step: {}". \
        #         format(run_num, time.trial, time.pass_, time.time_step)
        #     phase_string = 'Learning Phase - '

        elif ContextFlags.CONTROL in execution_phase:
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


    # ******************************************************************************************************************
    #                                           EXECUTION
    # ******************************************************************************************************************

    @handle_external_context()
    def run(
            self,
            inputs=None,
            scheduler=None,
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
            reinitialize_nodes_when=Never(),
            runtime_params=None,
            skip_initialization=False,
            skip_analyze_graph=False,
            animate=False,
            context=None,
            base_context=Context(execution_id=None),
            ):
        """Pass inputs to Composition, then execute sets of nodes that are eligible to run until termination
        conditions are met.  See `Run` for details of formatting input specifications. See `Run` for details of
        formatting input specifications. Use **animate** to generate a gif of the execution sequence.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` : list } or { `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each Node in the composition that receives inputs from
                the user. For each pair, the key is the Node and the value is a list of inputs. Each input in the
                list corresponds to a certain `TRIAL`.

            scheduler : Scheduler
                the scheduler object that owns the conditions that will instruct the execution of the Composition.
                If not specified, the Composition will use its automatically generated scheduler.

            context
                context will be set to self.default_execution_id if unspecified

            base_context
                the context corresponding to the execution context from which this execution will be initialized,
                if values currently do not exist for **context**

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
        if scheduler is None:
            scheduler = self.scheduler

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
            node.reinitialize(*reinitialize_values[node], context=context)

        # cache and set reinitialize_when conditions for nodes, matching
        # old System behavior
        # Validate
        if not isinstance(reinitialize_nodes_when, Condition):
            raise CompositionError(
                "{} is not a valid specification for "
                "reinitialize_nodes_when of {}. "
                "reinitialize_nodes_when must be a Condition.".format(
                    reinitialize_nodes_when,
                    self.name
                )
            )

        self._reinitialize_nodes_when_cache = {}
        for node in self.nodes:
            try:
                if isinstance(node.reinitialize_when, Never):
                    self._reinitialize_nodes_when_cache[node] = node.reinitialize_when
                    node.reinitialize_when = reinitialize_nodes_when
            except AttributeError:
                pass

        if ContextFlags.SIMULATION not in context.execution_phase:
            try:
                self.parameters.input_specification._set(copy(inputs), context)
            except:
                self.parameters.input_specification._set(inputs, context)

        # DS 1/7/20: Check to see if any Components are still in deferred init. If so, attempt to initialize them.
        # If they can not be initialized, raise a warning.
        if ContextFlags.SIMULATION not in context.execution_phase:
            self._check_initialization_status()

        # MODIFIED 8/27/19 OLD:
        # try:
        #     if ContextFlags.SIMULATION not in context.execution_phase:
        #         self._analyze_graph()
        # except AttributeError:
        #     # if context is None, it has not been created for this context yet, so it is not
        #     # in a simulation
        #     self._analyze_graph()
        # MODIFIED 8/27/19 NEW:
        # FIX: MODIFIED FEEDBACK -
        #      THIS IS NEEDED HERE (AND NO LATER) TO WORK WITH test_3_mechanisms_2_origins_1_additive_control_1_terminal
        # If a scheduler was passed in, first call _analyze_graph with default scheduler
        if scheduler is not self.scheduler:
            if not skip_analyze_graph:
                self._analyze_graph(context=context)
        # Then call _analyze graph with scheduler actually being used (passed in or default)
        try:
            if ContextFlags.SIMULATION not in context.execution_phase:
                if not skip_analyze_graph:
                    self._analyze_graph(scheduler=scheduler, context=context)
        except AttributeError:
            # if context is None, it has not been created for this context yet,
            # so it is not in a simulation
            if not skip_analyze_graph:
                self._analyze_graph(scheduler=scheduler, context=context)
        # MODIFIED 8/27/19 END

        # set auto logging if it's not already set, and if log argument is True
        if log:
            self.enable_logging()

        # Set animation attributes
        if animate is True:
            animate = {}
        self._animate = animate
        if self._animate is not False:
            self._set_up_animation(context)

        # SET UP EXECUTION -----------------------------------------------

        results = []

        self._assign_execution_ids(context)

        scheduler._init_counts(execution_id=context.execution_id)

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        # if inputs is a generator function, we should instantiate it now so that it will be properly handled
        # below
        if isgeneratorfunction(inputs):
            inputs = inputs()

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
        elif hasattr(inputs, '__next__'):
            num_inputs_sets = sys.maxsize
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
        if not callable(inputs) \
                and not hasattr(inputs, '__next__'):
            # Currently, no validation if 'inputs' arg is a function
            inputs, num_inputs_sets = self._adjust_stimulus_dict(inputs)

        if num_trials is not None:
            num_trials = num_trials
        else:
            num_trials = num_inputs_sets

        scheduler._reset_counts_total(TimeScale.RUN, context.execution_id)

        # KDM 3/29/19: run the following not only during LLVM Run compilation, due to bug where TimeScale.RUN
        # termination condition is checked and no data yet exists. Adds slight overhead as long as run is not
        # called repeatedly (this init is repeated in Composition.execute)
        # initialize from base context but don't overwrite any values already set for this context
        if (not skip_initialization
            and (context is None or ContextFlags.SIMULATION not in context.execution_phase)):
            self._initialize_from_context(context, base_context, override=False)

        context.composition = self

        is_simulation = (context is not None and
                         ContextFlags.SIMULATION in context.execution_phase)

        if (bin_execute is True or str(bin_execute).endswith('Run')):
            # There's no mode to run simulations.
            # Simulations are run as part of the controller node wrapper.
            assert not is_simulation
            try:
                comp_ex_tags = frozenset({"learning"}) if self._is_learning(context) else frozenset()
                if bin_execute is True or bin_execute.startswith('LLVM'):
                    _comp_ex = pnlvm.CompExecution(self, [context.execution_id], additional_tags=comp_ex_tags)
                    results += _comp_ex.run(inputs, num_trials, num_inputs_sets)
                elif bin_execute.startswith('PTX'):
                    self.__ptx_initialize(context, additional_tags=comp_ex_tags)
                    EX = self._compilation_data.ptx_execution._get(context)
                    results += EX.cuda_run(inputs, num_trials, num_inputs_sets)
                full_results = self.parameters.results._get(context)
                if full_results is None:
                    full_results = results
                else:
                    full_results.extend(results)

                self.parameters.results._set(full_results, context)

                if self._is_learning(context):
                    # copies back matrix to pnl from param struct (after learning)
                    _comp_ex._copy_params_to_pnl(context=context)

                # KAM added the [-1] index after changing Composition run()
                # behavior to return only last trial of run (11/7/18)
                self.most_recent_context = context

                return full_results[-1]

            except Exception as e:
                if bin_execute is not True:
                    raise e from None

                warnings.warn("Failed to run `{}': {}".format(self.name, str(e)))

        # Reset gym forager environment for the current trial
        if self.env:
            trial_output = np.atleast_2d(self.env.reset())

        # Loop over the length of the list of inputs - each input represents a TRIAL
        for trial_num in range(num_trials):
            # Execute call before trial "hook" (user defined function)
            if call_before_trial:
                call_with_pruned_args(call_before_trial, context=context)

            if termination_processing[TimeScale.RUN].is_satisfied(
                scheduler=scheduler,
                context=context
            ):
                break

            # PROCESSING ------------------------------------------------------------------------
            # Prepare stimuli from the outside world  -- collect the inputs for this TRIAL and store them in a dict
            if callable(inputs):
                next_inputs = inputs(trial_num)
            elif isgenerator(inputs):
                try:
                    next_inputs = inputs.__next__()
                except StopIteration:
                    break

            if callable(inputs) or isgenerator(inputs):
                next_inputs, num_inputs_sets = self._adjust_stimulus_dict(next_inputs)
                execution_stimuli = {}
                for node in next_inputs:
                    if len(next_inputs[node]) == 1:
                        execution_stimuli[node] = next_inputs[node][0]
                        continue
                    else:
                        input_type = 'Generator' if isgenerator(inputs) else 'Function'
                        raise CompositionError(f"{input_type}s used for Composition input must return one trial's "
                                               f"worth of input on each call. Current generator returned "
                                               f"{len(next_inputs[node])} trials' worth of input on its last call.")
            else:
                execution_stimuli = {}
                stimulus_index = trial_num % num_inputs_sets
                for node in inputs:
                    if len(inputs[node]) == 1:
                        execution_stimuli[node] = inputs[node][0]
                        continue
                    execution_stimuli[node] = inputs[node][stimulus_index]

            for node in self.nodes:
                if hasattr(node, "reinitialize_when") and node.parameters.has_initializers._get(context):
                    if node.reinitialize_when.is_satisfied(scheduler=self.scheduler,
                                                           context=context):
                        node.reinitialize(None, context=context)

            # execute processing
            # pass along the stimuli for this trial

            trial_output = self.execute(inputs=execution_stimuli,
                                        scheduler=scheduler,
                                        termination_processing=termination_processing,
                                        call_before_time_step=call_before_time_step,
                                        call_before_pass=call_before_pass,
                                        call_after_time_step=call_after_time_step,
                                        call_after_pass=call_after_pass,
                                        context=context,
                                        base_context=base_context,
                                        clamp_input=clamp_input,
                                        runtime_params=runtime_params,
                                        skip_initialization=True,
                                        bin_execute=bin_execute,
                                        )

            # ---------------------------------------------------------------------------------
            # store the result of this execute in case it will be the final result

            # object.results.append(result)
            if isinstance(trial_output, collections.abc.Iterable):
                result_copy = trial_output.copy()
            else:
                result_copy = trial_output

            if ContextFlags.SIMULATION not in context.execution_phase:
                results.append(result_copy)

                if not self.parameters.retain_old_simulation_data._get():
                    if self.controller is not None:
                        # if any other special parameters store simulation info that needs to be cleaned up
                        # consider dedicating a function to it here
                        # this will not be caught above because it resides in the base context (context)
                        if not self.parameters.simulation_results.retain_old_simulation_data:
                            self.parameters.simulation_results._get(context).clear()

                        if not self.controller.parameters.simulation_ids.retain_old_simulation_data:
                            self.controller.parameters.simulation_ids._get(context).clear()

            if call_after_trial:
                call_with_pruned_args(call_after_trial, context=context)

        # Reset input spec for next trial
        self.parameters.input_specification._set(None, context)

        scheduler.get_clock(context)._increment_time(TimeScale.RUN)

        full_results = self.parameters.results._get(context)
        if full_results is None:
            full_results = results
        else:
            full_results.extend(results)

        self.parameters.results._set(full_results, context)

        self.most_recent_context = context

        if self._animate is not False:
            # Save list of gifs in self._animation as movie file
            movie_path = self._animation_directory + '/' + self._movie_filename
            self._animation[0].save(fp=movie_path,
                                    format='GIF',
                                    save_all=True,
                                    append_images=self._animation[1:],
                                    duration=self._image_duration * 1000,
                                    loop=0)
            # print(f'\nSaved movie for {self.name} in {self._animation_directory}/{self._movie_filename}')
            print(f"\nSaved movie for '{self.name}' in '{self._movie_filename}'")
            if self._show_animation:
                movie = Image.open(movie_path)
                movie.show()

        # Undo override of reinitialize_when conditions
        for node in self.nodes:
            try:
                node.reinitialize_when = self._reinitialize_nodes_when_cache[node]
            except KeyError:
                pass

        return trial_output

    @handle_external_context()
    def learn(
            self,
            inputs: dict,
            targets: dict = None,
            num_trials: int = None,
            epochs: int = 1,
            minibatch_size: int = 1,
            patience: int = None,
            min_delta: int = 0,
            context: Context = None,
            bin_execute=False,
            randomize_minibatches=False,
            call_before_minibatch = None,
            call_after_minibatch = None,
            *args,
            **kwargs
            ):
        """
            Runs the composition in learning mode - that is, any components with disable_learning False will be executed in learning mode. See `Composition_Learning` for details.

            Arguments
            ---------

            inputs: { `Mechanism <Mechanism>` or `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each node in the composition that receives inputs from
                the user. There are several equally valid ways that this dict could be structured:
                1. For each pair, the key is the node (Mechanism or Composition) and the value is an input,
                the shape of which must match the node's default variable. This is identical to the input dict in `the run method <Composition.run>`
                2. A dict with keys 'inputs', 'targets', and 'epochs'. The `inputs` key stores a dict that is the same structure as input specification (1) of learn. The `targets` and `epochs` keys
                should contain values of the same shape as `targets <Composition.learn>` and `epochs` <Composition.learn>

            targets: { `Mechanism <Mechanism>` or `Composition <Composition>` : list }
                a dictionary containing a key-value pair for each node in the composition that receives target values to train on from
                the user. This could be either target mechanisms or output nodes in backpropagation learning pathways.

            num_trials : int (default=None)
                typically, the composition will infer the number of trials from the length of its input specification.
                To reuse the same inputs across many trials, you may specify an input dictionary with lists of length 1,
                or use default inputs, and select a number of trials with num_trials.

            epochs : int (default=1)
                specifies the number of training epochs (that is, repetitions of the batched input set) to run with

            minibatch_size : int (default=1)
                specifies the size of the minibatches to use. The input trials will be batched and ran, after which learning mechanisms with learning mode TRIAL will update weights

            randomize_minibatch: bool (default=False)
                specifies whether the order of the input trials should be randomized on each epoch

            patience : int or None (default=None)
                **patience** allows the model to stop training early, if training stops reducing loss. The model tracks how many
                consecutive epochs of training have failed to reduce the model's loss. When this number exceeds **patience**,
                the model stops training early. If **patience** is ``None``, the model will train for the number
                of specified epochs and will not stop training early.

             min_delta : float (default=0)
                the minimum reduction in average loss that an epoch must provide in order to qualify as a 'good' epoch.
                Used for early stopping of training, in combination with **patience**.

            scheduler : Scheduler
                the scheduler object that owns the conditions that will instruct the execution of this Composition
                If not specified, the Composition will use its automatically generated scheduler.

            context
                context will be set to self.default_execution_id if unspecified

            call_before_minibatch : callable
                called before each minibatch is executed

            call_after_minibatch : callable
                called after each minibatch is executed

            Returns
            ---------

            the results of the final epoch of training
        """
        from psyneulink.library.compositions import CompositionRunner
        runner = CompositionRunner(self)

        self._analyze_graph()
        context.add_flag(ContextFlags.LEARNING_MODE)

        learning_results = runner.run_learning(
            inputs=inputs,
            targets=targets,
            num_trials=num_trials,
            epochs=epochs,
            minibatch_size=minibatch_size,
            patience=patience,
            min_delta=min_delta,
            randomize_minibatches=randomize_minibatches,
            call_before_minibatch=call_before_minibatch,
            call_after_minibatch=call_after_minibatch,
            context=context,
            bin_execute=bin_execute,
            *args, **kwargs)

        context.remove_flag(ContextFlags.LEARNING_MODE)
        return learning_results

    @handle_external_context(execution_phase=ContextFlags.PROCESSING)
    def execute(
            self,
            inputs=None,
            scheduler=None,
            termination_processing=None,
            call_before_time_step=None,
            call_before_pass=None,
            call_after_time_step=None,
            call_after_pass=None,
            context=None,
            base_context=Context(execution_id=None),
            clamp_input=SOFT_CLAMP,
            runtime_params=None,
            skip_initialization=False,
            bin_execute=False,
            ):
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

            scheduler : Scheduler
                the scheduler object that owns the conditions that will instruct the execution of this Composition
                If not specified, the Composition will use its automatically generated scheduler.

            context
                context will be set to self.default_execution_id if unspecified

            base_context
                the context corresponding to the execution context from which this execution will be initialized,
                if values currently do not exist for **context**

            call_before_time_step : callable
                called before each `TIME_STEP` is executed
                passed the current *context* (but it is not necessary for your callable to take)

            call_after_time_step : callable
                called after each `TIME_STEP` is executed
                passed the current *context* (but it is not necessary for your callable to take)

            call_before_pass : callable
                called before each `PASS` is executed
                passed the current *context* (but it is not necessary for your callable to take)

            call_after_pass : callable
                called after each `PASS` is executed
                passed the current *context* (but it is not necessary for your callable to take)

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
        self._assign_execution_ids(context)

        context.composition = self

        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)

        execution_scheduler = scheduler or self.scheduler

        context.source = ContextFlags.COMPOSITION

        if termination_processing is None:
            termination_processing = self.termination_processing

        # Skip initialization if possible (for efficiency):
        # - and(context has not changed
        # -     structure of the graph has not changed
        # -     not a nested composition
        # -     its not a simulation)
        # - or(gym forage env is being used)
        # (e.g., when run is called externally repeated for the same environment)
        # KAM added HACK below "or self.env is None" in order to merge in interactive inputs fix for speed improvement
        # TBI: Clean way to call _initialize_from_context if context has not changed, BUT composition has changed
        # for example:
        # comp.run()
        # comp.add_node(new_node)
        # comp.run().
        # context has not changed on the comp, BUT new_node's execution id needs to be set from None --> ID
        if self.most_recent_context != context or self.env is None:
            # initialize from base context but don't overwrite any values already set for this context
            if (
                not skip_initialization
                and not nested
                or context is None
                and context.execution_phase is not ContextFlags.SIMULATION
            ):
                self._initialize_from_context(context, base_context, override=False)
                context.composition = self

        # Generate first frame of animation without any active_items
        if self._animate is not False:
            # If context fails, the scheduler has no data for it yet.
            # It also may be the first, so fall back to default execution_id
            try:
                self._animate_execution(INITIAL_FRAME, context)
            except KeyError:
                old_eid = context.execution_id
                context.execution_id = self.default_execution_id
                self._animate_execution(INITIAL_FRAME, context)
                context.execution_id = old_eid

        # Reinitialize any nodes that have satisfied 'reinitialize_when'
        # conditions. This mimics the behavior used for Systems. (only
        # checking for reinitialization at the beginning of a trial)
        for node in self.nodes:
            if node.parameters.has_initializers._get(context):
                try:
                    if (
                        node.reinitialize_when.is_satisfied(
                            scheduler=execution_scheduler,
                            context=context
                        )
                    ):
                        node.reinitialize(None, context=context)
                except AttributeError:
                    pass

        # EXECUTE INPUT CIM ********************************************************************************************

        # FIX: 6/12/19 MOVE TO EXECUTE BELOW? (i.e., with bin_execute / _comp_ex.execute_node(self.input_CIM, inputs))
        # Handles Input CIM and Parameter CIM execution.
        #
        # FIX: 8/21/19
        # If self is a nested composition, its input CIM will obtain its value in one of two ways,
        # depending on whether or not it is being executed within a simulation.
        # If it is a simulation, then we need to use the _assign_values_to_input_CIM method, which parses the inputs
        # argument of the execute method into a suitable shape for the input ports of the input_CIM.
        # If it is not a simulation, we can simply execute the input CIM.
        #
        # If self is an unnested composition, we must update the input ports for any input nodes that are Compositions.
        # This is done to update the variable for their input CIMs, which allows the _adjust_execution_stimuli
        # method to properly validate input for those nodes.
        # -DS
        context.add_flag(ContextFlags.PROCESSING)
        if nested:
            # check that inputs are specified - autodiff does not in some cases
            if ContextFlags.SIMULATION in context.execution_phase and inputs is not None:
                inputs = self._adjust_execution_stimuli(inputs)
                self._assign_values_to_input_CIM(inputs, context=context)
            else:
                self.input_CIM.execute(context=context)
            self.parameter_CIM.execute(context=context)
        else:
            inputs = self._adjust_execution_stimuli(inputs)
            self._assign_values_to_input_CIM(inputs, context=context)

        # FIX: 6/12/19 Deprecate?
        # Manage input clamping

        # 1 because call_before_pass is called before the main
        # scheduler loop to ensure it happens regardless of whether
        # the scheduler terminates a trial immediately
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
        if self._animate is not False and SHOW_CIM in self._animate and self._animate[SHOW_CIM]:
            self._animate_execution(self.input_CIM, context)
        # FIX: END
        context.remove_flag(ContextFlags.PROCESSING)

        # EXECUTE CONTROLLER (if specified for BEFORE) *****************************************************************

        # Compile controller execution (if compilation is specified) --------------------------------

        if bin_execute:
            is_simulation = (context is not None and
                             ContextFlags.SIMULATION in context.execution_phase)
            # Try running in Exec mode first
            if (bin_execute is True or str(bin_execute).endswith('Exec')):
                # There's no mode to execute simulations.
                # Simulations are run as part of the controller node wrapper.
                assert not is_simulation
                try:
                    if bin_execute is True or bin_execute.startswith('LLVM'):
                        _comp_ex = pnlvm.CompExecution(self, [context.execution_id])
                        _comp_ex.execute(inputs)
                        return _comp_ex.extract_node_output(self.output_CIM)
                    elif bin_execute.startswith('PTX'):
                        self.__ptx_initialize(context)
                        __execution = self._compilation_data.ptx_execution._get(context)
                        __execution.cuda_execute(inputs)
                        return __execution.extract_node_output(self.output_CIM)
                except Exception as e:
                    if bin_execute is not True:
                        raise e from None

                    warnings.warn("Failed to execute `{}': {}".format(self.name, str(e)))

            # Exec failed for some reason, we can still try node level bin_execute
            try:
                # Filter out mechanisms. Nested compositions are not executed in this mode
                # Filter out controller. Compilation of controllers is not supported yet
                mechanisms = [n for n in self._all_nodes
                              if isinstance(n, Mechanism) and (n is not self.controller or not is_simulation)]
                # Generate all mechanism wrappers
                for m in mechanisms:
                    self._get_node_wrapper(m)

                _comp_ex = pnlvm.CompExecution(self, [context.execution_id])
                # Compile all mechanism wrappers
                for m in mechanisms:
                    _comp_ex._set_bin_node(m)

                bin_execute = True
            except Exception as e:
                if bin_execute is not True:
                    raise e from None

                warnings.warn("Failed to compile wrapper for `{}' in `{}': {}".format(m.name, self.name, str(e)))
                bin_execute = False

        # Execute controller --------------------------------------------------------

        if (self.enable_controller and
            self.controller_mode is BEFORE and
            self.controller_condition.is_satisfied(scheduler=execution_scheduler,
                                                   context=context)):

            # control phase
            # FIX: SHOULD SET CONTEXT AS CONTROL HERE AND RESET AT END (AS DONE FOR animation BELOW)
            if (
                    self.initialization_status != ContextFlags.INITIALIZING
                    and ContextFlags.SIMULATION not in context.execution_phase
            ):
                if self.controller and not bin_execute:
                    # FIX: REMOVE ONCE context IS SET TO CONTROL ABOVE
                    # FIX: END REMOVE
                    context.add_flag(ContextFlags.PROCESSING)
                    self.controller.execute(context=context)

                if bin_execute:
                    _comp_ex.execute_node(self.controller)

                context.remove_flag(ContextFlags.PROCESSING)

                # Animate controller (before execution)
                context.add_flag(ContextFlags.CONTROL)
                if self._animate != False and SHOW_CONTROLLER in self._animate and self._animate[SHOW_CONTROLLER]:
                    self._animate_execution(self.controller, context)
                context.remove_flag(ContextFlags.CONTROL)

        # EXECUTE (each execution_set) *********************************************************************************

        # PREPROCESS (get inputs, call_before_pass, animate first frame) ----------------------------------

        context.add_flag(ContextFlags.PROCESSING)

        if bin_execute:
            _comp_ex.execute_node(self.input_CIM, inputs)
        #              WHY DO BOTH?  WHY NOT if-else?

        if call_before_pass:
            call_with_pruned_args(call_before_pass, context=context)

        # GET execution_set -------------------------------------------------------------------------
        # run scheduler to receive sets of nodes that may be executed at this time step in any order
        for next_execution_set in execution_scheduler.run(termination_conds=termination_processing,
                                                          context=context,
                                                          skip_trial_time_increment=True,
                                                          ):

            # SETUP EXECUTION ----------------------------------------------------------------------------

            # FIX: 6/12/19 WHY IS call_*after*_pass BEING CALLED BEFORE THE PASS?
            # KDM 1/15/20: Because we can't tell at the end of this
            # code block whether a PASS has ended or not. The scheduler
            # only modifies the pass after we receive an execution_set.
            # So, we only know a PASS has ended in retrospect after the
            # scheduler has changed the clock to indicate it. So, we
            # have to run call_after_pass before the next PASS (here) or
            # after this code block (see the call to call_after_pass
            # below)
            if call_after_pass:
                if next_pass_after == \
                        execution_scheduler.get_clock(context).get_total_times_relative(TimeScale.PASS,
                                                                                          TimeScale.TRIAL):
                    logger.debug('next_pass_after {0}\tscheduler pass {1}'.
                                 format(next_pass_after,
                                        execution_scheduler.get_clock(
                                            context).get_total_times_relative(
                                                TimeScale.PASS, TimeScale.TRIAL)))
                    call_with_pruned_args(call_after_pass, context=context)
                    next_pass_after += 1

            if call_before_pass:
                if next_pass_before == \
                        execution_scheduler.get_clock(context).get_total_times_relative(TimeScale.PASS,
                                                                                          TimeScale.TRIAL):
                    call_with_pruned_args(call_before_pass, context=context)
                    logger.debug('next_pass_before {0}\tscheduler pass {1}'.
                                 format(next_pass_before,
                                        execution_scheduler.get_clock(
                                            context).get_total_times_relative(
                                                TimeScale.PASS,
                                                TimeScale.TRIAL)))
                    next_pass_before += 1

            if call_before_time_step:
                call_with_pruned_args(call_before_time_step, context=context)

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
            if not self._is_learning(context):
                next_execution_set = next_execution_set - set(self.get_nodes_by_role(NodeRole.LEARNING))

            # ANIMATE execution_set ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self._animate is not False and self._animate_unit is EXECUTION_SET:
                self._animate_execution(next_execution_set, context)

            # EXECUTE (each node) --------------------------------------------------------------------------

            # execute each node with EXECUTING in context
            for node in next_execution_set:

                # Store values of all nodes in this execution_set for use by other nodes in the execution set
                #    throughout this timestep (e.g., for recurrent Projections)
                frozen_values[node] = node.get_output_values(context)

                # FIX: 6/12/19 Deprecate?
                # Handle input clamping
                if node in input_nodes:
                    if clamp_input:
                        if node in hard_clamp_inputs:
                            # clamp = HARD_CLAMP --> "turn off" recurrent projection
                            if hasattr(node, "recurrent_projection"):
                                node.recurrent_projection.sender.parameters.value._set([0.0], context)
                        elif node in no_clamp_inputs:
                            for input_port in node.input_ports:
                                self.input_CIM_ports[input_port][1].parameters.value._set(0.0, context)

                # EXECUTE A MECHANISM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                if isinstance(node, Mechanism):

                    execution_runtime_params = {}

                    if node in runtime_params:
                        for param in runtime_params[node]:
                            if runtime_params[node][param][1].is_satisfied(scheduler=execution_scheduler,
                                               # KAM 5/15/18 - not sure if this will always be the correct execution id:
                                                                           context=context):
                                execution_runtime_params[param] = runtime_params[node][param][0]

                    # Set context.execution_phase

                    # Set to PROCESSING by default
                    context.add_flag(ContextFlags.PROCESSING)

                    # Set to LEARNING if Mechanism receives any PathwayProjections that are being learned
                    #    for which learning_enabled == True or ONLINE (i.e., not False or AFTER)
                    #    Implementation Note: RecurrentTransferMechanisms are special cased as the AutoAssociativeMechanism
                    #    should be handling learning - not the RTM itself.
                    if self._is_learning(context) and not isinstance(node, RecurrentTransferMechanism):
                        projections = set(self.projections).intersection(set(node.path_afferents))
                        if any([p for p in projections if
                                any([a for a in p.parameter_ports[MATRIX].mod_afferents
                                     if (hasattr(a, 'learning_enabled') and a.learning_enabled in {True, ONLINE})])]):
                            context.replace_flag(ContextFlags.PROCESSING, ContextFlags.LEARNING)

                    # Execute node
                    if bin_execute:
                        _comp_ex.execute_node(node)
                    else:
                        if node is not self.controller:
                            if nested and node in self.get_nodes_by_role(NodeRole.INPUT):
                                for port in node.input_ports:
                                    port._update(context=context)
                            node.execute(
                                context=context,
                                runtime_params=execution_runtime_params,
                            )

                    # Reset runtime_params for node and its function if specified
                        if context.execution_id in node._runtime_params_reset:
                            for key in node._runtime_params_reset[context.execution_id]:
                                node._set_parameter_value(key, node._runtime_params_reset[context.execution_id][key],
                                                          context)
                        node._runtime_params_reset[context.execution_id] = {}

                        if context.execution_id in node.function._runtime_params_reset:
                            for key in node.function._runtime_params_reset[context.execution_id]:
                                node.function._set_parameter_value(
                                        key,
                                        node.function._runtime_params_reset[context.execution_id][key],
                                        context)

                        node.function._runtime_params_reset[context.execution_id] = {}

                    # Set execution_phase for node's context back to IDLE
                    if self._is_learning(context):
                        context.replace_flag(ContextFlags.LEARNING, ContextFlags.PROCESSING)
                    context.remove_flag(ContextFlags.PROCESSING)

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
                                srnode.output_ports[i].parameters.value._set(v, context, skip_history=True,
                                                                             skip_log=True)

                    # Pass outer context to nested Composition
                    context.composition = node
                    if ContextFlags.SIMULATION in context.execution_phase:
                        is_simulating = True
                        context.remove_flag(ContextFlags.SIMULATION)
                    else:
                        is_simulating = False


                    ret = node.execute(context=context)

                    if is_simulating:
                        context.add_flag(ContextFlags.SIMULATION)

                    context.composition = self

                    # Get output info from compiled execution
                    if bin_execute:
                        # Update result in binary data structure
                        _comp_ex.insert_node_output(node, ret)
                        for i, v in enumerate(ret):
                            # Set current output. This will be stored to "new_values" below
                            node.output_CIM.output_ports[i].parameters.value._set(v, context, skip_history=True,
                                                                                  skip_log=True)

                # ANIMATE node ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if self._animate is not False and self._animate_unit is COMPONENT:
                    self._animate_execution(node, context)


                # MANAGE INPUTS (for next execution_set)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # FIX: 6/12/19 Deprecate?
                # Handle input clamping
                if node in input_nodes:
                    if clamp_input:
                        if node in pulse_clamp_inputs:
                            for input_port in node.input_ports:
                                # clamp = None --> "turn off" input node
                                self.input_CIM_ports[input_port][1].parameters.value._set(0, context)

                # Store new value generated by node,
                #    then set back to frozen value for use by other nodes in execution_set
                new_values[node] = node.get_output_values(context)
                for i in range(len(node.output_ports)):
                    node.output_ports[i].parameters.value._set(frozen_values[node][i], context,
                                                               skip_history=True, skip_log=True)


            # Set all nodes to new values
            for node in next_execution_set:
                for i in range(len(node.output_ports)):
                    node.output_ports[i].parameters.value._set(new_values[node][i], context,
                                                               skip_history=True, skip_log=True)

            if call_after_time_step:
                call_with_pruned_args(call_after_time_step, context=context)

        context.remove_flag(ContextFlags.PROCESSING)

        #Update matrix parameter of PathwayProjections being learned with learning_enabled==AFTER
        from psyneulink.library.compositions.autodiffcomposition import AutodiffComposition
        if self._is_learning(context) and not isinstance(self, AutodiffComposition):
            context.add_flag(ContextFlags.LEARNING)
            for projection in [p for p in self.projections if
                               hasattr(p, 'has_learning_projection') and p.has_learning_projection]:
                matrix_parameter_port = projection.parameter_ports[MATRIX]
                if any([lp for lp in matrix_parameter_port.mod_afferents if lp.learning_enabled == AFTER]):
                    matrix_parameter_port._update(context=context)
            context.remove_flag(ContextFlags.LEARNING)

        if call_after_pass:
            call_with_pruned_args(call_after_pass, context=context)


        # Animate output_CIM
        # FIX: NOT SURE WHETHER IT CAN BE LEFT IN PROCESSING AFTER THIS -
        #      COORDINATE WITH REFACTORING OF PROCESSING/CONTROL CONTEXT
        if self._animate is not False and SHOW_CIM in self._animate and self._animate[SHOW_CIM]:
            self._animate_execution(self.output_CIM, context)
        # FIX: END

        # EXECUTE CONTROLLER (if controller_mode == AFTER) ************************************************************

        if (self.enable_controller and
                self.controller_mode == AFTER and
                self.controller_condition.is_satisfied(scheduler=execution_scheduler,
                                                       context=context)):
            # control phase
            if (
                    self.initialization_status != ContextFlags.INITIALIZING
                    and ContextFlags.SIMULATION not in context.execution_phase
            ):
                context.add_flag(ContextFlags.CONTROL)
                if self.controller and not bin_execute:
                    self.controller.execute(context=context)

                if bin_execute:
                    _comp_ex.freeze_values()
                    _comp_ex.execute_node(self.controller)



                # Animate controller (after execution)
                if self._animate is not False and SHOW_CONTROLLER in self._animate and self._animate[SHOW_CONTROLLER]:
                    self._animate_execution(self.controller, context)

                context.remove_flag(ContextFlags.CONTROL)

        execution_scheduler.get_clock(context)._increment_time(TimeScale.TRIAL)

        # REPORT RESULTS ***********************************************************************************************

        # Extract result here
        if bin_execute:
            _comp_ex.freeze_values()
            _comp_ex.execute_node(self.output_CIM)
            return _comp_ex.extract_node_output(self.output_CIM)

        context.add_flag(ContextFlags.PROCESSING)
        self.output_CIM.execute(context=context)
        context.remove_flag(ContextFlags.PROCESSING)

        output_values = []
        for port in self.output_CIM.output_ports:
            output_values.append(port.parameters.value._get(context))

        return output_values

    def _update_learning_parameters(self, context):
        pass

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, values, context=NotImplemented):
        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id

        for i in range(self.stateful_nodes):
            self.stateful_nodes[i].reinitialize(values[i], context=context)

    def disable_all_history(self):
        """
            When run, disables history tracking for all Parameters of all Components used in this Composition
        """
        self._set_all_parameter_properties_recursively(history_max_length=0)

    def _get_processing_condition_set(self, node):
        dep_group = []
        for group in self.scheduler.consideration_queue:
            if node in group:
                break
            dep_group = group

        # NOTE: This is not ideal we don't need to depend on
        # the entire previous group. Only our dependencies
        cond = [EveryNCalls(dep, 1) for dep in dep_group]
        if node not in self.scheduler.conditions:
            cond.append(Always())
        else:
            node_conds = self.scheduler.conditions[node]
            cond.append(node_conds)

        return All(*cond)

    def _input_matches_variable(self, input_value, var):
        # input_value ports are uniform
        if np.shape(np.atleast_2d(input_value)) == np.shape(var):
            return "homogeneous"
        # input_value ports have different lengths
        elif len(np.shape(var)) == 1 and isinstance(var[0], (list, np.ndarray)):
            for i in range(len(input_value)):
                if len(input_value[i]) != len(var[i]):
                    return False
            return "heterogeneous"
        return False

    def _is_learning(self, context):
        """Returns true if this composition can learn in the given context"""
        return (not self.disable_learning) and (ContextFlags.LEARNING_MODE in context.runmode)

    def _adjust_stimulus_dict(self, stimuli):

        # STEP 1A: Check that all of the nodes listed in the inputs dict are INPUT nodes in the composition
        input_nodes = self.get_nodes_by_role(NodeRole.INPUT)
        for node in stimuli.keys():
            if not node in input_nodes:
                if not isinstance(node, (Mechanism, Composition)):
                    raise CompositionError(f'{node} in "inputs" dict for {self.name} is not a '
                                           f'{Mechanism.__name__} or {Composition.__name__}.')
                else:
                    raise CompositionError(f"{node.name} in inputs dict for {self.name} is not one of its INPUT nodes.")

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

                    adjusted_stimulus_dict, num_trials = node._adjust_stimulus_dict(stim_list)
                    translated_stimulus_dict = {}

                    # first time through the stimulus dictionary, assemble a dictionary in which the keys are input CIM
                    # InputPorts and the values are lists containing the first input value
                    for nested_input_node, values in adjusted_stimulus_dict.items():
                        first_value = values[0]
                        for i in range(len(first_value)):
                            input_port = nested_input_node.external_input_ports[i]
                            input_cim_input_port = node.input_CIM_ports[input_port][0]
                            translated_stimulus_dict[input_cim_input_port] = [first_value[i]]
                            # then loop through the stimulus dictionary again for each remaining trial
                            for trial in range(1, num_trials):
                                translated_stimulus_dict[input_cim_input_port].append(values[trial][i])

                    adjusted_stimulus_list = []
                    for trial in range(num_trials):
                        trial_adjusted_stimulus_list = []
                        for port in node.external_input_ports:
                            trial_adjusted_stimulus_list.append(translated_stimulus_dict[port][trial])
                        adjusted_stimulus_list.append(trial_adjusted_stimulus_list)
                    stimuli[node] = adjusted_stimulus_list
                    stim_list = adjusted_stimulus_list  # ADDED CW 12/21/18: This line fixed a bug, but it might be a hack

            # excludes any input ports marked "internal_only" (usually recurrent)
            # KDM 3/29/19: changed to use defaults equivalent of node.external_input_values
            input_must_match = [input_port.defaults.value for input_port in node.input_ports
                                if not input_port.internal_only]

            if input_must_match == []:
                # all input ports are internal_only
                continue

            check_spec_type = self._input_matches_variable(stim_list, input_must_match)
            # If a node provided a single input, wrap it in one more list in order to represent trials
            if check_spec_type == "homogeneous" or check_spec_type == "heterogeneous":
                if check_spec_type == "homogeneous":
                    # np.atleast_2d will catch any single-input ports specified without an outer list
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
                                                " to represent the outside stimulus for the inhibition InputPort, and " \
                                                "for systems, put your inputs"
                        raise RunError(err_msg)
                    elif check_spec_type == "homogeneous":
                        # np.atleast_2d will catch any single-input ports specified without an outer list
                        # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                        adjusted_stimuli[node].append(np.atleast_2d(stim))
                    else:
                        adjusted_stimuli[node].append(stim)
                nums_input_sets.add(len(stimuli[node]))

        num_trials = max(nums_input_sets)
        for node, stim in adjusted_stimuli.items():
            if len(stim) == 1:
                adjusted_stimuli[node] *= max(nums_input_sets)
        nums_input_sets.discard(1)
        if len(nums_input_sets) > 1:
            raise CompositionError("The input dictionary for {} contains input specifications of different "
                                    "lengths ({}). The same number of inputs must be provided for each node "
                                    "in a Composition.".format(self.name, nums_input_sets))
        return adjusted_stimuli, num_trials

    def _adjust_execution_stimuli(self, stimuli):
        adjusted_stimuli = {}
        for node, stimulus in stimuli.items():
            if isinstance(node, Composition):
                input_must_match = node.default_external_input_values
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
                    # np.atleast_2d will catch any single-input ports specified without an outer list
                    # e.g. [2.0, 2.0] --> [[2.0, 2.0]]
                    adjusted_stimuli[node] = np.atleast_2d(stimulus)
                else:
                    adjusted_stimuli[node] = stimulus
            else:
                raise CompositionError("Input stimulus ({}) for {} is incompatible with its variable ({})."
                                       .format(stimulus, node.name, input_must_match))
        return adjusted_stimuli

    def _assign_values_to_input_CIM(self, inputs, context=None):
        """
            Assign values from input dictionary to the InputPorts of the Input CIM, then execute the Input CIM

        """

        build_CIM_input = []

        for input_port in self.input_CIM.input_ports:
            # "input_port" is an InputPort on the input CIM

            for key in self.input_CIM_ports:
                # "key" is an InputPort on an origin Node of the Composition
                if self.input_CIM_ports[key][0] == input_port:
                    origin_input_port = key
                    origin_node = key.owner
                    index = origin_node.input_ports.index(origin_input_port)

                    if isinstance(origin_node, CompositionInterfaceMechanism):
                        index = origin_node.input_ports.index(origin_input_port)
                        origin_node = origin_node.composition

                    if origin_node in inputs:
                        value = inputs[origin_node][index]

                    else:
                        value = origin_node.defaults.variable[index]

            build_CIM_input.append(value)

        self.input_CIM.execute(build_CIM_input, context=context)

    def _assign_execution_ids(self, context=None):
        """
            assigns the same execution id to each Node in the composition's processing graph as well as the CIMs.
            he execution id is either specified in the user's call to run(), or from the Composition's
            **default_execution_id**
        """

        # Traverse processing graph and assign one execution_id to all of its nodes
        if context.execution_id is None:
            context.execution_id = self.default_execution_id

        if context.execution_id not in self.execution_ids:
            self.execution_ids.add(context.execution_id)

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

    def _after_agent_rep_execution(self, context=None):
        pass


    # ******************************************************************************************************************
    #                                           LLVM
    # ******************************************************************************************************************

    @property
    def _inner_projections(self):
        # PNL considers afferent projections to input_CIM to be part
        # of the nested composition. Filter them out.
        return (p for p in self.projections
                  if p.receiver.owner is not self.input_CIM and
                     p.receiver.owner is not self.parameter_CIM)

    def _get_param_struct_type(self, ctx):
        node_param_type_list = (ctx.get_param_struct_type(m) for m in self._all_nodes)
        proj_param_type_list = (ctx.get_param_struct_type(p) for p in self._inner_projections)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(node_param_type_list),
            pnlvm.ir.LiteralStructType(proj_param_type_list)))

    def _get_state_struct_type(self, ctx):
        node_state_type_list = (ctx.get_state_struct_type(m) for m in self._all_nodes)
        proj_state_type_list = (ctx.get_state_struct_type(p) for p in self._inner_projections)
        return pnlvm.ir.LiteralStructType((
            pnlvm.ir.LiteralStructType(node_state_type_list),
            pnlvm.ir.LiteralStructType(proj_state_type_list)))

    def _get_input_struct_type(self, ctx):
        pathway = ctx.get_input_struct_type(self.input_CIM)
        if not self.parameter_CIM.afferents:
            return pathway
        modulatory = ctx.get_input_struct_type(self.parameter_CIM)
        return pnlvm.ir.LiteralStructType((pathway, modulatory))

    def _get_output_struct_type(self, ctx):
        return ctx.get_output_struct_type(self.output_CIM)

    def _get_data_struct_type(self, ctx):
        output_type_list = (ctx.get_output_struct_type(n) for n in self._all_nodes)
        output_type = pnlvm.ir.LiteralStructType(output_type_list)
        nested_types = (ctx.get_data_struct_type(n) for n in self._all_nodes)
        return pnlvm.ir.LiteralStructType((output_type, *nested_types))

    def _get_state_initializer(self, context):
        node_states = (m._get_state_initializer(context=context) for m in self._all_nodes)
        proj_states = (p._get_state_initializer(context=context) for p in self._inner_projections)
        return (tuple(node_states), tuple(proj_states))

    def _get_param_initializer(self, context):
        node_params = (m._get_param_initializer(context) for m in self._all_nodes)
        proj_params = (p._get_param_initializer(context) for p in self._inner_projections)
        return (tuple(node_params), tuple(proj_params))

    def _get_data_initializer(self, context):
        output_data = ((os.parameters.value.get(context) for os in m.output_ports) for m in self._all_nodes)
        nested_data = (getattr(node, '_get_data_initializer', lambda _: ())(context)
                       for node in self._all_nodes)
        return (pnlvm._tupleize(output_data), *nested_data)

    def _get_node_index(self, node):
        node_list = list(self._all_nodes)
        return node_list.index(node)

    def _gen_llvm_function(self, *, tags:frozenset):
        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            if "run" in tags:
                return ctx.gen_composition_run(self, tags=tags)
            else:
                return ctx.gen_composition_exec(self, tags=tags)

    @handle_external_context(execution_id=NotImplemented)
    def reinitialize(self, context=None):
        if context.execution_id is NotImplemented:
            context.execution_id = self.most_recent_context.execution_id

        self._compilation_data.ptx_execution.set(None, context)
        self._compilation_data.parameter_struct.set(None, context)
        self._compilation_data.state_struct.set(None, context)
        self._compilation_data.data_struct.set(None, context)
        self._compilation_data.scheduler_conditions.set(None, context)

    def __ptx_initialize(self, context=None, additional_tags=frozenset()):
        if self._compilation_data.ptx_execution._get(context) is None:
            self._compilation_data.ptx_execution._set(pnlvm.CompExecution(self, [context.execution_id], additional_tags=additional_tags), context)

    def _get_node_wrapper(self, node):
        """Return a (memoized) wrapper instance that generates node invocation sequence.

           Since nodes in general don't know about their owning composition this structure ties together composition and its nodes.
        """
        if node not in self.__generated_node_wrappers:
            class node_wrapper():
                def __init__(self, node, gen_f):
                    self._node = node
                    self._gen_f = gen_f
                def _gen_llvm_function(self, *, tags:frozenset):
                    return self._gen_f(self._node, tags=tags)
            wrapper = node_wrapper(node, self._gen_node_wrapper)
            self.__generated_node_wrappers[node] = wrapper
            return wrapper

        return self.__generated_node_wrappers[node]

    def _gen_node_wrapper(self, node, *, tags:frozenset):
        assert "node_wrapper" in tags
        func_tags = tags.difference({"node_wrapper"})
        is_mech = isinstance(node, Mechanism)

        with pnlvm.LLVMBuilderContext.get_global() as ctx:
            node_function = ctx.import_llvm_function(node, tags=func_tags)

            data_struct_ptr = ctx.get_data_struct_type(self).as_pointer()
            args = [
                ctx.get_state_struct_type(self).as_pointer(),
                ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_input_struct_type(self).as_pointer(),
                data_struct_ptr, data_struct_ptr]

            if not is_mech and "reinitialize" not in tags:
                # Add condition struct of the parent composition
                # This includes structures of all nested compositions
                cond_gen = pnlvm.helpers.ConditionGenerator(ctx, self)
                cond_ty = cond_gen.get_condition_struct_type().as_pointer()
                args.append(cond_ty)

            builder = ctx.create_llvm_function(args, node, "comp_wrap_" + node_function.name)
            llvm_func = builder.function
            llvm_func.attributes.add('alwaysinline')
            for a in llvm_func.args:
                a.attributes.add('nonnull')

            context, params, comp_in, data_in, data_out = llvm_func.args[:5]

            if node is self.input_CIM:
                # if there are incoming modulatory projections,
                # the input structure is shared
                if self.parameter_CIM.afferents:
                    node_in = builder.gep(comp_in, [ctx.int32_ty(0), ctx.int32_ty(0)])
                else:
                    node_in = comp_in
                incoming_projections = []
            elif node is self.parameter_CIM and node.afferents:
                # if parameter_CIM has afferent projections,
                # their values are in comp_in[1]
                node_in = builder.gep(comp_in, [ctx.int32_ty(0), ctx.int32_ty(1)])
                # And we run no further projection
                incoming_projections = []
            elif not is_mech:
                node_in = builder.alloca(node_function.args[2].type.pointee)
                incoming_projections = node.input_CIM.afferents + node.parameter_CIM.afferents
            else:
                # this path also handles parameter_CIM with no afferent
                # projections. 'comp_in' does not include any extra values,
                # and the entire call should be optimized out.
                node_in = builder.alloca(node_function.args[2].type.pointee)
                incoming_projections = node.afferents

            if "reinitialize" in tags:
                incoming_projections = []

            # Execute all incoming projections
            inner_projections = list(self._inner_projections)
            for proj in incoming_projections:
                # Skip autoassociative projections
                if proj.sender.owner is proj.receiver.owner:
                    continue

                # Get location of projection input data
                par_mech = proj.sender.owner
                if par_mech in self._all_nodes:
                    par_idx = self._get_node_index(par_mech)
                else:
                    comp = par_mech.composition
                    assert par_mech is comp.output_CIM
                    par_idx = self.nodes.index(comp)

                output_s = proj.sender
                assert output_s in par_mech.output_ports
                output_port_idx = par_mech.output_ports.index(output_s)
                proj_in = builder.gep(data_in, [ctx.int32_ty(0),
                                                ctx.int32_ty(0),
                                                ctx.int32_ty(par_idx),
                                                ctx.int32_ty(output_port_idx)])

                # Get location of projection output (in mechanism's input structure
                rec_port = proj.receiver
                assert rec_port.owner is node or rec_port.owner is node.input_CIM or rec_port.owner is node.parameter_CIM
                indices = [0]
                if proj in rec_port.owner.path_afferents:
                    rec_port_idx = rec_port.owner.input_ports.index(rec_port)

                    assert proj in rec_port.pathway_projections
                    projection_idx = rec_port.pathway_projections.index(proj)

                    # Adjust for AutoAssociative projections
                    for i in range(projection_idx):
                        if isinstance(rec_port.pathway_projections[i], AutoAssociativeProjection):
                            projection_idx -= 1
                    if not is_mech and node.parameter_CIM.afferents:
                        # If there are afferent projections to parameter_CIM
                        # the input structure is split between input_CIM
                        # and parameter_CIM
                        if proj in node.parameter_CIM.afferents:
                            # modulatory projection
                            indices.append(1)
                        else:
                            # pathway projection
                            indices.append(0)
                    indices.extend([rec_port_idx, projection_idx])
                elif proj in rec_port.owner.mod_afferents:
                    # Only mechanism ports list mod projections in mod_afferents
                    assert is_mech
                    projection_idx = rec_port.owner.mod_afferents.index(proj)
                    indices.extend([len(rec_port.owner.input_ports), projection_idx])
                else:
                    assert False, "Projection neither pathway nor modulatory"

                proj_out = builder.gep(node_in, [ctx.int32_ty(i) for i in indices])

                # Get projection parameters and state
                proj_idx = inner_projections.index(proj)
                # Projections are listed second in param and state structure
                proj_params = builder.gep(params, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(proj_idx)])
                proj_context = builder.gep(context, [ctx.int32_ty(0), ctx.int32_ty(1), ctx.int32_ty(proj_idx)])
                proj_function = ctx.import_llvm_function(proj, tags=func_tags)

                if proj_out.type != proj_function.args[3].type:
                    warnings.warn("Shape mismatch: Projection ({}) results does not match the receiver state({}) input: {} vs. {}".format(proj, proj.receiver, proj.defaults.value, proj.receiver.defaults.variable))
                    proj_out = builder.bitcast(proj_out, proj_function.args[3].type)
                builder.call(proj_function, [proj_params, proj_context, proj_in, proj_out])

            idx = ctx.int32_ty(self._get_node_index(node))
            zero = ctx.int32_ty(0)
            node_params = builder.gep(params, [zero, zero, idx])
            node_context = builder.gep(context, [zero, zero, idx])
            node_out = builder.gep(data_out, [zero, zero, idx])
            if is_mech:
                call_args = [node_params, node_context, node_in, node_out]
                if len(node_function.args) > 4:
                    assert node is self.controller
                    call_args += [params, context, data_in]
                builder.call(node_function, call_args)
            elif "reinitialize" not in tags:
                # FIXME: reinitialization of compositions is not supported
                # Condition and data structures includes parent first
                nested_idx = ctx.int32_ty(self._get_node_index(node) + 1)
                node_data = builder.gep(data_in, [zero, nested_idx])
                node_cond = builder.gep(llvm_func.args[5], [zero, nested_idx])
                builder.call(node_function, [node_context, node_params, node_in,
                                             node_data, node_cond])
                # Copy output of the nested composition to its output place
                output_idx = node._get_node_index(node.output_CIM)
                result = builder.gep(node_data, [zero, zero, ctx.int32_ty(output_idx)])
                builder.store(builder.load(result), node_out)

            builder.ret_void()

        return llvm_func

    def enable_logging(self):
        for item in self.nodes + self.projections:
            if isinstance(item, Composition):
                item.enable_logging()
            elif not isinstance(item, CompositionInterfaceMechanism):
                for param in item.parameters:
                    if param.loggable and param.log_condition is LogCondition.OFF:
                        param.log_condition = LogCondition.EXECUTION

    @property
    def _dict_summary(self):
        scheduler_dict = {
            str(ContextFlags.PROCESSING): self.scheduler._dict_summary
        }

        super_summary = super()._dict_summary

        try:
            super_summary[self._model_spec_id_parameters][MODEL_SPEC_ID_PSYNEULINK]['schedulers'] = scheduler_dict
        except KeyError:
            super_summary[self._model_spec_id_parameters][MODEL_SPEC_ID_PSYNEULINK] = {}
            super_summary[self._model_spec_id_parameters][MODEL_SPEC_ID_PSYNEULINK]['schedulers'] = scheduler_dict

        nodes_dict = {MODEL_SPEC_ID_PSYNEULINK: {}}
        projections_dict = {MODEL_SPEC_ID_PSYNEULINK: {}}

        additional_projections = []
        additional_nodes = (
            [self.controller]
            if self.controller is not None
            else []
        )

        for n in list(self.nodes) + additional_nodes:
            if not isinstance(n, CompositionInterfaceMechanism):
                nodes_dict[n.name] = n._dict_summary

            # consider making this more general in the future
            try:
                additional_projections.extend(n.control_projections)
            except AttributeError:
                pass

        for p in list(self.projections) + additional_projections:
            has_cim_sender = isinstance(
                p.sender.owner,
                CompositionInterfaceMechanism
            )
            has_cim_receiver = isinstance(
                p.receiver.owner,
                CompositionInterfaceMechanism
            )

            # filter projections to/from CIMs, unless they are to embedded
            # compositions (any others should be automatically generated)
            if (
                (not has_cim_sender or p.sender.owner.composition in self.nodes)
                and (
                    not has_cim_receiver
                    or p.receiver.owner.composition in self.nodes
                )
            ):
                p_summary = p._dict_summary

                if has_cim_sender:
                    p_summary[MODEL_SPEC_ID_SENDER_MECH] = p.sender.owner.composition.name

                if has_cim_receiver:
                    p_summary[MODEL_SPEC_ID_RECEIVER_MECH] = p.receiver.owner.composition.name

                projections_dict[p.name] = p_summary

        if len(nodes_dict[MODEL_SPEC_ID_PSYNEULINK]) == 0:
            del nodes_dict[MODEL_SPEC_ID_PSYNEULINK]

        if len(projections_dict[MODEL_SPEC_ID_PSYNEULINK]) == 0:
            del projections_dict[MODEL_SPEC_ID_PSYNEULINK]

        return {
            MODEL_SPEC_ID_COMPOSITION: [{
                **super_summary,
                **{
                    MODEL_SPEC_ID_NODES: nodes_dict,
                    MODEL_SPEC_ID_PROJECTIONS: projections_dict,
                    'controller': self.controller,
                }
            }]
        }

    # ******************************************************************************************************************
    #                                           PROPERTIES
    # ******************************************************************************************************************

    @property
    def input_ports(self):
        """Returns all InputPorts that belong to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_ports

    @property
    def output_ports(self):
        """Returns all OutputPorts that belong to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_ports

    @property
    def output_values(self):
        """Returns values of all OutputPorts that belong to the Output CompositionInterfaceMechanism"""
        return self.get_output_values()

    def get_output_values(self, context=None):
        return [output_port.parameters.value.get(context) for output_port in self.output_CIM.output_ports]

    @property
    def input_port(self):
        """Returns the index 0 InputPort that belongs to the Input CompositionInterfaceMechanism"""
        return self.input_CIM.input_ports[0]

    @property
    def input_values(self):
        """Returns values of all InputPorts that belong to the Input CompositionInterfaceMechanism"""
        return self.get_input_values()

    def get_input_values(self, context=None):
        return [input_port.parameters.value.get(context) for input_port in self.input_CIM.input_ports]

    @property
    def runs_simulations(self):
        return True

    @property
    def simulation_results(self):
        return self.parameters.simulation_results.get(self.default_execution_id)

    #  For now, external_input_ports == input_ports and external_input_values == input_values
    #  They could be different in the future depending on new features (ex. if we introduce recurrent compositions)
    #  Useful to have this property for treating Compositions the same as Mechanisms in run & execute
    @property
    def external_input_ports(self):
        """Returns all external InputPorts that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_port for input_port in self.input_CIM.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def external_input_values(self):
        """Returns values of all external InputPorts that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_port.value for input_port in self.input_CIM.input_ports if not input_port.internal_only]
        except (TypeError, AttributeError):
            return None

    @property
    def default_external_input_values(self):
        """Returns the default values of all external InputPorts that belong to the Input CompositionInterfaceMechanism"""
        try:
            return [input_port.defaults.value for input_port in self.input_CIM.input_ports if
                    not input_port.internal_only]
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
    def output_port(self):
        """Returns the index 0 OutputPort that belongs to the Output CompositionInterfaceMechanism"""
        return self.output_CIM.output_ports[0]

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
            [self.input_CIM, self.output_CIM, self.parameter_CIM],
            [self.controller] if self.controller is not None else []
        ))

    @property
    def learning_components(self):
        return [node for node in self.nodes if (NodeRole.LEARNING in self.nodes_to_roles[node] or
                                                NodeRole.AUTOASSOCIATIVE_LEARNING in self.nodes_to_roles[node])]

    @property
    def learned_components(self):
        learned_projections = [proj for proj in self.projections
                               if hasattr(proj, 'has_learning_projection') and proj.has_learning_projection]
        related_processing_mechanisms = [mech for mech in self.nodes
                                         if (isinstance(mech, Mechanism)
                                             and (any([mech in learned_projections for mech in mech.afferents])
                                                  or any([mech in learned_projections for mech in mech.efferents])))]
        return related_processing_mechanisms + learned_projections

    @property
    def afferents(self):
        return ContentAddressableList(component_type=Projection,
                                      list=[proj for proj in self.input_CIM.afferents])

    @property
    def efferents(self):
        return ContentAddressableList(component_type=Projection,
                                      list=[proj for proj in self.output_CIM.efferents])

    @property
    def _all_nodes(self):
        for n in self.nodes:
            yield n
        yield self.input_CIM
        yield self.output_CIM
        yield self.parameter_CIM
        if self.controller:
            yield self.controller
