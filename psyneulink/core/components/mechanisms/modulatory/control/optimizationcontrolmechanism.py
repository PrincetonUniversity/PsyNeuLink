# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  OptimizationControlMechanism *************************************************

# FIX: REWORK WITH REFERENCES TO `outcome <OptimizationControlMechanism.outcome>`
#      INTRODUCE SIMULATION INTO DISCUSSION OF COMPOSITION-BASED

"""

Contents
--------

  * `OptimizationControlMechanism_Overview`
      - `Expected Value of Control <OptimizationControlMechanism_EVC>`
      - `Agent Representation and Types of Optimization <OptimizationControlMechanism_Agent_Representation_Types>`
          - `"Model-Free" Optimization <OptimizationControlMechanism_Model_Free>`
          - `Model-Based Optimization <OptimizationControlMechanism_Model_Based>`
  * `OptimizationControlMechanism_Creation`
      - `Agent Rep <OptimizationControlMechanism_Agent_Rep_Arg>`
      - `State Features <OptimizationControlMechanism_State_Features_Arg>`
          - `agent_rep Composition <OptimizationControlMechanism_Agent_Rep_Composition>`
          - `agent_rep CompositionFunctionApproximator <OptimizationControlMechanism_Agent_Rep_CFA>`
      - `State Feature Functions <OptimizationControlMechanism_State_Feature_Function_Arg>`
      - `Outcome  <OptimizationControlMechanism_Outcome_Args>`
  * `OptimizationControlMechanism_Structure`
      - `Agent Representation <OptimizationControlMechanism_Agent_Rep>`
          - `State <OptimizationControlMechanism_State>`
      - `Input <OptimizationControlMechanism_Input>`
          - `state_input_ports <OptimizationControlMechanism_State_Features>`
          - `outcome_input_ports <OptimizationControlMechanism_Outcome>`
              - `objective_mechanism <OptimizationControlMechanism_ObjectiveMechanism>`
              - `monitor_for_control <OptimizationControlMechanism_Monitor_for_Control>`
      - `Function <OptimizationControlMechanism_Function>`
          - `OptimizationControlMechanism_Custom_Function`
          - `OptimizationControlMechanism_Search_Functions`
          - `OptimizationControlMechanism_Default_Function`
      - `Output <OptimizationControlMechanism_Output>`
          - `Randomization ControlSignal <OptimizationControlMechanism_Randomization_Control_Signal>`
  * `OptimizationControlMechanism_Execution`
      - `OptimizationControlMechanism_Execution_Timing`
      - `OptimizationControlMechanism_Optimization_Procedure`
      - `OptimizationControlMechanism_Estimation_Randomization`
  * `OptimizationControlMechanism_Class_Reference`


.. _OptimizationControlMechanism_Overview:

Overview
--------

An OptimizationControlMechanism is a `ControlMechanism <ControlMechanism>` that uses an `OptimizationFunction` to
optimize the performance of the `Composition` for which it is a `controller <Composition_Controller>`.  It does so
by using the `OptimizationFunction` (assigned as its `function <OptimizationControlMechanism.function>`) to execute
its `agent_rep <OptimizationControlMechanism.agent_rep>` -- a representation of the Composition to be optimized --
under different `control_allocations <ControlMechanism.control_allocation>`, and selecting the one that optimizes
its `net_outcome <ControlMechanism.net_outcome>`.  A OptimizationControlMechanism can be configured to implement
forms of optimization, ranging from fully `model-based optimization <OptimizationControlMechanism_Model_Based>`
that uses the Composition itself as the  `agent_rep <OptimizationControlMechanism.agent_rep>` to simulate the
outcome for a given `state <OptimizationControlMechanism_State>` (i.e., a combination of the current input and a
particular `control_allocation <ControlMechanism.control_allocation>`), to fully `model-free optimization
<OptimizationControlMechanism_Model_Free>` by using a `CompositionFunctionApproximator` as the `agent_rep
<OptimizationControlMechanism.agent_rep>` that learns to  predict the outcomes for a state. Intermediate forms of
optimization can also be implemented, that use simpler Compositions to approximate the dynamics of the full Composition.
The outcome of executing the `agent_rep <OptimizationControlMechanism.agent_rep>` is used to compute a `net_outcome
<ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>`, that takes into account
the `costs <ControlMechanism_Costs_NetOutcome>` associated with the `control_allocation, and is used to determine
the optimal `control_allocations <ControlMechanism.control_allocation>`.

.. _OptimizationControlMechanism_EVC:

**Expected Value of Control**

The `net_outcome <ControlMechanism.net_outcome>` of an OptimizationControlMechanism's `agent_rep
<OptimizationControlMechanism.agent_rep>` is computed -- for a given `state <OptimizationControlMechanism_State>`
(i.e., set of `state_feature_values <OptimizationControlMechanism.state_feature_values>` and a `control_allocation
<ControlMechanism.control_allocation>` -- as the difference between the `outcome <ControlMechanism.outcome>` computed
by its `objective_mechanism <ControlMechanism.objective_mechanism>` and the aggregated `costs <ControlMechanism.costs>`
of its `control_signals <OptimizationControlMechanism.control_signals>` computed by its `combine_costs
<ControlMechanism.combine_costs>` function.  If the `outcome <ControlMechanism.outcome>` computed by the
`objective_mechanism <ControlMechanism.objective_mechanism>` is configured to measure the value of processing (e.g.,
reward received, time taken to respond, or a combination of these, etc.), and the `OptimizationFunction` assigned as
the OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` is configured find the
`control_allocation <ControlMechanism.control_allocation>` that maximizes its `net_outcome
<ControlMechanism.net_outcome>` (that is, the `outcome <ControlMechanism.outcome>` discounted by the
result of the `combine_costs <ControlMechanism.combine_costs>` function, then the OptimizationControlMechanism is
said to be maximizing the `Expected Value of Control (EVC) <https://www.ncbi.nlm.nih.gov/pubmed/23889930>`_.  That
is, it implements a cost-benefit analysis that weighs the `costs <ControlMechanism.costs>` of the ControlSignal
`values <ControlSignal.value>` associated with a `control_allocation <ControlMechanism.control_allocation>` against
the `outcome <ControlMechanism.outcome>` expected to result from it.  The costs are computed based on the
`cost_options <ControlSignal.cost_options>` specified for each of the OptimizationControlMechanism's `control_signals
<OptimizationControlMechanism.control_signals>` and its `combine_costs <ControlMechanism.combine_costs>` function.
The EVC is determined by its `compute_net_outcome <ControlMechanism.compute_net_outcome>` function (assigned to its
`net_outcome <ControlMechanism.net_outcome>` attribute), which is computed for a given `state
<OptimizationControlMechanism_State>` by the OptimizationControlMechanism's `evaluate_agent_rep
<OptimizationControlMechanism.evaluate_agent_rep>` method. In these respects, optimization of a Composition's
performance by its OptimizationControlMechanism -- as indexed by its `net_outcome <ControlMechanism.net_outcome>`
attribute -- implement a form of `Bounded Rationality <https://psycnet.apa.org/buy/2009-18254-003>`_,
also referred to as `Resource Rationality <https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/resourcerational-analysis-understanding-human-cognition-as-the-optimal-use-of-limited-computational-resources/586866D9AD1D1EA7A1EECE217D392F4A>`_,
in which the constraints imposed by the "bounds" or resources are reflected in the `costs` of the ControlSignals
(also see `Computational Rationality <https://onlinelibrary.wiley.com/doi/full/10.1111/tops.12086>`_ and `Toward a
Rational and Mechanistic Account of Mental Effort
<https://www.annualreviews.org/doi/abs/10.1146/annurev-neuro-072116-031526?casa_token=O2pFelbmqvsAAAAA:YKjdIbygP5cj_O7vAj4KjIvfHehHSh82xm44I5VS6TdTtTELtTypcBeET4BGdAy0U33BnDXBasfqcQ>`_).

COMMENT:
The table `below <OptimizationControlMechanism_Examples>` lists different
parameterizations of OptimizationControlMechanism that implement various models of EVC Optimization.
COMMENT

.. _OptimizationControlMechanism_Agent_Representation_Types:

**Agent Representation and Types of Optimization**

Much of the functionality described above is supported by a `ControlMechanism` (the parent class of an
OptimizationControlMechanism). The defining  characteristic of an OptimizationControlMechanism is its `agent
representation <OptimizationControlMechanism_Agent_Rep>`, that is used to determine the `net_outcome
<ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>`, and find the
`control_allocation <ControlMechanism.control_allocation>` that optimizes this.  The `agent_rep
<OptimizationControlMechanism.agent_rep>` can be the `Composition` to which the OptimizationControlMechanism
belongs (and controls), another (presumably simpler) one, or a `CompositionFunctionApproximator`) that is used to
estimate the `net_outcome <ControlMechanism.net_outcome>` Composition of which the OptimizationControlMechanism is
the `controller <Composition.controller>`.  These different types of `agent representation
<OptimizationControlMechanism_Agent_Rep>` correspond closely to the distinction between *model-based* and
*model-free* optimization in the `machine learning
<https://www.google.com/books/edition/Reinforcement_Learning_second_edition/uWV0DwAAQBAJ?hl=en&gbpv=1&dq=Sutton,+R.+S.,+%26+Barto,+A.+G.+(2018).+Reinforcement+learning:+An+introduction.+MIT+press.&pg=PR7&printsec=frontcover>`_
and `cognitive neuroscience <https://www.nature.com/articles/nn1560>`_ literatures, as described below.

.. figure:: _static/Optimization_fig.svg
   :scale: 50%
   :alt: OptimizationControlMechanism

   **Functional Anatomy of an OptimizationControlMechanism.** *Panel A:* Examples of use in fully model-based
   and model-free optimization.  Note that in the example of `model-based optimization
   <OptimizationControlMechanism_Model_Based>` (left), the OptimizationControlMechanism uses the entire
   `Composition` that it controls as its `agent_rep <OptimizationControlMechanism.agent_rep>`, whereas in
   the example of `model-free optimization <OptimizationControlMechanism_Model_Free>` (right) the
   the `agent_rep <OptimizationControlMechanism.agent_rep>` is a `CompositionFunctionApproximator`. The `agent_rep
   <OptimizationControlMechanism.agent_rep>` can also be another (presumably simpler) Composition that can be used
   to implement forms of optimization intermediate between fully model-based and model-free. *Panel B:* Flow of
   execution during optimization.  In both panels, faded items show process of adaptation when using a
   `CompositionFunctionApproximator` as the `agent_rep <OptimizationControlMechanism.agent_rep>`.
|
.. _OptimizationControlMechanism_Model_Based:

*Model-Based Optimization*

The fullest form of this is implemented by assigning as the `agent_rep  <OptimizationControlMechanism.agent_rep>`
the Composition for which the OptimizationControlMechanism is the `controller <Composition.controller>`).
On each `TRIAL <TimeScale.TRIAL>`, that Composition *itself* is provided with either the most recent inputs
to the Composition, or ones predicted for the upcoming trial (as determined by the `state_feature_values
<OptimizationControlMechanism.state_feature_values>` of the OptimizationControlMechanism), and then used to simulate
processing on that trial in order to find the `control_allocation <ControlMechanism.control_allocation>` that yields
the best `net_outcome <ControlMechanism.net_outcome>` for that trial.  A different Composition can also be assigned as
the `agent_rep  <OptimizationControlMechanism.agent_rep>`, that approximates in simpler form the dynamics of processing
in the Composition for which the OptimizationControlMechanism is the `controller <Composition.controller>`,
implementing a more restricted form of model-based optimization.

.. _OptimizationControlMechanism_Model_Free:

*"Model-Free" Optimization*

    .. note::
       The term *model-free* is placed in apology quotes to reflect the fact that, while this term is
       used widely (e.g., in machine learning and cognitive science) to distinguish it from *model-based* forms of
       processing, model-free processing nevertheless relies on *some* form of model -- albeit usually a much simpler
       one -- for learning, planning and decision making.  In the context of a OptimizationControlMechanism, this is
       addressed by use of the term "agent_rep", and how it is implemented, as described below.

This clearest form of this uses a `CompositionFunctionApproximator`, that learns to predict the `net_outcome
`net_outcome <ControlMechanism.net_outcome>` for a given state (e.g., using reinforcement learning or other forms
of function approximation, , such as a `RegressionCFA`).  In each `TRIAL <TimeScale.TRIAL>` the  `agent_rep
<OptimizationControlMechanism.agent_rep>` is used to search over `control_allocation
<ControlMechanism.control_allocation>`\\s, to find the one that yields the best predicted `net_outcome
<ControlMechanism.net_outcome>` of processing on the upcoming trial, based on the current or (expected)
`state_feature_values <OptimizationControlMechanism.state_feature_values>` for that trial.  The `agent_rep
<OptimizationControlMechanism.agent_rep>` is also given the chance to adapt in order to improve its prediction
of its `net_outcome <ControlMechanism.net_outcome>` based on the `state <OptimizationControlMechanism_State>`,
and `net_outcome <ControlMechanism.net_outcome>` of the prior trial.  A Composition can also be used to generate
such predictions permitting, as noted above, forms of optimization intermediate between the extreme examples of
model-based and model-free.

.. _OptimizationControlMechanism_Creation:

Creating an OptimizationControlMechanism
----------------------------------------

The constructor has the same arguments as a `ControlMechanism <ControlMechanism>`, with the following
exceptions/additions, which are specific to the OptimizationControlMechanism:

.. _OptimizationControlMechanism_Agent_Rep_Arg:

* **agent_rep** -- specifies the `Composition` used by the OptimizationControlMechanism's `evaluate_agent_rep
  <OptimizationControlMechanism.evaluate_agent_rep>` method to calculate the predicted `net_outcome
  <ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>` (see `below
  <OptimizationControlMechanism_Agent_Rep>` for additional details). If it is not specified, then the
  `Composition` to which the OptimizationControlMechanism is assigned becomes its `agent_rep
  <OptimizationControlMechanism.agent_rep>`, and the OptimizationControlMechanism is assigned as that Composition's
  `controller <Composition.controller>`, implementing fully `model-based <OptimizationControlMechanism_Model_Based>`
  optimization.  If that Composition already has a `controller <Composition.controller>` specified,
  the OptimizationControlMechanism is disabled. If another Composition is specified, it must conform to the
  specifications for an `agent_rep <OptimizationControlMechanism.agent_rep>` as described `below
  <OptimizationControlMechanism_Agent_Rep>`.  The  `agent_rep <OptimizationControlMechanism.agent_rep>` can also be
  a `CompositionFunctionApproximator` for `model-free <OptimizationControlMechanism_Model_Free>` forms of
  optimization.  The type of Component assigned as the `agent_rep <OptimizationControlMechanism.agent_rep>` is
  identified in the OptimizationControlMechanism's `agent_rep_type <OptimizationControlMechanism.agent_rep_type>`
  attribute.

.. _OptimizationControlMechanism_State_Features_Arg:

* **state_features** -- specifies the values provided by the OptimizationControlMechanism as the input to its
  `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method, together
  with a selected `control_allocation <ControlMechanism.control_allocation>`, when that is called to estimate
  or predict the Composition's `net_outcome <ControlMechanism.net_outcome>`. These are used to construct the
  `state_input_ports <OptimizationControlMechanism.state_input_ports>` for the OptimizationControlMechanism,
  the `values <InputPort.value>` of which are assigned as the `state_feature_values
  <OptimizationControlMechanism.state_feature_values>` and provided to the **predicted_inputs** argument of the
  `evaluate <Composition.evaluate>` method if `agent_rep <OptimizationControlMechanism.agent_rep>` is a `Composition`,
  or the **feature_values** argument if it is a `CompositionFunctionApproximator`.  Accordingly, the specification
  requirements for **state_features** depend on whether the `agent_rep<OptimizationControlMechanism.agent_rep>` is a
  `Composition` or a `CompositionFunctionApproximator`, as described in each of the two sections below.

  |

  .. _OptimizationControlMechanism_Agent_Rep_Composition:

  **state_features** *for an agent_rep that is a* **Composition**

  |

  .. _OptimizationControlMechanism_State_Features_Automatic_Assignment:

  *Automatic assignment.* The **state_features** specify the inputs to the Composition assigned as the `agent_rep
  <OptimizationControlMechanism.agent_rep>` when it is executed by the OptimizationControlMechanism to
  `evaluate <OptimizationControlMechanism_Evaluation>` its performance.  The default is for the evaluation to use the
  same values received by the `agent_rep <OptimizationControlMechanism.agent_rep>` as its `external inputs
  <Composition_Execution_Inputs>` during its last `TRIAL <TimeScale.TRIAL>` of execution. Accordingly, if
  **state_features** is not specified, a set of `state_input_ports <OptimizationControlMechanism.state_input_ports>`
  is constructed automatically that `shadow the input <InputPort_Shadow_Inputs>` to every `InputPort` of every
  `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>`
  Composition.

  |

  .. _OptimizationControlMechanism_State_Features_Explicit_Specification:

  *Explicit specification.* The **state_features** argument can also be specified explicitly, using the formats
  described below.  This is useful if values other than the `external inputs <Composition_Execution_Inputs>` to
  the `agent_rep <OptimizationControlMechanism.agent_rep>` Composition are to be used to evaluate it; to restrict
  evaluation to a subset of inputs (while others are held constant); and/or to assign specific functions to one or
  more `state_input_ports <OptimizationControlMechanism.state_input_ports>` that allow them to process the inputs
  (e.g., modulate and/or integrate them) before they are assigned as to `state_feature_values
  <OptimizationControlMechanism.state_feature_values>` (see `below
  <OptimizationControlMechanism_State_Feature_Function_Arg>`). Note that assigning any **state_features** explicitly
  overrides their automatic assignment, so that *all* required ones must be specified explicitly, as described below.
  Any that are *not* specified will be assigned the value of their `default_variable <Mechanism_Base.default_variable>`
  when the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method is
  executed.

  .. _OptimizationControlMechanism_State_Features_Shapes:

      .. note::
         If **state_features** are specified explicitly, the shapes of the `value <Component.value>`\\s of the
         specified Components must match those required as `external inputs <Composition_Execution_Inputs>` to
         the corresponding `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the `agent_rep
         <OptimizationControlMechanism.agent_rep>`. An example of the input format required by the `INPUT
         <NodeRole.INPUT>` `Nodes <Composition_Nodes>` can be generated using the `agent_rep
         <OptimizationControlMechanism.agent_rep>`\\'s `get_input_format <Composition.get_input_format>` method.
         A failure to properly meet these requirements generates an error.

      COMMENT:
      .. _OptimizationControlMechanism_Selective_Input:

      .. hint::
         For cases in which only a subset of the inputs to the Composition are relevant to its optimization (e.g.,
         the others should be held constant), it is still the case that all must be specified as **state_features**
         (see note above).  This can be handled several ways.  One is by specifying (as required) **state_features**
         for all of the inputs, and assigning *state_feature_function** (see `below
         <OptimizationControlMechanism_State_feature_function_Arg>`) such that those assigned to the desired
         inputs pass their values unmodified, while those for the inputs that are to be ignored return a constant value.
         Another approach, for cases in which the desired inputs pertain to a subset of Components in the Composition
         solely responsible for determining its `net_outcome <ControlMechanism.net_outcome>`, is to assign those
         Components to a `nested Composition <Composition_Nested>` and assign that Composition as the `agent_rep
         <OptimizationControlMechanism.agent_rep>`.  A third, more sophisticated approach, is to assign
         ControlSignals to the InputPorts for the irrelevant features, and specify them to suppress their values.
      COMMENT

  .. _OptimizationControlMechanism_State_Features_Shadow_Inputs:

  The **state_features** argument can be specified using any of the following formats:

  .. _Optimization_Control_Mechanism_State_Feature_Input_Dict:

  * *Inputs dictionary* -- a dictionary that conforms to the format used to `specify external inputs
    <Composition_Input_Dictionary>` to the `agent_rep <OptimizationControlMechanism.agent_rep>`, in which entries
    consist of a key specifying an `INPUT <NodeRole.INPUT>` Node of `agent_rep
    <OptimizationControlMechanism.agent_rep>`, and its value is the source of the input, that can be any of the forms
    of individual input specifications listed `below <Optimization_Control_Mechanism_State_Feature_Individual_Inputs>`.
    This is the most straightforward and reliable way to specify **state_features**.  The full format required for
    inputs to `agent_rep <OptimizationControlMechanism.agent_rep>` can be seen using its `get_input_format
    <Composition.get_input_format>` method.  If any `INPUT <NodeRole.INPUT>` Nodes are not specified or assigned None
    as their value, their `default variable <Mechanism_Base.default_variable>` is used for their input when the
    `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method is executed,
    irrespective of the input to the `agent_rep <OptimizationControlMechanism.agent_rep>` during the last `TRIAL
    <TimeScale.TRIAL>`.  If a nested Composition is specified (that is an `INPUT <NodeRole.INPUT>` Node of `agent_rep
    <OptimizationControlMechanism.agent_rep>`), the value assigned to it is used for *all* of the `INPUT
    <NodeRole.INPUT>` Nodes for the nested Composition and any nested within it, at all levels of nesting.  If one or
    more `INPUT <NodeRole.INPUT>` Nodes of a nested Composition (that are INPUT Nodes at all levels of nesting) are
    specified, then any unspecified INPUT Nodes of the corresponding Compositions are assigned None as their values.

  .. _Optimization_Control_Mechanism_State_Feature_List_Inputs:

  * *List* -- a list of individual input source specifications, that can be any of the forms of individual input
    specifications listed `below <Optimization_Control_Mechanism_State_Feature_Individual_Inputs>`. The items
    must be listed in the order that `INPUT <NodeRole.INPUT>` Nodes are listed in the `agent_rep
    <OptimizationControlMechanism.agent_rep>`\\'s `nodes <Composition.nodes>` attribute (and returned by its
    `get_nodes_by_role(NodeRole.INPUT) <Composition.get_nodes_by_role>` method).  If the list is incomplete,
    the remaining INPUT Nodes are assigned their `default variable <Component.defaults>` as input when the `agent_rep
    <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method is called; ``None`` can
    be used as an entry to "skip" items in the list (i.e., specify that they receive their `default variable
    <Component.defaults>` as input). Items can be specified in the list that have not yet been added to the
    OptimizationControlMechanism's Composition or its `agent_rep <OptimizationControlMechanism.agent_rep>`, that are
    either sources of input to `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s  `INPUT <NodeRole.INPUT>`
    `Nodes <Composition_Nodes>`, or those Nodes themselves.  However, these must be added before Composition is
    executed, and must appear in the list in the same position that the `INPUT Nodes to which they pertain are
    list in the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `nodes <Composition.nodes>` attribute.

  .. _Optimization_Control_Mechanism_State_Feature_Set_Inputs:

  * *Set* -- a set of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the `agent_rep
    <OptimizationControlMechanism.agent_rep>` to receive the same inputs during evaluation as when
    the `agent_rep  <OptimizationControlMechanism.agent_rep>` is fully executed; the `state_input_ports
    <OptimizationControlMechanism.state_input_ports>` constructed for these state_features are assigned
    Projections that `shadow <InputPort_Shadow_Inputs>` the specified `INPUT <NodeRole.INPUT>` `Node
    <Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>`. The order of their
    specification does not matter;  however, any of the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s
    `INPUT <NodeRole.INPUT>` Nodes that are *not* included in the set will be assigned their `default variable
    <Component.defaults>` when the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate
    <Composition.evaluate>` method is called.

  .. _Optimization_Control_Mechanism_State_Feature_Individual_Inputs:

  * *Individual inputs* -- any of the forms below can be used singly, or in a dict or list as described
    `above <Optimization_Control_Mechanism_State_Feature_Input_Dict>`, to configure a `state_input_port
    <OptimizationControlMechanism.state_input_ports>`, the value of which is assigned as the corresponding element
    of `state_feature_values <OptimizationControlMechanism.state_feature_values>` provided as input to the `INPUT
    <NodeRole.INPUT>` `Node <Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>` when it
    is `evaluated <Composition.evaluate>` method is called.

      .. note::
         If only a single input specification is provided to **state_features**, it is treated as a list with a single
         item (see `above <Optimization_Control_Mechanism_State_Feature_List_Inputs>`), and assigned as the input to the
         first `INPUT <NodeRole.INPUT>` Node of `agent_rep <OptimizationControlMechanism.agent_rep>`; if the latter has
         any additional `INPUT <NodeRole.INPUT>` Nodes, they are assigned their `default variable <Component.defaults>`
         as inputs when the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>`
         method is executed.

    .. _Optimization_Control_Mechanism_Numeric_State_Feature:

    * *numeric value* -- create an `InputPort` with the specified value as its `default variable <Component.defaults>`
      and no `afferent Projections <Mechanism_Base.afferents>`;  as a result, the specified value is assigned as the
      input to the corresponding `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of the `agent_rep
      <OptimizationControlMechanism.agent_rep>` each time it is `evaluated <Composition.evaluate>`.

    .. _Optimization_Control_Mechanism_Tuple_State_Feature:

    * *2-item tuple* -- the first item must be a `Port` or `Mechanism` specification, as described below;
      the second item must be a `Function`, that is assigned as the `function <InputPort.function>` of the
      corresponding `state_input_port <OptimizationControlMechanism.state_input_ports>`;
      this takes precedence over any other state_feature_function specifications (e.g., in an `InputPort
      specification dictionary <InputPort_Specification_Dictionary>` or the **state_feature_function** argument
      of the OptimizationControlMechanism's constructor; see `state_feature_function
      <OptimizationControlMechanism_State_Feature_Function_Arg>` for additional details).

    .. _Optimization_Control_Mechanism_Input_Port_Dict_State_Feature:

    * *specification dictionary* -- an `InputPort specification dictionary <InputPort_Specification_Dictionary>`
      can be used to configure the corresponding `state_input_port <OptimizationControlMechanism.state_input_ports>`,
      if `Parameters <Parameter>` other than its `function <InputPort.function>` need to be specified (e.g., its `name
      <InputPort.name>` or more than a single `afferent Projection <Mechanism_Base.afferents>`).

    .. _Optimization_Control_Mechanism_Input_Port_State_Feature:

    * *InputPort specification* -- creates an `InputPort` that `shadows <InputPort_Shadow_Inputs>` the input to
      the specified InputPort, the `value <InputPort.value>` of which is used as the corresponding value of the
      OptimizationControlMechanism's `state_feature_values <OptimizationControlMechanism.state_feature_values>`.

      .. note::
         Only the `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of a `nested Composition <Composition_Nested>`
         can shadowed.  Therefore, if the Composition that an OptimizationControlMechanism controls contains any
         nested Compositions, only its `INPUT <NodeRole.INPUT>` Nodes can be specified for shadowing in the
         **state_features** argument of the OptimizationControlMechanism's constructor.

      .. hint::
         Shadowing the input to a Node of a `nested Composition <Composition_Nested>` that is not an `INTERNAL
         <NodeRole.INTERNAL>` Node of that Composition can be accomplished one or of two ways, by: a) assigning it
         `INPUT <NodeRole.INPUT>` as a `required NodeRole <Composition_Node_Role_Assignment>` where it is added to
         the nested Composition; and/or b) adding an additional Node to that Composition that shadows the desired one
         (this is allowed *within* the *same* Composition), and is assigned as an `OUTPUT <NodeRole.OUTPUT>` Node of
         that Composition, the `OutputPort` of which which can then be specified in the **state_features** argument of
         the OptimizationControlMechanism's constructor (see below).

      .. technical_note::
        The InputPorts specified as state_features are marked as `internal_only <InputPort.internal_only>` = `True`.

    .. _Optimization_Control_Mechanism_Output_Port_State_Feature:

    * *OutputPort specification* -- this creates an `InputPort` that receives a `MappingProjection` from the
      specified `OutputPort`;  it can be any form of `OutputPort specification <OutputPort_Specification>`
      for any `OutputPort` of another `Mechanism <Mechanism>` in the Composition. The `value <OutputPort.value>`
      of the specified OutputPort is used as the corresponding value of the OptimizationControlMechanism's
      `state_feature_values <OptimizationControlMechanism.state_feature_values>`.

    .. _Optimization_Control_Mechanism_Mechanism_State_Feature:

    * *Mechanism* -- if the `agent_rep <OptimizationControlMechanism.agent_rep>` is a Composition, the Mechanism's
      `primary InputPort <InputPort_Primary>` is shadowed;  that is, it is assumed that its' input should be used
      as the corresponding value of the OptimizationControlMechanism's `state_feature_values
      <OptimizationControlMechanism.state_feature_values>`. This has the same result as explicitly specifying the
      Mechanism's  input_port, as described `above <Optimization_Control_Mechanism_Input_Port_State_Feature>`.  If
      the Mechanism is in a `nested Composition <Composition_Nested>`, it must be an `INPUT <NodeRole.INPUT>` `Node
      <Composition_Nodes>` of that Composition (see note above).  If its OutputPort needs to be used, it must be
      specified explicitly (as described `above <Optimization_Control_Mechanism_Output_Port_State_Feature>`).

  |

  .. _OptimizationControlMechanism_Agent_Rep_CFA:

  **state_features** *for an agent_rep that is a* **CompositionFunctionApproximator**

  |

  The **state_features** specify the **feature_values** argument to the `CompositionFunctionApproximator`\\'s
  `evaluate <CompositionFunctionApproximator.evaluate>` method. These cannot be determined automatically and so
  they *must be specified explicitly*, in a list, with the correct number of items in the same order and with
  the same shapes they are expected have in the array passed to the **feature_values** argument of the
  `evaluate <CompositionFunctionApproximator.evaluate>` method (see warning below).

      .. warning::
         The **state_features** for an `agent_rep <OptimizationControlMechanism.agent_rep>` that is a
         `CompositionFunctionApproximator` cannot be created automatically nor can they be validated;
         thus specifying the wrong number or invalid **state_features**, or specifying them in an incorrect
         order may produce errors that are unexpected or difficult to interpret.

  The list of specifications can contain any of the forms of specification used for an `agent_rep
  <OptimizationControlMechanism.agent_rep>` that is a Composition as described `above
  <Optimization_Control_Mechanism_State_Feature_Input_Dict>`, with the following exception: if a
  `Mechanism` is specified, its `primary OutputPort <OutputPort_Primary>` is used (rather than
  shadowing its primary InputPort), since that is more typical usage, and there are no assumptions
  made about the state features of a `CompositionFunctionApproximator` (as there are about a Composition
  as `agent_rep <OptimizationControlMechanism.agent_rep>`); if the input to the Mechanism *is* to be
  `shadowed <InputPort_Shadow_Inputs>`, then its InputPort must be specified explicitly (as described
  `above <Optimization_Control_Mechanism_Input_Port_State_Feature>`).

  COMMENT:
      FIX: CONFIRM THAT THE FOLLOWING WORKS
  State features can also be added to an existing OptimizationControlMechanism using its `add_state_features` method.
  COMMENT

|

.. _OptimizationControlMechanism_State_Feature_Function_Arg:

* **state_feature_function** -- specifies a `function <InputPort.function>` to be used as the default
  function for `state_input_ports <OptimizationControlMechanism.state_input_ports>`. This is assigned as
  the `function <InputPort.function>` to any state_input_ports for which no other `Function` is specified;
  i.e., in an InputPort specification dictionary <InputPort_Specification_Dictionary>` or `2-item tuple
  <Optimization_Control_Mechanism_Tuple_State_Feature>` in the **state_features** argument (see `state_features
  <OptimizationControlMechanism_State_Features_Arg>`).  If either of the latter is specified, they override
  the specification in **state_feature_function**.  If it is *not* specified, then `LinearCombination`
  (the standard default `Function` for an `InputPort`) is assigned to any `state_input_ports
  <OptimizationControlMechanism.state_input_ports>` that are not otherwise assigned a `Function`.
  Specifying functions for `state_input_ports <OptimizationControlMechanism.state_input_ports>` can be useful,
  for example to provide an average or integrated value of prior inputs to the `agent_rep
  <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method during the optimization
  process, or to use a generative model of the environment to provide those inputs.

    .. note::
       The value returned by a function assigned to the **state_feature_function** argument must preserve the
       shape of its input, and must also accommodate the shape of the inputs to all of the `state_input_ports
       <OptimizationControlMechanism.state_input_ports>` to which it is assigned (see `note
       <OptimizationControlMechanism_State_Features_Shapes>` above).

.. _OptimizationControlMechanism_Outcome_Args:

* **Outcome arguments** -- these specify the Components, the values of which are assigned to the `outcome
  <ControlMechanism.outcome>` attribute, and used to compute the `net_outcome <ControlMechanism.net_outcome>` for a
  given `control_allocation <ControlMechanism.control_allocation>` (see `OptimizationControlMechanism_Execution`).
  As with a ControlMechanism, these can be sepcified directly in the **monitor_for_control** argument, or through the
  use of `ObjectiveMechanism` specified in the  **objecctive_mechanism** argument (see
  ControlMechanism_Monitor_for_Control for additional details).  However, an OptimizationControlMechanism places some
  restrictions on the specification of these arguments that, as with specification of `state_features
  <OptimizationControlMechanism_State_Features_Arg>`, depend on the nature of the `agent_rep
  <OptimizationControlMechanism.agent_rep>`, as described below.

  * *agent_rep is a Composition* -- the items specified to be monitored for control must belong to the `agent_rep
    <OptimizationControlMechanism.agent_rep>`, since those are the only ones that will be executed when the
    `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` is called; an error will be generated
    identifying any Components that do not belong to the `agent_rep <OptimizationControlMechanism.agent_rep>`.

  * *agent_rep is a CompositionFunctionApproximator* -- the items specified to be monitored for control can be any
    within the Composition for which the OptimizationControlMechanism is the `controller <Composition_Controller>`;
    this is because their values during the last execution of the Composition are used to determine the `net_outcome
    <ControlMechanism.net_outcome>` that the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s
    `adapt <CompositionFunctionApproximator.adapt>` method -- if it has one -- seeks to predict.  Accordingly,
    the values of the items specified to be monitored control must match, in shape and order, the
    **net_outcome** of that `adapt <CompositionFunctionApproximator.adapt>` method.

* **Optimization arguments** -- these specify parameters that determine how the OptimizationControlMechanism's
  `function <OptimizationControlMechanism.function>` searches for and determines the optimal `control_allocation
  <ControlMechanism.control_allocation>` (see `OptimizationControlMechanism_Execution`); this includes specification
  of the `num_estimates <OptimizationControlMechanism.num_estimates>` and `num_trials_per_estimate
  <OptimizationControlMechanism.num_trials_per_estimate>` parameters, as well as the `random_variables
  <OptimizationControlMechanism.random_variables>`, `initial_seed <OptimizationControlMechanism.initial_seed>` and
  `same_seed_for_all_allocations <OptimizationControlMechanism.same_seed_for_all_allocations>` Parameters, which
  determine how the `net_outcome <ControlMechanism.net_outcome>` is estimated for a given `control_allocation
  <ControlMechanism.control_allocation>` (see `OptimizationControlMechanism_Estimation_Randomization` for additional
  details).

.. _OptimizationControlMechanism_Structure:

Structure
---------

An OptimizationControlMechanism conforms to the structure of a `ControlMechanism`, with the following exceptions
and additions.

.. _OptimizationControlMechanism_Agent_Rep:

*Agent Representation*
^^^^^^^^^^^^^^^^^^^^^^

The defining feature of an OptimizationControlMechanism is its agent representation, specified in the **agent_rep**
argument of its constructor, and assigned to its `agent_rep <OptimizationControlMechanism.agent_rep>` attribute.  This
designates a representation of the `Composition` (or parts of it) that the OptimizationControlMechanism uses to
evaluate sample `control_allocations <ControlMechanism.control_allocation>` in order to find one that optimizes the
the `net_outcome <ControlMechanism.net_outcome>` of the Composition when it is fully executed. The `agent_rep
<OptimizationControlMechanism.agent_rep>` can be the Composition itself for which the OptimizationControlMechanism is
the `controller <Composition_Controller>` (fully `model-based optimization <OptimizationControlMechanism_Model_Based>`,
or another one `model-free optimization <OptimizationControlMechanism_Model_Free>`), that is usually a simpler
Composition or a `CompositionFunctionApproximator`  used to estimate the `net_outcome <ControlMechanism.net_outcome>`
for the full Composition (see `above <OptimizationControlMechanism_Agent_Representation_Types>`).  The `evaluate
<Composition.evaluate>` method of the `agent_rep <OptimizationControlMechanism.agent_rep>` is assigned as the
`evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method of the OptimizationControlMechanism.
If the `agent_rep <OptimizationControlMechanism.agent_rep>` is not the Composition for which the
OptimizationControlMechanism is the controller, then it must meet the following requirements:

* Its `evaluate <Composition.evaluate>` method must accept as its first four positional arguments:

  - values that correspond in shape to  the `state_feature_values
    <OptimizationControlMechanism.state_feature_values>` (inputs for estimate);
  - `control_allocation <ControlMechanism.control_allocation>` (the set of parameters for which estimates
    of `net_outcome <ControlMechanism.net_outcome>` are made);
  - `num_trials_per_estimate <OptimizationControlMechanism.num_trials_per_estimate>` (number of trials executed by
    agent_rep for each estimate).
..
* If it has an `adapt <CompositionFunctionApproximator.adapt>` method, that must accept as its first three
  arguments, in order:

  - values that correspond to the shape of the  `state_feature_values
    <OptimizationControlMechanism.state_feature_values>` (inputs that led to the net_come);
  - `control_allocation <ControlMechanism.control_allocation>` (set of parameters that led to the net_outcome);
  - `net_outcome <ControlMechanism.net_outcome>` (the net_outcome that resulted from the `state_feature_values
    <OptimizationControlMechanism.state_feature_values>` and `control_allocation
    <ControlMechanism.control_allocation>`) that must match the shape of `outcome <ControlMechanism.outcome>`.
  COMMENT:
  - `num_estimates <OptimizationControlMechanism.num_trials_per_estimate>` (number of estimates of `net_outcome
    <ControlMechanism.net_outcome>` made for each `control_allocation <ControlMechanism.control_allocation>`).
  COMMENT

 .. _OptimizationControlMechanism_State:

*State*
~~~~~~~

The current state of the OptimizationControlMechanism -- or, more properly, of its `agent_rep
<OptimizationControlMechanism.agent_rep>` -- is determined by the OptimizationControlMechanism's current
`state_feature_values <OptimizationControlMechanism.state_feature_values>` (see `below
<OptimizationControlMechanism_State_Features>`) and `control_allocation <ControlMechanism.control_allocation>`.
These are used by the `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method,
the results of which are combined with the `costs <ControlMechanism_Costs_NetOutcome>` associated with the
`control_allocation <ControlMechanism.control_allocation>`, to evaluate the `net_outcome
<ControlMechanism.net_outcome>` for that state. The current state is listed in the OptimizationControlMechanism's
`state <OptimizationControlMechanism.state>` attribute, and `state_dict <OptimizationControlMechanism.state_dict>`
contains the Components associated with each value of `state <OptimizationControlMechanism.state>`.

.. _OptimizationControlMechanism_Input:

*Input*
^^^^^^^

An OptimizationControlMechanism has two types of `input_ports <Mechanism_Base.input_ports>`, corresponding to the two
forms of input it requires: `state_input_ports <OptimizationControlMechanism.state_input_ports>` that provide the values
of the Components specified as its `state_features <OptimizationControlMechanism_State_Features_Arg>`, and that are used
as inputs to the `agent_rep <OptimizationControlMechanism.agent_rep>` when its `evaluate <Composition.evaluate>` method
is used to execute it; and `outcome_input_ports <OptimizationControlMechanism.outcome_input_ports>` that provide the
outcome of executing the `agent_rep <OptimizationControlMechanism.agent_rep>`, that is used to compute the `net_outcome
<ControlMechanism.net_outcome>` for the `control_allocation <ControlMechanism.control_allocation>` under which the
execution occurred. Each of these is described below.

.. _OptimizationControlMechanism_State_Features:

*state_input_ports*
~~~~~~~~~~~~~~~~~~~

The `state_input_ports <OptimizationControlMechanism.state_input_ports>` receive `Projections <Projection>`
from the Components specified as the OptimizationControlMechanism's `state_features
<OptimizationControlMechanism_State_Features_Arg>`, the values of which are assigned as the `state_feature_values
<OptimizationControlMechanism.state_feature_values>`, and conveyed to the `agent_rep
<OptimizationControlMechanism.agent_rep>` when it is `executed <OptimizationControlMechanism_Execution>`. If the
`agent_rep is a Composition <OptimizationControlMechanism_Agent_Rep_Composition>`, then the
OptimizationControlMechanism has a state_input_port for every `InputPort` of every `INPUT <NodeRole.INPUT>` `Node
<Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>` Composition, each of which receives
a `Projection` that `shadows the input <InputPort_Shadow_Inputs>` of the corresponding state_feature. If the
`agent_rep is a CompositionFunctionApproximator <OptimizationControlMechanism_Agent_Rep_CFA>`,
then the OptimizationControlMechanism has a state_input_port that receives a Projection from each Component specified
in the **state_features** arg of its constructor.

COMMENT:
In either, case the the `values <InputPort.value>` of the
`state_input_ports <OptimizationControlMechanism.state_input_ports>` are assigned to the `state_feature_values
<OptimizationControlMechanism.state_feature_values>` attribute that is used, in turn, by the
OptimizationControlMechanism's `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method to
estimate or predict the `net_outcome <ControlMechanism.net_outcome>` for a given `control_allocation
<ControlMechanism.control_allocation>` (see `OptimizationControlMechanism_Execution`).

State features can be of two types:

* *Input Features* -- these are values that shadow the input received by a `Mechanisms <Mechanism>` in the
  `Composition` for which the OptimizationControlMechanism is a `controller <Composition.controller>` (irrespective
  of whether that is the OptimizationControlMechanism`s `agent_rep <OptimizationControlMechanism.agent_rep>`).
  They are implemented as `shadow InputPorts <InputPort_Shadow_Inputs>` (see
  `OptimizationControlMechanism_State_Features_Shadow_Inputs` for specification) that receive a
  `Projection` from the same source as the Mechanism being shadowed.
..
* *Output Features* -- these are the `value <OutputPort.value>` of an `OutputPort` of  `Mechanism <Mechanism>` in the
  `Composition` for which the OptimizationControlMechanism is a `controller <Composition.controller>` (again,
  irrespective of whether it is the OptimizationControlMechanism`s `agent_rep
  <OptimizationControlMechanism.agent_rep>`); and each is assigned a
  `Projection` from the specified OutputPort(s) to the InputPort of the OptimizationControlMechanism for that feature.

The InputPorts assigned to the **state_features** are listed in the OptimizationControlMechanism's `state_input_port
<OptimizationControlMechanism's.state_input_port>` attribute, and their current `values <InputPort.value>` are
listed in its `state_feature_values <OptimizationControlMechanism.state_feature_values>` attribute.

The InputPorts assigned to the **state_features** are listed in the OptimizationControlMechanism's `state_input_port
<OptimizationControlMechanism's.state_input_port>` attribute, and their current `values <InputPort.value>` are
listed in its `state_feature_values <OptimizationControlMechanism.state_feature_values>` attribute.
COMMENT

.. _OptimizationControlMechanism_Outcome:

*outcome_input_ports*
~~~~~~~~~~~~~~~~~~~~~

The `outcome_input_ports <OptimizationControlMechanism.outcome_input_ports>` comprise either a single `OutputPort`
that receives a `Projection` from the OptimizationControlMechanism's `objective_mechanism
<OptimizationControlMechanism_ObjectiveMechanism>` if has one; or, if it does not, then an OutputPort for each
Component it `monitors <OptimizationControlMechanism_Monitor_for_Control>` to determine the `net_outcome
<ControlMechanism.net_outcome>` of executing its `agent_rep <OptimizationControlMechanism.agent_rep>` (see `outcome
arguments <OptimizationControlMechanism_Outcome_Args>` for how these are specified). The value(s) of the
`outcome_input_ports <OptimizationControlMechanism.outcome_input_ports>` are assigned to the
OptimizationControlMechanism's `outcome <ControlMechanism.outcome>` attribute.

.. _OptimizationControlMechanism_ObjectiveMechanism:

*objective_mechanism*

If an OptimizationControlMechanism has an `objective_mechanism <ControlMechanism.objective_mechanism>`, it is
assigned a single outcome_input_port, named *OUTCOME*, that receives a Projection from the objective_mechanism's
`OUTCOME OutputPort <ObjectiveMechanism_Output>`. The OptimizationControlMechanism's `objective_mechanism
<ControlMechanism>` is used to evaluate the outcome of executing its `agent_rep
<OptimizationControlMechanism.agent_rep>` for a given `state <OptimizationControlMechanism_State>`. This passes
the result to the OptimizationControlMechanism's *OUTCOME* InputPort, that is placed in its `outcome
<ControlMechanism.outcome>` attribute.

    .. note::
        An OptimizationControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>` and its `function
        <ObjectiveMechanism.function>` are distinct from, and should not be confused with the `objective_function
        <OptimizationFunction.objective_function>` parameter of the OptimizationControlMechanism's `function
        <OptimizationControlMechanism.function>`.  The `objective_mechanism <ControlMechanism.objective_mechanism>`\\'s
        `function <ObjectiveMechanism.function>` evaluates the `outcome <ControlMechanism.outcome>` of processing
        without taking into account the `costs <ControlMechanism.costs>` of the OptimizationControlMechanism's
        `control_signals <OptimizationControlMechanism.control_signals>`.  In contrast, its `evaluate_agent_rep
        <OptimizationControlMechanism.evaluate_agent_rep>` method, which is assigned as the `objective_function`
        parameter of its `function <OptimizationControlMechanism.function>`, takes the `costs <ControlMechanism.costs>`
        of the OptimizationControlMechanism's `control_signals <OptimizationControlMechanism.control_signals>` into
        account when calculating the `net_outcome` that it returns as its result.

COMMENT:
ADD HINT HERE RE: USE OF CONCATENATION

the items specified by `monitor_for_control
<ControlMechanism.monitor_for_control>` are all assigned `MappingProjections <MappingProjection>` to a single
*OUTCOME* InputPort.  This is assigned `Concatenate` as it `function <InputPort.function>`, which concatenates the
`values <Projection_Base.value>` of its Projections into a single array (that is, it is automatically configured
to use the *CONCATENATE* option of a ControlMechanism's `outcome_input_ports_option
<ControlMechanism.outcome_input_ports_option>` Parameter). This ensures that the input to the
OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` has the same format as when an
`objective_mechanism <ControlMechanism.objective_mechanism>` has been specified, as described below.
COMMENT

.. _OptimizationControlMechanism_Monitor_for_Control:

*monitor_for_control*

If an OptimizationControlMechanism is not assigned an `objective_mechanism <ControlMechanism.objective_mechanism>`,
then its `outcome_input_ports <OptimizationControlMechanism.outcome_input_ports>` are determined by its
`monitor_for_control <ControlMechanism.monitor_for_control>` and `outcome_input_ports_option
<ControlMechanism.outcome_input_ports_option>` attributes, specified in the corresponding arguments of its
constructor (see `Outcomes arguments <OptimizationControlMechanism_Outcome_Args>`). The value(s) of the specified
Components are assigned as the OptimizationControlMechanism's `outcome <ControlMechanism.outcome>` attribute,
which is used to compute the `net_outcome <ControlMechanism.net_outcome>` of executing its `agent_rep
<OptimizationControlMechanism.agent_rep>`.

    .. note::
       If a `Node <Composition_Nodes>` other than an `OUTPUT <NodeRole.OUTPUT>` of a `nested <Composition_Nested>`
       Composition is `specified to be monitored <ControlMechanism_Monitor_for_Control>`, it is assigned as a `PROBE
       <NodeRole.PROBE>` of that nested Composition. Although `PROBE <NodeRole.PROBE>` Nodes are generally treated
       like `OUTPUT <NodeRole.OUTPUT>` Nodes (since they project out of the Composition to which they belong), their
       `value <Mechanism_Base.value>` is not included in the `output_values <Composition.output_values>` or `results
       <Composition.results>` attributes of the Composition for which the OptimizationControlMechanism is the
       `controller <Composition.controller>`, unless that Composition's `include_probes_in_output
       <Composition.include_probes_in_output>` attribute is set to True (see Probes `Composition_Probes` for additional
       information).

.. _OptimizationControlMechanism_Function:

*Function*
^^^^^^^^^^

The `function <OptimizationControlMechanism.function>` of an OptimizationControlMechanism is used to find the
`control_allocation <ControlMechanism.control_allocation>` that optimizes the `net_outcome
<ControlMechanism.net_outcome>` for the current (or expected) `state <OptimizationControlMechanism_State>`.
It is generally an `OptimizationFunction`, which in turn has `objective_function
<OptimizationFunction.objective_function>`, `search_function <OptimizationFunction.search_function>` and
`search_termination_function <OptimizationFunction.search_termination_function>` methods, as well as a `search_space
<OptimizationFunction.search_space>` attribute.  The `objective_function <OptimizationFunction.objective_function>`
is automatically assigned the OptimizationControlMechanism's `evaluate_agent_rep
<OptimizationControlMechanism.evaluate_agent_rep>` method, that is used to evaluate each `control_allocation
<ControlMechanism.control_allocation>` sampled from the `search_space <OptimizationFunction.search_space>` by the
`search_function <OptimizationFunction.search_function>` until the `search_termination_function
<OptimizationFunction.search_termination_function>` returns `True` (see `OptimizationControlMechanism_Execution`
for additional details).

.. _OptimizationControlMechanism_Custom_Function:

*Custom Function*
~~~~~~~~~~~~~~~~~

A custom function can be assigned as the OptimizationControlMechanism's `function
<OptimizationControlMechanism.function>`, however it must meet the following requirements:

  - It must accept as its first argument and return as its result an array with the same shape as the
    OptimizationControlMechanism's `control_allocation <ControlMechanism.control_allocation>`.
  ..
  - It must be able to execute the OptimizationControlMechanism's `evaluate_agent_rep
    <OptimizationControlMechanism.evaluate_agent_rep>` `num_estimates <OptimizationControlMechanism.num_estimates>`
    times, and aggregate the results in computing the `net_outcome <ControlMechanism.net_outcome>` for a given
    `control_allocation <ControlMechanism.control_allocation>` (see
    `OptimizationControlMechanism_Estimation_Randomization` for additional details).
  ..
  - It must implement a `reset` method that can accept as keyword arguments **objective_function**,
    **search_function**, **search_termination_function**, and **search_space**, and implement attributes
    with corresponding names.

.. _OptimizationControlMechanism_Search_Functions:

*Search Function, Search Space and Search Termination Function*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclasses of OptimizationControlMechanism may implement their own `search_function
<OptimizationControlMechanism.search_function>` and `search_termination_function
<OptimizationControlMechanism.search_termination_function>` methods, as well as a
`control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>` attribute, that are
passed as parameters to the `OptimizationFunction` when it is constructed.  These can be specified in the
constructor for an `OptimizationFunction` assigned as the **function** argument in the
OptimizationControlMechanism's constructor, as long as they are compatible with the requirements of
the OptimizationFunction and OptimizationControlMechanism.  If they are not specified, then defaults specified
either by the OptimizationControlMechanism or the OptimizationFunction are used.

.. _OptimizationControlMechanism_Default_Function:

*Default Function: GridSearch*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the **function** argument is not specified, the `GridSearch` `OptimizationFunction` is assigned as the default,
which evaluates the `net_outcome <ControlMechanism.net_outcome>` using the OptimizationControlMechanism's
`control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>` as its
`search_space <OptimizationFunction.search_space>`, and returns the `control_allocation
<ControlMechanism.control_allocation>` that yields the greatest `net_outcome <ControlMechanism.net_outcome>`,
thus implementing a computation of `EVC <OptimizationControlMechanism_EVC>`.


.. _OptimizationControlMechanism_Output:

*Output*
^^^^^^^^

The output of OptimizationControlMechanism are its `control_signals <ControlMechanism.control_signals>` that implement
the `control_allocations <ControlMechanism.control_allocation>` it evaluates and optimizes. These their effects are
estimated over variation in the values of Components with random variables, then the OptimizationControlMechanism's
`control_signals <ControlMechanism.control_signals>` include an additional *RANDOMIZATION_CONTROL_SIGNAL* that
implements that variablity for the relevant Components, as described below.

.. _OptimizationControlMechanism_Randomization_Control_Signal:

*Randomization ControlSignal*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If `num_estimates <OptimizationControlMechanism.num_estimates>` is specified (that is, it is not None), and
`agent_rep <OptimizationControlMechanism.agent_rep>` has any `Components <Component>` with random variables
(that is, that call a randomization function) specified in the OptimizationControlMechanism's `random_variables
<OptimizationControlMechanism.random_variables>` attribute, then a `ControlSignal` is automatically added to the
OptimizationControlMechanism's `control_signals <OptimizationControlMechanism.control_signals>`, named
*RANDOMIZATION_CONTROL_SIGNAL*, that randomizes the values of the `random variables
<OptimizationControlMechanism.random_variables>` over estimates of its `net_outcome <ControlMechanism.net_outcome>`
for each `control_allocation <ControlMechanism.control_allocation>` If `num_estimates
<OptimizationControlMechanism.num_estimates>` is specified but `agent_rep <OptimizationControlMechanism.agent_rep>`
has not random variables, then a warning is issued and no *RANDOMIZATION_CONTROL_SIGNAL* is constructed. The
`initial_seed <OptimizationControlMechanism.initial_seed>` and `same_seed_for_all_allocations
<OptimizationControlMechanism.same_seed_for_all_allocations>` Parameters can also be used to further refine
randomization (see `OptimizationControlMechanism_Estimation_Randomization` for additional details).

.. technical_note::

    The *RANDOMIZATION_CONTROL_SIGNAL* ControlSignal sends a `ControlProjection` to the `ParameterPort` for the
    see `Parameter` of Components specified either in the OptimizationControlMechanism's `random_variables
    <OptimizationControlMechanism.random_variables>` attribute or that of the `agent_rep
    <OptimizationControlMechanism.agent_rep>` (see above). The *RANDOMIZATION_CONTROL_SIGNAL* is also included when
    constructing the `control_allocation_search_space <OptimizationFunction.control_allocation_search_space>` passed
    to the constructor for OptimizationControlMechanism's `function <OptimizationControlMechanism.function>`,
    as its **search_space** argument, along with the index of the *RANDOMIZATION_CONTROL_SIGNAL* as its
    **randomization_dimension** argument.

.. _OptimizationControlMechanism_Execution:

Execution
---------

When an OptimizationControlMechanism is executed, the `OptimizationFunction` assigned as it's `function
<OptimizationControlMechanism.function>` is used evaluate the effects of different `control_allocations
<ControlMechanism.control_allocation>` to find one that optimizes the `net_outcome <ControlMechanism.net_outcome>`;
that `control_allocation <ControlMechanism.control_allocation>` is then used when the Composition controlled by the
OptimizationControlMechanism is next executed.  The OptimizationFunction does this by either simulating performance
of the Composition or executing the CompositionFunctionApproximator that is its `agent_rep
<OptimizationControlMechanism.agent_rep>`.

.. _OptimizationControlMechanism_Execution_Timing:

*Timing of Execution*
^^^^^^^^^^^^^^^^^^^^^

When the OptimizationControlMechanism is executed is determined by the `controller_mode <Composition.controller_mode>`
of the Composition for which the OptimizationControlMechanism is the `controller <Composition_Controller>`:  if it is
set to *AFTER* (the default), the OptimizationControlMechanism is executed at the end of a `TRIAL <TimeScale.TRIAL>`,
after the Composition has executed, using `state_feature_value <OptimizationControlMechanism.state_feature_values>`
(including any inputs to the Composition) for that `TRIAL <TimeScale.TRIAL>`; if the `controller_mode
<Composition.controller_mode>` is *BEFORE*, then the OptimizationControlMechanism is executed before the Composition
that it controls, using `state_feature_value <OptimizationControlMechanism.state_feature_values>` (including any inputs
to the Composition) from the previous `TRIAL <TimeScale.TRIAL>`.

.. _OptimizationControlMechanism_Optimization_Procedure:

*Optimization Procedure*
^^^^^^^^^^^^^^^^^^^^^^^^

When an OptimizationControlMechanism is executed, it carries out the following steps to find a `control_allocation
<ControlMechanism.control_allocation>` that optmimzes performance of the Composition that it controls:

  .. _OptimizationControlMechanism_Adaptation:

  * *Adaptation* -- if the `agent_rep <OptimizationControlMechanism.agent_rep>` is a `CompositionFunctionApproximator`,
    its `adapt <CompositionFunctionApproximator.adapt>` method, allowing it to modify its parameters in order to better
    predict the `net_outcome <ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>`,
    based the state and `net_outcome <ControlMechanism.net_outcome>` of the previous `TRIAL <TimeScale.TRIAL>`.

  .. _OptimizationControlMechanism_Evaluation:

  * *Evaluation* -- the OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` is
    called to find the `control_allocation <ControlMechanism.control_allocation>` that optimizes `net_outcome
    <ControlMechanism.net_outcome>` of its `agent_rep <OptimizationControlMechanism.agent_rep>` for the current
    `state <OptimizationControlMechanism_State>`. The way in which it searches for the best `control_allocation
    <ControlMechanism.control_allocation>` is determined by the type of `OptimizationFunction` assigned to `function
    <OptimizationControlMechanism.function>`, whereas the way that it evaluates each one is determined by the
    OptimizationControlMechanism's `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method.
    More specifically, it carries out the following procedure:

    .. _OptimizationControlMechanism_Estimation:

    * *Estimation* - the `function <OptimizationControlMechanism.function>` selects a sample `control_allocation
      <ControlMechanism.control_allocation>` (using its `search_function <OptimizationFunction.search_function>`
      to select one from its `search_space <OptimizationFunction.search_space>`), and evaluates the `net_outcome
      <ControlMechanism.net_outcome>` for that `control_allocation <ControlMechanism.control_allocation>`.
      It does this by calling the OptimizationControlMechanism's `evaluate_agent_rep
      <OptimizationControlMechanism.evaluate_agent_rep>` method `num_estimates <OptimizationControlMechanism>`
      times, each of which uses the `state_feature_values <OptimizationControlMechanism.state_feature_values>`
      and `control_allocation <ControlMechanism.control_allocation>` as the input to the `agent_rep
      <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method, executing it for
      `num_trials_per_estimate <OptimizationControlMechanism.num_trials_per_estimate>` trials for each estimate.
      The `state_feature_values <OptimizationControlMechanism.state_feature_values>` and `control_allocation
      <ControlMechanism.control_allocation>` remain fixed for each estimate, but the random seeds of any Parameters
      that rely on randomization are varied, so that the values of those Parameters are randomly sampled for every
      estimate (see `OptimizationControlMechanism_Estimation_Randomization`).

    * *Aggregation* - the `function <OptimizationControlMechanism.function>`\\'s `aggregation_function
      <OptimizationFunction.aggregation_function>` is used to aggregate the `net_outcome
      <ControlMechanism.net_outcome>` over the all the estimates for a given `control_allocation
      <ControlMechanism.control_allocation>`, and the aggregated value is returned as the `outcome
      <ControlMechanism.outcome>` and used to the compute the `net_outcome <ControlMechanism.net_outcome>`
      for that `control_allocation <ControlMechanism.control_allocation>`.

    * *Termination* - the `function <OptimizationControlMechanism.function>` continues to evaluate samples of
      `control_allocations <ControlMechanism.control_allocation>` provided by its `search_function
      <OptimizationFunction.search_function>` until its `search_termination_function
      <OptimizationFunction.search_termination_function>` returns `True`.

  .. _OptimizationControlMechanism_Control_Assignment:

  * *Assignment* - when the search completes, the `function <OptimizationControlMechanism.function>`
    assigns the `control_allocation <ControlMechanism.control_allocation>` that yielded the optimal value of
    `net_outcome <ControlMechanism.net_outcome>` to the OptimizationControlMechanism's `control_signals,
    that compute their `values <ControlSignal.value>` which, in turn, are assigned to their `ControlProjections
    <ControlProjection>` to `modulate the Parameters <ModulatorySignal_Modulation>` they control when the
    Composition is next executed.

.. _OptimizationControlMechanism_Estimation_Randomization:

*Randomization of Estimation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If `num_estimates <OptimizationControlMechanism.num_estimates>` is specified (i.e., it is not None), then each
`control_allocation <ControlMechanism.control_allocation>` is independently evaluated `num_estimates
<OptimizationControlMechanism.num_estimates>` times (i.e., by that number of calls to the
OptimizationControlMechanism's `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method).
The values of Components listed in the OptimizationControlMechanism's `random_variables
<OptimizationControlMechanism.random_variables>` attribute are randomized over those estimates.  By default,
this includes all Components in the `agent_rep <OptimizationControlMechanism.agent_rep>` with random variables
(listed in its `random_variables <Composition.random_variables>` attribute).  However, if particular Components
are specified in the **random_variables** argument of the OptimizationControlMechanism's constructor, then
randomization is restricted to their values. Randomization over estimates can be further configured using the
`initial_seed <OptimizationControlMechanism.initial_seed>` and `same_seed_for_all_allocations
<OptimizationControlMechanism.same_seed_for_all_allocations>` attributes. The results of all the estimates for a
given `control_allocation <ControlMechanism.control_allocation>` are aggregated by the `aggregation_function
<OptimizationFunction.aggregation_function>` of the `OptimizationFunction` assigned to the
OptimizationControlMechanism's `function <OptimizationControlMechanism>`, and used to compute the `net_outcome
<ControlMechanism.net_outcome>` over the estimates for that `control_allocation <ControlMechanism.control_allocation>`
(see `OptimizationControlMechanism_Execution` for additional details).

COMMENT:
.. _OptimizationControlMechanism_Examples:

Examples
--------

The table below lists `model-free <OptimizationControlMechanism_Model_Free>` and `model-based
<ModelBasedOptimizationControlMechanism` subclasses of OptimizationControlMechanisms.

.. table:: **Model-Free and Model-Based OptimizationControlMechanisms**

   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |                         |     *Model-Free*     |                           *Model-Based*                                                         |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |**Functions**            |`LVOCControlMechanism`| LVOMControlMechanism | MDPControlMechanism |`EVCControlMechanism`| ParameterEstimationMechanism |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |**learning_function**    |     `BayesGLM`       |        `pymc`        |    `BeliefUpdate`   |       *None*        |           `pymc`             |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |**function** *(primary)* |`GradientOptimization`|     `GridSearch`     |       `Sample`      |    `GridSearch`     |           `Sample`           |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |       *search_function* |  *follow_gradient*   |   *traverse_grid*    | *sample_from_dist*  |   *traverse_grid*   |      *sample_from_dist*      |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |    *objective_function* |    *compute_EVC*     |  *evaluate*,   |  *evaluate*,  |  *evaluate*,  |    *evaluate*,         |
   |                         |                      |  *compute_EVC*       |  *compute_EVC*      |  *compute_EVC*      |    *compute_likelihood*      |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+
   |             *execution* | *iterate w/in trial* |  *once per trial*    | *iterate w/in trial*| *iterate w/in trial*|     *iterate w/in trial*     |
   +-------------------------+----------------------+----------------------+---------------------+---------------------+------------------------------+

The following models provide examples of implementing the OptimizationControlMechanisms in the table above:

`LVOCControlMechanism`\\:  `BustamanteStroopXORLVOCModel`
`EVCControlMechanism`\\:  `UmemotoTaskSwitchingEVCModel`
COMMENT



.. _OptimizationControlMechanism_Class_Reference:

Class Reference
---------------

"""
import ast
import copy
import warnings
from collections.abc import Iterable
from typing import Union

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility, Component
from psyneulink.core.components.functions.function import is_function_type
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import \
    GridSearch, OBJECTIVE_FUNCTION, SEARCH_SPACE, RANDOMIZATION_DIMENSION
from psyneulink.core.components.functions.nonstateful.transferfunctions import CostFunctions
from psyneulink.core.components.mechanisms.mechanism import Mechanism
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import \
    ControlMechanism, ControlMechanismError
from psyneulink.core.components.ports.inputport import InputPort, _parse_shadow_inputs
from psyneulink.core.components.ports.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.ports.outputport import OutputPort
from psyneulink.core.components.ports.port import _parse_port_spec, _instantiate_port, Port
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    ALL, COMPOSITION, COMPOSITION_FUNCTION_APPROXIMATOR, CONCATENATE, DEFAULT_INPUT, DEFAULT_VARIABLE, EID_FROZEN, \
    FUNCTION, INTERNAL_ONLY, NAME, OPTIMIZATION_CONTROL_MECHANISM, NODE, OWNER_VALUE, PARAMS, PORT, PROJECTIONS, \
    SHADOW_INPUTS, SHADOW_INPUT_NAME, VALUE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.sampleiterator import SampleIterator, SampleSpec
from psyneulink.core.globals.utilities import convert_to_list, convert_to_np_array, ContentAddressableList, is_numeric
from psyneulink.core.llvm.debug import debug_env

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError',
    'AGENT_REP', 'STATE_FEATURES', 'STATE_FEATURE_FUNCTION', 'RANDOMIZATION_CONTROL_SIGNAL', 'NUM_ESTIMATES'
]

AGENT_REP = 'agent_rep'
STATE_FEATURES = 'state_features'
STATE_FEATURE_FUNCTION = 'state_feature_function'
RANDOMIZATION_CONTROL_SIGNAL = 'RANDOMIZATION_CONTROL_SIGNAL'
RANDOM_VARIABLES = 'random_variables'
NUM_ESTIMATES = 'num_estimates'

def _state_feature_values_getter(owning_component=None, context=None):
    # If no state_input_ports return empty list
    if (not owning_component.num_state_input_ports):
        return []
    # If OptimizationControlMechanism is still under construction, use items from input_values as placemarkers
    elif context.source == ContextFlags.CONSTRUCTOR:
        return owning_component.input_values[owning_component.num_outcome_input_ports:]

    # Otherwise, use current values of state_input_ports
    state_input_port_values = [p.parameters.value.get(context) for p in owning_component.state_input_ports]

    if (not owning_component.state_feature_specs
            or owning_component.num_state_input_ports == len(owning_component.state_feature_specs)):
        # Automatically assigned state_features or full set of specs for all INPUT Nodes,
        #   so use values of state_input_ports (since there is one for every INPUT Node of agent_rep)
        state_feature_values = state_input_port_values
    else:
        # Specified state_features for a subset of INPUT Nodes so use those
        j = 0
        state_feature_values = []
        for node, spec in zip(owning_component._specified_input_ports_in_order,
                              owning_component.state_feature_specs):
            if spec is not None:
                state_feature_values.append(state_input_port_values[j])
                j += 1
            else:
                # FIX: 1/29/22 - HANDLE VARIBLE AS 1d vs 2d
                # assert node.defaults.variable.ndim == 2 and len(node.defaults.variable)==1
                # state_feature_values.append(node.defaults.variable[0])
                state_feature_values.append(node.defaults.variable)

    return convert_to_np_array(state_feature_values)

class OptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _control_allocation_search_space_getter(owning_component=None, context=None):
    search_space = owning_component.parameters.search_space._get(context)
    if search_space is None:
        return [c.parameters.allocation_samples._get(context) for c in owning_component.control_signals]
    else:
        return search_space


class OptimizationControlMechanism(ControlMechanism):
    """OptimizationControlMechanism(                    \
        agent_rep=None,                                 \
        state_features=None,                            \
        state_feature_function=None,                    \
        monitor_for_control=None,                       \
        objective_mechanism=None,                       \
        function=GridSearch,                            \
        num_estimates=1,                                \
        random_variables=ALL,                           \
        initial_seed=None,                              \
        same_seed_for_all_parameter_combinations=False  \
        num_trials_per_estimate=None,                   \
        search_function=None,                           \
        search_termination_function=None,               \
        search_space=None,                              \
        control_signals=None,                           \
        modulation=MULTIPLICATIVE,                      \
        combine_costs=np.sum,                           \
        compute_reconfiguration_cost=None,              \
        compute_net_outcome=lambda x,y:x-y)

    Subclass of `ControlMechanism <ControlMechanism>` that adjusts its `ControlSignals <ControlSignal>` to optimize
    performance of the `Composition` to which it belongs.  See `ControlMechanism <ControlMechanism_Class_Reference>`
    for arguments not described here.

    Arguments
    ---------

    state_features : Mechanism, InputPort, OutputPort, Projection, numeric value, dict, or list containing any of these
        specifies the Components from which `state_input_ports <OptimizationControlMechanism.state_input_ports>`
        receive their inputs, the `values <InputPort.value>` of which are assigned to `state_feature_values
        <OptimizationControlMechanism.state_feature_values>` and provided as input to the `agent_rep
        <OptimizationControlMechanism.agent_rep>'s `evaluate <Composition.evaluate>` method when it is executed.
        See `state_features <OptimizationControlMechanism_State_Features_Arg>` for details of specification.

    state_feature_function : Function or function : default None
        specifies the `function <InputPort.function>` to use as the default function for the `state_input_ports
        <OptimizationControlMechanism.state_input_ports>` created for the corresponding **state_features** (see
        `state_feature_function <OptimizationControlMechanism_State_Feature_Function_Arg>` for additional details).

    agent_rep : None or Composition  : default None or Composition to which OptimizationControlMechanism is assigned
        specifies the `Composition` used by `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>`
        to predict the `net_outcome <ControlMechanism.net_outcome>` for a given `state
        <OptimizationControlMechanism_State>`.  If a Composition is specified, it must be suitably configured
        (see `agent_rep <OptimizationControlMechanism_Agent_Rep_Arg>` for additional details).  It can also be a
        `CompositionFunctionApproximator`, or subclass of one, used for `model-free
        <OptimizationControlMechanism_Model_Free>` optimization. If **agent_rep** is not specified, the
        OptimizationControlMechanism is placed in `deferred_init <Component_Deferred_Init>` status until it is assigned
        as the `controller <Composition.controller>` of a Composition, at which time that Composition is assigned as
        the `agent_rep <OptimizationControlMechanism.agent_rep>`.

    num_estimates : int : 1
        specifies the number independent runs of `agent_rep <OptimizationControlMechanism.agent_rep>` randomized
        over **random_variables** and used to estimate its `net_outcome <ControlMechanism.net_outcome>` for each
        `control_allocation <ControlMechanism.control_allocation>` sampled (see `num_estimates
        <OptimizationControlMechanism.num_estimates>` for additional information).

    random_variables : Parameter or list[Parameter] : default ALL
        specifies the Components of `agent_rep <OptimizationControlMechanism.agent_rep>` with random variables to be
        randomized over different estimates of each `control_allocation <ControlMechanism.control_allocation>`;  these
        must be in the `agent_rep <OptimizationControlMechanism.agent_rep>` and have a `seed` `Parameter`. By default,
        all such Components (listed in its `random_variables <Composition.random_variables>` attribute) are included
        (see `random_variables <OptimizationControlMechanism.random_variables>` for additional information).

        .. note::
           if **num_estimates** is specified but `agent_rep <OptimizationControlMechanism.agent_rep>` has no
           `random variables <Composition.random_variables>`, a warning is generated and `num_estimates
           <OptimizationControlMechanism.num_estimates>` is set to None.

    initial_seed : int : default None
        specifies the seed used to initialize the random number generator at construction.
        If it is not specified then then the seed is set to a random value (see `initial_seed
        <OptimizationControlMechanism.initial_seed>` for additional information).

    same_seed_for_all_parameter_combinations :  bool : default False
        specifies whether the random number generator is re-initialized to the same value when estimating each
        `control_allocation <ControlMechanism.control_allocation>` (see `same_seed_for_all_parameter_combinations
        <OptimizationControlMechanism.same_seed_for_all_allocations>` for additional information).

    num_trials_per_estimate : int : default None
        specifies the number of trials to execute in each run of `agent_rep
        <OptimizationControlMechanism.agent_rep>` by a call to `evaluate_agent_rep
        <OptimizationControlMechanism.evaluate_agent_rep>` (see `num_trials_per_estimate
        <OptimizationControlMechanism.num_trials_per_estimate>` for additional information).

    search_function : function or method
        specifies the function assigned to `function <OptimizationControlMechanism.function>` as its
        `search_function <OptimizationFunction.search_function>` parameter, unless that is specified in a
        constructor for `function <OptimizationControlMechanism.function>`.  It must take as its arguments
        an array with the same shape as `control_allocation <ControlMechanism.control_allocation>` and an integer
        (indicating the iteration of the `optimization process <OptimizationFunction_Procedure>`), and return
        an array with the same shape as `control_allocation <ControlMechanism.control_allocation>`.

    search_termination_function : function or method
        specifies the function assigned to `function <OptimizationControlMechanism.function>` as its
        `search_termination_function <OptimizationFunction.search_termination_function>` parameter, unless that is
        specified in a constructor for `function <OptimizationControlMechanism.function>`.  It must take as its
        arguments an array with the same shape as `control_allocation <ControlMechanism.control_allocation>` and two
        integers (the first representing the `net_outcome <ControlMechanism.net_outcome>` for the current
        `control_allocation <ControlMechanism.control_allocation>`, and the second the current iteration of the
        `optimization process <OptimizationFunction_Procedure>`);  it must return `True` or `False`.

    search_space : iterable [list, tuple, ndarray, SampleSpec, or SampleIterator] | list, tuple, ndarray, SampleSpec, or SampleIterator
        specifies the `search_space <OptimizationFunction.search_space>` parameter for `function
        <OptimizationControlMechanism.function>`, unless that is specified in a constructor for `function
        <OptimizationControlMechanism.function>`.  An element at index i should correspond to an element at index i in
        `control_allocation <ControlMechanism.control_allocation>`. If
        `control_allocation <ControlMechanism.control_allocation>` contains only one element, then search_space can be
        specified as a single element without an enclosing iterable.

    function : OptimizationFunction, function or method
        specifies the function used to optimize the `control_allocation <ControlMechanism.control_allocation>`;
        must take as its sole argument an array with the same shape as `control_allocation
        <ControlMechanism.control_allocation>`, and return a similar array (see `Function
        <OptimizationControlMechanism_Function>` for additional details).

    Attributes
    ----------

    agent_rep : Composition
        determines the `Composition` used by the `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>`
        method to predict the `net_outcome <ControlMechanism.net_outcome>` for a given `state
        <OptimizationControlMechanism_State>`; see `Agent Representation <OptimizationControlMechanism_Agent_Rep>`
        for additional details.

    agent_rep_type : None, COMPOSITION or COMPOSITION_FUNCTION_APPROXIMATOR
        identifies whether the agent_rep is a `Composition`, a `CompositionFunctionApproximator` or
        one of its subclasses, or it has not been assigned (None) (see `Agent Representation and Types
        of Optimization <OptimizationControlMechanism_Agent_Representation_Types>` for additional details).

    state_features : Dict[Node:source]
        dictionary in which keys are Mechanism's that are `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of
        `agent_rep <OptimizationControlMechanism.agent_rep>` and/or any `nested Compositions <Composition_Nested>`
        within it, and values are sources of input specified in **state_features** (or determined automatically). The
        latter are provided as the inputs to `state_input_ports <OptimizationControlMechanism.state_input_ports>`, the
        values of which are assigned to `state_feature_values <OptimizationControlMechanism.state_feature_values>` and
        provided as input to the `agent_rep <OptimizationControlMechanism.agent_rep>'s `evaluate <Composition.evaluate>`
        method when it is executed (see `state_features <OptimizationControlMechanism_State_Features_Arg>` and
        `OptimizationControlMechanism_State_Features` for additional details).

        .. technical_note::
            the state_features dict is used by the _build_predicted_inputs() method of an `agent_rep
            <OptimizationControlMechanism>` Composition to construct inputs for its `evaluate
            <Composition.evaluate_method>` method. Only Mechanisms are used as keys, to accommodate the possibility
            that some but not all of the `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of any nested
            composition(s) are specified in the `state_features <OptimizationControlMechanism_State_Features_Arg>`
            of the OptimizationControlMechanism's constructor, allowing _build_predicted_inputs() to identify and
            provide defaults for any that are not specified. Accordingly, if **state_features** is not specified in
            the constructor, and thus assigned automatically, the state_features dictionary will contain entries for
            shadowing the InputPorts of all Mechanisms that are `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>`
            in either `agent_rep <OptimizationControlMechanism.agent_rep>` and/or of any Compositions nested at any
            level within it.

    state_feature_values : 2d array
        the current value of each item of the OptimizationControlMechanism's `state_input_ports
        <OptimizationControlMechanism.state_input_ports>  (see `OptimizationControlMechanism_State_Features`
        for additional details).

    state_feature_function : Function of function
        determines the `function <InputPort.function>` used as the default function for
        `state_input_ports <OptimizationControlMechanism.state_input_ports>` (see `state_feature_function
        <OptimizationControlMechanism_State_Feature_Function_Arg>` for additional details).

    state_input_ports : ContentAddressableList
        lists the OptimizationControlMechanism's `InputPorts <InputPort>` that receive `Projections <Projection>`
        from the items specified in the **state_features** argument in the OptimizationControlMechanism's constructor,
        or constructed automatically (see `state_features <OptimizationControlMechanism_State_Features_Arg>`), the
        values of which are assigned to `state_feature_values <OptimizationControlMechanism.state_feature_values>`
        and provided as input to the `agent_rep <OptimizationControlMechanism.agent_rep>'s `evaluate
        <Composition.evaluate>` method (see `OptimizationControlMechanism_State_Features` for additional details).

    num_state_input_ports : int
        contains the number of `state_input_ports <OptimizationControlMechanism.state_input_ports>`.

    outcome_input_ports : ContentAddressableList
        lists the OptimizationControlMechanism's `OutputPorts <OutputPort>` that receive `Projections <Projection>`
        from either its `objective_mechanism <ControlMechanism.objective_mechanism>` or the Components listed in
        its `monitor_for_control <ControlMechanism.monitor_for_control>` attribute, the values of which are used
        to compute the `net_outcome <ControlMechanism.net_outcome>` of executing the `agent_rep
        <OptimizationControlMechanism.agent_rep>` in a given `OptimizationControlMechanism_State`
        (see `Outcome <OptimizationControlMechanism_Outcome>` for additional details).

    state : ndarray
        lists the values of the current state -- a concatenation of the `state_feature_values
        <OptimizationControlMechanism.state_feature_values>` and `control_allocation
        <ControlMechanism.control_allocation>` following the last execution of `agent_rep
        <OptimizationControlMechanism.agent_rep>`.

    state_dict : Dict[(Port, Mechanism, Composition, index)):value]
        dictionary containing information about the Components corresponding to the values in `state
        <OptimizationControlMechanism.state>`.  Keys are (`Port`, `Mechanism`, `Composition`, index) tuples,
        identifying the source of the value for each item at the corresponding index in
        `state <OptimizationControlMechanism.state>`, and values are its value in `state
        <OptimizationControlMechanism.state>`. The initial entries are for the OptimizationControlMechanism's
        `state features <OptimizationControlMechanism.state_features>`, that are the sources of its
        `state_feature_values <OptimizationControlMechanism.state_feature_values>`;  they are followed
        by entries for the parameters modulated by the OptimizationControlMechanism's `control_signals
        <OptimizationControlMechanism_Output>` with the corresponding `control_allocation
        <ControlMechanism.control_allocation>` values.

    num_estimates : int
        determines the number independent runs of `agent_rep <OptimizationControlMechanism.agent_rep>` (i.e., calls to
        `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>`) used to estimate the `net_outcome
        <ControlMechanism.net_outcome>` of each `control_allocation <ControlMechanism.control_allocation>` evaluated
        by the OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` (i.e.,
        that are specified by its `search_space <OptimizationFunction.search_space>`); see
        `OptimizationControlMechanism_Estimation_Randomization` for additional details.

    random_variables : Parameter or List[Parameter]
        list of the `Parameters <Parameter>` in `agent_rep <OptimizationControlMechanism.agent_rep>` with random
        variables (that is, ones that call a randomization function) that are randomized over estimates for a given
        `control_allocation <ControlMechanism.control_allocation>`;  by default, all Components in the `agent_rep
        <OptimizationControlMechanism.agent_rep>` with random variables are included (listed in its `random_variables
        <Composition.random_variables>` attribute);  see `OptimizationControlMechanism_Estimation_Randomization`
        for additional details.

    initial_seed : int or None
        determines the seed used to initialize the random number generator at construction.
        If it is not specified then then the seed is set to a random value, and different runs of a
        Composition containing the OptimizationControlMechanism will yield different results, which should be roughly
        comparable if the estimation process is stable.  If **initial_seed** is specified, then running the Composition
        should yield identical results for the estimation process, which can be useful for debugging.

    same_seed_for_all_allocations :  bool
        determines whether the random number generator used to select seeds for each estimate of the `agent_rep
        <OptimizationControlMechanism.agent_rep>`\\'s `net_outcome <ControlMechanism.net_outcome>` is re-initialized
        to the same value for each `control_allocation <ControlMechanism.control_allocation>` evaluated.
        If same_seed_for_all_allocations is True, then any differences in the estimates made of `net_outcome
        <ControlMechanism.net_outcome>` for each `control_allocation <ControlMechanism.control_allocation>` will
        reflect exclusively the influence of the different control_allocations on the execution of the `agent_rep
        <OptimizationControlMechanism.agent_rep>`, and *not* any variability intrinsic to the execution of
        the Composition itself (e.g., any of its Components). This can be confirmed by identical results for repeated
        executions of the OptimizationControlMechanism's `evaluate_agent_rep
        <OptimizationControlMechanism.evaluate_agent_rep>` method for the same `control_allocation
        <ControlMechanism.control_allocation>`. If same_seed_for_all_allocations is False, then each time a
        `control_allocation <ControlMechanism.control_allocation>` is estimated, it will use a different set of seeds.
        This can be confirmed by differing results for repeated executions of the OptimizationControlMechanism's
        `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method with the same `control_allocation
        <ControlMechanism.control_allocation>`). Small differences in results suggest
        stability of the estimation process across `control_allocations <ControlMechanism.control_allocation>`, while
        substantial differences indicate instability, which may be helped by increasing `num_estimates
        <OptimizationControlMechanism.num_estimates>`.

    num_trials_per_estimate : int or None
        imposes an exact number of trials to execute in each run of `agent_rep <OptimizationControlMechanism.agent_rep>`
        used to evaluate its `net_outcome <ControlMechanism.net_outcome>` by a call to the
        OptimizationControlMechanism's `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method.
        If it is None (the default), then either the number of **inputs** or the value specified for **num_trials** in
        the Composition's `run <Composition.run>` method used to determine the number of trials executed (see
        `number of trials <Composition_Execution_Num_Trials>` for additional information).

    function : OptimizationFunction, function or method
        takes current `control_allocation <ControlMechanism.control_allocation>` (as initializer),
        uses its `search_function <OptimizationFunction.search_function>` to select samples of `control_allocation
        <ControlMechanism.control_allocation>` from its `search_space <OptimizationFunction.search_space>`,
        evaluates these using its `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method by
        calling it `num_estimates <OptimizationControlMechanism.num_estimates>` times to estimate its `net_outcome
        `net_outcome <ControlMechanism.net_outcome>` for a given `control_allocation
        <ControlMechanism.control_allocation>`, and returns the one that yields the optimal `net_outcome
        <ControlMechanism.net_outcome>` (see `Function <OptimizationControlMechanism_Function>` for additional details).

    evaluate_agent_rep : function or method
        returns the `net_outcome(s) <ControlMechanism.net_outcome>` for a given `state
        <OptimizationControlMechanism_State>` (i.e., combination of `state_feature_values
        <OptimizationControlMechanism.state_feature_values>` and `control_allocation
        <ControlMechanism.control_allocation>`). It is assigned as the `objective_function
        <OptimizationFunction.objective_function>` parameter of `function
        <OptimizationControlMechanism.function>`, and calls the `evaluate` method of the OptimizationControlMechanism's
        `agent_rep <OptimizationControlMechanism.agent_rep>` with the current `state_feature_values
        <OptimizationControlMechanism.state_feature_values>` and a specified `control_allocation
        <ControlMechanism.control_allocation>`, which runs of the `agent_rep
        <OptimizationControlMechanism.agent_rep>` for `num_trials_per_estimate
        <OptimizationControlMechanism.num_trials_per_estimate>` trials. It returns an array containing the
        `net_outcome <ControlMechanism.net_outcome>` of the run and, if the **return_results** argument is True,
        an array containing the `results <Composition.results>` of the run. This method is `num_estimates
        <OptimizationControlMechanism>` times by the OptimizationControlMechanism's `function
        <OptimizationControlMechanism.function>`, which aggregates the `net_outcome <ControlMechanism.net_outcome>`
        over those in evaluating a given `control_allocation <ControlMechanism.control_allocation>`
        (see `OptimizationControlMechanism_Function` for additional details).

    search_function : function or method
        `search_function <OptimizationFunction.search_function>` assigned to `function
        <OptimizationControlMechanism.function>`; used to select samples of `control_allocation
        <ControlMechanism.control_allocation>` to evaluate by `evaluate_agent_rep
        <OptimizationControlMechanism.evaluate_agent_rep>`.

    search_termination_function : function or method
        `search_termination_function <OptimizationFunction.search_termination_function>` assigned to
        `function <OptimizationControlMechanism.function>`;  determines when to terminate the
        `optimization process <OptimizationFunction_Procedure>`.

    control_signals : ContentAddressableList[ControlSignal]
        list of the `ControlSignals <ControlSignal>` for the OptimizationControlMechanism for the Parameters being
        optimized by the OptimizationControlMechanism, including any inherited from the `Composition` for which it is
        the `controller <Composition.controller>` (this is the same as ControlMechanism's `output_ports
        <Mechanism_Base.output_ports>` attribute). Each sends a `ControlProjection` to the `ParameterPort` for the
        Parameter it controls when evaluating a `control_allocation <ControlMechanism.control_allocation>`. If
        `num_estimates <OptimizationControlMechanism.num_estimates>` is specified (that is, it is not None), a
        `ControlSignal` is added to control_signals, named *RANDOMIZATION_CONTROL_SIGNAL*, that is used to randomize
        estimates of `outcome <ControlMechanism.outcome>` for a given `control_allocation
        <ControlMechanism.control_allocation>` (see `OptimizationControlMechanism_Estimation_Randomization` for
        details.)

    control_allocation_search_space : list of SampleIterators
        `search_space <OptimizationFunction.search_space>` assigned by default to the
        OptimizationControlMechanism's `function <OptimizationControlMechanism.function>`, that determines the
        samples of `control_allocation <ControlMechanism.control_allocation>` evaluated by the `evaluate_agent_rep
        <OptimizationControlMechanism.evaluate_agent_rep>` method.  This is a property that, unless overridden,
        returns a list of the `SampleIterators <SampleIterator>` generated from the `allocation_samples
        <ControlSignal.allocation_samples>` specifications for each of the OptimizationControlMechanism's
        `control_signals <OptimizationControlMechanism.control_signals>`, and includes the
        *RANDOMIZATION_CONTROL_SIGNAL* used to randomize estimates of each `control_allocation
        <ControlMechanism.control_allocation>` (see `note <OptimizationControlMechanism_Randomization_Control_Signal>`
        above).

    saved_samples : list
        contains all values of `control_allocation <ControlMechanism.control_allocation>` sampled by `function
        <OptimizationControlMechanism.function>` if its `save_samples <OptimizationFunction.save_samples>` parameter
        is `True`;  otherwise list is empty.

    saved_values : list
        contains values of `net_outcome <ControlMechanism.net_outcome>` associated with all samples of
        `control_allocation <ControlMechanism.control_allocation>` evaluated by by `function
        <OptimizationControlMechanism.function>` if its `save_values <OptimizationFunction.save_samples>` parameter
        is `True`;  otherwise list is empty.

    search_statefulness : bool : True
        if True (the default), calls to `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>`
        by the OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` for each
        `control_allocation <ControlMechanism.control_allocation>` will run as simulations in their own
        `execution contexts <Composition_Execution_Context>`.  If *search_statefulness* is False, calls for each
        `control_allocation <ControlMechanism.control_allocation>` will not be executed as independent simulations;
        rather, all will be run in the same (original) execution context.
    """

    componentType = OPTIMIZATION_CONTROL_MECHANISM
    # initMethod = INIT_FULL_EXECUTE_METHOD
    # initMethod = INIT_EXECUTE_METHOD_ONLY

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # classPreferenceLevel = PreferenceLevel.TYPE
    # Any preferences specified below will override those specified in TYPE_DEFAULT_PREFERENCES
    # Note: only need to specify setting;  level will be assigned to Type automatically
    # classPreferences = {
    #     PREFERENCE_SET_NAME: 'DefaultControlMechanismCustomClassPreferences',
    #     PREFERENCE_KEYWORD<pref>: <setting>...}

    class Parameters(ControlMechanism.Parameters):
        """
            Attributes
            ----------

                agent_rep
                    see `agent_rep <OptimizationControlMechanism_Agent_Rep>`

                    :default value: None
                    :type:

                comp_execution_mode
                    see `comp_execution_mode <OptimizationControlMechanism.comp_execution_mode>`

                    :default value: `PYTHON`
                    :type: ``str``

                control_allocation_search_space
                    see `control_allocation_search_space <OptimizationControlMechanism.control_allocation_search_space>`

                    :default value: None
                    :type:

                function
                    see `function <OptimizationControlMechanism_Function>`

                    :default value: None
                    :type:

                input_ports
                    see `input_ports <Mechanism_Base.input_ports>`

                    :default value: ["{name: OUTCOME, params: {internal_only: True}}"]
                    :type: ``list``
                    :read only: True

                num_estimates
                    see `num_estimates <OptimizationControlMechanism.num_estimates>`

                    :default value: None
                    :type:

                num_trials_per_estimate
                    see `num_trials_per_estimate <OptimizationControlMechanism.num_trials_per_estimate>`

                    :default value: None
                    :type:

                outcome_input_ports_option
                    see `outcome_input_ports_option <OptimizationControlMechanism.outcome_input_ports_option>`

                    :default value: None
                    :type:

                saved_samples
                    see `saved_samples <OptimizationControlMechanism.saved_samples>`

                    :default value: None
                    :type:

                saved_values
                    see `saved_values <OptimizationControlMechanism.saved_values>`

                    :default value: None
                    :type:

                search_function
                    see `search_function <OptimizationControlMechanism.search_function>`

                    :default value: None
                    :type:

                search_statefulness
                    see `search_statefulness <OptimizationControlMechanism.search_statefulness>`

                    :default value: True
                    :type: ``bool``

                search_termination_function
                    see `search_termination_function <OptimizationControlMechanism.search_termination_function>`

                    :default value: None
                    :type:

                state_features
                    see `state_features <Optimization.state_features>`

                    :default value: None
                    :type: ``dict``

                state_feature_function
                    see `state_feature_function <OptimizationControlMechanism_Feature_Function>`

                    :default value: None
                    :type:

                state_input_ports
                    see `state_input_ports <OptimizationControlMechanism.state_input_ports>`

                    :default value: None
                    :type:  ``list``
        """
        outcome_input_ports_option = Parameter(CONCATENATE, stateful=False, loggable=False, structural=True)
        state_input_ports = Parameter(None, reference=True, stateful=False, loggable=False, read_only=True)
        state_feature_specs = Parameter(None, stateful=False, loggable=False, read_only=True,
                                        structural=True, parse_spec=True)
        state_feature_function = Parameter(None, reference=True, stateful=False, loggable=False)
        function = Parameter(GridSearch, stateful=False, loggable=False)
        search_function = Parameter(None, stateful=False, loggable=False)
        search_space = Parameter(None, read_only=True)
        search_termination_function = Parameter(None, stateful=False, loggable=False)
        comp_execution_mode = Parameter('Python', stateful=False, loggable=False, pnl_internal=True)
        search_statefulness = Parameter(True, stateful=False, loggable=False)

        agent_rep = Parameter(None, stateful=False, loggable=False, pnl_internal=True, structural=True)

        state_feature_values = Parameter(None, getter=_state_feature_values_getter, user=False, pnl_internal=True)

        # FIX: Should any of these be stateful?
        random_variables = ALL
        initial_seed = None
        same_seed_for_all_allocations = False
        num_estimates = None
        num_trials_per_estimate = None

        # search_space = None
        control_allocation_search_space = Parameter(
            None,
            read_only=True,
            getter=_control_allocation_search_space_getter,
            dependencies='search_space'
        )

        saved_samples = None
        saved_values = None

        def _parse_state_feature_specs(self, state_features):
            # return (state_features if isinstance(state_features, (dict, set)) else convert_to_list(state_features))
            from psyneulink.core.compositions.composition import Composition
            return (state_features if (isinstance(state_features, set)
                                       or (isinstance(state_features, dict)
                                           and (any(isinstance(key, (Port, Mechanism, Composition))
                                                    for key in state_features)
                                                or SHADOW_INPUTS in state_features)))
                    else convert_to_list(state_features))

    @handle_external_context()
    @tc.typecheck
    def __init__(self,
                 agent_rep=None,
                 state_features: tc.optional(tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort))) =
                 None,
                 state_feature_function: tc.optional(tc.optional(tc.any(dict, is_function_type))) = None,
                 function=None,
                 num_estimates = None,
                 random_variables = None,
                 initial_seed=None,
                 same_seed_for_all_allocations=None,
                 num_trials_per_estimate = None,
                 search_function: tc.optional(tc.optional(tc.any(is_function_type))) = None,
                 search_termination_function: tc.optional(tc.optional(tc.any(is_function_type))) = None,
                 search_statefulness=None,
                 context=None,
                 **kwargs):
        """Implement OptimizationControlMechanism"""

        from psyneulink.core.compositions.composition import Composition

        # Legacy warnings and conversions
        for k in kwargs.copy():
            if k == 'features':
                if state_features:
                    warnings.warn(f"Both 'features' and '{STATE_FEATURES}' were specified in the constructor "
                                  f"for an {self.__class__.__name__}. Note: 'features' has been deprecated; "
                                  f"'{STATE_FEATURES}' ({state_features}) will be used.")
                else:
                    warnings.warn(f"'features' was specified in the constructor for an {self.__class__.__name__}; "
                                  f"Note: 'features' has been deprecated; please use '{STATE_FEATURES}' in the future.")
                    state_features = kwargs['features']
                kwargs.pop('features')
                continue
            if k == 'feature_function':
                if state_feature_function:
                    warnings.warn(f"Both 'feature_function' and 'state_feature_function' were specified in the "
                                  f"constructor for an {self.__class__.__name__}. Note: 'feature_function' has been "
                                  f"deprecated; 'state_feature_function' ({state_feature_function}) will be used.")
                else:
                    warnings.warn(f"'feature_function' was specified in the constructor for an"
                                  f"{self.__class__.__name__}; Note: 'feature_function' has been deprecated; "
                                  f"please use 'state_feature_function' in the future.")
                    state_feature_function = kwargs['feature_function']
                kwargs.pop('feature_function')
                continue

        function = function or GridSearch

        # If agent_rep hasn't been specified, put into deferred init
        if agent_rep is None:
            if context.source==ContextFlags.COMMAND_LINE:
                # Temporarily name InputPort
                self._assign_deferred_init_name(self.__class__.__name__)
                # Store args for deferred initialization
                self._store_deferred_init_args(**locals())
                self._init_args['state_feature_specs'] = state_features

                # Flag for deferred initialization
                self.initialization_status = ContextFlags.DEFERRED_INIT
                return
            # If constructor is called internally (i.e., for controller of Composition),
            # agent_rep needs to be specified
            else:
                assert False, f"PROGRAM ERROR: 'agent_rep' arg should have been specified " \
                              f"in internal call to constructor for {self.name}."
        elif agent_rep.componentCategory=='Composition':
            from psyneulink.core.compositions.composition import NodeRole
            # If there are more state_features than INPUT Nodes in agent_rep, defer initialization until they are added
            if (state_features
                    and len(convert_to_list(state_features)) > len(agent_rep.get_nodes_by_role(NodeRole.INPUT))):
                # Temporarily name InputPort
                self._assign_deferred_init_name(self.__class__.__name__)
                # Store args for deferred initialization
                self._store_deferred_init_args(**locals())
                self._init_args['state_feature_specs'] = state_features

                # Flag for deferred initialization
                self.initialization_status = ContextFlags.DEFERRED_INIT
                return

        super().__init__(
            agent_rep=agent_rep,
            state_feature_specs=state_features,
            state_feature_function=state_feature_function,
            function=function,
            num_estimates=num_estimates,
            num_trials_per_estimate = num_trials_per_estimate,
            random_variables=random_variables,
            initial_seed=initial_seed,
            same_seed_for_all_allocations=same_seed_for_all_allocations,
            search_statefulness=search_statefulness,
            search_function=search_function,
            search_termination_function=search_termination_function,
            **kwargs
        )

    def _validate_params(self, request_set, target_set=None, context=None):
        """Insure that specification of ObjectiveMechanism has projections to it"""

        super()._validate_params(request_set=request_set, target_set=target_set, context=context)

        from psyneulink.core.compositions.composition import Composition
        if request_set[AGENT_REP] is None:
            raise OptimizationControlMechanismError(f"The '{AGENT_REP}' arg of an {self.__class__.__name__} must "
                                                    f"be specified and be a {Composition.__name__}")

        elif not (isinstance(request_set[AGENT_REP], Composition)
                  or (isinstance(request_set[AGENT_REP], type) and issubclass(request_set[AGENT_REP], Composition))):
            raise OptimizationControlMechanismError(f"The '{AGENT_REP}' arg of an {self.__class__.__name__} "
                                                    f"must be either a {Composition.__name__} or a sublcass of one")

        elif request_set[STATE_FEATURE_FUNCTION]:
            state_feats = request_set.pop(STATE_FEATURES, None)
            state_feat_fcts = request_set.pop(STATE_FEATURE_FUNCTION, None)
            # If no or only one item is specified in state_features, only one state_function is allowed
            if ((not state_feats or len(convert_to_list(state_feats))==1)
                    and len(convert_to_list(state_feat_fcts))!=1):
                raise OptimizationControlMechanismError(f"Only one function is allowed to be specified for "
                                                        f"the '{STATE_FEATURE_FUNCTION}' arg of {self.name} "
                                                        f"if either no only one items is specified for its "
                                                        f"'{STATE_FEATURES}' arg.")
            if len(convert_to_list(state_feat_fcts))>1 and not isinstance(state_feat_fcts, dict):
                raise OptimizationControlMechanismError(f"The '{STATE_FEATURES}' arg of {self.name} contains more "
                                                        f"than one item, so its '{STATE_FEATURE_FUNCTION}' arg "
                                                        f"must be either only a single function (applied to all "
                                                        f"{STATE_FEATURES}) or a dict with entries of the form "
                                                        f"<state_feature>:<function>.")
            if len(convert_to_list(state_feat_fcts))>1:
                invalid_fct_specs = [fct_spec for fct_spec in state_feat_fcts if fct_spec not in state_feats]
                if invalid_fct_specs:
                    raise OptimizationControlMechanismError(f"The following entries of the dict specified for "
                                                            f"'{STATE_FEATURE_FUNCTION} of {self.name} have keys that "
                                                            f"do not match any InputPorts specified in its "
                                                            f"{STATE_FEATURES} arg: {invalid_fct_specs}.")


        if self.random_variables is not ALL:
            invalid_params = [param.name for param in self.random_variables
                              if param not in self.agent_rep.random_variables]
            if invalid_params:
                raise OptimizationControlMechanismError(f"The following Parameters were specified for the "
                                                        f"{RANDOM_VARIABLES} arg of {self.name} that are do"
                                                        f"randomizable (i.e., they do not have a 'seed' attribute: "
                                                        f"{invalid_params}.")

    # FIX: CONSIDER GETTING RID OF THIS METHOD ENTIRELY, AND LETTING state_input_ports
    #      BE HANDLED ENTIRELY BY _update_state_input_ports_for_controller
    def _instantiate_input_ports(self, context=None):
        """Instantiate InputPorts for state_features (with state_feature_function if specified).

        This instantiates the OptimizationControlMechanism's `state_input_ports;
             these are used to provide input to the agent_rep when its evaluate method is called
             (see Composition._build_predicted_inputs_dict).
        The OptimizationControlMechanism's outcome_input_ports are instantiated by
            ControlMechanism._instantiate_input_ports in the call to super().

        InputPorts are constructed for **state_features** by calling _parse_state_feature_specs
            with them and **state_feature_function** arguments of the OptimizationControlMechanism constructor.
        The constructed state_input_ports  are passed to ControlMechanism_instantiate_input_ports(),
             which appends them to the InputPort(s) that receive input from the **objective_mechanism* (if specified)
             or **monitor_for_control** ports (if **objective_mechanism** is not specified).
        Also ensures that:
             - every state_input_port has only a single Projection;
             - every outcome_input_ports receive Projections from within the agent_rep if it is a Composition.

        If no **state_features** are specified in the constructor, assign ones for INPUT Nodes of owner.
          - warn for use of CompositionFunctionApproximator as agent_rep;
          - ignore here for Composition as agent_rep
            (handled in _update_state_input_ports_for_controller).

        See `state_features <OptimizationControlMechanism_State_Features_Arg>` and
        `OptimizationControlMechanism_State_Features` for additional details.
        """

        # If any state_features were specified parse them and pass to ControlMechanism._instantiate_input_ports()
        state_input_ports_specs = None

        # FIX: 11/3/21 :
        #    ADD CHECK IN _parse_state_feature_specs THAT IF A NODE RATHER THAN InputPort IS SPECIFIED,
        #    ITS PRIMARY IS USED (SEE SCRATCH PAD FOR EXAMPLES)
        if not self.state_feature_specs:
            # If agent_rep is CompositionFunctionApproximator, warn if no state_features specified.
            # Note: if agent rep is Composition, state_input_ports and any state_feature_function specified
            #       are assigned in _update_state_input_ports_for_controller.
            if self.agent_rep_type == COMPOSITION_FUNCTION_APPROXIMATOR:
                warnings.warn(f"No '{STATE_FEATURES}' specified for use with `agent_rep' of {self.name}")

        else:
            # Implement any specified state_features
            state_input_ports_specs = self._parse_state_feature_specs()
            # Note:
            #   if state_features were specified and agent_rep is a CompositionFunctionApproximator,
            #   assume they are OK (no way to check their validity for agent_rep.evaluate() method, and skip assignment

        # Pass state_input_ports_sepcs to ControlMechanism for instantiation and addition to OCM's input_ports
        super()._instantiate_input_ports(state_input_ports_specs, context=context)

        # Assign to self.state_input_ports attribute
        start = self.num_outcome_input_ports # FIX: 11/3/21 NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
        stop = start + len(state_input_ports_specs) if state_input_ports_specs else 0
        # FIX 11/3/21: THIS SHOULD BE MADE A PARAMETER
        self.parameters.state_input_ports.set(ContentAddressableList(component_type=InputPort,
                                                                     list=self.input_ports[start:stop]),
                                              override=True)

        # Ensure that every state_input_port has no more than one afferent projection
        # FIX: NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
        for i in range(self.num_outcome_input_ports, self.num_state_input_ports):
            port = self.input_ports[i]
            if len(port.path_afferents) > 1:
                raise OptimizationControlMechanismError(f"Invalid {type(port).__name__} on {self.name}. "
                                                        f"{port.name} should receive exactly one projection, "
                                                        f"but it receives {len(port.path_afferents)} projections.")

    def _get_agent_rep_input_receivers(self, comp=None, type=PORT, comp_as_node=False):
        if not self.agent_rep_type or self.agent_rep_type == COMPOSITION_FUNCTION_APPROXIMATOR:
            return [None]
        comp = comp or self.agent_rep
        return comp._get_input_receivers(comp=comp, type=type, comp_as_node=comp_as_node)

    def _get_specs_not_in_agent_rep(self, state_feature_specs):
        from psyneulink.core.compositions.composition import Composition
        agent_rep_nodes = self.agent_rep._get_all_nodes()
        return [spec for spec in state_feature_specs
                if ((isinstance(spec, (Mechanism, Composition))
                     and spec not in agent_rep_nodes)
                    or (isinstance(spec, Port)
                        and spec.owner not in agent_rep_nodes))]

    def _validate_input_nodes(self, nodes, enforce=None):
        """Check that nodes are INPUT Nodes of agent_rep
        INPUT Nodes are those at the top level of agent_rep as well as those of any Compositions nested within it
            that are themselves INPUT Nodes of their enclosing Composition.
        Raise exception for non-INPUT Nodes if **enforce** is specified; otherwise just issue warning.
        """
        from psyneulink.core.compositions.composition import Composition
        agent_rep_input_nodes = self._get_agent_rep_input_receivers(type=NODE,comp_as_node=ALL)
        agent_rep_input_ports = self._get_agent_rep_input_receivers(type=PORT)
        agent_rep_all_nodes = self.agent_rep._get_all_nodes()
        non_input_node_specs = [node for node in nodes
                                # if node not in self._get_agent_rep_input_receivers(type=NODE, comp_as_node=ALL)]
                                if ((isinstance(node, (Mechanism, Composition)) and node not in agent_rep_input_nodes)
                                    or (isinstance(node, Port) and (not isinstance(node, InputPort)
                                                                    or node not in agent_rep_input_ports)))]
        non_agent_rep_node_specs = [node for node in nodes
                                    if ((isinstance(node, (Mechanism, Composition)) and node not in agent_rep_all_nodes) or
                                        (isinstance(node, Port) and node.owner not in agent_rep_all_nodes))]

        # Deal with Nodes that are in agent_rep but not INPUT Nodes
        if non_input_node_specs:
            items = ', '.join([n._name for n in non_input_node_specs])
            if len(non_input_node_specs) == 1:
                items_str = f"contains an item ({items}) that is not an INPUT Node"
            else:
                items_str = f"contains items ({items}) that are not INPUT Nodes"
            message = f"The '{STATE_FEATURES}' specified for '{self.name}' {items_str} " \
                      f"within its {AGENT_REP} ('{self.agent_rep.name}'); only INPUT Nodes can be in a set " \
                      f"or used as keys in a dict used to specify '{STATE_FEATURES}'."
            if enforce:
                raise OptimizationControlMechanismError(message)
            else:
                warnings.warn(message)

        # Deal with Nodes that are not in agent_rep
        if non_agent_rep_node_specs:
            items = ', '.join([n._name for n in non_agent_rep_node_specs])
            singular = len(non_agent_rep_node_specs) == 1
            if singular:
                items_str = f"contains an item ({items}) that is"
            else:
                items_str = f"contains items ({items}) that are"
            message = f"The '{STATE_FEATURES}' specified for '{self.name}' {items_str} not in its {AGENT_REP} " \
                      f"('{self.agent_rep.name}'). Executing '{self.agent_rep.name}' before " \
                      f"{'they are' if singular else 'it is'} added will generate an error ."
            if enforce:
                raise OptimizationControlMechanismError(message)
            else:
                warnings.warn(message)

    # FIX: 1/29/22 - REFACTOR TO SUPPORT TUPLE AND InportPort SPECIFICATION DICT FOR MULT. PROJS. TO STATE_INPUT_PORT
    # FIX: 2/25/22 - REFACTOR TO SUPPORT InputPort SPECIFICATIONS IN dict, set and list specs
    def _parse_state_feature_specs(self, context=None):
        """Parse entries of state_features specifications used to construct state_input_ports.

        Called from _instantiate_input_ports()

        Parse **state_features** arg of constructor for OptimizationControlMechanism, assigned to state_feature_specs.

        state_feature_specs lists sources of inputs to *all* INPUT Nodes of agent_rep, at all levels of nesting; there
          is one entry for every INPUT Node in agent_rep, and every INPUT Node of any nested Composition that is
          itself an INPUT Node at any level of nesting.
            
        Construct a state_input_port for every entry in state_feature_specs that is not None:
          the value of those state_input_ports comprise the state_feature_values attribute, and are provided as the
              input to the INPUT Nodes of agent_rep when its evaluate() method is executed (as the **predicted_inputs**
              argument if agent_rep is a Composition, and the **feature_values** argument if it is a 
              CompositionFunctionApproximator); for INPUT;
          for None entries in state_feature_specs, the corresponding INPUT Nodes are provided their
              default_external_input_shape as their input when agent_rep.evaluate() executes.

        Projection(s) to state_input_ports from sources specified in state_feature_specs can be direct,
            or indirect by way of a CIM if the source is in a nested Composition.
        
        Handle four formats:

        - dict {INPUT Node: source or None, INPUT Node: source or None...}:
            - every key must be an INPUT Node of agent_rep or an INPUT Node of a nested Composition within it that 
                is itself an INPUT Node of its enclosing Composition, at any level of nesting; 
            - if a Composition is specified as a key, construct a state_input_port for each InputPort of each of its
                INPUT Nodes, and those of any Compositions nested within it at all levels of nesting, and assign the
                the value of the dict entry as the source for all of them;
            - for INPUT Nodes not specified or assigned None as their value, assign corresponding entries in
                state_feature_specs as None, and don't construct a state_input_port for them;
            - if only one or some of the INPUT Nodes of a nested Composition are specified,
                for the remaining ones assign the corresponding entries in state_feature_specs as None,
                and don't construct state_input_ports for them;
        - list [source, None, source...]: specifies source specs for INPUT Nodes
            - must be listed in same order as *expanded* list of agent_rep INPUT Nodes to which they correspond (i.e.,
                nested Compositions that are INPUT Nodes replaced by their INPUT Nodes, for all levels of nesting);
            - if there are fewer sources listed than INPUT Nodes, assign None to the entries in state_feature_specs
               corresponding to the remaining INPUT Nodes and don't construct state_input_ports for them;
            - if there more sources listed than INPUT Nodes, leave the excess ones, and label them as
               'EXPECTED INPUT NODE n' for later resolution (see below). 

        - set {INPUT Node, Input Node...}: specifies INPUT Nodes to be shadowed
            - every item must be an INPUT Node of agent_rep or an INPUT Node of a nested Composition within it that 
                is itself an INPUT Node of its enclosing Composition, at any level of nesting; 
            - if a Composition is specified, construct a state_input_port for each of its INPUT Nodes, and those of 
                any Compositions nested within it at all levels of nesting, each of which shadows the input of the
                corresponding INPUT Node (see _InputPort_Shadow_Inputs).
            - if only one or some of the INPUT Nodes of a nested Composition are specified, don't state_input_ports
                for the remaining ones, and assign them their default inputs in evaluate().   

        IMPLEMENTATION NOTE: this is a legacy format for consistency with generic specification of shadowing inputs
        - SHADOW_INPUTS dict {"SHADOW_INPUTS":[shadowable input, None, shadowable input...]}:
            - all items must be a Mechanism (or one of its InputPorts) that is an INPUT Node of agent_rep or
                 of a nested Composition within it that is itself an INPUT Node; 
            - must be listed in same order as *expanded* list of agent_rep INPUT Nodes to which they correspond
                (see list format above);
            - construct a state_input_port for each non-None spec, and assign it a Projection that shadows the spec.
                (see _InputPort_Shadow_Inputs).

        If shadowing is specified for an INPUT Node, set INTERNAL_ONLY to True in entry of params dict in
            specification dictionary for corresponding state_input_port (so that inputs to Composition are not 
            required if the specified source is itself an INPUT Node).
            
        If an INPUT Node is specified that is not (yet) in agent_rep, and/or a source is specified that is not yet
            in self.composition, warn and defer creating a state_input_port;  final check is made, and error(s)
            generated for unresolved specifications at run time.

        Assign functions specified in **state_feature_function** to InputPorts for all state_features

        Return list of InputPort specification dictionaries for state_input_ports
        """

        from psyneulink.core.compositions.composition import Composition, NodeRole
        # Agent rep's input Nodes and their names
        agent_rep_input_ports = self._get_agent_rep_input_receivers(type=PORT)
        self._specified_input_ports_in_order = []

        # List of assigned state_feature_function (vs. user provided specs)
        self._state_feature_functions = []

        # VALIDATION AND WARNINGS -----------------------------------------------------------------------------------

        # Only list spec allowed if agent_rep is a CompositionFunctionApproximator
        if self.agent_rep_type == COMPOSITION_FUNCTION_APPROXIMATOR and not isinstance(self.state_feature_specs, list):
            agent_rep_name = f" ({self.agent_rep.name})" if not isinstance(self.agent_rep, type) else ''
            raise OptimizationControlMechanismError(
                f"The {AGENT_REP} specified for {self.name}{agent_rep_name} is a {COMPOSITION_FUNCTION_APPROXIMATOR}, "
                f"so its '{STATE_FEATURES}' argument must be a list, not a {type(self.state_feature_specs).__name__} "
                f"({self.state_feature_specs}).")

        # agent_rep has not yet been (fully) constructed
        if not agent_rep_input_ports and self.agent_rep_type is COMPOSITION:
            if (isinstance(self.state_feature_specs, set)
                    or isinstance(self.state_feature_specs, dict) and SHADOW_INPUTS not in self.state_feature_specs):
                # Dict and set specs reference Nodes of agent_rep, and so must that must be constructed first
                raise OptimizationControlMechanismError(
                    f"The '{STATE_FEATURES}' arg for {self.name} has been assigned a dict or set specification "
                    f"before any Nodes have been assigned to its {AGENT_REP} ('{self.agent_rep.name}').  Either"
                    f"those should be assigned before construction of {self.name}, or a list specification "
                    f"should be used (though that is not advised).")
            else:
                # List and SHADOW_INPUTS specs are dangerous before agent_rep has been fully constructed
                warnings.warn(f"The {STATE_FEATURES}' arg for {self.name} has been specified before any Nodes have "
                              f"been assigned to its {AGENT_REP} ('{self.agent_rep.name}').  Their order must be the "
                              f"same as the order of the corresponding INPUT Nodes for '{self.agent_rep.name}' once "
                              f"they are added, or unexpected results may occur.  It is safer to assign all Nodes to "
                              f"the {AGENT_REP} of a controller before specifying its '{STATE_FEATURES}'.")

        # HELPER METHODS ------------------------------------------------------------------------------------------

        def expand_nested_input_comp_to_input_nodes(comp):
            input_nodes = []
            for node in comp.get_nodes_by_role(NodeRole.INPUT):
                if isinstance(node, Composition):
                    input_nodes.extend(expand_nested_input_comp_to_input_nodes(node))
                else:
                    input_nodes.append(node)
            return input_nodes

        # FIX: 3/4/22 - DELETE IF NOT USED:
        # def get_inputs_for_nested_comp(comp):
        #     # FIX: 1/18/22 - NEEDS TO BE MODIFIED TO RETURN TUPLE IF > INPUT NODE, ONCE THAT CAN BE HANDLED BY LIST SPEC
        #     return comp.get_nodes_by_role(NodeRole.INPUT)

        # PARSE SPECS  ------------------------------------------------------------------------------------------
        # Generate parallel lists of INPUT Nodes and corresponding feature specs (for sources of inputs)

        def _parse_specs(state_feature_specs, spec_str="list"):
            """Validate and parse INPUT Node specs assigned to construct state_feature_specs
            Validate number and identity of specs relative to agent_rep INPUT Nodes.
            Assign spec for every INPUT Mechanism (nested) within agent_rep (i.e., for all nested Compositions)
                as entries in state_feature_specs
            Return names for use as input_port_names in main body of method
            """
            parsed_feature_specs = []

            if self.agent_rep_type == COMPOSITION:

                # FIX: 3/4/22 - THESE SEEM DUPLICATIVE OF _validate_state_features;  JUST CALL THAT HERE?
                #               ALSO, "fewer" WARNING IS TRIGGERED IF MECHANIMS RATHER THAN ITS INPUT_PORTS ARE SPEC'D
                # Too FEW specs for number of agent_rep receivers
                if len(self.state_feature_specs) < len(agent_rep_input_ports):
                    warnings.warn(f"There are fewer '{STATE_FEATURES}' specified for '{self.name}' than the number "
                                  f"of {InputPort.__name__}'s for all of the INPUT Nodes of its {AGENT_REP} "
                                  f"('{self.agent_rep.name}'); the remaining inputs will be assigned default values "
                                  f"when '{self.agent_rep.name}`s 'evaluate' method is executed. If this is not the "
                                  f"desired behavior, use its get_inputs_format() method to see the format for its "
                                  f"inputs.")

                # Too MANY specs for number of agent_rep receivers
                if len(state_feature_specs) > len(agent_rep_input_ports):
                    # specs_not_in_agent_rep = [f"'{spec.name if isinstance(spec, Mechanism) else spec.owner.name}'"
                    #                           for spec in self._get_specs_not_in_agent_rep(state_feature_specs)]
                    specs_not_in_agent_rep = [f"'{spec.name if isinstance(spec,(Mechanism, Composition)) else spec.owner.name}'"
                                              for spec in self._get_specs_not_in_agent_rep(user_specs)]

                    if specs_not_in_agent_rep:
                        spec_str = ", ".join(specs_not_in_agent_rep)
                        warnings.warn(
                            f"The '{STATE_FEATURES}' specified for {self.name} is associated with a number of "
                            f"{InputPort.__name__}s ({len(state_feature_specs)}) that is greater than for the "
                            f"{InputPort.__name__}s of the INPUT Nodes ({len(agent_rep_input_ports)}) for the "
                            f"Composition assigned as its {AGENT_REP} ('{self.agent_rep.name}'), which includes "
                            f"the following that are not (yet) in '{self.agent_rep.name}': {spec_str}. Executing "
                            f"{self.name} before the additional item(s) are added as (part of) INPUT Nodes will "
                            f"generate an error.")
                    else:
                        warnings.warn(
                            f"The '{STATE_FEATURES}' specified for {self.name} is associated with a number of "
                            f"{InputPort.__name__}s ({len(state_feature_specs)}) that is greater than for the "
                            f"{InputPort.__name__}s of the INPUT Nodes ({len(agent_rep_input_ports)}) for the "
                            f"Composition assigned as its {AGENT_REP} ('{self.agent_rep.name}'). Executing "
                            f"{self.name} before the additional item(s) are added as (part of) INPUT Nodes will "
                            f"generate an error.")

            # Nested Compositions not allowed to be specified in a list spec
            nested_comps = [node for node in state_feature_specs if isinstance(node, Composition)]
            if nested_comps:
                comp_names = ", ".join([f"'{n.name}'" for n in nested_comps])
                raise OptimizationControlMechanismError(
                    f"The '{STATE_FEATURES}' argument for '{self.name}' includes one or more Compositions "
                    f"({comp_names}) in the {spec_str} specified for its '{STATE_FEATURES}' argument; these must be "
                    f"replaced by direct references to the Mechanisms (or their InputPorts) within them to be "
                    f"shadowed.")
            spec_names = []
            num_specs = len(state_feature_specs)
            num_ports = len(agent_rep_input_ports)
            self._num_state_feature_specs = max(num_specs, num_ports)
            for i in range(self._num_state_feature_specs):
                # PORT & PORT_NAME
                # (and specs for CFA and any nodes not yet in agent_rep)
                spec_name = None
                state_feature_fct = None
                if i < num_ports:
                    # Node should be in agent_rep, so use that to be sure
                    if self.agent_rep_type == COMPOSITION:
                        node = agent_rep_input_ports[i]
                    # Assign spec as node for CompositionFunctionApproximator
                    else:
                        spec = state_feature_specs[i]
                        node = spec if isinstance(spec, (Mechanism, Composition)) else spec.owner
                    node_name = node.full_name
                else:
                    # Node not (yet) in agent_rep, so "DEFERRED n" as node name
                    spec = state_feature_specs[i]
                    node = None
                    node_name = f'DEFFERED {str(i-num_ports)}'
                # SPEC
                # Assign specs
                # Only process specs for which there are already INPUT Nodes in agent_rep
                #     (others may be added to Composition later)
                if i < num_specs:
                    spec = state_feature_specs[i]
                    # Assign input_port name
                    if is_numeric(spec):
                        spec_name = f"{node_name} {DEFAULT_VARIABLE.upper()}"
                    elif isinstance(spec, (Port, Mechanism, Composition)):
                        if hasattr(spec, 'full_name'):
                            spec_name = spec.full_name
                        else:
                            spec_name = spec.name
                    elif isinstance(spec, dict):
                        spec_name = spec[NAME] if NAME in spec else f"STATE FEATURE INPUT for {node_name}"
                        if FUNCTION in spec:
                            state_feature_fct = spec[FUNCTION]
                        elif PARAMS in spec and FUNCTION in spec[PARAMS]:
                            state_feature_fct = spec[PARAMS][FUNCTION]
                    elif isinstance(spec, tuple):
                        state_feature_fct = spec[1]
                        spec = spec[0]
                    elif spec is not None:
                        assert False, f"PROGRAM ERROR: unrecognized form of state_feature specification for {self.name}"
                else:
                    # Fewer specifications than number of INPUT Nodes,
                    #  the remaining ones may be specified later, but for now assume they are meant to be ignored
                    spec = None

                parsed_feature_specs.append(spec)
                self._state_feature_functions.append(state_feature_fct)
                self._specified_input_ports_in_order.append(node)
                spec_names.append(spec_name)

            self.parameters.state_feature_specs.set(parsed_feature_specs, override=True)
            return spec_names or []

        user_specs = self.parameters.state_feature_specs.spec

        # LIST OR SHADOW_INPUTS DICT: source specs
        # Source specs but not INPUT Nodes specified; spec is either:
        # - list:  [spec, spec...]
        # - SHADOW_INPUTS dict (with list spec as its only entry): {SHADOW_INPUTS: {[spec, spec...]}}
        # Treat specs as sources of input to INPUT Nodes of agent_rep (in corresponding order):
        # Call _parse_specs to construct a regular dict using INPUT Nodes as keys and specs as values
        if isinstance(user_specs, list) or (isinstance(user_specs, dict) and SHADOW_INPUTS in user_specs):
            if isinstance(user_specs, list):
                specs = user_specs
                spec_str = 'list'
            else:
                # SHADOW_INPUTS spec:
                if isinstance(user_specs[SHADOW_INPUTS], set):
                    # Set not allowed as SHADOW_INPUTS spec; catch here to provide context-relevant error message
                    raise OptimizationControlMechanismError(
                        f"The '{STATE_FEATURES}' argument for '{self.name}' uses a set in a '{SHADOW_INPUTS.upper()}' "
                        f"dict;  this must be a single item or list of specifications in the order of the INPUT Nodes"
                        f"of its '{AGENT_REP}' ({self.agent_rep.name}) to which they correspond." )
                # All specifications in list specified for SHADOW_INPUTS must be shadowable
                #     (i.e., either an INPUT Node or the InputPort of one) or None
                # FIX: 3/4/22: NEED TO ACCOMODATE input_port IN SET OR AS KEY IN DICT HERE:
                bad_specs = [spec for spec in user_specs[SHADOW_INPUTS]
                             if ((spec.owner not in self._get_agent_rep_input_receivers(type=NODE, comp_as_node=ALL)
                                  if isinstance(spec, InputPort)
                                  else spec not in self._get_agent_rep_input_receivers(type=NODE, comp_as_node=ALL))
                                 and spec is not None)]
                if bad_specs:
                    bad_spec_names = [f"'{item.owner.name}'" if hasattr(item, 'owner')
                                      else f"'{item.name}'" for item in bad_specs]
                    raise OptimizationControlMechanismError(
                        f"The '{STATE_FEATURES}' argument for '{self.name}' has one or more items in the list "
                        f"specified for '{SHADOW_INPUTS.upper()}' ({', '.join([name for name in bad_spec_names])}) "
                        f"that are not (part of) any INPUT Nodes of its '{AGENT_REP}' ('{self.agent_rep.name}')." )
                specs = user_specs[SHADOW_INPUTS]
                spec_str = f"{SHADOW_INPUTS.upper()} dict"
            input_port_names = _parse_specs(specs, spec_str=spec_str)

        # FIX: 2/25/22 - ?ITEMS IN set ARE SHADOWED, BUT UNSPECIFIED ITEMS IN SET AND DICT ARE ASSIGNED DEFAULT VALUES
        # SET OR DICT: specification by INPUT Nodes
        # INPUT Nodes of agent_rep specified in either a:
        # - set, without source specs: {node, node...}
        # - dict, with source specs for each: {node: spec, node: spec...}
        # Call _parse_specs to convert to or flesh out dict with INPUT Nodes as keys and any specs as values
        #  adding any unspecified INPUT Nodes of agent_rep and assiging default values for any unspecified values
        elif isinstance(user_specs, (set, dict)):

            # FIX: 3/4/22 REINSTATE & MOVE TO ABOVE??
            self._validate_input_nodes(set(user_specs))

            # MODIFIED 3/4/22 NEW:
            #  FIX: MOVE TO _validate_input_nodes
            # Validate user_specs
            internal_input_port_specs = [f"'{spec.full_name}'" for spec in user_specs
                                         if isinstance(spec, InputPort) and spec.internal_only]
            if internal_input_port_specs:
                raise OptimizationControlMechanismError(
                    f"The following {InputPort.__name__}s specified in the '{STATE_FEATURES}' arg for {self.name} "
                    f"do not receive any external inputs and thus cannot be assigned '{STATE_FEATURES}': "
                    f"{', '.join(internal_input_port_specs)}.")
            non_input_port_specs = [f"'{spec.full_name}'" for spec in user_specs
                                    if isinstance(spec, Port) and not isinstance(spec, InputPort)]
            if non_input_port_specs:
                raise OptimizationControlMechanismError(
                    f"The following {Port.__name__}s specified in the '{STATE_FEATURES}' arg for {self.name} "
                    f"are not {InputPort.__name__} and thus cannot be assigned '{STATE_FEATURES}': "
                    f"{', '.join(non_input_port_specs)}.")

            expanded_specified_ports = []
            expanded_dict_with_ports = {}
            dict_format = isinstance(user_specs, dict)

            # Expand any Nodes specified in user_specs set or keys of dict to corresponding InputPorts
            for spec in user_specs:
                # Expand any specified Compositions into corresponding InputPorts for all INPUT Nodes (including nested)
                if isinstance(spec, Composition):
                    ports = self._get_agent_rep_input_receivers(spec)
                # Expand any specified Mechanisms into corresponding InputPorts except any internal ones
                elif isinstance(spec, Mechanism):
                    ports = [input_port for input_port in spec.input_ports if not input_port.internal_only]
                else:
                    ports = [spec]
                expanded_specified_ports.extend(ports)
                if dict_format:
                    # Assign values specified in user_specs dict to corresponding InputPorts
                    expanded_dict_with_ports.update({port:user_specs[spec] for port in ports})

            # # Get specified ports in order of agent_rep INPUT Nodes, with None assigned to any unspecified INPUT Nodes
            all_specified_ports = [port if port in expanded_specified_ports else None for port in agent_rep_input_ports]
            # Get any not found anywhere (including nested) in agent_rep, which are placed at the end of list
            all_specified_ports.extend([port for port in expanded_specified_ports if port not in agent_rep_input_ports])

            if isinstance(user_specs, set):
                # Just pass ports;  _parse_specs assigns shadowing projections for them and None to all unspecified ones
                specs = all_specified_ports
            else:
                # Pass values from user_spec dict to be parsed;
                #    corresponding ports are safely in all_specified_ports
                #    unspecified ports are assigned None per requirements of list format
                specs = [expanded_dict_with_ports[port] if port in all_specified_ports else None
                         for port in agent_rep_input_ports]
            # MODIFIED 3/4/22 END

            input_port_names = _parse_specs(specs)

        # CONSTRUCT InputPort SPECS -----------------------------------------------------------------------------

        state_input_port_specs = []
        for i in range(self._num_state_feature_specs):
            # Note: state_feature_specs have now been parsed (user's specs are in parameters.state_feature_specs.spec);
            #       state_feature_specs correspond to all InputPorts of agent_rep's INPUT Nodes (including nested ones)
            spec = self.state_feature_specs[i]
            if spec is None:
                continue
            spec = _parse_shadow_inputs(self, spec)
            # If spec is numeric, assign as default value and InputPort function that simply returns that value
            if is_numeric(spec):
                spec_val = copy.copy(spec)
                spec = {VALUE: spec_val,
                        PARAMS: {DEFAULT_INPUT: DEFAULT_VARIABLE}
                }
            else:
                spec = spec[0] # _parse_shadow_inputs(self, spec) returns a list, even when passed a single item
            # If optimization uses Composition, assume that shadowing a Mechanism means shadowing its primary InputPort
            if isinstance(spec, Mechanism):
                if self.agent_rep_type == COMPOSITION:
                    # FIX: 11/29/21: MOVE THIS TO _parse_shadow_inputs
                    #      (ADD ARG TO THAT FOR DOING SO, OR RESTRICT TO InputPorts IN GENERAL)
                    if len(spec.input_ports)!=1:
                        raise OptimizationControlMechanismError(
                            f"A Mechanism ({spec.name}) is specified to be shadowed in the '{STATE_FEATURES}' arg "
                            f"for '{self.name}', but it has more than one {InputPort.__name__}; a specific one of its "
                            f"{InputPort.__name__}s must be specified to be shadowed.")
                    spec = spec.input_port
                else:
                    spec = spec.output_port
                # Update Mechanism spec with Port
                self.state_feature_specs[i] = spec
            if isinstance(spec, dict):
                # Note: clear any functions specified; will be assigned in _assign_state_feature_function
                if self._state_feature_functions[i]:
                    spec.pop(FUNCTION, None)
                    if PARAMS in spec:
                        spec[PARAMS].pop(FUNCTION, None)
            parsed_spec = _parse_port_spec(owner=self, port_type=InputPort, port_spec=spec)

            if not parsed_spec[NAME]:
                parsed_spec[NAME] = input_port_names[i]

            if parsed_spec[PARAMS] and SHADOW_INPUTS in parsed_spec[PARAMS]:
                # Composition._update_shadow_projections will take care of PROJECTIONS specification
                parsed_spec[PARAMS][INTERNAL_ONLY]=True,
                parsed_spec[PARAMS][PROJECTIONS]=None

            # Assign function for state_input_port if specified---------------------------------------------------
            parsed_spec = self._assign_state_feature_function(parsed_spec, i)

            parsed_spec = [parsed_spec] # so that extend works below
            state_input_port_specs.extend(parsed_spec)

        return state_input_port_specs

    def _assign_state_feature_function(self, specification_dict, idx=None):
        """Assign any specified state_feature_function to corresponding state_input_ports
        idx is index into self._state_feature_functions; if None, use self.state_feature_function specified by user
        Specification in InputPort specification dictionary or **state_features** tuple
            takes precedence over **state_feature_function** specification.
        Assignment of function to dict specs handled above, so skip here
        Return state_input_port_dicts with FUNCTION entries added as appropriate.
        """

        # Note: state_feature_function has been validated in _validate_params
        default_function = self.state_feature_function  # User specified in constructor
        try:
            if self._state_feature_functions:
                assert len(self._state_feature_functions) == self._num_state_feature_specs, \
                    f"PROGRAM ERROR: Length of _state_feature_functions for {self.name} should be same " \
                    f"as number of state_input_port_dicts passed to _assign_state_feature_function"
            state_feature_functions = self._state_feature_functions
        except AttributeError:
            # state_features assigned automatically in _update_state_input_ports_for_controller,
            #    so _state_feature_functions (for individual state_features) not created
            state_feature_functions = None
        fct = state_feature_functions[idx] if state_feature_functions else None
        if fct:
            specification_dict[FUNCTION] = self._parse_state_feature_function(fct)
        elif default_function and FUNCTION not in specification_dict[PARAMS]:
            # Assign **state_feature_function** (aka default_function) if specified and no other has been specified
            specification_dict[FUNCTION] = self._parse_state_feature_function(default_function)
        return specification_dict

    def _parse_state_feature_function(self, feature_function):
        if isinstance(feature_function, Function):
            return copy.deepcopy(feature_function)
        else:
            return feature_function

    def _update_state_features_dict(self):
        agent_rep_input_ports = self._get_agent_rep_input_receivers()
        specified_input_ports = self._specified_input_ports_in_order

        for i, port in enumerate(self.state_input_ports):
            # Get value (need first, to determine whether it belongs to a nested Comp, for assigning key)
            feature = self.state_feature_specs[i]
            # Get INPUT Node of agent_rep as key:
            if (isinstance(feature, Component) and
                    feature.owner in [n[0] for n in self.agent_rep._get_nested_nodes()]):
                node = feature.owner
            # # MODIFIED 3/4/22 OLD:
            # elif specified_input_nodes[i]:
            #     node = specified_input_nodes[i]
            # elif i < len(agent_rep_input_nodes):
            #     node = specified_input_nodes[i] = agent_rep_input_nodes[i]
            # MODIFIED 3/4/22 NEW:
            elif specified_input_ports[i]:
                node = specified_input_ports[i]
            elif i < len(agent_rep_input_ports):
                node = specified_input_ports[i] = agent_rep_input_ports[i]
            # MODIFIED 3/4/22 END
            else:
                node = None
            if not (isinstance(node, str) and 'DEFERRED' in node):
                continue
            if feature.owner not in self._get_agent_rep_input_receivers(comp_as_node=ALL):
                # Don't add to dict, will be dealt with or raise an error at run time
                continue
            self.state_feature_specs[i] = feature

    def _update_state_input_ports_for_controller(self, context=None):
        """Check and update state_input_ports for model-based optimization (agent_rep==Composition)

        If no agent_rep has been specified or it is a CompositionFunctionApproximator, return
            (note: validation of state_features specified for CompositionFunctionApproximator optimization
            is up to the CompositionFunctionApproximator)

        For agent_rep that is a Composition):

        - ensure that state_input_ports for all specified state_features are either for InputPorts of INPUT Nodes of
          agent_rep, or from Nodes of it or any nested Compositions;
          raise an error if any receive a Projection that is not a shadow Projection from an INPUT Node of agent_rep
          or from the output_port of a Node that is not somewhere in the agent_rep Composition.
          (note: there should already be state_input_ports for any **state_features** specified in the constructor).

        - if no state_features specified, assign a state_input_port for every InputPort of every INPUT Node of agent_rep
          (note: shadow Projections for all state_input_ports are created in Composition._update_shadow_projections()).

        - assign state_feature_function to relevant state_input_ports (same function for all if no state_features
          are specified or only one state_function is specified;  otherwise, use dict for specifications).

        Return True if successful, None if not performed.
        """

        # Don't instantiate unless being called by Composition.run()
        # This avoids error messages if called prematurely (i.e., before construction of Composition is complete)
        if context.flags & ContextFlags.PROCESSING:
            return

        # Don't bother for agent_rep that is not a Composition, since state_input_ports specified can be validated
        #    or assigned by default for a CompositionApproximator, and so are either up to its implementation or to
        #    do the validation and/or default assignment (this contrasts with agent_rep that is a Composition, for
        #    which there must be a state_input_port for every InputPort of every INPUT node of the agent_rep.
        if self.agent_rep_type != COMPOSITION:
            return

        if self.state_feature_specs:
            # Restrict validation and any further instantiation of state_input_ports
            #    until run time, when the Composition is expected to be fully constructed
            if context._execution_phase == ContextFlags.PREPARING:
                # MODIFIED 1/30/22 NEW:
                # FIX: 1/30/22 - NEEDS TO EXECUTE ON UPDATES WITHOUT RUN,
                #                BUT MANAGE ERRORS WRT TO _validate_state_features
                self._update_state_features_dict()
                # MODIFIED 1/30/22 END
                self._validate_state_features(context)
            # MODIFIED 1/30/22 OLD:
            return
            # # MODIFIED 1/30/22 NEW:
            # return True
            # MODIFIED 1/30/22 END

        elif not self.state_input_ports:
            # agent_rep is Composition, but no state_features have been specified,
            #   so assign a state_input_port to shadow every InputPort of every INPUT node of agent_rep

            # Get list of nodes with any nested Comps that are INPUT Nodes replaced with their respective INPUT Nodes
            #   (as those are what need to be shadowed)
            shadowed_input_ports = []
            # # MODIFIED 3/4/22 OLD:
            # for node in self._get_agent_rep_input_receivers(comp_as_node=False):
            #     for input_port in node.input_ports:
            #         if input_port.internal_only:
            #             continue
            #         # if isinstance(input_port.owner, CompositionInterfaceMechanism):
            #         #     input_port = input_port.
            #         shadowed_input_ports.append(input_port)
            # MODIFIED 3/4/22 NEW:
            for port in self._get_agent_rep_input_receivers():
                if port.internal_only:
                    continue
                # if isinstance(input_port.owner, CompositionInterfaceMechanism):
                #     input_port = input_port.
                shadowed_input_ports.append(port)
            # MODIFIED 3/4/22 END

            #  Instantiate state_input_ports
            local_context = Context(source=ContextFlags.METHOD)
            state_input_ports = []
            # for input_port in input_ports_not_specified:
            for input_port in shadowed_input_ports:
                input_port_name = f"{SHADOW_INPUT_NAME}{input_port.owner.name}[{input_port.name}]"
                params = {SHADOW_INPUTS: input_port,
                          INTERNAL_ONLY:True}
                if self.state_feature_function:
                    # Use **state_feature_function** if specified by user in constructor
                    params = self._assign_state_feature_function(params)
                state_input_port = _instantiate_port(name=input_port_name,
                                                     port_type=InputPort,
                                                     owner=self,
                                                     reference_value=input_port.value,
                                                     params=params,
                                                     context=local_context)
                state_input_ports.append(state_input_port)

            self.add_ports(state_input_ports,
                                 update_variable=False,
                                 context=local_context)

            # Assign OptimizationControlMechanism attributes
            self.state_input_ports.data = state_input_ports
            self._num_state_feature_specs = len(self.state_input_ports)
            # # MODIFIED 3/4/22 OLD:
            # self._specified_input_nodes_in_order = self._get_agent_rep_input_receivers(comp_as_node=False)
            # # MODIFIED 3/4/22 NEW:
            self._specified_input_ports_in_order = self._get_agent_rep_input_receivers()
            # MODIFIED 3/4/22 END
            self.parameters.state_feature_specs.set([input_port.shadow_inputs for input_port in self.state_input_ports],
                                                    override=True)
            return True

    def _validate_state_features(self, context):
        """Validate that state_features are legal and consistent with agent_rep.

        Called by _update_state_input_ports_for_controller,
        - after new Nodes have been added to Composition
        - and/or in run() as final check before execution.

        Ensure that:
        - if state_feature_specs are speified as a user dict, keys are valid INPUT Nodes of agent_rep;
        - all InputPorts shadowed by specified state_input_ports are in agent_rep or one of its nested Compositions;
        - any Projections received from output_ports are from Nodes in agent_rep or its nested Compositions;
        - all InputPorts shadowed by state_input_ports reference INPUT Nodes of agent_rep or of a nested Composition;
        - state_features are compatible with input format for agent_rep Composition
        """

        # FIX: 3/4/22 - DO user_specs HAVE TO BE RE-VALIDATED? ?IS IT BECAUSE THEY MAY REFER TO NEWLY ADDED NODES?
        from psyneulink.core.compositions.composition import \
            Composition, CompositionInterfaceMechanism, CompositionError, RunError, NodeRole

        comp = self.agent_rep
        user_specs = self.parameters.state_feature_specs.spec
        # user_specs = self.state_feature_specs

        if isinstance(user_specs, dict) and SHADOW_INPUTS in user_specs:
            state_feature_specs = user_specs[SHADOW_INPUTS]
        else:
            state_feature_specs = user_specs

        if isinstance(state_feature_specs, list):
            # FIX: 2/20/22: CONSIDER ALLOWING PARTIAL SPECIFICATION OF INPUT NODES IN NESTED COMPS:
            #               NEED TO DETERMINE WHETHER NODE IS NESTED AND, IF SO, AND ONLY PARTIAL
            #               FLESH OUT SPEC FOR REST OF NEST COMP
            # Convert list to dict, assuming list is in order of INPUT Nodes,
            #    and assigning the corresponding INPUT Nodes as keys for use in comp._build_predicted_inputs_dict()
            input_nodes = comp.get_nodes_by_role(NodeRole.INPUT)
            if len(state_feature_specs) > len(input_nodes):
                nodes_not_in_agent_rep = [f"'{spec.name if isinstance(spec, Mechanism) else spec.owner.name}'"
                                          for spec in self._get_specs_not_in_agent_rep(state_feature_specs)]
                missing_nodes_str = (f", that includes the following: {', '.join(nodes_not_in_agent_rep)} "
                                     f"missing from {self.agent_rep.name}"
                                     if nodes_not_in_agent_rep else '')
                raise OptimizationControlMechanismError(
                    f"The number of '{STATE_FEATURES}' specified for {self.name} ({len(state_feature_specs)}) "
                    f"is more than the number of INPUT Nodes ({len(input_nodes)}) of the Composition assigned "
                    f"as its {AGENT_REP} ('{self.agent_rep.name}'){missing_nodes_str}.")
            input_dict = {}
            for i, spec in enumerate(state_feature_specs):
                input_dict[input_nodes[i]] = spec
            state_features = state_feature_specs
        # # MODIFIED 3/4/22 OLD:
        # elif isinstance(state_feature_specs, dict):
        #     # If user dict is specified, check that keys are legal INPUT nodes:
        #     self._validate_input_nodes(state_feature_specs.keys(), enforce=True)
        #     # If dict is specified, get values for checks below
        #     state_features = list(state_feature_specs.values())
        # elif isinstance(state_feature_specs, set):
        #     # If set is specified, check that all items are legal INPUT nodes:
        #     self._validate_input_nodes(state_feature_specs, enforce=True)
        #     # If dict is specified, get values for checks below
        #     state_features = list(state_feature_specs)
        # MODIFIED 3/4/22 NEW:
        elif isinstance(state_feature_specs, (set, dict)):
            # If set or dict is specified, check that items of set or keys of dict are legal INPUT nodes:
            self._validate_input_nodes(set(state_feature_specs), enforce=True)
            if isinstance(state_feature_specs, dict):
                # If dict is specified, get values for checks below
                state_features = list(state_feature_specs.values())
            else:
                state_features = list(state_feature_specs)
        # MODIFIED 3/4/22 END

        # Include agent rep in error messages if it is not the same as self.composition
        self_has_state_features_str = f"'{self.name}' has '{STATE_FEATURES}' specified "
        agent_rep_str = ('' if self.agent_rep == self.composition
                         else f"both its `{AGENT_REP}` ('{self.agent_rep.name}') as well as ")
        not_in_comps_str = f"that are missing from {agent_rep_str}'{self.composition.name}' and any " \
                      f"{Composition.componentCategory}s nested within it."

        # Ensure that all InputPorts shadowed by specified state_input_ports
        #    are in agent_rep or one of its nested Compositions
        invalid_state_features = [input_port for input_port in self.state_input_ports
                                  if (input_port.shadow_inputs
                                      and not (input_port.shadow_inputs.owner in
                                            list(comp.nodes) + [n[0] for n in comp._get_nested_nodes()])
                                      and (not [input_port.shadow_inputs.owner.composition is x for x in
                                                  comp._get_nested_compositions()
                                           if isinstance(input_port.shadow_inputs.owner,
                                                     CompositionInterfaceMechanism)]))]
        # Ensure any Projections received from output_ports are from Nodes in agent_rep or its nested Compositions
        for input_port in self.state_input_ports:
            if input_port.shadow_inputs:
                continue
            try:
                all(comp._get_source(p) for p in input_port.path_afferents)
            except CompositionError:
                invalid_state_features.append(input_port)
        if any(invalid_state_features):
            raise OptimizationControlMechanismError(
                self_has_state_features_str + f"({[d.name for d in invalid_state_features]}) " + not_in_comps_str)

        # Ensure that all InputPorts shadowed by specified state_input_ports
        #    reference INPUT Nodes of agent_rep or of a nested Composition
        invalid_state_features = [input_port for input_port in self.state_input_ports
                                  if (input_port.shadow_inputs
                                      and not (input_port.shadow_inputs.owner
                                               in self._get_agent_rep_input_receivers())
                                      and (isinstance(input_port.shadow_inputs.owner,
                                                      CompositionInterfaceMechanism)
                                           and not (input_port.shadow_inputs.owner.composition in
                                                    [nested_comp for nested_comp in comp._get_nested_compositions()
                                                     if nested_comp in comp.get_nodes_by_role(NodeRole.INPUT)])))]
        if any(invalid_state_features):
            raise OptimizationControlMechanismError(
                self_has_state_features_str + f"({[d.name for d in invalid_state_features]}) " + not_in_comps_str)

        # # FOLLOWING IS FOR DEBUGGING: (TO SEE CODING ERRORS DIRECTLY) -----------------------
        # print("****** DEBUGGING CODE STILL IN OCM -- REMOVE FOR PROPER TESTING ************")
        # inputs = self.agent_rep._build_predicted_inputs_dict(None, self, context)
        # inputs_dict, num_inputs = self.agent_rep._parse_input_dict(inputs)
        # # END DEBUGGING ---------------------------------------------------------------------

        # Ensure state_features are compatible with input format for agent_rep Composition
        try:
            # FIX: 1/10/22 - ?USE self.agent_rep.external_input_values FOR CHECK?
            # Call these to check for errors in constructing inputs dict
            inputs = self.agent_rep._build_predicted_inputs_dict(None, self, context)
            self.agent_rep._parse_input_dict(inputs)
        except RunError as error:
            raise OptimizationControlMechanismError(
                f"The '{STATE_FEATURES}' argument has been specified for '{self.name}' that is using a "
                f"{Composition.componentType} ('{self.agent_rep.name}') as its agent_rep, but "
                f"they are not compatible with the inputs required by its 'agent_rep': '{error.error_value}' "
                f"Use the get_inputs_format() method of '{self.agent_rep.name}' to see the required format, or "
                f"remove the specification of '{STATE_FEATURES}' from the constructor for {self.name} "
                f"to have them automatically assigned.")
        except KeyError as error:   # This occurs if a Node is illegal for a reason other than above,
            pass                    # and will issue the corresponding error message.
        except:  # Legal Node specifications, but incorrect for input to agent_rep
            specs = [f.full_name if hasattr(f, 'full_name') else (f.name if isinstance(f, Component) else f)
                     for f in state_features]
            raise OptimizationControlMechanismError(
                f"The '{STATE_FEATURES}' argument has been specified for '{self.name}' that is using a "
                f"{Composition.componentType} ('{self.agent_rep.name}') as its agent_rep, but the "
                f"'{STATE_FEATURES}' ({specs}) specified are not compatible with the inputs required by 'agent_rep' "
                f"when it is executed. Use its get_inputs_format() method to see the required format, "
                f"or remove the specification of '{STATE_FEATURES}' from the constructor for {self.name} "
                f"to have them automatically assigned.")

    def _validate_monitor_for_control(self, nodes):
        # Ensure all of the Components being monitored for control are in the agent_rep if it is Composition
        if self.agent_rep_type == COMPOSITION:
            try:
                super()._validate_monitor_for_control(self.agent_rep._get_all_nodes())
            except ControlMechanismError as e:
                raise OptimizationControlMechanismError(f"{self.name} has 'outcome_ouput_ports' that receive "
                                                        f"Projections from the following Components that do not belong "
                                                        f"to its {AGENT_REP} ({self.agent_rep.name}): {e.data}.")

    def _instantiate_output_ports(self, context=None):
        """Assign CostFunctions.DEFAULTS as default for cost_option of ControlSignals.
        """
        super()._instantiate_output_ports(context)

        for control_signal in self.control_signals:
            if control_signal.cost_options is None:
                control_signal.cost_options = CostFunctions.DEFAULTS
                control_signal._instantiate_cost_attributes(context)

    def _instantiate_control_signals(self, context):
        """Size control_allocation and assign modulatory_signals
        Set size of control_allocation equal to number of modulatory_signals.
        Assign each modulatory_signal sequentially to corresponding item of control_allocation.
        Assign RANDOMIZATION_CONTROL_SIGNAL for random_variables
        """

        control_signals = []
        for i, spec in list(enumerate(self.output_ports)):
            control_signal = self._instantiate_control_signal(spec, context=context)
            control_signal._variable_spec = (OWNER_VALUE, i)
            # MODIFIED 11/20/21 NEW:
            #  FIX - SHOULD MOVE THIS TO WHERE IT IS CALLED IN ControlSignal._instantiate_control_signal
            if self._check_for_duplicates(control_signal, control_signals, context):
                continue
            # MODIFIED 11/20/21 END
            self.output_ports[i] = control_signal

        self._create_randomization_control_signal(context)
        self.defaults.value = np.tile(control_signal.parameters.variable.default_value, (len(self.output_ports), 1))
        self.parameters.control_allocation._set(copy.deepcopy(self.defaults.value), context)

    def _create_randomization_control_signal(self, context):
        if self.num_estimates:
            # must be SampleSpec in allocation_samples arg
            randomization_seed_mod_values = SampleSpec(start=1, stop=self.num_estimates, step=1)

            # FIX: 11/3/21 noise PARAM OF TransferMechanism IS MARKED AS SEED WHEN ASSIGNED A DISTRIBUTION FUNCTION,
            #                BUT IT HAS NO PARAMETER PORT BECAUSE THAT PRESUMABLY IS FOR THE INTEGRATOR FUNCTION,
            #                BUT THAT IS NOT FOUND BY model.all_dependent_parameters
            # FIX: 1/4/22:  CHECK IF THIS WORKS WITH CFA
            #               (i.e., DOES IT [make sense to HAVE A random_variables ATTRIBUTE?)
            # Get Components with variables to be randomized across estimates
            #   and construct ControlSignal to modify their seeds over estimates
            if self.random_variables is ALL:
                self.random_variables = self.agent_rep.random_variables

            if not self.random_variables:
                warnings.warn(f"'{self.name}' has '{NUM_ESTIMATES} = {self.num_estimates}' specified, "
                              f"but its '{AGENT_REP}' ('{self.agent_rep.name}') has no random variables: "
                              f"'{RANDOMIZATION_CONTROL_SIGNAL}' will not be created, and num_estimates set to None.")
                self.num_estimates = None
                return

            randomization_control_signal = ControlSignal(name=RANDOMIZATION_CONTROL_SIGNAL,
                                                         modulates=[param.parameters.seed.port
                                                                    for param in self.random_variables],
                                                         allocation_samples=randomization_seed_mod_values)
            randomization_control_signal_index = len(self.output_ports)
            randomization_control_signal._variable_spec = (OWNER_VALUE, randomization_control_signal_index)

            randomization_control_signal = self._instantiate_control_signal(randomization_control_signal, context)

            self.output_ports.append(randomization_control_signal)

            # Otherwise, assert that num_estimates and number of seeds generated by randomization_control_signal are equal
            num_seeds = self.control_signals[RANDOMIZATION_CONTROL_SIGNAL].parameters.allocation_samples._get(context).num
            assert self.num_estimates == num_seeds, \
                    f"PROGRAM ERROR:  The value of the {NUM_ESTIMATES} Parameter of {self.name}" \
                    f"({self.num_estimates}) is not equal to the number of estimates that will be generated by " \
                    f"its {RANDOMIZATION_CONTROL_SIGNAL} ControlSignal ({num_seeds})."

            function_search_space = self.function.parameters.search_space._get(context)
            if randomization_control_signal_index >= len(function_search_space):
                # TODO: check here if search_space has an item for each
                # control_signal? or is allowing it through for future
                # checks the right way?

                # search_space must be a SampleIterator
                function_search_space.append(SampleIterator(randomization_seed_mod_values))

    def _instantiate_function(self, function, function_params=None, context=None):
        # this indicates a significant peculiarity of OCM, in that its function
        # corresponds to its value (control_allocation) rather than anything to
        # do with its variable. see _instantiate_attributes_after_function

        # Workaround this issue here, and explicitly allow the function's
        # default variable to be modified
        if isinstance(function, Function):
            function._variable_shape_flexibility = DefaultsFlexibility.FLEXIBLE

        super()._instantiate_function(function, function_params, context)

    def _instantiate_attributes_after_function(self, context=None):
        """Instantiate OptimizationControlMechanism's OptimizationFunction attributes"""

        super()._instantiate_attributes_after_function(context=context)

        search_space = self.parameters.search_space._get(context)
        if type(search_space) == np.ndarray:
            search_space = search_space.tolist()
        if search_space:
            corrected_search_space = []
            try:
                if type(search_space) == SampleIterator:
                    corrected_search_space.append(search_space)
                elif type(search_space) == SampleSpec:
                    corrected_search_space.append(SampleIterator(search_space))
                else:
                    for i in self.parameters.search_space._get(context):
                        if not type(i) == SampleIterator:
                            corrected_search_space.append(SampleIterator(specification=i))
                            continue
                        corrected_search_space.append(i)
            except AssertionError:
                corrected_search_space = [SampleIterator(specification=search_space)]
            self.parameters.search_space._set(corrected_search_space, context)

        try:
            randomization_control_signal_index = self.control_signals.names.index(RANDOMIZATION_CONTROL_SIGNAL)
        except ValueError:
            randomization_control_signal_index = None

        # Assign parameters to function (OptimizationFunction) that rely on OptimizationControlMechanism
        # NOTE: as in this call, randomization_dimension must be set
        # after search_space to avoid IndexError when getting
        # num_estimates of function
        self.function.reset(**{
            DEFAULT_VARIABLE: self.parameters.control_allocation._get(context),
            OBJECTIVE_FUNCTION: self.evaluate_agent_rep,
            # SEARCH_FUNCTION: self.search_function,
            # SEARCH_TERMINATION_FUNCTION: self.search_termination_function,
            SEARCH_SPACE: self.parameters.control_allocation_search_space._get(context),
            RANDOMIZATION_DIMENSION: randomization_control_signal_index
        })

        if isinstance(self.agent_rep, type):
            self.agent_rep = self.agent_rep()

        if self.agent_rep_type == COMPOSITION_FUNCTION_APPROXIMATOR:
            self._initialize_composition_function_approximator(context)

    def _execute(self, variable=None, context=None, runtime_params=None):
        """Find control_allocation that optimizes result of agent_rep.evaluate().
        """

        if self.is_initializing:
            return [defaultControlAllocation]

        # Assign default control_allocation if it is not yet specified (presumably first trial)
        control_allocation = self.parameters.control_allocation._get(context)
        if control_allocation is None:
            control_allocation = [c.defaults.variable for c in self.control_signals]
            self.parameters.control_allocation._set(control_allocation, context=None)

        # Give the agent_rep a chance to adapt based on last trial's state_feature_values and control_allocation
        if hasattr(self.agent_rep, "adapt"):
            # KAM 4/11/19 switched from a try/except to hasattr because in the case where we don't
            # have an adapt method, we also don't need to call the net_outcome getter
            net_outcome = self.parameters.net_outcome._get(context)

            self.agent_rep.adapt(self.parameters.state_feature_values._get(context),
                                 control_allocation,
                                 net_outcome,
                                 context=context)

        # freeze the values of current context, because they can be changed in between simulations,
        # and the simulations must start from the exact spot
        frozen_context = self._get_frozen_context(context)

        alt_controller = None
        if self.agent_rep.controller is None:
            try:
                alt_controller = context.composition.controller
            except AttributeError:
                pass

        self.agent_rep._initialize_as_agent_rep(
            frozen_context, base_context=context, alt_controller=alt_controller
        )

        # Get control_allocation that optimizes net_outcome using OptimizationControlMechanism's function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        optimal_control_allocation, optimal_net_outcome, saved_samples, saved_values = \
                                                super(ControlMechanism,self)._execute(
                                                    variable=control_allocation,
                                                    num_estimates=self.parameters.num_estimates._get(context),
                                                    context=context,
                                                    runtime_params=runtime_params
                                                )

        # clean up frozen values after execution
        self.agent_rep._clean_up_as_agent_rep(frozen_context, alt_controller=alt_controller)

        optimal_control_allocation = np.array(optimal_control_allocation).reshape((len(self.defaults.value), 1))
        if self.function.save_samples:
            self.saved_samples = saved_samples
        if self.function.save_values:
            self.saved_values = saved_values

        # Return optimal control_allocation
        return optimal_control_allocation

    def _get_frozen_context(self, context=None):
        return Context(execution_id=f'{context.execution_id}{EID_FROZEN}')

    def _set_up_simulation(
        self,
        base_context=Context(execution_id=None),
        control_allocation=None,
        alt_controller=None
    ):
        sim_context = copy.copy(base_context)
        sim_context.execution_id = self.get_next_sim_id(base_context, control_allocation)

        try:
            self.parameters.simulation_ids._get(base_context).append(sim_context.execution_id)
        except AttributeError:
            self.parameters.simulation_ids._set([sim_context.execution_id], base_context)

        self.agent_rep._initialize_as_agent_rep(
            sim_context,
            base_context=self._get_frozen_context(base_context),
            alt_controller=alt_controller
        )

        return sim_context

    def _tear_down_simulation(self, sim_context, alt_controller=None):
        if not self.agent_rep.parameters.retain_old_simulation_data._get():
            self.agent_rep._clean_up_as_agent_rep(sim_context, alt_controller=alt_controller)

    def evaluate_agent_rep(self, control_allocation, context=None, return_results=False):
        """Call `evaluate <Composition.evaluate>` method of `agent_rep <OptimizationControlMechanism.agent_rep>`

        Assigned as the `objective_function <OptimizationFunction.objective_function>` for the
        OptimizationControlMechanism's `function <OptimizationControlMechanism.function>`.

        Evaluates `agent_rep <OptimizationControlMechanism.agent_rep>` by calling its `evaluate <Composition.evaluate>`
        method, which executes its `agent_rep <OptimizationControlMechanism.agent_rep>` using the current
        `state_feature_values <OptimizationControlMechanism.state_feature_values>` as the input and the specified
        **control_allocation**.

        If the `agent_rep <OptimizationControlMechanism.agent_rep>` is a `Composition`, each execution is a call to
        its `run <Composition.run>` method that uses the `num_trials_per_estimate
        <OptimizationControlMechanism.num_trials_per_estimate>` as its **num_trials** argument, and the same
        `state_feature_values <OptimizationControlMechanism.state_feature_values>` and **control_allocation**
        but a different randomly chosen seed for the random number generator for each run.  It then returns an array of
        length **number_estimates** containing the `net_outcome <ControlMechanism.net_outcome>` of each execution
        and, if **return_results** is True, also an array with the `results <Composition.results>` of each run.

        COMMENT:
        FIX: THIS SHOULD BE REFACTORED TO BE HANDLED THE SAME AS A Composition AS agent_rep
        COMMENT
        If the `agent_rep <OptimizationControlMechanism.agent_rep>` is a CompositionFunctionApproximator,
        then `num_estimates <OptimizationControlMechanism.num_estimates>` is passed to it to handle execution and
        estimation as determined by its implementation, and returns a single estimated net_outcome.


        (See `evaluate <Composition.evaluate>` for additional details.)
        """

        # agent_rep is a Composition (since runs_simulations = True)
        if self.agent_rep.runs_simulations:
            alt_controller = None
            if self.agent_rep.controller is None:
                try:
                    alt_controller = context.composition.controller
                except AttributeError:
                    pass
            # KDM 5/20/19: crudely using default here because it is a stateless parameter
            # and there is a bug in setting parameter values on init, see TODO note above
            # call to self._instantiate_defaults around component.py:1115
            if self.defaults.search_statefulness:
                new_context = self._set_up_simulation(context, control_allocation, alt_controller)
            else:
                new_context = context

            old_composition = context.composition
            context.composition = self.agent_rep

            # We shouldn't get this far if execution mode is not Python
            assert self.parameters.comp_execution_mode._get(context) == "Python"
            exec_mode = pnlvm.ExecutionMode.Python
            ret_val = self.agent_rep.evaluate(self.parameters.state_feature_values._get(context),
                                              control_allocation,
                                              self.parameters.num_trials_per_estimate._get(context),
                                              base_context=context,
                                              context=new_context,
                                              execution_mode=exec_mode,
                                              return_results=return_results)
            context.composition = old_composition
            if self.defaults.search_statefulness:
                self._tear_down_simulation(new_context, alt_controller)

            # FIX: THIS SHOULD BE REFACTORED TO BE HANDLED THE SAME AS A Composition AS agent_rep
            # If results of the simulation should be returned then, do so. agent_rep's evaluate method will
            # return a tuple in this case in which the first element is the outcome as usual and the second
            # is the results of the composition run.
            if return_results:
                return ret_val[0], ret_val[1]
            else:
                return ret_val

        # FIX: 11/3/21 - ??REFACTOR CompositionFunctionApproximator TO NOT TAKE num_estimates
        #                (i.e., LET OptimzationFunction._grid_evaluate HANDLE IT)
        # agent_rep is a CompositionFunctionApproximator (since runs_simuluations = False)
        else:
            return self.agent_rep.evaluate(self.parameters.state_feature_values._get(context),
                                           control_allocation,
                                           self.parameters.num_estimates._get(context),
                                           self.parameters.num_trials_per_estimate._get(context),
                                           context=context
                                           )

    def _get_evaluate_input_struct_type(self, ctx):
        # We construct input from optimization function input
        return ctx.get_input_struct_type(self.function)

    def _get_evaluate_output_struct_type(self, ctx):
        # Returns a scalar that is the predicted net_outcome
        return ctx.float_ty

    def _get_evaluate_alloc_struct_type(self, ctx):
        return pnlvm.ir.ArrayType(ctx.float_ty,
                                  len(self.parameters.control_allocation_search_space.get()))

    def _gen_llvm_net_outcome_function(self, *, ctx, tags=frozenset()):
        assert "net_outcome" in tags
        args = [ctx.get_param_struct_type(self).as_pointer(),
                ctx.get_state_struct_type(self).as_pointer(),
                self._get_evaluate_alloc_struct_type(ctx).as_pointer(),
                ctx.float_ty.as_pointer(),
                ctx.float_ty.as_pointer()]

        builder = ctx.create_llvm_function(args, self, str(self) + "_net_outcome")
        llvm_func = builder.function
        for p in llvm_func.args:
            p.attributes.add('nonnull')
        params, state, allocation_sample, objective_ptr, arg_out = llvm_func.args

        op_params = pnlvm.helpers.get_param_ptr(builder, self, params,
                                                "output_ports")
        op_states = pnlvm.helpers.get_state_ptr(builder, self, state,
                                                "output_ports", None)

        # calculate cost function
        total_cost = builder.alloca(ctx.float_ty, name="total_cost")
        builder.store(ctx.float_ty(-0.0), total_cost)
        for i, op in enumerate(self.output_ports):
            op_i_params = builder.gep(op_params, [ctx.int32_ty(0),
                                                  ctx.int32_ty(i)])
            op_i_state = builder.gep(op_states, [ctx.int32_ty(0),
                                                 ctx.int32_ty(i)])

            op_f = ctx.import_llvm_function(op, tags=frozenset({"costs"}))

            op_in = builder.alloca(op_f.args[2].type.pointee,
                                   name="output_port_cost_in")

            # copy allocation_sample, the input is 1-element array in a struct
            data_in = builder.gep(allocation_sample, [ctx.int32_ty(0),
                                                      ctx.int32_ty(i)])
            data_out = builder.gep(op_in, [ctx.int32_ty(0), ctx.int32_ty(0),
                                           ctx.int32_ty(0)])
            if data_in.type != data_out.type:
                warnings.warn(f"Shape mismatch: Allocation sample '{i}' "
                              f"({self.parameters.control_allocation_search_space.get()}) "
                              f"doesn't match input port input ({op.defaults.variable}).")
                assert len(data_out.type.pointee) == 1
                data_out = builder.gep(data_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

            builder.store(builder.load(data_in), data_out)

            # Invoke cost function
            cost = builder.call(op_f, [op_i_params, op_i_state, op_in])

            # simplified version of combination fmax(cost, 0)
            ltz = builder.fcmp_ordered("<", cost, cost.type(0))
            cost = builder.select(ltz, cost.type(0), cost)

            # combine is not a PNL function
            assert self.combine_costs is np.sum
            val = builder.load(total_cost)
            val = builder.fadd(val, cost)
            builder.store(val, total_cost)

        # compute net_outcome
        objective = builder.load(objective_ptr)
        net_outcome = builder.fsub(objective, builder.load(total_cost))
        builder.store(net_outcome, arg_out)

        builder.ret_void()
        return llvm_func

    def _gen_llvm_evaluate_alloc_range_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags=frozenset()):
        assert "evaluate" in tags
        assert "alloc_range" in tags
        evaluate_f = ctx.import_llvm_function(self, tags=tags - {"alloc_range"})

        args = [*evaluate_f.type.pointee.args[:2],
                ctx.int32_ty, ctx.int32_ty,
                *evaluate_f.type.pointee.args[3:]]
        builder = ctx.create_llvm_function(args, self, str(self) + "_evaluate_range")
        llvm_func = builder.function

        params, state, start, stop, arg_out, arg_in, data = llvm_func.args
        for p in llvm_func.args:
            if isinstance(p.type, (pnlvm.ir.PointerType)):
                p.attributes.add('nonnull')

        nodes_params = pnlvm.helpers.get_param_ptr(builder, self.composition,
                                                   params, "nodes")
        my_idx = self.composition._get_node_index(self)
        my_params = builder.gep(nodes_params, [ctx.int32_ty(0),
                                               ctx.int32_ty(my_idx)])
        func_params = pnlvm.helpers.get_param_ptr(builder, self,
                                                  my_params, "function")
        search_space = pnlvm.helpers.get_param_ptr(builder, self.function,
                                                   func_params, "search_space")

        allocation = builder.alloca(evaluate_f.args[2].type.pointee, name="allocation")
        with pnlvm.helpers.for_loop(builder, start, stop, stop.type(1), "alloc_loop") as (b, idx):

            func_out = b.gep(arg_out, [idx])
            pnlvm.helpers.create_allocation(b, allocation, search_space, idx)

            b.call(evaluate_f, [params, state, allocation, func_out, arg_in, data])

        builder.ret_void()
        return llvm_func

    def _gen_llvm_evaluate_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags=frozenset()):
        assert "evaluate" in tags
        args = [ctx.get_param_struct_type(self.agent_rep).as_pointer(),
                ctx.get_state_struct_type(self.agent_rep).as_pointer(),
                self._get_evaluate_alloc_struct_type(ctx).as_pointer(),
                self._get_evaluate_output_struct_type(ctx).as_pointer(),
                self._get_evaluate_input_struct_type(ctx).as_pointer(),
                ctx.get_data_struct_type(self.agent_rep).as_pointer()]

        builder = ctx.create_llvm_function(args, self, str(self) + "_evaluate")
        llvm_func = builder.function
        for p in llvm_func.args:
            p.attributes.add('nonnull')

        comp_params, base_comp_state, allocation_sample, arg_out, arg_in, base_comp_data = llvm_func.args

        if "const_params" in debug_env:
            comp_params = builder.alloca(comp_params.type.pointee, name="const_params_loc")
            const_params = comp_params.type.pointee(self.agent_rep._get_param_initializer(None))
            builder.store(const_params, comp_params)

        # Create a simulation copy of composition state
        comp_state = builder.alloca(base_comp_state.type.pointee, name="state_copy")
        if "const_state" in debug_env:
            const_state = self.agent_rep._get_state_initializer(None)
            builder.store(comp_state.type.pointee(const_state), comp_state)
        else:
            builder = pnlvm.helpers.memcpy(builder, comp_state, base_comp_state)

        # Create a simulation copy of composition data
        comp_data = builder.alloca(base_comp_data.type.pointee, name="data_copy")
        if "const_data" in debug_env:
            const_data = self.agent_rep._get_data_initializer(None)
            builder.store(comp_data.type.pointee(const_data), comp_data)
        else:
            builder = pnlvm.helpers.memcpy(builder, comp_data, base_comp_data)

        # Evaluate is called on composition controller
        assert self.composition.controller is self
        assert self.composition is self.agent_rep
        nodes_states = pnlvm.helpers.get_state_ptr(builder, self.composition,
                                                   comp_state, "nodes", None)
        nodes_params = pnlvm.helpers.get_param_ptr(builder, self.composition,
                                                   comp_params, "nodes")

        controller_idx = self.composition._get_node_index(self)
        controller_state = builder.gep(nodes_states, [ctx.int32_ty(0),
                                                      ctx.int32_ty(controller_idx)])
        controller_params = builder.gep(nodes_params, [ctx.int32_ty(0),
                                                       ctx.int32_ty(controller_idx)])

        # Get simulation function
        sim_f = ctx.import_llvm_function(self.agent_rep,
                                         tags=frozenset({"run", "simulation"}))

        # Apply allocation sample to simulation data
        assert len(self.output_ports) == len(allocation_sample.type.pointee)
        idx = self.agent_rep._get_node_index(self)
        ocm_out = builder.gep(comp_data, [ctx.int32_ty(0), ctx.int32_ty(0),
                                          ctx.int32_ty(idx)])
        for i, _ in enumerate(self.output_ports):
            idx = ctx.int32_ty(i)
            sample_ptr = builder.gep(allocation_sample, [ctx.int32_ty(0), idx])
            sample_dst = builder.gep(ocm_out, [ctx.int32_ty(0), idx, ctx.int32_ty(0)])
            if sample_ptr.type != sample_dst.type:
                assert len(sample_dst.type.pointee) == 1
                sample_dst = builder.gep(sample_dst, [ctx.int32_ty(0),
                                                      ctx.int32_ty(0)])
            builder.store(builder.load(sample_ptr), sample_dst)

        # Construct input
        comp_input = builder.alloca(sim_f.args[3].type.pointee, name="sim_input")

        input_initialized = [False] * len(comp_input.type.pointee)
        for src_idx, ip in enumerate(self.input_ports):
            if ip.shadow_inputs is None:
                continue

            # shadow inputs point to an input port of of a node.
            # If that node takes direct input, it will have an associated
            # (input_port, output_port) in the input_CIM.
            # Take the former as an index to composition input variable.
            cim_in_port = self.agent_rep.input_CIM_ports[ip.shadow_inputs][0]
            dst_idx = self.agent_rep.input_CIM.input_ports.index(cim_in_port)

            # Check that all inputs are unique
            assert not input_initialized[dst_idx], "Double initialization of input {}".format(dst_idx)
            input_initialized[dst_idx] = True

            src = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(src_idx)])
            # Destination is a struct of 2d arrays
            dst = builder.gep(comp_input, [ctx.int32_ty(0),
                                           ctx.int32_ty(dst_idx),
                                           ctx.int32_ty(0)])
            builder.store(builder.load(src), dst)

        # Assert that we have populated all inputs
        assert all(input_initialized), \
          "Not all inputs to the simulated composition are initialized: {}".format(input_initialized)

        if "const_input" in debug_env:
            if not debug_env["const_input"]:
                input_init = [[os.defaults.variable.tolist()] for os in self.agent_rep.input_CIM.input_ports]
                print("Setting default input: ", input_init)
            else:
                input_init = ast.literal_eval(debug_env["const_input"])
                print("Setting user input in evaluate: ", input_init)

            builder.store(comp_input.type.pointee(input_init), comp_input)


        # Determine simulation counts
        num_trials_per_estimate_ptr = pnlvm.helpers.get_param_ptr(builder, self,
                                                        controller_params,
                                                        "num_trials_per_estimate")

        num_trials_per_estimate = builder.load(num_trials_per_estimate_ptr, "num_trials_per_estimate")

        # if num_trials_per_estimate is 0, run 1 trial
        param_is_zero = builder.icmp_unsigned("==", num_trials_per_estimate,
                                                    ctx.int32_ty(0))
        num_sims = builder.select(param_is_zero, ctx.int32_ty(1),
                                  num_trials_per_estimate, "corrected_estimates")

        num_runs = builder.alloca(ctx.int32_ty, name="num_runs")
        builder.store(num_sims, num_runs)

        # We only provide one input
        num_inputs = builder.alloca(ctx.int32_ty, name="num_inputs")
        builder.store(num_inputs.type.pointee(1), num_inputs)

        # Simulations don't store output
        comp_output = sim_f.args[4].type(None)
        builder.call(sim_f, [comp_state, comp_params, comp_data, comp_input,
                             comp_output, num_runs, num_inputs])

        # Extract objective mechanism value
        idx = self.agent_rep._get_node_index(self.objective_mechanism)
        # Mechanisms' results are stored in the first substructure
        objective_os_ptr = builder.gep(comp_data, [ctx.int32_ty(0),
                                                   ctx.int32_ty(0),
                                                   ctx.int32_ty(idx)])
        # Objective mech output shape should be 1 single element 2d array
        objective_val_ptr = builder.gep(objective_os_ptr,
                                        [ctx.int32_ty(0), ctx.int32_ty(0),
                                         ctx.int32_ty(0)], "obj_val_ptr")

        net_outcome_f = ctx.import_llvm_function(self, tags=tags.union({"net_outcome"}))
        builder.call(net_outcome_f, [controller_params, controller_state,
                                     allocation_sample, objective_val_ptr,
                                     arg_out])

        builder.ret_void()

        return llvm_func

    def _gen_llvm_function(self, *, ctx:pnlvm.LLVMBuilderContext, tags:frozenset):
        if "net_outcome" in tags:
            return self._gen_llvm_net_outcome_function(ctx=ctx, tags=tags)
        if "evaluate" in tags and "alloc_range" in tags:
            return self._gen_llvm_evaluate_alloc_range_function(ctx=ctx, tags=tags)
        if "evaluate" in tags:
            return self._gen_llvm_evaluate_function(ctx=ctx, tags=tags)

        is_comp = not isinstance(self.agent_rep, Function)
        if is_comp:
            extra_args = [ctx.get_param_struct_type(self.agent_rep).as_pointer(),
                          ctx.get_state_struct_type(self.agent_rep).as_pointer(),
                          ctx.get_data_struct_type(self.agent_rep).as_pointer()]
        else:
            extra_args = []

        f = super()._gen_llvm_function(ctx=ctx, extra_args=extra_args, tags=tags)
        if is_comp:
            for a in f.args[-len(extra_args):]:
                a.attributes.add('nonnull')

        return f

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, context, variable, *, tags:frozenset):
        fun = ctx.import_llvm_function(function)
        fun_out = builder.alloca(fun.args[3].type.pointee, name="func_out")

        args = [params, context, variable, fun_out]
        # If we're calling compiled version of Composition.evaluate,
        # we need to pass extra arguments
        if len(fun.args) > 4:
            args += builder.function.args[-3:]
        builder.call(fun, args)

        return fun_out, builder

    def _gen_llvm_output_port_parse_variable(self, ctx, builder, params, context, value, port):
        i = self.output_ports.index(port)
        # Allocate the only member of the port input struct
        oport_input = builder.alloca(ctx.get_input_struct_type(port).elements[0],
                                     name="output_port_in")
        # FIXME: workaround controller signals occasionally being 2d
        dest_ptr = pnlvm.helpers.unwrap_2d_array(builder, oport_input)
        dest_ptr = builder.gep(dest_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val_ptr = builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(i)])
        builder.store(builder.load(val_ptr), dest_ptr)
        return oport_input

    @property
    def agent_rep_type(self):
        from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator
        if (isinstance(self.agent_rep, CompositionFunctionApproximator)
                or self.agent_rep.componentCategory is COMPOSITION_FUNCTION_APPROXIMATOR):
            return COMPOSITION_FUNCTION_APPROXIMATOR
        elif self.agent_rep.componentCategory=='Composition':
            return COMPOSITION
        else:
            return None

    @property
    def num_state_input_ports(self):
        try:
            return len(self.state_input_ports)
        except:
            return 0

    @property
    def state_features(self):
        self._update_state_features_dict()
        agent_rep_input_ports = self._get_agent_rep_input_receivers()
        state_features_dict = {}
        # Use num_state_feature_specs here instead of num_state_input_ports as there may be some "null" (None) specs
        for i in range(self._num_state_feature_specs):
            # Assign keys as INPUT Nodes of agent_rep
            if self._specified_input_ports_in_order[i] in agent_rep_input_ports:
                k = self._specified_input_ports_in_order[i]
            else:
                k = f"EXPECTED INPUT NODE {i} OF {self.agent_rep.name}"
            state_features_dict[k] = self.state_feature_specs[i]
        return state_features_dict

    @property
    def state_feature_sources(self):
        """Dict with {INPUT Node: source} entries for sources specified in **state_features** arg of constructor."""
        # FIX: 1/30/22:
        # FIX: SUPPORT tuple SPECIFICATION FOR sources
        # FIX: REDUCE ALL MECHANISM SOURCES TO Port SPECS
        # FIX: ??REFACTOR TO CONSTRUCT THE FOLLOWING:
        #          state_feature_specs_dict: {INPUT Node: spec}
        #          state_features: {INPUT Node: source}
        if isinstance(self.state_feature_specs, dict) and SHADOW_INPUTS in self.state_feature_specs:
            # SHADOW_INPUTS dict
            state_feature_specs = self.state_feature_specs[SHADOW_INPUTS]
        else:
            # list and set specs
            state_feature_specs = self.state_feature_specs
        sources = [source_tuple[0] if source_tuple[0] != DEFAULT_VARIABLE else value
                   for source_tuple, value in self.state_distal_sources_and_destinations_dict.items()]
        return {k:v for k,v in zip(self.state_features, sources)}

    def _state(self, context=None):
        """Get context-specific state_feature and control_allocation values"""
        # Use self.state_feature_values Parameter if state_features specified; else use state_input_port values
        state_feat_vals = self.parameters.state_feature_values.get(context)
        state_feature_values = state_feat_vals if len(state_feat_vals) else self.state_input_ports.values
        # FIX: 1/30/22:  USE CONTEXT TO GET control_allocations IF THAT IS NOT ALREADY DONE ON THAT ATTRIBUTE
        return [v.tolist() for v in state_feature_values] + self.control_allocation.tolist()

    @property
    def state(self):
        """Array that is concatenation of state_feature_values and control_allocations"""
        # Use self.state_feature_values Parameter if state_features specified; else use state_input_port values
        return self.state_feature_values + self.control_allocation.tolist()

    @property
    def state_distal_sources_and_destinations_dict(self):
        """Return dict with (Port, Node, Composition, index) tuples as keys and corresponding state[index] as values.
        Initial entries are for sources of the state_feature_values (i.e., distal afferents for state_input_ports)
        and subsequent entries are for destination parameters modulated by the OptimizationControlMechanism's
        ControlSignals (i.e., distal efferents of its ControlProjections).
        Note: the index is required, since a state_input_port may have more than one afferent Projection
              (that is, a state_feature_value may be determined by Projections from more than one source),
              and a ControlSignal may have more than one ControlProjection (that is, a given element of the
              control_allocation may apply to more than one Parameter).  However, for state_input_ports that shadow
              a Node[InputPort], only that Node[InputPort] is listed in state_dict even if the Node[InputPort] being
              shadowed has more than one afferent Projection (this is because it is the value of the Node[InputPort]
              (after it has processed the value of its afferent Projections) that determines the input to the
              state_input_port.
        """

        state_dict = {}

        # FIX: 1/30/22 - RECONCILE AND DOCUMENT MULTIPLE POSSIBLE SOURCES/DESTINATIONS FOR
        #          - THIS DICT AND STATE_FEATURES DICTS, WHICH MAY HAVE MORE THAN ONE ENTRY PER INPUT_NODE
        #          - VS. STATE_FEATURE_VALUES AND STATE_FEATURE_SPECS_DICT (WHICH HAVE ONLY ONE ENTRY PER INPUT_NODE
        # Get sources for state_feature_values of state:
        for state_index, port in enumerate(self.state_input_ports):
            if not port.path_afferents:
                if port.default_input is DEFAULT_VARIABLE:
                    source_port = DEFAULT_VARIABLE
                    node = None
                    comp = None
                else:
                    continue
            else:
                get_info_method = self.composition._get_source
                # MODIFIED 1/8/22: ONLY ONE PROJECTION PER STATE FEATURE
                if port.shadow_inputs:
                    port = port.shadow_inputs
                    if port.owner in self.composition.nodes:
                        composition = self.composition
                    else:
                        composition = port.path_afferents[0].sender.owner.composition
                    get_info_method = composition._get_destination
                source_port, node, comp = get_info_method(port.path_afferents[0])
            state_dict.update({(source_port, node, comp, state_index):self.state[state_index]})
        state_index += 1
        # Get recipients of control_allocations values of state:
        for ctl_index, control_signal in enumerate(self.control_signals):
            for projection in control_signal.efferents:
                port, node, comp = self.composition._get_destination(projection)
                state_dict.update({(port, node, comp, state_index + ctl_index):self.state[state_index + ctl_index]})

        return state_dict

    @property
    def _model_spec_parameter_blacklist(self):
        # default_variable is hidden in constructor arguments,
        # and anyway assigning it is problematic because it is modified
        # several times when creating input ports, and assigning function that
        # fits the control allocation
        return super()._model_spec_parameter_blacklist.union({
            'variable',
        })

    # ******************************************************************************************************************
    # FIX:  THE FOLLOWING IS SPECIFIC TO CompositionFunctionApproximator AS agent_rep
    # ******************************************************************************************************************

    def _initialize_composition_function_approximator(self, context):
        """Initialize CompositionFunctionApproximator"""

        # CompositionFunctionApproximator needs to have access to control_signals to:
        # - to construct control_allocation_search_space from their allocation_samples attributes
        # - compute their values and costs for samples of control_allocations from control_allocation_search_space
        self.agent_rep.initialize(features_array=np.array(self.defaults.variable[1:]),
                                  control_signals = self.control_signals,
                                  context=context)

    # # FIX: NEEDS TO BE UPDATED / REFACTORED TO WORK WITH _parse_state_feature_specs
    # # FIX: THE FOLLOWING SHOULD BE MERGED WITH HANDLING OF PredictionMechanisms FOR ORIG MODEL-BASED APPROACH;
    # # FIX: SHOULD BE GENERALIZED AS SOMETHING LIKE update_feature_values
    # @tc.typecheck
    # @handle_external_context()
    # def add_state_features(self, features, context=None):
    #     """Add InputPorts and Projections to OptimizationControlMechanism for state_features used to
    #     predict `net_outcome <ControlMechanism.net_outcome>`
    #
    #     **state_features** argument can use any of the forms of specification allowed for InputPort(s)
    #     """
    #
    #     if features:
    #         features = self._parse_state_feature_specs(features=features,
    #                                                    context=context)
    #     self.add_ports(InputPort, features)
