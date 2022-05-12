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
          - `state_input_ports <OptimizationControlMechanism_State_Input_Ports>`
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
its `net_outcome <ControlMechanism.net_outcome>`.  An OptimizationControlMechanism can be configured to implement
various forms of optimization, ranging from fully `model-based optimization <OptimizationControlMechanism_Model_Based>`
that uses the Composition itself as the  `agent_rep <OptimizationControlMechanism.agent_rep>` to simulate the
outcome for a given `state <OptimizationControlMechanism_State>` (i.e., a combination of the current input and a
particular `control_allocation <ControlMechanism.control_allocation>`), to fully `model-free optimization
<OptimizationControlMechanism_Model_Free>` by using a `CompositionFunctionApproximator` as the `agent_rep
<OptimizationControlMechanism.agent_rep>` that learns to  predict the outcomes for a state. Intermediate forms of
optimization can also be implemented, that use simpler Compositions to approximate the dynamics of the full
Composition. The outcome of executing the `agent_rep <OptimizationControlMechanism.agent_rep>` is used to compute a
`net_outcome <ControlMechanism.net_outcome>` for a given `state <OptimizationControlMechanism_State>`, that takes
into account the `costs <ControlMechanism_Costs_NetOutcome>` associated with the `control_allocation
<ControlMechanism.control_allocation>`, and is used to determine the optimal `control_allocations
<ControlMechanism.control_allocation>`.

.. _OptimizationControlMechanism_EVC:

**Expected Value of Control**

The `net_outcome <ControlMechanism.net_outcome>` of an OptimizationControlMechanism's `agent_rep
<OptimizationControlMechanism.agent_rep>` is computed -- for a given `state <OptimizationControlMechanism_State>`
(i.e., set of `state_feature_values <OptimizationControlMechanism.state_feature_values>` and a `control_allocation
<ControlMechanism.control_allocation>`) -- as the difference between the `outcome <ControlMechanism.outcome>` computed
by its `objective_mechanism <ControlMechanism.objective_mechanism>` and the aggregated `costs <ControlMechanism.costs>`
of its `control_signals <OptimizationControlMechanism.control_signals>` computed by its `combine_costs
<ControlMechanism.combine_costs>` function.  If the `outcome <ControlMechanism.outcome>` computed by the
`objective_mechanism <ControlMechanism.objective_mechanism>` is configured to measure the value of processing (e.g.,
reward received, time taken to respond, or a combination of these, etc.), and the `OptimizationFunction` assigned as
the OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` is configured to find the
`control_allocation <ControlMechanism.control_allocation>` that maximizes its `net_outcome
<ControlMechanism.net_outcome>` (that is, the `outcome <ControlMechanism.outcome>` discounted by the
result of the `combine_costs <ControlMechanism.combine_costs>` function), then the OptimizationControlMechanism is
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
belongs (and controls), another (presumably simpler) one, or a `CompositionFunctionApproximator` that is used to
estimate the `net_outcome <ControlMechanism.net_outcome>` of the Composition of which the OptimizationControlMechanism
is the `controller <Composition.controller>`.  These different types of `agent representation
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

The fullest form of this is implemented by assigning the Composition for which the OptimizationControlMechanism is the
`controller <Composition.controller>`) as its a`agent_rep  <OptimizationControlMechanism.agent_rep>`
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
<ControlMechanism.net_outcome>` for a given state (e.g., using reinforcement learning or other forms of
function approximation, such as a `RegressionCFA`).  In each `TRIAL <TimeScale.TRIAL>` the  `agent_rep
<OptimizationControlMechanism.agent_rep>` is used to search over `control_allocation
<ControlMechanism.control_allocation>`\\s, to find the one that yields the best predicted `net_outcome
<ControlMechanism.net_outcome>` of processing on the upcoming trial, based on the current or (expected)
`state_feature_values <OptimizationControlMechanism.state_feature_values>` for that trial.  The `agent_rep
<OptimizationControlMechanism.agent_rep>` is also given the chance to adapt in order to improve its prediction of
its `net_outcome <ControlMechanism.net_outcome>` based on the `state <OptimizationControlMechanism_State>` and
`net_outcome <ControlMechanism.net_outcome>` of the prior `TRIAL <TimeScale.TRIAL>`.  A Composition can also be
used to generate such predictions, permitting forms of optimization that are intermediate between the extreme
examples of model-based and model-free, as noted above.

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

* **state_features** -- specifies the sources of input to the OptimizationControlMechanism's `agent_rep
  <OptimizationControlMechanism.agent_rep>` which, together with a selected `control_allocation
  <ControlMechanism.control_allocation>`, are provided as input to it's `evaluate <Composition.evaluate>` method
  when that is executed to estimate or predict the Composition's `net_outcome <ControlMechanism.net_outcome>`.
  Those sources of input are used to construct the OptimizationControlMechanism's `state_input_ports
  <OptimizationControlMechanism.state_input_ports>`, one for each `external InputPort
  <Composition_Input_External_InputPorts>` of the `agent_rep <OptimizationControlMechanism.agent_rep>`. The input to
  each `state_input_port <OptimizationControlMechanism.state_input_ports>`, after being processed by it `function
  <InputPort.function>`, is assigned as the corresponding value of `state_feature_values
  <OptimizationControlMechanism.state_feature_values>`, the values of which provided as the input to the corresponding
  InputPorts of the `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the agent_rep each time it is `evaluated
  <Composition.evaluate>`.  Accordingly, the specification requirements for **state_features** depend on whether the
  `agent_rep<OptimizationControlMechanism.agent_rep>` is a `Composition` or a `CompositionFunctionApproximator`,
  as described in each of the two sections below.

  |

  .. _OptimizationControlMechanism_Agent_Rep_Composition:

  **state_features** *for an agent_rep that is a* **Composition**

  |

  .. _OptimizationControlMechanism_State_Features_Automatic_Assignment:

  *Automatic assignment.*  By default, if **state_features**, **state_feature_default** and **state_feature_function**
  are not specified, the `state_input_ports <OptimizationControlMechanism.state_input_ports>` are configured to
  `shadow the inputs <InputPort_Shadow_Inputs>` of every `external InputPort <Composition_Input_External_InputPorts>`
  of the `agent_rep <OptimizationControlMechanism.agent_rep>` Composition;  as a result, each time `agent_rep
  <OptimizationControlMechanism.agent_rep>` is `evaluated <Composition.evaluate>`, it receives the same `external
  input <Composition_Execution_Inputs>`) it received during its last `TRIAL<TimeScale.TRIAL>` of execution.

  |

  .. _OptimizationControlMechanism_State_Features_Explicit_Specification:

  *Explicit specification.* Specifying the **state_features**, **state_feature_default** and/or
  **state_feature_function** arguments explicitly can be useful if: values need to be provided as input to the
  `agent_rep <OptimizationControlMechanism.agent_rep>` when it is evaluated other than its `external inputs
  <Composition_Execution_Inputs>`; to restrict evaluation to a subset of its inputs (while others are held constant);
  and/or to assign specific functions to one or more `state_input_ports
  <OptimizationControlMechanism.state_input_ports>` (see `below
  <OptimizationControlMechanism_State_Feature_Function_Arg>`) that allow them to process the inputs
  (e.g., modulate and/or integrate them) before they are assigned to `state_feature_values
  <OptimizationControlMechanism.state_feature_values>` and passed to the `agent_rep
  <OptimizationControlMechanism.agent_rep>`. Assignments can be made to **state_features** corresponding
  to any or all InputPorts of the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `INPUT <NodeRole.INPUT>`
  `Nodes <Composition_Nodes>`, as described `below <OptimizationControlMechanism_State_Features_Specification>`.
  Any that are not specified are assigned the value specified for **state_feature_default** (*SHADOW_INPUTS* by
  default; see `state_feature_default <OptimizationControlMechanism.state_feature_default>` for additional details).
  A single assignment can be made for all **state_features**, or they can be specified individually for each `INPUT
  <NodeRole.INPUT>` `Nodes <Composition_Nodes>` InputPort, as descdribed below.

  .. _OptimizationControlMechanism_State_Features_Shapes:

      .. note::
         If **state_features** are specified explicitly, the `value <Component.value>`\\s of the specified Components
         must match the `input_shape <InputPort.input_shape>` of the corresponding InputPorts of the `agent_rep
         <OptimizationControlMechanism.agent_rep>`\\'s `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>`. Those
         InputPorts are listed in the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s
         `external_input_ports_of_all_input_nodes <Composition.external_input_ports_of_all_input_nodes>` attribute
         and, together with examples of their values, in the OptimizationControlMechanism's `state_feature_values
         <OptimizationControlMechanism.state_feature_values>` attribute. A failure to properly meet these requirements
         produces an error.

  .. _OptimizationControlMechanism_State_Features_Specification:

  The **state_features** argument can be specified using any of the following formats:

  .. _OptimizationControlMechanism_State_Feature_Single_Spec:

  * *Single specification* -- any of the indivdiual specifications described `below
    <OptimizationControlMechanism_State_Feature_Individual_Specs>` can be directly to **state_features**, that is
    then used to construct *all* of the `state_input_ports <OptimizationControlMechanism.state_input_ports>`, one
    for each `external InputPort <Composition_Input_External_InputPorts>` of the `agent_rep
    <OptimizationControlMechanism.agent_rep>`.

  .. _OptimizationControlMechanism_State_Feature_Input_Dict:

  * *Inputs dictionary* -- specifies state_features (entry values) for individual `InputPorts <InputPort>` and/or
    `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>`
    (entry keys). It must conform to the format used to `specify external inputs <Composition_Input_Dictionary>`
    to the `agent_rep <OptimizationControlMechanism.agent_rep>`, in which entries consist of a key specifying either
    an `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>`
    or one of their `external InputPorts <Composition_Input_External_InputPorts>`, and a value that is the source of
    the input that can be any of the forms of individual input specifications listed `below
    <OptimizationControlMechanism_State_Feature_Individual_Specs>`. The format required for the entries can be seen
    using either the `agent_rep <OptimizationControlMechanism.agent_rep>`
    `get_input_format <Composition.get_input_format>` method (for inputs to its `INPUT <NodeRole.INPUT>` <Nodes
    <Composition_Nodes>`) or its `external_input_ports_of_all_input_nodes
    <Composition.external_input_ports_of_all_input_nodes>` (for all of their `external InputPorts
    <Composition_Input_External_InputPorts>`). If a nested Composition is specified (that is, one that is an `INPUT
    <NodeRole.INPUT>` Node of `agent_rep <OptimizationControlMechanism.agent_rep>`), the state_feature assigned to it
    is used to construct the `state_input_ports <OptimizationControlMechanism.state_input_ports>` for *all* of the
    `external InputPorts <Composition_Input_External_InputPorts>` for that nested Composition, and any nested within
    it at all levels of nesting. If any `INPUT <NodeRole.INPUT>` Nodes or their InputPorts are not specified in the
    dictionary, `state_feature_default <OptimizationControlMechanism.state_feature_default>` is assigned as their
    state_feature specification (this includes cases in which some but not all `INPUT <NodeRole.INPUT>` Nodes of a
    nested Composition, or their InputPorts, are specified; any unspecified INPUT Nodes of the corresponding
    Compositions are assigned `state_feature_default <OptimizationControlMechanism.state_feature_default>` as their
    state_feature specification).

  .. _OptimizationControlMechanism_State_Feature_List_Inputs:

  * *List* -- a list of individual state_feature specifications, that can be any of the forms of individual
    input specifications listed `below <OptimizationControlMechanism_State_Feature_Individual_Specs>`. The items
    correspond to all of the `external InputPorts <Composition_Input_External_InputPorts>` of the `agent_rep
    <OptimizationControlMechanism.agent_rep>`, and must be specified in the order they are listed in the
    `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `external_input_ports_of_all_input_nodes
    <Composition.external_input_ports_of_all_input_nodes>` attribute. If the list is incomplete, the remaining
    InputPorts are assigned `state_feature_default <OptimizationControlMechanism.state_feature_default>`
    as their state_feature specification, which by default is *SHADOW_INPUTS* (see `below
    <OptimizationControlMechanism_SHADOW_INPUTS_State_Feature>`. Items can be included in the list that
    have not yet been added to the OptimizationControlMechanism's Composition or its `agent_rep
    <OptimizationControlMechanism.agent_rep>`. However, these must be added before the Composition is executed,
    and must appear in the list in the same position that the InputPorts to which they pertain are listed in
    the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `external_input_ports_of_all_input_nodes
    <Composition.external_input_ports_of_all_input_nodes>` attribute, once construction of the `agent_rep
    <OptimizationControlMechanism.agent_rep>` is complete.

  .. _OptimizationControlMechanism_State_Feature_Set_Inputs:

  * *Set* -- a set of `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of the `agent_rep
    <OptimizationControlMechanism.agent_rep>` that are assigned *SHADOW_INPUTS* as their state_feature
    -- that is, that should receive the same inputs during evaluation as when the Composition of which
    the OptimizationControlMechanism is the `controller <Composition_Controller>` is fully executed
    (see `below <_OptimizationControlMechanism_SHADOW_INPUTS_State_Feature>`). The order of their specification
    does not matter;  however, any of the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `INPUT
    <NodeRole.INPUT>` Nodes that are *not* included in the set are assigned `state_feature_default
    <OptimizationControlMechanism.state_feature_default>` as their state_feature specification.  Note that,
    since the default for `state_feature_default <OptimizationControlMechanism.state_feature_default>` is
    *SHADOW_INPUTS*, unless this is specified otherwise omitting items from a set has no effect (i.e., they
    too are assigned *SHADOW_INPUTS*);  for omitted items to be treated differently, `state_feature_default
    <OptimizationControlMechanism.state_feature_default>` must be specified; for example by assigning it
    ``None`` so that items omitted from the set are assigned their default input value (see `below
    <OptimizationControlMechanism_None_State_Feature>`.

  .. _OptimizationControlMechanism_State_Feature_Individual_Specs:

  * *Individual state_feature specifications* -- any of the specifications listed below can be used singly,
    or in a dict, list or set as described `above <OptimizationControlMechanism_State_Feature_Input_Dict>`,
    to configure `state_input_ports <OptimizationControlMechanism.state_input_ports>`.

    .. _OptimizationControlMechanism_None_State_Feature:

    * *None* -- no `state_input_port <OptimizationControlMechanism.state_input_ports>` is constructed for
      the corresponding `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` InputPort, and its the value
      of its `default variable <Component.defaults>` is used as the input to that InputPort whenever the
      <OptimizationControlMechanism.agent_rep>` is `evaluated <Composition.evaluate>`, irrespective of its input when
      the `agent_rep <OptimizationControlMechanism.agent_rep>` was last executed.

    .. _OptimizationControlMechanism_Numeric_State_Feature:

    * *numeric value* -- create a `state_input_port <OptimizationControlMechanism.state_input_ports>` has
      no `afferent Projections <Mechanism_Base.afferents>`, and uses the specified value as the input to its
      `function <InputPort.function>`, the result of which is assigned to the corresponding value of
      `state_feature_values <OptimizationControlMechanism.state_feature_values>` and provided as the input to
      the corresponding `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` InputPort each time the `agent_rep
      <OptimizationControlMechanism.agent_rep>`  is `evaluated <Composition.evaluate>`. The specified value must
      be compatible with the shape of all of the `external InputPorts <Composition_Input_External_InputPorts>`
      of the `agent_rep <OptimizationControlMechanism.agent_rep>` (see `note
      <OptimizationControlMechanism_State_Features_Shapes>` above).

    .. _OptimizationControlMechanism_SHADOW_INPUTS_State_Feature:

    * *SHADOW_INPUTS* -- create a `state_input_port <OptimizationControlMechanism.state_input_ports>` that `shadows
      the input <InputPort_Shadow_Inputs>` of the InputPort to which the specification is assigned; that is, each time
      `agent_rep <OptimizationControlMechanism.agent_rep>` is  `evaluated <Composition.evaluate>`, the state_input_port
      receives the same input that the corresponding `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` InputPort
      received during the last `TRIAL <TimeScale.TRIAL>` of execution.

    .. _OptimizationControlMechanism_Input_Port_State_Feature:

    * *InputPort specification* -- create a `state_input_port <OptimizationControlMechanism.state_input_ports>` that
      `shadows <InputPort_Shadow_Inputs>` the input to the specified `InputPort`;  that is, each time `agent_rep
      <OptimizationControlMechanism.agent_rep>` is  `evaluated <Composition.evaluate>`, the state_input_port receives
      the same input that the specified InputPort received during the last `TRIAL <TimeScale.TRIAL>` in which the
      Composition for which the OptimizationControlMechanism is the `controller <Composition.controller>` was executed.
      The specification can be any form of `InputPort specification <InputPort_Specification>` for the `InpuPort`
      of any `Mechanism <Mechanism>` that is an `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` in the Composition
      (not limited to the `agent_rep <OptimizationControlMechanism.agent_rep>`).  This includes an
      `InputPort specification dictionary <InputPort_Specification_Dictionary>`, that can be used to configure the
      corresponding `state_input_port <OptimizationControlMechanism.state_input_ports>`, if `Parameters <Parameter>`
      other than its `function <InputPort.function>` need to be specified (which can be done directly using a
      `2-item tuple <OptimizationControlMechanism_Tuple_State_Feature>` specification or the **state_feature_function**
      arg as described `below <OptimizationControlMechanism_State_Feature_Function_Arg>`), such as the InputPort's
      `name <InputPort.name>` or more than a single `afferent Projection <Mechanism_Base.afferents>`.

      .. _OptimizationControlMechanism_INPUT_Node_Specification:

      .. note::
         Only the `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` of a `nested Composition <Composition_Nested>`
         can be shadowed.  Therefore, if the Composition that an OptimizationControlMechanism controls contains any
         nested Compositions, only its `INPUT <NodeRole.INPUT>` `Nodes <Composition_Nodes>` can be specified for
         shadowing in the **state_features** argument of the OptimizationControlMechanism's constructor.

      .. hint::
         Shadowing the input to a Node of a `nested Composition <Composition_Nested>` that is not an `INPUT
         <NodeRole.INPUT>` Node of that Composition can be accomplished in one or of two ways, by: a) assigning it
         `INPUT <NodeRole.INPUT>` as a `required NodeRole <Composition_Node_Role_Assignment>` where it is added to
         the nested Composition; and/or b) adding an additional Node to that Composition that shadows the desired one
         (this is allowed *within* the *same* Composition), and is assigned as an `OUTPUT <NodeRole.OUTPUT>` Node of
         that Composition, the `OutputPort` of which which can then be specified in the **state_features** argument of
         the OptimizationControlMechanism's constructor (see below).

      .. technical_note::
        The InputPorts specified as state_features are designated as `internal_only <InputPort.internal_only>` = `True`.

    .. _OptimizationControlMechanism_Output_Port_State_Feature:

    * *OutputPort specification* -- create a `state_input_port <OptimizationControlMechanism.state_input_ports>` that
      receives a `MappingProjection` from the specified `OutputPort`;  that is, each time `agent_rep
      <OptimizationControlMechanism.agent_rep>` is  `evaluated <Composition.evaluate>`, the state_input_port receives
      the `value <OutputPort.value>` of the specified OutputPort after the last `TRIAL <TimeScale.TRIAL>` in which the
      Composition for which the OptimizationControlMechanism is the `controller <Composition.controller>` was executed.
      The specification can be any form of `OutputPort specification <OutputPort_Specification>` for any `OutputPort`
      of a `Mechanism <Mechanism>` in the Composition (not limited to the `agent_rep
      <OptimizationControlMechanism.agent_rep>`.

    .. _OptimizationControlMechanism_Mechanism_State_Feature:

    * *Mechanism* -- create a `state_input_port <OptimizationControlMechanism.state_input_ports>` that `shadows
      <InputPort_Shadow_Inputs>` the input to the `primary InputPort <InputPort_Primary>` of the specified Mechanism
      (this is the same as explicitly specifying the Mechanism's  input_port, as described `above
      <OptimizationControlMechanism_Input_Port_State_Feature>`). If the Mechanism is in a `nested Composition
      <Composition_Nested>`, it must be an `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of that Composition
      (see `note <OptimizationControlMechanism_INPUT_Node_Specification>` above).  If the Mechanism's `OutputPort`
      needs to be used, it must be specified explicitly (as described `above
      <OptimizationControlMechanism_Output_Port_State_Feature>`).

      .. note::
         The use of a Mechanism to specify the shadowing of its `primary InputPort <InputPort_Primary>` is unique to
         its specification in the **state_features** argument of an OptimizationControlMechanism, and differs from the
         ordinary usage where it specifies a Projection from its `primary OutputPort <OutputPort_Primary>` (see
         `InputPort specification <InputPort_Projection_Source_Specification>`).  This difference extends to the use
         of a Mechanism in the *PROJECTIONS* entry of an `InputPort specification dictionary
         <InputPort_Specification_Dictionary>` used in the **state_features** argument, where there too
         it designates shadowing of its `primary InputPort <InputPort_Primary>` rather than a `Projection` from its
         `primary OutputPort <OutputPort_Primary>`.

    .. _OptimizationControlMechanism_Tuple_State_Feature:

    * *2-item tuple* -- the first item must be any of the forms of individual state_feature specifications
      described `above <OptimizationControlMechanism_State_Feature_Individual_Specs>`, and the second
      item must be a `Function`, that is assigned as the `function <InputPort.function>` of the corresponding
      `state_input_port <OptimizationControlMechanism.state_input_ports>`; this takes precedence over
      any other state_feature_function specifications (e.g., in an `InputPort specification dictionary
      <InputPort_Specification_Dictionary>` or the **state_feature_function** argument of the
      OptimizationControlMechanism's constructor; see `state_feature_function
      <OptimizationControlMechanism_State_Feature_Function_Arg>` for additional details).

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
  <OptimizationControlMechanism_State_Feature_Input_Dict>`, with the following exception: if a
  `Mechanism` is specified, its `primary OutputPort <OutputPort_Primary>` is used (rather than
  shadowing its primary InputPort), since that is more typical usage, and there are no assumptions
  made about the state features of a `CompositionFunctionApproximator` (as there are about a Composition
  as `agent_rep <OptimizationControlMechanism.agent_rep>`); if the input to the Mechanism *is* to be
  `shadowed <InputPort_Shadow_Inputs>`, then its InputPort must be specified explicitly (as described
  `above <OptimizationControlMechanism_Input_Port_State_Feature>`).

|

.. _OptimizationControlMechanism_State_Feature_Function_Arg:

* **state_feature_function** -- specifies a `function <InputPort.function>` to be used as the default
  function for `state_input_ports <OptimizationControlMechanism.state_input_ports>`. This is assigned as
  the `function <InputPort.function>` to any state_input_ports for which *no other* `Function` is specified --
  that is, in either an `InputPort specification dictionary <InputPort_Specification_Dictionary>` or a `2-item tuple
  <OptimizationControlMechanism_Tuple_State_Feature>` in the **state_features** argument (see `state_features
  <OptimizationControlMechanism_State_Features_Arg>`).  If either of the latter is specified, they override
  the specification in **state_feature_function**.  If **state_feature_function** is *not* specified, then
  `LinearCombination` (the standard default `Function` for an `InputPort`) is assigned to any `state_input_ports
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
<OptimizationControlMechanism_State_Input_Ports>`) and `control_allocation <ControlMechanism.control_allocation>`.
These are used by the `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method,
the results of which are combined with the `costs <ControlMechanism_Costs_NetOutcome>` associated with the
`control_allocation <ControlMechanism.control_allocation>`, to evaluate the `net_outcome
<ControlMechanism.net_outcome>` for that state. The current state is listed in the OptimizationControlMechanism's
`state <OptimizationControlMechanism.state>` attribute, and `state_dict <OptimizationControlMechanism.state_dict>`
contains the Components associated with each value of `state <OptimizationControlMechanism.state>`.

COMMENT:
          > Attributes that pertain to the state of the agent_rep for a given evaluation:
              state_feature_specs: a list of the sources for state_feature_values, one for each InputPort of each INPUT Node of the agent_rep
                                   (at all levels of nesting);  None for any sources not specified in **state_features**
                                   (corresponding InputPorts are are assigned their default input when agent_rep.evaluate() is executed).
              state_feature_values: a dict with entries for each item specified in **state_features** arg of constructor,
                                    in which the key of each entry is an InputPort of an INPUT Node of agent_rep (at any level of nesting)
                                    and the value is the current value of the corresponding state_input_port;  the dict is suitable for
                                    use as the **predicted_inputs** or **feature_values** arg of the agent_rep's evaluate() method
                                    (depending on whether the agent_rep is a Composition or a CFA);
                                    note:  there are not entries for InputPorts of INPUT Nodes that are not specified in **state_features**;
                                           those are assigned either their default input values (LINK XXX) or the shadowed input of
                                           the corresponding INPUT Node InputPort (LINK XXX), depending on how **state_features** was formatted;
                                           (see LINK XXX for details of formatting).
              state_features: a dict with entries corresponding to each item of state_feature_specs,
                              the keys of which are InputPorts of the INPUT Nodes of the agent_rep,
                              and values of which are the corresponding state_feature_specs
                              (i.e., sources of input for those InputPorts when evaluate() is called);
              control_allocation: a list of the current values of the OCM's control_signals that are used to modulate the Parameters
                                  specified for control when the agent_rep's evaluate() method is called;
              state: a list of the values of the current state, starting with state_feature_values and ending with
                     control_allocations
              state_dict: a dictionary with entries for each state_feature and ControlSignal, keys?? values??
                                   their source/destination, and their current values
          > Constituents of state specifications:
            - agent_rep_input_port:  an InputPort of an INPUT Node of the agent_rep,
                                     that will receive a value from state_feature_values passed to agent_rep.evaluate()
            - source: the source of the input to an agent_rep_input_port,
                      that sends a Projection to the corresponding state_input_port

          > Relationship of numeric spec to ignoring it (i.e. assigning it None):
               allows specification of value as input *just* for simulations (i.e., agent_rep_evaluate)
               and not normal execution of comp

COMMENT

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

.. _OptimizationControlMechanism_State_Input_Ports:

*state_input_ports*
~~~~~~~~~~~~~~~~~~~

The `state_input_ports <OptimizationControlMechanism.state_input_ports>` receive `Projections <Projection>`
from the Components specified as the OptimizationControlMechanism's `state_features
<OptimizationControlMechanism_State_Features_Arg>`, the values of which are assigned as the `state_feature_values
<OptimizationControlMechanism.state_feature_values>`, and conveyed to the `agent_rep
<OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method when it is `executed
<OptimizationControlMechanism_Execution>`.  The OptimizationControlMechanism has a `state_input_port
<OptimizationControlMechanism.state_input_ports>` for every specification in the **state_features** arg of its
constructor (see `above <OptimizationControlMechanism_State_Features_Arg>`).

COMMENT:
OLD
If the `agent_rep is a Composition <OptimizationControlMechanism_Agent_Rep_Composition>`, then the
OptimizationControlMechanism has a state_input_port for every specification in the **state_features** arg of its
constructor (see `above <OptimizationControlMechanism_State_Features_Arg>`). If the `agent_rep is a
CompositionFunctionApproximator <OptimizationControlMechanism_Agent_Rep_CFA>`, then the OptimizationControlMechanism
has a state_input_port that receives a Projection from each Component specified in the **state_features** arg of its
constructor.
COMMENT

COMMENT:
In either, case the the `values <InputPort.value>` of the
`state_input_ports <OptimizationControlMechanism.state_input_ports>` are assigned to the `state_feature_values
<OptimizationControlMechanism.state_feature_values>` attribute that is used, in turn, by the
OptimizationControlMechanism's `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method to
estimate or predict the `net_outcome <ControlMechanism.net_outcome>` for a given `control_allocation
<ControlMechanism.control_allocation>` (see `OptimizationControlMechanism_Execution`).

State features can be of two types:

* *Input Features* -- these are values that either shadow the input received by an `InputPort` of a `Mechanisms
  <Mechanism>` in the `Composition` for which the OptimizationControlMechanism is a `controller
  <Composition.controller>` (irrespective of whether that is the OptimizationControlMechanism`s `agent_rep
  <OptimizationControlMechanism.agent_rep>`). They are implemented as `shadow InputPorts <InputPort_Shadow_Inputs>`
  (see `OptimizationControlMechanism_SHADOW_INPUTS_State_Feature` for specification) that receive a `Projection`
  from the same source as the Mechanism being shadowed.
..
* *Output Features* -- these are the `value <OutputPort.value>` of an `OutputPort` of a `Mechanism <Mechanism>` in
  the `Composition` for which the OptimizationControlMechanism is a `controller <Composition.controller>` (again,
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
<ControlMechanism.objective_mechanism>` is used to evaluate the outcome of executing its `agent_rep
<OptimizationControlMechanism.agent_rep>` for a given `state <OptimizationControlMechanism_State>`. This passes
the result to the OptimizationControlMechanism's *OUTCOME* InputPort, that is placed in its `outcome
<ControlMechanism.outcome>` attribute.

    .. note::
        An OptimizationControlMechanism's `objective_mechanism <ControlMechanism.objective_mechanism>` and the `function
        <ObjectiveMechanism.function>` of that Mechanism, are distinct from and should not be confused with the
        `objective_function <OptimizationFunction.objective_function>` parameter of the OptimizationControlMechanism's
        `function <OptimizationControlMechanism.function>`.  The `objective_mechanism
        <ControlMechanism.objective_mechanism>`\\'s `function <ObjectiveMechanism.function>` evaluates the `outcome
        <ControlMechanism.outcome>` of processing without taking into account the `costs <ControlMechanism.costs>` of
        the OptimizationControlMechanism's `control_signals <OptimizationControlMechanism.control_signals>`.  In
        contrast, its `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method, which is assigned
        as the `objective_function` parameter of its `function <OptimizationControlMechanism.function>`, takes the
        `costs <ControlMechanism.costs>` of the OptimizationControlMechanism's `control_signals
        <OptimizationControlMechanism.control_signals>` into account when calculating the `net_outcome` that it
        returns as its result.

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
    FUNCTION, INPUT_PORT, INTERNAL_ONLY, NAME, OPTIMIZATION_CONTROL_MECHANISM, NODE, OWNER_VALUE, PARAMS, PORT, \
    PROJECTIONS, SHADOW_INPUTS, VALUE
from psyneulink.core.globals.parameters import Parameter, check_user_specified
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.registry import rename_instance_in_registry
from psyneulink.core.globals.sampleiterator import SampleIterator, SampleSpec
from psyneulink.core.globals.utilities import convert_to_list, ContentAddressableList, is_numeric
from psyneulink.core.llvm.debug import debug_env

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError',
    'AGENT_REP', 'STATE_FEATURES', 'STATE_FEATURE_FUNCTION', 'RANDOMIZATION_CONTROL_SIGNAL', 'NUM_ESTIMATES',
    'DEFERRED_STATE_INPUT_PORT_PREFIX', 'NUMERIC_STATE_INPUT_PORT_PREFIX', 'SHADOWED_INPUT_STATE_INPUT_PORT_PREFIX',
    'INPUT_SOURCE_FOR_STATE_INPUT_PORT_PREFIX'
]

# constructor arguments
AGENT_REP = 'agent_rep'
STATE_FEATURES = 'state_features'
STATE_FEATURE_FUNCTION = 'state_feature_function'
RANDOMIZATION_CONTROL_SIGNAL = 'RANDOMIZATION_CONTROL_SIGNAL'
RANDOM_VARIABLES = 'random_variables'
NUM_ESTIMATES = 'num_estimates'

# state_input_port names
NUMERIC_STATE_INPUT_PORT_PREFIX = "NUMERIC INPUT FOR "
INPUT_SOURCE_FOR_STATE_INPUT_PORT_PREFIX = "SOURCE OF INPUT FOR "
SHADOWED_INPUT_STATE_INPUT_PORT_PREFIX = "SHADOWED INPUT OF "
# SHADOWED_INPUT_STATE_INPUT_PORT_PREFIX = "Shadowed input of "
DEFERRED_STATE_INPUT_PORT_PREFIX = 'DEFERRED INPUT NODE InputPort '

def _state_input_port_name(source_port_name, agent_rep_input_port_name):
    return f"INPUT FROM {source_port_name} FOR {agent_rep_input_port_name}"

def _shadowed_state_input_port_name(shadowed_port_name, agent_rep_input_port_name):
    return f"{SHADOWED_INPUT_STATE_INPUT_PORT_PREFIX}{shadowed_port_name} FOR {agent_rep_input_port_name}"

def _numeric_state_input_port_name(agent_rep_input_port_name):
    return f"{NUMERIC_STATE_INPUT_PORT_PREFIX}{agent_rep_input_port_name}"

def _deferred_agent_rep_input_port_name(node_name, agent_rep_name):
    # return f"{DEFERRED_STATE_INPUT_PORT_PREFIX}{node_name} OF {agent_rep_name}"
    return f"{DEFERRED_STATE_INPUT_PORT_PREFIX}OF {agent_rep_name} ({node_name})"

def _deferred_state_feature_spec_msg(spec_str, comp_name):
    return f"{spec_str} NOT (YET) IN {comp_name}"

def _not_specified_state_feature_spec_msg(spec_str, comp_name):
    return f"NO SPECIFICATION (YET) FOR {spec_str} IN {comp_name}"

def _state_feature_values_getter(owning_component=None, context=None):
    """Return dict {agent_rep INPUT Node InputPort: value} suitable for **predicted_inputs** arg of evaluate method.
    Only include entries for sources specified in **state_features**, corresponding to OCM's state_input_ports;
       default input will be assigned for all other INPUT Node InputPorts in composition._instantiate_input_dict().
    """

    # FIX: REFACTOR TO DO VALIDATIONS ON INIT AND INITIAL RUN-TIME CHECK
    #      AND SIMPLY RETURN VALUES (WHICH SHOULD ALL BE ASSIGNED BY RUN TIME) DURING EXECUTION / SIMULATIONS

    # If no state_input_ports return empty list
    if (not owning_component.num_state_input_ports):
        return {}

    # Get sources specified in **state_features**
    specified_state_features = [spec for spec in owning_component.state_feature_specs if spec is not None]
    # Get INPUT Node InputPorts for sources specified in **state_features**
    specified_INPUT_Node_InputPorts = [port for port, spec
                                       in zip(owning_component._specified_INPUT_Node_InputPorts_in_order,
                                              owning_component.state_feature_specs)
                                       if spec is not None]
    num_agent_rep_input_ports = len(owning_component._get_agent_rep_input_receivers())

    assert len(specified_state_features) == \
           len(specified_INPUT_Node_InputPorts) == \
           owning_component.num_state_input_ports

    # Construct state_feature_values dict
    state_feature_values = {}
    for i in range(owning_component.num_state_input_ports):
        key = specified_INPUT_Node_InputPorts[i]
        state_input_port = owning_component.state_input_ports[i]
        spec = specified_state_features[i]

        # Get key
        if not isinstance(key, InputPort):
            # INPUT Node InputPort is not fully or properly specified
            key = _deferred_agent_rep_input_port_name((key or str(i - num_agent_rep_input_ports)),
                                                  owning_component.agent_rep.name)
        elif key not in owning_component._get_agent_rep_input_receivers():
            # INPUT Node InputPort is not (yet) in agent_rep
            key = _deferred_agent_rep_input_port_name(key.full_name, owning_component.agent_rep.name)

        # Get state_feature_value
        if spec is None:
            # state_feature not specified; default input will be assigned in _instantiate_input_dict()
            state_feature_value = _not_specified_state_feature_spec_msg((key if isinstance(key, str) else key.full_name),
                                                                       owning_component.composition.name)
        elif is_numeric(spec):
            # if spec is numeric, use that
            state_feature_value = state_input_port.function(spec)
        elif (hasattr(owning_component, 'composition')
              and not owning_component.composition._is_in_composition(spec)):
            # spec is not in ocm.composition
            state_feature_value = _deferred_state_feature_spec_msg(spec.full_name, owning_component.agent_rep.name)
        elif state_input_port.parameters.value._get(context) is not None:
            # if state_input_port returns a value, use that
            state_feature_value = state_input_port.parameters.value._get(context)
        else:
            # otherwise use state_input_port's default input value
            state_feature_value = state_input_port.default_input_shape

        state_feature_values[key] = state_feature_value

    return state_feature_values


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

    state_feature_default : same as state_features : default None
        specifies the default used if a state_feature is not otherwise specified for the `InputPort` of an
        `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of `agent_rep <OptimizationControlMechanism.agent_rep>`.
        (see `state_feature_default <OptimizationControlMechanism.state_feature_default>` and
        `state_features <OptimizationControlMechanism_State_Features_Arg>` for additional details).

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
        dictionary in which keys are all `external InputPorts <Composition_Input_External_InputPorts>` for `agent_rep
        <OptimizationControlMechanism.agent_rep>`, and values are the sources of their input specified in
        **state_features**. These are provided as the inputs to `state_input_ports
        <OptimizationControlMechanism.state_input_ports>`, the `values <InputPort.value>`
        of which are assigned to `state_feature_values <OptimizationControlMechanism.state_feature_values>` and
        provided to the `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>`
        method when it is executed (see `state_features <OptimizationControlMechanism_State_Features_Arg>` and
        `OptimizationControlMechanism_State_Input_Ports` for additional details).

    state_feature_default : Mechanism, InputPort, OutputPort, Projection, dict, SHADOW_INPUTS, numeric value
        determines the default used if the state_feature (i.e. source) is not otherwise specified for the `InputPort` of
        an `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of `agent_rep <OptimizationControlMechanism.agent_rep>`.
        If it is None, then no corresponding `state_input_port <OptimizationControlMechanism.state_input_ports>`
        is created for that InputPort, and its `default variable <Component.defaults>` is used as its input when the
        `agent_rep <OptimizationControlMechanism.agent_rep>`\\'s `evaluate <Composition.evaluate>` method is executed
        (see `state_features <OptimizationControlMechanism_State_Features_Arg>` for additional details).

    state_feature_values : 2d array
        a dict containing the current values assigned as the input to the InputPorts of the `INPUT <NodeRole.INPUT>`
        `Nodes <Composition_Nodes>` of the `agent_rep <OptimizationControlMechanism.agent_rep>` when its `evaluate
        <Composition.evaluate>` method is executed.  For each such InputPort, if a `state_feature
        <OptimizationControlMechanism_State_Features_Arg>` has been specified for it, then its value in
        state_feature_values is the `value <InputPort.value>` of the corresponding `state_input_port
        <OptimizationControlMechanism.state_input_ports>`.  There are no entries for InputPorts for which the
        **state_features** specification is ``None`` or it has not been otherwise specified;  for those InputPorts,
        their `default_variable <Component.default_variable>` is assigned directly as their input when `agent_rep
        <OptimizationControlMechanism.agent_rep>` is `evaluated <Composition.evaluate>` (see
        `OptimizationControlMechanism_State_Input_Ports` for additional details).

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
        <Composition.evaluate>` method (see `OptimizationControlMechanism_State_Input_Ports` for additional details).

    num_state_input_ports : int
        contains the number of `state_input_ports <OptimizationControlMechanism.state_input_ports>`.

    outcome_input_ports : ContentAddressableList
        lists the OptimizationControlMechanism's `OutputPorts <OutputPort>` that receive `Projections <Projection>`
        from either its `objective_mechanism <ControlMechanism.objective_mechanism>` or the Components listed in
        its `monitor_for_control <ControlMechanism.monitor_for_control>` attribute, the values of which are used
        to compute the `net_outcome <ControlMechanism.net_outcome>` of executing the `agent_rep
        <OptimizationControlMechanism.agent_rep>` in a given `OptimizationControlMechanism_State`
        (see `objective_mechanism <OptimizationControlMechanism_ObjectiveMechanism>` and `outcome_input_ports
        <OptimizationControlMechanism_Outcome>` for additional details).

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

                state_feature_specs
                    This is for internal use only, including population of the state_features property
                    (see `state_features <OptimizationControlMechanism.state_features>`)

                    :default value: SHADOW_INPUTS
                    :type: ``dict``

                state_feature_default_spec
                    This is a shell parameter to validate its assignment and explicity user specification of None
                    to override Parameter default;  its .spec attribute is assigned to the user-facing
                    self.state_feature_default (see `state_feature_default <Optimization.state_feature_default>`).

                    :default value: SHADOW_INPUTS
                    :type:

                state_feature_function
                    see `state_feature_function <OptimizationControlMechanism_Feature_Function>`

                    :default value: None
                    :type:

                state_input_ports
                    see `state_input_ports <OptimizationControlMechanism.state_input_ports>`

                    :default value: None
                    :type:  ``list``
        """
        agent_rep = Parameter(None, stateful=False, loggable=False, pnl_internal=True, structural=True)
        state_input_ports = Parameter(None, reference=True, stateful=False, loggable=False, read_only=True)
        state_feature_specs = Parameter(SHADOW_INPUTS, stateful=False, loggable=False, read_only=True,
                                        structural=True, parse_spec=True)
        state_feature_default_spec = Parameter(SHADOW_INPUTS, stateful=False, loggable=False, read_only=True,
                                               structural=True)
        state_feature_function = Parameter(None, reference=True, stateful=False, loggable=False)
        state_feature_values = Parameter(None,getter=_state_feature_values_getter,
                                         user=False,  pnl_internal=True, read_only=True)
        outcome_input_ports_option = Parameter(CONCATENATE, stateful=False, loggable=False, structural=True)
        function = Parameter(GridSearch, stateful=False, loggable=False)
        search_function = Parameter(None, stateful=False, loggable=False)
        search_space = Parameter(None, read_only=True)
        search_termination_function = Parameter(None, stateful=False, loggable=False)
        comp_execution_mode = Parameter('Python', stateful=False, loggable=False, pnl_internal=True)
        search_statefulness = Parameter(True, stateful=False, loggable=False)

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

        def _validate_state_feature_default_spec(self, state_feature_default):
            if not (isinstance(state_feature_default, (InputPort, OutputPort, Mechanism))
                    or state_feature_default in {SHADOW_INPUTS}
                    or is_numeric(state_feature_default)
                    or state_feature_default is None):
                return f"must be an InputPort, OutputPort, Mechanism, Composition, SHADOW_INPUTS or a list or array " \
                       f"with a shape appropriate for all of the INPUT Nodes or InputPorts to which it will be applied."

    @handle_external_context()
    @check_user_specified
    @tc.typecheck
    def __init__(self,
                 agent_rep=None,
                 state_features: tc.optional((tc.any(str, Iterable, InputPort,
                                                     OutputPort, Mechanism)))=SHADOW_INPUTS,
                 # state_feature_default=None,
                 state_feature_default: tc.optional((tc.any(str, Iterable,
                                                            InputPort, OutputPort,Mechanism)))=SHADOW_INPUTS,
                 state_feature_function: tc.optional(tc.optional(tc.any(dict, is_function_type)))=None,
                 function=None,
                 num_estimates=None,
                 random_variables=None,
                 initial_seed=None,
                 same_seed_for_all_allocations=None,
                 num_trials_per_estimate=None,
                 search_function: tc.optional(tc.optional(tc.any(is_function_type)))=None,
                 search_termination_function: tc.optional(tc.optional(tc.any(is_function_type)))=None,
                 search_statefulness=None,
                 context=None,
                 **kwargs):
        """Implement OptimizationControlMechanism"""

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
                self._init_args['state_feature_default_spec'] = state_feature_default

                # Flag for deferred initialization
                self.initialization_status = ContextFlags.DEFERRED_INIT
                return
            # If constructor is called internally (i.e., for controller of Composition),
            # agent_rep needs to be specified
            else:
                assert False, f"PROGRAM ERROR: 'agent_rep' arg should have been specified " \
                              f"in internal call to constructor for {self.name}."

        # If agent_rep is a Composition, but there are more state_features than INPUT Nodes,
        #     defer initialization until they are added
        elif agent_rep.componentCategory=='Composition':
            from psyneulink.core.compositions.composition import NodeRole
            if (state_features
                    and len(convert_to_list(state_features)) > len(agent_rep.get_nodes_by_role(NodeRole.INPUT))):
                # Temporarily name InputPort
                self._assign_deferred_init_name(self.__class__.__name__)
                # Store args for deferred initialization
                self._store_deferred_init_args(**locals())
                self._init_args['state_feature_specs'] = state_features
                self._init_args['state_feature_default_spec'] = state_feature_default
                # Flag for deferred initialization
                self.initialization_status = ContextFlags.DEFERRED_INIT
                return

        super().__init__(
            agent_rep=agent_rep,
            state_feature_specs=state_features,
            state_feature_default_spec=state_feature_default,
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
        The OptimizationControlMechanism's outcome_input_ports are instantiated by
            ControlMechanism._instantiate_input_ports in the call to super().

        InputPorts are constructed for **state_features** by calling _parse_state_feature_specs()
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
        `OptimizationControlMechanism_State_Input_Ports` for additional details.
        """

        # FIX: 11/3/21 :
        #    ADD CHECK IN _parse_state_feature_specs() THAT IF A NODE RATHER THAN InputPort IS SPECIFIED,
        #    ITS PRIMARY IS USED (SEE SCRATCH PAD FOR EXAMPLES)
        if not self.state_feature_specs:
            # If agent_rep is CompositionFunctionApproximator, warn if no state_features specified.
            # Note: if agent rep is Composition, state_input_ports and any state_feature_function specified
            #       are assigned in _update_state_input_ports_for_controller.
            if self.agent_rep_type == COMPOSITION_FUNCTION_APPROXIMATOR:
                warnings.warn(f"No '{STATE_FEATURES}' specified for use with `agent_rep' of {self.name}")

        # Implement any specified state_features
        state_input_ports_specs = self._parse_state_feature_specs(context)
        # Note:
        #   if state_features were specified and agent_rep is a CompositionFunctionApproximator,
        #   assume they are OK (no way to check their validity for agent_rep.evaluate() method, and skip assignment

        # Pass state_input_ports_sepcs to ControlMechanism for instantiation and addition to OCM's input_ports
        super()._instantiate_input_ports(state_input_ports_specs, context=context)

        # Assign to self.state_input_ports attribute
        start = self.num_outcome_input_ports # FIX: 11/3/21 NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
        stop = start + len(state_input_ports_specs) if state_input_ports_specs else 0
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
                      f"{'it is' if singular else 'they are'} added will generate an error ."
            if enforce:
                raise OptimizationControlMechanismError(message)
            else:
                warnings.warn(message)

    # FIX: 1/29/22 - REFACTOR TO SUPPORT InportPort SPECIFICATION DICT FOR MULT. PROJS. TO STATE_INPUT_PORT
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

        - dict {INPUT Node: source or None, INPUT Node or InputPort: source or None...}:
            - every key must be an INPUT Node of agent_rep or an INPUT Node of a nested Composition within it that is
                itself an INPUT Node of its enclosing Composition, or the external InputPort of one, at any level of
                nesting;
            - if a Mechanism is specified as a key, construct a state_input_port for each of its external InputPorts,
                and assign the value of the dict entry as the source for all of them;
            - if a Composition is specified as a key, construct a state_input_port for each external InputPort of each
                of its INPUT Nodes, and those of any Compositions nested within it at all levels of nesting,
                and assign the the value of the dict entry as the source for all of them;
            - for INPUT Nodes not specified or assigned None as their value, assign corresponding entries in
                state_feature_specs as state_feature_default
            - if only one or some of the INPUT Nodes of a nested Composition are specified,
                for the remaining ones assign the corresponding entries in state_feature_specs as state_feature_default
            - if None is specified, don't construct a state_input_port
        - list [source, None, source...]: specifies source specs for INPUT Node external InputPorts:
            - must be listed in same order as *expanded* list of agent_rep INPUT Node external InputPorts to which they
              correspond (i.e., nested Compositions that are INPUT Nodes replaced by their INPUT Nodes,
              for all levels of nesting);
            - if there are fewer sources listed than INPUT Node external InputPorts, assign state_feature_default to
                the entries in state_feature_specs corresponding to the remaining INPUT Node external InputPorts
            - if there more sources listed than INPUT Nodes, leave the excess ones, and label them as
               'EXPECT <specified INPUT Node InputPort name>' for later resolution (see below).

        - set {INPUT Node, Input Node...}: specifies INPUT Nodes to be shadowed
            - every item must be an INPUT Node of agent_rep or an INPUT Node of a nested Composition within it that
                is itself an INPUT Node of its enclosing Composition, at any level of nesting;
            - if a Composition is specified, construct a state_input_port for each of its INPUT Node extenal InputPorts,
                and those of any Compositions nested within it at all levels of nesting, each of which shadows the
                input of the corresponding INPUT Node (see _InputPort_Shadow_Inputs).
            - if only one or some of the INPUT Nodes of a nested Composition are specified, use state_feature_default.

        IMPLEMENTATION NOTE: this is a legacy format for consistency with generic specification of shadowing inputs
        - SHADOW_INPUTS dict {"SHADOW_INPUTS":[shadowable input, None, shadowable input...]}:
            - all items must be a Mechanism (or one of its external InputPorts) that is an INPUT Node of agent_rep or
                 of a nested Composition within it that is itself an INPUT Node;
            - must be listed in same order as *expanded* list of agent_rep INPUT Nodes to which they correspond
                (see list format above);
            - construct a state_input_port for each non-None spec, and assign it a Projection that shadows the spec.
                (see _InputPort_Shadow_Inputs).

        If shadowing is specified for an INPUT Node InputPort, set INTERNAL_ONLY to True in entry of params dict in
            specification dictionary for corresponding state_input_port (so that inputs to Composition are not
            required if the specified source is itself an INPUT Node).

        If an INPUT Node (or one of its external InputPorts) is specified that is not (yet) in agent_rep,
            and/or a source is specified that is not yet in self.composition, warn and defer creating a
            state_input_port;  final check is made, and error(s) generated for unresolved specifications at run time.

        Assign functions specified in **state_feature_function** to InputPorts for all state_features

        Return list of InputPort specification dictionaries for state_input_ports
        """

        from psyneulink.core.compositions.composition import Composition, NodeRole
        # Agent rep's input Nodes and their names
        agent_rep_input_ports = self._get_agent_rep_input_receivers(type=PORT)
        self._specified_INPUT_Node_InputPorts_in_order = []

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
            # FIX: 3/18/22 - ADD TESTS FOR THESE (in test_deferred_init or test_partial_deferred_init()?)
            if (isinstance(self.state_feature_specs, set)
                    or isinstance(self.state_feature_specs, dict) and SHADOW_INPUTS not in self.state_feature_specs):
                # Dict and set specs reference Nodes that are not yet in agent_rep
                warnings.warn(f"Nodes are specified in the {STATE_FEATURES}' arg for '{self.name}' that are not "
                              f"(yet) in its its {AGENT_REP} ('{self.agent_rep.name}'). They must all be assigned "
                              f"to it before the Composition is executed'.  It is generally safer to assign all "
                              f"Nodes to the {AGENT_REP} of a controller before specifying its '{STATE_FEATURES}'.")
            else:
                # List and SHADOW_INPUTS specs are dangerous before agent_rep has been fully constructed
                warnings.warn(f"The '{STATE_FEATURES}' arg for '{self.name}' has been specified before any Nodes have "
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

        def get_port_for_mech_spec(spec:Union[Port,Mechanism]):
            """Return port for Mechanism specified as state_feature
            This is used to override the standard interpretation of a Mechanism in an InputPort specification:
               - return Primary InputPort of Mechanism (to be shadowed) if agent_rep is Composition
               - return Primary OutputPort of Mechanism (standard behavior) if agent_rep is a CFA
            """
            # assert isinstance(mech, Mechanism), \
            #     f"PROGRAM ERROR: {mech} should be Mechanism in call to get_port_for_mech_spec() for '{self.name}'"
            if isinstance(spec, Port):
                return spec
            if self.agent_rep_type == COMPOSITION:
                # FIX: 11/29/21: MOVE THIS TO _parse_shadow_inputs
                #      (ADD ARG TO THAT FOR DOING SO, OR RESTRICT TO InputPorts IN GENERAL)
                if len(spec.input_ports)!=1:
                    raise OptimizationControlMechanismError(
                        f"A Mechanism ({spec.name}) is specified to be shadowed in the '{STATE_FEATURES}' arg "
                        f"for '{self.name}', but it has more than one {InputPort.__name__}; a specific one of its "
                        f"{InputPort.__name__}s must be specified to be shadowed.")
                return spec.input_port
            else:
                # agent_rep is a CFA
                return spec.output_port

        # PARSE SPECS  ------------------------------------------------------------------------------------------
        # Generate parallel lists of state feature specs (for sources of inputs)
        #                            and INPUT Nodes to which they (if specified in dict or set format)
        def _parse_specs(state_feature_specs, specified_input_ports=None, spec_type="list"):
            """Validate and parse INPUT Node specs assigned to construct state_feature_specs
            Validate number and identity of specs relative to agent_rep INPUT Nodes.
            Assign spec for every INPUT Mechanism (nested) within agent_rep (i.e., for all nested Compositions)
                as entries in state_feature_specs
            Return names for use as state_input_port_names in main body of method
            """

            parsed_feature_specs = []
            num_user_specs = len(state_feature_specs)
            num_specified_ports = len(specified_input_ports)
            num_agent_rep_input_ports = len(agent_rep_input_ports)
            # Total number of specs to be parsed:
            self._num_state_feature_specs = max(num_user_specs, num_agent_rep_input_ports)

            assert num_user_specs == num_specified_ports, f"ALERT: num state_feature_specs != num ports in _parse_spec()"
            # Note: there may be more state_feature_specs (i.e., ones for unspecified input_ports)
            #       than num_specified_ports

            if self.agent_rep_type == COMPOSITION:

                # FIX: 3/18/22 - THESE SEEM DUPLICATIVE OF _validate_state_features;  JUST CALL THAT HERE?
                #               ALSO, WARNING IS TRIGGERED IF MECHANIMS RATHER THAN ITS INPUT_PORTS ARE SPEC'D
                #              AT THE LEAST, MOVE TO THEIR OWN VALIDATION HELPER METHOD
                # Too FEW specs for number of agent_rep receivers
                if len(self.state_feature_specs) < num_agent_rep_input_ports:
                    warnings.warn(f"There are fewer '{STATE_FEATURES}' specified for '{self.name}' than the number "
                                  f"of {InputPort.__name__}'s for all of the INPUT Nodes of its {AGENT_REP} "
                                  f"('{self.agent_rep.name}'); the remaining inputs will be assigned default values "
                                  f"when '{self.agent_rep.name}`s 'evaluate' method is executed. If this is not the "
                                  f"desired behavior, use its get_inputs_format() method to see the format for its "
                                  f"inputs.")

                # Too MANY specs for number of agent_rep receivers
                if num_user_specs > num_agent_rep_input_ports:
                    # specs_not_in_agent_rep = [f"'{spec.name if isinstance(spec, Mechanism) else spec.owner.name}'"
                    #                           for spec in self._get_specs_not_in_agent_rep(state_feature_specs)]
                    specs_not_in_agent_rep = \
                        [f"'{spec.name if isinstance(spec,(Mechanism, Composition)) else spec.owner.name}'"
                         for spec in self._get_specs_not_in_agent_rep(user_specs)]

                    if specs_not_in_agent_rep:
                        spec_type = ", ".join(specs_not_in_agent_rep)
                        warnings.warn(
                            f"The '{STATE_FEATURES}' specified for {self.name} is associated with a number of "
                            f"{InputPort.__name__}s ({len(state_feature_specs)}) that is greater than for the "
                            f"{InputPort.__name__}s of the INPUT Nodes ({num_agent_rep_input_ports}) for the "
                            f"Composition assigned as its {AGENT_REP} ('{self.agent_rep.name}'), which includes "
                            f"the following that are not (yet) in '{self.agent_rep.name}': {spec_type}. Executing "
                            f"{self.name} before the additional item(s) are added as (part of) INPUT Nodes will "
                            f"generate an error.")
                    else:
                        warnings.warn(
                            f"The '{STATE_FEATURES}' specified for {self.name} is associated with a number of "
                            f"{InputPort.__name__}s ({len(state_feature_specs)}) that is greater than for the "
                            f"{InputPort.__name__}s of the INPUT Nodes ({num_agent_rep_input_ports}) for the "
                            f"Composition assigned as its {AGENT_REP} ('{self.agent_rep.name}'). Executing "
                            f"{self.name} before the additional item(s) are added as (part of) INPUT Nodes will "
                            f"generate an error.")

            # Nested Compositions not allowed to be specified in a list spec
            nested_comps = [node for node in state_feature_specs if isinstance(node, Composition)]
            if nested_comps:
                comp_names = ", ".join([f"'{n.name}'" for n in nested_comps])
                raise OptimizationControlMechanismError(
                    f"The '{STATE_FEATURES}' argument for '{self.name}' includes one or more Compositions "
                    f"({comp_names}) in the {spec_type} specified for its '{STATE_FEATURES}' argument; these must be "
                    f"replaced by direct references to the Mechanisms (or their InputPorts) within them to be "
                    f"shadowed.")

            state_input_port_names = []
            for i in range(self._num_state_feature_specs):

                state_input_port_name = None
                state_feature_fct = None

                # FIX: CONSOLIDATE THIS WITH PARSING OF SPEC BELOW
                # AGENT_REP INPUT NODE InputPort
                #    Assign it's name to be used in state_features
                # (and specs for CFA and any Nodes not yet in agent_rep)
                if self.agent_rep_type == COMPOSITION:
                    if i < num_agent_rep_input_ports:
                        # spec is for Input{ort of INPUT Node already in agent_rep
                        #    so get agent_rep_input_port and its name (spec will be parsed and assigned below)
                        agent_rep_input_port = agent_rep_input_ports[i]
                        agent_rep_input_port_name = agent_rep_input_port.full_name
                    else:
                        # spec is for deferred NODE InputPort (i.e., not (yet) in agent_rep)
                        #    so get specified value for spec, for later parsing and assignment (once Node is known)
                        agent_rep_input_port = specified_input_ports[i]
                        # - assign "DEFERRED n" as node name
                        agent_rep_input_port_name = \
                            _deferred_agent_rep_input_port_name(str(i - num_agent_rep_input_ports),
                                                                self.agent_rep.name)
                # For CompositionFunctionApproximator, assign spec as agent_rep_input_port
                else:
                    spec = state_feature_specs[i]
                    agent_rep_input_port = spec
                    agent_rep_input_port_name = spec.full_name if isinstance(spec, Port) else spec.name
                    # Assign state_input_port_name here as won't get done below (i can't be < num_user_specs for CFA)
                    state_input_port_name = f"FEATURE {i} FOR {self.agent_rep.name}"

                # SPEC and state_input_port_name
                # Parse and assign user specifications (note: may be for INPUT Node InputPorts not yet inagent_rep)
                if i < num_user_specs:               # i.e., if num_agent_rep_input_ports < num_user_specs)
                    spec = state_feature_specs[i]
                    # Unpack tuple

                    if isinstance(spec, tuple):
                        state_feature_fct = spec[1]
                        spec = spec[0]

                    # Assign spec and state_input_port name
                    if is_numeric(spec):
                        state_input_port_name = _numeric_state_input_port_name(agent_rep_input_port_name)

                    elif isinstance(spec, (InputPort, Mechanism)):
                        spec_name = spec.full_name if isinstance(spec, InputPort) else spec.input_port.full_name
                        state_input_port_name = _shadowed_state_input_port_name(spec_name,
                                                                                agent_rep_input_port_name)
                    elif isinstance(spec, OutputPort):
                        state_input_port_name = _state_input_port_name(spec.full_name,
                                                                       agent_rep_input_port_name)
                    elif isinstance(spec, Composition):
                        assert False, f"Composition spec ({spec}) made it to _parse_specs for {self.name}."

                    elif spec == SHADOW_INPUTS:
                        # Shadow the specified agent_rep_input_port (Name assigned where shadow input is parsed)
                        spec = agent_rep_input_port
                        state_input_port_name = _shadowed_state_input_port_name(agent_rep_input_port_name,
                                                                                agent_rep_input_port_name)

                    elif isinstance(spec, dict):
                        state_input_port_name = spec[NAME] if NAME in spec else f"INPUT FOR {agent_rep_input_port_name}"
                        # tuple specification of function (assigned above) overrides dictionary specification
                        if state_feature_fct is None:
                            if FUNCTION in spec:
                                state_feature_fct = spec[FUNCTION]
                            elif PARAMS in spec and FUNCTION in spec[PARAMS]:
                                state_feature_fct = spec[PARAMS][FUNCTION]

                    elif spec is not None:
                        assert False, f"PROGRAM ERROR: unrecognized form of state_feature specification for {self.name}"

                # Fewer specifications than number of INPUT Nodes, so assign state_feature_default to the rest
                else:
                    # Note: state_input_port name assigned above
                    spec = self.state_feature_default

                parsed_feature_specs.append(spec)
                self._state_feature_functions.append(state_feature_fct)
                self._specified_INPUT_Node_InputPorts_in_order.append(agent_rep_input_port)
                state_input_port_names.append(state_input_port_name)

            if not any(self._state_feature_functions):
                self._state_feature_functions = None
            self.parameters.state_feature_specs.set(parsed_feature_specs, override=True)
            return state_input_port_names or []

        # END OF PARSE SPECS  -----------------------------------------------------------------------------------

        user_specs = self.parameters.state_feature_specs.spec
        self.state_feature_default = self.parameters.state_feature_default_spec.spec

        # SINGLE ITEM spec, SO APPLY TO ALL agent_rep_input_ports
        if (user_specs is None
                or isinstance(user_specs, (str, tuple, InputPort, OutputPort, Mechanism, Composition))
                or (is_numeric(user_specs) and (np.array(user_specs).ndim < 2))):
            specs = [user_specs] * len(agent_rep_input_ports)
            # OK to assign here (rather than in _parse_secs()) since spec is intended for *all* state_input_ports
            self.parameters.state_feature_specs.set(specs, override=True)
            state_input_port_names = _parse_specs(state_feature_specs=specs,
                                                  specified_input_ports=agent_rep_input_ports,
                                                  spec_type='list')

        # LIST OR SHADOW_INPUTS DICT: source specs
        # Source specs but not INPUT Nodes specified; spec is either:
        # - list:  [spec, spec...]
        # - SHADOW_INPUTS dict (with list spec as its only entry): {SHADOW_INPUTS: {[spec, spec...]}}
        # Treat specs as sources of input to INPUT Nodes of agent_rep (in corresponding order):
        # Call _parse_specs to construct a regular dict using INPUT Nodes as keys and specs as values
        elif isinstance(user_specs, list) or (isinstance(user_specs, dict) and SHADOW_INPUTS in user_specs):
            if isinstance(user_specs, list):
                num_missing_specs = len(agent_rep_input_ports) - len(self.state_feature_specs)
                specs = user_specs + [self.state_feature_default] * num_missing_specs
                spec_type = 'list'
            else:
                # SHADOW_INPUTS spec:
                if isinstance(user_specs[SHADOW_INPUTS], set):
                    # Set not allowed as SHADOW_INPUTS spec; catch here to provide context-relevant error message
                    raise OptimizationControlMechanismError(
                        f"The '{STATE_FEATURES}' argument for '{self.name}' uses a set in a '{SHADOW_INPUTS.upper()}' "
                        f"dict;  this must be a single item or list of specifications in the order of the INPUT Nodes"
                        f"of its '{AGENT_REP}' ({self.agent_rep.name}) to which they correspond." )
                # FIX: 3/18/22 - ?DOES THIS NEED TO BE DONE HERE, OR CAN IT BE DONE IN A RELEVANT _validate METHOD?
                # All specifications in list specified for SHADOW_INPUTS must be shadowable
                #     (i.e., either an INPUT Node or the InputPort of one) or None;
                #     note: if spec is not in agent_rep, might be added later,
                #           so defer dealing with that until runtime.
                bad_specs = [spec for spec in user_specs[SHADOW_INPUTS]
                             if (    # spec is not an InputPort
                                     ((isinstance(spec, Port) and not isinstance(spec, InputPort))
                                      # spec is an InputPort of a Node in agent_rep but not one of its INPUT Nodes
                                      or (isinstance(spec, InputPort)
                                          and spec.owner in self.agent_rep._get_all_nodes()
                                          and spec.owner not in self._get_agent_rep_input_receivers(type=NODE,
                                                                                                    comp_as_node=ALL))
                                      )
                                     and spec is not None)]
                if bad_specs:
                    bad_spec_names = [f"'{item.owner.name}'" if hasattr(item, 'owner')
                                      else f"'{item.name}'" for item in bad_specs]
                    raise OptimizationControlMechanismError(
                        f"The '{STATE_FEATURES}' argument for '{self.name}' has one or more items in the list "
                        f"specified for '{SHADOW_INPUTS.upper()}' ({', '.join([name for name in bad_spec_names])}) "
                        f"that are not (part of) any INPUT Nodes of its '{AGENT_REP}' ('{self.agent_rep.name}')." )

                specs = user_specs[SHADOW_INPUTS]
                spec_type = f"{SHADOW_INPUTS.upper()} dict"

            specified_input_ports = agent_rep_input_ports + [None] * (len(specs) - len(agent_rep_input_ports))
            state_input_port_names = _parse_specs(state_feature_specs=specs,
                                                  specified_input_ports=specified_input_ports,
                                                  spec_type=spec_type)

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

            # # Get specified ports in order of agent_rep INPUT Nodes, with None assigned to any unspecified InputPorts
            all_specified_ports = [port if port in expanded_specified_ports
                                   else None for port in agent_rep_input_ports]
            # Get any not found anywhere (including nested) in agent_rep, which are placed at the end of list
            all_specified_ports.extend([port for port in expanded_specified_ports if port not in agent_rep_input_ports])

            if isinstance(user_specs, set):
                # Just pass ports;  _parse_specs assigns shadowing projections for them and None to all unspecified ones
                specs = all_specified_ports
            else:
                # Pass values from user_spec dict to be parsed;
                #    corresponding ports are safely in all_specified_ports
                #    unspecified ports are assigned state_feature_default per requirements of list format
                specs = [expanded_dict_with_ports[port] if port is not None and port in all_specified_ports
                         else self.state_feature_default for port in all_specified_ports]

            state_input_port_names = _parse_specs(state_feature_specs=specs,
                                                  specified_input_ports=list(all_specified_ports),
                                                  spec_type='dict')

        else:
            assert False, f"PROGRAM ERROR: Unanticipated type specified for '{STATE_FEATURES}' arg of '{self.name}: " \
                          f"'{user_specs}' ({type(user_specs)})."

        # CONSTRUCT InputPort SPECS -----------------------------------------------------------------------------

        state_input_port_specs = []

        for i in range(self._num_state_feature_specs):
            # Note: state_feature_specs have now been parsed (user's specs are in parameters.state_feature_specs.spec);
            #       state_feature_specs correspond to all InputPorts of agent_rep's INPUT Nodes (including nested ones)
            spec = self.state_feature_specs[i]

            if spec is None:
                continue

            # FIX: 3/22/22 - COULD/SHOULD ANY OF THE FOLLOWING BE DONE IN _parse_specs():

            if isinstance(self.state_feature_specs[i], dict):
                # If spec is an InputPort specification dict:
                # - if a Mechanism is specified as the source, specify for shadowing:
                #    - add SHADOW_INPUTS entry to specify shadowing of Mechanism's primary InputPort
                #    - process dict through _parse_shadow_inputs(),
                #      (preserves spec as dict in case other parameters (except function, handled below) are specified)
                #    - replace self.state_feature_specs[i] with the InputPort (required by state_feature_values)
                # - if a function is specified, clear from dict
                #    - it has already been assigned to self._state_feature_functions in _parse_spec()
                #    - it will be properly assigned to InputPort specification dict in _assign_state_feature_function
                def _validate_entries(spec=None, source=None):
                    if spec and source:
                        if (SHADOW_INPUTS in spec) or (PARAMS in spec and SHADOW_INPUTS in spec[PARAMS]):
                            error_msg = "both PROJECTIONS and SHADOW_INPUTS cannot specified"
                        elif len(convert_to_list(source)) != 1:
                            error_msg = "PROJECTIONS entry has more than one source"
                        return
                    else:
                        error_msg = f"missing a 'PROJECTIONS' entry specifying the source of the input."
                    raise OptimizationControlMechanismError(f"Error in InputPort specification dictionary used in '"
                                                            f"{STATE_FEATURES}' arg of '{self.name}': {error_msg}.")
                # Handle shadowing of Mechanism as source
                if PROJECTIONS in spec:
                    source = get_port_for_mech_spec(spec.pop(PROJECTIONS))
                    if spec:  # No need to check if nothing left in dict
                        _validate_entries(spec, source)
                    spec[SHADOW_INPUTS] = source
                elif PARAMS in spec and PROJECTIONS in spec[PARAMS]:
                    source = get_port_for_mech_spec(spec[PARAMS].pop(PROJECTIONS))
                    _validate_entries(spec, source)
                    spec[PARAMS][SHADOW_INPUTS] = source
                else:
                    _validate_entries()

                # Clear FUNCTION entry
                if self._state_feature_functions[i]:
                    spec.pop(FUNCTION, None)
                    if PARAMS in spec:
                        spec[PARAMS].pop(FUNCTION, None)

                spec = _parse_shadow_inputs(self, spec)[0] # _parse_shadow_inputs returns a list, so get item
                self.state_feature_specs[i] = source

            if is_numeric(spec):
                # Construct InputPort specification dict that configures it to use its InputPort.default_variable
                #    as its input, and assigns the spec's value to the VALUE entry,
                #    which assigns it as the value of the InputPort.default_variable
                spec_val = copy.copy(spec)
                spec = {VALUE: spec_val,
                        PARAMS: {DEFAULT_INPUT: DEFAULT_VARIABLE}}

            if isinstance(spec, Mechanism):
                # Replace with primary InputPort to be shadowed
                spec = get_port_for_mech_spec(spec)
                self.state_feature_specs[i] = spec

            # Get InputPort specification dictionary for state_input_port and update its entries
            parsed_spec = _parse_port_spec(owner=self, port_type=InputPort, port_spec=spec)
            parsed_spec[NAME] = state_input_port_names[i]
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

    def _update_state_input_ports_for_controller(self, context=None):
        """Check and update state_input_ports at run time if agent_rep is a Composition

        If no agent_rep has been specified or it is a CompositionFunctionApproximator, return
            (note: validation of state_features specified for CompositionFunctionApproximator optimization
            is up to the CompositionFunctionApproximator)

        If agent_rep is a Composition:
           - if  has any new INPUT Node InputPorts:
               - construct state_input_ports for them
               - add to _specified_INPUT_Node_InputPorts_in_order
           - call _validate_state_features()
           - call _update_state_input_port_names()
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

        from psyneulink.core.compositions.composition import Composition
        num_agent_rep_input_ports = len(self.agent_rep_input_ports)
        num_state_feature_specs = len(self.state_feature_specs)

        if num_state_feature_specs < num_agent_rep_input_ports:
            # agent_rep is Composition, but state_input_ports are missing for some agent_rep INPUT Node InputPorts
            #   so construct a state_input_port for each missing one, using state_feature_default;
            #   note: assumes INPUT Nodes added are at the end of the list in self.agent_rep_input_ports
            # FIX: 3/24/22 - REFACTOR THIS TO CALL _parse_state_feature_specs?
            state_input_ports = []
            local_context = Context(source=ContextFlags.METHOD)
            default = self.state_feature_default
            new_agent_rep_input_ports = self.agent_rep_input_ports[self.num_state_input_ports:]
            for input_port in new_agent_rep_input_ports:
                # Instantiate state_input_port for each agent_rep INPUT Node InputPort not already specified:
                params = {INTERNAL_ONLY:True,
                          PARAMS: {}}
                if default is None:
                    continue
                if default == SHADOW_INPUTS:
                    params[SHADOW_INPUTS] = input_port
                    input_port_name = _shadowed_state_input_port_name(input_port.full_name, input_port.full_name)
                    self.state_feature_specs.append(input_port)
                elif is_numeric(default):
                    params[VALUE]: default
                    input_port_name = _numeric_state_input_port_name(input_port.full_name)
                    self.state_feature_specs.append(default)
                elif isinstance(default, (Port, Mechanism, Composition)):
                    params[PROJECTIONS]: default
                    self.state_feature_specs.append(default)
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
                # FIX: 3/24/22 - MAKE THIS A PROPERTY? (OR NEED IT REMAIN STABLE FOR LOOPS?)
                self._num_state_feature_specs += 1

            self.add_ports(state_input_ports,
                                 update_variable=False,
                                 context=local_context)

            # Assign OptimizationControlMechanism attributes
            self.state_input_ports.extend(state_input_ports)

        # IMPLEMENTATION NOTE: Can't just assign agent_rep_input_ports to _specified_INPUT_Node_InputPorts_in_order
        #                      below since there may be specifications in _specified_INPUT_Node_InputPorts_in_order
        #                      for agent_rep INPUT Node InputPorts that have not yet been added to Composition
        #                      (i.e., they are deferred)
        # Update _specified_INPUT_Node_InputPorts_in_order with any new agent_rep_input_ports
        for i in range(num_agent_rep_input_ports):
            if i < len(self._specified_INPUT_Node_InputPorts_in_order):
                # Replace existing ones (in case any deferred ones are "placemarked" with None)
                self._specified_INPUT_Node_InputPorts_in_order[i] = self.agent_rep_input_ports[i]
            else:
                # Add any that have been added to Composition
                self._specified_INPUT_Node_InputPorts_in_order.append(self.agent_rep_input_ports[i])

        if context._execution_phase == ContextFlags.PREPARING:
            # Restrict validation until run time, when the Composition is expected to be fully constructed
            self._validate_state_features(context)

        self._update_state_input_port_names(context)

    def _update_state_input_port_names(self, context=None):
        """Update names of state_input_port for any newly instantiated INPUT Node InputPorts

        If its instantiation has NOT been DEFERRED, assert that:
            - corresponding agent_rep INPUT Node InputPort is in Composition
            - state_input_port either has path_afferents or it is for a numeric spec

        If it's instantiation HAS been DEFERRED, for any newly added agent_rep INPUT Node InputPorts:
            - add agent_rep INPUT Node InputPort to _specified_INPUT_Node_InputPorts_in_order
            - if state_input_port:
                - HAS path_afferents, get source and generate new name
                - does NOT have path_afferents, assert it is for a numeric spec and generate new name
            - assign new name
        """

        num_agent_rep_input_ports = len(self.agent_rep_input_ports)
        for i, state_input_port in enumerate(self.state_input_ports):

            if context and context.flags & ContextFlags.PREPARING:
                # By run time, state_input_port should either have path_afferents assigned or be for a numeric spec
                assert state_input_port.path_afferents or NUMERIC_STATE_INPUT_PORT_PREFIX in state_input_port.name, \
                    f"PROGRAM ERROR: state_input_port instantiated for '{self.name}' ({state_input_port.name}) " \
                    f"with a specification in '{STATE_FEATURES}' ({self.parameters.state_feature_specs.spec[i]}) " \
                    f"that is not numeric but has not been assigned any path_afferents."

            if DEFERRED_STATE_INPUT_PORT_PREFIX not in state_input_port.name:
                # state_input_port should be associated with existing agent_rep INPUT Node InputPort
                assert i < num_agent_rep_input_ports, \
                    f"PROGRAM ERROR: state_input_port instantiated for '{self.name}' ({state_input_port.name}) " \
                    f"but there is no corresponding INPUT Node in '{AGENT_REP}'."
                continue

            if i >= num_agent_rep_input_ports:
                # No more new agent_rep INPUT Node InputPorts
                break

            # Add new agent_rep INPUT Node InputPorts
            self._specified_INPUT_Node_InputPorts_in_order[i] = self.agent_rep_input_ports[i]
            agent_rep_input_port_name = self.agent_rep_input_ports[i].full_name

            if state_input_port.path_afferents:
                # Non-numeric spec, so get source and change name accordingly
                source_input_port_name = self.state_feature_specs[i].full_name
                if 'INPUT FROM' in state_input_port.name:
                    new_name = _state_input_port_name(source_input_port_name, agent_rep_input_port_name)
                elif SHADOWED_INPUT_STATE_INPUT_PORT_PREFIX in state_input_port.name:
                    new_name = _shadowed_state_input_port_name(source_input_port_name, agent_rep_input_port_name)
            elif NUMERIC_STATE_INPUT_PORT_PREFIX in state_input_port.name:
                # Numeric spec, so change name accordingly
                new_name = _numeric_state_input_port_name(agent_rep_input_port_name)
            else:
                # Non-numeric but path_afferents haven't yet been assigned (will get tested again at run time)
                continue

            # Change name of state_input_port
            state_input_port.name = rename_instance_in_registry(registry=self._portRegistry,
                                                                category=INPUT_PORT,
                                                                new_name= new_name,
                                                                component=state_input_port)

    def _validate_state_features(self, context):
        """Validate that state_features are legal and consistent with agent_rep.

        Called by _update_state_input_ports_for_controller,
        - after new Nodes have been added to Composition
        - and/or in run() as final check before execution.

        Ensure that:
        - the number of state_feature_specs equals the number of external InputPorts of INPUT Nodes of agent_rep;
        - if state_feature_specs are specified as a user dict, keys are valid INPUT Nodes of agent_rep;
        - all InputPorts shadowed by specified state_input_ports are in agent_rep or one of its nested Compositions;
        - any Projections received from output_ports are from Nodes in agent_rep or its nested Compositions;
        - all InputPorts shadowed by state_input_ports reference INPUT Nodes of agent_rep or Compositions nested in it;
        - state_features are compatible with input format for agent_rep Composition
        """

        # FIX: 3/4/22 - DO user_specs HAVE TO BE RE-VALIDATED? ?IS IT BECAUSE THEY MAY REFER TO NEWLY ADDED NODES?
        from psyneulink.core.compositions.composition import \
            Composition, CompositionInterfaceMechanism, CompositionError, RunError, NodeRole

        comp = self.agent_rep
        user_specs = self.parameters.state_feature_specs.spec

        if isinstance(user_specs, dict) and SHADOW_INPUTS in user_specs:
            state_feature_specs = user_specs[SHADOW_INPUTS]
        else:
            state_feature_specs = user_specs
        if isinstance(state_feature_specs, list):
            # Convert list to dict (assuming list is in order of InputPorts of INPUT Nodes)
            input_ports = comp.external_input_ports_of_all_input_nodes
            if len(state_feature_specs) > len(input_ports):
                nodes_not_in_agent_rep = [f"'{spec.name if isinstance(spec, Mechanism) else spec.owner.name}'"
                                          for spec in self._get_specs_not_in_agent_rep(state_feature_specs)]
                missing_nodes_str = (f", that includes the following: {', '.join(nodes_not_in_agent_rep)} "
                                     f"missing from {self.agent_rep.name}"
                                     if nodes_not_in_agent_rep else '')
                raise OptimizationControlMechanismError(
                    f"The number of '{STATE_FEATURES}' specified for {self.name} ({len(state_feature_specs)}) "
                    f"is more than the number of INPUT Nodes ({len(input_ports)}) of the Composition assigned "
                    f"as its {AGENT_REP} ('{self.agent_rep.name}'){missing_nodes_str}.")
            input_dict = {}
            for i, spec in enumerate(state_feature_specs):
                input_dict[input_ports[i]] = spec
            state_features = state_feature_specs
        elif isinstance(state_feature_specs, (set, dict)):
            # If set or dict is specified, check that items of set or keys of dict are legal INPUT nodes:
            self._validate_input_nodes(set(state_feature_specs), enforce=True)
            if isinstance(state_feature_specs, dict):
                # If dict is specified, get values for checks below
                state_features = list(state_feature_specs.values())
            else:
                state_features = list(state_feature_specs)

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
                                               in self.agent_rep_input_ports)
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
        # inputs_dict, num_inputs = self.agent_rep._parse_input_dict(self.parameters.state_feature_values._get(context))
        # #  END DEBUGGING ---------------------------------------------------------------------

        # Ensure state_features are compatible with input format for agent_rep Composition
        try:
            # Call this to check for errors in constructing inputs dict
            self.agent_rep._parse_input_dict(self.parameters.state_feature_values._get(context))
        except (RunError, CompositionError) as error:
            raise OptimizationControlMechanismError(
                f"The '{STATE_FEATURES}' argument has been specified for '{self.name}' that is using a "
                f"{Composition.componentType} ('{self.agent_rep.name}') as its agent_rep, but some of the "
                f"specifications are not compatible with the inputs required by its 'agent_rep': '{error.error_value}' "
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
        """Find control_allocation that optimizes net_outcome of agent_rep.evaluate().
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

            # Port input struct is {data, modulation} if modulation is present,
            # otherwise it's just data
            if len(op.mod_afferents) > 0:
                data_out = builder.gep(op_in, [ctx.int32_ty(0), ctx.int32_ty(0),
                                               ctx.int32_ty(0)])
            else:
                data_out = builder.gep(op_in, [ctx.int32_ty(0), ctx.int32_ty(0)])

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
            pnlvm.helpers.create_sample(b, allocation, search_space, idx)

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
                                  num_trials_per_estimate, "corrected_trials per_estimate")

        num_trials = builder.alloca(ctx.int32_ty, name="num_sim_trials")
        builder.store(num_sims, num_trials)

        # We only provide one input
        num_inputs = builder.alloca(ctx.int32_ty, name="num_sim_inputs")
        builder.store(num_inputs.type.pointee(1), num_inputs)

        # Simulations don't store output
        comp_output = sim_f.args[4].type(None)
        builder.call(sim_f, [comp_state, comp_params, comp_data, comp_input,
                             comp_output, num_trials, num_inputs])

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

    def _gen_llvm_invoke_function(self, ctx, builder, function, params, context,
                                  variable, out, *, tags:frozenset):
        fun = ctx.import_llvm_function(function)

        # The function returns (sample_optimal, value_optimal),
        # but the value of mechanism is only 'sample_optimal'
        # so we cannot reuse the space provided and need to explicitly copy
        # the results later.
        fun_out = builder.alloca(fun.args[3].type.pointee, name="func_out")
        value = builder.gep(fun_out, [ctx.int32_ty(0), ctx.int32_ty(0)])

        args = [params, context, variable, fun_out]
        # If we're calling compiled version of Composition.evaluate,
        # we need to pass extra arguments
        if len(fun.args) > 4:
            args += builder.function.args[-3:]
        builder.call(fun, args)


        # The mechanism also converts the value to array of arrays
        # e.g. [3 x double] -> [3 x [1 x double]]
        assert len(value.type.pointee) == len(out.type.pointee)
        assert value.type.pointee.element == out.type.pointee.element.element
        with pnlvm.helpers.array_ptr_loop(builder, out, id='mech_value_copy') as (b, idx):
            src = b.gep(value, [ctx.int32_ty(0), idx])
            dst = b.gep(out, [ctx.int32_ty(0), idx, ctx.int32_ty(0)])
            b.store(b.load(src), dst)

        return out, builder

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
    def agent_rep_input_ports(self):
        return self._get_agent_rep_input_receivers(type=PORT)

    @property
    def num_state_input_ports(self):
        try:
            return len(self.state_input_ports)
        except:
            return 0

    # @property
    # def _num_state_feature_specs(self):
    #     return len(self.state_feature_specs)

    @property
    def state_features(self):
        """Return {InputPort name: source name} for all state_features.
        If state_feature_spec is numeric for a Node, assign its value as the source
        If existing INPUT Node is not specified in state_feature_specs, assign state_feature_default as source
        If an InputPort is referenced in state_feature_specs that is not yet in agent_rep,
            assign "DEFERRED INPUT NODE <InputPort name> OF <agent_rep>" as key for the entry;
            (it should be resolved by runtime, or an error is generated).
        If a state_feature_spec is referenced that is not yet in ocm.composition,
            assign "<InputPort name> NOT (YET) IN <agent_rep>" as the value of the entry;
            (it should be resolved by runtime, or an error is generated).
        """

        self._update_state_input_port_names()

        agent_rep_input_ports = self.agent_rep.external_input_ports_of_all_input_nodes
        state_features_dict = {}
        state_input_port_num = 0

        # Process all state_feature_specs, that may include ones for INPUT Nodes not (yet) in agent_rep
        for i in range(self._num_state_feature_specs):
            spec = self.state_feature_specs[i]
            input_port = self._specified_INPUT_Node_InputPorts_in_order[i]
            # Get key for state_features dict
            if input_port in agent_rep_input_ports:
                # Specified InputPort belongs to an INPUT Node already in agent_rep, so use as key
                key = input_port.full_name
            else:
                # Specified InputPort is not (yet) in agent_rep
                input_port_name = (f"{input_port.full_name}" if input_port
                                   else f"{str(i-len(agent_rep_input_ports))}")
                key = _deferred_agent_rep_input_port_name(input_port_name, self.agent_rep.name)

            # Get source for state_features dict
            if spec is None:
                # no state_input_port has been constructed, so assign source as None
                source = None
            else:
                if is_numeric(spec):
                    # assign numeric spec as source
                    source = spec
                    # increment state_port_num, since one is implemented for numeric spec
                else:
                    # spec is a Component, so get distal source of Projection to state_input_port
                    #    (spec if it is an OutputPort; or ??input_CIM.output_port if it spec for shadowing an input??
                    state_input_port = self.state_input_ports[state_input_port_num]
                    if self.composition._is_in_composition(spec):
                        source = spec.full_name
                    else:
                        source = _deferred_state_feature_spec_msg(spec.full_name, self.composition.name)
                state_input_port_num += 1

            state_features_dict[key] = source

        return state_features_dict

    @property
    def state(self):
        """Array that is concatenation of state_feature_values and control_allocations"""
        # Use self.state_feature_values Parameter if state_features specified; else use state_input_port values
        return list(self.state_feature_values.values()) + list(self.control_allocation)

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
        sources_and_destinations = self.state_feature_sources
        sources_and_destinations.update(self.control_signal_destinations)
        return sources_and_destinations

    @property
    def state_feature_sources(self):
        """Dict with {InputPort: source} for all INPUT Nodes of agent_rep, and sources in **state_feature_specs.
        Used by state_distal_sources_and_destinations_dict()
        """
        state_dict = {}
        # FIX: 3/4/22 - THIS NEEDS TO HANDLE BOTH state_input_ports BUT ALSO state_feature_values FOR WHICH THERE ARE NO INPUTPORTS
        specified_state_features = [spec for spec in self.state_feature_specs if spec is not None]
        for state_index, port in enumerate(self.state_input_ports):
            if port.path_afferents:
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
            else:
                if port.default_input is DEFAULT_VARIABLE:
                    source_port = DEFAULT_VARIABLE
                    node = None
                    comp = None
                else:
                    source_port = specified_state_features[state_index]
                    node = None
                    comp = None
            state_dict.update({(source_port, node, comp, state_index):self.state[state_index]})
        return state_dict

    @property
    def control_signal_destinations(self):
        state_dict = {}
        state_index = self.num_state_input_ports
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
