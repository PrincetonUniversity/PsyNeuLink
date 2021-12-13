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
          - `Model-Free" Optimization <OptimizationControlMechanism_Model_Free>`
          - `Model-Based" Optimization <OptimizationControlMechanism_Model_Based>`
  * `OptimizationControlMechanism_Creation`
      - `Agent Rep <OptimizationControlMechanism_Agent_Rep_Arg>`
      - `State Features <OptimizationControlMechanism_State_Features_Arg>`
      - `State Feature Functions <OptimizationControlMechanism_State_Feature_Functions_Arg>`
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

* **state_features** -- specifies the values provided by the OptimizationControlMechanism as the input to the
  `agent_rep <OptimizationControlMechanism.agent_rep>` when used, together with a selected `control_allocation
  <ControlMechanism.control_allocation>`, to estimate or predict the Composition's `net_outcome
  <ControlMechanism.net_outcome>`.  These are used to construct the `state_input_ports
  <OptimizationControlMechanism.state_input_ports>` for the OptimizationControlMechanism, that provide the
  `agent_rep<OptimizationControlMechanism.agent_rep>` with its input, and thus the specification requirements for
  **state_features** depend on whether the `agent_rep<OptimizationControlMechanism.agent_rep>` is a `Composition`
  or a `CompositionFunctionApproximator`:

  .. _OptimizationControlMechanism_Agent_Rep_Composition:

  * *agent_rep is a Composition* -- the **state_features** specify the inputs to the Composition when it is executed
    by the OptimizationControlMechanism to `evaluate <OptimizationControlMechanism_Evaluation>` its performance.
    If **state_features** is not specified, this is done automatically by constructing a set of `state_input_ports
    <OptimizationControlMechanism.state_input_ports>` that `shadow the input <InputPort_Shadow_Inputs>` to every
    `InputPort` of every `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of the Composition assigned as
    the `agent_rep <OptimizationControlMechanism.agent_rep>`.  In this case, if `controller_mode
    <Composition.controller_mode>` of the Composition for which the OptimizationControlMechanism is the `controller
    <Composition_Controller>` is set to *AFTER* (the default), the `input <Composition.input_values>` to
    the Composition on the current trial is used as its input to the `agent_rep
    <OptimizationControlMechanism.agent_rep>` for the optimization process; if the `controller_mode
    <Composition.controller_mode>` is *BEFORE*, then the inputs from the previous trial are used.

    The **state_features** argument can also be specified explicitly, using the formats described below.  This is
    useful if different functions need to be assigned to different `state_input_ports
    <OptimizationControlMechanism.state_input_ports>` used to generate the corresponding `state_feature_values
    state_feature_values <OptimizationControlMechanism.state_feature_values>` (see `below
    <OptimizationControlMechanism_State_Feature_Functions_Arg>`). However, doing so overrides the automatic
    assignment of all state_features, and so a complete and appropriate set of specifications must be provided
    (see note below).

    .. _OptimizationControlMechanism_State_Features_Shapes:

        .. note::
           If **state_features** *are* specified explicitly when the `agent_rep <OptimizationControlMechanism.agent_rep>`
           is a Composition, there must be one for every `InputPort` of every `INPUT <NodeRole.INPUT>` `Node
           <Composition_Nodes>` in that Composition, and these must match -- both individually, and in their order --
           the `inputs to the Composition <Composition_Execution_Inputs>`) required by its `run <Composition.run>`
           method.  Failure to do so generates an error indicating this.

        .. _OptimizationControlMechanism_Selective_Input:

        .. hint::
           For cases in which only a subset of the inputs to the Composition are relevant to its optimization (e.g.,
           the others should be held constant), it is still the case that all must be specified as **state_features**
           (see note above).  This can be handled several ways.  One is by specifying (as required) **state_features**
           for all of the inputs, and assigning *state_feature_functions** (see `below
           <OptimizationControlMechanism_State_Feature_Functions_Arg>`) such that those assigned to the desired
           inputs pass their values unmodified, while those for the inputs that are to be ignored return a constant value.
           Another approach, for cases in which the desired inputs pertain to a subset of Components in the Composition
           that solely responsible for determining its `net_outcome <ControlMechanism.net_outcome>`, is to assign those
           Components to a `nested Composition <Composition_Nested>` and assign that Composition as the `agent_rep
           <OptimizationControlMechanism.agent_rep>`.  A third, more sophisticated approach, would be to assign
           ControlSignals to the InputPorts for the irrelevant features, and specify them to suppress their values.

  .. _OptimizationControlMechanism_Agent_Rep_CFA:

  * *agent_rep is a CompositionFunctionApproximator* -- the **state_features** specify the inputs to the
    CompositionFunctionApproximator's `evaluate <CompositionFunctionApproximator.evaluate>` method.  This is not
    done automatically (see warning below).

        .. warning::
           The **state_features** specified when the `agent_rep <OptimizationControlMechanism.agent_rep>` is a
           `CompositionFunctionApproximator` must align with the arguments of its `evaluate
           <CompositionFunctionApproximator.evaluate>` method.  Since the latter cannot always be determined automatically,
           the `state_input_ports <OptimizationControlMechanism.state_input_ports>` cannot be created automatically, nor
           can the **state_features** specification be validated;  thus, specifying inappropriate **state_features** may
           produce errors that are unexpected or difficult to interpret.

  COMMENT:
   FIX: CONFIRM (OR IMPLEMENT?) THE FOLLOWING
   If all of the inputs to the Composition are still required, these can be specified using the keyword *INPUTS*,
   in which case they are retained along with any others specified.
  COMMENT

  .. _OptimizationControlMechanism_State_Features_Shadow_Inputs:

  The specifications in the **state_features** argument are used to construct the `state_input_ports
  <OptimizationControlMechanism.state_input_ports>`, and can be any of the following, used either singly or in a list:

  * *InputPort specification* -- this creates an `InputPort` as one of the OptimizationControlMechanism's
    `state_input_ports <OptimizationControlMechanism.state_input_ports>` that `shadows <InputPort_Shadow_Inputs>` the
    input to the specified InputPort;  that is, the value of which is used as the corresponding value of the
    OptimizationControlMechanism's `state_feature_values <OptimizationControlMechanism.state_feature_values>`.

    .. technical_note::
      The InputPorts specified as state_features are marked as `internal_only <InputPort.internal_only>` = `True`.

  * *OutputPort specification* -- this can be any form of `OutputPort specification <OutputPort_Specification>`
    for any `OutputPort` of another `Mechanism <Mechanism>` in the Composition; the `value <OutputPort.value>`
    of the specified OutputPort is used as the corresponding value of the OptimizationControlMechanism's
    `state_feature_values <OptimizationControlMechanism.state_feature_values>`.

  * *Mechanism* -- if the `agent_rep <OptimizationControlMechanism.agent_rep>` is a Composition, it must be an
    `INPUT <NodeRole.INPUT>` `Node <Composition_Nodes>` of that Composition, and the Mechanism's `primary InputPort
    <InputPort_Primary>` is used (since in this case the state_feature must correspond to an input to the Composition).
    If the `agent_rep <OptimizationControlMechanism.agent_rep>` is a `CompositionFunctionApproximator`, then the
    Mechanism's `primary OutputPort <OutputPort_Primary>` is used (since is the typically usage for specifying an
    InputPort);  if the input to the Mechanism is to be shadowed, then its InputPort must be specified explicitly.

  COMMENT:
      FIX: CONFIRM THAT THE FOLLOWING ALL WORK
  COMMENT
  State features can also be added to an existing OptimizationControlMechanism using its `add_state_features` method.

.. _OptimizationControlMechanism_State_Feature_Functions_Arg:

* **state_feature_functions** -- specifies the `function(s) <InputPort.function>` assigned to the `state_input_ports
  <OptimizationControlMechanism.state_input_ports>` created for each of the corresponding **state_features**.
  If **state_feature_functions** is not specified, the identity function is assigned to all of the `state_input_ports
  <OptimizationControlMechanism.state_input_ports>` (whether those were created automatically or explicitly specified;
  see `above <OptimizationControlMechanism_State_Features_Arg>`).  However, other functions can be specified
  individually for the `state_input_ports <OptimizationControlMechanism.state_input_ports>` associated with each
  state_feature. This can be useful, for example to provide an average or integrated value of prior inputs, to
  select specific inputs for use (see `hint <OptimizationControlMechanism_Selective_Input>` above), and/or use a
  generative model of the environment to provide inputs to the `agent_rep <OptimizationControlMechanism.agent_rep>`
  during the optimization process. This can be done by specifying the **state_feature_functions** argument with a
  dict with keys that match each of the specifications in the **state_features** argument, and corresponding values
  that specify the function to use for each.

    .. note::
       A dict can be used to specify **state_feature_functions** only if **state_features** are specified explicitly
       (see `above <OptimizationControlMechanism_State_Features_Arg>`). The dict must contain one entry for
       each of the items specified in **state_features**, and the value returned by each function must preserve the
       shape of its input, which must match that of the corresponding input to the Composition's `run
       <Composition.run>` method (see `note <OptimizationControlMechanism_State_Features_Shapes>` above).

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

The current state of the OptimizationControlMechanism -- or, more properly, its `agent_rep
<OptimizationControlMechanism.agent_rep>` -- is determined by the OptimizationControlMechanism's current
`state_feature_values <OptimizationControlMechanism.state_feature_values>` (see `below
<OptimizationControlMechanism_State_Features>`) and `control_allocation <ControlMechanism.control_allocation>`.
These are provided as input to the `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>` method,
the results of which are used together with the `costs <ControlMechanism_Costs_NetOutcome>` associated with the
`control_allocation <ControlMechanism.control_allocation>`, to evaluate the `net_outcome
<ControlMechanism.net_outcome>` for that state.

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
execution occurred.  Each of these is described below.

.. _OptimizationControlMechanism_State_Features:

*state_input_ports*
~~~~~~~~~~~~~~~~~~~

The `state_input_ports <OptimizationControlMechanism.state_input_ports>` receive `Projections <Projection>`
from the Components specified as the OptimizationControlMechanism's `state_features
<OptimizationControlMechanism_State_Features_Arg>`, the values of which are assigned as the `state_feature_values
<OptimizationControlMechanism.state_feature_values>`, and conveyed to the `agent_rep
<OptimizationControlMechanism.agent_rep>` when it is `executed <OptimizationControlMechanism_Execution>`. If the
`agent_rep is a `Composition <OptimizationControlMechanism_Agent_Rep_Composition>`, then the
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
        `function <ObjectiveMechanism.funtion>` evaluates the `outcome <ControlMechanism.outcome>` of processing
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
constructor (see `Outcomes arguments <OptimizationControlMechanism_Outcome_Args>`), and the `allow_probes
<Composition.allow_probes>` attribute of the Composition for which the OptimizationControlMechanism is the
`controller <Composition.controller>`. The latter allows the values of the items listed in `monitor_for_control
<ControlMechanism.monitor_for_control>` to be `INPUT <NodeRole.INTERNAL>` or `INTERNAL <NodeRole.INTERNAL>` `Nodes
<Composition_Nodes>` of a `nested Composition <Composition_Nested>` to be monitored and included in the computation
of `outcome <ControlMechanism.outcome>` (ordinarily, those must be `OUTPUT <NodeRole.OUTPUT>` Nodes of a nested
Composition).  This can be thought of as providing access to "latent variables" of the Composition being evaluated;
that is, ones that do not contribute directly to the Composition's `results <Composition_Execution_Results>`. This
applies both to items that are monitored directly by the OptimizationControlMechanism or via its ObjectiveMechanism
(see `allow_probes <ControlMechanism_Allow_Probes>` above for additional details).

The value(s) of the specified Components are assigned as the OptimizationControlMechanism's `outcome
<ControlMechanism.outcome>` attribute, which is used to compute the `net_outcome <ControlMechanism.net_outcome>`
of executing its `agent_rep <OptimizationControlMechanism.agent_rep>`.

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
  - It must execute the OptimizationControlMechanism's `evaluate_agent_rep
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

If `num_estimates <OptimizationControlMechanism.num_estimates>` is specified (that is, it is not None),
a `ControlSignal` is automatically added to the OptimizationControlMechanism's `control_signals
<OptimizationControlMechanism.control_signals>`, named *RANDOMIZATION_CONTROL_SIGNAL*, that randomizes
the values of random variables in the `agent_rep <OptimizationControlMechanism.agent_rep>` over estimates of its
`net_outcome <ControlMechanism.net_outcome>`. The `initial_seed <OptimizationControlMechanism.initial_seed>` and
`same_seed_for_all_allocations <OptimizationControlMechanism.same_seed_for_all_allocations>` Parameters can also be
used to further refine randomization (see `OptimizationControlMechanism_Estimation_Randomization` for additional
details).

.. _technical_note::

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
      <OptimizationControlMechanism.evaluate_agent_rep>` method `num_estimates <OptimizationControlMechanism>` times,
      each with the current `state_feature_values <OptimizationControlMechanism.state_feature_values>` as its input,
      and executing it for `num_trials_per_estimate <OptimizationControlMechanism.num_trials_per_estimate>` trials
      for each estimate.  The `control_allocation <ControlMechanism.control_allocation>` remains fixed for each
      estimate, but the random seed of any Parameters that rely on randomization is varied, so that the values of those
      Parameters are randomly sampled for every estimate (see `OptimizationControlMechanism_Estimation_Randomization`).

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
<OptimizationControlMechanism.random_variables>` attribute are randomized over thoese estimates.  By default,
this includes all Components in the `agent_rep <OptimizationControlMechanism.agent_rep>` with random variables (listed
in its `random_variables <Composition.random_variables>` attribute).  However, if particular Components are specified
in the **random_variables** argument of the OptimizationControlMechanism's constructor, then randomization is
restricted to their values. Randomization over estimates can be further configured using the `initial_seed
<OptimizationControlMechanism.initial_seed>` and `same_seed_for_all_allocations
<OptimizationControlMechanism.same_seed_for_all_allocations>` attributes. The results of all the estimates for a given
`control_allocation <ControlMechanism.control_allocation>` are aggregated by the `aggregation_function
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

import numpy as np
import typecheck as tc

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.component import DefaultsFlexibility
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
from psyneulink.core.components.ports.port import _parse_port_spec, _instantiate_port
from psyneulink.core.components.shellclasses import Function
from psyneulink.core.globals.context import Context, ContextFlags
from psyneulink.core.globals.context import handle_external_context
from psyneulink.core.globals.defaults import defaultControlAllocation
from psyneulink.core.globals.keywords import \
    ALL, COMPOSITION, COMPOSITION_FUNCTION_APPROXIMATOR, CONCATENATE, DEFAULT_VARIABLE, EID_FROZEN, \
    FUNCTION, INTERNAL_ONLY, OPTIMIZATION_CONTROL_MECHANISM, OWNER_VALUE, PARAMS, PROJECTIONS, \
    SHADOW_INPUTS, SHADOW_INPUT_NAME
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.sampleiterator import SampleIterator, SampleSpec
from psyneulink.core.globals.utilities import convert_to_list, convert_to_np_array, ContentAddressableList
from psyneulink.core.llvm.debug import debug_env

__all__ = [
    'OptimizationControlMechanism', 'OptimizationControlMechanismError',
    'AGENT_REP', 'STATE_FEATURES', 'STATE_FEATURE_FUNCTIONS', 'RANDOMIZATION_CONTROL_SIGNAL'
]

AGENT_REP = 'agent_rep'
STATE_FEATURES = 'state_features'
STATE_FEATURE_FUNCTIONS = 'state_feature_functions'
RANDOMIZATION_CONTROL_SIGNAL = 'RANDOMIZATION_CONTROL_SIGNAL'
RANDOM_VARIABLES = 'random_variables'

def _parse_state_feature_values_from_variable(index, variable):
    """Return values of state_input_ports"""
    return convert_to_np_array(np.array(variable[index:]).tolist())

class OptimizationControlMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _control_allocation_search_space_getter(owning_component=None, context=None):
    search_space = owning_component.parameters.search_space._get(context)
    if not search_space:
        return [c.parameters.allocation_samples._get(context) for c in owning_component.control_signals]
    else:
        return search_space

class OptimizationControlMechanism(ControlMechanism):
    """OptimizationControlMechanism(                    \
        agent_rep=None,                                 \
        state_features=None,                            \
        state_feature_functions=None,                   \
        monitor_for_control=None,                       \
        allow_probes=False,                             \
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

    state_features : Mechanism, InputPort, OutputPort, Projection, dict, or list containing any of these
        specifies Components for which `state_input_ports <OptimizationControlMechanism.state_input_ports>`
        are created, the `values <InputPort.value>` of which are assigned to `state_feature_values
        <OptimizationControlMechanism.state_feature_values>` and used to predict `net_outcome
        <ControlMechanism.net_outcome>`. Any `InputPort specification <InputPort_Specification>`
        can be used that resolves to an `OutputPort` that projects to that InputPort (see
        `state_features <OptimizationControlMechanism_State_Features_Arg>` for additional details>).

    state_feature_functions : Function or function : default None
        specifies the `function <InputPort.function>` assigned the `InputPort` in `state_input_ports
        <OptimizationControlMechanism.state_input_ports>` assigned to each **state_feature**
        (see `state_feature_functions <OptimizationControlMechanism_State_Feature_Functions_Arg>` for additional details).

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
        specifies the number independent runs of `agent_rep <OptimizationControlMechanism.agent_rep>` used
        to estimate its `net_outcome <ControlMechanism.net_outcome>` for each `control_allocation
        <ControlMechanism.control_allocation>` sampled (see `num_estimates
        <OptimizationControlMechanism.num_estimates>` for additional information).

    random_variables : Parameter or list[Parameter] : default ALL
        specifies the Components with random variables to be randomized over different estimates
        of each `control_allocation <ControlMechanism.control_allocation>`;  these must be in the `agent_rep
        <OptimizationControlMechanism.agent_rep>` and have a `seed` `Parameter`. By default, all such Components in
        the `agent_rep <OptimizationControlMechanism.agent_rep>` (listed in its `random_variables
        <Composition.random_variables>` attribute) are included (see `random_variables
        <OptimizationControlMechanism.random_variables>` for additional information).

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
        one of its subclasses, or it has not been assigned (None); see `Agent Representation and Types
        of Optimization <OptimizationControlMechanism_Agent_Representation_Types>` for additional details.

    state_feature_values : 2d array
        the current value of each item of the OptimizationControlMechanism's
        `OptimizationControlMechanism_State_Features` (each of which is a 1d array).

    state_input_ports : ContentAddressableList
        lists the OptimizationControlMechanism's `InputPorts <InputPort>` that receive `Projections <Projection>`
        from the items specified in the **state_features** argument in the OptimizationControlMechanism's constructor
        or constructed automatically (see `state_features <OptimizationControlMechanism_State_Features_Arg>`), and
        that provide the `state_feature_values <OptimizationControlMechanism.state_feature_values>` to the `agent_rep
        <OptimizationControlMechanism>` (see `OptimizationControlMechanism_State_Features` for additional details).

    num_state_input_ports : int
        cantains the number of `state_input_ports <OptimizationControlMechanism.state_input_ports>`.

    outcome_input_ports : ContentAddressableList
        lists the OptimizationControlMechanism's `OutputPorts <OutputPort>` that receive `Projections <Projection>`
        from either its `objective_mechanism <ControlMechanism.objective_mechanism>` or the Components listed in
        its `monitor_for_control <ControlMechanism.monitor_for_control>` attribute, the values of which are used
        to compute the `net_outcome <ControlMechanism.net_outcome>` of executing the `agent_rep
        <OptimizationControlMechanism.agent_rep>` in a given `OptimizationControlMechanism_State`
        (see `Outcome <OptimizationControlMechanism_Outcome>` for additional details).

    num_estimates : int
        determines the number independent runs of `agent_rep <OptimizationControlMechanism.agent_rep>` (i.e., calls to
        `evaluate_agent_rep <OptimizationControlMechanism.evaluate_agent_rep>`) used to estimate the `net_outcome
        <ControlMechanism.net_outcome>` of each `control_allocation <ControlMechanism.control_allocation>` evaluated
        by the OptimizationControlMechanism's `function <OptimizationControlMechanism.function>` (i.e.,
        that are specified by its `search_space <OptimizationFunction.search_space>`); see
        `OptimizationControlMechanism_Estimation_Randomization` for additional details.

    random_variables : Parameter or List[Parameter]
        list of the Components with variables that are randomized over estimates for a given `control_allocation
        <ControlMechanism.control_allocation>`;  by default, all Components in the `agent_rep
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
        <ControlMechanism.control_allocation>` (see `note <OptimizationControlMechanism_Randomization_Control_Signal>` above).

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

    # FIX: ADD OTHER Parameters() HERE??
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

                state_feature_functions
                    see `state_feature_functions <OptimizationControlMechanism_Feature_Function>`

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
        """
        outcome_input_ports_option = Parameter(CONCATENATE, stateful=False, loggable=False, structural=True)
        function = Parameter(GridSearch, stateful=False, loggable=False)
        state_feature_functions = Parameter(None, reference=True, stateful=False, loggable=False)
        search_function = Parameter(None, stateful=False, loggable=False)
        search_space = Parameter(None, read_only=True)
        search_termination_function = Parameter(None, stateful=False, loggable=False)
        comp_execution_mode = Parameter('Python', stateful=False, loggable=False, pnl_internal=True)
        search_statefulness = Parameter(True, stateful=False, loggable=False)

        agent_rep = Parameter(None, stateful=False, loggable=False, pnl_internal=True, structural=True)

        # FIX: NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED (CHANGE 1 to 0? IF STATE_INPUT_PORTS ARE FIRST)
        state_feature_values = Parameter(_parse_state_feature_values_from_variable(1, [defaultControlAllocation]),
                                         user=False,
                                         pnl_internal=True)

        # FIX: Should any of these be stateful?
        random_variables = ALL
        initial_seed = None
        same_seed_for_all_allocations = False
        num_estimates = None
        num_trials_per_estimate = None

        # search_space = None
        control_allocation_search_space = Parameter(None, read_only=True, getter=_control_allocation_search_space_getter)

        saved_samples = None
        saved_values = None

    @handle_external_context()
    @tc.typecheck
    def __init__(self,
                 agent_rep=None,
                 state_features: tc.optional(tc.optional(tc.any(Iterable, Mechanism, OutputPort, InputPort))) = None,
                 state_feature_functions: tc.optional(tc.optional(tc.any(dict, is_function_type))) = None,
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

        # Legacy warnings and conversions
        for k in kwargs.copy():
            if k == 'features':
                if state_features:
                    warnings.warn(f"Both 'features' and 'state_features' were specified in the constructor for an"
                                  f" {self.__class__.__name__}. Note: 'features' has been deprecated; "
                                  f"'state_features' ({state_features}) will be used.")
                else:
                    warnings.warn(f"'features' was specified in the constructor for an {self.__class__.__name__}; "
                                  f"Note: 'features' has been deprecated; please use 'state_features' in the future.")
                    state_features = kwargs['features']
                kwargs.pop('features')
                continue
            if k == 'feature_function':
                if state_feature_functions:
                    warnings.warn(f"Both 'feature_function' and 'state_feature_functions' were specified in the "
                                  f"constructor for an {self.__class__.__name__}. Note: 'feature_function' has been "
                                  f"deprecated; 'state_feature_functions' ({state_feature_functions}) will be used.")
                else:
                    warnings.warn(f"'feature_function' was specified in the constructor for an"
                                  f"{self.__class__.__name__}; Note: 'feature_function' has been deprecated; "
                                  f"please use 'state_feature_functions' in the future.")
                    state_feature_functions = kwargs['feature_function']
                kwargs.pop('feature_function')
                continue
        self.state_features = convert_to_list(state_features)

        function = function or GridSearch

        # If agent_rep hasn't been specified, put into deferred init
        if agent_rep is None:
            if context.source==ContextFlags.COMMAND_LINE:
                # Temporarily name InputPort
                self._assign_deferred_init_name(self.__class__.__name__)
                # Store args for deferred initialization
                self._store_deferred_init_args(**locals())

                # Flag for deferred initialization
                self.initialization_status = ContextFlags.DEFERRED_INIT
                return
            # If constructor is called internally (i.e., for controller of Composition),
            # agent_rep needs to be specified
            else:
                assert False, f"PROGRAM ERROR: 'agent_rep' arg should have been specified " \
                              f"in internal call to constructor for {self.name}."

        super().__init__(
            function=function,
            state_feature_functions=state_feature_functions,
            num_estimates=num_estimates,
            num_trials_per_estimate = num_trials_per_estimate,
            random_variables=random_variables,
            initial_seed=initial_seed,
            same_seed_for_all_allocations=same_seed_for_all_allocations,
            search_statefulness=search_statefulness,
            search_function=search_function,
            search_termination_function=search_termination_function,
            agent_rep=agent_rep,
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

        elif request_set[STATE_FEATURE_FUNCTIONS]:
            state_feats = request_set.pop(STATE_FEATURES, None)
            state_feat_fcts = request_set.pop(STATE_FEATURE_FUNCTIONS, None)
            # If no or only one item is specified in state_features, only one state_function is allowed
            if ((not state_feats or len(convert_to_list(state_feats))==1)
                    and len(convert_to_list(state_feat_fcts))!=1):
                raise OptimizationControlMechanismError(f"Only one function is allowed to be specified for "
                                                        f"the '{STATE_FEATURE_FUNCTIONS}' arg of {self.name} "
                                                        f"if either no only one items is specified for its "
                                                        f"'{STATE_FEATURES}' arg.")
            if len(convert_to_list(state_feat_fcts))>1 and not isinstance(state_feat_fcts, dict):
                raise OptimizationControlMechanismError(f"The '{STATE_FEATURES}' arg of {self.name} contains more "
                                                        f"than one item, so its '{STATE_FEATURE_FUNCTIONS}' arg "
                                                        f"must be either only a single function (applied to all "
                                                        f"{STATE_FEATURES}) or a dict with entries of the form "
                                                        f"<state_feature>:<function>.")
            if len(convert_to_list(state_feat_fcts))>1:
                invalid_fct_specs = [fct_spec for fct_spec in state_feat_fcts if fct_spec not in state_feats]
                if invalid_fct_specs:
                    raise OptimizationControlMechanismError(f"The following entries of the dict specified for "
                                                            f"'{STATE_FEATURE_FUNCTIONS} of {self.name} have keys that "
                                                            f"do not match any InputPorts specified in its "
                                                            f"{STATE_FEATURES} arg: {invalid_fct_specs}.")


        if self.random_variables is not ALL:
            # invalid_params = [param.name for param in self.random_variables
            #                   if param not in [r._owner._owner for r in self.agent_rep.random_variables]]
            # if invalid_params:
            #     raise OptimizationControlMechanismError(f"The following Parameters were specified for the "
            #                                             f"{RANDOM_VARIABLES} arg of {self.name} that are do randomizable "
            #                                             f"(i.e., they do not have a 'seed' attribute: "
            #                                             f"{invalid_params}.")
            invalid_params = [param.name for param in self.random_variables
                              if param not in self.agent_rep.random_variables]
            if invalid_params:
                raise OptimizationControlMechanismError(f"The following Parameters were specified for the "
                                                        f"{RANDOM_VARIABLES} arg of {self.name} that are do randomizable "
                                                        f"(i.e., they do not have a 'seed' attribute: "
                                                        f"{invalid_params}.")

    # FIX: CONSIDER GETTING RID OF THIS METHOD ENTIRELY, AND LETTING state_input_ports
    #      BE HANDLED ENTIRELY BY _update_state_input_ports_for_controller
    def _instantiate_input_ports(self, context=None):
        """Instantiate InputPorts for state_features (with state_feature_functions if specified).

        This instantiates the OptimizationControlMechanism's `state_input_ports;
             these are used to provide input to the agent_rep when its evaluate method is called
             (see Composition._build_predicted_inputs_dict).
        The OptimizationCOntrolMechanism's outcome_input_ports are instantiated by
            ControlMechanism._instantiate_input_ports in the call to super().

        InputPorts are constructed for **state_features** by calling _parse_state_feature_specs
            with them and **state_feature_functions** arguments of the OptimizationControlMechanism constructor.
        The constructed state_input_ports  are passed to ControlMechanism_instantiate_input_ports(),
             which appends them to the InputPort(s) that receive input from the **objective_mechanism* (if specified)
             or **monitor_for_control** ports (if **objective_mechanism** is not specified).
        Also ensures that:
             - every state_input_port has only a single Projection;
             - every outcome_input_ports receive Projections from within the agent_rep if it is a Composition.

        If no **state_features** are specified in the constructor, assign ones for INPUT Nodes of owner.
          - warn for model-free `model-free optimization <OptimizationControlMechanism_Model_Based>`.
          - ignore here for `model-based optimization <OptimizationControlMechanism_Model_Based>`
            (handled in _update_state_input_ports_for_controller)

        See `state_features <OptimizationControlMechanism_State_Features_Arg>` and
        `OptimizationControlMechanism_State_Features` for additional details.
        """

        # If any state_features were specified parse them and pass to ControlMechanism._instantiate_input_ports()
        state_input_ports_specs = None

        # FIX: 11/3/21 :
        #    ADD CHECK IN _parse_state_feature_specs THAT IF A NODE RATHER THAN InputPort IS SPECIFIED,
        #    ITS PRIMARY IS USED (SEE SCRATCH PAD FOR EXAMPLES)
        if not self.state_features:
            # For model-free (agent_rep = CompositionFunctionApproximator), warn if no state_features specified.
            # Note: for model-based optimization, state_input_ports and any state_feature_functions specified
            #       are assigned in _update_state_input_ports_for_controller.
            if self.agent_rep_type == COMPOSITION_FUNCTION_APPROXIMATOR:
                warnings.warn(f"No 'state_features' specified for use with `agent_rep' of {self.name}")

        else:
            # FIX: 11/29/21: DISALLOW FOR COMPOSITION
            # Implement any specified state_features
            state_input_ports_specs = self._parse_state_feature_specs(self.state_features,
                                                                      self.state_feature_functions)
            # Note:
            #   if state_features were specified for model-free (i.e., agent_rep is a CompositionFunctionApproximator),
            #   assume they are OK (no way to check their validity for agent_rep.evaluate() method, and skip assignment

        # Pass state_input_ports_sepcs to ControlMechanism for instantiation and addition to OCM's input_ports
        super()._instantiate_input_ports(state_input_ports_specs, context=context)

        # Assign to self.state_input_ports attribute
        start = self.num_outcome_input_ports # FIX: 11/3/21 NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
        stop = start + len(state_input_ports_specs) if state_input_ports_specs else 0
        # FIX 11/3/21: THIS SHOULD BE MADE A PARAMETER
        self.state_input_ports = ContentAddressableList(component_type=InputPort,
                                                          list=self.input_ports[start:stop])

        # Ensure that every state_input_port has no more than one afferent projection
        # FIX: NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
        for i in range(self.num_outcome_input_ports, self.num_state_input_ports):
            port = self.input_ports[i]
            if len(port.path_afferents) > 1:
                raise OptimizationControlMechanismError(f"Invalid {type(port).__name__} on {self.name}. "
                                                        f"{port.name} should receive exactly one projection, "
                                                        f"but it receives {len(port.path_afferents)} projections.")

    def _validate_monitor_for_control(self, nodes):
        # Ensure all of the Components being monitored for control are in the agent_rep if it is Composition
        if self.agent_rep_type == COMPOSITION:
            try:
                super()._validate_monitor_for_control(self.agent_rep._get_all_nodes())
            except ControlMechanismError as e:
                raise OptimizationControlMechanismError(f"{self.name} has 'outcome_ouput_ports' that receive "
                                                        f"Projections from the following Components that do not "
                                                        f"belong to its {AGENT_REP} ({self.agent_rep.name}): {e.data}.")

    # FIX: 12/9/21 -- DEPRECATE DIRECT PROJECTIONS FROM PROBES, ELIMINATING THE NEED FOR THIS OVERRIDE
    # def _parse_monitor_for_control_input_ports(self, context):
    #     """Override ControlMechanism to implement allow_probes=DIRECT option
    #
    #     If is False (default), simply pass results of super()._parse_monitor_for_control_input_ports(context);
    #         this is restricted to the use of OUTPUT Nodes in nested Compositions, and routes Projections from nodes in
    #         nested Compositions through their respective output_CIMs.
    #
    #     If allow_probes option is True, any INTERNAL Nodes of nested Compositions specified in monitor_for_control
    #        are assigned NodeRole.OUTPUT, and Projections from them to the OptimizationControlMechanism are routed
    #        from the nested Composition(s) through the respective output_CIM(s).
    #
    #     If allow_probes option is DIRECT, Projection specifications are added to Port specification dictionaries,
    #        so that the call to super()._instantiate_input_ports in ControlMechanism instantiates Projections from
    #         monitored node to OptimizationControlMechanism. This allows *direct* Projections from monitored nodes in
    #         nested Compositions to the OptimizationControlMechanism, bypassing output_CIMs and preventing inclusion
    #         of their values in the results attribute of those Compositions.
    #
    #     Return port specification dictionaries (*with* Projection specifications), their value sizes and null list
    #     (to suppress Projection assignment to aux_components in ControlMechanism._instantiate_input_ports)
    #     """
    #
    #     outcome_input_port_specs, outcome_value_sizes, monitored_ports \
    #         = super()._parse_monitor_for_control_input_ports(context)
    #
    #     if self.allow_probes == DIRECT:
    #         # Add Projection specifications to port specification dictionaries for outcome_input_ports
    #         #    and return monitored_ports = []
    #
    #         if self.outcome_input_ports_option == SEPARATE:
    #             # Add port spec to to each outcome_input_port_spec (so that a Projection is specified directly to each)
    #             for i in range(self.num_outcome_input_ports):
    #                 outcome_input_port_specs[i].update({PROJECTIONS: monitored_ports[i]})
    #         else:
    #             # Add all ports specs as list to single outcome_input_port
    #             outcome_input_port_specs[0].update({PROJECTIONS: monitored_ports})
    #
    #         # Return [] for ports to suppress creation of Projections in _instantiate_input_ports
    #         monitored_ports = []
    #
    #     return outcome_input_port_specs, outcome_value_sizes, monitored_ports

    def _update_state_input_ports_for_controller(self, context=None):
        """Check and update state_input_ports for model-based optimization (agent_rep==Composition)

        If no agent_rep has been specified or it is model-free, return
            (note: validation of state_features specified for model-free optimization is up to the
            CompositionFunctionApproximator)

        For model-based optimization (agent_rep is a Composition):

        - ensure that state_input_ports for all specified state_features are for InputPorts of INPUT Nodes of agent_rep;
          raises an error if any receive a Projection that is not a shadow Projection from an INPUT Node of agent_rep
          (note: there should already be state_input_ports for any **state_features** specified in the constructor).

        - if no state_features specified, assign a state_input_port for every InputPort of every INPUT Node of agent_rep
          (note: shadow Projections for all state_input_ports are created in Composition._update_shadow_projections()).

        - assign state_feature_functions to relevant state_input_ports (same function for all if no state_features
          are specified or only one state_function is specified;  otherwise, use dict for specifications).
        """

        # FIX: 11/15/21 - REPLACE WITH ContextFlags.PROCESSING ??
        #               TRY TESTS WITHOUT THIS
        # Don't instantiate unless being called by Composition.run() (which does not use ContextFlags.METHOD)
        # This avoids error messages if called prematurely (i.e., before run is complete)
        # MODIFIED 11/29/21 OLD:
        if context.flags & ContextFlags.METHOD:
            return
        # MODIFIED 11/29/21 END

        # Don't bother for model-free optimization (see OptimizationControlMechanism_Model_Free)
        #    since state_input_ports specified or model-free optimization are entirely the user's responsibility;
        #    this is because they can't be programmatically validated against the agent_rep's evaluate() method.
        #    (This contrast with model-based optimization, for which there must be a state_input_port for every
        #    InputPort of every INPUT node of the agent_rep (see OptimizationControlMechanism_Model_Based).
        if self.agent_rep_type != COMPOSITION:
            return

        from psyneulink.core.compositions.composition import Composition, NodeRole, CompositionInterfaceMechanism

        def _get_all_input_nodes(comp):
            """Return all input_nodes, including those for any Composition nested one level down.
            Note: more deeply nested Compositions will either be served by their containing one(s) or own controllers
            """
            _input_nodes = comp.get_nodes_by_role(NodeRole.INPUT)
            input_nodes = []
            for node in _input_nodes:
                if isinstance(node, Composition):
                    input_nodes.extend(_get_all_input_nodes(node))
                else:
                    input_nodes.append(node)
            return input_nodes

        if self.state_features:
            # FIX: 11/26/21 - EXPLAIN THIS BEHAVIOR IN DOSCSTRING;
            warnings.warn(f"The 'state_features' argument has been specified for {self.name}, that is being "
                          f"configured as a model-based {self.__class__.__name__} (i.e, one that uses a "
                          f"{Composition.componentType} as its agent_rep).  This overrides automatic assignment of "
                          f"all inputs to its agent_rep ({self.agent_rep.name}) as the 'state_features'; only the "
                          f"ones specified will be used ({self.state_features}), and they must match the shape of the "
                          f"input to {self.agent_rep.name} when it is run.  Remove this specification from the "
                          f"constructor for {self.name} if automatic assignment is preferred.")

            comp = self.agent_rep
            # Ensure that all InputPorts shadowed by specified state_input_ports
            #    are in agent_rep or one of its nested Compositions
            invalid_state_features = [input_port for input_port in self.state_input_ports
                                      if (not (input_port.shadow_inputs.owner in
                                                list(comp.nodes) + [n[0] for n in comp._get_nested_nodes()])
                                          and (not [input_port.shadow_inputs.owner.composition is x for x in
                                                      comp._get_nested_compositions()
                                               if isinstance(input_port.shadow_inputs.owner,
                                                         CompositionInterfaceMechanism)]))]
            if any(invalid_state_features):
                raise OptimizationControlMechanismError(f"{self.name}, being used as controller for model-based "
                                                        f"optimization of {self.agent_rep.name}, has 'state_features' "
                                                        f"specified ({[d.name for d in invalid_state_features]}) that "
                                                        f"are missing from the Composition or any nested within it.")

            # Ensure that all  InputPorts shadowed by specified state_input_ports
            #    reference INPUT Nodes of agent_rep or of a nested Composition
            invalid_state_features = [input_port for input_port in self.state_input_ports
                                      if (not (input_port.shadow_inputs.owner in _get_all_input_nodes(self.agent_rep))
                                          and (isinstance(input_port.shadow_inputs.owner,
                                                         CompositionInterfaceMechanism)
                                               and not (input_port.shadow_inputs.owner.composition in
                                                        [nested_comp for nested_comp in comp._get_nested_compositions()
                                                         if nested_comp in comp.get_nodes_by_role(NodeRole.INPUT)])))]
            if any(invalid_state_features):
                raise OptimizationControlMechanismError(f"{self.name}, being used as controller for model-based "
                                                        f"optimization of {self.agent_rep.name}, has 'state_features' "
                                                        f"specified ({[d.name for d in invalid_state_features]}) that "
                                                        f"are not INPUT nodes for the Composition or any nested "
                                                        f"within it.")
            return

        # Model-based agent_rep, but no state_features have been specified,
        #   so assign a state_input_port to shadow every InputPort of every INPUT node of agent_rep
        shadow_input_ports = []
        for node in _get_all_input_nodes(self.agent_rep):
            for input_port in node.input_ports:
                if input_port.internal_only:
                    continue
                # if isinstance(input_port.owner, CompositionInterfaceMechanism):
                #     input_port = input_port.
                shadow_input_ports.append(input_port)

        local_context = Context(source=ContextFlags.METHOD)
        state_input_ports_to_add = []
        # for input_port in input_ports_not_specified:
        for input_port in shadow_input_ports:
            input_port_name = f"{SHADOW_INPUT_NAME} of {input_port.owner.name}[{input_port.name}]"
            params = {SHADOW_INPUTS: input_port,
                      INTERNAL_ONLY:True}
            # Note: state_feature_functions has been validated _validate_params
            #       to have only a single function in for model-based agent_rep
            if self.state_feature_functions:
                params.update({FUNCTION: self._parse_state_feature_function(self.state_feature_functions)})
            state_input_ports_to_add.append(_instantiate_port(name=input_port_name,
                                                              port_type=InputPort,
                                                              owner=self,
                                                              reference_value=input_port.value,
                                                              params=params,
                                                              context=local_context))
        self.add_ports(state_input_ports_to_add,
                             update_variable=False,
                             context=local_context)
        self.state_input_ports.extend(state_input_ports_to_add)

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

        # MODIFIED 11/21/21 NEW:
        #  FIX - PURPOSE OF THE FOLLOWING IS TO "CAPTURE" CONTROL SPECS MADE LOCALLY ON MECHANISMS IN THE COMP
        #        AND INSTANTIATE ControlSignals FOR THEM HERE, ALONG WITH THOSE SPECIFIED IN THE CONSTRUCTOR
        #         FOR THE OCM. ALSO CAPTURES DUPLICATES (SEE MOD BELOW).
        # FIX: WITHOUT THIS, GET THE mod param ERROR;  WITH IT, GET FAILURES IN test_control:
        #        TestModelBasedOptimizationControlMechanisms_Execution
        #            test_evc
        #            test_stateful_mechanism_in_simulation
        #        TestControlMechanisms:
        #            test_lvoc
        #            test_lvoc_both_prediction_specs
        #            test_lvoc_features_function
        # if self.agent_rep and self.agent_rep.componentCategory=='Composition':
        #     control_signals_from_composition = self.agent_rep._get_control_signals_for_composition()
        # self.output_ports.extend(control_signals_from_composition)
        # MODIFIED 11/21/21 END

        if self.num_estimates:

            randomization_seed_mod_values = SampleSpec(start=1,stop=self.num_estimates,step=1)

            # FIX: 11/3/21 noise PARAM OF TransferMechanism IS MARKED AS SEED WHEN ASSIGNED A DISTRIBUTION FUNCTION,
            #                BUT IT HAS NO PARAMETER PORT BECAUSE THAT PRESUMABLY IS FOR THE INTEGRATOR FUNCTION,
            #                BUT THAT IS NOT FOUND BY model.all_dependent_parameters
            # Get Components with variables to be randomized across estimates
            #   and construct ControlSignal to modify their seeds over estimates
            if self.random_variables is ALL:
                self.random_variables = self.agent_rep.random_variables
            self.output_ports.append(ControlSignal(name=RANDOMIZATION_CONTROL_SIGNAL,
                                                   modulates=[param.parameters.seed._port
                                                              for param in self.random_variables],
                                                   allocation_samples=randomization_seed_mod_values))

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

        self.defaults.value = np.tile(control_signal.parameters.variable.default_value, (i + 1, 1))
        self.parameters.control_allocation._set(copy.deepcopy(self.defaults.value), context)

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

        # If there is no randomization_control_signal, but num_estimates is 1 or None,
        #     pass None for randomization_control_signal_index (1 will be used by default by OptimizationFunction)
        if RANDOMIZATION_CONTROL_SIGNAL not in self.control_signals and self.num_estimates in {1, None}:
            randomization_control_signal_index = None
        # Otherwise, assert that num_estimates and number of seeds generated by randomization_control_signal are equal
        else:
            num_seeds = self.control_signals[RANDOMIZATION_CONTROL_SIGNAL].allocation_samples.base.num
            assert self.num_estimates == num_seeds, \
                    f"PROGRAM ERROR:  The value of the 'num_estimates' Parameter of {self.name}" \
                    f"({self.num_estimates}) is not equal to the number of estimates that will be generated by " \
                    f"its {RANDOMIZATION_CONTROL_SIGNAL} ControlSignal ({num_seeds})."
            randomization_control_signal_index = self.control_signals.names.index(RANDOMIZATION_CONTROL_SIGNAL)

        # Assign parameters to function (OptimizationFunction) that rely on OptimizationControlMechanism
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

        # # FIX: THESE NEED TO BE FOR THE PREVIOUS TRIAL;  ARE THEY FOR FUNCTION_APPROXIMATOR?
        # FIX: NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
        self.parameters.state_feature_values._set(_parse_state_feature_values_from_variable(
            self.num_outcome_input_ports,
            variable), context)

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

            # FIX: NEED TO MODIFY IF OUTCOME InputPorts ARE MOVED
            self.agent_rep.adapt(_parse_state_feature_values_from_variable(self.num_outcome_input_ports, variable),
                                 control_allocation,
                                 net_outcome,
                                 context=context)

        # freeze the values of current context, because they can be changed in between simulations,
        # and the simulations must start from the exact spot
        self.agent_rep._initialize_from_context(self._get_frozen_context(context),
                                                base_context=context,
                                                override=True)

        # Get control_allocation that optimizes net_outcome using OptimizationControlMechanism's function
        # IMPLEMENTATION NOTE: skip ControlMechanism._execute since it is a stub method that returns input_values
        optimal_control_allocation, optimal_net_outcome, saved_samples, saved_values = \
                                                super(ControlMechanism,self)._execute(variable=control_allocation,
                                                                                      num_estimates=self.parameters.num_estimates._get(context),
                                                                                      context=context,
                                                                                      runtime_params=runtime_params
                                                                                      )

        # clean up frozen values after execution
        self.agent_rep._delete_contexts(self._get_frozen_context(context))

        optimal_control_allocation = np.array(optimal_control_allocation).reshape((len(self.defaults.value), 1))
        if self.function.save_samples:
            self.saved_samples = saved_samples
        if self.function.save_values:
            self.saved_values = saved_values

        # Return optimal control_allocation
        return optimal_control_allocation

    def _get_frozen_context(self, context=None):
        return Context(execution_id=f'{context.execution_id}{EID_FROZEN}')

    def _set_up_simulation(self, base_context=Context(execution_id=None), control_allocation=None):
        sim_context = copy.copy(base_context)
        sim_context.execution_id = self.get_next_sim_id(base_context, control_allocation)

        try:
            self.parameters.simulation_ids._get(base_context).append(sim_context.execution_id)
        except AttributeError:
            self.parameters.simulation_ids._set([sim_context.execution_id], base_context)

        self.agent_rep._initialize_from_context(sim_context, self._get_frozen_context(base_context), override=False)

        return sim_context

    def _tear_down_simulation(self, sim_context=None):
        if not self.agent_rep.parameters.retain_old_simulation_data._get():
            self.agent_rep._delete_contexts(sim_context, check_simulation_storage=True)

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
            # KDM 5/20/19: crudely using default here because it is a stateless parameter
            # and there is a bug in setting parameter values on init, see TODO note above
            # call to self._instantiate_defaults around component.py:1115
            if self.defaults.search_statefulness:
                new_context = self._set_up_simulation(context, control_allocation)
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
                self._tear_down_simulation(new_context)

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
        total_cost = builder.alloca(ctx.float_ty)
        builder.store(ctx.float_ty(-0.0), total_cost)
        for i, op in enumerate(self.output_ports):
            op_i_params = builder.gep(op_params, [ctx.int32_ty(0),
                                                  ctx.int32_ty(i)])
            op_i_state = builder.gep(op_states, [ctx.int32_ty(0),
                                                 ctx.int32_ty(i)])

            op_f = ctx.import_llvm_function(op, tags=frozenset({"costs"}))

            op_in = builder.alloca(op_f.args[2].type.pointee)

            # copy allocation_sample, the input is 1-element array in a struct
            data_in = builder.gep(allocation_sample, [ctx.int32_ty(0),
                                                      ctx.int32_ty(i)])
            data_out = builder.gep(op_in, [ctx.int32_ty(0), ctx.int32_ty(0),
                                           ctx.int32_ty(0)])
            if data_in.type != data_out.type:
                warnings.warn("Shape mismatch: Allocation sample '{}' ({}) doesn't match input port input ({}).".format(i, self.parameters.control_allocation_search_space.get(), op.defaults.variable))
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

    def _gen_llvm_evaluate_alloc_range_function(self, *, ctx:pnlvm.LLVMBuilderContext,
                                                   tags=frozenset()):
        assert "evaluate" in tags
        assert "alloc_range" in tags
        evaluate_f = ctx.import_llvm_function(self,
                                              tags=tags - {"alloc_range"})


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

        allocation = builder.alloca(evaluate_f.args[2].type.pointee)
        with pnlvm.helpers.for_loop(builder, start, stop, stop.type(1), "alloc_loop") as (b, idx):

            func_out = b.gep(arg_out, [idx])
            pnlvm.helpers.create_allocation(b, allocation, search_space, idx)

            b.call(evaluate_f, [params, state, allocation, func_out, arg_in, data])

        builder.ret_void()
        return llvm_func

    def _gen_llvm_evaluate_function(self, *, ctx:pnlvm.LLVMBuilderContext,
                                             tags=frozenset()):
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
            builder.store(builder.load(base_comp_state), comp_state)

        # Create a simulation copy of composition data
        comp_data = builder.alloca(base_comp_data.type.pointee, name="data_copy")
        if "const_data" in debug_env:
            const_data = self.agent_rep._get_data_initializer(None)
            builder.store(comp_data.type.pointee(const_data), comp_data)
        else:
            builder.store(builder.load(base_comp_data), comp_data)

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
        fun_out = builder.alloca(fun.args[3].type.pointee)

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
        oport_input = builder.alloca(ctx.get_input_struct_type(port).elements[0])
        # FIXME: workaround controller signals occasionally being 2d
        dest_ptr = pnlvm.helpers.unwrap_2d_array(builder, oport_input)
        dest_ptr = builder.gep(dest_ptr, [ctx.int32_ty(0), ctx.int32_ty(0)])
        val_ptr = builder.gep(value, [ctx.int32_ty(0), ctx.int32_ty(0), ctx.int32_ty(i)])
        builder.store(builder.load(val_ptr), dest_ptr)
        return oport_input

    # @property
    # def state_feature_values(self):
    #     if hasattr(self.agent_rep, 'model_based_optimizer') and self.agent_rep.model_based_optimizer is self:
    #         return self.agent_rep._get_predicted_input()
    #     else:
    #         return np.array(np.array(self.variable[1:]).tolist())

    @property
    def agent_rep_type(self):
        from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator
        if isinstance(self.agent_rep, CompositionFunctionApproximator):
            return COMPOSITION_FUNCTION_APPROXIMATOR
        elif self.agent_rep.componentCategory=='Composition':
            return COMPOSITION
        else:
            return None

    def _parse_state_feature_function(self, feature_function):
        if isinstance(feature_function, Function):
            return copy.deepcopy(feature_function)
        else:
            return feature_function

    @tc.typecheck
    def _parse_state_feature_specs(self, state_features, feature_functions, context=None):
        """Parse entries of state_features into InputPort spec dictionaries
        Set INTERNAL_ONLY entry of params dict of InputPort spec dictionary to True
            (so that inputs to Composition are not required if the specified state is on an INPUT Mechanism)
        Assign functions specified in **state_feature_functions** to InputPorts for all state_features
        Return list of InputPort specification dictionaries
        """

        _state_input_ports = _parse_shadow_inputs(self, state_features)

        parsed_features = []

        for spec in _state_input_ports:
            # MODIFIED 11/29/21 NEW:
            # If optimization uses Composition, assume that shadowing a Mechanism means shadowing its primary InputPort
            if isinstance(spec, Mechanism) and self.agent_rep_type == COMPOSITION:
                # FIX: 11/29/21: MOVE THIS TO _parse_shadow_inputs
                #      (ADD ARG TO THAT FOR DOING SO, OR RESTRICTING TO INPUTPORTS IN GENERAL)
                if len(spec.input_ports)!=1:
                    raise OptimizationControlMechanismError(f"A Mechanism ({spec.name}) is specified in the "
                                                            f"'{STATE_FEATURES}' arg for {self.name} that has "
                                                            f"more than one InputPort; a specific one or subset "
                                                            f"of them must be specified.")
                spec = spec.input_port
            parsed_spec = _parse_port_spec(owner=self, port_type=InputPort, port_spec=spec)    # returns InputPort dict
            parsed_spec[PARAMS].update({INTERNAL_ONLY:True,
                                        PROJECTIONS:None})
            if feature_functions:
                if isinstance(feature_functions, dict) and spec in feature_functions:
                    feat_fct = feature_functions.pop(spec)
                else:
                    feat_fct = feature_functions
                parsed_spec.update({FUNCTION: self._parse_state_feature_function(feat_fct)})
            parsed_spec = [parsed_spec] # so that extend works below

            parsed_features.extend(parsed_spec)

        return parsed_features

    @property
    def num_state_input_ports(self):
        try:
            return len(self.state_input_ports)
        except:
            return 0

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

    # FIX: THE FOLLOWING SHOULD BE MERGED WITH HANDLING OF PredictionMechanisms FOR ORIG MODEL-BASED APPROACH;
    # FIX: SHOULD BE GENERALIZED AS SOMETHING LIKE update_feature_values
    @tc.typecheck
    @handle_external_context()
    def add_state_features(self, features, context=None):
        """Add InputPorts and Projections to OptimizationControlMechanism for state_features used to
        predict `net_outcome <ControlMechanism.net_outcome>`

        **state_features** argument can use any of the forms of specification allowed for InputPort(s)
        """

        if features:
            features = self._parse_state_feature_specs(features=features,
                                                       context=context)
        self.add_ports(InputPort, features)
